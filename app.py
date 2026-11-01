from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from src.examples import EXAMPLES
from src.prompts import REVIEW_TYPES, SYSTEM_PROMPT, build_prompt


load_dotenv(Path(__file__).resolve().parent / ".env")

# One container runs one product: "api" (Hugging Face Inference API) or
# "local" (Q4 GGUF model on the container CPU via llama.cpp).
PRODUCT_MODE = os.getenv("PRODUCT_MODE", "local").strip().lower()
if PRODUCT_MODE not in {"api", "local"}:
    raise RuntimeError(f"PRODUCT_MODE must be 'api' or 'local', not {PRODUCT_MODE!r}")

ROOT_PATH = os.getenv("ROOT_PATH", "")
APP_VERSION = os.getenv("APP_VERSION", "dev")
METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))
INFERENCE_TIMEOUT = float(os.getenv("INFERENCE_TIMEOUT", "300"))
LOCAL_CONTEXT_SIZE = int(os.getenv("LOCAL_CONTEXT_SIZE", "4096"))
LOCAL_THREADS = int(os.getenv("LOCAL_THREADS", "2"))

# Local models are Q4 GGUF files: a Hugging Face repo plus a filename pattern.
LOCAL_PRIMARY_MODEL = os.getenv("LOCAL_PRIMARY_MODEL", "unsloth/Qwen3-0.6B-GGUF")
LOCAL_PRIMARY_FILE = os.getenv("LOCAL_PRIMARY_FILE", "*Q4_K_M.gguf")
LOCAL_BACKUP_MODEL = os.getenv("LOCAL_BACKUP_MODEL", "LiquidAI/LFM2.5-1.2B-Instruct-GGUF")
LOCAL_BACKUP_FILE = os.getenv("LOCAL_BACKUP_FILE", "*QAD-Q4_0.gguf")
LOCAL_MODEL_FILES = {
    LOCAL_PRIMARY_MODEL: LOCAL_PRIMARY_FILE,
    LOCAL_BACKUP_MODEL: LOCAL_BACKUP_FILE,
}
REMOTE_PRIMARY_MODEL = os.getenv("REMOTE_PRIMARY_MODEL", "openai/gpt-oss-20b")
REMOTE_BACKUP_MODEL = os.getenv("REMOTE_BACKUP_MODEL", "zai-org/GLM-5.3-Flash")

PRODUCT_MODELS = {
    "local": (LOCAL_PRIMARY_MODEL, LOCAL_BACKUP_MODEL),
    "api": (REMOTE_PRIMARY_MODEL, REMOTE_BACKUP_MODEL),
}
MODEL_LABELS = {
    LOCAL_PRIMARY_MODEL: "Qwen3 0.6B Q4_K_M (primary)",
    LOCAL_BACKUP_MODEL: "LFM2.5 1.2B Q4_0 (backup)",
    REMOTE_PRIMARY_MODEL: "GPT-OSS 20B (primary)",
    REMOTE_BACKUP_MODEL: "GLM 5.3 Flash (backup)",
}


def product_models():
    """Return (primary, backup) model IDs for the active product."""
    return PRODUCT_MODELS[PRODUCT_MODE]


# --- Prometheus metrics (served on METRICS_PORT, scraped by Prometheus) ---
REQUESTS = Counter(
    "resumelens_requests_total",
    "Resume reviews handled, by outcome.",
    ["product", "result"],
)
REQUEST_DURATION = Histogram(
    "resumelens_request_duration_seconds",
    "End-to-end resume review time, including validation and failover.",
    ["product"],
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60, 120, 180, 300, 600),
)
IN_PROGRESS = Gauge(
    "resumelens_in_progress_requests",
    "Resume reviews currently being processed.",
    ["product"],
)
INPUT_CHARACTERS = Histogram(
    "resumelens_input_characters",
    "Characters submitted per review (resume plus job description).",
    ["product"],
    buckets=(100, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000),
)
OUTPUT_CHARACTERS = Histogram(
    "resumelens_output_characters",
    "Characters returned per successful review.",
    ["product"],
    buckets=(100, 250, 500, 1000, 2000, 4000, 8000, 16000),
)
MODEL_CALLS = Counter(
    "resumelens_model_calls_total",
    "Individual model attempts, including failover attempts.",
    ["product", "model", "result"],
)
ERRORS = Counter(
    "resumelens_errors_total",
    "Failed reviews, by error type.",
    ["product", "error_type"],
)
BUILD_INFO = Gauge(
    "resumelens_build_info",
    "Static build and configuration information (always 1).",
    ["product", "version", "primary_model", "backup_model"],
)


def _review_error(message, error_type):
    error = gr.Error(message)
    error.error_type = error_type
    return error


def _is_timeout(error):
    return isinstance(error, TimeoutError) or "timeout" in type(error).__name__.lower()


def require_hf_token(token):
    """Return the token typed into the UI; environment variables are never read."""
    token = (token or "").strip()
    if not token:
        raise _review_error(
            "A Hugging Face token is required. Enter it in the sidebar.", "missing_token"
        )
    return token


def _validated_prompt(resume_text, review_type, job_description):
    try:
        return build_prompt(resume_text, review_type, job_description)
    except ValueError as error:
        raise _review_error(str(error), "validation") from error


def _messages(prompt):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def _safe_error(error, token=""):
    # Provider exceptions can include request credentials or URLs.
    message = str(error)
    return message.replace(token, "[redacted]") if token else message


def _inference_information(model_id, execution, elapsed_time, fallback=False):
    fallback_line = "\nFailover: Backup model used\n" if fallback else ""
    return f"""
### Inference Information

Model: `{model_id}`

Execution: {execution}
{fallback_line}
Response time: {elapsed_time:.2f} seconds
"""


def _review_with_failover(candidates, run, execution, token=""):
    """Try each model in order, recording every attempt."""
    failures = []
    timeouts = 0
    start_time = time.perf_counter()

    for index, model_id in enumerate(candidates):
        try:
            response = run(model_id)
        except Exception as error:
            MODEL_CALLS.labels(PRODUCT_MODE, model_id, "error").inc()
            timeouts += _is_timeout(error)
            failures.append(f"{model_id}: {_safe_error(error, token)}")
            continue
        MODEL_CALLS.labels(PRODUCT_MODE, model_id, "success").inc()
        elapsed_time = time.perf_counter() - start_time
        return response, _inference_information(
            model_id, execution, elapsed_time, index > 0
        )

    error_type = "timeout" if timeouts == len(candidates) else "inference"
    raise _review_error(f"{execution} inference failed. " + " | ".join(failures), error_type)


# --- Local CPU inference (llama.cpp, Q4 GGUF) ---
# Only one model stays resident so the container fits on the shared VM.
_local_lock = threading.RLock()
_local_cache = {}


def _load_local_model(model_id):
    with _local_lock:
        if model_id not in _local_cache:
            # Imported lazily so the API product and tests do not need llama.cpp.
            from llama_cpp import Llama

            _local_cache.clear()
            _local_cache[model_id] = Llama.from_pretrained(
                model_id,
                filename=LOCAL_MODEL_FILES[model_id],
                n_ctx=LOCAL_CONTEXT_SIZE,
                n_threads=LOCAL_THREADS,
                verbose=False,
            )
        return _local_cache[model_id]


def _warm_local_model():
    try:
        _load_local_model(LOCAL_PRIMARY_MODEL)
        print(f"Loaded {LOCAL_PRIMARY_MODEL}", flush=True)
    except Exception as error:
        print(f"Could not preload {LOCAL_PRIMARY_MODEL}: {error}", flush=True)


def _run_local_model(model_id, prompt, max_tokens, temperature):
    messages = _messages(prompt)
    if "qwen3" in model_id.lower():
        # Qwen3 soft switch: answer directly instead of reasoning first.
        messages[-1]["content"] += "\n\n/no_think"
    with _local_lock:
        llm = _load_local_model(model_id)
        text = "\n".join(message["content"] for message in messages)
        # The margin covers the chat template's role markers.
        input_tokens = len(llm.tokenize(text.encode("utf-8"))) + 64
        if input_tokens + int(max_tokens) > LOCAL_CONTEXT_SIZE:
            raise RuntimeError("Input plus requested response exceeds local context; shorten input or reduce response tokens.")
        result = llm.create_chat_completion(
            messages=messages,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=0.95,
        )
    response = result["choices"][0]["message"].get("content") or ""
    response = response.split("</think>")[-1]
    if not response.strip():
        raise RuntimeError("the model returned an empty response")
    return response.strip()


def local_resume_review(
    resume_text, review_type, job_description, max_tokens, temperature, models=None
):
    """Analyze a resume on the container CPU, failing over to the backup model."""
    prompt = _validated_prompt(resume_text, review_type, job_description)
    return _review_with_failover(
        models or list(PRODUCT_MODELS["local"]),
        lambda model_id: _run_local_model(model_id, prompt, max_tokens, temperature),
        "Local CPU (llama.cpp, Q4)",
    )


# --- Remote inference (Hugging Face Inference API) ---
def _remote_extra_body(model_id):
    if model_id == REMOTE_PRIMARY_MODEL:
        return {"reasoning_effort": "low"}
    if model_id == REMOTE_BACKUP_MODEL:
        return {"reasoning_effort": "low", "clear_thinking": True}
    return None


def _run_remote_model(model_id, token, prompt, max_tokens, temperature):
    client = InferenceClient(token=token, model=model_id, timeout=INFERENCE_TIMEOUT)
    request = {
        "messages": _messages(prompt),
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": 0.95,
        "stream": False,
    }
    extra_body = _remote_extra_body(model_id)
    if extra_body:
        request["extra_body"] = extra_body
    completion = client.chat_completion(**request)
    choice = completion.choices[0]
    response = choice.message.content
    if not response or not response.strip():
        finish_reason = choice.finish_reason or "unknown"
        raise RuntimeError(f"empty response (finish reason: {finish_reason})")
    return response.strip()


def remote_resume_review(
    resume_text, review_type, job_description, max_tokens, temperature, token, models=None
):
    """Analyze a resume remotely with the user's token, failing over to the backup model."""
    prompt = _validated_prompt(resume_text, review_type, job_description)
    token = require_hf_token(token)
    return _review_with_failover(
        models or list(PRODUCT_MODELS["api"]),
        lambda model_id: _run_remote_model(model_id, token, prompt, max_tokens, temperature),
        "Remote Hugging Face API",
        token,
    )


def _route_review(
    resume_text, review_type, job_description, max_tokens, temperature,
    automatic_failover, selected_model, hf_token,
):
    models = None
    if not automatic_failover:
        if selected_model not in product_models():
            raise _review_error(f"Invalid model: {selected_model}", "validation")
        models = [selected_model]

    common_args = (resume_text, review_type, job_description, max_tokens, temperature)
    if PRODUCT_MODE == "api":
        return remote_resume_review(*common_args, hf_token, models=models)
    return local_resume_review(*common_args, models=models)


def analyze_resume(
    resume_text,
    review_type,
    job_description,
    max_tokens,
    temperature,
    automatic_failover,
    selected_model,
    hf_token="",
):
    """Gradio entry point: route the review and record product metrics."""
    start_time = time.perf_counter()
    IN_PROGRESS.labels(PRODUCT_MODE).inc()
    INPUT_CHARACTERS.labels(PRODUCT_MODE).observe(
        len(resume_text or "") + len(job_description or "")
    )
    try:
        response, information = _route_review(
            resume_text, review_type, job_description, max_tokens, temperature,
            automatic_failover, selected_model, hf_token,
        )
    except Exception as error:
        REQUESTS.labels(PRODUCT_MODE, "error").inc()
        ERRORS.labels(PRODUCT_MODE, getattr(error, "error_type", "inference")).inc()
        raise
    finally:
        REQUEST_DURATION.labels(PRODUCT_MODE).observe(time.perf_counter() - start_time)
        IN_PROGRESS.labels(PRODUCT_MODE).dec()

    REQUESTS.labels(PRODUCT_MODE, "success").inc()
    OUTPUT_CHARACTERS.labels(PRODUCT_MODE).observe(len(response))
    return response, information


def model_picker_state(automatic_failover):
    """Enable the model picker only when automatic failover is off."""
    return gr.Dropdown(interactive=not automatic_failover)


PRIMARY_MODEL, BACKUP_MODEL = product_models()
BUILD_INFO.labels(PRODUCT_MODE, APP_VERSION, PRIMARY_MODEL, BACKUP_MODEL).set(1)

if PRODUCT_MODE == "api":
    PRODUCT_DESCRIPTION = (
        "Analyze a resume with a hosted model through the Hugging Face Inference API. "
        "Enter your own Hugging Face token in the sidebar; it is used only for your request."
    )
else:
    PRODUCT_DESCRIPTION = (
        "Analyze a resume with a small 4-bit quantized model running on this container's CPU. "
        "The first review may wait while the model downloads."
    )


with gr.Blocks(title="ResumeLens AI") as demo:
    with gr.Sidebar():
        gr.Markdown("## Inference")
        hf_token = gr.Textbox(
            label="Hugging Face token",
            type="password",
            placeholder="hf_...",
            info="Needs inference-provider permission. Not stored.",
            visible=PRODUCT_MODE == "api",
        )

        with gr.Accordion("Advanced settings", open=False):
            automatic_failover = gr.Checkbox(
                value=True,
                label="Automatic failover",
                info="Try the primary model, then the backup model.",
            )
            selected_model = gr.Dropdown(
                choices=[(MODEL_LABELS.get(m, m), m) for m in (PRIMARY_MODEL, BACKUP_MODEL)],
                value=PRIMARY_MODEL,
                label="Model",
                interactive=False,
            )
            max_tokens = gr.Slider(
                minimum=256,
                maximum=2048,
                value=2048 if PRODUCT_MODE == "api" else 1024,
                step=32,
                label="Maximum Response Tokens",
            )
            temperature = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.3,
                step=0.1,
                label="Temperature",
            )

    gr.Markdown(
        f"""
# ResumeLens AI — {"API" if PRODUCT_MODE == "api" else "Local CPU"} product

{PRODUCT_DESCRIPTION}

**For this class demonstration, use a synthetic or sanitized resume whenever possible.**
"""
    )

    with gr.Row():
        with gr.Column():
            resume_text = gr.Textbox(
                label="Resume", placeholder="Paste your resume text here...", lines=20
            )
            review_type = gr.Dropdown(
                choices=REVIEW_TYPES,
                value="General Resume Review",
                label="Review Type",
            )
            job_description = gr.Textbox(
                label="Job Description",
                placeholder="Optional unless using 'Job Description Match'.",
                lines=10,
            )
            analyze_button = gr.Button("Analyze Resume", variant="primary")

        with gr.Column():
            feedback = gr.Markdown("### Resume Feedback\n\nYour analysis will appear here.")
            model_information = gr.Markdown()

    gr.Examples(examples=EXAMPLES, inputs=[resume_text, review_type, job_description])

    analyze_button.click(
        fn=analyze_resume,
        inputs=[
            resume_text,
            review_type,
            job_description,
            max_tokens,
            temperature,
            automatic_failover,
            selected_model,
            hf_token,
        ],
        outputs=[feedback, model_information],
    )
    automatic_failover.change(
        fn=model_picker_state,
        inputs=automatic_failover,
        outputs=selected_model,
    )

demo.queue(default_concurrency_limit=1, max_size=8)

if __name__ == "__main__":
    start_http_server(METRICS_PORT)
    if PRODUCT_MODE == "local":
        threading.Thread(target=_warm_local_model, daemon=True).start()
    demo.launch(server_name="0.0.0.0", server_port=7860, root_path=ROOT_PATH, share=False)
