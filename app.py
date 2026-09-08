from __future__ import annotations

import os
import time
from functools import lru_cache

import gradio as gr
import spaces
from huggingface_hub import InferenceClient

from src.examples import EXAMPLES
from src.prompts import REVIEW_TYPES, SYSTEM_PROMPT, build_prompt


LOCAL_PRIMARY_MODEL = os.getenv("LOCAL_PRIMARY_MODEL", "ibm-granite/granite-4.2-3b")
LOCAL_BACKUP_MODEL = os.getenv("LOCAL_BACKUP_MODEL", "Qwen/Qwen3.5-0.8B")
REMOTE_PRIMARY_MODEL = os.getenv("REMOTE_PRIMARY_MODEL", "openai/gpt-oss-20b")
REMOTE_BACKUP_MODEL = os.getenv("REMOTE_BACKUP_MODEL", "zai-org/GLM-5.3-Flash")

MODEL_EXECUTION = {
    LOCAL_PRIMARY_MODEL: "Local",
    LOCAL_BACKUP_MODEL: "Local",
    REMOTE_PRIMARY_MODEL: "Remote",
    REMOTE_BACKUP_MODEL: "Remote",
}
MODEL_CHOICES = [
    ("Remote — GPT-OSS 20B", REMOTE_PRIMARY_MODEL),
    ("Remote — GLM 5.3 Flash", REMOTE_BACKUP_MODEL),
    ("Local — Granite 4.2 3B", LOCAL_PRIMARY_MODEL),
    ("Local — Qwen 3.5 0.8B", LOCAL_BACKUP_MODEL),
]


@lru_cache(maxsize=2)
def get_local_pipeline(model_id):
    """Load a local model only when it is first requested."""
    import torch
    from transformers import pipeline

    task = "image-text-to-text" if model_id == LOCAL_BACKUP_MODEL else "text-generation"
    return pipeline(
        task=task,
        model=model_id,
        device="cuda",
        dtype=torch.bfloat16,
    )


def _validated_prompt(resume_text, review_type, job_description):
    try:
        return build_prompt(resume_text, review_type, job_description)
    except ValueError as error:
        raise gr.Error(str(error)) from error


def _messages(prompt):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def _generation_settings(max_tokens, temperature):
    settings = {"max_new_tokens": int(max_tokens), "return_full_text": False}
    if float(temperature) > 0:
        settings.update(do_sample=True, temperature=float(temperature), top_p=0.95)
    else:
        settings["do_sample"] = False
    return settings


def _extract_generated_text(result):
    """Normalize text and conversational pipeline response shapes."""
    generated = result[0]["generated_text"]
    if isinstance(generated, str):
        return generated.strip()
    if isinstance(generated, list) and generated:
        content = generated[-1].get("content", "")
        if isinstance(content, str):
            return content.strip()
    return ""


def _run_local_model(model_id, prompt, max_tokens, temperature):
    local_pipeline = get_local_pipeline(model_id)
    template_source = getattr(local_pipeline, "tokenizer", None)
    if template_source is None:
        template_source = local_pipeline.processor
    formatted_prompt = template_source.apply_chat_template(
        _messages(prompt),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    settings = _generation_settings(max_tokens, temperature)
    if model_id == LOCAL_BACKUP_MODEL:
        result = local_pipeline(text=formatted_prompt, **settings)
    else:
        result = local_pipeline(formatted_prompt, **settings)
    response = _extract_generated_text(result)
    if not response:
        raise RuntimeError("the model returned an empty response")
    return response


def _inference_information(model_id, execution, elapsed_time, fallback=False):
    fallback_line = "\nFailover: Backup model used\n" if fallback else ""
    return f"""
### Inference Information

Model: `{model_id}`

Execution: {execution}
{fallback_line}
Response time: {elapsed_time:.2f} seconds
"""


@spaces.GPU(duration=45)
def local_resume_review(
    resume_text, review_type, job_description, max_tokens, temperature, models=None
):
    """Analyze a resume locally, optionally failing over to a backup model."""
    prompt = _validated_prompt(resume_text, review_type, job_description)
    candidates = models or [LOCAL_PRIMARY_MODEL, LOCAL_BACKUP_MODEL]
    failures = []
    start_time = time.perf_counter()

    for index, model_id in enumerate(candidates):
        try:
            response = _run_local_model(model_id, prompt, max_tokens, temperature)
            elapsed_time = time.perf_counter() - start_time
            return response, _inference_information(
                model_id, "Local Hugging Face ZeroGPU", elapsed_time, index > 0
            )
        except Exception as error:
            failures.append(f"{model_id}: {error}")

    raise gr.Error("Local inference failed. " + " | ".join(failures))


def _remote_extra_body(model_id):
    if model_id == REMOTE_PRIMARY_MODEL:
        return {"reasoning_effort": "low"}
    if model_id == REMOTE_BACKUP_MODEL:
        return {"reasoning_effort": "low", "clear_thinking": True}
    return None


def _run_remote_model(model_id, token, prompt, max_tokens, temperature):
    client = InferenceClient(token=token, model=model_id)
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
    resume_text,
    review_type,
    job_description,
    max_tokens,
    temperature,
    hf_token: gr.OAuthToken | None,
    models=None,
):
    """Analyze a resume remotely, optionally failing over to a backup model."""
    if hf_token is None:
        raise gr.Error("Please sign in with Hugging Face before using a remote model.")

    prompt = _validated_prompt(resume_text, review_type, job_description)
    candidates = models or [REMOTE_PRIMARY_MODEL, REMOTE_BACKUP_MODEL]
    failures = []
    start_time = time.perf_counter()

    for index, model_id in enumerate(candidates):
        try:
            response = _run_remote_model(
                model_id, hf_token.token, prompt, max_tokens, temperature
            )
            elapsed_time = time.perf_counter() - start_time
            return response, _inference_information(
                model_id, "Remote Hugging Face API", elapsed_time, index > 0
            )
        except Exception as error:
            failures.append(f"{model_id}: {error}")

    raise gr.Error("Remote inference failed. " + " | ".join(failures))


def automatic_resume_review(
    resume_text,
    review_type,
    job_description,
    max_tokens,
    temperature,
    first_execution,
    hf_token: gr.OAuthToken | None,
):
    """Try both models in one execution mode, then cross to the other mode."""
    if first_execution not in {"Local", "Remote"}:
        raise gr.Error(f"Invalid inference mode: {first_execution}")

    # Reject user-input errors before attempting infrastructure failover.
    _validated_prompt(resume_text, review_type, job_description)
    common_args = (
        resume_text,
        review_type,
        job_description,
        max_tokens,
        temperature,
    )
    attempts = (
        ("Local", lambda: local_resume_review(*common_args)),
        ("Remote", lambda: remote_resume_review(*common_args, hf_token)),
    )
    if first_execution == "Remote":
        attempts = tuple(reversed(attempts))

    failures = []
    for index, (execution, review) in enumerate(attempts):
        try:
            response, information = review()
            if index > 0:
                information = information.replace(
                    "Response time:",
                    f"Failover: Switched from {first_execution} to {execution}\n\n"
                    "Response time:",
                )
            return response, information
        except Exception as error:
            failures.append(f"{execution}: {error}")

    raise gr.Error("All inference options failed. " + " || ".join(failures))


def analyze_resume(
    resume_text,
    review_type,
    job_description,
    max_tokens,
    temperature,
    automatic_failover,
    inference_mode,
    selected_model,
    hf_token: gr.OAuthToken | None,
):
    """Route a review using automatic failover or one explicitly selected model."""
    if automatic_failover:
        return automatic_resume_review(
            resume_text,
            review_type,
            job_description,
            max_tokens,
            temperature,
            inference_mode,
            hf_token,
        )

    execution = MODEL_EXECUTION.get(selected_model)
    if execution == "Local":
        return local_resume_review(
            resume_text,
            review_type,
            job_description,
            max_tokens,
            temperature,
            models=[selected_model],
        )
    if execution == "Remote":
        return remote_resume_review(
            resume_text,
            review_type,
            job_description,
            max_tokens,
            temperature,
            hf_token,
            models=[selected_model],
        )
    raise gr.Error(f"Invalid model: {selected_model}")


def execution_control_visibility(automatic_failover):
    """Show the mode toggle for failover or the model picker for direct use."""
    return (
        gr.Radio(visible=automatic_failover),
        gr.Dropdown(visible=not automatic_failover),
    )


with gr.Blocks(title="ResumeLens AI") as demo:
    with gr.Sidebar():
        gr.Markdown("## Inference")
        gr.Markdown("Sign in before using Remote inference.")
        gr.LoginButton()

        with gr.Accordion("Advanced settings", open=False):
            automatic_failover = gr.Checkbox(
                value=True,
                label="Automatic failover",
                info="Try all four models, starting with the selected execution.",
            )
            inference_mode = gr.Radio(
                choices=["Local", "Remote"],
                value="Remote",
                label="Start with",
            )
            selected_model = gr.Dropdown(
                choices=MODEL_CHOICES,
                value=REMOTE_PRIMARY_MODEL,
                label="Model",
                visible=False,
            )
            max_tokens = gr.Slider(
                minimum=256,
                maximum=2048,
                value=2048,
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
        """
# ResumeLens AI

Analyze a resume with either a model running locally on Hugging Face
ZeroGPU or a remotely hosted model through the Hugging Face Inference API.

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
            inference_mode,
            selected_model,
        ],
        outputs=[feedback, model_information],
    )
    automatic_failover.change(
        fn=execution_control_visibility,
        inputs=automatic_failover,
        outputs=[inference_mode, selected_model],
    )

demo.queue()

if __name__ == "__main__":
    demo.launch()
