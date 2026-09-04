from __future__ import annotations

import os
import time
from functools import lru_cache

import gradio as gr
import spaces
from huggingface_hub import InferenceClient

from prompts import REVIEW_TYPES, SYSTEM_PROMPT, build_prompt


LOCAL_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REMOTE_MODEL = os.getenv("REMOTE_MODEL", "openai/gpt-oss-20b")
INFERENCE_MODES = ["Local", "Remote"]


@lru_cache(maxsize=1)
def get_local_pipeline():
    """Load the local model only when Local inference is first requested."""
    import torch
    from transformers import pipeline

    return pipeline(
        task="text-generation",
        model=LOCAL_MODEL,
        device="cuda",
        dtype=torch.bfloat16,
    )


def _validated_prompt(resume_text, review_type, job_description):
    try:
        return build_prompt(resume_text, review_type, job_description)
    except ValueError as error:
        raise gr.Error(str(error)) from error


@spaces.GPU(duration=45)
def local_resume_review(
    resume_text, review_type, job_description, max_tokens, temperature
):
    """Analyze a resume with SmolLM2 on Hugging Face ZeroGPU."""
    prompt = _validated_prompt(resume_text, review_type, job_description)
    local_pipeline = get_local_pipeline()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    formatted_prompt = local_pipeline.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    generation_settings = {
        "max_new_tokens": int(max_tokens),
        "return_full_text": False,
    }
    if float(temperature) > 0:
        generation_settings.update(
            do_sample=True, temperature=float(temperature), top_p=0.9
        )
    else:
        generation_settings["do_sample"] = False

    start_time = time.perf_counter()
    try:
        result = local_pipeline(formatted_prompt, **generation_settings)
    except Exception as error:
        raise gr.Error(f"Local model inference failed: {error}") from error
    elapsed_time = time.perf_counter() - start_time

    response = result[0]["generated_text"].strip()
    if not response:
        raise gr.Error("The local model returned an empty response.")

    information = f"""
### Inference Information

Model: `{LOCAL_MODEL}`

Execution: Local Hugging Face ZeroGPU

Response time: {elapsed_time:.2f} seconds
"""
    return response, information


@spaces.GPU(duration=30)
def remote_resume_review(
    resume_text,
    review_type,
    job_description,
    max_tokens,
    temperature,
    hf_token: gr.OAuthToken | None,
):
    """Analyze a resume through the Hugging Face Inference API."""
    if hf_token is None:
        raise gr.Error("Please sign in with Hugging Face before using the remote model.")

    prompt = _validated_prompt(resume_text, review_type, job_description)
    client = InferenceClient(token=hf_token.token, model=REMOTE_MODEL)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    start_time = time.perf_counter()
    try:
        completion = client.chat_completion(
            messages=messages,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=0.9,
            stream=False,
        )
    except Exception as error:
        raise gr.Error(f"Remote inference failed: {error}") from error
    elapsed_time = time.perf_counter() - start_time

    response = completion.choices[0].message.content
    if not response or not response.strip():
        raise gr.Error("The remote model returned an empty response.")

    information = f"""
### Inference Information

Model: `{REMOTE_MODEL}`

Execution: Remote Hugging Face API

Response time: {elapsed_time:.2f} seconds
"""
    return response, information


def analyze_resume(
    resume_text,
    review_type,
    job_description,
    max_tokens,
    temperature,
    inference_mode,
    hf_token: gr.OAuthToken | None,
):
    """Route a review request to the selected inference implementation."""
    if inference_mode == "Local":
        return local_resume_review(
            resume_text, review_type, job_description, max_tokens, temperature
        )
    if inference_mode == "Remote":
        return remote_resume_review(
            resume_text,
            review_type,
            job_description,
            max_tokens,
            temperature,
            hf_token,
        )
    raise gr.Error(f"Invalid inference mode: {inference_mode}")


with gr.Blocks(title="ResumeLens AI") as demo:
    with gr.Sidebar():
        gr.Markdown("## Inference")
        inference_mode = gr.Radio(
            choices=INFERENCE_MODES, value="Remote", label="Model execution"
        )
        gr.Markdown("Sign in before using Remote inference.")
        gr.LoginButton()
        gr.Markdown("## Generation Settings")
        max_tokens = gr.Slider(
            minimum=64,
            maximum=512,
            value=300,
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

    gr.Examples(
        examples=[
            [
                """EDUCATION
B.S. Computer Science

EXPERIENCE
Research Assistant
- Worked with machine learning models.
- Helped analyze data.
- Used Python.""",
                "Bullet Point Strength",
                "",
            ],
            [
                """EDUCATION
M.S. Data Science

SKILLS
Python, PyTorch, SQL

EXPERIENCE
Research Assistant
- Developed predictive models for mobility data.
- Evaluated models using held-out datasets.""",
                "Job Description Match",
                """Machine Learning Engineer Intern

Requirements:
- Python
- PyTorch
- Git
- Docker
- AWS""",
            ],
        ],
        inputs=[resume_text, review_type, job_description],
    )

    analyze_button.click(
        fn=analyze_resume,
        inputs=[
            resume_text,
            review_type,
            job_description,
            max_tokens,
            temperature,
            inference_mode,
        ],
        outputs=[feedback, model_information],
    )

demo.queue()

if __name__ == "__main__":
    demo.launch()
