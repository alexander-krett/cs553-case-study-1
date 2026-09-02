from __future__ import annotations

import sys
import time
from pathlib import Path

import gradio as gr
import spaces
import torch
from transformers import pipeline

# Allow this app to import ../shared/prompts.py

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from shared.prompts import (  # noqa: E402
    REVIEW_TYPES,
    SYSTEM_PROMPT,
    build_prompt,
)


# Local model configuration

LOCAL_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"


# Load model at application startup.
# For ZeroGPU, Hugging Face recommends putting the model
# on CUDA at module level rather than moving it to CUDA
# inside the @spaces.GPU function.

# The Transformers pipeline handles:
#   downloading the model from Hugging Face,
#   loading the tokenizer,
#   loading the model weights,
#   generating text during inference.
# The model is configured for CUDA because this application will run
# on a Hugging Face ZeroGPU Space.


local_pipeline = pipeline(
    task="text-generation",
    model=LOCAL_MODEL,
    device="cuda",
    dtype=torch.bfloat16,
)


# Local ZeroGPU inference
# @spaces.GPU requests temporary GPU resources from Hugging Face
# ZeroGPU while this function is executing.
@spaces.GPU(duration=45)
def local_resume_review(
    resume_text,
    review_type,
    job_description,
    max_tokens,
    temperature,
):
    """
    Analyze a resume using SmolLM2 running directly
    on the Hugging Face Space with ZeroGPU.
    """

    # Build and validate the same prompt used by the remote model.
    try:
        prompt = build_prompt(
            resume_text=resume_text,
            review_type=review_type,
            job_description=job_description,
        )
    # Convert prompt-validation failures into a clear Gradio
    # error message.
    except ValueError as error:
        raise gr.Error(str(error)) from error

    # Format the request as a chat conversation
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


    # Convert chat messages into the model's expected
    # instruction format. apply_chat_template()
    # inserts the model-specific formatting tokens needed
    # for instruction-following generation.
    formatted_prompt = (
        local_pipeline
        .tokenizer
        .apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )


    # Measure local inference time.
    start_time = time.perf_counter()

    # Configure text generation
    generation_settings = {
        # Maximum number of tokens generated after the prompt.
        "max_new_tokens": int(max_tokens),
        # Only return newly generated model text rather than
        # including the original prompt in the output.
        "return_full_text": False,
    }

    # Configure sampling behavior
    # Only enable sampling when temperature > 0.
    # If temperature is zero, use deterministic greedy decoding.
    # Otherwise, enable sampling with the selected temperature.
    if float(temperature) > 0:
        generation_settings.update(
            {
                "do_sample": True,
                "temperature": float(temperature),
                "top_p": 0.9,
            }
        )

    else:
        generation_settings["do_sample"] = False

    # Run local model inference
    try:
        result = local_pipeline(
            formatted_prompt,
            **generation_settings,
        )

    except Exception as error:
        # Display model failures cleanly inside Gradio.
        raise gr.Error(
            f"Local model inference failed: {error}"
        ) from error

    # Stop the inference timer.
    elapsed_time = time.perf_counter() - start_time

    # Extract the generated text.
    response = result[0]["generated_text"].strip()


    if not response:
        raise gr.Error(
            "The local model returned an empty response."
        )

# Build inference metadata
# This information will later help us compare local and
# remote latency and execution behavior.
    model_information = f"""
### Inference Information

Model: `{LOCAL_MODEL}`

Execution: Local Hugging Face ZeroGPU

Response time: {elapsed_time:.2f} seconds
"""


    return response, model_information

# Gradio interface

with gr.Blocks(
    # Application title and privacy notice
    title="ResumeLens AI - Local"
) as demo:

    gr.Markdown(
        """
        # ResumeLens AI

        ### Local ZeroGPU Resume Reviewer

        Analyze a resume using a lightweight language model
        executed directly on this Hugging Face Space.

        **For this class demonstration, use a synthetic or sanitized
        resume whenever possible.**
        """
    )

    # Main two-column interface
    with gr.Row():

        # # Left side: Input columns

        with gr.Column():
            # Resume text supplied by the user.
            resume_text = gr.Textbox(
                label="Resume",
                placeholder=(
                    "Paste your resume text here..."
                ),
                lines=20,
            )

            # User selects which type of resume analysis to perform.
            review_type = gr.Dropdown(
                choices=REVIEW_TYPES,
                value="General Resume Review",
                label="Review Type",
            )

            # Optional unless the Job Description Match mode is used.
            job_description = gr.Textbox(
                label="Job Description",
                placeholder=(
                    "Optional unless using "
                    "'Job Description Match'."
                ),
                lines=10,
            )

            # Starts local ZeroGPU inference.
            analyze_button = gr.Button(
                "Analyze Resume",
                variant="primary",
            )

        # Right sides: Output column

        with gr.Column():
            # Main resume-review feedback from SmolLM2.
            feedback = gr.Markdown(
                """
                ### Resume Feedback

                Your analysis will appear here.
                """
            )

            # Displays the local model name and latency.
            model_information = gr.Markdown()


    # Generation controls
    # These are hidden inside an accordion so the main resume-review
    # experience remains simple.

    with gr.Accordion(
        "Generation Settings",
        open=False,
    ):
        # Maximum response length.
        max_tokens = gr.Slider(
            minimum=64,
            maximum=512,
            value=300,
            step=32,
            label="Maximum Response Tokens",
        )

        # Controls generation randomness.
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.3,
            step=0.1,
            label="Temperature",
        )


    # Examples for demonstrations
    # These provide quick test/demo cases and can later be reused
    # when comparing local and remote model performance.

    gr.Examples(
        examples=[
            [
                """
EDUCATION
B.S. Computer Science

EXPERIENCE
Research Assistant
- Worked with machine learning models.
- Helped analyze data.
- Used Python.
""",
                "Bullet Point Strength",
                "",
            ],

            [
                """
EDUCATION
M.S. Data Science

SKILLS
Python, PyTorch, SQL

EXPERIENCE
Research Assistant
- Developed predictive models for mobility data.
- Evaluated models using held-out datasets.
""",
                "Job Description Match",
                """
Machine Learning Engineer Intern

Requirements:
- Python
- PyTorch
- Git
- Docker
- AWS
""",
            ],
        ],

        inputs=[
            resume_text,
            review_type,
            job_description,
        ],
    )


    # Connect the Analyze Resume button to local inference

    analyze_button.click(
        fn=local_resume_review,

        inputs=[
            resume_text,
            review_type,
            job_description,
            max_tokens,
            temperature,
        ],

        outputs=[
            feedback,
            model_information,
        ],
    )

# Queue requests before ZeroGPU execution.
demo.queue()

# Launch the local application when this file is run directly.
if __name__ == "__main__":
    demo.launch()