# ResumeLens AI

ResumeLens AI is a Gradio application that reviews resumes against a job
description. It supports two inference modes:

- **Local** runs `Qwen/Qwen3-4B-Instruct-2507` on Hugging Face ZeroGPU.
- **Remote** uses `openai/gpt-oss-20b` through the Hugging Face Inference API.

Use synthetic or sanitized resumes only. Choose an inference mode, paste a
resume, select a review type, and click **Analyze Resume**. Remote inference
requires a Hugging Face sign-in; Local inference does not. Set `REMOTE_MODEL`
to choose a different remote model.

The app is deployed at [akrett/cs553-ml-ops](https://huggingface.co/spaces/akrett/cs553-ml-ops).

## Project layout

- `app.py`: Gradio application entrypoint.
- `src/prompts.py`: shared prompt construction and input validation.
- `src/examples.py`: example resumes and job description.
- `tests/`: unit tests for prompts, validation, examples, and model routing.

## Local development

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

The full app dependencies for a Hugging Face Space are in `requirements.txt`.

## CI/CD

Every push to `main` runs the test suite, syncs passing code to the Hugging
Face Space, and sends a Discord notification. The deployment uses the official
`huggingface/hub-sync` action and stages `README.huggingface.md` as the Space's
root `README.md`; this repository's `README.md` remains GitHub-specific.

Configure these GitHub Actions repository secrets before deploying:

- `HF_TOKEN`: a Hugging Face token with write permission to the Space.
- `DISCORD_WEBHOOK`: the Discord webhook URL for deployment notifications.
