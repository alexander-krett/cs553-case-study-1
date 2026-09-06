# ResumeLens AI

ResumeLens AI is a Gradio application that reviews resumes against a job
description. It supports two inference modes with automatic failover:

- **Local** runs `Qwen/Qwen3.5-4B` on Hugging Face ZeroGPU and falls back to
  `ibm-granite/granite-4.2-3b`.
- **Remote** uses `openai/gpt-oss-20b` through the Hugging Face Inference API
  and falls back to `zai-org/GLM-5.3-Flash`.

Use synthetic or sanitized resumes only. Choose an inference mode, paste a
resume, select a review type, and click **Analyze Resume**. Remote inference
requires a Hugging Face sign-in; Local inference does not. Automatic failover is
enabled by default. Disable it to select any of the four models directly.

The default model IDs can be changed with `LOCAL_PRIMARY_MODEL`,
`LOCAL_BACKUP_MODEL`, `REMOTE_PRIMARY_MODEL`, and `REMOTE_BACKUP_MODEL`.
Under **Advanced settings**, the unified **Execution** selector shows
Local/Remote chains when failover is on and individual models when it is off.
The shared generation controls are there as well, and every model uses the same
response-token budget.

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
