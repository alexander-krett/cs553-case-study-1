---
title: Resume Reviewer - CS553 - MLOps Case Study 1 Homework
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_scopes:
- inference-api
---

ResumeLens AI reviews resumes using a single Gradio interface with two inference
modes:

- **Local** runs `Qwen/Qwen3-4B-Instruct-2507` on Hugging Face ZeroGPU.
- **Remote** uses `openai/gpt-oss-20b` through the Hugging Face Inference API.

Choose the inference mode in the sidebar, paste a synthetic or sanitized resume,
select a review type, and click **Analyze Resume**. Remote inference requires a
Hugging Face sign-in; Local inference does not. Set the `REMOTE_MODEL` environment
variable to use a different remote model.

The root `app.py` is the deployment entrypoint, and `src/prompts.py` contains
prompt construction and input validation shared by both inference modes.
`src/examples.py` contains the sample resumes and job description.

## Testing & Deployment

Automated CI/CD runs on every push to `main` via GitHub Actions:
1. **Testing**: Runs the `pytest` test suite covering prompt construction, validation rules, example resumes, and model routing.
2. **Deployment**: Automatically mirrors to the [`akrett/cs553-ml-ops`](https://huggingface.co/spaces/akrett/cs553-ml-ops) Hugging Face Space upon passing tests, triggering a rebuild. The workflow does not verify application readiness.
3. **Discord Notification**: Sends a detailed Discord embed with commit metadata, author, test results, and live deployment links.

To run tests locally:
```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest
```

Before running the workflow, ensure these GitHub Actions repository secrets are configured:
- `HF_TOKEN`: a Hugging Face token with permission to write to the Space.
- `DISCORD_WEBHOOK`: the URL of the Discord webhook that receives notifications.
