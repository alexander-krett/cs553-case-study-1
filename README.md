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

- **Local** runs `HuggingFaceTB/SmolLM2-360M-Instruct` on Hugging Face ZeroGPU.
- **Remote** uses `openai/gpt-oss-20b` through the Hugging Face Inference API.

Choose the inference mode in the sidebar, paste a synthetic or sanitized resume,
select a review type, and click **Analyze Resume**. Remote inference requires a
Hugging Face sign-in; Local inference does not. Set the `REMOTE_MODEL` environment
variable to use a different remote model.

The root `app.py` is the deployment entrypoint, and `prompts.py` contains prompt
construction and input validation shared by both inference modes.

## Deployment

Pushes to `main` are automatically mirrored to the
[`akrett/cs553-ml-ops`](https://huggingface.co/spaces/akrett/cs553-ml-ops) Hugging Face
Space, then a deployment notification is sent to Discord. Before running the
workflow, add these GitHub Actions repository secrets:

- `HF_TOKEN`: a Hugging Face token with permission to write to the Space.
- `DISCORD_WEBHOOK`: the URL of the Discord webhook that receives notifications.
