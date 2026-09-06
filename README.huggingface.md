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
