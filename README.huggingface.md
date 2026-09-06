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
modes and automatic failover:

- **Local** runs `Qwen/Qwen3.5-4B` on Hugging Face ZeroGPU and falls back to
  `ibm-granite/granite-4.2-3b`.
- **Remote** uses `zai-org/GLM-5.3-Flash` through the Hugging Face Inference API
  and falls back to `openai/gpt-oss-20b`.

Choose the inference mode in the sidebar, paste a synthetic or sanitized resume,
select a review type, and click **Analyze Resume**. Remote inference requires a
Hugging Face sign-in; Local inference does not. Automatic failover is enabled by
default; disable it to select any model directly. Advanced settings contain the
Local/Remote failover mode and shared generation controls.

Override the defaults with `LOCAL_PRIMARY_MODEL`, `LOCAL_BACKUP_MODEL`,
`REMOTE_PRIMARY_MODEL`, and `REMOTE_BACKUP_MODEL`.

The root `app.py` is the deployment entrypoint, and `src/prompts.py` contains
prompt construction and input validation shared by both inference modes.
`src/examples.py` contains the sample resumes and job description.
