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

- **Local-first** tries `Qwen/Qwen3.5-4B` and
  `ibm-granite/granite-4.2-3b` on Hugging Face ZeroGPU, then crosses to the
  remote models if both fail.
- **Remote** uses `openai/gpt-oss-20b` through the Hugging Face Inference API
  and falls back to `zai-org/GLM-5.3-Flash`, then crosses to the local models
  if both fail.

Choose the inference mode in the sidebar, paste a synthetic or sanitized resume,
select a review type, and click **Analyze Resume**. Remote inference requires a
Hugging Face sign-in; Local inference does not. Automatic failover is enabled by
default. Under **Advanced settings**, a Local/Remote toggle chooses which mode
runs first when failover is on. Turning failover off replaces it with an
individual model selector. The shared generation controls are there as well.
Crossing into Remote requires Hugging Face sign-in.

Override the defaults with `LOCAL_PRIMARY_MODEL`, `LOCAL_BACKUP_MODEL`,
`REMOTE_PRIMARY_MODEL`, and `REMOTE_BACKUP_MODEL`.

The root `app.py` is the deployment entrypoint, and `src/prompts.py` contains
prompt construction and input validation shared by both inference modes.
`src/examples.py` contains the sample resumes and job description.
