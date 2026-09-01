---
title: HaikuGPT - CS553 - MLOps Case Study 1 Homework
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_scopes:
- inference-api
---

An example chatbot using [Gradio](https://gradio.app), [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/v0.22.2/en/index), and the [Hugging Face Inference API](https://huggingface.co/docs/api-inference/index).

## Deployment

Pushes to `main` are automatically mirrored to the
[`akrett/cs553-ml-ops`](https://huggingface.co/spaces/akrett/cs553-ml-ops) Hugging Face
Space. Add a GitHub Actions repository secret named `HF_TOKEN` with permission to
write to the Space before running the workflow.
