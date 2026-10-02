# ResumeLens AI — local CPU deployment

Gradio resume review app for Ubuntu 22.04 CPU hosts. Local-first automatic
failover tries Qwen3 0.6B, LFM2.5 1.2B, GPT-OSS 20B, then GLM 5.3 Flash.
Remote-first and individual model selection are available in Advanced settings.
Use synthetic or sanitized resumes for demonstrations.

## Development

Install uv, then run `uv sync --locked` and `uv run --locked python -m pytest -q`.
Python 3.11 and dependencies are recorded in `.python-version` and `uv.lock`.
Copy `.env.example` to `.env` and set `HF_TOKEN` to run `uv run app.py`.
The UI listens on all interfaces on port 7860 without authentication.

## Deployment

Deployment tooling lives in the sibling [case study 2 repository](https://github.com/derickhoward09/ML-OPS-Project-2/tree/alexander-dev/scripts).
The deployment reads the private, untracked `.hf.env` file in case study 2
(containing a bare token or `HF_TOKEN=...`). A root `hf-token` file here can
also be used by setting `TOKEN_FILE`. The token needs inference-provider permissions. Run
`scripts/deploy_resumelens.sh` from case study 2; it generates the deployed
app's `.env`. Do not commit token files or paste credentials into chat.

The app calls a CPU-only llama.cpp router at `LOCAL_BASE_URL` (default
`http://127.0.0.1:8080`). The deployment keeps one model resident, with two
threads and a 4096-token context. Qwen uses its official Q8_0 GGUF; LFM2.5
1.2B uses the official QAD Q4_0 GGUF. No CUDA, PyTorch, Transformers, Spaces,
or OAuth is needed.

Configuration: `HF_TOKEN` is required at startup; `LOCAL_BASE_URL`,
`LOCAL_CONTEXT_SIZE` (4096), `INFERENCE_TIMEOUT` (300 seconds), and the four
`LOCAL_PRIMARY_MODEL`, `LOCAL_BACKUP_MODEL`, `REMOTE_PRIMARY_MODEL`,
`REMOTE_BACKUP_MODEL` IDs can be overridden. Local IDs must match router presets;
context settings must match the runtime. The active model reuses matching prompt prefixes to reduce CPU work.
Requests are serialized, with eight
queued requests maximum. Response tokens default to 2,048. Input is not silently
truncated; automatic failover may use remote inference for oversized prompts.

CI runs tests only; this branch does not deploy to Hugging Face Spaces.
