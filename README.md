# ResumeLens AI — Case Study 3 (Docker + monitoring)

ResumeLens is a Gradio resume-review app. For Case Study 3 it ships as **two
products built from one image**, each in its own container on the shared VM:

| Product | `PRODUCT_MODE` | Models (primary → backup) | HF token |
|---|---|---|---|
| API-based | `api` | `openai/gpt-oss-20b` → `zai-org/GLM-5.3-Flash` (Hugging Face Inference API) | Typed into the web UI per request |
| Locally executed | `local` | Qwen3 0.6B Q4_K_M (`unsloth/Qwen3-0.6B-GGUF`) → LFM2.5 1.2B QAD-Q4_0 (`LiquidAI/LFM2.5-1.2B-Instruct-GGUF`), run with llama.cpp on the container CPU | Not needed |

Both containers also run Prometheus Node Exporter (container port 9100) and
expose application metrics with `prometheus_client` (container port 8000).
Prometheus, Grafana, an nginx gateway and an optional ngrok tunnel run as
separate containers. Use synthetic or sanitized resumes for demonstrations.

## Architecture and ports

The shared VM only allows host ports **2100–2110**. Container ports stay standard.

```
internet ──ngrok──▶ gateway:8080 (host 2100) ─┬─ /api/     ▶ api_product:7860
                                              ├─ /local/   ▶ local_product:7860
                                              └─ /grafana/ ▶ grafana:3000
prometheus:9090 (host 2103) scrapes api_product:{8000,9100}, local_product:{8000,9100}
grafana reads prometheus over the compose network
```

| Host port | Container | Purpose |
|---|---|---|
| 2100 | gateway → 8080 | Landing page plus `/api/`, `/local/` and `/grafana/`. The ngrok target. |
| 2101 | api_product → 7860 | API product UI (direct) |
| 2102 | local_product → 7860 | Local product UI (direct) |
| 2103 | prometheus → 9090 | Prometheus UI / query API |
| 2104 | grafana → 3000 | Grafana (direct; served under `/grafana/`) |
| 2105 / 2106 | api_product → 8000 / 9100 | Python metrics / Node Exporter |
| 2107 / 2108 | local_product → 8000 / 9100 | Python metrics / Node Exporter |
| 2109 | ngrok → 4040 | ngrok inspector API (127.0.0.1 only) |
| 2110 | — | Spare |

Each value can be overridden in `.env`. `BIND_ADDRESS` (default `0.0.0.0`)
controls whether the ports are reachable from the VM network or only from
the VM itself (`127.0.0.1`).

## Deploying on the shared VM

```bash
git clone <this repository> resumelens && cd resumelens
cp .env.example .env && chmod 600 .env
nano .env        # DOCKERHUB_USERNAME, COMPOSE_PROJECT_NAME, GRAFANA_ADMIN_PASSWORD, NGROK_AUTHTOKEN
scripts/up.sh    # docker compose up -d --build, then ngrok URL discovery
scripts/validate_stack.sh
scripts/show_container_ips.sh
```

`scripts/up.sh` builds the image, starts all services, and prints the
URLs. When `NGROK_AUTHTOKEN` is set, it also writes them to
`evidence/public_urls.txt`. The first local review downloads the models into
the `model_cache` volume. The primary model is preloaded in the background at
startup.

Useful commands: `docker compose ps`, `docker compose logs -f local_product`,
`docker compose down` (add `-v` to also delete the model cache and metrics
history).

From a laptop, an SSH tunnel reaches services that are bound to 127.0.0.1:

```bash
ssh -L 2100:127.0.0.1:2100 -L 2103:127.0.0.1:2103 <user>@<vm-host>
# then open http://localhost:2100/ and http://localhost:2103/targets
```

### Publishing with ngrok

The compose file has an `ngrok` service (official `ngrok/ngrok` image) that
tunnels to `gateway:8080`. One tunnel serves both products and Grafana:

1. Copy your authtoken from the ngrok dashboard into `NGROK_AUTHTOKEN` in `.env`.
2. Optionally set `NGROK_DOMAIN` to your free static domain so the URL stays
   the same across restarts.
3. Run `scripts/up.sh`. Public URLs are `<url>/api/`, `<url>/local/` and
   `<url>/grafana/`.

If `NGROK_AUTHTOKEN` is blank, the ngrok container logs
`ngrok disabled: NGROK_AUTHTOKEN not set` and exits cleanly. Stop the tunnel
with `docker compose stop ngrok`. Without a static domain, `up.sh` recreates
Grafana with the discovered URL so its redirects use the public address.
Free ngrok URLs show a one-time browser warning page before the app loads.

### Docker Hub

```bash
docker login
docker compose build
docker compose push api_product   # both products share the same image tag
docker image inspect --format '{{index .RepoDigests 0}}' "$DOCKERHUB_USERNAME/resumelens-cs3:v1"
```

## Modifications made for Docker

- **One image, two products.** `PRODUCT_MODE` picks the API or local product.
  Each container only offers its own two models, with primary → backup
  failover. The Case Study 1/2 cross-mode failover was removed.
- **Local inference runs 4-bit (Q4) GGUF models in-process** with
  `llama-cpp-python`, installed from prebuilt CPU wheels so the image needs no
  compiler. Case Study 2 used an external llama.cpp router instead. On 2 CPUs,
  Qwen3 0.6B Q4_K_M generates about 10 tokens/s and LFM2.5 1.2B Q4_0 about 6
  tokens/s. Unquantized bf16 transformers managed only about 1 token/s.
  Only one model is kept resident, and the container is limited to 3 GB of
  memory and 2 CPUs (`LOCAL_THREADS` matches the CPU limit). `llama_cpp` is
  imported lazily, so the API product never loads it. Qwen3 receives the
  `/no_think` switch so it answers directly.
- **The HF token comes from the UI.** The API product has a password field in
  the sidebar. The token is used for that request only, is never stored or
  logged, and is redacted from error messages. No container receives `HF_TOKEN`.
- **Subpath serving.** `ROOT_PATH` (`/api`, `/local`) lets Gradio generate
  correct asset URLs behind the nginx gateway. Grafana uses
  `GF_SERVER_SERVE_FROM_SUB_PATH`.
- **Container hardening.** The app binds `0.0.0.0` inside the container, runs
  as a non-root user under `tini`, and has a healthcheck and
  `no-new-privileges`. Secrets live only in the untracked `.env`.

## Monitoring

### Node Exporter (system metrics, container port 9100)

The `Dockerfile` installs the Debian `prometheus-node-exporter` apt package.
`docker-entrypoint.sh` starts it with `--web.listen-address=":9100"` next to
the app, and the container exits if either process dies. Check it with
`docker compose exec local_product curl -s localhost:9100/metrics | head`.
Node Exporter inside a container reports mostly host-level CPU and memory,
not per-container cgroup usage.

### prometheus_client (product metrics, container port 8000)

`app.py` defines the metrics and calls `start_http_server(8000)` at startup.
Every metric carries a `product` label (`api` or `local`):

| Metric | Type | Meaning |
|---|---|---|
| `resumelens_requests_total{result}` | Counter | Reviews by outcome (`success` / `error`) |
| `resumelens_request_duration_seconds` | Histogram | End-to-end review latency |
| `resumelens_in_progress_requests` | Gauge | Reviews currently running |
| `resumelens_input_characters` | Histogram | Resume + job description size |
| `resumelens_output_characters` | Histogram | Size of the generated feedback |
| `resumelens_model_calls_total{model,result}` | Counter | Every model attempt, including backup failovers |
| `resumelens_errors_total{error_type}` | Counter | `validation`, `missing_token`, `timeout`, `inference` |
| `resumelens_build_info{version,primary_model,backup_model}` | Gauge (1) | Deployed configuration |

`prometheus/prometheus.yml` scrapes all four targets (`api_product:8000`,
`api_product:9100`, `local_product:8000`, `local_product:9100`) every 15 s.
Grafana is provisioned with the Prometheus data source and the
**ResumeLens Case Study 3 Overview** dashboard (`grafana/dashboards/`).

`scripts/collect_monitoring_snapshot.py` saves a timestamped JSON/Markdown
snapshot of the key queries to `evidence/` for LLM analysis.

## Class Google Sheet (manual)

- Port range: **2100–2110**.
- Container IPs: the output of `scripts/show_container_ips.sh` (api_product, local_product).
- Public URLs: `evidence/public_urls.txt` (`/api/` and `/local/`).

## Development

Install uv, then run `uv sync --locked` and `uv run --locked python -m pytest -q`.
To run the local product outside Docker, run `uv sync --locked --group local`,
then `PRODUCT_MODE=local uv run app.py`. To run the API product, use
`PRODUCT_MODE=api uv run app.py` and enter a token in the UI. The UI listens
on port 7860, and metrics on `METRICS_PORT` (8000).
