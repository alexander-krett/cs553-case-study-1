# Start here - your first Docker session

Do these in order. Stop when a step fails and read the matching README section.

## Before the shared VM

- [ ] Give the professor your **public** SSH key.
- [ ] Claim ports **2100-2110** in the class Google Sheet.
- [ ] Create a Docker Hub account.
- [ ] Get an ngrok authtoken (and optionally a free static domain).
- [ ] Have a Hugging Face token with inference-provider permission. It is typed
      into the API product's web page, not into any file.

## On the shared VM

```bash
# 1. Enter the project directory after cloning your repository.
cd resumelens

# 2. Create the private configuration.
cp .env.example .env
nano .env          # DOCKERHUB_USERNAME, COMPOSE_PROJECT_NAME, GRAFANA_ADMIN_PASSWORD, NGROK_AUTHTOKEN
chmod 600 .env

# 3. Confirm Docker is available and check the resolved configuration.
docker version
docker compose version
docker compose config --quiet

# 4. Build, start, and publish (ngrok starts only if NGROK_AUTHTOKEN is set).
scripts/up.sh
docker compose ps

# 5. Validate both products and both kinds of monitoring.
scripts/validate_stack.sh
scripts/show_container_ips.sh
```

Do not share the full output of `docker compose config`. It contains the
Grafana password and the ngrok token.

## Check it works

- [ ] API product at `/api/` (enter your HF token in the sidebar).
- [ ] Local product at `/local/` (the first review downloads the model).
- [ ] Four Prometheus targets are UP at `http://<vm>:2103/targets`.
- [ ] The Grafana dashboard at `/grafana/` loads and shows both products.

## Public evidence

- [ ] Test the URLs in `evidence/public_urls.txt` from a non-WPI network.
- [ ] Enter both product URLs and both product-container IPs in the class sheet.
- [ ] Take the screenshots listed in `docs/RUBRIC_CHECKLIST.md`.
- [ ] Generate real traffic, collect a monitoring snapshot, and complete the LLM
      analysis plus your own human action decision.
- [ ] Push the application image to Docker Hub and record its tag/digest.

Read `README.md` for the explanation and exact commands behind each step.
