# Case Study 3 rubric checklist (100 points)

## 1. Docker deployment - 25 points

- [ ] API-backed ResumeLens works at `/api/` and a screenshot/test is saved. (10)
- [ ] Local CPU ResumeLens works at `/local/` and a screenshot/test is saved. (10)
- [ ] Group's port range (2100-2110) is entered in the class Google Sheet. (5)
- [ ] Docker Hub image/tag and commit hash are recorded.

## 2. Monitoring - 35 points

- [ ] Node Exporter runs in both product containers on container port 9100. (10)
- [ ] Each product exposes 4-10 custom Python metrics on container port 8000. (20)
- [ ] Prometheus Targets page shows all four scrape jobs UP.
- [ ] API and local container IPs from `scripts/show_container_ips.sh` are entered in the class sheet. (5)

## 3. Global product access - 10 points

- [ ] ngrok tunnel is running (`ngrok` compose service, started by `scripts/up.sh`). (5)
- [ ] `/api/` and `/local/` world-readable URLs are tested from a non-WPI network.
- [ ] Both product URLs are entered in the class Google Sheet. (5)

## 4. Grafana - 10 points

- [ ] Grafana loads at `/grafana/` through the public tunnel.
- [ ] Dashboard shows API/local traffic, success fraction, latency, CPU, memory, errors, and target health.
- [ ] Report includes readable screenshots with time range and panel legends. (10)

## 5. Report - 20 points

- [ ] Group member names. (1)
- [ ] Docker setup and modifications from Case Studies 1/2. (4)
- [ ] Prometheus and Node Exporter setup for both products. (3)
- [ ] ngrok setup and public URLs. (3)
- [ ] Challenges, lessons, and future improvements. (3)
- [ ] Complete LLM prompt and response analyzing system and product metrics. (3)
- [ ] Independent human action decision with accepted/rejected recommendations and metric-based justification. (3)

## Evidence bundle

- [ ] `docker compose ps`
- [ ] `scripts/validate_stack.sh`
- [ ] `scripts/show_container_ips.sh`
- [ ] Prometheus Targets screenshot
- [ ] Grafana overview screenshot after real requests
- [ ] API and local UI screenshots
- [ ] ngrok URLs and off-network tests
- [ ] timestamped monitoring snapshot, LLM prompt/response, and human decision
- [ ] Docker image name/tag/digest and Git commit
