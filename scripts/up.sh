#!/usr/bin/env bash
# Build and start the stack. If NGROK_AUTHTOKEN is set in .env, wait for the
# public tunnel, point Grafana at it, and record the public URLs.
set -Eeuo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

if [[ ! -f .env ]]; then
  echo "ERROR: copy .env.example to .env and fill it in first." >&2
  exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a

gateway_port=${PUBLIC_GATEWAY_PORT:-2100}
if [[ -n "${NGROK_AUTHTOKEN:-}" && -n "${NGROK_DOMAIN:-}" ]]; then
  export PUBLIC_BASE_URL="https://${NGROK_DOMAIN}"
fi

docker compose up -d --build
mkdir -p evidence

if [[ -z "${NGROK_AUTHTOKEN:-}" ]]; then
  host=$(hostname -I 2>/dev/null | awk '{print $1}')
  echo
  echo "ngrok disabled (NGROK_AUTHTOKEN is blank). Stack is available on the VM ports:"
  echo "  Gateway:    http://${host:-localhost}:${gateway_port}/"
  echo "  API:        http://${host:-localhost}:${gateway_port}/api/"
  echo "  Local:      http://${host:-localhost}:${gateway_port}/local/"
  echo "  Grafana:    http://${host:-localhost}:${gateway_port}/grafana/"
  echo "  Prometheus: http://${host:-localhost}:${PROMETHEUS_PORT:-2103}/"
  exit 0
fi

public_url=""
for _ in {1..30}; do
  public_url=$(curl --silent --max-time 2 "http://127.0.0.1:${NGROK_API_PORT:-2109}/api/tunnels" \
    | python3 -c 'import json,sys; t=json.load(sys.stdin).get("tunnels",[]); print(next((x["public_url"] for x in t if x["public_url"].startswith("https://")), ""))' \
    2>/dev/null || true)
  [[ -n "$public_url" ]] && break
  sleep 2
done

if [[ -z "$public_url" ]]; then
  echo "ERROR: ngrok did not publish a URL. Inspect: docker compose logs ngrok" >&2
  exit 1
fi

if [[ "${PUBLIC_BASE_URL:-}" != "$public_url" ]]; then
  # Recreate only Grafana so redirects and asset URLs use the public address.
  PUBLIC_BASE_URL="$public_url" docker compose up -d --no-deps --force-recreate grafana
fi

cat >evidence/public_urls.txt <<URLS
Base URL: $public_url/
API product: $public_url/api/
Local product: $public_url/local/
Grafana: $public_url/grafana/
URLS
echo
cat evidence/public_urls.txt
