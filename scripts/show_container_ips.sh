#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

printf '%-22s %s\n' 'SERVICE' 'CONTAINER IP'
for service in api_product local_product prometheus grafana gateway; do
  container_id=$(docker compose ps -q "$service")
  if [[ -z "$container_id" ]]; then
    printf '%-22s %s\n' "$service" 'NOT RUNNING'
    continue
  fi
  ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$container_id")
  printf '%-22s %s\n' "$service" "$ip"
done
