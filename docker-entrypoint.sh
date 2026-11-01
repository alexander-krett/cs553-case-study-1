#!/usr/bin/env bash
set -Eeuo pipefail

# The systemd collector needs D-Bus, which containers do not have.
prometheus-node-exporter --web.listen-address=":9100" --no-collector.systemd &
exporter_pid=$!

python /opt/resumelens/app.py &
app_pid=$!

shutdown() {
  kill -TERM "$app_pid" "$exporter_pid" 2>/dev/null || true
  wait "$app_pid" "$exporter_pid" 2>/dev/null || true
}
trap shutdown INT TERM EXIT

# Exit the container if either required process exits.
wait -n "$app_pid" "$exporter_pid"
