#!/usr/bin/env python3
"""Collect a compact, timestamped Prometheus snapshot for the report/LLM prompt."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


QUERIES = {
    "target_health": "up",
    "requests_15m": "sum by (product, result) (increase(resumelens_requests_total[15m]))",
    "p95_latency_15m": "histogram_quantile(0.95, sum by (le, product) (rate(resumelens_request_duration_seconds_bucket[15m])))",
    "mean_latency_15m": "sum by (product) (rate(resumelens_request_duration_seconds_sum[15m])) / clamp_min(sum by (product) (rate(resumelens_request_duration_seconds_count[15m])), 0.000001)",
    "model_calls_15m": "sum by (product, model, result) (increase(resumelens_model_calls_total[15m]))",
    "errors_15m": "sum by (product, error_type) (increase(resumelens_errors_total[15m]))",
    "cpu_percent": "100 * (1 - avg by (job) (rate(node_cpu_seconds_total{mode=\"idle\",job=~\"resumelens_.*_node\"}[5m])))",
    "memory_percent": "100 * (1 - node_memory_MemAvailable_bytes{job=~\"resumelens_.*_node\"} / node_memory_MemTotal_bytes{job=~\"resumelens_.*_node\"})",
}


def query(base_url: str, expression: str) -> list[dict]:
    url = f"{base_url.rstrip('/')}/api/v1/query?{urlencode({'query': expression})}"
    with urlopen(url, timeout=10) as response:
        payload = json.load(response)
    if payload.get("status") != "success":
        raise RuntimeError(payload)
    return payload["data"]["result"]


def main() -> None:
    base_url = os.getenv("PROMETHEUS_URL", f"http://127.0.0.1:{os.getenv('PROMETHEUS_PORT', '2103')}")
    timestamp = datetime.now(timezone.utc)
    results = {name: query(base_url, expression) for name, expression in QUERIES.items()}
    document = {
        "collected_at_utc": timestamp.isoformat(),
        "prometheus_url": base_url,
        "window": "15 minutes unless the query says 5 minutes",
        "queries": QUERIES,
        "results": results,
    }

    evidence = Path(__file__).resolve().parents[1] / "evidence"
    evidence.mkdir(exist_ok=True)
    stem = f"monitoring-snapshot-{timestamp.strftime('%Y%m%dT%H%M%SZ')}"
    json_path = evidence / f"{stem}.json"
    md_path = evidence / f"{stem}.md"
    json_path.write_text(json.dumps(document, indent=2) + "\n")

    lines = [
        "# ResumeLens monitoring snapshot",
        "",
        f"Collected at: {document['collected_at_utc']}",
        f"Window: {document['window']}",
        "",
    ]
    for name, expression in QUERIES.items():
        lines += [f"## {name}", "", f"PromQL: `{expression}`", "", "```json", json.dumps(results[name], indent=2), "```", ""]
    md_path.write_text("\n".join(lines))
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
