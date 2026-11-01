# LLM monitoring-analysis evidence

Run `scripts/collect_monitoring_snapshot.py` after generating both successful and
failed requests. Paste the resulting Markdown snapshot after this prompt. Keep the
complete prompt and response for the report.

## Prompt

You are reviewing a 15-minute Prometheus snapshot for two ResumeLens deployments:
an API-backed product and a local CPU product. Analyze only the supplied data.

1. Confirm whether every Prometheus target is up.
2. Compare request volume, success/failure counts, mean latency, and p95 latency.
3. Compare CPU and memory measurements, but state the limitations of Node Exporter
   running inside a Docker container.
4. Identify anomalies or missing evidence.
5. Recommend at most three actions, ordered by urgency.
6. Separate observations directly supported by metrics from inferences.

Do not invent thresholds, incidents, causes, or request counts. If a query is empty,
say that the snapshot does not contain enough evidence.

## Human decision to write yourself

- Which recommendations do we accept, reject, or defer?
- What exact metric values support that choice?
- Is an action needed now, or is continued observation more appropriate?
- What limitation of the snapshot could change the decision?
