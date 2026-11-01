FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRADIO_SERVER_NAME=0.0.0.0

# Install the professor-required Node Exporter before Python dependencies so
# changes to requirements.txt can reuse this Docker layer.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        prometheus-node-exporter \
        tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/resumelens

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY app.py docker-entrypoint.sh ./
COPY src/ ./src/
RUN chmod 0755 /opt/resumelens/docker-entrypoint.sh \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin resumelens \
    && mkdir -p /home/resumelens/.cache/huggingface \
    && chown -R resumelens:resumelens /opt/resumelens /home/resumelens

USER resumelens

EXPOSE 7860 8000 9100

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=4 \
  CMD curl --fail --silent http://127.0.0.1:7860/ >/dev/null || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/opt/resumelens/docker-entrypoint.sh"]
