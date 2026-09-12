# Production-grade, zero-dependency Spanda Enterprise Gateway
FROM python:3.11-slim AS builder

WORKDIR /app
COPY pyproject.toml README.md ./
COPY spanda/ ./spanda/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

FROM python:3.11-slim AS runner

# Create non-root system user for security compliance (SOC-2 / CIS)
RUN groupadd -r spanda && useradd -r -g spanda -s /bin/false spanda

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/spanda-gateway /usr/local/bin/spanda-gateway

USER spanda

ENV PYTHONUNBUFFERED=1
ENV SPANDA_PORT=8080
ENV SPANDA_UPSTREAM=https://api.openai.com/v1
ENV SPANDA_THRESHOLD=0.35
ENV SPANDA_BLOCK_MODE=0
ENV SPANDA_DEFAULT_K=3

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz')" || exit 1

ENTRYPOINT ["spanda-gateway"]
CMD ["--port", "8080"]
