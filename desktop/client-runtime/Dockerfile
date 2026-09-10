FROM python:3.11-slim-bookworm

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CC=/usr/bin/gcc
ENV PYTHONPATH=/app

COPY shared /app/shared
COPY client /app/client

CMD ["sh", "-c", "exec uvicorn client.main:app --host 127.0.0.1 --port ${CLIENT_AGENT_PORT:-8001}"]
