FROM python:3.11-slim-bookworm AS host-base

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY shared /app/shared
COPY host /app/host

ENV PYTHONPATH=/app

CMD ["uvicorn", "host.main:app", "--host", "0.0.0.0", "--port", "8002"]

FROM host-base AS host-ml

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CC=/usr/bin/gcc

COPY tests/__init__.py /app/tests/__init__.py
COPY tests/test_host_training.py /app/tests/test_host_training.py

FROM host-base AS host
