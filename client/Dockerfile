FROM python:3.11-slim-bookworm AS client-base

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY shared /app/shared
COPY client /app/client

ENV PYTHONPATH=/app

CMD ["uvicorn", "client.main:app", "--host", "0.0.0.0", "--port", "8001"]

FROM client-base AS client

FROM client-base AS client-ml

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CC=/usr/bin/gcc

COPY tests/__init__.py /app/tests/__init__.py
COPY tests/test_client_knowledge.py /app/tests/test_client_knowledge.py
COPY tests/test_client_real_model.py /app/tests/test_client_real_model.py
COPY tests/test_fedmkt_selection.py /app/tests/test_fedmkt_selection.py
COPY tests/test_fedmkt_sparse_targets.py /app/tests/test_fedmkt_sparse_targets.py
COPY tests/test_fedmkt_token_alignment.py /app/tests/test_fedmkt_token_alignment.py
COPY tests/test_fedmkt_trainer.py /app/tests/test_fedmkt_trainer.py
