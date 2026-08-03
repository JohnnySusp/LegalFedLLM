FROM python:3.11-slim-bookworm

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY shared /app/shared
COPY client /app/client

ENV PYTHONPATH=/app

CMD ["uvicorn", "client.main:app", "--host", "0.0.0.0", "--port", "8001"]
