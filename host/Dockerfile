FROM python:3.12-slim-bookworm

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY shared /app/shared
COPY host /app/host

ENV PYTHONPATH=/app

CMD ["uvicorn", "host.main:app", "--host", "0.0.0.0", "--port", "8000"]
