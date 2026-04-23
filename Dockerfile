FROM python:3.11-slim

WORKDIR /app

# System deps (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# App code — PYTHONPATH set below, so no `pip install -e .` needed
COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/src

EXPOSE 7860

# HF Spaces with Docker SDK expects the container to serve on port $PORT (7860)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
