# Forzar ARM64 explícitamente
FROM --platform=linux/arm64 python:3.11-slim

WORKDIR /app

ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV HTTP_PROXY=$HTTP_PROXY \
    HTTPS_PROXY=$HTTPS_PROXY \
    http_proxy=$HTTP_PROXY \
    https_proxy=$HTTPS_PROXY \
    OMP_NUM_THREADS=$CEREBRO_OMP_NUM_THREADS \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Dependencias del sistema para ARM64
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY cerebro/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY cerebro ./cerebro

EXPOSE 9000

# Quita --reload en producción
CMD ["uvicorn", "cerebro.main:app", "--host", "0.0.0.0", "--port", "9000", \
     "--workers", "4", "--loop", "uvloop"]