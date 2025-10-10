# Dockerfile (repo root) - use python:3.11-slim
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install required system packages (OCR + PDF)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    ghostscript \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    default-jre \
    curl \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements (root-level requirements.txt expected)
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r /app/requirements.txt

# Copy repository
COPY . /app

EXPOSE 8000

# Use the root-level main:app so Render or CLI uvicorn main:app works
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}"]








