# Dockerfile (repo root)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install OS-level dependencies required for OCR / PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    wget \
    git \
    pkg-config \
    poppler-utils \
    libpoppler-cpp-dev \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    ghostscript \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libopenjp2-7-dev \
    libtiff5-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install Python deps
COPY backend/requirements.txt ./requirements.txt
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r ./requirements.txt

# Copy application files
COPY backend/ ./backend
COPY frontend/ ./frontend

EXPOSE 8000

# Start FastAPI app (use Render's $PORT when present)
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]


