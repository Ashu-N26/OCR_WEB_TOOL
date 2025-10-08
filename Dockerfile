# Use a supported Python base
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install OS-level deps required by OCR/PDF/image libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    wget \
    git \
    default-jre \                       # (optional) for tabula-py if later used
    poppler-utils \                     # pdftoppm used by pdf2image
    libpoppler-cpp-dev \
    tesseract-ocr \                     # Tesseract binary for pytesseract
    tesseract-ocr-eng \                 # English language pack (add others as needed)
    libtesseract-dev \
    libleptonica-dev \
    ghostscript \                       # used by some PDF workflows
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libopenjp2-7-dev \
    libtiff5-dev \
    libgl1-mesa-glx \                   # libGL required by opencv headless in some images
    libglib2.0-0 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (cache friendly)
COPY backend/requirements.txt ./requirements.txt

# Upgrade pip and install Python deps
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r ./requirements.txt

# Copy app code (backend + frontend)
COPY backend/ ./backend
COPY frontend/ ./frontend

# expose (Render will route to $PORT)
EXPOSE 8000

# Use Render's $PORT env when available; default to 8000 locally
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]





