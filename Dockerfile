FROM python:3.11-slim

WORKDIR /app

# Install system packages required by OCR/table libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    default-jre \
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
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (helps layer caching)
COPY backend/requirements.txt .

# Install Python deps
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy application code
COPY backend/ ./backend
COPY frontend/ ./frontend

EXPOSE 8000

# Use Render's PORT env var at runtime
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port $PORT"]



