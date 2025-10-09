# Dockerfile (repo root) - robust for varying repo layouts
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies for OCR & PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    ghostscript \
    libleptonica-dev \
    libtesseract-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libopenjp2-7-dev \
    libtiff5-dev \
    libgl1 \
    libglib2.0-0 \
    default-jre \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire repo into the image (safe and simple)
COPY . /app

# Install Python dependencies from the first requirements file that exists:
# Priority: backend/requirements.txt -> requirements.txt (root)
RUN python -m pip install --upgrade pip setuptools wheel && \
    if [ -f backend/requirements.txt ]; then pip install -r backend/requirements.txt; \
    elif [ -f requirements.txt ]; then pip install -r requirements.txt; \
    else echo "ERROR: no requirements.txt found" && exit 1; fi

# Expose port (Render will supply $PORT)
EXPOSE 8000

# Start application
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]






