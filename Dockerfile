# Dockerfile (repo root) - robust: copy repo then install whichever requirements file exists
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System packages (OCR + PDF processing + minimal libs required by opencv)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    poppler-utils \
    ghostscript \
    libleptonica-dev \
    libtesseract-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libopenjp2-7-dev \
    libtiff5-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    default-jre \
    curl \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy whole repo (makes build robust to file/paths)
COPY . /app

# Upgrade pip & install whichever requirements file exists
RUN python -m pip install --upgrade pip setuptools wheel && \
    if [ -f backend/requirements.txt ]; then \
        pip install --no-cache-dir -r backend/requirements.txt; \
    elif [ -f requirements.txt ]; then \
        pip install --no-cache-dir -r requirements.txt; \
    else \
        echo "ERROR: No requirements.txt found at backend/requirements.txt or requirements.txt" && exit 1; \
    fi

# Expose port used by uvicorn (Render sets PORT environment variable)
EXPOSE 8000

# Start the app. This points to backend.main:app (adjustable)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}"]










