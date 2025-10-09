# -----------------------------
# Base Image
# -----------------------------
FROM python:3.10-slim

# -----------------------------
# Environment Setup
# -----------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------
# Install System Dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    ghostscript \
    libglib2.0-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    default-jre \
    curl \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Set Work Directory
# -----------------------------
WORKDIR /app

# -----------------------------
# Copy Requirements and Install
# -----------------------------
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# -----------------------------
# Copy Full Application
# -----------------------------
COPY . /app

# -----------------------------
# Expose Port and Start Server
# -----------------------------
EXPOSE 8080
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8080}"]





