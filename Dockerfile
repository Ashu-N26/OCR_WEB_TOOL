# ---------- Dockerfile (place at repo root) ----------
FROM python:3.10-slim

# set working dir
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install system dependencies required for PDF->image and OCR
# poppler-utils (pdftoppm), tesseract-ocr (binary), fonts, opencv libs
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    # add other lang packs if you need them, e.g. tesseract-ocr-por
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ghostscript \
    fonts-dejavu-core \
    fonts-liberation \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Copy only backend requirements first for layer caching
COPY backend/requirements.txt /app/backend/requirements.txt

# upgrade pip and install python deps
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy rest of the project
COPY . /app

# expose port used by uvicorn
EXPOSE 10000

# default command (Render overrides CMD if you put Docker command in UI)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "10000"]
# ----------------------------------------------------

