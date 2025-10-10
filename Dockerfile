# ============================================================
# Dockerfile — OCR Web Tool (FastAPI + Hybrid OCR Extractor)
# Base: python:3.10-slim
# Optimized for Render deployment
# ============================================================

# ---------------------------
# 1️⃣ Base image and system deps
# ---------------------------
FROM python:3.10-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update apt, install OS-level dependencies
# - tesseract-ocr: OCR engine
# - poppler-utils: for pdf2image (pdftoppm)
# - ghostscript: fallback PDF rasterizer
# - libgl1, libsm6, libxext6, libxrender1: required for OpenCV headless
# - default-jre: required by some PDF parsing tools (failsafe)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils \
        ghostscript \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libglib2.0-0 \
        default-jre && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------------------------
# 2️⃣ Set working directory
# ---------------------------
WORKDIR /app

# ---------------------------
# 3️⃣ Copy dependency file & install Python packages
# ---------------------------
# COPY backend/requirements.txt ./   # <- earlier failed because backend/requirements.txt didn't exist
COPY requirements.txt ./

# Upgrade pip (to avoid resolver bugs)
RUN pip install --upgrade pip

# Install dependencies
RUN pip install -r requirements.txt

# ---------------------------
# 4️⃣ Copy full project
# ---------------------------
COPY . .

# ---------------------------
# 5️⃣ Expose port and set env
# ---------------------------
ENV PORT=8000
EXPOSE 8000

# ---------------------------
# 6️⃣ Command to start FastAPI app
# ---------------------------
# Uvicorn loads backend.main:app entrypoint
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
