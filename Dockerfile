# -------------------------------
# Stage 1: Base Environment
# -------------------------------
FROM python:3.10-slim

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    ghostscript \
    libglib2.0-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    default-jre \
    curl \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# -------------------------------
# Stage 2: Application Setup
# -------------------------------
WORKDIR /app

# If your requirements.txt is in root (not backend/)
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Copy all your project files (backend + frontend if needed)
COPY . /app

# Expose port for FastAPI
EXPOSE 8080

# -------------------------------
# Stage 3: Run FastAPI Server
# -------------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]







