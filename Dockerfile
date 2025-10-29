# -----------------------------------------------
# Stage 1: Base image and system dependencies
# -----------------------------------------------
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Prevent Python from buffering output
ENV PYTHONUNBUFFERED=1

# Install system-level dependencies required for OCR and PDF processing
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------
# Stage 2: Copy and install backend dependencies
# -----------------------------------------------
# Copy only the requirements first (to optimize Docker caching)
COPY backend/requirements.txt ./backend/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# -----------------------------------------------
# Stage 3: Copy the entire project
# -----------------------------------------------
COPY . .

# -----------------------------------------------
# Stage 4: Expose FastAPI app port
# -----------------------------------------------
EXPOSE 10000

# -----------------------------------------------
# Stage 5: Start the FastAPI server
# -----------------------------------------------
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "10000"]
