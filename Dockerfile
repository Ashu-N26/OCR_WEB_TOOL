# Use lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PORT=8000

# Create and set working directory
WORKDIR /app

# Install system dependencies required for OCR + PDF
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

# Expose port for Render
EXPOSE $PORT

# Command to run app
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]




