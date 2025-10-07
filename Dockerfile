# Use official Python slim image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy backend requirements and install dependencies
COPY backend/requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy backend code
COPY backend/ .

# Expose the port (Render assigns dynamically via $PORT)
EXPOSE 8000

# Start FastAPI using uvicorn and Render's assigned port
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
