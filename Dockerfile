FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY backend/requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy all code (backend + frontend)
COPY backend/ ./backend
COPY frontend/ ./frontend

# Expose Render's dynamic port
EXPOSE 8000

# Start FastAPI (serves frontend + API)
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port $PORT"]

