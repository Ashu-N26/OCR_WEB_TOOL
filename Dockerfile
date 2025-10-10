# Use official lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only backend dependencies first for faster caching
COPY backend/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project to the container
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Run FastAPI from inside the backend folder
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]









