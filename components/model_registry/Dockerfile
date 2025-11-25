# Dockerfile for Model Registry Component
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY components/model_registry/model_registry.py .

# Create registry directory
RUN mkdir -p registry

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "model_registry.py"]

