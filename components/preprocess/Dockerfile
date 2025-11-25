# Dockerfile for Data Preprocessing Component
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
COPY components/preprocess/preprocess.py .

# Create data directories
RUN mkdir -p data processed_data

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "preprocess.py"]

