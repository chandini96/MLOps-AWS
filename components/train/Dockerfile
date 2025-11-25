# Dockerfile for Model Training Component (AWS SageMaker Compatible)
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt/ml

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy training script
COPY components/train/train.py /opt/program/train.py

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/program

# Create SageMaker directories
RUN mkdir -p /opt/ml/input/data/train \
    /opt/ml/model \
    /opt/ml/output

# Default command
ENTRYPOINT ["python", "/opt/program/train.py"]

