FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p /app/src /app/model

# Copy application code
COPY main.py /app/src/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set the entry point
ENTRYPOINT ["python", "/app/src/main.py"]