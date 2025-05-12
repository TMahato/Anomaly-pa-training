FROM python:3.10-slim

# Prevent Python from writing .pyc files to disk and from buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install necessary build tools with non-interactive mode
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    gcc \
    libatlas-base-dev \
    libopenblas-dev \
    liblapack-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create required directories and ensure they are writable
RUN mkdir -p /app/src /app/data /app/model && \
    chmod -R 777 /app/model

# Copy dependencies and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY main.py /app/src/main.py

# Set default command
CMD ["python", "/app/src/main.py"]
