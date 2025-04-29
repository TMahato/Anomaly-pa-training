FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables to reduce Python bytecode generation and ensure logs aren't buffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install minimal dependencies in stages to better utilize caching
# Start with core packages
RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    pandas>=1.3.5 \
    scikit-learn>=1.0.2 \
    joblib>=1.1.0 \
    boto3>=1.26.0 \
    python-dotenv>=0.21.0 \
    pyarrow>=7.0.0

# Install PyCaret separately with only required components
RUN pip install --no-cache-dir \
    pycaret[core,anomaly]>=2.3.10

# Create directories
RUN mkdir -p /app/src /app/model

# Copy application code
COPY main.py /app/src/

# Set the entry point
ENTRYPOINT ["python", "/app/src/main.py"]