# Base Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create app directories
WORKDIR /app
RUN mkdir -p /app/src /app/data /app/model

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY main.py /app/src/main.py

# Define entrypoint (used by AI Core)
CMD ["python", "/app/src/main.py"]
