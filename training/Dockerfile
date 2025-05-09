FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
RUN mkdir -p /app/src /app/data /app/model

# Add build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libatlas-base-dev \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py /app/src/main.py

CMD ["python", "/app/src/main.py"]
