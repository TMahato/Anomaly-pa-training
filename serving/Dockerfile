# Base Python image
FROM python:3.12
ENV LANG C.UTF-8

# Create app directory and copy files
WORKDIR /app
COPY main.py .
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port (optional but good practice)
EXPOSE 9001

# Start the Flask app
CMD ["python", "main.py"]
