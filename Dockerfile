# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the service code
COPY . .

# Expose the port your service listens on (change if different)
EXPOSE 2016

# Default command to run the service
CMD ["gunicorn", "--bind", "0.0.0.0:2016", "--workers", "2", "--preload", "--timeout", "120", "--keep-alive", "60", "--worker-class", "gevent", "src.app:app"]