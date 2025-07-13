# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the service code
COPY . /app/src

# Expose the port your service listens on (change if different)
EXPOSE 2016

# Default command to run the service
CMD ["python", "src/app.py"]