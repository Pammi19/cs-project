# Use lightweight Python 3.10 runtime base
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition and install packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and pre-trained models
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Configure Streamlit environment settings
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Launch application
CMD ["streamlit", "run", "app.py"]