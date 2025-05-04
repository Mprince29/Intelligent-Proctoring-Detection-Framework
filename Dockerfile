FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libx11-dev \
    libatlas-base-dev \
    libgtk-3-dev \
    libboost-python-dev \
    python3-dev \
    gfortran \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dlib separately with optimized settings
RUN pip install --no-binary :all: dlib==19.24.0

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Command to run
CMD ["gunicorn", "app:app"]