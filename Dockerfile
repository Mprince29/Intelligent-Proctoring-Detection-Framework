FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies with specific versions known to work with dlib
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
    wget \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install dlib directly from GitHub with specific commit known to work (19.24.0)
RUN git clone --depth 1 https://github.com/davisking/dlib.git dlib-repo && \
    cd dlib-repo && \
    python setup.py install && \
    cd .. && \
    rm -rf dlib-repo

# First install face-recognition-models which doesn't depend on dlib
RUN pip install --no-cache-dir face-recognition-models

# Then install face-recognition
RUN pip install --no-cache-dir face-recognition

# Copy requirements but remove face-recognition since we've already installed it
COPY requirements.txt .
RUN grep -v "face-recognition" requirements.txt > requirements_mod.txt

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements_mod.txt

# Copy project
COPY . .

# Command to run
CMD ["gunicorn", "app:app"]