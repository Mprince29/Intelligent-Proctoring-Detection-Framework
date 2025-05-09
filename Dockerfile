# Build stage
FROM python:3.9-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    OPENCV_VIDEOIO_PRIORITY_MSMF=0

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
    wget \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install dlib with optimizations
RUN git clone --depth 1 https://github.com/davisking/dlib.git dlib-repo && \
    cd dlib-repo && \
    python setup.py install --yes USE_AVX_INSTRUCTIONS --yes DLIB_USE_CUDA && \
    cd .. && \
    rm -rf dlib-repo

# Install face-recognition packages
RUN pip install --no-cache-dir face-recognition-models face-recognition

# Copy and install requirements
COPY requirements.txt .
RUN grep -v "face-recognition\|dlib" requirements.txt > requirements_mod.txt && \
    pip install --no-cache-dir -r requirements_mod.txt

# Final stage
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    OPENCV_VIDEOIO_PRIORITY_MSMF=0

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libx11-6 \
    libatlas-base-dev \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

# Create and set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project files
COPY . .

# Set proper permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Expose port
EXPOSE 8000

# Set entry point and default command
ENTRYPOINT ["gunicorn"]
CMD ["--bind", "0.0.0.0:8000", "--workers", "4", "--timeout", "120", "app:app"]