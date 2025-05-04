#!/bin/bash
set -e

echo "===== Starting build process ====="

echo "Installing system dependencies for dlib and ML libraries..."
apt-get update
apt-get install -y --no-install-recommends \
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
  libswscale-dev

echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "Installing dlib separately with specific compilation settings..."
pip install --no-cache-dir dlib>=19.24.0,<19.25.0

echo "Installing remaining Python dependencies..."
pip install --no-cache-dir -r requirements.txt

echo "===== Build completed ====="