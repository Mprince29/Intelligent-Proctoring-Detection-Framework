#!/bin/bash
set -e

# Install system dependencies
apt-get update
apt-get install -y cmake libx11-dev libatlas-base-dev libgtk-3-dev libboost-python-dev

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt