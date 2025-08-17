#!/bin/bash
# FedTinyProp Raspberry Pi Setup Script (Fixed version)
# This script sets up the environment for running FedTinyProp on Raspberry Pi

# Exit on error
set -e

echo "===== FedTinyProp Raspberry Pi Setup ====="

# Update system packages
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-venv \
    python3-wheel \
    build-essential \
    libatlas-base-dev \
    gfortran \
    git \
    htop \
    iotop

# Create Python virtual environment
echo "Setting up Python virtual environment..."
python3 -m venv fedtinyprop_env

# Activate virtual environment
echo "Activating virtual environment..."
source fedtinyprop_env/bin/activate

# Install PyTorch (CPU version for Raspberry Pi)
echo "Installing PyTorch (CPU version)..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
echo "Installing other dependencies..."
pip install -r rpi/requirements.txt

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p data
mkdir -p results

echo "===== Setup Complete! ====="
echo "To activate the environment: source fedtinyprop_env/bin/activate"
echo "To run benchmarks: python rpi/benchmark.py"
