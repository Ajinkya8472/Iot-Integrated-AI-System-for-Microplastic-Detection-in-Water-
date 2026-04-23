#!/bin/bash
# =============================================================================
#  Raspberry Pi Setup Script
#  IoT-Integrated AI System for Microplastic Detection in Water
# =============================================================================
#
#  This script sets up the Raspberry Pi 4 environment for running
#  the microplastic detection system. Run this after a fresh Raspberry Pi OS install.
#
#  Usage: chmod +x setup_pi.sh && ./setup_pi.sh
# =============================================================================

set -e  # Exit on any error

echo "=============================================="
echo "  🔬 Microplastic Detection - Pi Setup"
echo "=============================================="

# 1. System update
echo "[1/6] Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# 2. Install system dependencies
echo "[2/6] Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-opencv \
    libatlas-base-dev \
    libopenblas-dev \
    libhdf5-dev \
    libjpeg-dev \
    libpng-dev \
    git

# 3. Enable camera interface
echo "[3/6] Enabling camera interface..."
if ! grep -q "start_x=1" /boot/config.txt 2>/dev/null; then
    echo "start_x=1" | sudo tee -a /boot/config.txt
    echo "gpu_mem=128" | sudo tee -a /boot/config.txt
    echo "[INFO] Camera interface enabled. Reboot required after setup."
fi

# 4. Create virtual environment
echo "[4/6] Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 5. Install Python dependencies
echo "[5/6] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install picamera2

# 6. Verify installation
echo "[6/6] Verifying installation..."
python3 -c "
import cv2
import numpy as np
from ultralytics import YOLO
print('[OK] OpenCV version:', cv2.__version__)
print('[OK] NumPy version:', np.__version__)
print('[OK] Ultralytics YOLO loaded successfully')
print()
print('Setup complete! Run detection with:')
print('  python src/capture.py --weights results/weights/best.pt')
"

echo ""
echo "=============================================="
echo "  ✅ Setup Complete!"
echo "=============================================="
echo ""
echo "  Next steps:"
echo "  1. Place your trained weights in results/weights/best.pt"
echo "  2. Reboot if camera was just enabled: sudo reboot"
echo "  3. Start detection: python src/capture.py --weights results/weights/best.pt"
echo ""
