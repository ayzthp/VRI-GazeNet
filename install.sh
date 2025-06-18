#!/bin/bash

echo "🚀 Installing VRI-GazeNet System..."
echo "=================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.7+ first."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python $python_version detected"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv gazenet_env

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source gazenet_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Installation complete!"
echo "=================================="
echo "To run the system:"
echo "1. Activate the environment: source gazenet_env/bin/activate"
echo "2. Run the webcam script: python webcam_face_detection.py --snapshot models/VRI.pkl --gpu cpu"
echo ""
echo "For video recording:"
echo "python webcam_face_detection.py --snapshot models/VRI.pkl --gpu cpu --save"
echo ""
echo "To deactivate: deactivate" 