#!/bin/bash

echo "🎯 Starting Desktop VRI-GazeNet System..."
echo "========================================="

# Check if virtual environment exists
if [ ! -d "gazenet_env" ]; then
    echo "❌ Virtual environment not found. Please run install.sh first."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source gazenet_env/bin/activate

# Check if model exists
if [ ! -f "models/VRI.pkl" ]; then
    echo "❌ Model file not found: models/VRI.pkl"
    echo "Please ensure the VRI.pkl file is in the models/ directory."
    exit 1
fi

echo "✅ Starting desktop gaze detection..."
echo "📷 Using camera ID: 8 (connected desktop camera)"
echo "Controls:"
echo "- Press 'q' to quit"
echo "- Press 's' to save screenshot"
echo "- Show your LEFT hand for detection"
echo ""

# Run the desktop application
python webcam_desktop.py --snapshot models/VRI.pkl --gpu cpu --cam 8 --arrow-length 200 