#!/bin/bash

echo "🎯 Starting VRI-GazeNet System..."
echo "=================================="

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

echo "✅ Starting gaze detection..."
echo "Controls:"
echo "- Press 'q' to quit"
echo "- Press 's' to save screenshot"
echo "- Show your LEFT hand for detection"
echo ""

# Run the main application
python webcam_face_detection.py --snapshot models/VRI.pkl --gpu cpu --arrow-length 200 