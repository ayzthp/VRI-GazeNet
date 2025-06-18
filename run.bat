@echo off
echo 🎯 Starting VRI-GazeNet System...
echo ==================================

REM Check if virtual environment exists
if not exist "gazenet_env" (
    echo ❌ Virtual environment not found. Please run install.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call gazenet_env\Scripts\activate.bat

REM Check if model exists
if not exist "models\VRI.pkl" (
    echo ❌ Model file not found: models\VRI.pkl
    echo Please ensure the VRI.pkl file is in the models\ directory.
    pause
    exit /b 1
)

echo ✅ Starting gaze detection...
echo Controls:
echo - Press 'q' to quit
echo - Press 's' to save screenshot
echo - Show your LEFT hand for detection
echo.

REM Run the main application
python webcam_face_detection.py --snapshot models/VRI.pkl --gpu cpu --arrow-length 200 