@echo off
echo 🎯 Starting Desktop VRI-GazeNet System...
echo =========================================

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

echo ✅ Starting desktop gaze detection...
echo 📷 Using camera ID: 8 (connected desktop camera)
echo Controls:
echo - Press 'q' to quit
echo - Press 's' to save screenshot
echo - Show your LEFT hand for detection
echo.

REM Run the desktop application
python webcam_desktop.py --snapshot models/VRI.pkl --gpu cpu --cam 8 --arrow-length 200 