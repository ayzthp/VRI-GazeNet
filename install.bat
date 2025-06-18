@echo off
echo 🚀 Installing VRI-GazeNet System...
echo ==================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.7+ first.
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set python_version=%%i
echo ✅ Python %python_version% detected

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv gazenet_env

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call gazenet_env\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📚 Installing dependencies...
pip install -r requirements.txt

echo.
echo ✅ Installation complete!
echo ==================================
echo To run the system:
echo 1. Activate the environment: gazenet_env\Scripts\activate.bat
echo 2. Run the webcam script: python webcam_face_detection.py --snapshot models/VRI.pkl --gpu cpu
echo.
echo For video recording:
echo python webcam_face_detection.py --snapshot models/VRI.pkl --gpu cpu --save
echo.
echo To deactivate: deactivate
pause 