# VRI-GazeNet Installation Guide

## 🎯 Overview
VRI-GazeNet is a real-time gaze estimation system with hand detection and bisecting line analysis. It uses MediaPipe for face and hand detection, and a pre-trained neural network for gaze estimation.

## 📋 System Requirements

### Minimum Requirements:
- **Python**: 3.7 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **Camera**: Webcam for real-time detection
- **OS**: Windows 10+, macOS 10.14+, or Linux

### Recommended:
- **GPU**: CUDA-compatible GPU (optional, for faster processing)
- **RAM**: 16GB
- **Camera**: HD webcam (720p or higher)

## 🚀 Quick Installation

### Option 1: Automated Installation (Recommended)

#### For macOS/Linux:
```bash
chmod +x install.sh
./install.sh
```

#### For Windows:
```cmd
install.bat
```

### Option 2: Manual Installation

1. **Clone or download the project**
2. **Create virtual environment:**
   ```bash
   python -m venv gazenet_env
   ```

3. **Activate virtual environment:**
   - **macOS/Linux:**
     ```bash
     source gazenet_env/bin/activate
     ```
   - **Windows:**
     ```cmd
     gazenet_env\Scripts\activate.bat
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

### Basic Usage:
```bash
python webcam_face_detection.py --snapshot models/VRI.pkl --gpu cpu
```

### With Video Recording:
```bash
python webcam_face_detection.py --snapshot models/VRI.pkl --gpu cpu --save
```

### Advanced Options:
```bash
python webcam_face_detection.py \
  --snapshot models/VRI.pkl \
  --gpu cpu \
  --arrow-length 250 \
  --save
```

## 🎯 Features

### Real-time Detection:
- ✅ **Face Detection**: Green bounding box around detected face
- ✅ **Left Hand Detection**: Yellow bounding box around left hand
- ✅ **Gaze Estimation**: Red arrow from nose showing gaze direction
- ✅ **Gaze Strength**: Percentage-based focus measurement (0-100%)
- ✅ **Pupil Lines**: Blue lines from pupils to hand (when focused)
- ✅ **Bisecting Line**: Green line when perfect alignment detected

### Controls:
- **'q'**: Quit the application
- **'s'**: Save screenshot
- **Show LEFT hand**: Position your left hand in view for detection

### Visual Indicators:
- 🔴 **Red Arrow**: Gaze direction from nose
- 🟡 **Yellow Box**: Left hand detection
- 🔵 **Blue Lines**: Pupil-to-hand lines (when gaze strength > 60%)
- 🟢 **Green Line**: Perfect bisecting alignment
- 📊 **White Text**: Gaze strength, angles, and status

## 🔧 Troubleshooting

### Common Issues:

1. **"No face detected"**
   - Ensure good lighting
   - Face the camera directly
   - Check camera permissions

2. **"Show LEFT hand"**
   - Show your left hand to the camera
   - Keep hand open and visible
   - Ensure hand is in frame

3. **Low FPS**
   - Close other applications
   - Use CPU mode (--gpu cpu)
   - Reduce camera resolution

4. **Import errors**
   - Ensure virtual environment is activated
   - Reinstall requirements: `pip install -r requirements.txt`

### Performance Tips:
- Use good lighting for better detection
- Keep face and hand clearly visible
- Close unnecessary applications
- Use CPU mode if GPU is slow

## 📁 File Structure
```
GazeNet/
├── models/
│   └── VRI.pkl              # Pre-trained model
├── webcam_face_detection.py # Main application
├── requirements.txt         # Dependencies
├── install.sh              # macOS/Linux installer
├── install.bat             # Windows installer
└── README_INSTALL.md       # This file
```

## 🎬 Output Files
- **Screenshots**: `frame_[timestamp].jpg` (press 's')
- **Videos**: `output.mp4` (with --save flag)

## 🔄 Updates
To update the system:
1. Pull latest changes
2. Reactivate virtual environment
3. Run: `pip install -r requirements.txt --upgrade`

## 📞 Support
For issues or questions:
1. Check troubleshooting section
2. Ensure all requirements are met
3. Verify camera permissions
4. Check Python version compatibility

## 🎉 Ready to Use!
Once installed, you can start the system and begin real-time gaze detection with hand interaction analysis! 