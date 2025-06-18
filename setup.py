from setuptools import setup, find_packages

setup(
    name="vri-gazenet",
    version="1.0.0",
    description="Real-time gaze estimation with hand detection and bisecting line analysis",
    author="VRI-UFPR",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0", 
        "opencv-python>=4.5.0",
        "mediapipe>=0.8.0",
        "numpy>=1.21.0",
        "Pillow>=8.0.0",
        "scipy>=1.7.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
) 