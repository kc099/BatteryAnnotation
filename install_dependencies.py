#!/usr/bin/env python3
"""
Installation script for Battery Quality Inspection System

This script checks and installs all required dependencies.
"""

import subprocess
import sys
import importlib
import platform

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required!")
        return False
    
    print("✅ Python version OK")
    return True

def check_gpu():
    """Check GPU availability"""
    print("\n🖥️  GPU Check:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA Available: {torch.cuda.get_device_name()}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   PyTorch Version: {torch.__version__}")
        else:
            print("⚠️  CUDA not available - will use CPU")
    except ImportError:
        print("❌ PyTorch not installed")

def install_package(package_name):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def check_and_install_requirements():
    """Check and install all requirements"""
    print("\n📦 Checking dependencies...")
    
    # Core requirements (simplified list)
    core_requirements = [
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "pytorch-lightning>=2.0.0",
        "opencv-python>=4.8.0",
        "Pillow>=9.5.0",
        "albumentations>=1.3.0",
        "numpy>=1.24.0",
        "shapely>=2.0.0",
        "matplotlib>=3.6.0",
        "tqdm>=4.65.0"
    ]
    
    # Test basic imports
    test_imports = {
        'torch': 'torch',
        'torchvision': 'torchvision', 
        'pytorch_lightning': 'pytorch_lightning',
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'albumentations': 'albumentations',
        'numpy': 'numpy',
        'shapely': 'shapely',
        'matplotlib': 'matplotlib',
        'tqdm': 'tqdm'
    }
    
    missing = []
    installed = []
    
    for import_name, package_name in test_imports.items():
        try:
            importlib.import_module(import_name)
            print(f"✅ {package_name}")
            installed.append(package_name)
        except ImportError:
            print(f"❌ {package_name} - Missing")
            missing.append(package_name)
    
    if missing:
        print(f"\n🔧 Installing {len(missing)} missing packages...")
        
        for package in missing:
            print(f"Installing {package}...")
            if install_package(package):
                print(f"   ✅ {package} installed")
            else:
                print(f"   ❌ {package} installation failed")
    
    print(f"\n📊 Summary:")
    print(f"   ✅ Installed: {len(installed)}")
    print(f"   ❌ Missing: {len(missing)}")

def install_from_requirements():
    """Install from requirements.txt if available"""
    import os
    
    req_file = "requirements.txt"
    if os.path.exists(req_file):
        print(f"\n📄 Installing from {req_file}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", req_file
            ])
            print("✅ Requirements installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Installation failed: {e}")
    else:
        print(f"⚠️  {req_file} not found")

def test_installation():
    """Test the installation by importing key modules"""
    print("\n🧪 Testing installation...")
    
    try:
        # Test core imports
        import torch
        import torchvision
        import pytorch_lightning as pl
        import cv2
        import numpy as np
        import albumentations as A
        
        print("✅ Core dependencies OK")
        
        # Test GPU
        if torch.cuda.is_available():
            print(f"✅ GPU ready: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  GPU not available")
        
        # Test our modules (if available)
        try:
            from model import HierarchicalQualityModel
            from dataset import ComponentQualityDataset
            print("✅ Custom modules OK")
        except ImportError as e:
            print(f"⚠️  Custom modules: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def main():
    """Main installation function"""
    print("🔋 BATTERY QUALITY INSPECTION - DEPENDENCY INSTALLER")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check platform
    print(f"\n💻 Platform: {platform.system()} {platform.release()}")
    
    # Option 1: Install from requirements.txt
    install_from_requirements()
    
    # Option 2: Check individual packages
    check_and_install_requirements()
    
    # Check GPU
    check_gpu()
    
    # Test installation
    test_installation()
    
    print("\n🎉 Installation complete!")
    print("\nNext steps:")
    print("1. Run: python simple_debug.py")
    print("2. Or run: python debug_pipeline.py")
    print("3. For training: python example_usage.py")

if __name__ == "__main__":
    main() 