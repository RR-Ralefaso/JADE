import subprocess
import sys
import os

def check_and_install():
    """Check and install required packages"""
    required_packages = [
        'opencv-python>=4.8.0',
        'numpy>=1.24.0',
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'ultralytics>=8.0.0',
        'speechrecognition>=3.10.0',
        'pyttsx3>=2.90',
        'pyaudio>=0.2.11',
        'noisereduce>=1.0.0',
        'psutil>=5.9.0',
        'gputil>=1.4.0',
        'msgpack>=1.0.0',
        'scikit-learn>=1.3.0',
        'scikit-image>=0.21.0',
        'python-dotenv>=1.0.0',
        'sounddevice>=0.4.6',
        'boxmot>=10.0.0',
        'pyyaml>=6.0'
    ]
    
    print("📦 Checking/installing required packages...")
    
    for package in required_packages:
        package_name = package.split('>=')[0].split('[')[0]
        try:
            __import__(package_name.replace('-', '_'))
            print(f"✅ {package_name} already installed")
        except ImportError:
            print(f"⬇️  Installing {package_name}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ {package_name} installed successfully")
            except:
                print(f"❌ Failed to install {package_name}")
                print(f"   Try: pip install {package}")
    
    print("\n✅ Installation complete!")
    
    # Create directories
    directories = [
        'models', 'logs', 'voice_logs', 'reports', 'cache',
        'exports/images', 'exports/videos',
        'datasets/train/images', 'datasets/train/labels',
        'datasets/val/images', 'datasets/val/labels',
        'datasets/test/images', 'datasets/test/labels'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created: {directory}")
    
    # Download YOLO model
    print("\n📥 Downloading YOLO model...")
    try:
        from ultralytics import YOLO
        model = YOLO('yolo11n.pt')  # Standard model
        print("✅ Model downloaded")
    except Exception as e:
        print(f"❌ Model download failed: {e}")
        print("   Model will download on first run")
    
    # Create .env template
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write("""# JADE Configuration

# OpenAI API Key (Optional)
OPENAI_API_KEY=your_key_here_optional

# Camera Settings
# CAMERA_ID=0

# Voice Settings
# WAKE_WORD=hey jade
# VOICE_GENDER=female
""")
        print("📄 Created .env template")
    
    print("\n" + "="*60)
    print("🚀 JADE Setup Complete!")
    print("="*60)
    print("\nTo start JADE:")
    print("  python main.py")
    print("\nFor testing:")
    print("  python test_voice.py")
    print("="*60)

if __name__ == "__main__":
    check_and_install()