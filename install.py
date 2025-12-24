import subprocess
import sys
import os
import time

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
        'pyyaml>=6.0',
        'matplotlib>=3.7.0',      # New: For graphs
        'seaborn>=0.12.0',        # New: For graphs
        'pandas>=2.0.0',          # New: For data analysis
        'Pillow>=10.0.0'          # New: For image processing
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
    
    # Create enhanced directories
    directories = [
        'models', 'logs', 'voice_logs', 'reports',
        'reports/plots', 'reports/sessions', 'reports/detector',
        'reports/training', 'reports/knowledge_base', 'reports/tests',
        'cache', 'exports', 'exports/screenshots', 'exports/videos',
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
    
    # Create enhanced .env template
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write("""# JADE Enhanced Configuration

# AI API Keys (Optional)
OPENAI_API_KEY=your_openai_key_here_optional
DEEPSEEK_API_KEY=your_deepseek_key_here_optional

# Camera Settings
# CAMERA_ID=0

# Voice Settings
# WAKE_WORD=hey jade
# VOICE_GENDER=female
# SPEAKING_RATE=180

# Display Settings
# THEME=dark
# ENHANCED_GUI=true
# SHOW_PERFORMANCE_GRAPHS=true
# SHOW_PERFORMANCE_OVERLAY=true

# Performance Settings
# TARGET_FPS=60
# FRAME_SKIP=2
# ENABLE_DEEP_ANALYSIS=true
""")
        print("📄 Created enhanced .env template")
    
    print("\n" + "="*60)
    print("🚀 JADE Enhanced Setup Complete!")
    print("="*60)
    print("\n✨ ENHANCED FEATURES:")
    print("  • Performance graphs and visualizations")
    print("  • Modern dark theme GUI")
    print("  • Session comparison dashboard")
    print("  • Automatic report generation")
    print("  • Enhanced object analysis")
    
    print("\n📊 PERFORMANCE VISUALIZATION:")
    print("  • FPS trends and analysis")
    print("  • Object detection statistics")
    print("  • Confidence distribution")
    print("  • Inference time monitoring")
    print("  • Session comparison")
    
    print("\nTo start JADE Enhanced:")
    print("  python main.py")
    
    print("\nFor testing:")
    print("  python test_voice.py")
    
    print("\nFor training custom models:")
    print("  python train_jade.py --setup")
    print("  python train_jade.py --train")
    
    print("\n📁 Reports will be saved to: reports/ directory")
    print("📈 Graphs will be saved to: reports/plots/ directory")
    print("="*60)

if __name__ == "__main__":
    check_and_install()