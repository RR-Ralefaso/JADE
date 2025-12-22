import subprocess
import sys
import os

def check_and_install():
    """Check and install required packages"""
    required_packages = [
        'opencv-python',
        'numpy',
        'torch',
        'torchvision',
        'ultralytics',
        'speechrecognition',
        'pyttsx3',
        'pyaudio',
        'requests',
        'python-dotenv',
        'scikit-image',      # Added for GLCM features
        'scikit-learn',      # Added for KMeans clustering
        'pyyaml'            # Added for training config
    ]
    
    print("📦 Checking/installing required packages...")
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_').split('[')[0])
            print(f"✅ {package} already installed")
        except ImportError:
            print(f"⬇️  Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ {package} installed successfully")
            except:
                print(f"❌ Failed to install {package}")
                print(f"   Try: pip install {package}")
    
    print("\n✅ Installation complete!")
    
    # Create necessary directories
    directories = ['models', 'logs', 'voice_logs', 'train/images', 'train/labels', 
                   'val/images', 'val/labels', 'test/images', 'test/labels', 'reports']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Check for YOLO model
    model_path = 'models/yolo11n.pt'
    if not os.path.exists(model_path):
        print(f"\n⚠️  Model not found at {model_path}")
        print("   The model will be downloaded automatically on first run.")
    
    print("\n🎤 Voice System Check:")
    
    # Test speech recognition
    try:
        import speech_recognition as sr
        print("✅ SpeechRecognition: OK")
    except:
        print("❌ SpeechRecognition: Failed")
    
    # Test text-to-speech
    try:
        import pyttsx3
        engine = pyttsx3.init()
        print("✅ pyttsx3: OK")
    except:
        print("❌ pyttsx3: Failed")
    
    # Test pyaudio
    try:
        import pyaudio
        print("✅ pyaudio: OK")
    except:
        print("❌ pyaudio: Failed")
        print("   On Linux: sudo apt-get install python3-pyaudio")
        print("   On Mac: brew install portaudio")
        print("   On Windows: pip install pipwin then pipwin install pyaudio")
    
    print("\n🔧 Configuration Setup:")
    
    # Create .env template if not exists
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write("""# JADE Voice Assistant API Key
# Get key from: https://platform.openai.com/api-keys
# Note: OpenAI API key is optional for basic functionality

OPENAI_API_KEY=your_openai_key_here_optional

# Without OpenAI key, JADE will use built-in knowledge base
# for object analysis and basic conversation
""")
        print("📄 Created .env template file")
        print("ℹ️  OpenAI API key is optional for basic functionality")
    else:
        print("✅ .env file already exists")

if __name__ == "__main__":
    check_and_install()
    print("\n🚀 Setup complete! Run 'python main.py' to start JADE")