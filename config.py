import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    
    # Camera Settings
    CAMERA_ID = int(os.getenv('CAMERA_ID', '0'))
    PREVIEW_WIDTH = 1280
    PREVIEW_HEIGHT = 720
    TARGET_FPS = 60 #30
    
    # Model Settings
    MODEL_PATH = 'models/yolo11n.pt'
    CONFIDENCE = 0.5
    IOU_THRESHOLD = 0.45
    AGNOSTIC_NMS = True
    MAX_DETECTIONS = 60 #100
    
    # Performance Optimization
    FRAME_SKIP = 2  # Process every 2nd frame for better performance 
    HALF_PRECISION = True
    
    # Voice Settings
    WAKE_WORD = os.getenv('WAKE_WORD', 'hey jade')
    VOICE_GENDER = os.getenv('VOICE_GENDER', 'female')
    SPEAKING_RATE = int(os.getenv('SPEAKING_RATE', '180'))
    AUTO_START_VOICE = True
    
    # Audio Settings
    AUDIO_SAMPLE_RATE = 44100
    AUDIO_CHUNK_SIZE = 4096
    NOISE_REDUCTION_ENABLED = True
    
    # Analysis Settings
    ENABLE_DEEP_ANALYSIS = True
    ENABLE_REALTIME_TRACKING = True  # Disabled for better performance
    
    # Logging & Storage
    LOG_FILE = 'logs/detections.jsonl'
    MAX_LOG_SIZE_MB = 50
    REPORT_DIR = 'reports'
    
    # Display Settings
    SHOW_FPS = True
    SHOW_CONFIDENCE = True
    SHOW_TRACKING_IDS = True #disable for better performance
    SHOW_BOUNDING_BOXES = True
    THEME = "dark"
    
    @property
    def model_device(self):
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    
    @property
    def model_dtype(self):
        if self.HALF_PRECISION:
            import torch
            return torch.float16
        return None

# Create instance for easy access
config = Config()

def setup_directories():
    """Create all necessary directories"""
    directories = [
        'models',
        'logs',
        'voice_logs', 
        'reports',
        'exports',
        'exports/images',
        'exports/videos',
        'datasets/train/images', 
        'datasets/train/labels',
        'datasets/val/images', 
        'datasets/val/labels',
        'datasets/test/images', 
        'datasets/test/labels'
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            print(f"⚠️  Could not create {directory}: {e}")

def check_api_keys():
    """Check if required API keys are available"""
    print("🔑 Checking API keys...")
    
    # Check OpenAI API key
    if not config.OPENAI_API_KEY or config.OPENAI_API_KEY == 'your_openai_key_here_optional':
        print("⚠️  OpenAI API key not set or using placeholder")
        print("   Note: OpenAI API is optional for basic functionality")
        print("   Get key from: https://platform.openai.com/api-keys")
        return False
    
    # Check if key looks valid
    if config.OPENAI_API_KEY.startswith('sk-') and len(config.OPENAI_API_KEY) > 30:
        print("✅ OpenAI API key found")
        return True
    else:
        print("⚠️  OpenAI API key format looks invalid")
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    dependencies = [
        ('opencv-python', 'cv2'),
        ('numpy', 'numpy'),
        ('torch', 'torch'),
        ('ultralytics', 'ultralytics'),
        ('speechrecognition', 'speech_recognition'),
        ('pyttsx3', 'pyttsx3'),
        ('pyaudio', 'pyaudio'),
        ('python-dotenv', 'dotenv'),
        ('requests', 'requests'),
    ]
    
    print("📦 Checking dependencies...")
    missing = []
    
    for pip_name, import_name in dependencies:
        try:
            __import__(import_name)
            print(f"✅ {pip_name}")
        except ImportError:
            print(f"❌ {pip_name}")
            missing.append(pip_name)
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("   Run: pip install " + " ".join(missing))
        return False
    
    print("✅ All dependencies installed")
    return True

def check_camera():
    """Check camera availability"""
    print("📷 Checking camera...")
    
    try:
        import cv2
        
        # Try default camera
        cap = cv2.VideoCapture(config.CAMERA_ID)
        if not cap.isOpened():
            # Try alternative cameras
            for cam_id in [1, 2, 0]:
                cap = cv2.VideoCapture(cam_id)
                if cap.isOpened():
                    print(f"✅ Camera found: ID {cam_id}")
                    config.CAMERA_ID = cam_id
                    cap.release()
                    return True, cam_id
            print("❌ No camera found")
            return False, config.CAMERA_ID
        
        ret, frame = cap.read()
        if ret:
            print(f"✅ Camera {config.CAMERA_ID}: {frame.shape[1]}x{frame.shape[0]}")
        else:
            print(f"❌ Camera {config.CAMERA_ID} found but cannot read frames")
        
        cap.release()
        return ret, config.CAMERA_ID
        
    except Exception as e:
        print(f"❌ Camera check error: {e}")
        return False, config.CAMERA_ID

def check_model():
    """Check if YOLO model exists or can be downloaded"""
    print("🤖 Checking model...")
    
    if os.path.exists(config.MODEL_PATH):
        file_size = os.path.getsize(config.MODEL_PATH) / (1024 * 1024)  # MB
        print(f"✅ Model found: {config.MODEL_PATH} ({file_size:.1f} MB)")
        return True
    else:
        print(f"⚠️  Model not found: {config.MODEL_PATH}")
        print("   It will be downloaded automatically on first run")
        return False

def initialize_system():
    """Initialize the entire system"""
    print("="*60)
    print("🚀 JADE System Initialization")
    print("="*60)
    
    # Step 1: Create directories
    print("\n1. Setting up directories...")
    setup_directories()
    print("✅ Directories created")
    
    # Step 2: Check dependencies
    print("\n2. Checking dependencies...")
    deps_ok = check_dependencies()
    
    # Step 3: Check API keys
    print("\n3. Checking API keys...")
    api_ok = check_api_keys()
    
    # Step 4: Check camera
    print("\n4. Checking camera...")
    camera_ok, camera_id = check_camera()
    
    # Step 5: Check model
    print("\n5. Checking model...")
    model_ok = check_model()
    
    # Step 6: System info
    print("\n6. System Information:")
    import platform
    import sys
    
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   Platform: {platform.platform()}")
    print(f"   Processor: {platform.processor()}")
    
    try:
        import torch
        print(f"   PyTorch: {torch.__version__}")
        print(f"   Device: {config.model_device}")
        if torch.cuda.is_available():
            print(f"   CUDA: {torch.version.cuda}")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    except:
        print("   PyTorch: Not available")
    
    print("\n" + "="*60)
    
    # Summary
    print("📊 INITIALIZATION SUMMARY:")
    print(f"   Dependencies: {'✅ OK' if deps_ok else '❌ Missing'}")
    print(f"   API Keys: {'✅ Found' if api_ok else '⚠️  Missing (optional)'}")
    print(f"   Camera: {'✅ Found' if camera_ok else '❌ Not found'}")
    print(f"   Model: {'✅ Found' if model_ok else '⚠️  Will download on first run'}")
    
    return camera_ok or deps_ok  # Return True if either camera or dependencies are OK

if __name__ == "__main__":
    # Run initialization
    success = initialize_system()
    
    if success:
        print("\n🎯 CONFIGURATION READY:")
        print(f"   Camera ID: {config.CAMERA_ID}")
        print(f"   Resolution: {config.PREVIEW_WIDTH}x{config.PREVIEW_HEIGHT}")
        print(f"   Model: {config.MODEL_PATH}")
        print(f"   Confidence: {config.CONFIDENCE}")
        print(f"   Voice: {config.VOICE_GENDER} at {config.SPEAKING_RATE} WPM")
        print(f"   Wake Word: '{config.WAKE_WORD}'")
        
        print("\n🚀 To start JADE:")
        print("   python main.py")
        
        print("\n🎤 Voice Commands (Continuous Listening):")
        print("   • 'Hey jade' - Wake phrase")
        print("   • 'Analyze object' - Analyze current view")
        print("   • 'What do you see' - Describe scene")
        print("   • Speak naturally - I'm always listening")
    else:
        print("\n❌ INITIALIZATION FAILED")
        print("   Please fix the issues above and try again")
    
    print("="*60)

# Export individual variables for backward compatibility (keep at end of file)
LOG_FILE = config.LOG_FILE
MAX_LOG_SIZE_MB = config.MAX_LOG_SIZE_MB
OPENAI_API_KEY = config.OPENAI_API_KEY
WAKE_WORD = config.WAKE_WORD
VOICE_GENDER = config.VOICE_GENDER
SPEAKING_RATE = config.SPEAKING_RATE
CAMERA_ID = config.CAMERA_ID
PREVIEW_WIDTH = config.PREVIEW_WIDTH
PREVIEW_HEIGHT = config.PREVIEW_HEIGHT
MODEL_PATH = config.MODEL_PATH
CONFIDENCE = config.CONFIDENCE
TARGET_FPS = config.TARGET_FPS
FRAME_SKIP = config.FRAME_SKIP

# Enhanced detection settings
MAX_DETECTIONS = config.MAX_DETECTIONS
IOU_THRESHOLD = config.IOU_THRESHOLD
AGNOSTIC_NMS = config.AGNOSTIC_NMS

# Analysis settings
ENABLE_DEEP_ANALYSIS = config.ENABLE_DEEP_ANALYSIS
ENABLE_REALTIME_TRACKING = config.ENABLE_REALTIME_TRACKING
REPORT_DIR = config.REPORT_DIR

# Voice Settings dictionary
VOICE_SETTINGS = {
    'wake_word': config.WAKE_WORD,
    'voice_gender': config.VOICE_GENDER,
    'speaking_rate': config.SPEAKING_RATE,
    'auto_start': config.AUTO_START_VOICE,
    'noise_reduction': config.NOISE_REDUCTION_ENABLED,
    'sample_rate': config.AUDIO_SAMPLE_RATE
}

# Detection Settings dictionary
DETECTION_SETTINGS = {
    'model_path': config.MODEL_PATH,
    'confidence': config.CONFIDENCE,
    'max_detections': config.MAX_DETECTIONS,
    'iou_threshold': config.IOU_THRESHOLD,
    'agnostic_nms': config.AGNOSTIC_NMS,
    'device': config.model_device
}