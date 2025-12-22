import os
from dotenv import load_dotenv

load_dotenv()

# Enhanced configuration
class Config:
    # API Keys (set in .env file) - Only OpenAI needed
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    
    # Voice Settings
    WAKE_WORD = "hey jade"
    VOICE_GENDER = "female"  # "male" or "female"
    SPEAKING_RATE = 180
    
    # Voice Activation Settings
    WAKE_WORD_SENSITIVITY = 0.7
    SPEECH_TIMEOUT = 5
    PHRASE_TIME_LIMIT = 7
    
    # Camera Settings
    CAMERA_ID = 2  # Default camera(0,1,2,3,4)
    PREVIEW_WIDTH = 1280
    PREVIEW_HEIGHT = 720
    
    # Detection Settings
    MODEL_PATH = 'models/yolo11n.pt'
    CONFIDENCE = 0.5  # Lowered for more detections
    MAX_DETECTIONS = 50
    IOU_THRESHOLD = 0.45
    AGNOSTIC_NMS = True
    
    # Analysis Settings
    ENABLE_DETAILED_ANALYSIS = True
    SAVE_ANALYSIS_REPORTS = True
    REPORT_DIR = 'reports'
    
    # Application Settings
    TARGET_FPS = 30
    FRAME_SKIP = 2  # Process every 2nd frame for performance
    
    # Logging
    LOG_FILE = 'logs/detections.jsonl'
    MAX_LOG_SIZE_MB = 10
    
    # GUI Settings
    PREVIEW_SIZE = (1280, 720)

# Create instance for easy access
config = Config()

# Export individual variables for backward compatibility
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
PREVIEW_SIZE = config.PREVIEW_SIZE

# Enhanced detection settings
MAX_DETECTIONS = config.MAX_DETECTIONS
IOU_THRESHOLD = config.IOU_THRESHOLD
AGNOSTIC_NMS = config.AGNOSTIC_NMS

# Voice activation settings
WAKE_WORD_SENSITIVITY = config.WAKE_WORD_SENSITIVITY
SPEECH_TIMEOUT = config.SPEECH_TIMEOUT
PHRASE_TIME_LIMIT = config.PHRASE_TIME_LIMIT

# Analysis settings
ENABLE_DETAILED_ANALYSIS = config.ENABLE_DETAILED_ANALYSIS
SAVE_ANALYSIS_REPORTS = config.SAVE_ANALYSIS_REPORTS
REPORT_DIR = config.REPORT_DIR

# API Keys dictionary (only OpenAI)
API_KEYS = {
    'openai': config.OPENAI_API_KEY
}

# Enhanced Voice Settings dictionary
VOICE_SETTINGS = {
    'wake_word': config.WAKE_WORD,
    'voice_gender': config.VOICE_GENDER,
    'speaking_rate': config.SPEAKING_RATE,
    'auto_start': True,
    'wake_word_sensitivity': config.WAKE_WORD_SENSITIVITY,
    'speech_timeout': config.SPEECH_TIMEOUT,
    'phrase_time_limit': config.PHRASE_TIME_LIMIT
}

# Detection Settings dictionary
DETECTION_SETTINGS = {
    'model_path': config.MODEL_PATH,
    'confidence': config.CONFIDENCE,
    'max_detections': config.MAX_DETECTIONS,
    'iou_threshold': config.IOU_THRESHOLD,
    'agnostic_nms': config.AGNOSTIC_NMS
}

# Create .env template if not exists
def create_env_template():
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write("""# JADE Voice Assistant API Keys
# Get key from: https://platform.openai.com/api-keys

OPENAI_API_KEY=your_openai_key_here

# Optional keys (not required for basic functionality)
# GOOGLE_SEARCH_API_KEY=your_key_here
""")
        print("📄 Created .env template file")
        print("⚠️  Please edit .env with your OpenAI API key")

# Create reports directory if not exists
def create_directories():
    directories = ['logs', 'voice_logs', 'models', 'reports', 
                   'train/images', 'train/labels',
                   'val/images', 'val/labels',
                   'test/images', 'test/labels']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

if __name__ == "__main__":
    create_env_template()
    create_directories()
    print("✅ Configuration loaded and directories created")