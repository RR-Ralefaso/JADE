import os
from dotenv import load_dotenv

load_dotenv()

# Simple configuration
class Config:
    # API Keys (set in .env file) - Only OpenAI needed
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    
    # Voice Settings
    WAKE_WORD = "hey jade"
    VOICE_GENDER = "female"  # "male" or "female"
    SPEAKING_RATE = 180
    
    # Camera Settings
    CAMERA_ID = 2 # Default camera(0,1,2,3,4)
    PREVIEW_WIDTH = 1280
    PREVIEW_HEIGHT = 720
    
    # Detection Settings
    MODEL_PATH = 'models/yolo11n.pt'
    CONFIDENCE = 0.5
    
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

# API Keys dictionary (only OpenAI)
API_KEYS = {
    'openai': config.OPENAI_API_KEY
}

# Voice Settings dictionary
VOICE_SETTINGS = {
    'wake_word': config.WAKE_WORD,
    'voice_gender': config.VOICE_GENDER,
    'speaking_rate': config.SPEAKING_RATE,
    'auto_start': True
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

if __name__ == "__main__":
    create_env_template()
    print("✅ Configuration loaded")
