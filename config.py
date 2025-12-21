import os

# Model Configuration
MODEL_PATH = 'models/yolo11n.pt'
MODEL_SIZE = 640

# Detection Parameters
CONFIDENCE = 0.5
DETECTION_PARAMS = {
    'iou': 0.45,
    'agnostic_nms': False,
    'max_det': 50,
    'classes': None,
    'half': True,
    'augment': False
}

# Application Settings
WINDOW_NAME = 'JADE Universal Analyzer v3.0'
CAMERA_ID = 3

# Performance Settings
TARGET_FPS = 30
FRAME_SKIP = 1
PREVIEW_SIZE = (1280, 720)

# Analysis Settings
ENABLE_OBJECT_ANALYSIS = True
ANALYSIS_COOLDOWN = 10  # Seconds between repeated analysis
MIN_CONFIDENCE_ANALYSIS = 0.4

# Display Settings
ENABLE_SIDEBAR = True
SIDEBAR_WIDTH = 350
SHOW_VALUE_ESTIMATES = True
SHOW_CONDITION_INDICATORS = True

# Logging Configuration
LOG_FILE = 'logs/jade_analysis.jsonl'
MAX_LOG_SIZE_MB = 100
LOG_LEVEL = 'INFO'

# Value Estimation Settings
BASE_VALUES = {
    'car': 20000,
    'laptop': 800,
    'cell phone': 500,
    'bicycle': 300,
    'handbag': 200,
    'backpack': 50,
    'chair': 150,
    'bottle': 25,
    'book': 20,
    'clock': 100,
    'vase': 75
}

# File Paths
LOG_DIR = 'logs'
MODEL_DIR = 'models'
EXPORT_DIR = 'exports'
REPORT_DIR = 'reports'

# Create directories
for directory in [LOG_DIR, MODEL_DIR, EXPORT_DIR, REPORT_DIR]:
    os.makedirs(directory, exist_ok=True)
