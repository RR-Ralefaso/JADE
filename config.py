import os

# Model Configuration
MODEL_PATH = 'models/yolo11n.pt'
MODEL_SIZE = 640  # Input size for better accuracy

# Detection Parameters
CONFIDENCE = 0.5
DETECTION_PARAMS = {
    'iou': 0.45,           # Intersection over Union threshold
    'agnostic_nms': False, # Class-agnostic NMS
    'max_det': 100,        # Maximum detections per image
    'classes': None,       # Filter by class (None = all)
    'half': True,          # Use FP16 on supported devices
    'augment': False       # Test-time augmentation
}

# Application Settings
WINDOW_NAME = 'JADE - Object Detection System'
CAMERA_ID = 2  # Default camera (0, 1, 2, etc.)

# Performance Settings
TARGET_FPS = 30
FRAME_SKIP = 1  # Process every nth frame (1 = all frames)
PREVIEW_SIZE = (1280, 720)  # Display resolution

# Logging Configuration
LOG_FILE = 'logs/jade_detections.jsonl'
MAX_LOG_SIZE_MB = 50  # Rotate log when it reaches 50MB
LOG_LEVEL = 'INFO'    # DEBUG, INFO, WARNING, ERROR

# Knowledge Base Settings
ENABLE_KNOWLEDGE = True
KNOWLEDGE_COOLDOWN = 5.0  # Seconds between knowledge announcements
MIN_CONFIDENCE_KNOWLEDGE = 0.6  # Minimum confidence for knowledge display

# System Settings
ENABLE_GPU_MONITOR = True
GPU_MEMORY_THRESHOLD_MB = 100  # Warn if GPU memory below this
CPU_THREADS = 4  # Number of CPU threads for inference

# File Paths
LOG_DIR = 'logs'
MODEL_DIR = 'models'
EXPORT_DIR = 'exports'

# Create directories if they don't exist
for directory in [LOG_DIR, MODEL_DIR, EXPORT_DIR]:
    os.makedirs(directory, exist_ok=True)
