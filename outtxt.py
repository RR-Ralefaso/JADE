import os
from datetime import datetime

LOG_FILE = "jade_detections.txt"

def reset_output_file():
    """Delete and create fresh log file"""
    with open(LOG_FILE, 'w') as f:
        f.write(f"JADE Log Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 50 + "\n")

def write_detection_output(detection_info):
    """Write JADE detection to log file"""
    with open(LOG_FILE, 'a') as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"[{timestamp}] {detection_info}\n")
