import os
import json
from datetime import datetime
import logging
import config  # Changed from: from config import LOG_FILE, MAX_LOG_SIZE_MB

class DetectionLogger:
    def __init__(self):
        """Initialize detection logger with rotation support"""
        self.log_file = config.LOG_FILE  # Changed
        self.max_size_mb = config.MAX_LOG_SIZE_MB  # Changed
        self._setup_logging()
        self._check_log_rotation()
        
    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Setup file logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
    def _check_log_rotation(self):
        """Rotate log file if it exceeds maximum size"""
        if os.path.exists(self.log_file):
            size_mb = os.path.getsize(self.log_file) / (1024 * 1024)
            if size_mb > self.max_size_mb:
                self._rotate_log()
    
    def _rotate_log(self):
        """Rotate log file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        rotated_file = f"{self.log_file}.{timestamp}"
        os.rename(self.log_file, rotated_file)
        logging.info(f"Log rotated: {rotated_file}")
    
    def log_detection(self, detection_data, include_image_data=False):
        """Log detection event with structured data"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'detections': detection_data['detections'],
            'frame_info': {
                'width': detection_data.get('frame_width', 0),
                'height': detection_data.get('frame_height', 0),
                'fps': detection_data.get('fps', 0.0)
            },
            'device_info': detection_data.get('device_info', {}),
            'session_id': detection_data.get('session_id', 'default')
        }
        
        if include_image_data and detection_data.get('image_hash'):
            log_entry['image_hash'] = detection_data['image_hash']
        
        # Write to JSONL file
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Also log to console
        if detection_data['detections']:
            for det in detection_data['detections']:
                logging.info(f"Detected: {det['class_name']} ({det['confidence']:.2f})")
        else:
            logging.debug("No detections in frame")
    
    def log_system_event(self, event_type, details):
        """Log system events (startup, errors, etc.)"""
        event_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details
        }
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event_entry) + '\n')
        
        if event_type == 'ERROR':
            logging.error(f"System error: {details}")
        elif event_type == 'WARNING':
            logging.warning(f"System warning: {details}")
        else:
            logging.info(f"System event: {details}")
    
    def get_recent_detections(self, limit=100):
        """Retrieve recent detections for analysis"""
        detections = []
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if 'detections' in entry:
                            detections.append(entry)
                    except json.JSONDecodeError:
                        continue
        
        return detections[-limit:] if detections else []
    
    def export_statistics(self, output_file='detection_stats.json'):
        """Export detection statistics"""
        detections = self.get_recent_detections()
        
        stats = {
            'total_detections': 0,
            'unique_objects': set(),
            'confidence_stats': {
                'average': 0.0,
                'min': 1.0,
                'max': 0.0
            },
            'detection_by_class': {},
            'timeline': []
        }
        
        confidences = []
        
        for entry in detections:
            for det in entry.get('detections', []):
                stats['total_detections'] += 1
                stats['unique_objects'].add(det['class_name'])
                confidences.append(det['confidence'])
                
                # Count by class
                class_name = det['class_name']
                stats['detection_by_class'][class_name] = stats['detection_by_class'].get(class_name, 0) + 1
        
        if confidences:
            stats['confidence_stats'] = {
                'average': sum(confidences) / len(confidences),
                'min': min(confidences),
                'max': max(confidences)
            }
        
        stats['unique_objects'] = list(stats['unique_objects'])
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        return stats
