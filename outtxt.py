import os
import json
import logging
import numpy as np
from datetime import datetime
from queue import Queue
from threading import Thread
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import msgpack

@dataclass
class LogEntry:
    timestamp: str
    session_id: str
    detections: List[Dict]
    frame_info: Dict
    metadata: Dict

class DetectionLogger:
    def __init__(self):
        """Initialize detection logger with async writing"""
        from config import config
        
        self.log_file = config.LOG_FILE
        self.max_size_mb = config.MAX_LOG_SIZE_MB
        self.report_dir = config.REPORT_DIR
        
        # Async writing queue
        self.queue = Queue(maxsize=1000)
        self.writer_thread = None
        self.running = False
        
        self._setup_logging()
        self._check_log_rotation()
        
        # Create report directory
        os.makedirs(self.report_dir, exist_ok=True)
        
        # Start writer thread
        self.start()
        
        print(f"📊 Logger initialized: {self.log_file}")
    
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
        
        self.logger = logging.getLogger(__name__)
    
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
        self.logger.info(f"Log rotated: {rotated_file}")
    
    def start(self):
        """Start the async writer thread"""
        self.running = True
        self.writer_thread = Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
    
    def stop(self):
        """Stop the async writer thread"""
        self.running = False
        self.queue.put(None)
        if self.writer_thread:
            self.writer_thread.join(timeout=2)
    
    def _writer_loop(self):
        """Writer thread loop"""
        while self.running:
            try:
                entry = self.queue.get(timeout=1)
                if entry is None:
                    break
                
                self._write_entry_sync(entry)
                self.queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Writer error: {e}")
    
    def _write_entry_sync(self, entry):
        """Write log entry synchronously"""
        try:
            # Try msgpack first (faster)
            msgpack_file = f"{self.log_file}.msgpack"
            with open(msgpack_file, 'ab') as f:
                msgpack.dump(asdict(entry), f)
        except:
            # Fallback to JSON
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(asdict(entry), cls=NumpyEncoder) + '\n')
    
    def log_detection(self, detection_data, include_image_data=False):
        """Async log detection event"""
        # Convert numpy types
        detection_data = self._convert_numpy_types(detection_data)
        
        log_entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            session_id=detection_data.get('session_id', 'default'),
            detections=detection_data['detections'],
            frame_info={
                'width': detection_data.get('frame_width', 0),
                'height': detection_data.get('frame_height', 0),
                'fps': detection_data.get('fps', 0.0)
            },
            metadata={
                'analysis_mode': detection_data.get('analysis_mode', 'basic'),
                'include_image_data': include_image_data,
                'image_hash': detection_data.get('image_hash', '')
            }
        )
        
        # Queue for async writing
        try:
            self.queue.put_nowait(log_entry)
        except:
            self.logger.warning("Log queue full, dropping entry")
        
        # Also log to console
        if detection_data['detections']:
            for det in detection_data['detections'][:5]:  # Limit console output
                self.logger.info(f"Detected: {det.get('class_name', 'unknown')} ({det.get('confidence', 0):.2f})")
        
        return asdict(log_entry)
    
    def _convert_numpy_types(self, data):
        """Recursively convert numpy types to Python native types"""
        if isinstance(data, dict):
            return {k: self._convert_numpy_types(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_numpy_types(item) for item in data]
        elif isinstance(data, np.float32):
            return float(data)
        elif isinstance(data, np.float64):
            return float(data)
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.bool_):
            return bool(data)
        else:
            return data
    
    def log_system_event(self, event_type, details):
        """Log system events"""
        event_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details
        }
        
        # Write sync for important events
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event_entry, cls=NumpyEncoder) + '\n')
        
        if event_type == 'ERROR':
            self.logger.error(f"System error: {details}")
        elif event_type == 'WARNING':
            self.logger.warning(f"System warning: {details}")
        else:
            self.logger.info(f"System event: {details}")
        
        return event_entry
    
    def get_recent_detections(self, limit=100):
        """Retrieve recent detections"""
        detections = []
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()[-limit*2:]  # Read extra lines
                for line in lines:
                    try:
                        entry = json.loads(line.strip())
                        if 'detections' in entry:
                            detections.append(entry)
                    except:
                        continue
        
        return detections[-limit:] if detections else []
    
    def export_statistics(self, output_file='detection_stats.json'):
        """Export detection statistics"""
        detections = self.get_recent_detections(limit=1000)
        
        stats = {
            'total_detections': 0,
            'unique_objects': set(),
            'confidence_stats': {
                'average': 0.0,
                'min': 1.0,
                'max': 0.0,
                'std': 0.0
            },
            'detection_by_class': {},
            'detection_by_hour': {},
            'timeline': {},
            'generated_at': datetime.now().isoformat()
        }
        
        confidences = []
        
        for entry in detections:
            try:
                timestamp = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                hour = timestamp.hour
                
                stats['detection_by_hour'][hour] = stats['detection_by_hour'].get(hour, 0) + 1
                
                for det in entry.get('detections', []):
                    stats['total_detections'] += 1
                    stats['unique_objects'].add(det.get('class_name', 'unknown'))
                    confidences.append(float(det.get('confidence', 0)))
                    
                    class_name = det.get('class_name', 'unknown')
                    stats['detection_by_class'][class_name] = stats['detection_by_class'].get(class_name, 0) + 1
            except:
                continue
        
        if confidences:
            stats['confidence_stats'] = {
                'average': sum(confidences) / len(confidences),
                'min': min(confidences),
                'max': max(confidences),
                'std': np.std(confidences) if len(confidences) > 1 else 0
            }
        
        stats['unique_objects'] = list(stats['unique_objects'])
        
        output_path = os.path.join(self.report_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, cls=NumpyEncoder)
        
        self.logger.info(f"Statistics exported to {output_path}")
        return stats
    
    def __del__(self):
        """Cleanup"""
        self.stop()

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)