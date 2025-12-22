import os
import json
from datetime import datetime
import logging
import numpy as np
from config import LOG_FILE, MAX_LOG_SIZE_MB, REPORT_DIR

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64, np.int8, np.int16, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)

class DetectionLogger:
    def __init__(self):
        """Initialize detection logger with rotation support"""
        self.log_file = LOG_FILE
        self.max_size_mb = MAX_LOG_SIZE_MB
        self.report_dir = REPORT_DIR
        self._setup_logging()
        self._check_log_rotation()
        
        # Create report directory if not exists
        os.makedirs(self.report_dir, exist_ok=True)
        
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
        # Convert numpy types to Python native types
        detection_data = self._convert_numpy_types(detection_data)
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'detections': detection_data['detections'],
            'frame_info': {
                'width': detection_data.get('frame_width', 0),
                'height': detection_data.get('frame_height', 0),
                'fps': detection_data.get('fps', 0.0)
            },
            'device_info': detection_data.get('device_info', {}),
            'session_id': detection_data.get('session_id', 'default'),
            'analysis_mode': detection_data.get('analysis_mode', 'basic')
        }
        
        if include_image_data and detection_data.get('image_hash'):
            log_entry['image_hash'] = detection_data['image_hash']
        
        # Write to JSONL file with custom encoder
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, cls=NumpyEncoder) + '\n')
        
        # Also log to console
        if detection_data['detections']:
            for det in detection_data['detections']:
                logging.info(f"Detected: {det['class_name']} ({det['confidence']:.2f}) at {det['bbox']}")
        else:
            logging.debug("No detections in frame")
        
        return log_entry
    
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
        """Log system events (startup, errors, etc.)"""
        event_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details
        }
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event_entry, cls=NumpyEncoder) + '\n')
        
        if event_type == 'ERROR':
            logging.error(f"System error: {details}")
        elif event_type == 'WARNING':
            logging.warning(f"System warning: {details}")
        else:
            logging.info(f"System event: {details}")
        
        return event_entry
    
    def log_voice_command(self, command, response, mode):
        """Log voice command interactions"""
        voice_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'voice_command',
            'command': command,
            'response': response,
            'mode': mode
        }
        
        voice_log_file = os.path.join('voice_logs', f"voice_{datetime.now().strftime('%Y%m%d')}.jsonl")
        os.makedirs('voice_logs', exist_ok=True)
        
        with open(voice_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(voice_entry, cls=NumpyEncoder) + '\n')
        
        return voice_entry
    
    def log_comprehensive_analysis(self, assessment):
        """Log comprehensive object analysis"""
        # Convert numpy types in assessment
        assessment = self._convert_numpy_types(assessment)
        
        analysis_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'comprehensive_analysis',
            'assessment': assessment
        }
        
        analysis_log_file = os.path.join(self.report_dir, f"analysis_{datetime.now().strftime('%Y%m%d')}.jsonl")
        
        with open(analysis_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(analysis_entry, cls=NumpyEncoder) + '\n')
        
        logging.info(f"Comprehensive analysis logged for {assessment['identification']['object_type']}")
        return analysis_entry
    
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
    
    def get_voice_logs(self, date=None, limit=50):
        """Retrieve voice command logs"""
        if date is None:
            date = datetime.now().strftime('%Y%m%d')
        
        voice_log_file = os.path.join('voice_logs', f"voice_{date}.jsonl")
        voice_logs = []
        
        if os.path.exists(voice_log_file):
            with open(voice_log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        voice_logs.append(entry)
                    except json.JSONDecodeError:
                        continue
        
        return voice_logs[-limit:] if voice_logs else []
    
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
            'detection_by_hour': {},
            'timeline': [],
            'session_summary': {}
        }
        
        confidences = []
        sessions = {}
        
        for entry in detections:
            try:
                timestamp = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                hour = timestamp.hour
                
                # Track by hour
                stats['detection_by_hour'][hour] = stats['detection_by_hour'].get(hour, 0) + 1
                
                # Track by session
                session_id = entry.get('session_id', 'unknown')
                sessions[session_id] = sessions.get(session_id, 0) + len(entry.get('detections', []))
                
                for det in entry.get('detections', []):
                    stats['total_detections'] += 1
                    stats['unique_objects'].add(det['class_name'])
                    confidences.append(float(det['confidence']))
                    
                    # Count by class
                    class_name = det['class_name']
                    stats['detection_by_class'][class_name] = stats['detection_by_class'].get(class_name, 0) + 1
            except (KeyError, ValueError) as e:
                continue
        
        if confidences:
            stats['confidence_stats'] = {
                'average': sum(confidences) / len(confidences),
                'min': min(confidences),
                'max': max(confidences),
                'std': np.std(confidences) if len(confidences) > 1 else 0
            }
        
        stats['unique_objects'] = list(stats['unique_objects'])
        stats['session_summary'] = {
            'total_sessions': len(sessions),
            'detections_per_session': sessions,
            'most_active_session': max(sessions, key=sessions.get) if sessions else None
        }
        
        # Add timeline data
        if detections:
            try:
                # Get first and last detection times
                first_entry = detections[0]
                last_entry = detections[-1]
                first_time = datetime.fromisoformat(first_entry['timestamp'].replace('Z', '+00:00'))
                last_time = datetime.fromisoformat(last_entry['timestamp'].replace('Z', '+00:00'))
                
                stats['timeline'] = {
                    'first_detection': first_time.isoformat(),
                    'last_detection': last_time.isoformat(),
                    'duration_hours': (last_time - first_time).total_seconds() / 3600
                }
            except (KeyError, ValueError):
                stats['timeline'] = {}
        
        output_path = os.path.join(self.report_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, cls=NumpyEncoder)
        
        logging.info(f"Statistics exported to {output_path}")
        return stats
    
    def export_comprehensive_report(self, assessments, output_file='comprehensive_report.json'):
        """Export comprehensive analysis report"""
        # Convert numpy types in assessments
        assessments = self._convert_numpy_types(assessments)
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_assessments': len(assessments),
            'assessments': assessments,
            'summary': {
                'total_objects': len(assessments),
                'unique_objects': set(),
                'average_condition_score': 0.0,
                'total_estimated_value': 0.0
            }
        }
        
        condition_scores = []
        unique_objects = set()
        total_value = 0.0
        
        for assessment in assessments:
            try:
                unique_objects.add(assessment['identification']['object_type'])
                condition_scores.append(float(assessment['condition']['score']))
                
                # Extract numeric value from estimated value string
                value_str = assessment['value_assessment']['estimated_value']
                try:
                    # Remove currency symbols and convert
                    if 'M' in value_str:
                        value = float(value_str.replace('$', '').replace('M', '')) * 1000000
                    elif 'K' in value_str:
                        value = float(value_str.replace('$', '').replace('K', '')) * 1000
                    else:
                        value = float(value_str.replace('$', '').replace(',', ''))
                    total_value += value
                except (ValueError, AttributeError):
                    pass
            except (KeyError, ValueError):
                continue
        
        if condition_scores:
            report['summary']['average_condition_score'] = sum(condition_scores) / len(condition_scores)
        
        report['summary']['unique_objects'] = list(unique_objects)
        report['summary']['total_estimated_value'] = f"${total_value:,.2f}"
        
        output_path = os.path.join(self.report_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        
        logging.info(f"Comprehensive report exported to {output_path}")
        return report
    
    def generate_daily_report(self):
        """Generate daily summary report"""
        today = datetime.now().strftime('%Y%m%d')
        
        # Get today's detections
        all_detections = self.get_recent_detections(limit=1000)
        today_detections = [
            d for d in all_detections 
            if datetime.fromisoformat(d['timestamp'].replace('Z', '+00:00')).strftime('%Y%m%d') == today
        ]
        
        # Get today's voice logs
        today_voice_logs = self.get_voice_logs(date=today)
        
        report = {
            'date': today,
            'detection_summary': {
                'total_frames': len(today_detections),
                'total_objects': sum(len(d.get('detections', [])) for d in today_detections),
                'unique_objects': set()
            },
            'voice_summary': {
                'total_commands': len(today_voice_logs),
                'commands_by_type': {}
            },
            'hourly_activity': {}
        }
        
        # Process detections
        for entry in today_detections:
            try:
                for det in entry.get('detections', []):
                    report['detection_summary']['unique_objects'].add(det['class_name'])
                
                # Track hourly activity
                hour = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00')).hour
                report['hourly_activity'][hour] = report['hourly_activity'].get(hour, 0) + 1
            except (KeyError, ValueError):
                continue
        
        report['detection_summary']['unique_objects'] = list(report['detection_summary']['unique_objects'])
        
        # Process voice logs
        for log in today_voice_logs:
            try:
                cmd_type = log.get('type', 'unknown')
                report['voice_summary']['commands_by_type'][cmd_type] = report['voice_summary']['commands_by_type'].get(cmd_type, 0) + 1
            except (KeyError, ValueError):
                continue
        
        # Save daily report
        daily_report_file = os.path.join(self.report_dir, f"daily_report_{today}.json")
        with open(daily_report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        
        logging.info(f"Daily report generated for {today}")
        return report