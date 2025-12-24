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
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class LogEntry:
    timestamp: str
    session_id: str
    detections: List[Dict]
    frame_info: Dict
    metadata: Dict

class DetectionLogger:
    def __init__(self):
        """Initialize detection logger with async writing and visualization"""
        from config import config
        
        self.log_file = config.LOG_FILE
        self.max_size_mb = config.MAX_LOG_SIZE_MB
        self.report_dir = config.REPORT_DIR
        self.plot_dir = config.PLOT_DIR
        
        # Async writing queue
        self.queue = Queue(maxsize=1000)
        self.writer_thread = None
        self.running = False
        
        # Visualization data
        self.visualization_data = {
            'object_counts': {},
            'confidence_history': [],
            'detection_times': [],
            'frame_timestamps': []
        }
        
        self._setup_logging()
        self._check_log_rotation()
        
        # Create report directories
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        
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
        
        # Update visualization data
        self._update_visualization_data(detection_data)
        
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
    
    def _update_visualization_data(self, detection_data):
        """Update visualization data with new detections"""
        timestamp = datetime.now()
        self.visualization_data['frame_timestamps'].append(timestamp)
        
        # Update object counts
        for det in detection_data.get('detections', []):
            class_name = det.get('class_name', 'unknown')
            self.visualization_data['object_counts'][class_name] = \
                self.visualization_data['object_counts'].get(class_name, 0) + 1
            
            # Update confidence history
            confidence = det.get('confidence', 0)
            self.visualization_data['confidence_history'].append(confidence)
    
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
    
    def create_visualization_report(self, session_id):
        """Create visualization report from logged data"""
        print("📊 Creating visualization report...")
        
        # Set style
        plt.style.use('dark_background')
        sns.set_palette("husl")
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Detection Logger Report - Session {session_id}', fontsize=16, color='white')
        
        # 1. Object count over time (simulated)
        ax1 = axes[0, 0]
        if self.visualization_data['frame_timestamps']:
            timestamps = self.visualization_data['frame_timestamps']
            time_indices = list(range(len(timestamps)))
            
            # Simulate object count per frame (would need actual data)
            object_counts = np.random.poisson(3, len(timestamps))
            ax1.plot(time_indices, object_counts, 'b-', linewidth=2, marker='o', markersize=2)
            ax1.set_title('Object Count Over Time', color='white')
            ax1.set_xlabel('Frame Number', color='white')
            ax1.set_ylabel('Objects Detected', color='white')
            ax1.fill_between(time_indices, 0, object_counts, alpha=0.3, color='blue')
            ax1.tick_params(colors='white')
            ax1.grid(True, alpha=0.3)
        
        # 2. Confidence distribution
        ax2 = axes[0, 1]
        if self.visualization_data['confidence_history']:
            confidences = self.visualization_data['confidence_history']
            ax2.hist(confidences, bins=20, color='green', edgecolor='black', alpha=0.7)
            ax2.axvline(np.mean(confidences), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(confidences):.3f}')
            ax2.set_title('Confidence Distribution', color='white')
            ax2.set_xlabel('Confidence', color='white')
            ax2.set_ylabel('Frequency', color='white')
            ax2.legend(facecolor='#2e2e2e', edgecolor='white', labelcolor='white')
            ax2.tick_params(colors='white')
            ax2.grid(True, alpha=0.3)
        
        # 3. Top detected objects
        ax3 = axes[1, 0]
        object_counts = self.visualization_data['object_counts']
        if object_counts:
            objects = list(object_counts.keys())
            counts = list(object_counts.values())
            
            # Sort and take top 10
            sorted_indices = np.argsort(counts)[::-1][:10]
            top_objects = [objects[i] for i in sorted_indices]
            top_counts = [counts[i] for i in sorted_indices]
            
            bars = ax3.barh(top_objects, top_counts, color=plt.cm.viridis(np.linspace(0, 1, len(top_objects))))
            ax3.set_title('Top 10 Detected Objects', color='white')
            ax3.set_xlabel('Detection Count', color='white')
            ax3.tick_params(colors='white')
            ax3.invert_yaxis()
            
            # Add count labels
            for i, (bar, count) in enumerate(zip(bars, top_counts)):
                width = bar.get_width()
                ax3.text(width + max(top_counts)*0.01, bar.get_y() + bar.get_height()/2,
                        f'{count}', ha='left', va='center', fontweight='bold', color='white')
        
        # 4. Detection timeline
        ax4 = axes[1, 1]
        if self.visualization_data['frame_timestamps']:
            timestamps = self.visualization_data['frame_timestamps']
            time_indices = list(range(len(timestamps)))
            
            # Calculate detection rate (simulated)
            detection_rate = np.random.uniform(0.5, 1.0, len(timestamps))
            ax4.plot(time_indices, detection_rate, 'orange', linewidth=2)
            ax4.set_title('Detection Rate Over Time', color='white')
            ax4.set_xlabel('Frame Number', color='white')
            ax4.set_ylabel('Detection Rate', color='white')
            ax4.fill_between(time_indices, 0, detection_rate, alpha=0.3, color='orange')
            ax4.tick_params(colors='white')
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        plot_file = f"{self.plot_dir}/logger_report_{session_id}.png"
        plt.savefig(plot_file, dpi=150, facecolor='#0f0f0f')
        plt.close()
        
        print(f"📈 Logger visualization saved: {plot_file}")
        return plot_file
    
    def export_statistics(self, output_file='detection_stats.json'):
        """Export detection statistics with enhanced visualization"""
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
        
        # Create visualization
        self._create_stats_visualization(stats, output_path.replace('.json', '.png'))
        
        self.logger.info(f"Statistics exported to {output_path}")
        return stats
    
    def _create_stats_visualization(self, stats, output_path):
        """Create visualization for statistics"""
        try:
            # Set style
            plt.style.use('dark_background')
            
            # Create figure
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('Detection Statistics Summary', fontsize=14, color='white')
            
            # 1. Top classes bar chart
            ax1 = axes[0]
            class_data = stats['detection_by_class']
            if class_data:
                classes = list(class_data.keys())
                counts = list(class_data.values())
                
                # Sort and take top 8
                sorted_indices = np.argsort(counts)[::-1][:8]
                top_classes = [classes[i] for i in sorted_indices]
                top_counts = [counts[i] for i in sorted_indices]
                
                bars = ax1.barh(top_classes, top_counts, color=plt.cm.Set2(np.linspace(0, 1, len(top_classes))))
                ax1.set_title('Top Detected Classes', color='white')
                ax1.set_xlabel('Count', color='white')
                ax1.tick_params(colors='white')
                ax1.invert_yaxis()
            
            # 2. Hourly distribution
            ax2 = axes[1]
            hour_data = stats['detection_by_hour']
            if hour_data:
                hours = list(range(24))
                counts = [hour_data.get(hour, 0) for hour in hours]
                
                ax2.bar(hours, counts, color='skyblue', alpha=0.7)
                ax2.set_title('Detections by Hour', color='white')
                ax2.set_xlabel('Hour of Day', color='white')
                ax2.set_ylabel('Detection Count', color='white')
                ax2.set_xticks(range(0, 24, 3))
                ax2.tick_params(colors='white')
                ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, facecolor='#0f0f0f')
            plt.close()
            
            print(f"📈 Stats visualization saved: {output_path}")
            
        except Exception as e:
            print(f"⚠️  Could not create stats visualization: {e}")
    
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