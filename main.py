import cv2
import time
import numpy as np
from datetime import datetime
import hashlib
import uuid
import config
from Jade import JadeImageDetector
from outtxt import DetectionLogger
from knowledge_base import JADE_KNOWLEDGE

class JADESystem:
    def __init__(self):
        """Initialize JADE detection system"""
        self.session_id = str(uuid.uuid4())[:8]
        print(f"🚀 JADE Session ID: {self.session_id}")
        
        # Initialize logger
        self.logger = DetectionLogger()
        self.logger.log_system_event('SESSION_START', {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat()
        })
        
        # Initialize detector
        self.detector = JadeImageDetector()
        self.logger.log_system_event('DETECTOR_INITIALIZED', {
            'device': self.detector.device,
            'model': config.MODEL_PATH
        })
        
        # Initialize camera
        self.camera_id = config.CAMERA_ID
        self.cap = self._initialize_camera()
        if not self.cap:
            return
        
        # Performance tracking
        self.fps_history = []
        self.frame_count = 0
        self.processing_times = []
        self.last_knowledge_time = {}
        
        # Detection tracking
        self.current_detections = []
        self.detection_history = []
        self.object_presence = {}  # Track object presence over time
        
        print("✅ JADE System Initialized")
        print("📊 Press 'q' to quit, 's' to save stats, 'p' to pause")
    
    def _initialize_camera(self):
        """Initialize camera with optimal settings"""
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"❌ Camera {self.camera_id} not accessible!")
            self.logger.log_system_event('CAMERA_ERROR', {
                'camera_id': self.camera_id,
                'error': 'Failed to open camera'
            })
            
            # Try alternative cameras
            for alt_id in [1, 2, 0]:
                if alt_id == self.camera_id:
                    continue
                cap = cv2.VideoCapture(alt_id)
                if cap.isOpened():
                    print(f"✅ Using camera {alt_id} as fallback")
                    self.camera_id = alt_id
                    break
        
        if not cap.isOpened():
            self.logger.log_system_event('FATAL_ERROR', {
                'error': 'No camera available'
            })
            return None
        
        # Configure camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.PREVIEW_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.PREVIEW_SIZE[1])
        cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"📷 Camera: {actual_width}x{actual_height} @ {actual_fps:.1f}FPS")
        
        self.logger.log_system_event('CAMERA_INITIALIZED', {
            'resolution': f"{actual_width}x{actual_height}",
            'fps': actual_fps,
            'camera_id': self.camera_id
        })
        
        return cap
    
    def _calculate_image_hash(self, frame):
        """Calculate hash for duplicate frame detection"""
        # Convert to grayscale and resize for faster hashing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (8, 8))
        avg = resized.mean()
        hash_str = ''.join(['1' if pixel > avg else '0' for pixel in resized.flatten()])
        return hash_str
    
    def _update_object_presence(self, detections):
        """Track object presence over time"""
        current_time = time.time()
        current_objects = {det['class_name'] for det in detections}
        
        # Update presence tracking
        for obj_name in current_objects:
            if obj_name not in self.object_presence:
                self.object_presence[obj_name] = {
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'total_frames': 1
                }
            else:
                self.object_presence[obj_name]['last_seen'] = current_time
                self.object_presence[obj_name]['total_frames'] += 1
        
        # Remove objects not seen for 5 seconds
        stale_objects = []
        for obj_name, data in self.object_presence.items():
            if current_time - data['last_seen'] > 5.0 and obj_name not in current_objects:
                stale_objects.append(obj_name)
        
        for obj_name in stale_objects:
            duration = self.object_presence[obj_name]['last_seen'] - self.object_presence[obj_name]['first_seen']
            print(f"📤 {obj_name} left scene after {duration:.1f}s")
            del self.object_presence[obj_name]
    
    def _display_knowledge(self, class_name, confidence):
        """Display knowledge about detected object"""
        current_time = time.time()
        
        # Check cooldown and confidence
        if class_name in self.last_knowledge_time:
            if current_time - self.last_knowledge_time[class_name] < config.KNOWLEDGE_COOLDOWN:
                return
        
        if confidence < config.MIN_CONFIDENCE_KNOWLEDGE:
            return
        
        if class_name in JADE_KNOWLEDGE and config.ENABLE_KNOWLEDGE:
            knowledge = JADE_KNOWLEDGE[class_name]
            print(f"🧠 {class_name.upper()}: {knowledge['info']}")
            print(f"   💰 Value: {knowledge['value']}")
            print(f"   💡 Tip: {knowledge['tip']}")
            print("-" * 50)
            
            self.last_knowledge_time[class_name] = current_time
    
    def _draw_detection_info(self, frame, detections, fps):
        """Draw detection information on frame"""
        h, w = frame.shape[:2]
        
        # Draw header
        cv2.putText(frame, f"JADE v1.0 | Session: {self.session_id}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", 
                   (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw device info
        device_text = f"Device: {self.detector.device}"
        cv2.putText(frame, device_text, 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Draw detection count
        detection_text = f"Detections: {len(detections)}"
        cv2.putText(frame, detection_text, 
                   (w - 150, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Draw detection boxes and labels
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = det['confidence']
            class_name = det['class_name']
            
            # Draw bounding box
            color = (0, 255, 0) if conf > 0.7 else (0, 165, 255) if conf > 0.5 else (0, 0, 255)
            thickness = 2 if conf > 0.7 else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{class_name} {conf:.2f}"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                        (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Main detection loop"""
        last_time = time.time()
        frame_skip_counter = 0
        
        while True:
            start_time = time.time()
            
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Failed to read frame")
                time.sleep(0.1)
                continue
            
            self.frame_count += 1
            frame_skip_counter += 1
            
            # Skip frames if configured
            if frame_skip_counter % config.FRAME_SKIP != 0:
                continue
            
            # Process frame
            processed_frame, detections = self.detector.detect(frame)
            
            # Update object presence tracking
            self._update_object_presence(detections)
            
            # Display knowledge for new objects
            for det in detections:
                self._display_knowledge(det['class_name'], det['confidence'])
            
            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - last_time)
            last_time = current_time
            self.fps_history.append(fps)
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
            
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            
            # Log detections
            log_data = {
                'session_id': self.session_id,
                'detections': detections,
                'frame_width': frame.shape[1],
                'frame_height': frame.shape[0],
                'fps': avg_fps,
                'device_info': {
                    'device': self.detector.device,
                    'frame_count': self.frame_count
                },
                'image_hash': self._calculate_image_hash(frame)
            }
            self.logger.log_detection(log_data)
            
            # Draw UI
            display_frame = self._draw_detection_info(processed_frame, detections, avg_fps)
            
            # Show frame
            cv2.imshow(config.WINDOW_NAME, display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                stats = self.logger.export_statistics()
                print(f"📊 Stats saved: {stats['total_detections']} total detections")
            elif key == ord('p'):
                print("⏸️ Paused. Press any key to continue...")
                cv2.waitKey(0)
            
            # Performance monitoring
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Log session statistics
        if self.processing_times:
            avg_process_time = sum(self.processing_times) / len(self.processing_times)
            print(f"📈 Average processing time: {avg_process_time*1000:.1f}ms")
        
        print(f"📊 Total frames processed: {self.frame_count}")
        
        self.logger.log_system_event('SESSION_END', {
            'session_id': self.session_id,
            'total_frames': self.frame_count,
            'duration': time.time() - self.start_time if hasattr(self, 'start_time') else 0
        })

def main():
    """Main entry point"""
    print("=" * 60)
    print("🤖 JADE - Just Another Detection Engine")
    print("=" * 60)
    
    try:
        jade = JADESystem()
        jade.start_time = time.time()
        jade.run()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ System error: {str(e)}")
    finally:
        if 'jade' in locals():
            jade.cleanup()
        print("✅ JADE shutdown complete")

if __name__ == "__main__":
    main()
