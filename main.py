
import cv2
import time
import numpy as np
from datetime import datetime
import hashlib
import uuid
import config
from Jade import JadeImageDetector
from outtxt import DetectionLogger
from knowledge_base import (
    ENHANCED_KNOWLEDGE,
    analyze_object_visual_features,
    estimate_condition_from_features,
    generate_detailed_report,
    format_report_for_display
)

class JADEUniversalAnalyzer:
    def __init__(self):
        """Initialize universal JADE analyzer for all objects"""
        self.session_id = str(uuid.uuid4())[:8]
        print(f"🚀 JADE Universal Analyzer Session: {self.session_id}")
        
        # Initialize logger
        self.logger = DetectionLogger()
        self.logger.log_system_event('SESSION_START', {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'version': '3.0 - Universal Object Analysis'
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
        
        # Analysis tracking
        self.object_reports = {}  # Cache of analyzed objects
        self.report_history = []
        self.last_report_time = {}
        self.report_cooldown = 10  # Seconds between repeated reports for same object
        
        # Display tracking
        self.display_mode = "normal"  # normal, detailed, minimal
        self.selected_object = None
        self.display_text = []
        self.text_expiry = 0
        
        # Performance tracking
        self.fps_history = []
        self.frame_count = 0
        self.processing_times = []
        
        print("✅ JADE Universal Analyzer Initialized")
        print("🔍 All objects will receive detailed analysis reports")
        print("📊 Commands: 'q'=quit, 's'=stats, 'd'=toggle detail, 'c'=clear display")
        print("             1-9=select object, 'r'=refresh analysis")
    
    def _initialize_camera(self):
        """Initialize camera with optimal settings"""
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"❌ Camera {self.camera_id} not accessible!")
            
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
            print("❌ No camera available!")
            return None
        
        # Configure camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.PREVIEW_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.PREVIEW_SIZE[1])
        cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📷 Camera: {actual_width}x{actual_height}")
        
        return cap
    
    def _analyze_object(self, frame, detection):
        """
        Perform comprehensive analysis on any object
        """
        object_type = detection['class_name']
        confidence = detection['confidence']
        bbox = detection['bbox']
        
        # Check cache first
        cache_key = f"{object_type}_{int(bbox[0])}_{int(bbox[1])}"
        current_time = time.time()
        
        if cache_key in self.object_reports:
            report_time, report = self.object_reports[cache_key]
            if current_time - report_time < self.report_cooldown:
                return report
        
        # Analyze visual features
        features = analyze_object_visual_features(frame, bbox, object_type)
        
        # Estimate condition
        condition, condition_confidence = estimate_condition_from_features(features, object_type)
        
        # Generate detailed report
        report = generate_detailed_report(object_type, condition, features, min(confidence, condition_confidence))
        
        # Add detection metadata
        report.update({
            "detection_confidence": confidence,
            "bbox": bbox,
            "timestamp": datetime.now().isoformat(),
            "analysis_id": str(uuid.uuid4())[:8]
        })
        
        # Cache report
        self.object_reports[cache_key] = (current_time, report)
        
        # Log report
        self._log_object_report(report)
        
        return report
    
    def _log_object_report(self, report):
        """Log detailed object analysis"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'OBJECT_ANALYSIS',
            'object': report['object'],
            'condition': report['condition'],
            'estimated_value': report['estimated_value'],
            'confidence': report['analysis_confidence'],
            'session_id': self.session_id
        }
        
        self.logger.log_system_event('OBJECT_ANALYSIS', log_entry)
        self.report_history.append(report)
        
        # Keep history manageable
        if len(self.report_history) > 100:
            self.report_history.pop(0)
    
    def _display_object_report(self, report, frame):
        """Display object report on screen"""
        if not report:
            return frame
        
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, report['bbox'])
        
        # Draw bounding box with condition-based color
        condition_colors = {
            "excellent": (0, 255, 0),      # Green
            "good": (100, 255, 100),       # Light Green
            "fair": (255, 255, 0),         # Yellow
            "poor": (255, 0, 0),           # Red
            "unknown": (255, 165, 0)       # Orange
        }
        
        color = condition_colors.get(report['condition'], (255, 165, 0))
        thickness = 2
        
        # Draw main bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw corner markers
        marker_size = 10
        cv2.rectangle(frame, (x1, y1), (x1 + marker_size, y1 + marker_size), color, -1)
        cv2.rectangle(frame, (x2 - marker_size, y1), (x2, y1 + marker_size), color, -1)
        cv2.rectangle(frame, (x1, y2 - marker_size), (x1 + marker_size, y2), color, -1)
        cv2.rectangle(frame, (x2 - marker_size, y2 - marker_size), (x2, y2), color, -1)
        
        # Draw object label
        label = f"{report['object']} ({report['condition']})"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Background for label
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Label text
        cv2.putText(frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Value indicator
        value_text = f"Value: {report['estimated_value']}"
        value_size, _ = cv2.getTextSize(value_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y2), 
                     (x1 + value_size[0] + 10, y2 + value_size[1] + 10), color, -1)
        cv2.putText(frame, value_text, (x1 + 5, y2 + value_size[1] + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Confidence indicator
        conf_text = f"Conf: {report['analysis_confidence']:.0%}"
        cv2.putText(frame, conf_text, (x2 - 70, y1 - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def _draw_sidebar(self, frame, detections, reports, fps):
        """Draw sidebar with object information"""
        h, w = frame.shape[:2]
        sidebar_width = 350
        
        # Create sidebar background
        sidebar = np.zeros((h, sidebar_width, 3), dtype=np.uint8)
        sidebar[:, :] = (40, 40, 40)  # Dark gray
        
        # Sidebar title
        cv2.putText(sidebar, "JADE OBJECT ANALYSIS", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Session info
        cv2.putText(sidebar, f"Session: {self.session_id}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(sidebar, f"FPS: {fps:.1f}", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(sidebar, f"Objects: {len(detections)}", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Object list
        y_offset = 130
        cv2.putText(sidebar, "DETECTED OBJECTS:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        y_offset += 30
        
        for i, (det, report) in enumerate(zip(detections, reports)):
            if y_offset > h - 50:
                break
                
            # Object entry
            obj_text = f"{i+1}. {det['class_name']}: {report['estimated_value']}"
            color = (0, 255, 0) if report['condition'] == 'excellent' else \
                   (255, 255, 0) if report['condition'] == 'good' else \
                   (255, 165, 0) if report['condition'] == 'fair' else \
                   (255, 0, 0)
            
            cv2.putText(sidebar, obj_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Condition indicator
            cond_text = f"   [{report['condition'][0].upper()}]"
            cv2.putText(sidebar, cond_text, (sidebar_width - 50, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            y_offset += 20
        
        # Detailed report if object selected
        if self.selected_object is not None and self.display_mode == "detailed":
            if self.selected_object < len(reports):
                report = reports[self.selected_object]
                y_offset = h - 250
                
                cv2.putText(sidebar, "DETAILED REPORT:", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                y_offset += 25
                
                report_lines = format_report_for_display(report)
                for line in report_lines[:6]:  # Show first 6 lines
                    if y_offset > h - 20:
                        break
                    cv2.putText(sidebar, line, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    y_offset += 20
        
        # Commands help
        y_offset = h - 30
        cv2.putText(sidebar, "1-9:Select D:Detail Q:Quit", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Combine sidebar with main frame
        combined = np.hstack([frame, sidebar])
        
        return combined
    
    def _draw_status_overlay(self, frame):
        """Draw status overlay on main frame"""
        h, w = frame.shape[:2]
        
        # Top status bar
        status_bar = np.zeros((40, w, 3), dtype=np.uint8)
        status_bar[:, :] = (30, 30, 30)
        
        # Status text
        mode_text = f"Mode: {self.display_mode.upper()}"
        cv2.putText(status_bar, mode_text, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        if self.selected_object is not None:
            sel_text = f"Selected: Object {self.selected_object + 1}"
            cv2.putText(status_bar, sel_text, (w // 3, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # Combine with frame
        combined = np.vstack([status_bar, frame])
        
        return combined
    
    def run(self):
        """Main detection loop"""
        last_time = time.time()
        
        while True:
            start_time = time.time()
            
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Failed to read frame")
                time.sleep(0.1)
                continue
            
            self.frame_count += 1
            
            # Process frame
            processed_frame, detections = self.detector.detect(frame)
            
            # Analyze each object
            reports = []
            for det in detections:
                report = self._analyze_object(frame, det)
                reports.append(report)
                
                # Update display if object is selected
                if self.selected_object == detections.index(det):
                    self.display_text = format_report_for_display(report)
                    self.text_expiry = time.time() + 5  # Show for 5 seconds
            
            # Draw object annotations
            for report in reports:
                processed_frame = self._display_object_report(report, processed_frame)
            
            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - last_time)
            last_time = current_time
            self.fps_history.append(fps)
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
            
            avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else fps
            
            # Draw UI elements
            final_frame = self._draw_status_overlay(processed_frame)
            final_frame = self._draw_sidebar(final_frame, detections, reports, avg_fps)
            
            # Show frame
            cv2.imshow(config.WINDOW_NAME, final_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.display_mode = "detailed" if self.display_mode != "detailed" else "normal"
                print(f"📊 Display mode: {self.display_mode}")
            elif key == ord('c'):
                self.selected_object = None
                self.display_text = []
                print("🧹 Display cleared")
            elif key == ord('r'):
                self.object_reports.clear()
                print("🔄 Analysis cache cleared")
            elif key == ord('s'):
                stats = self.logger.export_statistics()
                print(f"📈 Stats saved: {len(self.report_history)} analyses")
            elif ord('1') <= key <= ord('9'):
                obj_num = key - ord('1')
                if obj_num < len(detections):
                    self.selected_object = obj_num
                    print(f"🎯 Selected object {obj_num + 1}: {detections[obj_num]['class_name']}")
            
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
        
        # Summary report
        print("\n" + "="*60)
        print("📊 JADE SESSION SUMMARY")
        print("="*60)
        
        if self.report_history:
            # Group by object type
            object_counts = {}
            total_value = 0
            
            for report in self.report_history:
                obj_type = report['object']
                object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
                
                # Extract numeric value from string like "$1,234.56"
                try:
                    value_str = report['estimated_value'].replace('$', '').replace(',', '')
                    value = float(value_str)
                    total_value += value
                except:
                    pass
            
            print(f"Total objects analyzed: {len(self.report_history)}")
            print(f"Unique object types: {len(object_counts)}")
            print(f"Estimated total value: ${total_value:,.2f}")
            print("\nTop objects detected:")
            for obj, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {obj}: {count} times")
        
        print(f"\nTotal frames processed: {self.frame_count}")
        
        if self.processing_times:
            avg_process_time = sum(self.processing_times) / len(self.processing_times)
            print(f"Average processing time: {avg_process_time*1000:.1f}ms")
        
        self.logger.log_system_event('SESSION_END', {
            'session_id': self.session_id,
            'total_frames': self.frame_count,
            'total_analyses': len(self.report_history),
            'duration': time.time() - self.start_time if hasattr(self, 'start_time') else 0
        })

def main():
    """Main entry point"""
    print("="*70)
    print("🤖 JADE UNIVERSAL ANALYZER v3.0")
    print("🔍 Intelligent analysis for EVERY detected object")
    print("="*70)
    
    try:
        jade = JADEUniversalAnalyzer()
        jade.start_time = time.time()
        jade.run()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ System error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if 'jade' in locals():
            jade.cleanup()
        print("\n✅ JADE shutdown complete")

if __name__ == "__main__":
    main()
