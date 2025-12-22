import cv2
import time
import numpy as np
from datetime import datetime
import json
import os
from config import config
from JadeAssistant import JadeAssistant
from knowledge_base import knowledge_base, generate_detailed_report, analyze_object_visual_features
from outtxt import DetectionLogger
from Jade import JADEBaseAssistant, JADEVoiceAssistant

class JADEUniversalAnalyzer:
    def __init__(self):
        """Initialize JADE analyzer"""
        self.session_id = f"session_{int(time.time())}"
        print(f"🚀 JADE Universal Analyzer Session: {self.session_id}")
        
        # Initialize components
        self.logger = DetectionLogger()
        self.detector = JadeAssistant(
            model_path=config.MODEL_PATH,
            confidence=config.CONFIDENCE
        )
        
        self.assistant = JADEBaseAssistant()
        self.voice_assistant = JADEVoiceAssistant(
            self.assistant,
            wake_word=config.WAKE_WORD,
            voice_gender=config.VOICE_GENDER,
            speaking_rate=config.SPEAKING_RATE
        )
        
        # Initialize camera
        self.cap = self._initialize_camera()
        if not self.cap:
            raise RuntimeError("Failed to initialize camera")
        
        # State
        self.voice_enabled = False  # Start with voice disabled to avoid audio issues
        self.show_sidebar = True
        self.frame_count = 0
        self.start_time = time.time()
        
        print("✅ JADE Object Analyzer Initialized")
        print("⚠️  Voice disabled by default (press 'V' to enable)")
        print("🎯 Point camera at objects to analyze them")
    
    def _initialize_camera(self):
        """Initialize camera"""
        cap = cv2.VideoCapture(config.CAMERA_ID)
        
        if not cap.isOpened():
            for cam_id in [1, 2, 0]:
                cap = cv2.VideoCapture(cam_id)
                if cap.isOpened():
                    print(f"✅ Using camera {cam_id}")
                    break
        
        if not cap.isOpened():
            print("❌ No camera available")
            return None
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.PREVIEW_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.PREVIEW_HEIGHT)
        
        return cap
    
    def _process_frame(self, frame):
        """Process a single frame"""
        # Run detection
        display_frame, detections = self.detector.detect(frame)
        
        # Analyze detected objects
        object_assessments = []
        for det in detections[:5]:  # Limit to 5 for performance
            try:
                features = analyze_object_visual_features(
                    frame, det.bbox, det.class_name
                )
                condition = features.get('condition_indicators', {}).get('overall_condition', 'unknown') if features else 'unknown'
                assessment = generate_detailed_report(
                    det.class_name, 
                    condition,
                    features,
                    det.confidence
                )
                assessment['bbox'] = det.bbox
                object_assessments.append(assessment)
            except Exception as e:
                continue
        
        # Log detection
        detection_data = []
        for det in detections:
            detection_data.append({
                'class_name': det.class_name,
                'confidence': det.confidence,
                'bbox': det.bbox,
                'area': det.area
            })
        
        self.logger.log_detection({
            'detections': detection_data,
            'frame_width': frame.shape[1],
            'frame_height': frame.shape[0],
            'fps': 0,
            'session_id': self.session_id
        })
        
        return display_frame, detections, object_assessments
    
    def _draw_interface(self, frame, detections, assessments):
        """Draw user interface"""
        h, w = frame.shape[:2]
        
        if not self.show_sidebar:
            return frame
        
        # Create sidebar
        sidebar_width = 300
        sidebar = np.zeros((h, sidebar_width, 3), dtype=np.uint8)
        sidebar[:, :] = (40, 40, 40)
        
        # Title
        voice_icon = "🎤" if self.voice_enabled else "🔇"
        cv2.putText(sidebar, f"JADE {voice_icon}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Performance info
        perf_stats = self.detector.get_performance_stats()
        fps = perf_stats.get('avg_fps', 0)
        
        y_offset = 60
        cv2.putText(sidebar, f"FPS: {fps:.1f}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 25
        
        cv2.putText(sidebar, f"Objects: {len(detections)}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 25
        
        cv2.putText(sidebar, f"Mode: {self.assistant.current_mode}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 40
        
        # Voice status
        if self.voice_enabled:
            cv2.putText(sidebar, f"Voice: READY", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 25
            cv2.putText(sidebar, "Press 'L' to listen", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            y_offset += 20
        
        # Object list
        if assessments:
            y_offset += 10
            cv2.putText(sidebar, "Detected Objects:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            y_offset += 25
            
            for i, assessment in enumerate(assessments[:6]):
                if y_offset > h - 50:
                    break
                
                obj_name = assessment['object'][:12] if len(assessment['object']) > 12 else assessment['object']
                value = assessment.get('estimated_value', 'Unknown')
                
                cv2.putText(sidebar, f"{i+1}. {obj_name}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                if value != "Unknown":
                    cv2.putText(sidebar, value, (sidebar_width - 80, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                y_offset += 25
        
        # Help text
        cv2.putText(sidebar, "V:Voice L:Listen H:Help Q:Quit", (10, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Combine with frame
        combined = np.hstack([frame, sidebar])
        
        # Add FPS to main frame
        cv2.putText(combined, f"FPS: {fps:.1f}", (w - 120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return combined
    
    def run(self):
        """Main application loop"""
        print("\n🎯 JADE is running! Press:")
        print("   V: Toggle voice")
        print("   L: Listen for command (when voice enabled)")
        print("   H: Show help")
        print("   S: Save screenshot")
        print("   Q: Quit")
        
        last_frame_time = time.time()
        
        while True:
            try:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Camera error")
                    break
                
                # Calculate timing
                current_time = time.time()
                frame_time = current_time - last_frame_time
                last_frame_time = current_time
                
                # Process frame
                processed_frame, detections, assessments = self._process_frame(frame)
                
                # Draw interface
                display_frame = self._draw_interface(processed_frame, detections, assessments)
                
                # Show frame
                cv2.imshow('JADE - Object Analyzer', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('v'):
                    self._toggle_voice()
                elif key == ord('l'):
                    self._listen_command()
                elif key == ord('h'):
                    self._show_help()
                elif key == ord('s'):
                    self._save_screenshot(display_frame)
                elif key == ord('1'):
                    self.voice_assistant.test_voice()
                elif key == ord('2'):
                    # Force speak about detected objects
                    if assessments:
                        obj = assessments[0]
                        self.voice_assistant.speak(
                            f"I see a {obj['object']} with estimated value {obj['estimated_value']}"
                        )
                
                self.frame_count += 1
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Main loop error: {e}")
                time.sleep(0.1)
    
    def _toggle_voice(self):
        """Toggle voice assistant"""
        self.voice_enabled = not self.voice_enabled
        
        if self.voice_enabled:
            self.voice_assistant.start_listening()
            self.voice_assistant.speak("Voice assistant activated!")
        else:
            self.voice_assistant.stop_listening()
            self.voice_assistant.speak("Voice assistant deactivated.")
        
        print(f"Voice assistant: {'ON' if self.voice_enabled else 'OFF'}")
    
    def _listen_command(self):
        """Listen for a single command"""
        if not self.voice_enabled:
            print("⚠️  Voice assistant is disabled. Press 'V' to enable.")
            return
        
        print("🎤 Listening for command...")
        text = self.voice_assistant.listen_once()
        
        if text:
            # Process the command
            command = type('Command', (), {'text': text, 'confidence': 1.0, 'timestamp': time.time()})()
            self.voice_assistant._handle_command(command)
    
    def _show_help(self):
        """Show help information"""
        help_text = """
        JADE OBJECT ANALYZER - HELP
        
        Voice Commands (when voice enabled):
          • "Analyze object" - Analyze current view
          • "What do you see" - Describe scene
          • "Switch to analysis mode" - Detailed analysis
          • "Switch to conversational mode" - Chat mode
        
        Keyboard Shortcuts:
          • V: Toggle voice assistant
          • L: Listen for command (when voice enabled)
          • H: Show this help
          • S: Save screenshot
          • 1: Test voice output
          • 2: Speak about detected object
          • Q: Quit application
        
        Tips for Best Performance:
          1. Ensure good lighting
          2. Keep objects centered in frame
          3. Speak clearly and at moderate pace
        """
        
        print(help_text)
        
        if self.voice_enabled:
            self.voice_assistant.speak("Help information displayed.")
    
    def _save_screenshot(self, frame):
        """Save screenshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create exports directory if it doesn't exist
        os.makedirs('exports', exist_ok=True)
        
        filename = f"exports/screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"✅ Screenshot saved: {filename}")
        
        if self.voice_enabled:
            self.voice_assistant.speak("Screenshot saved.")
    
    def cleanup(self):
        """Cleanup resources"""
        print("\n🛑 Shutting down JADE...")
        
        if hasattr(self, 'cap'):
            self.cap.release()
        
        if hasattr(self, 'voice_assistant'):
            self.voice_assistant.cleanup()
        
        if hasattr(self, 'logger'):
            self.logger.stop()
        
        cv2.destroyAllWindows()
        
        # Export final report
        try:
            total_time = time.time() - self.start_time
            avg_fps = self.frame_count / total_time if total_time > 0 else 0
            
            final_report = {
                'session_id': self.session_id,
                'duration_seconds': total_time,
                'total_frames': self.frame_count,
                'average_fps': avg_fps,
                'end_time': datetime.now().isoformat()
            }
            
            # Create reports directory if it doesn't exist
            os.makedirs('reports', exist_ok=True)
            
            with open(f'reports/session_{self.session_id}.json', 'w') as f:
                json.dump(final_report, f, indent=2)
            
            print(f"📄 Session report saved")
            
        except Exception as e:
            print(f"❌ Error saving report: {e}")
        
        print("✅ JADE shutdown complete")

def main():
    """Main entry point"""
    print("="*60)
    print("🤖 JADE OBJECT ANALYZER")
    print("="*60)
    print("\n🎤 Voice Features (optional):")
    print("  • Press 'V' to enable voice")
    print("  • Press 'L' to listen for command")
    print("  • Press '1' to test voice output")
    print("\n⌨️  Keyboard Shortcuts:")
    print("  • V: Toggle voice")
    print("  • L: Listen for command")
    print("  • H: Show help")
    print("  • S: Save screenshot")
    print("  • Q: Quit")
    print("="*60)
    print("📦 Using built-in knowledge base")
    print("🎯 Point camera at objects to analyze them")
    print("="*60)
    
    try:
        analyzer = JADEUniversalAnalyzer()
        analyzer.run()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'analyzer' in locals():
            analyzer.cleanup()

if __name__ == "__main__":
    main()