import cv2
import time
import numpy as np
from datetime import datetime
import uuid
import threading
import config
from Jade import JADEVoiceAssistant, JADEBaseAssistant
from outtxt import DetectionLogger
from knowledge_base import (
    analyze_object_visual_features,
    estimate_condition_from_features,
    generate_detailed_report,
    format_report_for_display
)
from JadeAssistant import JadeAssistant  # Import the detector

class JADEUniversalAnalyzer:
    def __init__(self):
        """Initialize JADE analyzer with voice capabilities"""
        self.session_id = str(uuid.uuid4())[:8]
        print(f"🚀 JADE Universal Analyzer Session: {self.session_id}")
        
        # Initialize logger
        self.logger = DetectionLogger()
        
        # Initialize detector
        self.detector = JadeAssistant(model_path=config.MODEL_PATH, 
                                    confidence=config.CONFIDENCE)
        
        # Initialize AI Assistant (using base class)
        self.assistant = JADEBaseAssistant()
        
        # Initialize Voice Assistant
        self.voice_assistant = JADEVoiceAssistant(
            self.assistant,
            wake_word=config.VOICE_SETTINGS['wake_word'],
            voice_gender=config.VOICE_SETTINGS['voice_gender'],
            speaking_rate=config.VOICE_SETTINGS['speaking_rate']
        )
        
        # Initialize camera
        self.camera_id = config.CAMERA_ID
        self.cap = self._initialize_camera()
        if not self.cap:
            return
        
        # State tracking
        self.voice_active = config.VOICE_SETTINGS['auto_start']
        self.show_chat = False
        self.chat_input = ""
        self.selected_object = None
        self.current_detections = []
        self.object_reports = {}
        
        print("✅ JADE Voice-Enabled Analyzer Initialized")
        print("🎤 Voice assistant ready - Say 'hey jade' to start!")
        print("📦 Using built-in knowledge base for object analysis")
    
    def _initialize_camera(self):
        """Initialize camera"""
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            for alt_id in [1, 2, 0]:
                cap = cv2.VideoCapture(alt_id)
                if cap.isOpened():
                    self.camera_id = alt_id
                    break
        
        if not cap.isOpened():
            print("❌ No camera available!")
            return None
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.PREVIEW_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.PREVIEW_HEIGHT)
        
        return cap
    
    def start_voice_assistant(self):
        """Start voice assistant"""
        if self.voice_active:
            self.voice_assistant.start_listening()
            return "🎤 Voice assistant started!"
        return "Voice is disabled. Press 'V' to enable."
    
    def stop_voice_assistant(self):
        """Stop voice assistant"""
        self.voice_assistant.stop_listening()
        return "🎤 Voice assistant stopped."
    
    def toggle_voice(self):
        """Toggle voice on/off"""
        self.voice_active = not self.voice_active
        
        if self.voice_active:
            return self.start_voice_assistant()
        else:
            return self.stop_voice_assistant()
    
    def _process_frame(self, frame):
        """Process a single frame"""
        processed_frame, detections = self.detector.detect(frame)
        self.current_detections = detections
        
        # Log detection
        detection_data = {
            'detections': detections,
            'frame_width': frame.shape[1],
            'frame_height': frame.shape[0],
            'fps': 0,  # Will be calculated elsewhere
            'session_id': self.session_id
        }
        self.logger.log_detection(detection_data)
        
        # Analyze detected objects
        reports = []
        for det in detections:
            report = self._analyze_object(frame, det)
            reports.append(report)
        
        return processed_frame, detections, reports
    
    def _analyze_object(self, frame, detection):
        """Analyze a detected object"""
        object_type = detection['class_name']
        confidence = detection['confidence']
        bbox = detection['bbox']
        
        # Analyze visual features
        features = analyze_object_visual_features(frame, bbox, object_type)
        
        # Estimate condition
        condition, _ = estimate_condition_from_features(features, object_type)
        
        # Generate report
        report = generate_detailed_report(object_type, condition, features, confidence)
        report['bbox'] = bbox
        
        return report
    
    def _draw_interface(self, frame, detections, reports, fps):
        """Draw user interface"""
        h, w = frame.shape[:2]
        
        # Create sidebar
        sidebar_width = 300
        sidebar = np.zeros((h, sidebar_width, 3), dtype=np.uint8)
        sidebar[:, :] = (40, 40, 40)
        
        # Title
        voice_icon = "🎤" if self.voice_active else "🔇"
        cv2.putText(sidebar, f"JADE {voice_icon}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Status info
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
        if self.voice_active:
            status = "ACTIVE" if self.voice_assistant.is_awake else "LISTENING"
            color = (0, 255, 0) if self.voice_assistant.is_awake else (255, 255, 0)
            cv2.putText(sidebar, f"Voice: {status}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 30
            
            # Quick commands
            cv2.putText(sidebar, "Say 'hey jade':", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_offset += 20
            
            commands = ["analyze object", "what do you see", "switch mode"]
            for cmd in commands:
                cv2.putText(sidebar, f"• {cmd}", (15, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                y_offset += 18
        
        # Object list with value estimates
        if detections and reports:
            y_offset += 10
            cv2.putText(sidebar, "Detected Objects:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            y_offset += 25
            
            for i, (det, report) in enumerate(zip(detections[:4], reports[:4])):
                if y_offset > h - 50:
                    break
                
                # Truncate object name if too long
                obj_name = det['class_name']
                if len(obj_name) > 12:
                    obj_name = obj_name[:12] + "..."
                
                # Get value from report
                value_text = "Unknown"
                if 'estimated_value' in report:
                    value_text = report['estimated_value']
                
                obj_text = f"{i+1}. {obj_name}"
                value_text_display = f"  {value_text}"
                
                cv2.putText(sidebar, obj_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw value in green if available
                if value_text != "Unknown":
                    cv2.putText(sidebar, value_text_display, (120, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                y_offset += 25
        
        # Help text
        cv2.putText(sidebar, "V:Voice T:Chat Q:Quit", (10, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Combine with frame
        combined = np.hstack([frame, sidebar])
        
        return combined
    
    def run(self):
        """Main application loop"""
        last_time = time.time()
        fps_history = []
        
        # Start voice assistant
        if self.voice_active:
            self.start_voice_assistant()
        
        while True:
            start_time = time.time()
            
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Camera error")
                break
            
            # Process frame
            processed_frame, detections, reports = self._process_frame(frame)
            
            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - last_time)
            last_time = current_time
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            
            avg_fps = sum(fps_history) / len(fps_history) if fps_history else fps
            
            # Draw interface
            display_frame = self._draw_interface(processed_frame, detections, reports, avg_fps)
            
            # Show frame
            cv2.imshow('JADE - Voice Enabled Object Analyzer', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('v'):
                response = self.toggle_voice()
                print(response)
            elif key == ord('t'):
                # Toggle text chat
                self.show_chat = not self.show_chat
                if self.show_chat:
                    print("💬 Chat mode enabled. Type in console.")
                    print("Type 'exit' to return to camera view.")
                    # Simple console chat
                    while self.show_chat:
                        try:
                            user_input = input("You: ")
                            if user_input.lower() == 'exit':
                                self.show_chat = False
                                print("💬 Chat mode disabled.")
                                break
                            response = self.assistant.chat(user_input)
                            print(f"JADE: {response}")
                        except KeyboardInterrupt:
                            self.show_chat = False
                            print("\n💬 Chat mode disabled.")
                            break
                else:
                    print("💬 Chat mode disabled.")
            elif key == ord('s'):
                # Record audio sample
                if self.voice_active:
                    print("🎙️ Recording 3 seconds of audio...")
                    audio_data = self.voice_assistant.record_audio_numpy(duration=3)
                    print(f"✅ Recorded {len(audio_data)} samples")
            elif key == ord('1') and self.voice_active:
                # Test voice
                self.voice_assistant.speak("Voice test successful!")
            elif key == ord('h'):
                # Show help
                print("\n" + "="*50)
                print("JADE HELP")
                print("="*50)
                print("Voice Commands:")
                print("  • 'Hey jade' - Wake phrase")
                print("  • 'Analyze object' - Analyze current view")
                print("  • 'What do you see' - Describe scene")
                print("  • 'Switch to analysis mode' - Detailed object analysis")
                print("  • 'Switch to conversational mode' - Chat mode")
                print("\nKeyboard Shortcuts:")
                print("  • V: Toggle voice")
                print("  • T: Toggle text chat")
                print("  • S: Record audio sample")
                print("  • H: Show this help")
                print("  • Q: Quit")
                print("="*50)
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'cap'):
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        if hasattr(self, 'voice_assistant'):
            self.voice_assistant.stop_listening()
            self.voice_assistant.cleanup()
        
        print("\n✅ JADE shutdown complete")

def main():
    """Main entry point"""
    print("="*60)
    print("🤖 JADE VOICE-ENABLED OBJECT ANALYZER")
    print("="*60)
    print("\n🎤 Voice Commands:")
    print("  • 'Hey jade' - Wake phrase")
    print("  • 'Analyze object' - Analyze current view")
    print("  • 'What do you see' - Describe scene")
    print("  • 'Switch to [mode]' - Change mode")
    print("\n⌨️  Keyboard Shortcuts:")
    print("  • V: Toggle voice")
    print("  • T: Toggle text chat")
    print("  • S: Record audio sample")
    print("  • H: Show help")
    print("  • Q: Quit")
    print("="*60)
    print("📦 Using built-in knowledge base (no API required)")
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
