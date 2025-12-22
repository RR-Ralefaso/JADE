import cv2
import time
import numpy as np
from datetime import datetime
import json
import os
import threading
from config import config
from JadeAssistant import JadeAssistant
from knowledge_base import knowledge_base, generate_detailed_report, analyze_object_deep_features
from outtxt import DetectionLogger
from Jade import JADEBaseAssistant, JADEVoiceAssistant

class JADEUniversalAnalyzer:
    def __init__(self):
        """Initialize JADE analyzer with JARVIS-like capabilities"""
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
        
        # Initialize camera with optimized settings
        self.cap = self._initialize_camera_optimized()
        if not self.cap:
            raise RuntimeError("Failed to initialize camera")
        
        # Enhanced state
        self.voice_enabled = True  # Voice enabled by default
        self.show_sidebar = True
        self.frame_count = 0
        self.start_time = time.time()
        self.last_analysis_time = 0
        self.analysis_interval = 3  # Seconds between automatic analysis
        self.auto_analysis_enabled = True
        
        # Detection cache for performance
        self.last_detections = []
        self.last_assessments = []
        self.detection_cache_time = 0
        
        # Start voice assistant immediately
        if self.voice_enabled:
            self.voice_assistant.start_continuous_listening()
            print("✅ Voice assistant started with continuous listening")
        
        print("✅ JADE Initialized")
        print("👩 Female Voice: ACTIVE (Continuous Listening)")
        print("🔍 Automatic Object Analysis: ENABLED")
        print("🎯 Point camera at objects for instant analysis")
    
    def _initialize_camera_optimized(self):
        """Initialize camera with optimized settings for speed"""
        cap = cv2.VideoCapture(config.CAMERA_ID)
        
        if not cap.isOpened():
            for cam_id in [1, 2, 0]:
                cap = cv2.VideoCapture(cam_id)
                if cap.isOpened():
                    print(f"✅ Using camera {cam_id}")
                    config.CAMERA_ID = cam_id
                    break
        
        if not cap.isOpened():
            print("❌ No camera available")
            return None
        
        # Optimize camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.PREVIEW_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.PREVIEW_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Small buffer for lower latency
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"📷 Camera: {actual_width}x{actual_height} @ {actual_fps}FPS")
        
        return cap
    
    def _process_frame_optimized(self, frame):
        """Optimized frame processing with smart detection"""
        # Apply frame skipping for better performance
        if self.frame_count % config.FRAME_SKIP != 0:
            # Return cached results if available
            if time.time() - self.detection_cache_time < 0.5:
                return frame, self.last_detections, self.last_assessments
        
        # Store original frame for display
        display_frame = frame.copy()
        
        # Create processing frame (resized for speed)
        processing_height = 480
        processing_width = int(frame.shape[1] * (processing_height / frame.shape[0]))
        processing_frame = cv2.resize(frame, (processing_width, processing_height))
        
        # Run detection
        start_detect = time.time()
        processed_frame, detections = self.detector.detect(processing_frame)
        detect_time = time.time() - start_detect
        
        # Scale detections back to original frame size
        scale_x = frame.shape[1] / processing_width
        scale_y = frame.shape[0] / processing_height
        
        scaled_detections = []
        for det in detections:
            # Scale bounding box
            scaled_bbox = [
                int(det.bbox[0] * scale_x),
                int(det.bbox[1] * scale_y),
                int(det.bbox[2] * scale_x),
                int(det.bbox[3] * scale_y)
            ]
            
            # Create new detection with scaled bbox
            scaled_det = type('Detection', (), {
                'class_id': det.class_id,
                'class_name': det.class_name,
                'confidence': det.confidence,
                'bbox': scaled_bbox,
                'area': (scaled_bbox[2] - scaled_bbox[0]) * (scaled_bbox[3] - scaled_bbox[1]),
                'track_id': det.track_id,
                'centroid': det.centroid
            })()
            scaled_detections.append(scaled_det)
            
            # Draw on display frame
            color = self.detector.class_colors[det.class_id % len(self.detector.class_colors)]
            x1, y1, x2, y2 = scaled_bbox
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det.class_name} {det.confidence:.2f}"
            if det.track_id:
                label = f"ID:{det.track_id} {label}"
            
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            cv2.rectangle(display_frame, 
                         (x1, y1 - text_height - 10),
                         (x1 + text_width, y1),
                         color, -1)
            
            cv2.putText(display_frame, label,
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 255), 1)
        
        # Smart analysis - only analyze when needed
        object_assessments = []
        current_time = time.time()
        
        # Analyze objects if enough time passed since last analysis and auto analysis is enabled
        if (scaled_detections and self.auto_analysis_enabled and 
            (current_time - self.last_analysis_time > self.analysis_interval)):
            
            self.last_analysis_time = current_time
            
            # Analyze up to 3 main objects (highest confidence or largest)
            detections_to_analyze = sorted(
                scaled_detections, 
                key=lambda x: x.confidence * x.area, 
                reverse=True
            )[:3]
            
            for det in detections_to_analyze:
                try:
                    # Analyze with knowledge base
                    features = analyze_object_deep_features(
                        frame, det.bbox, det.class_name
                    )
                    
                    condition = features.get('condition_indicators', {}).get('overall_condition', 'unknown') if features else 'unknown'
                    assessment = generate_detailed_report(
                        det.class_name, 
                        condition,
                        features,
                        det.confidence
                    )
                    
                    # Add online suggestions
                    try:
                        online_info = self.assistant.get_online_suggestions(det.class_name)
                        assessment['online_suggestions'] = online_info[:2]
                    except:
                        assessment['online_suggestions'] = []
                    
                    assessment['bbox'] = det.bbox
                    assessment['confidence'] = det.confidence
                    object_assessments.append(assessment)
                    
                except Exception as e:
                    print(f"Analysis error: {e}")
                    continue
            
            # Speak analysis results if we have assessments
            if object_assessments and self.voice_enabled:
                # Use threading to avoid blocking
                threading.Thread(
                    target=self.voice_assistant.process_object_analysis,
                    args=(object_assessments,),
                    daemon=True
                ).start()
        
        # Cache results
        self.last_detections = scaled_detections
        self.last_assessments = object_assessments
        self.detection_cache_time = current_time
        
        # Log detection data
        detection_data = []
        for det in scaled_detections:
            detection_data.append({
                'class_name': det.class_name,
                'confidence': det.confidence,
                'bbox': det.bbox,
                'area': det.area
            })
        
        if detection_data:
            self.logger.log_detection({
                'detections': detection_data,
                'frame_width': frame.shape[1],
                'frame_height': frame.shape[0],
                'fps': 1.0 / max(detect_time, 0.001),
                'session_id': self.session_id
            })
        
        return display_frame, scaled_detections, object_assessments
    
    def _draw_enhanced_interface(self, frame, detections, assessments):
        """Draw enhanced user interface with real-time info"""
        h, w = frame.shape[:2]
        
        if not self.show_sidebar:
            return frame
        
        # Create sidebar
        sidebar_width = 350
        sidebar = np.zeros((h, sidebar_width, 3), dtype=np.uint8)
        sidebar[:, :] = (30, 30, 40)  # Dark blue theme
        
        # Title with voice status
        voice_status = "🔊 ACTIVE" if self.voice_enabled else "🔇 MUTED"
        analysis_status = "🔍 ON" if self.auto_analysis_enabled else "👁️ OFF"
        cv2.putText(sidebar, f"JADE", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(sidebar, f"Voice: {voice_status}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if self.voice_enabled else (100, 100, 100), 1)
        cv2.putText(sidebar, f"Auto-Analysis: {analysis_status}", (10, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200) if self.auto_analysis_enabled else (100, 100, 100), 1)
        
        # Performance info
        perf_stats = self.detector.get_performance_stats()
        fps = perf_stats.get('avg_fps', 0)
        inference_time = perf_stats.get('avg_inference_time', 0) * 1000
        
        y_offset = 120
        info_items = [
            (f"FPS: {fps:.1f}", (200, 200, 200)),
            (f"Latency: {inference_time:.1f}ms", (200, 200, 200)),
            (f"Objects: {len(detections)}", (200, 255, 200)),
            (f"Mode: {self.assistant.current_mode.upper()}", (255, 200, 100))
        ]
        
        for text, color in info_items:
            cv2.putText(sidebar, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 25
        
        y_offset += 10
        
        # Object analysis section
        if assessments:
            cv2.putText(sidebar, "CURRENT ANALYSIS:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            y_offset += 25
            
            for i, assessment in enumerate(assessments[:4]):  # Show up to 4 objects
                if y_offset > h - 100:
                    break
                
                # Object name and value
                obj_name = assessment['object'][:15]
                value = assessment.get('estimated_value', 'Unknown')
                condition = assessment.get('condition', 'Unknown').capitalize()
                confidence = assessment.get('confidence', 0)
                
                # Object line
                obj_text = f"• {obj_name}"
                if confidence > 0:
                    obj_text += f" ({confidence:.0%})"
                
                cv2.putText(sidebar, obj_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Value with color coding
                value_color = (0, 255, 0) if value != "Unknown" else (150, 150, 150)
                cv2.putText(sidebar, value, (sidebar_width - 100, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, value_color, 1)
                
                y_offset += 20
                
                # Condition
                condition_color = {
                    'excellent': (0, 255, 0),
                    'good': (100, 255, 100),
                    'fair': (255, 200, 0),
                    'poor': (255, 100, 0)
                }.get(condition.lower(), (200, 200, 200))
                
                cond_text = f"  Condition: {condition}"
                if 'condition_score' in assessment.get('visual_features', {}).get('condition_indicators', {}):
                    score = assessment['visual_features']['condition_indicators']['condition_score']
                    cond_text += f" ({score}/10)"
                
                cv2.putText(sidebar, cond_text, (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, condition_color, 1)
                
                y_offset += 25
        else:
            cv2_text = "Point camera at objects"
            text_size = cv2.getTextSize(cv2_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = (sidebar_width - text_size[0]) // 2
            
            cv2.putText(sidebar, cv2_text, (text_x, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)
            y_offset += 30
            
            cv2_text = "for automatic analysis"
            text_size = cv2.getTextSize(cv2_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = (sidebar_width - text_size[0]) // 2
            
            cv2.putText(sidebar, cv2_text, (text_x, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)
            y_offset += 40
        
        # Voice commands help
        y_offset = h - 140
        cv2.putText(sidebar, "VOICE COMMANDS:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        y_offset += 20
        
        commands = [
            "'Hey Jade' - Get attention",
            "'Analyze object' - Detailed analysis",
            "'What do you see' - Scene description",
            "'Switch mode' - Change operation mode"
        ]
        
        for i, cmd in enumerate(commands[:4]):
            cv2.putText(sidebar, cmd, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            y_offset += 18
        
        # Keyboard shortcuts
        shortcuts = [
            "V: Toggle voice",
            "M: Toggle analysis", 
            "L: Manual listen",
            "S: Screenshot",
            "Q: Quit"
        ]
        
        y_offset = h - 40
        for i, shortcut in enumerate(shortcuts):
            text_size = cv2.getTextSize(shortcut, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
            x_pos = 10 + (i % 2) * 170
            y_pos = y_offset + (i // 2) * 18
            
            cv2.putText(sidebar, shortcut, (x_pos, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        
        # Combine with frame
        combined = np.hstack([frame, sidebar])
        
        # Add FPS and status to main frame
        status_text = f"JADE - {fps:.0f} FPS"
        cv2.putText(combined, status_text, (w - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add mode indicator
        mode_text = f"MODE: {self.assistant.current_mode.upper()}"
        cv2.putText(combined, mode_text, (20, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
        
        return combined
    
    def run(self):
        """Enhanced main application loop"""
        print("\n" + "="*60)
        print("🎯 JADE  is running!")
        print("👩 Voice: Always Active (Like JARVIS)")
        print("🔍 Automatic Object Analysis: Every 3 seconds")
        print("\n📢 You can speak naturally - I'm always listening")
        print("🎤 Just say 'Hey Jade' to get my attention")
        print("\n⌨️  Keyboard Shortcuts:")
        print("   V: Toggle voice mute")
        print("   M: Toggle automatic analysis")
        print("   L: Manual listen (single command)")
        print("   S: Save screenshot")
        print("   H: Show help")
        print("   Q: Quit")
        print("="*60)
        
        last_frame_time = time.time()
        frame_times = []
        
        # Initial greeting
        if self.voice_enabled:
            self.voice_assistant.speak("JADE activated. I'm always listening. Point the camera at objects for analysis.")
        
        while True:
            try:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Camera error - trying to reconnect...")
                    time.sleep(1)
                    self.cap.release()
                    self.cap = self._initialize_camera_optimized()
                    if not self.cap:
                        break
                    continue
                
                # Calculate timing
                current_time = time.time()
                frame_time = current_time - last_frame_time
                last_frame_time = current_time
                frame_times.append(frame_time)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                
                # Process frame
                processed_frame, detections, assessments = self._process_frame_optimized(frame)
                
                # Draw interface
                display_frame = self._draw_enhanced_interface(processed_frame, detections, assessments)
                
                # Show frame
                cv2.imshow('JADE - Object Analyzer', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('v'):
                    self._toggle_voice_mute()
                elif key == ord('m'):
                    self._toggle_analysis_mute()
                elif key == ord('l'):
                    self._manual_listen()
                elif key == ord('h'):
                    self._show_enhanced_help()
                elif key == ord('s'):
                    self._save_screenshot(display_frame)
                elif key == ord('1'):
                    self.voice_assistant.test_voice()
                elif key == ord('2'):
                    # Force immediate analysis
                    self.last_analysis_time = 0
                    if assessments:
                        self.voice_assistant.process_object_analysis(assessments)
                    elif detections:
                        self.voice_assistant.speak(f"I see {len(detections)} objects. Point at a specific object for detailed analysis.")
                
                self.frame_count += 1
                
            except KeyboardInterrupt:
                print("\n🛑 Keyboard interrupt detected")
                break
            except Exception as e:
                print(f"❌ Main loop error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
    
    def _toggle_voice_mute(self):
        """Toggle voice assistant mute"""
        if self.voice_enabled:
            self.voice_assistant.stop_listening()
            self.voice_enabled = False
            print("🔇 Voice muted")
            cv2.displayOverlay('JADE - Object Analyzer', "Voice: MUTED", 1000)
        else:
            self.voice_assistant.start_continuous_listening()
            self.voice_enabled = True
            print("🔊 Voice activated")
            cv2.displayOverlay('JADE - Object Analyzer', "Voice: ACTIVE", 1000)
    
    def _toggle_analysis_mute(self):
        """Toggle automatic analysis"""
        self.auto_analysis_enabled = not self.auto_analysis_enabled
        status = "ON" if self.auto_analysis_enabled else "OFF"
        print(f"🔍 Automatic analysis: {status}")
        
        overlay_text = f"Auto-Analysis: {status}"
        cv2.displayOverlay('JADE - Object Analyzer', overlay_text, 1000)
        
        if self.voice_enabled:
            self.voice_assistant.speak(f"Automatic analysis {status}")
    
    def _manual_listen(self):
        """Manual listen for command"""
        if not self.voice_enabled:
            print("⚠️  Voice assistant is disabled. Press 'V' to enable.")
            return
        
        print("🎤 Manual listening activated...")
        text = self.voice_assistant.listen_once()
        
        if text:
            # Process the command
            command = type('Command', (), {'text': text, 'confidence': 1.0, 'timestamp': time.time()})()
            self.voice_assistant._handle_command_advanced(command)
        else:
            if self.voice_enabled:
                self.voice_assistant.speak("I didn't catch that. Please try again.")
    
    def _show_enhanced_help(self):
        """Show enhanced help information"""
        help_text = """
        JADE - HELP GUIDE
        
        VOICE FEATURES (Always Active):
          • "Hey Jade" - Get my attention
          • "What do you see" - Scene description
          • "Analyze object" - Detailed analysis
          • "What's this" - Object identification
          • "Switch to analysis mode" - Detailed mode
          • "Switch to conversational mode" - Chat mode
          • "Go to sleep" - Mute voice temporarily
        
        AUTOMATIC ANALYSIS:
          • Objects analyzed every 3 seconds
          • Condition assessment with value estimation
          • Material identification
          • Online information lookup
        
        KEYBOARD SHORTCUTS:
          • V: Toggle voice on/off
          • M: Toggle automatic analysis
          • L: Manual listen (single command)
          • S: Save screenshot
          • 1: Test voice output
          • 2: Force immediate analysis
          • H: Show this help
          • Q: Quit application
        
        TIPS FOR BEST RESULTS:
          1. Good lighting improves analysis
          2. Center objects in frame
          3. Speak clearly at normal volume
          4. Allow 3 seconds for automatic analysis
        
        I'm your personal AI assistant - always ready to help!
        """
        
        print(help_text)
        
        if self.voice_enabled:
            self.voice_assistant.speak("Help information displayed. You can ask me anything!")
    
    def _save_screenshot(self, frame):
        """Save screenshot with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create exports directory if it doesn't exist
        os.makedirs('exports', exist_ok=True)
        
        filename = f"exports/jade_screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"✅ Screenshot saved: {filename}")
        
        if self.voice_enabled:
            self.voice_assistant.speak("Screenshot saved successfully.")
    
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
        
        # Export enhanced session report
        try:
            total_time = time.time() - self.start_time
            avg_fps = self.frame_count / total_time if total_time > 0 else 0
            
            final_report = {
                'session_id': self.session_id,
                'duration_seconds': total_time,
                'total_frames': self.frame_count,
                'average_fps': avg_fps,
                'voice_active': self.voice_enabled,
                'analysis_mode': self.assistant.current_mode,
                'auto_analysis': self.auto_analysis_enabled,
                'end_time': datetime.now().isoformat(),
                'performance_stats': self.detector.get_performance_stats(),
                'objects_detected': len(set([d.class_name for d in self.last_detections])) if self.last_detections else 0
            }
            
            # Create reports directory if it doesn't exist
            os.makedirs('reports', exist_ok=True)
            
            report_file = f'reports/session_{self.session_id}.json'
            with open(report_file, 'w') as f:
                json.dump(final_report, f, indent=2)
            
            print(f"📄 Session report saved: {report_file}")
            
        except Exception as e:
            print(f"❌ Error saving report: {e}")
        
        print("✅ JADE shutdown complete")

def main():
    """Main entry point"""
    print("="*60)
    print("🤖 JADE ")
    print("="*60)
    print("\n👩 VOICE ASSISTANT FEATURES:")
    print("  • Female voice (always active)")
    print("  • Continuous listening ")
    print("  • Natural conversation")
    print("  • Automatic object analysis")
    print("  • Online information lookup")
    
    print("\n🔍 OBJECT ANALYSIS CAPABILITIES:")
    print("  • Real-time detection & tracking")
    print("  • Condition assessment")
    print("  • Value estimation")
    print("  • Material identification")
    print("  • Maintenance suggestions")
    
    print("\n🎤 HOW TO USE:")
    print("  1. Just speak naturally - I'm always listening")
    print("  2. Say 'Hey Jade' to get my attention")
    print("  3. Point camera at objects for automatic analysis")
    print("  4. Ask questions about what you see")
    
    print("\n⌨️  KEYBOARD SHORTCUTS:")
    print("  V: Toggle voice  M: Toggle analysis  L: Manual listen")
    print("  S: Screenshot    H: Help             Q: Quit")
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