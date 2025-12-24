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
        """Initialize JADE analyzer with enhanced GUI and performance tracking"""
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
        self.voice_enabled = True
        self.show_sidebar = True
        self.frame_count = 0
        self.start_time = time.time()
        self.last_analysis_time = 0
        self.analysis_interval = 3
        self.auto_analysis_enabled = True
        
        # GUI enhancement
        self.sidebar_width = 350
        self.main_width = config.PREVIEW_WIDTH
        self.total_width = self.main_width + self.sidebar_width
        self.height = config.PREVIEW_HEIGHT
        
        # Performance tracking
        self.fps_history = []
        self.object_count_history = []
        self.confidence_history = []
        self.last_fps_update = time.time()
        self.fps = 0
        
        # Detection cache for performance
        self.last_detections = []
        self.last_assessments = []
        self.detection_cache_time = 0
        
        # Enhanced GUI colors (modern dark theme)
        self.colors = {
            'primary': (0, 150, 255),      # Blue
            'secondary': (255, 100, 0),    # Orange
            'success': (0, 200, 100),      # Green
            'warning': (255, 200, 0),      # Yellow
            'danger': (255, 50, 50),       # Red
            'dark': (15, 15, 20),          # Dark background
            'darker': (10, 10, 15),        # Darker background
            'light': (220, 220, 220),      # Light text
            'medium': (150, 150, 160),     # Medium text
            'card': (25, 25, 35),          # Card background
            'card_highlight': (40, 40, 50) # Card highlight
        }
        
        # Start voice assistant immediately
        if self.voice_enabled:
            self.voice_assistant.start_continuous_listening()
            print("✅ Voice assistant started with continuous listening")
        
        print("✅ JADE Initialized with Enhanced GUI")
        print("👩 Female Voice: ACTIVE (Continuous Listening)")
        print("📊 Performance Graphs: ENABLED")
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
    
    def _draw_enhanced_gui(self, frame, detections, assessments):
        """Draw enhanced GUI with modern design"""
        # Create main display with sidebar
        display = np.zeros((self.height, self.total_width, 3), dtype=np.uint8)
        display[:, :, :] = self.colors['dark']
        
        # Add camera frame
        if frame.shape[0] == self.height and frame.shape[1] == self.main_width:
            display[0:self.height, 0:self.main_width] = frame
        
        # Add sidebar
        self._draw_sidebar(display, detections, assessments)
        
        # Add top status bar
        self._draw_status_bar(display)
        
        # Add FPS counter
        self._draw_fps_counter(display)
        
        return display
    
    def _draw_sidebar(self, display, detections, assessments):
        """Draw enhanced sidebar"""
        sidebar_start = self.main_width
        
        # Sidebar background
        cv2.rectangle(display, 
                     (sidebar_start, 0),
                     (self.total_width, self.height),
                     self.colors['darker'], -1)
        
        # Sidebar header
        header_text = "JADE ANALYZER"
        cv2.putText(display, header_text, (sidebar_start + 20, 30),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, self.colors['primary'], 2)
        
        # Status indicators
        y_offset = 70
        status_items = [
            (f"Voice: {'🔊 ON' if self.voice_enabled else '🔇 OFF'}", 
             self.colors['success'] if self.voice_enabled else self.colors['medium']),
            (f"Analysis: {'🔍 ON' if self.auto_analysis_enabled else '👁️ OFF'}", 
             self.colors['primary'] if self.auto_analysis_enabled else self.colors['medium']),
            (f"Mode: {self.assistant.current_mode.upper()}", self.colors['secondary']),
            (f"FPS: {self.fps:.1f}", 
             self.colors['success'] if self.fps > 20 else self.colors['warning'] if self.fps > 10 else self.colors['danger'])
        ]
        
        for text, color in status_items:
            cv2.putText(display, text, (sidebar_start + 20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 25
        
        y_offset += 10
        
        # Performance stats
        cv2.putText(display, "PERFORMANCE:", (sidebar_start + 20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['light'], 1)
        y_offset += 25
        
        perf_stats = self.detector.get_performance_stats()
        perf_items = [
            (f"Frames: {self.frame_count}", self.colors['light']),
            (f"Objects: {len(detections)}", self.colors['light']),
            (f"Inference: {perf_stats.get('avg_inference_time', 0)*1000:.1f}ms", 
             self.colors['success'] if perf_stats.get('avg_inference_time', 0)*1000 < 30 else self.colors['warning'])
        ]
        
        for text, color in perf_items:
            cv2.putText(display, text, (sidebar_start + 20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            y_offset += 20
        
        y_offset += 10
        
        # Detected objects
        cv2.putText(display, "DETECTED OBJECTS:", (sidebar_start + 20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['light'], 1)
        y_offset += 25
        
        if detections:
            # Show top 5 objects
            for i, det in enumerate(detections[:5]):
                if y_offset > self.height - 100:
                    break
                
                obj_name = det.get('class_name', 'object')[:15]
                confidence = det.get('confidence', 0)
                
                # Color based on confidence
                color = self.colors['success'] if confidence > 0.7 else self.colors['warning'] if confidence > 0.5 else self.colors['danger']
                
                cv2.putText(display, f"• {obj_name}", (sidebar_start + 20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.colors['light'], 1)
                
                # Confidence bar
                bar_width = 60
                bar_x = sidebar_start + 120
                fill_width = int(bar_width * confidence)
                
                # Background
                cv2.rectangle(display, (bar_x, y_offset - 10), (bar_x + bar_width, y_offset - 3),
                             (50, 50, 50), -1)
                # Fill
                cv2.rectangle(display, (bar_x, y_offset - 10), (bar_x + fill_width, y_offset - 3),
                             color, -1)
                
                # Percentage
                cv2.putText(display, f"{confidence:.0%}", (bar_x + bar_width + 5, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                y_offset += 20
        else:
            cv2_text = "No objects detected"
            text_size = cv2.getTextSize(cv2_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            text_x = sidebar_start + (self.sidebar_width - text_size[0]) // 2
            cv2.putText(display, cv2_text, (text_x, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.colors['medium'], 1)
            y_offset += 25
        
        y_offset += 10
        
        # Analysis results
        if assessments:
            cv2.putText(display, "ANALYSIS:", (sidebar_start + 20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['light'], 1)
            y_offset += 25
            
            for i, assessment in enumerate(assessments[:3]):
                if y_offset > self.height - 50:
                    break
                
                obj_name = assessment.get('object', 'object')[:12]
                value = assessment.get('estimated_value', 'Unknown')
                condition = assessment.get('condition', 'unknown').capitalize()
                
                cv2.putText(display, f"📦 {obj_name}", (sidebar_start + 20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.colors['light'], 1)
                
                # Condition with color coding
                condition_color = {
                    'Excellent': self.colors['success'],
                    'Good': (100, 255, 100),
                    'Fair': self.colors['warning'],
                    'Poor': self.colors['danger']
                }.get(condition, self.colors['medium'])
                
                cv2.putText(display, condition, (sidebar_start + 100, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, condition_color, 1)
                
                # Value
                cv2.putText(display, value, (sidebar_start + 180, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['primary'], 1)
                
                y_offset += 20
        
        # Quick actions
        y_offset = self.height - 80
        cv2.putText(display, "QUICK ACTIONS:", (sidebar_start + 20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['light'], 1)
        y_offset += 25
        
        actions = [
            ("V", "Voice", self.colors['primary']),
            ("M", "Analysis", self.colors['secondary']),
            ("S", "Screenshot", self.colors['success']),
            ("P", "Graphs", self.colors['warning']),
            ("Q", "Quit", self.colors['danger'])
        ]
        
        for i, (key, label, color) in enumerate(actions):
            x_pos = sidebar_start + 20 + i * 60
            y_pos = y_offset
            
            # Button background
            cv2.rectangle(display, (x_pos, y_pos), (x_pos + 40, y_pos + 40), color, -1)
            
            # Key
            cv2.putText(display, key, (x_pos + 15, y_pos + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Label
            cv2.putText(display, label, (x_pos, y_pos + 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.colors['light'], 1)
    
    def _draw_status_bar(self, display):
        """Draw top status bar"""
        bar_height = 40
        
        # Draw bar background
        cv2.rectangle(display, (0, 0), (self.total_width, bar_height), self.colors['darker'], -1)
        
        # Logo
        cv2.putText(display, "JADE AI", (20, 25),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, self.colors['primary'], 2)
        
        # Session info
        session_text = f"Session: {self.session_id}"
        cv2.putText(display, session_text, (self.total_width - 200, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['medium'], 1)
        
        # Separator line
        cv2.line(display, (0, bar_height), (self.total_width, bar_height), self.colors['card'], 1)
    
    def _draw_fps_counter(self, display):
        """Draw FPS counter"""
        fps_text = f"{self.fps:.1f} FPS"
        fps_color = self.colors['success'] if self.fps > 20 else self.colors['warning'] if self.fps > 10 else self.colors['danger']
        
        # Draw background
        text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        bg_x = self.main_width - text_size[0] - 20
        bg_y = 10
        
        cv2.rectangle(display, (bg_x - 10, bg_y - 10), 
                     (bg_x + text_size[0] + 10, bg_y + text_size[1] + 10),
                     self.colors['darker'], -1)
        
        # Draw FPS text
        cv2.putText(display, fps_text, (bg_x, bg_y + text_size[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
    
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
            
            # Create detection dict
            scaled_det = {
                'class_id': det.class_id,
                'class_name': det.class_name,
                'confidence': det.confidence,
                'bbox': scaled_bbox,
                'area': (scaled_bbox[2] - scaled_bbox[0]) * (scaled_bbox[3] - scaled_bbox[1]),
                'track_id': det.track_id
            }
            scaled_detections.append(scaled_det)
            
            # Draw on display frame with enhanced visualization
            color = self.detector.class_colors[det.class_id % len(self.detector.class_colors)]
            x1, y1, x2, y2 = scaled_bbox
            
            # Draw glowing box effect
            for i in range(3):
                thickness = 2 - i
                alpha = 0.5 - i * 0.15
                temp_frame = display_frame.copy()
                cv2.rectangle(temp_frame, (x1-i, y1-i), (x2+i, y2+i), color, thickness)
                cv2.addWeighted(temp_frame, alpha, display_frame, 1-alpha, 0, display_frame)
            
            # Draw label with modern style
            label = f"{det.class_name} {det.confidence:.0%}"
            if det.track_id:
                label = f"ID:{det.track_id} {label}"
            
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Label background
            label_bg = np.array([self.colors['card']]) * 0.7
            cv2.rectangle(display_frame, 
                         (x1, y1 - text_height - 10),
                         (x1 + text_width + 10, y1),
                         label_bg.tolist()[0], -1)
            
            # Label text
            cv2.putText(display_frame, label,
                       (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       self.colors['light'], 2)
            
            # Confidence bar
            bar_width = 60
            bar_height = 4
            bar_x = x1
            bar_y = y2 + 5
            
            # Background
            cv2.rectangle(display_frame,
                         (bar_x, bar_y),
                         (bar_x + bar_width, bar_y + bar_height),
                         (50, 50, 50), -1)
            
            # Fill
            fill_width = int(bar_width * det.confidence)
            bar_color = self.colors['success'] if det.confidence > 0.7 else self.colors['warning'] if det.confidence > 0.5 else self.colors['danger']
            cv2.rectangle(display_frame,
                         (bar_x, bar_y),
                         (bar_x + fill_width, bar_y + bar_height),
                         bar_color, -1)
        
        # Update performance tracking
        current_time = time.time()
        if current_time - self.last_fps_update > 0.5:  # Update FPS every 0.5 seconds
            if detect_time > 0:
                self.fps = 1.0 / detect_time
                self.fps_history.append(self.fps)
            self.last_fps_update = current_time
        
        self.object_count_history.append(len(scaled_detections))
        if scaled_detections:
            avg_confidence = np.mean([d['confidence'] for d in scaled_detections])
            self.confidence_history.append(avg_confidence)
        
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
                key=lambda x: x['confidence'] * x['area'], 
                reverse=True
            )[:3]
            
            for det in detections_to_analyze:
                try:
                    # Analyze with knowledge base
                    features = analyze_object_deep_features(
                        frame, det['bbox'], det['class_name']
                    )
                    
                    condition = features.get('condition_indicators', {}).get('overall_condition', 'unknown') if features else 'unknown'
                    assessment = generate_detailed_report(
                        det['class_name'], 
                        condition,
                        features,
                        det['confidence']
                    )
                    
                    # Update performance data in assistant
                    self.assistant.update_performance_data(scaled_detections, detect_time)
                    
                    # Add online suggestions
                    try:
                        online_info = self.assistant.get_online_suggestions(det['class_name'])
                        assessment['online_suggestions'] = online_info[:2]
                    except:
                        assessment['online_suggestions'] = []
                    
                    assessment['bbox'] = det['bbox']
                    assessment['confidence'] = det['confidence']
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
                'class_name': det['class_name'],
                'confidence': det['confidence'],
                'bbox': det['bbox'],
                'area': det['area']
            })
        
        if detection_data:
            self.logger.log_detection({
                'detections': detection_data,
                'frame_width': frame.shape[1],
                'frame_height': frame.shape[0],
                'fps': self.fps,
                'session_id': self.session_id
            })
        
        return display_frame, scaled_detections, object_assessments
    
    def run(self):
        """Enhanced main application loop with performance graphs"""
        print("\n" + "="*60)
        print("🎯 JADE Enhanced is running!")
        print("👩 Voice: Always Active (Like JARVIS)")
        print("📊 Performance Graphs: ENABLED")
        print("🎨 Enhanced GUI: ACTIVE")
        print("\n📢 You can speak naturally - I'm always listening")
        print("🎤 Just say 'Hey Jade' to get my attention")
        print("\n⌨️  Enhanced Keyboard Shortcuts:")
        print("   V: Toggle voice mute")
        print("   M: Toggle automatic analysis")
        print("   L: Manual listen (single command)")
        print("   S: Save screenshot")
        print("   P: Generate performance graphs")
        print("   G: Show live performance overlay")
        print("   H: Show help")
        print("   Q: Quit")
        print("="*60)
        
        last_frame_time = time.time()
        
        # Initial greeting
        if self.voice_enabled:
            self.voice_assistant.speak("JADE Enhanced activated. I'm always listening. Point the camera at objects for analysis.")
        
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
                
                # Process frame
                processed_frame, detections, assessments = self._process_frame_optimized(frame)
                
                # Create enhanced display
                display_frame = self._draw_enhanced_gui(processed_frame, detections, assessments)
                
                # Show frame
                cv2.imshow('JADE Enhanced - Object Analyzer', display_frame)
                
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
                elif key == ord('p'):
                    self._generate_performance_graphs()
                elif key == ord('g'):
                    self._show_live_performance()
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
            cv2.displayOverlay('JADE Enhanced - Object Analyzer', "Voice: MUTED", 1000)
        else:
            self.voice_assistant.start_continuous_listening()
            self.voice_enabled = True
            print("🔊 Voice activated")
            cv2.displayOverlay('JADE Enhanced - Object Analyzer', "Voice: ACTIVE", 1000)
    
    def _toggle_analysis_mute(self):
        """Toggle automatic analysis"""
        self.auto_analysis_enabled = not self.auto_analysis_enabled
        status = "ON" if self.auto_analysis_enabled else "OFF"
        print(f"🔍 Automatic analysis: {status}")
        
        overlay_text = f"Auto-Analysis: {status}"
        cv2.displayOverlay('JADE Enhanced - Object Analyzer', overlay_text, 1000)
        
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
    
    def _generate_performance_graphs(self):
        """Generate and show performance graphs"""
        print("📊 Generating performance graphs...")
        
        # Generate reports from all components
        total_time = time.time() - self.start_time
        
        # 1. Assistant performance report
        assistant_report = self.assistant.generate_performance_report(
            self.session_id, self.frame_count, total_time
        )
        
        # 2. Detector performance report
        detector_report = self.detector.create_performance_report(self.session_id)
        
        # 3. Logger visualization
        logger_report = self.logger.create_visualization_report(self.session_id)
        
        # 4. Knowledge base performance (if available)
        try:
            kb_report = knowledge_base.create_performance_report(self.session_id)
        except:
            kb_report = None
        
        print("\n✅ Performance graphs generated:")
        print(f"   • Assistant: {assistant_report}")
        print(f"   • Detector: {detector_report}")
        print(f"   • Logger: {logger_report}")
        if kb_report:
            print(f"   • Knowledge Base: {kb_report}")
        
        if self.voice_enabled:
            self.voice_assistant.speak("Performance graphs generated and saved to reports folder.")
    
    def _show_live_performance(self):
        """Show live performance overlay"""
        print("📈 Showing live performance stats...")
        
        # Create simple performance overlay
        overlay_text = f"""
        Live Performance Stats:
        
        Frames Processed: {self.frame_count}
        Current FPS: {self.fps:.1f}
        Objects Detected: {len(self.last_detections)}
        Analysis Mode: {self.assistant.current_mode}
        
        Session Duration: {time.time() - self.start_time:.1f}s
        Voice: {'ACTIVE' if self.voice_enabled else 'MUTED'}
        Auto-Analysis: {'ON' if self.auto_analysis_enabled else 'OFF'}
        """
        
        print(overlay_text)
        
        if self.voice_enabled:
            self.voice_assistant.speak(f"Live performance: {self.fps:.0f} FPS, {len(self.last_detections)} objects detected.")
    
    def _show_enhanced_help(self):
        """Show enhanced help information"""
        help_text = """
        JADE ENHANCED - HELP GUIDE
        
        ENHANCED FEATURES:
          • Modern dark theme GUI
          • Real-time performance tracking
          • Automatic graph generation
          • Enhanced object visualization
          • Performance statistics overlay
        
        VOICE FEATURES (Always Active):
          • "Hey Jade" - Get my attention
          • "What do you see" - Scene description
          • "Analyze object" - Detailed analysis
          • "Show performance" - Display statistics
          • "Generate graphs" - Create visualizations
        
        PERFORMANCE GRAPHS:
          • FPS trends and analysis
          • Object detection statistics
          • Confidence distribution
          • Inference time monitoring
          • Session comparison
        
        KEYBOARD SHORTCUTS:
          • V: Toggle voice on/off
          • M: Toggle automatic analysis
          • L: Manual listen (single command)
          • S: Save screenshot
          • P: Generate performance graphs
          • G: Show live performance overlay
          • H: Show this help
          • Q: Quit application
        
        TIPS FOR BEST RESULTS:
          1. Good lighting improves analysis
          2. Center objects in frame
          3. Speak clearly at normal volume
          4. Allow 3 seconds for automatic analysis
          5. Press 'P' to generate performance graphs
        
        Performance graphs are saved to 'reports/' directory
        """
        
        print(help_text)
        
        if self.voice_enabled:
            self.voice_assistant.speak("Enhanced help information displayed. You can ask me anything or generate performance graphs!")
    
    def _save_screenshot(self, frame):
        """Save screenshot with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create exports directory if it doesn't exist
        os.makedirs('exports/screenshots', exist_ok=True)
        
        filename = f"exports/screenshots/jade_enhanced_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"✅ Screenshot saved: {filename}")
        
        if self.voice_enabled:
            self.voice_assistant.speak("Screenshot saved successfully.")
    
    def cleanup(self):
        """Cleanup resources and generate final report"""
        print("\n🛑 Shutting down JADE Enhanced...")
        
        # Stop components
        if hasattr(self, 'cap'):
            self.cap.release()
        
        if hasattr(self, 'voice_assistant'):
            self.voice_assistant.cleanup()
        
        if hasattr(self, 'logger'):
            self.logger.stop()
        
        cv2.destroyAllWindows()
        
        # Generate final performance report
        try:
            total_time = time.time() - self.start_time
            avg_fps = self.frame_count / total_time if total_time > 0 else 0
            
            final_report = {
                'session_id': self.session_id,
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': total_time,
                'total_frames': self.frame_count,
                'average_fps': avg_fps,
                'voice_active': self.voice_enabled,
                'analysis_mode': self.assistant.current_mode,
                'auto_analysis': self.auto_analysis_enabled,
                'performance_stats': self.detector.get_performance_stats(),
                'objects_detected': len(set([d['class_name'] for d in self.last_detections])) if self.last_detections else 0,
                'total_detections': sum([len(self.last_detections)]),
                'fps_history': list(self.fps_history)[-50:],  # Last 50 FPS values
                'object_count_history': list(self.object_count_history)[-50:],
                'confidence_history': list(self.confidence_history)[-50:]
            }
            
            # Create reports directory if it doesn't exist
            os.makedirs('reports/sessions', exist_ok=True)
            
            report_file = f'reports/sessions/session_{self.session_id}.json'
            with open(report_file, 'w') as f:
                json.dump(final_report, f, indent=2)
            
            print(f"📄 Session report saved: {report_file}")
            
            # Generate final graphs
            print("\n📊 Generating final performance graphs...")
            self._generate_performance_graphs()
            
            # Show summary
            print(f"\n📋 SESSION SUMMARY:")
            print(f"   Duration: {total_time:.1f} seconds")
            print(f"   Frames processed: {self.frame_count}")
            print(f"   Average FPS: {avg_fps:.1f}")
            print(f"   Objects detected: {final_report['objects_detected']}")
            print(f"   Performance graphs: reports/plots/")
            
        except Exception as e:
            print(f"❌ Error saving final report: {e}")
        
        print("✅ JADE Enhanced shutdown complete")

def main():
    """Main entry point"""
    print("="*60)
    print("🤖 JADE ENHANCED")
    print("="*60)
    print("\n✨ ENHANCED FEATURES:")
    print("  • Modern Dark Theme GUI")
    print("  • Real-time Performance Graphs")
    print("  • Automatic Report Generation")
    print("  • Enhanced Object Visualization")
    print("  • Session Statistics Tracking")
    
    print("\n🎯 CAPABILITIES:")
    print("  • Real-time object detection & tracking")
    print("  • Condition assessment & value estimation")
    print("  • Voice-controlled operation")
    print("  • Performance monitoring & visualization")
    print("  • Automatic report generation")
    
    print("\n🚀 STARTING ENHANCED VERSION...")
    
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