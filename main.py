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
        """Initialize JADE analyzer with optimized display"""
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
        
        # Get actual camera resolution
        self.camera_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.camera_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📷 Camera resolution: {self.camera_width}x{self.camera_height}")
        
        # Display dimensions - keep original camera resolution
        self.display_scale = 0.7  # Scale down for better performance
        self.display_width = int(self.camera_width * self.display_scale)
        self.display_height = int(self.camera_height * self.display_scale)
        
        self.sidebar_width = 350
        self.total_width = self.display_width + self.sidebar_width
        self.total_height = self.display_height
        
        # Enhanced state
        self.voice_enabled = True
        self.show_sidebar = True
        self.frame_count = 0
        self.start_time = time.time()
        self.last_analysis_time = 0
        self.analysis_interval = 3
        self.auto_analysis_enabled = True
        
        # Performance tracking
        self.fps_history = []
        self.object_count_history = []
        self.confidence_history = []
        self.last_fps_update = time.time()
        self.fps = 0
        self.last_frame_time = time.time()
        
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
        
        print("✅ JADE Initialized with Optimized Display")
        print("👩 Female Voice: ACTIVE (Continuous Listening)")
        print(f"📺 Display: {self.display_width}x{self.display_height}")
        print("🎯 Point camera at objects for instant analysis")
    
    def _initialize_camera_optimized(self):
        """Initialize camera with optimized settings for clarity"""
        cap = cv2.VideoCapture(config.CAMERA_ID)
        
        if not cap.isOpened():
            print(f"❌ Camera {config.CAMERA_ID} not accessible, trying alternatives...")
            for cam_id in [1, 2, 0, 3, 4]:
                cap = cv2.VideoCapture(cam_id)
                if cap.isOpened():
                    print(f"✅ Using camera {cam_id}")
                    config.CAMERA_ID = cam_id
                    break
        
        if not cap.isOpened():
            print("❌ No camera available")
            print("⚠️  Creating test pattern for debugging...")
            return None
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Try for higher resolution
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Better compression
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus if available
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
        cap.set(cv2.CAP_PROP_CONTRAST, 0.5)
        cap.set(cv2.CAP_PROP_SATURATION, 0.5)
        
        # Test if we can read a frame
        ret, test_frame = cap.read()
        if not ret:
            print("❌ Cannot read from camera")
            cap.release()
            return None
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✅ Camera initialized: {actual_width}x{actual_height}")
        
        return cap
    
    def _enhance_frame_quality(self, frame):
        """Enhance frame quality with denoising and sharpening"""
        if frame is None or frame.size == 0:
            return frame
        
        # Convert to float for processing
        frame_float = frame.astype(np.float32) / 255.0
        
        # 1. Denoise
        denoised = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        
        # 2. Adaptive histogram equalization for contrast
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # 3. Mild sharpening
        kernel = np.array([[0, -0.5, 0],
                          [-0.5, 3, -0.5],
                          [0, -0.5, 0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # 4. Adjust brightness and contrast
        alpha = 1.1  # Contrast control (1.0-3.0)
        beta = 10    # Brightness control (0-100)
        adjusted = cv2.convertScaleAbs(sharpened, alpha=alpha, beta=beta)
        
        return adjusted
    
    def _draw_enhanced_gui(self, frame, detections, assessments):
        """Draw enhanced GUI with clear camera display"""
        # Enhance frame quality
        enhanced_frame = self._enhance_frame_quality(frame)
        
        # Resize frame for display (maintain aspect ratio)
        display_frame = cv2.resize(enhanced_frame, (self.display_width, self.display_height), 
                                  interpolation=cv2.INTER_LINEAR)
        
        # Create main display with sidebar
        display = np.zeros((self.total_height, self.total_width, 3), dtype=np.uint8)
        display[:, :, :] = self.colors['dark']
        
        # Add camera frame to left side
        display[0:self.display_height, 0:self.display_width] = display_frame
        
        # Add sidebar
        self._draw_sidebar(display, detections, assessments)
        
        # Add top status bar
        self._draw_status_bar(display)
        
        # Add FPS counter
        self._draw_fps_counter(display)
        
        # Add grid overlay for better visual reference
        self._draw_grid_overlay(display)
        
        return display
    
    def _draw_grid_overlay(self, display):
        """Add subtle grid overlay for better visual reference"""
        grid_color = (40, 40, 50)
        grid_alpha = 0.3
        
        # Create grid overlay
        overlay = display.copy()
        
        # Vertical lines
        for x in range(0, self.display_width, 50):
            cv2.line(overlay, (x, 0), (x, self.display_height), grid_color, 1)
        
        # Horizontal lines
        for y in range(0, self.display_height, 50):
            cv2.line(overlay, (0, y), (self.display_width, y), grid_color, 1)
        
        # Center crosshair
        center_x = self.display_width // 2
        center_y = self.display_height // 2
        cv2.line(overlay, (center_x - 20, center_y), (center_x + 20, center_y), 
                (255, 100, 100), 2)
        cv2.line(overlay, (center_x, center_y - 20), (center_x, center_y + 20), 
                (255, 100, 100), 2)
        
        # Blend overlay
        cv2.addWeighted(overlay, grid_alpha, display, 1 - grid_alpha, 0, display)
    
    def _draw_sidebar(self, display, detections, assessments):
        """Draw enhanced sidebar"""
        sidebar_start = self.display_width
        
        # Sidebar background with subtle gradient
        for x in range(sidebar_start, self.total_width):
            alpha = (x - sidebar_start) / self.sidebar_width
            color = tuple(int(c * (0.8 + 0.2 * alpha)) for c in self.colors['darker'])
            cv2.line(display, (x, 0), (x, self.total_height), color, 1)
        
        # Sidebar header
        header_height = 50
        header_bg = np.zeros((header_height, self.sidebar_width, 3), dtype=np.uint8)
        header_bg[:, :, :] = self.colors['primary']
        
        # Add gradient to header
        for y in range(header_height):
            alpha = y / header_height
            color = tuple(int(c * (0.3 + 0.7 * alpha)) for c in self.colors['primary'])
            cv2.line(header_bg, (0, y), (self.sidebar_width, y), color, 1)
        
        display[0:header_height, sidebar_start:self.total_width] = header_bg
        
        # Sidebar header text
        header_text = "JADE ANALYZER"
        (text_width, text_height), _ = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
        text_x = sidebar_start + (self.sidebar_width - text_width) // 2
        cv2.putText(display, header_text, (text_x, 35),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        
        # Status indicators
        y_offset = header_height + 20
        status_items = [
            (f"🔊 Voice: {'ON' if self.voice_enabled else 'OFF'}", 
             self.colors['success'] if self.voice_enabled else self.colors['medium']),
            (f"🔍 Analysis: {'ON' if self.auto_analysis_enabled else 'OFF'}", 
             self.colors['primary'] if self.auto_analysis_enabled else self.colors['medium']),
            (f"🎯 Mode: {self.assistant.current_mode.upper()}", self.colors['secondary']),
            (f"⚡ FPS: {self.fps:.1f}", 
             self.colors['success'] if self.fps > 20 else self.colors['warning'] if self.fps > 10 else self.colors['danger'])
        ]
        
        for text, color in status_items:
            cv2.putText(display, text, (sidebar_start + 20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 25
        
        y_offset += 10
        
        # Performance stats card
        card_y = y_offset
        card_height = 90
        card_width = self.sidebar_width - 20
        
        # Card background
        cv2.rectangle(display, 
                     (sidebar_start + 10, card_y),
                     (sidebar_start + 10 + card_width, card_y + card_height),
                     self.colors['card'], -1)
        cv2.rectangle(display,
                     (sidebar_start + 10, card_y),
                     (sidebar_start + 10 + card_width, card_y + card_height),
                     self.colors['card_highlight'], 2)
        
        # Card title
        cv2.putText(display, "📊 PERFORMANCE", (sidebar_start + 20, card_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['light'], 1)
        
        # Performance stats
        perf_stats = self.detector.get_performance_stats()
        perf_items = [
            (f"Frames: {self.frame_count}", self.colors['light']),
            (f"Objects: {len(detections)}", self.colors['light']),
            (f"Inference: {perf_stats.get('avg_inference_time', 0)*1000:.1f}ms", 
             self.colors['success'] if perf_stats.get('avg_inference_time', 0)*1000 < 30 else self.colors['warning']),
            (f"Confidence: {np.mean([d.get('confidence', 0) for d in detections]):.0%}" if detections else "Confidence: N/A",
             self.colors['primary'])
        ]
        
        for i, (text, color) in enumerate(perf_items):
            cv2.putText(display, text, (sidebar_start + 20, card_y + 40 + i*15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        
        y_offset += card_height + 20
        
        # Detected objects card
        if y_offset < self.total_height - 200:
            max_card_height = self.total_height - y_offset - 70
            card_height = min(200, max_card_height)
            
            # Card background
            cv2.rectangle(display,
                         (sidebar_start + 10, y_offset),
                         (sidebar_start + 10 + card_width, y_offset + card_height),
                         self.colors['card'], -1)
            cv2.rectangle(display,
                         (sidebar_start + 10, y_offset),
                         (sidebar_start + 10 + card_width, y_offset + card_height),
                         self.colors['card_highlight'], 2)
            
            # Card title
            cv2.putText(display, "🎯 DETECTED OBJECTS", (sidebar_start + 20, y_offset + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['light'], 1)
            
            obj_y = y_offset + 40
            
            if detections:
                # Show detected objects with confidence bars
                for i, det in enumerate(detections[:6]):  # Show max 6 objects
                    if obj_y > y_offset + card_height - 20:
                        break
                    
                    obj_name = det.get('class_name', 'object')[:12]
                    confidence = det.get('confidence', 0)
                    
                    # Color based on confidence
                    color = self.colors['success'] if confidence > 0.7 else self.colors['warning'] if confidence > 0.5 else self.colors['danger']
                    
                    # Object name
                    cv2.putText(display, f"• {obj_name}", (sidebar_start + 20, obj_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.colors['light'], 1)
                    
                    # Confidence bar
                    bar_width = 60
                    bar_x = sidebar_start + 100
                    fill_width = int(bar_width * confidence)
                    
                    # Background
                    cv2.rectangle(display, (bar_x, obj_y - 10), 
                                 (bar_x + bar_width, obj_y - 3),
                                 (50, 50, 50), -1)
                    # Fill
                    cv2.rectangle(display, (bar_x, obj_y - 10), 
                                 (bar_x + fill_width, obj_y - 3),
                                 color, -1)
                    
                    # Percentage
                    cv2.putText(display, f"{confidence:.0%}", 
                               (bar_x + bar_width + 5, obj_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
                    obj_y += 20
            else:
                cv2_text = "No objects detected"
                text_size = cv2.getTextSize(cv2_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
                text_x = sidebar_start + (self.sidebar_width - text_size[0]) // 2
                cv2.putText(display, cv2_text, (text_x, obj_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.colors['medium'], 1)
        
        # Quick actions at bottom
        actions_y = self.total_height - 60
        cv2.putText(display, "⚡ QUICK ACTIONS", (sidebar_start + 20, actions_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['light'], 1)
        
        actions = [
            ("V", "Voice", self.colors['primary']),
            ("M", "Analysis", self.colors['secondary']),
            ("S", "Screenshot", self.colors['success']),
            ("P", "Graphs", self.colors['warning']),
            ("Q", "Quit", self.colors['danger'])
        ]
        
        button_size = 40
        button_spacing = 55
        
        for i, (key, label, color) in enumerate(actions):
            x_pos = sidebar_start + 20 + i * button_spacing
            y_pos = actions_y + 20
            
            # Button background with shadow
            cv2.rectangle(display, (x_pos + 2, y_pos + 2), 
                         (x_pos + button_size + 2, y_pos + button_size + 2),
                         (0, 0, 0), -1)
            cv2.rectangle(display, (x_pos, y_pos), 
                         (x_pos + button_size, y_pos + button_size),
                         color, -1)
            cv2.rectangle(display, (x_pos, y_pos), 
                         (x_pos + button_size, y_pos + button_size),
                         (255, 255, 255), 1)
            
            # Key
            (key_width, key_height), _ = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            key_x = x_pos + (button_size - key_width) // 2
            key_y = y_pos + (button_size + key_height) // 2
            
            cv2.putText(display, key, (key_x, key_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Label
            (label_width, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            label_x = x_pos + (button_size - label_width) // 2
            cv2.putText(display, label, (label_x, y_pos + button_size + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.colors['light'], 1)
    
    def _draw_status_bar(self, display):
        """Draw top status bar"""
        bar_height = 40
        
        # Draw bar background with gradient
        for y in range(bar_height):
            alpha = y / bar_height
            color = tuple(int(c * (0.2 + 0.8 * alpha)) for c in self.colors['darker'])
            cv2.line(display, (0, y), (self.total_width, y), color, 1)
        
        # Logo with glow effect
        logo_text = "JADE AI"
        for i in range(3, 0, -1):
            cv2.putText(display, logo_text, (20 + i, 25 + i),
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(display, logo_text, (20, 25),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, self.colors['primary'], 2)
        
        # Session info
        session_display = f"Session: {self.session_id[:8]}..."
        cv2.putText(display, session_display, (self.total_width - 200, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['medium'], 1)
        
        # Separator line
        cv2.line(display, (0, bar_height), (self.total_width, bar_height), 
                self.colors['card_highlight'], 2)
    
    def _draw_fps_counter(self, display):
        """Draw FPS counter"""
        fps_text = f"{self.fps:.1f} FPS"
        
        # Determine color based on FPS
        if self.fps > 25:
            fps_color = self.colors['success']
        elif self.fps > 15:
            fps_color = self.colors['warning']
        else:
            fps_color = self.colors['danger']
        
        # Draw background with rounded corners effect
        text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        bg_x = self.display_width - text_size[0] - 30
        bg_y = 10
        bg_width = text_size[0] + 20
        bg_height = text_size[1] + 10
        
        # Background with shadow
        cv2.rectangle(display, (bg_x + 2, bg_y + 2), 
                     (bg_x + bg_width + 2, bg_y + bg_height + 2),
                     (0, 0, 0), -1)
        
        # Main background
        cv2.rectangle(display, (bg_x, bg_y), 
                     (bg_x + bg_width, bg_y + bg_height),
                     self.colors['darker'], -1)
        cv2.rectangle(display, (bg_x, bg_y), 
                     (bg_x + bg_width, bg_y + bg_height),
                     fps_color, 2)
        
        # FPS text
        cv2.putText(display, fps_text, (bg_x + 10, bg_y + text_size[1] + 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _process_frame_optimized(self, frame):
        """Optimized frame processing maintaining quality"""
        if frame is None or frame.size == 0:
            return None, [], []
        
        # Apply frame skipping for better performance
        current_time = time.time()
        if self.frame_count % config.FRAME_SKIP != 0:
            # Return cached results if available
            if current_time - self.detection_cache_time < 0.5:
                return frame, self.last_detections, self.last_assessments
        
        # Store original frame for display
        display_frame = frame.copy()
        
        # Calculate FPS
        frame_time = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        if frame_time > 0:
            self.fps = 1.0 / frame_time
            self.fps_history.append(self.fps)
            
            # Keep FPS history manageable
            if len(self.fps_history) > 100:
                self.fps_history.pop(0)
        
        # Create processing frame (maintain reasonable size for speed)
        processing_height = 640  # Fixed size for consistent processing
        processing_width = int(frame.shape[1] * (processing_height / frame.shape[0]))
        
        # Use high-quality interpolation for resizing
        processing_frame = cv2.resize(frame, (processing_width, processing_height), 
                                     interpolation=cv2.INTER_LANCZOS4)
        
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
            
            # Draw on display frame with clear visualization
            color = self.detector.class_colors[det.class_id % len(self.detector.class_colors)]
            x1, y1, x2, y2 = scaled_bbox
            
            # Draw bounding box with shadow for better visibility
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with clear background
            label = f"{det.class_name} {det.confidence:.0%}"
            if det.track_id:
                label = f"ID:{det.track_id} {label}"
            
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Label background
            cv2.rectangle(display_frame, 
                         (x1, y1 - text_height - 10),
                         (x1 + text_width + 10, y1),
                         (0, 0, 0), -1)
            cv2.rectangle(display_frame,
                         (x1, y1 - text_height - 10),
                         (x1 + text_width + 10, y1),
                         color, 1)
            
            # Label text
            cv2.putText(display_frame, label,
                       (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 255), 2)
            
            # Draw tracking centroid
            if det.track_id:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv2.circle(display_frame, (cx, cy), 5, (0, 255, 255), -1)
                cv2.circle(display_frame, (cx, cy), 7, (0, 0, 0), 1)
        
        # Update performance tracking
        self.object_count_history.append(len(scaled_detections))
        if scaled_detections:
            avg_confidence = np.mean([d['confidence'] for d in scaled_detections])
            self.confidence_history.append(avg_confidence)
            
            # Keep history manageable
            if len(self.object_count_history) > 100:
                self.object_count_history.pop(0)
            if len(self.confidence_history) > 100:
                self.confidence_history.pop(0)
        
        # Smart analysis - only analyze when needed
        object_assessments = []
        
        if (scaled_detections and self.auto_analysis_enabled and 
            (current_time - self.last_analysis_time > self.analysis_interval)):
            
            self.last_analysis_time = current_time
            
            # Analyze up to 3 main objects
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
        """Main application loop with optimized display"""
        print("\n" + "="*60)
        print("🎯 JADE Enhanced is running!")
        print("👩 Voice: Always Active (Like JARVIS)")
        print("📊 Performance Graphs: ENABLED")
        print("🎨 Enhanced GUI: ACTIVE")
        print(f"📺 Display: {self.total_width}x{self.total_height}")
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
        
        # Initial greeting
        if self.voice_enabled:
            self.voice_assistant.speak("JADE Enhanced activated. I'm always listening. Point the camera at objects for analysis.")
        
        # Create window with proper properties
        cv2.namedWindow('JADE Enhanced - Object Analyzer', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('JADE Enhanced - Object Analyzer', self.total_width, self.total_height)
        
        while True:
            try:
                # Read frame
                ret = False
                frame = None
                
                if self.cap is not None:
                    ret, frame = self.cap.read()
                
                if not ret or frame is None or frame.size == 0:
                    print("❌ Camera error or empty frame")
                    
                    # Create debug display
                    debug_height = 480
                    debug_width = 640
                    debug_frame = np.zeros((debug_height, debug_width, 3), dtype=np.uint8)
                    debug_frame[:, :, :] = (30, 30, 40)
                    
                    # Add debug message
                    cv2.putText(debug_frame, "CAMERA FEED UNAVAILABLE", (120, 200),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(debug_frame, "Press 'H' for troubleshooting", (140, 250),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    
                    # Show debug display
                    cv2.imshow('JADE Enhanced - Camera Debug', debug_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('h'):
                        self._show_camera_help()
                    
                    time.sleep(0.1)
                    continue
                
                # Process frame
                processed_frame, detections, assessments = self._process_frame_optimized(frame)
                
                if processed_frame is None:
                    continue
                
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
    
    def _show_camera_help(self):
        """Show camera troubleshooting help"""
        help_text = """
        CAMERA TROUBLESHOOTING:
        
        1. Check if camera is connected
        2. Try different camera IDs:
           - Change CAMERA_ID in config.py
           - Common IDs: 0, 1, 2
        
        3. Test camera with OpenCV:
           python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera found' if cap.isOpened() else 'No camera')"
        
        4. On Linux, check video devices:
           ls -la /dev/video*
        
        5. On Windows, check Device Manager
        
        Press 'Q' to quit, or try different camera ID.
        """
        
        print(help_text)
        
        # Create help window
        help_frame = np.zeros((400, 600, 3), dtype=np.uint8)
        help_frame[:, :, :] = (30, 30, 40)
        
        y_pos = 30
        for line in help_text.strip().split('\n'):
            cv2.putText(help_frame, line.strip(), (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
            y_pos += 20
        
        cv2.imshow('Camera Help', help_frame)
        cv2.waitKey(3000)
        cv2.destroyWindow('Camera Help')
    
    def _toggle_voice_mute(self):
        """Toggle voice assistant mute"""
        if self.voice_enabled:
            self.voice_assistant.stop_listening()
            self.voice_enabled = False
            print("🔇 Voice muted")
            
            # Show notification
            notif = np.zeros((80, 250, 3), dtype=np.uint8)
            notif[:, :, :] = (50, 50, 50)
            cv2.putText(notif, "Voice: MUTED", (60, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
            cv2.imshow('Notification', notif)
            cv2.waitKey(1000)
            cv2.destroyWindow('Notification')
        else:
            self.voice_assistant.start_continuous_listening()
            self.voice_enabled = True
            print("🔊 Voice activated")
            
            # Show notification
            notif = np.zeros((80, 250, 3), dtype=np.uint8)
            notif[:, :, :] = (50, 50, 50)
            cv2.putText(notif, "Voice: ACTIVE", (60, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
            cv2.imshow('Notification', notif)
            cv2.waitKey(1000)
            cv2.destroyWindow('Notification')
    
    def _toggle_analysis_mute(self):
        """Toggle automatic analysis"""
        self.auto_analysis_enabled = not self.auto_analysis_enabled
        status = "ON" if self.auto_analysis_enabled else "OFF"
        print(f"🔍 Automatic analysis: {status}")
        
        # Show notification
        notif = np.zeros((80, 300, 3), dtype=np.uint8)
        notif[:, :, :] = (50, 50, 50)
        color = (100, 255, 100) if self.auto_analysis_enabled else (255, 100, 100)
        cv2.putText(notif, f"Auto-Analysis: {status}", (60, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow('Notification', notif)
        cv2.waitKey(1000)
        cv2.destroyWindow('Notification')
        
        if self.voice_enabled:
            self.voice_assistant.speak(f"Automatic analysis {status}")
    
    def _manual_listen(self):
        """Manual listen for command"""
        if not self.voice_enabled:
            print("⚠️  Voice assistant is disabled. Press 'V' to enable.")
            return
        
        print("🎤 Manual listening activated...")
        
        # Show listening indicator
        listen_frame = np.zeros((100, 350, 3), dtype=np.uint8)
        listen_frame[:, :, :] = (30, 100, 200)
        cv2.putText(listen_frame, "🎤 LISTENING...", (80, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imshow('Listening', listen_frame)
        cv2.waitKey(100)
        
        text = self.voice_assistant.listen_once()
        cv2.destroyWindow('Listening')
        
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
        
        # Show generating indicator
        gen_frame = np.zeros((120, 450, 3), dtype=np.uint8)
        gen_frame[:, :, :] = (30, 100, 150)
        cv2.putText(gen_frame, "📈 Generating Graphs...", (80, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imshow('Generating', gen_frame)
        cv2.waitKey(100)
        
        # Generate reports
        total_time = time.time() - self.start_time
        
        try:
            # 1. Assistant performance report
            assistant_report = self.assistant.generate_performance_report(
                self.session_id, self.frame_count, total_time
            )
            
            # 2. Detector performance report
            detector_report = self.detector.create_performance_report(self.session_id)
            
            # 3. Logger visualization
            logger_report = self.logger.create_visualization_report(self.session_id)
            
            cv2.destroyWindow('Generating')
            
            # Show completion message
            complete_frame = np.zeros((120, 450, 3), dtype=np.uint8)
            complete_frame[:, :, :] = (50, 150, 50)
            cv2.putText(complete_frame, "✅ Graphs Generated!", (100, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imshow('Complete', complete_frame)
            cv2.waitKey(1000)
            cv2.destroyWindow('Complete')
            
            print("\n✅ Performance graphs generated:")
            print(f"   • Assistant: {assistant_report}")
            print(f"   • Detector: {detector_report}")
            print(f"   • Logger: {logger_report}")
            
            if self.voice_enabled:
                self.voice_assistant.speak("Performance graphs generated and saved to reports folder.")
                
        except Exception as e:
            print(f"❌ Error generating graphs: {e}")
            cv2.destroyWindow('Generating')
    
    def _show_live_performance(self):
        """Show live performance overlay"""
        print("📈 Showing live performance stats...")
        
        # Create performance overlay
        perf_frame = np.zeros((350, 500, 3), dtype=np.uint8)
        perf_frame[:, :, :] = (20, 20, 30)
        
        # Title
        cv2.putText(perf_frame, "📊 LIVE PERFORMANCE", (120, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.colors['primary'], 2)
        
        # Stats with icons
        stats = [
            f"📷 Frames Processed: {self.frame_count}",
            f"⚡ Current FPS: {self.fps:.1f}",
            f"🎯 Objects Detected: {len(self.last_detections)}",
            f"🔍 Analysis Mode: {self.assistant.current_mode}",
            f"⏱️  Session Duration: {time.time() - self.start_time:.1f}s",
            f"🔊 Voice: {'ACTIVE' if self.voice_enabled else 'MUTED'}",
            f"🤖 Auto-Analysis: {'ON' if self.auto_analysis_enabled else 'OFF'}"
        ]
        
        y_pos = 80
        for stat in stats:
            cv2.putText(perf_frame, stat, (30, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
            y_pos += 40
        
        cv2.imshow('Live Performance', perf_frame)
        cv2.waitKey(3000)
        cv2.destroyWindow('Live Performance')
        
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
        
        # Display help in a window
        help_frame = np.zeros((550, 750, 3), dtype=np.uint8)
        help_frame[:, :, :] = (20, 20, 30)
        
        y_pos = 40
        for line in help_text.strip().split('\n'):
            cv2.putText(help_frame, line.strip(), (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
            y_pos += 25
        
        cv2.imshow('JADE Help Guide', help_frame)
        cv2.waitKey(5000)
        cv2.destroyWindow('JADE Help Guide')
        
        if self.voice_enabled:
            self.voice_assistant.speak("Enhanced help information displayed. You can ask me anything or generate performance graphs!")
    
    def _save_screenshot(self, frame):
        """Save screenshot with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create exports directory if it doesn't exist
        os.makedirs('exports/screenshots', exist_ok=True)
        
        filename = f"exports/screenshots/jade_enhanced_{timestamp}.jpg"
        
        # Save with high quality
        cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"✅ Screenshot saved: {filename}")
        
        # Show confirmation
        confirm_frame = np.zeros((100, 350, 3), dtype=np.uint8)
        confirm_frame[:, :, :] = (50, 150, 50)
        cv2.putText(confirm_frame, "📸 Screenshot Saved!", (60, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imshow('Confirmation', confirm_frame)
        cv2.waitKey(1000)
        cv2.destroyWindow('Confirmation')
        
        if self.voice_enabled:
            self.voice_assistant.speak("Screenshot saved successfully.")
    
    def cleanup(self):
        """Cleanup resources and generate final report"""
        print("\n🛑 Shutting down JADE Enhanced...")
        
        # Stop components
        if hasattr(self, 'cap') and self.cap is not None:
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
                'fps_history': list(self.fps_history)[-50:],
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