import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')
import colorsys
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    mask: Optional[np.ndarray] = None
    track_id: Optional[int] = None
    centroid: Optional[Tuple[int, int]] = None
    area: float = 0.0

class JadeAssistant:
    """Ultra-fast and accurate object detector with tracking and performance visualization"""
    
    def __init__(self, model_path='models/yolo11n.pt', confidence=0.35):
        self.model_path = model_path
        self.confidence = confidence
        self.device = self._get_device()
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Load model with optimizations
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # Fuse layers for speed if available
        try:
            self.model.fuse()
        except:
            pass
        
        # Warm up model
        self._warmup()
        
        # Tracking and performance monitoring
        self.tracker = self._init_tracker()
        self.fps_history = deque(maxlen=100)
        self.inference_times = deque(maxlen=100)
        self.detection_counts = deque(maxlen=100)
        
        # Performance tracking
        self.performance_data = {
            'inference_times': [],
            'fps_values': [],
            'detection_counts': [],
            'confidence_values': [],
            'start_time': time.time()
        }
        
        # Class names and colors
        self.class_names = self.model.names
        self.class_colors = self._generate_colors(len(self.class_names))
        
        print(f"✅ Detector loaded with model: {model_path}")
        print(f"🖥️  Using device: {self.device}")
        print(f"📊 Class count: {len(self.class_names)}")
        print(f"🎯 Confidence threshold: {confidence}")
        print(f"⚡ Precision: {self.dtype}")
    
    def _get_device(self):
        """Get the best available device"""
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    
    def _warmup(self):
        """Warm up the model with dummy inference"""
        print("🔥 Warming up model...")
        import torch
        dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
        if self.dtype == torch.float16:
            dummy_input = dummy_input.half()
        
        for _ in range(3):
            with torch.no_grad():
                _ = self.model(dummy_input, verbose=False)
        print("✅ Model warmed up")
    
    def _init_tracker(self):
        """Initialize object tracker"""
        try:
            from boxmot import BYTETracker
            tracker = BYTETracker(
                track_thresh=0.45,
                match_thresh=0.8,
                frame_rate=30
            )
            print("✅ Object tracker initialized")
            return tracker
        except ImportError:
            print("⚠️  BYTETracker not available, tracking disabled")
            return None
    
    def _generate_colors(self, n):
        """Generate distinct colors for each class"""
        colors = []
        for i in range(n):
            hue = i * 137.508  # Golden angle approximation
            r, g, b = colorsys.hsv_to_rgb((hue % 360) / 360, 0.8, 0.9)
            colors.append((int(b * 255), int(g * 255), int(r * 255)))
        return colors
    
    def _preprocess_frame(self, frame):
        """Optimized frame preprocessing"""
        # Convert to RGB (YOLO expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Enhance contrast using CLAHE in LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Sharpen slightly
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened
    
    def detect(self, frame):
        """Ultra-fast detection with tracking and performance monitoring"""
        start_time = time.time()
        
        # Preprocess
        processed_frame = self._preprocess_frame(frame)
        
        # Run inference with optimizations
        with torch.no_grad():
            results = self.model(
                processed_frame,
                conf=self.confidence,
                iou=0.45,
                device=self.device,
                half=(self.dtype == torch.float16),
                verbose=False,
                max_det=100,
                agnostic_nms=True,
                retina_masks=True
            )
        
        detections = []
        display_frame = frame.copy()
        
        confidence_values = []
        
        if results and len(results) > 0:
            result = results[0]
            
            # Extract boxes, scores, classes
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                # Extract masks if available
                masks = None
                if result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                
                # Apply tracking if available
                track_id_map = {}
                if self.tracker is not None and len(boxes) > 0:
                    tracker_inputs = []
                    for i, box in enumerate(boxes):
                        tracker_inputs.append([
                            box[0], box[1], box[2], box[3], scores[i], class_ids[i]
                        ])
                    
                    tracker_inputs = np.array(tracker_inputs)
                    tracked_detections = self.tracker.update(tracker_inputs, frame)
                    
                    # Map tracking IDs
                    for det in tracked_detections:
                        tlbr = det.tlbr
                        track_id = det.track_id
                        # Find matching detection
                        for i, box in enumerate(boxes):
                            if self._iou(box, tlbr) > 0.5:
                                track_id_map[i] = track_id
                                break
                
                # Process each detection
                for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Skip very small detections
                    area = (x2 - x1) * (y2 - y1)
                    if area < 100:  # Minimum 100 pixels
                        continue
                    
                    confidence_values.append(score)
                    
                    # Get mask if available
                    mask = None
                    if masks is not None and i < len(masks):
                        mask = masks[i]
                        # Resize mask to original bbox
                        mask = cv2.resize(mask, (x2 - x1, y2 - y1))
                        mask = (mask > 0.5).astype(np.uint8) * 255
                    
                    # Get tracking ID
                    track_id = track_id_map.get(i)
                    
                    # Create detection object
                    detection = Detection(
                        class_id=class_id,
                        class_name=self.class_names[class_id],
                        confidence=float(score),
                        bbox=[x1, y1, x2, y2],
                        mask=mask,
                        track_id=track_id,
                        centroid=((x1 + x2) // 2, (y1 + y2) // 2),
                        area=area
                    )
                    detections.append(detection)
                    
                    # Draw on frame
                    display_frame = self._draw_detection(display_frame, detection)
        
        # Calculate inference time
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # Update performance data
        self.performance_data['inference_times'].append(inference_time)
        self.performance_data['detection_counts'].append(len(detections))
        self.performance_data['confidence_values'].extend(confidence_values)
        
        # Calculate FPS
        if inference_time > 0:
            fps = 1.0 / inference_time
            self.fps_history.append(fps)
            self.performance_data['fps_values'].append(fps)
        
        return display_frame, detections
    
    def _iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return inter_area / (box1_area + box2_area - inter_area + 1e-6)
    
    def _draw_detection(self, frame, detection):
        """Draw detection with optimized rendering"""
        x1, y1, x2, y2 = detection.bbox
        color = self.class_colors[detection.class_id % len(self.class_colors)]
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw mask if available
        if detection.mask is not None:
            # Create colored overlay
            overlay = frame.copy()
            mask_color = color + (50,)  # Add transparency
            overlay[y1:y2, x1:x2][detection.mask > 0] = mask_color[:3]
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Prepare label
        label_parts = []
        if detection.track_id is not None:
            label_parts.append(f"ID:{detection.track_id}")
        
        label_parts.append(detection.class_name)
        label_parts.append(f"{detection.confidence:.2f}")
        
        label = " ".join(label_parts)
        
        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw label background
        cv2.rectangle(frame, 
                     (x1, y1 - text_height - 10),
                     (x1 + text_width, y1),
                     color, -1)
        
        # Draw label text
        cv2.putText(frame, label,
                   (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (255, 255, 255), 1)
        
        # Draw centroid if tracking
        if detection.centroid and detection.track_id is not None:
            cx, cy = detection.centroid
            cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)
        
        return frame
    
    def create_performance_report(self, session_id):
        """Create performance visualization report"""
        print("📊 Creating detector performance report...")
        
        # Create reports directory if it doesn't exist
        os.makedirs('reports/detector', exist_ok=True)
        
        # Set style
        plt.style.use('dark_background')
        sns.set_palette("husl")
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Detector Performance Report - Session {session_id}', fontsize=16, color='white')
        
        # 1. FPS over time
        ax1 = axes[0, 0]
        fps_values = self.performance_data['fps_values']
        if fps_values:
            time_indices = list(range(len(fps_values)))
            ax1.plot(time_indices, fps_values, 'b-', linewidth=2, marker='o', markersize=2)
            ax1.set_title('FPS Over Time', color='white')
            ax1.set_xlabel('Frame Number', color='white')
            ax1.set_ylabel('Frames Per Second', color='white')
            ax1.fill_between(time_indices, 0, fps_values, alpha=0.3, color='blue')
            ax1.tick_params(colors='white')
            ax1.grid(True, alpha=0.3)
            
            # Add average line
            avg_fps = np.mean(fps_values)
            ax1.axhline(y=avg_fps, color='red', linestyle='--', linewidth=2,
                       label=f'Average: {avg_fps:.1f} FPS')
            ax1.legend(facecolor='#2e2e2e', edgecolor='white', labelcolor='white')
        
        # 2. Inference time distribution
        ax2 = axes[0, 1]
        inference_times = self.performance_data['inference_times']
        if inference_times:
            # Convert to milliseconds
            inference_ms = [t * 1000 for t in inference_times]
            ax2.hist(inference_ms, bins=20, color='green', edgecolor='black', alpha=0.7)
            ax2.axvline(np.mean(inference_ms), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(inference_ms):.1f}ms')
            ax2.set_title('Inference Time Distribution', color='white')
            ax2.set_xlabel('Inference Time (ms)', color='white')
            ax2.set_ylabel('Frequency', color='white')
            ax2.legend(facecolor='#2e2e2e', edgecolor='white', labelcolor='white')
            ax2.tick_params(colors='white')
            ax2.grid(True, alpha=0.3)
        
        # 3. Detection count over time
        ax3 = axes[1, 0]
        detection_counts = self.performance_data['detection_counts']
        if detection_counts:
            time_indices = list(range(len(detection_counts)))
            ax3.plot(time_indices, detection_counts, 'orange', linewidth=2, marker='s', markersize=2)
            ax3.set_title('Detection Count Over Time', color='white')
            ax3.set_xlabel('Frame Number', color='white')
            ax3.set_ylabel('Objects Detected', color='white')
            ax3.fill_between(time_indices, 0, detection_counts, alpha=0.3, color='orange')
            ax3.tick_params(colors='white')
            ax3.grid(True, alpha=0.3)
            
            # Add average line
            avg_detections = np.mean(detection_counts)
            ax3.axhline(y=avg_detections, color='red', linestyle='--', linewidth=2,
                       label=f'Average: {avg_detections:.1f} objects')
            ax3.legend(facecolor='#2e2e2e', edgecolor='white', labelcolor='white')
        
        # 4. Confidence distribution
        ax4 = axes[1, 1]
        confidence_values = self.performance_data['confidence_values']
        if confidence_values:
            ax4.hist(confidence_values, bins=20, color='purple', edgecolor='black', alpha=0.7)
            ax4.axvline(np.mean(confidence_values), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(confidence_values):.3f}')
            ax4.set_title('Confidence Distribution', color='white')
            ax4.set_xlabel('Confidence', color='white')
            ax4.set_ylabel('Frequency', color='white')
            ax4.legend(facecolor='#2e2e2e', edgecolor='white', labelcolor='white')
            ax4.tick_params(colors='white')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = f"reports/detector/detector_performance_{session_id}.png"
        plt.savefig(plot_file, dpi=150, facecolor='#0f0f0f')
        plt.close()
        
        print(f"📈 Detector performance plot saved: {plot_file}")
        
        # Create summary dashboard
        self._create_summary_dashboard(session_id)
        
        return plot_file
    
    def _create_summary_dashboard(self, session_id):
        """Create summary dashboard for detector performance"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate statistics
        total_frames = len(self.performance_data['fps_values'])
        total_detections = sum(self.performance_data['detection_counts'])
        avg_fps = np.mean(self.performance_data['fps_values']) if self.performance_data['fps_values'] else 0
        avg_inference = np.mean(self.performance_data['inference_times']) * 1000 if self.performance_data['inference_times'] else 0
        avg_confidence = np.mean(self.performance_data['confidence_values']) if self.performance_data['confidence_values'] else 0
        
        # Create summary text
        summary_text = f"""Detector Performance Summary - Session {session_id}
        
        Performance Metrics:
        • Total Frames Processed: {total_frames}
        • Total Detections: {total_detections}
        • Average FPS: {avg_fps:.1f}
        • Average Inference Time: {avg_inference:.1f}ms
        • Average Confidence: {avg_confidence:.3f}
        
        Device Information:
        • Device: {self.device.upper()}
        • Precision: {str(self.dtype).split('.')[-1]}
        • Model: {os.path.basename(self.model_path)}
        • Confidence Threshold: {self.confidence}
        
        Detection Statistics:
        • Classes Available: {len(self.class_names)}
        • Tracking: {'Enabled' if self.tracker else 'Disabled'}
        • Warm-up Completed: Yes
        
        Performance Insights:"""
        
        # Add insights based on performance
        if avg_fps < 10:
            summary_text += "\n• ⚠️  FPS is low - consider reducing resolution or frame skip"
        elif avg_fps < 20:
            summary_text += "\n• ℹ️  FPS is moderate - system is running adequately"
        else:
            summary_text += "\n• ✅ FPS is excellent - system is running optimally"
        
        if avg_inference > 50:
            summary_text += "\n• ⚠️  Inference time is high - consider optimizing model"
        elif avg_inference > 30:
            summary_text += "\n• ℹ️  Inference time is moderate"
        else:
            summary_text += "\n• ✅ Inference time is excellent"
        
        if avg_confidence < 0.5:
            summary_text += "\n• ⚠️  Average confidence is low - consider adjusting confidence threshold"
        elif avg_confidence < 0.7:
            summary_text += "\n• ℹ️  Average confidence is acceptable"
        else:
            summary_text += "\n• ✅ Average confidence is excellent"
        
        # Display summary
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#2e2e2e', 
                                                 edgecolor='white', alpha=0.9),
               color='white', fontfamily='monospace')
        
        ax.axis('off')
        ax.set_facecolor('#0f0f0f')
        
        # Add logo/watermark
        ax.text(0.98, 0.02, 'JADE Object Detector', transform=ax.transAxes,
               fontsize=10, color='gray', ha='right', va='bottom', alpha=0.5)
        
        plt.tight_layout()
        summary_file = f"reports/detector/detector_summary_{session_id}.png"
        plt.savefig(summary_file, dpi=150, facecolor='#0f0f0f')
        plt.close()
        print(f"📋 Detector summary saved: {summary_file}")
    
    def detect_specific_classes(self, frame, class_names):
        """Detect only specific classes"""
        # Get class IDs for the specified names
        class_ids = []
        for name in class_names:
            for class_id, class_name in self.class_names.items():
                if class_name == name:
                    class_ids.append(class_id)
                    break
        
        if not class_ids:
            return frame, []
        
        # Run inference with class filter
        results = self.model(
            frame,
            conf=self.confidence,
            classes=class_ids,
            verbose=False,
            device=self.device
        )
        
        detections = []
        processed_frame = frame.copy()
        
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.class_names[class_id]
                    
                    # Create detection object
                    detection = Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=[int(x1), int(y1), int(x2), int(y2)],
                        area=(x2 - x1) * (y2 - y1)
                    )
                    detections.append(detection)
                    
                    # Draw
                    processed_frame = self._draw_detection(processed_frame, detection)
        
        return processed_frame, detections
    
    def train_custom_model(self, data_yaml, epochs=50, imgsz=640):
        """Train custom model for specific objects"""
        print(f"🎯 Starting training on: {data_yaml}")
        print(f"⏱️  Epochs: {epochs}")
        print(f"📐 Image size: {imgsz}")
        
        # Train the model
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=16,
            device=self.device,
            workers=4,
            save=True,
            save_period=10,
            pretrained=True,
            optimizer='AdamW',
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            verbose=True
        )
        
        print("✅ Training completed!")
        print(f"📁 Model saved in: runs/detect/train/")
        return results
    
    def export_detection_statistics(self, detections):
        """Generate statistics about detected objects"""
        if not detections:
            return None
        
        stats = {
            'total_objects': len(detections),
            'unique_classes': set(),
            'confidence_avg': 0.0,
            'confidence_min': 1.0,
            'confidence_max': 0.0,
            'largest_object': None,
            'class_distribution': {},
            'object_density': len(detections) / (1280 * 720)
        }
        
        confidences = []
        max_area = 0
        
        for det in detections:
            class_name = det.class_name
            confidence = det.confidence
            area = det.area
            
            # Update stats
            stats['unique_classes'].add(class_name)
            confidences.append(confidence)
            
            # Class distribution
            stats['class_distribution'][class_name] = stats['class_distribution'].get(class_name, 0) + 1
            
            # Largest object
            if area > max_area:
                max_area = area
                stats['largest_object'] = {
                    'class': class_name,
                    'confidence': confidence,
                    'area': area,
                    'bbox': det.bbox
                }
        
        if confidences:
            stats['confidence_avg'] = sum(confidences) / len(confidences)
            stats['confidence_min'] = min(confidences)
            stats['confidence_max'] = max(confidences)
        
        stats['unique_classes'] = list(stats['unique_classes'])
        
        return stats
    
    def save_detections_image(self, frame, detections, output_path):
        """Save frame with detections to file"""
        display_frame = frame.copy()
        
        for det in detections:
            display_frame = self._draw_detection(display_frame, det)
        
        cv2.imwrite(output_path, display_frame)
        print(f"💾 Saved detection image to: {output_path}")
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.inference_times:
            return {}
        
        times = list(self.inference_times)
        fps_values = list(self.fps_history)
        
        return {
            'avg_inference_time': sum(times) / len(times),
            'min_inference_time': min(times),
            'max_inference_time': max(times),
            'avg_fps': sum(fps_values) / len(fps_values) if fps_values else 0,
            'device': self.device,
            'precision': str(self.dtype),
            'total_frames_processed': len(self.performance_data['fps_values']),
            'total_detections': sum(self.performance_data['detection_counts']),
            'avg_confidence': np.mean(self.performance_data['confidence_values']) if self.performance_data['confidence_values'] else 0
        }