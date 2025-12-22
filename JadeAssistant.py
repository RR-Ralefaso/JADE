import cv2
import numpy as np
from ultralytics import YOLO
import torch

class JadeAssistant:
    """Enhanced object detector using YOLO with better accuracy and training capabilities"""
    def __init__(self, model_path='models/yolo11n.pt', confidence=0.3):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.class_names = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        print(f"✅ Detector loaded with model: {model_path}")
        print(f"🖥️  Using device: {self.device}")
        print(f"📊 Class count: {len(self.class_names)}")
        print(f"🎯 Confidence threshold: {confidence}")
    
    def detect(self, frame):
        """Detect objects in frame with improved accuracy"""
        # Preprocess for better detection
        processed_frame = self._preprocess_frame(frame)
        
        # Run inference
        results = self.model(
            processed_frame, 
            conf=self.confidence, 
            verbose=False,
            device=self.device,
            agnostic_nms=True,  # Better for overlapping objects
            max_det=50,  # Increase max detections
            iou=0.45
        )
        
        detections = []
        display_frame = frame.copy()
        
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # Extract bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.class_names[class_id]
                    
                    # Filter very small detections (noise)
                    bbox_area = (x2 - x1) * (y2 - y1)
                    frame_area = frame.shape[0] * frame.shape[1]
                    
                    if bbox_area < frame_area * 0.0005:  # Skip very small objects
                        continue
                    
                    # Enhanced drawing
                    color = self._get_class_color(class_id)
                    thickness = max(1, int(2 * (confidence ** 2)))  # Thicker for higher confidence
                    
                    # Draw bounding box with gradient color
                    cv2.rectangle(display_frame, 
                                (int(x1), int(y1)), 
                                (int(x2), int(y2)), 
                                color, thickness)
                    
                    # Draw label with background
                    label = f"{class_name} {confidence:.2f}"
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    )
                    
                    # Background for text
                    cv2.rectangle(display_frame,
                                (int(x1), int(y1) - text_height - 10),
                                (int(x1) + text_width, int(y1)),
                                color, -1)
                    
                    # Text
                    cv2.putText(display_frame, label, 
                              (int(x1), int(y1) - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # Store detection with more info
                    detections.append({
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'class_id': class_id,
                        'area': bbox_area,
                        'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                    })
        
        return display_frame, detections
    
    def _preprocess_frame(self, frame):
        """Preprocess frame for better detection"""
        # Enhance contrast
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge back
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # Optional: resize for consistent input
        if frame.shape[:2] != (640, 640):
            enhanced = cv2.resize(enhanced, (640, 640))
        
        return enhanced
    
    def _get_class_color(self, class_id):
        """Get consistent color for each class"""
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (0, 128, 128),  # Teal
            (128, 128, 0),  # Olive
            (128, 0, 0),    # Maroon
        ]
        return colors[class_id % len(colors)]
    
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
            'object_density': len(detections) / (1280 * 720)  # objects per pixel
        }
        
        confidences = []
        max_area = 0
        
        for det in detections:
            class_name = det['class_name']
            confidence = det['confidence']
            area = det['area']
            
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
                    'bbox': det['bbox']
                }
        
        if confidences:
            stats['confidence_avg'] = sum(confidences) / len(confidences)
            stats['confidence_min'] = min(confidences)
            stats['confidence_max'] = max(confidences)
        
        stats['unique_classes'] = list(stats['unique_classes'])
        
        return stats
    
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
                    
                    # Draw bounding box
                    cv2.rectangle(processed_frame, 
                                (int(x1), int(y1)), 
                                (int(x2), int(y2)), 
                                (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"{class_name} {confidence:.2f}"
                    cv2.putText(processed_frame, label, 
                              (int(x1), int(y1)-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Store detection
                    detections.append({
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'class_id': class_id
                    })
        
        return processed_frame, detections
    
    def save_detections_image(self, frame, detections, output_path):
        """Save frame with detections to file"""
        display_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']
            
            # Draw bounding box
            cv2.rectangle(display_frame, 
                        (x1, y1), 
                        (x2, y2), 
                        (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(display_frame, label, 
                      (x1, y1-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imwrite(output_path, display_frame)
        print(f"💾 Saved detection image to: {output_path}")