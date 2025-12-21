import cv2
import numpy as np
from ultralytics import YOLO

class JadeAssistant:
    """Simple object detector using YOLO"""
    def __init__(self, model_path='models/yolo11n.pt', confidence=0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence
        print(f"✅ Detector loaded with model: {model_path}")
    
    def detect(self, frame):
        """Detect objects in frame"""
        results = self.model(frame, conf=self.confidence, verbose=False)
        
        detections = []
        processed_frame = frame.copy()
        
        if results and len(results) > 0:
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract bounding box
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = result.names[class_id]
                        
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