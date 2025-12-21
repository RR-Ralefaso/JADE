import cv2
import time
import numpy as np
from Jade import JadeImageDetector
from outtxt import reset_output_file, write_detection_output
from knowledge_base import JADE_KNOWLEDGE
from config import CAMERA_ID, WINDOW_NAME, CONFIDENCE

def main():
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("Camera error! Try CAMERA_ID = 0, 1, 2 in config.py")
        return
    
    reset_output_file()
    
    # Ultra-fast camera setup
    max_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    max_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = min(max_width, 1280)
    height = min(max_height, 720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"SMART JADE ALWAYS-ON: {width}x{height} @ {fps:.1f}FPS")
    
    detector = JadeImageDetector()
    
    print("SMART JADE ALWAYS DETECTING! 'q' quit")
    fps_counter = 0
    start_time = time.time()
    last_spoken = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display_frame = frame.copy()
        
        # **ALWAYS DETECT** - No toggle needed!
        small_frame = cv2.resize(frame, (320, 240))
        
        raw_results = detector.model(
            small_frame, verbose=False, conf=CONFIDENCE,
            device=detector.device, imgsz=320, half=True
        )
        
        # **JADE'S INTELLIGENCE** (always active)
        if raw_results[0].boxes is not None and len(raw_results[0].boxes) > 0:
            scale_x = width / 320
            scale_y = height / 240
            current_detections = []
            
            for box in raw_results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                class_name = detector.model.names[cls_id]
                
                x1, y1, x2, y2 = int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)
                
                # Draw box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                
                # **SPEAK KNOWLEDGE**
                if class_name in JADE_KNOWLEDGE and class_name not in last_spoken:
                    info = JADE_KNOWLEDGE[class_name]["info"]
                    value = JADE_KNOWLEDGE[class_name]["value"]
                    tip = JADE_KNOWLEDGE[class_name]["tip"]
                    
                    knowledge_log = f"JADE ID: {class_name} | {info} | {value} | Tip: {tip}"
                    write_detection_output(knowledge_log)
                    print(f"🧠 {class_name}: {info}")
                    
                    last_spoken[class_name] = True
                
                # Show on frame
                label = f'{class_name} {conf:.1f}'
                cv2.putText(display_frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                current_detections.append(class_name)
            
            # Reset spoken cache for new objects
            for cls in current_detections:
                if cls not in last_spoken:
                    last_spoken = {}
                    break
            
            # Status: DETECTING
            cv2.putText(display_frame, "JADE DETECTING", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            # Status: SCANNING (still detecting, no objects)
            cv2.putText(display_frame, "JADE SCANNING", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # FPS overlay
        fps_counter += 1
        if fps_counter % 30 == 0:
            fps = 30 / (time.time() - start_time)
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, height-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            start_time = time.time()
        
        cv2.imshow(WINDOW_NAME, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            write_detection_output("SMART JADE session ended.")
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
