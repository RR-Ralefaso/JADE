import torch
import numpy as np
from ultralytics import YOLO as yolo
from config import MODEL_PATH, CONFIDENCE

class JadeImageDetector:
    def __init__(self):
        print("🔍 JADE Hardware Detection...")
        
        # PRIORITY: Intel GPU > CUDA > CPU
        self.device = 'cpu'
        
        # 1. CHECK INTEL GPU (OpenVINO iGPU/Arc)
        try:
            import openvino
            self.device = 'openvino:GPU'  # Intel GPU first
            print("✅ INTEL GPU (OpenVINO) SELECTED")
        except ImportError:
            print("⚠️ Install: pip install openvino")
        
        # 2. CHECK CUDA GPU
        if torch.cuda.is_available():
            self.device = 'cuda:0'
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ CUDA GPU: {gpu_name} ({vram_gb:.1f}GB)")
        
        # 3. CPU FALLBACK
        else:
            self.device = 'cpu'
            print("⚠️ CPU ONLY - Install OpenVINO or CUDA for GPU speed")
        
        print(f"🚀 Loading YOLO11n on {self.device}...")
        
        # LOAD MODEL
        self.model = yolo(MODEL_PATH)
        self.model.to(self.device)
        self.model.fuse()  # Optimize inference
        
        # CONFIRM LOADING
        param_device = next(self.model.parameters()).device.type
        print(f"✅ JADE LOADED on {param_device.upper()}")
        print(f"   Device: {self.device}")
        
        if param_device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")

    def detect(self, frame):
        """Ultra-fast detection optimized for your hardware"""
        if frame is None or frame.size == 0:
            return frame
        
        # HARDWARE-SPECIFIC OPTIMIZATIONS
        results = self.model(
            frame,
            verbose=False,           # No console spam
            conf=CONFIDENCE,         # From config.py
            device=self.device,      # Auto-detected hardware
            imgsz=320,               # Fast input size
            half=(self.device != 'cpu'),  # FP16 for GPU only
            max_det=10               # Limit detections for speed
        )
        return results[0].plot()  # Annotated frame
