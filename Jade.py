
import torch
import numpy as np
from ultralytics import YOLO as yolo
import logging
from config import MODEL_PATH, CONFIDENCE, DETECTION_PARAMS

class JadeImageDetector:
    def __init__(self):
        """Initialize JADE detector with hardware optimization"""
        self.logger = logging.getLogger('JADE')
        self.logger.info("🔍 Initializing JADE Hardware Detection...")
        
        # Hardware detection priority: Intel GPU > CUDA > CPU
        self.device = self._detect_optimal_device()
        self.logger.info(f"🚀 Selected device: {self.device}")
        
        # Load model with error handling
        try:
            self.model = self._load_model()
            self._verify_model_loading()
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
        
    def _detect_optimal_device(self):
        """Auto-detect and select optimal inference device"""
        device = 'cpu'
        
        # 1. Check Intel GPU (OpenVINO)
        try:
            import openvino
            import openvino.runtime as ov
            core = ov.Core()
            gpu_devices = core.available_devices
            if 'GPU' in str(gpu_devices):
                device = 'openvino'
                self.logger.info("✅ Intel GPU detected (OpenVINO)")
                return device
        except ImportError:
            self.logger.debug("OpenVINO not installed")
        except Exception as e:
            self.logger.warning(f"OpenVINO GPU detection failed: {str(e)}")
        
        # 2. Check CUDA GPU with memory validation
        if torch.cuda.is_available():
            try:
                # Check if CUDA has enough memory (at least 1GB)
                cuda_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                if cuda_memory_gb >= 1.0:  # Minimum 1GB required
                    device = 'cuda:0'
                    gpu_name = torch.cuda.get_device_name(0)
                    self.logger.info(f"✅ CUDA GPU detected: {gpu_name} ({cuda_memory_gb:.1f}GB)")
                    return device
                else:
                    self.logger.warning(f"Insufficient CUDA memory: {cuda_memory_gb:.1f}GB")
            except Exception as e:
                self.logger.warning(f"CUDA validation failed: {str(e)}")
        
        # 3. CPU fallback with optimization
        device = 'cpu'
        self.logger.warning("⚠️ Using CPU inference - Install OpenVINO/CUDA for GPU acceleration")
        return device
    
    def _load_model(self):
        """Load YOLO model with device-specific optimizations"""
        self.logger.info(f"📦 Loading model: {MODEL_PATH}")
        
        # Load model
        model = yolo(MODEL_PATH)
        
        # Apply device-specific optimizations
        if self.device == 'openvino':
            # OpenVINO specific optimizations
            model.export(format='openvino', dynamic=False, half=False)
            model = yolo(MODEL_PATH.replace('.pt', '_openvino_model/'))
        elif 'cuda' in self.device:
            # CUDA specific optimizations
            model.to(self.device)
            model.fuse()  # Optimize model for inference
            if torch.cuda.get_device_properties(0).major >= 7:  # Tensor cores
                model.half()  # Use FP16 for modern GPUs
        else:
            # CPU optimizations
            model.fuse()
            torch.set_num_threads(4)  # Optimize CPU threads
        
        return model
    
    def _verify_model_loading(self):
        """Verify model loaded correctly on target device"""
        if hasattr(self.model.model, 'device'):
            loaded_device = self.model.model.device.type
        else:
            loaded_device = 'cpu'
        
        self.logger.info(f"✅ Model loaded on: {loaded_device.upper()}")
        
        if 'cuda' in str(loaded_device):
            gpu_mem = torch.cuda.memory_allocated() / 1e6
            self.logger.info(f"   GPU memory allocated: {gpu_mem:.1f}MB")
    
    def detect(self, frame, confidence=None):
        """Perform object detection with hardware optimizations"""
        if frame is None or frame.size == 0:
            self.logger.warning("Empty frame received for detection")
            return frame, []
        
        # Use custom confidence or default
        conf = confidence if confidence is not None else CONFIDENCE
        
        try:
            # Hardware-specific inference parameters
            inference_params = {
                'verbose': False,
                'conf': conf,
                'iou': DETECTION_PARAMS.get('iou', 0.45),
                'agnostic_nms': DETECTION_PARAMS.get('agnostic_nms', False),
                'max_det': DETECTION_PARAMS.get('max_det', 100),
                'classes': DETECTION_PARAMS.get('classes', None)  # Filter by class
            }
            
            # Add device-specific parameters
            if self.device == 'openvino':
                inference_params['device'] = 'openvino'
            elif 'cuda' in self.device:
                inference_params['device'] = self.device
                inference_params['half'] = True  # FP16 for CUDA
            else:
                inference_params['device'] = 'cpu'
            
            # Perform inference
            results = self.model(frame, **inference_params)
            
            if len(results) == 0:
                return frame, []
            
            result = results[0]
            annotated_frame = result.plot() if hasattr(result, 'plot') else frame
            
            # Extract detection data
            detections = []
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    detection = {
                        'bbox': box.xyxy[0].cpu().numpy().tolist(),
                        'confidence': float(box.conf[0].cpu().numpy()),
                        'class_id': int(box.cls[0].cpu().numpy()),
                        'class_name': self.model.names[int(box.cls[0].cpu().numpy())]
                    }
                    detections.append(detection)
            
            return annotated_frame, detections
            
        except Exception as e:
            self.logger.error(f"Detection error: {str(e)}")
            return frame, []
