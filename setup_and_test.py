import os
import cv2
import numpy as np
import time
from JadeAssistant import JadeAssistant
from knowledge_base import knowledge_base
from config import config

def test_camera():
    print("📷 Testing camera...")
    print("-" * 40)
    
    available_cameras = []
    
    for camera_id in range(5):
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append({
                    'id': camera_id,
                    'resolution': f"{frame.shape[1]}x{frame.shape[0]}",
                    'fps': cap.get(cv2.CAP_PROP_FPS)
                })
                print(f"✅ Camera {camera_id} detected")
                print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
            cap.release()
        else:
            print(f"❌ Camera {camera_id} not available")
    
    cv2.destroyAllWindows()
    
    if available_cameras:
        print(f"\n🎯 Recommended camera ID: {available_cameras[0]['id']}")
    else:
        print("\n❌ No cameras detected!")
    
    return available_cameras

def test_detection():
    print("\n🔍 Testing object detection...")
    print("-" * 40)
    
    try:
        # Create test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (100, 100), (300, 300), (0, 255, 0), -1)
        cv2.circle(test_image, (450, 150), 50, (255, 0, 0), -1)
        
        # Initialize detector
        detector = JadeAssistant()
        
        # Test detection
        start_time = time.time()
        processed_frame, detections = detector.detect(test_image)
        detection_time = time.time() - start_time
        
        print(f"✅ Detector initialized")
        print(f"⏱️  Detection time: {detection_time:.3f} seconds")
        print(f"📊 Detections: {len(detections)} objects")
        
        if detections:
            stats = detector.export_detection_statistics(detections)
            print(f"   Unique classes: {len(stats['unique_classes'])}")
            print(f"   Confidence average: {stats['confidence_avg']:.3f}")
        
        # Test performance
        perf_stats = detector.get_performance_stats()
        if perf_stats:
            print(f"\n📊 Performance Stats:")
            print(f"   Avg FPS: {perf_stats.get('avg_fps', 0):.1f}")
            print(f"   Avg inference: {perf_stats.get('avg_inference_time', 0)*1000:.1f}ms")
        
    except Exception as e:
        print(f"❌ Detection test failed: {e}")

def test_knowledge_base():
    print("\n📚 Testing knowledge base...")
    print("-" * 40)
    
    print(f"✅ Knowledge base loaded: {len(knowledge_base.objects)} objects")
    
    # Test with sample objects
    test_objects = ['car', 'laptop', 'chair', 'bottle']
    
    for obj in test_objects:
        info = knowledge_base.get_object_info(obj)
        if info:
            print(f"   {obj}: {info.category} - Base value: ${info.base_value:,.2f}")
        else:
            print(f"   {obj}: Not in knowledge base")
    
    # Test value estimation
    print("\n💰 Testing value estimation...")
    for obj in ['car', 'laptop']:
        value = knowledge_base.estimate_object_value(obj, 'good')
        print(f"   {obj} (good condition): {value}")

def test_system_integration():
    print("\n🤖 Testing system integration...")
    print("-" * 40)
    
    try:
        from Jade import JADEBaseAssistant, JADEVoiceAssistant
        from outtxt import DetectionLogger
        
        # Test config
        print("✅ Configuration loaded")
        print(f"   Camera ID: {config.CAMERA_ID}")
        print(f"   Model: {config.MODEL_PATH}")
        
        # Test assistant
        assistant = JADEBaseAssistant()
        print(f"✅ Assistant initialized")
        print(f"   Current mode: {assistant.current_mode}")
        
        # Test voice assistant
        voice_assistant = JADEVoiceAssistant(assistant)
        print(f"✅ Voice assistant initialized")
        
        # Test logger
        logger = DetectionLogger()
        print(f"✅ Logger initialized")
        
        print("\n✅ All system components loaded successfully!")
        
    except Exception as e:
        print(f"❌ System integration test failed: {e}")

def main():
    print("="*60)
    print("🤖 JADE SYSTEM DIAGNOSTIC TEST")
    print("="*60)
    
    # Run tests
    test_camera()
    test_detection()
    test_knowledge_base()
    test_system_integration()
    
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print("✅ All tests completed!")
    print("\n🚀 To start JADE:")
    print("   python main.py")
    print("\n🎯 Tips for best results:")
    print("   • Ensure good lighting")
    print("   • Speak clearly")
    print("   • Position objects clearly in view")
    print("="*60)

if __name__ == "__main__":
    main()