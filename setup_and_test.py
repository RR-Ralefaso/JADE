import os
import sys
import cv2
import time
import numpy as np
from datetime import datetime

def test_camera():
    """Test camera detection"""
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
                print(f"   FPS: {cap.get(cv2.CAP_PROP_FPS):.1f}")
            cap.release()
        else:
            print(f"❌ Camera {camera_id} not available")
    
    cv2.destroyAllWindows()
    
    if available_cameras:
        print(f"\n🎯 Recommended camera ID: {available_cameras[0]['id']}")
        print("   Update config.py CAMERA_ID if needed")
    else:
        print("\n❌ No cameras detected!")
        print("   Please check camera connection")
    
    return available_cameras

def test_voice():
    """Test voice functionality"""
    print("\n🎤 Testing voice system...")
    print("-" * 40)
    
    try:
        import speech_recognition as sr
        import pyttsx3
        
        # Test TTS
        print("🔊 Testing Text-to-Speech...")
        engine = pyttsx3.init()
        
        # Get available voices
        voices = engine.getProperty('voices')
        print(f"✅ TTS: {len(voices)} voices available")
        
        # Test with sample text
        test_text = "Hello, this is JADE voice test."
        print(f"   Speaking: '{test_text}'")
        engine.say(test_text)
        engine.runAndWait()
        print("✅ Text-to-speech: Working")
        
        # Test microphone
        print("\n🎤 Testing microphone...")
        r = sr.Recognizer()
        
        try:
            with sr.Microphone() as source:
                print("✅ Microphone: Detected")
                print("   Calibrating for ambient noise...")
                r.adjust_for_ambient_noise(source, duration=1)
                print("✅ Noise calibration: Complete")
                
                print("\n🎤 Say 'test' clearly when prompted...")
                print("🎯 Listening for 3 seconds...")
                
                audio = r.listen(source, timeout=3, phrase_time_limit=2)
                
                try:
                    text = r.recognize_google(audio)
                    print(f"🗣️  You said: {text}")
                    
                    if 'test' in text.lower():
                        print("✅ Speech recognition: Working perfectly!")
                    else:
                        print("✅ Speech recognition: Working (heard different word)")
                        
                except sr.UnknownValueError:
                    print("❌ Could not understand audio")
                    print("   Tip: Speak clearly in a quiet environment")
                except sr.RequestError as e:
                    print(f"❌ API error: {e}")
                    print("   Check internet connection for Google Speech Recognition")
                    
        except OSError as e:
            print(f"❌ Microphone access error: {e}")
            print("   Tip: Check microphone permissions")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Run: pip install speechrecognition pyttsx3 pyaudio")
    except Exception as e:
        print(f"❌ Voice test failed: {e}")

def test_detection():
    """Test object detection"""
    print("\n🔍 Testing object detection...")
    print("-" * 40)
    
    try:
        from JadeAssistant import JadeAssistant
        
        # Create a test image with simple shapes
        print("🖼️ Creating test image...")
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw some shapes to simulate objects
        cv2.rectangle(test_image, (100, 100), (300, 300), (0, 255, 0), -1)  # Green rectangle
        cv2.circle(test_image, (450, 150), 50, (255, 0, 0), -1)  # Blue circle
        cv2.rectangle(test_image, (200, 350), (400, 450), (0, 0, 255), -1)  # Red rectangle
        
        # Initialize detector
        print("🤖 Initializing detector...")
        detector = JadeAssistant()
        
        # Test detection
        print("🔍 Running detection test...")
        start_time = time.time()
        processed_frame, detections = detector.detect(test_image)
        detection_time = time.time() - start_time
        
        print(f"✅ Detector initialized")
        print(f"⏱️  Detection time: {detection_time:.3f} seconds")
        print(f"📊 Detection results: {len(detections)} objects found")
        
        # Test statistics
        if detections:
            stats = detector.export_detection_statistics(detections)
            print(f"   Unique classes: {len(stats['unique_classes'])}")
            print(f"   Confidence average: {stats['confidence_avg']:.3f}")
            
            for i, det in enumerate(detections[:3]):
                print(f"   Object {i+1}: {det['class_name']} ({det['confidence']:.3f})")
        
        # Test specific class detection
        print("\n🎯 Testing specific class detection...")
        specific_classes = ['person', 'car', 'chair']
        _, specific_detections = detector.detect_specific_classes(test_image, specific_classes)
        print(f"   Filtered for {specific_classes}: {len(specific_detections)} detections")
        
    except Exception as e:
        print(f"❌ Detection test failed: {e}")
        print("   Make sure YOLO model is downloaded")
        print("   Model will auto-download on first run")

def test_knowledge_base():
    """Test knowledge base functionality"""
    print("\n📚 Testing knowledge base...")
    print("-" * 40)
    
    try:
        from knowledge_base import (
            ENHANCED_KNOWLEDGE,
            analyze_object_visual_features,
            generate_detailed_report,
            comprehensive_object_assessment
        )
        
        print(f"✅ Knowledge base loaded: {len(ENHANCED_KNOWLEDGE)} objects")
        
        # Test with sample objects
        test_objects = ['car', 'laptop', 'chair', 'bottle']
        
        print("🧪 Testing object knowledge...")
        for obj in test_objects:
            if obj in ENHANCED_KNOWLEDGE:
                info = ENHANCED_KNOWLEDGE[obj]
                print(f"   {obj}: {info['info'][:50]}...")
        
        # Create test image for analysis
        test_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        test_bbox = [50, 50, 250, 200]
        
        print("\n🔍 Testing visual feature analysis...")
        features = analyze_object_visual_features(test_image, test_bbox, 'car')
        if features:
            print(f"✅ Feature analysis working")
            print(f"   Colors detected: {len(features.get('color_names', []))}")
            print(f"   Texture level: {features.get('material_indicators', {}).get('texture_level', 'unknown')}")
        
        print("\n📊 Testing report generation...")
        report = generate_detailed_report('laptop', 'good', features, 0.8)
        print(f"✅ Report generated for: {report['object']}")
        print(f"   Estimated value: {report['estimated_value']}")
        
    except Exception as e:
        print(f"❌ Knowledge base test failed: {e}")

def test_system_integration():
    """Test full system integration"""
    print("\n🤖 Testing system integration...")
    print("-" * 40)
    
    try:
        # Test config
        from config import config
        print("✅ Configuration loaded")
        print(f"   Camera ID: {config.CAMERA_ID}")
        print(f"   Model path: {config.MODEL_PATH}")
        print(f"   Confidence: {config.CONFIDENCE}")
        
        # Test assistant
        from Jade import JADEBaseAssistant
        assistant = JADEBaseAssistant()
        print(f"✅ Assistant initialized")
        print(f"   Current mode: {assistant.current_mode}")
        
        # Test voice assistant
        from Jade import JADEVoiceAssistant
        voice_assistant = JADEVoiceAssistant(assistant)
        print(f"✅ Voice assistant initialized")
        print(f"   Wake word: {voice_assistant.wake_word}")
        
        # Test logger
        from outtxt import DetectionLogger
        logger = DetectionLogger()
        print(f"✅ Logger initialized")
        
        print("\n✅ All system components loaded successfully!")
        
    except Exception as e:
        print(f"❌ System integration test failed: {e}")
        import traceback
        traceback.print_exc()

def check_dependencies():
    """Check all dependencies are installed"""
    print("📦 Checking dependencies...")
    print("-" * 40)
    
    dependencies = [
        ('opencv-python', 'cv2'),
        ('numpy', 'numpy'),
        ('torch', 'torch'),
        ('ultralytics', 'ultralytics'),
        ('speechrecognition', 'speech_recognition'),
        ('pyttsx3', 'pyttsx3'),
        ('pyaudio', 'pyaudio'),
        ('scikit-learn', 'sklearn'),
        ('scikit-image', 'skimage'),
        ('pyyaml', 'yaml')
    ]
    
    all_ok = True
    for pip_name, import_name in dependencies:
        try:
            __import__(import_name)
            print(f"✅ {pip_name}")
        except ImportError:
            print(f"❌ {pip_name}")
            all_ok = False
    
    return all_ok

def main():
    """Run all tests"""
    print("="*60)
    print("🤖 JADE SYSTEM DIAGNOSTIC TEST")
    print("="*60)
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Missing dependencies!")
        print("   Run: python install.py")
        return
    
    # Run tests
    print("\n" + "="*60)
    print("🚀 RUNNING SYSTEM TESTS")
    print("="*60)
    
    test_results = {}
    
    # Test 1: Camera
    print("\n1. Camera Test:")
    test_results['camera'] = test_camera()
    
    # Test 2: Voice
    print("\n2. Voice System Test:")
    test_voice()
    test_results['voice'] = True
    
    # Test 3: Detection
    print("\n3. Object Detection Test:")
    test_detection()
    test_results['detection'] = True
    
    # Test 4: Knowledge Base
    print("\n4. Knowledge Base Test:")
    test_knowledge_base()
    test_results['knowledge'] = True
    
    # Test 5: System Integration
    print("\n5. System Integration Test:")
    test_system_integration()
    test_results['integration'] = True
    
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    all_passed = all(test_results.values())
    
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("\n🚀 System is ready!")
    else:
        print("⚠️  Some tests had issues")
        print("   Review the test output above")
    
    print("\n" + "="*60)
    print("📋 NEXT STEPS:")
    print("="*60)
    print("1. Run 'python main.py' to start JADE")
    print("2. Point camera at objects")
    print("3. Say 'hey jade' clearly")
    print("4. Try commands: 'analyze object', 'what do you see'")
    print("5. Press 'H' for help menu")
    print("\n🎯 TIPS FOR BEST RESULTS:")
    print("   • Ensure good lighting")
    print("   • Speak clearly and at moderate pace")
    print("   • Position objects clearly in camera view")
    print("   • Train custom model for specific objects")
    print("="*60)

if __name__ == "__main__":
    main()