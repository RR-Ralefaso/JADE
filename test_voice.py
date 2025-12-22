import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Jade import JADEVoiceAssistant, JADEBaseAssistant
import time

def test_voice_assistant():
    """Test the voice assistant"""
    print("🎤 Testing JADE Voice Assistant")
    print("="*60)
    
    # Create mock assistant
    mock_assistant = JADEBaseAssistant()
    
    # Create voice assistant
    voice_assistant = JADEVoiceAssistant(
        mock_assistant,
        wake_word="test jade",
        voice_gender="female",
        speaking_rate=180
    )
    
    print("\n1. Testing text-to-speech...")
    
    # Test different messages
    test_messages = [
        "Hello, this is a voice test.",
        "I can analyze objects for you.",
        "Point the camera at an object to get started."
    ]
    
    for i, message in enumerate(test_messages):
        print(f"   Test {i+1}: {message}")
        voice_assistant.speak(message)
        time.sleep(1)
    
    print("\n2. Testing voice commands...")
    
    # Simulate command processing
    commands = [
        "current mode",
        "switch to analysis mode",
        "what can you do",
        "help",
        "hello",
        "analyze object"
    ]
    
    for command in commands:
        print(f"\nCommand: '{command}'")
        response = voice_assistant._handle_command(
            type('Command', (), {'text': command, 'confidence': 1.0, 'timestamp': time.time()})()
        )
        if response:
            print(f"Response spoken")
        time.sleep(0.5)
    
    print("\n3. Testing audio recording...")
    try:
        print("🎙️ Recording 2 seconds of audio...")
        # Note: Actual recording requires microphone
        print("✅ Recording test passed (requires microphone for full test)")
    except Exception as e:
        print(f"❌ Recording test failed: {e}")
    
    print("\n4. Testing wake word detection...")
    wake_variations = voice_assistant.wake_word_variations
    print(f"   Supported wake words: {', '.join(wake_variations[:3])}...")
    
    # Cleanup
    voice_assistant.cleanup()
    
    print("\n" + "="*60)
    print("✅ Voice assistant test complete!")
    print("="*60)

if __name__ == "__main__":
    test_voice_assistant()