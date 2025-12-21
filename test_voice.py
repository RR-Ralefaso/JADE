
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Jade import JADEVoiceAssistant, JADEBaseAssistant

def test_voice_assistant():
    """Test the voice assistant"""
    print("🎤 Testing JADE Voice Assistant")
    print("="*50)
    
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
    voice_assistant.speak("Hello! This is a test of the JADE voice system.")
    
    print("\n2. Testing voice commands...")
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
        response = voice_assistant._execute_command(command)
        print(f"Response: {response}")
    
    print("\n3. Testing audio recording...")
    try:
        audio_data = voice_assistant.record_audio_numpy(duration=2)
        print(f"✅ Recorded {len(audio_data)} audio samples")
        
        # Save the recording
        filename = voice_assistant.save_audio_numpy(audio_data)
        print(f"💾 Saved to: {filename}")
    except Exception as e:
        print(f"❌ Recording failed: {e}")
        print("Note: Audio recording may require microphone permissions")
    
    print("\n4. Testing voice profiles...")
    profiles = ['friendly', 'professional', 'analytical']
    for profile in profiles:
        response = voice_assistant.change_voice_profile(profile)
        print(f"  {profile}: {response}")
    
    # Cleanup
    voice_assistant.cleanup()
    
    print("\n✅ Voice assistant test complete!")

if __name__ == "__main__":
    test_voice_assistant()
