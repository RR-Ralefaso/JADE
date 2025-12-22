import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Jade import JADEVoiceAssistant, JADEBaseAssistant

def test_voice_assistant():
    """Enhanced test for the voice assistant"""
    print("🎤 Testing JADE Enhanced Voice Assistant")
    print("="*60)
    
    # Create mock assistant
    mock_assistant = JADEBaseAssistant()
    
    # Create voice assistant with enhanced settings
    voice_assistant = JADEVoiceAssistant(
        mock_assistant,
        wake_word="test jade",
        voice_gender="female",
        speaking_rate=180
    )
    
    print("\n1. Testing text-to-speech with different profiles...")
    profiles = ['friendly', 'professional', 'analytical', 'calm']
    
    for profile in profiles:
        print(f"\n   Profile: {profile}")
        voice_assistant.change_voice_profile(profile)
        voice_assistant.speak(f"This is the {profile} voice profile.")
    
    print("\n2. Testing voice commands...")
    commands = [
        "current mode",
        "switch to analysis mode",
        "switch to conversational mode",
        "switch to detection mode",
        "what can you do",
        "help",
        "hello",
        "analyze object",
        "what do you see",
        "go to sleep",
        "wake up"
    ]
    
    for command in commands:
        print(f"\nCommand: '{command}'")
        response = voice_assistant._execute_command(command)
        print(f"Response: {response}")
    
    print("\n3. Testing audio recording and analysis...")
    try:
        print("🎙️ Recording 2 seconds of audio...")
        audio_data = voice_assistant.record_audio_numpy(duration=2)
        print(f"✅ Recorded {len(audio_data)} audio samples")
        
        # Analyze audio quality
        analysis = voice_assistant.analyze_audio_quality(audio_data)
        print(f"📊 Audio analysis:")
        print(f"   RMS: {analysis['rms']:.1f}")
        print(f"   Peak: {analysis['peak']}")
        print(f"   SNR: {analysis['snr']:.1f} dB")
        print(f"   Clear audio: {'Yes' if analysis['is_clear'] else 'No'}")
        
        # Save the recording
        filename = voice_assistant.save_audio_numpy(audio_data)
        print(f"💾 Saved to: {filename}")
        
    except Exception as e:
        print(f"❌ Recording failed: {e}")
        print("Note: Audio recording may require microphone permissions")
    
    print("\n4. Testing wake word variations...")
    wake_variations = voice_assistant.wake_word_variations
    print(f"   Supported wake words: {', '.join(wake_variations)}")
    
    print("\n5. Testing conversation context...")
    test_conversation = [
        "What mode are you in?",
        "Tell me about your capabilities",
        "Can you analyze objects?",
        "How do I use you?"
    ]
    
    for message in test_conversation:
        print(f"\nYou: {message}")
        response = voice_assistant.jade_assistant.chat(message)
        print(f"JADE: {response}")
    
    print("\n6. Getting conversation summary...")
    summary = voice_assistant.get_conversation_summary()
    print(summary)
    
    # Test microphone functionality
    print("\n7. Testing microphone functionality...")
    try:
        response = voice_assistant._test_microphone()
        print(f"Microphone test: {response}")
    except Exception as e:
        print(f"Microphone test error: {e}")
    
    # Cleanup
    voice_assistant.cleanup()
    
    print("\n" + "="*60)
    print("✅ Enhanced voice assistant test complete!")
    print("="*60)

if __name__ == "__main__":
    test_voice_assistant()