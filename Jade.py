import speech_recognition as sr
import pyttsx3
import threading
import queue
import time
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, List

@dataclass
class VoiceCommand:
    text: str
    confidence: float
    timestamp: float

class JADEBaseAssistant:
    """Base assistant class"""
    def __init__(self):
        self.current_mode = 'conversational'
        self.object_analysis_enabled = True
    
    def change_mode(self, mode):
        valid_modes = ['analysis', 'conversational', 'detection_only']
        if mode in valid_modes:
            self.current_mode = mode
            return f"Mode changed to {mode} mode"
        return f"Invalid mode. Available: {', '.join(valid_modes)}"
    
    def chat(self, message, context=None):
        message_lower = message.lower()
        
        responses = {
            "hello": "Hello! I'm JADE, your voice-enabled object analyzer.",
            "hi": "Hi there! Ready to analyze objects for you.",
            "how are you": "I'm functioning optimally! Ready to analyze objects.",
            "thank": "You're welcome! I'm here to help.",
            "what can you do": "I can analyze objects, estimate their value, and chat with you!",
            "analyze": "I can analyze objects detected by the camera. Point the camera at an object.",
            "help": "I can analyze objects, estimate value, and chat. Try 'analyze object' or 'what do you see'.",
        }
        
        for key in responses:
            if key in message_lower:
                return responses[key]
        
        if "mode" in message_lower:
            if "analysis" in message_lower:
                return self.change_mode('analysis')
            elif "conversational" in message_lower:
                return self.change_mode('conversational')
            elif "detection" in message_lower:
                return self.change_mode('detection_only')
        
        return f"I heard: '{message}'. I'm in {self.current_mode} mode. How can I assist you?"

class JADEVoiceAssistant:
    def __init__(self, jade_assistant=None, wake_word="hey jade", voice_gender='female', speaking_rate=180):
        self.jade_assistant = jade_assistant if jade_assistant else JADEBaseAssistant()
        self.wake_word = wake_word.lower()
        
        # Wake word variations
        self.wake_word_variations = [
            "hey jade", "jade", "okay jade", "listen jade",
            "attention jade", "hello jade", "hi jade"
        ]
        
        # Speech recognition - using a simpler approach
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 1000  # Higher to reduce background noise
        self.recognizer.dynamic_energy_threshold = False  # Disable to avoid issues
        
        # Text-to-speech
        self.tts_engine = pyttsx3.init()
        self.setup_tts(voice_gender, speaking_rate)
        
        # State management
        self.is_listening = False
        self.is_awake = False
        self.command_queue = queue.Queue(maxsize=50)
        
        # Threads
        self.listening_thread = None
        self.processing_thread = None
        
        # Performance
        self.last_command_time = 0
        self.command_cooldown = 1.0
        
        print(f"🎤 JADE Voice Assistant Initialized")
        print(f"📢 Wake word: '{self.wake_word}'")
    
    def setup_tts(self, gender='female', rate=180):
        """Setup text-to-speech engine"""
        try:
            voices = self.tts_engine.getProperty('voices')
            
            if gender.lower() == 'female':
                for voice in voices:
                    if 'female' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            else:
                for voice in voices:
                    if 'male' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            
            self.tts_engine.setProperty('rate', rate)
            self.tts_engine.setProperty('volume', 0.8)  # Lower volume to avoid issues
            
        except Exception as e:
            print(f"⚠️  TTS setup error: {e}")
    
    def start_listening(self):
        """Start voice listening (simplified version)"""
        if self.is_listening:
            return
        
        self.is_listening = True
        
        # Start processing thread only (no continuous listening to avoid audio issues)
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self.processing_thread.start()
        
        print("✅ Voice assistant ready - press '1' to test voice")
        print("⚠️  Continuous listening disabled to avoid audio issues")
    
    def stop_listening(self):
        """Stop voice listening"""
        self.is_listening = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        
        print("🛑 Voice assistant stopped")
    
    def listen_once(self):
        """Listen for a single command"""
        try:
            print("🎤 Listening for command...")
            
            with sr.Microphone() as source:
                # Adjust for ambient noise
                print("   Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                # Listen for audio
                print("   Speak now...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                
                # Recognize speech
                try:
                    text = self.recognizer.recognize_google(audio).lower()
                    print(f"🗣️  Heard: {text}")
                    return text
                except sr.UnknownValueError:
                    print("🤔 Could not understand audio")
                    return None
                except sr.RequestError as e:
                    print(f"❌ API error: {e}")
                    return None
                    
        except Exception as e:
            print(f"❌ Listening error: {e}")
            return None
    
    def _processing_loop(self):
        """Process voice commands from queue"""
        while self.is_listening:
            try:
                command = self.command_queue.get(timeout=0.5)
                if command:
                    self._handle_command(command)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Processing error: {e}")
                time.sleep(0.1)
    
    def _handle_command(self, command):
        """Handle voice command"""
        text = command.text.lower()
        
        responses = {
            'what do you see': "I see objects through the camera. Please point at something.",
            'analyze object': "I'll analyze the objects in view. Please point the camera.",
            'switch to analysis mode': "Switching to analysis mode.",
            'switch to conversational mode': "Switching to conversational mode.",
            'stop listening': lambda: self._go_to_sleep(),
            'help': "I can analyze objects, estimate value, and provide information.",
            'hello': "Hello! I'm JADE, ready to help.",
            'test': "Voice test successful! I can hear you clearly.",
        }
        
        for key, response in responses.items():
            if key in text:
                if callable(response):
                    result = response()
                else:
                    result = response
                
                self.speak(result)
                return
        
        # Check for patterns
        if 'mode' in text:
            if 'analysis' in text:
                self.speak("Switching to analysis mode.")
            elif 'conversational' in text:
                self.speak("Switching to conversational mode.")
            else:
                self.speak("Available modes: analysis and conversational.")
        
        elif 'how are you' in text:
            self.speak("I'm functioning optimally! Ready to analyze objects.")
        
        else:
            self.speak(f"I heard: {text}. How can I help with object analysis?")
    
    def _go_to_sleep(self):
        """Put assistant to sleep"""
        self.is_awake = False
        return "Going to sleep. Say 'hey jade' to wake me up."
    
    def speak(self, text):
        """Speak text aloud"""
        if not text:
            return
        
        print(f"🗣️  Speaking: {text[:100]}...")
        
        def _speak():
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"❌ TTS error: {e}")
        
        # Use threading for non-blocking speech
        threading.Thread(target=_speak, daemon=True).start()
    
    def test_voice(self):
        """Test voice functionality"""
        print("🔊 Testing voice system...")
        self.speak("Hello! This is a voice test. JADE is working correctly.")
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_listening()
        print("✅ Audio resources cleaned up")