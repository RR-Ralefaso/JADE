import speech_recognition as sr
import pyttsx3
import threading
import queue
import time
import json
import os
from datetime import datetime
import numpy as np
import pyaudio
from pathlib import Path

class JADEBaseAssistant:
    """Base assistant class that JadeVoiceAssistant can interact with"""
    def __init__(self):
        self.current_mode = 'conversational'
    
    def change_mode(self, mode):
        self.current_mode = mode
        return f"Mode changed to {mode}"
    
    def chat(self, message, context=None):
        # Simple responses without API calls
        if "hello" in message.lower() or "hi" in message.lower():
            return "Hello! I'm JADE, your voice-enabled object analyzer."
        elif "how are you" in message.lower():
            return "I'm functioning optimally! Ready to analyze objects for you."
        elif "thank" in message.lower():
            return "You're welcome! I'm here to help."
        elif "analyze" in message.lower():
            return "I can analyze objects detected by the camera. Point the camera at an object."
        elif "what can you do" in message.lower():
            return "I can analyze objects through camera, identify them, estimate their value, and chat with you!"
        else:
            return f"I heard: '{message}'. I'm currently in {self.current_mode} mode. How can I assist you with object analysis?"
    
    def clear_conversation(self):
        return "Conversation cleared"
    
    def _explain_capabilities(self):
        return "I can analyze objects through camera, identify them, estimate value, and have conversations."
    
    def _explain_modes(self):
        return "Available modes: analysis (detailed object analysis), conversational (chat mode)"

class JADEVoiceAssistant:
    def __init__(self, jade_assistant=None, wake_word="hey jade", voice_gender='female', speaking_rate=180):
        """Initialize enhanced voice assistant for JADE"""
        # Use provided assistant or create a base one
        self.jade_assistant = jade_assistant if jade_assistant else JADEBaseAssistant()
        self.wake_word = wake_word.lower()
        
        # Speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.listening = False
        self.is_awake = False
        
        # Text-to-speech
        self.tts_engine = pyttsx3.init()
        self.setup_tts(voice_gender, speaking_rate)
        
        # Audio recording
        self.audio = pyaudio.PyAudio()
        self.is_recording = False
        self.audio_queue = queue.Queue()
        
        # Thread management
        self.listening_thread = None
        self.processing_thread = None
        self.speaking_thread = None
        self.command_queue = queue.Queue()
        
        # Voice settings
        self.voice_commands = self._load_voice_commands()
        self.response_cache = {}
        self.conversation_context = []
        self.max_context_length = 10
        
        # Audio processing
        self.audio_chunk = 1024
        self.audio_format = pyaudio.paInt16
        self.audio_channels = 1
        self.audio_rate = 16000
        
        # Voice profiles
        self.voice_profiles = {
            'professional': {'rate': 170, 'volume': 0.9},
            'friendly': {'rate': 160, 'volume': 1.0},
            'analytical': {'rate': 150, 'volume': 0.8},
            'enthusiastic': {'rate': 190, 'volume': 1.0}
        }
        
        # Create voice logs directory
        self.voice_logs_dir = Path("voice_logs")
        self.voice_logs_dir.mkdir(exist_ok=True)
        
        print(f"🎤 JADE Voice Assistant Initialized")
        print(f"📢 Wake word: '{self.wake_word}'")
        print(f"🗣️  Voice profile: {voice_gender}")
    
    def setup_tts(self, gender='female', rate=180):
        """Setup text-to-speech engine"""
        voices = self.tts_engine.getProperty('voices')
        
        if gender.lower() == 'female':
            # Try to find a female voice
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
        else:
            # Use default male voice
            for voice in voices:
                if 'male' in voice.name.lower() or 'david' in voice.name.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
        
        self.tts_engine.setProperty('rate', rate)
        self.tts_engine.setProperty('volume', 1.0)
    
    def _load_voice_commands(self):
        """Load predefined voice commands"""
        commands = {
            # Mode switching
            "switch to analysis mode": lambda: self.jade_assistant.change_mode('analysis'),
            "switch to conversational mode": lambda: self.jade_assistant.change_mode('conversational'),
            
            # Analysis commands
            "analyze object": self._analyze_current_object,
            "what do you see": self._describe_scene,
            "identify objects": self._identify_objects,
            "describe scene": self._describe_scene,
            
            # System commands
            "stop listening": self.stop_listening,
            "start listening": self.start_listening,
            "clear conversation": lambda: self.jade_assistant.clear_conversation(),
            "what can you do": lambda: self.jade_assistant._explain_capabilities(),
            
            # Information requests
            "current mode": lambda: f"Current mode: {self.jade_assistant.current_mode}",
            "list modes": lambda: self.jade_assistant._explain_modes(),
            "help": lambda: self._provide_help(),
            
            # Voice control
            "speak faster": lambda: self._adjust_speech_rate(40),
            "speak slower": lambda: self._adjust_speech_rate(-40),
            "speak louder": lambda: self._adjust_volume(0.2),
            "speak softer": lambda: self._adjust_volume(-0.2),
            
            # Greetings
            "hello": lambda: "Hello! I'm JADE, ready to analyze objects.",
            "hi": lambda: "Hi there! How can I help you today?",
            "good morning": lambda: "Good morning! Ready for object analysis.",
            "good afternoon": lambda: "Good afternoon! Let's analyze some objects.",
            "good evening": lambda: "Good evening! I'm here to assist you.",
        }
        
        return commands
    
    def _adjust_speech_rate(self, delta):
        """Adjust speech rate"""
        current_rate = self.tts_engine.getProperty('rate')
        new_rate = max(80, min(400, current_rate + delta))
        self.tts_engine.setProperty('rate', new_rate)
        return f"Speech rate adjusted to {new_rate}"
    
    def _adjust_volume(self, delta):
        """Adjust volume"""
        current_volume = self.tts_engine.getProperty('volume')
        new_volume = max(0.1, min(1.0, current_volume + delta))
        self.tts_engine.setProperty('volume', new_volume)
        return f"Volume adjusted to {new_volume:.1f}"
    
    def start_listening(self):
        """Start continuous voice listening"""
        if self.listening:
            self.speak("I'm already listening!")
            return
        
        self.listening = True
        self.is_awake = True
        
        # Initialize microphone
        try:
            self.microphone = sr.Microphone()
            print("🎤 Microphone initialized successfully")
        except Exception as e:
            print(f"❌ Microphone error: {e}")
            self.speak("I cannot access the microphone. Please check your audio settings.")
            return
        
        # Start listening thread
        self.listening_thread = threading.Thread(target=self._continuous_listen, daemon=True)
        self.listening_thread.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_commands, daemon=True)
        self.processing_thread.start()
        
        self.speak(f"Voice assistant activated. Say '{self.wake_word}' to get my attention!")
        print("✅ Voice listening started")
    
    def stop_listening(self):
        """Stop voice listening"""
        self.listening = False
        self.is_awake = False
        
        if self.listening_thread:
            self.listening_thread.join(timeout=1)
        
        if self.processing_thread:
            self.processing_thread.join(timeout=1)
        
        self.speak("Voice assistant deactivated. Goodbye!")
        print("🛑 Voice listening stopped")
    
    def _continuous_listen(self):
        """Continuously listen for voice input"""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("🔊 Calibrated for ambient noise")
            
            while self.listening:
                try:
                    print("👂 Listening... (say 'hey jade' to wake me)")
                    
                    # Listen for audio with timeout
                    audio = self.recognizer.listen(
                        source, 
                        timeout=5, 
                        phrase_time_limit=10
                    )
                    
                    # Convert speech to text
                    try:
                        text = self.recognizer.recognize_google(audio)
                        print(f"🗣️  Heard: {text}")
                        
                        # Check for wake word
                        if self.wake_word in text.lower():
                            if not self.is_awake:
                                self.is_awake = True
                                self.speak("Yes, I'm listening!")
                                print(f"✅ Woken up by '{self.wake_word}'")
                            
                            # Extract command after wake word
                            command = text.lower().split(self.wake_word, 1)[1].strip()
                            if command:
                                self.command_queue.put(command)
                                print(f"📝 Command queued: {command}")
                        elif self.is_awake:
                            # Direct command if already awake
                            self.command_queue.put(text.lower())
                            print(f"📝 Direct command: {text}")
                            
                    except sr.UnknownValueError:
                        if self.is_awake:
                            self.speak("I didn't catch that. Could you repeat?")
                        print("🤔 Could not understand audio")
                    except sr.RequestError as e:
                        print(f"❌ Recognition error: {e}")
                        if self.is_awake:
                            self.speak("I'm having trouble with speech recognition.")
                
                except sr.WaitTimeoutError:
                    # No speech detected, check if we should go back to sleep
                    if self.is_awake:
                        print("⏰ No speech detected for 10 seconds, going back to sleep")
                        self.is_awake = False
                    continue
                except Exception as e:
                    print(f"❌ Listening error: {e}")
                    time.sleep(1)
    
    def _process_commands(self):
        """Process queued voice commands"""
        while self.listening:
            try:
                # Get command from queue with timeout
                command = self.command_queue.get(timeout=1)
                
                if command:
                    print(f"⚡ Processing command: {command}")
                    
                    # Add to conversation context
                    self.conversation_context.append({
                        'role': 'user',
                        'content': command,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Keep context manageable
                    if len(self.conversation_context) > self.max_context_length:
                        self.conversation_context.pop(0)
                    
                    # Process command
                    response = self._execute_command(command)
                    
                    # Speak response
                    if response:
                        self.speak(response)
                        
                        # Add to conversation context
                        self.conversation_context.append({
                            'role': 'assistant',
                            'content': response,
                            'timestamp': datetime.now().isoformat()
                        })
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Command processing error: {e}")
                self.speak("I encountered an error processing your command.")
    
    def _execute_command(self, command):
        """Execute voice command and return response"""
        # Check for exact command matches first
        for cmd_key, cmd_func in self.voice_commands.items():
            if cmd_key in command:
                try:
                    result = cmd_func()
                    return result if isinstance(result, str) else "Command executed successfully."
                except Exception as e:
                    print(f"❌ Command error: {e}")
                    return f"I couldn't execute that command: {str(e)}"
        
        # Check for patterns
        if "mode" in command:
            # Try to extract mode from command
            modes = ['analysis', 'conversational']
            for mode in modes:
                if mode in command:
                    return self.jade_assistant.change_mode(mode)
        
        # Default: Pass to JADE assistant for processing
        print(f"🤖 Passing to JADE assistant: {command}")
        return self.jade_assistant.chat(command, self.conversation_context)
    
    def _analyze_current_object(self):
        """Analyze currently selected object"""
        return "Please select an object through the camera interface, then ask me to analyze it. I can identify objects and estimate their value."
    
    def _describe_scene(self):
        """Describe current scene from camera"""
        return "I can see objects through the camera. Please point the camera at what you want me to analyze."
    
    def _identify_objects(self):
        """Identify objects in current view"""
        return "Looking at the camera feed now. I'll identify all visible objects using object detection."
    
    def _provide_help(self):
        """Provide voice command help"""
        help_text = """
        Here are some voice commands you can use:
        
        • "Hey Jade" followed by any question
        • "Switch to [analysis/conversational] mode" - Change my operation mode
        • "Analyze object" - Analyze current object
        • "What do you see" - Describe the scene
        • "Describe scene" - Describe what's in view
        • "Current mode" - Check my current mode
        • "Stop listening" - Deactivate voice assistant
        • "Speak faster/slower" - Adjust speech speed
        • "Speak louder/softer" - Adjust volume
        • "Hello/Hi" - Greet me
        • "What can you do" - Learn about my capabilities
        
        You can also ask me anything about objects!
        """
        return help_text
    
    def speak(self, text, voice_profile='friendly'):
        """Speak text aloud with voice profiling"""
        if not text:
            return
        
        print(f"🗣️  Speaking: {text[:100]}...")
        
        # Apply voice profile
        profile = self.voice_profiles.get(voice_profile, self.voice_profiles['friendly'])
        self.tts_engine.setProperty('rate', profile['rate'])
        self.tts_engine.setProperty('volume', profile['volume'])
        
        # Save to voice log
        self._log_voice_interaction(text)
        
        # Speak in a separate thread to not block
        def _speak():
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"❌ TTS error: {e}")
        
        speaking_thread = threading.Thread(target=_speak, daemon=True)
        speaking_thread.start()
        
        # Wait a bit for speech to start
        time.sleep(0.1)
    
    def _log_voice_interaction(self, text, user_input=None):
        """Log voice interactions"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'mode': self.jade_assistant.current_mode,
            'user_input': user_input,
            'response': text[:500]
        }
        
        log_file = self.voice_logs_dir / f"voice_log_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"❌ Failed to log voice interaction: {e}")
    
    def record_audio_numpy(self, duration=5):
        """Record audio and return as numpy array"""
        stream = self.audio.open(
            format=self.audio_format,
            channels=self.audio_channels,
            rate=self.audio_rate,
            input=True,
            frames_per_buffer=self.audio_chunk
        )
        
        print(f"🎙️ Recording audio for {duration} seconds...")
        frames = []
        
        for i in range(0, int(self.audio_rate / self.audio_chunk * duration)):
            data = stream.read(self.audio_chunk)
            frames.append(data)
        
        print("✅ Recording complete")
        
        stream.stop_stream()
        stream.close()
        
        # Convert to numpy array
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        
        return audio_data
    
    def save_audio_numpy(self, audio_data, filename=None):
        """Save numpy audio data to raw file"""
        if not filename:
            filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.raw"
        
        filepath = self.voice_logs_dir / filename
        
        # Save raw audio data
        audio_data.tofile(str(filepath))
        
        # Save metadata
        metadata = {
            'sample_rate': self.audio_rate,
            'channels': self.audio_channels,
            'dtype': str(audio_data.dtype),
            'samples': len(audio_data),
            'duration': len(audio_data) / self.audio_rate
        }
        
        metadata_file = filepath.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Saved audio to {filename}")
        return filepath
    
    def play_audio_numpy(self, audio_data):
        """Play numpy audio data"""
        stream = self.audio.open(
            format=self.audio_format,
            channels=self.audio_channels,
            rate=self.audio_rate,
            output=True
        )
        
        # Convert to bytes
        audio_bytes = audio_data.tobytes()
        
        stream.write(audio_bytes)
        stream.stop_stream()
        stream.close()
    
    def change_voice_profile(self, profile_name):
        """Change voice profile"""
        if profile_name in self.voice_profiles:
            profile = self.voice_profiles[profile_name]
            self.tts_engine.setProperty('rate', profile['rate'])
            self.tts_engine.setProperty('volume', profile['volume'])
            return f"Changed voice profile to {profile_name}"
        else:
            return f"Unknown profile. Available: {', '.join(self.voice_profiles.keys())}"
    
    def get_conversation_summary(self):
        """Get summary of voice conversation"""
        if not self.conversation_context:
            return "No voice conversation yet."
        
        summary = "Voice Conversation Summary:\n\n"
        for entry in self.conversation_context[-5:]:
            role = "You" if entry['role'] == 'user' else "JADE"
            summary += f"{role}: {entry['content'][:80]}...\n"
        
        return summary
    
    def toggle_wake_word(self, enable=True):
        """Enable or disable wake word requirement"""
        if enable:
            self.is_awake = False
            return "Wake word enabled. Say 'hey jade' to get my attention."
        else:
            self.is_awake = True
            return "Wake word disabled. I'm always listening for commands."
    
    def cleanup(self):
        """Cleanup audio resources"""
        self.audio.terminate()