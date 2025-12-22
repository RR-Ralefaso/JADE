import speech_recognition as sr
import pyttsx3
import threading
import queue
import time
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, List
import requests

@dataclass
class VoiceCommand:
    text: str
    confidence: float
    timestamp: float

class JADEBaseAssistant:
    """Enhanced JADE assistant with internet access and better analysis"""
    def __init__(self):
        self.current_mode = 'conversational'
        self.object_analysis_enabled = True
        self.conversation_context = []
        self.api_keys = {}
        
    def change_mode(self, mode):
        valid_modes = ['analysis', 'conversational', 'detection_only']
        if mode in valid_modes:
            self.current_mode = mode
            # Store context
            self.conversation_context.append(f"Mode changed to {mode}")
            return f"Mode changed to {mode} mode"
        return f"Invalid mode. Available: {', '.join(valid_modes)}"
    
    def analyze_object_with_context(self, object_name, features=None):
        """Enhanced object analysis with context"""
        analysis = {
            'object': object_name,
            'condition': features.get('condition_indicators', {}).get('overall_condition', 'unknown') if features else 'unknown',
            'materials': features.get('material_indicators', {}).get('possible_materials', []) if features else [],
            'color': features.get('color_name', 'unknown') if features else 'unknown',
            'value_estimate': 'Calculating...',
            'maintenance_tips': [],
            'online_suggestions': []
        }
        
        # Try to get online suggestions
        try:
            suggestions = self.get_online_suggestions(object_name)
            analysis['online_suggestions'] = suggestions[:3]
        except:
            pass
        
        return analysis
    
    def get_online_suggestions(self, query):
        """Get online suggestions for objects"""
        try:
            # Use DuckDuckGo Instant Answer API
            url = f"https://api.duckduckgo.com/?q={query}&format=json&pretty=1"
            response = requests.get(url, timeout=3)
            data = response.json()
            
            suggestions = []
            if data.get('Abstract'):
                abstract = data['Abstract']
                if abstract and abstract != "":
                    suggestions.append(f"Info: {abstract[:150]}...")
            if data.get('RelatedTopics'):
                for topic in data['RelatedTopics'][:2]:
                    if isinstance(topic, dict) and 'Text' in topic:
                        suggestions.append(f"Related: {topic['Text'][:100]}")
                    elif isinstance(topic, str):
                        suggestions.append(f"Topic: {topic[:80]}")
            
            return suggestions if suggestions else ["No additional information found online."]
        except Exception as e:
            print(f"Online lookup error: {e}")
            return ["Unable to fetch online data at the moment."]
    
    def chat(self, message, context=None):
        """Enhanced chat with context awareness"""
        message_lower = message.lower()
        
        # Store conversation context
        self.conversation_context.append(f"User: {message}")
        
        # Check for specific patterns
        if any(greeting in message_lower for greeting in ["hello", "hi", "hey"]):
            response = "Hello! I'm JADE, your AI assistant. I'm here to analyze objects and assist you."
        
        elif "how are you" in message_lower:
            response = "I'm functioning optimally! Ready to analyze objects and help you with anything."
        
        elif "thank" in message_lower:
            response = "You're welcome! I'm here to help you analyze and understand your surroundings."
        
        elif "what can you do" in message_lower or "capabilities" in message_lower:
            response = "I can analyze objects through the camera, estimate their value, assess condition, provide maintenance tips, and access online information. I'm your personal object analysis assistant!"
        
        elif "analyze" in message_lower:
            response = "I'm ready to analyze objects. Point the camera at any object, and I'll assess its condition, materials, and estimated value."
        
        elif "mode" in message_lower:
            if "analysis" in message_lower:
                response = self.change_mode('analysis')
            elif "conversational" in message_lower:
                response = self.change_mode('conversational')
            elif "detection" in message_lower:
                response = self.change_mode('detection_only')
            else:
                response = "Available modes: analysis mode for detailed object assessment, conversational mode for chatting, and detection mode for object identification only."
        
        elif "help" in message_lower:
            response = "I can help you with: 1) Object analysis - point camera at objects 2) Value estimation 3) Condition assessment 4) Maintenance suggestions 5) Online research 6) General conversation"
        
        elif "time" in message_lower:
            current_time = datetime.now().strftime("%I:%M %p")
            response = f"The current time is {current_time}"
        
        elif "date" in message_lower:
            current_date = datetime.now().strftime("%B %d, %Y")
            response = f"Today is {current_date}"
        
        else:
            # For other queries, provide intelligent response
            response = self.generate_intelligent_response(message)
        
        # Store response in context
        self.conversation_context.append(f"JADE: {response}")
        
        # Keep context manageable
        if len(self.conversation_context) > 10:
            self.conversation_context = self.conversation_context[-10:]
        
        return response
    
    def generate_intelligent_response(self, message):
        """Generate intelligent responses for complex queries"""
        message_lower = message.lower()
        
        # Object-related queries
        if any(word in message_lower for word in ["value", "worth", "price"]):
            return "I can estimate object values. Please point the camera at an object for analysis."
        
        elif any(word in message_lower for word in ["condition", "status", "quality"]):
            return "I can assess object condition through visual analysis. Show me the object through the camera."
        
        elif any(word in message_lower for word in ["maintenance", "care", "repair"]):
            return "I can provide maintenance tips based on object type and condition. Let me analyze the object first."
        
        elif any(word in message_lower for word in ["material", "made of", "composition"]):
            return "I can identify materials through visual analysis. Show me the object for assessment."
        
        # General responses
        responses = [
            "I understand. Could you be more specific about what you'd like to analyze?",
            "That's interesting. Would you like me to analyze an object for you?",
            "I'm listening. You can ask me to analyze objects or ask general questions.",
            "I'm designed to help with object analysis. Point the camera at something to get started.",
            "I can assist with object assessment and general queries. What would you like to know?"
        ]
        
        import random
        return random.choice(responses)
    
    def process_object_assessment(self, assessment):
        """Process object assessment into conversational response"""
        if not assessment:
            return "I don't see any objects to analyze. Please point the camera at an object."
        
        obj = assessment[0] if isinstance(assessment, list) else assessment
        
        response = f"I see a {obj.get('object', 'object')}. "
        
        if 'condition' in obj and obj['condition'] != 'unknown':
            response += f"It appears to be in {obj['condition']} condition. "
        
        if 'estimated_value' in obj and obj['estimated_value'] != 'Unknown':
            response += f"Estimated value is approximately {obj['estimated_value']}. "
        
        if 'materials' in obj and obj['materials']:
            materials = obj['materials'][:3]
            if isinstance(materials, list) and len(materials) > 0:
                response += f"It seems to be made of {', '.join(materials)}. "
        
        if 'maintenance' in obj:
            response += f"Maintenance tip: {obj['maintenance'][:100]}... "
        
        if 'online_suggestions' in obj and obj['online_suggestions']:
            response += f"Online information suggests: {obj['online_suggestions'][0]}"
        
        return response

class JADEVoiceAssistant:
    def __init__(self, jade_assistant=None, wake_word="hey jade", voice_gender='female', speaking_rate=180):
        self.jade_assistant = jade_assistant if jade_assistant else JADEBaseAssistant()
        self.wake_word = wake_word.lower()
        
        # Wake word variations for better detection
        self.wake_word_variations = [
            "hey jade", "jade", "okay jade", "listen jade",
            "attention jade", "hello jade", "hi jade", "ok jade", "hey j"
        ]
        
        # Enhanced speech recognition
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 3000
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.recognizer.operation_timeout = 5
        
        # Enhanced TTS with female voice
        self.tts_engine = pyttsx3.init()
        self.setup_female_tts(speaking_rate)
        
        # Continuous listening state
        self.is_listening = False
        self.is_awake = True  # Always awake in conversational mode
        self.command_queue = queue.Queue(maxsize=100)
        self.is_processing = False
        
        # Threads for continuous operation
        self.listening_thread = None
        self.processing_thread = None
        self.audio_thread = None
        
        # Performance monitoring
        self.last_command_time = 0
        self.command_cooldown = 1.5
        self.silence_timeout = 10  # Timeout after silence
        
        print(f"🎤 JADE Voice Assistant Initialized - Always Active")
        print(f"👩 Voice: Female (Continuous Listening)")
        print(f"🎯 Wake words: {', '.join(self.wake_word_variations[:3])}...")
    
    def setup_female_tts(self, rate=180):
        """Setup female text-to-speech engine"""
        try:
            voices = self.tts_engine.getProperty('voices')
            
            # Find female voices
            female_voices = []
            for voice in voices:
                voice_name = voice.name.lower()
                if 'female' in voice_name or 'zira' in voice_name or 'eva' in voice_name:
                    female_voices.append(voice)
                elif 'microsoft zira' in voice_name:
                    female_voices.insert(0, voice)  # Prefer Zira on Windows
            
            if female_voices:
                selected_voice = female_voices[0]
                self.tts_engine.setProperty('voice', selected_voice.id)
                print(f"✅ Female voice selected: {selected_voice.name}")
            else:
                print("⚠️  No female voice found, using default")
            
            # Optimize voice settings
            self.tts_engine.setProperty('rate', rate)
            self.tts_engine.setProperty('volume', 0.9)
            
            # Test voice
            self.tts_engine.say(" ")
            self.tts_engine.runAndWait()
            
        except Exception as e:
            print(f"⚠️  TTS setup error: {e}")
    
    def start_continuous_listening(self):
        """Start continuous voice listening (like JARVIS)"""
        if self.is_listening:
            return
        
        self.is_listening = True
        self.is_awake = True
        
        print("🔊 Starting continuous listening...")
        self.speak("Voice assistant activated. I'm always listening.")
        
        # Start listening thread
        self.listening_thread = threading.Thread(
            target=self._continuous_listening_loop,
            daemon=True,
            name="JADE-Listener"
        )
        self.listening_thread.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True,
            name="JADE-Processor"
        )
        self.processing_thread.start()
        
        print("✅ Continuous listening started")
    
    def stop_listening(self):
        """Stop voice listening"""
        self.is_listening = False
        self.is_awake = False
        
        if self.listening_thread:
            self.listening_thread.join(timeout=2)
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        
        print("🛑 Voice assistant stopped")
    
    def _continuous_listening_loop(self):
        """Continuous listening loop like JARVIS"""
        print("🎧 Continuous listening loop started")
        
        with sr.Microphone() as source:
            # Adjust for ambient noise once
            print("   Adjusting for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            
            last_activity = time.time()
            silence_count = 0
            
            while self.is_listening:
                try:
                    # Check for silence timeout
                    current_time = time.time()
                    if current_time - last_activity > self.silence_timeout:
                        silence_count += 1
                        if silence_count % 5 == 0:  # Every 50 seconds of silence
                            self.speak("I'm still listening.")
                        last_activity = current_time
                    
                    print("   Listening... (Speak now)")
                    audio = self.recognizer.listen(
                        source,
                        timeout=1,
                        phrase_time_limit=5
                    )
                    
                    # Recognize speech
                    try:
                        text = self.recognizer.recognize_google(audio).lower()
                        print(f"🗣️  Heard: {text}")
                        last_activity = time.time()
                        silence_count = 0
                        
                        # Check for wake word
                        if any(wake in text for wake in self.wake_word_variations):
                            print("✅ Wake word detected!")
                            self.speak("Yes, I'm here?")
                            continue
                        
                        # Queue command for processing
                        command = VoiceCommand(
                            text=text,
                            confidence=1.0,
                            timestamp=time.time()
                        )
                        
                        try:
                            self.command_queue.put_nowait(command)
                        except queue.Full:
                            print("⚠️  Command queue full, dropping oldest")
                            try:
                                self.command_queue.get_nowait()
                                self.command_queue.put_nowait(command)
                            except:
                                pass
                        
                    except sr.UnknownValueError:
                        continue  # No speech detected
                    except sr.RequestError as e:
                        print(f"❌ API error: {e}")
                        time.sleep(1)
                    
                except sr.WaitTimeoutError:
                    continue  # Timeout, continue listening
                except Exception as e:
                    print(f"❌ Listening error: {e}")
                    time.sleep(0.5)
    
    def _processing_loop(self):
        """Process voice commands from queue"""
        print("⚙️  Command processing started")
        
        while self.is_listening:
            try:
                command = self.command_queue.get(timeout=0.5)
                if command:
                    self._handle_command_advanced(command)
                    self.command_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Processing error: {e}")
                time.sleep(0.1)
    
    def _handle_command_advanced(self, command):
        """Advanced command handling with context awareness"""
        text = command.text.lower()
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_command_time < self.command_cooldown:
            return "Please wait a moment..."
        
        self.last_command_time = current_time
        
        # Special commands
        if "go to sleep" in text or "stop listening" in text:
            self.speak("Going to sleep. Say 'hey jade' to wake me up.")
            self.is_awake = False
            return
        
        if "wake up" in text or any(wake in text for wake in self.wake_word_variations):
            self.speak("I'm awake and listening.")
            self.is_awake = True
            return
        
        # Check for analysis commands
        if any(cmd in text for cmd in ["what do you see", "analyze", "what's this", "identify", "scan"]):
            response = "I'll analyze the objects in view. Please ensure the camera is pointed correctly."
            self.speak(response)
            return response
        
        # Mode switching
        if "mode" in text:
            if "analysis" in text or "analyze" in text:
                response = self.jade_assistant.change_mode('analysis')
            elif "conversation" in text or "chat" in text:
                response = self.jade_assistant.change_mode('conversational')
            elif "detection" in text or "detect" in text:
                response = self.jade_assistant.change_mode('detection_only')
            else:
                response = "Available modes: analysis, conversational, and detection."
            self.speak(response)
            return response
        
        # Check for object-related queries
        if any(word in text for word in ["value", "worth", "price", "cost"]):
            response = "I can estimate object values. Point the camera at an object for analysis."
            self.speak(response)
            return response
        
        if any(word in text for word in ["condition", "quality", "status"]):
            response = "I can assess object condition. Show me the object through the camera."
            self.speak(response)
            return response
        
        # Process through assistant
        response = self.jade_assistant.chat(text)
        self.speak(response)
        return response
    
    def speak(self, text):
        """Speak text aloud with improved stability"""
        if not text or not self.is_listening:
            return
        
        # Limit text length for TTS
        tts_text = text[:200] if len(text) > 200 else text
        display_text = text[:150] + "..." if len(text) > 150 else text
        print(f"🗣️  Speaking: {display_text}")
        
        def _speak_thread():
            try:
                self.tts_engine.say(tts_text)
                self.tts_engine.runAndWait()
            except RuntimeError:
                # Handle engine busy
                time.sleep(0.5)
                try:
                    self.tts_engine.say(tts_text)
                    self.tts_engine.runAndWait()
                except:
                    pass
            except Exception as e:
                print(f"❌ TTS error: {e}")
        
        # Use threading for non-blocking speech
        threading.Thread(target=_speak_thread, daemon=True, name="JADE-TTS").start()
    
    def process_object_analysis(self, assessments):
        """Process object assessments and speak results"""
        if not assessments:
            self.speak("I don't see any objects to analyze. Please point the camera at an object.")
            return
        
        response = self.jade_assistant.process_object_assessment(assessments)
        self.speak(response)
    
    def test_voice(self):
        """Test voice functionality"""
        print("🔊 Testing voice system...")
        self.speak("Hello! I am JADE, your personal AI assistant. I'm here to help you analyze objects and answer your questions.")
    
    def listen_once(self):
        """Listen for a single command (for keyboard shortcut)"""
        try:
            print("🎤 Listening for single command...")
            
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                
                try:
                    text = self.recognizer.recognize_google(audio).lower()
                    print(f"🗣️  Heard: {text}")
                    return text
                except sr.UnknownValueError:
                    print("🤔 Could not understand audio")
                    return None
                    
        except Exception as e:
            print(f"❌ Listening error: {e}")
            return None
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_listening()
        print("✅ Audio resources cleaned up")