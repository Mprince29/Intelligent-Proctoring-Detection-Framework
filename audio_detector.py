import speech_recognition as sr
import pyaudio
import threading
import time
import logging
import json
from collections import Counter
import numpy as np

class AudioDetector:
    def __init__(self):
        self.logger = logging.getLogger("audio_detector")
        
        # Audio recording parameters
        self.rate = 16000  # Standard rate for speech recognition
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        
        # Speech recognition - optimized settings
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 200  # Lowered from 300 for better sensitivity
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.dynamic_energy_adjustment_ratio = 1.5  # Added for better adaptation
        self.recognizer.pause_threshold = 0.5  # Lowered from 0.6 for faster detection
        self.recognizer.phrase_threshold = 0.3  # Added to improve phrase detection
        self.recognizer.non_speaking_duration = 0.3  # Added to detect speech end faster
        
        # State variables
        self.is_recording = False
        self.audio_thread = None
        self.pyaudio_instance = None
        self.stream = None
        self.device_index = None
        
        # Tracking detected speech
        self.spoken_words = []
        self.word_counter = Counter()
        self.spoken_sentences = []
        self.recording_start_time = None
        self.last_detection_time = None
        self.detection_timestamps = []
        
        # Metrics
        self.metrics = {
            "total_words_detected": 0,
            "total_sentences": 0,
            "last_spoken_text": "",
            "audio_monitoring_active": False,
            "headphones_detected": False,
            "headphones_with_mic": False,
            "background_noise_level": 0,
            "last_detection_time": None
        }
        
        # Ambient noise samples for continuous adaptation
        self.ambient_samples = []
        self.last_ambient_adjustment = 0
    
    def list_available_devices(self):
        """List all available audio input and output devices"""
        try:
            p = pyaudio.PyAudio()
            devices = []
            
            for i in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(i)
                device_type = []
                if device_info['maxInputChannels'] > 0:
                    device_type.append("Input")
                if device_info['maxOutputChannels'] > 0:
                    device_type.append("Output")
                    
                devices.append({
                    "index": i,
                    "name": device_info['name'],
                    "type": " & ".join(device_type),
                    "sample_rate": int(device_info['defaultSampleRate']),
                    "input_channels": device_info['maxInputChannels'],
                    "output_channels": device_info['maxOutputChannels']
                })
                
            p.terminate()
            self.logger.info(f"Available audio devices: {devices}")
            return devices
        except Exception as e:
            self.logger.error(f"Error listing audio devices: {e}")
            return []
    
    def get_default_input_device(self):
        """Get the default input device index with improved selection logic"""
        try:
            p = pyaudio.PyAudio()
            
            # Log all available devices for debugging
            all_devices = []
            for i in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(i)
                all_devices.append(device_info)
                self.logger.info(f"Device {i}: {device_info['name']}, Input: {device_info['maxInputChannels']}, Output: {device_info['maxOutputChannels']}")
            
            # Try to get the default input device
            try:
                default_index = p.get_default_input_device_info().get('index')
                device_info = p.get_device_info_by_index(default_index)
                self.logger.info(f"Default input device: {device_info['name']}")
                p.terminate()
                return default_index
            except Exception as e:
                self.logger.error(f"Error getting default input device: {e}")
            
            # If failed, try to find the best input device
            # Look for headset/mic devices first
            input_devices = []
            mic_keywords = ['mic', 'microphone', 'headset', 'input']
            
            for i in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    # Score the device based on name relevance and channel count
                    score = device_info['maxInputChannels']
                    name = device_info['name'].lower()
                    
                    # Increase score for devices that seem to be dedicated microphones
                    if any(keyword in name for keyword in mic_keywords):
                        score += 5
                    
                    input_devices.append((i, device_info['name'], score))
            
            p.terminate()
            
            if input_devices:
                # Sort by score (higher is better)
                input_devices.sort(key=lambda x: x[2], reverse=True)
                self.logger.info(f"Selected best available input device: {input_devices[0]}")
                return input_devices[0][0]  # Return the index of the best input device
            
            # No input devices found
            self.logger.error("No input devices found")
            return None
        except Exception as e:
            self.logger.error(f"Failed to find any input device: {e}")
            return None
    
    def start_audio_monitoring(self):
        """Start the audio monitoring thread with improved initialization"""
        if self.is_recording:
            self.logger.info("Audio monitoring already active")
            return True
        
        try:
            # List available devices first to help diagnose issues
            available_devices = self.list_available_devices()
            self.logger.info(f"Available devices before starting: {available_devices}")
            
            # Get the default input device
            self.device_index = self.get_default_input_device()
            
            if self.device_index is None:
                self.logger.error("No input devices available")
                return False
                
            self.logger.info(f"Selected device index: {self.device_index}")
            
            # Test if we can open a stream with this device
            try:
                test_pyaudio = pyaudio.PyAudio()
                test_stream = test_pyaudio.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.rate,
                    input=True,
                    input_device_index=self.device_index,
                    frames_per_buffer=self.chunk
                )
                
                # Read a small chunk to test and check audio levels
                data = test_stream.read(self.chunk)
                audio_array = np.frombuffer(data, dtype=np.int16)
                audio_level = np.abs(audio_array).mean()
                
                self.logger.info(f"Initial audio level: {audio_level}")
                
                # Auto-adjust initial energy threshold based on test reading
                if audio_level > 0:
                    # Set threshold to be sensitive but avoid false triggers
                    self.recognizer.energy_threshold = min(max(audio_level * 1.2, 50), 300)
                    self.logger.info(f"Auto-adjusted energy threshold to: {self.recognizer.energy_threshold}")
                
                test_stream.close()
                test_pyaudio.terminate()
                self.logger.info("Successfully tested audio input stream")
            except Exception as e:
                self.logger.error(f"Failed to test audio input stream: {e}")
                return False
            
            self.pyaudio_instance = pyaudio.PyAudio()
            self.is_recording = True
            self.recording_start_time = time.time()
            self.last_ambient_adjustment = time.time()
            self.audio_thread = threading.Thread(target=self._audio_monitoring_thread)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            self.metrics["audio_monitoring_active"] = True
            self.logger.info(f"Audio monitoring started with device_index: {self.device_index}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start audio monitoring: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def stop_audio_monitoring(self):
        """Stop the audio monitoring thread"""
        self.is_recording = False
        if self.stream and hasattr(self.stream, 'is_active') and self.stream.is_active():
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                self.logger.error(f"Error stopping stream: {e}")
        
        if self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
            except Exception as e:
                self.logger.error(f"Error terminating PyAudio: {e}")
            
        self.metrics["audio_monitoring_active"] = False
        self.logger.info("Audio monitoring stopped")
        return True
    
    def _adjust_ambient_noise_if_needed(self, source):
        """Periodically adjust for ambient noise to adapt to changing environments"""
        current_time = time.time()
        # Adjust every 60 seconds or on first run
        if current_time - self.last_ambient_adjustment > 60:
            try:
                self.logger.info("Periodically adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.3)  # Shorter duration for faster response
                current_threshold = self.recognizer.energy_threshold
                self.metrics["background_noise_level"] = current_threshold
                self.logger.info(f"Updated energy threshold: {current_threshold}")
                
                # Keep threshold within reasonable bounds
                if current_threshold < 50:
                    self.recognizer.energy_threshold = 50
                elif current_threshold > 2000:  # Lowered upper bound for better sensitivity
                    self.recognizer.energy_threshold = 2000
                
                self.last_ambient_adjustment = current_time
                return True
            except Exception as e:
                self.logger.error(f"Error adjusting for ambient noise: {e}")
                return False
        return True
    
    def _audio_monitoring_thread(self):
        """Optimized background thread to continuously monitor audio"""
        retry_delay = 1  # Initial retry delay
        max_retry_delay = 10  # Maximum retry delay
        consecutive_errors = 0
        
        while self.is_recording:
            try:
                # Use context manager for microphone to ensure proper cleanup
                with sr.Microphone(device_index=self.device_index, sample_rate=self.rate) as source:
                    self.logger.info("Microphone source created successfully")
                    
                    # Initial adjustment for ambient noise
                    if not self._adjust_ambient_noise_if_needed(source):
                        time.sleep(retry_delay)
                        continue
                    
                    while self.is_recording:
                        try:
                            # Periodically adjust for ambient noise
                            self._adjust_ambient_noise_if_needed(source)
                            
                            # Listen for audio with shorter timeout for more responsiveness
                            self.logger.info("Waiting for speech...")
                            audio = self.recognizer.listen(
                                source, 
                                timeout=3,  # Shorter timeout for faster retry
                                phrase_time_limit=8  # Longer phrase limit for complete sentences
                            )
                            
                            # Reset error counter on successful listen
                            consecutive_errors = 0
                            retry_delay = 1
                            
                            self.logger.info("Audio captured, trying to recognize...")
                            
                            # Try multiple recognition engines for better accuracy
                            text = None
                            try:
                                # Try Google first (most accurate but requires internet)
                                text = self.recognizer.recognize_google(audio)
                            except sr.RequestError:
                                # Fallback to Sphinx (offline, less accurate)
                                try:
                                    self.logger.info("Google API failed, trying Sphinx...")
                                    text = self.recognizer.recognize_sphinx(audio)
                                except:
                                    # If both fail, raise the original error
                                    raise
                            
                            current_time = time.time()
                            
                            if text:
                                self.logger.warning(f"DETECTED SPEECH: {text}")
                                
                                # Process the detected speech
                                self._process_detected_speech(text, current_time)
                                
                        except sr.WaitTimeoutError:
                            # No speech detected, just continue listening
                            self.logger.info("No speech detected within timeout period")
                            continue
                        except sr.UnknownValueError:
                            # Speech was unintelligible
                            self.logger.info("Speech not recognized - unclear audio")
                            continue
                        except Exception as inner_e:
                            self.logger.error(f"Error during speech recognition: {inner_e}")
                            # Don't sleep here, just continue the inner loop
                            continue
            
            except Exception as e:
                self.logger.error(f"Error in audio monitoring thread outer loop: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                
                # Exponential backoff for retries
                consecutive_errors += 1
                retry_delay = min(retry_delay * 1.5, max_retry_delay)
                
                # If too many consecutive errors, try to reinitialize the microphone
                if consecutive_errors >= 5:
                    self.logger.warning("Multiple consecutive errors, attempting to reinitialize microphone...")
                    try:
                        # Close and recreate PyAudio instance
                        if self.pyaudio_instance:
                            self.pyaudio_instance.terminate()
                        self.pyaudio_instance = pyaudio.PyAudio()
                        consecutive_errors = 0
                    except Exception as reinit_e:
                        self.logger.error(f"Failed to reinitialize PyAudio: {reinit_e}")
                
                time.sleep(retry_delay)
                
    def _process_detected_speech(self, text, timestamp):
        """Process and store detected speech"""
        # Store the full sentence
        self.spoken_sentences.append({
            "text": text,
            "timestamp": timestamp,
            "elapsed_time": timestamp - self.recording_start_time
        })
        
        # Process and count individual words
        words = text.lower().split()
        self.spoken_words.extend(words)
        
        # Update word counter
        for word in words:
            self.word_counter[word] += 1
        
        # Update metrics
        self.metrics["total_words_detected"] = len(self.spoken_words)
        self.metrics["total_sentences"] = len(self.spoken_sentences)
        self.metrics["last_spoken_text"] = text
        self.metrics["last_detection_time"] = timestamp
        
        # Store detection timestamp
        self.detection_timestamps.append(timestamp)
        self.last_detection_time = timestamp
    
    def get_audio_metrics(self):
        """Get current audio monitoring metrics"""
        # Calculate some additional metrics
        time_since_last = None
        if self.metrics["last_detection_time"]:
            time_since_last = time.time() - self.metrics["last_detection_time"]
        
        return {
            **self.metrics,
            "most_common_words": self.word_counter.most_common(10),
            "total_unique_words": len(self.word_counter),
            "time_since_last_detection": time_since_last,
            "speech_frequency": len(self.detection_timestamps) / (time.time() - self.recording_start_time) if self.recording_start_time else 0,
            "energy_threshold": self.recognizer.energy_threshold if hasattr(self.recognizer, 'energy_threshold') else None
        }
    
    def get_all_spoken_content(self):
        """Get all spoken content recorded so far"""
        return {
            "sentences": self.spoken_sentences,
            "words": self.spoken_words,
            "word_frequencies": dict(self.word_counter)
        }
    
    def detect_headphones(self):
        """Detect if headphones with microphone are connected"""
        try:
            p = pyaudio.PyAudio()
            
            # Check for input devices (microphones)
            input_devices = []
            output_devices = []
            
            for i in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    input_devices.append(device_info)
                if device_info['maxOutputChannels'] > 0:
                    output_devices.append(device_info)
            
            p.terminate()
            
            # Initialize variables
            found_headphones = False
            found_headphones_with_mic = False
            
            # Check for headphones with mic
            headphone_keywords = ['headphone', 'headset', 'earphone', 'headphones']
            mic_keywords = ['mic', 'microphone', 'input']
            
            # Look for headphone output devices
            for device in output_devices:
                name = device['name'].lower()
                if any(keyword in name for keyword in headphone_keywords):
                    found_headphones = True
                    break
            
            # Look for headphone microphones
            for device in input_devices:
                name = device['name'].lower()
                if any(keyword in name for keyword in headphone_keywords):
                    found_headphones_with_mic = True
                    break
            
            # If no specific headphone devices found but we have input devices, 
            # consider system microphone as available
            has_system_mic = len(input_devices) > 0
            
            self.metrics["headphones_detected"] = found_headphones
            self.metrics["headphones_with_mic"] = found_headphones_with_mic
            
            self.logger.info(f"Headphones detected: {found_headphones}, with mic: {found_headphones_with_mic}")
            self.logger.info(f"System has microphone: {has_system_mic}")
            
            if not found_headphones_with_mic and has_system_mic:
                self.logger.info("Using system microphone instead of headphone mic")
            
            return {
                "headphones_detected": found_headphones,
                "headphones_with_mic": found_headphones_with_mic,
                "system_microphone_available": has_system_mic,
                "input_devices": len(input_devices),
                "output_devices": len(output_devices)
            }
        except Exception as e:
            self.logger.error(f"Error detecting headphones: {e}")
            return {
                "headphones_detected": False,
                "headphones_with_mic": False,
                "system_microphone_available": False,
                "error": str(e)
            }
    
    def test_microphone(self, duration=3):
        """Test the microphone by recording briefly and checking audio levels"""
        try:
            # Get default input device
            device_index = self.get_default_input_device()
            if device_index is None:
                return {
                    "microphone_working": False,
                    "error": "No input device found"
                }
                
            self.logger.info(f"Testing microphone with device index: {device_index}")
            
            p = pyaudio.PyAudio()
            stream = p.open(format=self.format,
                          channels=self.channels,
                          rate=self.rate,
                          input=True,
                          input_device_index=device_index,
                          frames_per_buffer=self.chunk)
            
            frames = []
            audio_levels = []
            
            # Collect audio data and calculate levels in real-time
            for i in range(0, int(self.rate / self.chunk * duration)):
                data = stream.read(self.chunk, exception_on_overflow=False)
                frames.append(data)
                
                # Calculate audio level for this chunk
                audio_array = np.frombuffer(data, dtype=np.int16)
                audio_level = np.abs(audio_array).mean()
                audio_levels.append(audio_level)
                
                # Log every second
                if i % int(self.rate / self.chunk) == 0:
                    self.logger.info(f"Current audio level: {audio_level}")
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Calculate average and peak audio levels
            avg_audio_level = np.mean(audio_levels) if audio_levels else 0
            peak_audio_level = np.max(audio_levels) if audio_levels else 0
            
            self.logger.info(f"Microphone test completed. Avg level: {avg_audio_level}, Peak: {peak_audio_level}")
            
            # Auto-adjust energy threshold based on test results
            if avg_audio_level > 10:
                # Set threshold relative to ambient noise but ensuring it's responsive
                suggested_threshold = min(max(avg_audio_level * 1.2, 50), 300)
                self.logger.info(f"Suggested energy threshold based on test: {suggested_threshold}")
                
                if hasattr(self, 'recognizer'):
                    self.recognizer.energy_threshold = suggested_threshold
            
            return {
                "microphone_working": avg_audio_level > 10,  # Threshold to determine if picking up sound
                "avg_audio_level": float(avg_audio_level),
                "peak_audio_level": float(peak_audio_level),
                "duration": duration,
                "suggested_threshold": float(avg_audio_level * 1.2) if avg_audio_level > 10 else 200.0
            }
        except Exception as e:
            self.logger.error(f"Error testing microphone: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "microphone_working": False,
                "error": str(e)
            }
    
    def save_spoken_content_to_file(self, filename="spoken_content.json"):
        """Save all recorded spoken content to a JSON file"""
        try:
            content = self.get_all_spoken_content()
            with open(filename, 'w') as f:
                json.dump(content, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Error saving spoken content: {e}")
            return False
            
    def debug_audio_levels(self, duration=10):
        """Debug function to record audio and print levels"""
        try:
            device_index = self.get_default_input_device()
            if device_index is None:
                self.logger.error("No input device found for debugging")
                return False
                
            self.logger.info(f"Testing audio levels with device index: {device_index}")
            
            p = pyaudio.PyAudio()
            stream = p.open(format=self.format,
                          channels=self.channels,
                          rate=self.rate,
                          input=True,
                          input_device_index=device_index,
                          frames_per_buffer=self.chunk)
            
            self.logger.info(f"Recording for {duration} seconds to check audio levels...")
            
            # Record for the specified duration while printing audio levels
            start_time = time.time()
            audio_levels = []
            
            while time.time() - start_time < duration:
                data = stream.read(self.chunk, exception_on_overflow=False)
                audio_array = np.frombuffer(data, dtype=np.int16)
                audio_level = np.abs(audio_array).mean()
                audio_levels.append(audio_level)
                self.logger.info(f"Current audio level: {audio_level}")
                time.sleep(0.2)  # More frequent readings
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Calculate statistics
            if audio_levels:
                avg_level = np.mean(audio_levels)
                max_level = np.max(audio_levels)
                min_level = np.min(audio_levels)
                self.logger.info(f"Audio level stats - Avg: {avg_level}, Max: {max_level}, Min: {min_level}")
                
                # Suggest a good threshold
                suggested_threshold = min(max(avg_level * 1.2, 50), 300)
                self.logger.info(f"Suggested energy threshold: {suggested_threshold}")
            
            self.logger.info("Audio level test completed")
            return True
        except Exception as e:
            self.logger.error(f"Error in audio level debugging: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False