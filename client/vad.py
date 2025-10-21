import wave
import webrtcvad
import pyaudio
from typing import List, Optional
import io
import threading
import queue
import time

class VAD:
    """Voice Activity Detection using WebRTC VAD"""
    
    def __init__(self, aggressiveness: int = 3, sample_rate: int = 16000):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_duration = 30  # ms
        self.frame_size = int(sample_rate * self.frame_duration / 1000)
        self.is_recording = False
        self.audio_buffer = []
        self.silence_threshold = 20  # frames of silence before stopping
        
    def _frames_to_wav_bytes(self, frames: List[bytes]) -> bytes:
        """Convert audio frames to WAV format bytes"""
        buffer = io.BytesIO()
        
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(b''.join(frames))
        
        buffer.seek(0)
        return buffer.read()        
    
    def run_vad(self) -> bytes:
        """
        Capture audio with VAD and return audio data when speech ends
        Returns raw audio bytes ready for STT
        """
        audio = pyaudio.PyAudio()
        
        try:
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.frame_size
            )
            
            print("Listening for speech...")
            speech_frames = []
            silence_count = 0
            speech_detected = False
            
            while True:
                frame = stream.read(self.frame_size)
                is_speech = self.vad.is_speech(frame, self.sample_rate)
                
                if is_speech:
                    if not speech_detected:
                        print("Speech detected, recording...")
                        speech_detected = True
                    speech_frames.append(frame)
                    silence_count = 0
                elif speech_detected:
                    speech_frames.append(frame)  # Include some silence
                    silence_count += 1
                    
                    if silence_count > self.silence_threshold:
                        print("Speech ended, processing...")
                        break
                        
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
        
        # Convert frames to wav bytes
        if speech_frames:
            return self._frames_to_wav_bytes(speech_frames)
        return b""
    
class ContinuousVAD:
    """Continuous Voice Activity Detection using WebRTC VAD"""
    
    def __init__(self, aggressiveness: int = 3, sample_rate: int = 16000):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_duration = 30  # ms
        self.frame_size = int(sample_rate * self.frame_duration / 1000)
        
        # Threading and state management
        self.is_running = False
        self.listen_thread = None
        self.audio_stream = None
        self.audio_context = None
        
        # Audio processing
        self.speech_queue = queue.Queue()
        self.current_speech_frames = []
        self.is_speech_active = False
        self.silence_count = 0
        self.silence_threshold = 20  # frames of silence before ending speech
        self.min_speech_frames = 10  # minimum frames for valid speech
        
        # Buffer for pre-speech audio (to catch the beginning of words)
        self.pre_speech_buffer = []
        self.pre_speech_buffer_size = 10
        
    def _frames_to_wav_bytes(self, frames: List[bytes]) -> bytes:
        """Convert audio frames to WAV format bytes"""
        if not frames:
            return b""
            
        buffer = io.BytesIO()
        
        try:
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(b''.join(frames))
            
            buffer.seek(0)
            return buffer.read()
        except Exception as e:
            print(f"Error converting frames to WAV: {e}")
            return b""
    
    def _audio_processing_loop(self):
        """Main audio processing loop running in background thread"""
        try:
            while self.is_running:
                try:
                    # Read audio frame
                    frame = self.audio_stream.read(self.frame_size, exception_on_overflow=False)
                    is_speech = self.vad.is_speech(frame, self.sample_rate)
                    
                    if is_speech:
                        if not self.is_speech_active:
                            # Speech just started - include pre-speech buffer
                            print("Speech detected, starting recording...")
                            self.is_speech_active = True
                            self.current_speech_frames = list(self.pre_speech_buffer)  # Include lead-in
                            self.silence_count = 0
                        
                        self.current_speech_frames.append(frame)
                        self.silence_count = 0
                        
                    else:
                        # No speech detected
                        if self.is_speech_active:
                            # We're in the middle of recording speech
                            self.current_speech_frames.append(frame)  # Include some trailing silence
                            self.silence_count += 1
                            
                            if self.silence_count >= self.silence_threshold:
                                # Speech has ended
                                print("Speech ended, queuing for processing...")
                                
                                # Only queue if we have enough speech frames
                                if len(self.current_speech_frames) >= self.min_speech_frames:
                                    wav_data = self._frames_to_wav_bytes(self.current_speech_frames)
                                    if wav_data:
                                        try:
                                            self.speech_queue.put_nowait(wav_data)
                                        except queue.Full:
                                            print("Speech queue full, dropping audio")
                                
                                # Reset for next speech detection
                                self._reset_speech_detection()
                        else:
                            # Not recording, maintain pre-speech buffer
                            self.pre_speech_buffer.append(frame)
                            if len(self.pre_speech_buffer) > self.pre_speech_buffer_size:
                                self.pre_speech_buffer.pop(0)
                    
                except Exception as e:
                    if self.is_running:  # Only log if we're still supposed to be running
                        print(f"Error in audio processing loop: {e}")
                        time.sleep(0.1)  # Brief pause on error
                        
        except Exception as e:
            print(f"Fatal error in audio processing loop: {e}")
        finally:
            print("Audio processing loop ended")
    
    def _reset_speech_detection(self):
        """Reset speech detection state"""
        self.is_speech_active = False
        self.current_speech_frames = []
        self.silence_count = 0
    
    def start_listening(self):
        """Start continuous listening in background thread"""
        if self.is_running:
            print("Already listening")
            return
        
        try:
            # Initialize PyAudio
            self.audio_context = pyaudio.PyAudio()
            
            # Open audio stream
            self.audio_stream = self.audio_context.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.frame_size,
                stream_callback=None  # We'll use blocking read
            )
            
            # Clear any existing audio in queue
            while not self.speech_queue.empty():
                try:
                    self.speech_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Start processing
            self.is_running = True
            self.listen_thread = threading.Thread(target=self._audio_processing_loop, daemon=True)
            self.listen_thread.start()
            
            print("Continuous VAD started")
            
        except Exception as e:
            print(f"Failed to start VAD: {e}")
            self.stop_listening()
    
    def stop_listening(self):
        """Stop continuous listening"""
        print("Stopping continuous VAD...")
        self.is_running = False
        
        # Wait for thread to finish
        if self.listen_thread and self.listen_thread.is_alive():
            self.listen_thread.join(timeout=2.0)
        
        # Close audio resources
        if self.audio_stream:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            except Exception as e:
                print(f"Error closing audio stream: {e}")
            self.audio_stream = None
        
        if self.audio_context:
            try:
                self.audio_context.terminate()
            except Exception as e:
                print(f"Error terminating audio context: {e}")
            self.audio_context = None
        
        # Reset state
        self._reset_speech_detection()
        print("Continuous VAD stopped")
    
    def get_speech_audio(self, timeout: float = 0.1) -> Optional[bytes]:
        """
        Get detected speech audio if available
        Returns None if no speech is available within timeout
        """
        try:
            return self.speech_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def has_speech_available(self) -> bool:
        """Check if speech audio is available without blocking"""
        return not self.speech_queue.empty()
    
    def clear_speech_queue(self):
        """Clear any pending speech in the queue"""
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except queue.Empty:
                break