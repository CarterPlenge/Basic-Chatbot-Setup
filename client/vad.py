import wave
import webrtcvad
import pyaudio
from typing import List
import io

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