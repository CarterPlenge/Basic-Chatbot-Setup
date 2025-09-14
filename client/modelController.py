import aiohttp
from typing import AsyncGenerator, List, Dict
import base64
import io
import json

class ModelController:
    """Communicates with models in Docker containers"""
    
    def __init__(self, 
                 ollama_url: str = "http://localhost:11434",
                 stt_url: str = "http://localhost:8002", 
                 tts_url: str = "http://localhost:8001"):
        self.ollama_url = ollama_url
        self.stt_url = stt_url
        self.tts_url = tts_url
    
    async def stt_stream_response(self, audio_data: bytes) -> str:
        """Send audio to STT service and get transcription"""
        try:
            async with aiohttp.ClientSession() as session:
                # Create form data with audio file
                data = aiohttp.FormData()
                data.add_field('file', 
                              io.BytesIO(audio_data), 
                              filename='audio.wav',
                              content_type='audio/wav')
                
                async with session.post(f"{self.stt_url}/stt/transcribe", 
                                      data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('text', '').strip()
                    else:
                        error_text = await response.text()
                        print(f"STT Error {response.status}: {error_text}")
                        return ""
        except Exception as e:
            print(f"STT request failed: {e}")
            return ""
    
    async def gen_ai_stream_response(self, messages: List[Dict[str, str]], model: str = "Dan") -> AsyncGenerator[str, None]:
        """Stream response from Ollama"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": model,
                    "messages": messages,
                    "stream": True
                }
                
                async with session.post(f"{self.ollama_url}/api/chat",
                                      json=payload) as response:
                    if response.status == 200:
                        async for line in response.content:
                            if line.strip():
                                try:
                                    data = json.loads(line.decode('utf-8'))
                                    if 'message' in data and 'content' in data['message']:
                                        yield data['message']['content']
                                except json.JSONDecodeError:
                                    continue
                    else:
                        error_text = await response.text()
                        print(f"Ollama Error {response.status}: {error_text}")
        except Exception as e:
            print(f"Ollama request failed: {e}")
    
    async def tts_stream_response(self, text: str) -> AsyncGenerator[bytes, None]:
        """Stream TTS audio from text"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {"text": text}
                
                async with session.post(f"{self.tts_url}/tts/stream",
                                      json=payload) as response:
                    if response.status == 200:
                        async for line in response.content:
                            if line.strip():
                                try:
                                    # Decode base64 audio chunk
                                    audio_b64 = line.decode('utf-8').strip()
                                    audio_bytes = base64.b64decode(audio_b64)
                                    yield audio_bytes
                                except Exception:
                                    continue
                    else:
                        error_text = await response.text()
                        print(f"TTS Error {response.status}: {error_text}")
        except Exception as e:
            print(f"TTS request failed: {e}")