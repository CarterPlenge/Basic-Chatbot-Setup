import asyncio
import numpy as np
import pyaudio
from typing import AsyncGenerator, Tuple, Optional
from memory import Memory
from modelController import ModelController
from vad import VAD


class VoiceBot:
    """Main VoiceBot class orchestrating the conversation flow"""
    
    def __init__(self, model_name: str = "mistral:7b-instruct"):
        self.vad = VAD()
        self.memory = Memory()
        self.model_controller = ModelController()
        self.model_name = model_name
        self.audio_player = None
        self._init_audio_player()
    
    def _init_audio_player(self):
        """Initialize PyAudio for audio playback"""
        self.audio = pyaudio.PyAudio()
        self.audio_player = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=24000,
            output=True
        )
    
    def __del__(self):
        """Cleanup audio resources"""
        if self.audio_player:
            self.audio_player.close()
        if hasattr(self, 'audio'):
            self.audio.terminate()
    
    async def ask(self, prompt: str, with_tts: bool = True, context_id: str = "general") -> AsyncGenerator[Tuple[str, Optional[bytes]], None]:
        """
        Main conversation method
        
        Args:
            prompt: Text prompt
            with_tts: Whether to synthesize speech
            context_id: Use to create memory scopes
            
        Yields:
            Tuple of (text_chunk, audio_chunk) - audio_chunk is None if with_tts=False
        """
        
        print(f"User: {prompt}")
        
        # Add user message to memory
        self.memory.add_message(context_id, "user", prompt)
        
        # Prepare conversation context
        messages = self.memory.get_conversation_context(context_id)
        
        # Generate AI response and TTS simultaneously
        full_response = ""
        
        if with_tts:
            # Create queues for coordination
            ai_queue = asyncio.Queue()
            
            async def ai_generator():
                """Generate AI response and feed to coordination queue"""
                async for chunk in self.model_controller.gen_ai_stream_response(messages, self.model_name):
                    if chunk.strip():
                        await ai_queue.put(chunk)
                await ai_queue.put(None)  # Signal end
            
            # Start AI generation task
            ai_task = asyncio.create_task(ai_generator())
            
            # Process chunks for both text output and TTS
            sentence_buffer = ""
            
            while True:
                chunk = await ai_queue.get()
                if chunk is None:
                    # Process remaining buffer for TTS
                    if sentence_buffer.strip():
                        async for audio_chunk in self.model_controller.tts_stream_response(sentence_buffer):
                            yield "", audio_chunk
                    break
                
                # Yield text chunk immediately
                full_response += chunk
                yield chunk, None
                
                # Add to sentence buffer for TTS
                sentence_buffer += chunk
                
                # Check if we have a complete sentence for TTS
                if any(punct in sentence_buffer for punct in '.!?'):
                    # Find the last sentence boundary
                    for punct in '.!?':
                        if punct in sentence_buffer:
                            idx = sentence_buffer.rfind(punct) + 1
                            sentence = sentence_buffer[:idx].strip()
                            sentence_buffer = sentence_buffer[idx:].strip()
                            
                            if sentence:
                                # Stream TTS for this sentence
                                async for audio_chunk in self.model_controller.tts_stream_response(sentence):
                                    yield "", audio_chunk
                            break
            
            await ai_task
            
        else:
            # Text-only mode
            async for chunk in self.model_controller.gen_ai_stream_response(messages, self.model_name):
                if chunk.strip():
                    full_response += chunk
                    yield chunk, None
        
        # Add assistant response to memory
        if full_response:
            self.memory.add_message(context_id, "assistant", full_response)
    
async def main():
    """Main function for testing the VoiceBot"""
    bot = VoiceBot()
    print("Bot has started!")
    while True:
        user_input = input("\n> ").strip()
        
        # exit condition
        if user_input.lower() in ['quit', 'exit', 'stop']:
            break
        
        # Vad if no message was entered
        if not(user_input):
            print("Starting voice capture...")
            audio_data = bot.vad.run_vad()
                
            if not audio_data:
                print("No speech detected")
                continue
        
            print("Transcribing speech...")
            user_input = await bot.model_controller.stt_stream_response(audio_data)
            
            if not user_input:
                print("Could not transcribe speech")
                continue
        
        # ask the bot
        async for text_chunk, audio_chunk in bot.ask(user_input):
            if text_chunk:
                print(text_chunk, end="",flush=True)
            
            if audio_chunk and bot.audio_player:
                audio_array = np.frombuffer(audio_chunk, dtype=np.float32)
                bot.audio_player.write(audio_array.tobytes())

        print()


if __name__ == "__main__":
    asyncio.run(main())