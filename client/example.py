from voiceBot import VoiceBot
import numpy as np
import asyncio

async def main():
    bot = VoiceBot("Dan")
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