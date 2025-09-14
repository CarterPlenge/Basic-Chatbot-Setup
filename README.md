# Basic Chatbot
The Goal of this project is to create a simple bot class that can do the following while keeping its implementation simple. 
 - remember prior conversations
 - respond via text or audio
 - get input via text or audio

## Setup
Software needed
 - [Docker](https://www.docker.com/)
 - [Ollama](https://ollama.com/)
 - [Python](https://www.python.org/)

Termianl Commands
```
# 1: make sure your in your venv

# 2: ollama model setup
docker exec -it ollama ollama pull mistral:7b-instruct 
# you can use other models as well, but you might have to adjust it
# to handle model specific parameters

# 3: create/update custom model
docker exec -it basic-chatbot-setup-ollama ollama create Dan -f /models/Modelfile
# Dan is a name. name it whatever you want. 
# re-run this command to update the model after changing the Modelfile

# 3: install dependences
pip install -r .\client\requirements.txt

```

## How to use
### example.py
```
from voiceBot import VoiceBot
import numpy as np
import asyncio

async def main():
    bot = VoiceBot("Dan") # whatever you named it in ollama setup
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
```
### list of useful functions
```
bot.ask(prompt, with_tts, context_id) 
bot.vad.run_vad() 
bot.memory.clear_memory(context_id)
```
Thats all. simple as I could make it without taking too much flexablity away.


## Other Notes

 - Context_id will make it only pull on memories with the same context_id. If you dont want that just dont use it. Everything will default to "general" and will have the same context_id making it do nothing.
 - All of the models are ran in docker containers. If you want to use a differnt model adjust the corosponding container and everything should still work fine. probably

