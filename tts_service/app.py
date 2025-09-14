from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from kokoro import KPipeline
import numpy as np
import base64

app = FastAPI()
pipeline = KPipeline(lang_code="a", device="cuda") # yeilds tuple(gs, ps, audio)

def synthesize_sentence(sentence: str, voice="af_heart"):
    generator = pipeline(
        sentence,
        voice=voice,
        speed=1,
        split_pattern="\n+"
    )
    audio_segments = [audio for _, _, audio in generator]
    if not audio_segments:
        return np.zeros(0, dtype="float32")

    audio = np.concatenate(audio_segments).astype("float32")

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    return audio


@app.post("/tts/stream")
async def tts_stream(request: Request):
    """
    Input: {"text":"..."}
    Output: newline-delimited base64 PCM float32 chunks
    """
    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        return {"error": "No text provided"}

    audio = synthesize_sentence(text)

    async def pcm_gen():
        CHUNK = 4096
        for i in range(0, len(audio), CHUNK):
            chunk = audio[i:i+CHUNK]
            yield (
                base64.b64encode(chunk.tobytes()).decode("utf-8") + "\n"
            )

    return StreamingResponse(pcm_gen(), media_type="application/octet-stream")