from fastapi import FastAPI, File, UploadFile, HTTPException
import tempfile
import shutil
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="STT Service", version="1.0.0")

# Global model variable
model = None

def load_model():
    """Load the Whisper model"""
    global model
    if model is None:
        try:
            logger.info("Loading Whisper model (base for reliability)...")
            from faster_whisper import WhisperModel
            
            model = WhisperModel("base", compute_type="int8", device="cuda")
            logger.info("Whisper model loaded successfully")
            
        except ImportError as e:
            logger.error(f"faster_whisper not available: {e}")
            raise Exception("faster_whisper package not installed")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise Exception(f"Model loading failed: {e}")
    
    return model

@app.post("/stt/transcribe")
async def stt_transcribe(file: UploadFile = File(...)):
    """
    Transcribe uploaded audio file
    """
    logger.info(f"Received transcription request for file: {file.filename}")
    
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    tmp_path = None
    try:
        # Load model
        whisper_model = load_model()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            await file.seek(0)
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        logger.info(f"Audio saved to temporary file: {tmp_path}")
        
        # Check file size
        file_size = os.path.getsize(tmp_path)
        logger.info(f"Audio file size: {file_size} bytes")
        
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        # Transcribe audio
        logger.info("Starting transcription...")
        segments, info = whisper_model.transcribe(tmp_path)
        
        # Extract text from segments
        transcription_text = ""
        segment_count = 0
        
        for segment in segments:
            segment_count += 1
            transcription_text += segment.text.strip() + " "
            logger.info(f"Segment {segment_count}: [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        
        # Clean up final text
        transcription_text = transcription_text.strip()
        
        if not transcription_text:
            logger.warning("No speech detected in audio file")
            transcription_text = ""
        
        logger.info(f"Transcription completed: '{transcription_text}'")
        
        return {
            "text": transcription_text,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "segments": segment_count,
            "file_info": {
                "filename": file.filename,
                "size": file_size
            }
        }
    
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"STT processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"STT processing failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                logger.info("Temporary file cleaned up")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {e}")
                
@app.get("/")
async def root():
    return {"message": "STT Service is running", "model": "base"}

@app.get("/health")
async def health_check():
    try:
        # Try to load model to check if everything is working
        model_instance = load_model()
        model_loaded = model_instance is not None
        
        return {
            "status": "healthy",
            "service": "stt", 
            "model_loaded": model_loaded,
            "model_type": "base"
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")