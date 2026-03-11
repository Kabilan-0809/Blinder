import whisper
import base64
import tempfile
import os
import logging

logger = logging.getLogger(__name__)

# Load once at startup — 'base' is the best speed/accuracy trade-off for on-device use
_model = None

def get_model():
    global _model
    if _model is None:
        logger.info("[STT] Loading Whisper base model...")
        _model = whisper.load_model("base")
        logger.info("[STT] Whisper model ready.")
    return _model

def transcribe_audio(audio_b64: str) -> str:
    """
    Accepts a base64-encoded audio blob (webm/mp3/wav/any ffmpeg-compatible format).
    Returns the transcript string.
    """
    try:
        audio_bytes = base64.b64decode(audio_b64)
        
        # Write to a temp file because Whisper expects a file path
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        
        model = get_model()
        result = model.transcribe(tmp_path, fp16=False)
        transcript = result["text"].strip()
        
        logger.info(f"[STT] Transcript: '{transcript}'")
        return transcript
        
    except Exception as e:
        logger.error(f"[STT] Transcription error: {e}")
        return ""
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass
