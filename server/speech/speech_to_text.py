import whisper  # type: ignore
import base64  # type: ignore
import tempfile  # type: ignore
import os  # type: ignore
import logging  # type: ignore

logger = logging.getLogger(__name__)

# Load once at startup — 'base' is the best speed/accuracy trade-off for on-device use
_model = None

def get_model():  # type: ignore
    global _model
    if _model is None:
        logger.info("[STT] Loading Whisper base model...")
        _model = whisper.load_model("base")
        logger.info("[STT] Whisper model ready.")
    return _model  # type: ignore

def transcribe_audio(audio_b64: str) -> str:  # type: ignore
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
        result = model.transcribe(tmp_path, fp16=False)  # type: ignore
        transcript = result["text"].strip()
        
        logger.info(f"[STT] Transcript: '{transcript}'")
        return transcript  # type: ignore
        
    except Exception as e:
        logger.error(f"[STT] Transcription error: {e}")
        return ""  # type: ignore
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass
