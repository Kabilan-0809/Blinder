"""
text_to_speech.py

Async text-to-speech using Microsoft Edge TTS.
Generates speech audio from text and returns it as base64-encoded MP3.

Edge TTS is free, fast (~200ms for short sentences), and supports
natural-sounding neural voices.

Usage:
    from speech.text_to_speech import synthesize_speech, get_available_voices

    # Async
    audio_b64 = await synthesize_speech("Stop. Chair ahead.")

    # With custom voice
    audio_b64 = await synthesize_speech("Turn left here.", voice="en-US-GuyNeural")
"""

import asyncio  # type: ignore
import base64  # type: ignore
import tempfile  # type: ignore
import os  # type: ignore
import logging  # type: ignore

logger = logging.getLogger("speech.tts")

# Default voice — Aria is clear, warm, and natural for assistive use
DEFAULT_VOICE = "en-US-AriaNeural"
DEFAULT_RATE = "+10%"      # slightly faster for urgency
DEFAULT_VOLUME = "+0%"

# Supported assistive voices (warm, clear neural voices)
VOICE_OPTIONS = {
    "aria":   "en-US-AriaNeural",       # Female, warm and clear
    "guy":    "en-US-GuyNeural",        # Male, calm and steady
    "jenny":  "en-US-JennyNeural",      # Female, friendly
    "davis":  "en-US-DavisNeural",      # Male, professional
    "sara":   "en-US-SaraNeural",       # Female, natural
}


async def synthesize_speech(
    text: str,
    *,
    voice: str = DEFAULT_VOICE,
    rate: str = DEFAULT_RATE,
    volume: str = DEFAULT_VOLUME,
) -> str | None:  # type: ignore
    """
    Convert text to speech using Edge TTS.

    Args:
        text:    Text to synthesize
        voice:   Edge TTS voice name (e.g. "en-US-AriaNeural")
        rate:    Speed adjustment (e.g. "+10%", "-5%")
        volume:  Volume adjustment (e.g. "+0%", "+20%")

    Returns:
        Base64-encoded MP3 audio string, or None on error
    """
    if not text or not text.strip():
        return None

    try:
        import edge_tts  # type: ignore

        communicate = edge_tts.Communicate(
            text=text.strip(),
            voice=voice,
            rate=rate,
            volume=volume,
        )

        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name

        await communicate.save(tmp_path)

        # Read and encode
        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        logger.info(
            f"[TTS] Synthesized {len(text)} chars → {len(audio_bytes)//1024}KB MP3"
        )
        return audio_b64  # type: ignore

    except ImportError:
        logger.warning("[TTS] edge-tts not installed. Install with: pip install edge-tts")
        return None  # type: ignore
    except Exception as e:
        logger.error(f"[TTS] Synthesis error: {e}")
        return None  # type: ignore
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


async def synthesize_speech_bytes(
    text: str,
    *,
    voice: str = DEFAULT_VOICE,
    rate: str = DEFAULT_RATE,
) -> bytes | None:  # type: ignore
    """
    Synthesize speech and return raw MP3 bytes (for streaming).
    """
    b64 = await synthesize_speech(text, voice=voice, rate=rate)
    if b64:
        return base64.b64decode(b64)
    return None  # type: ignore


def get_available_voices() -> dict:  # type: ignore
    """Return the map of short voice names to Edge TTS identifiers."""
    return dict(VOICE_OPTIONS)
