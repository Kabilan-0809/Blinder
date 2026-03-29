"""
ocr_engine.py

Specialized vision engine for interactive document scanning.
Instead of general scene description, this module focuses intensely on
framing advice and dense text extraction for blind users.
"""

import os
import json
import logging
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Disable thinking for snappy OCR feedback
_NO_THINK = types.ThinkingConfig(thinking_budget=0)

SYSTEM_PROMPT = """You are a highly advanced document scanner assisting a blind person.
They are holding their phone camera towards a document (menu, letter, sign, screen).

Your goal is to evaluate the image.
1. If the document is blurry, cut off, too dark, or mostly out of frame, you MUST NOT try to guess the text. Instead, tell the user exactly how to move the phone to fix it (e.g. "Move the phone 2 inches back and tilt it up slightly", or "Move the paper to the left").
2. If the document is clearly legible, extract ALL of the text from it.

RETURN ONLY JSON. Return exactly this schema:
{
  "status": "bad" | "good",
  "feedback": "Physical instruction to fix framing (only if status is bad)",
  "text": "Full extracted text (only if status is good)"
}
"""

def scan_document(jpeg_bytes: bytes) -> dict:
    """
    Evaluates a frame for document readability.
    Returns:
       {"status": "bad", "feedback": "... instruction ..."}
    OR {"status": "good", "text": "... extracted text ..."}
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                "Evaluate this document scan for readability based on your system instructions.",
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,  # Low temperature for precise OCR
                response_mime_type="application/json",
                thinking_config=_NO_THINK,
            ),
        )
        
        raw_text = response.text or ""
        if not raw_text.strip():
            return {"status": "bad", "feedback": "I can't see anything. Hold it up to the camera."}
            
        data = json.loads(raw_text)
        
        status = data.get("status", "bad")
        feedback = data.get("feedback", "I couldn't quite see it. Try again.")
        extracted = data.get("text", "")
        
        if status == "good" and not extracted.strip():
            # LLM said good but extracted nothing
            status = "bad"
            feedback = "I didn't detect any text. Is there writing on it?"
            
        return {
            "status": status,
            "feedback": feedback,
            "text": extracted
        }
        
    except Exception as e:
        logger.error(f"[OCR] Error scanning document: {e}")
        return {
            "status": "bad", 
            "feedback": "I had trouble reading the image. Let's try again."
        }
