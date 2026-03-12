import os  # type: ignore
import base64  # type: ignore
import logging  # type: ignore
from google import genai  # type: ignore
from dotenv import load_dotenv  # type: ignore

load_dotenv()
logger = logging.getLogger(__name__)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

VISION_PROMPT = """You are an AI vision system assisting a blind person navigate.
Analyze this camera frame and describe what is relevant for navigation.

Focus on:
- Signs and labels (terminal numbers, directions, exits, restrooms)
- Paths and corridors (open, blocked, narrow)
- Crowds of people (density, direction of movement)
- Obstacles (chairs, poles, walls, vehicles)
- Flooring hazards (stairs, ramps, curbs)

Be concise. One to three sentences. Do not describe colors or aesthetics.
If the image is too blurry or dark to analyze, say: UNCLEAR
"""

def analyze_frame(jpeg_bytes: bytes, goal: str = "") -> str:  # type: ignore
    """
    Sends a JPEG frame to Gemini 1.5 Flash for multimodal visual scene reasoning.
    Returns: A short scene description focused on navigation.
    """
    try:
        import google.genai.types as gtypes  # type: ignore
        
        goal_context = f"The user's current goal is: {goal}. " if goal else ""
        prompt = goal_context + VISION_PROMPT
        
        response = client.models.generate_content(
            model="gemini-1.5-flash",   # Use the full vision model
            contents=[
                gtypes.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                gtypes.Part.from_text(prompt)
            ]
        )
        
        description = response.text.strip()
        logger.info(f"[VISION] Scene: {description}")
        return description  # type: ignore
        
    except Exception as e:
        logger.error(f"[VISION] Frame analysis error: {e}")
        return "Unable to analyze frame."  # type: ignore
