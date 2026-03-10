import os
import logging
from google import genai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# The persona prompt. Strict scene describer, NOT object lister.
SYSTEM_PROMPT = """You are an elite mobility guide for a blind person holding a phone camera.
You are calm, intelligent, and sound exactly like a human walking beside them.

SCENE INPUT: You receive structured data about what the phone camera sees — objects, positions (left, center, right), distance (0.0=far, 1.0=extremely close).

YOUR JOB:
- Describe the scene as one flowing, natural spoken sentence.
- Focus on what matters for WALKING SAFETY. 
- Mention obstacles by feel and direction (e.g. "something on your right", "a wall ahead", "two people ahead").
- Guide them through free space.

STRICT RULES:
1. ONE sentence only. Maximum 15 words.
2. NEVER list objects robotically ("person left, car center"). Be conversational.
3. If NOTHING has meaningfully changed from the last instruction, respond with exactly: SKIP
4. DO NOT say SKIP if this is the first instruction or something new has appeared.
5. Output ONLY the spoken text. No quotes, no markdown."""

def generate_guidance(scene_json: str, history: list) -> dict:
    """
    Call the LLM to generate a natural guidance sentence based on the scene.
    Returns {"instruction": str, "skip": bool}
    """
    history_text = "No previous instructions."
    if history:
        lines = [f"[{i+1}s ago]: '{h['instruction']}'" for i, h in enumerate(reversed(history[-3:]))]
        history_text = "\n".join(lines)

    prompt = f"""RECENT INSTRUCTIONS (do NOT repeat these):
{history_text}

CURRENT SCENE:
{scene_json}

Your response:"""

    try:
        # gemini-1.5-flash-8b is ~3x faster than gemini-2.5-flash with good enough quality
        response = client.models.generate_content(
            model="gemini-1.5-flash-8b",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.4,
                max_output_tokens=60,
            )
        )
        text = response.text.strip().strip('"').strip("'")
        logger.info(f"LLM response: '{text}'")
        
        if text.upper() == "SKIP":
            return {"instruction": None, "skip": True}
        return {"instruction": text, "skip": False}
    
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return {"instruction": None, "skip": True}
