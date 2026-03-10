import os
import logging
import json
from google import genai
from dotenv import load_dotenv
from memory import get_session

load_dotenv()
logger = logging.getLogger(__name__)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
# Using flash-lite for fastest navigation reasoning
MODEL_NAME = "gemini-2.0-flash-lite"

SYSTEM_PROMPT = """You are the reasoning engine for an autonomous blind navigation system.
Your goal is to guide the user safely.

RULES:
1. Mention obstacles affecting walking.
2. Provide directional guidance.
3. Avoid repeating previous instructions.
4. Keep sentences under 10 words.
5. Sound natural and calm.

BAD outputs:
- Long sentences
- Visual descriptions (colors, lighting)
- Repeating the same sentence

INPUT DATA:
- JSON string containing objects and free space.
- Previous instruction given to the user.

OUTPUT:
- Produce ONE short sentence. (Example: "Pole ahead. Move slightly left.")
- MUST returning a valid JSON object matching the following format exactly:
  {"instruction": "<Your 10-word instruction here>"}
"""

def generate_navigation_instruction(session_id: str, scene_json: str) -> dict:
    """
    Stage 5: Navigation Reasoning Engine
    Consumes the scene JSON and session memory, then outputs a calm,
    short directional instruction via the LLM context.
    """
    state = get_session(session_id)
    last_instruction = state["last_instruction"]
    
    prompt = f"""
Current Scene:
{scene_json}

Previous Instruction: {last_instruction or 'None'}

Generate guidance (under 10 words as JSON):"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,
                max_output_tokens=40,
                response_mime_type="application/json"
            )
        )
        
        # Gemini JSON mode guarantees standard parsing
        result = json.loads(response.text)
        
        # Double check length rule
        text = result.get("instruction", "")
        if len(text.split()) > 12: # small buffer
            logger.warning(f"LLM verbosity violation: '{text}'")
            
        return result
        
    except Exception as e:
        logger.error(f"Navigation Reasoning Error: {e}")
        return {"instruction": None}
