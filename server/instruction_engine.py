import os
import logging
from google import genai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

SYSTEM_PROMPT = """You are a calm, intelligent mobility guide for a blind person walking with a phone.
You describe what the camera sees as if you are the person's eyes — naturally, like a calm friend beside them.

RULES:
1. Describe the scene holistically in ONE short sentence (max 15 words). 
   Use the provided SCENE ANALYSIS text to know where there is safe free space.
   Bad: "Chair left, table center, person right."
   Good: "You're in a messy room — walk carefully, there's things on both sides."
   Good: "Someone's coming right at you, stay to the right where it's clear."
   Good: "The center path looks clear, you can walk straight ahead."
2. NEVER repeat exactly what you said in the recent history. If nothing important is new, reply ONLY with the word: SKIP
3. On the very FIRST call (history is empty), you MUST describe the scene — never SKIP.
4. Focus heavily on where the user CAN walk safely, not just naming objects.
5. Output ONLY the spoken text. No quotes, no markdown."""

def generate_guidance(scene_json: str, history: list, is_first: bool = False) -> dict:
    history_text = "No previous instructions — this is the very first time. You MUST describe the scene."
    if history and not is_first:
        lines = [f"- '{h['instruction']}'" for h in history[-3:]]
        history_text = "Recent instructions (do NOT repeat these):\n" + "\n".join(lines)

    prompt = f"""{history_text}

Current scene detected by camera:
{scene_json}

Your one-sentence guidance:"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.5,
                max_output_tokens=60,
            )
        )
        text = response.text.strip().strip('"').strip("'")
        logger.info(f"LLM → '{text}'")

        if text.upper() == "SKIP":
            return {"instruction": None, "skip": True}
        return {"instruction": text, "skip": False}

    except Exception as e:
        logger.error(f"LLM error: {e}")
        return {"instruction": None, "skip": True}
