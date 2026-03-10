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
   Bad: "Chair left, table center, person right."
   Good: "You're in a cluttered room — walk carefully, there are things on both sides."
   Good: "Someone's ahead of you, stay to the right."
   Good: "The path looks clear, you can walk straight."
2. NEVER repeat what you said in the recent history. If nothing important is new, reply ONLY with the word: SKIP
3. On the very FIRST call (history is empty), you MUST describe the scene — never SKIP.
4. Focus on what matters for safe walking, not random objects.
5. Output ONLY the spoken text. No quotes, no punctuation beyond natural speech."""

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
            model="gemini-1.5-flash-8b",
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
