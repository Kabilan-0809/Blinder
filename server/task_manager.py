import os
import json
import logging
from google import genai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

TASK_EXTRACT_PROMPT = """You are an AI assistant for a blind navigation system.
The user has spoken. Extract their primary intent from their speech.

Intents:
- "navigate": they are explicitly stating a destination they want to go to
- "query": they are asking a question about their current physical environment/surroundings
- "interrupt": they want to stop or pause the current navigation
- "chat": casual conversation, asking how the app works, how you are, or anything unrelated to physical movement

Return ONLY a JSON object with this format:
For navigation: {"intent": "navigate", "goal": "terminal A10"}
For a query: {"intent": "query", "question": "Is there a crowd ahead?"}
For an interrupt: {"intent": "interrupt", "new_request": "pause for a second"}
For chat: {"intent": "chat", "text": "how to control this app"}
"""

def extract_task(transcript: str) -> dict:
    """
    Parses the user's voice transcript into a structured task using Gemini.
    Returns: {"intent": str, "goal": str} or {"intent": "query", "question": str}
    """
    if not transcript:
        return {"intent": "unknown"}
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=f"User said: \"{transcript}\"",
            config=genai.types.GenerateContentConfig(
                system_instruction=TASK_EXTRACT_PROMPT,
                temperature=0.1,
                max_output_tokens=80,
                response_mime_type="application/json"
            )
        )
        result = json.loads(response.text)
        logger.info(f"[TASK] Extracted: {result}")
        return result
        
    except Exception as e:
        logger.error(f"[TASK] Extraction error: {e}")
        # Fallback: treat the whole thing as general chat
        return {"intent": "chat", "text": transcript}
