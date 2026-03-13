"""
task_manager.py

Parses the user's voice transcript into a structured task using Gemini.

Enhanced to extract:
  - Primary intent (navigate | query | interrupt | chat)
  - Navigation goal
  - Long-running tasks (persist across session, e.g. "warn about poles")
  - Short tasks (one-shot queries, e.g. "is there a crowd?")

Example input: "Guide me to the supermarket and warn me if there are poles ahead"
Example output:
  {
    "intent": "navigate",
    "goal": "nearest supermarket",
    "long_running_tasks": ["warn about poles ahead"],
    "short_tasks": [],
    "question": null,
    "text": null
  }
"""

import os  # type: ignore
import json  # type: ignore
import logging  # type: ignore
from google import genai  # type: ignore
from dotenv import load_dotenv  # type: ignore

load_dotenv()
logger = logging.getLogger(__name__)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

TASK_EXTRACT_PROMPT = """You are an AI assistant for a blind navigation system.
Parse the user's spoken sentence and return ONLY a JSON object.

Intents:
- "navigate": user wants to go somewhere
- "query": user is asking a ONE-TIME question about surroundings (answered once, then done)
- "interrupt": user wants to stop, pause, or cancel navigation
- "chat": casual conversation unrelated to physical movement

Also extract tasks embedded in the sentence:
- "long_running_tasks": ongoing monitoring tasks that persist throughout navigation
  e.g. "warn me about poles", "tell me if there are crowds", "alert me to steps"
- "short_tasks": one-time queries embedded alongside a navigation request
  e.g. "is there a cafe nearby?"

Return JSON matching ONE of these schemas:

Navigate with tasks:
{"intent":"navigate","goal":"<destination>","long_running_tasks":["..."],"short_tasks":[],"question":null,"text":null}

Query only:
{"intent":"query","goal":null,"long_running_tasks":[],"short_tasks":[],"question":"<question>","text":null}

Interrupt:
{"intent":"interrupt","goal":null,"long_running_tasks":[],"short_tasks":[],"question":null,"text":"<what user said>"}

Chat:
{"intent":"chat","goal":null,"long_running_tasks":[],"short_tasks":[],"question":null,"text":"<what user said>"}

Rules:
- Always include all fields, even if null or empty list
- long_running_tasks and short_tasks are lists of strings (empty list if none)
- goal: extract the exact destination phrase, not paraphrased
- question: the exact question, if intent is "query"
"""


def extract_task(transcript: str) -> dict:  # type: ignore
    """
    Parse a voice transcript into a structured task + sub-task dict.
    Returns a safe fallback dict on any error.
    """
    if not transcript or not transcript.strip():
        return _fallback(transcript or "")

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=f'User said: "{transcript}"',
            config=genai.types.GenerateContentConfig(
                system_instruction=TASK_EXTRACT_PROMPT,
                temperature=0.05,
                max_output_tokens=150,
                response_mime_type="application/json",
            ),
        )
        result = json.loads(response.text)  # type: ignore

        # Normalise — ensure all expected fields are present
        result.setdefault("long_running_tasks", [])
        result.setdefault("short_tasks", [])
        result.setdefault("goal", None)
        result.setdefault("question", None)
        result.setdefault("text", None)

        logger.info(f"[TASK] intent={result.get('intent')} goal={result.get('goal')} "
                    f"long={result.get('long_running_tasks')} short={result.get('short_tasks')}")
        return result  # type: ignore

    except Exception as e:
        logger.error(f"[TASK] Extraction error: {e}")
        return _fallback(transcript)


def _fallback(transcript: str) -> dict:  # type: ignore
    return {
        "intent": "chat",
        "goal": None,
        "long_running_tasks": [],
        "short_tasks": [],
        "question": None,
        "text": transcript,
    }
