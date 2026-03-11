import os
import json
import logging
from google import genai
from dotenv import load_dotenv
from environment_memory import get_memory

load_dotenv()
logger = logging.getLogger(__name__)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

GUIDE_SYSTEM_PROMPT = """You are a calm, human-like AI guide walking beside a blind person.
You help them reach their goal by analyzing their environment and giving clear guidance.

RULES:
1. Speak in first-person, like a calm friend: "I can see...", "You're heading..."
2. Focus on the user's current GOAL. Direct your guidance toward achieving it.
3. If the goal appears in the observations, announce it enthusiastically: "You've reached [goal]!"
4. Keep responses under 20 words.
5. Be warm and reassuring, never robotic.
6. Reference what you actually see in the observations, don't hallucinate.
"""

def generate_guidance(session_id: str, latest_observation: str = None) -> str:
    """
    Main reasoning engine: takes the current session memory and generates
    a contextual, goal-driven navigation instruction.
    """
    mem = get_memory(session_id)
    goal = mem.get("current_goal", "no specific destination")
    task_status = mem.get("task_status", "idle")
    observations = mem.get("environment_observations", [])
    history = mem.get("conversation_history", [])
    
    if latest_observation:
        observations = observations + [latest_observation]  # Include latest without saving yet
    
    # Build context string for the LLM
    obs_text = "\n".join([f"- {o}" for o in observations[-3:]]) if observations else "No observations yet."
    history_text = "\n".join([f"{t['role'].upper()}: {t['text']}" for t in history[-4:]]) if history else ""
    
    prompt = f"""User's goal: "{goal}"
Task status: {task_status}

Recent environment observations:
{obs_text}

Conversation so far:
{history_text}

Now generate your next navigation instruction or response:"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=GUIDE_SYSTEM_PROMPT,
                temperature=0.4,
                max_output_tokens=60
            )
        )
        reply = response.text.strip()
        logger.info(f"[GUIDE] → {reply}")
        return reply
        
    except Exception as e:
        err = str(e)
        if "429" in err:
            logger.warning("[GUIDE] Rate limited. Skipping.")
        else:
            logger.error(f"[GUIDE] Error: {e}")
        return None

def answer_question(session_id: str, question: str, latest_observation: str) -> str:
    """
    Handles an ad-hoc vision query from the user mid-navigation.
    """
    mem = get_memory(session_id)
    obs = mem.get("environment_observations", [])
    combined = obs[-2:] + [latest_observation] if latest_observation else obs[-3:]
    obs_text = "\n".join([f"- {o}" for o in combined])
    
    prompt = f"""Environment observations:
{obs_text}

User question: "{question}"

Answer the user's question based on the observations above. Be direct and concise (under 15 words)."""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction="You are a helpful AI vision assistant for a blind user. Answer very short.",
                temperature=0.2,
                max_output_tokens=40
            )
        )
        reply = response.text.strip()
        logger.info(f"[GUIDE-Q] → {reply}")
        return reply
    except Exception as e:
        logger.error(f"[GUIDE-Q] Error: {e}")
        return "I couldn't check that right now."
