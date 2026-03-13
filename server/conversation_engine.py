"""
conversation_engine.py

LLM-backed conversational layer for the Blind AI navigation assistant.

Functions:
  generate_guidance()     — proactive navigation + task-aware scene guidance
  answer_question()       — answer a direct user query using scene context
  handle_chat()           — pure conversational response
  generate_arrival()      — warm arrival confirmation when goal is reached
  check_long_running_tasks() — evaluate long-running tasks against scene
"""

import os  # type: ignore
import logging  # type: ignore
from google import genai  # type: ignore
from dotenv import load_dotenv  # type: ignore
import memory_manager as mem_mgr  # type: ignore

load_dotenv()
logger = logging.getLogger(__name__)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

GUIDE_SYSTEM_PROMPT = """You are a calm, warm AI guide walking beside a blind person.
You speak like a trusted human companion — clear, reassuring, never robotic.

Rules:
1. Always speak in first person: "I can see...", "You're approaching...", "Just ahead..."
2. Keep ALL responses under 25 words. Brevity is critical.
3. Your primary job: help the user reach their goal.
4. React to what you actually see in the observations. Never hallucinate.
5. If the goal is visible, announce enthusiastically: "You've reached [goal]!"
6. Address active monitoring tasks proactively in your guidance.
"""


# ─────────────────────────────────────────────────────────────
# Proactive navigation guidance
# ─────────────────────────────────────────────────────────────

def generate_guidance(session_id: str, scene: dict | None = None) -> str | None:  # type: ignore
    """
    Generates a contextual navigation instruction based on:
    - Current goal
    - Recent scene observation (from scene_reasoning)
    - Long-running monitoring tasks
    - Conversation history

    Returns guidance string or None if rate-limited / error.
    """
    mem = mem_mgr.get_memory(session_id)
    goal = mem.get("navigation_goal") or "no specific destination"
    task_status = mem.get("task_status", "idle")
    history = mem.get("conversation_history", [])
    long_tasks = mem.get("active_tasks", {}).get("long_running", [])
    env = mem.get("environment_memory", {})

    scene_desc = (scene or {}).get("description") or env.get("last_scene_description") or "No scene data."

    history_text = "\n".join([f"{t['role'].upper()}: {t['text']}" for t in history[-4:]]) if history else ""
    tasks_text = "\n".join([f"- {t}" for t in long_tasks]) if long_tasks else "None."

    prompt = f"""Navigation goal: "{goal}"
Status: {task_status}

Current scene: {scene_desc}

Active monitoring tasks:
{tasks_text}

Recent conversation:
{history_text}

Provide your next navigation instruction or scene update (max 25 words):"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=GUIDE_SYSTEM_PROMPT,
                temperature=0.35,
                max_output_tokens=60,
            ),
        )
        reply = response.text.strip()  # type: ignore
        logger.info(f"[GUIDE] → {reply}")
        return reply  # type: ignore

    except Exception as e:
        err = str(e)
        if "429" in err:
            logger.warning("[GUIDE] Rate limited.")
        else:
            logger.error(f"[GUIDE] Error: {e}")
        return None  # type: ignore


# ─────────────────────────────────────────────────────────────
# Answer a direct user question
# ─────────────────────────────────────────────────────────────

def answer_question(session_id: str, question: str, scene: dict | None = None) -> str:  # type: ignore
    """
    Answers a user's specific question about the current environment.
    After answering, the caller should remove the question from short_tasks.
    """
    mem = mem_mgr.get_memory(session_id)
    env = mem.get("environment_memory", {})
    scene_desc = (scene or {}).get("description") or env.get("last_scene_description") or "No scene data."
    signs = ", ".join((scene or {}).get("signs_seen", env.get("observed_signs", [])))  # type: ignore
    crowd = (scene or {}).get("crowd_density") or env.get("crowd_density", "unknown")

    prompt = f"""Scene description: {scene_desc}
Visible signs: {signs or 'none'}
Crowd density: {crowd}

User question: "{question}"

Answer directly and concisely (under 15 words). Use only what you can see."""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction="You are a helpful AI vision assistant for a blind user. Answer very short and factually.",
                temperature=0.2,
                max_output_tokens=50,
            ),
        )
        reply = response.text.strip()  # type: ignore
        logger.info(f"[Q&A] Q: '{question}' → '{reply}'")
        return reply  # type: ignore

    except Exception as e:
        logger.error(f"[Q&A] Error: {e}")
        return "I couldn't check that right now."  # type: ignore


# ─────────────────────────────────────────────────────────────
# Check long-running monitoring tasks against scene
# ─────────────────────────────────────────────────────────────

def check_long_running_tasks(session_id: str, scene: dict) -> str | None:  # type: ignore
    """
    Evaluates whether any long-running tasks are triggered by the current scene.
    Returns a spoken report or None if nothing to report.
    Example: task="detect poles ahead", scene shows pole → "There's a pole ahead."
    """
    mem = mem_mgr.get_memory(session_id)
    long_tasks = mem.get("active_tasks", {}).get("long_running", [])
    if not long_tasks:
        return None  # type: ignore

    scene_desc = scene.get("description", "")
    if not scene_desc or scene_desc == "UNCLEAR":
        return None  # type: ignore

    tasks_text = "\n".join([f"- {t}" for t in long_tasks])
    prompt = f"""Scene: {scene_desc}

Monitoring tasks (check if any are triggered by the scene):
{tasks_text}

If ANY task is triggered, write ONE short spoken alert (max 12 words).
If NONE are triggered, respond with exactly: NONE"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction="You are a safety monitor for a blind navigation system. Be precise and brief.",
                temperature=0.1,
                max_output_tokens=40,
            ),
        )
        result = response.text.strip()  # type: ignore
        if result == "NONE" or not result:
            return None  # type: ignore
        logger.info(f"[TASK-CHECK] → {result}")
        return result  # type: ignore

    except Exception as e:
        logger.error(f"[TASK-CHECK] Error: {e}")
        return None  # type: ignore


# ─────────────────────────────────────────────────────────────
# General conversational chat
# ─────────────────────────────────────────────────────────────

def handle_chat(session_id: str, user_text: str) -> str:  # type: ignore
    """Handles general conversational messages unrelated to navigation."""
    mem = mem_mgr.get_memory(session_id)
    goal = mem.get("navigation_goal") or "none"
    history = mem.get("conversation_history", [])
    history_text = "\n".join([f"{t['role'].upper()}: {t['text']}" for t in history[-5:]]) if history else ""

    prompt = f"""User's goal: {goal}
Conversation:
{history_text}
User: "{user_text}"

Respond warmly as their AI guide. Keep it under 25 words."""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction="You are a calm, friendly AI guide for a blind user. Be conversational and warm.",
                temperature=0.6,
                max_output_tokens=60,
            ),
        )
        reply = response.text.strip()  # type: ignore
        logger.info(f"[CHAT] → {reply}")
        return reply  # type: ignore

    except Exception as e:
        logger.error(f"[CHAT] Error: {e}")
        return "I'm here with you, ready to help!"  # type: ignore


# ─────────────────────────────────────────────────────────────
# Arrival message
# ─────────────────────────────────────────────────────────────

def generate_arrival(goal: str) -> str:  # type: ignore
    """Generate a warm arrival announcement."""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f'The user has arrived at their destination: "{goal}". Announce this warmly and enthusiastically in under 15 words.',
            config=genai.types.GenerateContentConfig(
                system_instruction=GUIDE_SYSTEM_PROMPT,
                temperature=0.7,
                max_output_tokens=40,
            ),
        )
        return response.text.strip()  # type: ignore
    except Exception:
        return f"You've arrived at {goal}!"  # type: ignore
