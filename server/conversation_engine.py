"""
conversation_engine.py

LLM-backed conversational layer for the Blind AI navigation assistant.
Now powered by Iris — a warm, personalized AI companion.

Functions:
  generate_guidance()          — proactive navigation + task-aware scene guidance
  answer_question()            — answer a direct user query using scene context
  handle_chat()                — pure conversational response
  generate_arrival()           — warm arrival confirmation when goal is reached
  check_long_running_tasks()   — evaluate long-running tasks against scene
"""

import os  # type: ignore
import logging  # type: ignore
from google import genai  # type: ignore
from google.genai import types as gtypes  # type: ignore
from dotenv import load_dotenv  # type: ignore
import memory_manager as mem_mgr  # type: ignore
import companion_personality as iris  # type: ignore

load_dotenv()
logger = logging.getLogger(__name__)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Disable thinking for fast, cheap responses
_NO_THINK = gtypes.ThinkingConfig(thinking_budget=0)


def _safe_text(response) -> str:  # type: ignore
    """Safely extract text from a Gemini response, handling None."""
    text = getattr(response, 'text', None)
    if text is None:
        return ""
    return text.strip()


def _build_system_prompt(session_id: str, mem: dict, extra_context: str = "") -> str:  # type: ignore
    """Build a full Iris system prompt with personality context."""
    profile = mem.get("user_profile", {})
    ps = mem.get("personality_state", {})

    personality_ctx = iris.build_personality_context(
        session_id=session_id,
        user_name=profile.get("name"),
        user_mood=ps.get("current_mood", "calm"),
        scene_mood=ps.get("scene_mood", "neutral"),
    )

    return f"""{iris.IRIS_PERSONALITY}

{personality_ctx}
{extra_context}"""


# ─────────────────────────────────────────────────────────────
# Proactive navigation guidance
# ─────────────────────────────────────────────────────────────

def generate_guidance(session_id: str, scene: dict | None = None) -> str | None:  # type: ignore
    mem = mem_mgr.get_memory(session_id)
    goal = mem.get("navigation_goal") or "no specific destination"
    task_status = mem.get("task_status", "idle")
    history = mem.get("conversation_history", [])
    env = mem.get("environment_memory", {})

    scene_desc = (scene or {}).get("description") or env.get("last_scene_description") or "No scene data."

    # Get long-running tasks from task engine
    long_tasks: list = []
    try:
        import task_manager  # type: ignore
        engine = task_manager.get_engine(session_id)
        long_tasks = [t["description"] for t in engine.get_active_tasks() if t["type"] == "LONG_RUNNING"]
    except Exception:
        pass

    history_text = "\n".join([f"{t['role'].upper()}: {t['text']}" for t in history[-4:]]) if history else ""
    tasks_text = "\n".join([f"- {t}" for t in long_tasks]) if long_tasks else "None."

    prompt = f"""Navigation goal: "{goal}"
Status: {task_status}

Current scene: {scene_desc}

Active monitoring tasks:
{tasks_text}

Recent conversation:
{history_text}

Give your next navigation instruction or scene update. Be natural & varied (max 25 words):"""

    system_prompt = _build_system_prompt(session_id, mem)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=gtypes.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.5,
                max_output_tokens=100,
                thinking_config=_NO_THINK,
            ),
        )
        reply = _safe_text(response)
        if not reply:
            return None  # type: ignore

        iris._track_response(session_id, reply)
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
    mem = mem_mgr.get_memory(session_id)
    env = mem.get("environment_memory", {})
    scene_desc = (scene or {}).get("description") or env.get("last_scene_description") or "No scene data."
    signs = ", ".join((scene or {}).get("signs_seen", env.get("observed_signs", [])))  # type: ignore
    crowd = (scene or {}).get("crowd_density") or env.get("crowd_density", "unknown")

    # Track stats
    stats = mem.get("session_stats", {})
    stats["questions_asked"] = stats.get("questions_asked", 0) + 1

    prompt = f"""Scene description: {scene_desc}
Visible signs: {signs or 'none'}
Crowd density: {crowd}

User question: "{question}"

Answer directly and naturally (under 20 words). Be warm and helpful, like talking to a friend."""

    system_prompt = _build_system_prompt(
        session_id, mem,
        extra_context="You are answering a specific question. Be direct and factual but warm."
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=gtypes.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3,
                max_output_tokens=100,
                thinking_config=_NO_THINK,
            ),
        )
        reply = _safe_text(response)
        if not reply:
            return "Hmm, I couldn't see clearly. Can you ask again?"  # type: ignore

        iris._track_response(session_id, reply)
        logger.info(f"[Q&A] Q: '{question}' → '{reply}'")
        return reply  # type: ignore

    except Exception as e:
        logger.error(f"[Q&A] Error: {e}")
        return "Sorry, I'm having trouble right now. Let me try again in a moment."  # type: ignore


# ─────────────────────────────────────────────────────────────
# Check long-running monitoring tasks against scene
# ─────────────────────────────────────────────────────────────

def check_long_running_tasks(session_id: str, scene: dict) -> str | None:  # type: ignore
    mem = mem_mgr.get_memory(session_id)

    long_tasks: list = []
    try:
        import task_manager  # type: ignore
        engine = task_manager.get_engine(session_id)
        long_tasks = [t["description"] for t in engine.get_active_tasks() if t["type"] == "LONG_RUNNING"]
    except Exception:
        pass

    if not long_tasks:
        return None  # type: ignore

    scene_desc = scene.get("description", "")
    if not scene_desc or scene_desc == "UNCLEAR":
        return None  # type: ignore

    tasks_text = "\n".join([f"- {t}" for t in long_tasks])
    prompt = f"""Scene: {scene_desc}

Monitoring tasks (check if any are triggered by the scene):
{tasks_text}

If ANY task is triggered, write ONE short alert in Iris's warm voice (max 12 words).
If NONE are triggered, respond with exactly: NONE"""

    system_prompt = _build_system_prompt(session_id, mem)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=gtypes.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.15,
                max_output_tokens=60,
                thinking_config=_NO_THINK,
            ),
        )
        result = _safe_text(response)
        if result == "NONE" or not result:
            return None  # type: ignore

        iris._track_response(session_id, result)
        logger.info(f"[TASK-CHECK] → {result}")
        return result  # type: ignore

    except Exception as e:
        logger.error(f"[TASK-CHECK] Error: {e}")
        return None  # type: ignore


# ─────────────────────────────────────────────────────────────
# General conversational chat
# ─────────────────────────────────────────────────────────────

def handle_chat(session_id: str, user_text: str) -> str:  # type: ignore
    mem = mem_mgr.get_memory(session_id)
    goal = mem.get("navigation_goal") or "none"
    history = mem.get("conversation_history", [])
    history_text = "\n".join([f"{t['role'].upper()}: {t['text']}" for t in history[-5:]]) if history else ""

    # Detect and update mood
    mood = iris.detect_mood(user_text)
    ps = mem.get("personality_state", {})
    ps["current_mood"] = mood

    prompt = f"""User's goal: {goal}
Conversation:
{history_text}
User: "{user_text}"

Respond warmly as Iris, their AI companion. Keep it under 25 words. Be natural and varied."""

    system_prompt = _build_system_prompt(session_id, mem)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=gtypes.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.65,
                max_output_tokens=100,
                thinking_config=_NO_THINK,
            ),
        )
        reply = _safe_text(response)
        if not reply:
            return "I'm right here with you. What do you need?"  # type: ignore

        iris._track_response(session_id, reply)
        logger.info(f"[CHAT] → {reply}")
        return reply  # type: ignore

    except Exception as e:
        logger.error(f"[CHAT] Error: {e}")
        return "I'm right here with you. What can I help with?"  # type: ignore


# ─────────────────────────────────────────────────────────────
# Arrival message
# ─────────────────────────────────────────────────────────────

def generate_arrival(goal: str) -> str:  # type: ignore
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f'The user has arrived at their destination: "{goal}". Announce this warmly and enthusiastically as Iris, their AI companion. Under 20 words. Celebrate!',
            config=gtypes.GenerateContentConfig(
                system_instruction=iris.IRIS_PERSONALITY,
                temperature=0.75,
                max_output_tokens=60,
                thinking_config=_NO_THINK,
            ),
        )
        reply = _safe_text(response)
        return reply or f"You made it to {goal}! Great job!"  # type: ignore
    except Exception:
        return f"You've arrived at {goal}! Well done!"  # type: ignore
