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
import re  # type: ignore
import agent.environment_memory as mem_mgr  # type: ignore
import personality.companion_personality as iris  # type: ignore

load_dotenv()
logger = logging.getLogger(__name__)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Disable thinking for fast, cheap responses
_NO_THINK = gtypes.ThinkingConfig(thinking_budget=0)

# Stop-words to ignore when comparing guidance for similarity
_STOP = {
    'a', 'an', 'the', 'is', 'are', 'was', 'be', 'been', 'being',
    'to', 'of', 'and', 'in', 'on', 'at', 'for', 'with', 'you',
    'your', 'i', 'it', 'its', "it's", 'there', 'here', 'this',
    'that', 'just', 'now', 'so', 'let', "let's", 'about', 'up',
    'go', 'can', 'will', 'me', 'my', 'by', "you're", 'or', 'if',
}


def _content_words(text: str) -> set:  # type: ignore
    """Extract meaningful content words from a string for similarity comparison."""
    tokens = re.sub(r"[^a-z0-9']", ' ', text.lower()).split()
    return {t for t in tokens if t not in _STOP and len(t) > 2}


def _is_too_similar(new_text: str, history: list, threshold: float = 0.50) -> bool:  # type: ignore
    """
    Returns True if new_text overlaps >threshold with ANY of the last 4 guidance strings.
    Uses Jaccard index on content words.
    """
    new_words = _content_words(new_text)
    if not new_words:
        return False
    for past in history[-4:]:        # only compare against recent 4
        past_words = _content_words(past)
        if not past_words:
            continue
        intersection = len(new_words & past_words)
        union = len(new_words | past_words)
        if union > 0 and (intersection / union) >= threshold:
            return True
    return False


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

def generate_guidance(
    session_id: str,
    scene: dict | None = None,
    nav_progress: dict | None = None,
) -> str | None:  # type: ignore
    """
    Generate a single navigation instruction fusing:
      - GPS route step (if route is loaded)
      - Visual scene description
      - Semantic dedup against recent guidance history
    Returns None if the new guidance is too similar to something recently said.
    """
    mem = mem_mgr.get_memory(session_id)
    goal = mem.get("navigation_goal") or "no specific destination"
    task_status = mem.get("task_status", "idle")
    history = mem.get("conversation_history", [])
    env = mem.get("environment_memory", {})
    guidance_history = mem_mgr.get_guidance_history(session_id)

    scene_desc = (scene or {}).get("description") or env.get("last_scene_description") or "No scene data."

    # ── GPS route step context ─────────────────────────────────
    nav = nav_progress or mem.get("navigation_progress", {})
    route_steps = nav.get("route_steps", [])
    current_step_idx = nav.get("current_step", 0)
    distance_to_turn = nav.get("distance_to_turn")
    route_context = ""
    if route_steps and current_step_idx < len(route_steps):
        step_instr = route_steps[current_step_idx].get("instruction", "")
        if distance_to_turn is not None:
            route_context = (
                f"GPS direction: {step_instr} "
                f"(turn in approximately {distance_to_turn:.0f} metres)."
            )
        else:
            route_context = f"GPS direction: {step_instr}."

    # ── What has Iris said recently (avoid repetition)? ────────
    recent_said = guidance_history[-4:] if guidance_history else []
    avoid_block = ""
    if recent_said:
        bullet_list = "\n".join(f'  - "{g}"' for g in recent_said)
        avoid_block = (
            f"\nYou recently said:\n{bullet_list}\n"
            "Do NOT repeat these ideas or phrasings. "
            "Give a NEW, different instruction or observation.\n"
        )

    # ── Long-running task context ──────────────────────────────
    long_tasks: list = []
    try:
        import agent.task_manager as task_manager  # type: ignore
        engine = task_manager.get_engine(session_id)
        long_tasks = [t["description"] for t in engine.get_active_tasks() if t["type"] == "LONG_RUNNING"]
    except Exception:
        pass

    tasks_text = "\n".join([f"- {t}" for t in long_tasks]) if long_tasks else "None."
    history_text = "\n".join([f"{t['role'].upper()}: {t['text']}" for t in history[-4:]]) if history else ""

    prompt = f"""Navigation goal: "{goal}"
Status: {task_status}
{route_context}

Current spatial scene: {scene_desc}

Active monitoring tasks:
{tasks_text}

Recent conversation:
{history_text}
{avoid_block}
Give your next navigation instruction. RULES:
- State DIRECTION (left/right/ahead/behind) and DISTANCE in metres for anything you mention.
- Use clock positions ("at your 9 o'clock") when helpful.
- NEVER mention colors, textures, or patterns.
- Be warm and natural but spatially precise. Max 30 words:"""

    system_prompt = _build_system_prompt(session_id, mem)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=gtypes.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.6,
                max_output_tokens=100,
                thinking_config=_NO_THINK,
            ),
        )
        reply = _safe_text(response)
        if not reply:
            return None  # type: ignore

        # ── Semantic dedup: suppress if too similar to recent guidance ──
        if _is_too_similar(reply, guidance_history):
            logger.info(f"[GUIDE] Suppressed (too similar to recent): '{reply[:60]}'")
            return None  # type: ignore

        mem_mgr.push_guidance(session_id, reply)
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

    prompt = f"""Spatial scene data: {scene_desc}
Visible signs: {signs or 'none'}
Crowd density: {crowd}

User question: "{question}"

Answer ONLY in spatial terms a blind person can act on. RULES:
- State exact direction: left / right / ahead / behind / slightly left / slightly right.
- State a CLOCK POSITION (e.g. "at your 9 o'clock") for every object.
- State ESTIMATED DISTANCE in metres or steps.
- NEVER mention color, texture, or visual appearance.
- Be concise and actionable (under 25 words). Example: 'Your bag is 1 metre to your left, at your 9 o'clock. Reach down slightly.'"""

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
        import agent.task_manager as task_manager  # type: ignore
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
