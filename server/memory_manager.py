"""
memory_manager.py

Production-grade session memory for the Blind AI navigation assistant.
Each WebSocket session gets its own fully isolated memory store.

Memory structure:
  navigation_goal:       str | None
  navigation_progress:   { route_steps, current_step, distance_to_turn }
  active_tasks:          { long_running: [...], short: [...] }
  environment_memory:    { observed_signs, crowd_density, corridor_direction, last_clear_path_meters }
  conversation_history:  [{ role, text }, ...]   (rolling last 10)
  last_location:         { lat, lng } | None
  task_status:           idle | active | paused | completed
  pending_question:      str | None
"""

import logging  # type: ignore

logger = logging.getLogger(__name__)

# Global session store — session_id → memory dict
_sessions: dict = {}


# ─────────────────────────────────────────────────────────────
# Core session access
# ─────────────────────────────────────────────────────────────

def get_memory(session_id: str) -> dict:  # type: ignore
    """Return (creating if needed) the full memory dict for a session."""
    if session_id not in _sessions:
        _sessions[session_id] = _new_session()
        logger.info(f"[MEM] Created new session: {session_id}")
    return _sessions[session_id]


def _new_session() -> dict:  # type: ignore
    return {
        "navigation_goal": None,
        "navigation_progress": {
            "route_steps": [],
            "current_step": 0,
            "distance_to_turn": None,
            "last_announced_threshold": None,
        },
        "active_tasks": {
            "long_running": [],   # persist until user cancels
            "short": [],          # answered once, then removed
        },
        "environment_memory": {
            "observed_signs": [],
            "crowd_density": "unknown",
            "corridor_direction": "unknown",
            "last_clear_path_meters": None,
            "last_scene_description": None,
        },
        "conversation_history": [],
        "last_location": None,
        "task_status": "idle",      # idle | active | paused | completed
        "pending_question": None,
    }


def clear_session(session_id: str):  # type: ignore
    """Fully reset a session (e.g. user starts over)."""
    _sessions[session_id] = _new_session()
    logger.info(f"[MEM] Cleared session: {session_id}")


# ─────────────────────────────────────────────────────────────
# Navigation
# ─────────────────────────────────────────────────────────────

def set_navigation_goal(session_id: str, goal: str):  # type: ignore
    """Set a new navigation goal and activate the session."""
    mem = get_memory(session_id)
    mem["navigation_goal"] = goal
    mem["task_status"] = "active"
    mem["navigation_progress"] = {
        "route_steps": [],
        "current_step": 0,
        "distance_to_turn": None,
        "last_announced_threshold": None,
    }
    mem["environment_memory"]["observed_signs"] = []
    logger.info(f"[MEM] Goal set: '{goal}'")


def clear_navigation_goal(session_id: str):  # type: ignore
    """Called when user reaches destination."""
    mem = get_memory(session_id)
    mem["navigation_goal"] = None
    mem["task_status"] = "completed"
    mem["active_tasks"]["long_running"] = []
    logger.info(f"[MEM] Navigation complete for session {session_id}")


# ─────────────────────────────────────────────────────────────
# Task management
# ─────────────────────────────────────────────────────────────

def add_long_running_task(session_id: str, task: str):  # type: ignore
    """Add a task that persists for the entire navigation (e.g. 'warn about poles')."""
    mem = get_memory(session_id)
    if task not in mem["active_tasks"]["long_running"]:
        mem["active_tasks"]["long_running"].append(task)
        logger.info(f"[MEM] Long-running task added: '{task}'")


def add_short_task(session_id: str, task: str):  # type: ignore
    """Add a one-shot query task (e.g. 'is there a crowd ahead?')."""
    mem = get_memory(session_id)
    if task not in mem["active_tasks"]["short"]:
        mem["active_tasks"]["short"].append(task)
        logger.info(f"[MEM] Short task added: '{task}'")


def remove_short_task(session_id: str, task: str):  # type: ignore
    """Remove a short task after it has been answered."""
    mem = get_memory(session_id)
    try:
        mem["active_tasks"]["short"].remove(task)
        logger.info(f"[MEM] Short task answered & removed: '{task}'")
    except ValueError:
        pass


def get_active_tasks(session_id: str) -> dict:  # type: ignore
    """Return the full active_tasks dict."""
    return get_memory(session_id)["active_tasks"]


def set_pending_question(session_id: str, question: str):  # type: ignore
    mem = get_memory(session_id)
    mem["pending_question"] = question
    mem["task_status"] = "paused"


def clear_pending_question(session_id: str):  # type: ignore
    mem = get_memory(session_id)
    mem["pending_question"] = None
    if mem["task_status"] == "paused":
        mem["task_status"] = "active"


# ─────────────────────────────────────────────────────────────
# Environment memory
# ─────────────────────────────────────────────────────────────

def update_scene(session_id: str, scene: dict):  # type: ignore
    """
    Update environment memory from a structured scene result.
    Expected keys: description, signs_seen, decision_point, crowd_density,
                   estimated_clear_path_m, corridor_direction
    """
    mem = get_memory(session_id)
    env = mem["environment_memory"]

    env["last_scene_description"] = scene.get("description", "")
    env["crowd_density"] = scene.get("crowd_density", env["crowd_density"])
    env["last_clear_path_meters"] = scene.get("estimated_clear_path_m")

    if scene.get("corridor_direction"):
        env["corridor_direction"] = scene["corridor_direction"]

    for sign in scene.get("signs_seen", []):
        if sign and sign not in env["observed_signs"]:
            env["observed_signs"].append(sign)
            # Keep only the last 10 unique signs
            if len(env["observed_signs"]) > 10:
                env["observed_signs"].pop(0)


# ─────────────────────────────────────────────────────────────
# Conversation history
# ─────────────────────────────────────────────────────────────

def add_turn(session_id: str, role: str, text: str):  # type: ignore
    """Add a conversation turn; keep rolling window of last 10."""
    mem = get_memory(session_id)
    history = mem["conversation_history"]
    history.append({"role": role, "text": text})
    if len(history) > 10:
        history.pop(0)
