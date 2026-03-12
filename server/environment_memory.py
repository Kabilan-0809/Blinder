"""
environment_memory.py

Maintains persistent session memory during a navigation task.
Each connected WebSocket session gets its own isolated memory store.
"""

sessions: dict[str, dict] = {}

def get_memory(session_id: str) -> dict:  # type: ignore
    if session_id not in sessions:
        sessions[session_id] = {
            "current_goal": None,
            "task_status": "idle",          # idle | active | paused | completed
            "environment_observations": [], # Rolling list of last 5 vision descriptions
            "conversation_history": [],     # Rolling list of last 8 turns {"role": "user"/"assistant", "text": ...}
            "recent_locations": [],         # GPS history
            "pending_question": None        # User asked a question mid-task
        }
    return sessions[session_id]  # type: ignore

def set_goal(session_id: str, goal: str):  # type: ignore
    mem = get_memory(session_id)
    mem["current_goal"] = goal
    mem["task_status"] = "active"
    mem["environment_observations"] = []
    mem["conversation_history"] = []
    
def add_observation(session_id: str, description: str):  # type: ignore
    """Add a new vision frame description, keeping only the last 5."""
    mem = get_memory(session_id)
    obs = mem["environment_observations"]
    obs.append(description)
    if len(obs) > 5:
        obs.pop(0)

def add_turn(session_id: str, role: str, text: str):  # type: ignore
    """Add a conversation turn, keeping only the last 8 turns."""
    mem = get_memory(session_id)
    history = mem["conversation_history"]
    history.append({"role": role, "text": text})
    if len(history) > 8:
        history.pop(0)

def complete_goal(session_id: str):  # type: ignore
    mem = get_memory(session_id)
    mem["task_status"] = "completed"
    
def pause_task(session_id: str, question: str):  # type: ignore
    mem = get_memory(session_id)
    old_status = mem["task_status"]
    mem["task_status"] = "paused"
    mem["pending_question"] = question
    return old_status  # type: ignore

def resume_task(session_id: str):  # type: ignore
    mem = get_memory(session_id)
    mem["task_status"] = "active"
    mem["pending_question"] = None
