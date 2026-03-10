import time

# Redis-ready dictionary. Each session_id maps to state.
sessions = {}

def get_session(session_id: str) -> dict:
    if session_id not in sessions:
        sessions[session_id] = {
            "history": [],            # Rolling list of last 5 spoken instructions
            "last_spoken_time": 0.0,  # When we last actually SPOKE something
            "last_object_types": set(), # Object type set from last spoken frame
            "is_first": True,          # First instruction of the session?
        }
    return sessions[session_id]

def scene_has_changed(session_id: str, objects_info: list) -> bool:
    """
    Returns True if the scene has meaningfully changed since the last spoken instruction.
    We compare the SET of object types, not exact positions (positions fluctuate per frame).
    """
    state = get_session(session_id)
    current_types = set(o["type"] for o in objects_info)
    last_types = state["last_object_types"]
    
    # New objects appeared or old ones disappeared
    if current_types != last_types:
        return True
    
    return False

def is_within_cooldown(session_id: str, cooldown_secs: float = 8.0) -> bool:
    """Returns True if we spoke recently and should be quiet."""
    state = get_session(session_id)
    return (time.time() - state["last_spoken_time"]) < cooldown_secs

def update_session(session_id: str, instruction: str, objects_info: list):
    """Record a spoken instruction."""
    state = get_session(session_id)
    
    state["history"].append({
        "instruction": instruction,
        "time": time.time(),
    })
    if len(state["history"]) > 5:
        state["history"].pop(0)
    
    state["last_spoken_time"] = time.time()
    state["last_object_types"] = set(o["type"] for o in objects_info)
    state["is_first"] = False
    sessions[session_id] = state