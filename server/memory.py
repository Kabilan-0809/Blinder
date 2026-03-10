import time
import hashlib
import json

# Redis-ready dictionary. Each session_id maps to state.
sessions = {}

def get_session(session_id: str) -> dict:
    if session_id not in sessions:
        sessions[session_id] = {
            "history": [],          # Rolling list of last 5 {instruction, scene, time}
            "last_scene_hash": "",  # Hash of last spoken scene (for dedup)
            "last_spoken_time": 0.0, # Timestamp of the last spoken instruction
        }
    return sessions[session_id]

def compute_scene_hash(scene_json: str) -> str:
    """A quick hash to detect if the scene has meaningfully changed."""
    return hashlib.md5(scene_json.encode()).hexdigest()

def should_skip(session_id: str, scene_json: str, min_cooldown_secs: float = 6.0) -> bool:
    """
    Returns True if we should NOT process the LLM for this frame.
    We skip if: the scene hash is identical AND we're within the cooldown window.
    """
    state = get_session(session_id)
    new_hash = compute_scene_hash(scene_json)
    time_since_last = time.time() - state["last_spoken_time"]
    
    if new_hash == state["last_scene_hash"] and time_since_last < min_cooldown_secs:
        return True  # Skip — scene hasn't changed and we just spoke
    return False

def update_session(session_id: str, instruction: str, scene_json: str):
    state = get_session(session_id)
    state["history"].append({
        "instruction": instruction,
        "scene": scene_json,
        "time": time.time(),
    })
    # Rolling window — keep only last 5
    if len(state["history"]) > 5:
        state["history"].pop(0)
    
    state["last_scene_hash"] = compute_scene_hash(scene_json)
    state["last_spoken_time"] = time.time()
    sessions[session_id] = state