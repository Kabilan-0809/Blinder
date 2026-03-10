sessions = {}

def get_session(session_id: str) -> dict:
    if session_id not in sessions:
        sessions[session_id] = {
            "last_instruction": "",
            "last_scene_summary": "",
            "recent_obstacles": []
        }
    return sessions[session_id]

def update_session(session_id: str, instruction: str, scene_summary: str, objects_info: list):
    """
    Updates the session memory with the results of the current frame so the
    navigation engine can avoid repetitive instructions.
    """
    state = get_session(session_id)
    
    state["last_instruction"] = instruction
    state["last_scene_summary"] = scene_summary
    
    # Track the unique classes of objects recently encountered
    state["recent_obstacles"] = list(set([o["type"] for o in objects_info]))
    
    sessions[session_id] = state