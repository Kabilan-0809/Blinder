# Redis-ready dictionary for state management
# In production, replace `sessions` with a Redis client connection
sessions = {}

def get_session(session_id):
    if session_id not in sessions:
        sessions[session_id] = {
            "last_instruction": None,
            "last_scene_summary": None,
            "recent_objects": []
        }
    return sessions[session_id]

def update_session(session_id, instruction, scene_summary, objects):
    state = get_session(session_id)
    state["last_instruction"] = instruction
    state["last_scene_summary"] = scene_summary
    state["recent_objects"] = objects
    sessions[session_id] = state