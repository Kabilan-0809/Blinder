import time

# Redis-ready dictionary for state management
sessions = {}

def get_session(session_id):
    if session_id not in sessions:
        sessions[session_id] = {
            "history": [] # Will store up to 5 dicts of {timestamp, instruction, scene}
        }
    return sessions[session_id]

def update_session(session_id, instruction, scene_summary, objects):
    state = get_session(session_id)
    
    # Append the latest event
    state["history"].append({
        "time": time.time(),
        "instruction": instruction,
        "scene": scene_summary
    })
    
    # Enforce rolling window of the last 5 interactions to prevent context overflow
    if len(state["history"]) > 5:
        state["history"].pop(0)
        
    sessions[session_id] = state