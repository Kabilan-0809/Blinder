sessions = {}

def create_session(session_id):
    sessions[session_id] = {
        "last_instruction": None,
        "history": []
    }

def get_last_instruction(session_id):
    return sessions.get(session_id, {}).get("last_instruction")

def update_instruction(session_id, instruction):
    sessions[session_id]["last_instruction"] = instruction
    sessions[session_id]["history"].append(instruction)