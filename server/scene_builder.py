import json

def build_scene(objects_info):
    """
    Constructs a structured JSON representation of the immediate scene
    to provide to the instruction engine / LLM.
    """
    
    # Analyze the positional distribution to hint at free space to the LLM
    positions = [obj["position"] for obj in objects_info]
    
    if "center" not in positions:
        free_space = "center"
    elif "left" not in positions and "right" not in positions:
        free_space = "sides"
    elif "right" not in positions:
        free_space = "right"
    elif "left" not in positions:
        free_space = "left"
    else:
        free_space = "none"

    return json.dumps({
        "objects": objects_info,
        "free_space": free_space
    })
