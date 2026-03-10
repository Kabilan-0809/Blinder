import json

def build_scene(objects_info):
    """
    Constructs a rich, descriptive text representation of the scene
    to help the LLM generate more natural, less robotic language.
    Instead of just JSON, we provide interpreted spatial context.
    """
    if not objects_info:
        return "The camera sees nothing but open space. The path is completely clear."

    descriptions = []
    
    # Analyze the positional distribution
    positions = [obj["position"] for obj in objects_info]
    
    if "center" not in positions:
        free_space = "The center path ahead is CLEAR."
    elif "left" not in positions and "right" not in positions:
        free_space = "The sides are CLEAR, but something is in the center."
    elif "right" not in positions:
        free_space = "The right side is CLEAR."
    elif "left" not in positions:
        free_space = "The left side is CLEAR."
    else:
        free_space = "There is NO clear path. Objects are on the left, right, and center."

    # Build object specific descriptions
    for obj in objects_info:
        t = obj["type"]
        pos = obj["position"]
        dist = obj["distance"] # this is fill_ratio (0.0 to 1.0)
        
        # Interpret distance
        if dist > 0.4:
            dist_str = "VERY CLOSE"
        elif dist > 0.15:
            dist_str = "moderately close"
        else:
            dist_str = "far away"

        descriptions.append(f"- A {t} on the {pos} ({dist_str}).")

    scene_text = "SCENE ANALYSIS:\n"
    scene_text += free_space + "\n\n"
    scene_text += "DETECTED OBJECTS:\n"
    scene_text += "\n".join(descriptions)

    return scene_text
