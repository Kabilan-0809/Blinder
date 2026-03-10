import json

def build_scene(objects_info, terrain_hazards=None):
    """
    Stage 4: Scene Builder
    Merges YOLO bounding boxes with MiDaS depth estimates to create a structured
    spatial representation for the navigation engine.
    """
    if terrain_hazards is None:
        terrain_hazards = []
        
    scene_objects = []
    positions = set()
    
    for obj in objects_info:
        scene_objects.append({
            "type": obj["type"],
            "position": obj["position"],
            "distance": obj["distance"]
        })
        positions.add(obj["position"])
        
    # Determine general free space direction
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
        "objects": scene_objects,
        "terrain_hazards": terrain_hazards,
        "free_space": free_space
    })
