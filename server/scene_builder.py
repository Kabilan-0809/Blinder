import json

def build_scene(objects_info, terrain_hazards=None, path_recommendation=None):
    """
    Stage 4: Scene Builder
    Merges YOLO bounding boxes with MiDaS depth estimates to create a structured
    spatial representation for the navigation engine.
    """
    if terrain_hazards is None:
        terrain_hazards = []
    if path_recommendation is None:
        path_recommendation = {"free_space": "none", "confidence": 0.0}
        
    scene_objects = []
    
    for obj in objects_info:
        scene_objects.append({
            "type": obj["type"],
            "position": obj["position"],
            "distance": obj["distance"]
        })
        
    return json.dumps({
        "objects": scene_objects,
        "terrain_hazards": terrain_hazards,
        "path_recommendation": path_recommendation
    })
