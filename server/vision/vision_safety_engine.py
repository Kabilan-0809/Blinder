"""
vision_safety.py

Fast, always-on obstacle detection using YOLOv8n.
Target latency: <100ms per frame.

Returns structured result:
  {
    "alert":                  str | None,       # spoken warning text, or None if clear
    "objects":                [...],             # list of detected objects with positions
    "clear_path":             bool,             # True if nothing dangerously close
    "estimated_clear_distance_m": float | None, # proxy distance using bbox area heuristic
    "object_types":           set               # set of label strings (for change detection)
  }

Distance proxy:
  Instead of running the full MiDaS depth model on every safety frame
  (too slow), we estimate proximity from bounding box height fraction:
    proximity_score = bbox_height / frame_height
  Objects with score > DANGER_THRESHOLD are considered dangerously close (~<1m).
"""

import cv2  # type: ignore
import numpy as np  # type: ignore
import logging  # type: ignore
from ultralytics import YOLO  # type: ignore

logger = logging.getLogger(__name__)

# Load once at module import
_yolo = YOLO("yolov8n.pt")

# Proximity thresholds (fraction of frame height)
DANGER_THRESHOLD  = 0.55    # bbox occupies >55% height → very close (<~1m)
CAUTION_THRESHOLD = 0.35    # bbox occupies >35% height → approaching (~1-2m)

# Frame size for fast inference
INFER_W, INFER_H = 320, 240


def run_safety_check(jpeg_bytes: bytes, safety_enabled: bool = True) -> dict:  # type: ignore
    """
    Run YOLOv8n on a JPEG frame and return structured safety result.
    If safety_enabled is False, returns clear path without generating UI alerts.
    """
    frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return _clear_result()

    frame = cv2.resize(frame, (INFER_W, INFER_H))
    h, w = frame.shape[:2]

    try:
        results = _yolo(frame, conf=0.30, verbose=False)
    except Exception as e:
        logger.warning(f"[SAFETY] YOLO inference error: {e}")
        return _clear_result()

    objects = []
    object_types = set()
    closest_score = 0.0

    for r in results:
        for box in r.boxes:  # type: ignore
            label = _yolo.names[int(box.cls)]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])

            # Proximity score: relative height of bounding box
            bbox_h = y2 - y1
            proximity_score = bbox_h / h

            # Horizontal position for human-readable alerts
            cx = (x1 + x2) / 2
            pos = "left" if cx < w * 0.33 else ("right" if cx > w * 0.67 else "ahead of you")

            objects.append({
                "type": label,
                "position": pos,
                "proximity_score": round(float(proximity_score), 2),  # type: ignore
                "conf": round(conf, 2),
                "bbox": [round(x1), round(y1), round(x2), round(y2)],
            })
            object_types.add(label)
            closest_score = max(closest_score, float(proximity_score))

    # No objects detected
    if not objects:
        return {
            **_clear_result(),
            "object_types": object_types,
        }

    # Estimate a rough clear distance from proximity score
    # At proximity_score=1.0 → ~0.3m; at 0.1 → ~10m (inverse linear heuristic)
    estimated_clear_m = max(0.3, min(15.0, 1.5 / closest_score)) if closest_score > 0.0 else 15.0  # type: ignore

    # ── Danger check ────────────────────────────────────────
    danger_objects = [o for o in objects if o["proximity_score"] >= DANGER_THRESHOLD]
    
    # If the user toggled safety off via UI, suppress the alert
    if not safety_enabled:
        if danger_objects:
            logger.debug("[SAFETY] Danger detected but supressed by UI toggle.")
        danger_objects = []

    if danger_objects:
        # Report the single closest threat
        threat = max(danger_objects, key=lambda o: o["proximity_score"])
        label = threat["type"].capitalize()
        pos = threat["position"]
        alert = f"Stop. {label} {pos}."
        logger.warning(f"[SAFETY] 🚨 {alert}")
        return {
            "alert": alert,
            "objects": objects,
            "clear_path": False,
            "estimated_clear_distance_m": round(estimated_clear_m, 1),  # type: ignore
            "object_types": object_types,
        }

    return {
        "alert": None,
        "objects": objects,
        "clear_path": True,
        "estimated_clear_distance_m": round(estimated_clear_m, 1),  # type: ignore
        "object_types": object_types,
    }


def _clear_result() -> dict:  # type: ignore
    return {
        "alert": None,
        "objects": [],
        "clear_path": True,
        "estimated_clear_distance_m": 15.0,
        "object_types": set(),
    }
