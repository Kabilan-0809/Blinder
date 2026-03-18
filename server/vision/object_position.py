"""
object_position.py

Determines the spatial position of detected objects relative to the user.

Uses the horizontal position of the bounding box center within the frame
to classify objects into directional zones:

    ┌──────────┬──────────┬──────────┐
    │   LEFT   │  CENTER  │  RIGHT   │
    │  0-33%   │  33-67%  │  67-100% │
    └──────────┴──────────┴──────────┘

For vertical position (close/far):
    ┌────────────────────────────────┐
    │         FAR  (top 40%)         │
    │       MEDIUM (middle 30%)      │
    │        NEAR  (bottom 30%)      │
    └────────────────────────────────┘

Usage:
    from vision.object_position import get_object_position
    pos = get_object_position(bbox, frame_width=640, frame_height=480)
    # → {"horizontal": "left", "vertical": "near", "label": "on your left, close by"}
"""

import logging  # type: ignore

logger = logging.getLogger("vision.position")


# ── Zone thresholds ──────────────────────────────────────────────────────────

HORIZONTAL_ZONES = {
    "left":   (0.0, 0.33),
    "center": (0.33, 0.67),
    "right":  (0.67, 1.0),
}

VERTICAL_ZONES = {
    "far":    (0.0, 0.40),
    "medium": (0.40, 0.70),
    "near":   (0.70, 1.0),
}

# Human-readable labels combining horizontal + vertical
POSITION_LABELS = {
    ("left",   "far"):    "far to your left",
    ("left",   "medium"): "to your left",
    ("left",   "near"):   "close on your left",
    ("center", "far"):    "ahead in the distance",
    ("center", "medium"): "ahead of you",
    ("center", "near"):   "directly ahead, close",
    ("right",  "far"):    "far to your right",
    ("right",  "medium"): "to your right",
    ("right",  "near"):   "close on your right",
}


def get_object_position(
    bbox: list | tuple,
    frame_width: int,
    frame_height: int,
) -> dict:  # type: ignore
    """
    Compute the spatial position of an object from its bounding box.

    Args:
        bbox:         [x1, y1, x2, y2] bounding box coordinates
        frame_width:  Width of the camera frame in pixels
        frame_height: Height of the camera frame in pixels

    Returns:
        {
            "horizontal": "left" | "center" | "right",
            "vertical":   "far" | "medium" | "near",
            "label":      str,   # human-readable position
            "center_x":   float, # normalized center x (0-1)
            "center_y":   float, # normalized center y (0-1)
        }
    """
    x1, y1, x2, y2 = bbox
    center_x = ((x1 + x2) / 2) / frame_width
    center_y = ((y1 + y2) / 2) / frame_height

    # Clamp to [0, 1]
    center_x = max(0.0, min(1.0, center_x))
    center_y = max(0.0, min(1.0, center_y))

    horizontal = _classify(center_x, HORIZONTAL_ZONES)
    vertical = _classify(center_y, VERTICAL_ZONES)
    label = POSITION_LABELS.get((horizontal, vertical), "nearby")

    return {
        "horizontal": horizontal,
        "vertical": vertical,
        "label": label,
        "center_x": round(center_x, 3),
        "center_y": round(center_y, 3),
    }


def get_position_label(
    center_x: float,
    frame_width: int,
) -> str:  # type: ignore
    """
    Quick horizontal-only position label (left/center/right).

    Args:
        center_x:    Horizontal center of bounding box in pixels
        frame_width: Frame width in pixels

    Returns:
        "left", "ahead of you", or "right"
    """
    ratio = center_x / frame_width
    if ratio < 0.33:
        return "left"
    elif ratio > 0.67:
        return "right"
    return "ahead of you"


def _classify(value: float, zones: dict) -> str:  # type: ignore
    """Classify a normalized value into a named zone."""
    for name, (lo, hi) in zones.items():
        if lo <= value < hi:
            return name  # type: ignore
    # Fall through to last zone
    return list(zones.keys())[-1]  # type: ignore
