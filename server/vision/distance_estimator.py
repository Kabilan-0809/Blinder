"""
distance_estimator.py

Estimates real-world distance to detected objects using two strategies:

Strategy 1 — Focal-Length Based (when camera intrinsics are known):
    distance = (focal_length × real_height) / bbox_height_pixels

Strategy 2 — Bounding Box Heuristic (default fallback):
    Uses the proportion of frame height occupied by the bounding box as a
    proximity proxy. Larger bbox = closer object.

    proximity_score = bbox_height / frame_height
    estimated_meters ≈ 1.5 / proximity_score  (capped 0.3m–15m)

Reference real-world heights (meters) for common objects:
    person:       1.7
    car:          1.5
    chair:        0.9
    bicycle:      1.0
    dog:          0.5
    stop sign:    2.1
    traffic light: 0.6

Usage:
    from vision.distance_estimator import estimate_distance, estimate_from_bbox
    dist = estimate_distance("person", bbox_height_px=200, frame_height=480)
    dist = estimate_from_bbox(bbox_height_px=200, frame_height=480)
"""

import logging  # type: ignore

logger = logging.getLogger("vision.distance")


# ── Known real-world heights (meters) for focal-length estimation ────────────

REAL_HEIGHTS_M = {
    "person":         1.70,
    "bicycle":        1.00,
    "car":            1.50,
    "motorcycle":     1.10,
    "bus":            3.00,
    "truck":          3.50,
    "dog":            0.50,
    "cat":            0.30,
    "chair":          0.90,
    "dining table":   0.75,
    "bench":          0.85,
    "fire hydrant":   0.75,
    "stop sign":      2.10,
    "parking meter":  1.20,
    "traffic light":  0.60,
    "backpack":       0.50,
    "umbrella":       1.00,
    "suitcase":       0.70,
    "potted plant":   0.60,
    "bottle":         0.30,
    "cup":            0.15,
    "laptop":         0.03,
    "cell phone":     0.15,
    "tv":             0.60,
    "book":           0.25,
}

# Default camera focal length (pixels) — typical for 480p webcam, ~60° FoV
DEFAULT_FOCAL_LENGTH_PX = 500.0

# Distance clamping
MIN_DISTANCE_M = 0.3
MAX_DISTANCE_M = 15.0


def estimate_distance(
    object_type: str,
    bbox_height_px: float,
    frame_height: int = 480,
    focal_length_px: float = DEFAULT_FOCAL_LENGTH_PX,
) -> dict:  # type: ignore
    """
    Estimate distance using focal-length method when object type is known.
    Falls back to bbox heuristic when object type has no known height.

    Args:
        object_type:     YOLO class name (e.g. "person", "chair")
        bbox_height_px:  Height of bounding box in pixels
        frame_height:    Camera frame height in pixels
        focal_length_px: Camera focal length in pixels

    Returns:
        {
            "distance_m":  float,          # estimated meters
            "method":      str,            # "focal_length" or "bbox_heuristic"
            "confidence":  str,            # "high" | "medium" | "low"
            "proximity":   str,            # "danger" | "caution" | "safe"
        }
    """
    if bbox_height_px <= 0:
        return _make_result(MAX_DISTANCE_M, "error", "low", "safe")

    real_h = REAL_HEIGHTS_M.get(object_type.lower())

    if real_h is not None:
        # ── Strategy 1: Focal-length based ────────────────────────────
        distance_m = (focal_length_px * real_h) / bbox_height_px
        distance_m = max(MIN_DISTANCE_M, min(MAX_DISTANCE_M, distance_m))
        method = "focal_length"
        confidence = "high" if real_h > 0.5 else "medium"
    else:
        # ── Strategy 2: Bbox heuristic fallback ──────────────────────
        result = estimate_from_bbox(bbox_height_px, frame_height)
        return result

    proximity = _classify_proximity(distance_m)
    logger.debug(
        f"[DIST] {object_type}: {distance_m:.1f}m ({method}, {confidence})"
    )
    return _make_result(distance_m, method, confidence, proximity)


def estimate_from_bbox(
    bbox_height_px: float,
    frame_height: int = 480,
) -> dict:  # type: ignore
    """
    Quick distance estimate using only bounding box height ratio.

    This is the default fallback when object type is unknown or has no
    reference height. Less accurate but always available.

    Args:
        bbox_height_px: Height of bounding box in pixels
        frame_height:   Camera frame height in pixels

    Returns:
        Same dict format as estimate_distance()
    """
    if bbox_height_px <= 0 or frame_height <= 0:
        return _make_result(MAX_DISTANCE_M, "error", "low", "safe")

    proximity_score = bbox_height_px / frame_height
    distance_m = max(MIN_DISTANCE_M, min(MAX_DISTANCE_M, 1.5 / proximity_score))

    proximity = _classify_proximity(distance_m)
    return _make_result(
        round(distance_m, 1),
        "bbox_heuristic",
        "medium" if proximity_score > 0.2 else "low",
        proximity,
    )


def _classify_proximity(distance_m: float) -> str:  # type: ignore
    """Classify distance into danger/caution/safe zones."""
    if distance_m <= 1.0:
        return "danger"
    elif distance_m <= 2.5:
        return "caution"
    return "safe"


def _make_result(
    distance_m: float, method: str, confidence: str, proximity: str
) -> dict:  # type: ignore
    return {
        "distance_m": round(distance_m, 1),
        "method": method,
        "confidence": confidence,
        "proximity": proximity,
    }
