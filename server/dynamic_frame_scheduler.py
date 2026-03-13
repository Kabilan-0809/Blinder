"""
dynamic_frame_scheduler.py

Intelligent dynamic frame capture scheduler for the Blind AI navigation assistant.

PURPOSE
-------
Decide WHEN to send a camera frame to the multimodal LLM for scene reasoning.
Fixed-interval capture wastes tokens. This module targets 3–10 LLM calls/minute
(80%+ reduction vs fixed 1fps capture) while keeping the assistant safe and aware.

ENVIRONMENT ASSUMPTIONS
-----------------------
  • User walking speed  : ~1.2 m/s
  • Camera FoV range    : 10–25 m ahead
  • YOLO safety system  : runs on EVERY frame (<100ms), always-on
  • LLM scene reasoning : expensive (~1–3s), called only when needed

═══════════════════════════════════════════════════════════════════════
DECISION ALGORITHM  (pseudocode)
═══════════════════════════════════════════════════════════════════════

function should_capture_frame(scene_state, navigation_state, user_query):

    # Hard rate limit — never flood the LLM
    if time_since_last_llm < MIN_GAP_SEC:
        return False, "min_gap_throttle"

    # ── RULE 3: User Question ─────────────────────────────────────────
    if user_query is not None:
        return True, "rule3_user_question"

    # ── Session start / forced ────────────────────────────────────────
    if force:
        return True, "force"

    # ── RULE 2: New Environment / Decision Point ──────────────────────
    if scene_state.decision_point or scene_state.open_area or scene_state.building_entrance:
        return True, "rule2_new_environment"

    # ── RULE 5: Upcoming Navigation Turn ─────────────────────────────
    if navigation_state.next_turn_distance <= NAV_TURN_THRESHOLD_M:
        return True, "rule5_nav_turn"

    # ── RULE 4: Sign Detected by YOLO ────────────────────────────────
    if any sign-like class in scene_state.yolo_classes:
        return True, "rule4_sign_detected"

    # ── RULE 6: Sudden Obstacle Density ──────────────────────────────
    if len(scene_state.yolo_objects) >= CROWD_THRESHOLD:
        return True, "rule6_high_density"

    if scene_state.new_object_types_vs_previous:
        return True, "rule6_new_objects"

    # ── RULE 7: Memory Awareness ──────────────────────────────────────
    if scene_has_not_changed_significantly(scene_state, last_scene_memory):
        path_ttl = estimate_walk_time(last_known_clear_path_m) × 0.75
        if time_since_last_llm < path_ttl:
            return False, "rule7_scene_unchanged"

    # ── RULE 1: Long Clear Path ───────────────────────────────────────
    if scene_state.clear_path:
        path_m   = estimate_visible_path_length(depth_map or yolo_proxy)
        wait_sec = estimate_walk_time(path_m) × 0.90
        if time_since_last_llm < wait_sec:
            return False, "rule1_clear_path"

    # ── Keepalive ─────────────────────────────────────────────────────
    if time_since_last_llm >= KEEPALIVE_SEC:
        return True, "keepalive"

    return False, "stable"

═══════════════════════════════════════════════════════════════════════
"""

import time   # type: ignore
import math   # type: ignore
import logging  # type: ignore
import numpy as np  # type: ignore

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# TUNING CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

WALK_SPEED_MPS       = 1.2    # average user walking speed (m/s)
CAMERA_FOV_MAX_M     = 25.0   # max visible path the camera can see (m)

MIN_GAP_SEC          = 6.0    # hard minimum between LLM calls (anti-flood) — raised to avoid rate limits
KEEPALIVE_SEC        = 20.0   # maximum silence — always call LLM within this

MIN_CLEAR_M          = 10.0   # below this, treat as approaching decision point
NAV_TURN_THRESHOLD_M = 20.0   # fire if navigation turn is within this distance

CROWD_THRESHOLD      = 5      # ≥ N overlapping yolo objects = crowd density event

# YOLO class names that suggest readable signage
SIGN_CLASSES = {
    "stop sign", "sign", "billboard", "parking meter",
    "traffic sign", "exit sign", "banner", "board",
}

# Path TTL safety margin: use 90% of estimated walk time before re-checking
PATH_TTL_FACTOR      = 0.90


# ─────────────────────────────────────────────────────────────────────────────
# PATH LENGTH ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def estimate_visible_path_length(
    depth_map: "np.ndarray | None" = None,  # type: ignore
    yolo_clear_distance_m: float | None = None,
) -> float:  # type: ignore
    """
    Estimate usable clear path length in meters ahead of the user.

    Two strategies (first available wins):

    Strategy A — MiDaS depth map (preferred, more accurate):
      1. Extract the central vertical strip (middle 20% of frame width)
      2. Average depth values in that strip for each row, top→bottom
      3. Find the first row where depth jumps significantly (wall / obstacle)
      4. Map pixel row → real-world meters using the camera FoV model
      Returns: estimated meters to first obstacle

    Strategy B — YOLO bounding-box proxy (fallback, no depth model needed):
      Uses the pre-computed `yolo_clear_distance_m` from vision_safety.
      Returns: that value directly, capped to CAMERA_FOV_MAX_M

    Args:
        depth_map:             H×W float32 array from MiDaS (inverse depth, 0–1),
                               or None to use the YOLO proxy.
        yolo_clear_distance_m: Optional float from vision_safety.run_safety_check().

    Returns:
        Estimated clear path length in meters (float, 0.3–CAMERA_FOV_MAX_M).
    """
    # ── Strategy A: depth map ─────────────────────────────────────────
    if depth_map is not None and depth_map.size > 0:
        try:
            h, w = depth_map.shape[:2]
            # Central vertical strip (middle 20% of width)
            cx = w // 2
            half = max(1, w // 10)
            strip = depth_map[:, cx - half: cx + half]        # H x strip_w

            # Average inverse-depth per row
            row_depths = strip.mean(axis=1)                   # shape: (H,)

            # Normalise to [0, 1]
            d_min, d_max = row_depths.min(), row_depths.max()
            if d_max > d_min:
                row_norm = (row_depths - d_min) / (d_max - d_min)
            else:
                row_norm = row_depths

            # Find first row (top→bottom) with a sharp depth jump (obstacle)
            JUMP_THRESHOLD = 0.25
            obstacle_row = h  # default = no obstacle found
            for i in range(1, h):
                if (row_norm[i] - row_norm[i - 1]) > JUMP_THRESHOLD:
                    obstacle_row = i
                    break

            # Map row fraction → distance using linear FoV model
            # Row 0 = far end (CAMERA_FOV_MAX_M), row h = near (0.3 m)
            row_fraction = 1.0 - (obstacle_row / h)           # 1.0 = far end  # type: ignore
            path_m = max(0.3, row_fraction * CAMERA_FOV_MAX_M)
            logger.debug(f"[SCHED] depth path estimate: {path_m:.1f}m (obstacle_row={obstacle_row}/{h})")
            return float(min(path_m, CAMERA_FOV_MAX_M))  # type: ignore

        except Exception as e:
            logger.warning(f"[SCHED] depth map estimation failed: {e}")

    # ── Strategy B: YOLO bbox proxy ───────────────────────────────────
    if yolo_clear_distance_m is not None:
        return float(min(yolo_clear_distance_m, CAMERA_FOV_MAX_M))  # type: ignore

    # ── No data: assume moderate visibility ───────────────────────────
    return float(CAMERA_FOV_MAX_M / 2)  # type: ignore


def estimate_walk_time(path_length_m: float) -> float:  # type: ignore
    """
    Estimate seconds until the user reaches a given distance ahead.

    Args:
        path_length_m: Distance in meters.

    Returns:
        Time in seconds at WALK_SPEED_MPS.

    Example:
        estimate_walk_time(20.0) → 16.67 seconds
    """
    if path_length_m <= 0:
        return 0.0
    return float(path_length_m / WALK_SPEED_MPS)  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# STATEFUL SCHEDULER (one instance per session)
# ─────────────────────────────────────────────────────────────────────────────

class DynamicFrameScheduler:
    """
    Per-session scheduler instance.

    Usage in websocket_server.py:
        sched = DynamicFrameScheduler()  # once per connection

        # Each camera frame:
        result = sched.should_capture_frame(
            scene_state     = yolo_result,
            navigation_state= nav_state,
            user_query      = pending_question,   # str or None
            session_memory  = mem,
            depth_map       = depth_arr,          # or None
        )
        if result["capture"]:
            scene = analyze_frame(...)
    """

    def __init__(self):  # type: ignore
        self._last_llm_time     : float = 0.0
        self._prev_object_types : set   = set()
        self._prev_scene_hash   : int   = 0
        self._force_next        : bool  = True   # capture on very first frame

    # ── Public API ────────────────────────────────────────────────────

    def should_capture_frame(
        self,
        *,
        scene_state     : dict,          # from vision_safety.run_safety_check()
        navigation_state: dict,          # from navigation_engine / nav_progress
        user_query      : str | None = None,
        session_memory  : dict,
        depth_map       = None,          # np.ndarray | None  (MiDaS output)
        force           : bool = False,
    ) -> dict:  # type: ignore
        """
        Main entry point.  Call for every camera frame.

        Args:
            scene_state:      YOLO result dict with keys:
                                clear_path (bool)
                                estimated_clear_distance_m (float | None)
                                object_types (set[str])
                                objects (list of detected obj dicts)
            navigation_state: Dict with optional keys:
                                next_turn_distance_m (float)
                                at_decision_point (bool)
            user_query:       Pending question string (Rule 3), or None
            session_memory:   Full session memory dict (Rule 7)
            depth_map:        MiDaS depth array or None (Rule 1 precision)
            force:            Hard override — always capture

        Returns:
            {
                "capture": bool,
                "rule":    str,     # which rule triggered / suppressed
                "delay_s": float,   # estimated seconds until next needed capture
            }
        """
        now     = time.time()
        elapsed = now - self._last_llm_time

        # ── Hard minimum gap (anti-flood) ─────────────────────────────
        if elapsed < MIN_GAP_SEC:
            return self._skip("min_gap_throttle", 0.0)

        # ── Session init / hard force ─────────────────────────────────
        if force or self._force_next:
            self._force_next = False
            return self._capture(now, "force", 0.0)

        # ════════════════════════════════════════════════════════════════
        # TRIGGER RULES  (capture = True)
        # ════════════════════════════════════════════════════════════════

        # ── Rule 3: User question → immediate ────────────────────────
        if user_query:
            logger.info(f"[SCHED] Rule 3 — user query: '{user_query}'")
            return self._capture(now, "rule3_user_question", 0.0)

        # ── Rule 2: New intersection / environment change ─────────────
        if navigation_state.get("at_decision_point"):
            logger.info("[SCHED] Rule 2 — navigation decision point")
            return self._capture(now, "rule2_nav_decision_point", 0.0)

        env = session_memory.get("environment_memory", {})
        if session_memory.get("_decision_point_flagged"):
            session_memory["_decision_point_flagged"] = False
            logger.info("[SCHED] Rule 2 — visual decision point flagged")
            return self._capture(now, "rule2_visual_decision_point", 0.0)

        # ── Rule 5: Navigation turn within 20m ───────────────────────
        next_turn = navigation_state.get("next_turn_distance_m")
        if next_turn is not None and next_turn <= NAV_TURN_THRESHOLD_M:
            logger.info(f"[SCHED] Rule 5 — nav turn in {next_turn:.0f}m")
            return self._capture(now, f"rule5_nav_turn_{next_turn:.0f}m", 0.0)

        # ── Rule 4: Sign/label detected by YOLO ──────────────────────
        obj_types = scene_state.get("object_types", set())
        sign_hits = obj_types & SIGN_CLASSES
        if sign_hits:
            logger.info(f"[SCHED] Rule 4 — sign detected: {sign_hits}")
            return self._capture(now, f"rule4_sign:{','.join(sign_hits)}", 0.0)

        # ── Rule 6: Sudden obstacle density (crowd/cluster) ───────────
        n_objects = len(scene_state.get("objects", []))
        if n_objects >= CROWD_THRESHOLD:
            logger.info(f"[SCHED] Rule 6 — high density: {n_objects} objects")
            return self._capture(now, f"rule6_density:{n_objects}", 0.0)

        # Rule 6b: Only trigger on HIGH-VALUE new object classes, not routine ones
        # (laptop, keyboard, cell phone, cup etc. are furniture — not navigation hazards)
        HIGH_VALUE_CLASSES = {
            'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
            'dog', 'horse', 'cow', 'sheep',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
            'bench', 'chair', 'dining table',
        }
        new_types = (obj_types - self._prev_object_types) & HIGH_VALUE_CLASSES
        self._prev_object_types = obj_types
        if new_types:
            logger.info(f"[SCHED] Rule 6 — new significant objects: {new_types}")
            return self._capture(now, f"rule6_new_objects:{','.join(new_types)}", 0.0)

        # ════════════════════════════════════════════════════════════════
        # SUPPRESSION RULES  (capture = False)
        # ════════════════════════════════════════════════════════════════

        # ── Rule 1: Long clear path → estimate safe delay ─────────────
        clear_m = estimate_visible_path_length(
            depth_map            = depth_map,
            yolo_clear_distance_m= scene_state.get("estimated_clear_distance_m"),
        )

        if scene_state.get("clear_path", False) and clear_m >= MIN_CLEAR_M:
            walk_time = estimate_walk_time(clear_m)
            delay     = walk_time * PATH_TTL_FACTOR           # 90% safety margin
            if elapsed < delay:
                remaining = delay - elapsed
                logger.debug(
                    f"[SCHED] Rule 1 — clear {clear_m:.1f}m "
                    f"walk={walk_time:.1f}s skip={remaining:.1f}s remaining"
                )
                return self._skip(f"rule1_clear_path_{clear_m:.0f}m", remaining)

        # ── Rule 7: Memory awareness — scene hasn't changed significantly ──
        scene_hash = self._hash_scene(scene_state)
        if scene_hash == self._prev_scene_hash:
            last_clear = env.get("last_clear_path_meters") or MIN_CLEAR_M
            walk_time  = estimate_walk_time(last_clear)
            delay      = walk_time * PATH_TTL_FACTOR
            if elapsed < delay:
                remaining = delay - elapsed
                logger.debug(f"[SCHED] Rule 7 — scene unchanged, skip {remaining:.1f}s")
                return self._skip(f"rule7_scene_unchanged", remaining)

        self._prev_scene_hash = scene_hash

        # ── Keepalive — at least one call per KEEPALIVE_SEC ───────────
        if elapsed >= KEEPALIVE_SEC:
            logger.info(f"[SCHED] Keepalive after {elapsed:.1f}s")
            return self._capture(now, "keepalive", 0.0)

        # ── Default stable window ─────────────────────────────────────
        if elapsed >= 10.0:
            return self._capture(now, "time_elapsed_10s", 0.0)

        return self._skip(f"stable_{elapsed:.1f}s", KEEPALIVE_SEC - elapsed)

    # ── Backwards-compatible alias used by websocket_server ──────────

    def should_send_to_llm(  # type: ignore
        self,
        *,
        pending_question: str | None,
        yolo_result: dict,
        session_memory: dict,
        force: bool = False,
        depth_map=None,
    ) -> dict:  # type: ignore
        """Legacy alias — delegates to should_capture_frame."""
        result = self.should_capture_frame(
            scene_state      = yolo_result,
            navigation_state = {},
            user_query       = pending_question,
            session_memory   = session_memory,
            depth_map        = depth_map,
            force            = force,
        )
        return {"should_send": result["capture"], "reason": result["rule"]}

    def reset(self):  # type: ignore
        """Call when a new navigation session/goal starts."""
        self._last_llm_time     = 0.0
        self._prev_object_types = set()
        self._prev_scene_hash   = 0
        self._force_next        = True
        logger.info("[SCHED] Reset for new session")

    # ── Internal helpers ──────────────────────────────────────────────

    def _capture(self, now: float, rule: str, delay_s: float) -> dict:  # type: ignore
        self._last_llm_time = now
        logger.debug(f"[SCHED] ✅ CAPTURE ({rule})")
        return {"capture": True, "rule": rule, "delay_s": delay_s}

    @staticmethod
    def _skip(rule: str, delay_s: float) -> dict:  # type: ignore
        logger.debug(f"[SCHED] ⏭️ SKIP ({rule}) next≈{delay_s:.1f}s")
        return {"capture": False, "rule": rule, "delay_s": delay_s}

    @staticmethod
    def _hash_scene(scene_state: dict) -> int:  # type: ignore
        """
        Fast scene fingerprint for Rule 7 (memory awareness).
        Hashes: object type set + clear_path bool + density bucket.
        """
        obj_types  = frozenset(scene_state.get("object_types", []))
        clear      = scene_state.get("clear_path", True)
        n_objects  = len(scene_state.get("objects", []))
        density    = n_objects // 2    # bucket to 0/1/2/3+ to avoid hash churn
        return hash((obj_types, clear, density))
