"""
instruction_fusion.py

Merges outputs from all subsystems into a single final instruction.

Priority order (lower number = higher priority):
  1  SAFETY   — YOLO obstacle alert  (always spoken, never suppressed)
  2  NAV      — GPS turn instruction
  3  SCENE    — LLM scene insight / sign reading
  4  TASK     — Long-running task report (e.g. "no poles detected ahead")
  5  CONV     — Conversational response / guidance

Rules:
  - Safety alerts always fire immediately, regardless of recency
  - Only ONE non-safety instruction fires per cycle
  - A minimum gap of MIN_SPEAK_GAP_SEC seconds is enforced between non-safety outputs
    to avoid overwhelming the user with rapid-fire speech
"""

import time  # type: ignore
import logging  # type: ignore

logger = logging.getLogger(__name__)

MIN_SPEAK_GAP_SEC = 3.0     # min seconds between non-safety spoken outputs


def fuse(
    *,
    safety_alert: str | None = None,
    nav_instruction: str | None = None,
    scene_insight: str | None = None,
    task_report: str | None = None,
    conv_response: str | None = None,
    last_spoken_time: float = 0.0,
) -> dict:  # type: ignore
    """
    Stateless fusion function. Returns the highest-priority non-None instruction.

    Returns:
        {
            "text": str | None,
            "priority": int,           # 1–5, matching explanation above
            "source": str,             # "safety"|"nav"|"scene"|"task"|"conv"
            "should_speak": bool       # False if too soon after last utterance
        }
    """
    now = time.time()

    # ── Priority 1: Safety — always speak regardless of timing ──
    if safety_alert:
        logger.info(f"[FUSION] 🚨 SAFETY [{safety_alert}]")
        return {
            "text": safety_alert,
            "priority": 1,
            "source": "safety",
            "should_speak": True,
        }

    # ── Respect minimum gap for all non-safety outputs ──────────
    time_since_last = now - last_spoken_time
    if time_since_last < MIN_SPEAK_GAP_SEC:
        return {
            "text": None,
            "priority": 0,
            "source": "throttled",
            "should_speak": False,
        }

    # ── Priority 2: Navigation turn instruction ─────────────────
    if nav_instruction:
        logger.info(f"[FUSION] 🗺️ NAV [{nav_instruction}]")
        return {
            "text": nav_instruction,
            "priority": 2,
            "source": "nav",
            "should_speak": True,
        }

    # ── Priority 3: Scene insight from LLM ─────────────────────
    if scene_insight:
        logger.info(f"[FUSION] 👁️ SCENE [{scene_insight[:60]}...]")
        return {
            "text": scene_insight,
            "priority": 3,
            "source": "scene",
            "should_speak": True,
        }

    # ── Priority 4: Long-running task report ───────────────────
    if task_report:
        logger.info(f"[FUSION] ✅ TASK [{task_report}]")
        return {
            "text": task_report,
            "priority": 4,
            "source": "task",
            "should_speak": True,
        }

    # ── Priority 5: Conversational guidance ────────────────────
    if conv_response:
        logger.info(f"[FUSION] 💬 CONV [{conv_response[:60]}...]")
        return {
            "text": conv_response,
            "priority": 5,
            "source": "conv",
            "should_speak": True,
        }

    return {
        "text": None,
        "priority": 0,
        "source": "none",
        "should_speak": False,
    }
