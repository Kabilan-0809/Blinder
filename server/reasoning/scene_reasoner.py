"""
scene_reasoning.py

Multimodal LLM scene analysis using Gemini Flash.
Called only when the DynamicFrameScheduler decides a frame is worth processing.

Returns structured JSON:
  {
    "description":           str,           # short narrative scene description
    "signs_seen":            [str, ...],    # any visible signs or labels
    "decision_point":        bool,          # intersection, exit, staircase, etc.
    "crowd_density":         "low|medium|high|none",
    "corridor_direction":    "forward|left|right|unknown",
    "estimated_clear_path_m": float | None  # rough estimate to nearest decision point
  }
"""

import os  # type: ignore
import json  # type: ignore
import logging  # type: ignore
from google import genai  # type: ignore
from dotenv import load_dotenv  # type: ignore

load_dotenv()
logger = logging.getLogger(__name__)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

SCENE_SYSTEM_PROMPT = """You are the visual awareness system for Iris, an AI companion for a blind person.
Analyze the camera image and return a JSON object with accurate, navigation-relevant information.

Return ONLY a JSON object in exactly this format:
{
  "description": "One to three natural, friendly sentences about what you see. Focus on useful details: paths, obstacles, objects, people, signs. Speak as if describing the scene to a friend.",
  "signs_seen": ["list of exact text from any signs or labels visible"],
  "decision_point": true or false,
  "crowd_density": "none" or "low" or "medium" or "high",
  "corridor_direction": "forward" or "left" or "right" or "unknown",
  "estimated_clear_path_m": null or a number (e.g. 15.0),
  "mood_hint": "calm" or "busy" or "tense" or "peaceful" or "alert"
}

Rules:
- description: be natural and warm. Say "There's a chair on your left" not "Chair detected at position left."
- mood_hint: how would a person FEEL in this environment? busy street = "busy", park = "peaceful", construction = "tense"
- decision_point: true if you see intersection, staircase, escalator, elevator, door, exit, or fork.
- estimated_clear_path_m: estimate visual distance to nearest obstacle or decision point.
- If the image is too blurry or dark, set description to "UNCLEAR" and all other fields to defaults.
- Do NOT include any text outside the JSON object.
"""


def analyze_frame(jpeg_bytes: bytes, goal: str = "", active_tasks: list | None = None) -> dict:  # type: ignore
    """
    Sends a JPEG frame to Gemini for structured scene analysis.

    Args:
        jpeg_bytes:    Raw JPEG image bytes
        goal:          Current navigation goal string (adds context to prompt)
        active_tasks:  List of long-running task strings (e.g. ["detect poles"])

    Returns structured scene dict. Falls back to a minimal dict on error.
    """
    try:
        from google.genai import types as gtypes  # type: ignore

        # Build context prefix
        context_parts = []
        if goal:
            context_parts.append(f"Navigation goal: {goal}.")
        if active_tasks:
            context_parts.append(f"Monitor for: {', '.join(active_tasks)}.")
        context_parts.append("Analyze the image and return JSON as instructed.")
        prompt = " ".join(context_parts)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=gtypes.Content(
                role="user",
                parts=[
                    gtypes.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                    gtypes.Part(text=prompt),
                ]
            ),
            config=gtypes.GenerateContentConfig(
                system_instruction=SCENE_SYSTEM_PROMPT,
                temperature=0.1,
                max_output_tokens=1024,
                thinking_config=gtypes.ThinkingConfig(thinking_budget=0),
            ),
        )

        raw = (response.text or "").strip()  # type: ignore
        if not raw:
            logger.warning("[SCENE] Empty response from Gemini")
            return _fallback_scene()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            parts = raw.split("```")
            if len(parts) >= 2:
                raw = parts[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()  # type: ignore
        scene = json.loads(raw)
        
        desc = str(scene.get("description", ""))
        logger.info(f"[SCENE] {desc[:80]}")  # type: ignore
        return scene  # type: ignore

    except json.JSONDecodeError as e:
        err_raw = raw[:100] if 'raw' in locals() else 'N/A'  # type: ignore
        logger.warning(f"[SCENE] JSON parse error: {e} — raw: {err_raw}")
        return _fallback_scene()
    except Exception as e:
        logger.error(f"[SCENE] Analysis error: {e}")
        return _fallback_scene()


def _fallback_scene() -> dict:  # type: ignore
    return {
        "description": "Unable to analyze scene.",
        "signs_seen": [],
        "decision_point": False,
        "crowd_density": "unknown",
        "corridor_direction": "unknown",
        "estimated_clear_path_m": None,
    }


def build_scene_insight(scene: dict, long_running_tasks: list) -> str | None:  # type: ignore
    """
    Convert the structured scene dict into a concise spoken insight.
    Returns None if there's nothing useful to say.
    """
    if not scene or scene.get("description") in ("UNCLEAR", "Unable to analyze scene."):
        return None  # type: ignore

    parts = []

    desc = scene.get("description", "")
    if desc and desc != "UNCLEAR":
        parts.append(desc)

    # Report any visible signs
    signs = scene.get("signs_seen", [])
    if signs:
        sign_text = ", ".join(signs[:3])
        parts.append(f"I can see: {sign_text}.")

    # Decision point highlight
    if scene.get("decision_point"):
        parts.append("There's a junction or exit ahead.")

    if not parts:
        return None  # type: ignore

    return " ".join(parts)  # type: ignore
