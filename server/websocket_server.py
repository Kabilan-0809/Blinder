"""
websocket_server.py

Thin coordinator for the Blind AI navigation assistant.
Delegates all logic to specialized subsystem modules.

WebSocket message protocol (client → server):
  { "type": "audio",  "data": "<base64 webm>" }         STT → task extraction
  { "type": "frame",  "data": "<base64 JPEG>" }          safety + optional scene LLM
  { "type": "gps",    "lat": float, "lng": float }       GPS nav step check
  { "type": "setting","danger_detection": bool }         UI toggle
  { "type": "route",  "start": {...}, "dest": "..." }    Load a Google Maps route

WebSocket message protocol (server → client):
  { "type": "response",   "text": "...", "source": "...", "priority": int }
  { "type": "transcript", "text": "..." }
  { "type": "safety",     "text": "..." }
  { "type": "error",      "text": "..." }
"""

import base64  # type: ignore
import json  # type: ignore
import time  # type: ignore
import asyncio  # type: ignore
import logging  # type: ignore

from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # type: ignore
from fastapi.responses import HTMLResponse  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore

# ── Subsystem imports ──────────────────────────────────────────────────────────
from speech_to_text import transcribe_audio  # type: ignore
import memory_manager as mem_mgr  # type: ignore
from task_manager import extract_task  # type: ignore
from navigation_engine import load_route, get_next_navigation_step  # type: ignore
from vision_safety import run_safety_check  # type: ignore
from dynamic_frame_scheduler import DynamicFrameScheduler  # type: ignore
from scene_reasoning import analyze_frame, build_scene_insight  # type: ignore
from conversation_engine import (  # type: ignore
    generate_guidance, answer_question, handle_chat,
    generate_arrival, check_long_running_tasks,
)
from instruction_fusion import fuse  # type: ignore

logger = logging.getLogger("blind_ai")

app = FastAPI(title="Blind AI Navigation Assistant")
app.mount("/app", StaticFiles(directory="client", html=True), name="client")


@app.get("/")
async def root():  # type: ignore
    return HTMLResponse(  # type: ignore
        "<h1>Blind AI Navigation Assistant</h1>"
        "<p><a href='/app'>Open App</a></p>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# PER-SESSION STATE  (held in the WebSocket handler stack frame)
# ─────────────────────────────────────────────────────────────────────────────

class _SessionState:
    """Transient per-connection state (complements the persistent memory dict)."""
    def __init__(self):  # type: ignore
        self.scheduler    = DynamicFrameScheduler()
        self.last_spoken  = 0.0      # timestamp of last spoken output (any source)
        self.first_frame  = True     # force LLM on very first frame
        self.danger_detection_enabled = True  # UI toggle state
        self.last_safety_alert_time = 0.0     # throttle safety alert repeats


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WEBSOCKET HANDLER
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/stream")
async def stream(ws: WebSocket):  # type: ignore
    await ws.accept()

    # Prefer an explicit session-id header; fall back to remote IP
    session_id: str = ws.headers.get("x-session-id") or (ws.client.host if ws.client else "unknown")  # type: ignore
    logger.info(f"🔗 [CONNECT] session={session_id}")

    state = _SessionState()

    try:
        while True:
            message = await ws.receive()

            if message.get("type") == "websocket.disconnect":
                break

            raw = message.get("text") or ""
            if not raw:
                continue

            try:
                payload = json.loads(raw)
            except Exception:
                logger.warning("[WS] Bad JSON payload — skipping")
                continue

            msg_type = payload.get("type")

            # ─────────────────────────────────────────────────────────────
            # SETTING — UI toggles sent by the client
            # ─────────────────────────────────────────────────────────────
            if msg_type == "setting":
                if "danger_detection" in payload:
                    state.danger_detection_enabled = bool(payload["danger_detection"])
                    logger.info(f"⚙️ [SETTING] Danger Detection = {state.danger_detection_enabled}")
                continue

            # ─────────────────────────────────────────────────────────────
            # AUDIO — user spoke
            # ─────────────────────────────────────────────────────────────
            if msg_type == "audio":
                try:
                    await _handle_audio(ws, session_id, state, payload)
                except Exception as e:
                    logger.error(f"💥 [AUDIO ERROR] {e}", exc_info=True)
                    await _send(ws, {"type": "error", "text": "Sorry, something went wrong processing your voice."})

            # ─────────────────────────────────────────────────────────────
            # FRAME — camera image from mobile
            # ─────────────────────────────────────────────────────────────
            elif msg_type == "frame":
                try:
                    await _handle_frame(ws, session_id, state, payload)
                except Exception as e:
                    logger.error(f"💥 [FRAME ERROR] {e}", exc_info=True)

            # ─────────────────────────────────────────────────────────────
            # GPS — location update
            # ─────────────────────────────────────────────────────────────
            elif msg_type == "gps":
                try:
                    await _handle_gps(ws, session_id, state, payload)
                except Exception as e:
                    logger.error(f"💥 [GPS ERROR] {e}", exc_info=True)

            # ─────────────────────────────────────────────────────────────
            # ROUTE — pre-load a route before navigation starts
            # ─────────────────────────────────────────────────────────────
            elif msg_type == "route":
                try:
                    await _handle_route(ws, session_id, payload)
                except Exception as e:
                    logger.error(f"💥 [ROUTE ERROR] {e}", exc_info=True)

    except WebSocketDisconnect:
        logger.info(f"🔌 [DISCONNECT] session={session_id}")
    except Exception as e:
        logger.error(f"💥 [WS FATAL] {e}", exc_info=True)


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO HANDLER
# ─────────────────────────────────────────────────────────────────────────────

async def _handle_audio(ws: WebSocket, session_id: str, state: _SessionState, payload: dict):  # type: ignore
    audio_b64 = payload.get("data", "")
    if not audio_b64:
        return

    kb = len(audio_b64) * 3 // 4 // 1024
    logger.info(f"🎙️ [AUDIO] {kb}KB → Whisper STT...")
    t0 = time.time()

    transcript = await asyncio.to_thread(transcribe_audio, audio_b64)
    if not transcript:
        logger.warning("❌ [STT] Empty transcript")
        await _send(ws, {"type": "error", "text": "Sorry, I didn't catch that. Please try again."})
        return

    logger.info(f"✅ [STT] {(time.time()-t0)*1000:.0f}ms: '{transcript}'")
    await _send(ws, {"type": "transcript", "text": transcript})

    # ── Extract structured task ──────────────────────────────
    try:
        task = await asyncio.to_thread(extract_task, transcript)
    except Exception as e:
        logger.warning(f"[TASK] extract_task failed: {e} — falling back to chat")
        task = {"intent": "chat", "text": transcript}

    intent = task.get("intent", "chat")
    logger.info(f"🎯 [TASK] intent={intent} | {task}")

    mem = mem_mgr.get_memory(session_id)

    # ── Apply long-running tasks & short tasks from this utterance ──
    for t in task.get("long_running_tasks", []):
        mem_mgr.add_long_running_task(session_id, t)
    for t in task.get("short_tasks", []):
        mem_mgr.add_short_task(session_id, t)

    reply = ""

    if intent == "navigate":
        goal = task.get("goal") or transcript
        mem_mgr.set_navigation_goal(session_id, goal)
        mem_mgr.add_turn(session_id, "user", transcript)
        state.scheduler.reset()
        state.first_frame = True

        reply = await asyncio.to_thread(
            handle_chat, session_id,
            f"I want to go to {goal}. Acknowledge my goal warmly and tell me you'll guide me there."
        )
        logger.info(f"🗺️ [NAV] Goal set: '{goal}'")

    elif intent == "query":
        question = task.get("question") or transcript
        mem_mgr.set_pending_question(session_id, question)
        mem_mgr.add_short_task(session_id, question)
        mem_mgr.add_turn(session_id, "user", transcript)
        reply = "Let me look around for you..."

    elif intent == "interrupt":
        mem["task_status"] = "paused"
        mem_mgr.add_turn(session_id, "user", transcript)
        reply = await asyncio.to_thread(
            handle_chat, session_id,
            f"User wants to pause: {task.get('text', transcript)}"
        )

    else:  # chat / unknown
        mem_mgr.add_turn(session_id, "user", transcript)
        # If it looks like a question about surroundings, treat as visual query
        # This handles fallback from rate-limited task extraction
        q_words = ("what", "where", "is there", "can you see", "do you see", "how many", "read", "find", "look", "describe")
        is_question = "?" in transcript or any(transcript.lower().startswith(w) for w in q_words)
        if is_question:
            mem_mgr.set_pending_question(session_id, transcript)
            reply = "Let me look around for you..."
        else:
            reply = await asyncio.to_thread(handle_chat, session_id, transcript)

    if reply:
        mem_mgr.add_turn(session_id, "assistant", reply)
        state.last_spoken = time.time()
        await _send(ws, {
            "type": "response",
            "text": reply,
            "source": "conv",
            "priority": 5,
            "intent": intent,
            "transcript": transcript,
        })


# ─────────────────────────────────────────────────────────────────────────────
# FRAME HANDLER
# ─────────────────────────────────────────────────────────────────────────────

async def _handle_frame(ws: WebSocket, session_id: str, state: _SessionState, payload: dict):  # type: ignore
    frame_b64 = payload.get("data", "")
    if not frame_b64:
        return
    try:
        jpeg_bytes = base64.b64decode(frame_b64)
    except Exception:
        logger.warning("[FRAME] Failed to decode base64 frame")
        return

    mem = mem_mgr.get_memory(session_id)
    goal = mem.get("navigation_goal")
    task_status = mem.get("task_status", "idle")
    pending_q = mem.get("pending_question")

    # Get long-running tasks safely (guard against missing key)
    long_tasks: list = []
    try:
        import task_manager  # type: ignore
        engine = task_manager.get_engine(session_id)
        long_tasks = [t["description"] for t in engine.get_active_tasks() if t["type"] == "LONG_RUNNING"]
    except Exception:
        pass

    # ── 1. SAFETY — always runs, fast (<100ms) ─────────────────
    try:
        safety_result = await asyncio.to_thread(
            run_safety_check,
            jpeg_bytes,
            state.danger_detection_enabled
        )
    except Exception as e:
        logger.error(f"[SAFETY] YOLO error: {e}")
        safety_result = {"alert": None, "objects": [], "clear_path": True,
                         "estimated_clear_distance_m": 15.0, "object_types": set()}

    safety_alert = safety_result.get("alert")

    # Throttle repeated safety alerts (max 1 every 4 seconds)
    if safety_alert:
        now = time.time()
        if now - state.last_safety_alert_time < 4.0:
            safety_alert = None  # suppress repeat
        else:
            state.last_safety_alert_time = now
            # Send safety as immediate priority message
            await _send(ws, {"type": "safety", "text": safety_alert})

    # ── 2. DECIDE WHETHER TO CALL LLM ─────────────────────────
    # Mode A: Navigation active → use dynamic scheduler (3-10 calls/min)
    # Mode B: User asked a question → fire exactly 1 LLM call then clear
    # Mode C: Idle (no goal, no question) → NEVER call LLM (save quota)
    should_call_llm = False

    if pending_q:
        # User explicitly asked something — always answer with a single LLM call
        should_call_llm = True
        logger.info(f"👁️ [SCHED] Question mode — single LLM call for: '{pending_q}'")

    elif goal:
        # Navigation active — use the dynamic scheduler
        try:
            sched_result = state.scheduler.should_send_to_llm(
                pending_question=None,
                yolo_result=safety_result,
                session_memory=mem,
                force=state.first_frame,
            )
            should_call_llm = sched_result.get("should_send", False)
        except Exception as e:
            logger.error(f"[SCHED] Scheduler error: {e}")
            should_call_llm = False
    else:
        # Idle mode — no goal, no question. Do NOT call LLM.
        # YOLO safety check still runs on every frame (above).
        pass

    state.first_frame = False

    # ── 3. SCENE REASONING — only when scheduler says yes ──────
    # NOTE: Works even without a navigation goal (user just wants object detection)
    scene = None
    scene_insight = None
    task_report = None

    if should_call_llm:
        logger.info(f"👁️ [VISION] Analyzing frame (goal={goal or 'none'})...")
        t_vis = time.time()
        try:
            # Use a generic task if no navigation goal
            effective_goal = goal or "describe what you see and warn of any obstacles"
            scene = await asyncio.to_thread(
                analyze_frame, jpeg_bytes, effective_goal, long_tasks
            )
            logger.info(f"👁️ [VISION] {(time.time()-t_vis)*1000:.0f}ms: {str(scene.get('description',''))[:60]}")  # type: ignore
        except Exception as e:
            logger.error(f"[VISION] analyze_frame error: {e}")
            scene = None

        if scene:
            # Update session environment memory
            mem_mgr.update_scene(session_id, scene)

            # Flag decision point for scheduler on next frame
            if scene.get("decision_point"):
                mem["_decision_point_flagged"] = True

            # Check goal arrival via scene
            if goal and goal.lower() in (scene.get("description") or "").lower():
                try:
                    arrival_msg = await asyncio.to_thread(generate_arrival, goal)
                    mem_mgr.clear_navigation_goal(session_id)
                    mem_mgr.add_turn(session_id, "assistant", arrival_msg)
                    state.last_spoken = time.time()
                    await _send(ws, {"type": "response", "text": arrival_msg, "source": "nav", "priority": 2})
                    return
                except Exception as e:
                    logger.error(f"[NAV] Arrival msg error: {e}")

            # Answer pending short-task question
            if pending_q:
                try:
                    answer = await asyncio.to_thread(answer_question, session_id, pending_q, scene)
                    mem_mgr.clear_pending_question(session_id)
                    mem_mgr.remove_short_task(session_id, pending_q)
                    mem_mgr.add_turn(session_id, "assistant", answer)
                    state.last_spoken = time.time()
                    await _send(ws, {"type": "response", "text": answer, "source": "conv", "priority": 5})
                    return
                except Exception as e:
                    logger.error(f"[CONV] answer_question error: {e}")

            # Scene insight (spoken description)
            try:
                scene_insight = build_scene_insight(scene, long_tasks)
            except Exception:
                scene_insight = scene.get("description")

            # Long-running task check
            if scene and long_tasks:
                try:
                    task_report = await asyncio.to_thread(check_long_running_tasks, session_id, scene)
                except Exception as e:
                    logger.error(f"[TASK] check_long_running_tasks error: {e}")

    # ── 4. GUIDANCE — if goal is active ────────────────────────
    conv_response = None
    if task_status == "active" and goal and scene:
        try:
            conv_response = await asyncio.to_thread(generate_guidance, session_id, scene)
        except Exception as e:
            logger.error(f"[CONV] generate_guidance error: {e}")

    # ── 5. INSTRUCTION FUSION — pick the winner ────────────────
    try:
        fused = fuse(
            safety_alert=None,           # already sent above as "safety" type
            nav_instruction=None,        # GPS handler sends nav instructions separately
            scene_insight=scene_insight,
            task_report=task_report,
            conv_response=conv_response,
            last_spoken_time=state.last_spoken,
        )

        if fused["should_speak"] and fused["text"]:
            if fused["source"] != "throttled":
                mem_mgr.add_turn(session_id, "assistant", fused["text"])
                state.last_spoken = time.time()
                await _send(ws, {
                    "type": "response",
                    "text": fused["text"],
                    "source": fused["source"],
                    "priority": fused["priority"],
                })
    except Exception as e:
        logger.error(f"[FUSION] fuse error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# GPS HANDLER
# ─────────────────────────────────────────────────────────────────────────────

async def _handle_gps(ws: WebSocket, session_id: str, state: _SessionState, payload: dict):  # type: ignore
    lat = payload.get("lat")
    lng = payload.get("lng")
    if lat is None or lng is None:
        return

    mem = mem_mgr.get_memory(session_id)
    mem["last_location"] = {"lat": lat, "lng": lng}

    nav_progress = mem.get("navigation_progress", {})
    user_loc = {"lat": lat, "lng": lng}

    try:
        nav_instruction = await asyncio.to_thread(get_next_navigation_step, user_loc, nav_progress)
    except Exception as e:
        logger.error(f"[GPS] nav step error: {e}")
        return

    if nav_instruction:
        step = nav_progress.get("current_step", 0)
        dist = nav_progress.get("distance_to_turn", "?")
        logger.info(f"🗺️ [GPS] step={step} dist={dist}m → '{nav_instruction}'")

        try:
            fused = fuse(nav_instruction=nav_instruction, last_spoken_time=state.last_spoken)
        except Exception:
            fused = {"should_speak": True}
        if fused.get("should_speak", True):
            mem_mgr.add_turn(session_id, "assistant", nav_instruction)
            state.last_spoken = time.time()
            await _send(ws, {
                "type": "response",
                "text": nav_instruction,
                "source": "nav",
                "priority": 2,
            })
    else:
        dist = nav_progress.get("distance_to_turn", "?")
        logger.debug(f"📍 [GPS] lat={lat:.5f} lng={lng:.5f} dist={dist}m")


# ─────────────────────────────────────────────────────────────────────────────
# ROUTE HANDLER
# ─────────────────────────────────────────────────────────────────────────────

async def _handle_route(ws: WebSocket, session_id: str, payload: dict):  # type: ignore
    start = payload.get("start")
    destination = payload.get("dest") or payload.get("destination", "")
    if not start or not destination:
        await _send(ws, {"type": "error", "text": "Missing start or destination for route."})
        return

    mem = mem_mgr.get_memory(session_id)
    nav_progress = mem["navigation_progress"]

    try:
        ok = await asyncio.to_thread(load_route, start, destination, nav_progress)
    except Exception as e:
        logger.error(f"[ROUTE] load_route error: {e}")
        ok = False

    step_count = len(nav_progress.get("route_steps", []))
    logger.info(f"🗺️ [ROUTE] loaded={ok} steps={step_count}")

    await _send(ws, {
        "type": "route_loaded",
        "ok": ok,
        "steps": step_count,
        "destination": destination,
    })


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

async def _send(ws: WebSocket, data: dict):  # type: ignore
    try:
        await ws.send_text(json.dumps(data))
    except Exception as e:
        logger.warning(f"[WS] Send failed: {e}")
