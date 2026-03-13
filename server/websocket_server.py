"""
websocket_server.py

Thin coordinator for the Blind AI navigation assistant.
Delegates all logic to specialized subsystem modules.

WebSocket message protocol (client → server):
  { "type": "audio",  "data": "<base64 webm>" }         STT → task extraction
  { "type": "frame",  "data": "<base64 JPEG>" }          safety + optional scene LLM
  { "type": "gps",    "lat": float, "lng": float }       GPS nav step check
  { "type": "route",  "start": {...}, "dest": "..." }    Load a Google Maps route

WebSocket message protocol (server → client):
  { "type": "response",   "text": "...", "source": "...", "priority": int }
  { "type": "transcript", "text": "..." }
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
        self.danger_detection_enabled = True # UI toggle state


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
                    state.danger_detection_enabled = payload["danger_detection"]
                    logger.info(f"⚙️ [SETTING] Danger Detection = {state.danger_detection_enabled}")
                continue

            # ─────────────────────────────────────────────────────────────
            # AUDIO — user spoke (pressed STOP)
            # ─────────────────────────────────────────────────────────────
            if msg_type == "audio":
                await _handle_audio(ws, session_id, state, payload)

            # ─────────────────────────────────────────────────────────────
            # FRAME — camera image from mobile
            # ─────────────────────────────────────────────────────────────
            elif msg_type == "frame":
                await _handle_frame(ws, session_id, state, payload)

            # ─────────────────────────────────────────────────────────────
            # GPS — location update
            # ─────────────────────────────────────────────────────────────
            elif msg_type == "gps":
                await _handle_gps(ws, session_id, state, payload)

            # ─────────────────────────────────────────────────────────────
            # ROUTE — pre-load a route before navigation starts
            # ─────────────────────────────────────────────────────────────
            elif msg_type == "route":
                await _handle_route(ws, session_id, payload)

    except WebSocketDisconnect:
        logger.info(f"🔌 [DISCONNECT] session={session_id}")
    except Exception as e:
        logger.error(f"💥 [WS ERROR] {e}", exc_info=True)


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
        await _send(ws, {"type": "error", "text": "Sorry, I didn't catch that."})
        return

    logger.info(f"✅ [STT] {(time.time()-t0)*1000:.0f}ms: '{transcript}'")
    await _send(ws, {"type": "transcript", "text": transcript})

    # ── Extract structured task ──────────────────────────────
    task = await asyncio.to_thread(extract_task, transcript)
    intent = task.get("intent", "unknown")
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

        # Kick off route loading in background (needs GPS — client sends route msg)
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
        reply = "Let me look..."

    elif intent == "interrupt":
        mem["task_status"] = "paused"
        mem_mgr.add_turn(session_id, "user", transcript)
        reply = await asyncio.to_thread(
            handle_chat, session_id,
            f"User wants to pause or interrupt: {task.get('text', transcript)}"
        )

    else:  # chat / unknown
        mem_mgr.add_turn(session_id, "user", transcript)
        reply = await asyncio.to_thread(handle_chat, session_id, transcript)

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
        return

    mem = mem_mgr.get_memory(session_id)
    goal = mem.get("navigation_goal")
    task_status = mem.get("task_status", "idle")
    pending_q = mem.get("pending_question")
    long_tasks = mem.get("active_tasks", {}).get("long_running", [])

    # ── 1. SAFETY — always runs, fast (<100ms) ─────────────────
    safety_result = await asyncio.to_thread(
        run_safety_check, 
        jpeg_bytes, 
        state.danger_detection_enabled
    )
    safety_alert = safety_result.get("alert")

    # ── 2. DYNAMIC SCHEDULER — should we call the LLM? ────────
    sched_result = state.scheduler.should_send_to_llm(
        pending_question=pending_q,
        yolo_result=safety_result,
        session_memory=mem,
        force=state.first_frame,
    )
    state.first_frame = False

    # ── 3. SCENE REASONING — only when scheduler says yes ──────
    scene = None
    scene_insight = None
    task_report = None

    if sched_result["should_send"] and goal:
        logger.info(f"👁️ [VISION] Analyzing frame (reason={sched_result['reason']})...")
        t_vis = time.time()
        scene = await asyncio.to_thread(
            analyze_frame, jpeg_bytes, goal, long_tasks
        )
        logger.info(f"👁️ [VISION] {(time.time()-t_vis)*1000:.0f}ms: {scene.get('description','')[:60]}")

        # Update session environment memory
        mem_mgr.update_scene(session_id, scene)

        # Flag decision point for scheduler on next frame
        if scene.get("decision_point"):
            mem["_decision_point_flagged"] = True

        # Check goal arrival via scene (complement to GPS arrival)
        if goal and goal.lower() in (scene.get("description") or "").lower():
            arrival_msg = await asyncio.to_thread(generate_arrival, goal)
            mem_mgr.clear_navigation_goal(session_id)
            mem_mgr.add_turn(session_id, "assistant", arrival_msg)
            state.last_spoken = time.time()
            await _send(ws, {"type": "response", "text": arrival_msg, "source": "nav", "priority": 2})
            return

        # Answer pending short-task question
        if pending_q:
            answer = await asyncio.to_thread(answer_question, session_id, pending_q, scene)
            mem_mgr.clear_pending_question(session_id)
            mem_mgr.remove_short_task(session_id, pending_q)
            mem_mgr.add_turn(session_id, "assistant", answer)
            state.last_spoken = time.time()
            await _send(ws, {"type": "response", "text": answer, "source": "conv", "priority": 5})
            return

        # Scene insight (spoken description)
        scene_insight = build_scene_insight(scene, long_tasks)

        # Long-running task check
        if scene and long_tasks:
            task_report = await asyncio.to_thread(check_long_running_tasks, session_id, scene)

    # ── 4. GUIDANCE — if goal is active ────────────────────────
    conv_response = None
    if task_status == "active" and goal:
        conv_response = await asyncio.to_thread(generate_guidance, session_id, scene)

    # ── 5. INSTRUCTION FUSION — pick the winner ────────────────
    fused = fuse(
        safety_alert=safety_alert,
        nav_instruction=None,          # GPS handler sends nav instructions separately
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

    nav_instruction = await asyncio.to_thread(get_next_navigation_step, user_loc, nav_progress)

    if nav_instruction:
        step = nav_progress.get("current_step", 0)
        dist = nav_progress.get("distance_to_turn", "?")
        logger.info(f"🗺️ [GPS] step={step} dist={dist}m → '{nav_instruction}'")

        fused = fuse(
            nav_instruction=nav_instruction,
            last_spoken_time=state.last_spoken,
        )
        if fused["should_speak"]:
            mem_mgr.add_turn(session_id, "assistant", nav_instruction)
            state.last_spoken = time.time()
            await _send(ws, {
                "type": "response",
                "text": nav_instruction,
                "source": "nav",
                "priority": 2,
            })
    else:
        step = nav_progress.get("current_step", 0)
        dist = nav_progress.get("distance_to_turn", "?")
        logger.debug(f"📍 [GPS] lat={lat:.5f} lng={lng:.5f} step={step} dist={dist}m")


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

    ok = await asyncio.to_thread(load_route, start, destination, nav_progress)
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
