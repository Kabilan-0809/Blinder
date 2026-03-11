from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import json
import time
import asyncio
import logging

from speech_to_text import transcribe_audio
from task_manager import extract_task
from vision_reasoning import analyze_frame
from environment_memory import (
    get_memory, set_goal, add_observation, add_turn,
    complete_goal, pause_task, resume_task
)
from conversation_engine import generate_guidance, answer_question
from safety_detector import run_safety_check
from route_navigation import load_route, get_next_navigation_step

logger = logging.getLogger("blind_ai_conv")

app = FastAPI()
app.mount("/app", StaticFiles(directory="client", html=True), name="client")

@app.get("/")
async def root():
    return HTMLResponse("<h1>Blind Navigation AI — Conversational Mode</h1><p><a href='/app'>Open App</a></p>")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN WEBSOCKET
# Message types from client:
#   {"type": "audio",  "data": "<base64 webm>"}    → STT → task extraction
#   {"type": "frame",  "data": "<base64 JPEG>"}    → safety + vision reasoning
#   {"type": "gps",    "lat": x, "lng": y}         → GPS navigation
#   {"type": "route",  "start": {...}, "destination": "..."} → load a route
# ─────────────────────────────────────────────────────────────────────────────
@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    session_id = ws.client.host
    logger.info(f"[CONNECT] Session: {session_id}")

    last_vision_time = 0.0   # Rate limit vision calls (max 1 per 2 seconds)
    last_guide_time  = 0.0   # Rate limit guidance calls (max 1 per 4 seconds)

    try:
        while True:
            message = await ws.receive()

            # ── Graceful disconnect ──────────────────────────────────────────
            if message.get("type") == "websocket.disconnect":
                logger.info(f"[DISCONNECT] {session_id}")
                break

            raw = message.get("text") or ""
            if not raw:
                # Binary frame — not expected in new architecture (all base64 over JSON)
                continue

            try:
                payload = json.loads(raw)
            except Exception:
                logger.warning("[WS] Could not parse payload as JSON")
                continue

            msg_type = payload.get("type")
            mem = get_memory(session_id)

            # ────────────────────────────────────────────────────────────────
            # TYPE: AUDIO — user pressed STOP after speaking
            # ────────────────────────────────────────────────────────────────
            if msg_type == "audio":
                audio_b64 = payload.get("data", "")
                if not audio_b64:
                    continue

                # Run STT in background thread (Whisper is CPU-bound)
                transcript = await asyncio.to_thread(transcribe_audio, audio_b64)
                if not transcript:
                    await ws.send_text(json.dumps({"type": "error", "text": "Sorry, I didn't catch that."}))
                    continue

                logger.info(f"[USER] '{transcript}'")

                # Extract task/intent
                task = await asyncio.to_thread(extract_task, transcript)
                intent = task.get("intent", "unknown")

                if intent == "navigate":
                    goal = task.get("goal", transcript)
                    set_goal(session_id, goal)
                    add_turn(session_id, "user", transcript)
                    reply = f"Got it! I'll guide you to {goal}. Please show me the surroundings."
                    add_turn(session_id, "assistant", reply)
                    logger.info(f"[TASK] Goal set: {goal}")

                elif intent == "query":
                    question = task.get("question", transcript)
                    pause_task(session_id, question)
                    add_turn(session_id, "user", transcript)
                    # Will be answered on the next frame
                    reply = "Let me look..."
                    add_turn(session_id, "assistant", reply)

                elif intent == "interrupt":
                    new_req = task.get("new_request", transcript)
                    pause_task(session_id, new_req)
                    add_turn(session_id, "user", transcript)
                    reply = "Sure, hold on."
                    add_turn(session_id, "assistant", reply)

                else:
                    # Unknown — treat as navigation goal
                    set_goal(session_id, transcript)
                    add_turn(session_id, "user", transcript)
                    reply = f"I'll try to help with that."
                    add_turn(session_id, "assistant", reply)

                await ws.send_text(json.dumps({
                    "type": "response",
                    "text": reply,
                    "transcript": transcript,
                    "intent": intent
                }))

            # ────────────────────────────────────────────────────────────────
            # TYPE: FRAME — camera image
            # ────────────────────────────────────────────────────────────────
            elif msg_type == "frame":
                import base64
                frame_b64 = payload.get("data", "")
                if not frame_b64:
                    continue

                try:
                    jpeg_bytes = base64.b64decode(frame_b64)
                except Exception:
                    continue

                # ── SAFETY CHECK — runs on every frame, always ───────────────
                safety_alert = await asyncio.to_thread(run_safety_check, jpeg_bytes)
                if safety_alert:
                    logger.warning(f"[SAFETY] → {safety_alert}")
                    await ws.send_text(json.dumps({"type": "safety", "text": safety_alert}))
                    continue   # Don't run vision/guidance when there's a danger

                now = time.time()
                goal = mem.get("current_goal")
                task_status = mem.get("task_status", "idle")

                # ── VISION REASONING — max 1 per 2 seconds ───────────────────
                vision_description = None
                if now - last_vision_time > 2.0 and goal:
                    vision_description = await asyncio.to_thread(
                        analyze_frame, jpeg_bytes, goal
                    )
                    last_vision_time = now

                    if vision_description and vision_description != "UNCLEAR":
                        add_observation(session_id, vision_description)

                        # Check goal completion by seeing if goal string appears in vision
                        if goal and goal.lower() in vision_description.lower():
                            complete_goal(session_id)
                            reply = f"You've reached {goal}!"
                            add_turn(session_id, "assistant", reply)
                            await ws.send_text(json.dumps({
                                "type": "response",
                                "text": reply,
                                "vision": vision_description
                            }))
                            continue

                # ── ANSWER PENDING QUESTION ───────────────────────────────────
                pending_q = mem.get("pending_question")
                if pending_q and vision_description:
                    reply = await asyncio.to_thread(
                        answer_question, session_id, pending_q, vision_description
                    )
                    add_turn(session_id, "assistant", reply)
                    resume_task(session_id)
                    await ws.send_text(json.dumps({"type": "response", "text": reply}))
                    last_guide_time = now
                    continue

                # ── GUIDANCE — max 1 per 4 seconds while goal is active ───────
                if task_status == "active" and goal and (now - last_guide_time > 4.0):
                    guidance = await asyncio.to_thread(
                        generate_guidance, session_id, vision_description
                    )
                    if guidance:
                        add_turn(session_id, "assistant", guidance)
                        last_guide_time = now
                        await ws.send_text(json.dumps({
                            "type": "response",
                            "text": guidance,
                            "vision": vision_description or ""
                        }))

            # ────────────────────────────────────────────────────────────────
            # TYPE: GPS — location update
            # ────────────────────────────────────────────────────────────────
            elif msg_type == "gps":
                lat, lng = payload.get("lat"), payload.get("lng")
                if lat and lng:
                    mem["last_location"] = {"lat": lat, "lng": lng}
                    nav = get_next_navigation_step({"lat": lat, "lng": lng}, mem)
                    if nav:
                        await ws.send_text(json.dumps({"type": "navigation", "text": nav}))

            # ────────────────────────────────────────────────────────────────
            # TYPE: ROUTE — load a Google Maps route into session
            # ────────────────────────────────────────────────────────────────
            elif msg_type == "route":
                ok = await asyncio.to_thread(
                    load_route, payload.get("start"), payload.get("destination"), mem
                )
                await ws.send_text(json.dumps({
                    "type": "route_loaded",
                    "ok": ok,
                    "steps": len(mem.get("route_steps", []))
                }))

    except WebSocketDisconnect:
        logger.info(f"[DISCONNECT] {session_id}")
    except Exception as e:
        logger.error(f"[WS ERROR] {e}")
