from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # type: ignore
from fastapi.responses import HTMLResponse  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore
import json  # type: ignore
import time  # type: ignore
import asyncio  # type: ignore
import logging  # type: ignore

from speech_to_text import transcribe_audio  # type: ignore
from task_manager import extract_task  # type: ignore
from vision_reasoning import analyze_frame  # type: ignore
from environment_memory import (  # type: ignore
    get_memory, set_goal, add_observation, add_turn,
    complete_goal, pause_task, resume_task
)
from conversation_engine import generate_guidance, answer_question, handle_chat  # type: ignore
from safety_detector import run_safety_check  # type: ignore
from route_navigation import load_route, get_next_navigation_step  # type: ignore

logger = logging.getLogger("blind_ai_conv")

app = FastAPI()
app.mount("/app", StaticFiles(directory="client", html=True), name="client")

@app.get("/")
async def root():
    return HTMLResponse("<h1>Blind Navigation AI — Conversational Mode</h1><p><a href='/app'>Open App</a></p>")  # type: ignore

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
    logger.info(f"🔗 [CONNECT] New session: {session_id}")
    logger.info(f"   Waiting for audio/frame/gps messages...")

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
                audio_size_kb = len(audio_b64) * 3 // 4 // 1024
                logger.info(f"🎙️  [AUDIO] Received {audio_size_kb}KB audio blob → running Whisper STT...")
                t0 = time.time()
                transcript = await asyncio.to_thread(transcribe_audio, audio_b64)
                stt_ms = (time.time() - t0) * 1000
                if not transcript:
                    logger.warning("❌  [STT] Empty transcript — could not understand audio")
                    await ws.send_text(json.dumps({"type": "error", "text": "Sorry, I didn't catch that."}))
                    continue
                logger.info(f"✅  [STT] Transcript ({stt_ms:.0f}ms): '{transcript}'")

                # Extract task/intent
                logger.info("🧠  [TASK] Extracting intent from transcript...")
                t1 = time.time()
                task = await asyncio.to_thread(extract_task, transcript)
                intent = task.get("intent", "unknown")
                logger.info(f"🎯  [TASK] Intent={intent} | Data={task} ({(time.time()-t1)*1000:.0f}ms)")

                if intent == "navigate":
                    goal = task.get("goal", transcript)
                    set_goal(session_id, goal)
                    add_turn(session_id, "user", transcript)
                    # Use LLM to generate a natural confirmation instead of hardcoding
                    reply = await asyncio.to_thread(handle_chat, session_id, f"I want to go to {goal}. Please confirm and say you will guide me.")
                    add_turn(session_id, "assistant", reply)
                    logger.info(f"🗺️  [NAVIGATE] Goal set → '{goal}'")

                elif intent == "query":
                    question = task.get("question", transcript)
                    pause_task(session_id, question)
                    add_turn(session_id, "user", transcript)
                    reply = "Let me look..."
                    add_turn(session_id, "assistant", reply)

                elif intent == "interrupt":
                    new_req = task.get("new_request", transcript)
                    pause_task(session_id, new_req)
                    add_turn(session_id, "user", transcript)
                    reply = await asyncio.to_thread(handle_chat, session_id, f"I need to pause navigation: {new_req}")
                    add_turn(session_id, "assistant", reply)

                else:
                    # Chat or unknown — let the LLM handle it naturally
                    add_turn(session_id, "user", transcript)
                    reply = await asyncio.to_thread(handle_chat, session_id, transcript)
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
                import base64  # type: ignore
                frame_b64 = payload.get("data", "")
                if not frame_b64:
                    continue

                try:
                    jpeg_bytes = base64.b64decode(frame_b64)
                except Exception:
                    continue

                # ── SAFETY CHECK — runs on every frame, always ───────────────
                # [DISABLED FOR CONVERSATION TESTING]
                # t_safe = time.time()
                # safety_alert = await asyncio.to_thread(run_safety_check, jpeg_bytes)
                # safe_ms = (time.time() - t_safe) * 1000
                # if safety_alert:
                #     logger.warning(f"🚨  [SAFETY] {safety_alert}  ({safe_ms:.0f}ms)")
                #     await ws.send_text(json.dumps({"type": "safety", "text": safety_alert}))
                #     continue   # Don't run vision/guidance when there's a danger
                # else:
                #     logger.info(f"✅  [SAFETY] Clear ({safe_ms:.0f}ms)")

                now = time.time()
                goal = mem.get("current_goal")
                task_status = mem.get("task_status", "idle")

                # ── VISION REASONING — max 1 per 2 seconds ───────────────────
                vision_description = None
                if now - last_vision_time > 2.0 and goal:
                    logger.info(f"👁️  [VISION] Analyzing frame for goal='{goal}'...")
                    t_vis = time.time()
                    vision_description = await asyncio.to_thread(
                        analyze_frame, jpeg_bytes, goal
                    )
                    last_vision_time = now
                    vis_ms = (time.time() - t_vis) * 1000
                    logger.info(f"👁️  [VISION] ({vis_ms:.0f}ms): {vision_description[:80]}...")

                    if vision_description and vision_description != "UNCLEAR":
                        add_observation(session_id, vision_description)
                        # Check goal completion
                        if goal and goal.lower() in vision_description.lower():
                            logger.info(f"🏁  [GOAL] REACHED: '{goal}'")
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
                    logger.info("💬  [GUIDE] Generating navigation guidance...")
                    # Update rate limit timer IMMEDIATELY to prevent infinite loop on 429s
                    last_guide_time = now 
                    t_guide = time.time()
                    guidance = await asyncio.to_thread(
                        generate_guidance, session_id, vision_description
                    )
                    guide_ms = (time.time() - t_guide) * 1000
                    if guidance:
                        add_turn(session_id, "assistant", guidance)
                        logger.info(f"💬  [GUIDE] ({guide_ms:.0f}ms): '{guidance}'")
                        await ws.send_text(json.dumps({
                            "type": "response",
                            "text": guidance,
                            "vision": vision_description or ""
                        }))
                    else:
                        logger.info(f"💬  [GUIDE] Skipped or failed (rate limited, {guide_ms:.0f}ms)")

            # ────────────────────────────────────────────────────────────────
            # TYPE: GPS — location update
            # ────────────────────────────────────────────────────────────────
            elif msg_type == "gps":
                lat, lng = payload.get("lat"), payload.get("lng")
                if lat and lng:
                    mem["last_location"] = {"lat": lat, "lng": lng}
                    nav = get_next_navigation_step({"lat": lat, "lng": lng}, mem)
                    if nav:
                        logger.info(f"🗺️  [GPS NAV] → '{nav}'")
                        await ws.send_text(json.dumps({"type": "navigation", "text": nav}))
                    else:
                        step = mem.get('current_route_step', 0)
                        dist = mem.get('distance_to_next_turn')
                        logger.info(f"📍  [GPS] lat={lat:.4f} lng={lng:.4f} | step={step} dist={dist}m")

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
        logger.info(f"🔌  [DISCONNECT] Session ended: {session_id}")
    except Exception as e:
        logger.error(f"💥  [WS ERROR] Unhandled exception: {e}", exc_info=True)
