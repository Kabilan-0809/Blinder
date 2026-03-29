"""
agent_controller.py

Central orchestrator for the Blind AI agent pipeline.

This is the "brain" that ties all subsystems together. The WebSocket server
calls into AgentController methods; this module handles all orchestration
logic, keeping the transport layer thin.

Pipeline (per frame):
    1. Vision Safety Engine    — YOLO obstacle detection (always, <100ms)
    2. Dynamic Frame Scheduler — should we call the VLM?
    3. Scene Reasoner          — multimodal LLM analysis (only when triggered)
    4. Environment Memory      — update session state from scene
    5. Task evaluation         — check long-running tasks against scene
    6. Navigation guidance     — LLM guidance if nav goal active
    7. Instruction Fusion      — priority-merge all outputs
    8. Ambient Observations    — Iris personality layer

Pipeline (per audio):
    1. Speech-to-Text          — Whisper transcription
    2. Mood Detection          — update Iris personality state
    3. Task Extraction         — LLM intent + task parsing
    4. Task Manager            — register / route tasks
    5. Response Generation     — LLM reply based on intent
"""

import base64  # type: ignore
import time  # type: ignore
import asyncio  # type: ignore
import logging  # type: ignore
from dataclasses import dataclass, field  # type: ignore

# ── Subsystem imports from domain packages ───────────────────────────────────
from speech.speech_to_text import transcribe_audio  # type: ignore
import agent.environment_memory as mem_mgr  # type: ignore
from agent.task_manager import extract_task, get_engine as get_task_engine  # type: ignore
from navigation.navigation_engine import load_route, get_next_navigation_step, resolve_place  # type: ignore
from vision.vision_safety_engine import run_safety_check  # type: ignore
from scheduler.dynamic_frame_scheduler import DynamicFrameScheduler  # type: ignore
from reasoning.scene_reasoner import analyze_frame, build_scene_insight  # type: ignore
from personality.conversation_engine import (  # type: ignore
    generate_guidance, answer_question, handle_chat,
    generate_arrival, check_long_running_tasks,
)
from agent.instruction_fusion import fuse  # type: ignore
import personality.companion_personality as iris  # type: ignore

logger = logging.getLogger("agent")


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE DATACLASS — clean output from every pipeline call
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentResponse:
    """Structured output from the agent controller."""
    messages: list = field(default_factory=list)  # type: ignore
    # Each message: {"type": str, "text": str, ...}

    def add(self, msg_type: str, text: str, **kwargs):  # type: ignore
        """Add a message to the response batch."""
        if text:
            self.messages.append({"type": msg_type, "text": text, **kwargs})

    def has_messages(self) -> bool:  # type: ignore
        return len(self.messages) > 0


# ─────────────────────────────────────────────────────────────────────────────
# AGENT CONTROLLER — one instance per WebSocket session
# ─────────────────────────────────────────────────────────────────────────────

class AgentController:
    """
    Central orchestrator tying all agent subsystems together.

    Usage:
        controller = AgentController(session_id)

        # On WebSocket connect:
        welcome = controller.get_welcome()

        # Per message:
        response = await controller.process_audio(audio_b64)
        response = await controller.process_frame(jpeg_bytes)
        response = await controller.process_gps(lat, lng)
    """

    def __init__(self, session_id: str):  # type: ignore
        self.session_id = session_id
        self.scheduler = DynamicFrameScheduler()
        self.last_spoken = 0.0
        self.first_frame = True
        self.danger_detection_enabled = True
        self.last_safety_alert_time = 0.0

        # Ensure session memory exists
        mem_mgr.get_memory(session_id)
        logger.info(f"🤖 [AGENT] Controller initialized for session={session_id}")

    # ─────────────────────────────────────────────────────────────────────
    # WELCOME — called once on connection
    # ─────────────────────────────────────────────────────────────────────

    def get_welcome(self) -> dict:  # type: ignore
        """Generate Iris's personalized welcome message."""
        mem = mem_mgr.get_memory(self.session_id)
        profile = mem.get("user_profile", {})
        try:
            welcome_text = iris.get_welcome_message(
                user_name=profile.get("name"),
                journey_count=profile.get("journey_count", 0),
            )
            profile["journey_count"] = profile.get("journey_count", 0) + 1
            logger.info(f"🫂 [IRIS] Welcome: '{welcome_text}'")
            return {
                "type": "welcome",
                "text": welcome_text,
                "ask_name": profile.get("name") is None,
            }
        except Exception as e:
            logger.error(f"[IRIS] Welcome error: {e}")
            return {
                "type": "welcome",
                "text": "Hi! I'm Iris, your walking companion. Where would you like to go?",
                "ask_name": True,
            }

    # ─────────────────────────────────────────────────────────────────────
    # SETTINGS — UI toggles and profile updates
    # ─────────────────────────────────────────────────────────────────────

    def set_danger_detection(self, enabled: bool):  # type: ignore
        """Toggle danger detection on/off."""
        self.danger_detection_enabled = enabled
        logger.info(f"⚙️ [SETTING] Danger Detection = {enabled}")

    def set_user_name(self, name: str):  # type: ignore
        """Store the user's name in their profile."""
        name = name.strip()
        if name:
            mem = mem_mgr.get_memory(self.session_id)
            mem.get("user_profile", {})["name"] = name
            logger.info(f"👤 [PROFILE] User name set to: {name}")

    # ─────────────────────────────────────────────────────────────────────
    # AUDIO PIPELINE — Speech → Intent → Task → Response
    # ─────────────────────────────────────────────────────────────────────

    async def process_audio(self, audio_b64: str) -> AgentResponse:  # type: ignore
        """
        Full audio processing pipeline:
        1. Whisper STT
        2. Mood detection
        3. Task extraction (LLM)
        4. Intent routing (navigate/query/interrupt/chat)
        5. Response generation
        """
        response = AgentResponse()

        if not audio_b64:
            return response

        kb = len(audio_b64) * 3 // 4 // 1024
        logger.info(f"🎙️ [AUDIO] {kb}KB → Whisper STT...")
        t0 = time.time()

        # ── 1. Speech-to-Text ─────────────────────────────────
        transcript = await asyncio.to_thread(transcribe_audio, audio_b64)
        if not transcript:
            logger.warning("❌ [STT] Empty transcript")
            response.add("error", "Sorry, I didn't catch that. Please try again.")
            return response

        logger.info(f"✅ [STT] {(time.time()-t0)*1000:.0f}ms: '{transcript}'")
        response.add("transcript", transcript)

        # ── 2. Mood detection ─────────────────────────────────
        try:
            mood = iris.detect_mood(transcript)
            mem = mem_mgr.get_memory(self.session_id)
            ps = mem.get("personality_state", {})
            ps["current_mood"] = mood
            if mood != "calm":
                logger.info(f"💭 [MOOD] Detected: {mood}")
        except Exception:
            pass

        # ── 3. Task extraction (LLM) ──────────────────────────
        try:
            task = await asyncio.to_thread(extract_task, transcript)
        except Exception as e:
            logger.warning(f"[TASK] extract_task failed: {e} — falling back to chat")
            task = {"intent": "chat", "text": transcript}

        intent = task.get("intent", "chat")
        logger.info(f"🎯 [TASK] intent={intent} | {task}")

        mem = mem_mgr.get_memory(self.session_id)

        # ── 4. Register tasks ─────────────────────────────────
        for t in task.get("long_running_tasks", []):
            mem_mgr.add_long_running_task(self.session_id, t)
        for t in task.get("short_tasks", []):
            mem_mgr.add_short_task(self.session_id, t)

        # ── 5. Intent routing + response generation ───────────
        reply = ""

        if intent == "navigate":
            goal = task.get("goal") or transcript
            mem_mgr.set_navigation_goal(self.session_id, goal)
            mem_mgr.add_turn(self.session_id, "user", transcript)
            self.scheduler.reset()
            self.first_frame = True

            # ── Auto load route if we already have GPS ─────────────
            last_loc = mem["last_location"]
            route_status_suffix = ""
            if last_loc and last_loc.get("lat") and last_loc.get("lng"):
                try:
                    ok = await asyncio.to_thread(
                        load_route,
                        last_loc,
                        goal,
                        mem["navigation_progress"],
                    )
                    if ok:
                        step_count = len(mem["navigation_progress"].get("route_steps", []))
                        route_status_suffix = f" I've loaded a {step_count}-step walking route."
                        logger.info(f"🗺️ [NAV] Auto-loaded route: {step_count} steps to '{goal}'")
                    else:
                        route_status_suffix = " I couldn't find an exact route, but I'll guide you visually."
                        logger.warning(f"[NAV] Auto route load failed for goal='{goal}'")
                except Exception as e:
                    logger.error(f"[NAV] Auto route load error: {e}")
                    route_status_suffix = " I'll guide you visually for now."
            else:
                route_status_suffix = " Please enable GPS so I can load turn-by-turn directions."
                logger.info("[NAV] No GPS yet — skipping auto route load")

            reply = await asyncio.to_thread(
                handle_chat, self.session_id,
                f"I want to go to {goal}. Acknowledge my goal warmly and tell me you'll guide me there."
            )
            if route_status_suffix:
                reply = (reply or "").rstrip('. ') + route_status_suffix
            logger.info(f"🗺️ [NAV] Goal set: '{goal}'")

        elif intent == "query":
            question = task.get("question") or transcript
            mem_mgr.set_pending_question(self.session_id, question)
            mem_mgr.add_short_task(self.session_id, question)
            mem_mgr.add_turn(self.session_id, "user", transcript)
            reply = "Let me look around for you..."

        elif intent == "interrupt":
            mem["task_status"] = "paused"
            mem_mgr.add_turn(self.session_id, "user", transcript)
            reply = await asyncio.to_thread(
                handle_chat, self.session_id,
                f"User wants to pause: {task.get('text', transcript)}"
            )

        else:  # chat / unknown
            mem_mgr.add_turn(self.session_id, "user", transcript)
            q_words = ("what", "where", "is there", "can you see", "do you see",
                       "how many", "read", "find", "look", "describe")
            is_question = "?" in transcript or any(
                transcript.lower().startswith(w) for w in q_words
            )
            if is_question:
                mem_mgr.set_pending_question(self.session_id, transcript)
                reply = "Let me look around for you..."
            else:
                reply = await asyncio.to_thread(handle_chat, self.session_id, transcript)

        if reply:
            mem_mgr.add_turn(self.session_id, "assistant", reply)
            self.last_spoken = time.time()
            response.add("response", reply,
                         source="conv", priority=5,
                         intent=intent, transcript=transcript)

        return response

    # ─────────────────────────────────────────────────────────────────────
    # FRAME PIPELINE — Safety → Scheduler → VLM → Memory → Fusion
    # ─────────────────────────────────────────────────────────────────────

    async def process_frame(self, jpeg_bytes: bytes) -> AgentResponse:  # type: ignore
        """
        Full frame processing pipeline:
        1. Vision Safety Engine  (YOLO, always runs, <100ms)
        2. Frame Scheduler       (should we call VLM?)
        3. Scene Reasoner        (multimodal LLM, only when triggered)
        4. Environment Memory    (update state from scene)
        5. Task evaluation       (long-running task checks)
        6. Navigation guidance   (LLM guidance for active goal)
        7. Instruction Fusion    (priority-merge all outputs)
        8. Ambient Observations  (Iris personality, periodic)
        """
        response = AgentResponse()

        if not jpeg_bytes:
            return response

        mem = mem_mgr.get_memory(self.session_id)
        goal = mem.get("navigation_goal")
        task_status = mem.get("task_status", "idle")
        pending_q = mem.get("pending_question")

        # Get long-running tasks
        long_tasks: list = []
        try:
            engine = get_task_engine(self.session_id)
            long_tasks = [
                t["description"] for t in engine.get_active_tasks()
                if t["type"] == "LONG_RUNNING"
            ]
        except Exception:
            pass

        # ── 1. VISION SAFETY ENGINE — always runs, fast (<100ms) ──────
        try:
            safety_result = await asyncio.to_thread(
                run_safety_check, jpeg_bytes, self.danger_detection_enabled
            )
        except Exception as e:
            logger.error(f"[SAFETY] YOLO error: {e}")
            safety_result = {
                "alert": None, "objects": [], "clear_path": True,
                "estimated_clear_distance_m": 15.0, "object_types": set()
            }

        safety_alert = safety_result.get("alert")

        # Throttle repeated safety alerts (max 1 every 4 seconds)
        if safety_alert:
            now = time.time()
            if now - self.last_safety_alert_time < 4.0:
                safety_alert = None  # suppress repeat
            else:
                self.last_safety_alert_time = now
                response.add("safety", safety_alert)

        # ── 2. DYNAMIC FRAME SCHEDULER — should we call VLM? ─────────
        should_call_llm = False

        if pending_q:
            should_call_llm = True
            logger.info(f"👁️ [SCHED] Question mode — single LLM call for: '{pending_q}'")
        elif goal:
            try:
                sched_result = self.scheduler.should_send_to_llm(
                    pending_question=None,
                    yolo_result=safety_result,
                    session_memory=mem,
                    force=self.first_frame,
                )
                should_call_llm = sched_result.get("should_send", False)
            except Exception as e:
                logger.error(f"[SCHED] Scheduler error: {e}")
        # Idle mode: no goal, no question → do NOT call LLM

        self.first_frame = False

        # ── 3. SCENE REASONER — only when scheduler says yes ──────────
        scene = None
        scene_insight = None
        task_report = None

        if should_call_llm:
            logger.info(f"👁️ [VISION] Analyzing frame (goal={goal or 'none'})...")
            t_vis = time.time()
            try:
                effective_goal = goal or "describe what you see and warn of any obstacles"
                scene = await asyncio.to_thread(
                    analyze_frame, jpeg_bytes, effective_goal, long_tasks
                )
                logger.info(
                    f"👁️ [VISION] {(time.time()-t_vis)*1000:.0f}ms: "
                    f"{str(scene.get('description',''))[:60]}"  # type: ignore
                )
            except Exception as e:
                logger.error(f"[VISION] analyze_frame error: {e}")
                scene = None

            if scene:
                # ── 4. ENVIRONMENT MEMORY — update from scene ─────────
                mem_mgr.update_scene(self.session_id, scene)

                # Update scene mood for Iris personality
                try:
                    ps = mem.get("personality_state", {})
                    ps["scene_mood"] = scene.get("mood_hint") or iris.infer_scene_mood(scene)
                except Exception:
                    pass

                # Flag decision point
                if scene.get("decision_point"):
                    mem["_decision_point_flagged"] = True

                # Check goal arrival
                if goal and goal.lower() in (scene.get("description") or "").lower():
                    try:
                        arrival_msg = await asyncio.to_thread(generate_arrival, goal)
                        mem_mgr.clear_navigation_goal(self.session_id)
                        mem_mgr.add_turn(self.session_id, "assistant", arrival_msg)
                        self.last_spoken = time.time()
                        response.add("response", arrival_msg, source="nav", priority=2)
                        return response
                    except Exception as e:
                        logger.error(f"[NAV] Arrival msg error: {e}")

                # Answer pending question
                if pending_q:
                    try:
                        answer = await asyncio.to_thread(
                            answer_question, self.session_id, pending_q, scene
                        )
                        mem_mgr.clear_pending_question(self.session_id)
                        mem_mgr.remove_short_task(self.session_id, pending_q)
                        mem_mgr.add_turn(self.session_id, "assistant", answer)
                        self.last_spoken = time.time()
                        response.add("response", answer, source="conv", priority=5)
                        return response
                    except Exception as e:
                        logger.error(f"[CONV] answer_question error: {e}")

                # Scene insight
                try:
                    scene_insight = build_scene_insight(scene, long_tasks)
                except Exception:
                    scene_insight = scene.get("description")

                # ── 5. TASK EVALUATION — long-running task check ──────
                if long_tasks:
                    try:
                        task_report = await asyncio.to_thread(
                            check_long_running_tasks, self.session_id, scene
                        )
                    except Exception as e:
                        logger.error(f"[TASK] check_long_running_tasks error: {e}")

        # ── 6. NAVIGATION GUIDANCE — if goal is active ────────────────
        conv_response = None
        if task_status == "active" and goal and scene:
            try:
                nav_progress = mem.get("navigation_progress", {})
                conv_response = await asyncio.to_thread(
                    generate_guidance, self.session_id, scene, nav_progress
                )
            except Exception as e:
                logger.error(f"[CONV] generate_guidance error: {e}")

        # ── 7. INSTRUCTION FUSION — pick the winner ──────────────────
        try:
            fused = fuse(
                safety_alert=None,  # already sent above as "safety" type
                nav_instruction=None,
                scene_insight=scene_insight,
                task_report=task_report,
                conv_response=conv_response,
                last_spoken_time=self.last_spoken,
            )
            if fused["should_speak"] and fused["text"]:
                if fused["source"] != "throttled":
                    mem_mgr.add_turn(self.session_id, "assistant", fused["text"])
                    self.last_spoken = time.time()
                    response.add("response", fused["text"],
                                 source=fused["source"],
                                 priority=fused["priority"])
        except Exception as e:
            logger.error(f"[FUSION] fuse error: {e}")

        # ── 8. AMBIENT COMPANION OBSERVATION (Iris personality) ───────
        if goal and not pending_q:
            try:
                ps = mem.get("personality_state", {})
                last_amb = ps.get("last_ambient_time", 0.0)
                env = mem.get("environment_memory", {})
                amb = iris.get_ambient_observation(
                    crowd_density=env.get("crowd_density", "unknown"),
                    clear_path=safety_result.get("clear_path", True),
                    journey_progress=0.0,
                    last_ambient_time=last_amb,
                )
                if amb:
                    ps["last_ambient_time"] = time.time()
                    response.add("ambient", amb)
                    logger.info(f"🫂 [IRIS] Ambient: '{amb}'")
            except Exception as e:
                logger.error(f"[IRIS] Ambient error: {e}")

        return response

    # ─────────────────────────────────────────────────────────────────────
    # GPS PIPELINE — Location → Navigation Step → Fusion
    # ─────────────────────────────────────────────────────────────────────

    async def process_gps(self, lat: float, lng: float) -> AgentResponse:  # type: ignore
        """Process a GPS location update and return navigation instructions."""
        response = AgentResponse()

        mem = mem_mgr.get_memory(self.session_id)
        mem["last_location"] = {"lat": lat, "lng": lng}

        nav_progress = mem.get("navigation_progress", {})
        user_loc = {"lat": lat, "lng": lng}

        try:
            nav_instruction = await asyncio.to_thread(
                get_next_navigation_step, user_loc, nav_progress
            )
        except Exception as e:
            logger.error(f"[GPS] nav step error: {e}")
            return response

        if nav_instruction:
            step = nav_progress.get("current_step", 0)
            dist = nav_progress.get("distance_to_turn", "?")
            logger.info(f"🗺️ [GPS] step={step} dist={dist}m → '{nav_instruction}'")

            try:
                fused = fuse(nav_instruction=nav_instruction,
                             last_spoken_time=self.last_spoken)
            except Exception:
                fused = {"should_speak": True}

            if fused.get("should_speak", True):
                mem_mgr.add_turn(self.session_id, "assistant", nav_instruction)
                self.last_spoken = time.time()
                response.add("response", nav_instruction,
                             source="nav", priority=2)
        else:
            dist = nav_progress.get("distance_to_turn", "?")
            logger.debug(f"📍 [GPS] lat={lat:.5f} lng={lng:.5f} dist={dist}m")

        return response

    # ─────────────────────────────────────────────────────────────────────
    # ROUTE LOADING
    # ─────────────────────────────────────────────────────────────────────

    async def load_route(self, start: dict, destination: str) -> dict:  # type: ignore
        """Load a Google Maps walking route."""
        mem = mem_mgr.get_memory(self.session_id)
        nav_progress = mem["navigation_progress"]

        try:
            ok = await asyncio.to_thread(load_route, start, destination, nav_progress)
        except Exception as e:
            logger.error(f"[ROUTE] load_route error: {e}")
            ok = False

        step_count = len(nav_progress.get("route_steps", []))
        logger.info(f"🗺️ [ROUTE] loaded={ok} steps={step_count}")

        return {
            "type": "route_loaded",
            "ok": ok,
            "steps": step_count,
            "destination": destination,
        }
