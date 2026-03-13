"""
task_manager.py

Production-grade task lifecycle manager for the Blind AI navigation assistant.

TASK TYPES
──────────
  LONG_RUNNING  — Active for the entire navigation session until user cancels.
                  Executed on every LLM reasoning cycle.
                  Examples: "guide me to X", "warn about poles", "alert if crowd"

  SHORT         — One-time task answered from a single frame + LLM call.
                  Auto-removed after answer is delivered.
                  Examples: "Is there a crowd ahead?", "What sign is that?"

  TEMP_CONTEXT  — Conversational question answered from session memory alone.
                  Does NOT trigger vision processing.
                  Examples: "Which direction were we heading?", "What's my goal?"

TASK MEMORY STRUCTURE (stored in memory_manager)
────────────────────────────────────────────────
  {
    "navigation_goal":  str | None,
    "tasks": [
      {
        "id":           int,
        "type":         "LONG_RUNNING" | "SHORT" | "TEMP_CONTEXT",
        "priority":     int,           # 1=safety, 2=nav, 3=query, 4=info
        "description":  str,
        "subsystem":    "safety" | "navigation" | "scene" | "memory" | "conversation",
        "status":       "pending" | "active" | "paused" | "done",
        "created_at":   float,
        "answered":     bool,
      }
    ],
    "interrupted_tasks": [],           # tasks paused by a user interruption
    "conversation_context": []         # rolling conversation turns
  }

PRIORITY ORDER
──────────────
  1 — Safety monitoring (obstacle/crowd/hazard detection)
  2 — Navigation (route following, turn-by-turn)
  3 — User questions (one-shot or context queries)
  4 — Informational / ambient awareness

PIPELINE INTEGRATION
────────────────────
  Speech → extract_tasks() → TaskEngine.add_task()
         → websocket_server dispatches to subsystems based on task.subsystem
         → on completion → TaskEngine.mark_done() / auto-remove SHORT tasks
"""

import os  # type: ignore
import json  # type: ignore
import time  # type: ignore
import uuid  # type: ignore
import logging  # type: ignore
from google import genai  # type: ignore
from dotenv import load_dotenv  # type: ignore

load_dotenv()
logger = logging.getLogger(__name__)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

PRIORITY = {
    "safety":       1,
    "navigation":   2,
    "query":        3,
    "informational":4,
}

SUBSYSTEM_MAP = {
    # task keyword → subsystem that handles it
    "pole":       "safety",
    "obstacle":   "safety",
    "crowd":      "safety",
    "people":     "safety",
    "wall":       "safety",
    "steps":      "safety",
    "stairs":     "safety",
    "navigate":   "navigation",
    "guide":      "navigation",
    "take me":    "navigation",
    "go to":      "navigation",
    "sign":       "scene",
    "read":       "scene",
    "what is":    "scene",
    "direction":  "memory",
    "heading":    "memory",
    "goal":       "memory",
    "where":      "memory",
}

# ─────────────────────────────────────────────────────────────────────────────
# LLM EXTRACTION PROMPT
# ─────────────────────────────────────────────────────────────────────────────

TASK_EXTRACT_PROMPT = """You are a task parser for a blind navigation AI assistant.

Parse the user's spoken sentence and return ONLY a JSON object with this exact schema:

{
  "intent": "navigate" | "query" | "interrupt" | "chat",
  "navigation_goal": "<destination string>" | null,
  "tasks": [
    {
      "type": "LONG_RUNNING" | "SHORT" | "TEMP_CONTEXT",
      "description": "<clear task description>",
      "subsystem": "safety" | "navigation" | "scene" | "memory" | "conversation"
    }
  ]
}

TASK TYPE RULES:
- LONG_RUNNING: continuous monitoring throughout the session
  e.g. "warn about poles", "detect crowds", "alert me to steps"
  subsystem: safety (for hazards), scene (for ambient awareness)

- SHORT: a one-time question about the current physical environment — needs fresh camera frame
  e.g. "is there a crowd?", "what sign is that?", "is the road clear?"
  subsystem: scene

- TEMP_CONTEXT: a question answerable from session memory only — no camera needed
  e.g. "which direction were we heading?", "what is my goal?", "how far to the turn?"
  subsystem: memory

INTENT RULES:
- "navigate": user states a destination to go to
  → always create a LONG_RUNNING navigation task in tasks[]
- "query": user asks a one-time question about surroundings → SHORT task
- "interrupt": user wants to stop/pause/cancel current navigation
- "chat": casual conversation → TEMP_CONTEXT or conversation task

PRIORITY ASSIGNMENT (do not include in output, computed internally):
  safety tasks → priority 1
  navigation → priority 2
  scene queries → priority 3
  memory/conversation → priority 4

RULES:
- Always return valid JSON with no text outside the object
- tasks[] must contain ALL tasks embedded in the sentence, even if multiple
- navigation_goal: only if intent is "navigate", else null
- description: be specific and action-oriented (e.g. "Monitor for poles and warn user")

EXAMPLES:

Input: "Guide me to Surya Super Market and warn me about poles"
Output: {
  "intent": "navigate",
  "navigation_goal": "Surya Super Market",
  "tasks": [
    {"type": "LONG_RUNNING", "description": "Navigate to Surya Super Market", "subsystem": "navigation"},
    {"type": "LONG_RUNNING", "description": "Monitor for poles and warn user", "subsystem": "safety"}
  ]
}

Input: "Is there a crowd ahead?"
Output: {
  "intent": "query",
  "navigation_goal": null,
  "tasks": [
    {"type": "SHORT", "description": "Check if there is a crowd ahead", "subsystem": "scene"}
  ]
}

Input: "Which direction were we heading?"
Output: {
  "intent": "chat",
  "navigation_goal": null,
  "tasks": [
    {"type": "TEMP_CONTEXT", "description": "Recall current navigation direction from memory", "subsystem": "memory"}
  ]
}

Input: "Stop, I need a minute"
Output: {
  "intent": "interrupt",
  "navigation_goal": null,
  "tasks": []
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# TASK ENGINE — the core lifecycle manager
# ─────────────────────────────────────────────────────────────────────────────

class TaskEngine:
    """
    Stateful task lifecycle manager — one instance per session.

    Maintains a structured task list with priorities, lifecycle states,
    and interruption handling.

    Usage:
        engine = TaskEngine(session_id)

        # From speech input:
        parsed  = engine.extract_tasks(transcript)
        created = engine.apply_extracted(parsed)

        # During execution loop:
        task = engine.get_highest_priority_task()

        # After completion:
        engine.mark_done(task["id"])

        # On user interruption:
        engine.interrupt_and_push(new_task)
        engine.resume_interrupted()
    """

    def __init__(self, session_id: str):  # type: ignore
        self.session_id     : str   = session_id
        self._tasks         : list  = []         # all active task dicts
        self._interrupted   : list  = []         # tasks paused by interruption
        self._nav_task_id   : int | None = None  # id of current nav task
        self._id_counter    : int   = 0

    # ── Task creation ─────────────────────────────────────────────────

    def add_task(self, description: str, task_type: str, subsystem: str) -> dict:  # type: ignore
        """
        Create and register a task.

        Args:
            description: Human-readable task description
            task_type:   "LONG_RUNNING" | "SHORT" | "TEMP_CONTEXT"
            subsystem:   "safety" | "navigation" | "scene" | "memory" | "conversation"

        Returns:
            The newly created task dict.
        """
        self._id_counter += 1
        priority = self._infer_priority(task_type, subsystem)

        task = {
            "id":           self._id_counter,
            "type":         task_type,
            "priority":     priority,
            "description":  description,
            "subsystem":    subsystem,
            "status":       "active" if task_type != "TEMP_CONTEXT" else "pending",
            "created_at":   time.time(),
            "answered":     False,
        }

        # Deduplicate: don't add if a sufficiently similar task already exists
        if not self._is_duplicate(description):
            self._tasks.append(task)
            logger.info(
                f"[TASKS] +{task_type}[{task['id']}] p{priority} "
                f"'{description}' → {subsystem}"
            )

            # Track the navigation task id for later removal on arrival
            if subsystem == "navigation":
                self._nav_task_id = int(task["id"])  # type: ignore
        else:
            logger.debug(f"[TASKS] Duplicate skipped: '{description}'")

        return task  # type: ignore

    def remove_task(self, task_id: int) -> bool:  # type: ignore
        """
        Remove a task by id. Returns True if found and removed.
        """
        before = len(self._tasks)
        self._tasks = [t for t in self._tasks if t["id"] != task_id]
        removed = len(self._tasks) < before
        if removed:
            logger.info(f"[TASKS] Removed task id={task_id}")
        return removed  # type: ignore

    def mark_done(self, task_id: int):  # type: ignore
        """
        Mark a task as completed.
        SHORT tasks are auto-removed immediately.
        LONG_RUNNING tasks remain active until explicitly removed.
        """
        for t in self._tasks:
            if t["id"] == task_id:
                t["answered"] = True
                if t["type"] == "SHORT":
                    self.remove_task(task_id)
                    logger.info(f"[TASKS] SHORT task id={task_id} answered & removed")
                elif t["type"] == "TEMP_CONTEXT":
                    self.remove_task(task_id)
                    logger.info(f"[TASKS] TEMP_CONTEXT task id={task_id} answered & removed")
                else:
                    t["status"] = "done"
                    logger.info(f"[TASKS] LONG_RUNNING task id={task_id} marked done (kept)")
                return

    def update_task_state(self, task_id: int, status: str):  # type: ignore
        """
        Update the status field of a task.
        Valid statuses: "pending" | "active" | "paused" | "done"
        """
        for t in self._tasks:
            if t["id"] == task_id:
                t["status"] = status
                logger.debug(f"[TASKS] id={task_id} → status={status}")
                return

    # ── Query helpers ─────────────────────────────────────────────────

    def get_active_tasks(self, subsystem: str | None = None) -> list:  # type: ignore
        """
        Return all non-done active tasks, optionally filtered by subsystem.
        Sorted by priority ascending (1 = highest).
        """
        tasks = [t for t in self._tasks if t["status"] in ("active", "pending")]
        if subsystem:
            tasks = [t for t in tasks if t["subsystem"] == subsystem]
        return sorted(tasks, key=lambda t: t["priority"])  # type: ignore

    def get_highest_priority_task(self) -> dict | None:  # type: ignore
        """
        Returns the single highest-priority active task, or None.
        """
        active = self.get_active_tasks()
        return active[0] if active else None  # type: ignore

    def get_tasks_for_subsystem(self, subsystem: str) -> list:  # type: ignore
        """
        Returns all active tasks that should be executed by a given subsystem.
        """
        return self.get_active_tasks(subsystem=subsystem)  # type: ignore

    def has_navigation_task(self) -> bool:  # type: ignore
        return any(t["subsystem"] == "navigation" and t["status"] == "active"
                   for t in self._tasks)

    def get_navigation_goal(self) -> str | None:  # type: ignore
        """Extract the navigation goal string from the active nav task."""
        for t in self._tasks:
            if t["subsystem"] == "navigation" and t["status"] == "active":
                # Description is "Navigate to <goal>"
                desc = t["description"]
                prefixes = ("Navigate to ", "navigate to ", "Guide to ", "guide to ")
                for pfx in prefixes:
                    if desc.startswith(pfx):
                        return desc[len(pfx):]
                return desc
        return None  # type: ignore

    def complete_navigation(self):  # type: ignore
        """
        Called when user reaches the destination.
        Removes navigation task and all associated long-running tasks.
        Preserves any TEMP_CONTEXT tasks.
        """
        to_remove = [
            t["id"] for t in self._tasks
            if t["subsystem"] in ("navigation", "safety", "scene")
            and t["type"] == "LONG_RUNNING"
        ]
        for task_id in to_remove:
            self.remove_task(task_id)
        logger.info(f"[TASKS] Navigation complete — cleared {len(to_remove)} tasks")

    # ── Interruption handling ─────────────────────────────────────────

    def interrupt_current(self):  # type: ignore
        """
        User spoke mid-navigation. Pause all active tasks and push them
        onto the interrupted stack to resume later.
        Called before processing a new user request.
        """
        active = [t for t in self._tasks if t["status"] == "active"]
        for t in active:
            t["status"] = "paused"
        self._interrupted.extend(active)
        logger.info(f"[TASKS] Interrupted {len(active)} tasks → paused")

    def resume_interrupted(self):  # type: ignore
        """
        Resume all previously interrupted tasks after handling the
        user's interrupt request.
        """
        for t in self._interrupted:
            t["status"] = "active"
        count = len(self._interrupted)
        self._interrupted.clear()
        logger.info(f"[TASKS] Resumed {count} interrupted tasks")

    def has_interrupted(self) -> bool:  # type: ignore
        return len(self._interrupted) > 0

    # ── LLM-based extraction ──────────────────────────────────────────

    def extract_tasks(self, transcript: str) -> dict:  # type: ignore
        """
        Use Gemini to parse a voice transcript into a structured task spec.

        Returns:
            {
              "intent":           str,
              "navigation_goal":  str | None,
              "tasks": [
                { "type", "description", "subsystem" }, ...
              ]
            }

        Falls back to a safe chat response on any error.
        """
        if not transcript or not transcript.strip():
            return self._fallback(transcript or "")

        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=f'User said: "{transcript}"',
                config=genai.types.GenerateContentConfig(
                    system_instruction=TASK_EXTRACT_PROMPT,
                    temperature=0.05,
                    max_output_tokens=250,
                    response_mime_type="application/json",
                ),
            )
            parsed = json.loads(response.text)  # type: ignore

            # Normalise fields
            parsed.setdefault("intent", "chat")
            parsed.setdefault("navigation_goal", None)
            parsed.setdefault("tasks", [])

            logger.info(
                f"[TASKS] Extracted intent={parsed['intent']} "
                f"goal={parsed['navigation_goal']} "
                f"tasks={[t['description'][:30] for t in parsed['tasks']]}"
            )
            return parsed  # type: ignore

        except Exception as e:
            logger.error(f"[TASKS] Extraction error: {e}")
            return self._fallback(transcript)

    def apply_extracted(self, parsed: dict) -> list:  # type: ignore
        """
        Apply a parsed task spec from extract_tasks() to the engine.
        Creates all tasks, returns list of created task dicts.

        Also handles interruption: if nav is already active and a new
        request comes in, existing tasks are interrupted first.
        """
        created = []
        intent  = parsed.get("intent", "chat")

        # Interruption: new navigation while navigating → interrupt then restart
        if intent == "navigate" and self.has_navigation_task():
            logger.info("[TASKS] New nav goal while navigating → interrupting current")
            self.interrupt_current()

        for t_spec in parsed.get("tasks", []):
            task = self.add_task(
                description = t_spec.get("description", "Unknown task"),
                task_type   = t_spec.get("type", "SHORT"),
                subsystem   = t_spec.get("subsystem", "conversation"),
            )
            created.append(task)

        return created  # type: ignore

    def snapshot(self) -> dict:  # type: ignore
        """
        Return a full JSON-serialisable snapshot of the task engine state.
        Useful for debugging, logging, and storing in session memory.
        """
        return {
            "session_id":   self.session_id,
            "tasks":        list(self._tasks),
            "interrupted":  list(self._interrupted),
            "nav_task_id":  self._nav_task_id,
        }

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _infer_priority(task_type: str, subsystem: str) -> int:  # type: ignore
        if subsystem == "safety":
            return PRIORITY["safety"]
        if subsystem == "navigation":
            return PRIORITY["navigation"]
        if subsystem in ("scene",) and task_type == "SHORT":
            return PRIORITY["query"]
        return PRIORITY["informational"]

    def _is_duplicate(self, description: str) -> bool:  # type: ignore
        """
        Simple duplicate check: is this exact task already active?
        Prevents re-registering "warn about poles" after every utterance.
        """
        key = description.lower().strip()
        return any(
            t["description"].lower().strip() == key
            and t["status"] in ("active", "pending")
            for t in self._tasks
        )

    @staticmethod
    def _fallback(transcript: str) -> dict:  # type: ignore
        return {
            "intent":          "chat",
            "navigation_goal": None,
            "tasks": [
                {
                    "type":        "TEMP_CONTEXT",
                    "description": transcript or "unknown",
                    "subsystem":   "conversation",
                }
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL CONVENIENCE (backwards compatibility for websocket_server.py)
# ─────────────────────────────────────────────────────────────────────────────

# Per-session engine registry
_engines: dict = {}


def get_engine(session_id: str) -> "TaskEngine":  # type: ignore
    """
    Get or create the TaskEngine for a session.
    This is the recommended integration point.
    """
    if session_id not in _engines:
        _engines[session_id] = TaskEngine(session_id)
    return _engines[session_id]


def extract_task(transcript: str) -> dict:  # type: ignore
    """
    Legacy single-call shorthand. Returns backwards-compatible dict with:
      intent, navigation_goal, long_running_tasks, short_tasks, question, text

    Internally uses TaskEngine.extract_tasks() for extraction.
    """
    engine = TaskEngine("_temp")
    parsed = engine.extract_tasks(transcript)

    # Map back to old format for websocket_server compatibility
    long_running = [
        t["description"] for t in parsed.get("tasks", [])  # type: ignore
        if (t if isinstance(t, dict) else {}).get("type") == "LONG_RUNNING" and (t if isinstance(t, dict) else {}).get("subsystem") != "navigation"
    ]
    short = [
        t["description"] for t in parsed.get("tasks", [])  # type: ignore
        if (t if isinstance(t, dict) else {}).get("type") == "SHORT"
    ]
    question = short[0] if short else None

    return {
        "intent":              parsed["intent"],
        "goal":                parsed.get("navigation_goal"),
        "navigation_goal":     parsed.get("navigation_goal"),
        "long_running_tasks":  long_running,
        "short_tasks":         short,
        "question":            question,
        "text":                transcript if parsed["intent"] in ("chat", "interrupt") else None,
        # New field: pass the full parsed task list for callers that use TaskEngine
        "_tasks":              parsed.get("tasks", []),
    }
