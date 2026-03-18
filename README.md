# Conversational AI Vision Assistant for the Blind

A real-time, multimodal AI assistant designed to guide visually impaired users by analyzing camera input and providing conversational spoken guidance.

This project implements a **production-grade modular AI agent pipeline**.

## Architecture Overview

The system is organized into domain-specific packages that run asynchronously to provide low-latency safety alerts alongside high-level environmental reasoning.

### Folder Structure
```text
server/
├── agent/            # Core central orchestrator and task management
├── vision/           # Real-time YOLO obstacle and position detection
├── navigation/       # GPS and Google Maps turn-by-turn routing
├── scheduler/        # Intelligent LLM call gating
├── reasoning/        # VLM scene analysis (Gemini/OpenAI)
├── speech/           # Audio processing (Whisper STT, Edge TTS)
├── personality/      # Iris companion persona and conversational engine
├── camera/           # OpenCV async video stream handling
├── tests/            # Simulation tools
├── utils/            # Config loading and structured logging
├── main.py           # FastAPI entry point
└── websocket_server.py # Thin WebSocket transport layer
```

---

## Core Agent Pipeline

The primary orchestrator is `AgentController` (in `agent/agent_controller.py`), which executes two parallel pipelines triggered by the WebSocket transport layer:

### 1. Frame Processing Pipeline (Video)
Triggered several times per second:
1. **Vision Safety Engine** — YOLO runs on every frame (<100ms) to detect immediate proximity threats.
2. **Dynamic Frame Scheduler** — Decides if the frame should be sent to the Vision Language Model (VLM) based on time elapsed, detected intersections, or active questions.
3. **Scene Reasoner** — Multimodal VLM extracts structured context (crowd density, signs, decision points).
4. **Environment Memory** — Session state is updated.
5. **Navigation Guidance** — Provides progress updates if a goal is active.
6. **Instruction Fusion** — Priority-merges outputs (Safety > Navigation > Conversational).

### 2. Audio Processing Pipeline (Voice)
Triggered when the user speaks:
1. **Speech-to-Text** — Whisper local transcription.
2. **Task Extraction** — LLM extracts intent (navigate, query, chat, pause) and distinct tasks.
3. **Task Manager** — Long-running and temporary tasks are registered.
4. **Response Generation** — AI companion response is generated.

---

## Key Modules

- **Dynamic Frame Scheduler**: Prevents API spam by gating VLM calls (targeting ~3-10 calls per minute) while ensuring critical moments (like intersections) are captured.
- **Vision Safety Engine**: An "always-on" local YOLO model providing a safety net that bypasses the latency of cloud LLMs.
- **Environment Memory**: Fully isolated per-session memory storing active tasks, navigation progress, and recent scene descriptions.
- **Instruction Fusion**: Ensures critical safety warnings interrupt conversational responses.
- **Simulation Runner**: Allows full pipeline testing against pre-recorded videos without needing a live camera.

## Setup & Running

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure `.env`:
   ```env
   GEMINI_API_KEY=your_key_here
   GOOGLE_MAPS_API_KEY=your_key_here
   ```
3. Start the server:
   ```bash
   uvicorn server.main:app --host 0.0.0.0 --port 8000
   ```

To run a simulation on a video file:
```bash
python -m tests.simulation_runner path/to/video.mp4 --goal "navigate to park"
```