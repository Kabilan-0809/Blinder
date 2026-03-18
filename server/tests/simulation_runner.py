"""
simulation_runner.py

Test the full AI agent pipeline using pre-recorded video files
instead of a live camera feed.

Usage:
    python -m tests.simulation_runner path/to/video.mp4
    python -m tests.simulation_runner path/to/video.mp4 --goal "navigate to exit"
    python -m tests.simulation_runner --dry-run

This simulates the full pipeline:
    Camera Frame → Safety → Scheduler → VLM → Memory → Fusion → TTS
"""

import sys  # type: ignore
import os  # type: ignore
import time  # type: ignore
import asyncio  # type: ignore
import argparse  # type: ignore
import logging  # type: ignore
import json  # type: ignore

# Ensure server/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger("simulation")


class SimulationStats:
    """Track simulation performance metrics."""

    def __init__(self):  # type: ignore
        self.frames_processed = 0
        self.safety_alerts = 0
        self.llm_calls = 0
        self.nav_instructions = 0
        self.total_objects_detected = 0
        self.start_time = time.time()

    def report(self) -> str:  # type: ignore
        elapsed = time.time() - self.start_time
        fps = self.frames_processed / max(elapsed, 0.001)
        return (
            f"\n{'='*60}\n"
            f"  SIMULATION REPORT\n"
            f"{'='*60}\n"
            f"  Duration:            {elapsed:.1f}s\n"
            f"  Frames processed:    {self.frames_processed}\n"
            f"  Effective FPS:       {fps:.1f}\n"
            f"  Safety alerts:       {self.safety_alerts}\n"
            f"  LLM calls:           {self.llm_calls}\n"
            f"  Nav instructions:    {self.nav_instructions}\n"
            f"  Objects detected:    {self.total_objects_detected}\n"
            f"{'='*60}\n"
        )


async def run_simulation(
    video_path: str,
    goal: str = "",
    max_frames: int = 300,
    target_fps: float = 5.0,
    dry_run: bool = False,
) -> SimulationStats:  # type: ignore
    """
    Run the full agent pipeline against a video file.

    Args:
        video_path:  Path to video file (MP4, AVI, etc.)
        goal:        Optional navigation goal string
        max_frames:  Maximum frames to process
        target_fps:  Processing rate (frames per second)
        dry_run:     If True, only test imports without processing

    Returns:
        SimulationStats with performance metrics
    """
    stats = SimulationStats()

    if dry_run:
        logger.info("DRY RUN — testing imports only")
        _test_imports()
        return stats

    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return stats

    # Import agent modules
    from camera.camera_stream import CameraStream  # type: ignore
    from vision.vision_safety_engine import run_safety_check  # type: ignore
    from scheduler.dynamic_frame_scheduler import DynamicFrameScheduler  # type: ignore
    from agent.environment_memory import get_memory, update_scene  # type: ignore
    from agent.instruction_fusion import fuse  # type: ignore

    session_id = f"sim-{int(time.time())}"
    cam = CameraStream(source=video_path, target_fps=target_fps)
    scheduler = DynamicFrameScheduler()
    mem = get_memory(session_id)

    if goal:
        mem["navigation_goal"] = goal
        mem["task_status"] = "active"
        logger.info(f"🗺️ Goal set: '{goal}'")

    logger.info(f"▶️  Starting simulation: {video_path}")
    logger.info(f"   Max frames: {max_frames}, FPS: {target_fps}")

    last_spoken = 0.0

    async for jpeg_bytes in cam.stream():
        if stats.frames_processed >= max_frames:
            break

        stats.frames_processed += 1
        frame_num = stats.frames_processed

        # ── 1. Safety check (always runs) ────────────────────────────
        try:
            safety = await asyncio.to_thread(run_safety_check, jpeg_bytes, True)
        except Exception as e:
            logger.error(f"[F{frame_num}] Safety error: {e}")
            continue

        n_objects = len(safety.get("objects", []))
        stats.total_objects_detected += n_objects

        if safety.get("alert"):
            stats.safety_alerts += 1
            logger.warning(f"[F{frame_num}] 🚨 {safety['alert']}")

        # ── 2. Scheduler decision ────────────────────────────────────
        sched = scheduler.should_send_to_llm(
            pending_question=None,
            yolo_result=safety,
            session_memory=mem,
            force=(frame_num == 1),
        )

        if sched.get("should_send"):
            stats.llm_calls += 1
            logger.info(
                f"[F{frame_num}] 👁️ LLM triggered ({sched['reason']}) "
                f"— {n_objects} objects"
            )

            # In simulation, we skip actual LLM call but log what would happen
            logger.info(f"[F{frame_num}]    → Would call scene_reasoner.analyze_frame()")
        else:
            if frame_num % 30 == 0:  # Log every 30th skip
                logger.debug(f"[F{frame_num}] ⏭️ Skipped ({sched['reason']})")

        # ── 3. Fusion ────────────────────────────────────────────────
        fused = fuse(
            safety_alert=safety.get("alert"),
            last_spoken_time=last_spoken,
        )
        if fused.get("should_speak") and fused.get("text"):
            last_spoken = time.time()
            logger.info(f"[F{frame_num}] 🔊 SPEAK: '{fused['text']}'")

    logger.info(stats.report())
    return stats


def _test_imports():  # type: ignore
    """Verify all modules can be imported."""
    modules = [
        ("utils.config", "get_config"),
        ("utils.logger", "get_logger"),
        ("vision.object_position", "get_object_position"),
        ("vision.distance_estimator", "estimate_distance"),
        ("reasoning.llm_interface", "LLMClient"),
        ("agent.task_manager", "TaskEngine"),
        ("agent.environment_memory", "get_memory"),
        ("agent.instruction_fusion", "fuse"),
        ("scheduler.dynamic_frame_scheduler", "DynamicFrameScheduler"),
        ("navigation.navigation_engine", "load_route"),
        ("personality.companion_personality", "detect_mood"),
    ]

    for mod_path, attr_name in modules:
        try:
            mod = __import__(mod_path, fromlist=[attr_name])
            obj = getattr(mod, attr_name)
            print(f"  ✅ {mod_path}.{attr_name}")
        except Exception as e:
            print(f"  ❌ {mod_path}.{attr_name} — {e}")

    print("\nImport test complete.")


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def main():  # type: ignore
    parser = argparse.ArgumentParser(
        description="Run Vision Assistant simulation on a video file"
    )
    parser.add_argument(
        "video", nargs="?", default=None,
        help="Path to video file (MP4, AVI)",
    )
    parser.add_argument("--goal", default="", help="Navigation goal")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true", help="Test imports only")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.dry_run:
        _test_imports()
        return

    if not args.video:
        parser.error("Video path required (or use --dry-run)")

    asyncio.run(
        run_simulation(
            args.video,
            goal=args.goal,
            max_frames=args.max_frames,
            target_fps=args.fps,
        )
    )


if __name__ == "__main__":
    main()
