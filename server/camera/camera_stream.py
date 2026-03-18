"""
camera_stream.py

Async camera frame producer using OpenCV.

Captures frames from a live camera or video file and yields them as
JPEG bytes for processing by the vision pipeline.

Supports:
  - Live camera (webcam, USB camera)
  - Video file playback (for testing / simulation)
  - Configurable FPS limiting
  - Async iteration via async generator

Usage:
    from camera.camera_stream import CameraStream

    # Live camera
    cam = CameraStream(source=0)

    # Video file
    cam = CameraStream(source="test_corridor.mp4")

    async for jpeg_bytes in cam.stream():
        result = vision_safety_engine.run_safety_check(jpeg_bytes)
"""

import cv2  # type: ignore
import asyncio  # type: ignore
import time  # type: ignore
import logging  # type: ignore
import numpy as np  # type: ignore
from typing import AsyncGenerator  # type: ignore

logger = logging.getLogger("camera")


class CameraStream:
    """
    Async camera frame producer.

    Captures from a live camera or video file, encodes as JPEG,
    and yields frames at a configurable rate.
    """

    def __init__(
        self,
        source: int | str = 0,
        *,
        target_fps: float = 10.0,
        jpeg_quality: int = 70,
        resolution: tuple = (640, 480),
    ):
        """
        Args:
            source:       Camera index (0, 1, ...) or video file path
            target_fps:   Maximum frames per second to yield
            jpeg_quality:  JPEG compression quality (0-100)
            resolution:   Target capture resolution (width, height)
        """
        self.source = source
        self.target_fps = target_fps
        self.jpeg_quality = jpeg_quality
        self.resolution = resolution
        self._cap: cv2.VideoCapture | None = None
        self._running = False
        self._frame_count = 0
        self._start_time = 0.0

    def open(self) -> bool:  # type: ignore
        """Open the camera/video source."""
        self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            logger.error(f"[CAM] Failed to open source: {self.source}")
            return False  # type: ignore

        # Set resolution for live cameras
        if isinstance(self.source, int):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(
            f"[CAM] Opened source={self.source} "
            f"resolution={actual_w}x{actual_h} target_fps={self.target_fps}"
        )
        self._running = True
        self._frame_count = 0
        self._start_time = time.time()
        return True  # type: ignore

    def close(self):  # type: ignore
        """Release the camera."""
        self._running = False
        if self._cap and self._cap.isOpened():
            self._cap.release()
            elapsed = time.time() - self._start_time
            fps = self._frame_count / max(elapsed, 0.001)
            logger.info(
                f"[CAM] Closed. {self._frame_count} frames in {elapsed:.1f}s "
                f"({fps:.1f} fps)"
            )
        self._cap = None

    def capture_frame(self) -> bytes | None:  # type: ignore
        """
        Capture a single frame and return as JPEG bytes.
        Returns None if capture fails.
        """
        if not self._cap or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        ok, buf = cv2.imencode(".jpg", frame, encode_params)
        if not ok:
            return None

        self._frame_count += 1
        return buf.tobytes()  # type: ignore

    async def stream(self) -> AsyncGenerator[bytes, None]:  # type: ignore
        """
        Async generator yielding JPEG frames at target FPS.

        Usage:
            async for jpeg in cam.stream():
                process(jpeg)
        """
        if not self.open():
            return

        frame_interval = 1.0 / self.target_fps

        try:
            while self._running:
                t0 = time.time()

                jpeg = await asyncio.to_thread(self.capture_frame)
                if jpeg is None:
                    if isinstance(self.source, str):
                        logger.info("[CAM] End of video file")
                        break
                    logger.warning("[CAM] Frame capture failed")
                    await asyncio.sleep(0.1)
                    continue

                yield jpeg

                # Rate limiting
                elapsed = time.time() - t0
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        finally:
            self.close()

    @property
    def is_running(self) -> bool:  # type: ignore
        return self._running

    @property
    def frame_count(self) -> int:  # type: ignore
        return self._frame_count

    def __enter__(self):  # type: ignore
        self.open()
        return self

    def __exit__(self, *args):  # type: ignore
        self.close()
