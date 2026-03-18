"""
config.py

Centralized configuration for the Vision Assistant.
Loads environment variables and provides typed access to all settings.
"""

import os  # type: ignore
from dataclasses import dataclass, field  # type: ignore
from dotenv import load_dotenv  # type: ignore

load_dotenv()


@dataclass
class VisionConfig:
    """YOLO and camera settings."""
    model_path: str = "yolov8n.pt"
    infer_width: int = 320
    infer_height: int = 240
    confidence_threshold: float = 0.30
    danger_threshold: float = 0.55       # bbox height fraction → <1m
    caution_threshold: float = 0.35      # bbox height fraction → 1-2m


@dataclass
class SchedulerConfig:
    """Dynamic frame scheduler tuning."""
    walk_speed_mps: float = 1.2
    camera_fov_max_m: float = 25.0
    min_gap_sec: float = 6.0             # hard minimum between LLM calls
    keepalive_sec: float = 20.0          # max silence before forced call
    min_clear_m: float = 10.0
    nav_turn_threshold_m: float = 20.0
    crowd_threshold: int = 5


@dataclass
class LLMConfig:
    """LLM/VLM configuration."""
    provider: str = "gemini"             # "gemini" or "openai"
    model_name: str = "gemini-2.5-flash"
    api_key: str = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY", ""))
    temperature: float = 0.3
    max_output_tokens: int = 1024
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))


@dataclass
class NavigationConfig:
    """Google Maps navigation settings."""
    google_maps_api_key: str = field(
        default_factory=lambda: os.environ.get("GOOGLE_MAPS_API_KEY", "")
    )
    announce_thresholds: list = field(default_factory=lambda: [30, 10, 5, 2])


@dataclass
class SpeechConfig:
    """Speech processing settings."""
    whisper_model: str = "base"
    tts_voice: str = "en-US-AriaNeural"  # Edge TTS voice
    tts_rate: str = "+10%"               # slightly faster for urgency
    tts_volume: str = "+0%"


@dataclass
class ServerConfig:
    """Server and transport settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"


@dataclass
class AppConfig:
    """Top-level configuration container."""
    vision: VisionConfig = field(default_factory=VisionConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


# ── Singleton ────────────────────────────────────────────────────────────────
_config: AppConfig | None = None


def get_config() -> AppConfig:  # type: ignore
    """Return the global AppConfig singleton."""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config  # type: ignore
