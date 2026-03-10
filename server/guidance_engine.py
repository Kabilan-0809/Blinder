import logging

logger = logging.getLogger(__name__)


def fuse_guidance(
    safety_alert: str | None,
    vision_instruction: str | None,
    navigation_instruction: str | None
) -> dict:
    """
    The Guidance Fusion Engine.

    Combines the three instruction streams using strict priority ordering:
        Priority 1: Immediate Safety Alerts (STOP, Step Down, Vehicle Approaching)
        Priority 2: Vision Guidance (Pole ahead, Move left, Person approaching)
        Priority 3: Route Navigation (Turn right in 10 meters)

    Never speaks two things at once; always picks the highest-priority available.

    Returns a structured output dict:
    {
        "instruction": "<primary spoken instruction>",
        "navigation": "<current route step or None>"
    }
    """
    spoken = None
    nav_text = navigation_instruction  # always carry the nav instruction for UI display

    # Priority 1: Safety takes absolute precedence
    if safety_alert:
        spoken = safety_alert
        logger.warning(f"[FUSE] Safety override: {spoken}")

    # Priority 2: Vision guidance, only if no immediate safety alert
    elif vision_instruction:
        spoken = vision_instruction
        logger.info(f"[FUSE] Vision guidance: {spoken}")

    # Priority 3: GPS route instruction is the default if vision is clear
    elif navigation_instruction:
        spoken = navigation_instruction
        logger.info(f"[FUSE] Route guidance: {spoken}")

    return {
        "instruction": spoken,
        "navigation": nav_text
    }
