"""personality/ — Iris companion personality and conversational AI."""
from personality.companion_personality import (  # type: ignore
    get_welcome_message, get_farewell_message, detect_mood,
    get_ambient_observation, build_personality_context,
)
from personality.conversation_engine import (  # type: ignore
    generate_guidance, answer_question, handle_chat,
    generate_arrival, check_long_running_tasks,
)
