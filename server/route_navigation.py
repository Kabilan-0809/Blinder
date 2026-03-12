import os  # type: ignore
import math  # type: ignore
import logging  # type: ignore
import requests  # type: ignore
from html.parser import HTMLParser  # type: ignore
from dotenv import load_dotenv  # type: ignore

load_dotenv()
logger = logging.getLogger(__name__)

GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")

# Distance thresholds (in meters) at which to remind the user about upcoming turns
ANNOUNCE_DISTANCES = [10, 5, 2]


class _HTMLStripper(HTMLParser):  # type: ignore
    """Helper to strip HTML tags from Google Maps instruction strings like <b>Turn left</b>."""
    def __init__(self):  # type: ignore
        super().__init__()
        self.text = []

    def handle_data(self, data):  # type: ignore
        self.text.append(data)

    def strip(self, html: str) -> str:  # type: ignore
        self.feed(html)
        return " ".join(self.text).strip()  # type: ignore


def _haversine_distance(lat1, lng1, lat2, lng2) -> float:  # type: ignore
    """Returns the straight-line distance between two GPS coordinates in meters."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))  # type: ignore


def load_route(start_location: dict, destination: str, session_state: dict) -> bool:  # type: ignore
    """
    Fetches a walking route from Google Directions API.
    Parses the polyline steps into a list of {lat, lng, instruction, distance_m}.
    Stores them in session_state["route_steps"].

    Args:
        start_location: {"lat": float, "lng": float}
        destination:    Free-text address string OR "lat,lng"
        session_state:  The mutable dict from memory.py stored for this session.

    Returns:
        True if route was successfully loaded, False on error.
    """
    if not GOOGLE_MAPS_API_KEY:
        logger.error("[ROUTE] GOOGLE_MAPS_API_KEY not set.")
        return False  # type: ignore

    origin = f"{start_location['lat']},{start_location['lng']}"

    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": origin,
        "destination": destination,
        "mode": "walking",
        "key": GOOGLE_MAPS_API_KEY
    }

    try:
        resp = requests.get(url, params=params, timeout=5)
        data = resp.json()
    except Exception as e:
        logger.error(f"[ROUTE] Directions API request failed: {e}")
        return False  # type: ignore

    if data.get("status") != "OK":
        logger.error(f"[ROUTE] API returned status: {data.get('status')}")
        return False  # type: ignore

    steps = []
    stripper = _HTMLStripper()

    for leg in data["routes"][0]["legs"]:
        for step in leg["steps"]:
            end_loc = step["end_location"]
            raw_instruction = step.get("html_instructions", "Continue")
            clean_instruction = stripper.strip(raw_instruction)
            steps.append({
                "lat": end_loc["lat"],
                "lng": end_loc["lng"],
                "instruction": clean_instruction,
                "distance_m": step["distance"]["value"]
            })

    session_state["route_steps"] = steps
    session_state["current_route_step"] = 0
    session_state["last_nav_distance_announced"] = None
    logger.info(f"[ROUTE] Loaded route with {len(steps)} steps.")
    return True  # type: ignore


def get_next_navigation_step(user_location: dict, session_state: dict) -> str | None:  # type: ignore
    """
    Compares the user's current GPS position with the next turn.
    Fires an instruction at 10m, 5m, and 2m announcement thresholds.

    Args:
        user_location:  {"lat": float, "lng": float}
        session_state:  The mutable session dict from memory.py.

    Returns:
        A short instruction string, or None if no announcement is needed yet.
    """
    steps = session_state.get("route_steps", [])
    step_idx = session_state.get("current_route_step", 0)

    if not steps or step_idx >= len(steps):
        return None  # type: ignore

    current_step = steps[step_idx]

    dist = _haversine_distance(
        user_location["lat"], user_location["lng"],
        current_step["lat"], current_step["lng"]
    )

    session_state["distance_to_next_turn"] = round(dist, 1)  # type: ignore

    # Determine which announce band we're in (10m, 5m, 2m)
    trigger_dist = None
    for threshold in ANNOUNCE_DISTANCES:
        if dist <= threshold:
            trigger_dist = threshold
            break

    if trigger_dist is None:
        return None  # type: ignore

    # Suppress repeat for the same threshold band
    if session_state.get("last_nav_distance_announced") == trigger_dist:
        return None  # type: ignore

    session_state["last_nav_distance_announced"] = trigger_dist

    # Advance to next step if we've passed this waypoint
    if dist <= 2:
        session_state["current_route_step"] = step_idx + 1
        session_state["last_nav_distance_announced"] = None
        return f"{current_step['instruction']} now."  # type: ignore

    return f"In {trigger_dist} meters, {current_step['instruction'].lower()}."  # type: ignore
