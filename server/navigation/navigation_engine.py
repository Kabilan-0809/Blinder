"""
navigation_engine.py

Handles all Google Maps interaction:
  - resolve_place(): Google Places API text search → lat/lng
  - load_route():    Google Directions API → list of steps
  - get_next_navigation_step(): compares user GPS vs route, fires turn instructions
"""

import os  # type: ignore
import math  # type: ignore
import logging  # type: ignore
import requests  # type: ignore
from html.parser import HTMLParser  # type: ignore
from dotenv import load_dotenv  # type: ignore

load_dotenv()
logger = logging.getLogger(__name__)

GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")

# Distance thresholds (meters) at which to announce upcoming turn
ANNOUNCE_THRESHOLDS = [30, 10, 5, 2]


# ─────────────────────────────────────────────────────────────
# HTML tag stripper (Google Directions returns HTML instructions)
# ─────────────────────────────────────────────────────────────

class _HTMLStripper(HTMLParser):  # type: ignore
    def __init__(self):  # type: ignore
        super().__init__()
        self._parts: list = []

    def handle_data(self, data):  # type: ignore
        self._parts.append(data)

    def get_text(self) -> str:  # type: ignore
        return " ".join(self._parts).strip()


def _strip_html(html: str) -> str:  # type: ignore
    s = _HTMLStripper()
    s.feed(html)
    return s.get_text()  # type: ignore


# ─────────────────────────────────────────────────────────────
# Haversine distance
# ─────────────────────────────────────────────────────────────

def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:  # type: ignore
    """Returns straight-line distance in meters between two GPS coords."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return float(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


# ─────────────────────────────────────────────────────────────
# Google Places — resolve free-text destination to lat/lng
# ─────────────────────────────────────────────────────────────

def resolve_place(destination: str) -> dict:  # type: ignore
    """
    Uses Google Places Text Search to resolve a natural-language destination
    like "Surya Super Market" into { lat, lng, name }.
    Falls back to treating destination as a raw address if Places call fails.
    """
    if not GOOGLE_MAPS_API_KEY:
        logger.warning("[NAV] No API key — skipping Places resolve, using raw text.")
        return {"lat": None, "lng": None, "name": destination}

    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": destination, "key": GOOGLE_MAPS_API_KEY}
    try:
        resp = requests.get(url, params=params, timeout=5)
        data = resp.json()
        if data.get("status") == "OK" and data.get("results"):
            result = data["results"][0]
            loc = result["geometry"]["location"]
            name = result.get("name", destination)
            logger.info(f"[NAV] Place resolved: '{destination}' → '{name}' @ {loc}")
            return {"lat": loc["lat"], "lng": loc["lng"], "name": name}
    except Exception as e:
        logger.error(f"[NAV] Places API error: {e}")

    return {"lat": None, "lng": None, "name": destination}


# ─────────────────────────────────────────────────────────────
# Google Directions — fetch and parse walking route
# ─────────────────────────────────────────────────────────────

def load_route(start_location: dict, destination: str, nav_progress: dict) -> bool:  # type: ignore
    """
    Fetches a walking route from Google Directions API.
    Stores steps into nav_progress (the 'navigation_progress' sub-dict of session memory).

    Each step: { lat, lng, instruction, distance_m }

    Returns True on success, False on error.
    """
    if not GOOGLE_MAPS_API_KEY:
        logger.error("[NAV] GOOGLE_MAPS_API_KEY not set.")
        return False  # type: ignore

    # Try to resolve destination to lat/lng first
    place = resolve_place(destination)
    dest_param = f"{place['lat']},{place['lng']}" if place["lat"] else destination

    origin = f"{start_location['lat']},{start_location['lng']}"
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": origin,
        "destination": dest_param,
        "mode": "walking",
        "key": GOOGLE_MAPS_API_KEY,
    }

    try:
        resp = requests.get(url, params=params, timeout=5)
        data = resp.json()
    except Exception as e:
        logger.error(f"[NAV] Directions API request failed: {e}")
        return False  # type: ignore

    if data.get("status") != "OK":
        logger.error(f"[NAV] API returned status: {data.get('status')}")
        return False  # type: ignore

    steps = []
    for leg in data["routes"][0]["legs"]:
        for step in leg["steps"]:
            end_loc = step["end_location"]
            raw = step.get("html_instructions", "Continue")
            steps.append({
                "lat": end_loc["lat"],
                "lng": end_loc["lng"],
                "instruction": _strip_html(raw),
                "distance_m": step["distance"]["value"],
            })

    nav_progress["route_steps"] = steps
    nav_progress["current_step"] = 0
    nav_progress["distance_to_turn"] = None
    nav_progress["last_announced_threshold"] = None
    logger.info(f"[NAV] Route loaded: {len(steps)} steps to '{destination}'")
    return True  # type: ignore


# ─────────────────────────────────────────────────────────────
# Turn-by-turn guidance from GPS
# ─────────────────────────────────────────────────────────────

def get_next_navigation_step(user_location: dict, nav_progress: dict) -> str | None:  # type: ignore
    """
    Compare user GPS position against the current route step.
    Returns a spoken instruction string if an announcement threshold is crossed,
    or None if nothing should be said yet.

    Mutates nav_progress to track progress.
    """
    steps = nav_progress.get("route_steps", [])
    idx = nav_progress.get("current_step", 0)

    if not steps or idx >= len(steps):
        return None  # type: ignore

    step = steps[idx]
    dist = _haversine(
        user_location["lat"], user_location["lng"],
        step["lat"], step["lng"],
    )
    nav_progress["distance_to_turn"] = round(dist, 1)  # type: ignore

    # ── Arrival at waypoint ──────────────────────────────────
    if dist <= 2:
        nav_progress["current_step"] = idx + 1
        nav_progress["last_announced_threshold"] = None
        instruction = step["instruction"]

        # Check if this was the final step
        if nav_progress["current_step"] >= len(steps):
            return "You have arrived at your destination!"  # type: ignore

        next_step = steps[nav_progress["current_step"]]
        return f"{instruction} now. Then, in {next_step['distance_m']} meters, {next_step['instruction'].lower()}."  # type: ignore

    # ── Announcement thresholds ──────────────────────────────
    triggered = None
    for threshold in ANNOUNCE_THRESHOLDS:
        if dist <= threshold:
            triggered = threshold
            break

    if triggered is None:
        return None  # type: ignore

    # Suppress repeat for same threshold
    if nav_progress.get("last_announced_threshold") == triggered:
        return None  # type: ignore

    nav_progress["last_announced_threshold"] = triggered
    instruction = step["instruction"].lower() if triggered > 2 else step["instruction"]
    dist_text = f"{triggered} meters" if triggered >= 5 else f"{int(dist)} meters"
    return f"In {dist_text}, {instruction}."  # type: ignore
