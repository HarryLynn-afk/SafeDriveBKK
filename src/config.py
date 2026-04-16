"""
Centralised configuration: file paths, environment variables, and constants.
Import from here instead of scattering os.getenv / Path(...) calls across modules.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

load_dotenv(BASE_DIR / ".env")

# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------
_LEGACY_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
GOOGLE_MAPS_BROWSER_KEY: str = os.getenv("GOOGLE_MAPS_BROWSER_KEY") or _LEGACY_KEY or ""
GOOGLE_MAPS_SERVER_KEY: str = os.getenv("GOOGLE_MAPS_SERVER_KEY") or _LEGACY_KEY or ""

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
BANGKOK_ACCIDENTS_PATH = DATA_DIR / "bangkok_accidents_clean.csv"
GRID_STATS_PATH = DATA_DIR / "bangkok_grid_stats.csv"
RISK_MAP_HTML_PATH = DATA_DIR / "bangkok_risk_map.html"
RISK_HEATMAP_HTML_PATH = BASE_DIR / "static" / "bangkok_risk_heatmap.html"
ROUTE_PRESETS_PATH = BASE_DIR / "static" / "route_presets.json"
NLP_CACHE_PATH = BASE_DIR / "static" / "nlp_cache.json"

# ---------------------------------------------------------------------------
# External service URLs
# ---------------------------------------------------------------------------
OSRM_BASE_URL = "https://router.project-osrm.org"

# ---------------------------------------------------------------------------
# Bangkok spatial grid
# ---------------------------------------------------------------------------
BKK_LAT_MIN = 13.6
BKK_LAT_MAX = 13.95
BKK_LON_MIN = 100.35
BKK_LON_MAX = 100.9
CELL_SIZE_DEG = 0.009

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------
DAY_NAME_MAP: dict[int, str] = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}

# Approximate district centroids for nearest-neighbour display in analytics.
BKK_DISTRICTS: list[tuple[str, float, float]] = [
    ("Chatuchak",       13.8038, 100.5503),
    ("Din Daeng",       13.7698, 100.5523),
    ("Don Mueang",      13.9101, 100.5993),
    ("Dusit",           13.7831, 100.5103),
    ("Huai Khwang",     13.7747, 100.5733),
    ("Khan Na Yao",     13.8200, 100.6700),
    ("Khlong Sam Wa",   13.8715, 100.7130),
    ("Khlong San",      13.7324, 100.4962),
    ("Khlong Toei",     13.7186, 100.5601),
    ("Lak Si",          13.8820, 100.5700),
    ("Lat Krabang",     13.7250, 100.7490),
    ("Lat Phrao",       13.8090, 100.5714),
    ("Min Buri",        13.8018, 100.7482),
    ("Nong Chok",       13.8580, 100.8440),
    ("Nong Khaem",      13.6908, 100.4074),
    ("On Nut",          13.7007, 100.6020),
    ("Pathum Wan",      13.7440, 100.5297),
    ("Phasi Charoen",   13.7120, 100.4537),
    ("Phra Khanong",    13.7071, 100.6010),
    ("Phra Nakhon",     13.7543, 100.4977),
    ("Phra Pradaeng",   13.6603, 100.5295),
    ("Pom Prap",        13.7540, 100.5190),
    ("Prawet",          13.6779, 100.6534),
    ("Rat Burana",      13.6757, 100.5066),
    ("Ratchathewi",     13.7564, 100.5347),
    ("Sai Mai",         13.9106, 100.6590),
    ("Saphan Sung",     13.7550, 100.6780),
    ("Sathon",          13.7195, 100.5267),
    ("Suan Luang",      13.7336, 100.6266),
    ("Taling Chan",     13.7709, 100.4445),
    ("Thawi Watthana",  13.7578, 100.3955),
    ("Thon Buri",       13.7320, 100.4872),
    ("Thung Khru",      13.6461, 100.4912),
    ("Wang Thonglang",  13.7659, 100.5956),
    ("Watthana",        13.7380, 100.5600),
    ("Yan Nawa",        13.6974, 100.5310),
]

ROUTE_DISPLAY_NAMES: dict[str, str] = {
    "7":   "Motorway 7",
    "9":   "Outer Ring Rd 9",
    "31":  "Vibhavadi Rangsit Rd",
    "35":  "Rama II Rd",
    "338": "Borommaratchachonnani Rd",
    "304": "Suwinthawong Rd",
}
