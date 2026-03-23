# SafeDrive BKK — AI-Powered Road Safety System

## What This Project Is

SafeDrive BKK predicts dangerous road zones in Bangkok using Thai government accident data (2,669 Bangkok records from 23,952 nationwide in 2024). It helps drivers find safer routes by combining XGBoost ML, Thai NLP, and interactive data visualization.

**Hackathon:** AIIC2026 (Tracks: NLP + Data Visualization)
**Live URL:** Runs locally via Flask on port 5000

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Flask 3.0 (Python 3.11) |
| ML | XGBoost 2.0.3, scikit-learn 1.8, pandas, numpy |
| NLP | pythainlp 5.0+ (Thai tokenization), TF-IDF + Logistic Regression |
| Maps/Routing | Google Maps API, OSRM, Folium, Leaflet.js |
| Charts | Chart.js, matplotlib |
| Frontend | Bootstrap 5.3, vanilla JavaScript |
| Chatbot | Groq API (LLM), intent detection (pattern matching) |

---

## How to Run

```bash
cd /Users/kyawnainglin/Desktop/Codex/hackathon
source venv/bin/activate
python src/app.py
# Opens at http://localhost:5000
```

Environment variables needed in `.env`:
- `GROQ_API_KEY` — For AI chatbot
- `GOOGLE_MAPS_BROWSER_KEY` — For Maps JavaScript + Places autocomplete in the browser
- `GOOGLE_MAPS_SERVER_KEY` — For server-side routing and geolocation calls
- `GOOGLE_MAPS_API_KEY` — Legacy fallback supported for local/dev compatibility

---

## File Structure

```
hackathon/
├── src/                          # All Python backend code
│   ├── app.py                    # Main Flask app (1,451 lines, 18+ API endpoints)
│   ├── ml_model.py               # RiskPredictor class wrapping XGBoost
│   ├── nlp_analysis.py           # Thai NLP pipeline (pythainlp + TF-IDF)
│   ├── feature_engineering.py    # 44-feature extraction per grid cell
│   ├── prepare_ml_data.py        # Temporal train/test split
│   ├── train_xgboost.py          # XGBoost training script
│   ├── train_random_forest.py    # Random Forest baseline
│   ├── train_text_classifier.py  # Thai cause classifier
│   ├── data_processing.py        # CSV loading, cleaning, Bangkok filtering
│   ├── grid_aggregation.py       # 1km x 1km grid system (276 cells)
│   ├── temporal_analysis.py      # Hour/day accident patterns
│   ├── evidence_metrics.py       # Validation metrics calculation
│   ├── find_danger_zones.py      # Haversine danger zone detection
│   ├── chatbot_data_processor.py # Aggregate data for chatbot
│   ├── chatbot_groq.py           # Groq AI wrapper
│   ├── chatbot_intents.py        # Intent detection + scripted responses
│   ├── create_heatmap_data.py    # Heat map JSON generator
│   ├── create_heatmap_html.py    # Folium interactive map generator
│   └── create_risk_map.py        # Visual risk map creator
│
├── templates/                    # Jinja2 HTML templates
│   ├── base.html                 # Bootstrap layout (navbar, footer)
│   ├── index.html                # Homepage: hero, metrics, heatmap embed
│   ├── trip_planner.html         # Route planner: map, comparison, driving sim
│   ├── analytics.html            # 12+ Chart.js dashboards
│   ├── nlp.html                  # NLP analysis showcase
│   ├── about.html                # Methodology, evidence, responsible AI
│   └── floating_chat.html        # Chat widget modal
│
├── static/
│   ├── css/
│   │   ├── style.css             # Main theme (8.7 KB)
│   │   ├── trip-planner.css      # Trip planner styles (51.7 KB)
│   │   ├── driving-mode.css      # Driving mode styles
│   │   ├── chatbot.css           # Chat widget styles
│   │   └── floating-chat.css     # Floating chat button
│   ├── js/
│   │   ├── app.js                # General app logic, charts, modals
│   │   ├── trip-planner.js       # Route planner + driving simulation (1,800+ lines)
│   │   ├── chatbot.js            # Chat message handling
│   │   └── floating-chat.js      # FAB button logic
│   ├── bangkok_risk_heatmap.html # Pre-rendered Folium map (593 KB)
│   ├── heatmap_data.json         # 2,669 accident coordinate points
│   ├── chatbot_data.json         # Aggregated zone/NLP data for chatbot
│   ├── nlp_cache.json            # Pre-computed NLP results
│   ├── demo_routes.json          # Pre-computed route waypoints
│   └── favicon.svg
│
├── models/                       # Trained ML artifacts
│   ├── xgboost_model.pkl         # Primary risk predictor (124 KB)
│   ├── random_forest.pkl         # Fallback model (328 KB)
│   ├── cause_classifier.pkl      # Thai cause classifier
│   ├── tfidf_vectorizer.pkl      # Text vectorizer
│   ├── xgb_results.json          # XGBoost metrics
│   ├── rf_results.json           # Random Forest metrics
│   └── xgb_feature_importance.csv
│
├── data/processed/               # Cleaned data (3.5 MB total)
│   ├── bangkok_accidents_clean.csv   # 2,669 accidents with GPS, datetime, Thai text
│   ├── bangkok_grid_stats.csv        # 276 grid cells with risk levels
│   ├── ml_features.csv               # 276 grids x 44 features
│   ├── X_train.csv / X_test.csv      # Train/test features
│   ├── y_train.csv / y_test.csv      # Train/test labels
│   └── feature_info.json / split_info.json
│
├── notebooks/
│   └── 01_data_exploration.ipynb     # EDA notebook
│
├── requirements.txt
├── runtime.txt                   # Python 3.11
└── .env                          # API keys
```

---

## Core Architecture

### ML Pipeline

```
Thai MOT accident CSV (23,952 records)
    → Bangkok filtering (2,669 records)
    → 1km x 1km grid aggregation (276 cells)
    → 44-feature extraction per cell:
        - Spatial: lat, lon, grid density, neighboring risk
        - Temporal: hour distribution, day-of-week patterns
        - Severity: death ratio, injury counts
        - Vehicle: motorcycle %, truck %, bus %
        - Weather: rain incidents, visibility
    → XGBoost classifier (binary: dangerous if 5+ accidents)
    → Result: 39/39 danger zones detected, 83.8% accuracy, 17.5x lift
```

### Risk Scoring (how routes get scored)

Location: `src/app.py` — `calculate_route_risk_density_aware()` (line ~808)

```
Route waypoints → Map each to nearest grid cell → Get accident count per cell
→ Distance-weighted average: blended = (weighted_avg * 0.8) + (max_accidents * 0.2)
→ Log-scale: risk_score = 100 * (1 - e^(-blended/25))
→ High-zone penalty: +sqrt(high_risk_zones) * 3 (max +15)
→ Risk level: >=60 HIGH, 30-59 MEDIUM, <30 LOW
```

### Route Comparison Flow

Location: `src/app.py` — `/api/compare-routes` (line ~1425)

```
User enters start + end locations
    → Google Maps API returns multiple alternative routes
    → Each route: decode polyline → map to grid cells → XGBoost risk score
    → Pick FASTEST (lowest time) and SAFEST (lowest risk)
    → If Google Maps unavailable:
        → OSRM for fast route
        → generate_safe_route() tries OSRM alternatives, then detour midpoints
    → Return both routes with waypoints, risk scores, time, distance
```

### Trip Planner UI Flow

Location: `static/js/trip-planner.js` + `templates/trip_planner.html`

```
1. User picks start/end (autocomplete or demo chip)
2. calculateRoute() → POST /api/compare-routes
3. Display two route cards (fast vs safe) on map
4. User selects one → selectRoute() highlights it
5. "Start Navigation" → POST /api/analyze-route → get risk segments
6. startNavigation() → fullscreen map with:
   - Risk-colored polyline segments (red/yellow/green)
   - Moving arrow marker (simulated driving)
   - Top banner showing next risk zone ahead
   - Toast pill alerts when entering danger zones
   - Bottom bar with ETA, distance, progress
7. Arrival → 3-stage animated summary modal
```

### Navigation Simulation

Location: `static/js/trip-planner.js` — `startNavigationSimulation()` (line ~1498)

```
- SIMULATION_TARGET_SECONDS = 18 (whole trip in 18s)
- SIMULATION_TICK_MS = 100 (update every 100ms)
- Speed = totalDistance / (targetSeconds / tickInterval)
- Each tick: advance distance → interpolate position → update marker
- checkNavigationAlerts() fires when entering new risk segments
- updateNavTopBanner() looks ahead 4 segments for upcoming danger
- On completion: showTripSummaryModal() with animated stats
```

---

## API Endpoints

### Route Planning
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/compare-routes` | Compare fast vs safe route (Google Maps + risk scoring) |
| POST | `/api/analyze-route` | Full segment-by-segment risk analysis for navigation |

**compare-routes request:**
```json
{
    "start_lat": 13.7648, "start_lon": 100.5378,
    "end_lat": 13.7300, "end_lon": 100.7500
}
```

**compare-routes response:**
```json
{
    "fast_route": {
        "waypoints": [[lat, lon], ...],
        "distance_km": 24.3,
        "time_min": 35,
        "risk_score": 73,
        "risk_level": "high",
        "high_risk_zones": 4,
        "medium_risk_zones": 4
    },
    "safe_route": { ... same structure, lower risk ... },
    "recommendation": "Safe route reduces risk by 45% for +8 minutes."
}
```

**analyze-route response:**
```json
{
    "success": true,
    "route": { "waypoints": [...], "distance_km": 24.3, "duration_min": 35 },
    "segments": [
        {
            "start_km": 0.0, "end_km": 1.0,
            "waypoints": [[lat, lon], ...],
            "risk_score": 10, "risk_level": "low", "color": "#28a745"
        },
        {
            "start_km": 1.0, "end_km": 2.0,
            "risk_score": 70, "risk_level": "high", "color": "#dc3545"
        }
    ],
    "alerts": [
        { "severity": "danger", "trigger_km": 4.5, "detail": "High accident zone" }
    ]
}
```

### Analytics
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/analytics-data` | Full accident dataset for charts |
| GET | `/api/grid-stats` | 276 grid cells with risk levels |
| GET | `/api/temporal-stats` | Hour/day accident patterns |
| GET | `/api/heatmap-data` | Accident coordinate points |
| GET | `/api/model-info` | ML model accuracy and feature importances |

### NLP
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/nlp-analysis` | Cause classification results |
| POST | `/api/nlp/classify-cause` | Classify Thai text into accident cause |

### Chatbot
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/chat` | Send message, get response (intent detection + Groq AI) |
| GET | `/api/chatbot-data` | Pre-aggregated data for chatbot context |

### Pages
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | Homepage |
| GET | `/trip-planner` | Trip planner page |
| GET | `/analytics` | Analytics dashboard |
| GET | `/nlp` | NLP analysis page |
| GET | `/about` | About/methodology page |

---

## Key Data Structures

### Grid Cell (bangkok_grid_stats.csv)
```
grid_id, lat_idx, lon_idx, center_lat, center_lon,
total_accidents, risk_score, risk_level (low/medium/high),
avg_severity, death_count, injury_count
```

### ML Features (44 per grid cell)
Top features by importance:
1. `severity_mean` (53.6%) — Average accident severity
2. `grid_density` (19.6%) — Accidents per km^2
3. `hour_night_ratio` (4.2%) — Night vs day accident ratio
4. `motorcycle_pct` (3.8%) — Motorcycle involvement
5. `neighboring_risk` (3.1%) — Adjacent cell risk average

### Segment (from analyze-route)
```json
{
    "start_km": 0.0,
    "end_km": 1.0,
    "waypoints": [[13.78, 100.52], [13.78, 100.51], ...],
    "risk_score": 70,
    "risk_level": "high",
    "color": "#dc3545",
    "high_risk_zones": 2,
    "max_accidents": 45
}
```

Colors: `#dc3545` = high (red), `#ffc107` = medium (yellow), `#28a745` = low (green)

---

## Important Global Variables (trip-planner.js)

```javascript
let tripCompareMap;              // Google Maps instance for route comparison
let tripNavigationMap;           // Google Maps instance for driving mode
let routeLayers = {};            // { fast: Polyline, safe: Polyline, _start: Marker, _end: Marker }
let routeOptions = {};           // { fast: routeData, safe: routeData } from compare-routes
let selectedRoute = null;        // "fast" or "safe"
let tripPlannerRouteData = null; // Full route data with segments for navigation
let navPolylines = [];           // Polylines drawn on navigation map
let tripMovingMarker = null;     // Moving arrow during simulation
let tripCurrentDistance = 0;     // Current km in simulation
let tripWaypointCumulativeKm = []; // Distance profile for interpolation

const SIMULATION_TARGET_SECONDS = 18;
const SIMULATION_TICK_MS = 100;
```

---

## High-Risk Areas in Bangkok (from accident data)

### Cluster 1: Western Corridor (HIGHEST)
- Location: lat ~13.7845, lon 100.37-100.41
- Peak: 85 accidents in single grid cell (risk score 100)
- Total: 639 accidents across 17 zones

### Cluster 2: Eastern Corridor (Sukhumvit-Bang Na)
- Location: lat ~13.7305, lon 100.70-100.75
- Peak: 69 accidents per cell
- Total: 475 accidents across 11 zones

### Demo Routes (configured in trip_planner.html)
1. **Dusit → Bang Kruai** — risk 93, crosses western hotspot
2. **Chatuchak → Taling Chan** — risk 87, long western commute
3. **Ekkamai → Bang Na** — risk 78, eastern corridor
4. **Victory Monument → Lat Krabang** — risk 73, cross-city

---

## Common Bugs and Their Fixes

### routeLayers mixing polylines and markers
`routeLayers` stores both Polylines (keys: `fast`, `safe`) and Markers (keys: `_start`, `_end`). When iterating to change styles, skip marker keys:
```javascript
Object.keys(routeLayers).forEach((id) => {
    if (id.startsWith("_")) return; // skip markers
    routeLayers[id].setOptions({ ... });
});
```

### Empty segments on navigation fallback
When `/api/analyze-route` fails, `startNavigationWithSelected()` uses `buildSyntheticSegments()` to create risk-colored segments from compare-routes data instead of falling back to `segments: []`.

### Navigation alerts not firing
`findSegmentIndexForDistance()` returns -1 when segments array is empty, causing `checkNavigationAlerts()` to bail. Fix: ensure segments are always populated (see above).

### Summary card assumes safe route
`showTripSummaryModal()` detects `selectedRoute === "fast"` vs `"safe"` and adapts all stats/labels/colors accordingly.

---

## CSS Architecture (trip-planner.css)

### Navigation Fullscreen Mode
- `.navigation-fullscreen` — `display: block` (map fills everything)
- `#navigation-map` — `position: absolute; inset: 0` (fullscreen map)
- `#nav-top-banner` — Dark glassmorphism pill at top (shows next risk zone)
- `#nav-alert-box` — Centered toast pill with spring animation + colored glow
- `.nav-bottom-bar` — Dark pill at bottom with exit/ETA/distance/progress
- `.nav-progress-bar-wrap` — 4px thin strip at very bottom

### Trip Summary Modal (3 stages)
- `.summary-s1` — SVG checkmark animation (circle draws itself)
- `.summary-s2` — Stats count-up (avoided zones, % safer arc, time cost)
- `.summary-s3` — Comparison bars (red fast vs green safe) + verdict + buttons
- Stages transition via `.visible` class with CSS opacity/transform transitions

---

## Cache Busting

Templates reference CSS/JS with version params:
```html
<link href="{{ url_for('static', filename='css/trip-planner.css', v='20260320e') }}">
<script src="{{ url_for('static', filename='js/trip-planner.js', v='20260320e') }}"></script>
```
Increment the version (e.g., `20260320e` → `20260320f`) after any CSS/JS changes.

---

## Key Functions Reference

### src/app.py
| Function | Line | Purpose |
|----------|------|---------|
| `calculate_route_risk_density_aware()` | ~808 | Score route risk from grid accident density |
| `calculate_route_risk_ml()` | ~770 | Score route risk using XGBoost model |
| `analyze_route_segments()` | ~926 | Divide route into 1km segments with risk colors |
| `generate_alerts_from_segments()` | ~999 | Create danger zone alerts with advance warning |
| `generate_fast_route()` | ~555 | Get fastest route (OSRM → cached → interpolation) |
| `generate_safe_route()` | ~617 | Find safer alternative (OSRM alternatives → detour midpoints) |
| `compare_routes()` | ~1425 | API: compare fast vs safe (Google Maps + risk) |
| `analyze_route()` | ~1297 | API: full segment analysis for navigation |
| `_find_grid_for_point()` | ~480 | Map lat/lon to nearest grid cell |
| `_load_grid_stats()` | ~450 | Load 276 grid cells from CSV (cached) |

### static/js/trip-planner.js
| Function | Line | Purpose |
|----------|------|---------|
| `calculateRoute()` | ~396 | Main entry: calls /api/compare-routes, shows results |
| `displayMultipleRoutes()` | ~530 | Draw fast + safe routes on map with cards |
| `selectRoute()` | ~878 | Highlight selected route, fade other |
| `changeRoute()` | ~932 | Reset route selection |
| `startNavigationWithSelected()` | ~1052 | Fetch segments + start driving mode |
| `buildSyntheticSegments()` | ~956 | Fallback: create segments from compare data |
| `startNavigation()` | ~1491 | Show fullscreen, init map, start simulation |
| `initializeNavigationMap()` | ~1511 | Draw risk-colored segments on nav map |
| `startNavigationSimulation()` | ~1598 | Animate driving marker along route |
| `checkNavigationAlerts()` | ~1711 | Fire alerts when entering danger zones |
| `updateNavTopBanner()` | ~1680 | Look ahead for upcoming risk zones |
| `showTripSummaryModal()` | ~1760 | 3-stage animated end-of-trip summary |
| `selectDemoRoute()` | ~1350 | Auto-fill + trigger route calculation |

---

## When Editing This Project

### Before changing trip-planner.js:
1. `routeLayers` has marker keys (`_start`, `_end`) — always filter with `if (id.startsWith("_")) return`
2. `tripPlannerRouteData.segments` may be empty if API fails — handle gracefully
3. Bump cache buster version in `trip_planner.html` after JS/CSS changes
4. Navigation simulation relies on `tripWaypointCumulativeKm` array — must be built before simulation starts

### Before changing app.py:
1. Grid stats are cached in memory via `_load_grid_stats()` — no need to reload
2. Risk scoring uses TWO methods: ML (XGBoost) and spatial (density-aware) — ML is preferred when available
3. Route generation has 3 fallback levels: Google Maps → OSRM → interpolation
4. All coordinates are `[lat, lon]` arrays (not objects)

### Before changing CSS:
1. Navigation mode uses absolute positioning (not flexbox) for overlays on map
2. Alert toast uses `transform: translateX(-50%)` for centering — don't change to flexbox
3. Summary modal stages use `.visible` class toggle — transitions are CSS-driven, timing is JS-driven
