# Ubiquitous Language — SafeDrive BKK

> This file is the single source of truth for domain terminology.
> Both the developer and Claude use these terms identically.
> Update this file whenever a new concept is introduced.
> Pass this file to Claude at the start of every session.

---

## Core Domain Terms

| Term | Definition | What it is NOT | Used in code as |
|------|------------|----------------|-----------------|
| **Accident** | A road collision event recorded in the Bangkok dataset. The primary domain event the system is built around. | Not "crash", "incident", or "impact" — those words appear in a few places but are not the canonical term. | `accident`, `total_accidents`, `load_bangkok_accidents()` |
| **Grid** | A fixed 0.009° × 0.009° spatial cell (~1 km²) used to aggregate accidents and compute risk. Identified by `grid_id` like `G020_045` (lat-index _ lon-index). | Not "zone" (which is a Grid rendered for display) and not "cell" (an internal math term in grid_aggregation.py only). | `grid_id`, `grid_df`, `load_grid_stats()`, `latlon_to_grid_id()` |
| **Zone** | A Grid that has at least one accident — used in API responses and the UI. The public-facing label for a risky grid cell. | Not a synonym for Grid in general. A Zone is always a Grid, but not every Grid is a Zone. | `risk_zones`, `zone_name`, `build_zone_display_name()` |
| **Risk Score** | An integer 0–100 representing accident probability at a location. 0 = no risk; 100 = maximum. Derived from XGBoost probability × 100, or from raw accident count × 2 in spatial fallback. | Not a percentage of model accuracy. Not interchangeable with Confidence. | `risk_score` |
| **Risk Level** | A categorical label derived from Risk Score: `"low"` (≤ 30), `"medium"` (31–70), `"high"` (≥ 71). | Not a freeform string. Always exactly one of the three values above. | `risk_level` |
| **Severity** | A weighted measure of harm per accident or grid: deaths × 10 + serious injuries × 5 + minor injuries × 1. Distinct from accident count. | Not "risk". A grid with many minor accidents may have lower severity than one with fewer fatalities. | `severity_score`, `severity_total`, `severity_mean`, `has_fatality` |
| **Prediction** | The full output of a single risk assessment: `{risk_score, risk_level, confidence, method, grid_id}`. Always includes the method that produced it. | Not just a number. A Prediction is always the complete dict, not only the score. | `predict_risk()`, `predict_batch()`, result dicts in `/api/compare-predictions` |
| **Confidence** | A 0–1 float indicating how reliable a Prediction is. XGBoost predictions: distance of probability from 0.5. Spatial fallback: fixed 0.4. Out-of-bounds: fixed 0.1. | Not model accuracy (83.8%). Accuracy is a population-level metric; Confidence is per-prediction. | `confidence` |
| **Route** | A complete path from a start coordinate to an end coordinate, including its geometry (Waypoints), distance, duration, Risk Score, and Risk Level. | Not just a list of coordinates. A Route always has aggregated metadata. | `route`, `route_fast`, `route_safe`, `route_recommended` |
| **Waypoints** | The ordered array of `[lat, lon]` coordinate pairs that make up a Route's geometry. Produced by OSRM, Google Maps, or interpolation. | Not the Route itself. Waypoints are the raw shape; the Route wraps them with risk metadata. | `waypoints`, field in all route dicts and API responses |
| **Segment** | A ~1 km subdivision of a Route, annotated with its own Risk Score, Risk Level, and position along the route. Used to pinpoint where on a route the danger is. | Not a Route. A Route contains many Segments. | `segments`, `analyze_route_segments()`, `generate_alerts_from_segments()` |
| **Alternative** | Any competing Route option for the same origin–destination pair. The system always produces at least two: a fast Alternative and a safe Alternative. | Not "detour" (an internal routing strategy). From the API's perspective, all candidates are Alternatives; "detour" is an implementation detail inside `routing.py`. | `alternatives_analyzed`, `all_routes`, `display_routes` |
| **Traffic Source** | A string tag on a Route or analysis result identifying where timing and geometry came from: `"google_maps_traffic"`, `"google_maps"`, `"osrm"`, `"cached_osrm"`, `"estimated"`. | Not a quality rating. A higher-fidelity source doesn't guarantee a safer route. | `traffic_source` |
| **Intent** | A named chatbot pattern (e.g. `"top_danger_corridors"`, `"route_safety"`) detected by keyword matching before the LLM is invoked. Intents trigger scripted responses. | Not a user goal in the product-management sense. An Intent is specifically a chatbot routing signal. | `intent`, `detect_intent()`, `get_scripted_response()` |
| **Cause** | A categorical accident causation type (e.g. speeding, drunk driving) extracted by the NLP pipeline from accident text. Distinct from Intent — Cause is about accidents, Intent is about chatbot routing. | Not an Intent. Not a Risk Level. Cause explains *why* accidents happen; Risk Level describes *where* they concentrate. | `cause_key`, `detect_cause_from_question()`, `handle_ml_cause_response()` |

---

## Relationships Between Terms

- A **Grid** is the fundamental spatial unit. The entire Bangkok area is divided into a fixed set of Grids. Each Grid has a stable `grid_id`.
- A **Zone** is a Grid that has recorded Accidents. The analytics dashboard and `/api/risk-zones` only surface Zones (grids with `total_accidents > 0`).
- Every Grid has a **Risk Score** and **Risk Level**, computed either by the XGBoost model (from 44 pre-computed features) or by the spatial fallback (raw accident count × 2).
- A **Prediction** packages the Risk Score + Risk Level + Confidence + method for a single point lookup. `predict_risk(lat, lon)` returns one Prediction.
- **Severity** is computed per Grid from the raw accident records. It is an input *feature* to the ML model, not the same as Risk Score (which is the model's output).
- A **Route** is built from **Waypoints** and scored by sampling those Waypoints against the Grid risk data. A Route's Risk Score is a density-weighted aggregate of the Grids it crosses.
- A **Segment** breaks a Route into ~1 km chunks, each scored independently against Grids. Segments are used for the turn-by-turn alert system.
- **Alternatives** are multiple Routes for the same trip. The system ranks them via TOPSIS, producing a recommended Alternative that balances risk and travel time.
- The **Chatbot** detects an **Intent** first. If no scripted Intent matches, it tries **Cause** detection. If neither matches with sufficient confidence, it calls the Groq LLM.
- A **District** is a human-readable Bangkok administrative area (e.g. "Chatuchak"). Districts are *display labels only* — they have no role in risk computation. The nearest District centroid is looked up for each Zone to produce `zone_name`.

---

## Boundaries

- **Payment, authentication, and user accounts** are outside this system. There are no users, sessions, or login flows — the app is stateless and publicly accessible.
- **Real-time accident reporting** is out of scope. The dataset is historical (Bangkok 2024). The system predicts risk based on patterns, not live events.
- **Turn-by-turn navigation** is not modeled. The system scores routes and raises alerts but does not give driving instructions.
- **Weather data** does not feed the model. Weather appears in chatbot responses as a contextual note, not as a prediction feature.
- **Districts are display only.** District boundaries are approximated by centroid proximity, not by official polygon boundaries. They are never used in risk computation.
- **Outside Bangkok** returns a safe default (Risk Score = 35, method = `"out_of_bounds"`). The model was trained only on Bangkok data.

---

## Resolved Inconsistencies

| Inconsistency found | Resolution |
|---------------------|------------|
| "grid" vs "cell" | **grid** everywhere in public APIs and inter-module code. "cell" is acceptable only inside `grid_aggregation.py` for internal spatial math. |
| "accident" vs "crash" vs "incident" | **accident** is canonical. "crash" and "incident" should not be introduced in new code or API responses. |
| "confidence" vs "accuracy" | **confidence** (0–1 float, per-prediction). **accuracy** (percentage, model-level statistic). Never swap them. |
| "alternative" vs "detour" | **alternative** in all API responses and user-facing language. "detour" is an internal routing strategy in `routing.py` only. |
| "path" vs "route" vs "waypoints" | **route** = the full object with metadata. **waypoints** = the coordinate array. "path" is acceptable only as a local variable in geometry functions, not in API responses. |

---

## Changelog

| Date | Term added/changed | Reason |
|------|--------------------|--------|
| 2026-05-16 | All 15 core terms | Initial extraction from codebase scan |
