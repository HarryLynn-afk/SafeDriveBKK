# SafeDrive BKK
> Claude reads this file at the start of every session.
> Update this file at the end of every session using /end-session.

---

## Stack
- Backend: Flask 3.0 (application factory pattern), Gunicorn (WSGI)
- ML: XGBoost 2.0 (native `.ubj` format), scikit-learn, joblib
- NLP/Chatbot: Groq LLM API, pythainlp (Thai NLP), TF-IDF cause classifier
- Routing: OSRM (public API), Google Maps Directions + Geolocation APIs
- Data: pandas, numpy — processed CSVs in `data/processed/`
- Frontend: Jinja2 templates, vanilla JS, Chart.js (CDN), Google Maps JS SDK
- Deployment: Docker (multi-stage, non-root user), GitHub Actions CI
- Testing: pytest 8+
- Python: 3.11 (pinned in `runtime.txt` and `.python-version`)

---

## Folder Structure
```
app.py                → WSGI entry point — calls create_app() only
src/
  app.py              → Application factory (create_app), blueprint registration, startup init
  config.py           → All paths, env vars, and constants — import from here
  extensions.py       → Shared app state: ml_predictor, gmaps_client, CHATBOT_DATA, USE_ML
  ml_model.py         → RiskPredictor class (XGBoost + spatial fallback)
  risk.py             → Route risk scoring, TOPSIS ranking, segment analysis
  routing.py          → OSRM/GMaps clients, fast/safe route generation, fallback chain
  data_processing.py  → Raw accident data cleaning
  feature_engineering.py → ML feature computation
  grid_aggregation.py → Bangkok spatial grid logic (0.009° cells)
  nlp_analysis.py     → NLP pipeline over accident text
  chatbot_groq.py     → Groq LLM wrapper
  chatbot_intents.py  → Scripted intent detection + ML cause matching
  chatbot_utils.py    → Chatbot payload sanitization helpers
  logging_config.py   → Centralised logging setup
  blueprints/
    main.py           → Page routes: /, /trip-planner, /analytics, /about, /nlp, /risk-map
    api.py            → /api/risk-zones, /api/temporal-stats, /api/analytics-data,
                        /api/model-info, /api/geolocate, /api/compare-predictions
    routing_api.py    → /api/analyze-route, /api/compare-routes
    nlp_bp.py         → /api/nlp/*
    chatbot_bp.py     → /api/chatbot, /api/chatbot/suggestions
  utils/
    data.py           → load_bangkok_accidents, load_grid_stats, calculate_temporal_stats
    geo.py            → latlon_to_grid_id, nearest_district, haversine_km, assign_bangkok_area
    validators.py     → validate_coordinates (call at every API boundary)
    rate_limit.py     → In-memory per-IP rate limiter decorator
    display.py        → build_zone_display_name
data/processed/       → bangkok_accidents_clean.csv, bangkok_grid_stats.csv, ml_features.csv,
                        X_train.csv, feature_info.json (source of truth for ML features)
models/               → xgboost_model.ubj (primary), xgboost_model.pkl (legacy fallback)
static/               → chatbot_data.json, heatmap_data.json, route_presets.json,
                        nlp_cache.json, bangkok_risk_heatmap.html, CSS, JS
templates/            → Jinja2: base.html, index.html, trip_planner.html, analytics.html,
                        about.html, nlp.html, floating_chat.html
tests/                → conftest.py (flask app + sample fixtures), test_api.py,
                        test_risk.py, test_routing.py, test_data.py, test_geo.py,
                        test_chatbot_utils.py
```

---

## Architecture Rules
- **All env vars and file paths go through `src/config.py`** — never scatter `os.getenv()` or `Path()` calls across modules.
- **All shared app state lives in `src/extensions.py`** — `ml_predictor`, `USE_ML`, `gmaps_client`, `CHATBOT_DATA`. These are mutated once inside `create_app()`.
- **All API routes go through blueprints** — no routes registered directly on the `app` object.
- **Validate all coordinates at API boundaries** using `src/utils/validators.py:validate_coordinates` before passing to routing or ML logic.
- **Rate-limit all public POST endpoints** using the `@rate_limit` decorator from `src/utils/rate_limit.py`.
- **Data loading goes through `src/utils/data.py`** — never read CSVs directly in blueprints or risk/routing modules.
- **XGBoost model loads from `.ubj` first, `.pkl` as fallback** — never change this precedence order.
- **ML predictions always have a spatial fallback** via `RiskPredictor.fallback_to_spatial()` — the app must degrade gracefully when the model file is missing.
- **Security headers are set globally** in `_register_request_hooks` — do not set them per-route.
- **Max request body is 2 MB** — enforced in `create_app()`. Do not override.

---

## Key Decisions Made
- **Application factory pattern** (`create_app()`) — chosen so tests can spin up isolated instances without global state bleeding between test runs.
- **XGBoost stored as native `.ubj` not `.pkl`** — eliminates pickle security warning and is the XGBoost-native format. The `.pkl` exists only for backward compatibility; `.ubj` always takes precedence.
- **OSRM → cached preset → dense interpolation fallback chain** — because OSRM is a public API that can be slow or unavailable; presets in `static/route_presets.json` cover common Bangkok corridors.
- **Google Maps traffic layered on top of OSRM geometry** — OSRM provides the route shape; GMaps provides live travel time when the API key is present. They're intentionally decoupled.
- **Chatbot: scripted → ML cause → Groq LLM** — scripted patterns handle high-confidence known intents without burning Groq quota. Groq is only called when scripted + ML both fail.
- **TOPSIS for multi-route ranking** — applied in `/api/compare-routes` to score alternatives on risk vs. time simultaneously, avoiding the need to hard-code a weighting scheme.
- **Non-root Docker user** — the container runs as `appuser`; the `/app` directory is chowned accordingly.
- **Bangkok spatial grid: 0.009° cells** — this constant is replicated in `config.py`, `ml_model.py`, and `grid_aggregation.py`. All three must stay in sync.
- **Consistent API response shape** — all success responses include a `success: true` key or a top-level data key; all errors return `{"error": "..."}` with an appropriate HTTP status.

---

## Current State
- Core app is **complete and working**: heatmap, trip planner, analytics dashboard, NLP page, chatbot
- XGBoost model loaded from `models/xgboost_model.ubj` — 83.8% holdout crash capture, 17.5x lift
- Thai-language chatbot support via pythainlp + Groq
- Docker image builds clean (multi-stage, non-root)
- GitHub Actions CI runs pytest + Docker build on every push to `main`
- Pytest suite has 6 test modules covering API endpoints, risk scoring, routing, data loading, geo utilities, and chatbot utils
- Deployed/targetted for **Render** (Gunicorn, `PORT` env var set automatically)
- Google Maps API key is optional — app degrades to OSRM + estimated travel times when absent

---

## Open Tasks
1. Add more edge-case test coverage (e.g. OSRM timeout paths, out-of-bounds coordinates)
2. Consider caching `load_bangkok_accidents()` / `load_grid_stats()` results in `extensions.py` to avoid re-reading CSVs on every request
3. NLP cache (`static/nlp_cache.json`) has no invalidation strategy — may serve stale results after data updates
4. Route presets (`static/route_presets.json`) are static; no automated refresh from live OSRM data

---

## What NOT to do
- **Never read CSVs directly in blueprints** — go through `src/utils/data.py`
- **Never call `os.getenv()` or `Path()` outside `src/config.py`** for app-level configuration
- **Never register routes directly on `app`** — use blueprints
- **Never skip coordinate validation** (`validate_coordinates`) before passing user-supplied lat/lon into routing or ML code
- **Never add new API keys** to the codebase without adding them to `src/config.py` and `.env.example`
- **Never change the grid cell size (0.009°)** in one file without updating all three locations: `config.py`, `ml_model.py`, `grid_aggregation.py`
- **Never load the XGBoost model with pickle** when `.ubj` is available — the native format is intentional
- **Never commit secrets** — `.env` is gitignored; `.env.example` documents the required keys

---

## Session Log
| Date | Task completed | Decisions made |
|------|---------------|----------------|
| 2026-05-16 | /init-project — generated CLAUDE.md from full codebase scan | Documented all architecture rules and key decisions |
