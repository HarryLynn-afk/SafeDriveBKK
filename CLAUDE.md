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
docs/
  ubiquitous-language.md → Shared domain vocabulary (15 core terms, pass to Claude each session)
src/
  app.py              → Application factory (create_app), blueprint registration, startup init
  config.py           → All paths, env vars, and constants — import from here
  extensions.py       → Shared app state: ml_predictor, gmaps_client, CHATBOT_DATA, USE_ML
  ml_model.py         → RiskPredictor class (XGBoost + spatial fallback)
  risk.py             → Route risk scoring, TOPSIS ranking, segment analysis; score_route() is the main entry point
  routing.py          → OSRM/GMaps clients, route generation; resolve_route() is the main entry point
  data_processing.py  → Raw accident data cleaning
  feature_engineering.py → ML feature computation
  grid_aggregation.py → Bangkok spatial grid logic (0.009° cells)
  nlp_analysis.py     → NLP pipeline over accident text; load_nlp_cache() is the public entry point
  chatbot.py          → Chatbot orchestration; respond_to_chat_message() is the public entry point
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
                        test_chatbot_utils.py, test_chatbot_intents.py,
                        test_chatbot_pipeline.py,
                        test_nlp_cache.py, test_refresh_presets.py
scripts/
  refresh_route_presets.py → refresh_presets() re-queries OSRM and rewrites route_presets.json
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
- **Route resolution goes through `resolve_route()`** — never re-implement the OSRM → cache → interpolation fallback chain inline in a blueprint.
- **Route risk scoring goes through `score_route()`** — never call `calculate_route_risk_ml()` and `calculate_route_risk_density_aware()` directly from blueprints; `score_route()` owns the ML/spatial branching decision.
- **Route comparison goes through `get_route_alternatives()` + `rank_and_annotate_routes()`** — blueprints must not orchestrate the dedupe → TOPSIS → annotate → display chain inline.
- **Analytics aggregation goes through `build_analytics_payload()`** — never replicate per-aggregation pandas logic in blueprints.
- **Chatbot orchestration goes through `respond_to_chat_message()`** in `src/chatbot.py` — never re-implement the scripted → ML cause → Groq tier dispatch, trip-planner CTA attachment, or sanitization chain inside a blueprint.
- **Single-route analysis goes through `build_route_analysis_payload()`** in `src/risk.py` — never re-implement sampling, segment analysis, alert generation, scoring, bucket counting, or model_accuracy mapping inside `/api/analyze-route`. The blueprint receives a `ResolvedRoute` and a `grid_df` and delegates the entire response shape to this function.

---

## Key Decisions Made
- **Application factory pattern** (`create_app()`) — chosen so tests can spin up isolated instances without global state bleeding between test runs.
- **XGBoost stored as native `.ubj` not `.pkl`** — eliminates pickle security warning and is the XGBoost-native format. The `.pkl` exists only for backward compatibility; `.ubj` always takes precedence.
- **`resolve_route()` in `routing.py` owns the full fallback chain** — OSRM → cached preset → dense interpolation → GMaps timing overlay. Returns a `ResolvedRoute` dataclass. Blueprints call this one function; they never manage the chain themselves.
- **`score_route()` in `risk.py` owns the ML/spatial branch** — calls XGBoost internally and falls back to spatial automatically. Returns a complete dict including `prediction_method`. Blueprints call this one function.
- **`load_bangkok_accidents()` is now cached in memory** — same pattern as `load_grid_stats()`; CSV read only once per process.
- **Google Maps traffic layered on top of OSRM geometry** — OSRM provides the route shape; GMaps provides live travel time when the API key is present. They're intentionally decoupled.
- **Chatbot: scripted → ML cause → Groq LLM** — scripted patterns handle high-confidence known intents without burning Groq quota. Groq is only called when scripted + ML both fail.
- **TOPSIS for multi-route ranking** — applied in `/api/compare-routes` to score alternatives on risk vs. time simultaneously, avoiding the need to hard-code a weighting scheme.
- **`get_route_alternatives()` in `routing.py` owns alternatives generation** — GMaps Directions first; falls back to fast + safe pair with GMaps timing layered on. Returns scored routes ready to rank.
- **`rank_and_annotate_routes()` in `risk.py` owns the comparison pipeline** — dedupe → TOPSIS → fast/safe/recommended selection → confidence/peak labels → display routes → recommendation text. Single entry point for the `/api/compare-routes` response body.
- **`build_analytics_payload()` in `utils/data.py` owns the analytics response** — internally composed of small private aggregators (temporal, top zones, area, vehicle, severity, risk distribution, weekend split). Thai column names live here, not in the blueprint.
- **`load_nlp_cache()` in `nlp_analysis.py` is the NLP cache entry point** — in-process singleton with mtime-based staleness detection. Logs a warning but never regenerates synchronously (pipeline is too slow for a web request). Regenerate offline with `python -m src.nlp_analysis`.
- **`reload_route_presets()` + `scripts/refresh_route_presets.py`** — two-part refresh: script re-queries OSRM and writes the file; `reload_route_presets()` busts the in-process cache. Both are needed for a live process to see updated presets.
- **Scripted intents are registered via `@intent(name, patterns)` in `chatbot_intents.py`** — patterns and handler live together; there is no separate `INTENTS` dict or `_HANDLERS` dispatch table. Matching priority is encoded by `_INTENT_ORDER` (a list of intent names) because pattern overlap means some intents must be tried before others.
- **`respond_to_chat_message()` in `src/chatbot.py` owns the chatbot pipeline** — scripted → ML cause (≥ 0.6 confidence) → Groq, then trip-planner CTA attachment and payload sanitization. Reads `extensions.CHATBOT_DATA` internally. The `/api/chatbot` blueprint is now an 8-line wrapper. Same architectural pattern as `resolve_route()` / `score_route()` / `build_analytics_payload()`.
- **`build_route_analysis_payload(resolved, grid_df)` in `src/risk.py` owns the /api/analyze-route response** — sampling, `analyze_route_segments`, `generate_alerts_from_segments`, `score_route`, high/medium/low bucketing, and the `prediction_method → model_accuracy` mapping all live inside. The blueprint is now a 4-line wrapper. Same architectural pattern as `rank_and_annotate_routes()` for `/api/compare-routes`.
- **`calculate_segment_risk` trusts `predict_risk`'s internal fallback** — the outer try/except chain that called `fallback_to_spatial()` separately was redundant (predict_risk already does that internally when the model fails). Collapsed to a single call per branch; the misleading `{"method": "fallback"}` drift is gone — that string is now only returned when `ml_predictor` is None.
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
- **220 tests passing** across 10 modules — all previous + 4 `calculate_segment_risk` boundary tests + 9 `build_route_analysis_payload` boundary tests
- `resolve_route()` and `score_route()` are the deep module interfaces for routing and risk; fully tested
- `get_route_alternatives()` + `rank_and_annotate_routes()` are the deep module interfaces for `/api/compare-routes`; blueprint is now ~30 lines; fully unit-tested
- `build_analytics_payload()` is the deep module interface for `/api/analytics-data`; blueprint is now ~7 lines; fully unit-tested
- `respond_to_chat_message()` in `src/chatbot.py` is the deep module interface for `/api/chatbot`; blueprint is now an 8-line wrapper; 8 boundary tests in `test_chatbot_pipeline.py`
- `build_route_analysis_payload()` in `src/risk.py` is the deep module interface for `/api/analyze-route`; blueprint analyze_route handler is now ~4 lines (resolve → build → jsonify); 9 boundary tests in `test_risk.py::TestBuildRouteAnalysisPayload`
- `load_nlp_cache()` in `src/nlp_analysis.py` is the public entry point for NLP results — in-process singleton, detects staleness via mtime comparison, never raises, warns when accident data is newer than cache
- `reload_route_presets()` in `src/routing.py` busts the in-process presets cache; `scripts/refresh_route_presets.py` re-queries OSRM and rewrites the file — run with `python -m scripts.refresh_route_presets`
- `static/route_presets.json` refreshed from live OSRM (all 4 presets updated 2026-05-20)
- **One High-priority architecture task open from 2026-05-23 review** — `/api/compare-predictions` blueprint still inlines ML/spatial dispatch (last blueprint doing this). See Open Tasks.
- Deployed/targeted for **Render** (Gunicorn, `PORT` env var set automatically)
- Google Maps API key is optional — app degrades to OSRM + estimated travel times when absent
- Project docs in `docs/ubiquitous-language.md` — 15 canonical domain terms, pass to Claude at session start

---

## Open Tasks
Surfaced by 2026-05-23 `/improve-codebase-architecture` Phase 1 report; one High held for next focused session, two Lows queued.
1. **High — `/api/compare-predictions` still inlines ML/spatial dispatch** — handler in `src/blueprints/api.py:137-179` reaches into `extensions.ml_predictor` directly, runs two parallel `try/except` blocks for `predict_risk()` and `fallback_to_spatial()`, and computes the difference dict inline. This is the last blueprint doing manual ML/spatial branching — same shape we extracted for `score_route` / `respond_to_chat_message` / `build_route_analysis_payload` / `calculate_segment_risk`. Extract to a `compare_predictions_payload(lat, lon)` (or a method on `RiskPredictor`) that returns both predictions and the diff dict.
2. **Low — `/api/risk-zones` data shaping in blueprint** — `df[df["total_accidents"] > 0].copy()` + `df.where(pd.notna(df), None)` lives in the handler; belongs next to `load_grid_stats()` in `src/utils/data.py` (or a new `load_active_risk_zones()`).
3. **Low — accuracy strings drift between `_MODEL_ACCURACY_BY_METHOD` (risk.py) and `model_info` (api.py)** — `"83.8% holdout crash capture"` (risk.py) vs `"83.8% holdout crash capture rate"` (api.py) already disagree on wording. Single-source-of-truth fix when one of those endpoints is next touched.
4. Run `python -m src.nlp_analysis` to regenerate `nlp_cache.json` if accident data is updated (script exists; no CI automation yet).
5. Add `scripts/refresh_route_presets.py` to a scheduled CI job so presets stay fresh automatically.

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
- **Never re-implement the route fallback chain inline** — always call `resolve_route()` from `src/routing.py`
- **Never call `calculate_route_risk_ml()` or `calculate_route_risk_density_aware()` directly from blueprints** — call `score_route()` instead
- **Never orchestrate `dedupe_scored_routes` / `apply_topsis` / `annotate_route_roles` / `build_display_routes` from blueprints** — call `rank_and_annotate_routes()` instead
- **Never replicate analytics aggregation in blueprints** — call `build_analytics_payload()` from `src/utils/data.py`
- **Never reintroduce `INTENTS` or `_HANDLERS` dicts in `chatbot_intents.py`** — register new scripted intents with the `@intent(name, patterns)` decorator and add the intent name to `_INTENT_ORDER` at the desired priority position
- **Never call `_load_nlp_cache()` in blueprints** — call `load_nlp_cache()` from `src.nlp_analysis` instead; the blueprint must own zero file I/O logic for NLP data
- **Never re-query OSRM for route preset refresh inside a web request** — use `python -m scripts.refresh_route_presets` offline; call `reload_route_presets()` after the file is written
- **Never orchestrate chatbot tiers inside a blueprint** — call `respond_to_chat_message()` from `src.chatbot`. The blueprint must not import from `chatbot_intents`, `chatbot_groq`, or `chatbot_utils` directly.
- **Never inline sampling / segment analysis / alert generation / scoring inside `/api/analyze-route`** — call `build_route_analysis_payload(resolved, grid_df)` from `src.risk`. The blueprint must not import `analyze_route_segments`, `generate_alerts_from_segments`, or `score_route` directly.
- **Never re-add the outer `fallback_to_spatial()` chain to `calculate_segment_risk`** — `predict_risk()` already handles its own spatial fallback. The function trusts that contract and only catches uncaught exceptions from the ML path.

---

## Session Log
| Date | Task completed | Decisions made |
|------|---------------|----------------|
| 2026-05-16 | /init-project — generated CLAUDE.md from full codebase scan | Documented all architecture rules and key decisions |
| 2026-05-16 | /ubiquitous-language — generated docs/ubiquitous-language.md with 15 core terms | Resolved 5 terminology inconsistencies (grid/cell, accident/crash, confidence/accuracy, alternative/detour, route/path/waypoints) |
| 2026-05-16 | /improve-codebase-architecture — extracted resolve_route(), score_route(), added accidents cache | resolve_route() owns fallback chain; score_route() owns ML/spatial branch; both are the required entry points from blueprints |
| 2026-05-16 | /tdd — added 18 tests for resolve_route() and score_route() | Test suite: 112 → 130; patches now target correct module boundaries |
| 2026-05-16 | /improve-codebase-architecture — extracted `get_route_alternatives()` + `rank_and_annotate_routes()` (route comparison) and `build_analytics_payload()` (analytics) | routing_api compare_routes: 145 → ~30 lines; api analytics_data: 124 → 7 lines. All 130 tests still passing. |
| 2026-05-16 | /improve-codebase-architecture — converted chatbot_intents.py from `INTENTS` dict + standalone handlers + `_HANDLERS` dispatch table to `@intent(name, patterns)` decorator | 3-place pattern → 2-place pattern (decorator + `_INTENT_ORDER` list). Patterns now live with their handlers. 130 tests + `python -m src.chatbot_intents` smoke test (18/21) confirm behavioral parity. |
| 2026-05-16 | /tdd — added 10 tests for `@intent` registry boundary in `test_chatbot_intents.py` | Covers known-pattern matches, no-match/empty cases, `_INTENT_ORDER` priority, decorator registration, and `get_scripted_response` dispatch. Test suite: 130 → 140. |
| 2026-05-20 | /end-session — review session; corrected test count 130 → 140 in CLAUDE.md, added `test_chatbot_intents.py` to folder structure | No new code changes; all commits were already on main. |
| 2026-05-20 | /tdd — added 34 unit tests for `get_route_alternatives()`, `rank_and_annotate_routes()`, and `build_analytics_payload()` | Test suite: 140 → 174. Open Task 3 closed. GMaps vs fallback paths, confidence bounds, prediction_method branching, temporal/risk/weekend aggregations all covered at module boundary. |
| 2026-05-20 | Closed all open tasks: NLP cache invalidation (`load_nlp_cache()` with mtime staleness), route preset refresh (`reload_route_presets()` + `scripts/refresh_route_presets.py`), 59 new unit tests | Test suite: 140 → 199. No remaining open tasks. `scripts/` package added. `nlp_bp.py` now delegates all file I/O to `load_nlp_cache()`. |
| 2026-05-20 | /improve-codebase-architecture + /tdd — extracted `respond_to_chat_message()` to `src/chatbot.py`; added 8 boundary tests in `test_chatbot_pipeline.py` | Chatbot blueprint: 50 → 8 lines in handler body. 9 cross-module imports → 1. Same pattern as `resolve_route()` / `build_analytics_payload()`. Test suite: 199 → 207. Two medium-priority items (`calculate_segment_risk` fallback dup, `analyze_route` handler inlining) surfaced and held for next session. |
| 2026-05-23 | /build-plan + /design-check + /tdd — closed both held items: collapsed `calculate_segment_risk` to a single call (no more `{"method": "fallback"}` drift) and extracted `build_route_analysis_payload(resolved, grid_df)` to `src/risk.py` | `/api/analyze-route` handler: 60 → 4 lines. 4 cross-module imports → 1. Added 4 boundary tests for `calculate_segment_risk` and 9 for `build_route_analysis_payload`. Test suite: 207 → 220. Open architecture tasks: 0. |
| 2026-05-23 | /improve-codebase-architecture Phase 1 — scanned for remaining shallow modules | 1 High (`/api/compare-predictions` inline ML/spatial dispatch), 2 Lows (`/api/risk-zones` shaping, accuracy-string drift). Held all three for a focused next session. |
