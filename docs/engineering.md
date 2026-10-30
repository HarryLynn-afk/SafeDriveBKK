# Engineering notes

[Back to the project overview](../README.md)

## Request flow

```
Browser
  │
  ├── GET  /                  → main_bp      (home, analytics, trip planner, NLP, about)
  ├── GET  /api/analytics-data → api_bp      (analytics payload)
  ├── POST /api/analyze-route → routing_api  (single-route risk analysis)
  ├── POST /api/compare-routes→ routing_api  (TOPSIS route comparison)
  ├── POST /api/chatbot       → chatbot_bp   (hybrid assistant)
  ├── GET  /api/nlp/*         → nlp_bp       (NLP pipeline preview, word cloud, metrics)
  └── POST /api/nlp/classify  → nlp_bp       (cause categorization)

Flask Application Factory (src/app.py)
  │
  ├── Security layer          (Content-Type enforcement, security headers, rate limiting)
  │
  ├── src/blueprints/
  │   ├── main.py             page routes
  │   ├── api.py              data / analytics API
  │   ├── routing_api.py      route analysis API
  │   ├── chatbot_bp.py       chatbot API
  │   └── nlp_bp.py           NLP API
  │
  ├── src/risk.py             route risk scoring, TOPSIS, segment analysis
  ├── src/routing.py          OSRM → cached preset → interpolation fallback chain
  ├── src/ml_model.py         XGBoost risk predictor
  ├── src/nlp_analysis.py     Thai NLP cause classifier
  ├── src/chatbot_groq.py     Groq LLM integration
  │
  └── src/utils/
      ├── geo.py              spatial helpers (grid lookup, haversine, area assignment)
      ├── data.py             data loading with module-level cache
      ├── validators.py       coordinate range validation
      └── rate_limit.py       in-memory fixed-window rate limiter
```

## Design decisions

**In-memory rate limiting instead of Redis**
The rate limiter uses a thread-safe in-memory counter rather than Redis or Flask-Limiter. Docker starts two Gunicorn workers, each with its own counters, so configured limits apply per worker rather than across the deployment. Limits also reset when workers restart. A shared deployment-wide limit would require a distributed store such as Redis.

**Grid-based spatial model instead of graph-based routing risk**
Risk is scored per approximately 1 km grid cell rather than per road segment. Cells use a 0.009° latitude/longitude step, approximately 1.00 km north–south and 0.97 km east–west near central Bangkok. Two routes through the same cell receive the same grid score regardless of which specific road they use. Road-segment scoring would require road-network data and a separate evaluation.

Known boundary limitation: grid generation includes partial cells at the northern and eastern bounds, but runtime coordinate lookup clamps indices using complete-cell counts. The centers of 100 of the 2,418 saved cells consequently map to adjacent inner cells. This needs a separate scoring fix before relying on results at those edges; the [spatial holdout calculation](evaluation.md#spatial-baseline) uses its own assignment with partial cells included.

**OSRM public instance instead of self-hosted**
Routing calls go to the public OSRM demo server (`router.project-osrm.org`). This is rate-limited and not suitable for production traffic. A production deployment would self-host OSRM or use a paid routing API. The fallback chain (live OSRM → cached preset → dense interpolation) was built specifically to handle public instance unavailability.

**TF-IDF + Logistic Regression instead of a Thai language model**
The NLP classifier uses TF-IDF features and Logistic Regression rather than a fine-tuned Thai BERT model (e.g., WangchanBERTa). It runs without a GPU and categorizes recorded cause descriptions. The [evaluation limitations](evaluation.md) must be addressed before comparing predictive performance with another model; a transformer has not been evaluated here.

**No persistent database**
Accident data is loaded from pre-processed CSV files at startup and held in memory. There is no database write path — the app is read-only at runtime. This was intentional: the dataset is static (2024 snapshot), so a database would add operational complexity without meaningful benefit. A live-updating system would require a write pipeline and a proper database.

**Model artifact compatibility**
The risk predictor prefers the checked-in native XGBoost `.ubj` artifact. When that file is absent, it can load the legacy `.pkl` artifact. The Thai NLP classifier and vectorizer also use pickle-based joblib artifacts. These depend on compatible Python/package versions and must only be loaded from trusted sources.

## Project layout

```
SafeDriveBKK/
├── app.py                   WSGI entry point
├── Dockerfile
├── requirements.txt
├── .env.example
│
├── src/
│   ├── app.py               Flask application factory
│   ├── config.py            Constants, paths, env vars
│   ├── extensions.py        Shared app state (ML predictor, gmaps client)
│   ├── logging_config.py    Structured logging setup
│   ├── risk.py              Route risk scoring, TOPSIS
│   ├── routing.py           OSRM / Google Maps routing clients
│   ├── ml_model.py          XGBoost risk predictor
│   ├── nlp_analysis.py      Thai NLP classifier
│   ├── chatbot_groq.py      Groq LLM integration
│   ├── chatbot_intents.py   Scripted intent matching
│   ├── blueprints/
│   │   ├── main.py          Page routes
│   │   ├── api.py           Data / analytics API
│   │   ├── routing_api.py   Route analysis API
│   │   ├── chatbot_bp.py    Chatbot API
│   │   └── nlp_bp.py        NLP API
│   └── utils/
│       ├── geo.py           Spatial helpers
│       ├── data.py          Data loading + cache
│       ├── validators.py    Input validation
│       └── rate_limit.py    Rate limiter decorator
│
├── data/processed/          Pre-processed accident + grid data
├── models/                  Trained XGBoost model
├── templates/               Jinja2 HTML templates
├── static/                  CSS, JS, images
└── tests/                   pytest suite
```
