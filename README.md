# SafeDrive BKK

SafeDrive BKK turns Bangkok road-accident data into a practical road-safety product. It combines machine learning, Thai-language NLP, and route visualisation to help users explore high-risk zones, compare route options, and understand when and why crashes are more likely to happen.

![CI](https://github.com/HarryLynn-afk/hackathon-version-1/actions/workflows/ci.yml/badge.svg)

---

## What It Does

| Feature | Detail |
|---|---|
| Risk map | XGBoost model scores 276 Bangkok grid zones; 83.8% holdout crash-capture rate |
| Trip planner | Compares fast vs. safe routes using TOPSIS multi-criteria scoring |
| Analytics dashboard | Temporal and spatial crash-pattern charts |
| Hybrid chatbot | Scripted intents → ML cause detection → Groq LLM fallback |
| Thai NLP | TF-IDF + Logistic Regression classifies accident causes from Thai text |

---

## Architecture

```
Browser
  │
  ├── GET  /                  → main_bp      (home, analytics, trip planner, NLP, about)
  ├── GET  /api/analytics-data→ api_bp       (risk zones, temporal stats, model info)
  ├── POST /api/analyze-route → routing_api  (single-route risk analysis)
  ├── POST /api/compare-routes→ routing_api  (TOPSIS route comparison)
  ├── POST /api/chatbot       → chatbot_bp   (hybrid assistant)
  └── GET  /api/nlp/*         → nlp_bp       (NLP pipeline preview, word cloud, classify)

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

---

## Stack

- **Backend**: Python 3.11, Flask, gunicorn
- **ML**: XGBoost, scikit-learn, pandas, NumPy
- **NLP**: pythainlp, TF-IDF, Logistic Regression
- **Routing**: OSRM (open-source), Google Maps Directions API (optional)
- **LLM**: Groq API — `llama-3.3-70b-versatile` (optional)
- **Frontend**: HTML/CSS/JS, Chart.js, Google Maps JS API
- **Testing**: pytest — 112 tests
- **CI/CD**: GitHub Actions — tests + Docker build on every push

---

## Dataset

- **Source**: Thai Ministry of Transport — [datagov.mot.go.th](https://datagov.mot.go.th/en/dataset/roadaccident)
- **Year**: 2024
- **Nationwide records**: 23,952
- **Bangkok records analysed**: 2,669
- **Output zones monitored**: 276

---

## Local Setup

### 1. Clone and install

```bash
git clone https://github.com/HarryLynn-afk/hackathon-version-1.git
cd hackathon-version-1
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env — all keys are optional (see .env.example for details)
```

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | No | Open-ended chatbot responses via Groq LLM |
| `GOOGLE_MAPS_SERVER_KEY` | No | Traffic-aware routing; falls back to OSRM |
| `GOOGLE_MAPS_BROWSER_KEY` | No | Trip planner map display |
| `SECRET_KEY` | No | Flask session secret; auto-generated if unset |
| `LOG_LEVEL` | No | `DEBUG`/`INFO`/`WARNING` (default: `INFO`) |
| `LOG_FILE` | No | Path for rotating log file; console-only if unset |

### 3. Run

```bash
python3 app.py
# Open http://localhost:5000
```

---

## Docker

```bash
# Build
docker build -t safedrive-bkk .

# Run
docker run -p 8000:8000 \
  -e GROQ_API_KEY=your_key \
  -e SECRET_KEY=your_secret \
  safedrive-bkk

# Open http://localhost:8000
```

---

## Tests

```bash
pytest --tb=short -q
# 112 passed
```

---

## Project Layout

```
hackathon-version-1/
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
└── tests/                   pytest suite (112 tests)
```
