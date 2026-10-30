# SafeDrive BKK

**A Bangkok road-safety web application for exploring accident hotspots and comparing route risk.**

[![CI](https://github.com/HarryLynn-afk/SafeDriveBKK/actions/workflows/ci.yml/badge.svg)](https://github.com/HarryLynn-afk/SafeDriveBKK/actions/workflows/ci.yml)

SafeDrive BKK turns **2,669 Bangkok accident records from 2024** into an interactive risk map, route comparisons, and crash-pattern analytics. It brings together a Flask backend, XGBoost grid scoring, Thai-language NLP, and a browser interface in one application.

[Features](#features) · [Run locally](#run-locally) · [Engineering](#engineering-highlights) · [Evaluation](#evaluation) · [Tests](#tests)

## The problem

An accident dataset tells you where crashes happened. Comparing two journeys also requires mapping those records to locations, resolving route geometry, and weighing travel time against historical accident concentration.

This project connects those steps: explore a hotspot, compare routes through the city, and inspect the patterns behind the scores. It uses historical data; it does not provide live accident detection or guarantee a safe route.

## Features

| Feature | What you can explore |
|---|---|
| Risk map | Historical accident concentration across 276 grids with recorded accidents |
| Trip planner | Route alternatives ranked by travel time and estimated risk using TOPSIS multi-criteria scoring |
| Analytics | Accident patterns by time, location, severity, and recorded cause |
| Safety assistant | Scripted answers, ML-assisted cause categorization, and optional Groq responses |
| Thai NLP | Tokenization and TF-IDF + Logistic Regression categorization of recorded cause descriptions |

## Engineering highlights

- **Modular Flask backend:** an application factory and separate blueprints for pages, analytics, routing, NLP, and chatbot endpoints.
- **Route scoring:** route geometry is sampled across approximately 1 km grids, with segment risk and alerts returned to the interface.
- **Dependency fallbacks:** routing can use live services, cached route presets, or interpolated geometry; risk scoring falls back to spatial counts when ML is unavailable. Interpolated geometry is an estimate and may not follow roads.
- **Input handling:** coordinate validation, JSON request checks, payload size limits, security headers, and per-worker rate limiting.
- **Automated checks:** pytest covers API behavior, routing, scoring, spatial helpers, feature engineering, and chatbot behavior. GitHub Actions runs tests and builds the Docker image for changes targeting `main`.

Read the [architecture, design decisions, and project layout](docs/engineering.md).

## Technology

| Area | Tools |
|---|---|
| Backend | Python 3.11, Flask, Gunicorn |
| Data and ML | pandas, NumPy, scikit-learn, XGBoost |
| Thai NLP | pythainlp, TF-IDF, Logistic Regression |
| Frontend | Jinja2, HTML, CSS, JavaScript, Chart.js, Google Maps |
| External services | OSRM, optional Google Maps APIs, optional Groq API |
| Packaging and checks | Docker, pytest, GitHub Actions |

## Run locally

Use **Python 3.11**, the version configured in CI and Docker. Preprocessed data and inference model artifacts are included; retraining is not required to start the app.

```bash
git clone https://github.com/HarryLynn-afk/SafeDriveBKK.git
cd SafeDriveBKK
python3.11 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python -m flask --app app run --port 5000
```

On Windows, activate the environment with `venv\Scripts\activate` instead.

Open **http://localhost:5000**. Start with the homepage risk map, then explore [analytics](http://localhost:5000/analytics) and the [safety assistant](http://localhost:5000/nlp).

### Optional API keys

The app starts without API keys. The trip planner's Google Maps display requires a browser key; server-side route requests can fall back to OSRM. Open-ended chatbot responses require Groq, while scripted responses remain available without it. Map tiles, charts loaded from CDNs, and live routing require network access.

To enable integrations, copy the example configuration:

```bash
cp .env.example .env
```

Replace the example values with your own, and **leave unused API keys empty**. Configuration details are in [.env.example](.env.example).

| Variable | Purpose |
|---|---|
| `GOOGLE_MAPS_BROWSER_KEY` | Trip planner map; restrict the key to your allowed HTTP referrers |
| `GOOGLE_MAPS_SERVER_KEY` | Google Maps routing and traffic estimates |
| `GROQ_API_KEY` | Open-ended assistant responses |
| `SECRET_KEY` | Fixed Flask signing secret; generated automatically when unset |
| `LOG_LEVEL`, `LOG_FILE` | Optional logging configuration |

### Docker

```bash
docker build -t safedrive-bkk .
docker run --rm -p 8000:8000 safedrive-bkk
```

Open **http://localhost:8000**. To supply configured keys, add `--env-file .env` before the image name in the run command.

## Evaluation

The verified result is for the **count-based spatial baseline**:

| Evaluation detail | Result |
|---|---|
| Rule | Flag grids with at least five accidents in Jan–Sep 2024 |
| Evaluation period | Oct–Dec 2024 |
| Later mapped accidents inside flagged grids | **477 / 569 (83.8%)** |
| Share of Jan–Sep observed grids flagged | **112 / 260 (43.1%)** |
| Lift versus uniform selection among those grids | **1.95×** |

Reproduce it from the repository root:

```bash
python -c 'from src.evidence_metrics import holdout_validation; print(holdout_validation())'
```

These figures do not establish XGBoost performance. Its evaluation artifacts are missing from this checkout. The NLP evaluation includes the recorded cause used to construct its labels, so its scores do not establish prediction of unknown causes.

**Known scoring bug:** coordinate lookup maps the centers of 100 partial boundary cells to adjacent inner cells. Route scoring at those edges needs correction. See [evaluation methods and known limitations](docs/evaluation.md), including the distinction between historical concentration and predictive performance.

## Tests

From the activated virtual environment:

```bash
python -m pytest --tb=short -q
```

The [CI workflow](.github/workflows/ci.yml) runs this suite on Python 3.11 and builds the Docker image after tests pass.

## Data

The dataset comes from the [Thai Ministry of Transport road-accident dataset](https://datagov.mot.go.th/en/dataset/roadaccident).

- **Period:** 2024.
- **Bangkok records:** 2,669; spatial evaluation excludes records without valid dates or coordinates inside the configured bounds.
- **Spatial representation:** 2,418 approximately 1 km grid cells, including 276 with recorded accidents.
- **Storage:** preprocessed CSV files loaded into memory; no runtime database is required.

The project explores recorded accident concentration. Without traffic-volume or distance-driven exposure data, its scores should not be interpreted as an individual driver's accident probability.
