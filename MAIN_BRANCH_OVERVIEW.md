# SafeDrive BKK — Main Branch Overview

## What Is This Project?

**SafeDrive BKK** (branded as "Smart City Dashboard" on the frontend) is an AI-powered accident risk prediction system for Bangkok. It uses Thai government accident data and XGBoost machine learning to predict next-week high-risk road zones, plan safer routes, and provide real-time navigation alerts.

**Data Source:** Thai Ministry of Transport (2024) — 23,952 accident records nationwide, 2,669 focused on Bangkok.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Flask Web App (app.py)                  │
│                     1,451 lines                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Pages:                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Dashboard /  │  │ Trip Planner │  │  Analytics   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│  ┌──────────────┐  ┌──────────────┐                     │
│  │  NLP /nlp    │  │  About       │                     │
│  └──────────────┘  └──────────────┘                     │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                 18+ API Endpoints                       │
│  /api/compare-routes    /api/analyze-route              │
│  /api/risk-zones        /api/temporal-stats             │
│  /api/model-info        /api/analytics-data             │
│  /api/nlp/*                                             │
│  /api/compare-predictions                               │
├─────────────────────────────────────────────────────────┤
│            ML Prediction (ml_model.py)                  │
│  XGBoost → 39/39 danger zones │ 83.8% accuracy          │
│  44 features per grid cell    │ 17.5x lift vs random    │
│  Fallback: Random Forest (37/39 zones)                  │
├─────────────────────────────────────────────────────────┤
│                    Data Layer                            │
│  bangkok_accidents_clean.csv    (2,669 records)         │
│  bangkok_grid_stats.csv         (276 grid cells)        │
│  ml_features.csv                (44 features × 276)     │
│  Bangkok grid: 39×55 cells, ~1km each                   │
└─────────────────────────────────────────────────────────┘
```

---

## What's Built on Main

### 1. Dashboard (Home Page)
- Hero section with project description
- 4 metric cards (accidents analyzed, danger zones, model accuracy, data points)
- Embedded interactive Bangkok risk heatmap (Folium)
- Quick links to NLP analysis and About page

### 2. Trip Planner (`/trip-planner`)
- **Grab-style UI** with start/end location inputs
- Location autocomplete via Nominatim geocoding
- "Detect My Location" browser geolocation
- Pre-defined Bangkok landmarks for quick selection
- **Multi-route comparison**: Fast route vs. Safe route
  - OSRM real-world routing for fast route
  - Angle-based detour algorithm for safe route
  - ML confidence scores and risk badges on each option
- **Full-screen navigation mode**:
  - Simulated driving along selected route
  - Real-time risk alerts with color coding (green/yellow/red)
  - Text-to-speech voice warnings
  - Progress bar, ETA, distance tracking
  - 1km segment-based risk analysis with 0.5km advance warnings
- **Frontend**: 1,461 lines of JavaScript (`trip-planner.js`)

### 3. Analytics Dashboard (`/analytics`)
- 12+ interactive Chart.js charts:
  - Accidents by hour of day
  - Accidents by day of week
  - Monthly trends
  - Top accident causes (Thai categories)
  - Vehicle type breakdown
  - Risk level distribution
  - Severity analysis

### 4. Risk Heatmap
- Pre-rendered Folium interactive map (`bangkok_risk_heatmap.html`)
- Color-coded 1km grid cells by risk level
- Embedded as iframe in dashboard

### 5. About / Methodology (`/about`)
- ML methodology explanation
- Evidence metrics display
- Responsible AI section
- Data source attribution
- Model limitations acknowledgment

### 6. NLP Analysis (`/nlp`) — Track 1
- Thai text tokenization using `pythainlp`
- Accident cause extraction from Thai-language reports
- Cause category classification
- **Key insight**: NLP output maps directly to XGBoost features (cause categories = 3.8% model importance)
- Pre-cached results in `nlp_cache.json`

---

## ML Model Details

### XGBoost Classifier
- **Performance**: 39/39 danger zones detected (100% recall on dangerous grids)
- **Accuracy**: 83.8% temporal capture
- **Lift**: 17.5x better than random selection
- **Improvement**: +16.2% over spatial baseline

### Top Feature Importances

| Feature | Importance | Category |
|---------|-----------|----------|
| `severity_mean` | 53.6% | Severity |
| `grid_density` | 19.6% | Spatial |
| `month_spread` | 9.8% | Temporal |
| `pickup_count` | 7.4% | Vehicle type |
| `rush_hour_pct` | 2.1% | Temporal |
| `type_ชนด้านข้าง` (sideswipe) | 1.8% | Accident type |
| `type_ชนท้าย` (rear-end) | 1.8% | Accident type |
| `pickup_pct` | 1.7% | Vehicle % |

### 44 Features Per Grid Cell
- **Spatial**: accident counts in 8 compass directions, neighbor statistics
- **Temporal**: peak hour, peak day, rush hour percentage, month spread
- **Vehicle**: motorcycle/car/pickup counts and percentages
- **Weather**: clear/rain percentages
- **Severity**: mean severity, total severity, fatality presence
- **Accident types**: collision, rear-end, sideswipe categories

---

## File Structure (Main Branch)

```
hackathon/
├── src/
│   ├── app.py                    # Flask app — 1,451 lines, 18+ routes
│   ├── ml_model.py               # RiskPredictor class — 301 lines
│   ├── nlp_analysis.py           # Thai NLP pipeline — 424 lines
│   ├── data_processing.py        # Raw data cleaning
│   ├── feature_engineering.py    # 44-feature extraction
│   ├── grid_aggregation.py       # 1km grid system
│   ├── train_xgboost.py          # Model training script
│   ├── train_random_forest.py    # Fallback model training
│   └── test_ml_integration.py    # 7/7 passing tests
├── models/
│   ├── xgboost_model.pkl         # Primary model (126 KB)
│   ├── random_forest.pkl         # Fallback model (335 KB)
│   ├── xgb_results.json          # Training metrics
│   └── xgb_feature_importance.csv
├── data/processed/
│   ├── bangkok_accidents_clean.csv   # 2,669 records
│   ├── bangkok_grid_stats.csv        # 276 grid cells
│   ├── ml_features.csv              # Feature matrix
│   ├── X_train.csv / y_train.csv    # Train split
│   ├── X_test.csv / y_test.csv      # Test split
│   └── bangkok_risk_map.html        # Pre-rendered heatmap
├── templates/
│   ├── base.html                 # Layout with Bootstrap 5 nav
│   ├── index.html                # Dashboard
│   ├── trip_planner.html         # Grab-style trip planner
│   ├── analytics.html            # Charts dashboard
│   ├── nlp.html                  # NLP analysis page
│   ├── about.html                # Methodology
│   ├── route_analyze.html        # Route analysis
│   └── route_compare.html        # Route comparison
├── static/
│   ├── css/style.css, trip-planner.css, driving-mode.css
│   ├── js/app.js, trip-planner.js (1,461 lines)
│   ├── bangkok_risk_heatmap.html
│   ├── heatmap_data.json
│   ├── demo_routes.json
│   └── nlp_cache.json
├── notebooks/
│   └── 01_data_exploration.ipynb     # EDA notebook
├── scripts/
│   ├── generate_dense_routes.py      # OSRM route fetcher
│   └── generate_dense_routes.py      # OSRM route fetcher
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Flask 3.0 |
| ML | XGBoost 2.0.3, scikit-learn 1.4 |
| NLP | pythainlp 5.0+ (Thai tokenization) |
| Maps | Folium 0.15, Leaflet.js |
| Charts | Chart.js |
| Routing | OSRM (OpenStreetMap Routing Machine) |
| Frontend | Bootstrap 5, vanilla JS |
| Data | pandas 2.1, numpy 1.26 |

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Total lines of code | ~5,000+ |
| API endpoints | 18+ |
| HTML templates | 9 pages |
| Bangkok accidents | 2,669 records |
| Grid cells | 276 (39×55 grid) |
| ML features | 44 per grid |
| Danger zones detected | 39/39 (100% recall) |
| Model accuracy | 83.8% |
| Prediction lift | 17.5× vs random |
| Trained models | 2 (XGBoost + Random Forest) |

---

## The 2-Track Strategy

The project competes across **2 hackathon tracks** through one unified system:

| Track | Component | How It Connects to XGBoost |
|-------|-----------|---------------------------|
| **Track 1: NLP** | Thai accident text analysis | Cause categories → 3.8% feature importance |
| **Track 2: Data Visualization** | Heatmap + Analytics + Trip Planner | Core prediction display |

Both tracks feed into the **same XGBoost model**, creating a unified narrative: "NLP + Data Visualization, one prediction engine."
