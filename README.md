# SafeDrive BKK

SafeDrive BKK is a portfolio project that turns Bangkok road-accident data into a practical road-safety product. It combines machine learning, Thai-language NLP, and route visualization to help users explore high-risk zones, compare route options, and understand when and why crashes are more likely to happen.

## What It Does

- Predicts zone-level road risk across Bangkok with an XGBoost model
- Compares faster and safer route options in the trip planner
- Answers road-safety questions through a hybrid scripted + LLM assistant
- Surfaces temporal and spatial crash patterns in an analytics dashboard

## Why This Project Is Strong Portfolio Work

- It is built around a real public-safety use case instead of a toy dataset
- The ML model is integrated into a working web application, not left in notebooks
- Thai-language NLP is part of the product experience, not just a side experiment
- The system includes runtime fallbacks for routing and assistant behavior when external services are unavailable

## Stack

- Python, Flask, pandas, NumPy
- XGBoost for zone-level risk scoring
- Thai NLP classification for accident-cause extraction
- HTML, CSS, JavaScript, Chart.js, Folium
- Optional Groq API integration for open-ended assistant responses
- Optional Google Maps keys for richer routing and travel-time behavior

## Dataset

- Source: Thai Ministry of Transport
- Year used in this build: 2024
- Nationwide source records: 23,952
- Bangkok records analyzed in the app: 2,669
- Output zones monitored: 276

Data source:
https://datagov.mot.go.th/en/dataset/roadaccident

## Local Run

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create an `.env` file if you want optional API integrations:

```bash
cp .env.example .env
```

`GROQ_API_KEY` is optional. Without it, the assistant falls back to deterministic responses. Google Maps keys are also optional; without them, the app uses OSRM and estimated travel times.

4. Start the app:

```bash
python3 app.py
```

5. Open `http://localhost:5000`.

## Project Layout

```text
.
├── app.py
├── data/processed/
├── models/
├── src/
├── static/
├── templates/
├── requirements.txt
└── README.md
```
