# SafeDrive BKK: AI-Powered Road Safety Prediction

Predicts high-risk road zones in Bangkok using NLP and Data Visualization on real Thai government accident data.

## Features

1. **ML Risk Prediction Map** - XGBoost model trained on 23,952 accident records, 44 engineered features
2. **Interactive Trip Planner** - Compare fast vs. safe routes with segment-level risk scores
3. **Safety Assistant Chatbot** - NLP-powered Q&A about Bangkok road safety
4. **Analytics Dashboard** - Accident patterns, temporal trends, and spatial insights
5. **Danger Calendar** - Time-based risk patterns highlighting holidays and peak hours

## Tech Stack

- **Data**: 23,952 accident records with GPS coordinates (Thai Ministry of Transport, 2024)
- **ML**: XGBoost gradient boosting (83.8% accuracy, 17.5x lift, 39/39 zone detection)
- **NLP**: PythaiNLP for Thai text tokenization + Groq AI chatbot
- **Backend**: Flask REST API
- **Visualization**: Folium interactive maps, Chart.js dashboards
- **Language**: Python 3.10+

## Installation

```bash
pip install -r requirements.txt
python src/app.py
```

## Data Source

**Official Source:** Thai Ministry of Transport
**Year:** 2024 (most recent complete year)
**Total Records:** 23,952 accidents

**Bangkok Analysis:**
- Accidents analyzed: 2,669
- High-risk zones: 126
- Medium-risk zones: 85
- Low-risk zones: 65
- GPS accuracy: 97.6%

https://datagov.mot.go.th/en/dataset/roadaccident

## Project Structure

```
hackathon/
├── data/
│   ├── raw/          # Original CSV files
│   └── processed/    # Cleaned data
├── models/           # Trained XGBoost + Random Forest models
├── src/
│   ├── app.py        # Flask web application
│   ├── ml_model.py   # Risk prediction model
│   ├── nlp_analysis.py  # Thai NLP pipeline
│   └── ...
├── static/           # CSS, JS, maps
├── templates/        # HTML templates
├── requirements.txt
└── README.md
```
