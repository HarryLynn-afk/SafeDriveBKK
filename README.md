# RoadRisk: AI-Powered Accident Risk Prediction

Predicts next-week high-risk road zones for Bangkok and Phuket using Thai government accident data.

## Features

1. **Risk Prediction Map** - Interactive map showing predicted high-risk zones (1km grid)
2. **Danger Calendar** - Weekly risk heatmap highlighting holidays and festivals
3. **Route Comparison** - Compare fast vs. safe routes with risk tradeoffs
4. **Risk Coach** - Real-time alerts with specific safety actions
5. **Ops Dashboard** - Enterprise safety planning with PDF export

## Tech Stack

- **Data**: 23,952 accident records with GPS coordinates (Thai Ministry of Transport, 2024)
- **ML**: Random Forest classifier (scikit-learn)
- **Backend**: Flask REST API
- **Visualization**: Folium interactive maps, Seaborn charts
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

✅ Real government data  
✅ Most comprehensive Bangkok risk study

https://datagov.mot.go.th/en/dataset/roadaccident

## Project Structure

```
road-risk-app/
├── data/
│   ├── raw/          # Original CSV files
│   ├── processed/    # Cleaned data
│   └── models/       # Trained models
├── src/
│   ├── data_processing.py
│   ├── risk_model.py
│   ├── map_generator.py
│   └── app.py
├── notebooks/        # Jupyter notebooks for exploration
├── static/           # CSS, JS files
├── templates/        # HTML templates
├── requirements.txt
└── README.md
```
