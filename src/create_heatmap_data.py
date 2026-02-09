import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
ACCIDENTS_PATH = BASE_DIR / "data" / "processed" / "bangkok_accidents_clean.csv"
OUTPUT_PATH = BASE_DIR / "static" / "heatmap_data.json"


def _get_coordinate_columns(df):
    if {"LATITUDE", "LONGITUDE"}.issubset(df.columns):
        return "LATITUDE", "LONGITUDE"
    if {"latitude", "longitude"}.issubset(df.columns):
        return "latitude", "longitude"
    raise ValueError("Could not find coordinate columns in processed accidents data")


def create_heatmap_data():
    """Generate heat map data points from processed Bangkok accidents."""
    accidents = pd.read_csv(ACCIDENTS_PATH, encoding="utf-8-sig")
    lat_col, lon_col = _get_coordinate_columns(accidents)

    coords = accidents[[lat_col, lon_col]].apply(pd.to_numeric, errors="coerce").dropna()
    heat_data = [[float(lat), float(lon), 1.0] for lat, lon in coords.to_numpy()]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file_handle:
        json.dump(heat_data, file_handle)

    print(f"Generated {len(heat_data)} heat points")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    create_heatmap_data()
