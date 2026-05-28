"""
Create an interactive Bangkok accident risk map from grid statistics.
"""

import logging
from pathlib import Path

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

LOGGER = logging.getLogger(__name__)

# Grid constants single-sourced from config.py; dual import supports both
# `python -m src.create_risk_map` and `python src/create_risk_map.py`.
try:
    from .config import (
        BKK_LAT_MAX,
        BKK_LAT_MIN,
        BKK_LON_MAX,
        BKK_LON_MIN,
        DEGREES_PER_KM,
    )
except ImportError:
    from config import (
        BKK_LAT_MAX,
        BKK_LAT_MIN,
        BKK_LON_MAX,
        BKK_LON_MIN,
        DEGREES_PER_KM,
    )

CELL_SIZE_KM = 1.0
CELL_SIZE_DEG = CELL_SIZE_KM * DEGREES_PER_KM

RISK_COLORS = {
    "low": "#90EE90",
    "medium": "#FFD700",
    "high": "#FF6B6B",
}


def _add_grid_bounds(grid_df):
    """
    Ensure each row has lat/lon grid bounds.
    If bounds are not present, infer from grid_id pattern G{lat_idx}_{lon_idx}.
    """
    required_bounds = {"lat_min", "lat_max", "lon_min", "lon_max"}
    if required_bounds.issubset(grid_df.columns):
        return grid_df.copy()

    parsed = grid_df["grid_id"].astype(str).str.extract(r"^G(\d{3})_(\d{3})$")
    if parsed.isna().any().any():
        bad_rows = grid_df.loc[parsed.isna().any(axis=1), "grid_id"].head(5).tolist()
        raise ValueError(f"Cannot parse grid_id format for rows like: {bad_rows}")

    lat_idx = parsed[0].astype(int)
    lon_idx = parsed[1].astype(int)

    with_bounds = grid_df.copy()
    with_bounds["lat_min"] = BKK_LAT_MIN + (lat_idx * CELL_SIZE_DEG)
    with_bounds["lat_max"] = (with_bounds["lat_min"] + CELL_SIZE_DEG).clip(upper=BKK_LAT_MAX)
    with_bounds["lon_min"] = BKK_LON_MIN + (lon_idx * CELL_SIZE_DEG)
    with_bounds["lon_max"] = (with_bounds["lon_min"] + CELL_SIZE_DEG).clip(upper=BKK_LON_MAX)
    return with_bounds


def _add_title_and_legend(risk_map):
    title_html = """
    <div style="
        position: fixed;
        top: 10px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 9999;
        background-color: rgba(255,255,255,0.92);
        padding: 8px 14px;
        border-radius: 6px;
        font-size: 16px;
        font-weight: bold;
        border: 1px solid #ccc;">
        Bangkok Accident Risk Zones (1km Grid)
    </div>
    """

    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        z-index: 9999;
        background-color: rgba(255,255,255,0.95);
        padding: 12px;
        border-radius: 6px;
        border: 1px solid #ccc;
        font-size: 13px;">
        <div style="font-weight: bold; margin-bottom: 8px;">Risk Level</div>
        <div><span style="display:inline-block;width:12px;height:12px;background:#90EE90;margin-right:8px;"></span>Low Risk (0-1 accidents)</div>
        <div><span style="display:inline-block;width:12px;height:12px;background:#FFD700;margin-right:8px;"></span>Medium Risk (2-4 accidents)</div>
        <div><span style="display:inline-block;width:12px;height:12px;background:#FF6B6B;margin-right:8px;"></span>High Risk (5+ accidents)</div>
    </div>
    """

    risk_map.get_root().html.add_child(folium.Element(title_html))
    risk_map.get_root().html.add_child(folium.Element(legend_html))


def create_static_png(grid_stats_df, output_path):
    """Create static PNG of risk map using matplotlib."""
    grid_data = grid_stats_df[grid_stats_df["total_accidents"] > 0].copy()

    colors = {
        "high": "#FF6B6B",
        "medium": "#FFD700",
        "low": "#90EE90",
    }

    risk_counts = grid_data["risk_level"].str.lower().value_counts()
    low_count = int(risk_counts.get("low", 0))
    medium_count = int(risk_counts.get("medium", 0))
    high_count = int(risk_counts.get("high", 0))
    total_count = int(len(grid_data))

    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)

    for _, row in grid_data.iterrows():
        rect = patches.Rectangle(
            (row["lon_min"], row["lat_min"]),
            row["lon_max"] - row["lon_min"],
            row["lat_max"] - row["lat_min"],
            linewidth=0.5,
            edgecolor="white",
            facecolor=colors.get(str(row["risk_level"]).lower(), "#90EE90"),
            alpha=0.6,
        )
        ax.add_patch(rect)

    ax.set_xlim(100.3, 100.9)
    ax.set_ylim(13.5, 14.0)
    ax.set_aspect("equal")

    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title(
        "Bangkok Accident Risk Zones (2024 Data - 1km Grid)",
        fontsize=16,
        fontweight="bold",
    )

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#90EE90", label=f"Low Risk (0-1 accidents): {low_count} zones"),
        Patch(facecolor="#FFD700", label=f"Medium Risk (2-4 accidents): {medium_count} zones"),
        Patch(facecolor="#FF6B6B", label=f"High Risk (5+ accidents): {high_count} zones"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()

    print(f"PNG saved: {output_path}")
    print("Resolution: 1920x1080")
    print(f"Contains {total_count} zones from 2024 data")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    input_path = Path("data/processed/bangkok_grid_stats.csv")
    output_path = Path("data/processed/bangkok_risk_map.html")

    try:
        if not input_path.exists():
            raise FileNotFoundError(f"Grid stats file not found: {input_path}")

        df = pd.read_csv(input_path, encoding="utf-8-sig")
        required_cols = {"grid_id", "total_accidents", "risk_score", "risk_level"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        df = _add_grid_bounds(df)
        non_empty = df[df["total_accidents"] > 0].copy()
        LOGGER.info("Loaded %d grid rows, visualizing %d cells with accidents", len(df), len(non_empty))

        risk_map = folium.Map(location=[13.75, 100.56], zoom_start=11, tiles="CartoDB positron")

        for row in non_empty.itertuples(index=False):
            risk_level = str(row.risk_level).lower()
            color = RISK_COLORS.get(risk_level, "#90EE90")
            popup_html = (
                f"<b>Grid ID:</b> {row.grid_id}<br>"
                f"<b>Total accidents:</b> {int(row.total_accidents)}<br>"
                f"<b>Risk score:</b> {float(row.risk_score):.2f}<br>"
                f"<b>Risk level:</b> {risk_level.title()}"
            )

            folium.Rectangle(
                bounds=[[row.lat_min, row.lon_min], [row.lat_max, row.lon_max]],
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                weight=1,
                popup=folium.Popup(popup_html, max_width=250),
            ).add_to(risk_map)

        _add_title_and_legend(risk_map)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        risk_map.save(output_path)
        create_static_png(df, "data/processed/bangkok_risk_map.png")

        level_counts = non_empty["risk_level"].str.lower().value_counts()
        low_count = int(level_counts.get("low", 0))
        medium_count = int(level_counts.get("medium", 0))
        high_count = int(level_counts.get("high", 0))

        LOGGER.info("Saved interactive risk map to %s", output_path)
        print(f"Total grids visualized: {len(non_empty)}")
        print(f"Risk breakdown -> low: {low_count}, medium: {medium_count}, high: {high_count}")
        print(f"HTML output: {output_path}")

    except Exception as exc:
        LOGGER.exception("Failed to create risk map: %s", exc)
        raise


if __name__ == "__main__":
    main()
