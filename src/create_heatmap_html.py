from pathlib import Path

import folium
import pandas as pd
from folium.plugins import HeatMap


BASE_DIR = Path(__file__).resolve().parent.parent
ACCIDENTS_PATH = BASE_DIR / "data" / "processed" / "bangkok_accidents_clean.csv"
GRID_STATS_PATH = BASE_DIR / "data" / "processed" / "bangkok_grid_stats.csv"
OUTPUT_PATH = BASE_DIR / "static" / "bangkok_risk_heatmap.html"
CELL_SIZE_DEG = 0.009


def _get_coordinate_columns(df):
    if {"LATITUDE", "LONGITUDE"}.issubset(df.columns):
        return "LATITUDE", "LONGITUDE"
    if {"latitude", "longitude"}.issubset(df.columns):
        return "latitude", "longitude"
    raise ValueError("Could not find coordinate columns in processed accidents data")


def _prepare_grid_bounds(grid_df):
    required_cols = {"lat_min", "lat_max", "lon_min", "lon_max"}
    if required_cols.issubset(grid_df.columns):
        return grid_df.copy()

    if not {"center_lat", "center_lon"}.issubset(grid_df.columns):
        raise ValueError("Grid stats missing center_lat/center_lon for bounds derivation")

    result = grid_df.copy()
    half_cell = CELL_SIZE_DEG / 2
    result["lat_min"] = pd.to_numeric(result["center_lat"], errors="coerce") - half_cell
    result["lat_max"] = pd.to_numeric(result["center_lat"], errors="coerce") + half_cell
    result["lon_min"] = pd.to_numeric(result["center_lon"], errors="coerce") - half_cell
    result["lon_max"] = pd.to_numeric(result["center_lon"], errors="coerce") + half_cell
    return result


def create_heatmap_html():
    """Create dashboard heat map HTML from processed Bangkok accidents."""
    accidents = pd.read_csv(ACCIDENTS_PATH, encoding="utf-8-sig")
    lat_col, lon_col = _get_coordinate_columns(accidents)

    coords = accidents[[lat_col, lon_col]].apply(pd.to_numeric, errors="coerce").dropna()
    heat_data = [[float(lat), float(lon)] for lat, lon in coords.to_numpy()]
    grid_stats = pd.read_csv(GRID_STATS_PATH, encoding="utf-8-sig")
    grid_stats = _prepare_grid_bounds(grid_stats)
    grid_stats["total_accidents"] = pd.to_numeric(grid_stats["total_accidents"], errors="coerce").fillna(0)
    grid_stats["risk_score"] = pd.to_numeric(grid_stats["risk_score"], errors="coerce").fillna(0)
    grid_stats = grid_stats[grid_stats["total_accidents"] > 0].copy()
    grid_stats["risk_level"] = grid_stats["risk_level"].astype(str).str.lower()

    bangkok_map = folium.Map(location=[13.75, 100.53], zoom_start=11, tiles="OpenStreetMap")

    heat_layer = folium.FeatureGroup(name="Accident Density Heatmap", show=True)
    HeatMap(
        heat_data,
        min_opacity=0.3,
        max_opacity=0.8,
        radius=15,
        blur=20,
        max_zoom=13,
        gradient={
            0.0: "#28a745",
            0.4: "#a3e635",
            0.6: "#fbbf24",
            0.8: "#f97316",
            1.0: "#dc2626",
        },
    ).add_to(heat_layer)
    heat_layer.add_to(bangkok_map)

    risk_layer_config = {
        "high": {"name": "High-Risk Segments", "color": "#dc2626"},
        "medium": {"name": "Medium-Risk Segments", "color": "#fbbf24"},
        "low": {"name": "Low-Risk Segments", "color": "#28a745"},
    }
    risk_layers = {
        level: folium.FeatureGroup(name=config["name"], show=(level != "low"))
        for level, config in risk_layer_config.items()
    }

    for _, row in grid_stats.iterrows():
        risk_level = row.get("risk_level", "low")
        if risk_level not in risk_layers:
            risk_level = "low"
        level_color = risk_layer_config[risk_level]["color"]

        bounds = [
            [float(row["lat_min"]), float(row["lon_min"])],
            [float(row["lat_max"]), float(row["lon_max"])],
        ]
        popup_html = (
            f"<b>Grid:</b> {row.get('grid_id', 'N/A')}<br>"
            f"<b>Risk level:</b> {risk_level.title()}<br>"
            f"<b>Risk score:</b> {float(row['risk_score']):.0f}/100<br>"
            f"<b>Accidents:</b> {int(row['total_accidents'])}"
        )
        folium.Rectangle(
            bounds=bounds,
            color=level_color,
            fill=True,
            fill_color=level_color,
            fill_opacity=0.18,
            weight=1.5,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"{risk_level.title()} risk segment",
        ).add_to(risk_layers[risk_level])

    for layer in risk_layers.values():
        layer.add_to(bangkok_map)

    title_html = """
    <div style="position: fixed;
                top: 10px;
                left: 50px;
                width: 400px;
                background: white;
                padding: 10px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                z-index: 9999;
                font-family: Arial, sans-serif;">
        <h4 style="margin: 0; font-size: 16px;">Bangkok Accident Risk Map (Heat + Segments)</h4>
        <p style="margin: 5px 0 0 0; font-size: 12px; color: #666;">
            Heat intensity plus color-coded risk segments from 2024 records
        </p>
    </div>
    """
    bangkok_map.get_root().html.add_child(folium.Element(title_html))

    legend_html = """
    <div style="position: fixed;
                bottom: 50px;
                left: 50px;
                background: white;
                padding: 10px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                z-index: 9999;
                font-family: Arial, sans-serif;">
        <div style="margin-bottom: 5px; font-weight: bold; font-size: 14px;">Segment Risk Level</div>
        <div style="display: flex; align-items: center; margin: 3px 0;">
            <div style="width: 20px; height: 20px; background: #28a745; margin-right: 8px; border-radius: 3px;"></div>
            <span style="font-size: 12px;">Low (Green)</span>
        </div>
        <div style="display: flex; align-items: center; margin: 3px 0;">
            <div style="width: 20px; height: 20px; background: #fbbf24; margin-right: 8px; border-radius: 3px;"></div>
            <span style="font-size: 12px;">Medium (Yellow)</span>
        </div>
        <div style="display: flex; align-items: center; margin: 3px 0;">
            <div style="width: 20px; height: 20px; background: #dc2626; margin-right: 8px; border-radius: 3px;"></div>
            <span style="font-size: 12px;">High (Red)</span>
        </div>
        <div style="margin-top: 6px; font-size: 11px; color: #666;">Use top-right layer control to toggle segment layers.</div>
    </div>
    """
    bangkok_map.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(bangkok_map)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    bangkok_map.save(str(OUTPUT_PATH))
    print(f"Created: {OUTPUT_PATH}")
    print(f"Heat points: {len(heat_data)}")
    print(f"Risk segments: {len(grid_stats)}")


if __name__ == "__main__":
    create_heatmap_html()
