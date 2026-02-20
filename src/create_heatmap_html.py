from pathlib import Path

import folium
import numpy as np
import pandas as pd
from folium.plugins import HeatMap


BASE_DIR = Path(__file__).resolve().parent.parent
ACCIDENTS_PATH = BASE_DIR / "data" / "processed" / "bangkok_accidents_clean.csv"
GRID_STATS_PATH = BASE_DIR / "data" / "processed" / "bangkok_grid_stats.csv"
OUTPUT_PATH = BASE_DIR / "static" / "bangkok_risk_heatmap.html"
CELL_SIZE_DEG = 0.009
BKK_LAT_MIN = 13.6
BKK_LON_MIN = 100.35


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


def _assign_grid_id(lat, lon):
    """Convert lat/lon to grid_id matching the training pipeline."""
    lat_idx = int((lat - BKK_LAT_MIN) / CELL_SIZE_DEG)
    lon_idx = int((lon - BKK_LON_MIN) / CELL_SIZE_DEG)
    return f"G{lat_idx:03d}_{lon_idx:03d}"


def _compute_grid_details(accidents, lat_col, lon_col):
    """Compute per-grid detailed stats from raw accident data."""
    df = accidents.copy()
    df["_lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["_lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=["_lat", "_lon"])
    df["_grid_id"] = df.apply(lambda r: _assign_grid_id(r["_lat"], r["_lon"]), axis=1)

    day_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}

    details = {}
    for grid_id, group in df.groupby("_grid_id"):
        n = len(group)

        # Peak hour
        hours = pd.to_numeric(group.get("hour", pd.Series(dtype=float)), errors="coerce").dropna()
        peak_hour = f"{int(hours.mode().iloc[0]):02d}:00" if len(hours) > 0 else "N/A"

        # Peak day
        days = pd.to_numeric(group.get("day_of_week", pd.Series(dtype=float)), errors="coerce").dropna().astype(int)
        peak_day = day_map.get(int(days.mode().iloc[0]), "N/A") if len(days) > 0 else "N/A"

        # Severity
        deaths = int(pd.to_numeric(group.get("ผู้เสียชีวิต", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        serious = int(pd.to_numeric(group.get("ผู้บาดเจ็บสาหัส", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        minor = int(pd.to_numeric(group.get("ผู้บาดเจ็บเล็กน้อย", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())

        # Top cause
        cause_col = "มูลเหตุสันนิษฐาน"
        top_cause = "N/A"
        if cause_col in group.columns:
            causes = group[cause_col].dropna()
            if len(causes) > 0:
                top_cause = str(causes.mode().iloc[0])
                if len(top_cause) > 25:
                    top_cause = top_cause[:25] + "..."

        # Motorcycle involvement
        moto_col = "รถจักรยานยนต์"
        moto_pct = 0
        if moto_col in group.columns:
            moto_involved = int((pd.to_numeric(group[moto_col], errors="coerce").fillna(0) > 0).sum())
            moto_pct = round((moto_involved / n) * 100) if n > 0 else 0

        details[grid_id] = {
            "peak_hour": peak_hour,
            "peak_day": peak_day,
            "deaths": deaths,
            "serious": serious,
            "minor": minor,
            "top_cause": top_cause,
            "moto_pct": moto_pct,
        }

    return details


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

    # Pre-compute detailed per-grid stats for richer popups
    grid_details = _compute_grid_details(accidents, lat_col, lon_col)

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
        grid_id = row.get("grid_id", "N/A")
        detail = grid_details.get(grid_id, {})
        acc_count = int(row["total_accidents"])
        score = float(row["risk_score"])

        # Build rich popup with drill-down stats
        severity_line = ""
        if detail:
            parts = []
            if detail.get("deaths", 0) > 0:
                parts.append(f"<span style='color:#dc2626;font-weight:bold'>{detail['deaths']} deaths</span>")
            if detail.get("serious", 0) > 0:
                parts.append(f"<span style='color:#ea580c'>{detail['serious']} serious</span>")
            if detail.get("minor", 0) > 0:
                parts.append(f"{detail['minor']} minor")
            if parts:
                severity_line = f"<div style='margin:4px 0'><b>Injuries:</b> {', '.join(parts)}</div>"

        popup_html = (
            f"<div style='font-family:Arial,sans-serif;font-size:13px;min-width:220px'>"
            f"<div style='font-size:15px;font-weight:bold;margin-bottom:6px;border-bottom:2px solid {level_color};padding-bottom:4px'>"
            f"{risk_level.upper()} RISK — {grid_id}</div>"
            f"<div><b>Accidents:</b> {acc_count} &nbsp;|&nbsp; <b>Score:</b> {score:.0f}/100</div>"
            f"{severity_line}"
        )
        if detail:
            popup_html += (
                f"<div style='margin:4px 0'><b>Peak hour:</b> {detail.get('peak_hour', 'N/A')} "
                f"&nbsp;|&nbsp; <b>Peak day:</b> {detail.get('peak_day', 'N/A')}</div>"
            )
            if detail.get("top_cause", "N/A") != "N/A":
                popup_html += f"<div><b>Top cause:</b> {detail['top_cause']}</div>"
            if detail.get("moto_pct", 0) > 0:
                popup_html += f"<div><b>Motorcycle:</b> {detail['moto_pct']}% of accidents</div>"
        popup_html += "</div>"

        folium.Rectangle(
            bounds=bounds,
            color=level_color,
            fill=True,
            fill_color=level_color,
            fill_opacity=0.18,
            weight=1.5,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"{risk_level.title()} risk — {acc_count} accidents (click for details)",
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
