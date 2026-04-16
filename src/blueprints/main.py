"""
Page-level routes: home, trip planner, analytics, about, NLP, risk map.
"""

import logging

from flask import Blueprint, jsonify, render_template, send_file

from ..config import GOOGLE_MAPS_BROWSER_KEY, RISK_HEATMAP_HTML_PATH, RISK_MAP_HTML_PATH

logger = logging.getLogger(__name__)
from ..extensions import CHATBOT_DATA
from ..utils.data import calculate_temporal_stats, load_bangkok_accidents, load_grid_stats

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def home():
    try:
        accidents_df = load_bangkok_accidents()
        grid_df = load_grid_stats()
        temporal = calculate_temporal_stats(accidents_df)

        total = int(len(accidents_df))
        high_risk = int(
            ((grid_df["risk_level"].str.lower() == "high") & (grid_df["total_accidents"] > 0)).sum()
        )
        peak_time = temporal["peak_hours"][0]["time_range"] if temporal["peak_hours"] else "N/A"
        dangerous_day = temporal["dangerous_days"][0]["day_name"] if temporal["dangerous_days"] else "N/A"

        return render_template(
            "index.html",
            total_accidents=total,
            high_risk_zones=high_risk,
            peak_time=peak_time,
            dangerous_day=dangerous_day,
            total_accidents_card=f"{total:,} Accidents Analyzed",
            high_risk_card=f"{high_risk:,} High-Risk Zones",
            peak_card=f"Peak: {peak_time}",
            dangerous_day_card=f"{dangerous_day} Most Dangerous",
            map_available=RISK_HEATMAP_HTML_PATH.exists(),
        )
    except Exception as exc:
        logger.exception("Error loading home page data")
        return render_template(
            "index.html",
            total_accidents=0,
            high_risk_zones=0,
            peak_time="N/A",
            dangerous_day="N/A",
            total_accidents_card="0 Accidents Analyzed",
            high_risk_card="0 High-Risk Zones",
            peak_card="Peak: N/A",
            dangerous_day_card="N/A Most Dangerous",
            map_available=False,
            error_message="Data temporarily unavailable.",
        ), 500


@main_bp.route("/trip-planner")
def trip_planner():
    return render_template("trip_planner.html", google_maps_api_key=GOOGLE_MAPS_BROWSER_KEY)


@main_bp.route("/about")
def about():
    return render_template("about.html")


@main_bp.route("/analytics")
def analytics_page():
    return render_template("analytics.html")


@main_bp.route("/nlp")
def nlp():
    assistant_stats = CHATBOT_DATA.get("stats", {}) if isinstance(CHATBOT_DATA, dict) else {}
    return render_template("nlp.html", assistant_stats=assistant_stats)


@main_bp.route("/risk-map")
def risk_map():
    if not RISK_MAP_HTML_PATH.exists():
        return jsonify({"error": "Risk map is not available."}), 404
    return send_file(RISK_MAP_HTML_PATH)
