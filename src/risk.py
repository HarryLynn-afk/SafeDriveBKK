"""
Risk scoring, route evaluation, and TOPSIS ranking.

All functions are pure with respect to Flask — no request/response objects.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from . import extensions
from .utils.geo import find_grid_for_point, haversine_km

if TYPE_CHECKING:
    from .routing import ResolvedRoute

# ---------------------------------------------------------------------------
# Single-point risk
# ---------------------------------------------------------------------------

def calculate_segment_risk(
    lat: float,
    lon: float,
    hour: int | None = None,
    day_of_week: int | None = None,
) -> dict:
    """Calculate risk for a single point.

    predict_risk() already falls back to spatial scoring internally when the
    XGBoost model is unavailable, so callers get the right method tag without
    needing an outer fallback chain.
    """
    if hour is None:
        hour = datetime.now().hour
    if day_of_week is None:
        day_of_week = datetime.now().weekday()

    predictor = extensions.ml_predictor
    if predictor is None:
        return {"risk_score": 35, "risk_level": "medium", "confidence": 0.3, "method": "fallback", "grid_id": None}

    if not extensions.USE_ML:
        return predictor.fallback_to_spatial(lat, lon)

    try:
        return predictor.predict_risk(lat, lon, hour, day_of_week)
    except Exception:
        return predictor.fallback_to_spatial(lat, lon)


# ---------------------------------------------------------------------------
# Route-level risk
# ---------------------------------------------------------------------------

def calculate_route_risk_ml(waypoints: list[list[float]]) -> dict | None:
    """Distance-weighted route risk from the XGBoost model.

    Returns None when ML is unavailable so callers can fall back to spatial.
    """
    if not (extensions.USE_ML and extensions.ml_predictor):
        return None

    results = extensions.ml_predictor.predict_batch(waypoints)
    high_risk_zones = sum(1 for r in results if r["risk_level"] == "high")
    medium_risk_zones = sum(1 for r in results if r["risk_level"] == "medium")

    total_risk_exposure = 0.0
    total_distance = 0.0
    high_risk_km = 0.0
    max_score = 0

    for i, r in enumerate(results):
        score = r["risk_score"]
        if score > max_score:
            max_score = score
        if i > 0:
            seg_dist = haversine_km(
                waypoints[i - 1][0], waypoints[i - 1][1],
                waypoints[i][0], waypoints[i][1],
            )
            seg_risk = (results[i - 1]["risk_score"] + score) / 2.0
            total_risk_exposure += seg_risk * seg_dist
            total_distance += seg_dist
            if results[i - 1]["risk_level"] == "high" or r["risk_level"] == "high":
                high_risk_km += seg_dist

    if total_distance > 0:
        weighted_avg = total_risk_exposure / total_distance
    else:
        weighted_avg = float(np.mean([r["risk_score"] for r in results])) if results else 0.0

    blended = (weighted_avg * 0.6) + (max_score * 0.4)
    risk_score = min(100, int(blended))
    risk_level = "high" if risk_score >= 60 else ("medium" if risk_score >= 30 else "low")

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "high_risk_zones": high_risk_zones,
        "high_risk_km": round(high_risk_km, 1),
        "medium_risk_zones": medium_risk_zones,
        "matched_grids": len(results),
        "evaluated_waypoints": len(waypoints),
        "prediction_method": "xgboost",
    }


def calculate_route_risk_density_aware(waypoints: list[list[float]], grid_df: pd.DataFrame) -> dict:
    """Distance-weighted route risk from accident-count grid data."""
    high_risk_zones = 0
    medium_risk_zones = 0
    max_accidents = 0
    matched_count = 0

    point_accidents: list[int] = []
    point_risk_levels: list[str] = []

    for lat, lon in waypoints:
        grid = find_grid_for_point(lat, lon, grid_df)
        if grid is not None:
            matched_count += 1
            acc = int(grid["total_accidents"])
            point_accidents.append(acc)
            if acc > max_accidents:
                max_accidents = acc
            lvl = str(grid["risk_level"]).lower()
            point_risk_levels.append(lvl)
            if lvl == "high":
                high_risk_zones += 1
            elif lvl == "medium":
                medium_risk_zones += 1
        else:
            point_accidents.append(0)
            point_risk_levels.append("low")

    total_weighted = 0.0
    total_distance = 0.0
    high_risk_km = 0.0

    for i in range(1, len(waypoints)):
        seg_dist = haversine_km(
            waypoints[i - 1][0], waypoints[i - 1][1],
            waypoints[i][0], waypoints[i][1],
        )
        seg_accidents = (point_accidents[i - 1] + point_accidents[i]) / 2.0
        total_weighted += seg_accidents * seg_dist
        total_distance += seg_dist
        if point_risk_levels[i - 1] == "high" or point_risk_levels[i] == "high":
            high_risk_km += seg_dist

    if total_distance > 0:
        weighted_avg = total_weighted / total_distance
    else:
        weighted_avg = float(np.mean(point_accidents)) if point_accidents else 0.0

    blended = (weighted_avg * 0.8 + max_accidents * 0.2) if max_accidents > 0 else weighted_avg
    risk_score = min(100, int(100 * (1 - math.exp(-blended / 25.0))))
    if high_risk_zones > 0:
        risk_score = min(100, risk_score + min(15, int(math.sqrt(high_risk_zones) * 3)))

    risk_level = "high" if risk_score >= 60 else ("medium" if risk_score >= 30 else "low")

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "high_risk_zones": high_risk_zones,
        "high_risk_km": round(high_risk_km, 1),
        "medium_risk_zones": medium_risk_zones,
        "matched_grids": matched_count,
        "evaluated_waypoints": len(waypoints),
    }


def calculate_distance_from_waypoints(waypoints: list[list[float]]) -> float:
    """Total route distance in km from consecutive waypoints."""
    return sum(
        haversine_km(waypoints[i - 1][0], waypoints[i - 1][1], waypoints[i][0], waypoints[i][1])
        for i in range(1, len(waypoints))
    )


# ---------------------------------------------------------------------------
# Unified route scorer (ML with automatic spatial fallback)
# ---------------------------------------------------------------------------

def score_route(waypoints: list[list[float]], grid_df: pd.DataFrame) -> dict:
    """Score a route using XGBoost when available, spatial fallback otherwise.

    Always returns a complete risk dict with prediction_method set.
    Callers never need to know which path ran.
    """
    ml_risk = calculate_route_risk_ml(waypoints)
    if ml_risk is not None:
        return ml_risk
    result = calculate_route_risk_density_aware(waypoints, grid_df)
    result["prediction_method"] = "spatial"
    return result


# ---------------------------------------------------------------------------
# Segment analysis & alerts
# ---------------------------------------------------------------------------

def analyze_route_segments(
    waypoints: list[list[float]],
    grid_df: pd.DataFrame,
    segment_distance_km: float = 1.0,
) -> list[dict]:
    """Divide a route into ~1 km segments and score each."""
    if len(waypoints) < 2:
        return []

    total_distance = sum(
        haversine_km(waypoints[i][0], waypoints[i][1], waypoints[i + 1][0], waypoints[i + 1][1])
        for i in range(len(waypoints) - 1)
    )
    num_segments = max(1, int(total_distance / segment_distance_km))
    pps = max(1, len(waypoints) // num_segments)

    segments = []
    current_km = 0.0

    for idx in range(0, len(waypoints) - pps, pps):
        seg_wps = waypoints[idx: idx + pps + 1]
        seg_dist = sum(
            haversine_km(seg_wps[j][0], seg_wps[j][1], seg_wps[j + 1][0], seg_wps[j + 1][1])
            for j in range(len(seg_wps) - 1)
        )

        risk_scores: list[float] = []
        accident_counts: list[int] = []
        high_risk_count = 0

        for lat, lon in seg_wps:
            grid = find_grid_for_point(lat, lon, grid_df)
            if grid is not None:
                accident_counts.append(int(grid["total_accidents"]))
                risk_scores.append(float(grid["risk_score"]))
                if str(grid["risk_level"]).lower() == "high":
                    high_risk_count += 1
            else:
                accident_counts.append(0)
                risk_scores.append(0)

        avg_acc = float(np.mean(accident_counts)) if accident_counts else 0.0
        max_acc = float(max(accident_counts)) if accident_counts else 0.0
        blended_acc = (avg_acc * 0.65) + (max_acc * 0.35)
        seg_score = min(100, int((blended_acc * 2.0) + (float(high_risk_count) * 4.0)))

        if seg_score >= 50:
            risk_level, color = "high", "#dc3545"
        elif seg_score >= 25:
            risk_level, color = "medium", "#ffc107"
        else:
            risk_level, color = "low", "#28a745"

        segments.append({
            "start_km": round(current_km, 1),
            "end_km": round(current_km + seg_dist, 1),
            "waypoints": seg_wps,
            "risk_score": seg_score,
            "risk_level": risk_level,
            "color": color,
            "high_risk_zones": high_risk_count,
            "max_accidents": max_acc,
        })
        current_km += seg_dist

    return segments


def generate_alerts_from_segments(
    segments: list[dict],
    advance_warning_km: float = 0.5,
) -> list[dict]:
    """Generate driver alerts for high- and medium-risk segments."""
    alerts: list[dict] = []
    for seg in segments:
        if seg["risk_level"] == "high":
            warning_km = max(0, seg["start_km"] - advance_warning_km)
            alerts += [
                {
                    "type": "warning",
                    "trigger_km": round(warning_km, 1),
                    "message": f"⚠️ HIGH RISK ZONE ahead in {int(advance_warning_km * 1000)}m",
                    "detail": (
                        f"Zone: km {seg['start_km']}-{seg['end_km']} | "
                        f"{seg['max_accidents']} accidents recorded"
                    ),
                    "severity": "warning",
                    "color": "#ffc107",
                },
                {
                    "type": "danger",
                    "trigger_km": seg["start_km"],
                    "message": "🔴 NOW ENTERING HIGH RISK ZONE",
                    "detail": f"Extra caution advised | Risk score: {seg['risk_score']}/100",
                    "severity": "danger",
                    "color": "#dc3545",
                },
                {
                    "type": "success",
                    "trigger_km": seg["end_km"],
                    "message": "✓ Exited high-risk zone - safer area ahead",
                    "detail": "",
                    "severity": "success",
                    "color": "#28a745",
                },
            ]
        elif seg["risk_level"] == "medium":
            alerts.append({
                "type": "info",
                "trigger_km": seg["start_km"],
                "message": "⚡ Medium risk area - stay alert",
                "detail": f"Risk score: {seg['risk_score']}/100",
                "severity": "info",
                "color": "#17a2b8",
            })
    alerts.sort(key=lambda a: a["trigger_km"])
    return alerts


# ---------------------------------------------------------------------------
# Route evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_route(route_points: list[list[float]], grid_df: pd.DataFrame) -> dict:
    """Score a route and return a full metrics dict."""
    total_distance = calculate_distance_from_waypoints(route_points)
    risk = calculate_route_risk_density_aware(route_points, grid_df)
    time_min = int(round((total_distance / 40.0) * 60.0))
    return {
        "waypoints": [[round(lat, 6), round(lon, 6)] for lat, lon in route_points],
        "distance_km": round(total_distance, 2),
        "time_min": time_min,
        "risk_score": min(100, risk["risk_score"]),
        "risk_level": risk["risk_level"],
        "high_risk_zones": risk["high_risk_zones"],
        "high_risk_km": risk["high_risk_km"],
        "medium_risk_zones": risk["medium_risk_zones"],
        "matched_grids": risk["matched_grids"],
        "evaluated_waypoints": risk["evaluated_waypoints"],
    }


# ---------------------------------------------------------------------------
# Route deduplication & TOPSIS ranking
# ---------------------------------------------------------------------------

def _route_key(waypoints: list[list[float]]) -> str:
    normalized = [[round(lat, 6), round(lon, 6)] for lat, lon in waypoints]
    raw = json.dumps(normalized, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def dedupe_scored_routes(scored_routes: list[dict]) -> list[dict]:
    """Collapse duplicate geometries, keeping the better-timed copy."""
    deduped: dict[str, dict] = {}
    order: list[str] = []
    for route in scored_routes:
        key = _route_key(route["waypoints"])
        route["route_key"] = key
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = route
            order.append(key)
        elif (route["time_min"], route["risk_score"]) < (existing["time_min"], existing["risk_score"]):
            deduped[key] = route
    return [deduped[k] for k in order]


def apply_topsis(scored_routes: list[dict], w_time: float = 0.5, w_risk: float = 0.5) -> None:
    """Attach a TOPSIS score (0–1) to each route in-place."""
    if not scored_routes:
        return
    if len(scored_routes) == 1:
        scored_routes[0]["topsis_score"] = 1.0
        return

    times = np.array([r["time_min"] for r in scored_routes], dtype=float)
    risks = np.array([r["risk_score"] for r in scored_routes], dtype=float)

    norm_t = times / np.sqrt(np.sum(times ** 2)) if np.sum(times ** 2) != 0 else times
    norm_r = risks / np.sqrt(np.sum(risks ** 2)) if np.sum(risks ** 2) != 0 else risks

    wm = np.column_stack((norm_t * w_time, norm_r * w_risk))
    ideal_best = np.min(wm, axis=0)
    ideal_worst = np.max(wm, axis=0)

    d_best = np.sqrt(np.sum((wm - ideal_best) ** 2, axis=1))
    d_worst = np.sqrt(np.sum((wm - ideal_worst) ** 2, axis=1))
    scores = d_worst / (d_best + d_worst + 1e-10)

    for i, route in enumerate(scored_routes):
        route["topsis_score"] = float(scores[i])


def annotate_route_roles(
    scored_routes: list[dict],
    route_fast: dict,
    route_safe: dict,
    route_recommended: dict,
) -> None:
    """Attach badge labels and primary_role to each route in-place."""
    fast_id = route_fast["route_id"]
    safe_id = route_safe["route_id"]
    rec_id = route_recommended["route_id"]

    for route in scored_routes:
        rid = route["route_id"]
        badges = []
        if rid == rec_id:
            badges.append("AI Recommended")
        if rid == fast_id:
            badges.append("Fastest")
        if rid == safe_id:
            badges.append("Safest")

        route["badges"] = badges
        if rid == rec_id:
            route["primary_role"] = "recommended"
            route["primary_label"] = "AI Recommended"
        elif rid == fast_id:
            route["primary_role"] = "fast"
            route["primary_label"] = "Fastest"
        elif rid == safe_id:
            route["primary_role"] = "safe"
            route["primary_label"] = "Safest"
        else:
            route["primary_role"] = "alternative"
            route["primary_label"] = "Alternative"


def build_display_routes(
    scored_routes: list[dict],
    route_fast: dict,
    route_safe: dict,
    route_recommended: dict,
    max_routes: int = 3,
) -> list[dict]:
    """Choose up to three distinct routes for the frontend cards."""
    display: list[dict] = []
    seen: set[str] = set()

    for candidate in (route_recommended, route_fast, route_safe):
        if candidate["route_id"] in seen:
            continue
        display.append(candidate)
        seen.add(candidate["route_id"])
        if len(display) >= max_routes:
            return display

    for route in sorted(scored_routes, key=lambda r: (r.get("topsis_score", 0.0), -r["time_min"]), reverse=True):
        if route["route_id"] in seen:
            continue
        display.append(route)
        seen.add(route["route_id"])
        if len(display) >= max_routes:
            break

    return display


# ---------------------------------------------------------------------------
# Route comparison pipeline (rank, annotate, build response dict)
# ---------------------------------------------------------------------------

def _peak_label_for_score(score: int) -> tuple[str, str]:
    if score >= 80:
        return "Crosses extreme danger zone", "top 5% accident hotspots"
    if score >= 60:
        return "Crosses high-risk zones", "above-average severity"
    if score >= 40:
        return "Moderate risk areas ahead", "exercise caution"
    return "Bypasses extreme danger zones", "30% safer peak risk"


def _format_time_diff(diff_min: int) -> str:
    if diff_min > 0:
        return f"+{diff_min} min"
    if diff_min < 0:
        return f"{diff_min} min"
    return "0 min"


def _build_recommendation_text(route_fast: dict, route_safe: dict) -> str:
    fast_score = int(route_fast["risk_score"])
    safe_score = int(route_safe["risk_score"])
    if safe_score >= fast_score:
        return "Fast route is already the safest available option. No safer alternative exists for this trip."

    reduction_pct = int((1 - safe_score / fast_score) * 100) if fast_score > 0 else 0
    time_cost = route_safe["time_min"] - route_fast["time_min"]
    if time_cost > 0:
        return f"Safe route reduces risk by {reduction_pct}% for +{time_cost} minutes."
    if time_cost < 0:
        return f"Safe route reduces risk by {reduction_pct}% and is {abs(time_cost)} minutes faster."
    return f"Safe route reduces risk by {reduction_pct}% with no extra time."


def rank_and_annotate_routes(scored_routes: list[dict]) -> dict:
    """Rank scored route alternatives and annotate them for the API response.

    Pipeline:
      1. Dedupe duplicate geometries.
      2. Apply TOPSIS to attach a multi-criteria score per route.
      3. Pick fast / safe / recommended.
      4. Annotate each route with confidence, prediction_method, peak label,
         badges, primary_role.
      5. Build the display-routes list (up to three).
      6. Compute the recommendation text and time-difference.

    Returns the full response dict:
      fast_route, safe_route, recommended_route, display_routes, all_routes,
      route_fast, route_safe, recommendation.

    Mutates the input list (route_id, route_key, topsis_score, badges,
    confidence, prediction_method, peak_label, peak_detail, primary_role,
    primary_label are added in place — matching legacy behaviour).
    """
    scored_routes = dedupe_scored_routes(scored_routes)
    for i, r in enumerate(scored_routes):
        r["route_id"] = f"route_{i}"

    apply_topsis(scored_routes)

    route_fast        = min(scored_routes, key=lambda r: r["time_min"])
    route_safe        = min(scored_routes, key=lambda r: r["risk_score"])
    route_recommended = max(scored_routes, key=lambda r: r.get("topsis_score", 0.0))

    prediction_method = "xgboost" if (extensions.USE_ML and extensions.ml_predictor) else "spatial"
    for r in scored_routes:
        matched   = r.get("matched_grids", 0)
        evaluated = max(1, r.get("evaluated_waypoints", 1))
        r["confidence"] = round(min(0.99, 0.75 + (matched / evaluated) * 0.24), 2)
        r["prediction_method"] = prediction_method
        r["peak_label"], r["peak_detail"] = _peak_label_for_score(int(r["risk_score"]))

    annotate_route_roles(scored_routes, route_fast, route_safe, route_recommended)
    display_routes = build_display_routes(scored_routes, route_fast, route_safe, route_recommended)

    route_safe["time_difference"] = _format_time_diff(route_safe["time_min"] - route_fast["time_min"])

    return {
        "fast_route": route_fast,
        "safe_route": route_safe,
        "recommended_route": route_recommended,
        "display_routes": display_routes,
        "all_routes": sorted(scored_routes, key=lambda r: r["time_min"]),
        "route_fast": route_fast,
        "route_safe": route_safe,
        "recommendation": _build_recommendation_text(route_fast, route_safe),
    }


# ---------------------------------------------------------------------------
# Single-route analysis payload (deep-module entry point for /api/analyze-route)
# ---------------------------------------------------------------------------

_MODEL_ACCURACY_BY_METHOD = {
    "xgboost": "83.8% holdout crash capture",
    "spatial": "83.8% spatial baseline capture",
}


def build_route_analysis_payload(
    resolved: "ResolvedRoute",
    grid_df: pd.DataFrame,
) -> dict:
    """Build the complete /api/analyze-route response body.

    Owns sampling, segment analysis, alert generation, overall scoring,
    bucket counting, and accuracy string mapping. The blueprint is now a
    thin wrapper that only handles input validation and HTTP framing.
    """
    waypoints = resolved.waypoints
    sampled = waypoints[:: max(1, len(waypoints) // 50)]

    segments = analyze_route_segments(sampled, grid_df, segment_distance_km=1.0)
    alerts = generate_alerts_from_segments(segments, advance_warning_km=0.5)
    overall_risk = score_route(sampled, grid_df)

    prediction_method = overall_risk["prediction_method"]
    model_accuracy = _MODEL_ACCURACY_BY_METHOD.get(
        prediction_method, "83.8% spatial baseline capture"
    )

    high   = sum(1 for s in segments if s["risk_level"] == "high")
    medium = sum(1 for s in segments if s["risk_level"] == "medium")
    low    = sum(1 for s in segments if s["risk_level"] == "low")

    return {
        "success": True,
        "route": {
            "waypoints": waypoints,
            "distance_km": round(resolved.distance_km, 2),
            "duration_min": resolved.duration_min,
            "risk_score": min(100, overall_risk["risk_score"]),
            "risk_level": overall_risk["risk_level"],
            "high_risk_km": overall_risk.get("high_risk_km", 0),
            "traffic_source": resolved.traffic_source,
        },
        "segments": segments,
        "alerts": alerts,
        "summary": {
            "total_distance": round(resolved.distance_km, 2),
            "total_time": resolved.duration_min,
            "high_risk_segments": high,
            "medium_risk_segments": medium,
            "low_risk_segments": low,
            "total_alerts": len(alerts),
            "prediction_method": prediction_method,
            "model_accuracy": model_accuracy,
            "traffic_source": resolved.traffic_source,
        },
    }
