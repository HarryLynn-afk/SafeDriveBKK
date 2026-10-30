#!/usr/bin/env python3
"""
Chatbot Data Processor
Aggregates data from multiple sources into a single chatbot-friendly JSON file.

Sources:
  - static/nlp_cache.json         (NLP cause analysis, Thai keywords, recommendations)
  - data/processed/ml_features.csv (per-zone features, cause breakdowns, vehicle mix)
  - data/processed/bangkok_grid_stats.csv (risk scores, risk levels)
  - data/processed/bangkok_accidents_clean.csv (raw temporal/severity stats)
  - static/camera_locations.json  (camera GPS + location names)

Output:
  - static/chatbot_data.json
"""

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
STATIC_DIR = BASE_DIR / "static"

# Known landmark names for top grid zones (reverse-geocoded from coordinates)
GRID_LOCATION_NAMES = {
    "G020_002": "Thon Buri / Phetkasem Highway (Km 24)",
    "G014_041": "Lat Krabang / Motorway (Eastern)",
    "G014_043": "Lat Krabang / Motorway (East End)",
    "G014_039": "Lat Krabang / Highway 3242",
    "G014_040": "Lat Krabang / Motorway Junction",
    "G014_042": "Lat Krabang / Industrial Zone",
    "G020_005": "Phutthamonthon / Borommaratchachonnani Rd",
    "G020_001": "Thon Buri / Phetkasem Outer Ring",
    "G020_003": "Thon Buri / Phetkasem Highway (Km 16)",
    "G020_006": "Phutthamonthon / Highway 338",
    "G020_008": "Taling Chan / Kanchanaphisek Expressway",
    "G020_010": "Taling Chan / Si Rat Expressway",
    "G020_007": "Phutthamonthon / Borommaratchachonnani Soi",
    "G024_036": "Min Buri / Ramkhamhaeng Rd (East)",
    "G020_012": "Bang Kruai / Charansanitwong Rd",
    "G025_036": "Nong Chok / Suwinthawong Rd",
    "G004_006": "Bang Khun Thian / Rama II Highway",
    "G018_038": "Saphan Sung / Ramkhamhaeng Rd",
    "G020_000": "Thon Buri / Phetkasem Outer Ring (West)",
    "G020_011": "Taling Chan / Ratchaphruek Rd",
    "G002_004": "Bang Bon / Ekkachai Rd",
    "G005_007": "Bang Khae / Kanchanaphisek Rd",
    "G014_044": "Lat Krabang / Chaloem Phra Kiat Rd",
    "G020_004": "Thon Buri / Phetkasem Highway (Km 20)",
    "G007_010": "Nong Khaem / Petchkasem 81",
    "G023_036": "Khlong Sam Wa / Hathairat Rd",
    "G020_037": "Prawet / Srinakarin Rd",
    "G014_048": "Lat Krabang Airport Zone",
    "G006_009": "Nong Khaem / Kanchanaphisek (South)",
    "G020_009": "Taling Chan / Ratchaphruek Intersection",
}

# Canonical primary-cause table — single source of truth for the 5 cause_*
# columns in ml_features.csv. All builders derive their lookups from this;
# never re-list the column names or keys per function (they drift silently).
PRIMARY_CAUSES = {
    "speeding": {
        "thai": "ขับรถเร็วเกินอัตรากำหนด",
        "column": "cause_ขับรถเร็วเกินอัตรากำหนด",
        "english": "Speeding",
    },
    "cutting_in_front": {
        "thai": "คน/รถ/สัตว์ตัดหน้ากระชั้นชิด",
        "column": "cause_คน/รถ/สัตว์ตัดหน้ากระชั้นชิด",
        "english": "Cutting in front",
    },
    "red_light": {
        "thai": "ฝ่าฝืนสัญญาณไฟ/เครื่องหมายจราจร",
        "column": "cause_ฝ่าฝืนสัญญาณไฟ/เครื่องหมายจราจร",
        "english": "Running red light / traffic sign violation",
    },
    "tire_blowout": {
        "thai": "ยางเสื่อมสภาพ/ยางแตก",
        "column": "cause_ยางเสื่อมสภาพ/ยางแตก",
        "english": "Tire blowout / worn tires",
    },
    "drowsy_driving": {
        "thai": "หลับใน",
        "column": "cause_หลับใน",
        "english": "Drowsy driving",
    },
}

# English cause name mapping
CAUSE_ENGLISH = {
    "ขับรถเร็วเกินอัตรากำหนด": "Speeding",
    "คน/รถ/สัตว์ตัดหน้ากระชั้นชิด": "Cutting in front",
    "ยางเสื่อมสภาพ/ยางแตก": "Tire blowout",
    "หลับใน": "Drowsy driving",
    "ฝ่าฝืนสัญญาณไฟ/เครื่องหมายจราจร": "Running red light",
    "ถนนลื่น": "Slippery road",
    "ไม่ยอมให้รถที่มีสิทธิ์ไปก่อน": "Failing to yield",
    "ไม่คุ้นเคยเส้นทาง/ขับรถไม่ชำนาญ": "Unfamiliar route",
    "อุปกรณ์ยานพาหนะบกพร่อง": "Vehicle malfunction",
    "ระบบห้ามล้อขัดข้อง/ระบบเบรกชำรุด": "Brake failure",
    "เมาสุรา": "Drunk driving",
    "ใช้โทรศัพท์เคลื่อนที่ขณะขับรถ": "Phone use while driving",
    "ขับรถย้อนศร": "Wrong-way driving",
    "เปลี่ยนช่องทางกะทันหัน": "Sudden lane change",
    "ขับรถตามกระชั้นชิด": "Tailgating",
    "แซงรถอย่างผิดกฎหมาย": "Illegal overtaking",
}

MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def load_csv(path):
    """Load CSV file into list of dicts."""
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_json(path):
    """Load JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def safe_int(val, default=0):
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def get_location_name(grid_id):
    """Get human-readable location name for a grid cell."""
    return GRID_LOCATION_NAMES.get(grid_id, f"Zone {grid_id}")


def build_stats(accidents):
    """Build overall project statistics."""
    total = len(accidents)
    deaths = sum(safe_int(a.get("ผู้เสียชีวิต", 0)) for a in accidents)
    serious = sum(safe_int(a.get("ผู้บาดเจ็บสาหัส", 0)) for a in accidents)
    minor = sum(safe_int(a.get("ผู้บาดเจ็บเล็กน้อย", 0)) for a in accidents)

    return {
        "total_accidents": total,
        "total_zones": 2418,
        "high_risk_zones": 126,
        "date_range": "January - December 2024",
        "data_source": "Thai Ministry of Transport",
        "nlp_processed": True,
        "deaths": deaths,
        "serious_injuries": serious,
        "minor_injuries": minor,
        "total_injuries": serious + minor,
        # Legacy keys describe the count-based spatial holdout, not XGBoost accuracy.
        "model_accuracy": 83.8,
        "model_lift": 1.95,
        "danger_zones_detected": "Not verified",
    }


def build_causes(nlp_cache, ml_features, grid_stats):
    """Build cause breakdown with top zones per cause."""
    cause_analysis = nlp_cache.get("cause_analysis", {})
    categories = cause_analysis.get("categories", [])

    # Build grid_id -> risk_score lookup
    risk_lookup = {}
    for row in grid_stats:
        gid = row["grid_id"]
        risk_lookup[gid] = safe_float(row.get("risk_score", 0))

    # Cause column mapping (Thai cause name -> ml_features column)
    cause_col_map = {m["thai"]: m["column"] for m in PRIMARY_CAUSES.values()}

    causes = {}
    for cat in categories:
        thai = cat["thai_name"]
        english = cat["english_name"]
        key = english.lower().replace(" ", "_").replace("/", "_")

        # Find top zones for this cause
        col = cause_col_map.get(thai)
        top_zones = []
        if col:
            zone_scores = []
            for row in ml_features:
                gid = row["grid_id"]
                cause_pct = safe_float(row.get(col, 0))
                total_acc = safe_int(row.get("total_accidents", 0))
                if cause_pct > 0 and total_acc > 0:
                    cause_count = int(round(cause_pct * total_acc / 100))
                    zone_scores.append({
                        "grid_id": gid,
                        "location": get_location_name(gid),
                        "accidents": total_acc,
                        "cause_count": cause_count,
                        "cause_pct": round(cause_pct, 1),
                        "risk_score": round(risk_lookup.get(gid, 0), 1),
                    })
            zone_scores.sort(key=lambda x: x["cause_count"], reverse=True)
            top_zones = zone_scores[:10]

        causes[key] = {
            "thai_name": thai,
            "english_name": english,
            "count": cat["count"],
            "percentage": cat["percentage"],
            "feature_importance": cat.get("feature_importance", 0),
            "top_zones": top_zones,
        }

    return causes


def build_time_patterns(accidents):
    """Build temporal analysis from raw accident data."""
    hour_counts = defaultdict(int)
    day_counts = defaultdict(int)
    month_counts = defaultdict(int)
    weekend_count = 0
    weekday_count = 0
    rush_hour_count = 0

    for a in accidents:
        hour = safe_int(a.get("hour", -1))
        if 0 <= hour <= 23:
            hour_counts[hour] += 1

        day = a.get("day_name", "")
        if day:
            day_counts[day] += 1

        month = safe_int(a.get("month", 0))
        if 1 <= month <= 12:
            month_counts[month] += 1

        if a.get("is_weekend", "").lower() == "true":
            weekend_count += 1
        else:
            weekday_count += 1

        if a.get("is_rush_hour", "").lower() == "true":
            rush_hour_count += 1

    total = len(accidents)

    # Find peaks
    peak_hour = max(hour_counts, key=lambda h: hour_counts[h]) if hour_counts else 10
    peak_day = max(day_counts, key=lambda d: day_counts[d]) if day_counts else "Saturday"
    peak_month = max(month_counts, key=lambda m: month_counts[m]) if month_counts else 3
    safest_hour = min(hour_counts, key=lambda h: hour_counts[h]) if hour_counts else 3

    # Build hourly breakdown
    hourly = {}
    for h in range(24):
        count = hour_counts.get(h, 0)
        hourly[f"{h:02d}:00"] = {
            "count": count,
            "percentage": round(count / total * 100, 1) if total else 0,
        }

    # Build daily breakdown
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily = {}
    for d in day_order:
        count = day_counts.get(d, 0)
        daily[d] = {
            "count": count,
            "percentage": round(count / total * 100, 1) if total else 0,
        }

    # Build monthly breakdown
    monthly = {}
    for m in range(1, 13):
        count = month_counts.get(m, 0)
        monthly[MONTH_NAMES[m]] = {
            "count": count,
            "percentage": round(count / total * 100, 1) if total else 0,
        }

    return {
        "most_dangerous_hour": f"{peak_hour}:00-{peak_hour + 1}:00",
        "most_dangerous_hour_count": hour_counts.get(peak_hour, 0),
        "safest_hour": f"{safest_hour}:00-{safest_hour + 1}:00",
        "safest_hour_count": hour_counts.get(safest_hour, 0),
        "most_dangerous_day": peak_day,
        "most_dangerous_day_count": day_counts.get(peak_day, 0),
        "most_dangerous_month": MONTH_NAMES[peak_month],
        "most_dangerous_month_count": month_counts.get(peak_month, 0),
        "rush_hour_accidents": rush_hour_count,
        "rush_hour_pct": round(rush_hour_count / total * 100, 1) if total else 0,
        "weekend_accidents": weekend_count,
        "weekend_pct": round(weekend_count / total * 100, 1) if total else 0,
        "weekday_accidents": weekday_count,
        "weekend_vs_weekday": f"Weekend {round(weekend_count / total * 100, 1)}% vs Weekday {round(weekday_count / total * 100, 1)}%" if total else "",
        "hourly": hourly,
        "daily": daily,
        "monthly": monthly,
    }


def build_zones(ml_features, grid_stats):
    """Build top 30 zone profiles with cause breakdowns."""
    # Risk score lookup
    risk_lookup = {}
    risk_level_lookup = {}
    for row in grid_stats:
        gid = row["grid_id"]
        risk_lookup[gid] = safe_float(row.get("risk_score", 0))
        risk_level_lookup[gid] = row.get("risk_level", "low")

    # Sort by total_accidents descending
    sorted_features = sorted(
        ml_features,
        key=lambda r: safe_int(r.get("total_accidents", 0)),
        reverse=True,
    )

    cause_cols = {key: m["column"] for key, m in PRIMARY_CAUSES.items()}
    cause_cols["other"] = "cause_other"

    zones = {}
    for row in sorted_features[:30]:
        gid = row["grid_id"]
        total = safe_int(row.get("total_accidents", 0))

        # Build cause breakdown (percentages)
        cause_breakdown = {}
        dominant_cause = "speeding"
        dominant_pct = 0
        for eng_key, col_name in cause_cols.items():
            pct = safe_float(row.get(col_name, 0))
            cause_breakdown[eng_key] = round(pct, 1)
            if pct > dominant_pct:
                dominant_pct = pct
                dominant_cause = eng_key

        zones[gid] = {
            "location": get_location_name(gid),
            "center_lat": safe_float(row.get("center_lat")),
            "center_lon": safe_float(row.get("center_lon")),
            "risk_score": round(risk_lookup.get(gid, 0), 1),
            "risk_level": risk_level_lookup.get(gid, "low"),
            "total_accidents": total,
            "dominant_cause": dominant_cause,
            "dominant_cause_pct": round(dominant_pct, 1),
            "cause_breakdown": cause_breakdown,
            "severity_mean": safe_float(row.get("severity_mean")),
            "has_fatality": row.get("has_fatality", "0") in ("1", "1.0", "True"),
            "motorcycle_pct": round(safe_float(row.get("motorcycle_pct")), 1),
            "rush_hour_pct": round(safe_float(row.get("rush_hour_pct")), 1),
            "weekend_pct": round(safe_float(row.get("weekend_pct")), 1),
        }

    return zones


def build_recommendations(nlp_cache, ml_features):
    """Build prioritized policy recommendations from NLP zone analysis."""
    # Count zones where each cause is dominant (>30% of accidents) from ml_features.
    # Keyed by PRIMARY_CAUSES short key so the lookups below can't drift.
    cause_zone_counts = defaultdict(int)
    for row in ml_features:
        total = safe_int(row.get("total_accidents", 0))
        if total < 5:
            continue
        for key, m in PRIMARY_CAUSES.items():
            pct = safe_float(row.get(m["column"], 0))
            if pct > 30:
                cause_zone_counts[key] += 1

    recommendations = [
        {
            "priority": 1,
            "title": "Speed Camera Deployment",
            "description": "Deploy automated speed enforcement in high-speeding zones. 82% of top danger zones have speeding as the dominant cause.",
            "target_cause": "Speeding",
            "zones_affected": cause_zone_counts.get("speeding", 0),
            "expected_impact": "Targets Bangkok's dominant crash cause across the top danger zones",
            "cost_estimate": "฿2.5M per camera installation",
            "recommendation_type": "enforcement",
        },
        {
            "priority": 2,
            "title": "Intersection Safety Redesign",
            "description": "Redesign dangerous intersections with better signage, barriers, and traffic signals to reduce cutting-in-front accidents.",
            "target_cause": "Cutting in front",
            "zones_affected": cause_zone_counts.get("cutting_in_front", 0),
            "expected_impact": "Targets cutting-in-front hotspots at busy intersections",
            "cost_estimate": "฿5-15M per intersection",
            "recommendation_type": "infrastructure",
        },
        {
            "priority": 3,
            "title": "Rumble Strips & Road Surface Treatment",
            "description": "Install rumble strips on highway stretches with high rear-end collision rates. Target drowsy driving zones.",
            "target_cause": "Drowsy driving",
            "zones_affected": cause_zone_counts.get("drowsy_driving", 0),
            "expected_impact": "Targets highway corridors with elevated drowsy-driving risk",
            "cost_estimate": "฿500K per km of highway",
            "recommendation_type": "infrastructure",
        },
        {
            "priority": 4,
            "title": "Red Light Camera Network",
            "description": "Install red-light cameras at intersections with high traffic signal violations.",
            "target_cause": "Running red light",
            "zones_affected": cause_zone_counts.get("red_light", 0),
            "expected_impact": "Targets repeated signal-violation intersections",
            "cost_estimate": "฿1.5M per intersection",
            "recommendation_type": "enforcement",
        },
        {
            "priority": 5,
            "title": "Vehicle Inspection Checkpoints",
            "description": "Set up mobile inspection points in zones with high tire blowout and mechanical failure rates.",
            "target_cause": "Tire blowout / mechanical",
            "zones_affected": cause_zone_counts.get("tire_blowout", 0),
            "expected_impact": "Targets mechanical-failure and tire-related trouble spots",
            "cost_estimate": "฿200K per checkpoint per month",
            "recommendation_type": "enforcement",
        },
    ]

    return recommendations


def build_thai_keywords(nlp_cache):
    """Build Thai keyword mapping from word cloud data."""
    word_cloud = nlp_cache.get("word_cloud", {}).get("words", [])

    # Group keywords by cause category
    keyword_groups = {
        "speeding": {
            "thai_terms": ["ขับรถเร็วเกินอัตรากำหนด"],
            "related_words": [],
        },
        "cutting_in_front": {
            "thai_terms": ["คน/รถ/สัตว์ตัดหน้ากระชั้นชิด", "เปลี่ยนช่องทางกะทันหัน", "ขับรถตามกระชั้นชิด"],
            "related_words": [],
        },
        "drowsy_driving": {
            "thai_terms": ["หลับใน"],
            "related_words": [],
        },
        "drunk_driving": {
            "thai_terms": ["เมาสุรา"],
            "related_words": [],
        },
        "red_light": {
            "thai_terms": ["ฝ่าฝืนสัญญาณไฟ/เครื่องหมายจราจร"],
            "related_words": [],
        },
        "mechanical": {
            "thai_terms": ["ยางเสื่อมสภาพ/ยางแตก", "อุปกรณ์ยานพาหนะบกพร่อง", "ระบบห้ามล้อขัดข้อง/ระบบเบรกชำรุด"],
            "related_words": [],
        },
        "road_conditions": {
            "thai_terms": ["ถนนลื่น", "ถนนชำรุด", "มีกองวัสดุ/สิ่งกีดขวาง", "แสงสว่างไม่เพียงพอ"],
            "related_words": [],
        },
        "reckless_driving": {
            "thai_terms": ["ขับรถย้อนศร", "แซงรถอย่างผิดกฎหมาย", "ความประมาท/เร่งรีบ (วิ่งตัดหน้า)", "สูญเสียการควบคุม"],
            "related_words": [],
        },
        "collision_types": {
            "thai_terms": ["ชนท้าย", "ชนด้านข้าง", "พลิกคว่ำ/ตกถนนในทางตรง", "ชนสิ่งกีดขวาง (บนผิวจราจร)", "ชนคนเดินเท้า"],
            "related_words": [],
        },
    }

    # Add frequency data from word cloud
    word_freq = {w["text"]: w["frequency"] for w in word_cloud}
    for group_key, group in keyword_groups.items():
        for term in group["thai_terms"]:
            freq = word_freq.get(term, 0)
            if freq > 0:
                group["related_words"].append({
                    "thai": term,
                    "english": CAUSE_ENGLISH.get(term, term),
                    "frequency": freq,
                })

    return keyword_groups


def build_cameras(camera_locations):
    """Build camera info for chatbot context."""
    cameras = []
    for cam in camera_locations:
        cameras.append({
            "camera_id": cam["camera_id"],
            "location": cam["location_name"],
            "lat": cam["latitude"],
            "lon": cam["longitude"],
            "image": cam.get("image_file", ""),
        })
    return cameras


def build_feature_importance(nlp_cache):
    """Extract model feature importance summary."""
    fi = nlp_cache.get("feature_importance", {})
    return {
        "severity": fi.get("severity", {}).get("importance_pct", 53.7),
        "spatial": fi.get("spatial", {}).get("importance_pct", 20.66),
        "temporal": fi.get("temporal", {}).get("importance_pct", 11.83),
        "vehicle": fi.get("vehicle", {}).get("importance_pct", 9.99),
        "nlp_text": fi.get("nlp_text", {}).get("importance_pct", 3.82),
        "top_feature": "severity_mean (53.6%)",
        "key_insight": "Severity of past accidents is the strongest predictor of future danger, not just accident count.",
    }


def main():
    print("Loading source data...")

    # Load all data sources
    accidents = load_csv(DATA_DIR / "bangkok_accidents_clean.csv")
    ml_features = load_csv(DATA_DIR / "ml_features.csv")
    grid_stats = load_csv(DATA_DIR / "bangkok_grid_stats.csv")
    nlp_cache = load_json(STATIC_DIR / "nlp_cache.json")
    camera_locations = load_json(STATIC_DIR / "camera_locations.json")

    print(f"  Accidents:   {len(accidents)} records")
    print(f"  ML Features: {len(ml_features)} zones")
    print(f"  Grid Stats:  {len(grid_stats)} zones")
    print(f"  Cameras:     {len(camera_locations)} locations")

    print("\nBuilding chatbot data...")

    chatbot_data = {
        "metadata": {
            "generated_by": "chatbot_data_processor.py",
            "project": "SafeDrive BKK",
            "description": "Aggregated data for the Bangkok Road Safety chatbot",
        },
        "stats": build_stats(accidents),
        "causes": build_causes(nlp_cache, ml_features, grid_stats),
        "time_patterns": build_time_patterns(accidents),
        "zones": build_zones(ml_features, grid_stats),
        "recommendations": build_recommendations(nlp_cache, ml_features),
        "thai_keywords": build_thai_keywords(nlp_cache),
        "cameras": build_cameras(camera_locations),
        "model_importance": build_feature_importance(nlp_cache),
    }

    # Write output
    output_path = STATIC_DIR / "chatbot_data.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chatbot_data, f, ensure_ascii=False, indent=2)

    file_size = os.path.getsize(output_path)
    print(f"\nOutput: {output_path}")
    print(f"Size:   {file_size / 1024:.1f} KB")

    # Summary
    data = chatbot_data
    print(f"\n--- Summary ---")
    print(f"Stats:           {len(data['stats'])} fields")
    print(f"Causes:          {len(data['causes'])} categories")
    print(f"Time patterns:   {len(data['time_patterns'])} fields")
    print(f"Zones:           {len(data['zones'])} zone profiles")
    print(f"Recommendations: {len(data['recommendations'])} policies")
    print(f"Thai keywords:   {len(data['thai_keywords'])} groups")
    print(f"Cameras:         {len(data['cameras'])} locations")
    print(f"Done!")


if __name__ == "__main__":
    main()
