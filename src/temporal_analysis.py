"""
Temporal pattern analysis for Bangkok road accidents.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LOGGER = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/processed")
INPUT_PATH = OUTPUT_DIR / "bangkok_accidents_clean.csv"

HOUR_CHART_PATH = OUTPUT_DIR / "accidents_by_hour.png"
DAY_CHART_PATH = OUTPUT_DIR / "accidents_by_day.png"
SUMMARY_PATH = OUTPUT_DIR / "temporal_analysis.txt"

DAY_NAME_MAP = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}


def _validate_columns(df, required_columns):
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")


def _save_figure(fig, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def analyze_by_hour(df):
    """
    Analyze accidents grouped by hour of day.

    Returns:
        dict: {"peak_hours": [...], "counts": [...]}
    """
    _validate_columns(df, ["hour"])

    hour_series = pd.to_numeric(df["hour"], errors="coerce")
    hour_counts = (
        hour_series.dropna()
        .astype(int)
        .value_counts()
        .reindex(range(24), fill_value=0)
        .sort_index()
    )

    peak = hour_counts.sort_values(ascending=False).head(3)
    peak_hours = [int(hour) for hour in peak.index]
    peak_counts = [int(value) for value in peak.values]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(hour_counts.index, hour_counts.values, color="#3b82f6", edgecolor="#1d4ed8")
    ax.set_title("Bangkok Accidents by Hour of Day", fontsize=18, pad=12)
    ax.set_xlabel("Hour of Day", fontsize=14)
    ax.set_ylabel("Number of Accidents", fontsize=14)
    ax.set_xticks(range(24))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, HOUR_CHART_PATH)

    return {"peak_hours": peak_hours, "counts": peak_counts}


def analyze_by_day_of_week(df):
    """
    Analyze accidents grouped by day of week.

    Returns:
        dict: {"dangerous_days": [...], "counts": [...]}
    """
    _validate_columns(df, ["day_of_week"])

    day_series = pd.to_numeric(df["day_of_week"], errors="coerce")
    day_counts = (
        day_series.dropna()
        .astype(int)
        .value_counts()
        .reindex(range(7), fill_value=0)
        .sort_index()
    )
    day_names = [DAY_NAME_MAP[i] for i in range(7)]

    top_days = day_counts.sort_values(ascending=False).head(3)
    dangerous_days = [DAY_NAME_MAP[int(day)] for day in top_days.index]
    dangerous_counts = [int(value) for value in top_days.values]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(day_names, day_counts.values, color="#f59e0b", edgecolor="#b45309")
    ax.set_title("Bangkok Accidents by Day of Week", fontsize=18, pad=12)
    ax.set_xlabel("Day of Week", fontsize=14)
    ax.set_ylabel("Number of Accidents", fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, DAY_CHART_PATH)

    return {"dangerous_days": dangerous_days, "counts": dangerous_counts}


def analyze_rush_hour_impact(df):
    """
    Analyze accident distribution between rush and non-rush hours.

    Returns:
        dict: {rush_hour_count, non_rush_count, rush_hour_percentage}
    """
    _validate_columns(df, ["is_rush_hour"])

    rush_col = df["is_rush_hour"]
    if rush_col.dtype == bool:
        rush_mask = rush_col.fillna(False)
    else:
        rush_mask = rush_col.astype(str).str.lower().isin({"true", "1", "yes", "y", "t"})

    rush_hour_count = int(rush_mask.sum())
    non_rush_count = int(len(df) - rush_hour_count)
    rush_hour_percentage = (rush_hour_count / len(df) * 100) if len(df) else 0.0

    return {
        "rush_hour_count": rush_hour_count,
        "non_rush_count": non_rush_count,
        "rush_hour_percentage": round(rush_hour_percentage, 2),
    }


def _format_hour_range(hour_value):
    start_hour = int(hour_value)
    end_hour = (start_hour + 1) % 24
    return f"{start_hour:02d}:00-{end_hour:02d}:00"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        if not INPUT_PATH.exists():
            raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

        df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
        LOGGER.info("Loaded %d Bangkok accidents from %s", len(df), INPUT_PATH)

        hour_result = analyze_by_hour(df)
        day_result = analyze_by_day_of_week(df)
        rush_result = analyze_rush_hour_impact(df)

        peak_lines = [
            f"- {_format_hour_range(hour)}: {count} accidents"
            for hour, count in zip(hour_result["peak_hours"], hour_result["counts"])
        ]
        day_lines = [
            f"- {day}: {count} accidents"
            for day, count in zip(day_result["dangerous_days"], day_result["counts"])
        ]

        summary_lines = [
            "TEMPORAL ANALYSIS SUMMARY - BANGKOK",
            "=" * 50,
            f"Total accidents analyzed: {len(df)}",
            "",
            "Peak danger hours:",
            *peak_lines,
            "",
            "Most dangerous days of week:",
            *day_lines,
            "",
            "Rush hour statistics:",
            f"- Rush hour accidents: {rush_result['rush_hour_count']}",
            f"- Non-rush accidents: {rush_result['non_rush_count']}",
            f"- Rush hour percentage: {rush_result['rush_hour_percentage']:.2f}%",
            "",
            f"Saved chart: {HOUR_CHART_PATH}",
            f"Saved chart: {DAY_CHART_PATH}",
        ]

        summary_text = "\n".join(summary_lines)
        print(summary_text)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.write_text(summary_text, encoding="utf-8")
        LOGGER.info("Saved temporal summary to %s", SUMMARY_PATH)

    except Exception as exc:
        LOGGER.exception("Temporal analysis failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
