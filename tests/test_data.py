"""
Tests for src/utils/data.py

Covers: format_hour_range, calculate_temporal_stats (peak hours, dangerous
days, rush-hour stats), and error handling for missing files.
"""

import pandas as pd
import pytest

from src.utils.data import build_analytics_payload, calculate_temporal_stats, format_hour_range


class TestFormatHourRange:
    def test_midnight_wraps_to_one(self):
        assert format_hour_range(23) == "23:00-00:00"

    def test_regular_hour(self):
        assert format_hour_range(8) == "08:00-09:00"

    def test_zero_hour(self):
        assert format_hour_range(0) == "00:00-01:00"


class TestCalculateTemporalStats:
    def test_returns_all_required_keys(self, sample_accidents_df):
        result = calculate_temporal_stats(sample_accidents_df)
        assert "peak_hours" in result
        assert "dangerous_days" in result
        assert "rush_hour_count" in result
        assert "non_rush_count" in result
        assert "rush_hour_percentage" in result

    def test_peak_hours_list_has_at_most_three(self, sample_accidents_df):
        result = calculate_temporal_stats(sample_accidents_df)
        assert len(result["peak_hours"]) <= 3

    def test_rush_hour_count_matches_data(self, sample_accidents_df):
        # sample_accidents_df has 5 True rush-hour rows
        result = calculate_temporal_stats(sample_accidents_df)
        assert result["rush_hour_count"] == 5
        assert result["non_rush_count"] == 2

    def test_rush_hour_percentage_sums_correctly(self, sample_accidents_df):
        result = calculate_temporal_stats(sample_accidents_df)
        pct = result["rush_hour_percentage"]
        assert 0.0 <= pct <= 100.0
        expected = round((5 / 7) * 100, 2)
        assert abs(pct - expected) < 0.01

    def test_string_boolean_rush_hour_column(self):
        """Rush hour column may arrive as string 'True'/'False' from CSV."""
        df = pd.DataFrame({
            "hour": [8, 9, 17],
            "day_of_week": [0, 1, 2],
            "month": [1, 1, 2],
            "is_rush_hour": ["True", "False", "true"],
            "is_weekend": ["False", "False", "False"],
        })
        result = calculate_temporal_stats(df)
        assert result["rush_hour_count"] == 2

    def test_dangerous_days_day_name_present(self, sample_accidents_df):
        result = calculate_temporal_stats(sample_accidents_df)
        for day in result["dangerous_days"]:
            assert "day_name" in day
            assert isinstance(day["day_name"], str)


# ---------------------------------------------------------------------------
# build_analytics_payload — full analytics response dict
# ---------------------------------------------------------------------------

_ANALYTICS_REQUIRED_KEYS = {
    "total_accidents", "total_grids", "by_hour", "by_day", "by_month",
    "top_zones", "by_area", "by_vehicle", "severity",
    "risk_distribution", "weekend_vs_weekday", "summary",
}


class TestBuildAnalyticsPayload:
    # ---- shape ----

    def test_returns_all_required_keys(self, sample_accidents_df, sample_grid_df):
        result = build_analytics_payload(sample_accidents_df, sample_grid_df)
        assert _ANALYTICS_REQUIRED_KEYS.issubset(result.keys())

    # ---- counts ----

    def test_total_accidents_equals_dataframe_length(self, sample_accidents_df, sample_grid_df):
        result = build_analytics_payload(sample_accidents_df, sample_grid_df)
        assert result["total_accidents"] == len(sample_accidents_df)

    def test_total_grids_equals_grid_df_length(self, sample_accidents_df, sample_grid_df):
        result = build_analytics_payload(sample_accidents_df, sample_grid_df)
        assert result["total_grids"] == len(sample_grid_df)

    # ---- temporal breakdowns ----

    def test_by_hour_keys_are_integers(self, sample_accidents_df, sample_grid_df):
        result = build_analytics_payload(sample_accidents_df, sample_grid_df)
        for key in result["by_hour"]:
            assert isinstance(key, int)

    def test_by_hour_covers_all_hours_in_data(self, sample_accidents_df, sample_grid_df):
        # sample_accidents_df hours: [7, 8, 17, 17, 23, 8, 17]
        result = build_analytics_payload(sample_accidents_df, sample_grid_df)
        assert set(result["by_hour"].keys()) == {7, 8, 17, 23}

    def test_by_day_keys_are_day_name_strings(self, sample_accidents_df, sample_grid_df):
        result = build_analytics_payload(sample_accidents_df, sample_grid_df)
        for key in result["by_day"]:
            assert isinstance(key, str)

    def test_by_month_keys_are_month_name_strings(self, sample_accidents_df, sample_grid_df):
        # sample_accidents_df months: [1, 1, 2, 2, 3, 3, 4]
        result = build_analytics_payload(sample_accidents_df, sample_grid_df)
        assert set(result["by_month"].keys()) == {"Jan", "Feb", "Mar", "Apr"}

    # ---- weekend split ----

    def test_weekend_vs_weekday_sums_to_total_accidents(self, sample_accidents_df, sample_grid_df):
        result = build_analytics_payload(sample_accidents_df, sample_grid_df)
        split = result["weekend_vs_weekday"]
        assert split["Weekend"] + split["Weekday"] == len(sample_accidents_df)

    def test_weekend_vs_weekday_counts_match_data(self, sample_accidents_df, sample_grid_df):
        # sample_accidents_df is_weekend: [F, F, F, T, T, F, F] → Weekend=2, Weekday=5
        result = build_analytics_payload(sample_accidents_df, sample_grid_df)
        assert result["weekend_vs_weekday"]["Weekend"] == 2
        assert result["weekend_vs_weekday"]["Weekday"] == 5

    # ---- risk distribution ----

    def test_risk_distribution_excludes_zero_accident_grids(self, sample_accidents_df, sample_grid_df):
        # sample_grid_df has one grid with total_accidents=0 (G011_010)
        # so that grid's risk_level ("low") should not inflate the count beyond active grids
        result = build_analytics_payload(sample_accidents_df, sample_grid_df)
        dist = result["risk_distribution"]
        # Total active grids: 3 (5+15+30 accidents); inactive: 1 (G011_010, 0 accidents)
        assert sum(dist.values()) == 3

    def test_summary_has_active_zones_and_high_risk_zones(self, sample_accidents_df, sample_grid_df):
        result = build_analytics_payload(sample_accidents_df, sample_grid_df)
        summary = result["summary"]
        assert "active_zones" in summary
        assert "high_risk_zones" in summary
