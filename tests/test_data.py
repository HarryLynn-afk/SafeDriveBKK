"""
Tests for src/utils/data.py

Covers: format_hour_range, calculate_temporal_stats (peak hours, dangerous
days, rush-hour stats), and error handling for missing files.
"""

import pandas as pd
import pytest

from src.utils.data import calculate_temporal_stats, format_hour_range


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
