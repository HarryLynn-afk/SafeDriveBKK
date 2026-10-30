"""
Tests for src/utils/geo.py

Covers: haversine accuracy, grid ID generation, bounds clipping,
district lookup, area assignment, grid bounds derivation, and point lookup.
"""

import math

import pandas as pd
import pytest

from src.utils.geo import (
    add_grid_bounds,
    assign_bangkok_area,
    clip_to_bangkok_bounds,
    find_grid_for_point,
    haversine_km,
    latlon_to_grid_id,
    nearest_district,
)
from src.config import BKK_LAT_MIN, BKK_LAT_MAX, BKK_LON_MIN, BKK_LON_MAX


class TestHaversineKm:
    def test_same_point_returns_zero(self):
        assert haversine_km(13.75, 100.5, 13.75, 100.5) == 0.0

    def test_known_distance_approximately_correct(self):
        # Chatuchak (13.8038, 100.5503) to Sathon (13.7195, 100.5267) ≈ 9.5 km
        dist = haversine_km(13.8038, 100.5503, 13.7195, 100.5267)
        assert 9.0 < dist < 10.5

    def test_is_symmetric(self):
        d1 = haversine_km(13.75, 100.5, 13.80, 100.55)
        d2 = haversine_km(13.80, 100.55, 13.75, 100.5)
        assert abs(d1 - d2) < 1e-9

    def test_is_always_non_negative(self):
        assert haversine_km(0.0, 0.0, -1.0, -1.0) >= 0


class TestClipToBangkokBounds:
    def test_inside_bounds_unchanged(self):
        lat, lon = clip_to_bangkok_bounds(13.75, 100.5)
        assert lat == 13.75
        assert lon == 100.5

    def test_clamps_lat_below_min(self):
        lat, lon = clip_to_bangkok_bounds(10.0, 100.5)
        assert lat == BKK_LAT_MIN

    def test_clamps_lon_above_max(self):
        lat, lon = clip_to_bangkok_bounds(13.75, 200.0)
        assert lon == BKK_LON_MAX


class TestLatlonToGridId:
    def test_saved_grid_centers_map_to_their_own_cells(self):
        from src.config import GRID_STATS_PATH
        grid = pd.read_csv(GRID_STATS_PATH)
        mismatches = [row.grid_id for row in grid.itertuples()
                      if latlon_to_grid_id(row.center_lat, row.center_lon) != row.grid_id]
        assert mismatches == []

    @pytest.mark.parametrize("lat,lon,expected", [
        (13.95, 100.35, "G038_000"),
        (13.6, 100.9, "G000_061"),
        (13.95, 100.9, "G038_061"),
        (13.6, 100.35, "G000_000"),
        (13.950001, 100.9, None),
        (13.95, 100.900001, None),
    ])
    def test_outer_boundaries(self, lat, lon, expected):
        assert latlon_to_grid_id(lat, lon) == expected

    def test_outside_bangkok_returns_none(self):
        assert latlon_to_grid_id(0.0, 0.0) is None

    def test_nan_returns_none(self):
        assert latlon_to_grid_id(float("nan"), 100.5) is None

    def test_inside_bangkok_returns_valid_id(self):
        grid_id = latlon_to_grid_id(13.75, 100.5)
        assert grid_id is not None
        assert grid_id.startswith("G")
        assert "_" in grid_id

    def test_format_is_g_ddd_ddd(self):
        import re
        grid_id = latlon_to_grid_id(13.75, 100.5)
        assert re.match(r"^G\d{3}_\d{3}$", grid_id)


class TestNearestDistrict:
    def test_chatuchak_area(self):
        # Centroid of Chatuchak is (13.8038, 100.5503)
        district = nearest_district(13.8038, 100.5503)
        assert district == "Chatuchak"

    def test_returns_string(self):
        result = nearest_district(13.75, 100.5)
        assert isinstance(result, str)
        assert len(result) > 0


class TestAssignBangkokArea:
    def test_nan_returns_none(self):
        assert assign_bangkok_area(float("nan"), 100.5) is None

    def test_northern_area(self):
        label = assign_bangkok_area(13.90, 100.50)
        assert "Don Mueang" in label or "Lak Si" in label

    def test_central_area(self):
        label = assign_bangkok_area(13.74, 100.52)
        assert label is not None


class TestFindGridForPoint:
    def test_hit_returns_row(self, sample_grid_df):
        row = find_grid_for_point(13.695, 100.444, sample_grid_df)
        assert row is not None
        assert row["grid_id"] == "G010_010"

    def test_miss_returns_none(self, sample_grid_df):
        result = find_grid_for_point(0.0, 0.0, sample_grid_df)
        assert result is None


class TestAddGridBounds:
    def test_skips_derivation_when_bounds_present(self, sample_grid_df):
        result = add_grid_bounds(sample_grid_df)
        # Original bounds should be unchanged
        assert result["lat_min"].iloc[0] == sample_grid_df["lat_min"].iloc[0]

    def test_derives_bounds_from_grid_id(self):
        df = pd.DataFrame({"grid_id": ["G010_010"], "total_accidents": [3]})
        result = add_grid_bounds(df)
        assert "lat_min" in result.columns
        assert "lon_min" in result.columns

    def test_invalid_grid_id_raises(self):
        df = pd.DataFrame({"grid_id": ["INVALID"], "total_accidents": [0]})
        with pytest.raises(ValueError, match="Invalid grid_id"):
            add_grid_bounds(df)
