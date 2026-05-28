"""
Boundary tests for the grid_aggregation data pipeline.

These three functions produce the spatial grid + per-cell stats that the ML
model trains on, so their correctness matters. They are pure DataFrame
transforms — no file I/O (that lives in main()) — so no mocking is needed.
"""

import numpy as np
import pandas as pd
import pytest

from src.grid_aggregation import (
    create_grid_cells,
    assign_to_grid,
    calculate_grid_statistics,
)


class TestCreateGridCells:
    def test_invalid_bounds_raise(self):
        with pytest.raises(ValueError):
            create_grid_cells(13.95, 13.60, 100.35, 100.90)  # lat_min >= lat_max
        with pytest.raises(ValueError):
            create_grid_cells(13.60, 13.95, 100.90, 100.35)  # lon_min >= lon_max

    def test_nonpositive_cell_size_raises(self):
        with pytest.raises(ValueError):
            create_grid_cells(13.60, 13.95, 100.35, 100.90, cell_size_km=0)

    def test_has_expected_columns(self):
        grid = create_grid_cells(0.0, 0.027, 0.0, 0.018)
        assert set(grid.columns) == {
            "grid_id", "center_lat", "center_lon",
            "lat_min", "lat_max", "lon_min", "lon_max",
        }

    def test_cell_count_matches_bins(self):
        # 0.027 lat / 0.018 lon at 0.009 step → 3 x 2 = 6 cells.
        grid = create_grid_cells(0.0, 0.027, 0.0, 0.018)
        assert len(grid) == 6
        assert grid["grid_id"].is_unique
        assert "G000_000" in set(grid["grid_id"])

    def test_centers_within_bounds_and_are_midpoints(self):
        grid = create_grid_cells(0.0, 0.027, 0.0, 0.018)
        assert (grid["lat_min"] < grid["lat_max"]).all()
        assert (grid["lon_min"] < grid["lon_max"]).all()
        assert np.allclose(grid["center_lat"], (grid["lat_min"] + grid["lat_max"]) / 2)
        assert np.allclose(grid["center_lon"], (grid["lon_min"] + grid["lon_max"]) / 2)

    def test_no_zero_area_cells(self):
        # Regression: float drift in edge generation used to emit zero-width
        # boundary cells. Every cell must have positive area.
        grid = create_grid_cells(0.0, 0.027, 0.0, 0.018)
        assert (grid["lat_max"] > grid["lat_min"]).all()
        assert (grid["lon_max"] > grid["lon_min"]).all()


class TestAssignToGrid:
    def _small_grid(self):
        # 2 x 2 grid over a 0.018 x 0.018 box at 1 km (0.009) step.
        return create_grid_cells(13.60, 13.618, 100.35, 100.368, cell_size_km=1.0)

    def test_missing_accident_columns_raise(self):
        bad = pd.DataFrame({"LAT": [13.6], "LON": [100.35]})
        with pytest.raises(ValueError):
            assign_to_grid(bad, self._small_grid())

    def test_missing_grid_columns_raise(self):
        accidents = pd.DataFrame({"LATITUDE": [13.6045], "LONGITUDE": [100.3545]})
        bad_grid = pd.DataFrame({"grid_id": ["G000_000"]})  # missing bounds/centers
        with pytest.raises(ValueError):
            assign_to_grid(accidents, bad_grid)

    def test_assigns_point_to_its_cell(self):
        grid = self._small_grid()
        # Centre of the first cell (G000_000).
        accidents = pd.DataFrame({"LATITUDE": [13.6045], "LONGITUDE": [100.3545]})
        result = assign_to_grid(accidents, grid)
        assert result.loc[0, "grid_id"] == "G000_000"

    def test_out_of_bounds_point_is_unassigned(self):
        grid = self._small_grid()
        accidents = pd.DataFrame({"LATITUDE": [50.0], "LONGITUDE": [50.0]})
        result = assign_to_grid(accidents, grid)
        assert pd.isna(result.loc[0, "grid_id"])

    def test_adds_grid_columns(self):
        grid = self._small_grid()
        accidents = pd.DataFrame({"LATITUDE": [13.6045], "LONGITUDE": [100.3545]})
        result = assign_to_grid(accidents, grid)
        assert {"grid_id", "center_lat", "center_lon"}.issubset(result.columns)


class TestCalculateGridStatistics:
    def _assigned(self):
        # 6 accidents in GA, 3 in GB, 1 in GC.
        rows = (
            [{"grid_id": "GA", "center_lat": 1.0, "center_lon": 1.0}] * 6
            + [{"grid_id": "GB", "center_lat": 2.0, "center_lon": 2.0}] * 3
            + [{"grid_id": "GC", "center_lat": 3.0, "center_lon": 3.0}] * 1
        )
        return pd.DataFrame(rows)

    def test_missing_columns_raise(self):
        with pytest.raises(ValueError):
            calculate_grid_statistics(pd.DataFrame({"grid_id": ["GA"]}))

    def test_empty_returns_empty_schema(self):
        df = pd.DataFrame({"grid_id": [np.nan], "center_lat": [1.0], "center_lon": [1.0]})
        result = calculate_grid_statistics(df)
        assert len(result) == 0
        assert list(result.columns) == [
            "grid_id", "center_lat", "center_lon", "total_accidents", "risk_score", "risk_level"
        ]

    def test_totals_per_cell(self):
        stats = calculate_grid_statistics(self._assigned()).set_index("grid_id")
        assert stats.loc["GA", "total_accidents"] == 6
        assert stats.loc["GB", "total_accidents"] == 3
        assert stats.loc["GC", "total_accidents"] == 1

    def test_risk_levels_match_thresholds(self):
        stats = calculate_grid_statistics(self._assigned()).set_index("grid_id")
        assert stats.loc["GA", "risk_level"] == "high"     # >= 5
        assert stats.loc["GB", "risk_level"] == "medium"   # 2-4
        assert stats.loc["GC", "risk_level"] == "low"      # <= 1

    def test_risk_score_is_relative_to_busiest_cell(self):
        stats = calculate_grid_statistics(self._assigned()).set_index("grid_id")
        assert stats.loc["GA", "risk_score"] == 100.0      # busiest → 100
        assert stats.loc["GB", "risk_score"] == 50.0       # 3/6
        assert stats.loc["GC", "risk_score"] == pytest.approx(16.67, abs=0.01)
