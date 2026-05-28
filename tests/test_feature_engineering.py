"""
Boundary tests for the feature_engineering pipeline.

These functions compute the per-grid features the XGBoost model trains on, so
their correctness is load-bearing. They are pure transforms over in-memory
DataFrames — file I/O lives in load_data/save_outputs/main, which are not tested
here — so no mocking is needed.
"""

import pandas as pd
import pytest

from src.feature_engineering import (
    _parse_grid_indices,
    aggregate_accident_features,
    build_grid_accident_lookup,
    calculate_neighbor_features,
    calculate_spatial_features,
    BANGKOK_CENTER_LAT,
    BANGKOK_CENTER_LON,
    COL_DEATHS,
    COL_MINOR_INJURY,
    COL_MOTORCYCLES,
    COL_PASSENGER_CARS,
    COL_PICKUPS,
    COL_SERIOUS_INJURY,
    COL_TRUCKS_6W,
    COL_TRUCKS_10PLUS,
    COL_TRUCKS_10W,
    COL_VEHICLES,
)


class TestParseGridIndices:
    def test_parses_lat_and_lon_indices(self):
        assert _parse_grid_indices("G020_002") == (20, 2)
        assert _parse_grid_indices("G000_000") == (0, 0)
        assert _parse_grid_indices("G038_061") == (38, 61)


class TestBuildGridAccidentLookup:
    def test_maps_indices_to_total_accidents(self):
        grid_stats = pd.DataFrame({
            "grid_id": ["G001_002", "G000_000"],
            "total_accidents": [5, 12],
        })
        lookup = build_grid_accident_lookup(grid_stats)
        assert lookup[(1, 2)] == 5
        assert lookup[(0, 0)] == 12


class TestCalculateNeighborFeatures:
    def test_sums_the_eight_surrounding_cells(self):
        grid_stats = pd.DataFrame({"grid_id": ["G001_001"]})
        # Distinct counts in each of the 8 directions around (1, 1).
        lookup = {
            (0, 1): 1,  # n
            (2, 1): 2,  # s
            (1, 2): 3,  # e
            (1, 0): 4,  # w
            (0, 2): 5,  # ne
            (0, 0): 6,  # nw
            (2, 2): 7,  # se
            (2, 0): 8,  # sw
        }
        row = calculate_neighbor_features(grid_stats, lookup).iloc[0]
        assert row["accidents_neighbors_total"] == 36   # 1+...+8
        assert row["accidents_neighbors_mean"] == 4.5    # 36 / 8
        assert row["accidents_neighbors_max"] == 8

    def test_missing_neighbors_count_as_zero(self):
        grid_stats = pd.DataFrame({"grid_id": ["G005_005"]})
        row = calculate_neighbor_features(grid_stats, {}).iloc[0]
        assert row["accidents_neighbors_total"] == 0
        assert row["accidents_neighbors_max"] == 0


class TestCalculateSpatialFeatures:
    def _grid(self, total_accidents):
        return pd.DataFrame({
            "grid_id": ["G000_000"],
            "center_lat": [BANGKOK_CENTER_LAT],
            "center_lon": [BANGKOK_CENTER_LON],
            "total_accidents": [total_accidents],
        })

    def test_distance_is_zero_at_city_center(self):
        feats = calculate_spatial_features(self._grid(10))
        assert feats.loc[0, "distance_from_center_km"] == pytest.approx(0.0, abs=1e-3)

    def test_distance_grows_with_offset(self):
        grid = pd.DataFrame({
            "grid_id": ["G000_000"],
            "center_lat": [BANGKOK_CENTER_LAT + 0.1],
            "center_lon": [BANGKOK_CENTER_LON + 0.1],
            "total_accidents": [1],
        })
        assert calculate_spatial_features(grid).loc[0, "distance_from_center_km"] > 0

    def test_density_scales_linearly_with_accident_count(self):
        d10 = calculate_spatial_features(self._grid(10)).loc[0, "grid_density"]
        d20 = calculate_spatial_features(self._grid(20)).loc[0, "grid_density"]
        assert d20 == pytest.approx(2 * d10, rel=1e-6)


class TestAggregateAccidentFeatures:
    def _fixture(self):
        # Two accidents that both bin into G000_000 (lat 13.6045, lon 100.3545).
        accidents = pd.DataFrame([
            {
                "LATITUDE": 13.6045, "LONGITUDE": 100.3545,
                COL_DEATHS: 1, COL_SERIOUS_INJURY: 0, COL_MINOR_INJURY: 2,
                COL_VEHICLES: 1, COL_MOTORCYCLES: 1, COL_PASSENGER_CARS: 0, COL_PICKUPS: 0,
                COL_TRUCKS_6W: 0, COL_TRUCKS_10W: 0, COL_TRUCKS_10PLUS: 0,
                "is_rush_hour": True, "is_weekend": False, "hour": 8, "day_of_week": 1, "month": 1,
            },
            {
                "LATITUDE": 13.6045, "LONGITUDE": 100.3545,
                COL_DEATHS: 0, COL_SERIOUS_INJURY: 1, COL_MINOR_INJURY: 0,
                COL_VEHICLES: 2, COL_MOTORCYCLES: 1, COL_PASSENGER_CARS: 1, COL_PICKUPS: 0,
                COL_TRUCKS_6W: 0, COL_TRUCKS_10W: 0, COL_TRUCKS_10PLUS: 0,
                "is_rush_hour": False, "is_weekend": True, "hour": 18, "day_of_week": 5, "month": 2,
            },
        ])
        grid = pd.DataFrame({
            "grid_id": ["G000_000"], "center_lat": [13.6045],
            "center_lon": [100.3545], "total_accidents": [2],
        })
        return accidents, grid

    def test_severity_weights_deaths_serious_minor(self):
        # severity = deaths*10 + serious*5 + minor*1 → (12 + 5) / 2 rows = 8.5 mean.
        out = aggregate_accident_features(*self._fixture()).set_index("grid_id")
        assert out.loc["G000_000", "severity_total"] == 17
        assert out.loc["G000_000", "severity_mean"] == 8.5

    def test_fatality_flag_set_when_any_death(self):
        out = aggregate_accident_features(*self._fixture()).set_index("grid_id")
        assert out.loc["G000_000", "has_fatality"] == 1

    def test_vehicle_mix_is_percentage_of_total(self):
        # 2 motorcycles + 1 car out of 3 vehicles → 66.67% / 33.33%.
        out = aggregate_accident_features(*self._fixture()).set_index("grid_id")
        assert out.loc["G000_000", "motorcycle_pct"] == pytest.approx(66.67, abs=0.01)
        assert out.loc["G000_000", "car_pct"] == pytest.approx(33.33, abs=0.01)

    def test_temporal_percentages(self):
        out = aggregate_accident_features(*self._fixture()).set_index("grid_id")
        assert out.loc["G000_000", "rush_hour_pct"] == 50.0
        assert out.loc["G000_000", "weekend_pct"] == 50.0

    def test_accidents_outside_known_grids_are_dropped(self):
        accidents, grid = self._fixture()
        # Add an accident far outside the single known grid cell.
        extra = accidents.iloc[[0]].copy()
        extra["LATITUDE"] = 13.90
        extra["LONGITUDE"] = 100.80
        accidents = pd.concat([accidents, extra], ignore_index=True)
        out = aggregate_accident_features(accidents, grid)
        # Only the known grid survives the valid-grid filter.
        assert set(out["grid_id"]) == {"G000_000"}
