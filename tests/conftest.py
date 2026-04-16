"""
Shared pytest fixtures used across all test modules.
"""

import pandas as pd
import pytest

from src.app import create_app


# ---------------------------------------------------------------------------
# Flask test client
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def flask_app():
    app = create_app()
    app.config["TESTING"] = True
    return app


@pytest.fixture(scope="session")
def client(flask_app):
    return flask_app.test_client()


# ---------------------------------------------------------------------------
# Minimal grid DataFrame (no file I/O required)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_grid_df():
    """Four-row grid covering a small area inside Bangkok for unit tests."""
    return pd.DataFrame({
        "grid_id":        ["G010_010", "G010_011", "G011_010", "G011_011"],
        "lat_min":        [13.690,     13.690,     13.699,     13.699],
        "lat_max":        [13.699,     13.699,     13.708,     13.708],
        "lon_min":        [100.440,    100.449,    100.440,    100.449],
        "lon_max":        [100.449,    100.458,    100.449,    100.458],
        "center_lat":     [13.6945,    13.6945,    13.7035,    13.7035],
        "center_lon":     [100.4445,   100.4535,   100.4445,   100.4535],
        "total_accidents":[5,          15,         0,          30],
        "risk_level":     ["low",      "medium",   "low",      "high"],
        "risk_score":     [10,         45,         0,          75],
    })


# ---------------------------------------------------------------------------
# Minimal accidents DataFrame
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_accidents_df():
    return pd.DataFrame({
        "hour":         [7, 8, 17, 17, 23, 8, 17],
        "day_of_week":  [0, 4,  4,  5,  5, 0,  4],
        "month":        [1, 1,  2,  2,  3, 3,  4],
        "is_rush_hour": [True, True, True, False, False, True, True],
        "is_weekend":   [False, False, False, True, True, False, False],
    })
