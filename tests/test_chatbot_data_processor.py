"""
Contract tests for the chatbot_data_processor ETL builders.

These cover the deep builders that derive their cause lookups from the single
canonical PRIMARY_CAUSES table. The headline test is the regression guard for
the build_recommendations key-mismatch bug, where red_light / tire_blowout were
silently stuck at zones_affected: 0.

The builders are pure functions over already-loaded rows (lists of dicts, string
values like csv.DictReader yields), so no file I/O or mocking is needed.
"""

from src.chatbot_data_processor import (
    PRIMARY_CAUSES,
    build_causes,
    build_recommendations,
    build_zones,
)


def _feature_row(grid_id, total, **cause_pcts):
    """Build one ml_features row with all cause columns zeroed, then overridden.

    cause_pcts is keyed by PRIMARY_CAUSES short key (e.g. red_light=50).
    """
    row = {"grid_id": grid_id, "total_accidents": str(total)}
    for meta in PRIMARY_CAUSES.values():
        row[meta["column"]] = "0"
    row["cause_other"] = "0"
    for key, pct in cause_pcts.items():
        row[PRIMARY_CAUSES[key]["column"]] = str(pct)
    return row


class TestBuildRecommendations:
    def test_all_five_causes_resolve(self):
        """Regression: every recommendation's zones_affected must track its cause.

        Under the old key mismatch, priority-4 (red_light) and priority-5
        (tire_blowout) read counts under long english keys that were never
        populated, so they were permanently 0. This proves all five resolve.
        """
        ml_features = [
            _feature_row("G1", 10, speeding=50),
            _feature_row("G2", 10, cutting_in_front=50),
            _feature_row("G3", 10, drowsy_driving=50),
            _feature_row("G4", 10, red_light=50),
            _feature_row("G5", 10, tire_blowout=50),
        ]
        recs = build_recommendations({}, ml_features)
        affected = {r["priority"]: r["zones_affected"] for r in recs}

        assert affected[1] == 1  # speeding
        assert affected[2] == 1  # cutting_in_front
        assert affected[3] == 1  # drowsy_driving
        assert affected[4] == 1  # red_light  — was 0 under the bug
        assert affected[5] == 1  # tire_blowout — was 0 under the bug

    def test_zone_below_minimum_accidents_excluded(self):
        """A dominant cause in a zone with < 5 accidents is not counted."""
        ml_features = [_feature_row("G1", 4, red_light=50)]
        recs = build_recommendations({}, ml_features)
        affected = {r["priority"]: r["zones_affected"] for r in recs}
        assert affected[4] == 0

    def test_threshold_is_strictly_above_30(self):
        """A cause is counted only when its share exceeds 30%, not at exactly 30."""
        at_threshold = build_recommendations({}, [_feature_row("G1", 10, speeding=30)])
        above = build_recommendations({}, [_feature_row("G1", 10, speeding=31)])
        assert {r["priority"]: r["zones_affected"] for r in at_threshold}[1] == 0
        assert {r["priority"]: r["zones_affected"] for r in above}[1] == 1


class TestBuildZones:
    def _grid_stats(self):
        return [
            {"grid_id": "G1", "risk_score": "72.4", "risk_level": "high"},
        ]

    def _feature(self):
        row = _feature_row("G1", 20, speeding=60, red_light=10)
        row.update({
            "center_lat": "13.70", "center_lon": "100.50",
            "severity_mean": "1.5", "has_fatality": "1",
            "motorcycle_pct": "40", "rush_hour_pct": "25", "weekend_pct": "30",
        })
        return row

    def test_cause_breakdown_keys_are_canonical_plus_other(self):
        zones = build_zones([self._feature()], self._grid_stats())
        keys = list(zones["G1"]["cause_breakdown"].keys())
        assert keys == [
            "speeding", "cutting_in_front", "red_light",
            "tire_blowout", "drowsy_driving", "other",
        ]

    def test_dominant_cause_is_highest_pct(self):
        zones = build_zones([self._feature()], self._grid_stats())
        assert zones["G1"]["dominant_cause"] == "speeding"
        assert zones["G1"]["dominant_cause_pct"] == 60.0

    def test_risk_fields_joined_from_grid_stats(self):
        zones = build_zones([self._feature()], self._grid_stats())
        assert zones["G1"]["risk_score"] == 72.4
        assert zones["G1"]["risk_level"] == "high"


class TestBuildCauses:
    def _nlp_cache(self):
        return {
            "cause_analysis": {
                "categories": [
                    {
                        "thai_name": PRIMARY_CAUSES["red_light"]["thai"],
                        "english_name": "Running red light / traffic sign violation",
                        "count": 100,
                        "percentage": 12.3,
                        "feature_importance": 0.4,
                    },
                ]
            }
        }

    def test_output_key_derived_from_english_name(self):
        """Key matches the slug chatbot_intents looks up (spaces/slashes -> _)."""
        causes = build_causes(self._nlp_cache(), [], [])
        assert "running_red_light___traffic_sign_violation" in causes

    def test_top_zones_sorted_by_cause_count_and_capped(self):
        col = PRIMARY_CAUSES["red_light"]["column"]
        ml_features = [
            {"grid_id": f"G{i}", "total_accidents": "100", col: str(i)}
            for i in range(1, 15)  # 14 zones with non-zero cause share
        ]
        causes = build_causes(self._nlp_cache(), ml_features, [])
        top = causes["running_red_light___traffic_sign_violation"]["top_zones"]
        assert len(top) == 10  # capped at 10
        counts = [z["cause_count"] for z in top]
        assert counts == sorted(counts, reverse=True)  # descending
