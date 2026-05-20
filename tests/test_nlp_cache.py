"""
Tests for src/nlp_analysis.load_nlp_cache

Interface boundary: load_nlp_cache() -> dict

Covers: fresh-cache load, stale-source warning, missing file, invalid JSON,
and in-process cache reuse (second call must not re-read disk).
"""

import json
import logging
import os
import time
from unittest.mock import patch

import pytest

import src.nlp_analysis as nlp_mod
from src.nlp_analysis import load_nlp_cache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_nlp_cache():
    """Wipe the module-level in-process cache before and after each test."""
    nlp_mod._nlp_cache = None
    yield
    nlp_mod._nlp_cache = None


_SAMPLE_CACHE = {
    "pipeline_preview": {"steps": 3},
    "cause_analysis": {"categories": []},
    "recommendations": ["wear a helmet"],
    "word_cloud": {},
}


def _write_cache(path, content=None):
    if content is None:
        content = _SAMPLE_CACHE
    path.write_text(json.dumps(content), encoding="utf-8")


def _set_mtime(path, offset_seconds: float):
    """Shift a file's mtime by offset_seconds relative to now."""
    t = time.time() + offset_seconds
    os.utime(path, (t, t))


# ---------------------------------------------------------------------------
# TestLoadNlpCache
# ---------------------------------------------------------------------------

class TestLoadNlpCache:

    # ---- file exists, accidents absent or older than cache ----

    def test_returns_dict_when_cache_file_exists(self, tmp_path):
        cache_file = tmp_path / "nlp_cache.json"
        _write_cache(cache_file)
        with (
            patch.object(nlp_mod, "NLP_CACHE_PATH", cache_file),
            patch.object(nlp_mod, "ACCIDENTS_PATH", tmp_path / "missing.csv"),
        ):
            result = load_nlp_cache()
        assert isinstance(result, dict)

    def test_returns_full_cache_contents(self, tmp_path):
        cache_file = tmp_path / "nlp_cache.json"
        _write_cache(cache_file)
        with (
            patch.object(nlp_mod, "NLP_CACHE_PATH", cache_file),
            patch.object(nlp_mod, "ACCIDENTS_PATH", tmp_path / "missing.csv"),
        ):
            result = load_nlp_cache()
        assert result["pipeline_preview"] == {"steps": 3}
        assert result["recommendations"] == ["wear a helmet"]

    # ---- staleness detection ----

    def test_logs_warning_when_accidents_file_is_newer_than_cache(self, tmp_path, caplog):
        cache_file = tmp_path / "nlp_cache.json"
        accidents_file = tmp_path / "accidents.csv"
        _write_cache(cache_file)
        accidents_file.write_text("data", encoding="utf-8")
        # Force accidents mtime to be 5 seconds ahead of cache
        _set_mtime(cache_file, -5)
        _set_mtime(accidents_file, 0)

        with (
            patch.object(nlp_mod, "NLP_CACHE_PATH", cache_file),
            patch.object(nlp_mod, "ACCIDENTS_PATH", accidents_file),
            caplog.at_level(logging.WARNING, logger="src.nlp_analysis"),
        ):
            load_nlp_cache()

        assert "stale" in caplog.text.lower()

    def test_stale_warning_still_returns_cached_data(self, tmp_path, caplog):
        cache_file = tmp_path / "nlp_cache.json"
        accidents_file = tmp_path / "accidents.csv"
        _write_cache(cache_file)
        accidents_file.write_text("data", encoding="utf-8")
        _set_mtime(cache_file, -5)
        _set_mtime(accidents_file, 0)

        with (
            patch.object(nlp_mod, "NLP_CACHE_PATH", cache_file),
            patch.object(nlp_mod, "ACCIDENTS_PATH", accidents_file),
            caplog.at_level(logging.WARNING, logger="src.nlp_analysis"),
        ):
            result = load_nlp_cache()

        assert result["pipeline_preview"] == {"steps": 3}

    def test_no_warning_when_cache_is_newer_than_accidents(self, tmp_path, caplog):
        cache_file = tmp_path / "nlp_cache.json"
        accidents_file = tmp_path / "accidents.csv"
        accidents_file.write_text("data", encoding="utf-8")
        _write_cache(cache_file)
        # Force cache to be 5 seconds newer
        _set_mtime(accidents_file, -5)
        _set_mtime(cache_file, 0)

        with (
            patch.object(nlp_mod, "NLP_CACHE_PATH", cache_file),
            patch.object(nlp_mod, "ACCIDENTS_PATH", accidents_file),
            caplog.at_level(logging.WARNING, logger="src.nlp_analysis"),
        ):
            load_nlp_cache()

        assert "stale" not in caplog.text.lower()

    def test_no_warning_when_accidents_file_absent(self, tmp_path, caplog):
        cache_file = tmp_path / "nlp_cache.json"
        _write_cache(cache_file)
        with (
            patch.object(nlp_mod, "NLP_CACHE_PATH", cache_file),
            patch.object(nlp_mod, "ACCIDENTS_PATH", tmp_path / "missing.csv"),
            caplog.at_level(logging.WARNING, logger="src.nlp_analysis"),
        ):
            load_nlp_cache()
        assert "stale" not in caplog.text.lower()

    # ---- missing / corrupt cache file ----

    def test_returns_empty_dict_when_cache_file_missing(self, tmp_path):
        with (
            patch.object(nlp_mod, "NLP_CACHE_PATH", tmp_path / "missing.json"),
            patch.object(nlp_mod, "ACCIDENTS_PATH", tmp_path / "missing.csv"),
        ):
            result = load_nlp_cache()
        assert result == {}

    def test_returns_empty_dict_when_json_is_invalid(self, tmp_path):
        cache_file = tmp_path / "nlp_cache.json"
        cache_file.write_text("{ not valid json {{", encoding="utf-8")
        with (
            patch.object(nlp_mod, "NLP_CACHE_PATH", cache_file),
            patch.object(nlp_mod, "ACCIDENTS_PATH", tmp_path / "missing.csv"),
        ):
            result = load_nlp_cache()
        assert result == {}

    # ---- in-process cache ----

    def test_second_call_returns_same_object(self, tmp_path):
        cache_file = tmp_path / "nlp_cache.json"
        _write_cache(cache_file)
        with (
            patch.object(nlp_mod, "NLP_CACHE_PATH", cache_file),
            patch.object(nlp_mod, "ACCIDENTS_PATH", tmp_path / "missing.csv"),
        ):
            first = load_nlp_cache()
            second = load_nlp_cache()
        assert second is first

    def test_second_call_does_not_pick_up_disk_changes(self, tmp_path):
        cache_file = tmp_path / "nlp_cache.json"
        _write_cache(cache_file)
        with (
            patch.object(nlp_mod, "NLP_CACHE_PATH", cache_file),
            patch.object(nlp_mod, "ACCIDENTS_PATH", tmp_path / "missing.csv"),
        ):
            first = load_nlp_cache()
            cache_file.write_text(json.dumps({"pipeline_preview": {"steps": 99}}), encoding="utf-8")
            second = load_nlp_cache()
        assert second["pipeline_preview"]["steps"] == 3  # stale on-disk change not visible

    # ---- never raises ----

    def test_never_raises_on_missing_files(self, tmp_path):
        with (
            patch.object(nlp_mod, "NLP_CACHE_PATH", tmp_path / "gone.json"),
            patch.object(nlp_mod, "ACCIDENTS_PATH", tmp_path / "gone.csv"),
        ):
            result = load_nlp_cache()
        assert isinstance(result, dict)
