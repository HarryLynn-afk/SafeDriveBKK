"""
Pre-generate NLP and Object Detection caches.

Run this BEFORE the hackathon to:
1. Generate NLP analysis cache (Thai text processing)
2. Download YOLOv8n model and run detection on sample images

Usage:
    python scripts/setup_nlp_detection.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def setup_nlp():
    """Generate NLP analysis cache."""
    LOGGER.info("=== NLP Analysis Cache ===")
    from nlp_analysis import NLPAnalyzer

    analyzer = NLPAnalyzer()
    cache = analyzer.generate_cache()
    LOGGER.info(
        "NLP cache: %d cause categories, %d recommendations, %d word cloud entries",
        len(cache["cause_analysis"]["categories"]),
        len(cache["recommendations"]),
        len(cache["word_cloud"]["words"]),
    )


def setup_detection():
    """Generate Object Detection cache (requires sample images)."""
    LOGGER.info("=== Object Detection Cache ===")
    sample_dir = Path(__file__).resolve().parent.parent / "static" / "sample_images"
    images = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))

    if not images:
        LOGGER.warning(
            "No sample images found in %s. "
            "Add .jpg/.png images and re-run to generate detection cache.",
            sample_dir,
        )
        return

    from object_detection import VehicleDetector

    detector = VehicleDetector()
    cache = detector.generate_cache(sample_dir)
    LOGGER.info("Detection cache: %d images processed", cache.get("total_images", 0))


if __name__ == "__main__":
    setup_nlp()
    print()
    setup_detection()
    print()
    LOGGER.info("Setup complete.")
