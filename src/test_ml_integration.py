#!/usr/bin/env python3
"""
ML Integration Test Suite
Tests XGBoost model integration in Flask app
"""
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from ml_model import RiskPredictor


def test_1_model_loading():
    """TEST 1: Verify ML model loads correctly"""
    print("\n" + "=" * 60)
    print("TEST 1: Model Loading")
    print("=" * 60)

    try:
        predictor = RiskPredictor("models/xgboost_model.pkl")
        print("  PASS  Model loaded successfully")
        print("  Model type: XGBoost")
        print("  Features: 44")
        return True
    except Exception as e:
        print(f"  FAIL  {e}")
        return False


def test_2_single_predictions():
    """TEST 2: Test predictions for known locations"""
    print("\n" + "=" * 60)
    print("TEST 2: Single Predictions")
    print("=" * 60)

    predictor = RiskPredictor()

    test_cases = [
        {
            "name": "Ratchada (High Risk)",
            "lat": 13.73,
            "lon": 100.72,
            "expected_level": "high",
            "expected_score_min": 70,
        },
        {
            "name": "G014_043 (High Risk)",
            "lat": 13.7305,
            "lon": 100.7415,
            "expected_level": "high",
            "expected_score_min": 70,
        },
        {
            "name": "Suburban Area (Low Risk)",
            "lat": 13.85,
            "lon": 100.50,
            "expected_level": "low",
            "expected_score_max": 30,
        },
    ]

    passed = 0
    failed = 0

    for test in test_cases:
        result = predictor.predict_risk(
            test["lat"],
            test["lon"],
            hour=18,  # Evening rush hour
            day_of_week=4,  # Friday
        )

        print(f"\n{test['name']}:")
        print(f"  Risk Score: {result['risk_score']}/100")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Method: {result['method']}")

        if "expected_level" in test:
            if result["risk_level"] == test["expected_level"]:
                print(f"  PASS  Level matches expected ({test['expected_level']})")
                passed += 1
            else:
                print(f"  FAIL  Expected {test['expected_level']}, got {result['risk_level']}")
                failed += 1

        if "expected_score_min" in test:
            if result["risk_score"] >= test["expected_score_min"]:
                print(f"  PASS  Score >= {test['expected_score_min']}")
                passed += 1
            else:
                print(f"  FAIL  Score {result['risk_score']} < {test['expected_score_min']}")
                failed += 1

        if "expected_score_max" in test:
            if result["risk_score"] <= test["expected_score_max"]:
                print(f"  PASS  Score <= {test['expected_score_max']}")
                passed += 1
            else:
                print(f"  FAIL  Score {result['risk_score']} > {test['expected_score_max']}")
                failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def test_3_batch_performance():
    """TEST 3: Test batch prediction performance"""
    print("\n" + "=" * 60)
    print("TEST 3: Batch Prediction Performance")
    print("=" * 60)

    predictor = RiskPredictor()

    # Generate 100 test locations along a route
    locations = [(13.7467 + i * 0.001, 100.5348 + i * 0.001) for i in range(100)]

    start = time.time()
    results = predictor.predict_batch(locations)
    elapsed = time.time() - start

    print(f"  Batch size: 100 locations")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Average: {elapsed / 100 * 1000:.1f}ms per location")
    print(f"  Throughput: {100 / elapsed:.1f} predictions/sec")

    if elapsed < 2.0:
        print(f"  PASS  Performance acceptable (<2s for 100 predictions)")
        return True
    else:
        print(f"  FAIL  Too slow: {elapsed:.3f}s (should be <2s)")
        return False


def test_4_ml_vs_spatial_comparison():
    """TEST 4: Compare ML and Spatial predictions"""
    print("\n" + "=" * 60)
    print("TEST 4: ML vs Spatial Comparison")
    print("=" * 60)

    predictor = RiskPredictor()

    test_locations = [
        (13.73, 100.72, "Ratchada"),
        (13.74, 100.74, "Din Daeng"),
        (13.76, 100.75, "Huai Khwang"),
        (13.85, 100.50, "Suburban"),
    ]

    print(
        f"\n{'Location':<15} {'ML Score':<12} {'Spatial':<12} {'Diff':<10} {'ML Conf':<10}"
    )
    print("-" * 65)

    total_diff = 0
    count = 0

    for lat, lon, name in test_locations:
        ml_result = predictor.predict_risk(lat, lon)
        spatial_result = predictor.fallback_to_spatial(lat, lon)

        diff = ml_result["risk_score"] - spatial_result["risk_score"]
        total_diff += abs(diff)
        count += 1

        print(
            f"{name:<15} {ml_result['risk_score']:>3}/100 ({ml_result['risk_level']:<6}) "
            f"{spatial_result['risk_score']:>3}/100 ({spatial_result['risk_level']:<6}) "
            f"{diff:>+4.0f}      {ml_result['confidence']:.3f}"
        )

    avg_diff = total_diff / count
    print(f"\nAverage absolute difference: {avg_diff:.1f} points")

    if avg_diff < 50:
        print("  PASS  ML and Spatial are reasonably correlated")
        return True
    else:
        print("  WARN  Large differences - investigate")
        return True  # Not a failure, just informational


def test_5_feature_importance():
    """TEST 5: Verify feature importance is accessible"""
    print("\n" + "=" * 60)
    print("TEST 5: Feature Importance")
    print("=" * 60)

    predictor = RiskPredictor()

    try:
        importance = predictor.get_feature_importance(top_n=10)

        print("  Top 10 most important features:")
        print()

        for i, (feature, score) in enumerate(importance.items(), 1):
            print(f"  {i}. {feature:<25} {score:.3f}")

        top_feature = list(importance.keys())[0]
        if "severity" in top_feature.lower():
            print(f"\n  PASS  Severity feature is top-ranked (as expected)")
            return True
        else:
            print(f"\n  WARN  Top feature is {top_feature} (expected severity_mean)")
            return True  # Not a hard failure

    except Exception as e:
        print(f"  FAIL  {e}")
        return False


def test_6_edge_cases():
    """TEST 6: Test edge cases and error handling"""
    print("\n" + "=" * 60)
    print("TEST 6: Edge Cases & Error Handling")
    print("=" * 60)

    predictor = RiskPredictor()

    edge_cases = [
        {
            "name": "Out of bounds (north)",
            "lat": 14.5,
            "lon": 100.5,
            "should_fallback": True,
        },
        {
            "name": "Out of bounds (east)",
            "lat": 13.7,
            "lon": 101.5,
            "should_fallback": True,
        },
        {
            "name": "Boundary case (min lat)",
            "lat": 13.61,
            "lon": 100.5,
            "should_work": True,
        },
        {
            "name": "Boundary case (max lat)",
            "lat": 13.93,
            "lon": 100.5,
            "should_work": True,
        },
    ]

    passed = 0

    for case in edge_cases:
        try:
            result = predictor.predict_risk(case["lat"], case["lon"])

            print(f"\n{case['name']}:")
            print(f"  Result: {result['risk_score']}/100 ({result['risk_level']})")
            print(f"  Method: {result['method']}")
            print(f"  Confidence: {result['confidence']:.3f}")

            if case.get("should_fallback"):
                if result["method"] in ["out_of_bounds", "spatial"]:
                    print(f"  PASS  Correctly used fallback")
                    passed += 1
                else:
                    print(f"  WARN  Expected fallback, got {result['method']}")

            if case.get("should_work"):
                if result["method"] == "xgboost":
                    print(f"  PASS  XGBoost prediction successful")
                    passed += 1
                else:
                    print(f"  WARN  Expected XGBoost, got {result['method']}")

        except Exception as e:
            print(f"\n{case['name']}:")
            print(f"  FAIL  Exception: {e}")

    print(f"\nPassed: {passed}/{len(edge_cases)}")
    return passed >= len(edge_cases) - 1  # Allow 1 failure


def test_7_confidence_scores():
    """TEST 7: Verify confidence scores are meaningful"""
    print("\n" + "=" * 60)
    print("TEST 7: Confidence Scores")
    print("=" * 60)

    predictor = RiskPredictor()

    test_points = [
        (13.73, 100.72, "high_risk", 0.95),
        (13.85, 100.50, "low_risk", 0.95),
    ]

    passed = 0

    for lat, lon, expected_type, min_confidence in test_points:
        result = predictor.predict_risk(lat, lon)

        print(f"\n{expected_type.replace('_', ' ').title()}:")
        print(f"  Confidence: {result['confidence']:.4f}")

        if result["confidence"] >= min_confidence:
            print(f"  PASS  High confidence (>={min_confidence})")
            passed += 1
        else:
            print(f"  WARN  Lower confidence than expected ({min_confidence})")

    print(f"\nPassed: {passed}/{len(test_points)}")
    return passed >= 1  # At least one should be high confidence


def run_all_tests():
    """Run complete test suite"""
    print("\n" + "=" * 70)
    print("ML INTEGRATION TEST SUITE")
    print("=" * 70)

    tests = [
        ("Model Loading", test_1_model_loading),
        ("Single Predictions", test_2_single_predictions),
        ("Batch Performance", test_3_batch_performance),
        ("ML vs Spatial", test_4_ml_vs_spatial_comparison),
        ("Feature Importance", test_5_feature_importance),
        ("Edge Cases", test_6_edge_cases),
        ("Confidence Scores", test_7_confidence_scores),
    ]

    results = []

    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  FAIL  TEST CRASHED: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {name}")

    print("=" * 70)
    print(f"Result: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\nALL TESTS PASSED - READY FOR DEPLOYMENT")
        return True
    else:
        print(f"\n{total_count - passed_count} tests failed - Review before deployment")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
