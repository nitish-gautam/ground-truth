#!/usr/bin/env python3
"""
Standalone ML Model Test
Tests trained models without requiring full backend infrastructure
"""
import sys
import os
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

import numpy as np
import tifffile
from backend.app.ml.inference.predictor import HyperspectralPredictor

def load_test_sample(material_type: str, sample_num: int = 1):
    """Load a test hyperspectral sample"""
    dataset_path = Path("datasets/raw/hyperspectral/umkc-material-surfaces")

    if material_type.lower() == "concrete":
        folder = dataset_path / "Concrete" / "HSI_TIFF_50x50"
        files = sorted(folder.glob("Auto*.tiff"))
    else:  # asphalt
        folder = dataset_path / "Asphalt" / "HSI_TIFF_50x50"
        files = sorted(folder.glob("Auto*.tiff"))

    if not files:
        raise FileNotFoundError(f"No samples found in {folder}")

    sample_file = files[min(sample_num - 1, len(files) - 1)]
    print(f"Loading: {sample_file.name}")

    cube = tifffile.imread(sample_file)
    return cube, sample_file.name

def test_ml_predictions():
    """Test ML predictions with real hyperspectral data"""
    print("=" * 70)
    print("ML Model Validation Test")
    print("=" * 70)

    # Initialize predictor
    print("\n[1/5] Loading ML predictor...")
    try:
        predictor = HyperspectralPredictor()
        print("✅ ML predictor loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load predictor: {e}")
        return False

    # Test with concrete samples
    print("\n[2/5] Testing concrete classification...")
    concrete_results = []
    for i in range(1, 4):
        cube, filename = load_test_sample("concrete", i)
        result = predictor.predict(cube)
        concrete_results.append(result)
        print(f"  Sample {i} ({filename}):")
        print(f"    Material: {result['material_type']} (expected: Concrete)")
        print(f"    Quality: {result['quality_score']:.2f}")
        print(f"    Confidence: {result['confidence']:.1f}%")

    concrete_correct = sum(1 for r in concrete_results if r['material_type'] == 'Concrete')
    print(f"\n  Result: {concrete_correct}/3 correct")

    # Test with asphalt samples
    print("\n[3/5] Testing asphalt classification...")
    asphalt_results = []
    for i in range(1, 4):
        cube, filename = load_test_sample("asphalt", i)
        result = predictor.predict(cube)
        asphalt_results.append(result)
        print(f"  Sample {i} ({filename}):")
        print(f"    Material: {result['material_type']} (expected: Asphalt)")
        print(f"    Quality: {result['quality_score']:.2f}")
        print(f"    Confidence: {result['confidence']:.1f}%")

    asphalt_correct = sum(1 for r in asphalt_results if r['material_type'] == 'Asphalt')
    print(f"\n  Result: {asphalt_correct}/3 correct")

    # Test deterministic behavior
    print("\n[4/5] Testing deterministic predictions...")
    cube, filename = load_test_sample("concrete", 1)
    predictions = []
    inference_times = []

    for i in range(5):
        start = time.time()
        result = predictor.predict(cube)
        inference_time = (time.time() - start) * 1000
        inference_times.append(inference_time)
        predictions.append(result)

    # Check if all predictions are identical
    first = predictions[0]
    all_identical = all(
        p['material_type'] == first['material_type'] and
        abs(p['quality_score'] - first['quality_score']) < 1e-6 and
        abs(p['confidence'] - first['confidence']) < 1e-6
        for p in predictions
    )

    avg_inference_time = np.mean(inference_times)

    print(f"  Ran 5 predictions on same sample:")
    print(f"    All identical: {'✅ Yes' if all_identical else '❌ No'}")
    print(f"    Avg inference time: {avg_inference_time:.1f}ms")

    # Summary
    print("\n[5/5] Test Summary")
    print("-" * 70)

    total_correct = concrete_correct + asphalt_correct
    total_samples = 6
    accuracy = (total_correct / total_samples) * 100

    tests_passed = 0
    tests_total = 4

    print(f"  Classification Accuracy: {total_correct}/{total_samples} ({accuracy:.1f}%)")
    if accuracy == 100:
        print("  ✅ Classification test PASSED")
        tests_passed += 1
    else:
        print("  ❌ Classification test FAILED")

    print(f"  Deterministic Predictions: {'Yes' if all_identical else 'No'}")
    if all_identical:
        print("  ✅ Deterministic test PASSED")
        tests_passed += 1
    else:
        print("  ❌ Deterministic test FAILED")

    print(f"  Avg Inference Time: {avg_inference_time:.1f}ms")
    if avg_inference_time < 1000:
        print("  ✅ Performance test PASSED (<1000ms target)")
        tests_passed += 1
    else:
        print("  ❌ Performance test FAILED (>1000ms)")

    print(f"  Concrete Accuracy: {concrete_correct}/3 ({concrete_correct/3*100:.1f}%)")
    if concrete_correct == 3:
        print("  ✅ Concrete test PASSED")
        tests_passed += 1
    else:
        print("  ❌ Concrete test FAILED")

    print("\n" + "=" * 70)
    print(f"FINAL RESULT: {tests_passed}/{tests_total} tests passed")
    print("=" * 70)

    if tests_passed == tests_total:
        print("\n🎉 ALL TESTS PASSED - ML models are working correctly!")
        return True
    else:
        print(f"\n⚠️  {tests_total - tests_passed} test(s) failed")
        return False

if __name__ == "__main__":
    try:
        success = test_ml_predictions()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
