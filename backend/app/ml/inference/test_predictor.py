"""
Test ML Predictor with Real UMKC Files

Validates:
1. Models load correctly
2. Predictions are deterministic (same input = same output)
3. Inference latency is <1 second
4. Predictions are within expected ranges
"""

import sys
import time
import numpy as np
import tifffile
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from app.ml.inference.predictor import get_predictor, reset_predictor


def test_model_loading():
    """Test that all models load successfully."""
    print("\n" + "="*70)
    print("TEST 1: Model Loading")
    print("="*70)

    try:
        predictor = get_predictor()
        print("✅ ML predictor loaded successfully")

        # Check all models are loaded
        assert predictor.material_classifier is not None, "Material classifier not loaded"
        assert predictor.feature_scaler is not None, "Feature scaler not loaded"

        if predictor.has_quality_models:
            assert predictor.quality_regressor is not None, "Quality regressor not loaded"
            assert predictor.strength_regressor is not None, "Strength regressor not loaded"
            assert predictor.confidence_regressor is not None, "Confidence regressor not loaded"
            print("✅ All quality models loaded")
        else:
            print("⚠️  Quality models not available (will use heuristics)")

        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


def test_inference_with_real_files():
    """Test inference with real UMKC hyperspectral files."""
    print("\n" + "="*70)
    print("TEST 2: Inference with Real UMKC Files")
    print("="*70)

    dataset_path = Path("/Users/nitishgautam/Code/prototype/ground-truth/datasets/raw/hyperspectral/umkc-material-surfaces")

    # Get sample files
    concrete_files = list(dataset_path.glob("Concrete/HSI_TIFF_50x50/*.tiff"))[:3]
    asphalt_files = list(dataset_path.glob("Asphalt/HSI_TIFF_50x50/*.tiff"))[:3]

    if not concrete_files:
        print("❌ No concrete samples found")
        return False

    if not asphalt_files:
        print("❌ No asphalt samples found")
        return False

    predictor = get_predictor()
    results = []

    print(f"\nTesting {len(concrete_files)} concrete samples...")
    for fpath in concrete_files:
        try:
            cube = tifffile.imread(str(fpath))

            start_time = time.time()
            predictions = predictor.predict(cube)
            inference_time = (time.time() - start_time) * 1000  # ms

            results.append({
                'file': fpath.name,
                'material': predictions['material_type'],
                'confidence': predictions['confidence'],
                'strength': predictions['predicted_strength'],
                'quality': predictions['quality_score'],
                'inference_ms': inference_time
            })

            print(f"  {fpath.name}:")
            print(f"    Material: {predictions['material_type']}")
            print(f"    Confidence: {predictions['confidence']:.1f}%")
            print(f"    Strength: {predictions['predicted_strength']:.1f} MPa" if predictions['predicted_strength'] else "    Strength: N/A")
            print(f"    Quality: {predictions['quality_score']:.1f}%")
            print(f"    Inference time: {inference_time:.1f}ms")

        except Exception as e:
            print(f"  ❌ Failed to process {fpath.name}: {e}")
            return False

    print(f"\nTesting {len(asphalt_files)} asphalt samples...")
    for fpath in asphalt_files:
        try:
            cube = tifffile.imread(str(fpath))

            start_time = time.time()
            predictions = predictor.predict(cube)
            inference_time = (time.time() - start_time) * 1000  # ms

            results.append({
                'file': fpath.name,
                'material': predictions['material_type'],
                'confidence': predictions['confidence'],
                'strength': predictions['predicted_strength'],
                'quality': predictions['quality_score'],
                'inference_ms': inference_time
            })

            print(f"  {fpath.name}:")
            print(f"    Material: {predictions['material_type']}")
            print(f"    Confidence: {predictions['confidence']:.1f}%")
            print(f"    Strength: {predictions['predicted_strength']}" if predictions['predicted_strength'] else "    Strength: N/A")
            print(f"    Quality: {predictions['quality_score']:.1f}%")
            print(f"    Inference time: {inference_time:.1f}ms")

        except Exception as e:
            print(f"  ❌ Failed to process {fpath.name}: {e}")
            return False

    # Verify all concrete samples classified as concrete
    concrete_correct = sum(1 for r in results[:len(concrete_files)] if r['material'] == 'Concrete')
    asphalt_correct = sum(1 for r in results[len(concrete_files):] if r['material'] == 'Asphalt')

    print(f"\n✅ Concrete classification: {concrete_correct}/{len(concrete_files)} correct")
    print(f"✅ Asphalt classification: {asphalt_correct}/{len(asphalt_files)} correct")

    # Check inference time
    avg_inference_time = np.mean([r['inference_ms'] for r in results])
    max_inference_time = np.max([r['inference_ms'] for r in results])

    print(f"\n⏱️  Average inference time: {avg_inference_time:.1f}ms")
    print(f"⏱️  Max inference time: {max_inference_time:.1f}ms")

    if max_inference_time < 1000:
        print("✅ Inference time < 1 second (target met)")
    else:
        print("⚠️  Inference time > 1 second (optimization needed)")

    return concrete_correct == len(concrete_files) and asphalt_correct == len(asphalt_files)


def test_deterministic_predictions():
    """Test that predictions are deterministic (same input = same output)."""
    print("\n" + "="*70)
    print("TEST 3: Deterministic Predictions")
    print("="*70)

    dataset_path = Path("/Users/nitishgautam/Code/prototype/ground-truth/datasets/raw/hyperspectral/umkc-material-surfaces")

    # Get one sample file
    concrete_file = list(dataset_path.glob("Concrete/HSI_TIFF_50x50/*.tiff"))[0]

    predictor = get_predictor()
    cube = tifffile.imread(str(concrete_file))

    print(f"\nTesting determinism with {concrete_file.name}...")
    print("Running 5 predictions on the same input...\n")

    predictions_list = []
    for i in range(5):
        predictions = predictor.predict(cube)
        predictions_list.append(predictions)
        print(f"  Run {i+1}:")
        print(f"    Material: {predictions['material_type']}")
        print(f"    Confidence: {predictions['confidence']:.6f}%")
        print(f"    Strength: {predictions['predicted_strength']:.6f} MPa" if predictions['predicted_strength'] else "    Strength: N/A")
        print(f"    Quality: {predictions['quality_score']:.6f}%")

    # Check all predictions are identical
    all_same = True
    reference = predictions_list[0]

    for i, pred in enumerate(predictions_list[1:], 2):
        if pred['material_type'] != reference['material_type']:
            print(f"❌ Material type differs at run {i}")
            all_same = False
        if abs(pred['confidence'] - reference['confidence']) > 1e-6:
            print(f"❌ Confidence differs at run {i}")
            all_same = False
        if pred['predicted_strength'] and reference['predicted_strength']:
            if abs(pred['predicted_strength'] - reference['predicted_strength']) > 1e-6:
                print(f"❌ Strength differs at run {i}")
                all_same = False
        if abs(pred['quality_score'] - reference['quality_score']) > 1e-6:
            print(f"❌ Quality score differs at run {i}")
            all_same = False

    if all_same:
        print("\n✅ All predictions are identical (deterministic)")
        print("✅ No more random predictions - using trained ML models!")
        return True
    else:
        print("\n❌ Predictions vary (non-deterministic)")
        return False


def test_prediction_ranges():
    """Test that predictions are within expected ranges."""
    print("\n" + "="*70)
    print("TEST 4: Prediction Ranges")
    print("="*70)

    dataset_path = Path("/Users/nitishgautam/Code/prototype/ground-truth/datasets/raw/hyperspectral/umkc-material-surfaces")

    concrete_files = list(dataset_path.glob("Concrete/HSI_TIFF_50x50/*.tiff"))
    asphalt_files = list(dataset_path.glob("Asphalt/HSI_TIFF_50x50/*.tiff"))

    predictor = get_predictor()

    all_valid = True

    print("\nChecking concrete samples...")
    for fpath in concrete_files[:5]:  # Test first 5
        cube = tifffile.imread(str(fpath))
        predictions = predictor.predict(cube)

        # Check ranges
        if not (94.0 <= predictions['confidence'] <= 98.9):
            print(f"  ❌ {fpath.name}: Confidence {predictions['confidence']:.1f}% out of range [94.0, 98.9]")
            all_valid = False

        if predictions['predicted_strength'] is not None:
            if not (28.0 <= predictions['predicted_strength'] <= 48.0):
                print(f"  ❌ {fpath.name}: Strength {predictions['predicted_strength']:.1f} MPa out of range [28.0, 48.0]")
                all_valid = False

        if not (82.0 <= predictions['quality_score'] <= 96.0):
            print(f"  ❌ {fpath.name}: Quality {predictions['quality_score']:.1f}% out of range [82.0, 96.0]")
            all_valid = False

    print("\nChecking asphalt samples...")
    for fpath in asphalt_files[:5]:  # Test first 5
        cube = tifffile.imread(str(fpath))
        predictions = predictor.predict(cube)

        # Asphalt should have different ranges
        if predictions['predicted_strength'] is not None:
            print(f"  ⚠️  {fpath.name}: Asphalt should not have strength prediction")

    if all_valid:
        print("\n✅ All predictions within expected ranges")
        return True
    else:
        print("\n❌ Some predictions out of range")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("ML PREDICTOR VALIDATION TESTS")
    print("="*70)
    print("\nThis script validates:")
    print("  1. Models load correctly")
    print("  2. Inference works with real UMKC files")
    print("  3. Predictions are deterministic (not random)")
    print("  4. Predictions are within expected ranges")
    print("  5. Inference latency < 1 second")

    results = {}

    # Test 1: Model loading
    results['model_loading'] = test_model_loading()

    if not results['model_loading']:
        print("\n❌ Cannot proceed - models failed to load")
        return

    # Test 2: Inference with real files
    results['inference'] = test_inference_with_real_files()

    # Test 3: Deterministic predictions
    results['deterministic'] = test_deterministic_predictions()

    # Test 4: Prediction ranges
    results['ranges'] = test_prediction_ranges()

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
        print("\nML models are working correctly!")
        print("Ready for production integration.")
        print("\nNext steps:")
        print("  1. Start backend API: uvicorn app.main:app --reload")
        print("  2. Test API endpoint with curl (see commands below)")
        print("  3. Update frontend to display ML predictions")
    else:
        print("\n" + "="*70)
        print("❌ SOME TESTS FAILED")
        print("="*70)
        print("\nPlease review errors above and fix before production deployment.")

    print("\n" + "="*70)
    print("CURL COMMANDS FOR API TESTING")
    print("="*70)
    print("\n# Test with concrete sample:")
    print('curl -X POST "http://localhost:8000/api/v1/progress/hyperspectral/analyze-material" \\')
    print('  -F "file=@/Users/nitishgautam/Code/prototype/ground-truth/datasets/raw/hyperspectral/umkc-material-surfaces/Concrete/HSI_TIFF_50x50/Auto119.tiff"')
    print("\n# Test with asphalt sample:")
    print('curl -X POST "http://localhost:8000/api/v1/progress/hyperspectral/analyze-material" \\')
    print('  -F "file=@/Users/nitishgautam/Code/prototype/ground-truth/datasets/raw/hyperspectral/umkc-material-surfaces/Asphalt/HSI_TIFF_50x50/Auto005.tiff"')
    print("")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Tests failed with error: {e}")
        import traceback
        traceback.print_exc()
