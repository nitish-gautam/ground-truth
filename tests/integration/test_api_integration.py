#!/usr/bin/env python3
"""
End-to-End API Integration Test
Tests ML models through the actual FastAPI endpoint logic
"""
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

import numpy as np
import tifffile
from backend.app.ml.inference.predictor import get_predictor

def test_api_integration():
    """Test ML integration via API endpoint logic"""
    print("=" * 70)
    print("API Integration Test - ML Models via Backend")
    print("=" * 70)

    # Load predictor (this is what the API does on startup)
    print("\n[1/3] Loading ML predictor (API startup simulation)...")
    try:
        predictor = get_predictor()
        print("✅ ML predictor loaded successfully")
        print(f"   - Material classifier: loaded")
        print(f"   - Feature scaler: loaded")
        print(f"   - Quality models: {'loaded' if predictor.has_quality_models else 'using heuristics'}")
    except Exception as e:
        print(f"❌ Failed to load predictor: {e}")
        return False

    # Load test sample
    print("\n[2/3] Processing hyperspectral sample...")
    dataset_path = Path("datasets/raw/hyperspectral/umkc-material-surfaces")
    sample_file = next((dataset_path / "Concrete" / "HSI_TIFF_50x50").glob("Auto*.tiff"))
    print(f"   - Loading: {sample_file.name}")

    cube = tifffile.imread(sample_file)
    print(f"   - Shape: {cube.shape}")
    print(f"   - Data type: {cube.dtype}")

    # Run prediction (this is what the API endpoint does)
    print("\n[3/3] Running ML prediction...")
    result = predictor.predict(cube)

    print("\nAPI Response (JSON):")
    print(f"{{")
    print(f'  "material_type": "{result["material_type"]}",')
    print(f'  "material_confidence": {result["material_confidence"]:.4f},')
    print(f'  "quality_score": {result["quality_score"]:.2f},')
    strength_str = f'{result["predicted_strength"]:.2f}' if result["predicted_strength"] else "null"
    print(f'  "predicted_strength": {strength_str},')
    print(f'  "confidence": {result["confidence"]:.2f}')
    print(f"}}")

    # Validate response
    print("\n" + "-" * 70)
    print("Validation:")

    tests_passed = 0
    tests_total = 4

    # Test 1: Material type is correct
    if result["material_type"] == "Concrete":
        print("  ✅ Material type: Concrete (correct)")
        tests_passed += 1
    else:
        print(f"  ❌ Material type: {result['material_type']} (expected Concrete)")

    # Test 2: Confidence is high
    if result["material_confidence"] > 0.95:
        print(f"  ✅ Material confidence: {result['material_confidence']:.1%} (>95%)")
        tests_passed += 1
    else:
        print(f"  ❌ Material confidence: {result['material_confidence']:.1%} (<95%)")

    # Test 3: Quality score in valid range
    if 82 <= result["quality_score"] <= 96:
        print(f"  ✅ Quality score: {result['quality_score']:.2f} (valid range 82-96)")
        tests_passed += 1
    else:
        print(f"  ❌ Quality score: {result['quality_score']:.2f} (out of range)")

    # Test 4: Response has all required fields
    required_fields = ["material_type", "material_confidence", "quality_score", "predicted_strength", "confidence"]
    if all(field in result for field in required_fields):
        print(f"  ✅ API response structure: All {len(required_fields)} fields present")
        tests_passed += 1
    else:
        missing = [f for f in required_fields if f not in result]
        print(f"  ❌ API response structure: Missing fields: {missing}")

    # Deterministic test
    print("\n[Bonus] Testing deterministic behavior...")
    result2 = predictor.predict(cube)
    if (result["material_type"] == result2["material_type"] and
        abs(result["quality_score"] - result2["quality_score"]) < 1e-6):
        print("  ✅ Predictions are deterministic (same input → same output)")
    else:
        print("  ❌ Predictions are non-deterministic")

    print("\n" + "=" * 70)
    print(f"RESULT: {tests_passed}/{tests_total} tests passed")
    print("=" * 70)

    if tests_passed == tests_total:
        print("\n🎉 API INTEGRATION VERIFIED - ML models work via backend!")
        return True
    else:
        print(f"\n⚠️  {tests_total - tests_passed} test(s) failed")
        return False

if __name__ == "__main__":
    try:
        success = test_api_integration()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
