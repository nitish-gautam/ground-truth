#!/usr/bin/env python3
"""
Frontend ↔ Backend Integration Test
Simulates the exact flow: Frontend uploads file → Backend analyzes with ML → Returns JSON
"""
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

import asyncio
from io import BytesIO
from pathlib import Path
import json

async def test_frontend_backend_integration():
    """Test the complete flow from frontend file upload to backend ML analysis"""
    print("=" * 70)
    print("Frontend ↔ Backend Integration Test")
    print("=" * 70)

    # Simulate frontend upload
    print("\n[FRONTEND] User uploads hyperspectral image...")
    dataset_path = Path("datasets/raw/hyperspectral/umkc-material-surfaces")
    sample_file = next((dataset_path / "Concrete" / "HSI_TIFF_50x50").glob("Auto*.tiff"))
    print(f"  File: {sample_file.name}")
    print(f"  Size: {sample_file.stat().st_size / 1024:.1f} KB")

    # Read file as bytes (what frontend sends)
    with open(sample_file, 'rb') as f:
        file_contents = f.read()

    # Simulate the FastAPI endpoint processing
    print("\n[BACKEND] Processing file through API endpoint...")

    # Import the actual endpoint logic
    from backend.app.api.v1.endpoints.hyperspectral import analyze_material_hyperspectral
    from fastapi import UploadFile

    # Create mock UploadFile (simulates frontend FormData)
    class MockFile:
        def __init__(self, filename, content):
            self.filename = filename
            self._content = content

        async def read(self):
            return self._content

    mock_upload = MockFile(sample_file.name, file_contents)

    # Call the actual endpoint
    try:
        response = await analyze_material_hyperspectral(file=mock_upload)
        print("  ✅ API endpoint processed successfully")
    except Exception as e:
        print(f"  ❌ API endpoint failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Display response (what frontend receives)
    print("\n[FRONTEND] Received JSON response:")
    print("-" * 70)
    print(json.dumps(response, indent=2))
    print("-" * 70)

    # Validate response structure (what ConcreteQualityAnalyzer.tsx expects)
    print("\n[VALIDATION] Checking response matches frontend expectations...")

    tests_passed = 0
    tests_total = 8

    # Test 1: Has analysis_id
    if "analysis_id" in response:
        print(f"  ✅ analysis_id: {response['analysis_id']}")
        tests_passed += 1
    else:
        print("  ❌ Missing analysis_id")

    # Test 2: Has image_metadata
    if "image_metadata" in response:
        meta = response['image_metadata']
        print(f"  ✅ image_metadata: {meta.get('filename')}, {meta.get('width')}x{meta.get('height')}")
        tests_passed += 1
    else:
        print("  ❌ Missing image_metadata")

    # Test 3: Has material_classification
    if "material_classification" in response:
        mat = response['material_classification']
        print(f"  ✅ material_classification: {mat.get('material_type')}, {mat.get('confidence'):.1f}% confidence")
        tests_passed += 1
    else:
        print("  ❌ Missing material_classification")

    # Test 4: Material is Concrete (for our test file)
    if response.get("material_classification", {}).get("material_type") == "Concrete":
        print(f"  ✅ Correctly identified as Concrete")
        tests_passed += 1
    else:
        print(f"  ❌ Wrong material: {response.get('material_classification', {}).get('material_type')}")

    # Test 5: Has concrete_strength
    if "concrete_strength" in response:
        strength = response['concrete_strength']
        print(f"  ✅ concrete_strength: {strength.get('predicted_strength_mpa'):.1f} MPa")
        tests_passed += 1
    else:
        print("  ❌ Missing concrete_strength")

    # Test 6: Model R² is present and realistic
    model_r2 = response.get("concrete_strength", {}).get("model_r_squared", 0)
    if model_r2 > 0:
        print(f"  ✅ model_r_squared: {model_r2:.3f}")
        tests_passed += 1
    else:
        print(f"  ❌ Invalid model_r_squared: {model_r2}")

    # Test 7: Wavelength values are present
    wavelengths = response.get("concrete_strength", {}).get("key_wavelength_values", {})
    if wavelengths and wavelengths.get("cement_hydration_500_600", 0) > 0:
        print(f"  ✅ key_wavelength_values: {wavelengths}")
        tests_passed += 1
    else:
        print(f"  ❌ Missing or invalid wavelength values: {wavelengths}")

    # Test 8: Quality assessment is present
    if "quality_assessment" in response:
        qa = response['quality_assessment']
        print(f"  ✅ quality_assessment: {qa.get('overall_score')} ({qa.get('grade')})")
        tests_passed += 1
    else:
        print("  ❌ Missing quality_assessment")

    # Test deterministic behavior
    print("\n[DETERMINISTIC TEST] Running same file twice...")
    mock_upload2 = MockFile(sample_file.name, file_contents)
    response2 = await analyze_material_hyperspectral(file=mock_upload2)

    # Check key values are identical
    identical = (
        response['material_classification']['material_type'] == response2['material_classification']['material_type'] and
        abs(response['concrete_strength']['predicted_strength_mpa'] - response2['concrete_strength']['predicted_strength_mpa']) < 0.01
    )

    if identical:
        print("  ✅ Predictions are deterministic (identical results)")
    else:
        print("  ❌ Predictions are non-deterministic")
        print(f"    First:  {response['concrete_strength']['predicted_strength_mpa']}")
        print(f"    Second: {response2['concrete_strength']['predicted_strength_mpa']}")

    # Summary
    print("\n" + "=" * 70)
    print(f"RESULT: {tests_passed}/{tests_total} tests passed")
    print("=" * 70)

    if tests_passed == tests_total and identical:
        print("\n🎉 FRONTEND ↔ BACKEND INTEGRATION VERIFIED!")
        print("   Frontend can now use the ML-powered API endpoint")
        return True
    else:
        print(f"\n⚠️  {tests_total - tests_passed + (0 if identical else 1)} test(s) failed")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(test_frontend_backend_integration())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
