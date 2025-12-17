# ML Models Directory

**Status**: Model files excluded from git (too large)
**Training Required**: Yes, run training scripts to generate models

---

## Why Models Are Not Committed

ML model files (.pkl) are **excluded from git** because:
- ✅ Binary files are large (11MB total)
- ✅ Not suitable for version control
- ✅ Should be trained locally or stored in model registry
- ✅ Keeps repository size manageable

---

## How to Train Models

### Prerequisites

```bash
# Ensure you have the UMKC dataset
ls datasets/raw/hyperspectral/umkc-material-surfaces/

# Should show:
# Concrete/HSI_TIFF_50x50/ (34 files)
# Asphalt/HSI_TIFF_50x50/ (16 files)
```

### Training Commands

```bash
# Navigate to project root
cd /Users/nitishgautam/Code/prototype/ground-truth

# Train material classifier (100% accuracy)
python3 backend/app/ml/training/train_material_classifier.py

# Train quality regressors
python3 backend/app/ml/training/train_quality_regressor.py
```

### Expected Output

After training, you should have **5 model files** in this directory:

```
ml_artifacts/models/
├── material_classifier_v1.pkl     # ~5 MB - 100% accuracy
├── feature_scaler.pkl              # ~10 KB - Normalization
├── quality_regressor_v1.pkl        # ~2 MB - Quality scores
├── strength_regressor_v1.pkl       # ~2 MB - Strength prediction
├── confidence_regressor_v1.pkl     # ~2 MB - Confidence scores
├── training_metrics.json           # Committed to git ✓
└── quality_metrics.json            # Committed to git ✓
```

### Training Time

- **Material Classifier**: ~2 seconds (500 trees, 350 samples)
- **Quality Regressors**: ~1.5 seconds (3 models, 200 trees each)
- **Total**: ~3.5 seconds on modern laptop

---

## Verifying Models

### Check Models Exist

```bash
ls -lh ml_artifacts/models/*.pkl

# Should show 5 files totaling ~11 MB
```

### Run Validation Tests

```bash
python3 backend/app/ml/inference/test_predictor.py

# Expected output:
# ✅ Model Loading: PASS
# ✅ Inference: PASS (3/3 concrete, 3/3 asphalt correct)
# ✅ Deterministic: PASS (5/5 identical predictions)
# ✅ Ranges: PASS (all within expected bounds)
# ⏱️  Average inference time: 75.2ms
```

### Test in Platform

```bash
# Start platform with ML models
./START_HS2_WITH_ML.sh

# Should show:
# ✅ ML models found: 5 files
# ✅ Backend API is ready
# ✅ ML predictor loaded successfully
```

---

## Model Performance Metrics

### Material Classifier

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Accuracy** | 100% | 89% | ✅ **+11% over target** |
| **Cross-Validation** | 100% (5-fold) | >85% | ✅ Pass |
| **Inference Time** | <10ms | <100ms | ✅ Pass |

### Quality Regressors

| Model | R² | MAE | RMSE |
|-------|-----|-----|------|
| **Quality Score** | 1.0000 | 0.00 | 0.00 |
| **Strength** | 1.0000 | 0.00 MPa | 0.00 MPa |
| **Confidence** | 0.6531 | 0.06% | 0.09% |

**Note**: Perfect R²=1.0 for quality/strength is due to pseudo-labels (physics-based heuristics). Real-world performance will be validated with actual lab-tested ground truth data.

---

## Alternative: Download Pre-Trained Models

If you don't want to train locally, you can download pre-trained models:

### Option 1: From Team Member

```bash
# Ask team member to share ml_artifacts/models/ folder
# Copy to your project:
cp -r /path/to/shared/ml_artifacts/models/* ml_artifacts/models/
```

### Option 2: From Cloud Storage (Future)

```bash
# Download from cloud storage (AWS S3, Azure Blob, etc.)
# aws s3 sync s3://hs2-platform-models/v1/ ml_artifacts/models/
# az storage blob download-batch --source ml-models --destination ml_artifacts/models/
```

### Option 3: From Model Registry (Production)

```bash
# Download from MLflow, Weights & Biases, or similar
# mlflow artifacts download --run-id <run-id> --dst-path ml_artifacts/models/
```

---

## Troubleshooting

### Error: "FileNotFoundError: Model directory not found"

**Solution**: Train models first
```bash
python3 backend/app/ml/training/train_material_classifier.py
python3 backend/app/ml/training/train_quality_regressor.py
```

### Error: "ModuleNotFoundError: No module named 'tifffile'"

**Solution**: Install ML dependencies
```bash
pip install -r backend/requirements.txt
```

### Error: "Dataset not found"

**Solution**: Ensure UMKC dataset exists
```bash
# Dataset should be at:
datasets/raw/hyperspectral/umkc-material-surfaces/
```

---

## Model Version Control

For production, consider using:

1. **MLflow**: Track experiments and model versions
2. **DVC (Data Version Control)**: Git-like versioning for models
3. **Weights & Biases**: Experiment tracking and model registry
4. **Cloud Storage**: S3, Azure Blob, GCS with versioned buckets

**Current approach**: Local training, git tracks code + metrics only

---

## Documentation

For complete ML implementation details, see:
- **[ML_TECHNICAL_GUIDE.md](../../docs/ML_TECHNICAL_GUIDE.md)** - Comprehensive ML guide
- **Training scripts**: `backend/app/ml/training/`
- **Inference code**: `backend/app/ml/inference/`
- **API integration**: `backend/app/api/v1/endpoints/hyperspectral.py`

---

**Last Updated**: December 2025
**Model Version**: v1.0
**Status**: Training required (models not committed to git)
