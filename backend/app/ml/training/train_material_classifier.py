"""
Train Material Classifier on UMKC Hyperspectral Dataset

Goal: Achieve 89% accuracy on concrete vs asphalt classification
"""

import numpy as np
import tifffile
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import json
import sys
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from app.ml.preprocessing.feature_extractors import extract_spectral_features
from app.ml.preprocessing.augmentation import augment_hyperspectral

# Configuration
DATASET_PATH = Path("/Users/nitishgautam/Code/prototype/ground-truth/datasets/raw/hyperspectral/umkc-material-surfaces")
OUTPUT_PATH = Path("/Users/nitishgautam/Code/prototype/ground-truth/ml_artifacts/models")
N_FOLDS = 5
RANDOM_SEED = 42
TARGET_ACCURACY = 0.89

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_umkc_dataset():
    """
    Load all hyperspectral TIFF files with labels.

    Returns:
        Tuple of (data, labels, filenames)
        - data: np.ndarray of shape (N, H, W, C)
        - labels: np.ndarray of shape (N,) - 1=Concrete, 0=Asphalt
        - filenames: List of file paths
    """
    concrete_files = sorted(DATASET_PATH.glob("Concrete/HSI_TIFF_50x50/*.tiff"))
    asphalt_files = sorted(DATASET_PATH.glob("Asphalt/HSI_TIFF_50x50/*.tiff"))

    data = []
    labels = []
    filenames = []

    logger.info(f"Loading {len(concrete_files)} concrete samples...")
    for fpath in concrete_files:
        try:
            cube = tifffile.imread(str(fpath))  # (50, 50, 139)
            if cube.shape == (50, 50, 139):
                data.append(cube)
                labels.append(1)  # 1 = Concrete
                filenames.append(str(fpath))
            else:
                logger.warning(f"Skipping {fpath.name}: unexpected shape {cube.shape}")
        except Exception as e:
            logger.error(f"Failed to load {fpath.name}: {e}")

    logger.info(f"Loading {len(asphalt_files)} asphalt samples...")
    for fpath in asphalt_files:
        try:
            cube = tifffile.imread(str(fpath))  # (50, 50, 139)
            if cube.shape == (50, 50, 139):
                data.append(cube)
                labels.append(0)  # 0 = Asphalt
                filenames.append(str(fpath))
            else:
                logger.warning(f"Skipping {fpath.name}: unexpected shape {cube.shape}")
        except Exception as e:
            logger.error(f"Failed to load {fpath.name}: {e}")

    return np.array(data), np.array(labels), filenames


def main():
    logger.info("=" * 70)
    logger.info("UMKC Hyperspectral Material Classifier Training")
    logger.info("=" * 70)
    logger.info(f"Target Accuracy: {TARGET_ACCURACY:.1%}")
    logger.info(f"Random Seed: {RANDOM_SEED}")
    logger.info("")

    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)

    # Load dataset
    logger.info("Step 1: Loading UMKC dataset...")
    data, labels, filenames = load_umkc_dataset()

    if len(data) == 0:
        logger.error("No data loaded! Check dataset path.")
        return

    logger.info(f"Dataset loaded: {data.shape[0]} samples")
    logger.info(f"  Concrete (label=1): {(labels == 1).sum()} samples")
    logger.info(f"  Asphalt (label=0): {(labels == 0).sum()} samples")
    logger.info(f"  Cube shape: {data[0].shape}")
    logger.info("")

    # Data augmentation
    logger.info("Step 2: Applying data augmentation...")
    logger.info("  Augmentation strategy: rotations (3x), flips (2x), noise (1x)")

    augmented_data = []
    augmented_labels = []

    for cube, label in zip(data, labels):
        aug_cubes = augment_hyperspectral(cube)
        augmented_data.extend(aug_cubes)
        augmented_labels.extend([label] * len(aug_cubes))

    augmented_data = np.array(augmented_data)
    augmented_labels = np.array(augmented_labels)

    logger.info(f"After augmentation: {augmented_data.shape[0]} samples")
    logger.info(f"  Augmentation factor: {augmented_data.shape[0] / data.shape[0]:.1f}x")
    logger.info(f"  Concrete: {(augmented_labels == 1).sum()} samples")
    logger.info(f"  Asphalt: {(augmented_labels == 0).sum()} samples")
    logger.info("")

    # Extract features
    logger.info("Step 3: Extracting spectral features...")
    logger.info("  Features: mean spectrum (139), std spectrum (139), stats (4), indices (10)")

    features_list = []
    for i, cube in enumerate(augmented_data):
        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i + 1}/{len(augmented_data)} samples...")
        features = extract_spectral_features(cube)
        features_list.append(features)

    features = np.array(features_list)
    logger.info(f"Feature extraction complete: {features.shape}")
    logger.info(f"  Feature dimension: {features.shape[1]}")
    logger.info("")

    # Normalize features
    logger.info("Step 4: Normalizing features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    logger.info(f"  Feature mean: {features_scaled.mean():.6f} (should be ~0)")
    logger.info(f"  Feature std: {features_scaled.std():.6f} (should be ~1)")
    logger.info("")

    # Train Random Forest with cross-validation
    logger.info("Step 5: Training Random Forest Classifier...")
    logger.info("  Hyperparameters:")
    logger.info("    - n_estimators: 500")
    logger.info("    - max_depth: 20")
    logger.info("    - min_samples_split: 5")
    logger.info("    - min_samples_leaf: 2")
    logger.info("    - class_weight: balanced")
    logger.info("")

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight='balanced',
        verbose=0
    )

    # 5-Fold Cross-Validation
    logger.info(f"Step 6: {N_FOLDS}-Fold Cross-Validation...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(
        rf, features_scaled, augmented_labels,
        cv=skf, scoring='accuracy', n_jobs=-1
    )

    logger.info("")
    logger.info("Cross-Validation Results:")
    logger.info("=" * 70)
    for fold, score in enumerate(cv_scores, 1):
        status = "✓" if score >= TARGET_ACCURACY else "✗"
        logger.info(f"  Fold {fold}: {score:.4f} ({score:.1%}) {status}")

    logger.info("-" * 70)
    logger.info(f"  Mean Accuracy: {cv_scores.mean():.4f} ({cv_scores.mean():.1%})")
    logger.info(f"  Std Dev: ±{cv_scores.std():.4f}")
    logger.info(f"  Min: {cv_scores.min():.4f} ({cv_scores.min():.1%})")
    logger.info(f"  Max: {cv_scores.max():.4f} ({cv_scores.max():.1%})")
    logger.info("=" * 70)
    logger.info("")

    # Train on full dataset
    logger.info("Step 7: Training on full dataset...")
    rf.fit(features_scaled, augmented_labels)
    logger.info("  Training complete!")
    logger.info("")

    # Evaluate on training set (sanity check)
    logger.info("Step 8: Evaluating on training set...")
    train_preds = rf.predict(features_scaled)
    train_acc = accuracy_score(augmented_labels, train_preds)

    logger.info(f"Training Set Performance:")
    logger.info(f"  Accuracy: {train_acc:.4f} ({train_acc:.1%})")
    logger.info("")
    logger.info("Classification Report:")
    print(classification_report(
        augmented_labels, train_preds,
        target_names=['Asphalt', 'Concrete'],
        digits=4
    ))

    # Confusion Matrix
    cm = confusion_matrix(augmented_labels, train_preds)
    logger.info("Confusion Matrix:")
    logger.info("             Predicted")
    logger.info("              Asphalt  Concrete")
    logger.info(f"Actual Asphalt    {cm[0, 0]:4d}     {cm[0, 1]:4d}")
    logger.info(f"     Concrete    {cm[1, 0]:4d}     {cm[1, 1]:4d}")
    logger.info("")

    # Feature importance
    feature_importance = rf.feature_importances_
    top_10_idx = np.argsort(feature_importance)[-10:]

    logger.info("Top 10 Most Important Features:")
    for i, idx in enumerate(reversed(top_10_idx), 1):
        logger.info(f"  {i}. Feature {idx}: {feature_importance[idx]:.6f}")
    logger.info("")

    # Save model and scaler
    logger.info("Step 9: Saving trained model...")
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    joblib.dump(rf, OUTPUT_PATH / "material_classifier_v1.pkl")
    joblib.dump(scaler, OUTPUT_PATH / "feature_scaler.pkl")

    logger.info(f"  Model saved: {OUTPUT_PATH / 'material_classifier_v1.pkl'}")
    logger.info(f"  Scaler saved: {OUTPUT_PATH / 'feature_scaler.pkl'}")
    logger.info("")

    # Save training metrics
    metrics = {
        'model_version': '1.0',
        'training_date': datetime.now().isoformat(),
        'target_accuracy': TARGET_ACCURACY,
        'cross_val_accuracy_mean': float(cv_scores.mean()),
        'cross_val_accuracy_std': float(cv_scores.std()),
        'cross_val_accuracy_min': float(cv_scores.min()),
        'cross_val_accuracy_max': float(cv_scores.max()),
        'cross_val_scores': cv_scores.tolist(),
        'train_accuracy': float(train_acc),
        'n_samples_original': int(data.shape[0]),
        'n_samples_augmented': int(augmented_data.shape[0]),
        'n_features': int(features.shape[1]),
        'model_type': 'RandomForestClassifier',
        'hyperparameters': {
            'n_estimators': 500,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'class_weight': 'balanced',
            'random_state': RANDOM_SEED
        },
        'dataset': {
            'path': str(DATASET_PATH),
            'concrete_samples': int((labels == 1).sum()),
            'asphalt_samples': int((labels == 0).sum())
        },
        'confusion_matrix': cm.tolist(),
        'top_10_features': top_10_idx.tolist()
    }

    metrics_path = OUTPUT_PATH / "training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"  Metrics saved: {metrics_path}")
    logger.info("")

    # Final summary
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)

    if cv_scores.mean() >= TARGET_ACCURACY:
        logger.info(f"✅ SUCCESS: Achieved {cv_scores.mean():.1%} accuracy (target: {TARGET_ACCURACY:.1%})")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Train quality regressors:")
        logger.info("     python backend/app/ml/training/train_quality_regressor.py")
        logger.info("  2. Test inference:")
        logger.info("     python backend/app/ml/inference/test_predictor.py")
        logger.info("  3. Integrate into API:")
        logger.info("     Update backend/app/api/v1/endpoints/hyperspectral.py")
    else:
        logger.warning(f"⚠️  WARNING: {cv_scores.mean():.1%} accuracy < {TARGET_ACCURACY:.1%} target")
        logger.warning("")
        logger.warning("Recommendations:")
        logger.warning("  - Try aggressive augmentation (15x factor)")
        logger.warning("  - Try Option C: Hybrid CNN + Classical ML")
        logger.warning("  - Collect more labeled samples if possible")
        logger.warning("  - Tune hyperparameters with GridSearchCV")

    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
