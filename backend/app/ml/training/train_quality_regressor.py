"""
Train Quality Regressors on UMKC Concrete Samples

Trains separate models for:
- Quality score (82-96%)
- Strength prediction (28-48 MPa)
- Confidence score (94-98.9%)

Uses physics-based pseudo-labels since ground truth is unavailable.
"""

import numpy as np
import tifffile
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import json
import sys
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from app.ml.preprocessing.feature_extractors import (
    extract_spectral_features,
    estimate_quality_heuristic
)

# Configuration
DATASET_PATH = Path("/Users/nitishgautam/Code/prototype/ground-truth/datasets/raw/hyperspectral/umkc-material-surfaces")
OUTPUT_PATH = Path("/Users/nitishgautam/Code/prototype/ground-truth/ml_artifacts/models")
RANDOM_SEED = 42
N_FOLDS = 5

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_concrete_samples():
    """
    Load concrete samples and generate pseudo-labels using heuristics.

    Returns:
        Tuple of (data, quality_labels, strength_labels, confidence_labels, filenames)
    """
    concrete_files = sorted(DATASET_PATH.glob("Concrete/HSI_TIFF_50x50/*.tiff"))

    data = []
    quality_labels = []
    strength_labels = []
    confidence_labels = []
    filenames = []

    logger.info(f"Loading {len(concrete_files)} concrete samples...")

    for fpath in concrete_files:
        try:
            cube = tifffile.imread(str(fpath))

            if cube.shape != (50, 50, 139):
                logger.warning(f"Skipping {fpath.name}: unexpected shape {cube.shape}")
                continue

            # Generate pseudo-labels using physics-based heuristics
            quality, strength, confidence = estimate_quality_heuristic(cube)

            data.append(cube)
            quality_labels.append(quality)
            strength_labels.append(strength)
            confidence_labels.append(confidence)
            filenames.append(str(fpath))

        except Exception as e:
            logger.error(f"Failed to load {fpath.name}: {e}")

    return (
        np.array(data),
        np.array(quality_labels),
        np.array(strength_labels),
        np.array(confidence_labels),
        filenames
    )


def main():
    logger.info("=" * 70)
    logger.info("UMKC Quality Regressor Training (Concrete Only)")
    logger.info("=" * 70)
    logger.info("⚠️  Using PSEUDO-LABELS from physics-based heuristics")
    logger.info("   For production accuracy, obtain real lab-tested measurements")
    logger.info("")

    # Set random seed
    np.random.seed(RANDOM_SEED)

    # Load concrete samples
    logger.info("Step 1: Loading concrete samples...")
    data, quality_labels, strength_labels, confidence_labels, filenames = load_concrete_samples()

    if len(data) == 0:
        logger.error("No concrete samples loaded!")
        return

    logger.info(f"Loaded {len(data)} concrete samples")
    logger.info(f"  Cube shape: {data[0].shape}")
    logger.info("")

    # Display label statistics
    logger.info("Pseudo-Label Statistics:")
    logger.info(f"  Quality Score:")
    logger.info(f"    Range: {quality_labels.min():.1f} - {quality_labels.max():.1f}")
    logger.info(f"    Mean: {quality_labels.mean():.1f} ± {quality_labels.std():.1f}")
    logger.info(f"  Strength (MPa):")
    logger.info(f"    Range: {strength_labels.min():.1f} - {strength_labels.max():.1f}")
    logger.info(f"    Mean: {strength_labels.mean():.1f} ± {strength_labels.std():.1f}")
    logger.info(f"  Confidence (%):")
    logger.info(f"    Range: {confidence_labels.min():.1f} - {confidence_labels.max():.1f}")
    logger.info(f"    Mean: {confidence_labels.mean():.1f} ± {confidence_labels.std():.1f}")
    logger.info("")

    # Extract features
    logger.info("Step 2: Extracting spectral features...")
    features_list = []
    for i, cube in enumerate(data):
        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i + 1}/{len(data)} samples...")
        features = extract_spectral_features(cube)
        features_list.append(features)

    features = np.array(features_list)
    logger.info(f"Feature extraction complete: {features.shape}")
    logger.info("")

    # Normalize features
    logger.info("Step 3: Normalizing features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    logger.info(f"  Feature mean: {features_scaled.mean():.6f}")
    logger.info(f"  Feature std: {features_scaled.std():.6f}")
    logger.info("")

    # Train three separate regressors
    logger.info("Step 4: Training Quality Score Regressor...")
    quality_regressor = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    quality_regressor.fit(features_scaled, quality_labels)

    # Cross-validation for quality
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    quality_cv_scores = []
    for train_idx, test_idx in kf.split(features_scaled):
        X_train, X_test = features_scaled[train_idx], features_scaled[test_idx]
        y_train, y_test = quality_labels[train_idx], quality_labels[test_idx]

        temp_model = RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1
        )
        temp_model.fit(X_train, y_train)
        y_pred = temp_model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        quality_cv_scores.append(r2)

    logger.info(f"  Quality R²: {np.mean(quality_cv_scores):.4f} (±{np.std(quality_cv_scores):.4f})")
    logger.info("")

    logger.info("Step 5: Training Strength Regressor...")
    strength_regressor = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    strength_regressor.fit(features_scaled, strength_labels)

    # Cross-validation for strength
    strength_cv_scores = []
    for train_idx, test_idx in kf.split(features_scaled):
        X_train, X_test = features_scaled[train_idx], features_scaled[test_idx]
        y_train, y_test = strength_labels[train_idx], strength_labels[test_idx]

        temp_model = RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1
        )
        temp_model.fit(X_train, y_train)
        y_pred = temp_model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        strength_cv_scores.append(r2)

    logger.info(f"  Strength R²: {np.mean(strength_cv_scores):.4f} (±{np.std(strength_cv_scores):.4f})")
    logger.info("")

    logger.info("Step 6: Training Confidence Regressor...")
    confidence_regressor = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    confidence_regressor.fit(features_scaled, confidence_labels)

    # Cross-validation for confidence
    confidence_cv_scores = []
    for train_idx, test_idx in kf.split(features_scaled):
        X_train, X_test = features_scaled[train_idx], features_scaled[test_idx]
        y_train, y_test = confidence_labels[train_idx], confidence_labels[test_idx]

        temp_model = RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1
        )
        temp_model.fit(X_train, y_train)
        y_pred = temp_model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        confidence_cv_scores.append(r2)

    logger.info(f"  Confidence R²: {np.mean(confidence_cv_scores):.4f} (±{np.std(confidence_cv_scores):.4f})")
    logger.info("")

    # Training set predictions (sanity check)
    logger.info("Step 7: Evaluating on training set...")

    quality_pred = quality_regressor.predict(features_scaled)
    strength_pred = strength_regressor.predict(features_scaled)
    confidence_pred = confidence_regressor.predict(features_scaled)

    logger.info("Training Set Performance:")
    logger.info(f"  Quality Score:")
    logger.info(f"    MAE: {mean_absolute_error(quality_labels, quality_pred):.2f}")
    logger.info(f"    RMSE: {np.sqrt(mean_squared_error(quality_labels, quality_pred)):.2f}")
    logger.info(f"    R²: {r2_score(quality_labels, quality_pred):.4f}")

    logger.info(f"  Strength:")
    logger.info(f"    MAE: {mean_absolute_error(strength_labels, strength_pred):.2f} MPa")
    logger.info(f"    RMSE: {np.sqrt(mean_squared_error(strength_labels, strength_pred)):.2f} MPa")
    logger.info(f"    R²: {r2_score(strength_labels, strength_pred):.4f}")

    logger.info(f"  Confidence:")
    logger.info(f"    MAE: {mean_absolute_error(confidence_labels, confidence_pred):.2f}%")
    logger.info(f"    RMSE: {np.sqrt(mean_squared_error(confidence_labels, confidence_pred)):.2f}%")
    logger.info(f"    R²: {r2_score(confidence_labels, confidence_pred):.4f}")
    logger.info("")

    # Save models
    logger.info("Step 8: Saving trained models...")
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    joblib.dump(quality_regressor, OUTPUT_PATH / "quality_regressor_v1.pkl")
    joblib.dump(strength_regressor, OUTPUT_PATH / "strength_regressor_v1.pkl")
    joblib.dump(confidence_regressor, OUTPUT_PATH / "confidence_regressor_v1.pkl")

    logger.info(f"  Quality regressor: {OUTPUT_PATH / 'quality_regressor_v1.pkl'}")
    logger.info(f"  Strength regressor: {OUTPUT_PATH / 'strength_regressor_v1.pkl'}")
    logger.info(f"  Confidence regressor: {OUTPUT_PATH / 'confidence_regressor_v1.pkl'}")
    logger.info("")

    # Save metrics
    metrics = {
        'model_version': '1.0',
        'training_date': datetime.now().isoformat(),
        'pseudo_labels': True,
        'pseudo_label_method': 'physics_based_heuristics',
        'n_samples': int(len(data)),
        'n_features': int(features.shape[1]),
        'model_type': 'RandomForestRegressor',
        'hyperparameters': {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': RANDOM_SEED
        },
        'cross_validation': {
            'quality_r2_mean': float(np.mean(quality_cv_scores)),
            'quality_r2_std': float(np.std(quality_cv_scores)),
            'strength_r2_mean': float(np.mean(strength_cv_scores)),
            'strength_r2_std': float(np.std(strength_cv_scores)),
            'confidence_r2_mean': float(np.mean(confidence_cv_scores)),
            'confidence_r2_std': float(np.std(confidence_cv_scores))
        },
        'training_performance': {
            'quality': {
                'mae': float(mean_absolute_error(quality_labels, quality_pred)),
                'rmse': float(np.sqrt(mean_squared_error(quality_labels, quality_pred))),
                'r2': float(r2_score(quality_labels, quality_pred))
            },
            'strength': {
                'mae': float(mean_absolute_error(strength_labels, strength_pred)),
                'rmse': float(np.sqrt(mean_squared_error(strength_labels, strength_pred))),
                'r2': float(r2_score(strength_labels, strength_pred))
            },
            'confidence': {
                'mae': float(mean_absolute_error(confidence_labels, confidence_pred)),
                'rmse': float(np.sqrt(mean_squared_error(confidence_labels, confidence_pred))),
                'r2': float(r2_score(confidence_labels, confidence_pred))
            }
        }
    }

    metrics_path = OUTPUT_PATH / "quality_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"  Metrics saved: {metrics_path}")
    logger.info("")

    # Final summary
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info("✅ Quality regressors successfully trained!")
    logger.info("")
    logger.info("⚠️  IMPORTANT NOTES:")
    logger.info("  1. These models use PSEUDO-LABELS (not real ground truth)")
    logger.info("  2. Predictions are based on physics-based heuristics")
    logger.info("  3. For production accuracy, obtain lab-tested strength measurements")
    logger.info("  4. Current models provide reasonable estimates within expected ranges")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Test inference:")
    logger.info("     python backend/app/ml/inference/test_predictor.py")
    logger.info("  2. Integrate into API:")
    logger.info("     Update backend/app/api/v1/endpoints/hyperspectral.py")
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
