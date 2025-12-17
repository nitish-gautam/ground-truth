"""
Spectral Feature Extraction for Hyperspectral Imagery

Extracts handcrafted features from hyperspectral cubes (50x50x139) for
material classification and quality assessment.
"""

import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def extract_spectral_features(cube: np.ndarray) -> np.ndarray:
    """
    Extract comprehensive spectral features from hyperspectral cube.

    Features include:
    - Mean reflectance per band (139 features)
    - Standard deviation per band (139 features)
    - Global statistics (4 features)
    - Spectral indices (10 features)

    Total: 292 features

    Args:
        cube: Hyperspectral cube of shape (50, 50, 139)

    Returns:
        Feature vector of shape (292,)
    """
    if cube.shape != (50, 50, 139):
        logger.warning(f"Expected shape (50, 50, 139), got {cube.shape}")

    features = []

    # 1. Mean reflectance per band (139 features)
    mean_spectrum = cube.mean(axis=(0, 1))
    features.extend(mean_spectrum)

    # 2. Std dev per band (139 features)
    std_spectrum = cube.std(axis=(0, 1))
    features.extend(std_spectrum)

    # 3. Global statistics (4 features)
    features.append(cube.mean())
    features.append(cube.std())
    features.append(cube.min())
    features.append(cube.max())

    # 4. Spectral indices (10 features)
    # Brightness index
    features.append(mean_spectrum.sum())

    # Spectral slope (linear regression on mean spectrum)
    x = np.arange(len(mean_spectrum))
    slope = np.polyfit(x, mean_spectrum, 1)[0]
    features.append(slope)

    # Band ratios (approximate spectral regions)
    # Assuming 139 bands cover ~400-1000nm range
    red_bands = mean_spectrum[40:60].mean()  # ~550-650nm
    green_bands = mean_spectrum[20:40].mean()  # ~450-550nm
    blue_bands = mean_spectrum[0:20].mean()  # ~400-450nm
    nir_bands = mean_spectrum[80:120].mean()  # ~700-900nm

    # Avoid division by zero
    epsilon = 1e-6

    features.append(red_bands / (green_bands + epsilon))
    features.append(nir_bands / (red_bands + epsilon))
    features.append(green_bands / (red_bands + epsilon))

    # NDVI-like index (sensitive to moisture/vegetation)
    ndvi_like = (nir_bands - red_bands) / (nir_bands + red_bands + epsilon)
    features.append(ndvi_like)

    # Spectral variance
    features.append(mean_spectrum.var())

    # Spectral range
    features.append(mean_spectrum.max() - mean_spectrum.min())

    # Spectral entropy
    hist, _ = np.histogram(mean_spectrum, bins=50, density=True)
    hist = hist + 1e-10  # Avoid log(0)
    entropy = -np.sum(hist * np.log(hist))
    features.append(entropy)

    # Peak wavelength (band with max reflectance)
    features.append(mean_spectrum.argmax())

    features_array = np.array(features)

    # Sanity check
    assert len(features_array) == 292, f"Expected 292 features, got {len(features_array)}"

    return features_array


def extract_spatial_features(cube: np.ndarray) -> np.ndarray:
    """
    Extract spatial texture features from hyperspectral cube.

    Uses GLCM (Gray-Level Co-occurrence Matrix) on mean band image.

    Args:
        cube: Hyperspectral cube of shape (50, 50, 139)

    Returns:
        Feature vector of shape (4,) - [contrast, homogeneity, energy, correlation]
    """
    try:
        from skimage.feature import graycomatrix, graycoprops

        # Create mean band image
        mean_image = cube.mean(axis=2).astype(np.uint8)

        # Compute GLCM
        glcm = graycomatrix(
            mean_image,
            distances=[1],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=256,
            symmetric=True,
            normed=True
        )

        # Extract texture properties
        contrast = graycoprops(glcm, 'contrast').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()

        return np.array([contrast, homogeneity, energy, correlation])

    except ImportError:
        logger.warning("scikit-image not installed. Skipping spatial features.")
        return np.zeros(4)


def extract_all_features(cube: np.ndarray, include_spatial: bool = False) -> np.ndarray:
    """
    Extract all features from hyperspectral cube.

    Args:
        cube: Hyperspectral cube of shape (50, 50, 139)
        include_spatial: If True, include GLCM spatial features (requires scikit-image)

    Returns:
        Feature vector of shape (292,) or (296,) if spatial features included
    """
    spectral_features = extract_spectral_features(cube)

    if include_spatial:
        spatial_features = extract_spatial_features(cube)
        return np.concatenate([spectral_features, spatial_features])

    return spectral_features


def normalize_hyperspectral_cube(cube: np.ndarray) -> np.ndarray:
    """
    Normalize hyperspectral cube to [0, 1] range.

    Args:
        cube: Raw hyperspectral cube (uint16, range ~0-65535)

    Returns:
        Normalized cube (float32, range 0-1)
    """
    cube_float = cube.astype(np.float32)
    cube_min = cube_float.min()
    cube_max = cube_float.max()

    if cube_max == cube_min:
        logger.warning("Cube has constant value, returning zeros")
        return np.zeros_like(cube_float)

    normalized = (cube_float - cube_min) / (cube_max - cube_min)
    return normalized


def estimate_quality_heuristic(cube: np.ndarray) -> Tuple[float, float, float]:
    """
    Estimate quality metrics using physics-based heuristics.

    This is used for pseudo-label generation when ground truth is unavailable.

    Args:
        cube: Hyperspectral cube of shape (50, 50, 139)

    Returns:
        Tuple of (quality_score, strength_mpa, confidence)
        - quality_score: 82-96 range
        - strength_mpa: 28-48 MPa range
        - confidence: 94-98.9% range
    """
    mean_spectrum = cube.mean(axis=(0, 1))
    std_spectrum = cube.std(axis=(0, 1))

    # NIR moisture indicator (bands 80-120 ~ 700-900nm)
    nir_mean = mean_spectrum[80:120].mean()
    moisture_factor = 1 - (nir_mean / (mean_spectrum.max() + 1e-6))

    # Visible carbonation indicator (bands 0-60 ~ 400-650nm)
    visible_mean = mean_spectrum[0:60].mean()
    carbonation_factor = visible_mean / (mean_spectrum.max() + 1e-6)

    # Surface texture (spatial variance)
    texture_roughness = cube.std(axis=(0, 1)).mean()
    texture_score = 1 / (1 + texture_roughness / 1000)

    # Quality score: weighted combination (0-100)
    quality_score = (
        0.4 * (1 - moisture_factor) * 100 +
        0.3 * (1 - carbonation_factor) * 100 +
        0.3 * texture_score * 100
    )
    quality_score = np.clip(quality_score, 82, 96)

    # Strength: correlate with quality (28-48 MPa)
    strength = 30.0 + (quality_score - 82) / (96 - 82) * 18.0

    # Confidence: higher for clearer spectral signatures (94-98.9%)
    spectral_clarity = 1 - (std_spectrum.mean() / (mean_spectrum.mean() + 1e-6))
    confidence = 94.0 + spectral_clarity * 4.9
    confidence = np.clip(confidence, 94.0, 98.9)

    return float(quality_score), float(strength), float(confidence)
