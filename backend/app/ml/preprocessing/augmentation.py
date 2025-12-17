"""
Data Augmentation for Hyperspectral Images

Provides augmentation strategies to increase training data from limited samples.
"""

import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


def augment_hyperspectral(cube: np.ndarray, noise_level: float = 0.05) -> List[np.ndarray]:
    """
    Augment hyperspectral cube with geometric and spectral transformations.

    Augmentations:
    - Original (1x)
    - Rotations: 90°, 180°, 270° (3x)
    - Flips: horizontal, vertical (2x)
    - Spectral noise (1x)

    Total: 7x augmentation factor

    Args:
        cube: Hyperspectral cube of shape (H, W, C)
        noise_level: Gaussian noise level as fraction of std dev (default: 5%)

    Returns:
        List of 7 augmented cubes
    """
    augmented = []

    # Original
    augmented.append(cube.copy())

    # Rotations (90°, 180°, 270°)
    augmented.append(np.rot90(cube, k=1, axes=(0, 1)))
    augmented.append(np.rot90(cube, k=2, axes=(0, 1)))
    augmented.append(np.rot90(cube, k=3, axes=(0, 1)))

    # Flips
    augmented.append(np.flip(cube, axis=0))  # Vertical flip
    augmented.append(np.flip(cube, axis=1))  # Horizontal flip

    # Spectral noise (Gaussian)
    cube_std = cube.std()
    noisy = cube + np.random.normal(0, cube_std * noise_level, cube.shape)

    # Clip to valid range (assuming uint16 range)
    if cube.dtype == np.uint16:
        noisy = np.clip(noisy, 0, 65535).astype(np.uint16)
    else:
        noisy = np.clip(noisy, cube.min(), cube.max())

    augmented.append(noisy)

    logger.debug(f"Augmented 1 sample into {len(augmented)} samples")

    return augmented


def augment_aggressive(cube: np.ndarray) -> List[np.ndarray]:
    """
    More aggressive augmentation strategy for very small datasets.

    Adds:
    - Diagonal flips
    - Multiple noise levels
    - Brightness adjustments
    - Spectral band shifts

    Total: ~15x augmentation factor

    Args:
        cube: Hyperspectral cube of shape (H, W, C)

    Returns:
        List of ~15 augmented cubes
    """
    augmented = augment_hyperspectral(cube)  # Start with 7 basic augmentations

    # Diagonal flips
    augmented.append(np.flip(cube, axis=(0, 1)))  # Both axes

    # Multiple noise levels
    for noise in [0.03, 0.07]:
        cube_std = cube.std()
        noisy = cube + np.random.normal(0, cube_std * noise, cube.shape)
        if cube.dtype == np.uint16:
            noisy = np.clip(noisy, 0, 65535).astype(np.uint16)
        augmented.append(noisy)

    # Brightness adjustments (simulate lighting variations)
    for brightness_factor in [0.9, 1.1]:
        bright = cube * brightness_factor
        if cube.dtype == np.uint16:
            bright = np.clip(bright, 0, 65535).astype(np.uint16)
        augmented.append(bright)

    # Spectral band shift (simulate sensor calibration variations)
    # Shift all bands by ±5% of mean value
    for shift_factor in [-0.05, 0.05]:
        shifted = cube + (cube.mean() * shift_factor)
        if cube.dtype == np.uint16:
            shifted = np.clip(shifted, 0, 65535).astype(np.uint16)
        augmented.append(shifted)

    logger.debug(f"Aggressive augmentation: 1 → {len(augmented)} samples")

    return augmented


def create_patches(cube: np.ndarray, patch_size: int = 25, stride: int = 10) -> List[np.ndarray]:
    """
    Extract overlapping patches from hyperspectral cube.

    Useful for increasing dataset size from single large images.

    Args:
        cube: Hyperspectral cube of shape (H, W, C)
        patch_size: Size of square patches (default: 25x25)
        stride: Stride between patches (default: 10, giving overlap)

    Returns:
        List of patches, each of shape (patch_size, patch_size, C)
    """
    H, W, C = cube.shape
    patches = []

    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch = cube[i:i + patch_size, j:j + patch_size, :]
            patches.append(patch)

    logger.debug(f"Extracted {len(patches)} patches of size {patch_size}x{patch_size} from {H}x{W} cube")

    return patches


def balance_dataset(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Balance dataset by oversampling minority class.

    Args:
        X: Feature array of shape (N, D)
        y: Label array of shape (N,)

    Returns:
        Tuple of (X_balanced, y_balanced)
    """
    unique, counts = np.unique(y, return_counts=True)

    if len(unique) != 2:
        logger.warning(f"Expected binary classification, got {len(unique)} classes")
        return X, y

    majority_class = unique[np.argmax(counts)]
    minority_class = unique[np.argmin(counts)]

    majority_count = counts.max()
    minority_count = counts.min()

    logger.info(f"Class imbalance: {majority_class}={majority_count}, {minority_class}={minority_count}")

    if majority_count == minority_count:
        logger.info("Dataset already balanced")
        return X, y

    # Oversample minority class
    minority_indices = np.where(y == minority_class)[0]
    majority_indices = np.where(y == majority_class)[0]

    # Randomly sample minority class with replacement
    oversample_indices = np.random.choice(
        minority_indices,
        size=majority_count - minority_count,
        replace=True
    )

    # Combine
    balanced_indices = np.concatenate([majority_indices, minority_indices, oversample_indices])
    np.random.shuffle(balanced_indices)

    X_balanced = X[balanced_indices]
    y_balanced = y[balanced_indices]

    logger.info(f"Balanced dataset: {len(y_balanced)} samples")

    return X_balanced, y_balanced
