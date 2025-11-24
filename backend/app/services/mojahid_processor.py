"""
Mojahid image dataset processing service
=======================================

Comprehensive service for processing Mojahid GPR images with feature extraction,
classification, and integration with the ML pipeline.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from sklearn.feature_extraction import image as sklearn_image
from sklearn.preprocessing import StandardScaler
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..core.config import settings
from ..core.logging_config import LoggerMixin
from ..models.gpr_data import GPRSurvey, GPRScan
from ..models.ml_analytics import FeatureVector, MLModel
from ..schemas.dataset import MojahidDatasetStatus


class MojahidImageProcessor(LoggerMixin):
    """Enhanced processor for Mojahid GPR images dataset."""

    def __init__(self):
        super().__init__()
        self.dataset_path = settings.GPR_MOJAHID_PATH / "GPR_data"
        self.feature_extractors = self._initialize_feature_extractors()

    def _initialize_feature_extractors(self) -> Dict[str, Any]:
        """Initialize feature extraction methods."""
        return {
            "statistical": self._extract_statistical_features,
            "texture": self._extract_texture_features,
            "edge": self._extract_edge_features,
            "histogram": self._extract_histogram_features,
            "spatial": self._extract_spatial_features
        }

    async def process_category_comprehensive(
        self,
        db: AsyncSession,
        category: str,
        max_images: Optional[int] = None,
        extract_features: bool = True,
        create_augmented: bool = False
    ) -> Dict[str, Any]:
        """Comprehensively process images from a specific category."""
        self.log_operation_start("process_mojahid_category", category=category, max_images=max_images)

        try:
            category_path = self.dataset_path / category
            if not category_path.exists():
                raise FileNotFoundError(f"Category path not found: {category_path}")

            # Get image files
            image_files = self._get_image_files(category_path, max_images)
            self.log_data_processing(f"mojahid_{category}", len(image_files))

            # Create survey for this category
            survey = await self._create_or_get_survey(db, category)

            results = {
                "category": category,
                "total_images": len(image_files),
                "processed_images": 0,
                "features_extracted": 0,
                "augmented_created": 0,
                "errors": []
            }

            # Process images in batches
            batch_size = 50
            for i in range(0, len(image_files), batch_size):
                batch = image_files[i:i + batch_size]
                batch_results = await self._process_image_batch(
                    db, survey.id, batch, category, extract_features, create_augmented
                )

                results["processed_images"] += batch_results["processed"]
                results["features_extracted"] += batch_results["features_extracted"]
                results["augmented_created"] += batch_results["augmented_created"]
                results["errors"].extend(batch_results["errors"])

                self.log_batch_processing(
                    batch_id=f"mojahid_{category}_batch_{i//batch_size + 1}",
                    total_items=len(image_files),
                    processed_items=min(i + batch_size, len(image_files))
                )

            await db.commit()
            self.log_operation_complete("process_mojahid_category", 0, **results)

            return results

        except Exception as e:
            await db.rollback()
            self.log_operation_error("process_mojahid_category", e, category=category)
            raise

    async def _process_image_batch(
        self,
        db: AsyncSession,
        survey_id: str,
        image_files: List[Path],
        category: str,
        extract_features: bool,
        create_augmented: bool
    ) -> Dict[str, Any]:
        """Process a batch of images."""
        results = {
            "processed": 0,
            "features_extracted": 0,
            "augmented_created": 0,
            "errors": []
        }

        for image_file in image_files:
            try:
                # Create scan record for this image
                scan = await self._create_image_scan(db, survey_id, image_file, category)

                # Extract features if requested
                if extract_features:
                    features = await self._extract_comprehensive_features(image_file)
                    await self._store_features(db, scan.id, features, category)
                    results["features_extracted"] += 1

                # Create augmented versions if requested
                if create_augmented:
                    augmented_count = await self._create_augmented_images(image_file, category)
                    results["augmented_created"] += augmented_count

                results["processed"] += 1

            except Exception as e:
                error_info = {
                    "file": str(image_file),
                    "error": str(e)
                }
                results["errors"].append(error_info)
                self.logger.error(f"Error processing image {image_file}: {e}")

        return results

    def _get_image_files(self, category_path: Path, max_images: Optional[int]) -> List[Path]:
        """Get list of image files from category directory."""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []

        for ext in extensions:
            image_files.extend(category_path.glob(f"*{ext}"))
            image_files.extend(category_path.glob(f"*{ext.upper()}"))

        # Sort for consistent processing order
        image_files.sort()

        if max_images:
            image_files = image_files[:max_images]

        return image_files

    async def _create_or_get_survey(self, db: AsyncSession, category: str) -> GPRSurvey:
        """Create or get survey for the category."""
        survey_name = f"Mojahid_{category}"

        # Check if survey already exists
        result = await db.execute(
            select(GPRSurvey).where(GPRSurvey.survey_name == survey_name)
        )
        survey = result.scalar_one_or_none()

        if not survey:
            survey = GPRSurvey(
                survey_name=survey_name,
                location_id=f"mojahid_{category}",
                survey_objective=f"Mojahid GPR images classification - {category}",
                status="completed"
            )
            db.add(survey)
            await db.flush()

        return survey

    async def _create_image_scan(
        self,
        db: AsyncSession,
        survey_id: str,
        image_file: Path,
        category: str
    ) -> GPRScan:
        """Create scan record for an image."""
        scan = GPRScan(
            survey_id=survey_id,
            scan_number=1,  # Will be updated with proper numbering
            scan_name=image_file.name,
            file_path=str(image_file),
            file_size_bytes=image_file.stat().st_size,
            data_format="IMAGE",
            is_processed=True,
            processing_status="completed"
        )

        db.add(scan)
        await db.flush()

        return scan

    async def _extract_comprehensive_features(self, image_file: Path) -> Dict[str, Any]:
        """Extract comprehensive features from an image."""
        try:
            features = {}

            # Load image
            image_rgb = np.array(Image.open(image_file).convert('RGB'))
            image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

            # Extract different types of features
            for feature_type, extractor in self.feature_extractors.items():
                try:
                    feature_values = await extractor(image_rgb, image_gray)
                    features[feature_type] = feature_values
                except Exception as e:
                    self.logger.warning(f"Failed to extract {feature_type} features from {image_file}: {e}")
                    features[feature_type] = []

            return features

        except Exception as e:
            self.logger.error(f"Error extracting features from {image_file}: {e}")
            return {}

    async def _extract_statistical_features(self, image_rgb: np.ndarray, image_gray: np.ndarray) -> List[float]:
        """Extract statistical features from image."""
        features = []

        # Grayscale statistics
        features.extend([
            float(np.mean(image_gray)),
            float(np.std(image_gray)),
            float(np.min(image_gray)),
            float(np.max(image_gray)),
            float(np.median(image_gray)),
            float(np.percentile(image_gray, 25)),
            float(np.percentile(image_gray, 75)),
            float(np.var(image_gray))
        ])

        # RGB channel statistics
        for channel in range(3):
            channel_data = image_rgb[:, :, channel]
            features.extend([
                float(np.mean(channel_data)),
                float(np.std(channel_data)),
                float(np.var(channel_data))
            ])

        # Intensity distribution features
        hist, _ = np.histogram(image_gray, bins=50, range=(0, 255))
        hist_normalized = hist / np.sum(hist)
        features.extend([
            float(np.argmax(hist_normalized)),  # Peak intensity
            float(np.sum(hist_normalized[:25])),  # Dark pixels ratio
            float(np.sum(hist_normalized[25:]))   # Bright pixels ratio
        ])

        return features

    async def _extract_texture_features(self, image_rgb: np.ndarray, image_gray: np.ndarray) -> List[float]:
        """Extract texture features using Local Binary Patterns and other methods."""
        features = []

        try:
            # Simple texture measures
            # Sobel gradient magnitude
            sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

            features.extend([
                float(np.mean(gradient_magnitude)),
                float(np.std(gradient_magnitude)),
                float(np.max(gradient_magnitude))
            ])

            # Laplacian variance (measure of texture)
            laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
            features.append(float(np.var(laplacian)))

            # Local standard deviation
            kernel = np.ones((5, 5)) / 25
            local_mean = cv2.filter2D(image_gray.astype(np.float64), -1, kernel)
            local_variance = cv2.filter2D((image_gray.astype(np.float64) - local_mean)**2, -1, kernel)
            features.append(float(np.mean(np.sqrt(local_variance))))

        except Exception as e:
            self.logger.warning(f"Error extracting texture features: {e}")
            features = [0.0] * 6

        return features

    async def _extract_edge_features(self, image_rgb: np.ndarray, image_gray: np.ndarray) -> List[float]:
        """Extract edge-based features."""
        features = []

        try:
            # Canny edge detection
            edges = cv2.Canny(image_gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            features.append(float(edge_density))

            # Edge direction histogram
            sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_angles = np.arctan2(sobel_y, sobel_x)

            # Histogram of edge orientations
            hist, _ = np.histogram(edge_angles[edges > 0], bins=8, range=(-np.pi, np.pi))
            hist_normalized = hist / (np.sum(hist) + 1e-8)
            features.extend(hist_normalized.tolist())

        except Exception as e:
            self.logger.warning(f"Error extracting edge features: {e}")
            features = [0.0] * 9

        return features

    async def _extract_histogram_features(self, image_rgb: np.ndarray, image_gray: np.ndarray) -> List[float]:
        """Extract histogram-based features."""
        features = []

        try:
            # Grayscale histogram features
            hist_gray, _ = np.histogram(image_gray, bins=32, range=(0, 255))
            hist_gray_norm = hist_gray / (np.sum(hist_gray) + 1e-8)
            features.extend(hist_gray_norm.tolist())

            # RGB histogram features (reduced bins)
            for channel in range(3):
                hist_channel, _ = np.histogram(image_rgb[:, :, channel], bins=16, range=(0, 255))
                hist_channel_norm = hist_channel / (np.sum(hist_channel) + 1e-8)
                features.extend(hist_channel_norm.tolist())

        except Exception as e:
            self.logger.warning(f"Error extracting histogram features: {e}")
            features = [0.0] * 80  # 32 + 16*3

        return features

    async def _extract_spatial_features(self, image_rgb: np.ndarray, image_gray: np.ndarray) -> List[float]:
        """Extract spatial features like moments and shape descriptors."""
        features = []

        try:
            # Image moments
            moments = cv2.moments(image_gray)

            # Hu moments (shape descriptors)
            hu_moments = cv2.HuMoments(moments)
            features.extend([float(hu) for hu in hu_moments.flatten()])

            # Aspect ratio and basic shape features
            h, w = image_gray.shape
            features.extend([
                float(w / h),  # Aspect ratio
                float(h * w),  # Area
                float(2 * (h + w))  # Perimeter approximation
            ])

        except Exception as e:
            self.logger.warning(f"Error extracting spatial features: {e}")
            features = [0.0] * 10

        return features

    async def _store_features(
        self,
        db: AsyncSession,
        scan_id: str,
        features: Dict[str, Any],
        category: str
    ) -> None:
        """Store extracted features in the database."""
        try:
            # Flatten all features into a single vector
            feature_vector = []
            feature_names = []

            for feature_type, feature_values in features.items():
                if isinstance(feature_values, list):
                    feature_vector.extend(feature_values)
                    feature_names.extend([f"{feature_type}_{i}" for i in range(len(feature_values))])

            if not feature_vector:
                return

            # Create feature vector record
            feature_record = FeatureVector(
                scan_id=scan_id,
                feature_set_name="mojahid_comprehensive",
                feature_version="1.0",
                extraction_method="computer_vision",
                feature_vector=feature_vector,
                feature_names=feature_names,
                feature_dimensions=len(feature_vector),
                statistical_features=features.get("statistical"),
                frequency_features={},  # Not applicable for images
                spatial_features=features.get("spatial"),
                target_label=category,
                target_confidence=1.0,
                ground_truth_available=True,
                extraction_timestamp=datetime.now()
            )

            db.add(feature_record)

        except Exception as e:
            self.log_operation_error("store_features", e)
            raise

    async def _create_augmented_images(self, image_file: Path, category: str) -> int:
        """Create augmented versions of an image (placeholder implementation)."""
        # This would create augmented versions using:
        # - Rotation
        # - Scaling
        # - Brightness/contrast adjustment
        # - Noise addition
        # - Horizontal/vertical flipping

        # For now, return 0 as this is a placeholder
        return 0

    async def get_processing_statistics(self, db: AsyncSession) -> Dict[str, Any]:
        """Get comprehensive processing statistics for Mojahid dataset."""
        try:
            # Count processed images by category
            category_stats = {}
            categories = settings.get_mojahid_categories()

            for category in categories:
                survey_name = f"Mojahid_{category}"
                result = await db.execute(
                    select(GPRScan).join(GPRSurvey).where(
                        GPRSurvey.survey_name == survey_name
                    )
                )
                scans = result.scalars().all()
                category_stats[category] = len(scans)

            # Count feature vectors
            feature_result = await db.execute(
                select(FeatureVector).where(
                    FeatureVector.feature_set_name == "mojahid_comprehensive"
                )
            )
            total_features = len(feature_result.scalars().all())

            return {
                "category_statistics": category_stats,
                "total_features_extracted": total_features,
                "last_updated": datetime.now().isoformat()
            }

        except Exception as e:
            self.log_operation_error("get_processing_statistics", e)
            raise