"""
Hyperspectral Imaging Processing Service
========================================

CRITICAL SERVICE for HS2 concrete quality assessment.

This service processes hyperspectral images from Specim IQ camera (204 bands, 400-1000nm)
to predict concrete strength, detect defects, and assess material quality.

Target Performance:
- Lab Conditions: R²=0.89, MAE=3.2 MPa
- Field Conditions: R²=0.82, MAE=4.2 MPa

Key Wavelength Bands:
- 500-600nm: Cement hydration (curing quality)
- 700-850nm: Moisture content (strength predictor)
- 900-1000nm: Aggregate composition (spec compliance)
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID
import base64
import io
from PIL import Image
from sqlalchemy.orm import Session
from datetime import datetime

from app.models.hyperspectral import (
    HyperspectralMaterialSample,
    HyperspectralAnalysis,
    ConcreteStrengthCalibration,
    MaterialType,
    QualityLabel
)
from app.schemas.hyperspectral import (
    DefectSeverity,
    MaterialClassification,
    DefectInfo
)

logger = logging.getLogger(__name__)


class HyperspectralProcessor:
    """
    Main hyperspectral image processing service.

    Handles material classification, concrete strength prediction,
    defect detection, and spectral signature extraction.
    """

    def __init__(self, db: Session):
        self.db = db
        self.num_bands = 204  # Specim IQ standard
        self.wavelength_min = 400.0  # nm
        self.wavelength_max = 1000.0  # nm
        self.wavelength_step = (self.wavelength_max - self.wavelength_min) / (self.num_bands - 1)

        logger.info(
            f"HyperspectralProcessor initialized: {self.num_bands} bands, "
            f"{self.wavelength_min}-{self.wavelength_max}nm"
        )

    def get_wavelengths(self) -> List[float]:
        """Generate wavelength array for Specim IQ (204 bands)."""
        return [
            self.wavelength_min + i * self.wavelength_step
            for i in range(self.num_bands)
        ]

    async def analyze_image(
        self,
        image_data: Optional[str] = None,
        image_path: Optional[str] = None,
        predict_material: bool = True,
        predict_strength: bool = True,
        detect_defects: bool = True,
        extract_signature: bool = True
    ) -> Dict[str, Any]:
        """
        MAIN ANALYSIS PIPELINE - End-to-end hyperspectral image analysis.

        Args:
            image_data: Base64 encoded image string
            image_path: Path to image file
            predict_material: Run material classification
            predict_strength: Predict concrete strength (if material is concrete)
            detect_defects: Run defect detection
            extract_signature: Extract spectral signature

        Returns:
            Dictionary containing all analysis results

        Raises:
            ValueError: If neither image_data nor image_path provided
        """
        logger.info(
            f"Starting hyperspectral analysis: "
            f"material={predict_material}, strength={predict_strength}, "
            f"defects={detect_defects}, signature={extract_signature}"
        )

        # Load image
        if image_data:
            image = self._load_image_from_base64(image_data)
        elif image_path:
            image = self._load_image_from_path(image_path)
        else:
            raise ValueError("Either image_data or image_path must be provided")

        logger.info(f"Image loaded: size={image.size}, mode={image.mode}")

        results = {
            "image_metadata": {
                "width": image.size[0],
                "height": image.size[1],
                "mode": image.mode,
                "analyzed_at": datetime.utcnow().isoformat()
            }
        }

        # Convert image to numpy array for processing
        img_array = np.array(image)

        # Material classification
        if predict_material:
            logger.info("Running material classification")
            material_result = await self.classify_material(img_array)
            results["material_classification"] = material_result
            logger.info(
                f"Material classified: {material_result['material_type']} "
                f"(confidence={material_result['confidence']:.3f})"
            )

        # Concrete strength prediction (only if material is concrete)
        if predict_strength and results.get("material_classification", {}).get("material_type") == "concrete":
            logger.info("Running concrete strength prediction")
            # Extract spectral signature first if needed
            if not extract_signature:
                spectral_sig = await self.extract_spectral_signature(img_array)
            else:
                spectral_sig = results.get("spectral_signature", await self.extract_spectral_signature(img_array))

            strength_result = await self.predict_concrete_strength(
                spectral_sig["reflectance_values"]
            )
            results["concrete_strength"] = strength_result
            logger.info(
                f"Strength predicted: {strength_result['predicted_strength_mpa']:.1f} MPa "
                f"(confidence={strength_result['confidence']:.3f})"
            )

        # Defect detection
        if detect_defects:
            logger.info("Running defect detection")
            defect_results = await self.detect_defects(img_array)
            results["defects"] = defect_results
            logger.info(f"Defects detected: {len(defect_results['defects_detected'])} defects")

        # Spectral signature extraction
        if extract_signature:
            logger.info("Extracting spectral signature")
            signature = await self.extract_spectral_signature(img_array)
            results["spectral_signature"] = signature
            logger.info(
                f"Signature extracted: {signature['num_bands']} bands, "
                f"key wavelengths analyzed"
            )

        logger.info("Hyperspectral analysis completed successfully")
        return results

    async def classify_material(
        self,
        image_array: np.ndarray
    ) -> Dict[str, Any]:
        """
        Classify material type from hyperspectral image.

        Uses trained ML model based on UMKC dataset (150 → 1,500 samples).

        Args:
            image_array: Numpy array of image data

        Returns:
            Dictionary with material_type, confidence, and probabilities
        """
        logger.info("Starting material classification")

        # Extract spectral features
        spectral_features = self._extract_spectral_features(image_array)

        # TODO: Load trained material classification model
        # For now, simulate classification based on spectral characteristics

        # Analyze spectral characteristics
        mean_intensity = np.mean(image_array)
        std_intensity = np.std(image_array)

        logger.debug(f"Spectral features: mean={mean_intensity:.2f}, std={std_intensity:.2f}")

        # Simplified classification logic (replace with actual ML model)
        if mean_intensity > 150 and std_intensity < 30:
            material_type = MaterialType.CONCRETE
            confidence = 0.87
        elif mean_intensity > 100 and std_intensity > 40:
            material_type = MaterialType.ASPHALT
            confidence = 0.82
        else:
            material_type = MaterialType.UNKNOWN
            confidence = 0.65

        # Calculate probability distribution
        probabilities = {
            MaterialType.CONCRETE.value: confidence if material_type == MaterialType.CONCRETE else 0.1,
            MaterialType.ASPHALT.value: confidence if material_type == MaterialType.ASPHALT else 0.1,
            MaterialType.STEEL.value: 0.05,
            MaterialType.WOOD.value: 0.05,
            MaterialType.SOIL.value: 0.05,
            MaterialType.VEGETATION.value: 0.05,
            MaterialType.WATER.value: 0.05,
            MaterialType.UNKNOWN.value: confidence if material_type == MaterialType.UNKNOWN else 0.05,
        }

        # Normalize probabilities
        total_prob = sum(probabilities.values())
        probabilities = {k: v / total_prob for k, v in probabilities.items()}

        result = {
            "material_type": material_type.value,
            "confidence": confidence,
            "probabilities": probabilities,
            "spectral_features": {
                "mean_intensity": float(mean_intensity),
                "std_intensity": float(std_intensity)
            }
        }

        logger.info(f"Material classification complete: {material_type.value} (conf={confidence:.3f})")
        return result

    async def predict_concrete_strength(
        self,
        spectral_signature: List[float],
        calibration_id: Optional[UUID] = None
    ) -> Dict[str, float]:
        """
        Predict concrete strength from spectral signature.

        Target Performance:
        - Lab: R²=0.89, MAE=3.2 MPa
        - Field: R²=0.82, MAE=4.2 MPa

        Key wavelengths analyzed:
        - 500-600nm: Cement hydration quality
        - 700-850nm: Moisture content
        - 900-1000nm: Aggregate composition

        Args:
            spectral_signature: List of reflectance values (204 bands)
            calibration_id: Optional specific calibration model to use

        Returns:
            Dictionary with predicted strength, confidence, and range
        """
        logger.info(
            f"Predicting concrete strength from spectral signature "
            f"(calibration_id={calibration_id})"
        )

        # Validate input
        if len(spectral_signature) != self.num_bands:
            logger.warning(
                f"Spectral signature has {len(spectral_signature)} bands, "
                f"expected {self.num_bands}. Attempting interpolation."
            )
            spectral_signature = self._interpolate_spectral_signature(spectral_signature)

        # Get wavelengths
        wavelengths = self.get_wavelengths()

        # Extract key wavelength regions
        cement_hydration = self._get_wavelength_region_value(
            wavelengths, spectral_signature, 500, 600
        )
        moisture_content = self._get_wavelength_region_value(
            wavelengths, spectral_signature, 700, 850
        )
        aggregate_quality = self._get_wavelength_region_value(
            wavelengths, spectral_signature, 900, 1000
        )

        logger.debug(
            f"Key wavelength analysis: cement_hydration={cement_hydration:.3f}, "
            f"moisture={moisture_content:.3f}, aggregate={aggregate_quality:.3f}"
        )

        # Load calibration model
        if calibration_id:
            calibration = self.db.query(ConcreteStrengthCalibration).filter(
                ConcreteStrengthCalibration.id == calibration_id
            ).first()
        else:
            # Use the best validated calibration
            calibration = self.db.query(ConcreteStrengthCalibration).filter(
                ConcreteStrengthCalibration.is_validated == True
            ).order_by(ConcreteStrengthCalibration.r_squared.desc()).first()

        if calibration:
            logger.info(
                f"Using calibration: {calibration.calibration_name} "
                f"(R²={calibration.r_squared:.3f}, MAE={calibration.mae:.2f} MPa)"
            )
            model_r_squared = calibration.r_squared
            model_mae = calibration.mae
        else:
            logger.warning("No calibration found, using default model parameters")
            model_r_squared = 0.85  # Conservative estimate
            model_mae = 4.5  # Conservative estimate

        # TODO: Load trained regression model
        # For now, use simplified empirical formula based on key wavelengths

        # Simplified strength prediction (replace with actual ML model)
        # Formula: strength correlates with cement hydration and inversely with moisture
        base_strength = 40.0  # MPa baseline
        cement_factor = (cement_hydration - 0.5) * 30.0  # +/- 15 MPa
        moisture_penalty = (moisture_content - 0.3) * -20.0  # Moisture reduces strength
        aggregate_bonus = (aggregate_quality - 0.5) * 10.0  # +/- 5 MPa

        predicted_strength = base_strength + cement_factor + moisture_penalty + aggregate_bonus
        predicted_strength = max(20.0, min(60.0, predicted_strength))  # Clamp to realistic range

        # Calculate confidence based on model accuracy
        confidence = model_r_squared * 0.95  # Scale R² to confidence

        # Calculate prediction range based on MAE
        strength_range_min = predicted_strength - (model_mae * 1.5)
        strength_range_max = predicted_strength + (model_mae * 1.5)

        result = {
            "predicted_strength_mpa": round(predicted_strength, 1),
            "confidence": round(confidence, 3),
            "strength_range_min": round(strength_range_min, 1),
            "strength_range_max": round(strength_range_max, 1),
            "model_r_squared": round(model_r_squared, 3),
            "model_mae": round(model_mae, 2),
            "key_wavelength_values": {
                "cement_hydration_500_600": round(cement_hydration, 3),
                "moisture_content_700_850": round(moisture_content, 3),
                "aggregate_quality_900_1000": round(aggregate_quality, 3)
            }
        }

        logger.info(
            f"Strength prediction complete: {predicted_strength:.1f} MPa "
            f"(range: {strength_range_min:.1f}-{strength_range_max:.1f} MPa, "
            f"confidence={confidence:.3f})"
        )

        return result

    async def detect_defects(
        self,
        image_array: np.ndarray,
        sensitivity: float = 0.7
    ) -> Dict[str, Any]:
        """
        Detect surface defects in concrete/material samples.

        Detects: voids, cracks, spalling, delamination, staining

        Args:
            image_array: Numpy array of image data
            sensitivity: Detection threshold (0.0-1.0), higher = more sensitive

        Returns:
            Dictionary with detected defects and overall severity
        """
        logger.info(f"Starting defect detection (sensitivity={sensitivity})")

        # TODO: Implement advanced defect detection using:
        # - Edge detection (Canny, Sobel)
        # - Morphological operations
        # - Spectral anomaly detection
        # - Trained defect classification model

        # Simplified defect detection (replace with actual CV/ML model)
        defects_detected = []

        # Analyze image variance for anomalies
        height, width = image_array.shape[:2]

        # Divide image into grid for local analysis
        grid_size = 50  # pixels
        for y in range(0, height - grid_size, grid_size):
            for x in range(0, width - grid_size, grid_size):
                patch = image_array[y:y+grid_size, x:x+grid_size]
                patch_variance = np.var(patch)

                # High variance may indicate defects
                if patch_variance > (100 * sensitivity):
                    defect_type = self._classify_defect_type(patch)
                    severity = self._calculate_defect_severity(patch_variance, sensitivity)

                    defects_detected.append({
                        "defect_type": defect_type,
                        "location_x": x + grid_size // 2,
                        "location_y": y + grid_size // 2,
                        "size_pixels": grid_size * grid_size,
                        "confidence": min(0.95, patch_variance / 200.0),
                        "severity": severity
                    })

        # Calculate overall severity
        if not defects_detected:
            overall_severity = DefectSeverity.NONE
        else:
            severity_counts = {}
            for defect in defects_detected:
                sev = defect["severity"]
                severity_counts[sev] = severity_counts.get(sev, 0) + 1

            if severity_counts.get(DefectSeverity.SEVERE, 0) > 0:
                overall_severity = DefectSeverity.SEVERE
            elif severity_counts.get(DefectSeverity.MODERATE, 0) > 2:
                overall_severity = DefectSeverity.MODERATE
            elif severity_counts.get(DefectSeverity.MINOR, 0) > 5:
                overall_severity = DefectSeverity.MINOR
            else:
                overall_severity = DefectSeverity.MINOR

        result = {
            "defects_detected": defects_detected,
            "num_defects": len(defects_detected),
            "overall_severity": overall_severity.value,
            "defect_locations": [
                {"x": d["location_x"], "y": d["location_y"]}
                for d in defects_detected
            ]
        }

        logger.info(
            f"Defect detection complete: {len(defects_detected)} defects found, "
            f"overall severity={overall_severity.value}"
        )

        return result

    async def extract_spectral_signature(
        self,
        image_array: np.ndarray,
        region: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict[str, Any]:
        """
        Extract spectral signature from image.

        For Specim IQ: 204 bands, 400-1000nm range

        Args:
            image_array: Numpy array of image data
            region: Optional (x, y, width, height) region to analyze

        Returns:
            Dictionary with wavelengths, reflectance values, and key band analysis
        """
        logger.info(f"Extracting spectral signature (region={region})")

        # Select region of interest
        if region:
            x, y, w, h = region
            roi = image_array[y:y+h, x:x+w]
        else:
            roi = image_array

        # TODO: For full hyperspectral cube, extract actual band values
        # For now, simulate spectral signature from RGB image

        wavelengths = self.get_wavelengths()

        # Simulate reflectance values based on RGB channels
        # In production, this would read actual hyperspectral cube data
        mean_rgb = np.mean(roi, axis=(0, 1))

        # Interpolate RGB to 204 bands (simplified simulation)
        reflectance_values = self._simulate_spectral_signature_from_rgb(mean_rgb)

        # Analyze key wavelength regions
        cement_hydration = self._get_wavelength_region_value(
            wavelengths, reflectance_values, 500, 600
        )
        moisture_content = self._get_wavelength_region_value(
            wavelengths, reflectance_values, 700, 850
        )
        aggregate_quality = self._get_wavelength_region_value(
            wavelengths, reflectance_values, 900, 1000
        )

        result = {
            "wavelengths_nm": wavelengths,
            "reflectance_values": reflectance_values,
            "num_bands": self.num_bands,
            "wavelength_range": f"{self.wavelength_min}-{self.wavelength_max}nm",
            "cement_hydration_500_600": round(cement_hydration, 3),
            "moisture_content_700_850": round(moisture_content, 3),
            "aggregate_quality_900_1000": round(aggregate_quality, 3),
            "mean_reflectance": round(float(np.mean(reflectance_values)), 3),
            "std_reflectance": round(float(np.std(reflectance_values)), 3)
        }

        logger.info(
            f"Spectral signature extracted: {self.num_bands} bands, "
            f"mean reflectance={result['mean_reflectance']:.3f}"
        )

        return result

    async def augment_training_sample(
        self,
        sample_id: UUID,
        augmentation_methods: List[str]
    ) -> List[UUID]:
        """
        Generate augmented training samples from existing sample.

        Goal: Expand 150 UMKC samples to 1,500+ with augmentation.

        Augmentation methods:
        - rotation: Rotate 90/180/270 degrees
        - flip: Horizontal/vertical flip
        - brightness: Adjust brightness +/- 10%
        - noise: Add Gaussian noise
        - crop: Random crop and resize

        Args:
            sample_id: UUID of original sample
            augmentation_methods: List of methods to apply

        Returns:
            List of UUIDs for newly created augmented samples
        """
        logger.info(
            f"Augmenting training sample {sample_id} with methods: "
            f"{', '.join(augmentation_methods)}"
        )

        # Load original sample
        original_sample = self.db.query(HyperspectralMaterialSample).filter(
            HyperspectralMaterialSample.id == sample_id
        ).first()

        if not original_sample:
            logger.error(f"Sample {sample_id} not found")
            raise ValueError(f"Sample {sample_id} not found")

        logger.info(
            f"Original sample loaded: {original_sample.sample_id} "
            f"({original_sample.material_type.value})"
        )

        # TODO: Implement actual image augmentation
        # For now, create placeholder augmented samples

        augmented_ids = []

        for method in augmentation_methods:
            # Create augmented sample record
            augmented_sample = HyperspectralMaterialSample(
                sample_id=f"{original_sample.sample_id}_aug_{method}",
                sample_name=f"{original_sample.sample_name} (augmented - {method})",
                material_type=original_sample.material_type,
                material_subtype=original_sample.material_subtype,
                surface_condition=original_sample.surface_condition,
                surface_age=original_sample.surface_age,
                moisture_level=original_sample.moisture_level,
                image_path=f"{original_sample.image_path}_aug_{method}",
                image_format=original_sample.image_format,
                resolution=original_sample.resolution,
                spectral_signature=original_sample.spectral_signature,
                num_bands=original_sample.num_bands,
                wavelength_range_nm=original_sample.wavelength_range_nm,
                is_specim_compatible=original_sample.is_specim_compatible,
                spectral_resolution_nm=original_sample.spectral_resolution_nm,
                source=original_sample.source,
                dataset_name=original_sample.dataset_name,
                quality_label=original_sample.quality_label,
                is_augmented=True,
                parent_sample_id=sample_id,
                augmentation_method=method,
                ground_truth_strength_mpa=original_sample.ground_truth_strength_mpa,
                ground_truth_moisture_pct=original_sample.ground_truth_moisture_pct,
                ground_truth_defects=original_sample.ground_truth_defects,
                metadata={"augmentation_method": method, "parent_id": str(sample_id)}
            )

            self.db.add(augmented_sample)
            augmented_ids.append(augmented_sample.id)

        self.db.commit()

        logger.info(
            f"Created {len(augmented_ids)} augmented samples from {sample_id}"
        )

        return augmented_ids

    async def calculate_quality_scores(
        self,
        spectral_signature: List[float]
    ) -> Dict[str, float]:
        """
        Calculate quality scores from spectral signature.

        Scores:
        - curing_quality: 0.0-1.0 (based on 500-600nm cement hydration)
        - moisture_level: 0.0-1.0 (based on 700-850nm water absorption)
        - aggregate_quality: 0.0-1.0 (based on 900-1000nm aggregate composition)

        Args:
            spectral_signature: List of reflectance values (204 bands)

        Returns:
            Dictionary with quality scores
        """
        logger.info("Calculating quality scores from spectral signature")

        wavelengths = self.get_wavelengths()

        # Extract key wavelength regions
        cement_hydration = self._get_wavelength_region_value(
            wavelengths, spectral_signature, 500, 600
        )
        moisture_content = self._get_wavelength_region_value(
            wavelengths, spectral_signature, 700, 850
        )
        aggregate_quality = self._get_wavelength_region_value(
            wavelengths, spectral_signature, 900, 1000
        )

        # Calculate quality scores (0.0-1.0 scale)
        # Higher cement hydration = better curing
        curing_quality = min(1.0, cement_hydration / 0.8)

        # Lower moisture = better quality (inverse relationship)
        moisture_quality = max(0.0, 1.0 - (moisture_content / 0.5))

        # Higher aggregate reflectance = better quality
        aggregate_score = min(1.0, aggregate_quality / 0.7)

        result = {
            "curing_quality_score": round(curing_quality, 3),
            "moisture_quality_score": round(moisture_quality, 3),
            "aggregate_quality_score": round(aggregate_score, 3),
            "overall_quality_score": round((curing_quality + moisture_quality + aggregate_score) / 3.0, 3)
        }

        logger.info(
            f"Quality scores calculated: curing={curing_quality:.3f}, "
            f"moisture={moisture_quality:.3f}, aggregate={aggregate_score:.3f}, "
            f"overall={result['overall_quality_score']:.3f}"
        )

        return result

    # Helper methods

    def _load_image_from_base64(self, base64_str: str) -> Image.Image:
        """Load image from base64 encoded string."""
        try:
            image_data = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_data))
            logger.debug(f"Image loaded from base64: size={image.size}, mode={image.mode}")
            return image
        except Exception as e:
            logger.error(f"Failed to load image from base64: {e}")
            raise ValueError(f"Invalid base64 image data: {e}")

    def _load_image_from_path(self, image_path: str) -> Image.Image:
        """Load image from file path."""
        try:
            image = Image.open(image_path)
            logger.debug(f"Image loaded from path: {image_path}, size={image.size}")
            return image
        except Exception as e:
            logger.error(f"Failed to load image from {image_path}: {e}")
            raise ValueError(f"Failed to load image from path: {e}")

    def _extract_spectral_features(self, image_array: np.ndarray) -> Dict[str, float]:
        """Extract spectral features for classification."""
        # Calculate basic statistical features
        return {
            "mean": float(np.mean(image_array)),
            "std": float(np.std(image_array)),
            "min": float(np.min(image_array)),
            "max": float(np.max(image_array)),
            "median": float(np.median(image_array))
        }

    def _get_wavelength_region_value(
        self,
        wavelengths: List[float],
        reflectance: List[float],
        wl_min: float,
        wl_max: float
    ) -> float:
        """Get average reflectance value for wavelength region."""
        region_values = [
            reflectance[i]
            for i, wl in enumerate(wavelengths)
            if wl_min <= wl <= wl_max
        ]
        return float(np.mean(region_values)) if region_values else 0.0

    def _interpolate_spectral_signature(self, signature: List[float]) -> List[float]:
        """Interpolate spectral signature to 204 bands."""
        if len(signature) == self.num_bands:
            return signature

        # Linear interpolation
        x_old = np.linspace(0, 1, len(signature))
        x_new = np.linspace(0, 1, self.num_bands)
        interpolated = np.interp(x_new, x_old, signature)
        return interpolated.tolist()

    def _simulate_spectral_signature_from_rgb(self, rgb: np.ndarray) -> List[float]:
        """
        Simulate 204-band spectral signature from RGB values.

        NOTE: This is a placeholder. In production, actual hyperspectral
        cube data would be read from Specim IQ .raw files.
        """
        # Normalize RGB to 0-1 range
        rgb_normalized = rgb / 255.0

        # Create spectral signature with Gaussian curves centered on RGB peaks
        wavelengths = self.get_wavelengths()
        signature = np.zeros(self.num_bands)

        # Red peak at ~650nm
        red_peak = 650
        signature += rgb_normalized[0] * np.exp(-((np.array(wavelengths) - red_peak) ** 2) / (2 * 50 ** 2))

        # Green peak at ~550nm
        green_peak = 550
        signature += rgb_normalized[1] * np.exp(-((np.array(wavelengths) - green_peak) ** 2) / (2 * 50 ** 2))

        # Blue peak at ~450nm
        blue_peak = 450
        signature += rgb_normalized[2] * np.exp(-((np.array(wavelengths) - blue_peak) ** 2) / (2 * 50 ** 2))

        # Add baseline
        signature += 0.1

        # Normalize to 0-1 range
        signature = np.clip(signature, 0, 1)

        return signature.tolist()

    def _classify_defect_type(self, patch: np.ndarray) -> str:
        """Classify type of defect from image patch."""
        # Simplified classification based on patch characteristics
        variance = np.var(patch)
        mean_val = np.mean(patch)

        if variance > 150:
            return "crack"
        elif mean_val < 50:
            return "void"
        elif variance > 100:
            return "spalling"
        else:
            return "staining"

    def _calculate_defect_severity(self, variance: float, sensitivity: float) -> DefectSeverity:
        """Calculate defect severity from variance."""
        threshold = 100 * sensitivity

        if variance > threshold * 3:
            return DefectSeverity.SEVERE
        elif variance > threshold * 2:
            return DefectSeverity.MODERATE
        else:
            return DefectSeverity.MINOR
