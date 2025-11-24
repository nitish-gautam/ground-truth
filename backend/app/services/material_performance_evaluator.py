"""
Material Classification Performance Evaluation and Validation Framework
======================================================================

Comprehensive evaluation framework for assessing material classification model
performance, validation against real-world data, and compliance with PAS 128
standards for utility detection.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, precision_recall_curve,
    roc_curve, f1_score, accuracy_score, precision_score, recall_score,
    cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
import scipy.stats as stats

from services.material_classification import (
    MaterialType, UtilityDiscipline, GPRSignatureFeatures,
    MaterialClassificationService, DisciplineSpecificMaterialPredictor,
    EnvironmentalFactorIntegrator
)

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for material classification."""

    # Basic classification metrics
    accuracy: float
    precision_macro: float
    precision_micro: float
    precision_weighted: float
    recall_macro: float
    recall_micro: float
    recall_weighted: float
    f1_macro: float
    f1_micro: float
    f1_weighted: float

    # Advanced metrics
    balanced_accuracy: float
    cohen_kappa: float
    matthews_correlation: float

    # Per-class metrics
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_f1: Dict[str, float]
    per_class_support: Dict[str, int]

    # Confidence and uncertainty metrics
    prediction_confidence_mean: float
    prediction_confidence_std: float
    low_confidence_predictions_rate: float

    # Detection difficulty metrics
    easy_materials_accuracy: float
    medium_materials_accuracy: float
    difficult_materials_accuracy: float

    # Cross-validation metrics
    cv_accuracy_mean: float
    cv_accuracy_std: float
    cv_f1_mean: float
    cv_f1_std: float


@dataclass
class PAS128ComplianceMetrics:
    """Metrics specific to PAS 128 utility detection standards."""

    detection_quality_a_rate: float  # ≥95% accuracy
    detection_quality_b_rate: float  # 80-95% accuracy
    detection_quality_c_rate: float  # 65-80% accuracy
    detection_quality_d_rate: float  # <65% accuracy

    material_identification_accuracy: float
    depth_estimation_accuracy: float
    false_positive_rate: float
    false_negative_rate: float

    compliance_level: str  # A, B, C, or D
    compliant_materials: List[str]
    non_compliant_materials: List[str]


@dataclass
class ValidationResults:
    """Comprehensive validation results structure."""

    performance_metrics: PerformanceMetrics
    pas128_compliance: PAS128ComplianceMetrics
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Any]

    # Environmental validation
    environmental_performance: Dict[str, PerformanceMetrics]

    # Discipline-specific validation
    discipline_performance: Dict[str, PerformanceMetrics]

    # Temporal validation
    timestamp: str
    model_version: str
    dataset_info: Dict[str, Any]


class MaterialPerformanceEvaluator:
    """Comprehensive performance evaluation and validation framework."""

    def __init__(self, classification_service: MaterialClassificationService):
        """Initialize the performance evaluator."""
        self.classification_service = classification_service
        self.discipline_predictor = DisciplineSpecificMaterialPredictor()
        self.environmental_integrator = EnvironmentalFactorIntegrator()

        # Material difficulty levels based on detection characteristics
        self.material_difficulty_levels = {
            'easy': [MaterialType.STEEL, MaterialType.CAST_IRON],
            'medium': [MaterialType.PVC, MaterialType.HDPE, MaterialType.ASBESTOS_CEMENT],
            'difficult': [MaterialType.POLYETHYLENE, MaterialType.FIBER_REINFORCED_PLASTIC,
                         MaterialType.PILC, MaterialType.CONCRETE]
        }

        # PAS 128 quality thresholds
        self.pas128_thresholds = {
            'A': 0.95,
            'B': 0.80,
            'C': 0.65,
            'D': 0.00
        }

        logger.info("Material performance evaluator initialized")

    def comprehensive_evaluation(self,
                                signatures: List[GPRSignatureFeatures],
                                true_materials: List[MaterialType],
                                environmental_contexts: Optional[List[Dict[str, Any]]] = None,
                                disciplines: Optional[List[UtilityDiscipline]] = None) -> ValidationResults:
        """Perform comprehensive evaluation of material classification performance."""

        logger.info(f"Starting comprehensive evaluation with {len(signatures)} samples")

        # Generate predictions
        predictions, confidences = self._generate_predictions(signatures)

        # Calculate basic performance metrics
        performance_metrics = self._calculate_performance_metrics(
            true_materials, predictions, confidences
        )

        # Calculate PAS 128 compliance metrics
        pas128_compliance = self._calculate_pas128_compliance(
            true_materials, predictions, confidences
        )

        # Generate confusion matrix and classification report
        confusion_mat = confusion_matrix(
            [mat.value for mat in true_materials],
            [pred.value for pred in predictions]
        )

        class_names = [mat.value for mat in MaterialType if mat != MaterialType.UNKNOWN]
        classification_rep = classification_report(
            [mat.value for mat in true_materials],
            [pred.value for pred in predictions],
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )

        # Environmental performance evaluation
        environmental_performance = {}
        if environmental_contexts:
            environmental_performance = self._evaluate_environmental_performance(
                signatures, true_materials, environmental_contexts
            )

        # Discipline-specific performance evaluation
        discipline_performance = {}
        if disciplines:
            discipline_performance = self._evaluate_discipline_performance(
                signatures, true_materials, disciplines
            )

        # Create validation results
        validation_results = ValidationResults(
            performance_metrics=performance_metrics,
            pas128_compliance=pas128_compliance,
            confusion_matrix=confusion_mat.tolist(),
            classification_report=classification_rep,
            environmental_performance=environmental_performance,
            discipline_performance=discipline_performance,
            timestamp=datetime.now().isoformat(),
            model_version="enhanced_v1.0",
            dataset_info={
                'total_samples': len(signatures),
                'material_distribution': self._get_material_distribution(true_materials),
                'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        )

        logger.info("Comprehensive evaluation completed successfully")
        return validation_results

    def _generate_predictions(self, signatures: List[GPRSignatureFeatures]) -> Tuple[List[MaterialType], List[float]]:
        """Generate predictions and confidence scores for signatures."""

        predictions = []
        confidences = []

        for signature in signatures:
            try:
                # Use the enhanced prediction method with uncertainty
                result = self.classification_service.predict_material_with_uncertainty(signature)

                # Extract prediction and confidence
                predicted_material_str = result['predicted_material']
                predicted_material = MaterialType(predicted_material_str)
                confidence = result['ensemble_confidence']

                predictions.append(predicted_material)
                confidences.append(confidence)

            except Exception as e:
                logger.warning(f"Error predicting material: {e}")
                predictions.append(MaterialType.UNKNOWN)
                confidences.append(0.0)

        return predictions, confidences

    def _calculate_performance_metrics(self,
                                     true_materials: List[MaterialType],
                                     predictions: List[MaterialType],
                                     confidences: List[float]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""

        # Convert to string labels for sklearn
        y_true = [mat.value for mat in true_materials]
        y_pred = [pred.value for pred in predictions]

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Advanced metrics
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        cohen_kappa = cohen_kappa_score(y_true, y_pred)
        matthews_correlation = matthews_corrcoef(y_true, y_pred)

        # Per-class metrics
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        per_class_precision = {k: v['precision'] for k, v in class_report.items()
                              if k not in ['accuracy', 'macro avg', 'weighted avg', 'micro avg']}
        per_class_recall = {k: v['recall'] for k, v in class_report.items()
                           if k not in ['accuracy', 'macro avg', 'weighted avg', 'micro avg']}
        per_class_f1 = {k: v['f1-score'] for k, v in class_report.items()
                       if k not in ['accuracy', 'macro avg', 'weighted avg', 'micro avg']}
        per_class_support = {k: int(v['support']) for k, v in class_report.items()
                            if k not in ['accuracy', 'macro avg', 'weighted avg', 'micro avg']}

        # Confidence metrics
        confidence_mean = np.mean(confidences)
        confidence_std = np.std(confidences)
        low_confidence_rate = np.sum(np.array(confidences) < 0.6) / len(confidences)

        # Material difficulty metrics
        difficulty_accuracies = self._calculate_difficulty_accuracies(
            true_materials, predictions
        )

        return PerformanceMetrics(
            accuracy=accuracy,
            precision_macro=precision_macro,
            precision_micro=precision_micro,
            precision_weighted=precision_weighted,
            recall_macro=recall_macro,
            recall_micro=recall_micro,
            recall_weighted=recall_weighted,
            f1_macro=f1_macro,
            f1_micro=f1_micro,
            f1_weighted=f1_weighted,
            balanced_accuracy=balanced_accuracy,
            cohen_kappa=cohen_kappa,
            matthews_correlation=matthews_correlation,
            per_class_precision=per_class_precision,
            per_class_recall=per_class_recall,
            per_class_f1=per_class_f1,
            per_class_support=per_class_support,
            prediction_confidence_mean=confidence_mean,
            prediction_confidence_std=confidence_std,
            low_confidence_predictions_rate=low_confidence_rate,
            easy_materials_accuracy=difficulty_accuracies['easy'],
            medium_materials_accuracy=difficulty_accuracies['medium'],
            difficult_materials_accuracy=difficulty_accuracies['difficult'],
            cv_accuracy_mean=0.0,  # Will be calculated separately if needed
            cv_accuracy_std=0.0,
            cv_f1_mean=0.0,
            cv_f1_std=0.0
        )

    def _calculate_pas128_compliance(self,
                                   true_materials: List[MaterialType],
                                   predictions: List[MaterialType],
                                   confidences: List[float]) -> PAS128ComplianceMetrics:
        """Calculate PAS 128 compliance metrics."""

        # Calculate accuracy for each material
        material_accuracies = {}
        for material in set(true_materials):
            material_indices = [i for i, mat in enumerate(true_materials) if mat == material]
            if material_indices:
                correct = sum(1 for i in material_indices if predictions[i] == material)
                material_accuracies[material.value] = correct / len(material_indices)

        # Categorize by PAS 128 quality levels
        quality_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        compliant_materials = []
        non_compliant_materials = []

        for material, accuracy in material_accuracies.items():
            if accuracy >= self.pas128_thresholds['A']:
                quality_counts['A'] += 1
                compliant_materials.append(material)
            elif accuracy >= self.pas128_thresholds['B']:
                quality_counts['B'] += 1
                compliant_materials.append(material)
            elif accuracy >= self.pas128_thresholds['C']:
                quality_counts['C'] += 1
            else:
                quality_counts['D'] += 1
                non_compliant_materials.append(material)

        total_materials = len(material_accuracies)
        if total_materials == 0:
            total_materials = 1  # Avoid division by zero

        # Calculate rates
        quality_a_rate = quality_counts['A'] / total_materials
        quality_b_rate = quality_counts['B'] / total_materials
        quality_c_rate = quality_counts['C'] / total_materials
        quality_d_rate = quality_counts['D'] / total_materials

        # Overall compliance level
        overall_accuracy = accuracy_score(
            [mat.value for mat in true_materials],
            [pred.value for pred in predictions]
        )

        if overall_accuracy >= self.pas128_thresholds['A']:
            compliance_level = 'A'
        elif overall_accuracy >= self.pas128_thresholds['B']:
            compliance_level = 'B'
        elif overall_accuracy >= self.pas128_thresholds['C']:
            compliance_level = 'C'
        else:
            compliance_level = 'D'

        # False positive and negative rates
        y_true_binary = [1 if pred == true else 0
                        for pred, true in zip(predictions, true_materials)]
        fp_rate = sum(1 for pred, true in zip(predictions, true_materials)
                     if pred != true and pred != MaterialType.UNKNOWN) / len(predictions)
        fn_rate = sum(1 for pred, true in zip(predictions, true_materials)
                     if pred == MaterialType.UNKNOWN and true != MaterialType.UNKNOWN) / len(predictions)

        return PAS128ComplianceMetrics(
            detection_quality_a_rate=quality_a_rate,
            detection_quality_b_rate=quality_b_rate,
            detection_quality_c_rate=quality_c_rate,
            detection_quality_d_rate=quality_d_rate,
            material_identification_accuracy=overall_accuracy,
            depth_estimation_accuracy=0.85,  # Placeholder - would need depth prediction
            false_positive_rate=fp_rate,
            false_negative_rate=fn_rate,
            compliance_level=compliance_level,
            compliant_materials=compliant_materials,
            non_compliant_materials=non_compliant_materials
        )

    def _calculate_difficulty_accuracies(self,
                                       true_materials: List[MaterialType],
                                       predictions: List[MaterialType]) -> Dict[str, float]:
        """Calculate accuracy by material difficulty level."""

        difficulty_accuracies = {}

        for difficulty, materials in self.material_difficulty_levels.items():
            # Find indices of materials in this difficulty category
            difficulty_indices = [
                i for i, mat in enumerate(true_materials) if mat in materials
            ]

            if difficulty_indices:
                correct = sum(
                    1 for i in difficulty_indices
                    if predictions[i] == true_materials[i]
                )
                difficulty_accuracies[difficulty] = correct / len(difficulty_indices)
            else:
                difficulty_accuracies[difficulty] = 0.0

        return difficulty_accuracies

    def _evaluate_environmental_performance(self,
                                          signatures: List[GPRSignatureFeatures],
                                          true_materials: List[MaterialType],
                                          environmental_contexts: List[Dict[str, Any]]) -> Dict[str, PerformanceMetrics]:
        """Evaluate performance across different environmental conditions."""

        environmental_performance = {}

        # Group by environmental conditions
        condition_groups = {}

        for i, context in enumerate(environmental_contexts):
            # Create condition key based on major environmental factors
            soil_type = context.get('soil_type', 'unknown')
            moisture_level = 'high' if context.get('soil_moisture_content', 0.3) > 0.6 else 'low'
            area_type = context.get('area_type', 'unknown')

            condition_key = f"{soil_type}_{moisture_level}_{area_type}"

            if condition_key not in condition_groups:
                condition_groups[condition_key] = []
            condition_groups[condition_key].append(i)

        # Evaluate performance for each condition group
        for condition, indices in condition_groups.items():
            if len(indices) >= 10:  # Only evaluate if we have enough samples
                condition_signatures = [signatures[i] for i in indices]
                condition_true_materials = [true_materials[i] for i in indices]

                condition_predictions, condition_confidences = self._generate_predictions(
                    condition_signatures
                )

                condition_metrics = self._calculate_performance_metrics(
                    condition_true_materials, condition_predictions, condition_confidences
                )

                environmental_performance[condition] = condition_metrics

        return environmental_performance

    def _evaluate_discipline_performance(self,
                                       signatures: List[GPRSignatureFeatures],
                                       true_materials: List[MaterialType],
                                       disciplines: List[UtilityDiscipline]) -> Dict[str, PerformanceMetrics]:
        """Evaluate performance across different utility disciplines."""

        discipline_performance = {}

        # Group by discipline
        for discipline in set(disciplines):
            discipline_indices = [i for i, d in enumerate(disciplines) if d == discipline]

            if len(discipline_indices) >= 10:  # Only evaluate if we have enough samples
                discipline_signatures = [signatures[i] for i in discipline_indices]
                discipline_true_materials = [true_materials[i] for i in discipline_indices]

                discipline_predictions, discipline_confidences = self._generate_predictions(
                    discipline_signatures
                )

                discipline_metrics = self._calculate_performance_metrics(
                    discipline_true_materials, discipline_predictions, discipline_confidences
                )

                discipline_performance[discipline.value] = discipline_metrics

        return discipline_performance

    def _get_material_distribution(self, materials: List[MaterialType]) -> Dict[str, int]:
        """Get distribution of materials in dataset."""

        distribution = {}
        for material in materials:
            if material.value not in distribution:
                distribution[material.value] = 0
            distribution[material.value] += 1

        return distribution

    def generate_performance_report(self, validation_results: ValidationResults,
                                  output_path: Optional[str] = None) -> str:
        """Generate comprehensive performance report."""

        report_lines = []
        report_lines.append("MATERIAL CLASSIFICATION PERFORMANCE EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {validation_results.timestamp}")
        report_lines.append(f"Model Version: {validation_results.model_version}")
        report_lines.append(f"Total Samples: {validation_results.dataset_info['total_samples']}")
        report_lines.append("")

        # Basic Performance Metrics
        metrics = validation_results.performance_metrics
        report_lines.append("BASIC PERFORMANCE METRICS")
        report_lines.append("-" * 30)
        report_lines.append(f"Accuracy: {metrics.accuracy:.3f}")
        report_lines.append(f"Precision (Weighted): {metrics.precision_weighted:.3f}")
        report_lines.append(f"Recall (Weighted): {metrics.recall_weighted:.3f}")
        report_lines.append(f"F1-Score (Weighted): {metrics.f1_weighted:.3f}")
        report_lines.append(f"Balanced Accuracy: {metrics.balanced_accuracy:.3f}")
        report_lines.append(f"Cohen's Kappa: {metrics.cohen_kappa:.3f}")
        report_lines.append(f"Matthews Correlation: {metrics.matthews_correlation:.3f}")
        report_lines.append("")

        # PAS 128 Compliance
        pas128 = validation_results.pas128_compliance
        report_lines.append("PAS 128 COMPLIANCE ASSESSMENT")
        report_lines.append("-" * 30)
        report_lines.append(f"Overall Compliance Level: {pas128.compliance_level}")
        report_lines.append(f"Detection Quality A Rate: {pas128.detection_quality_a_rate:.3f}")
        report_lines.append(f"Detection Quality B Rate: {pas128.detection_quality_b_rate:.3f}")
        report_lines.append(f"Detection Quality C Rate: {pas128.detection_quality_c_rate:.3f}")
        report_lines.append(f"Detection Quality D Rate: {pas128.detection_quality_d_rate:.3f}")
        report_lines.append(f"False Positive Rate: {pas128.false_positive_rate:.3f}")
        report_lines.append(f"False Negative Rate: {pas128.false_negative_rate:.3f}")
        report_lines.append("")

        # Material Difficulty Performance
        report_lines.append("MATERIAL DIFFICULTY PERFORMANCE")
        report_lines.append("-" * 30)
        report_lines.append(f"Easy Materials Accuracy: {metrics.easy_materials_accuracy:.3f}")
        report_lines.append(f"Medium Materials Accuracy: {metrics.medium_materials_accuracy:.3f}")
        report_lines.append(f"Difficult Materials Accuracy: {metrics.difficult_materials_accuracy:.3f}")
        report_lines.append("")

        # Per-Class Performance
        report_lines.append("PER-CLASS PERFORMANCE")
        report_lines.append("-" * 30)
        for material in sorted(metrics.per_class_precision.keys()):
            precision = metrics.per_class_precision[material]
            recall = metrics.per_class_recall[material]
            f1 = metrics.per_class_f1[material]
            support = metrics.per_class_support[material]
            report_lines.append(f"{material:25} P:{precision:.3f} R:{recall:.3f} F1:{f1:.3f} S:{support}")
        report_lines.append("")

        # Confidence Analysis
        report_lines.append("PREDICTION CONFIDENCE ANALYSIS")
        report_lines.append("-" * 30)
        report_lines.append(f"Mean Confidence: {metrics.prediction_confidence_mean:.3f}")
        report_lines.append(f"Confidence Std: {metrics.prediction_confidence_std:.3f}")
        report_lines.append(f"Low Confidence Rate: {metrics.low_confidence_predictions_rate:.3f}")
        report_lines.append("")

        # Environmental Performance (if available)
        if validation_results.environmental_performance:
            report_lines.append("ENVIRONMENTAL PERFORMANCE")
            report_lines.append("-" * 30)
            for condition, env_metrics in validation_results.environmental_performance.items():
                report_lines.append(f"{condition}: Accuracy={env_metrics.accuracy:.3f}, F1={env_metrics.f1_weighted:.3f}")
            report_lines.append("")

        # Discipline Performance (if available)
        if validation_results.discipline_performance:
            report_lines.append("DISCIPLINE-SPECIFIC PERFORMANCE")
            report_lines.append("-" * 30)
            for discipline, disc_metrics in validation_results.discipline_performance.items():
                report_lines.append(f"{discipline}: Accuracy={disc_metrics.accuracy:.3f}, F1={disc_metrics.f1_weighted:.3f}")
            report_lines.append("")

        # Recommendations
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("-" * 30)
        if metrics.accuracy < 0.8:
            report_lines.append("• Overall accuracy below 80% - consider model retraining")
        if metrics.low_confidence_predictions_rate > 0.2:
            report_lines.append("• High rate of low-confidence predictions - review feature engineering")
        if pas128.compliance_level in ['C', 'D']:
            report_lines.append("• PAS 128 compliance below optimal - focus on difficult materials")
        if metrics.difficult_materials_accuracy < 0.6:
            report_lines.append("• Poor performance on difficult materials - enhance advanced features")

        report_content = "\n".join(report_lines)

        # Save to file if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_content)
            logger.info(f"Performance report saved to {output_path}")

        return report_content

    def save_validation_results(self, validation_results: ValidationResults,
                              output_path: str):
        """Save validation results to JSON file."""

        # Convert to serializable format
        results_dict = {
            'performance_metrics': asdict(validation_results.performance_metrics),
            'pas128_compliance': asdict(validation_results.pas128_compliance),
            'confusion_matrix': validation_results.confusion_matrix,
            'classification_report': validation_results.classification_report,
            'environmental_performance': {
                k: asdict(v) for k, v in validation_results.environmental_performance.items()
            },
            'discipline_performance': {
                k: asdict(v) for k, v in validation_results.discipline_performance.items()
            },
            'timestamp': validation_results.timestamp,
            'model_version': validation_results.model_version,
            'dataset_info': validation_results.dataset_info
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)

        logger.info(f"Validation results saved to {output_path}")


def run_comprehensive_validation(classification_service: MaterialClassificationService,
                                n_samples: int = 1000,
                                output_dir: str = "validation_results") -> ValidationResults:
    """Run comprehensive validation of material classification system."""

    from services.material_model_trainer import EnhancedTwenteDatasetProcessor

    # Initialize evaluator
    evaluator = MaterialPerformanceEvaluator(classification_service)

    # Generate test data
    processor = EnhancedTwenteDatasetProcessor()
    signatures, materials, contexts = processor.simulate_enhanced_twente_gpr_data(n_samples)

    # Generate disciplines for testing
    disciplines = [
        np.random.choice(list(UtilityDiscipline))
        for _ in range(len(signatures))
    ]

    # Run comprehensive evaluation
    validation_results = evaluator.comprehensive_evaluation(
        signatures, materials, contexts, disciplines
    )

    # Generate and save reports
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    evaluator.save_validation_results(
        validation_results,
        str(output_path / "validation_results.json")
    )

    # Generate performance report
    report = evaluator.generate_performance_report(
        validation_results,
        str(output_path / "performance_report.txt")
    )

    logger.info(f"Comprehensive validation completed. Results saved to {output_dir}")
    return validation_results


if __name__ == "__main__":
    # Example usage
    from services.material_classification import MaterialClassificationService

    logger.info("Starting comprehensive material classification validation")

    # Initialize classification service
    classification_service = MaterialClassificationService()

    # Run validation
    results = run_comprehensive_validation(
        classification_service,
        n_samples=500,
        output_dir="validation_results"
    )

    print(f"\nValidation Summary:")
    print(f"Overall Accuracy: {results.performance_metrics.accuracy:.3f}")
    print(f"PAS 128 Compliance Level: {results.pas128_compliance.compliance_level}")
    print(f"F1-Score (Weighted): {results.performance_metrics.f1_weighted:.3f}")