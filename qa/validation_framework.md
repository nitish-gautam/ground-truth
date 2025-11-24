# Phase 1A Quality Assurance Framework
## GPR Signal Processing Validation and Performance Benchmarking

---

## Validation Framework Overview

### Validation Strategy Layers

```
Validation Framework Architecture:
├── Signal Processing Validation
│   ├── Time-Zero Correction Accuracy
│   ├── Noise Removal Effectiveness
│   ├── Frequency Analysis Validation
│   └── Feature Extraction Quality
├── Detection Algorithm Validation
│   ├── Utility Detection Accuracy
│   ├── Depth Estimation Precision
│   ├── Classification Performance
│   └── Confidence Calibration
├── Ground Truth Validation
│   ├── Twente Dataset Benchmarking
│   ├── Manual Interpretation Comparison
│   ├── Field Validation Campaigns
│   └── Cross-validation Studies
├── Environmental Impact Assessment
│   ├── Soil Type Performance Analysis
│   ├── Weather Condition Impact
│   ├── Surface Material Effects
│   └── Contamination Factor Analysis
└── Compliance Validation
    ├── PAS 128 Standard Compliance
    ├── Quality Level Assignment
    ├── Documentation Completeness
    └── Audit Trail Verification
```

---

## 1. Signal Processing Validation Framework

### Time-Zero Correction Validation

```python
class TimeZeroCorrectionValidator:
    """
    Validate accuracy of time-zero correction algorithms
    """

    def __init__(self):
        self.reference_datasets = self._load_reference_datasets()
        self.tolerance_ns = 0.5  # nanoseconds

    async def validate_time_zero_correction(
        self,
        original_signal: np.ndarray,
        corrected_signal: np.ndarray,
        expected_shift: float = None
    ) -> ValidationResult:

        # Test 1: Surface reflection removal
        surface_removal_score = self._assess_surface_reflection_removal(
            original_signal, corrected_signal
        )

        # Test 2: Air gap correction
        air_gap_correction_score = self._assess_air_gap_correction(
            corrected_signal
        )

        # Test 3: Depth calibration accuracy
        depth_calibration_score = self._assess_depth_calibration(
            corrected_signal, expected_shift
        )

        # Test 4: Signal preservation
        signal_preservation_score = self._assess_signal_preservation(
            original_signal, corrected_signal
        )

        return ValidationResult(
            overall_score=np.mean([
                surface_removal_score,
                air_gap_correction_score,
                depth_calibration_score,
                signal_preservation_score
            ]),
            detailed_scores={
                "surface_reflection_removal": surface_removal_score,
                "air_gap_correction": air_gap_correction_score,
                "depth_calibration": depth_calibration_score,
                "signal_preservation": signal_preservation_score
            },
            passed=all(score >= 0.8 for score in [
                surface_removal_score, air_gap_correction_score,
                depth_calibration_score, signal_preservation_score
            ])
        )

    def _assess_surface_reflection_removal(
        self,
        original: np.ndarray,
        corrected: np.ndarray
    ) -> float:
        """
        Assess how well surface reflection was removed
        """
        # Calculate energy reduction in first 10 samples
        original_surface_energy = np.sum(original[:10, :]**2)
        corrected_surface_energy = np.sum(corrected[:10, :]**2)

        reduction_ratio = 1 - (corrected_surface_energy / original_surface_energy)
        return min(1.0, max(0.0, reduction_ratio))

    def _assess_air_gap_correction(self, corrected: np.ndarray) -> float:
        """
        Verify air gap has been properly corrected
        """
        # Check if first arrivals are now at time zero
        first_break_times = []
        for trace in corrected.T:
            first_break = self._detect_first_break(trace)
            first_break_times.append(first_break)

        # Calculate standard deviation of first break times
        std_first_breaks = np.std(first_break_times)

        # Score based on consistency (lower std = better score)
        return max(0.0, 1.0 - (std_first_breaks / 5.0))  # 5 samples tolerance

    def _assess_depth_calibration(
        self,
        corrected: np.ndarray,
        expected_shift: float
    ) -> float:
        """
        Validate depth calibration accuracy
        """
        if expected_shift is None:
            return 1.0  # Cannot validate without ground truth

        # Calculate actual shift applied
        actual_shift = self._calculate_applied_shift(corrected)

        # Score based on accuracy
        error = abs(actual_shift - expected_shift)
        return max(0.0, 1.0 - (error / self.tolerance_ns))
```

### Noise Removal Validation

```python
class NoiseRemovalValidator:
    """
    Validate effectiveness of noise removal algorithms
    """

    async def validate_noise_removal(
        self,
        original_signal: np.ndarray,
        denoised_signal: np.ndarray,
        noise_type: str = "background_removal"
    ) -> ValidationResult:

        # Test 1: SNR improvement
        snr_improvement = self._calculate_snr_improvement(
            original_signal, denoised_signal
        )

        # Test 2: Signal preservation
        signal_preservation = self._assess_signal_preservation(
            original_signal, denoised_signal
        )

        # Test 3: Noise reduction effectiveness
        noise_reduction = self._assess_noise_reduction(
            original_signal, denoised_signal, noise_type
        )

        # Test 4: Artifact introduction check
        artifact_score = self._check_for_artifacts(denoised_signal)

        return ValidationResult(
            overall_score=np.mean([
                snr_improvement, signal_preservation,
                noise_reduction, artifact_score
            ]),
            detailed_scores={
                "snr_improvement": snr_improvement,
                "signal_preservation": signal_preservation,
                "noise_reduction": noise_reduction,
                "artifact_introduction": artifact_score
            },
            passed=all(score >= 0.7 for score in [
                snr_improvement, signal_preservation,
                noise_reduction, artifact_score
            ])
        )

    def _calculate_snr_improvement(
        self,
        original: np.ndarray,
        denoised: np.ndarray
    ) -> float:
        """
        Calculate SNR improvement ratio
        """
        # Estimate signal and noise power
        original_snr = self._estimate_snr(original)
        denoised_snr = self._estimate_snr(denoised)

        # SNR improvement in dB
        snr_improvement_db = denoised_snr - original_snr

        # Normalize to 0-1 scale (0-20dB improvement = 0-1 score)
        return min(1.0, max(0.0, snr_improvement_db / 20.0))

    def _assess_noise_reduction(
        self,
        original: np.ndarray,
        denoised: np.ndarray,
        noise_type: str
    ) -> float:
        """
        Assess specific noise reduction effectiveness
        """
        if noise_type == "background_removal":
            return self._assess_background_removal(original, denoised)
        elif noise_type == "random_noise":
            return self._assess_random_noise_removal(original, denoised)
        elif noise_type == "coherent_noise":
            return self._assess_coherent_noise_removal(original, denoised)
        else:
            return 0.5  # Unknown noise type

    def _assess_background_removal(
        self,
        original: np.ndarray,
        denoised: np.ndarray
    ) -> float:
        """
        Assess horizontal background noise removal
        """
        # Calculate trace-to-trace similarity reduction
        original_similarity = self._calculate_trace_similarity(original)
        denoised_similarity = self._calculate_trace_similarity(denoised)

        # Good background removal should reduce trace similarity
        similarity_reduction = original_similarity - denoised_similarity
        return min(1.0, max(0.0, similarity_reduction))
```

### Feature Extraction Quality Validation

```python
class FeatureExtractionValidator:
    """
    Validate quality and consistency of extracted features
    """

    async def validate_feature_extraction(
        self,
        signal_data: np.ndarray,
        extracted_features: Dict[str, Any],
        feature_type: str
    ) -> ValidationResult:

        validation_results = {}

        if feature_type == "hyperbola_detection":
            validation_results.update(
                await self._validate_hyperbola_features(
                    signal_data, extracted_features
                )
            )

        elif feature_type == "amplitude_analysis":
            validation_results.update(
                await self._validate_amplitude_features(
                    signal_data, extracted_features
                )
            )

        elif feature_type == "frequency_analysis":
            validation_results.update(
                await self._validate_frequency_features(
                    signal_data, extracted_features
                )
            )

        overall_score = np.mean(list(validation_results.values()))

        return ValidationResult(
            overall_score=overall_score,
            detailed_scores=validation_results,
            passed=overall_score >= 0.8
        )

    async def _validate_hyperbola_features(
        self,
        signal_data: np.ndarray,
        features: Dict[str, Any]
    ) -> Dict[str, float]:

        results = {}

        # Validate hyperbola detection consistency
        if "hyperbola_apex_depth" in features:
            depth = features["hyperbola_apex_depth"]
            # Check if depth is within reasonable range
            results["depth_reasonableness"] = 1.0 if 0.1 <= depth <= 3.0 else 0.0

        # Validate hyperbola geometry
        if "hyperbola_curvature" in features:
            curvature = features["hyperbola_curvature"]
            # Check if curvature is physically reasonable
            results["curvature_validity"] = 1.0 if 0.1 <= curvature <= 10.0 else 0.0

        # Validate symmetry score
        if "hyperbola_symmetry" in features:
            symmetry = features["hyperbola_symmetry"]
            results["symmetry_validity"] = 1.0 if 0.0 <= symmetry <= 1.0 else 0.0

        return results

    async def _validate_amplitude_features(
        self,
        signal_data: np.ndarray,
        features: Dict[str, Any]
    ) -> Dict[str, float]:

        results = {}

        # Validate statistical consistency
        if "mean_amplitude" in features and "peak_amplitude" in features:
            mean_amp = features["mean_amplitude"]
            peak_amp = features["peak_amplitude"]

            # Peak should be >= mean
            results["amplitude_consistency"] = 1.0 if peak_amp >= mean_amp else 0.0

        # Validate against actual signal statistics
        actual_mean = np.mean(signal_data)
        actual_peak = np.max(signal_data)

        if "mean_amplitude" in features:
            mean_error = abs(features["mean_amplitude"] - actual_mean) / actual_mean
            results["mean_accuracy"] = max(0.0, 1.0 - mean_error)

        if "peak_amplitude" in features:
            peak_error = abs(features["peak_amplitude"] - actual_peak) / actual_peak
            results["peak_accuracy"] = max(0.0, 1.0 - peak_error)

        return results
```

---

## 2. Detection Algorithm Validation

### Utility Detection Accuracy Assessment

```python
class UtilityDetectionValidator:
    """
    Comprehensive validation of utility detection algorithms
    """

    def __init__(self):
        self.twente_ground_truth = self._load_twente_ground_truth()
        self.validation_thresholds = {
            "position_tolerance_m": 0.2,
            "depth_tolerance_m": 0.1,
            "confidence_threshold": 0.7
        }

    async def validate_detection_performance(
        self,
        detections: List[DetectedUtility],
        ground_truth: List[GroundTruthUtility],
        survey_metadata: SurveyMetadata
    ) -> DetectionValidationResult:

        # Match detections with ground truth
        matches = self._match_detections_to_ground_truth(
            detections, ground_truth
        )

        # Calculate performance metrics
        metrics = self._calculate_detection_metrics(matches)

        # Environmental impact analysis
        environmental_impact = await self._analyze_environmental_impact(
            detections, ground_truth, survey_metadata
        )

        # Confidence calibration assessment
        confidence_calibration = self._assess_confidence_calibration(matches)

        return DetectionValidationResult(
            metrics=metrics,
            environmental_impact=environmental_impact,
            confidence_calibration=confidence_calibration,
            detailed_matches=matches
        )

    def _match_detections_to_ground_truth(
        self,
        detections: List[DetectedUtility],
        ground_truth: List[GroundTruthUtility]
    ) -> List[DetectionMatch]:

        matches = []
        used_ground_truth = set()

        for detection in detections:
            best_match = None
            min_distance = float('inf')

            for i, gt_utility in enumerate(ground_truth):
                if i in used_ground_truth:
                    continue

                # Calculate spatial distance
                position_distance = self._calculate_position_distance(
                    detection, gt_utility
                )
                depth_distance = abs(detection.depth_m - gt_utility.depth_m)

                # Combined distance metric
                total_distance = (
                    position_distance / self.validation_thresholds["position_tolerance_m"] +
                    depth_distance / self.validation_thresholds["depth_tolerance_m"]
                )

                if total_distance < min_distance and total_distance <= 2.0:  # Within tolerance
                    min_distance = total_distance
                    best_match = i

            if best_match is not None:
                used_ground_truth.add(best_match)
                matches.append(DetectionMatch(
                    detection=detection,
                    ground_truth=ground_truth[best_match],
                    match_type="true_positive",
                    spatial_error=self._calculate_position_distance(
                        detection, ground_truth[best_match]
                    ),
                    depth_error=detection.depth_m - ground_truth[best_match].depth_m
                ))
            else:
                matches.append(DetectionMatch(
                    detection=detection,
                    ground_truth=None,
                    match_type="false_positive"
                ))

        # Add false negatives
        for i, gt_utility in enumerate(ground_truth):
            if i not in used_ground_truth:
                matches.append(DetectionMatch(
                    detection=None,
                    ground_truth=gt_utility,
                    match_type="false_negative"
                ))

        return matches

    def _calculate_detection_metrics(
        self,
        matches: List[DetectionMatch]
    ) -> DetectionMetrics:

        true_positives = [m for m in matches if m.match_type == "true_positive"]
        false_positives = [m for m in matches if m.match_type == "false_positive"]
        false_negatives = [m for m in matches if m.match_type == "false_negative"]

        tp_count = len(true_positives)
        fp_count = len(false_positives)
        fn_count = len(false_negatives)

        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Depth estimation accuracy
        depth_errors = [abs(m.depth_error) for m in true_positives if m.depth_error is not None]
        mean_depth_error = np.mean(depth_errors) if depth_errors else 0
        depth_accuracy = len([e for e in depth_errors if e <= 0.1]) / len(depth_errors) if depth_errors else 0

        # Position accuracy
        position_errors = [m.spatial_error for m in true_positives if m.spatial_error is not None]
        mean_position_error = np.mean(position_errors) if position_errors else 0
        position_accuracy = len([e for e in position_errors if e <= 0.2]) / len(position_errors) if position_errors else 0

        return DetectionMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            true_positives=tp_count,
            false_positives=fp_count,
            false_negatives=fn_count,
            mean_depth_error=mean_depth_error,
            depth_accuracy=depth_accuracy,
            mean_position_error=mean_position_error,
            position_accuracy=position_accuracy
        )
```

### Performance Benchmarking Framework

```python
class PerformanceBenchmarkingFramework:
    """
    Comprehensive benchmarking against established baselines
    """

    def __init__(self):
        self.benchmarks = self._load_benchmark_datasets()
        self.baseline_methods = self._initialize_baseline_methods()

    async def run_comprehensive_benchmark(
        self,
        detection_method: str,
        test_surveys: List[GPRSurvey]
    ) -> BenchmarkResult:

        results = {}

        # Benchmark against Twente dataset
        twente_results = await self._benchmark_against_twente(
            detection_method, test_surveys
        )
        results["twente_dataset"] = twente_results

        # Benchmark against manual interpretation
        manual_interpretation_results = await self._benchmark_against_manual_interpretation(
            detection_method, test_surveys
        )
        results["manual_interpretation"] = manual_interpretation_results

        # Benchmark against existing methods
        comparative_results = await self._benchmark_against_baselines(
            detection_method, test_surveys
        )
        results["baseline_comparison"] = comparative_results

        # Statistical significance testing
        statistical_analysis = await self._perform_statistical_analysis(results)

        return BenchmarkResult(
            method_name=detection_method,
            benchmark_results=results,
            statistical_analysis=statistical_analysis,
            overall_ranking=self._calculate_overall_ranking(results)
        )

    async def _benchmark_against_twente(
        self,
        detection_method: str,
        test_surveys: List[GPRSurvey]
    ) -> TwenteBenchmarkResult:

        # Filter Twente surveys from test set
        twente_surveys = [s for s in test_surveys if s.location_id.count('.') == 1]

        total_metrics = []
        stratified_results = {}

        for survey in twente_surveys:
            # Run detection
            detections = await self._run_detection_method(detection_method, survey)

            # Get ground truth
            ground_truth = await self._get_twente_ground_truth(survey.location_id)

            # Validate
            validation_result = await self._validate_detections(
                detections, ground_truth, survey
            )

            total_metrics.append(validation_result.metrics)

            # Stratify by environmental conditions
            soil_type = survey.ground_condition
            if soil_type not in stratified_results:
                stratified_results[soil_type] = []
            stratified_results[soil_type].append(validation_result.metrics)

        # Calculate aggregate metrics
        aggregate_metrics = self._aggregate_metrics(total_metrics)

        # Calculate stratified performance
        stratified_performance = {}
        for condition, metrics_list in stratified_results.items():
            stratified_performance[condition] = self._aggregate_metrics(metrics_list)

        return TwenteBenchmarkResult(
            total_surveys=len(twente_surveys),
            aggregate_performance=aggregate_metrics,
            stratified_performance=stratified_performance,
            individual_results=total_metrics
        )

    async def _benchmark_against_manual_interpretation(
        self,
        detection_method: str,
        test_surveys: List[GPRSurvey]
    ) -> ManualInterpretationBenchmark:

        # Select subset for manual interpretation (expensive process)
        sample_surveys = random.sample(test_surveys, min(20, len(test_surveys)))

        manual_vs_algorithm_results = []

        for survey in sample_surveys:
            # Run algorithm detection
            algorithm_detections = await self._run_detection_method(
                detection_method, survey
            )

            # Get manual interpretation results
            manual_detections = await self._get_manual_interpretation(survey.id)

            # Compare algorithm vs manual
            comparison_result = await self._compare_detection_methods(
                algorithm_detections, manual_detections, survey
            )

            manual_vs_algorithm_results.append(comparison_result)

        return ManualInterpretationBenchmark(
            surveys_compared=len(sample_surveys),
            algorithm_vs_manual=manual_vs_algorithm_results,
            inter_operator_variability=await self._assess_inter_operator_variability(),
            algorithm_consistency=await self._assess_algorithm_consistency(sample_surveys)
        )

    async def _perform_statistical_analysis(
        self,
        benchmark_results: Dict[str, Any]
    ) -> StatisticalAnalysis:

        # Collect all performance metrics
        all_metrics = []
        for benchmark_type, results in benchmark_results.items():
            if hasattr(results, 'aggregate_performance'):
                all_metrics.append(results.aggregate_performance)

        # Confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(all_metrics)

        # Statistical significance tests
        significance_tests = await self._perform_significance_tests(benchmark_results)

        # Effect size calculations
        effect_sizes = self._calculate_effect_sizes(benchmark_results)

        return StatisticalAnalysis(
            confidence_intervals=confidence_intervals,
            significance_tests=significance_tests,
            effect_sizes=effect_sizes,
            sample_sizes=self._get_sample_sizes(benchmark_results)
        )
```

---

## 3. Ground Truth Validation Methods

### Twente Dataset Ground Truth Integration

```python
class TwenteGroundTruthProcessor:
    """
    Process and validate against Twente dataset ground truth
    """

    async def process_twente_ground_truth(
        self,
        location_id: str
    ) -> List[GroundTruthUtility]:

        # Load metadata for this location
        metadata = await self._load_location_metadata(location_id)

        # Parse utility information from metadata
        utilities = []

        if metadata['Utility discipline']:
            utility_types = metadata['Utility discipline'].split('\n')
            materials = metadata['Utility material'].split('\n') if metadata['Utility material'] else []
            diameters = metadata['Utility diameter'].split('\n') if metadata['Utility diameter'] else []

            for i, utility_type in enumerate(utility_types):
                if utility_type.strip():
                    utility = GroundTruthUtility(
                        location_id=location_id,
                        utility_type=utility_type.strip(),
                        material=materials[i].strip() if i < len(materials) else None,
                        diameter_mm=int(diameters[i]) if i < len(diameters) and diameters[i].strip() else None,
                        validated=True,
                        validation_method="trial_trench"
                    )
                    utilities.append(utility)

        return utilities

    async def validate_twente_ground_truth_quality(self) -> GroundTruthQualityReport:

        # Load all Twente metadata
        all_metadata = await self._load_all_twente_metadata()

        quality_metrics = {
            "completeness": 0,
            "consistency": 0,
            "accuracy": 0,
            "spatial_coverage": 0
        }

        # Assess completeness
        complete_records = 0
        for record in all_metadata:
            if all([
                record.get('Utility discipline'),
                record.get('Amount of utilities'),
                record.get('Ground condition'),
                record.get('Weather condition')
            ]):
                complete_records += 1

        quality_metrics["completeness"] = complete_records / len(all_metadata)

        # Assess consistency
        consistency_score = await self._assess_metadata_consistency(all_metadata)
        quality_metrics["consistency"] = consistency_score

        # Assess spatial coverage
        spatial_coverage = await self._assess_spatial_coverage(all_metadata)
        quality_metrics["spatial_coverage"] = spatial_coverage

        return GroundTruthQualityReport(
            total_records=len(all_metadata),
            quality_metrics=quality_metrics,
            recommendations=self._generate_quality_recommendations(quality_metrics)
        )
```

### Field Validation Campaign Framework

```python
class FieldValidationCampaign:
    """
    Design and execute field validation campaigns
    """

    async def design_validation_campaign(
        self,
        project_id: UUID,
        survey_locations: List[UUID],
        validation_budget: float
    ) -> ValidationCampaignPlan:

        # Prioritize locations for validation
        prioritized_locations = await self._prioritize_validation_locations(
            survey_locations, validation_budget
        )

        # Design validation methodology
        validation_methods = await self._select_validation_methods(
            prioritized_locations
        )

        # Estimate costs and timeline
        cost_estimate = await self._estimate_validation_costs(
            prioritized_locations, validation_methods
        )

        # Generate validation plan
        campaign_plan = ValidationCampaignPlan(
            project_id=project_id,
            locations_to_validate=prioritized_locations,
            validation_methods=validation_methods,
            estimated_cost=cost_estimate,
            estimated_duration_days=cost_estimate.duration_days,
            required_equipment=cost_estimate.equipment_list,
            personnel_requirements=cost_estimate.personnel_requirements
        )

        return campaign_plan

    async def _prioritize_validation_locations(
        self,
        locations: List[UUID],
        budget: float
    ) -> List[PrioritizedLocation]:

        prioritized = []

        for location_id in locations:
            # Get survey data
            survey = await self._get_survey_data(location_id)

            # Calculate priority score
            priority_score = await self._calculate_priority_score(survey)

            prioritized.append(PrioritizedLocation(
                location_id=location_id,
                priority_score=priority_score,
                validation_cost=await self._estimate_location_validation_cost(survey),
                expected_value=priority_score * budget / 100  # Simplified calculation
            ))

        # Sort by value/cost ratio
        prioritized.sort(key=lambda x: x.expected_value / x.validation_cost, reverse=True)

        # Select locations within budget
        selected_locations = []
        total_cost = 0

        for location in prioritized:
            if total_cost + location.validation_cost <= budget:
                selected_locations.append(location)
                total_cost += location.validation_cost
            else:
                break

        return selected_locations

    async def _calculate_priority_score(self, survey: GPRSurvey) -> float:

        score = 0

        # Factor 1: Detection uncertainty
        if survey.confidence_score < 0.8:
            score += 30

        # Factor 2: Critical utility types
        critical_utilities = ['gas', 'electricity', 'water']
        detected_utilities = await self._get_detected_utilities(survey.id)
        for utility in detected_utilities:
            if utility.utility_discipline.lower() in critical_utilities:
                score += 20

        # Factor 3: Environmental complexity
        if survey.utility_crossing:
            score += 15
        if survey.rubble_presence or survey.tree_roots_presence:
            score += 10

        # Factor 4: Depth range variety
        utility_depths = [u.depth_m for u in detected_utilities]
        if len(set([int(d) for d in utility_depths])) > 2:  # Multiple depth ranges
            score += 10

        # Factor 5: Ground truth availability gap
        existing_validations = await self._count_existing_validations(survey.site_id)
        if existing_validations < 3:
            score += 15

        return min(100, score)  # Cap at 100
```

---

## 4. Performance Benchmarking System

### Automated Performance Monitoring

```python
class AutomatedPerformanceMonitor:
    """
    Continuous monitoring of detection algorithm performance
    """

    def __init__(self):
        self.performance_thresholds = {
            "precision": 0.85,
            "recall": 0.80,
            "f1_score": 0.82,
            "depth_accuracy": 0.90,
            "position_accuracy": 0.85
        }

    async def monitor_continuous_performance(
        self,
        model_id: UUID,
        monitoring_period_days: int = 30
    ) -> PerformanceMonitoringReport:

        # Get recent predictions and validations
        recent_predictions = await self._get_recent_predictions(
            model_id, monitoring_period_days
        )

        # Calculate performance metrics
        current_performance = await self._calculate_current_performance(
            recent_predictions
        )

        # Compare with historical performance
        historical_performance = await self._get_historical_performance(
            model_id, monitoring_period_days * 4  # 4x period for comparison
        )

        # Detect performance drift
        performance_drift = await self._detect_performance_drift(
            current_performance, historical_performance
        )

        # Generate alerts if needed
        alerts = await self._generate_performance_alerts(
            current_performance, performance_drift
        )

        return PerformanceMonitoringReport(
            model_id=model_id,
            monitoring_period=monitoring_period_days,
            current_performance=current_performance,
            historical_comparison=historical_performance,
            performance_drift=performance_drift,
            alerts=alerts,
            recommendations=await self._generate_performance_recommendations(
                current_performance, performance_drift
            )
        )

    async def _detect_performance_drift(
        self,
        current: PerformanceMetrics,
        historical: PerformanceMetrics
    ) -> PerformanceDriftAnalysis:

        drift_analysis = {}

        # Check each metric for significant drift
        for metric_name in ["precision", "recall", "f1_score", "depth_accuracy"]:
            current_value = getattr(current, metric_name)
            historical_value = getattr(historical, metric_name)

            # Calculate drift percentage
            drift_percentage = ((current_value - historical_value) / historical_value) * 100

            # Determine significance
            is_significant = abs(drift_percentage) > 5.0  # 5% threshold

            drift_analysis[metric_name] = {
                "current": current_value,
                "historical": historical_value,
                "drift_percentage": drift_percentage,
                "is_significant": is_significant,
                "direction": "improvement" if drift_percentage > 0 else "degradation"
            }

        return PerformanceDriftAnalysis(drift_analysis)

    async def _generate_performance_alerts(
        self,
        current_performance: PerformanceMetrics,
        drift_analysis: PerformanceDriftAnalysis
    ) -> List[PerformanceAlert]:

        alerts = []

        # Check for performance below thresholds
        for metric_name, threshold in self.performance_thresholds.items():
            current_value = getattr(current_performance, metric_name)
            if current_value < threshold:
                alerts.append(PerformanceAlert(
                    severity="warning",
                    metric=metric_name,
                    current_value=current_value,
                    threshold=threshold,
                    message=f"{metric_name} ({current_value:.3f}) below threshold ({threshold})"
                ))

        # Check for significant degradation
        for metric_name, drift_info in drift_analysis.analysis.items():
            if drift_info["is_significant"] and drift_info["direction"] == "degradation":
                alerts.append(PerformanceAlert(
                    severity="critical" if abs(drift_info["drift_percentage"]) > 10 else "warning",
                    metric=metric_name,
                    current_value=drift_info["current"],
                    threshold=drift_info["historical"],
                    message=f"Significant {metric_name} degradation: {drift_info['drift_percentage']:.1f}%"
                ))

        return alerts
```

### Cross-Validation Framework

```python
class CrossValidationFramework:
    """
    Comprehensive cross-validation for model robustness assessment
    """

    async def perform_stratified_cross_validation(
        self,
        dataset: GPRDataset,
        model_name: str,
        stratification_factors: List[str],
        k_folds: int = 5
    ) -> CrossValidationResult:

        # Stratify dataset based on specified factors
        stratified_folds = await self._create_stratified_folds(
            dataset, stratification_factors, k_folds
        )

        fold_results = []

        for fold_idx, (train_data, val_data) in enumerate(stratified_folds):

            # Train model on fold
            trained_model = await self._train_model_on_fold(
                model_name, train_data, fold_idx
            )

            # Validate on holdout
            validation_result = await self._validate_model_on_fold(
                trained_model, val_data, fold_idx
            )

            fold_results.append(validation_result)

        # Aggregate results across folds
        aggregated_results = await self._aggregate_fold_results(fold_results)

        # Calculate confidence intervals
        confidence_intervals = await self._calculate_cross_validation_confidence_intervals(
            fold_results
        )

        return CrossValidationResult(
            k_folds=k_folds,
            stratification_factors=stratification_factors,
            fold_results=fold_results,
            aggregated_performance=aggregated_results,
            confidence_intervals=confidence_intervals,
            model_stability=await self._assess_model_stability(fold_results)
        )

    async def _create_stratified_folds(
        self,
        dataset: GPRDataset,
        stratification_factors: List[str],
        k_folds: int
    ) -> List[Tuple[GPRDataset, GPRDataset]]:

        # Create stratification keys for each sample
        stratification_keys = []
        for sample in dataset.samples:
            key_parts = []
            for factor in stratification_factors:
                if factor == "ground_condition":
                    key_parts.append(sample.survey.ground_condition)
                elif factor == "utility_type":
                    key_parts.append(sample.primary_utility_type)
                elif factor == "depth_range":
                    depth_category = "shallow" if sample.max_depth < 1.0 else "deep"
                    key_parts.append(depth_category)
                elif factor == "weather_condition":
                    key_parts.append(sample.survey.weather_condition)

            stratification_keys.append("_".join(key_parts))

        # Use stratified k-fold splitting
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

        folds = []
        for train_idx, val_idx in skf.split(dataset.samples, stratification_keys):
            train_samples = [dataset.samples[i] for i in train_idx]
            val_samples = [dataset.samples[i] for i in val_idx]

            train_dataset = GPRDataset(samples=train_samples)
            val_dataset = GPRDataset(samples=val_samples)

            folds.append((train_dataset, val_dataset))

        return folds
```

This comprehensive validation framework provides:

1. **Signal Processing Validation** - Time-zero correction, noise removal, and feature extraction quality assessment
2. **Detection Algorithm Validation** - Utility detection accuracy, depth estimation precision, and confidence calibration
3. **Ground Truth Integration** - Twente dataset benchmarking and field validation campaigns
4. **Performance Monitoring** - Continuous monitoring with drift detection and alerting
5. **Cross-Validation** - Stratified cross-validation for robust model assessment
6. **Statistical Analysis** - Confidence intervals, significance testing, and effect size calculations

The framework ensures rigorous validation of all GPR processing components with automated monitoring and comprehensive benchmarking against established datasets.