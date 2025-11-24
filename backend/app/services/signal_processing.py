"""
Advanced GPR signal processing service
=====================================

Comprehensive signal processing pipeline for GPR data including filtering,
feature extraction, hyperbola detection, and environmental correlation analysis.
"""

import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import json

import numpy as np
import scipy.signal as signal
import scipy.fft as fft
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..core.config import settings
from ..core.logging_config import LoggerMixin
from ..models.gpr_data import GPRScan, GPRSignalData, GPRProcessingResult
from ..models.environmental import EnvironmentalData
from ..models.ml_analytics import FeatureVector


class GPRSignalProcessor(LoggerMixin):
    """Advanced GPR signal processing and analysis service."""

    def __init__(self):
        super().__init__()
        self.sampling_frequency = settings.SAMPLING_FREQUENCY
        self.time_window = settings.TIME_WINDOW
        self.depth_calibration = settings.DEPTH_CALIBRATION

    async def process_scan_comprehensive(
        self,
        db: AsyncSession,
        scan_id: str,
        processing_config: Dict[str, Any]
    ) -> GPRProcessingResult:
        """Comprehensive processing of a GPR scan."""
        self.log_signal_processing("comprehensive_scan_processing", 0, scan_id=scan_id)

        try:
            # Load scan and signal data
            scan = await self._load_scan_data(db, scan_id)
            if not scan:
                raise ValueError(f"Scan not found: {scan_id}")

            # Load raw signal data
            signal_data = await self._load_signal_data(db, scan_id)
            if signal_data is None or len(signal_data) == 0:
                raise ValueError(f"No signal data found for scan: {scan_id}")

            # Apply processing pipeline
            processing_steps = processing_config.get("steps", [
                "time_zero_correction",
                "noise_removal",
                "bandpass_filter",
                "gain_adjustment",
                "feature_extraction",
                "hyperbola_detection",
                "environmental_correlation"
            ])

            processed_results = {}
            current_data = signal_data

            for step in processing_steps:
                self.log_signal_processing(step, current_data.shape[0] if hasattr(current_data, 'shape') else 0)

                if step == "time_zero_correction":
                    current_data = await self._apply_time_zero_correction(current_data, processing_config)
                elif step == "noise_removal":
                    current_data = await self._apply_noise_removal(current_data, processing_config)
                elif step == "bandpass_filter":
                    current_data = await self._apply_bandpass_filter(current_data, processing_config)
                elif step == "gain_adjustment":
                    current_data = await self._apply_gain_adjustment(current_data, processing_config)
                elif step == "feature_extraction":
                    features = await self._extract_signal_features(current_data, processing_config)
                    processed_results["features"] = features
                elif step == "hyperbola_detection":
                    hyperbolas = await self._detect_hyperbolas(current_data, processing_config)
                    processed_results["hyperbolas"] = hyperbolas
                elif step == "environmental_correlation":
                    correlations = await self._analyze_environmental_correlation(
                        db, scan_id, current_data, processing_config
                    )
                    processed_results["environmental_correlations"] = correlations

            # Create processing result record
            processing_result = GPRProcessingResult(
                scan_id=scan_id,
                processing_algorithm="comprehensive_pipeline",
                processing_version="1.0",
                processing_timestamp=datetime.now(),
                parameters=processing_config,
                detected_features=processed_results.get("features", []),
                hyperbola_detections=processed_results.get("hyperbolas", []),
                overall_confidence=self._calculate_overall_confidence(processed_results),
                detection_count=len(processed_results.get("hyperbolas", [])),
                environmental_impact_score=processed_results.get("environmental_correlations", {}).get("impact_score"),
                status="completed"
            )

            db.add(processing_result)
            await db.commit()

            self.log_operation_complete("comprehensive_scan_processing", 0, **processed_results)

            return processing_result

        except Exception as e:
            await db.rollback()
            self.log_operation_error("comprehensive_scan_processing", e, scan_id=scan_id)
            raise

    async def _load_scan_data(self, db: AsyncSession, scan_id: str) -> Optional[GPRScan]:
        """Load scan metadata from database."""
        result = await db.execute(select(GPRScan).where(GPRScan.id == scan_id))
        return result.scalar_one_or_none()

    async def _load_signal_data(self, db: AsyncSession, scan_id: str) -> Optional[np.ndarray]:
        """Load signal data from database."""
        result = await db.execute(
            select(GPRSignalData).where(
                GPRSignalData.scan_id == scan_id,
                GPRSignalData.data_type == "raw"
            ).order_by(GPRSignalData.trace_number)
        )
        signal_records = result.scalars().all()

        if not signal_records:
            return None

        # Reconstruct signal array from stored data
        traces = []
        for record in signal_records:
            if record.signal_data:
                # Convert bytes back to numpy array
                trace_data = np.frombuffer(record.signal_data, dtype=np.float32)
                traces.append(trace_data)

        if traces:
            return np.column_stack(traces)

        return None

    async def _apply_time_zero_correction(
        self,
        signal_data: np.ndarray,
        config: Dict[str, Any]
    ) -> np.ndarray:
        """Apply time-zero correction to align signals."""
        time_zero_offset = config.get("time_zero_offset", 0)

        if time_zero_offset == 0:
            # Auto-detect time zero by finding first break
            time_zero_offset = await self._detect_time_zero(signal_data)

        # Shift signals by the offset
        if time_zero_offset > 0:
            samples_to_shift = int(time_zero_offset * self.sampling_frequency / 1000)  # Convert ns to samples
            corrected_data = np.zeros_like(signal_data)
            if samples_to_shift < signal_data.shape[0]:
                corrected_data[:-samples_to_shift] = signal_data[samples_to_shift:]
            return corrected_data

        return signal_data

    async def _detect_time_zero(self, signal_data: np.ndarray) -> float:
        """Automatically detect time zero position."""
        # Simple method: find first significant amplitude change
        mean_trace = np.mean(signal_data, axis=1)
        derivative = np.gradient(mean_trace)
        max_derivative_idx = np.argmax(np.abs(derivative))

        # Convert sample index to time in nanoseconds
        time_zero_ns = max_derivative_idx / self.sampling_frequency * 1000

        return time_zero_ns

    async def _apply_noise_removal(
        self,
        signal_data: np.ndarray,
        config: Dict[str, Any]
    ) -> np.ndarray:
        """Remove noise using various filtering techniques."""
        noise_method = config.get("noise_removal_method", "mean_subtraction")

        if noise_method == "mean_subtraction":
            # Remove DC component and horizontal striping
            mean_trace = np.mean(signal_data, axis=1, keepdims=True)
            return signal_data - mean_trace

        elif noise_method == "median_filter":
            # Apply median filter to each trace
            kernel_size = config.get("median_kernel_size", 5)
            filtered_data = np.zeros_like(signal_data)
            for i in range(signal_data.shape[1]):
                filtered_data[:, i] = signal.medfilt(signal_data[:, i], kernel_size)
            return filtered_data

        elif noise_method == "wiener_filter":
            # Apply Wiener filter for noise reduction
            filtered_data = np.zeros_like(signal_data)
            for i in range(signal_data.shape[1]):
                filtered_data[:, i] = signal.wiener(signal_data[:, i])
            return filtered_data

        return signal_data

    async def _apply_bandpass_filter(
        self,
        signal_data: np.ndarray,
        config: Dict[str, Any]
    ) -> np.ndarray:
        """Apply bandpass filter to enhance signal quality."""
        low_freq = config.get("bandpass_low_freq", 100)  # MHz
        high_freq = config.get("bandpass_high_freq", 800)  # MHz

        # Convert to normalized frequencies
        nyquist = self.sampling_frequency / 2
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist

        # Design Butterworth bandpass filter
        filter_order = config.get("filter_order", 4)
        b, a = signal.butter(filter_order, [low_norm, high_norm], btype='band')

        # Apply filter to each trace
        filtered_data = np.zeros_like(signal_data)
        for i in range(signal_data.shape[1]):
            filtered_data[:, i] = signal.filtfilt(b, a, signal_data[:, i])

        return filtered_data

    async def _apply_gain_adjustment(
        self,
        signal_data: np.ndarray,
        config: Dict[str, Any]
    ) -> np.ndarray:
        """Apply automatic gain control or time-varying gain."""
        gain_method = config.get("gain_method", "agc")

        if gain_method == "agc":
            # Automatic Gain Control
            window_size = config.get("agc_window_size", 50)
            target_amplitude = config.get("agc_target_amplitude", 1.0)

            gained_data = np.zeros_like(signal_data)

            for i in range(signal_data.shape[1]):
                trace = signal_data[:, i]
                gained_trace = np.zeros_like(trace)

                for j in range(len(trace)):
                    start_idx = max(0, j - window_size // 2)
                    end_idx = min(len(trace), j + window_size // 2)
                    window_rms = np.sqrt(np.mean(trace[start_idx:end_idx] ** 2))

                    if window_rms > 0:
                        gain_factor = target_amplitude / window_rms
                        gained_trace[j] = trace[j] * gain_factor
                    else:
                        gained_trace[j] = trace[j]

                gained_data[:, i] = gained_trace

            return gained_data

        elif gain_method == "time_varying":
            # Time-varying gain (exponential)
            time_constant = config.get("time_constant", 50)  # ns
            gain_factor = np.exp(np.arange(signal_data.shape[0]) / time_constant)
            return signal_data * gain_factor[:, np.newaxis]

        return signal_data

    async def _extract_signal_features(
        self,
        signal_data: np.ndarray,
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract comprehensive features from processed signals."""
        features = []

        # Statistical features
        features.extend(await self._extract_statistical_signal_features(signal_data))

        # Frequency domain features
        features.extend(await self._extract_frequency_features(signal_data))

        # Time-depth features
        features.extend(await self._extract_time_depth_features(signal_data))

        # Wavelet features
        if config.get("extract_wavelet_features", True):
            features.extend(await self._extract_wavelet_features(signal_data))

        return features

    async def _extract_statistical_signal_features(self, signal_data: np.ndarray) -> List[Dict[str, Any]]:
        """Extract statistical features from signal data."""
        features = []

        # Overall statistics
        features.append({
            "type": "statistical",
            "name": "signal_mean",
            "value": float(np.mean(signal_data)),
            "description": "Overall signal mean"
        })

        features.append({
            "type": "statistical",
            "name": "signal_std",
            "value": float(np.std(signal_data)),
            "description": "Overall signal standard deviation"
        })

        features.append({
            "type": "statistical",
            "name": "signal_rms",
            "value": float(np.sqrt(np.mean(signal_data ** 2))),
            "description": "Root mean square amplitude"
        })

        # Dynamic range
        features.append({
            "type": "statistical",
            "name": "dynamic_range",
            "value": float(np.max(signal_data) - np.min(signal_data)),
            "description": "Signal dynamic range"
        })

        # Trace-to-trace coherence
        coherence_values = []
        for i in range(signal_data.shape[1] - 1):
            correlation = np.corrcoef(signal_data[:, i], signal_data[:, i + 1])[0, 1]
            if not np.isnan(correlation):
                coherence_values.append(correlation)

        if coherence_values:
            features.append({
                "type": "statistical",
                "name": "trace_coherence",
                "value": float(np.mean(coherence_values)),
                "description": "Average trace-to-trace coherence"
            })

        return features

    async def _extract_frequency_features(self, signal_data: np.ndarray) -> List[Dict[str, Any]]:
        """Extract frequency domain features."""
        features = []

        # Compute average power spectrum
        freqs = fft.fftfreq(signal_data.shape[0], d=1/self.sampling_frequency)
        avg_spectrum = np.mean(np.abs(fft.fft(signal_data, axis=0)), axis=1)

        # Keep only positive frequencies
        positive_freqs = freqs[:len(freqs)//2]
        positive_spectrum = avg_spectrum[:len(avg_spectrum)//2]

        # Peak frequency
        peak_freq_idx = np.argmax(positive_spectrum)
        features.append({
            "type": "frequency",
            "name": "peak_frequency",
            "value": float(positive_freqs[peak_freq_idx]),
            "description": "Dominant frequency component"
        })

        # Bandwidth (frequency range containing 90% of energy)
        cumulative_energy = np.cumsum(positive_spectrum ** 2)
        total_energy = cumulative_energy[-1]
        low_idx = np.where(cumulative_energy >= 0.05 * total_energy)[0][0]
        high_idx = np.where(cumulative_energy >= 0.95 * total_energy)[0][0]

        features.append({
            "type": "frequency",
            "name": "bandwidth_90",
            "value": float(positive_freqs[high_idx] - positive_freqs[low_idx]),
            "description": "90% energy bandwidth"
        })

        # Spectral centroid
        spectral_centroid = np.sum(positive_freqs * positive_spectrum) / np.sum(positive_spectrum)
        features.append({
            "type": "frequency",
            "name": "spectral_centroid",
            "value": float(spectral_centroid),
            "description": "Spectral centroid frequency"
        })

        return features

    async def _extract_time_depth_features(self, signal_data: np.ndarray) -> List[Dict[str, Any]]:
        """Extract time-depth domain features."""
        features = []

        # Convert time to depth
        time_samples = np.arange(signal_data.shape[0])
        depth_m = time_samples * self.depth_calibration / self.sampling_frequency

        # Find significant reflections
        envelope = np.abs(signal.hilbert(np.mean(signal_data, axis=1)))
        peaks, _ = signal.find_peaks(envelope, height=np.max(envelope) * 0.1)

        features.append({
            "type": "time_depth",
            "name": "reflection_count",
            "value": len(peaks),
            "description": "Number of significant reflections"
        })

        if len(peaks) > 0:
            reflection_depths = depth_m[peaks]
            features.append({
                "type": "time_depth",
                "name": "first_reflection_depth",
                "value": float(reflection_depths[0]),
                "description": "Depth of first significant reflection"
            })

            features.append({
                "type": "time_depth",
                "name": "deepest_reflection_depth",
                "value": float(reflection_depths[-1]),
                "description": "Depth of deepest significant reflection"
            })

        return features

    async def _extract_wavelet_features(self, signal_data: np.ndarray) -> List[Dict[str, Any]]:
        """Extract wavelet-based features (placeholder implementation)."""
        # This would use PyWavelets for continuous or discrete wavelet transform
        # For now, return placeholder features
        features = [
            {
                "type": "wavelet",
                "name": "wavelet_energy",
                "value": float(np.sum(signal_data ** 2)),
                "description": "Total wavelet energy (placeholder)"
            }
        ]

        return features

    async def _detect_hyperbolas(
        self,
        signal_data: np.ndarray,
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect hyperbolic reflections indicating utilities."""
        hyperbolas = []

        # Simple hyperbola detection using envelope and curvature analysis
        # This is a simplified implementation

        for trace_idx in range(signal_data.shape[1]):
            trace = signal_data[:, trace_idx]
            envelope = np.abs(signal.hilbert(trace))

            # Find peaks in envelope
            peaks, properties = signal.find_peaks(
                envelope,
                height=np.max(envelope) * 0.2,
                distance=10
            )

            for peak in peaks:
                # Analyze local curvature around peak
                window_start = max(0, peak - 20)
                window_end = min(len(envelope), peak + 20)
                local_envelope = envelope[window_start:window_end]

                # Simple curvature measure
                if len(local_envelope) > 5:
                    second_derivative = np.gradient(np.gradient(local_envelope))
                    curvature = np.max(np.abs(second_derivative))

                    if curvature > config.get("hyperbola_curvature_threshold", 0.1):
                        hyperbola = {
                            "trace_index": int(trace_idx),
                            "sample_index": int(peak),
                            "depth_m": float(peak * self.depth_calibration / self.sampling_frequency),
                            "amplitude": float(envelope[peak]),
                            "curvature": float(curvature),
                            "confidence": min(1.0, curvature / 0.5)
                        }
                        hyperbolas.append(hyperbola)

        return hyperbolas

    async def _analyze_environmental_correlation(
        self,
        db: AsyncSession,
        scan_id: str,
        signal_data: np.ndarray,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze correlation between signal characteristics and environmental factors."""
        try:
            # Get environmental data for the survey
            result = await db.execute(
                select(EnvironmentalData)
                .join(GPRScan.survey)
                .where(GPRScan.id == scan_id)
            )
            env_data = result.scalar_one_or_none()

            if not env_data:
                return {"status": "no_environmental_data"}

            correlations = {}

            # Signal quality metrics
            signal_quality = {
                "mean_amplitude": float(np.mean(np.abs(signal_data))),
                "signal_to_noise": float(np.std(signal_data) / (np.mean(np.abs(signal_data)) + 1e-8)),
                "penetration_depth": float(signal_data.shape[0] * self.depth_calibration / self.sampling_frequency)
            }

            # Correlate with environmental factors
            environmental_factors = {
                "ground_permittivity": env_data.ground_relative_permittivity,
                "weather_impact": 1.0 if env_data.weather_condition == "Dry" else 0.5,
                "utility_density": env_data.amount_of_utilities or 0,
                "ground_moisture": 1.0 if "wet" in (env_data.ground_condition or "").lower() else 0.0
            }

            # Calculate correlations (simplified)
            for env_factor, env_value in environmental_factors.items():
                if env_value is not None:
                    # Simple correlation with signal amplitude
                    correlation_value = signal_quality["mean_amplitude"] * (env_value / 10.0)
                    correlations[env_factor] = {
                        "correlation": float(correlation_value),
                        "significance": 0.8  # Placeholder
                    }

            # Overall environmental impact score
            impact_score = np.mean([abs(corr["correlation"]) for corr in correlations.values()])

            self.log_environmental_correlation("overall_impact", impact_score)

            return {
                "signal_quality": signal_quality,
                "correlations": correlations,
                "impact_score": float(impact_score),
                "environmental_factors": environmental_factors
            }

        except Exception as e:
            self.log_operation_error("analyze_environmental_correlation", e)
            return {"status": "error", "message": str(e)}

    def _calculate_overall_confidence(self, processing_results: Dict[str, Any]) -> float:
        """Calculate overall confidence score from processing results."""
        confidence_factors = []

        # Hyperbola detection confidence
        hyperbolas = processing_results.get("hyperbolas", [])
        if hyperbolas:
            avg_hyperbola_confidence = np.mean([h.get("confidence", 0) for h in hyperbolas])
            confidence_factors.append(avg_hyperbola_confidence)

        # Environmental correlation strength
        env_corr = processing_results.get("environmental_correlations", {})
        if env_corr.get("impact_score") is not None:
            confidence_factors.append(min(1.0, env_corr["impact_score"]))

        # Signal quality indicators
        features = processing_results.get("features", [])
        snr_features = [f for f in features if "snr" in f.get("name", "").lower()]
        if snr_features:
            confidence_factors.append(min(1.0, max(0.0, snr_features[0]["value"] / 10.0)))

        return float(np.mean(confidence_factors)) if confidence_factors else 0.5