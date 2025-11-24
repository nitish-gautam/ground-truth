# Phase 1A API Architecture Plan
## GPR Data Processing and Analysis Pipeline

---

## API Architecture Overview

### Core Service Components

```
FastAPI Backend Architecture:
├── Data Ingestion Service
│   ├── Twente GPR Loader API
│   ├── Mojahid Image Processor
│   ├── SEG-Y File Parser
│   └── Metadata Validator
├── Signal Processing Service
│   ├── Time-Zero Correction
│   ├── Noise Removal Pipeline
│   ├── Frequency Analysis
│   └── Feature Extraction
├── ML Analysis Service
│   ├── Utility Detection Models
│   ├── Image Classification
│   ├── Depth Estimation
│   └── Confidence Scoring
├── Validation Service
│   ├── Ground Truth Comparison
│   ├── Performance Metrics
│   ├── Quality Assessment
│   └── PAS 128 Compliance
└── Reporting Service
    ├── Survey Reports
    ├── Compliance Documentation
    ├── Performance Analytics
    └── Data Export
```

---

## API Endpoints Design

### 1. Data Ingestion Endpoints

#### Twente Dataset Integration
```python
# Load 125 Twente GPR scans with metadata
POST /api/v1/datasets/twente/import
{
    "zip_file_path": "datasets/raw/twente_gpr/01.zip",
    "project_id": "uuid",
    "process_immediately": true,
    "validation_level": "full"
}

Response:
{
    "import_id": "uuid",
    "status": "processing",
    "files_found": {
        "radargrams": 15,
        "survey_maps": 15,
        "ground_truth": 15
    },
    "estimated_processing_time": "5-10 minutes"
}

# Check import status
GET /api/v1/datasets/imports/{import_id}/status

# Get imported survey details
GET /api/v1/surveys/location/{location_id}  # e.g., "01.1"
```

#### Mojahid Image Dataset Integration
```python
# Load Mojahid image dataset
POST /api/v1/datasets/mojahid/import
{
    "dataset_path": "datasets/raw/mojahid_images/GPR_data/",
    "categories": ["cavities", "utilities", "intact"],
    "include_augmented": true,
    "extract_features": true
}

# Batch image processing
POST /api/v1/images/process/batch
{
    "image_ids": ["uuid1", "uuid2", ...],
    "processing_pipeline": ["feature_extraction", "classification"],
    "model_version": "v1.0"
}
```

### 2. Signal Processing Endpoints

#### Raw GPR Data Processing
```python
# Process SEG-Y files
POST /api/v1/gpr/process/segy
{
    "file_path": "path/to/radargram.sgy",
    "survey_id": "uuid",
    "processing_options": {
        "time_zero_correction": true,
        "background_removal": true,
        "gain_correction": "automatic",
        "bandpass_filter": {
            "low_freq": 100,
            "high_freq": 800
        }
    }
}

# Get processing status
GET /api/v1/gpr/process/{process_id}/status

# Extract signal features
POST /api/v1/gpr/features/extract
{
    "survey_id": "uuid",
    "feature_types": [
        "amplitude_stats",
        "frequency_domain",
        "hyperbola_detection",
        "texture_features"
    ]
}
```

#### Environmental Correlation Analysis
```python
# Analyze environmental impact on detection
POST /api/v1/analysis/environmental-correlation
{
    "survey_ids": ["uuid1", "uuid2", ...],
    "factors": [
        "ground_condition",
        "moisture_level",
        "weather_condition",
        "surface_material"
    ],
    "correlation_method": "pearson"
}
```

### 3. ML Analysis Endpoints

#### Utility Detection
```python
# Run utility detection on GPR data
POST /api/v1/ml/detect/utilities
{
    "survey_id": "uuid",
    "model_name": "yolo_v8_gpr",
    "confidence_threshold": 0.7,
    "nms_threshold": 0.5
}

# Classify detected utilities
POST /api/v1/ml/classify/utility-type
{
    "detection_id": "uuid",
    "models": ["material_classifier", "discipline_classifier"]
}

# Estimate depth
POST /api/v1/ml/estimate/depth
{
    "detection_id": "uuid",
    "soil_velocity_model": "sandy_soil_default",
    "ground_permittivity": 9.0
}
```

#### Image Classification
```python
# Classify GPR images (Mojahid dataset)
POST /api/v1/ml/classify/images
{
    "image_ids": ["uuid1", "uuid2", ...],
    "model_name": "resnet50_gpr_classifier",
    "return_probabilities": true
}

# Feature similarity search
POST /api/v1/ml/similarity/search
{
    "query_image_id": "uuid",
    "similarity_threshold": 0.8,
    "max_results": 50
}
```

### 4. Validation and Quality Endpoints

#### Ground Truth Validation
```python
# Compare with ground truth
POST /api/v1/validation/ground-truth/compare
{
    "survey_id": "uuid",
    "validation_campaign_id": "uuid",
    "tolerance_depth_m": 0.1,
    "tolerance_position_m": 0.2
}

# Record validation results
POST /api/v1/validation/record
{
    "detection_id": "uuid",
    "validation_method": "trial_trench",
    "actual_findings": {
        "utility_present": true,
        "utility_type": "water",
        "material": "polyVinylChloride",
        "diameter_mm": 125,
        "depth_m": 1.2
    }
}
```

#### Performance Assessment
```python
# Model performance metrics
GET /api/v1/models/{model_id}/performance
{
    "evaluation_period": "2024-01-01:2024-12-31",
    "metrics": ["accuracy", "precision", "recall", "f1_score"],
    "breakdown_by": ["utility_type", "depth_range", "ground_condition"]
}

# Detection accuracy analysis
POST /api/v1/analysis/detection-accuracy
{
    "survey_ids": ["uuid1", "uuid2", ...],
    "stratify_by": ["depth_range", "utility_discipline", "ground_condition"]
}
```

### 5. Compliance and Reporting Endpoints

#### PAS 128 Compliance
```python
# Check PAS 128 compliance
POST /api/v1/compliance/pas128/validate
{
    "survey_id": "uuid",
    "standard_version": "PAS128:2022",
    "quality_level_target": "QL-B"
}

# Generate compliance report
POST /api/v1/reports/compliance/generate
{
    "project_id": "uuid",
    "report_type": "pas128_full",
    "include_appendices": true,
    "format": "pdf"
}
```

---

## Signal Processing Pipeline Architecture

### 1. Time-Zero Correction Pipeline

```python
class TimeZeroCorrectionService:
    """
    Corrects for air gap and surface reflection in GPR data
    """

    async def correct_time_zero(
        self,
        signal_data: np.ndarray,
        method: str = "first_break"
    ) -> ProcessedSignal:

        # Auto-detect first break
        first_break_sample = self.detect_first_break(signal_data)

        # Apply time-zero correction
        corrected_signal = self.shift_time_axis(
            signal_data,
            shift_samples=first_break_sample
        )

        # Update depth calculations
        depth_axis = self.calculate_depth_axis(
            corrected_signal,
            velocity_model=self.get_velocity_model()
        )

        return ProcessedSignal(
            data=corrected_signal,
            depth_axis=depth_axis,
            processing_metadata={
                "time_zero_sample": first_break_sample,
                "correction_method": method,
                "processing_timestamp": datetime.utcnow()
            }
        )
```

### 2. Noise Removal Pipeline

```python
class NoiseRemovalService:
    """
    Advanced noise filtering for GPR signals
    """

    async def remove_background_noise(
        self,
        signal_data: np.ndarray,
        method: str = "mean_trace_subtraction"
    ) -> np.ndarray:

        if method == "mean_trace_subtraction":
            # Remove horizontal banding
            mean_trace = np.mean(signal_data, axis=1, keepdims=True)
            return signal_data - mean_trace

        elif method == "adaptive_filtering":
            # Use adaptive noise cancellation
            return self.adaptive_noise_filter(signal_data)

        elif method == "wavelet_denoising":
            # Wavelet-based denoising
            return self.wavelet_denoise(signal_data)

    async def apply_gain_correction(
        self,
        signal_data: np.ndarray,
        gain_type: str = "automatic"
    ) -> np.ndarray:

        if gain_type == "automatic":
            # Automatic gain control
            return self.automatic_gain_control(signal_data)
        elif gain_type == "depth_dependent":
            # Depth-dependent gain
            return self.depth_dependent_gain(signal_data)
```

### 3. Feature Extraction Architecture

```python
class FeatureExtractionService:
    """
    Extract comprehensive features from GPR data
    """

    async def extract_hyperbola_features(
        self,
        signal_data: np.ndarray
    ) -> Dict[str, float]:

        # Hough transform for hyperbola detection
        hyperbolas = self.hough_hyperbola_detection(signal_data)

        features = {}
        for i, hyperbola in enumerate(hyperbolas):
            features.update({
                f"hyperbola_{i}_apex_depth": hyperbola.apex_depth,
                f"hyperbola_{i}_width": hyperbola.width,
                f"hyperbola_{i}_curvature": hyperbola.curvature,
                f"hyperbola_{i}_symmetry": hyperbola.symmetry_score
            })

        return features

    async def extract_amplitude_features(
        self,
        signal_data: np.ndarray
    ) -> Dict[str, float]:

        return {
            "peak_amplitude": np.max(signal_data),
            "mean_amplitude": np.mean(signal_data),
            "amplitude_variance": np.var(signal_data),
            "amplitude_skewness": scipy.stats.skew(signal_data.flatten()),
            "amplitude_kurtosis": scipy.stats.kurtosis(signal_data.flatten()),
            "rms_amplitude": np.sqrt(np.mean(signal_data**2))
        }

    async def extract_frequency_features(
        self,
        signal_data: np.ndarray,
        sampling_rate: float
    ) -> Dict[str, float]:

        # FFT analysis
        fft_data = np.fft.fft(signal_data, axis=0)
        power_spectrum = np.abs(fft_data)**2

        # Frequency axis
        freqs = np.fft.fftfreq(signal_data.shape[0], 1/sampling_rate)

        return {
            "dominant_frequency": self.find_dominant_frequency(power_spectrum, freqs),
            "bandwidth": self.calculate_bandwidth(power_spectrum, freqs),
            "spectral_centroid": self.spectral_centroid(power_spectrum, freqs),
            "spectral_rolloff": self.spectral_rolloff(power_spectrum, freqs),
            "spectral_flatness": self.spectral_flatness(power_spectrum)
        }
```

---

## Environmental Correlation Analysis System

### Correlation Analysis Service

```python
class EnvironmentalCorrelationService:
    """
    Analyze impact of environmental factors on GPR detection
    """

    async def analyze_moisture_impact(
        self,
        survey_data: List[GPRSurvey]
    ) -> CorrelationAnalysis:

        # Group by moisture conditions
        dry_surveys = [s for s in survey_data if s.weather_condition == "Dry"]
        wet_surveys = [s for s in survey_data if s.weather_condition == "Rainy"]

        # Calculate detection performance for each group
        dry_performance = await self.calculate_detection_performance(dry_surveys)
        wet_performance = await self.calculate_detection_performance(wet_surveys)

        return CorrelationAnalysis(
            factor="moisture",
            dry_conditions=dry_performance,
            wet_conditions=wet_performance,
            correlation_coefficient=self.calculate_correlation(
                [s.detection_accuracy for s in dry_surveys],
                [s.detection_accuracy for s in wet_surveys]
            )
        )

    async def analyze_soil_type_impact(
        self,
        survey_data: List[GPRSurvey]
    ) -> Dict[str, PerformanceMetrics]:

        soil_groups = {}
        for survey in survey_data:
            soil_type = survey.ground_condition
            if soil_type not in soil_groups:
                soil_groups[soil_type] = []
            soil_groups[soil_type].append(survey)

        results = {}
        for soil_type, surveys in soil_groups.items():
            results[soil_type] = await self.calculate_detection_performance(surveys)

        return results

    async def calculate_detection_performance(
        self,
        surveys: List[GPRSurvey]
    ) -> PerformanceMetrics:

        total_detections = 0
        correct_detections = 0
        false_positives = 0
        depth_errors = []

        for survey in surveys:
            validations = await self.get_ground_truth_validations(survey.id)

            for validation in validations:
                total_detections += 1
                if validation.utility_present and validation.detection_correct:
                    correct_detections += 1
                elif not validation.utility_present:
                    false_positives += 1

                if validation.depth_error is not None:
                    depth_errors.append(abs(validation.depth_error))

        return PerformanceMetrics(
            accuracy=correct_detections / total_detections if total_detections > 0 else 0,
            false_positive_rate=false_positives / total_detections if total_detections > 0 else 0,
            mean_depth_error=np.mean(depth_errors) if depth_errors else 0,
            std_depth_error=np.std(depth_errors) if depth_errors else 0
        )
```

---

## Data Loading and Processing Workflows

### Twente Dataset Loading Workflow

```python
class TwenteDatasetLoader:
    """
    Load and process 125 Twente GPR surveys with full metadata
    """

    async def import_twente_dataset(
        self,
        project_id: UUID,
        dataset_path: str = "datasets/raw/twente_gpr/"
    ) -> ImportResult:

        # Load metadata CSV
        metadata_df = pd.read_csv(f"{dataset_path}/Metadata.csv", sep=';')

        import_tasks = []
        for zip_file in range(1, 14):  # 01.zip to 013.zip
            zip_path = f"{dataset_path}/{zip_file:02d}.zip" if zip_file < 10 else f"{dataset_path}/{zip_file}.zip"
            task = self.process_zip_file(zip_path, metadata_df, project_id)
            import_tasks.append(task)

        # Process all zip files concurrently
        results = await asyncio.gather(*import_tasks)

        return ImportResult(
            total_surveys_imported=sum(r.surveys_count for r in results),
            total_files_processed=sum(r.files_count for r in results),
            processing_time=sum(r.processing_time for r in results),
            errors=flatten([r.errors for r in results])
        )

    async def process_zip_file(
        self,
        zip_path: str,
        metadata_df: pd.DataFrame,
        project_id: UUID
    ) -> ZipProcessingResult:

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract files to temporary directory
            temp_dir = f"/tmp/gpr_processing/{uuid4()}"
            zip_ref.extractall(temp_dir)

            # Find all .sgy files
            sgy_files = glob.glob(f"{temp_dir}/**/*.sgy", recursive=True)

            surveys_created = []
            for sgy_file in sgy_files:
                # Extract location ID from filename
                location_id = self.extract_location_id(sgy_file)

                # Get metadata for this location
                location_metadata = metadata_df[
                    metadata_df['LocationID'] == location_id
                ].iloc[0]

                # Create survey record
                survey = await self.create_survey_record(
                    project_id=project_id,
                    location_id=location_id,
                    metadata=location_metadata,
                    sgy_file_path=sgy_file
                )

                surveys_created.append(survey)

                # Process GPR data
                await self.process_gpr_data(survey.id, sgy_file)

                # Extract ground truth if available
                ground_truth_file = sgy_file.replace('.sgy', '_ground_truth.png')
                if os.path.exists(ground_truth_file):
                    await self.process_ground_truth(survey.id, ground_truth_file)

            # Cleanup
            shutil.rmtree(temp_dir)

            return ZipProcessingResult(
                surveys_count=len(surveys_created),
                files_count=len(sgy_files),
                processing_time=time.time() - start_time
            )
```

### Mojahid Image Dataset Processing

```python
class MojahidImageProcessor:
    """
    Process Mojahid GPR image dataset for ML training
    """

    async def import_mojahid_dataset(
        self,
        dataset_path: str = "datasets/raw/mojahid_images/GPR_data/"
    ) -> ImageImportResult:

        categories = [
            "cavities", "intact", "Utilities",
            "augmented_cavities", "augmented_intact", "augmented_utilities"
        ]

        total_processed = 0
        for category in categories:
            category_path = f"{dataset_path}/{category}/"
            image_files = glob.glob(f"{category_path}/*.jpg")

            # Process images in batches
            batch_size = 50
            for i in range(0, len(image_files), batch_size):
                batch = image_files[i:i+batch_size]
                await self.process_image_batch(batch, category)
                total_processed += len(batch)

        return ImageImportResult(
            total_images_processed=total_processed,
            categories_processed=len(categories)
        )

    async def process_image_batch(
        self,
        image_files: List[str],
        category: str
    ) -> None:

        for image_file in image_files:
            # Load image
            image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

            # Extract features
            features = await self.extract_image_features(image)

            # Create database record
            image_record = await self.create_image_record(
                filename=os.path.basename(image_file),
                file_path=image_file,
                category=category,
                features=features,
                image_shape=image.shape
            )

            # Generate embeddings for similarity search
            embedding = await self.generate_image_embedding(image)
            await self.store_image_embedding(image_record.id, embedding)

    async def extract_image_features(
        self,
        image: np.ndarray
    ) -> Dict[str, Any]:

        return {
            "mean_intensity": np.mean(image),
            "std_intensity": np.std(image),
            "entropy": self.calculate_entropy(image),
            "contrast": self.calculate_contrast(image),
            "homogeneity": self.calculate_homogeneity(image),
            "energy": self.calculate_energy(image),
            "correlation": self.calculate_correlation(image)
        }
```

---

## API Implementation Example

### FastAPI Application Structure

```python
# app/main.py
from fastapi import FastAPI, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import datasets, gpr, ml, validation, compliance
from app.core.config import settings
from app.core.database import engine, Base

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Underground Utility Detection Platform API",
    description="GPR data processing and ML analysis for utility detection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(datasets.router, prefix="/api/v1/datasets", tags=["datasets"])
app.include_router(gpr.router, prefix="/api/v1/gpr", tags=["gpr"])
app.include_router(ml.router, prefix="/api/v1/ml", tags=["ml"])
app.include_router(validation.router, prefix="/api/v1/validation", tags=["validation"])
app.include_router(compliance.router, prefix="/api/v1/compliance", tags=["compliance"])

@app.get("/")
async def root():
    return {"message": "Underground Utility Detection Platform API v1.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}
```

### Dataset Import Endpoint Implementation

```python
# app/api/v1/datasets.py
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from app.services.twente_loader import TwenteDatasetLoader
from app.services.mojahid_processor import MojahidImageProcessor
from app.schemas.datasets import TwenteImportRequest, ImportStatus

router = APIRouter()

@router.post("/twente/import", response_model=ImportStatus)
async def import_twente_dataset(
    request: TwenteImportRequest,
    background_tasks: BackgroundTasks,
    loader: TwenteDatasetLoader = Depends()
):
    """
    Import Twente GPR dataset with full metadata processing
    """

    # Validate project exists
    project = await loader.get_project(request.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Start background import task
    import_id = uuid4()
    background_tasks.add_task(
        loader.import_twente_dataset,
        import_id=import_id,
        project_id=request.project_id,
        dataset_path=request.dataset_path,
        process_immediately=request.process_immediately
    )

    return ImportStatus(
        import_id=import_id,
        status="started",
        message="Twente dataset import started"
    )

@router.get("/imports/{import_id}/status", response_model=ImportStatus)
async def get_import_status(
    import_id: UUID,
    loader: TwenteDatasetLoader = Depends()
):
    """
    Get status of dataset import operation
    """
    status = await loader.get_import_status(import_id)

    if not status:
        raise HTTPException(status_code=404, detail="Import not found")

    return status
```

This comprehensive API architecture plan provides:

1. **Complete endpoint design** for all GPR data operations
2. **Signal processing pipeline** with time-zero correction and noise removal
3. **Feature extraction architecture** for amplitude and frequency analysis
4. **Environmental correlation analysis** system
5. **ML model integration** for utility detection and classification
6. **Validation framework** with ground truth comparison
7. **Compliance checking** for PAS 128 standards
8. **Efficient data loading** for both Twente and Mojahid datasets

The architecture supports parallel development by providing clear service boundaries and comprehensive API specifications for frontend integration.