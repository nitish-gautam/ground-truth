# Underground Utility Detection Platform - FastAPI Backend

🔍 **Comprehensive GPR data processing and analysis platform for underground utility detection**

## Overview

This FastAPI backend provides a complete pipeline for processing Ground Penetrating Radar (GPR) data, implementing advanced signal processing, machine learning analytics, and environmental correlation analysis for underground utility detection.

## 🎯 Key Features

### 📊 **Dataset Processing**
- **Twente GPR Dataset**: Process 125 real GPR scans with ground truth data
- **Mojahid Images**: Handle 2,239+ labeled GPR images across 6 categories
- **Automated ZIP extraction** and batch processing capabilities
- **Comprehensive metadata parsing** from CSV files

### 🔧 **Signal Processing Pipeline**
- **Time-zero correction** and signal alignment
- **Advanced filtering**: Bandpass, noise removal, gain adjustment
- **Feature extraction**: Statistical, frequency, and wavelet features
- **Hyperbola detection** for utility identification
- **Environmental correlation** analysis

### 🌍 **Environmental Analysis**
- **Weather impact assessment** on detection accuracy
- **Ground condition correlation** with signal quality
- **Soil composition analysis** and contamination effects
- **Terrain characteristics** impact evaluation
- **Multivariate analysis** and predictive modeling

### ✅ **Validation & Accuracy Assessment**
- **Ground truth validation** against known utility locations
- **PAS 128 compliance** checking (QL-A through QL-D)
- **Comprehensive accuracy metrics**: precision, recall, F1-score
- **Position and depth error** analysis
- **Statistical significance** testing

### 🤖 **Machine Learning Integration**
- **Feature vector management** for ML training
- **Model performance tracking** and analytics
- **Training session management** with comprehensive logging
- **Automated feature engineering** from GPR signals

## 🏗️ Architecture

```
backend/
├── app/
│   ├── api/v1/endpoints/     # REST API endpoints
│   ├── core/                 # Configuration, database, logging
│   ├── models/               # SQLAlchemy ORM models
│   ├── schemas/              # Pydantic request/response models
│   └── services/             # Business logic services
├── start_server.py           # Production startup script
└── .env.example             # Environment configuration template
```

### 🗄️ **Database Schema**

**Core Tables:**
- `gpr_surveys` - Survey metadata and configuration
- `gpr_scans` - Individual scan records with file information
- `gpr_signal_data` - Processed signal data storage
- `environmental_data` - Environmental conditions and metadata
- `ground_truth_data` - Known utility locations and characteristics
- `validation_results` - Detection validation against ground truth
- `ml_models` - Machine learning model registry
- `feature_vectors` - Extracted features for ML training

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 14+ with PostGIS extension
- Required datasets (see Dataset Setup below)

### 1. Environment Setup

```bash
# Clone and navigate to backend directory
cd backend/

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r ../requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your database credentials and paths
```

### 2. Database Setup

```bash
# Initialize PostgreSQL database
createdb gpr_platform

# Run database initialization (from project root)
psql -d gpr_platform -f database/00_master_init.sql
psql -d gpr_platform -f database/deploy_database.sql
```

### 3. Dataset Setup

Ensure datasets are available at the configured paths:

```
datasets/
├── raw/
│   ├── twente_gpr/          # 125 GPR ZIP files + Metadata.csv
│   └── mojahid_images/      # GPR_data/ with 6 image categories
└── processed/               # Auto-generated processed data
```

### 4. Start the Server

```bash
# Development mode
python start_server.py

# Or use uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 📡 API Endpoints

### Dataset Management
- `GET /api/v1/datasets/info` - List available datasets
- `POST /api/v1/datasets/twente/load` - Load Twente GPR dataset
- `POST /api/v1/datasets/mojahid/load` - Load Mojahid images
- `POST /api/v1/datasets/upload/gpr` - Upload individual GPR files

### GPR Data Processing
- `GET /api/v1/gpr/surveys` - List GPR surveys
- `GET /api/v1/gpr/scans` - List GPR scans
- `POST /api/v1/processing/filter` - Apply signal filtering
- `POST /api/v1/processing/extract-features` - Extract signal features

### Environmental Analysis
- `GET /api/v1/environmental/conditions` - Environmental data
- `GET /api/v1/environmental/correlations` - Correlation analysis

### Validation & Analytics
- `GET /api/v1/validation/ground-truth` - Ground truth data
- `GET /api/v1/validation/accuracy-metrics` - Accuracy assessment
- `GET /api/v1/analytics/models` - ML model information
- `GET /api/v1/analytics/performance` - Model performance metrics

## 📚 API Documentation

When the server is running, comprehensive API documentation is available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/api/v1/openapi.json

## 🔧 Configuration

### Key Environment Variables

```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_USER=gpr_app_user
DB_PASSWORD=your_password
DB_NAME=gpr_platform

# Processing
SAMPLING_FREQUENCY=1000.0
TIME_WINDOW=100.0
DEPTH_CALIBRATION=0.1
BATCH_SIZE=10

# Paths
DATA_ROOT_PATH=/path/to/datasets
```

### Logging Configuration

The application uses structured logging with multiple output formats:
- Console output with colors (development)
- Rotating file logs (production)
- Separate error and performance logs
- GPR processing specific logs

## 🧪 Usage Examples

### 1. Load Twente Dataset

```python
import httpx

# Start background loading
response = httpx.post("http://localhost:8000/api/v1/datasets/twente/load",
                     params={"batch_size": 10})
print(response.json())

# Check status
status = httpx.get("http://localhost:8000/api/v1/datasets/twente/status")
print(status.json())
```

### 2. Process GPR Signals

```python
# Apply signal processing
processing_config = {
    "steps": ["time_zero_correction", "bandpass_filter", "feature_extraction"],
    "bandpass_low_freq": 100,
    "bandpass_high_freq": 800
}

response = httpx.post("http://localhost:8000/api/v1/processing/filter",
                     json=processing_config)
```

### 3. Environmental Correlation Analysis

```python
# Get environmental correlations
correlations = httpx.get("http://localhost:8000/api/v1/environmental/correlations")
weather_impact = correlations.json()["weather_impact_analysis"]
```

## 🔍 Key Services

### TwenteDatasetLoader
- Processes ZIP files containing GPR data
- Extracts metadata from CSV files
- Creates database records with proper relationships

### MojahidImageProcessor
- Comprehensive image feature extraction
- Statistical, texture, edge, and spatial features
- Integration with ML pipeline

### GPRSignalProcessor
- Advanced signal processing pipeline
- Time-zero correction and filtering
- Hyperbola detection algorithms
- Environmental correlation analysis

### EnvironmentalCorrelationAnalyzer
- Weather impact assessment
- Ground condition analysis
- Multivariate correlation studies
- Predictive model building

### ValidationService
- Ground truth comparison
- PAS 128 compliance checking
- Accuracy metrics calculation
- Statistical significance testing

## 📊 Monitoring & Logging

### Performance Monitoring
- Processing time tracking for all operations
- Memory usage monitoring
- Database query performance
- File processing throughput metrics

### Comprehensive Logging
- Structured logging with context
- Performance benchmarking
- Error tracking with full stack traces
- Operation-specific log streams

## 🔒 Security Features

- Role-based database access (admin, analyst, app_user, readonly)
- Input validation using Pydantic schemas
- SQL injection protection via SQLAlchemy ORM
- File upload size and type restrictions
- CORS configuration for web integration

## 🚀 Production Deployment

### Docker Support (Recommended)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY backend/ .
CMD ["python", "start_server.py"]
```

### Environment Configuration

```bash
# Production settings
DEBUG=false
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Database connection pooling
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40

# Security
SECRET_KEY=your-production-secret-key
ALLOWED_HOSTS=["your-domain.com"]
```

## 🤝 Contributing

1. Follow PEP 8 style guidelines
2. Add comprehensive logging to new services
3. Include Pydantic schemas for all API endpoints
4. Write unit tests for business logic
5. Update API documentation

## 📄 License

This project is part of the Underground Utility Detection Platform research initiative.

## 🆘 Support

For technical support or questions:

1. Check the API documentation at `/docs`
2. Review application logs in the `logs/` directory
3. Verify database connectivity and configuration
4. Ensure datasets are properly configured and accessible

---

**Built with FastAPI, SQLAlchemy, and comprehensive GPR signal processing capabilities** 🔍📡