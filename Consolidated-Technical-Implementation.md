# Underground Utility Detection Platform
## Consolidated Technical Implementation Guide

---

## Implementation Overview

This document consolidates the technical implementation details, dataset requirements, and development workflows for building the Underground Utility Detection Platform MVP in 8 weeks.

---

## Proposed Project Structure

```
underground-utility-detection/
├── backend/                           # Backend application code
│   ├── api/                          # FastAPI application
│   │   ├── main.py                   # Main FastAPI app with WebSocket support
│   │   └── routers/                  # API endpoint routers
│   │       ├── gpr_processing.py     # GPR data upload and processing
│   │       ├── utility_detection.py  # Utility detection endpoints
│   │       ├── compliance.py         # PAS 128 compliance endpoints
│   │       ├── reports.py            # Report generation endpoints
│   │       ├── risk_assessment.py    # Risk scoring and analytics
│   │       ├── projects.py           # Project management endpoints
│   │       ├── data_fusion.py        # Multi-source data correlation
│   │       ├── websocket.py          # Real-time updates via WebSocket
│   │       └── datasets.py           # Open dataset integration endpoints
│   ├── database/                     # Multi-database layer
│   │   ├── postgresql_db.py         # PostgreSQL + PostGIS ORM models
│   │   ├── vector_db.py             # ChromaDB/Pinecone vector operations
│   │   └── unified_data_manager.py  # Universal DB interface
│   ├── processing/                   # Data processing modules
│   │   ├── gpr/                     # GPR-specific processing
│   │   │   ├── parsers.py           # SEG-Y, DZT, DT1 file parsers
│   │   │   ├── signal_processing.py # Time-zero correction, filtering
│   │   │   ├── feature_extraction.py # GPR feature extraction
│   │   │   └── utility_detection.py # ML-based utility detection
│   │   ├── documents/               # Document processing
│   │   │   ├── pdf_processor.py     # OCR and text extraction
│   │   │   ├── cad_processor.py     # CAD file parsing (DWG, DXF)
│   │   │   └── pas128_processor.py  # PAS 128 document processing
│   │   ├── ml/                      # Machine learning models
│   │   │   ├── utility_classifier.py # Utility type classification
│   │   │   ├── depth_estimator.py   # Depth estimation models
│   │   │   ├── risk_scorer.py       # Strike risk assessment
│   │   │   └── model_trainer.py     # Training pipeline
│   │   └── datasets/                # Open dataset integrations
│   │       ├── twente_loader.py     # University of Twente dataset
│   │       ├── mojahid_loader.py    # Mojahid GPR images
│   │       ├── usag_loader.py       # USAG strike reports
│   │       ├── bgs_loader.py        # BGS geotechnical data
│   │       └── uk_networks_loader.py # UK utility networks data
│   ├── rag/                         # RAG pipeline for compliance
│   │   ├── document_chunker.py      # PAS 128 semantic chunking
│   │   ├── embeddings.py            # Text embedding generation
│   │   ├── vector_store.py          # Vector database operations
│   │   ├── retrieval.py             # Semantic search and retrieval
│   │   └── generation.py            # LLM-based report generation
│   ├── llm/                         # Multi-LLM integrations
│   │   ├── base_client.py           # Abstract LLM interface
│   │   ├── openai_client.py         # OpenAI GPT-4o integration
│   │   ├── compliance_agent.py      # PAS 128 compliance agent
│   │   └── report_generator.py      # Report generation agent
│   ├── compliance/                  # PAS 128 compliance engine
│   │   ├── quality_levels.py       # QL-A to QL-D classification
│   │   ├── validation_rules.py     # Compliance validation
│   │   ├── audit_trail.py          # CDM 2015 audit logging
│   │   └── report_templates.py     # Standardized report formats
│   └── settings.py                 # Application settings
├── frontend/                        # Frontend application
│   ├── react-frontend/             # Modern React PWA
│   │   ├── src/
│   │   │   ├── components/         # Reusable React components
│   │   │   │   ├── upload/         # File upload components
│   │   │   │   │   ├── GPRUploader.tsx    # GPR file upload
│   │   │   │   │   ├── DocumentUploader.tsx # PDF/CAD upload
│   │   │   │   │   └── BulkUploader.tsx   # Batch file processing
│   │   │   │   ├── visualization/  # Data visualization
│   │   │   │   │   ├── GPRViewer.tsx      # GPR radargram display
│   │   │   │   │   ├── UtilityMap.tsx     # Utility overlay map
│   │   │   │   │   ├── RiskHeatmap.tsx    # Risk assessment visualization
│   │   │   │   │   └── DepthProfile.tsx   # Depth estimation charts
│   │   │   │   ├── reports/        # Report components
│   │   │   │   │   ├── ReportViewer.tsx   # PAS 128 report display
│   │   │   │   │   ├── ReportBuilder.tsx  # Interactive report creation
│   │   │   │   │   └── ExportOptions.tsx  # PDF/Word export
│   │   │   │   ├── common/         # Common UI components
│   │   │   │   │   ├── Header.tsx         # Navigation header
│   │   │   │   │   ├── Sidebar.tsx        # Navigation sidebar
│   │   │   │   │   ├── LoadingSpinner.tsx # Loading indicators
│   │   │   │   │   └── ErrorBoundary.tsx  # Error handling
│   │   │   │   └── mobile/         # Mobile-specific components
│   │   │   │       ├── MobileSurvey.tsx   # Field survey interface
│   │   │   │       ├── GPSCapture.tsx     # GPS coordinate capture
│   │   │   │       └── OfflineSync.tsx    # Offline data synchronization
│   │   │   ├── pages/              # Page components
│   │   │   │   ├── Dashboard.tsx          # Main project dashboard
│   │   │   │   ├── ProjectsList.tsx       # Project management
│   │   │   │   ├── SurveyCapture.tsx      # Data collection interface
│   │   │   │   ├── DataProcessing.tsx     # Processing status/results
│   │   │   │   ├── UtilityDetection.tsx   # Detection results
│   │   │   │   ├── RiskAssessment.tsx     # Risk analysis dashboard
│   │   │   │   ├── ComplianceCheck.tsx    # PAS 128 compliance validation
│   │   │   │   ├── ReportGeneration.tsx   # Report creation interface
│   │   │   │   ├── DatasetExplorer.tsx    # Open dataset browser
│   │   │   │   └── Settings.tsx           # Application settings
│   │   │   ├── services/           # API integration
│   │   │   │   ├── api.ts                 # Centralized API client
│   │   │   │   ├── gprService.ts          # GPR processing API calls
│   │   │   │   ├── complianceService.ts   # Compliance API calls
│   │   │   │   ├── reportsService.ts      # Reports API calls
│   │   │   │   └── websocketService.ts    # Real-time updates
│   │   │   ├── hooks/              # Custom React hooks
│   │   │   │   ├── useGPRProcessing.ts    # GPR processing state
│   │   │   │   ├── useWebSocket.ts        # WebSocket connection
│   │   │   │   ├── useOfflineStorage.ts   # Offline data management
│   │   │   │   └── useGeolocation.ts      # GPS/location hooks
│   │   │   ├── utils/              # Utility functions
│   │   │   │   ├── gprUtils.ts            # GPR data manipulation
│   │   │   │   ├── mapUtils.ts            # Mapping utilities
│   │   │   │   ├── fileUtils.ts           # File handling
│   │   │   │   └── validationUtils.ts     # Data validation
│   │   │   └── types/              # TypeScript definitions
│   │   │       ├── gpr.ts                 # GPR data types
│   │   │       ├── utility.ts             # Utility data types
│   │   │       ├── compliance.ts          # PAS 128 types
│   │   │       ├── report.ts              # Report types
│   │   │       └── api.ts                 # API response types
│   │   ├── public/                 # Static assets
│   │   │   ├── manifest.json              # PWA manifest
│   │   │   ├── sw.js                      # Service worker for offline
│   │   │   └── icons/                     # PWA icons
│   │   ├── package.json            # Dependencies
│   │   ├── tsconfig.json           # TypeScript config
│   │   └── vite.config.ts          # Vite build configuration
│   └── mobile/                     # Optional React Native app
│       ├── src/                    # Mobile app source
│       ├── android/                # Android build files
│       ├── ios/                    # iOS build files
│       └── package.json            # Mobile dependencies
├── datasets/                        # Open source datasets integration
│   ├── download_scripts/           # Dataset download automation
│   │   ├── download_twente.py      # University of Twente GPR data
│   │   ├── download_mojahid.py     # Mojahid labeled images
│   │   ├── download_usag.py        # USAG strike reports
│   │   ├── download_bgs.py         # BGS geotechnical data
│   │   ├── download_uk_networks.py # UK utility network data
│   │   └── download_all.py         # Download all datasets
│   ├── raw/                        # Raw downloaded datasets
│   │   ├── twente_gpr/             # University of Twente GPR scans
│   │   ├── mojahid_images/         # Mojahid labeled GPR images
│   │   ├── usag_reports/           # USAG utility strike reports
│   │   ├── bgs_data/               # BGS geotechnical database
│   │   ├── uk_gas_networks/        # Northern Gas Networks data
│   │   ├── uk_power_networks/      # UK Power Networks data
│   │   └── pas128_docs/            # PAS 128 specification documents
│   ├── processed/                  # Processed and cleaned data
│   │   ├── training_data/          # ML training datasets
│   │   ├── validation_data/        # Model validation sets
│   │   ├── embeddings/             # Pre-computed embeddings
│   │   └── knowledge_base/         # RAG knowledge base
│   └── synthetic/                  # Synthetic data generation
│       ├── gprmax_models/          # gprMax simulation models
│       ├── synthetic_gpr/          # Generated GPR data
│       └── augmented_data/         # Data augmentation results
├── ml_models/                       # Machine learning models
│   ├── trained/                    # Trained model files
│   │   ├── utility_classifier.pkl  # Utility type classifier
│   │   ├── depth_estimator.pkl     # Depth estimation model
│   │   ├── risk_scorer.pkl         # Risk assessment model
│   │   └── embeddings.pkl          # Text embeddings model
│   ├── training/                   # Training scripts and configs
│   │   ├── train_classifier.py     # Utility classification training
│   │   ├── train_depth_model.py    # Depth estimation training
│   │   ├── train_risk_model.py     # Risk scoring training
│   │   └── training_configs/       # Model configuration files
│   ├── evaluation/                 # Model evaluation
│   │   ├── evaluate_models.py      # Model performance evaluation
│   │   ├── benchmark_accuracy.py   # Accuracy benchmarking
│   │   └── validation_reports/     # Evaluation reports
│   └── experiments/                # Experimental models
│       ├── transformer_models/     # Transformer-based approaches
│       ├── ensemble_models/        # Model ensemble experiments
│       └── fine_tuning/            # Fine-tuning experiments
├── compliance/                      # PAS 128 compliance resources
│   ├── specifications/             # PAS 128 specification files
│   │   ├── pas128_2022.pdf         # Main PAS 128:2022 document
│   │   ├── quality_levels.json     # QL-A to QL-D definitions
│   │   ├── decision_trees.json     # Compliance decision logic
│   │   └── report_templates/       # Standard report templates
│   ├── validation/                 # Compliance validation
│   │   ├── quality_checkers.py     # Quality level validation
│   │   ├── completeness_check.py   # Report completeness validation
│   │   └── cdm2015_audit.py        # CDM 2015 audit trail
│   └── examples/                   # Example compliant reports
│       ├── sample_reports/         # Sample PAS 128 reports
│       └── test_cases/             # Compliance test scenarios
├── infrastructure/                  # Local infrastructure setup
│   ├── docker/                     # Docker configuration
│   │   ├── docker-compose.yml      # Multi-service orchestration
│   │   ├── Dockerfile.backend      # Backend container
│   │   ├── Dockerfile.frontend     # Frontend container
│   │   └── Dockerfile.ml           # ML processing container
│   ├── database/                   # Database setup
│   │   ├── init_postgresql.sql     # PostgreSQL initialization
│   │   ├── create_spatial_index.sql # PostGIS spatial indexing
│   │   └── sample_data.sql         # Sample data for testing
│   ├── vector_db/                  # Vector database setup
│   │   ├── chromadb_init.py        # ChromaDB initialization
│   │   └── pinecone_setup.py       # Pinecone setup (if used)
│   └── monitoring/                 # Basic monitoring setup
│       ├── healthcheck.py          # Health check endpoints
│       └── logging_config.py       # Logging configuration
├── scripts/                        # Utility and automation scripts
│   ├── setup/                      # Setup and installation
│   │   ├── setup_complete_platform.sh # Full platform setup
│   │   ├── install_dependencies.sh     # Install all dependencies
│   │   ├── setup_database.sh           # Database initialization
│   │   └── download_datasets.sh        # Download all open datasets
│   ├── data_processing/            # Data processing utilities
│   │   ├── process_gpr_batch.py    # Batch GPR processing
│   │   ├── generate_embeddings.py  # Generate document embeddings
│   │   ├── train_models.py         # Train all ML models
│   │   └── validate_data.py        # Data validation and cleaning
│   ├── testing/                    # Testing utilities
│   │   ├── test_api_endpoints.sh   # API endpoint testing
│   │   ├── test_gpr_processing.py  # GPR processing tests
│   │   ├── test_compliance.py      # Compliance validation tests
│   │   └── performance_test.py     # Performance benchmarking
│   └── deployment/                 # Deployment utilities
│       ├── build_containers.sh     # Build Docker containers
│       ├── start_services.sh       # Start all services
│       └── backup_data.sh          # Data backup utility
├── tests/                          # Comprehensive test suite
│   ├── unit/                       # Unit tests
│   │   ├── test_gpr_processing.py  # GPR processing unit tests
│   │   ├── test_compliance.py      # Compliance engine tests
│   │   ├── test_ml_models.py       # ML model tests
│   │   └── test_rag_pipeline.py    # RAG pipeline tests
│   ├── integration/                # Integration tests
│   │   ├── test_api_endpoints.py   # API integration tests
│   │   ├── test_database.py        # Database integration tests
│   │   └── test_end_to_end.py      # Full workflow tests
│   ├── datasets/                   # Dataset validation tests
│   │   ├── test_twente_loader.py   # Twente dataset tests
│   │   ├── test_mojahid_loader.py  # Mojahid dataset tests
│   │   └── test_usag_loader.py     # USAG dataset tests
│   └── fixtures/                   # Test data and fixtures
│       ├── sample_gpr_files/       # Sample GPR files for testing
│       ├── mock_datasets/          # Mock dataset responses
│       └── test_compliance_docs/   # Test compliance documents
├── docs/                           # Documentation
│   ├── api/                        # API documentation
│   │   ├── openapi.json            # OpenAPI specification
│   │   └── endpoints.md            # Endpoint documentation
│   ├── datasets/                   # Dataset documentation
│   │   ├── data_sources.md         # Open data source documentation
│   │   ├── integration_guide.md    # Dataset integration guide
│   │   └── licensing.md            # Dataset licensing information
│   ├── compliance/                 # Compliance documentation
│   │   ├── pas128_guide.md         # PAS 128 implementation guide
│   │   ├── quality_levels.md       # Quality level documentation
│   │   └── audit_requirements.md   # CDM 2015 audit requirements
│   ├── deployment/                 # Deployment documentation
│   │   ├── local_setup.md          # Local development setup
│   │   ├── docker_guide.md         # Docker deployment guide
│   │   └── troubleshooting.md      # Common issues and solutions
│   └── user_guide/                 # User documentation
│       ├── getting_started.md      # Getting started guide
│       ├── gpr_processing.md       # GPR processing workflow
│       ├── report_generation.md    # Report generation guide
│       └── mobile_app.md           # Mobile app usage guide
├── config/                         # Configuration files
│   ├── development.env             # Development environment config
│   ├── production.env              # Production environment config
│   ├── model_configs/              # ML model configurations
│   └── compliance_configs/         # PAS 128 compliance configurations
├── requirements.txt                # Python backend dependencies
├── package.json                    # Node.js dependencies (if any)
├── .env.example                    # Environment variables template
├── .gitignore                      # Git ignore rules
├── .dockerignore                   # Docker ignore rules
├── CLAUDE.md                       # Claude Code configuration (existing)
└── README.md                       # Project documentation
```

### Key Features of This Structure

#### 🎯 **Domain-Specific Organization**
- **GPR Processing Pipeline**: Dedicated modules for SEG-Y, DZT, DT1 parsing and signal processing
- **Compliance Engine**: PAS 128 quality level classification and validation
- **Open Dataset Integration**: Automated loaders for University of Twente, Mojahid, USAG, and BGS datasets
- **RAG Pipeline**: Semantic chunking and retrieval for regulatory documents

#### 🔧 **Technical Architecture**
- **Multi-Database Layer**: PostgreSQL + PostGIS for spatial data, ChromaDB for vectors
- **Microservices Ready**: Modular backend structure supporting containerization
- **PWA Frontend**: React-based Progressive Web App for offline field work
- **ML Pipeline**: Complete training, evaluation, and inference pipeline

#### 📊 **Data Management**
- **Raw Dataset Storage**: Organized storage for all open source datasets
- **Processed Data**: Training/validation splits and pre-computed embeddings
- **Synthetic Data**: gprMax simulation models for data augmentation
- **Model Artifacts**: Trained models with versioning and evaluation reports

#### 🛡️ **Compliance & Quality**
- **PAS 128 Resources**: Specification documents, quality levels, and validation rules
- **Testing Framework**: Unit, integration, and dataset-specific tests
- **Documentation**: Comprehensive guides for API, datasets, and compliance
- **Configuration Management**: Environment-specific configs and model parameters

#### 🚀 **Development & Deployment**
- **Local Infrastructure**: Docker Compose setup for complete local development
- **Automation Scripts**: Dataset download, model training, and testing utilities
- **CI/CD Ready**: Structure supports automated testing and deployment pipelines
- **Monitoring**: Health checks and logging configuration

This structure provides a production-ready foundation while maintaining simplicity for local development and testing with real open source datasets.

---

## Dataset Requirements & Specifications

### Core Dataset Categories

#### 1. Regulatory & Compliance Documents

**PAS 128:2022 Specification**
- **Format**: PDF document (200+ pages)
- **Source**: British Standards Institution (BSI)
- **Processing Required**: Semantic chunking, hierarchical parsing
- **Volume**: Single document, 500+ semantic chunks
- **Usage**: RAG knowledge base for compliance checking

**Quality Level Decision Trees**
- **Format**: Flowcharts, decision matrices
- **Content**: QL-A, QL-B, QL-C, QL-D classification criteria
- **Processing**: Convert to algorithmic rules
- **Volume**: 20+ decision paths

#### 2. Geophysical Survey Data

**Ground Penetrating Radar (GPR) Data**
- **Format**: SEG-Y, GSSI DZT, Sensors & Software DT1
- **Content**: Radar waveform traces
- **Sample Rate**: 512-1024 samples per trace
- **File Size**: 50MB-2GB per survey
- **Volume Needed**:
  - Training: 10,000+ files
  - Validation: 2,000+ files
  - Testing: 1,000+ files
- **Labeling Required**: Utility type, depth, confidence

**Electromagnetic Induction (EMI) Data**
- **Format**: CSV, proprietary formats (RD8000, CAT4)
- **Content**: Signal strength, frequency, depth estimates
- **Data Points**: 1000+ per survey line
- **Volume Needed**: 5,000+ survey files

#### 3. Utility Records & Documentation

**Historical Utility Maps**
- **Format**: PDF (scanned), CAD, GIS shapefiles
- **Content**: Gas/electric/water/telecom/sewer networks
- **Quality Issues**: 30-50% positional accuracy, missing depth info
- **Volume Needed**: 10,000+ documents
- **Processing**: OCR, georeferencing, digitization

**As-Built Drawings**
- **Format**: PDF, DWG, DXF
- **Content**: Construction drawings with utility positions
- **Accuracy**: Variable (1-5m positional error)
- **Volume**: 5,000+ drawings

#### 4. Incident & Strike Data

**HSE RIDDOR Reports**
- **Format**: Structured database, CSV exports
- **Content**: Strike location, utility type, damage severity, cost impact
- **Volume**: 15,000+ UK incidents annually
- **Historical Data**: 10 years minimum

**Near-Miss Database**
- **Format**: Excel, PDF, proprietary systems
- **Content**: Near-miss location, utility exposed but not damaged
- **Volume**: 20,000+ reports
- **Value**: Identifies high-risk zones

### Sample Datasets for Development

**Sample Dataset List (Summary)**
To directly address development needs, here's a list of realistic sample datasets and sources to kick-start your development, focusing on UK-aligned content:

1. **University of Twente GPR Utility Survey Dataset** – 125 real GPR scans with trench-verified utility locations (open access)
   - URL: https://data.4tu.nl/datasets/96303227-5886-41c9-8607-70fdd2cfe7c1

2. **Mojahid et al. (2024) Utility GPR Images** – 2,239 labeled radar images of buried pipes/cables (open access, Mendeley)
   - URL: https://data.mendeley.com/datasets/ww7fd9t325/1

3. **USAG Utility Strike Reports (2019–2020)** – PDFs summarizing ~2k+ UK utility strike incidents per year (free reports)
   - URL: https://www.utilitystrikeavoidancegroup.org/reports/
   - 2019 Report: https://www.utilitystrikeavoidancegroup.org/reports/#:~:text=2019%20Utility%20Strike%20Damages%20Report

4. **HSE CDM 2015 Regulations Text** – Legal guidance for safe utility work (free HSE publication)
   - URL: https://www.hse.gov.uk/pubns/books/l153.htm

5. **PAS 128:2022 Standard** – Specification for utility detection (obtain from BSI, with free summary guides available)
   - Client Guide: https://www.cices.org/hawkfile/386/original/PAS128%20Client%20Specification%20Guide%20Sep%2022%20final.pdf

6. **USGS Open EMI Log Dataset (Example)** – Electromagnetic induction data logs (environmental, demonstrates EMI data structure)
   - URL: https://data.usgs.gov/datacatalog/data/USGS:598894cce4b05ba66e9ffe60

7. **CGA DIRT Annual Dataset** – (Optional) North American utility damage database (open report download, for broader benchmarking)

**Additional Public Data (Contextual)**

Beyond the priority categories above, consider these data sources to enhance realism:

**Ordnance Survey Open Data** – While not containing utilities, OS OpenMap and OpenStreetMap give you surface features (buildings, roads, coordinates) to simulate realistic survey environments. They can help georeference utility data on real maps of the UK. For more detailed mapping, Ordnance Survey MasterMap (not free) provides high-precision base layers which many PAS128 surveys use for drawings.

**Geotechnical and Soil Data** – Datasets like the British Geological Survey's open data (soil maps, borehole logs) or DEFRA soil data can approximate ground conditions. Since soil type and moisture affect GPR performance, including some soil parameters can make your simulations more realistic.

**Weather and Hydrology** – Public weather archives (e.g., Met Office historical data) can supply rainfall or moisture conditions for sites/dates, which is useful if calibrating GPR signal attenuation. Similarly, flood maps or water table data add context about high-saturation areas where detection is harder.

**Synthetic Data Tools** – Consider using open-source simulators like gprMax (which is open) to generate synthetic GPR data under controlled conditions. gprMax comes with example models (e.g. a synthetic sedimentary scenario). You can tweak these to simulate utilities (pipes of certain diameters/materials) buried in different soils, producing "realistic" radargrams to augment your training set.
- gprMax examples: https://emanuelhuber.github.io/RGPR/80_RGPR_GPR-data-free-to-download/

## Additional UK Open Source Datasets (Comprehensive Research)

### UK-Specific GPR & Geophysical Data

**De Montfort University GPR Dataset**
- **URL**: https://figshare.dmu.ac.uk/articles/dataset/Ground_penetrating_radar_dataset/8323049
- **DOI**: https://doi.org/10.21253/DMU.8323049.v1
- **Description**: GPR dataset with .dat files obtained via GPR with sampling points on equally spaced grids (50mm distance)
- **Licensing**: Creative Commons Attribution 4.0 International License
- **Relevance**: High for GPR pattern recognition and processing techniques

**GprMax Software (University of Edinburgh)**
- **URL**: www.gprmax.org
- **Contact**: Professor Antonis Giannopoulos
- **Description**: Free software for modeling GPR responses from arbitrarily complex targets with 2D and 3D examples
- **Licensing**: Free for academic and commercial use
- **Relevance**: High for GPR simulation and utility detection modeling

**BGS Ground Penetrating Radar Data**
- **URL**: https://www.data.gov.uk/dataset/68444cae-2613-4199-9785-17f7b46e3ef0/ground-penetrating-radar-data-from-bgs-iceland-glacier-observatory-project-2012-2014
- **DOI**: https://doi.org/10.5285/e2386bf1-926d-4c32-9b54-a3cf8f143cc6
- **Description**: GPR data using Sensors and Software PulseEKKO Pro GPR system (2012-2014)
- **Data Format**: .DT1 files, header (.HD) files, GPS (.GPS) files, GIS shapefiles
- **Licensing**: Requires permission from BGS, copyright NERC

### UK Utility Infrastructure Data

**UK Power Networks Open Data Portal**
- **URL**: https://ukpowernetworks.opendatasoft.com/explore/
- **Description**: 55 datasets containing over 2 million records of electricity network assets, locations, capacity, and usage
- **Access**: Requires login for full dataset access
- **Contact**: opendata@ukpowernetworks.co.uk
- **Relevance**: High for electrical utility infrastructure mapping

**Northern Gas Networks Open Data Portal**
- **URL**: https://northerngasopendataportal.co.uk/
- **Description**: 23 datasets covering gas network infrastructure in North England
- **Data Types**: Network Boundaries, Live Distribution Mains, Transmission Pipelines, Smart Meter Statistics
- **Data Formats**: Excel, PDF, GeoPackage, Geospatial files
- **Licensing**: Creative Commons BY 4.0
- **Relevance**: High for gas utility infrastructure mapping

**National Underground Asset Register (NUAR)**
- **URL**: https://www.gov.uk/guidance/national-underground-asset-register-nuar
- **Description**: Digital map of underground pipes and cables for gas, electric, water, internet, phone connections
- **Coverage**: England, Wales, Northern Ireland (expanding from North East England, Wales, London)
- **Access**: Available to eligible organizations in launch regions
- **Relevance**: Extremely high for comprehensive utility infrastructure mapping

### UK Geotechnical & Soil Data

**BGS Single Onshore Borehole Index (SOBI)**
- **URL**: https://www.bgs.ac.uk/datasets/boreholes-index/
- **Description**: Over 1 million records of boreholes, shafts, and wells from Great Britain dating back to 1790
- **Data Formats**: GIS point data (ESRI, MapInfo, others available by request)
- **Licensing**: Open Government Licence
- **Relevance**: Very high for geotechnical soil conditions and subsurface investigation

**BGS AGS Download Service**
- **URL**: https://agsapi.bgs.ac.uk/
- **Description**: Free access to geotechnical data in AGS format (industry standard)
- **Content**: Over 10,000 boreholes in AGS format, 2 terabytes of downloadable geoscience data
- **Data Format**: AGS version 4 standard
- **Licensing**: Open access with donor consent
- **Relevance**: Very high for geotechnical analysis and subsurface conditions

**BGS National Geotechnical Properties Database (NGPD)**
- **URL**: https://www.bgs.ac.uk/geological-research/science-facilities/engineering-geotechnical-capability/national-geotechnical-properties-database/
- **Description**: 7,370 projects, 178,436 holes, 3.6M in situ field records, 879,293 samples, 5.2M lab test records
- **Access**: Contact BGS for access requirements
- **Relevance**: Very high for detailed geotechnical properties

### UK Incident & Safety Data

**HSE Statistics Portal**
- **URL**: https://www.hse.gov.uk/statistics/
- **Description**: 45 years of incident, accident investigation and safety data archive
- **Notable**: 1,230 safety-related electrical incidents reported in 2019
- **Licensing**: Government data licensing
- **Relevance**: High for utility strike incident analysis

**LSBUD (LinesearchbeforeUdig) Data**
- **URL**: https://lsbud.co.uk/
- **Description**: Free search service covering over 2 million kilometres of underground and overhead assets
- **Coverage**: 71% of all UK digging work, 60% of utility providers (900,000 km of 1.5 million km total)
- **Relevance**: Very high for excavation planning and utility strike prevention

### Academic Research Data

**EPSRC "Mapping The Underworld" Project**
- **Grant Reference**: EP/F065965/1
- **Lead Institution**: University of Birmingham (Professor Chris Rogers)
- **Collaborators**: Universities of Bath, Leeds, Sheffield, Southampton
- **Description**: £3.5M project developing multi-sensor device using GPR, acoustics, and electromagnetic technologies
- **Relevance**: Very high for comprehensive utility detection research

**University of Edinburgh - Ground Penetrating Radar Modelling**
- **Lead**: Professor Antonis Giannopoulos
- **URL**: https://www.research.ed.ac.uk/en/publications/modelling-ground-penetrating-radar-by-gprmax
- **Software**: GprMax (www.gprmax.org)
- **Access**: Free download for academic and commercial use
- **Relevance**: High for GPR simulation and utility detection modeling

### Government & Regulatory Data

**Ordnance Survey Open Data**
- **URL**: https://osdatahub.os.uk/downloads/open
- **Key Datasets**: OS MasterMap Topography Layer, Boundary-Line, OS Open Greenspace, OS OpenMap Local, Code-Point Open
- **Licensing**: Open Government Licence
- **Relevance**: High for base mapping and spatial reference data

**London Datastore**
- **URL**: https://data.london.gov.uk/
- **Key Features**: City Hall's Infrastructure Mapping Application with data from 26 utilities
- **Benefits**: Saved 426 days of road disruption, £860k in construction costs
- **Relevance**: High for London-specific utility coordination data

### Access Priority Recommendations

**Immediately Accessible (Open Access)**:
1. BGS SOBI - Download directly under Open Government Licence
2. BGS AGS Download Service - Free access to geotechnical data
3. De Montfort University GPR Dataset - Creative Commons licensed
4. Northern Gas Networks Open Data - 23 datasets available
5. GprMax Software - Free GPR modeling tool

**Requires Registration/Contact**:
1. UK Power Networks Open Data - Requires login
2. NUAR - Eligible organizations only
3. HSE Statistics - May require formal data request
4. University research datasets - Contact researchers directly

---

## 8-Week MVP Development Plan

### Pre-Development (Week 0)

#### Team Formation
| Role | Requirement | Time Commitment | Cost |
|------|-------------|-----------------|------|
| Technical Lead | RAG/LLM expertise | Full-time | £60K (6 months) |
| ML Engineer | GPR data processing | Full-time | £60K (6 months) |
| Domain Expert | PAS 128 certified | Part-time consultant | £30K |
| Product Designer | Field UX experience | 4 months | £40K |
| Customer Success | Survey industry knowledge | Full-time | £40K (6 months) |

#### Infrastructure Setup
- AWS account with billing alerts
- GitHub organization with CI/CD
- Pinecone account (Starter plan)
- OpenAI API key ($500 initial credit)
- Development tools (SonarQube, monitoring)

### Phase 1: Data Foundation (Weeks 1-2)

#### Week 1: PAS 128 Knowledge Base
**Day 1-2: Document Processing**
- Parse PAS 128 specification
- Extract numbered sections and requirements
- Create requirement traceability matrix

**Day 3-4: Knowledge Structuring**
- Hierarchical taxonomy (1.0, 1.1, 1.1.1)
- Quality Level criteria (QL-A to QL-D)
- Survey method requirements
- Deliverable specifications

**Day 5: Embedding Generation**
- Create Pinecone index
- Generate embeddings for each section
- Test retrieval accuracy (target: 95% relevance)

#### Week 2: Data Ingestion Pipeline
**Day 6-7: File Parsers**
- GPR Parser: SEG-Y format reader, signal extraction
- PDF/OCR Pipeline: Tesseract integration, table extraction

**Day 8-9: CAD Processing**
- DXF/DWG parser setup
- Layer extraction and coordinate handling
- Geometry simplification

**Day 10: Integration Testing**
- End-to-end pipeline test
- Error handling verification
- Performance benchmarking

### Phase 2: Intelligence Layer (Weeks 3-4)

#### Week 3: RAG Engine Development
**Day 11-12: Core RAG Setup**
- Query processing with intent classification
- Query expansion logic with synonyms
- Abbreviation dictionary for utilities

**Day 13-14: Retrieval System**
- Multi-index search implementation
- Hybrid search (keyword + semantic)
- Reranking with Cohere
- Result caching

**Day 15: LLM Integration**
- GPT-4 API integration
- Prompt templates for PAS 128
- Citation system
- Hallucination prevention

#### Week 4: ML Models
**Day 16-17: GPR Interpretation**
- Hyperbola detection algorithm
- Depth estimation using velocity analysis
- Confidence scoring
- Initial model training

**Day 18-19: Risk Scoring**
- Feature engineering from historical data
- Scoring algorithm development
- Validation metrics

**Day 20: Model Integration**
- API endpoints for models
- Error handling and performance optimization
- Testing suite

### Phase 3: Compliance & Reporting (Weeks 5-6)

#### Week 5: Report Generation
**Day 21-22: Template System**
- PAS 128 compliant structure
- Section templates with dynamic content
- Citation formatting

**Day 23-24: Generation Logic**
- Data aggregation from multiple sources
- Narrative generation using LLM
- Quality level assignment automation
- Compliance checking

**Day 25: Export Functionality**
- PDF generation with client branding
- Word document export
- Excel data tables
- CAD file generation

#### Week 6: Compliance Framework
**Day 26-27: Audit System**
- Decision logging with timestamps
- Immutable storage in S3 Glacier
- Compliance reporting

**Day 28-29: Validation**
- PAS 128 checklist automation
- Completeness verification
- Accuracy checks and warnings

**Day 30: Testing**
- Generate 10 test reports
- Expert review and feedback
- Final adjustments

### Phase 4: Interface & Integration (Weeks 7-8)

#### Week 7: User Interface
**Day 31-32: Web Application**
- React frontend with authentication
- Dashboard design
- File upload interface

**Day 33-34: Core Features**
- Project management interface
- Data upload workflow
- Report viewer and export

**Day 35: Mobile PWA**
- PWA setup with offline capability
- Photo capture with GPS
- Field data collection interface

#### Week 8: System Integration
**Day 36-37: API Development**
- FastAPI endpoint development
- Authentication/authorization
- Rate limiting and documentation

**Day 38-39: Testing**
- End-to-end workflow testing
- Load testing (50 concurrent users)
- Security testing

**Day 40: Deployment**
- AWS production deployment
- DNS configuration and SSL
- Monitoring activation

---

## Technical Workflows

### Data Processing Pipeline

```python
# Simplified processing workflow
def process_survey_data(project_data):
    """
    Complete survey data processing workflow
    """
    # Stage 1: Data Ingestion
    gpr_data = parse_gpr_files(project_data.gpr_files)
    utility_records = process_utility_pdfs(project_data.utility_docs)
    cad_data = parse_cad_drawings(project_data.cad_files)

    # Stage 2: Data Correlation
    correlated_data = correlate_multiple_sources(
        gpr=gpr_data,
        records=utility_records,
        cad=cad_data
    )

    # Stage 3: AI Analysis
    detected_utilities = ml_interpret_gpr(gpr_data)
    risk_scores = predict_strike_risk(correlated_data)

    # Stage 4: Report Generation
    report = generate_pas128_report(
        utilities=detected_utilities,
        risks=risk_scores,
        compliance_data=correlated_data
    )

    # Stage 5: Validation
    validation_result = validate_compliance(report)

    return {
        'report': report,
        'validation': validation_result,
        'confidence_scores': extract_confidence_metrics(detected_utilities)
    }
```

### GPR Signal Processing Workflow

```python
class GPRSignalProcessor:
    """
    Ground Penetrating Radar signal processing pipeline
    """

    def process_radargram(self, segy_file):
        # 1. Load and validate SEG-Y data
        traces = self.load_segy(segy_file)
        metadata = self.extract_metadata(segy_file)

        # 2. Signal preprocessing
        filtered_traces = self.apply_bandpass_filter(traces)
        gained_traces = self.apply_gain_correction(filtered_traces)
        background_removed = self.remove_background(gained_traces)

        # 3. Feature detection
        hyperbolas = self.detect_hyperbolas(background_removed)
        utilities = self.classify_reflections(hyperbolas)

        # 4. Depth calculation
        depths = self.calculate_depths(
            utilities,
            velocity_model=metadata.soil_velocity
        )

        # 5. Confidence assessment
        confidence_scores = self.assess_confidence(
            signal_quality=self.calculate_snr(filtered_traces),
            feature_clarity=hyperbolas.quality_score,
            historical_accuracy=self.lookup_area_accuracy(metadata.location)
        )

        return ProcessedGPRData(
            utilities=utilities,
            depths=depths,
            confidence=confidence_scores,
            quality_metrics=self.calculate_quality_metrics(traces)
        )
```

### RAG Query Processing

```python
class UtilityRAGProcessor:
    """
    Specialized RAG processing for utility survey queries
    """

    def __init__(self):
        self.pas128_index = "pas128-compliance"
        self.project_index = "project-{project_id}"
        self.incident_index = "incident-database"

    def process_compliance_query(self, query, project_context):
        # 1. Query understanding and expansion
        intent = self.classify_intent(query)  # compliance, risk, procedure
        expanded_query = self.expand_technical_terms(query)

        # 2. Multi-source retrieval
        compliance_docs = self.search_index(
            self.pas128_index,
            expanded_query,
            filter={"section_type": intent}
        )

        project_history = self.search_index(
            self.project_index.format(project_id=project_context.id),
            expanded_query,
            filter={"confidence": ">0.8"}
        )

        incident_data = self.search_index(
            self.incident_index,
            expanded_query,
            filter={"location_bbox": project_context.bbox}
        )

        # 3. Context assembly with relevance scoring
        context = self.assemble_context(
            compliance_docs=compliance_docs,
            project_data=project_history,
            incident_data=incident_data,
            max_tokens=8000
        )

        # 4. Response generation with citations
        response = self.generate_response(
            query=query,
            context=context,
            response_type=intent,
            citations_required=True
        )

        return response
```

### Risk Assessment Workflow

```python
class UtilityStrikeRiskAssessor:
    """
    Comprehensive risk assessment for utility strikes
    """

    def assess_project_risk(self, project_data):
        # 1. Extract risk features
        features = self.extract_risk_features(
            detected_utilities=project_data.utilities,
            construction_plan=project_data.excavation_plan,
            soil_conditions=project_data.geotechnical_data,
            historical_incidents=project_data.area_incidents,
            detection_confidence=project_data.confidence_scores
        )

        # 2. Multiple risk models
        probability_models = {
            'historical': self.historical_incident_model(features),
            'geometric': self.geometric_conflict_model(features),
            'confidence': self.detection_confidence_model(features),
            'environmental': self.environmental_factor_model(features)
        }

        # 3. Ensemble prediction
        weighted_risk = self.combine_risk_scores(
            probability_models,
            weights={'historical': 0.3, 'geometric': 0.3,
                    'confidence': 0.2, 'environmental': 0.2}
        )

        # 4. Risk categorization and recommendations
        risk_level = self.categorize_risk(weighted_risk)
        mitigation_steps = self.generate_mitigation_recommendations(
            risk_level, features
        )

        return RiskAssessment(
            overall_score=weighted_risk,
            risk_level=risk_level,
            contributing_factors=self.explain_risk_factors(features),
            mitigation_recommendations=mitigation_steps,
            confidence_interval=self.calculate_prediction_confidence()
        )
```

---

## Implementation Checklist

### Critical Path Items

#### Pre-Development Setup
- [ ] **Legal & Compliance**
  - [ ] Company registration and IP protection
  - [ ] BSI membership for PAS 128 access
  - [ ] Insurance quotes (£5M professional indemnity)
  - [ ] GDPR compliance framework

- [ ] **Team Formation**
  - [ ] Technical Lead (RAG/LLM expertise)
  - [ ] Domain Expert (PAS 128 certified)
  - [ ] ML Engineer (GPR processing)

- [ ] **Infrastructure**
  - [ ] AWS account with security setup
  - [ ] Pinecone account (vector database)
  - [ ] OpenAI API access
  - [ ] Development environment

#### Development Phase Checklist

**Week 1-2: Data Foundation**
- [ ] PAS 128 specification processed and vectorized
- [ ] GPR parser for SEG-Y format
- [ ] PDF/OCR pipeline for utility records
- [ ] CAD file processing (DXF/DWG)
- [ ] Data validation and quality checks

**Week 3-4: Intelligence Layer**
- [ ] RAG engine with multi-index search
- [ ] Query processing and expansion
- [ ] LLM integration (GPT-4)
- [ ] GPR interpretation model
- [ ] Risk scoring algorithm

**Week 5-6: Compliance & Reporting**
- [ ] PAS 128 report templates
- [ ] Automated report generation
- [ ] Compliance validation system
- [ ] Multi-format export (PDF, Word, Excel)
- [ ] Audit trail implementation

**Week 7-8: Interface & Integration**
- [ ] React web application
- [ ] Mobile PWA for field use
- [ ] FastAPI backend
- [ ] Authentication and authorization
- [ ] Production deployment on AWS

#### Quality Assurance
- [ ] **Testing Strategy**
  - [ ] Unit tests (>80% coverage)
  - [ ] Integration tests
  - [ ] End-to-end workflow tests
  - [ ] Load testing (50 concurrent users)
  - [ ] Security testing

- [ ] **Performance Benchmarks**
  - [ ] GPR processing: <30 seconds
  - [ ] Report generation: <10 minutes
  - [ ] API response: <200ms P95
  - [ ] System uptime: >99.5%

### Customer Validation Phase

#### Lighthouse Customer Preparation
- [ ] **Murphy Group**
  - [ ] Pilot agreement signed
  - [ ] Success criteria defined (75% time savings)
  - [ ] Training session scheduled
  - [ ] Weekly feedback meetings

- [ ] **Kier Utilities**
  - [ ] Technical workshop completed
  - [ ] Integration requirements gathered
  - [ ] Pilot scope defined

- [ ] **Cardiff Council**
  - [ ] Security assessment passed
  - [ ] Compliance verification
  - [ ] Procurement process initiated

#### Success Metrics Tracking
- [ ] Time savings measurement (target: >75%)
- [ ] Accuracy verification (target: >95%)
- [ ] User satisfaction (NPS >50)
- [ ] System reliability (>99% uptime)
- [ ] Case study development

---

## Risk Management Framework

### Technical Risk Mitigation

**GPR Interpretation Accuracy**
- Risk: ML model fails to accurately interpret GPR data
- Mitigation: Human-in-loop validation, confidence thresholds
- Monitoring: False negative rate <5%, post-excavation accuracy

**LLM Hallucinations**
- Risk: Generated reports contain incorrect information
- Mitigation: Citation-only mode, template constraints
- Monitoring: Hallucination rate per report, customer corrections

**Data Quality Issues**
- Risk: Poor quality input data affecting results
- Mitigation: Validation pipeline, error handling, data redundancy
- Monitoring: Data quality scores, completeness metrics

### Business Risk Mitigation

**Slow Market Adoption**
- Risk: Longer than expected customer adoption
- Mitigation: Free pilots, education campaign, bottom-up adoption
- Contingency: Adjust pricing, extend pilots, pivot strategy

**Competitive Response**
- Risk: Established players launch competing solutions
- Mitigation: Fast execution, differentiation, customer lock-in
- Monitoring: Competitive analysis, market positioning

**Funding Challenges**
- Risk: Unable to raise sufficient capital
- Mitigation: Multiple funding sources, capital efficiency
- Contingency: Revenue-based financing, bootstrap approach

---

## Performance Optimization

### Caching Strategy
```
Multi-Layer Cache:
L1: Browser (1 hour) - Static assets
L2: CDN (CloudFront) - Reports, documents
L3: Application (Redis) - Query results, embeddings
L4: Database - Query cache, materialized views
L5: ML Model - Inference results, feature vectors
```

### Scaling Considerations
- Horizontal scaling with Kubernetes
- Auto-scaling policies based on load
- Database optimization and indexing
- CDN for global distribution
- Async processing for heavy workloads

---

## Success Metrics

### Development KPIs
- Story points completed per sprint
- Code coverage (target: >80%)
- Technical debt ratio (<10%)
- Bug discovery rate

### Business KPIs
- Customer acquisition (target: 30 Year 1)
- Revenue growth (target: £750K ARR Year 1)
- User satisfaction (NPS >50)
- Time to value (<30 days)

### Technical KPIs
- Report generation time (<10 minutes)
- System accuracy (>95% vs manual)
- API performance (<200ms P95)
- System reliability (>99.5% uptime)

---

*This consolidated technical implementation guide provides the complete roadmap for building and deploying the Underground Utility Detection Platform MVP within the 8-week timeline, including all technical specifications, implementation workflows, and success criteria.*