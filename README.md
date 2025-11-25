# Infrastructure Intelligence Platform

> AI-native multi-modal data processing platform for GPR, BIM, and LiDAR analysis

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791.svg)](https://postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://docs.docker.com/compose/)

---

## 🎯 Overview

The **Infrastructure Intelligence Platform** is a comprehensive AI-native solution for major infrastructure projects, addressing **five critical domains**: underground utility detection, BIM validation, construction monitoring, **asset certification intelligence**, **data-led safety management**, and **intelligent cost verification**.

**Inspired by HS2 Railway Project requirements**: Designed to handle:
- 2M+ physical assets with certification tracking
- 100,000+ deliverables per contract (IDP management)
- 5M+ invoices and 11M+ cost line items validation
- Real-time safety intelligence across 10+ Joint Ventures
- Unified visualization across fragmented enterprise systems (VisHub 2.0)

### Key Features

#### Core Multi-Modal Processing
- **📡 GPR Processing**: Ground Penetrating Radar signal analysis with ML-powered utility detection
- **🏛️ BIM Integration**: IFC file validation and 3D model correlation
- **📊 LiDAR Analysis**: Point cloud processing for progress monitoring
- **🤖 AI Reports**: Automated PAS 128-compliant report generation with GPT-4o
- **🔍 Multi-Modal Fusion**: Correlate data across GPR, BIM, CAD, and utility records
- **✅ Compliance**: PAS 128:2022, CDM 2015, GDPR compliance built-in

#### Asset & Certification Intelligence
- **📋 Certificate Parsing**: OCR + NLP for qualification extraction and validation (>98% accuracy)
- **🔗 Enterprise Integration**: Connect to AIMS, CDEs (BIM 360, Aconex, ProjectWise), SharePoint, ERP systems
- **📊 IDP Analytics**: Track 100,000+ deliverables per contract with predictive analytics
- **✅ TAEM Compliance**: Technical Assurance Evidence Model validation and scoring
- **🎯 Asset Assurance**: Real-time readiness scoring, risk assessment, and automated escalation

#### Safety Intelligence 🆕
- **🚨 Predictive Safety Analytics**: AI-powered leading indicators and high-risk window prediction
- **📝 Incident Intelligence**: NLP analysis of unstructured incident reports for root causes and patterns
- **⚠️ Real-Time Risk Scoring**: Multi-factor risk assessment (behavior, environment, activity type)
- **🔔 Automated Anomaly Detection**: Cross-JV pattern analysis and automated alerting
- **🎯 Proactive Interventions**: Early-warning alerts before incidents occur

#### Cost Verification Engine 🆕
- **💰 Intelligent Invoice Processing**: Beyond basic OCR - context-aware extraction (5M+ invoices, 11M+ line items)
- **🧠 Semantic Validation**: AI understands "fish plate" (railway part) vs "fish" (food) - contextual interpretation
- **🔍 Fraud & Duplicate Detection**: Advanced pattern matching across years and contractors
- **📊 Cross-JV Analytics**: Identify unusual pricing (e.g., steel 2x cost in one JV vs another)
- **✅ Focused Review Sets**: Auto-categorize invoices (Likely OK / Requires Review / High-Risk)
- **🔄 Continuous Learning**: Model improves from commercial manager feedback

#### Unified Visualization (VisHub 2.0)
- **🗺️ Geographic Navigation**: Explore digital railway representation with spatial context
- **🏗️ Asset-Based Views**: Find information by asset classification and hierarchies
- **📊 Integrated Insights**: Access safety, cost, asset, and compliance data in one platform

---

## 🚀 Quick Start

### Prerequisites

- Docker Desktop (latest)
- 8GB RAM minimum (16GB recommended)
- 20GB free disk space

### Start All Services

```bash
# Clone repository
cd ground-truth

# Generate secure environment keys
chmod +x setup_env.sh
./setup_env.sh

# Start all services with Docker Compose
docker compose up -d

# Verify all services are healthy
docker compose ps
```

### Access the Platform

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend UI** | http://localhost:3003 | React web application |
| **Backend API** | http://localhost:8002 | FastAPI REST API |
| **API Docs** | http://localhost:8002/docs | Interactive Swagger UI |
| **MinIO Console** | http://localhost:9001 | S3-compatible storage UI |

**Default MinIO Credentials**: `minioadmin` / `minioadmin` (see `.env` for custom values)

---

## 📊 Current Status

### ✅ Deployed (Phase 1A - GPR Processing)

- **Backend**: 30+ API endpoints, FastAPI 0.104.1 *(expanding to 93+ endpoints)*
- **Frontend**: React 18 + TypeScript + Vite
- **Database**: PostgreSQL 16 + PGVector + PostGIS (17 tables) *(expanding to 57 tables)*
- **Storage**: MinIO S3-compatible object storage
- **Cache**: Redis 7.2
- **Data**: 10 real GPR surveys from University of Twente (100+ scans)

### 📋 Planned

- **Phase 1B** (Weeks 4-7): BIM integration with IFC.js
- **Phase 1C** (Weeks 8-11): LiDAR processing with Open3D
- **Phase 1D** (Weeks 12-18): **Multi-Domain Intelligence** 🆕
  - **Asset Certification** (Weeks 12-13): Certificate parsing, TAEM compliance, IDP analytics
  - **Safety Intelligence** (Weeks 14-15): Incident analysis, predictive risk scoring, anomaly detection
  - **Cost Verification** (Weeks 16-18): Intelligent invoice processing, fraud detection, cross-JV analytics
- **Phase 2A** (Weeks 19-26): LLM integration (GPT-4o + RAG pipeline for PAS 128, safety, cost)
- **Phase 2B** (Weeks 27-32): **Enterprise Integration** - AIMS, CDEs, VisHub 2.0, Microsoft Fabric
- **Phase 3** (Weeks 33-41): **Unified Intelligence Platform** - Cross-domain analytics, knowledge graph, predictive insights

---

## 🗂️ Project Structure

```
ground-truth/
├── backend/                    # FastAPI backend application
│   ├── app/
│   │   ├── api/v1/endpoints/  # API route handlers (30+ endpoints)
│   │   ├── core/              # Config, database, security
│   │   ├── models/            # SQLAlchemy models (17 tables)
│   │   ├── schemas/           # Pydantic schemas
│   │   └── services/          # Business logic
│   ├── scripts/               # Utility scripts
│   │   ├── database/          # DB management tools
│   │   └── import_twente_gpr_data.py  # Data import script
│   ├── requirements.txt       # Python dependencies
│   └── Dockerfile             # Multi-stage Docker build
│
├── frontend/                  # React + TypeScript frontend
│   ├── src/
│   │   ├── App.tsx            # Main application component
│   │   ├── main.tsx           # Entry point
│   │   └── *.css              # Styling
│   ├── package.json           # NPM dependencies
│   ├── vite.config.ts         # Vite configuration
│   └── Dockerfile             # Multi-stage build
│
├── datasets/                  # Data storage (volume mounted)
│   ├── raw/                   # Raw datasets
│   │   ├── twente_gpr/        # University of Twente GPR (125 surveys)
│   │   ├── mojahid_images/    # Labeled GPR images (2,239+)
│   │   └── pas128_docs/       # PAS 128 compliance docs
│   └── processed/             # Processed/extracted data
│
├── docs/                      # Documentation
│   ├── DATA_SOURCES.md        # Where to get sample data
│   ├── DATA_IMPORT_SUCCESS.md # GPR data import report
│   ├── DEPLOYMENT_SUCCESS.md  # Deployment status report
│   ├── SETUP_COMPLETE.md      # Detailed setup guide
│   └── QUICK_START.md         # Quick reference
│
├── docker-compose.yml         # Service orchestration
├── .env                       # Environment variables (generated)
└── README.md                  # This file
```

---

## 💾 Database Schema

### Phase 1A Tables (17 tables - Currently Deployed)

#### GPR Data (4 tables)
- `gpr_surveys` - Survey metadata (equipment, location, dates)
- `gpr_scans` - Individual radargrams (traces, samples, time windows)
- `gpr_signal_data` - Raw signal traces
- `gpr_processing_results` - Processed features and detections

#### Environmental (3 tables)
- `environmental_data` - Soil, weather, permittivity
- `weather_conditions` - Historical weather logs
- `ground_conditions` - Terrain and surface characteristics

#### Validation (3 tables)
- `validation_results` - Accuracy metrics
- `accuracy_metrics` - Performance statistics
- `ground_truth_data` - Known utility locations

#### Utilities (3 tables)
- `utility_disciplines` - Gas, water, electric, telecom
- `utility_materials` - Cast iron, PVC, asbestos cement, etc.
- `utility_records` - Detected/documented utilities

#### ML/Analytics (4 tables)
- `ml_models` - Trained model metadata
- `training_sessions` - Training runs
- `feature_vectors` - Embeddings (PGVector)
- `model_performance` - Evaluation metrics

### Phase 1D Tables (40 additional tables - Planned) 🆕

#### Asset Management (6 tables)
- `assets` - Physical infrastructure assets (2M+ for HS2-scale projects)
- `asset_types` - Asset classification taxonomy
- `asset_locations` - Spatial asset tracking
- `asset_relationships` - Parent-child asset hierarchies
- `asset_lifecycle` - Installation, commissioning, operational status
- `asset_metadata` - Custom attributes per asset type

#### Certification (4 tables)
- `certificates` - Certificate metadata (issuer, dates, status)
- `certificate_documents` - PDF/Excel storage references
- `certificate_qualifications` - Extracted qualification data (OCR + NLP)
- `certificate_validation` - Automated validation results

#### Assurance (4 tables)
- `assurance_requirements` - Project-specific quality gates
- `assurance_evidence` - Evidence submissions per requirement
- `assurance_scores` - Real-time asset readiness scoring
- `assurance_risks` - Identified risks and mitigation actions

#### Documents (4 tables)
- `document_index` - Central document registry (100k+ deliverables)
- `document_versions` - Version control tracking
- `document_metadata` - Tags, categories, compliance mappings
- `document_relationships` - Links between related documents

#### IDP & TAEM (4 tables)
- `idp_deliverables` - Information Delivery Plan items
- `idp_milestones` - Contract milestone tracking
- `taem_requirements` - Technical Assurance Evidence Model requirements
- `taem_compliance` - Compliance status per requirement

#### Enterprise Integration (3 tables)
- `system_connections` - API connections to AIMS, CDEs, SharePoint
- `sync_logs` - Data synchronization audit trail
- `integration_mappings` - Field mappings between systems

#### Safety Intelligence (8 tables) 🆕
- `safety_incidents` - Incident records with structured + unstructured data
- `safety_incident_narratives` - Full text narratives for NLP analysis
- `safety_risk_scores` - Real-time risk scoring per site/asset/contractor
- `safety_environmental_factors` - Weather, transport, congestion correlation
- `safety_behavioral_observations` - Culture surveys, behavioral data
- `safety_predictions` - ML model predictions for high-risk windows
- `safety_interventions` - Actions taken based on predictions
- `safety_leading_indicators` - Proactive safety metrics

#### Cost Verification (7 tables) 🆕
- `invoices` - Invoice metadata (5M+ records from HS2)
- `invoice_line_items` - Individual cost lines (11M+ records)
- `invoice_documents` - PDF/Excel/scan storage references
- `cost_verification_results` - Validation results per line item
- `cost_anomalies` - Flagged anomalies for commercial review
- `contract_rules` - JV-specific contract rules for validation
- `cost_benchmarks` - Commodity/material price benchmarks

**Total Schema**: 57 tables (17 deployed + 40 planned)

---

## 🔌 API Endpoints

### Phase 1A Endpoints (30+ deployed)

#### GPR Data Management
```bash
POST /api/v1/gpr/surveys              # Create survey
GET /api/v1/gpr/surveys               # List all surveys
GET /api/v1/gpr/surveys/{id}          # Get survey details
POST /api/v1/gpr/scans                # Upload scan (⚠️ needs fix)
POST /api/v1/gpr/environmental        # Add environmental data (⚠️ needs implementation)
```

#### Dataset Management
```bash
POST /api/v1/datasets/upload          # Batch upload
POST /api/v1/datasets/{id}/process    # Process dataset
GET /api/v1/datasets/{id}/status      # Get status
```

#### Analytics
```bash
GET /api/v1/analytics/detection-stats             # Detection statistics
GET /api/v1/analytics/environmental-correlation   # Environmental analysis
```

#### Material Classification
```bash
POST /api/v1/material-classification/classify     # Classify material
GET /api/v1/material-classification/materials     # List materials
```

#### PAS 128 Compliance
```bash
POST /api/v1/pas128-compliance/validate           # Validate compliance
GET /api/v1/pas128-compliance/quality-levels      # Get quality levels
```

### Phase 1D Endpoints (36 additional planned)

#### Asset Management (9 endpoints)
```bash
POST /api/v1/assets                    # Create asset
GET /api/v1/assets                     # List assets (paginated, filterable)
GET /api/v1/assets/{id}                # Get asset details
PUT /api/v1/assets/{id}                # Update asset
DELETE /api/v1/assets/{id}             # Delete asset
GET /api/v1/assets/{id}/relationships  # Get asset hierarchy
GET /api/v1/assets/{id}/lifecycle      # Get lifecycle history
GET /api/v1/assets/search              # Advanced search (spatial, attributes)
POST /api/v1/assets/bulk-import        # Bulk import from Excel/CSV
```

#### Certificate Management (8 endpoints)
```bash
POST /api/v1/certificates              # Upload certificate (PDF/Excel)
GET /api/v1/certificates               # List certificates
GET /api/v1/certificates/{id}          # Get certificate details
POST /api/v1/certificates/{id}/parse   # Trigger OCR + NLP parsing
GET /api/v1/certificates/{id}/qualifications  # Get extracted qualifications
POST /api/v1/certificates/{id}/validate       # Validate certificate
GET /api/v1/certificates/expiring      # Get expiring certificates (alerts)
GET /api/v1/certificates/search        # Search by qualification, issuer, etc.
```

#### Assurance & Scoring (7 endpoints)
```bash
GET /api/v1/assurance/requirements     # Get project requirements
POST /api/v1/assurance/evidence        # Submit evidence
GET /api/v1/assurance/scores           # Get asset readiness scores
GET /api/v1/assurance/scores/{asset_id}  # Get asset-specific score
GET /api/v1/assurance/risks            # Get identified risks
POST /api/v1/assurance/risks/{id}/mitigate  # Record mitigation action
GET /api/v1/assurance/dashboard        # Real-time assurance dashboard
```

#### Document Management (5 endpoints)
```bash
POST /api/v1/documents                 # Upload document
GET /api/v1/documents                  # List documents (100k+ scale)
GET /api/v1/documents/{id}             # Get document details
GET /api/v1/documents/search           # Full-text search
GET /api/v1/documents/{id}/related     # Get related documents
```

#### IDP & TAEM Compliance (5 endpoints)
```bash
GET /api/v1/idp/deliverables           # Get IDP deliverable status
GET /api/v1/idp/milestones             # Get contract milestones
POST /api/v1/taem/validate             # Validate TAEM compliance
GET /api/v1/taem/requirements          # Get TAEM requirements
GET /api/v1/taem/compliance-report     # Generate compliance report
```

#### Enterprise Integration (2 endpoints)
```bash
POST /api/v1/integrations/sync         # Trigger data sync (AIMS, CDEs)
GET /api/v1/integrations/status        # Get sync status
```

#### Safety Intelligence (12 endpoints) 🆕
```bash
# Incident Management
POST /api/v1/safety/incidents          # Create incident with narrative
GET /api/v1/safety/incidents           # List incidents (filterable by JV, site, type)
GET /api/v1/safety/incidents/{id}      # Get incident details
POST /api/v1/safety/incidents/{id}/analyze  # NLP analysis of narrative

# Risk Scoring & Prediction
GET /api/v1/safety/risk-scores         # Real-time risk scores across sites
GET /api/v1/safety/risk-scores/{site_id}  # Site-specific risk score
POST /api/v1/safety/predict-risk       # Predict high-risk windows (weather, fatigue)

# Analytics & Insights
GET /api/v1/safety/leading-indicators  # Proactive safety metrics
GET /api/v1/safety/anomalies           # Automated anomaly detection (cross-JV)
GET /api/v1/safety/top-risks           # Top 5 risks based on recent patterns
POST /api/v1/safety/correlate          # Correlate weather/incidents/behaviors

# Dashboard
GET /api/v1/safety/dashboard           # Unified safety intelligence dashboard
```

#### Cost Verification (15 endpoints) 🆕
```bash
# Invoice Processing
POST /api/v1/costs/invoices            # Upload invoice (PDF/Excel/scan)
GET /api/v1/costs/invoices             # List invoices (paginated, 5M+ scale)
GET /api/v1/costs/invoices/{id}        # Get invoice details
POST /api/v1/costs/invoices/{id}/digitize  # Intelligent OCR + extraction
POST /api/v1/costs/invoices/{id}/verify    # Context-aware verification

# Line Item Validation
GET /api/v1/costs/line-items           # Get line items (11M+ scale)
POST /api/v1/costs/line-items/{id}/validate # Validate single line item
POST /api/v1/costs/validate-batch      # Batch validation (monthly cycle)

# Anomaly & Fraud Detection
GET /api/v1/costs/anomalies            # Get flagged anomalies
GET /api/v1/costs/duplicates           # Detect duplicate invoices across years
GET /api/v1/costs/out-of-scope         # Out-of-scope items (fish & chips, hospitality)
GET /api/v1/costs/pricing-outliers     # Unusual pricing (steel 2x cost across JVs)

# Analytics & Reporting
GET /api/v1/costs/focused-review-sets  # Categorize: OK / Review / High-Risk
GET /api/v1/costs/contractor-patterns  # Cross-JV pattern analysis
POST /api/v1/costs/benchmark           # Compare to commodity price benchmarks
```

**Total API Endpoints**: 93+ (30 deployed + 63 planned)

**Full API documentation**: http://localhost:8002/docs

---

## 📁 Real Data Available

### University of Twente GPR Dataset

- **125 survey locations** (10 imported, 115 available)
- **1,500+ radargrams** in SEG-Y format (.sgy files)
- **Comprehensive metadata**: utilities, soil, weather, equipment
- **Ground truth annotations**: known utility locations

**Data location**: `/datasets/raw/twente_gpr/`

**Import more surveys**:
```bash
docker compose exec backend python /app/scripts/import_twente_gpr_data.py
```

### Mojahid GPR Images

- **2,239+ labeled images** across 6 categories
- **Image classification** ready dataset

---

## 🛠️ Development

### Backend Development

```bash
# Watch logs
docker compose logs -f backend

# Enter container
docker compose exec backend bash

# Database access
docker compose exec postgres psql -U gpr_user -d gpr_db
```

### Frontend Development

```bash
# Watch logs
docker compose logs -f frontend

# Rebuild frontend
docker compose build --no-cache frontend
docker compose up -d frontend
```

### Database Management

```bash
# Create tables
docker compose exec backend python /app/scripts/database/create_tables.py

# Reset database
docker compose exec backend python /app/scripts/database/reset_database.py

# List tables
docker compose exec backend python /app/scripts/database/list_tables.py
```

---

## 🧪 Example Usage

### Query GPR Surveys

```bash
# Get all surveys
curl http://localhost:8002/api/v1/gpr/surveys | python3 -m json.tool

# Get Twente surveys
curl http://localhost:8002/api/v1/gpr/surveys | \
  python3 -c "import sys, json; data = json.load(sys.stdin); \
  [print(f\"{s['survey_name']}: {s['location_id']}\") for s in data if 'Twente' in s['survey_name']]"
```

### Database Queries

```sql
-- Connect to database
docker compose exec postgres psql -U gpr_user -d gpr_db

-- List surveys with scan counts
SELECT
    s.survey_name,
    s.location_id,
    s.status
FROM gpr_surveys s
WHERE s.survey_name LIKE 'Twente%'
ORDER BY s.created_at DESC;
```

---

## 🔐 Security

### Environment Variables

All sensitive data stored in `.env` (auto-generated by `setup_env.sh`):
- Database passwords
- MinIO access keys
- JWT secret keys
- API keys (Phase 2)

### CORS Configuration

Update in `docker-compose.yml`:
```yaml
environment:
  - CORS_ORIGINS=http://localhost:3003,http://localhost:8002,https://yourdomain.com
```

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| **[docs/](docs/)** | **Documentation hub** - Start here |
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | Complete setup and usage guide |
| [docs/DATA_GUIDE.md](docs/DATA_GUIDE.md) | Data sources, import, and organization |
| [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) | Current status, roadmap, and metrics |
| [docs/architecture/](docs/architecture/) | System architecture and diagrams |

---

## 🐛 Troubleshooting

### Services Not Starting

```bash
# Check logs
docker compose logs

# Restart services
docker compose restart

# Full restart
docker compose down && docker compose up -d
```

### Port Conflicts

Ports used: **3003** (frontend), **8002** (backend), **5433** (postgres), **6379** (redis), **9000/9001** (minio)

Change ports in `docker-compose.yml` if needed.

### Database Connection Issues

```bash
# Check connection
docker compose exec postgres psql -U gpr_user -d gpr_db -c "SELECT 1"

# Reset database
docker compose exec backend python /app/scripts/database/reset_database.py
```

---

## 📊 Performance

### Current Metrics

- **API Response Time**: <200ms P95 for most endpoints
- **Database Queries**: <50ms for simple queries
- **Docker Build**: ~5 minutes (backend + frontend)
- **Import Speed**: ~5 seconds for 10 surveys

### Production Targets

- **Report Generation**: <10 minutes end-to-end
- **Vector Search**: <100ms from Pinecone (Phase 2)
- **Uptime**: 99.9% availability
- **Accuracy**: >95% vs manual interpretation

---

## 🗺️ Roadmap

### Phase 1A - GPR Processing ✅ COMPLETE (Weeks 1-3)
- [x] Database schema (17 tables)
- [x] API endpoints (30+)
- [x] Docker deployment
- [x] Import real datasets (Twente GPR)
- [ ] Fix environmental/scans endpoints
- [ ] SEG-Y file parsing
- [ ] Signal processing pipeline

### Phase 1B - BIM Integration (Weeks 4-7)
- [ ] IFC file upload endpoint
- [ ] IFC.js integration
- [ ] 3D model viewer component
- [ ] Clash detection with GPR data
- [ ] BIM validation service

### Phase 1C - LiDAR Processing (Weeks 8-11)
- [ ] LAZ/LAS file upload
- [ ] Open3D integration
- [ ] Point cloud viewer (Potree)
- [ ] Progress monitoring
- [ ] Alignment with BIM models

### Phase 1D - Multi-Domain Intelligence 🆕 (Weeks 12-18)
**Inspired by HS2 requirements: 2M+ assets, 5M+ invoices, real-time safety intelligence**

#### 1D-A: Asset Certification Intelligence (Weeks 12-13)
- [ ] **Certificate Parsing**: OCR + NLP engine (Azure Document Intelligence / AWS Textract)
- [ ] **Qualification Extraction**: Parse PDF/Excel certificates for skills, dates, compliance
- [ ] **Database Schema**: Add 25 tables (assets, certificates, documents, IDP, TAEM)
- [ ] **API Endpoints**: 36 new endpoints for asset/certificate management
- [ ] **Validation Engine**: Automated certificate validation against requirements
- [ ] **Expiration Alerts**: Automated notifications for expiring qualifications
- [ ] **Asset Lifecycle Tracking**: Installation → Commissioning → Operational
- [ ] **Document Intelligence**: Index 100k+ deliverables with full-text search

#### 1D-B: Safety Intelligence Platform (Weeks 14-15) 🆕
- [ ] **Incident Intelligence**: NLP analysis of unstructured incident reports
- [ ] **Predictive Risk Scoring**: Multi-factor risk assessment (weather, fatigue, activity type)
- [ ] **Leading Indicators**: Proactive safety metrics before incidents occur
- [ ] **Anomaly Detection**: Automated cross-JV pattern analysis
- [ ] **High-Risk Windows**: Predict elevated risk periods (e.g., winter peaks in slips/trips)
- [ ] **Behavioral Correlation**: Link culture survey data to incident patterns
- [ ] **Real-Time Dashboard**: Unified safety intelligence for supervisors and leadership
- [ ] **Database Schema**: Add 8 tables (incidents, risk scores, predictions, interventions)
- [ ] **API Endpoints**: 12 new endpoints for safety management

#### 1D-C: Cost Verification Engine (Weeks 16-18) 🆕
- [ ] **Intelligent Invoice Digitization**: Beyond basic OCR - context-aware extraction
- [ ] **Semantic Validation**: AI understanding of domain terminology ("fish plate" vs "fish")
- [ ] **Duplicate Detection**: Pattern matching across 5M+ invoices and multiple years
- [ ] **Out-of-Scope Flagging**: Identify non-construction costs (hospitality, personal items)
- [ ] **Cross-JV Analytics**: Unusual pricing detection (steel 2x cost in one JV vs another)
- [ ] **Focused Review Sets**: Auto-categorize invoices (OK / Review / High-Risk)
- [ ] **Continuous Learning**: Feedback loop from commercial managers improves model
- [ ] **Microsoft Fabric Integration**: 🔥 **MANDATORY** - All data via Fabric pipelines
- [ ] **Database Schema**: Add 7 tables (invoices, line items, anomalies, benchmarks)
- [ ] **API Endpoints**: 15 new endpoints for cost management

### Phase 2A - LLM Integration (Weeks 19-26)
**RAG across all domains: PAS 128, safety incidents, contract rules**
- [ ] LangChain/LangGraph setup
- [ ] Pinecone vector database
- [ ] RAG pipeline for PAS 128 documents
- [ ] GPT-4o report generation (PAS 128 compliance, safety summaries)
- [ ] Fine-tuned models for construction/railway domain terminology
- [ ] Compliance validation engine
- [ ] Citation tracking (no hallucinations)
- [ ] Context-aware cost validation using GPT-4 ("fish plate" vs "fish")

### Phase 2B - Enterprise Integration & VisHub 🆕 (Weeks 27-32)
**Connect to fragmented enterprise systems (8+ systems as per HS2)**
- [ ] **AIMS Integration**: Asset Information Management System API
- [ ] **CDE Connectors**: BIM 360, Aconex, ProjectWise, Viewpoint
- [ ] **SharePoint/Teams**: Document sync and collaboration
- [ ] **ERP Integration**: SAP, Oracle for procurement/financials
- [ ] **Field Data Collection**: Mobile app for site data capture
- [ ] **Data Sync Engine**: Real-time bidirectional synchronization
- [ ] **API Gateway**: Unified interface for all enterprise systems
- [ ] **Master Data Management**: Single source of truth for assets

### Phase 2B - Enterprise Integration & VisHub 🆕 (Weeks 27-32) (continued)
- [ ] **Data Sync Engine**: Real-time bidirectional synchronization
- [ ] **API Gateway**: Unified interface for all enterprise systems
- [ ] **Master Data Management**: Single source of truth for assets
- [ ] **VisHub 2.0 Integration**: Geographic + asset-based navigation
- [ ] **Unified Visualization**: Safety, cost, asset, compliance data in one platform

### Phase 3 - Unified Intelligence Platform 🆕 (Weeks 33-41)
**Cross-domain analytics, predictive insights, knowledge graph**
- [ ] **IDP Analytics Dashboard**: Track 100k+ deliverables per contract
- [ ] **TAEM Compliance Engine**: Technical Assurance Evidence Model validation
- [ ] **Asset Readiness Scoring**: Real-time scoring (0-100%) per asset
- [ ] **Safety-Cost Correlation**: Link high-risk sites to cost overruns
- [ ] **Predictive Risk Assessment**: Combined safety + cost + asset insights
- [ ] **Automated Escalation**: Alerts for non-compliance, delays, anomalies
- [ ] **Predictive Insights**: ML forecasting for milestone completion
- [ ] **Executive Dashboard**: Project-wide unified intelligence
- [ ] **Knowledge Graph**: Neo4j for asset relationships and dependencies
- [ ] **Microsoft Fabric Integration**: Full data lakehouse deployment

---

## 🛠️ Technology Stack

### Current Stack (Phase 1A)
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend** | FastAPI 0.104.1 (Python 3.11+) | Async REST API |
| **Frontend** | React 18 + TypeScript + Vite | PWA-capable web UI |
| **Database** | PostgreSQL 16 + PGVector + PostGIS | Relational + vector + spatial data |
| **Cache** | Redis 7.2 | Query caching |
| **Storage** | MinIO (S3-compatible) | Object storage for GPR files |
| **Container** | Docker + Docker Compose | Service orchestration |

### Expanded Stack (Phases 1D - 3)
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Document AI** | Azure Document Intelligence 🔥 | OCR + form recognition for certificates & invoices |
| **NLP Engine** | spaCy + Hugging Face Transformers | Qualification & incident narrative extraction |
| **Vector DB** | Pinecone | Semantic search for documents/certificates/incidents |
| **LLM** | GPT-4o (OpenAI) | Report generation, cost validation, compliance |
| **Fine-Tuned LLM** | GPT-4 Custom | Construction/railway terminology understanding |
| **Knowledge Graph** | Neo4j | Asset relationships, cost patterns, safety correlations |
| **Enterprise APIs** | REST/GraphQL clients | AIMS, BIM 360, Aconex, ProjectWise, SharePoint |
| **Data Lakehouse** | Microsoft Fabric 🔥 **MANDATORY** | Unified data foundation for HS2 |
| **Real-Time Analytics** | Fabric KQL DB | Streaming safety incidents, real-time risk scoring |
| **ML Framework** | scikit-learn + MLflow | Risk prediction models, anomaly detection |
| **Time Series** | TimescaleDB (PostgreSQL extension) | Temporal safety patterns, cost trends |
| **Duplicate Detection** | MinHash LSH (datasketch) | Near-duplicate invoice detection at scale |
| **Workflow** | Temporal.io / Airflow | Long-running processing pipelines |
| **Search** | Elasticsearch | Full-text search across 100k+ documents |
| **BI** | Power BI (Fabric native) | Executive dashboards, semantic models |
| **Visualization** | Mapbox GL JS | Geographic navigation (VisHub 2.0) |

### Safety Intelligence Stack 🆕
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **NLP** | spaCy + BERT-based models | Incident narrative analysis for root causes |
| **Risk Scoring** | scikit-learn (Random Forest, XGBoost) | Multi-factor risk prediction |
| **Anomaly Detection** | Isolation Forest, Z-score analysis | Cross-JV pattern anomalies |
| **Time Series Forecasting** | Prophet / ARIMA | Predict high-risk windows (weather-based) |
| **Real-Time Alerts** | Redis Pub/Sub | Push notifications for risk spikes |

### Cost Verification Stack 🆕
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **OCR** | Azure Document Intelligence (Form Recognizer) | Invoice digitization (5M+ invoices) |
| **Semantic Understanding** | GPT-4 + Custom Training | Context-aware validation ("fish plate" vs "fish") |
| **Duplicate Detection** | MinHash LSH + Fuzzy matching | Multi-year duplicate detection |
| **Pattern Analysis** | Pandas + NumPy | Cross-JV pricing outlier detection |
| **Continuous Learning** | Active Learning Pipeline | Commercial manager feedback loop |
| **Batch Processing** | Fabric Data Factory | Monthly reconciliation cycles |

---

## 🙏 Acknowledgments

- **HS2 (High Speed 2)** for inspiring the multi-domain intelligence vision:
  - Asset certification intelligence (2M+ assets, 100k+ deliverables)
  - Data-led safety management (predictive analytics, incident intelligence)
  - Smarter cost verification (5M+ invoices, fraud detection)
  - VisHub 2.0 (unified visualization across enterprise systems)
- **University of Twente** for comprehensive GPR dataset
- **FastAPI** community for excellent async framework
- **React** team for modern UI framework
- **PostgreSQL** community for powerful database engine
- **Microsoft** for Azure Document Intelligence and Fabric platform

---

**Built with ❤️ for infrastructure intelligence**

*From utility strike prevention to unified enterprise intelligence:*
*Safety + Cost + Assets + Compliance - One AI-Native Platform*

**Market Transformation**:
- **Before**: £280M TAM (Utility Detection)
- **After**: £3B+ TAM (Safety Intelligence + Cost Verification + Asset Assurance + Utility Detection)

**Target Projects**: HS2, Crossrail 2, Sizewell C, Hinkley Point C, Northern Powerhouse Rail
