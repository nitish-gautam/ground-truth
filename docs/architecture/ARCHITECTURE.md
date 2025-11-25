# Infrastructure Intelligence Platform - Architecture Documentation

**Version**: 3.0 (Multi-Domain Intelligence: Safety + Cost + Assets + Utilities)
**Last Updated**: 2025-11-25
**Status**: Phase 1A Complete, Phase 1D Planning (Safety + Cost + Assets)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture Overview](#system-architecture-overview)
3. [Phase 1A: GPR Utility Detection (Current)](#phase-1a-gpr-utility-detection-current)
4. [Phase 1D: Multi-Domain Intelligence (Planned)](#phase-1d-multi-domain-intelligence-planned)
   - [Phase 1D-A: Asset Certification Intelligence](#phase-1d-a-asset-certification-intelligence)
   - [Phase 1D-B: Safety Intelligence Platform](#phase-1d-b-safety-intelligence-platform)
   - [Phase 1D-C: Cost Verification Engine](#phase-1d-c-cost-verification-engine)
   - [Phase 1D Integration: VisHub 2.0](#phase-1d-integration-vishub-20)
5. [Phase 2: BIM/LiDAR Integration (Planned)](#phase-2-bimlidar-integration-planned)
6. [Technology Stack](#technology-stack)
7. [Data Architecture](#data-architecture)
8. [API Architecture](#api-architecture)
9. [Security & Compliance](#security--compliance)
10. [Deployment Architecture](#deployment-architecture)

---

## Executive Summary

The Infrastructure Intelligence Platform has transformed from a specialized **GPR-based underground utility detection system** (Phase 1A) into a comprehensive **Multi-Domain Enterprise Intelligence Platform** that addresses **five critical infrastructure domains**:

### Current Capabilities (Phase 1A - COMPLETE)
- ✅ **GPR Data Processing**: SEG-Y, GSSI DZT file parsing and signal processing
- ✅ **Material Classification**: 10+ utility material types (steel, PVC, concrete, etc.)
- ✅ **PAS 128 Compliance**: Automated QL-A/B/C/D quality level determination
- ✅ **Environmental Analysis**: Soil conditions, weather correlation
- ✅ **Backend API**: 30+ FastAPI endpoints with async support
- ✅ **Database**: PostgreSQL with PostGIS + PGVector (17 tables)
- ✅ **Real Datasets**: 10 GPR surveys imported from University of Twente
- ✅ **Docker Deployment**: 5/5 services healthy, fully containerized

### Planned Capabilities (Phases 1B-3)

#### Phase 1B: BIM Integration (Weeks 4-7)
- ⏳ **IFC Processing**: IFC file parsing, 3D model visualization
- ⏳ **Clash Detection**: BIM vs detected utilities conflict analysis
- ⏳ **Model Validation**: Standards compliance (ISO 19650, PAS 1192)

#### Phase 1C: LiDAR Processing (Weeks 8-11)
- ⏳ **Point Cloud Analysis**: LAZ/LAS processing, progress monitoring
- ⏳ **Construction QA**: Planned vs. actual comparison
- ⏳ **Volume Calculations**: Earthwork quantification

#### Phase 1D: Multi-Domain Intelligence 🆕 (Weeks 12-18)
**Inspired by HS2 Railway Project requirements**

**1D-A: Asset Certification Intelligence** (Weeks 12-13)
- ⏳ **Certificate Parsing**: OCR + NLP for 2M+ assets, 100k+ deliverables
- ⏳ **TAEM Compliance**: Technical Assurance Evidence Model validation
- ⏳ **IDP Analytics**: Information Delivery Plan tracking
- ⏳ **Readiness Scoring**: Real-time asset assurance scores

**1D-B: Safety Intelligence** 🆕 (Weeks 14-15)
- ⏳ **Incident Intelligence**: NLP analysis of unstructured incident reports
- ⏳ **Predictive Risk Scoring**: Multi-factor risk assessment (weather, fatigue, activity)
- ⏳ **Leading Indicators**: Proactive safety metrics before incidents occur
- ⏳ **Anomaly Detection**: Cross-JV pattern analysis and automated alerting
- ⏳ **High-Risk Windows**: Predict elevated risk periods (e.g., winter slips/trips peaks)
- ⏳ **Behavioral Correlation**: Link culture surveys to incident patterns

**1D-C: Cost Verification Engine** 🆕 (Weeks 16-18)
- ⏳ **Intelligent Invoice Processing**: 5M+ invoices, 11M+ line items validation
- ⏳ **Semantic Validation**: Context-aware AI ("fish plate" vs "fish")
- ⏳ **Fraud Detection**: Duplicate and out-of-scope cost identification
- ⏳ **Cross-JV Analytics**: Unusual pricing pattern detection
- ⏳ **Focused Review Sets**: Auto-categorize (OK / Review / High-Risk)
- ⏳ **Microsoft Fabric Integration**: 🔥 **MANDATORY** for HS2

#### Phase 2A: LLM Integration (Weeks 19-26)
- ⏳ **RAG Pipeline**: PAS 128, safety incidents, contract rules
- ⏳ **GPT-4 Integration**: Report generation, cost validation
- ⏳ **Fine-Tuned Models**: Construction/railway terminology understanding

#### Phase 2B: Enterprise Integration & VisHub (Weeks 27-32)
- ⏳ **AIMS Integration**: Asset Information Management System API
- ⏳ **CDE Connectors**: BIM 360, Aconex, ProjectWise, Viewpoint
- ⏳ **VisHub 2.0**: Geographic + asset-based unified visualization
- ⏳ **Microsoft Fabric**: Full data lakehouse deployment

#### Phase 3: Unified Intelligence Platform (Weeks 33-41)
- ⏳ **Cross-Domain Analytics**: Safety ↔ Cost ↔ Assets correlations
- ⏳ **Knowledge Graph**: Neo4j for holistic project intelligence
- ⏳ **Predictive Insights**: ML forecasting for risks, milestones, overruns

### Business Value Transformation
**Market Evolution**:
- **Original TAM**: £280M (Utility detection)
- **Transformed TAM**: £3B+ (Multi-domain intelligence)
  - Safety Intelligence: £800M
  - Cost Verification: £700M (HS2 identified £100M+ savings opportunity)
  - Asset Assurance: £1.5B
  - Utility Detection: £280M

**Revenue Model**:
- **Safety Management**: £50k-200k per major project
- **Cost Verification**: £100k-500k per project (ROI: 10-50x from fraud prevention)
- **Asset Assurance**: £200k-1M per £20B+ programme
- **Utility Surveys**: £2k-5k per survey
- **SaaS Platform**: £500k-2M annual enterprise license

---

## System Architecture Overview

### High-Level Architecture (Phase 1A + Phase 2)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                                │
│  React PWA (Frontend) - MapLibre 2D + IFC.js 3D + CesiumJS         │
└────────────────┬────────────────────────────────────────────────────┘
                 │ HTTPS / WebSocket
┌────────────────┴────────────────────────────────────────────────────┐
│                       API GATEWAY                                    │
│  FastAPI Server | JWT Auth | Rate Limiting | CORS                   │
└────┬────────────────────────────────────────────────────────┬────────┘
     │                                                         │
┌────┴──────────────────────────┐    ┌──────────────────────┴────────┐
│    PROCESSING SERVICES        │    │    DATA STORAGE LAYER         │
│                               │    │                               │
│ ┌──────────────────────────┐ │    │ ┌──────────────────────────┐ │
│ │ GPR Processing (Phase 1A)│ │    │ │ PostgreSQL + PostGIS     │ │
│ │ - Signal Processing      │ │    │ │ + PGVector               │ │
│ │ - Material Classification│ │    │ │                          │ │
│ │ - PAS 128 Compliance     │ │    │ │ - GPR Surveys            │ │
│ └──────────────────────────┘ │    │ │ - Detected Utilities     │ │
│                               │    │ │ - BIM Models (Phase 2)   │ │
│ ┌──────────────────────────┐ │    │ │ - LiDAR Scans (Phase 2)  │ │
│ │ BIM Processing (Phase 2) │ │    │ └──────────────────────────┘ │
│ │ - IFC Parser             │ │    │                               │
│ │ - Model Validator        │ │    │ ┌──────────────────────────┐ │
│ │ - Clash Detection        │ │    │ │ MinIO S3 Storage         │ │
│ └──────────────────────────┘ │    │ │                          │ │
│                               │    │ │ - GPR Files (500MB)      │ │
│ ┌──────────────────────────┐ │    │ │ - BIM Models (IFC)       │ │
│ │ LiDAR Processing (P2)    │ │    │ │ - LiDAR Scans (100GB)    │ │
│ │ - Point Cloud Processor  │ │    │ │ - Documents (PDF/CAD)    │ │
│ │ - Progress Monitor       │ │    │ └──────────────────────────┘ │
│ └──────────────────────────┘ │    │                               │
│                               │    │ ┌──────────────────────────┐ │
│ ┌──────────────────────────┐ │    │ │ Redis Cache              │ │
│ │ LLM Services (Phase 2)   │ │    │ │ - Query Cache            │ │
│ │ - LangChain Agents       │ │    │ │ - Celery Broker          │ │
│ │ - Report Generator       │ │    │ └──────────────────────────┘ │
│ └──────────────────────────┘ │    └───────────────────────────────┘
└───────────────────────────────┘
         │
┌────────┴────────────────────┐
│  BACKGROUND TASK QUEUE      │
│  Celery Workers + Flower UI │
└─────────────────────────────┘
```

**See Also**: [DIAGRAMS.md](DIAGRAMS.md) for detailed Mermaid diagrams

---

## Phase 1A: GPR Utility Detection (Current)

### Dataset Integration

**Priority Datasets (Production-Ready)**:
1. **Twente GPR Dataset**: 125 scans, 500MB, 25+ metadata fields
2. **Mojahid Images**: 2,239+ labeled images, 6 categories
3. **PAS 128 Documents**: Compliance specifications
4. **USAG Reports**: Historical incident data

### Core Processing Pipeline

```python
# GPR Processing Workflow (Simplified)
class GPRProcessingWorkflow:
    def process_gpr_file(self, file_path: str) -> ProcessedGPR:
        # Stage 1: File Validation
        validated_file = self.validate_segy_format(file_path)

        # Stage 2: Signal Processing
        processed_signal = self.apply_filters(
            data=validated_file,
            filters=['bandpass', 'gain_correction', 'background_removal']
        )

        # Stage 3: Feature Detection
        features = self.detect_hyperbolas(
            signal=processed_signal,
            method='hough_transform'
        )

        # Stage 4: Depth Calculation
        depths = self.calculate_depths(
            features=features,
            velocity_model=self.soil_velocity_model
        )

        # Stage 5: Utility Classification
        utilities = self.classify_utilities(
            features=features,
            model=self.trained_classifier
        )

        # Stage 6: Confidence Scoring
        confidence = self.calculate_confidence(
            signal_quality=processed_signal.snr,
            feature_clarity=features.score,
            model_certainty=utilities.probability
        )

        return ProcessedGPR(
            utilities=utilities,
            depths=depths,
            confidence=confidence
        )
```

### Material Classification System

**Supported Material Types (10+)**:
- Steel (ferrous pipes)
- PVC/Plastic
- Concrete
- Cast iron
- Copper
- Clay
- Asbestos cement
- Ductile iron
- High-density polyethylene (HDPE)
- Mixed/Unknown

**Classification Approach**:
- Signal amplitude analysis
- Hyperbola shape characteristics
- Frequency domain features
- Statistical pattern recognition
- ML model ensemble (Random Forest + SVM + Neural Network)

### PAS 128 Compliance Automation

**Quality Levels**:
- **QL-A**: Highest accuracy (surveyed utility location)
- **QL-B**: Investigation of existing records
- **QL-C**: Electromagnetic location (EMI)
- **QL-D**: Desktop study

**Automated Assessment**:
```python
class QualityLevelAssessment:
    def determine_quality_level(self, survey_data: dict) -> str:
        score = 0

        # Survey method scoring
        if survey_data['method'] == 'GPR':
            score += 40
        if survey_data['has_trench_verification']:
            score += 30
        if survey_data['electromagnetic_confirmation']:
            score += 20
        if survey_data['existing_records_validation']:
            score += 10

        # Quality level determination
        if score >= 70:
            return 'QL-A'
        elif score >= 50:
            return 'QL-B'
        elif score >= 30:
            return 'QL-C'
        else:
            return 'QL-D'
```

### Environmental Analysis

**Factors Analyzed**:
- Soil composition (clay, sand, loam, peat, chalk)
- Moisture content
- Temperature effects
- Ground conductivity
- Terrain characteristics

**Impact on GPR Performance**:
- Signal attenuation in conductive soils
- Depth penetration variations
- Resolution changes with frequency
- Weather correlation (rain impact)

---

## Phase 1D: Multi-Domain Intelligence (Planned)

### Overview

Phase 1D expands the platform from utility detection into **three major HS2-inspired use cases**: Asset Certification Intelligence, Safety Intelligence, and Cost Verification. This transformation increases the Total Addressable Market from £280M to £3B+ across major UK infrastructure projects.

**Timeline**: Weeks 12-18 (7 weeks)
**Inspired by**: HS2 Railway Project operational requirements

---

### Phase 1D-A: Asset Certification Intelligence

**Timeline**: Weeks 12-13 (2 weeks)

#### Business Context
- **Challenge**: 2M+ physical assets, 100,000+ deliverables per contract (HS2 scale)
- **Current Process**: Manual certificate parsing, scattered TAEM compliance tracking
- **Value Proposition**: Automated certificate intelligence, real-time readiness scoring

#### Architecture

**Certificate Processing Pipeline**:
```python
class CertificateProcessingWorkflow:
    def process_certificate(self, pdf_path: str) -> ParsedCertificate:
        # Stage 1: OCR Extraction
        raw_text = self.azure_ocr.extract_text(
            file_path=pdf_path,
            mode='read'  # High-accuracy mode
        )

        # Stage 2: NLP Information Extraction
        structured_data = self.bert_ner.extract_entities(
            text=raw_text,
            entity_types=[
                'certificate_number', 'issue_date', 'expiry_date',
                'qualification_type', 'holder_name', 'issuing_body',
                'standard_compliance', 'restrictions'
            ]
        )

        # Stage 3: TAEM Validation
        taem_compliance = self.validate_taem_requirements(
            certificate=structured_data,
            asset_type=self.get_asset_type(),
            work_package=self.get_work_package()
        )

        # Stage 4: IDP Tracking
        idp_status = self.track_information_delivery(
            certificate=structured_data,
            project_schedule=self.get_project_schedule()
        )

        return ParsedCertificate(
            structured_data=structured_data,
            taem_status=taem_compliance,
            idp_readiness=idp_status
        )
```

**Database Schema (8 tables)**:
```sql
-- Asset tracking
CREATE TABLE assets (
    id UUID PRIMARY KEY,
    project_id UUID REFERENCES projects(id),
    asset_tag VARCHAR(100) UNIQUE NOT NULL,
    asset_type VARCHAR(100),  -- Bridge, Track, Tunnel, etc.
    work_package VARCHAR(100),
    location GEOMETRY(Point, 4326),
    status VARCHAR(50),  -- In Progress, Certified, Awaiting Documentation
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Certificate management
CREATE TABLE asset_certificates (
    id UUID PRIMARY KEY,
    asset_id UUID REFERENCES assets(id),
    certificate_type VARCHAR(100),  -- Structural Test, Material Cert, etc.
    certificate_number VARCHAR(100),
    issue_date DATE,
    expiry_date DATE,
    issuing_body VARCHAR(255),
    document_url TEXT,  -- S3/MinIO URL
    ocr_confidence FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- TAEM compliance tracking
CREATE TABLE taem_requirements (
    id UUID PRIMARY KEY,
    asset_type VARCHAR(100),
    work_package VARCHAR(100),
    required_certificates JSONB,  -- Array of certificate types
    compliance_criteria JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE taem_compliance_status (
    id UUID PRIMARY KEY,
    asset_id UUID REFERENCES assets(id),
    requirement_id UUID REFERENCES taem_requirements(id),
    compliance_percentage FLOAT,  -- 0-100
    missing_certificates JSONB,
    next_deliverable_due DATE,
    risk_level VARCHAR(20),  -- Low, Medium, High, Critical
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- IDP analytics
CREATE TABLE information_delivery_plans (
    id UUID PRIMARY KEY,
    project_id UUID REFERENCES projects(id),
    work_package VARCHAR(100),
    total_deliverables INTEGER,
    completed_deliverables INTEGER,
    milestone_date DATE,
    status VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE idp_deliverables (
    id UUID PRIMARY KEY,
    idp_id UUID REFERENCES information_delivery_plans(id),
    deliverable_name VARCHAR(255),
    deliverable_type VARCHAR(100),
    due_date DATE,
    submission_date DATE,
    status VARCHAR(50),  -- Pending, Submitted, Approved, Rejected
    asset_id UUID REFERENCES assets(id),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Asset readiness scoring
CREATE TABLE asset_readiness_scores (
    id UUID PRIMARY KEY,
    asset_id UUID REFERENCES assets(id),
    readiness_score FLOAT,  -- 0-100
    certification_status FLOAT,  -- 0-100
    documentation_status FLOAT,  -- 0-100
    taem_compliance FLOAT,  -- 0-100
    calculated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Asset relationships (knowledge graph)
CREATE TABLE asset_relationships (
    id UUID PRIMARY KEY,
    source_asset_id UUID REFERENCES assets(id),
    target_asset_id UUID REFERENCES assets(id),
    relationship_type VARCHAR(100),  -- depends_on, part_of, connects_to
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

**API Endpoints (10 endpoints)**:
```
# Asset Management
POST   /api/v1/assets                          # Create asset record
GET    /api/v1/assets                          # List assets (filterable)
GET    /api/v1/assets/{id}                     # Get asset details
PUT    /api/v1/assets/{id}                     # Update asset
GET    /api/v1/assets/{id}/readiness           # Get readiness score

# Certificate Processing
POST   /api/v1/certificates/upload             # Upload certificate PDF
POST   /api/v1/certificates/parse              # OCR + NLP extraction
GET    /api/v1/certificates/{id}               # Get certificate details

# TAEM Compliance
GET    /api/v1/compliance/taem/{asset_id}      # Get TAEM status
POST   /api/v1/compliance/taem/validate        # Validate compliance

# IDP Analytics
GET    /api/v1/idp/deliverables                # Get deliverables status
POST   /api/v1/idp/predict-delays              # Predict milestone risks
```

---

### Phase 1D-B: Safety Intelligence Platform

**Timeline**: Weeks 14-15 (2 weeks)

#### Business Context
- **Challenge**: Fragmented incident data across 10+ Joint Ventures, reactive safety management
- **Current Process**: Manual incident report analysis, delayed pattern recognition
- **Value Proposition**: Predictive risk scoring, automated anomaly detection, leading indicators

#### Architecture

**Incident Intelligence Pipeline**:
```python
class SafetyIntelligenceWorkflow:
    def process_incident_report(self, incident_data: dict) -> IncidentIntelligence:
        # Stage 1: Narrative NLP Analysis
        narrative_analysis = self.spacy_nlp(incident_data['description'])

        # Extract root causes using BERT NER
        root_causes = self.bert_ner.extract_entities(
            text=incident_data['description'],
            entity_types=[
                'primary_cause', 'contributing_factors',
                'equipment_involved', 'location_specifics',
                'procedural_violations', 'environmental_conditions'
            ]
        )

        # Stage 2: Multi-Factor Risk Scoring
        risk_score = self.calculate_risk_score(
            incident_type=incident_data['type'],
            severity=incident_data['severity'],
            weather_conditions=self.get_weather(incident_data['timestamp']),
            fatigue_indicators=self.assess_fatigue(incident_data),
            activity_type=incident_data['activity'],
            contractor_history=self.get_contractor_safety_record()
        )

        # Stage 3: Behavioral Correlation
        behavioral_factors = self.correlate_culture_survey(
            contractor=incident_data['contractor'],
            site=incident_data['site'],
            incident_date=incident_data['timestamp']
        )

        # Stage 4: Anomaly Detection (Cross-JV)
        is_anomaly, anomaly_score = self.detect_anomaly(
            incident=incident_data,
            jv_historical_patterns=self.get_jv_patterns(),
            method='isolation_forest'
        )

        # Stage 5: Predictive Alerting
        if risk_score > self.HIGH_RISK_THRESHOLD:
            self.trigger_alert(
                type='high_risk_detected',
                recipients=self.get_safety_managers(),
                incident=incident_data
            )

        return IncidentIntelligence(
            root_causes=root_causes,
            risk_score=risk_score,
            behavioral_factors=behavioral_factors,
            is_anomaly=is_anomaly
        )

    def predict_high_risk_windows(self, site_id: str) -> List[RiskWindow]:
        # Time series forecasting using Prophet
        historical_incidents = self.get_site_incidents(site_id)
        weather_forecast = self.get_14_day_weather_forecast(site_id)

        # Train Prophet model on seasonal patterns
        model = Prophet()
        model.add_regressor('temperature')
        model.add_regressor('precipitation')
        model.add_regressor('daylight_hours')

        model.fit(self.prepare_training_data(historical_incidents))

        # Predict next 14 days
        future = model.make_future_dataframe(periods=14)
        future['temperature'] = weather_forecast['temp']
        future['precipitation'] = weather_forecast['rain']
        future['daylight_hours'] = self.calculate_daylight(future['ds'])

        forecast = model.predict(future)

        # Identify high-risk windows (e.g., P90 > 2.0 incidents/day)
        high_risk_windows = forecast[forecast['yhat_upper'] > 2.0]

        return [
            RiskWindow(
                date=row['ds'],
                predicted_incident_rate=row['yhat'],
                confidence_interval=(row['yhat_lower'], row['yhat_upper']),
                primary_risk_factors=self.identify_risk_factors(row)
            )
            for _, row in high_risk_windows.iterrows()
        ]
```

**Database Schema (8 tables)**:
```sql
-- Core incident tracking
CREATE TABLE safety_incidents (
    id UUID PRIMARY KEY,
    project_id UUID REFERENCES projects(id),
    incident_number VARCHAR(50) UNIQUE,
    incident_type VARCHAR(100),  -- Near Miss, Lost Time Injury, etc.
    severity VARCHAR(50),  -- Minor, Moderate, Serious, Fatal
    incident_date TIMESTAMPTZ NOT NULL,
    site_location VARCHAR(255),
    contractor VARCHAR(255),
    joint_venture VARCHAR(100),
    activity_type VARCHAR(100),
    weather_conditions JSONB,
    structured_data JSONB,  -- Extracted fields
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Unstructured narratives for NLP
CREATE TABLE safety_incident_narratives (
    id UUID PRIMARY KEY,
    incident_id UUID REFERENCES safety_incidents(id),
    narrative_text TEXT NOT NULL,
    narrative_type VARCHAR(50),  -- Initial Report, Investigation, Witness Statement
    extracted_root_causes JSONB,
    nlp_confidence FLOAT,
    processed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Real-time risk scoring
CREATE TABLE safety_risk_scores (
    id UUID PRIMARY KEY,
    site_id VARCHAR(100),
    contractor_id VARCHAR(100),
    risk_score FLOAT,  -- 0-100
    incident_rate FLOAT,
    severity_index FLOAT,
    leading_indicator_score FLOAT,
    calculated_at TIMESTAMPTZ DEFAULT NOW(),
    valid_until TIMESTAMPTZ
);

-- Environmental correlation
CREATE TABLE safety_environmental_factors (
    id UUID PRIMARY KEY,
    incident_id UUID REFERENCES safety_incidents(id),
    temperature FLOAT,
    precipitation FLOAT,
    wind_speed FLOAT,
    visibility FLOAT,
    daylight_hours FLOAT,
    transport_disruptions JSONB,
    congestion_index FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Behavioral observations
CREATE TABLE safety_behavioral_observations (
    id UUID PRIMARY KEY,
    contractor VARCHAR(255),
    site_location VARCHAR(255),
    observation_date DATE,
    culture_survey_score FLOAT,  -- 0-10
    leadership_engagement FLOAT,
    procedural_compliance FLOAT,
    ppe_usage_rate FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Predictive models
CREATE TABLE safety_predictions (
    id UUID PRIMARY KEY,
    site_id VARCHAR(100),
    prediction_date DATE,
    predicted_incident_rate FLOAT,
    confidence_interval_lower FLOAT,
    confidence_interval_upper FLOAT,
    primary_risk_factors JSONB,
    model_version VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Interventions tracking
CREATE TABLE safety_interventions (
    id UUID PRIMARY KEY,
    site_id VARCHAR(100),
    intervention_type VARCHAR(100),  -- Toolbox Talk, Equipment Upgrade, etc.
    intervention_date DATE,
    triggered_by VARCHAR(100),  -- Prediction, Incident, Inspection
    cost DECIMAL(10, 2),
    effectiveness_score FLOAT,  -- Measured post-intervention
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Leading indicators
CREATE TABLE safety_leading_indicators (
    id UUID PRIMARY KEY,
    site_id VARCHAR(100),
    indicator_type VARCHAR(100),  -- Near Miss Reports, Safety Walks, Training Hours
    indicator_value FLOAT,
    measurement_date DATE,
    target_value FLOAT,
    status VARCHAR(50),  -- On Track, At Risk, Critical
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

**API Endpoints (12 endpoints)**:
```
# Incident Management
POST   /api/v1/safety/incidents                    # Create incident with narrative
GET    /api/v1/safety/incidents                    # List incidents (filterable by JV, site, type)
GET    /api/v1/safety/incidents/{id}               # Get incident details
POST   /api/v1/safety/incidents/{id}/analyze       # NLP analysis of narrative

# Risk Scoring & Prediction
GET    /api/v1/safety/risk-scores                  # Real-time risk scores across sites
GET    /api/v1/safety/risk-scores/{site_id}        # Site-specific risk score
POST   /api/v1/safety/predict-risk                 # Predict high-risk windows (weather, fatigue)

# Analytics & Insights
GET    /api/v1/safety/leading-indicators           # Proactive safety metrics
GET    /api/v1/safety/anomalies                    # Automated anomaly detection (cross-JV)
GET    /api/v1/safety/top-risks                    # Top 5 risks based on recent patterns
POST   /api/v1/safety/correlate                    # Correlate weather/incidents/behaviors

# Dashboard
GET    /api/v1/safety/dashboard                    # Unified safety intelligence dashboard
```

**Technology Stack**:
```
NLP & ML:
- spaCy 3.7.0 - Fast NLP for narrative parsing
- transformers 4.35.0 - BERT-based NER models
- scikit-learn 1.3.2 - Random Forest, Isolation Forest
- xgboost 2.0.0 - Gradient boosting for risk scoring
- prophet 1.1.5 - Time series forecasting

Data Processing:
- pandas 2.1.0 - Data manipulation
- numpy 1.26.0 - Numerical operations
- TimescaleDB extension - PostgreSQL time-series optimization

Real-Time Alerts:
- Redis Pub/Sub - Push notifications
- Celery Beat - Scheduled predictions (daily risk forecasts)
```

---

### Phase 1D-C: Cost Verification Engine

**Timeline**: Weeks 16-18 (3 weeks)

#### Business Context
- **Challenge**: 5M+ invoices, 11M+ line items, basic OCR failures, £100M+ savings opportunity
- **Current Process**: 6 commercial managers, 3-week monthly cycles, duplicate invoices undetected
- **Value Proposition**: Intelligent invoice processing, semantic validation, automated fraud detection

#### Architecture

**Invoice Processing Pipeline**:
```python
class CostVerificationWorkflow:
    def process_invoice(self, invoice_file: str) -> VerifiedInvoice:
        # Stage 1: Intelligent OCR (Azure Document Intelligence)
        digitized_invoice = self.azure_form_recognizer.analyze_document(
            file_path=invoice_file,
            model='prebuilt-invoice',  # Beyond basic OCR
            locale='en-GB'
        )

        # Stage 2: Semantic Validation (GPT-4 with custom training)
        semantic_analysis = self.gpt4.validate_context(
            invoice_data=digitized_invoice,
            context_rules=[
                "Distinguish 'fish plate' (railway component) from 'fish' (food)",
                "Validate steel grades match project specifications",
                "Check units of measurement for consistency",
                "Verify VAT calculations",
                "Detect out-of-scope items (hospitality, personal expenses)"
            ]
        )

        # Stage 3: Duplicate Detection (MinHash LSH)
        duplicate_candidates = self.minhash_lsh.query(
            invoice_text=digitized_invoice.full_text,
            threshold=0.85  # 85% similarity
        )

        if duplicate_candidates:
            # Cross-year analysis
            for candidate in duplicate_candidates:
                if self.is_near_duplicate(digitized_invoice, candidate):
                    semantic_analysis['anomalies'].append({
                        'type': 'potential_duplicate',
                        'similar_invoice_id': candidate.id,
                        'similarity_score': candidate.similarity,
                        'date_difference_days': (
                            digitized_invoice.date - candidate.date
                        ).days
                    })

        # Stage 4: Cross-JV Pricing Analysis
        pricing_outliers = self.detect_pricing_anomalies(
            invoice=digitized_invoice,
            line_items=digitized_invoice.line_items,
            jv_benchmarks=self.get_jv_pricing_benchmarks(),
            commodity_prices=self.get_commodity_market_prices()
        )

        # Stage 5: Contract Rules Validation
        contract_compliance = self.validate_against_contract(
            invoice=digitized_invoice,
            contract_id=digitized_invoice.contract_reference,
            rules=self.get_contract_rules()
        )

        # Stage 6: Focused Review Categorization
        review_category = self.categorize_for_review(
            semantic_score=semantic_analysis['confidence'],
            anomaly_count=len(semantic_analysis['anomalies']),
            pricing_outliers=pricing_outliers,
            contract_compliance=contract_compliance
        )

        return VerifiedInvoice(
            digitized_data=digitized_invoice,
            semantic_validation=semantic_analysis,
            duplicate_status=len(duplicate_candidates) > 0,
            pricing_analysis=pricing_outliers,
            review_category=review_category  # OK / Review / High-Risk
        )

    def categorize_for_review(self, semantic_score, anomaly_count,
                               pricing_outliers, contract_compliance):
        # Decision tree for review prioritization
        if (semantic_score > 0.95 and
            anomaly_count == 0 and
            len(pricing_outliers) == 0 and
            contract_compliance['compliant']):
            return 'LIKELY_OK'

        elif (anomaly_count > 2 or
              len([p for p in pricing_outliers if p['severity'] == 'high']) > 0 or
              not contract_compliance['compliant']):
            return 'HIGH_RISK'

        else:
            return 'REQUIRES_MANUAL_REVIEW'
```

**Database Schema (7 tables)**:
```sql
-- Invoice metadata
CREATE TABLE invoices (
    id UUID PRIMARY KEY,
    project_id UUID REFERENCES projects(id),
    invoice_number VARCHAR(100),
    contractor VARCHAR(255),
    joint_venture VARCHAR(100),
    invoice_date DATE,
    due_date DATE,
    total_amount DECIMAL(15, 2),
    currency VARCHAR(3) DEFAULT 'GBP',
    contract_reference VARCHAR(100),
    document_url TEXT,  -- S3/MinIO URL
    ocr_confidence FLOAT,
    semantic_confidence FLOAT,
    review_category VARCHAR(50),  -- LIKELY_OK, REQUIRES_MANUAL_REVIEW, HIGH_RISK
    processed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Line item details (11M+ records at HS2 scale)
CREATE TABLE invoice_line_items (
    id UUID PRIMARY KEY,
    invoice_id UUID REFERENCES invoices(id),
    line_number INTEGER,
    description TEXT,
    quantity DECIMAL(15, 4),
    unit_of_measurement VARCHAR(50),
    unit_price DECIMAL(15, 2),
    line_total DECIMAL(15, 2),
    commodity_category VARCHAR(100),  -- Steel, Concrete, Labour, Equipment, etc.
    gl_code VARCHAR(50),  -- General Ledger code
    is_out_of_scope BOOLEAN DEFAULT FALSE,
    validation_status VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Document storage references
CREATE TABLE invoice_documents (
    id UUID PRIMARY KEY,
    invoice_id UUID REFERENCES invoices(id),
    document_type VARCHAR(50),  -- PDF, Excel, Scanned Image
    file_path TEXT,
    file_size_bytes BIGINT,
    page_count INTEGER,
    upload_date TIMESTAMPTZ DEFAULT NOW()
);

-- Verification results
CREATE TABLE cost_verification_results (
    id UUID PRIMARY KEY,
    invoice_id UUID REFERENCES invoices(id),
    line_item_id UUID REFERENCES invoice_line_items(id),
    verification_type VARCHAR(100),  -- Semantic, Duplicate, Pricing, Contract
    status VARCHAR(50),  -- Pass, Fail, Warning
    confidence_score FLOAT,
    findings JSONB,
    verified_at TIMESTAMPTZ DEFAULT NOW()
);

-- Anomaly tracking
CREATE TABLE cost_anomalies (
    id UUID PRIMARY KEY,
    invoice_id UUID REFERENCES invoices(id),
    line_item_id UUID REFERENCES invoice_line_items(id),
    anomaly_type VARCHAR(100),  -- Duplicate, Out-of-Scope, Pricing Outlier, etc.
    severity VARCHAR(50),  -- Low, Medium, High, Critical
    description TEXT,
    similar_invoice_id UUID,  -- For duplicates
    pricing_benchmark DECIMAL(15, 2),  -- For outliers
    resolution_status VARCHAR(50),  -- Open, Investigating, Resolved, False Positive
    assigned_to VARCHAR(255),  -- Commercial manager
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Contract rules (JV-specific)
CREATE TABLE contract_rules (
    id UUID PRIMARY KEY,
    contract_id VARCHAR(100),
    joint_venture VARCHAR(100),
    rule_type VARCHAR(100),  -- Price Cap, Approved Vendor, Prohibited Item
    rule_definition JSONB,
    effective_date DATE,
    expiry_date DATE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Commodity price benchmarks
CREATE TABLE cost_benchmarks (
    id UUID PRIMARY KEY,
    commodity_category VARCHAR(100),
    unit_of_measurement VARCHAR(50),
    benchmark_price DECIMAL(15, 2),
    price_source VARCHAR(255),  -- Market Index, Historical Average, Contract Rate
    effective_date DATE,
    geographic_region VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

**API Endpoints (15 endpoints)**:
```
# Invoice Processing
POST   /api/v1/costs/invoices                      # Upload invoice (PDF/Excel/scan)
GET    /api/v1/costs/invoices                      # List invoices (paginated, 5M+ scale)
GET    /api/v1/costs/invoices/{id}                 # Get invoice details
POST   /api/v1/costs/invoices/{id}/digitize        # Intelligent OCR + extraction
POST   /api/v1/costs/invoices/{id}/verify          # Context-aware verification

# Line Item Validation
GET    /api/v1/costs/line-items                    # Get line items (11M+ scale)
POST   /api/v1/costs/line-items/{id}/validate      # Validate single line item
POST   /api/v1/costs/validate-batch                # Batch validation (monthly cycle)

# Anomaly & Fraud Detection
GET    /api/v1/costs/anomalies                     # Get flagged anomalies
GET    /api/v1/costs/duplicates                    # Detect duplicate invoices across years
GET    /api/v1/costs/out-of-scope                  # Out-of-scope items (fish & chips, hospitality)
GET    /api/v1/costs/pricing-outliers              # Unusual pricing (steel 2x cost across JVs)

# Analytics & Reporting
GET    /api/v1/costs/focused-review-sets           # Categorize: OK / Review / High-Risk
GET    /api/v1/costs/contractor-patterns           # Cross-JV pattern analysis
POST   /api/v1/costs/benchmark                     # Compare to commodity price benchmarks
```

**Technology Stack**:
```
OCR & Document Intelligence:
- Azure Document Intelligence (Form Recognizer) - 🔥 MANDATORY for HS2
- PyPDF2 3.0.0 - PDF parsing fallback
- python-docx 1.1.0 - Word document processing
- openpyxl 3.1.0 - Excel parsing

Semantic Validation:
- OpenAI GPT-4 API - Context-aware validation
- Custom fine-tuned model - Railway/construction terminology

Duplicate Detection:
- datasketch 1.6.0 - MinHash LSH implementation
- python-Levenshtein 0.23.0 - Fuzzy string matching

Data Processing:
- pandas 2.1.0 - Dataframe operations (11M+ rows)
- numpy 1.26.0 - Numerical operations

Microsoft Fabric Integration: 🔥 MANDATORY
- OneLake - Data Lake Gen2 storage
- KQL Database - Real-time anomaly queries
- Data Factory - Batch invoice processing pipelines
- ML Workspace - Model training and deployment
- Power BI Semantic Models - Commercial dashboards
```

**Microsoft Fabric Integration Architecture**:
```
┌─────────────────────────────────────────────────────────────┐
│              Invoice Sources (5M+ invoices)                  │
│  Supplier Portals | Email Attachments | Scanned Documents   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│          Azure Document Intelligence (Form Recognizer)       │
│          Intelligent OCR + Table Extraction                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                 Microsoft Fabric OneLake                     │
│                 (Central Data Lakehouse)                     │
│                                                              │
│  Raw Invoices → Bronze Layer → Silver Layer → Gold Layer    │
│  (PDF/Excel)    (OCR Output)    (Validated)   (Analytics)   │
└──────────┬─────────────────────────────────┬────────────────┘
           │                                 │
┌──────────▼───────────┐       ┌────────────▼──────────────┐
│  Data Factory        │       │  KQL Database (Real-Time) │
│  Batch Pipelines     │       │  - Anomaly detection      │
│  - Monthly cycles    │       │  - Duplicate queries      │
│  - Reconciliation    │       │  - Cross-JV analytics     │
└──────────┬───────────┘       └────────────┬──────────────┘
           │                                 │
┌──────────▼──────────────────────────────────▼─────────────┐
│          Fabric ML Workspace                               │
│          - GPT-4 semantic validation                       │
│          - Pricing outlier models                          │
│          - Continuous learning pipeline                    │
└──────────┬─────────────────────────────────────────────────┘
           │
┌──────────▼─────────────────────────────────────────────────┐
│          Power BI Semantic Models                          │
│          - Focused review sets dashboard                   │
│          - Commercial manager workload allocation          │
│          - £100M+ savings tracking                         │
└────────────────────────────────────────────────────────────┘
```

---

### Phase 1D Integration: VisHub 2.0

**Purpose**: Unified visualization platform across Safety + Cost + Assets + Utilities

#### Business Context
- **Challenge**: 8+ fragmented systems (AIMS, CDEs, SharePoint, standalone tools)
- **Current Process**: Context switching between systems, no unified intelligence
- **Value Proposition**: Single pane of glass for project health, cross-domain insights

#### Architecture

**Unified Data Access Layer**:
```python
class VisHubDataAggregator:
    def get_unified_project_view(self, project_id: str) -> UnifiedView:
        # Parallel data fetching from all domains
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                'gpr': executor.submit(self.get_gpr_data, project_id),
                'bim': executor.submit(self.get_bim_data, project_id),
                'assets': executor.submit(self.get_asset_data, project_id),
                'safety': executor.submit(self.get_safety_data, project_id),
                'costs': executor.submit(self.get_cost_data, project_id)
            }

            results = {key: future.result() for key, future in futures.items()}

        # Cross-domain correlation
        correlations = self.analyze_cross_domain_patterns(results)

        return UnifiedView(
            gpr_surveys=results['gpr'],
            bim_models=results['bim'],
            asset_readiness=results['assets'],
            safety_risks=results['safety'],
            cost_status=results['costs'],
            cross_domain_insights=correlations
        )
```

**Frontend Architecture**:
```
┌─────────────────────────────────────────────────────────────┐
│                    VisHub 2.0 Interface                      │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Geographic   │  │ Asset-Based  │  │ Timeline     │      │
│  │ Navigation   │  │ View         │  │ View         │      │
│  │              │  │              │  │              │      │
│  │ Mapbox GL JS │  │ Neo4j Graph  │  │ Gantt Chart  │      │
│  │ - GPR scans  │  │ - Asset deps │  │ - Milestones │      │
│  │ - BIM models │  │ - Cert links │  │ - Incidents  │      │
│  │ - Incidents  │  │ - Safety     │  │ - Invoices   │      │
│  │ - Invoices   │  │              │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Cross-Domain Intelligence Panel               │  │
│  │                                                        │  │
│  │  📊 Project Health Score: 78/100                      │  │
│  │  ⚠️ Safety Alerts: 2 High-Risk Sites                  │  │
│  │  💷 Cost Anomalies: 15 Flagged for Review            │  │
│  │  📦 Asset Readiness: 82% TAEM Compliant              │  │
│  │  🔍 Utilities: 127 Detected, 3 Clashes               │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**API Gateway Pattern**:
```
GET /api/v1/vishub/projects/{id}/unified-dashboard

Returns:
{
  "project_id": "hs2-phase1-package3",
  "project_health_score": 78,
  "domains": {
    "utilities": {
      "surveys_completed": 45,
      "utilities_detected": 127,
      "pas128_compliance": "95%",
      "clashes_detected": 3
    },
    "assets": {
      "total_assets": 12500,
      "certified_assets": 10250,
      "taem_compliance": 82,
      "critical_missing_certs": 15
    },
    "safety": {
      "current_risk_score": 65,
      "high_risk_sites": 2,
      "incidents_this_month": 8,
      "leading_indicator_trend": "improving"
    },
    "costs": {
      "invoices_processed": 45000,
      "anomalies_flagged": 15,
      "duplicate_detections": 3,
      "estimated_savings": "£2.5M"
    }
  },
  "cross_domain_insights": [
    {
      "insight_type": "correlation",
      "description": "High safety risk score at Site A correlates with 15% cost overrun on steel procurement",
      "affected_domains": ["safety", "costs"],
      "recommended_action": "Investigate contractor safety practices and procurement patterns"
    }
  ]
}
```

---

## Phase 2: BIM/LiDAR Integration (Planned)

### BIM Validation Module

**Capabilities**:
- IFC file parsing (Industry Foundation Classes)
- 3D model visualization in browser (IFC.js)
- Geometry validation
- Standards compliance checking
- Clash detection with detected utilities

**BIM Processing Workflow**:
```python
class BIMProcessingWorkflow:
    def process_ifc_file(self, file_path: str) -> ProcessedBIM:
        # Stage 1: IFC Parsing
        ifc_file = ifcopenshell.open(file_path)

        # Stage 2: Extract Elements
        elements = self.extract_elements(
            ifc_file=ifc_file,
            types=['IfcWall', 'IfcSlab', 'IfcBeam', 'IfcColumn']
        )

        # Stage 3: Spatial Indexing
        spatial_index = self.create_spatial_index(
            elements=elements,
            method='rtree'
        )

        # Stage 4: Validation
        validation_results = self.validate_model(
            elements=elements,
            standards=['ISO 19650', 'PAS 1192']
        )

        # Stage 5: Clash Detection
        clashes = self.detect_clashes(
            bim_elements=elements,
            detected_utilities=self.get_utilities_in_bounds(spatial_index.bounds)
        )

        return ProcessedBIM(
            element_count=len(elements),
            validation=validation_results,
            clashes=clashes,
            spatial_bounds=spatial_index.bounds
        )
```

### LiDAR Processing Module

**Capabilities**:
- LAS/LAZ point cloud parsing
- Large file streaming (100GB+ support)
- Progress monitoring over time
- Planned vs. actual comparison
- Volume calculations (earthwork)

**LiDAR Processing Workflow**:
```python
class LiDARProcessingWorkflow:
    def process_point_cloud(self, file_path: str) -> ProcessedLiDAR:
        # Stage 1: Stream Parse (Memory Efficient)
        point_cloud = self.stream_parse_las(
            file_path=file_path,
            chunk_size=10_000_000  # 10M points per chunk
        )

        # Stage 2: Downsampling
        downsampled = self.downsample(
            point_cloud=point_cloud,
            method='voxel_grid',
            voxel_size=0.05  # 5cm voxels
        )

        # Stage 3: Ground Classification
        ground, non_ground = self.classify_ground(
            point_cloud=downsampled,
            algorithm='cloth_simulation'
        )

        # Stage 4: Feature Extraction
        features = self.extract_features(
            point_cloud=non_ground,
            types=['buildings', 'vegetation', 'infrastructure']
        )

        # Stage 5: Progress Monitoring
        progress = self.compare_with_baseline(
            current_scan=point_cloud,
            baseline_scan=self.get_baseline(),
            bim_model=self.get_bim_model()
        )

        return ProcessedLiDAR(
            point_count=len(point_cloud),
            features=features,
            progress_metrics=progress
        )
```

### Integration Architecture

**Cross-Domain Data Links**:
```
projects (Unified Table)
├── gpr_surveys → detected_utilities
├── bim_models → bim_elements
├── lidar_scans → progress_snapshots
└── Integration Tables:
    ├── bim_utility_clashes (BIM ↔ GPR)
    ├── bim_lidar_alignment (BIM ↔ LiDAR)
    └── construction_progress (All domains)
```

**Unified Project API**:
```
GET  /api/v1/projects/{id}/dashboard
├── GPR survey status
├── Detected utilities count
├── BIM model metadata
├── Latest LiDAR scan date
├── Combined risk assessment
└── Compliance status (PAS 128 + ISO 19650)
```

---

## Technology Stack

### Backend (Python)
**Core Framework**:
- **FastAPI** 0.104.1 - Async web framework
- **Uvicorn** 0.24.0 - ASGI server
- **Pydantic** 2.5.0 - Data validation

**Database**:
- **PostgreSQL** 15 with extensions:
  - **PostGIS** - Spatial queries
  - **PGVector** - Vector similarity search
- **SQLAlchemy** 2.0.23 - Async ORM
- **Alembic** 1.13.1 - Migrations

**Data Processing**:
- **obspy** 1.4.0 - SEG-Y processing
- **segyio** 1.9.11 - GPR file parsing
- **scipy** 1.11.4 - Signal processing
- **scikit-learn** 1.3.2 - ML models

**Phase 2 Additions**:
- **ifcopenshell** 0.7.0 - IFC parsing
- **open3d** 0.17.0 - Point cloud processing
- **laspy** 2.5.0 - LAS/LAZ parsing
- **langchain** - LLM orchestration
- **minio** 7.2.0 - Object storage client

**Task Queue**:
- **Celery** 5.3.0 - Background tasks
- **Redis** 7.0 - Message broker + cache

### Frontend (Phase 2)
**Core Framework**:
- **React** 18.2.0 + TypeScript
- **Vite** 5.0.0 - Build tool
- **Redux Toolkit** 2.0.0 - State management

**Mapping & 3D**:
- **MapLibre GL JS** 3.0.0 - 2D mapping (open-source)
- **IFC.js** 0.0.126 - BIM 3D viewer
- **CesiumJS** (optional) - Terrain 3D

**UI Framework**:
- **TailwindCSS** 3.4.0 - Styling
- **Shadcn/ui** - Component library

### Infrastructure
**Local Development**:
- **Docker Compose** - Multi-container orchestration
- **MinIO** - S3-compatible object storage
- **TileServer-GL** - Basemap tile server

**Production (AWS)**:
- **ECS/EKS** - Container orchestration
- **RDS PostgreSQL** - Managed database
- **S3** - Object storage
- **CloudFront** - CDN

---

## Data Architecture

### Database Schema Overview

**Phase 1A Tables (Existing)**:
- `projects` - Project metadata
- `gpr_surveys` - Survey sessions
- `gpr_scans` - Individual scans
- `detected_utilities` - Utility locations
- `environmental_data` - Environmental factors
- `validation_results` - Ground truth validation
- `ml_models` - Model registry

**Phase 2 Tables (Planned)**:
- `bim_models` - IFC file metadata
- `bim_elements` - Building elements
- `bim_versions` - Version control
- `lidar_scans` - Point cloud metadata
- `progress_snapshots` - Time-series progress
- `volume_calculations` - Earthwork volumes
- `bim_utility_clashes` - Conflict detection
- `bim_lidar_alignment` - Registration data
- `document_embeddings` - LLM vector store

**See Also**: [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md) for complete schema reference

### Object Storage Structure

**MinIO Buckets** (Local) / **S3 Buckets** (Production):
```
gpr-data/
├── raw/                  # Original SEG-Y, DZT files
├── processed/            # Filtered, analyzed data
└── exports/              # Report attachments

bim-models/
├── ifc/                  # Industry Foundation Classes files
├── revit/                # Revit RVT files
├── converted/            # Converted formats (glTF)
└── thumbnails/           # Model preview images

lidar-scans/
├── raw/                  # LAS/LAZ point clouds
├── processed/            # Classified, downsampled
├── potree/               # Web-optimized octree format
└── comparisons/          # Progress comparison data

documents/
├── pdf/                  # Reports, specifications
├── cad/                  # DWG, DXF drawings
└── photos/               # Site photos, GeoTIFF

reports/
├── pas128/               # Compliance reports (PDF)
├── bim_validation/       # Model validation reports
└── progress/             # Construction progress reports
```

---

## API Architecture

### API Versioning & Organization

**Base URL**: `https://api.infrastructure-intelligence.com/api/v1/`

**Authentication**: JWT tokens via `/api/v1/auth/login`

### Endpoint Groups

#### 1. Authentication & User Management
```
POST   /api/v1/auth/register
POST   /api/v1/auth/login
POST   /api/v1/auth/refresh
GET    /api/v1/users/me
```

#### 2. Project Management
```
POST   /api/v1/projects
GET    /api/v1/projects
GET    /api/v1/projects/{id}
GET    /api/v1/projects/{id}/dashboard
DELETE /api/v1/projects/{id}
```

#### 3. GPR Data (Phase 1A)
```
POST   /api/v1/gpr/upload
GET    /api/v1/gpr/surveys
GET    /api/v1/gpr/surveys/{id}
POST   /api/v1/gpr/process
GET    /api/v1/gpr/scans
GET    /api/v1/gpr/statistics
```

#### 4. Material Classification (Phase 1A)
```
POST   /api/v1/material-classification/predict
POST   /api/v1/material-classification/analyze
GET    /api/v1/material-classification/models
```

#### 5. PAS 128 Compliance (Phase 1A)
```
POST   /api/v1/compliance/quality-level/determine
GET    /api/v1/compliance/reports/{id}
POST   /api/v1/compliance/generate-report
```

#### 6. BIM Validation (Phase 2 - Planned)
```
POST   /api/v1/bim/upload
GET    /api/v1/bim/models
GET    /api/v1/bim/models/{id}
POST   /api/v1/bim/validate
GET    /api/v1/bim/models/{id}/elements
POST   /api/v1/bim/clash-detection
```

#### 7. LiDAR Processing (Phase 2 - Planned)
```
POST   /api/v1/lidar/upload
GET    /api/v1/lidar/scans
POST   /api/v1/lidar/process
GET    /api/v1/lidar/scans/{id}/comparison
POST   /api/v1/lidar/progress-report
```

#### 8. Integration & Analytics (Phase 2 - Planned)
```
GET    /api/v1/analytics/risk-assessment
GET    /api/v1/analytics/project-health
POST   /api/v1/reports/generate-unified
```

**See Also**: [../api/README.md](../api/README.md) for detailed API documentation

---

## Security & Compliance

### Authentication & Authorization

**JWT Token Structure**:
```json
{
  "sub": "user_id",
  "email": "user@example.com",
  "role": "surveyor",
  "project_ids": ["proj_123", "proj_456"],
  "permissions": ["read:surveys", "write:surveys"],
  "exp": 1735689600
}
```

**Role-Based Access Control (RBAC)**:
- **Admin**: Full system access
- **Project Manager**: Project-level management
- **Surveyor**: Data collection and processing
- **Viewer**: Read-only access

### Data Protection

**GDPR Compliance**:
- Project-level data isolation
- Automated PII detection and redaction
- 7-year audit trail (CDM requirement)
- Right to erasure implementation

**PAS 128:2022 Compliance**:
- Quality level automation (QL-A through QL-D)
- Full documentation trail
- Method validation
- Accuracy standards (>95% target)

**Security Measures**:
- Encryption at rest (AES-256)
- Encryption in transit (TLS 1.3)
- Secrets management (AWS Secrets Manager / HashiCorp Vault)
- Regular security audits

---

## Deployment Architecture

### Local Development (Docker Compose)

**Services**:
1. Frontend (React - Port 3000)
2. Backend (FastAPI - Port 8000)
3. PostgreSQL (Port 5432)
4. MinIO (Ports 9000/9001)
5. Redis (Port 6379)
6. Celery Worker
7. Flower UI (Port 5555)
8. TileServer-GL (Port 8080)

**Setup Command**:
```bash
docker compose up -d
```

**See Also**: [../deployment/LOCAL_SETUP.md](../deployment/LOCAL_SETUP.md)

### Production (AWS)

**Architecture**:
```
Route53 → CloudFront → ALB → ECS (Fargate)
                                ├── Backend Services
                                └── Celery Workers
                      ↓
                   RDS PostgreSQL + S3 + ElastiCache Redis
```

**High Availability**:
- Multi-AZ database deployment
- Auto-scaling ECS services
- S3 cross-region replication
- CloudFront edge caching

**See Also**: [../deployment/AWS_MIGRATION.md](../deployment/AWS_MIGRATION.md)

---

## Performance Targets

### Phase 1A Metrics
- **API Latency**: <200ms P95
- **GPR Processing**: <5 minutes per 500MB file
- **Material Classification**: <2 seconds per utility
- **Report Generation**: <10 minutes (vs. 8 hours manual)

### Phase 2 Targets
- **BIM Upload**: <30 seconds for typical IFC
- **LiDAR Processing**: <10 minutes for 10GB point cloud
- **3D Rendering**: <3 seconds initial load
- **Unified Report**: <5 minutes for all domains

---

## References

- [DIAGRAMS.md](DIAGRAMS.md) - Visual architecture diagrams
- [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md) - Complete schema reference
- [API_DESIGN.md](API_DESIGN.md) - API patterns and conventions
- [../technical/INFRASTRUCTURE_MERGE_PLAN.md](../technical/INFRASTRUCTURE_MERGE_PLAN.md) - Phase 2 implementation plan

---

**Original Source Documents** (Archived):
- [Consolidated-Architecture.md](../archived/Consolidated-Architecture.md)
- [PHASE_1A_ARCHITECTURE_SPECIFICATION.md](../archived/PHASE_1A_ARCHITECTURE_SPECIFICATION.md)

---

Last Updated: 2025-11-25
Document Version: 3.0
