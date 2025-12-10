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

The Infrastructure Intelligence Platform has transformed from a specialized **GPR-based underground utility detection system** (Phase 1A) into a comprehensive **Multi-Domain Enterprise Intelligence Platform** that addresses **six critical infrastructure domains**:

**NEW: HS2 Automated Progress Assurance** - Patent-pending hyperspectral imaging for non-destructive material quality verification combined with LiDAR progress monitoring.

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

#### Phase 1C-Extended: HS2 Automated Progress Assurance 🆕 (Weeks 8-15)
**"Google Maps for Construction" - The Complete Solution**

**Hyperspectral Imaging Pipeline** 🔥 **PATENT-PENDING DIFFERENTIATOR**
- ⏳ **Material Quality Verification**: Capture 100+ spectral bands to verify concrete strength, detect internal defects
- ⏳ **Non-Destructive Testing**: Replace £500-£2,000 per core/destructive test with hyperspectral analysis
- ⏳ **Multi-Sensor Fusion**: Combine hyperspectral + LiDAR + BIM for complete assurance
- ⏳ **Quality Evidence**: Spectral signatures as material evidence in compliance reports
- ⏳ **Real-Time Processing**: <10 minutes from scan to report (vs days for manual verification)

**Progress Monitoring & BIM Comparison**
- ⏳ **Automated BIM-to-Reality**: ICP alignment, voxel comparison, deviation detection
- ⏳ **3D Color-Coded Visualization**: Green (match), yellow (minor deviation), red (critical issue)
- ⏳ **Progress Dashboards**: Interactive "Google Maps" style site navigation
- ⏳ **One-Click Reports**: Automated PDF generation with material quality + progress status
- ⏳ **Timeline Scrubber**: Historical progress playback over weeks/months

**Expected Value Delivery**
- 📊 **95% reduction** in reporting time (10 minutes vs days)
- 💰 **40+ hours saved** per site monthly
- ✅ **Material evidence** for regulatory compliance (no destructive testing)
- 🎯 **95%+ accuracy** in material quality predictions

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

## Phase 1C-Extended: HS2 Automated Progress Assurance Architecture 🆕

### Overview

**The Challenge**: "What have we actually built this month?" - Currently takes DAYS to answer.

**Our Solution**: Patent-pending **hyperspectral imaging** combined with LiDAR and AI to provide:
- Material quality verification WITHOUT destructive testing
- Real-time progress monitoring with BIM comparison
- One-click compliance reports in <10 minutes

**Key Differentiator vs Competitors** (Doxel, Buildots, LiDARit, Mach9):
- ✅ **They do**: LiDAR scanning, BIM comparison, visual progress tracking
- ❌ **They CANNOT do**: Material quality verification, internal defect detection, material evidence
- 🔥 **We do EVERYTHING they do + Hyperspectral Imaging + Usable Intelligence**

---

### Hyperspectral Imaging Pipeline Architecture

**What is Hyperspectral Imaging?**

Traditional cameras capture 3 spectral bands (Red, Green, Blue). Hyperspectral cameras capture **100-200+ spectral bands** across visible and near-infrared wavelengths. The **Specim IQ** camera (recommended for HS2) captures **204 spectral bands** across 400-1000nm with ~3nm spectral resolution. This allows us to "see inside" materials and detect properties invisible to the human eye or regular cameras.

**Technical Approach**:

```python
class HyperspectralProcessingWorkflow:
    """
    Patent-Pending: Multi-Spectral Data Fusion for Non-Destructive Quality Assurance
    """

    def process_hyperspectral_scan(self, scan_file: str, location: dict) -> MaterialQualityReport:
        # Stage 1: Hyperspectral Data Acquisition
        hyperspectral_cube = self.load_hyperspectral_data(
            file_path=scan_file,
            format='ENVI',  # Common format: BSQ/BIL/BIP
            wavelengths=range(400, 2500, 10)  # 210 spectral bands
        )

        # Stage 2: Atmospheric Correction & Calibration
        corrected_cube = self.apply_atmospheric_correction(
            data=hyperspectral_cube,
            method='empirical_line',  # Use reference panels in scene
            solar_angle=location['solar_angle'],
            atmospheric_conditions=location['weather']
        )

        # Stage 3: Spectral Unmixing (Identify Material Composition)
        material_maps = self.spectral_unmixing(
            data=corrected_cube,
            method='linear_unmixing',
            endmembers=self.get_spectral_library()  # Concrete, steel, asphalt signatures
        )

        # Stage 4: Material Property Prediction (Deep Learning)
        material_properties = self.predict_material_properties(
            spectral_data=corrected_cube,
            material_maps=material_maps,
            model=self.trained_cnn_model  # Trained on lab test data
        )

        # Predict concrete compressive strength
        concrete_regions = material_maps['concrete_mask']
        strength_map = self.predict_concrete_strength(
            spectral_signature=corrected_cube[concrete_regions],
            model=self.strength_prediction_model  # Trained on core samples
        )

        # Stage 5: Defect Detection (Anomaly Detection)
        defects = self.detect_internal_defects(
            spectral_data=corrected_cube,
            material_type='concrete',
            defect_types=['voids', 'cracks', 'delamination', 'moisture_ingress']
        )

        # Stage 6: Quality Scoring
        quality_scores = self.calculate_quality_scores(
            strength_map=strength_map,
            defects=defects,
            material_properties=material_properties,
            specification_requirements=self.get_project_specs()
        )

        # Stage 7: Generate Material Evidence
        material_evidence = self.generate_spectral_evidence(
            spectral_signature=corrected_cube,
            quality_scores=quality_scores,
            reference_library=self.spectral_library
        )

        return MaterialQualityReport(
            concrete_strength_map=strength_map,
            defect_locations=defects,
            quality_scores=quality_scores,
            material_evidence=material_evidence,
            confidence_intervals=self.calculate_confidence(quality_scores)
        )

    def predict_concrete_strength(self, spectral_signature: np.ndarray,
                                   model: torch.nn.Module) -> np.ndarray:
        """
        Predict concrete compressive strength (MPa) from hyperspectral signature

        Training Data Required:
        - 1,000+ concrete samples with:
          - Hyperspectral scans
          - Lab-tested compressive strength (destructive core tests)
          - Age, mix design, curing conditions

        Model: CNN trained on spectral bands 800-1200nm (sensitive to cement hydration)
        Expected Accuracy: 90-95% correlation with destructive tests (R² > 0.85)
        """
        # Preprocess spectral data
        features = self.extract_spectral_features(
            signature=spectral_signature,
            bands_of_interest=[800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200]
        )

        # Predict strength using trained model
        with torch.no_grad():
            strength_predictions = model(torch.tensor(features))

        return strength_predictions.numpy()

    def detect_internal_defects(self, spectral_data: np.ndarray,
                                material_type: str, defect_types: List[str]) -> List[Defect]:
        """
        Detect internal defects without destructive testing

        Physics Basis:
        - Voids: Higher reflectance in NIR (no material present)
        - Cracks: Edge effects in spectral gradients
        - Delamination: Subsurface anomalies (thermal signature difference)
        - Moisture: Strong absorption at 1450nm and 1950nm water bands
        """
        defects = []

        for defect_type in defect_types:
            if defect_type == 'voids':
                # Voids show as bright spots in NIR due to air gaps
                void_indicator = spectral_data[:, :, self.bands['nir']] > self.thresholds['void']
                defects.extend(self.extract_void_locations(void_indicator))

            elif defect_type == 'moisture_ingress':
                # Water absorption bands at 1450nm and 1950nm
                water_band_1 = spectral_data[:, :, self.bands['1450nm']]
                water_band_2 = spectral_data[:, :, self.bands['1950nm']]
                moisture_map = (water_band_1 + water_band_2) / 2
                defects.extend(self.extract_moisture_regions(moisture_map))

            elif defect_type == 'cracks':
                # Cracks detected via spectral gradient analysis
                gradients = self.calculate_spectral_gradients(spectral_data)
                crack_indicator = self.detect_edges(gradients)
                defects.extend(self.extract_crack_locations(crack_indicator))

        return defects
```

**Database Schema for Hyperspectral Data**:

```sql
-- Hyperspectral scans metadata
CREATE TABLE hyperspectral_scans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id),
    site_location VARCHAR(255) NOT NULL,
    scan_date TIMESTAMPTZ NOT NULL,

    -- Camera specifications
    camera_model VARCHAR(100),  -- e.g., 'Specim IQ', 'Corning microHSI'
    wavelength_range VARCHAR(50),  -- e.g., '400-2500nm'
    spectral_resolution FLOAT,  -- nm per band
    band_count INTEGER,  -- e.g., 204 bands
    spatial_resolution FLOAT,  -- meters per pixel

    -- Environmental conditions
    solar_angle FLOAT,
    atmospheric_conditions JSONB,  -- Temperature, humidity, visibility
    weather VARCHAR(50),

    -- File storage
    raw_file_path TEXT NOT NULL,  -- S3: hyperspectral-data/raw/
    processed_file_path TEXT,  -- S3: hyperspectral-data/processed/
    file_size_bytes BIGINT,
    format VARCHAR(20),  -- 'ENVI', 'HDF5', 'GeoTIFF'

    -- Geospatial
    location GEOGRAPHY(POINT, 4326),
    coverage_area GEOGRAPHY(POLYGON, 4326),
    elevation_m FLOAT,

    -- Processing status
    processing_status VARCHAR(50) DEFAULT 'pending',  -- pending, processing, completed, failed
    processed_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Material quality assessments from hyperspectral analysis
CREATE TABLE material_quality_assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scan_id UUID REFERENCES hyperspectral_scans(id) ON DELETE CASCADE,

    -- Material identification
    material_type VARCHAR(100),  -- 'concrete', 'steel', 'asphalt', 'soil'
    material_subtype VARCHAR(100),  -- 'C40 concrete', 'Grade 500 steel'

    -- Spatial location
    location_in_scan GEOGRAPHY(POINT, 4326),  -- GPS coordinates
    pixel_coordinates JSONB,  -- {x: 1024, y: 768} in scan image
    region_area_m2 FLOAT,

    -- Quality metrics (concrete-specific)
    predicted_strength_mpa FLOAT,  -- Compressive strength prediction
    strength_confidence FLOAT,  -- 0-100%
    specification_strength_mpa FLOAT,  -- Required strength from specs
    meets_specification BOOLEAN,

    -- Defects detected
    defects_detected JSONB,  -- Array of {type, severity, location, confidence}
    defect_count INTEGER DEFAULT 0,
    critical_defects INTEGER DEFAULT 0,

    -- Spectral evidence (material fingerprint)
    spectral_signature JSONB,  -- Key wavelengths and reflectance values
    spectral_match_score FLOAT,  -- Similarity to reference library (0-100)

    -- Overall quality score
    quality_score FLOAT,  -- 0-100 composite score
    quality_grade VARCHAR(10),  -- 'A', 'B', 'C', 'F'

    -- Model metadata
    model_version VARCHAR(50),
    model_confidence FLOAT,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Spectral library (reference materials)
CREATE TABLE spectral_library (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Material identity
    material_name VARCHAR(255) NOT NULL,  -- 'C40 Concrete, 28-day cure'
    material_category VARCHAR(100),  -- 'Concrete', 'Steel', 'Asphalt'
    material_properties JSONB,  -- Known physical properties (strength, density, etc.)

    -- Spectral signature
    wavelengths FLOAT[],  -- Array of wavelengths (nm)
    reflectance_values FLOAT[],  -- Corresponding reflectance (0-1)
    spectral_curve JSONB,  -- Full spectral curve data

    -- Acquisition conditions
    acquisition_date DATE,
    lab_conditions JSONB,  -- Lighting, angle, atmospheric
    calibration_method VARCHAR(100),

    -- Validation data
    lab_test_results JSONB,  -- Destructive test results for validation
    sample_source VARCHAR(255),
    validation_confidence FLOAT,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Hyperspectral-LiDAR fusion (combined analysis)
CREATE TABLE hyperspectral_lidar_fusion (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hyperspectral_scan_id UUID REFERENCES hyperspectral_scans(id),
    lidar_scan_id UUID REFERENCES lidar_scans(id),

    -- Alignment data
    transformation_matrix JSONB,  -- 4x4 transformation for co-registration
    alignment_error_m FLOAT,  -- RMS alignment error
    alignment_method VARCHAR(100),  -- 'ICP', 'Feature-based', 'Manual'

    -- Fused data products
    fused_point_cloud_path TEXT,  -- 3D points with spectral attributes
    material_mapped_mesh_path TEXT,  -- 3D mesh with material quality overlay

    -- Quality assessment
    alignment_quality VARCHAR(50),  -- 'Excellent', 'Good', 'Fair', 'Poor'
    coverage_percentage FLOAT,  -- % of LiDAR points with hyperspectral data

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_hyperspectral_scans_project ON hyperspectral_scans(project_id);
CREATE INDEX idx_hyperspectral_scans_date ON hyperspectral_scans(scan_date);
CREATE INDEX idx_hyperspectral_scans_status ON hyperspectral_scans(processing_status);
CREATE INDEX idx_material_quality_scan ON material_quality_assessments(scan_id);
CREATE INDEX idx_material_quality_type ON material_quality_assessments(material_type);
CREATE INDEX idx_material_quality_score ON material_quality_assessments(quality_score);
CREATE INDEX idx_spectral_library_category ON spectral_library(material_category);

-- Spatial indexes
CREATE INDEX idx_hyperspectral_location ON hyperspectral_scans USING GIST(location);
CREATE INDEX idx_hyperspectral_coverage ON hyperspectral_scans USING GIST(coverage_area);
CREATE INDEX idx_material_quality_location ON material_quality_assessments USING GIST(location_in_scan);
```

---

### BIM-to-Reality Comparison Pipeline

**Automated Deviation Detection**:

```python
class BIMRealityComparison:
    """
    Automated comparison of BIM models against LiDAR reality captures
    """

    def compare_bim_to_reality(self, bim_model_path: str,
                                lidar_scan_path: str) -> DeviationReport:
        # Stage 1: Load BIM Model (IFC format)
        ifc_model = ifcopenshell.open(bim_model_path)
        bim_elements = self.extract_bim_geometry(ifc_model)

        # Stage 2: Load LiDAR Point Cloud
        point_cloud = self.load_lidar_scan(lidar_scan_path)

        # Stage 3: Align LiDAR to BIM Coordinate System (ICP Algorithm)
        aligned_cloud, transformation = self.align_point_cloud_to_bim(
            point_cloud=point_cloud,
            bim_bounds=bim_elements.bounds,
            method='iterative_closest_point',
            max_iterations=50,
            convergence_threshold=0.001  # meters
        )

        # Stage 4: Voxel-Based Comparison
        deviations = self.compute_voxel_deviations(
            bim_geometry=bim_elements,
            reality_points=aligned_cloud,
            voxel_size=0.05  # 5cm voxels
        )

        # Stage 5: Semantic Element Comparison
        element_deviations = []
        for element in bim_elements:
            # For each BIM element (wall, beam, column), check reality
            reality_geometry = self.extract_reality_geometry(
                element_bounds=element.bounds,
                point_cloud=aligned_cloud
            )

            deviation = self.calculate_element_deviation(
                designed=element,
                built=reality_geometry
            )

            element_deviations.append({
                'element_id': element.id,
                'element_type': element.type,
                'deviation_mm': deviation.mean * 1000,
                'max_deviation_mm': deviation.max * 1000,
                'volume_difference_m3': deviation.volume_diff,
                'severity': self.classify_severity(deviation)
            })

        # Stage 6: Generate Color-Coded Visualization
        color_coded_mesh = self.generate_color_coded_visualization(
            deviations=deviations,
            color_scale='green_yellow_red'  # Green=match, Yellow=minor, Red=critical
        )

        return DeviationReport(
            element_deviations=element_deviations,
            overall_alignment_error=transformation.error,
            color_coded_mesh=color_coded_mesh,
            compliance_status=self.assess_compliance(element_deviations)
        )

    def align_point_cloud_to_bim(self, point_cloud, bim_bounds,
                                  method='iterative_closest_point',
                                  max_iterations=50, convergence_threshold=0.001):
        """
        ICP (Iterative Closest Point) algorithm to align LiDAR scan to BIM model

        Library: Open3D (open3d.org)
        """
        import open3d as o3d

        # Convert BIM geometry to point cloud
        bim_point_cloud = self.bim_to_point_cloud(bim_bounds)

        # Initial alignment using feature-based registration
        initial_transform = self.feature_based_alignment(
            source=point_cloud,
            target=bim_point_cloud
        )

        # Refine with ICP
        icp_result = o3d.pipelines.registration.registration_icp(
            source=point_cloud,
            target=bim_point_cloud,
            max_correspondence_distance=convergence_threshold,
            init=initial_transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iterations
            )
        )

        # Apply transformation
        aligned_cloud = point_cloud.transform(icp_result.transformation)

        return aligned_cloud, icp_result
```

---

### Progress Verification Database Schema

```sql
-- LiDAR scans for progress monitoring
CREATE TABLE progress_lidar_scans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id),
    site_location VARCHAR(255) NOT NULL,
    scan_date TIMESTAMPTZ NOT NULL,

    -- Scanner specifications
    scanner_model VARCHAR(100),  -- 'Leica RTC360', 'Faro Focus'
    point_count BIGINT,  -- Number of 3D points
    point_density FLOAT,  -- Points per m²
    scan_quality VARCHAR(50),  -- 'High', 'Medium', 'Low'

    -- File storage
    raw_file_path TEXT NOT NULL,  -- S3: lidar-scans/raw/ (LAZ/LAS)
    processed_file_path TEXT,  -- S3: lidar-scans/processed/
    potree_octree_path TEXT,  -- S3: lidar-scans/potree/ (web visualization)
    file_size_bytes BIGINT,

    -- Geospatial
    location GEOGRAPHY(POINT, 4326),
    coverage_area GEOGRAPHY(POLYGON, 4326),
    coordinate_system VARCHAR(100),  -- 'EPSG:27700' (British National Grid)

    -- Processing
    processing_status VARCHAR(50) DEFAULT 'pending',
    processed_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- BIM model uploads
CREATE TABLE bim_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id),
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(50),
    upload_date TIMESTAMPTZ DEFAULT NOW(),

    -- File metadata
    file_path TEXT NOT NULL,  -- S3: bim-models/ifc/
    file_format VARCHAR(20),  -- 'IFC', 'Revit', 'glTF'
    file_size_bytes BIGINT,

    -- Model metadata
    ifc_schema VARCHAR(50),  -- 'IFC4', 'IFC2x3'
    element_count INTEGER,
    discipline VARCHAR(100),  -- 'Architectural', 'Structural', 'MEP'

    -- Spatial bounds
    bounds_min_x FLOAT,
    bounds_min_y FLOAT,
    bounds_min_z FLOAT,
    bounds_max_x FLOAT,
    bounds_max_y FLOAT,
    bounds_max_z FLOAT,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- BIM-LiDAR alignment results
CREATE TABLE bim_lidar_alignments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    bim_model_id UUID REFERENCES bim_models(id),
    lidar_scan_id UUID REFERENCES progress_lidar_scans(id),

    -- Alignment transformation
    transformation_matrix JSONB,  -- 4x4 homogeneous transformation
    rotation JSONB,  -- Quaternion or Euler angles
    translation JSONB,  -- X, Y, Z offset
    scale FLOAT DEFAULT 1.0,

    -- Alignment quality
    alignment_method VARCHAR(100),  -- 'ICP', 'Feature-based', 'Manual'
    alignment_error_m FLOAT,  -- RMS error in meters
    iterations_required INTEGER,
    convergence_achieved BOOLEAN,

    -- Alignment metadata
    aligned_by VARCHAR(255),  -- User or system
    alignment_confidence FLOAT,  -- 0-100%

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Deviation analysis results
CREATE TABLE progress_deviation_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    alignment_id UUID REFERENCES bim_lidar_alignments(id),

    -- Element-level deviations
    bim_element_id VARCHAR(255),  -- IFC element GUID
    element_type VARCHAR(100),  -- 'IfcWall', 'IfcBeam', 'IfcColumn'
    element_name VARCHAR(255),

    -- Deviation metrics
    mean_deviation_mm FLOAT,
    max_deviation_mm FLOAT,
    std_deviation_mm FLOAT,
    volume_difference_m3 FLOAT,

    -- Severity classification
    severity VARCHAR(50),  -- 'None', 'Minor', 'Moderate', 'Major', 'Critical'
    within_tolerance BOOLEAN,
    tolerance_threshold_mm FLOAT,

    -- Spatial location
    location GEOGRAPHY(POINT, 4326),
    deviation_geometry GEOGRAPHY(POLYGON, 4326),

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Progress snapshots (time-series tracking)
CREATE TABLE progress_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id),
    snapshot_date TIMESTAMPTZ NOT NULL,

    -- Data sources
    lidar_scan_id UUID REFERENCES progress_lidar_scans(id),
    bim_model_id UUID REFERENCES bim_models(id),
    hyperspectral_scan_id UUID REFERENCES hyperspectral_scans(id),

    -- Progress metrics
    percent_complete FLOAT,  -- 0-100%
    completed_volume_m3 FLOAT,
    planned_volume_m3 FLOAT,
    variance_volume_m3 FLOAT,

    -- Schedule metrics
    planned_completion_date DATE,
    predicted_completion_date DATE,
    schedule_variance_days INTEGER,

    -- Quality metrics
    quality_score FLOAT,  -- 0-100 from hyperspectral analysis
    defects_detected INTEGER,
    critical_issues INTEGER,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Automated reports
CREATE TABLE progress_assurance_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id),
    snapshot_id UUID REFERENCES progress_snapshots(id),
    report_date TIMESTAMPTZ DEFAULT NOW(),

    -- Report metadata
    report_type VARCHAR(100),  -- 'Weekly', 'Monthly', 'Milestone', 'On-Demand'
    report_title VARCHAR(255),
    generated_by VARCHAR(100),  -- 'System' or user name

    -- Report content
    executive_summary TEXT,
    progress_analysis TEXT,
    material_quality_summary TEXT,
    deviation_summary TEXT,
    risk_assessment TEXT,
    recommendations TEXT,

    -- Visualizations
    progress_charts JSONB,  -- Chart data/config
    color_coded_3d_model_path TEXT,  -- S3 path to visualization
    material_quality_heatmap_path TEXT,

    -- File outputs
    pdf_report_path TEXT,  -- S3: reports/progress/
    excel_data_path TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_progress_lidar_project ON progress_lidar_scans(project_id);
CREATE INDEX idx_progress_lidar_date ON progress_lidar_scans(scan_date);
CREATE INDEX idx_bim_models_project ON bim_models(project_id);
CREATE INDEX idx_deviation_alignment ON progress_deviation_analysis(alignment_id);
CREATE INDEX idx_deviation_severity ON progress_deviation_analysis(severity);
CREATE INDEX idx_snapshots_project ON progress_snapshots(project_id);
CREATE INDEX idx_snapshots_date ON progress_snapshots(snapshot_date);
CREATE INDEX idx_reports_project ON progress_assurance_reports(project_id);

-- Spatial indexes
CREATE INDEX idx_progress_lidar_location ON progress_lidar_scans USING GIST(location);
CREATE INDEX idx_deviation_location ON progress_deviation_analysis USING GIST(location);
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

# Technology Decision Matrix & Architectural Justification

This section documents the comprehensive technology selection process for the Infrastructure Intelligence Platform (HS2 Progress Assurance). Each technology decision includes context, alternatives considered, pros/cons analysis, cost breakdown, and risk assessment to provide full architectural defense.

## Decision Summary Matrix

| Technology | Category | Chosen Solution | Key Alternatives | Annual Cost | Risk Level |
|-----------|----------|----------------|------------------|-------------|------------|
| **Backend Framework** | API Layer | **FastAPI** | Django, Flask, Express.js | £0 (open-source) | 🟢 LOW |
| **Frontend Framework** | UI Layer | **React 18 + TypeScript** | Vue.js, Angular, Svelte | £0 (open-source) | 🟢 LOW |
| **Database** | Data Storage | **PostgreSQL + PostGIS + PGVector** | MySQL, MongoDB, SQL Server | £0-£12,000 | 🟢 LOW |
| **Object Storage** | File Storage | **MinIO (S3-compatible)** | AWS S3, Azure Blob, Google Cloud Storage | £0-£3,600 | 🟢 LOW |
| **Task Queue** | Async Processing | **Celery + Redis** | AWS SQS, RabbitMQ, BullMQ | £0-£1,200 | 🟢 LOW |
| **Hyperspectral Camera** | Material Quality | **Specim IQ** | Resonon Pika L, Headwall Nano-Hyperspec | £1,200-£35,000 | 🟡 MEDIUM |
| **LiDAR Scanner** | 3D Capture | **Leica RTC360** | Faro Focus, Trimble X12, GeoSLAM ZEB | £1,500-£45,000 | 🟢 LOW |
| **BIM Viewer** | 3D Visualization | **IFC.js (web-based)** | Autodesk Forge, BIM360, Solibri | £0 (open-source) | 🟢 LOW |
| **Cloud Platform** | Infrastructure | **Microsoft Azure** | AWS, Google Cloud Platform | £2,400-£12,000 | 🟢 LOW |
| **Containerization** | Deployment | **Docker + Docker Compose** | Kubernetes, Nomad, bare metal | £0 (open-source) | 🟢 LOW |
| **Authentication** | Security | **OAuth 2.0 + JWT** | SAML, Auth0, Okta | £0 (self-hosted) | 🟢 LOW |
| **Vector Database** | Embeddings | **PGVector (PostgreSQL extension)** | Pinecone, Weaviate, Chroma | £0 (bundled) | 🟢 LOW |
| **LLM API** | AI/ML | **OpenAI GPT-4o** | Anthropic Claude, Meta Llama, Google Gemini | £1,200-£6,000 | 🟡 MEDIUM |
| **Monitoring** | Observability | **Grafana + Prometheus** | Datadog, New Relic, Dynatrace | £0 (open-source) | 🟢 LOW |
| **CI/CD** | DevOps | **GitHub Actions** | GitLab CI, Jenkins, CircleCI | £0 (open-source) | 🟢 LOW |
| **Documentation** | Knowledge Base | **Markdown + MkDocs** | Confluence, Notion, GitBook | £0 (open-source) | 🟢 LOW |

**Total Annual Cost Range**: £6,300 (conservative) to £115,800 (full purchase)

---

## 1. Backend Framework: FastAPI

### Decision
**Chosen**: FastAPI (Python 3.11+)

### Context
The Infrastructure Intelligence Platform requires:
- High-performance API to handle concurrent requests from field teams, dashboard users, and automated systems
- Machine learning integration for hyperspectral imaging, LiDAR processing, BIM deviation analysis
- Automatic API documentation for integration partners (HS2, tier-1 contractors)
- Modern async/await support for I/O-bound operations (file uploads, database queries, external APIs)

### Alternatives Considered

| Framework | Language | Performance | ML Integration | Auto Docs | Learning Curve |
|-----------|----------|-------------|----------------|-----------|----------------|
| **FastAPI** | Python | ⭐⭐⭐⭐⭐ (async) | ⭐⭐⭐⭐⭐ (native) | ⭐⭐⭐⭐⭐ (OpenAPI) | 🟢 Low |
| Django REST Framework | Python | ⭐⭐⭐ (sync default) | ⭐⭐⭐⭐ (good) | ⭐⭐⭐ (manual) | 🟡 Medium |
| Flask + extensions | Python | ⭐⭐⭐ (sync) | ⭐⭐⭐⭐ (good) | ⭐⭐ (third-party) | 🟢 Low |
| Express.js | Node.js | ⭐⭐⭐⭐ (async) | ⭐⭐ (TensorFlow.js) | ⭐⭐ (manual) | 🟢 Low |
| Spring Boot | Java | ⭐⭐⭐⭐ (threads) | ⭐⭐ (DL4J) | ⭐⭐⭐ (Swagger) | 🔴 High |

### Pros
1. **🚀 Performance**: Async/await support = handle 1000+ concurrent requests (field teams uploading data simultaneously)
2. **🤖 ML-Friendly**: Native Python = seamless integration with NumPy, Pandas, TensorFlow, PyTorch for hyperspectral analysis
3. **📚 Automatic Documentation**: OpenAPI/Swagger UI generated automatically = integration partners (HS2) get live API docs
4. **🔒 Type Safety**: Pydantic models for request/response validation = reduce bugs by 30-40% (based on industry research)
5. **⚡ Fast Development**: Minimal boilerplate = 2-3x faster development vs Django (based on team experience)

### Cons
1. **⚠️ Younger Ecosystem**: Released 2018 (vs Django 2005) = fewer third-party packages, less Stack Overflow answers
2. **⚠️ Async Learning Curve**: Team needs to learn async/await patterns (estimated 2-week onboarding)
3. **⚠️ Database ORM**: No built-in ORM (need SQLAlchemy) vs Django's integrated ORM

### Risk Mitigation
- **Ecosystem Risk**: Use battle-tested libraries (SQLAlchemy 2.0, Pydantic, Starlette)
- **Learning Curve**: Provide internal training on async patterns, create coding standards document
- **ORM Integration**: SQLAlchemy 2.0 has async support + mature ecosystem

### Cost Analysis
- **License**: Open-source (MIT License) = £0/year
- **Training**: 2-week team upskilling = £4,000 (8 developers × £500/week)
- **Development Speed**: +30% faster development = -£12,000/year savings
- **Net Cost**: -£8,000/year (ROI positive)

### Verdict
✅ **FastAPI is the optimal choice**: Performance + ML integration + Auto-docs outweigh ecosystem maturity concerns.

**Risk Level**: 🟢 **LOW** (Mature project with 70K+ GitHub stars, production use at Microsoft, Uber, Netflix)

---

## 2. Frontend Framework: React 18 + TypeScript

### Decision
**Chosen**: React 18 with TypeScript

### Context
The platform requires:
- 3D BIM visualization (IFC.js integration)
- Real-time dashboard updates (WebSocket for live site data)
- Progressive Web App (PWA) for field use on tablets/phones
- Component reusability across modules (GPR, BIM, Hyperspectral, LiDAR)

### Alternatives Considered

| Framework | Ecosystem Size | 3D Rendering | Learning Curve | TypeScript Support | PWA Support |
|-----------|----------------|--------------|----------------|-------------------|-------------|
| **React 18** | ⭐⭐⭐⭐⭐ (Largest) | ⭐⭐⭐⭐⭐ (Three.js, IFC.js) | 🟡 Medium | ⭐⭐⭐⭐⭐ (First-class) | ⭐⭐⭐⭐⭐ |
| Vue.js 3 | ⭐⭐⭐⭐ (Large) | ⭐⭐⭐⭐ (Good) | 🟢 Low | ⭐⭐⭐⭐ (Good) | ⭐⭐⭐⭐ |
| Angular 17 | ⭐⭐⭐ (Medium) | ⭐⭐⭐ (OK) | 🔴 High | ⭐⭐⭐⭐⭐ (Built-in) | ⭐⭐⭐⭐ |
| Svelte | ⭐⭐ (Small) | ⭐⭐⭐ (OK) | 🟢 Low | ⭐⭐⭐ (Third-party) | ⭐⭐⭐ |

### Pros
1. **🌍 Largest Ecosystem**: 12M+ weekly npm downloads = most third-party libraries (3D rendering, charting, mapping)
2. **🎨 Component Reusability**: Build once, use across GPR/BIM/Hyperspectral modules
3. **📱 PWA Support**: Service workers + offline caching = field teams can work without internet
4. **🔧 Developer Availability**: 70%+ frontend developers know React = easy hiring
5. **🎭 IFC.js Compatibility**: IFC.js (BIM viewer) has React bindings = seamless integration

### Cons
1. **⚠️ Learning Curve**: Hooks, context, state management = 3-4 week onboarding for junior developers
2. **⚠️ Bundle Size**: React + dependencies = 150-200KB min.js (vs Svelte 50KB)
3. **⚠️ Boilerplate**: Need to choose state management (Redux/Zustand), routing (React Router), etc.

### Risk Mitigation
- **Learning Curve**: Create component library + coding standards early
- **Bundle Size**: Use code-splitting, lazy loading for 3D modules (only load when needed)
- **Boilerplate**: Use opinionated starter (Vite + React + TypeScript template)

### Cost Analysis
- **License**: Open-source (MIT License) = £0/year
- **Training**: 3-week team upskilling = £6,000 (8 developers × £750/week)
- **Development Tools**: VSCode extensions, ESLint, Prettier = £0 (free)
- **Net Cost**: £6,000 one-time (amortized over 3 years = £2,000/year)

### Verdict
✅ **React 18 + TypeScript is the optimal choice**: Ecosystem + IFC.js integration + PWA support justify learning curve.

**Risk Level**: 🟢 **LOW** (Mature framework backed by Meta, used in production by Facebook, Netflix, Airbnb)

---

## 3. Database: PostgreSQL + PostGIS + PGVector

### Decision
**Chosen**: PostgreSQL 15+ with PostGIS (spatial) and PGVector (vector embeddings) extensions

### Context
The platform requires:
- **Relational Data**: Users, projects, sites, quality assessments (ACID transactions)
- **Spatial Data**: Utility locations (lat/lon, polygons), LiDAR point clouds, BIM coordinates
- **Vector Embeddings**: Text embeddings for regulatory compliance search (RAG pipeline)
- **Time-Series Data**: GPR scans, hyperspectral measurements over time

### Alternatives Considered

| Database | Relational | Spatial (GIS) | Vector Embeddings | Time-Series | License Cost |
|----------|------------|---------------|-------------------|-------------|--------------|
| **PostgreSQL + PostGIS + PGVector** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | £0 (open-source) |
| MySQL + Extensions | ⭐⭐⭐⭐⭐ | ⭐⭐ (limited) | ❌ (none) | ⭐⭐⭐ | £0 (open-source) |
| MongoDB | ⭐⭐⭐ (document) | ⭐⭐⭐ (geospatial) | ⭐⭐⭐ (Atlas) | ⭐⭐⭐ | £0-£12K/year |
| Microsoft SQL Server | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ (spatial) | ❌ (none) | ⭐⭐⭐⭐ | £12K-£60K/year |
| Pinecone (vector only) | ❌ | ❌ | ⭐⭐⭐⭐⭐ | ❌ | £6K-£30K/year |

### Pros
1. **🎯 All-in-One Solution**: RDBMS + Spatial + Vector = no need for separate databases (reduces complexity)
2. **🗺️ PostGIS Excellence**: Industry standard for spatial data (used by Uber, Foursquare for location services)
3. **🔍 Full-Text Search**: Built-in FTS + tsvector for regulatory document search
4. **💰 Cost-Effective**: Open-source = £0 license cost (vs £12K-£60K/year for SQL Server)
5. **🔒 ACID Compliance**: Strong consistency for quality assessments (vs eventual consistency in MongoDB)

### Cons
1. **⚠️ PGVector Maturity**: PGVector released 2021 = less mature than Pinecone (2020) for vector search
2. **⚠️ Scaling Limits**: Single-node limit ~5TB before sharding required (vs MongoDB auto-sharding)
3. **⚠️ Vector Performance**: PGVector HNSW index = 10-20% slower than Pinecone (but <100ms still acceptable)

### Risk Mitigation
- **PGVector Maturity**: Use proven vector search patterns, fallback to Pinecone if <100ms latency not met
- **Scaling**: Partition tables by project_id (multi-tenancy), use read replicas
- **Vector Performance**: Pre-filter with WHERE clauses before vector search (hybrid approach)

### Cost Analysis (100-site deployment)
- **Self-Hosted**: AWS RDS PostgreSQL db.r6g.xlarge = £2,400/year
- **Managed PostGIS**: Azure Database for PostgreSQL = £3,600/year
- **Pinecone Alternative**: Pinecone Standard = £6,000/year (vector only, still need RDBMS)
- **Net Savings**: £2,400-£3,600/year vs multi-database architecture

### Verdict
✅ **PostgreSQL + PostGIS + PGVector is the optimal choice**: All-in-one solution reduces operational complexity and cost.

**Risk Level**: 🟢 **LOW** (PostgreSQL has 30+ years of production use, PostGIS is OGC-certified)

---

## 4. Object Storage: MinIO (S3-Compatible)

### Decision
**Chosen**: MinIO (self-hosted S3-compatible object storage)

### Context
The platform requires storage for:
- **GPR Files**: SEG-Y, GSSI DZT formats (100MB-500MB per scan, 1000+ scans)
- **Hyperspectral Data**: ENVI format (50MB-200MB per scan)
- **LiDAR Point Clouds**: LAS/LAZ files (500MB-5GB per scan)
- **BIM Models**: IFC files (10MB-100MB per model)
- **Photos**: 360° panoramas (5MB-20MB each, 100+ per site)

**Total Storage Needs**: 50GB-500GB per site × 100 sites = 5TB-50TB

### Alternatives Considered

| Solution | Cost (5TB/year) | Cost (50TB/year) | Vendor Lock-In | Self-Hosted Option | API Compatibility |
|----------|-----------------|------------------|----------------|-------------------|-------------------|
| **MinIO** | £0 (self-hosted) | £0 (self-hosted) | 🟢 None (open-source) | ✅ Yes | S3-compatible |
| AWS S3 | £1,200 | £12,000 | 🔴 High (AWS-specific) | ❌ No | S3 native |
| Azure Blob Storage | £1,080 | £10,800 | 🔴 High (Azure-specific) | ❌ No | Azure-specific |
| Google Cloud Storage | £1,200 | £12,000 | 🔴 High (GCP-specific) | ❌ No | GCS-specific |
| Backblaze B2 | £360 | £3,600 | 🟡 Medium | ❌ No | S3-compatible |

### Pros
1. **💰 Cost-Effective**: £0 license cost + self-hosted = save £1,200-£12,000/year vs AWS S3
2. **🔓 No Vendor Lock-In**: S3-compatible API = can migrate to AWS S3/Wasabi/Backblaze without code changes
3. **🏠 Data Sovereignty**: Self-hosted = full control over data location (important for HS2 UK data residency)
4. **⚡ Performance**: Local storage = lower latency (10ms vs 50-100ms to AWS S3)
5. **🔒 Security**: On-premise deployment = no internet exposure for sensitive BIM models

### Cons
1. **⚠️ Operational Burden**: Need to manage storage infrastructure (disk space, backups, redundancy)
2. **⚠️ Scalability Ceiling**: Self-hosted = manual scaling (vs auto-scaling in AWS S3)
3. **⚠️ Availability**: Single-node MinIO = 99.9% uptime (vs AWS S3 99.99%)

### Risk Mitigation
- **Operational Burden**: Use MinIO in distributed mode (4+ nodes) for automatic replication
- **Scalability**: Start with 10TB capacity, scale horizontally by adding nodes
- **Availability**: Use MinIO distributed mode + regular backups to AWS S3 Glacier (cold storage)

### Cost Analysis (50TB storage, 100 sites)
- **MinIO (self-hosted)**: £0 license + £3,600/year infrastructure (4× 15TB drives + servers)
- **AWS S3 Standard**: £12,000/year storage + £2,400/year data transfer = £14,400/year
- **Hybrid (MinIO + S3 Glacier backup)**: £3,600/year MinIO + £600/year S3 Glacier = £4,200/year
- **Net Savings**: £10,200/year vs AWS S3 alone

### Verdict
✅ **MinIO is the optimal choice**: Cost savings + data sovereignty + no vendor lock-in justify operational burden.

**Risk Level**: 🟢 **LOW** (MinIO used in production by Alibaba, Tencent, DigitalOcean)

---

## 5. Task Queue: Celery + Redis

### Decision
**Chosen**: Celery (Python task queue) + Redis (message broker)

### Context
The platform requires asynchronous processing for:
- **GPR Processing**: 5-10 minutes per file (CPU-intensive signal processing)
- **Hyperspectral Analysis**: 30-90 minutes per scan (deep learning inference)
- **LiDAR Processing**: 10-20 minutes per point cloud (3D mesh generation)
- **BIM Deviation Analysis**: 5-15 minutes per model (geometric comparison)
- **Report Generation**: 2-5 minutes (PDF compilation, 3D rendering)

**Workload**: 100 sites × 10 tasks/week = 1,000 tasks/week = 150 tasks/day

### Alternatives Considered

| Solution | Language | Performance | Monitoring | Cost (100 tasks/day) | Cloud-Native |
|----------|----------|-------------|------------|----------------------|--------------|
| **Celery + Redis** | Python | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ (Flower) | £0 (self-hosted) | 🟡 Partial |
| AWS SQS + Lambda | Any | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ (CloudWatch) | £360/year | ✅ Full |
| RabbitMQ + Workers | Any | ⭐⭐⭐⭐ | ⭐⭐⭐ (Management UI) | £0 (self-hosted) | ❌ No |
| BullMQ + Redis | Node.js | ⭐⭐⭐⭐ | ⭐⭐⭐ (Bull Board) | £0 (self-hosted) | ❌ No |
| Azure Service Bus | Any | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ (Azure Monitor) | £480/year | ✅ Full |

### Pros
1. **🐍 Python-Native**: Seamless integration with FastAPI backend = no language context switching
2. **📊 Rich Monitoring**: Flower dashboard for real-time task monitoring (running/failed/pending tasks)
3. **⚙️ Feature-Rich**: Task retries, rate limiting, priority queues, periodic tasks (cron-like scheduling)
4. **💰 Cost-Effective**: Open-source = £0 license cost (vs £360-£480/year for cloud services)
5. **🔧 Battle-Tested**: 10+ years production use (Instagram, Pinterest, Mozilla)

### Cons
1. **⚠️ Redis Dependency**: Redis as message broker = single point of failure (need Redis Sentinel/Cluster)
2. **⚠️ Worker Management**: Need to manage worker processes (scaling, crashes, restarts)
3. **⚠️ Cloud-Native Limitations**: Not as cloud-native as AWS SQS (no auto-scaling)

### Risk Mitigation
- **Redis SPOF**: Use Redis Sentinel (automatic failover) or AWS ElastiCache Redis (managed)
- **Worker Management**: Use systemd/supervisor for automatic worker restarts
- **Scaling**: Deploy workers as Docker containers in ECS/Kubernetes for auto-scaling

### Cost Analysis (100 tasks/day)
- **Celery + Redis (self-hosted)**: £0 license + £1,200/year infrastructure (Redis server)
- **AWS SQS + Lambda**: £360/year SQS + £0 Lambda (free tier) = £360/year
- **Hybrid (Celery + ElastiCache Redis)**: £0 Celery + £1,200/year ElastiCache = £1,200/year
- **Net Cost**: £0-£1,200/year (breakeven with cloud services)

### Verdict
✅ **Celery + Redis is the optimal choice**: Python-native integration + feature richness justify operational overhead.

**Risk Level**: 🟢 **LOW** (Mature project with 20K+ GitHub stars, production use at Instagram, Pinterest)

---

## 6. Hyperspectral Camera: Specim IQ

### Decision
**Chosen**: Specim IQ (204 spectral bands, 400-1000nm, handheld)

### Context
The platform requires non-destructive material quality verification:
- **Concrete Strength Prediction**: Predict compressive strength (40-60 MPa) without core sampling
- **Defect Detection**: Identify voids, cracks, delamination, moisture ingress
- **Material Compliance**: Verify aggregate composition, cement hydration, curing quality
- **Cost Savings**: Reduce core sampling by 60-80% (£500-£2,000 per core sample)

### Alternatives Considered

| Camera Model | Spectral Bands | Wavelength Range | Form Factor | Price (Purchase) | Price (Lease/month) |
|--------------|----------------|------------------|-------------|------------------|---------------------|
| **Specim IQ** | 204 bands | 400-1000nm | Handheld | £35,000 | £1,200 |
| Resonon Pika L | 281 bands | 400-1000nm | Drone/tripod | £45,000 | £1,500 |
| Headwall Nano-Hyperspec | 270 bands | 400-1000nm | Drone/tripod | £50,000 | £1,800 |
| Cubert UHD-185 | 138 bands | 450-950nm | Handheld | £30,000 | £1,000 |
| XIMEA xiSpec | 200 bands | 470-900nm | Camera module | £15,000 | £500 |

### Pros
1. **📱 Handheld Form Factor**: Field technicians can use on-site without drones/tripods = 2-3x faster data capture
2. **🎯 Optimal Spectral Range**: 400-1000nm covers cement hydration (500-600nm), moisture (700-850nm), aggregate (900-1000nm)
3. **🔋 Battery-Powered**: 4-hour battery = full day on-site without recharging
4. **📊 Integrated Processing**: Built-in GPU for real-time spectral analysis = instant feedback to field teams
5. **🏆 Industry Standard**: Used by construction firms (Skanska, Balfour Beatty) for material QA

### Cons
1. **⚠️ High Cost**: £35,000 purchase or £1,200/month lease = significant upfront investment
2. **⚠️ Limited Wavelength Range**: 400-1000nm only (vs 400-2500nm for Resonon Pika L) = can't detect hydrocarbon contamination
3. **⚠️ Training Required**: 2-week training for field technicians to operate correctly

### Risk Mitigation
- **High Cost**: Lease 4 cameras (£4,800/month) instead of purchasing (£140,000) = lower upfront risk
- **Wavelength Range**: 400-1000nm sufficient for concrete/asphalt (90% of HS2 materials)
- **Training**: Specim provides 2-day on-site training + online resources

### Cost Analysis (4 cameras for 100 sites)
- **Purchase**: £140,000 upfront + £2,000/year maintenance = £142,000 Year 1, £2,000/year thereafter
- **Lease**: £4,800/month × 12 months = £57,600/year (no maintenance fees)
- **ROI**: Save £50K-£200K/year on core sampling (£500-£2,000 per sample, 100-400 samples reduced)
- **Net Savings**: £0-£142,000/year (breakeven at 70-100 core samples saved)

### Verdict
✅ **Specim IQ is the optimal choice**: Handheld form factor + optimal spectral range + industry adoption justify cost.

**Risk Level**: 🟡 **MEDIUM** (High cost + training required, but proven ROI in construction QA)

---

## 7. LiDAR Scanner: Leica RTC360

### Decision
**Chosen**: Leica RTC360 (2M points/sec, ±1mm accuracy, 360° field of view)

### Context
The platform requires 3D reality capture for:
- **As-Built Documentation**: Capture exact site conditions (dimensions, elevations, geometry)
- **BIM Deviation Analysis**: Compare 3D point cloud to BIM model = detect construction errors
- **Volume Measurement**: Calculate earthwork volumes, concrete pours, material stockpiles
- **Progress Tracking**: Weekly scans to measure construction progress (m² completed)

### Alternatives Considered

| Scanner Model | Points/Sec | Accuracy | Range | Price (Purchase) | Price (Lease/month) |
|---------------|------------|----------|-------|------------------|---------------------|
| **Leica RTC360** | 2M | ±1mm @ 10m | 130m | £45,000 | £1,500 |
| Faro Focus Premium | 2M | ±1mm @ 10m | 150m | £40,000 | £1,300 |
| Trimble X12 | 1M | ±2mm @ 40m | 160m | £50,000 | £1,700 |
| GeoSLAM ZEB Horizon | 300K | ±6mm @ 30m | 100m | £30,000 | £1,000 |
| iPhone 15 Pro (LiDAR) | 30K | ±1cm @ 5m | 5m | £1,200 | N/A |

### Pros
1. **⚡ Fast Scanning**: 2M points/sec = complete site scan in 10-15 minutes (vs 30-45 minutes for Trimble X12)
2. **🎯 High Accuracy**: ±1mm @ 10m = suitable for structural tolerance checks (±5mm HS2 spec)
3. **📸 Integrated Imaging**: 36MP HDR camera = automatic colorization of point clouds
4. **🔋 Long Battery**: 6-hour battery = 15-20 scans per day without recharging
5. **🏆 HS2 Standard**: HS2 specifies Leica scanners = guaranteed compatibility with HS2 workflows

### Cons
1. **⚠️ High Cost**: £45,000 purchase or £1,500/month lease = significant investment
2. **⚠️ Range Limitations**: 130m range = need multiple scan positions for large sites (vs Trimble X12 160m)
3. **⚠️ File Size**: 2M points/sec × 10 minutes = 1.2B points = 5-10GB per scan

### Risk Mitigation
- **High Cost**: Lease 2 scanners (£3,000/month) instead of purchasing (£90,000)
- **Range**: Use automated registration (scan targets) to combine multiple scans seamlessly
- **File Size**: Use LAZ compression (80% size reduction) + cloud processing for large scans

### Cost Analysis (2 scanners for 100 sites)
- **Purchase**: £90,000 upfront + £3,000/year maintenance = £93,000 Year 1, £3,000/year thereafter
- **Lease**: £3,000/month × 12 months = £36,000/year (no maintenance fees)
- **ROI**: Save £20K-£50K/year on manual surveying (vs traditional total station + CAD)
- **Net Savings**: £0-£17,000/year (breakeven at 50-80% manual survey reduction)

### Verdict
✅ **Leica RTC360 is the optimal choice**: Fast scanning + HS2 standard + integrated imaging justify cost.

**Risk Level**: 🟢 **LOW** (Industry standard with proven ROI in construction surveying)

---

## 8. BIM Viewer: IFC.js (Web-Based)

### Decision
**Chosen**: IFC.js (open-source, web-based BIM viewer)

### Context
The platform requires 3D BIM visualization for:
- **Model Review**: View IFC models directly in browser (no desktop software required)
- **Deviation Highlighting**: Overlay LiDAR point cloud on BIM to show construction errors
- **Mobile Access**: Field teams view BIM on tablets/phones (PWA)
- **Integration**: Embed BIM viewer in React dashboard (unified interface)

### Alternatives Considered

| Solution | Deployment | IFC Support | Custom Features | License Cost | Browser Support |
|----------|------------|-------------|-----------------|--------------|-----------------|
| **IFC.js** | Web (self-hosted) | ⭐⭐⭐⭐⭐ (IFC 2×3, 4) | ⭐⭐⭐⭐⭐ (open-source) | £0 | Chrome, Firefox, Safari |
| Autodesk Forge Viewer | Cloud (Autodesk) | ⭐⭐⭐⭐ (IFC + proprietary) | ⭐⭐⭐ (limited API) | £12K-£60K/year | Chrome, Edge |
| BIM360 | Cloud (Autodesk) | ⭐⭐⭐⭐ (IFC + RVT) | ⭐⭐ (closed ecosystem) | £24K-£120K/year | Chrome, Edge |
| Solibri Viewer | Desktop | ⭐⭐⭐⭐⭐ (IFC 2×3, 4) | ⭐⭐⭐⭐ (QA features) | £6K-£30K/year | Windows only |
| BIMcollab Zoom | Web (BIMcollab) | ⭐⭐⭐⭐ (IFC 2×3, 4) | ⭐⭐⭐ (collaboration) | £3K-£15K/year | Chrome, Firefox |

### Pros
1. **💰 Cost-Effective**: Open-source (MIT License) = £0 license cost (vs £3K-£120K/year for alternatives)
2. **🌐 Web-Native**: JavaScript library = runs in browser, no desktop software required
3. **📱 Mobile-Friendly**: Works on tablets/phones = field teams access BIM on-site
4. **🔧 Customizable**: Full API access = can add custom features (deviation heatmaps, quality overlays)
5. **⚛️ React Integration**: Official React bindings = seamless integration with React dashboard

### Cons
1. **⚠️ Limited Features**: Basic viewer only (no clash detection, quantity takeoff, etc.)
2. **⚠️ Performance**: Large IFC models (>100MB) = slow loading (10-20 seconds)
3. **⚠️ Support**: Community support only (no enterprise SLA)

### Risk Mitigation
- **Limited Features**: Focus on core use case (visualization + deviation overlay), not full BIM authoring
- **Performance**: Use IFC model optimization (remove irrelevant objects, simplify geometry)
- **Support**: Use IFC.js commercial support option (€5,000/year for priority support)

### Cost Analysis (100 sites)
- **IFC.js (open-source)**: £0 license + £0 infrastructure (self-hosted)
- **Autodesk Forge Viewer**: £12,000/year (50 users)
- **BIM360**: £24,000/year (50 users + 100GB storage)
- **Net Savings**: £12,000-£24,000/year vs commercial alternatives

### Verdict
✅ **IFC.js is the optimal choice**: Cost savings + customizability + React integration justify limited enterprise support.

**Risk Level**: 🟢 **LOW** (Active development, used in production by construction firms like Bouygues)

---

## 9. Cloud Platform: Microsoft Azure

### Decision
**Chosen**: Microsoft Azure (UK South region for data residency)

### Context
The platform requires cloud infrastructure for:
- **HS2 Alignment**: HS2 uses Microsoft Azure = seamless integration with HS2 systems
- **UK Data Residency**: GDPR compliance + HS2 security requirements
- **GPU Compute**: Machine learning inference for hyperspectral analysis
- **Global Availability**: 99.99% SLA for dashboard uptime

### Alternatives Considered

| Cloud Provider | HS2 Alignment | UK Regions | GPU Availability | Cost (100 sites) | Enterprise Support |
|----------------|---------------|------------|------------------|------------------|-------------------|
| **Microsoft Azure** | ⭐⭐⭐⭐⭐ (HS2 standard) | UK South, UK West | ⭐⭐⭐⭐⭐ (NCv3, NDv2) | £12,000/year | ⭐⭐⭐⭐⭐ |
| AWS | ⭐⭐ (limited HS2 use) | EU (London) | ⭐⭐⭐⭐⭐ (P3, G4dn) | £10,800/year | ⭐⭐⭐⭐⭐ |
| Google Cloud Platform | ⭐ (no HS2 use) | EU (London) | ⭐⭐⭐⭐ (T4, V100) | £12,000/year | ⭐⭐⭐⭐ |
| Self-Hosted (on-prem) | ⭐⭐⭐ (secure) | UK (physical) | ⭐⭐ (limited GPUs) | £24,000/year | ⭐ (DIY) |

### Pros
1. **🏗️ HS2 Standard**: HS2 mandates Azure for contractors = guaranteed compatibility + integration
2. **🇬🇧 UK Data Residency**: UK South region = GDPR compliance + HS2 security requirements met
3. **🤖 Azure ML**: Managed ML services = easy deployment of hyperspectral models
4. **🔐 Enterprise Security**: Azure Active Directory = SSO integration with HS2 systems
5. **💼 Enterprise Support**: 24/7 support + 99.99% SLA = suitable for mission-critical HS2 operations

### Cons
1. **⚠️ Higher Cost**: 10-15% more expensive than AWS for equivalent resources
2. **⚠️ Vendor Lock-In**: Azure-specific services (Azure ML, Cosmos DB) = migration risk
3. **⚠️ Learning Curve**: Team needs to learn Azure CLI/Portal (vs AWS familiarity)

### Risk Mitigation
- **Higher Cost**: Use Reserved Instances (30-40% discount for 1-year commitment)
- **Vendor Lock-In**: Use open-source tools (Kubernetes, PostgreSQL) where possible
- **Learning Curve**: Provide Azure training (£2,000 for 8 developers)

### Cost Analysis (100 sites, 50TB storage, 10 compute instances)
- **Azure**: £12,000/year (compute + storage + networking)
- **AWS**: £10,800/year (10% cheaper)
- **GCP**: £12,000/year (equivalent)
- **Self-Hosted**: £24,000/year (servers + electricity + maintenance)
- **Net Cost**: £12,000/year (HS2 alignment worth 10% premium vs AWS)

### Verdict
✅ **Microsoft Azure is the optimal choice**: HS2 alignment + UK data residency justify 10% cost premium vs AWS.

**Risk Level**: 🟢 **LOW** (Enterprise-grade platform with proven HS2 integration)

---

## 10. Containerization: Docker + Docker Compose

### Decision
**Chosen**: Docker + Docker Compose (development), Docker + ECS/Kubernetes (production)

### Context
The platform requires:
- **Dev/Prod Parity**: Developers run same environment locally as production
- **Microservices**: Independent services (backend, frontend, Celery workers, Redis, PostgreSQL, MinIO)
- **Scalability**: Horizontal scaling of Celery workers during peak load
- **Simplified Deployment**: Single-command deployment for local development

### Alternatives Considered

| Solution | Dev Experience | Production Scaling | Learning Curve | Operational Complexity | Cost |
|----------|----------------|-------------------|----------------|------------------------|------|
| **Docker + Docker Compose** | ⭐⭐⭐⭐⭐ (simple) | ⭐⭐⭐ (manual) | 🟢 Low | 🟢 Low | £0 |
| Kubernetes (K8s) | ⭐⭐ (complex) | ⭐⭐⭐⭐⭐ (auto-scale) | 🔴 High | 🔴 High | £0-£6K/year |
| Docker Swarm | ⭐⭐⭐⭐ (moderate) | ⭐⭐⭐⭐ (good) | 🟡 Medium | 🟡 Medium | £0 |
| Nomad (HashiCorp) | ⭐⭐⭐ (moderate) | ⭐⭐⭐⭐ (good) | 🟡 Medium | 🟡 Medium | £0 |
| Bare Metal (no containers) | ⭐⭐ (slow) | ⭐⭐ (manual) | 🟢 Low | 🔴 High | £0 |

### Pros
1. **🚀 Fast Development**: Docker Compose = spin up full stack in 30 seconds (`docker-compose up`)
2. **🔄 Dev/Prod Parity**: Same Dockerfile in dev and production = "works on my machine" eliminated
3. **📦 Microservices-Ready**: Each service in separate container = independent scaling
4. **🌍 Portable**: Runs on any OS (macOS, Windows, Linux) = team flexibility
5. **📚 Industry Standard**: 70%+ of development teams use Docker

### Cons
1. **⚠️ Production Scaling**: Docker Compose not suitable for production auto-scaling (need K8s/ECS)
2. **⚠️ Networking Complexity**: Docker networks = learning curve for inter-service communication
3. **⚠️ Storage Persistence**: Volumes need careful management to avoid data loss

### Risk Mitigation
- **Production Scaling**: Use Docker Compose for dev, migrate to AWS ECS/Azure Container Instances for production
- **Networking**: Use docker-compose.yml service names for DNS resolution (automatic)
- **Storage**: Use named volumes for persistence, backup volumes to S3

### Cost Analysis
- **Docker + Docker Compose**: £0 (open-source)
- **Kubernetes (managed)**: £0 (open-source) + £6,000/year for AWS EKS/AKS cluster
- **Bare Metal**: £0 (no containers) + £12,000/year operational overhead
- **Net Savings**: £6,000-£12,000/year vs alternatives

### Verdict
✅ **Docker + Docker Compose is the optimal choice**: Developer experience + industry standard justify production scaling limitations.

**Risk Level**: 🟢 **LOW** (Mature technology with 10+ years of production use)

---

## 11. Authentication: OAuth 2.0 + JWT

### Decision
**Chosen**: OAuth 2.0 (authorization) + JWT (JSON Web Tokens) for stateless authentication

### Context
The platform requires:
- **Multi-Tenant Security**: Project-level data isolation (HS2 cannot see Crossrail data)
- **SSO Integration**: HS2 uses Azure AD = need SAML/OAuth integration
- **Mobile Support**: Field teams use tablets/phones = need token-based auth (not cookies)
- **API Access**: Third-party integrations (BIM 360, Procore) = need OAuth 2.0 client credentials

### Alternatives Considered

| Solution | Multi-Tenancy | SSO Support | Mobile-Friendly | Cost (100 users) | Self-Hosted |
|----------|---------------|-------------|-----------------|------------------|-------------|
| **OAuth 2.0 + JWT** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ (SAML bridge) | ⭐⭐⭐⭐⭐ (tokens) | £0 (self-hosted) | ✅ Yes |
| Auth0 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ (built-in) | ⭐⭐⭐⭐⭐ | £6,000/year | ❌ No |
| Okta | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ (built-in) | ⭐⭐⭐⭐⭐ | £12,000/year | ❌ No |
| Azure AD B2C | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ (Azure AD) | ⭐⭐⭐⭐⭐ | £3,000/year | ❌ No (Azure) |
| Session Cookies | ⭐⭐⭐ | ⭐⭐ (custom SAML) | ⭐⭐ (not ideal) | £0 (self-hosted) | ✅ Yes |

### Pros
1. **📱 Mobile-Friendly**: JWT tokens = stateless auth, works on mobile apps, tablets, IoT devices
2. **🔐 SSO Support**: OAuth 2.0 = integrate with Azure AD, Google Workspace, Okta
3. **🌍 Scalable**: Stateless tokens = no server-side session storage = horizontal scaling
4. **💰 Cost-Effective**: Open-source libraries (Authlib, PyJWT) = £0 license cost
5. **🔒 Secure**: Industry standard (used by Google, Facebook, GitHub for API access)

### Cons
1. **⚠️ Token Revocation**: JWTs cannot be revoked (need blacklist or short expiry times)
2. **⚠️ Implementation Complexity**: Need to handle token refresh, expiry, storage securely
3. **⚠️ Security Risks**: XSS attacks can steal tokens (need httpOnly cookies + CSP headers)

### Risk Mitigation
- **Token Revocation**: Use 15-minute access tokens + 7-day refresh tokens (blacklist refresh tokens on logout)
- **Implementation**: Use battle-tested libraries (Authlib for OAuth 2.0, PyJWT for tokens)
- **Security**: Store tokens in httpOnly cookies (not localStorage) + implement CSRF protection

### Cost Analysis (100 users)
- **Self-Hosted OAuth 2.0 + JWT**: £0 license + £0 infrastructure (runs on existing backend)
- **Auth0**: £6,000/year (Essentials plan)
- **Okta**: £12,000/year (Workforce Identity)
- **Azure AD B2C**: £3,000/year (50,000 MAUs)
- **Net Savings**: £3,000-£12,000/year vs managed alternatives

### Verdict
✅ **OAuth 2.0 + JWT is the optimal choice**: Cost savings + mobile support + SSO integration justify implementation complexity.

**Risk Level**: 🟢 **LOW** (Industry standard with proven security when implemented correctly)

---

## 12. Vector Database: PGVector (PostgreSQL Extension)

### Decision
**Chosen**: PGVector (PostgreSQL extension for vector embeddings)

### Context
The platform requires vector search for:
- **Regulatory Compliance**: Search PAS 128, CDM 2015, HS2 specifications (semantic search)
- **RAG Pipeline**: Retrieve relevant context for GPT-4o report generation
- **Similar Defects**: Find similar hyperspectral anomalies across sites
- **Historical Incidents**: Search incident reports for risk assessment

**Embedding Size**: 1536 dimensions (OpenAI text-embedding-3-small)
**Dataset Size**: 10,000 document chunks + 50,000 defect signatures = 60,000 vectors

### Alternatives Considered

| Solution | Cost (60K vectors) | Query Latency | Integration | Self-Hosted | ACID Transactions |
|----------|-------------------|---------------|-------------|-------------|-------------------|
| **PGVector (PostgreSQL)** | £0 (bundled) | <100ms | ⭐⭐⭐⭐⭐ (SQL) | ✅ Yes | ✅ Yes |
| Pinecone | £6,000/year | <50ms | ⭐⭐⭐ (HTTP API) | ❌ No | ❌ No |
| Weaviate | £0-£3,000/year | <80ms | ⭐⭐⭐⭐ (GraphQL) | ✅ Yes | ❌ No |
| Chroma | £0 (open-source) | <100ms | ⭐⭐⭐⭐ (Python) | ✅ Yes | ❌ No |
| Milvus | £0 (open-source) | <60ms | ⭐⭐⭐ (gRPC) | ✅ Yes | ❌ No |

### Pros
1. **💰 Cost-Effective**: PostgreSQL extension = £0 cost (vs £6,000/year for Pinecone)
2. **🔗 Unified Database**: Vectors + relational data in one database = simplify architecture
3. **🔍 Hybrid Search**: Combine vector search + SQL filters (e.g., "similar defects in last 6 months")
4. **🔒 ACID Transactions**: Update vectors + relational data atomically (data consistency)
5. **🏠 Self-Hosted**: Full control over data (important for HS2 security)

### Cons
1. **⚠️ Performance**: 10-20% slower than Pinecone for pure vector search (<100ms vs <50ms)
2. **⚠️ Scaling Limits**: PGVector optimized for <1M vectors (vs Pinecone billions)
3. **⚠️ Indexing Speed**: HNSW index build = 5-10 minutes for 60K vectors (vs Pinecone <1 minute)

### Risk Mitigation
- **Performance**: Pre-filter with SQL WHERE clauses before vector search (reduce search space)
- **Scaling**: 60K vectors well within PGVector limits, can shard by project_id if needed
- **Indexing Speed**: Build index offline during low-traffic periods (overnight)

### Cost Analysis (60,000 vectors)
- **PGVector (PostgreSQL)**: £0 license + £0 infrastructure (bundled with PostgreSQL)
- **Pinecone Standard**: £6,000/year (100K vectors)
- **Weaviate (managed)**: £3,000/year (self-hosted)
- **Net Savings**: £3,000-£6,000/year vs alternatives

### Verdict
✅ **PGVector is the optimal choice**: Cost savings + unified database + hybrid search justify 10-20% performance trade-off.

**Risk Level**: 🟢 **LOW** (Actively developed, production use at Supabase, Neon)

---

## 13. LLM API: OpenAI GPT-4o

### Decision
**Chosen**: OpenAI GPT-4o (GPT-4 Omni, multimodal)

### Context
The platform requires LLM for:
- **Report Generation**: Compile progress reports, quality assessments, deviation summaries
- **Regulatory Compliance**: Answer questions about PAS 128, CDM 2015, HS2 specifications
- **Defect Explanation**: Generate human-readable explanations for hyperspectral anomalies
- **Risk Summaries**: Summarize construction risks based on historical incidents

**Usage**: 100 sites × 10 reports/month = 1,000 reports/month = 12,000 reports/year
**Tokens/Report**: 5,000 tokens input + 2,000 tokens output = 7,000 tokens/report

### Alternatives Considered

| LLM | Cost (12K reports/year) | Multimodal | Context Window | Latency | Fine-Tuning |
|-----|------------------------|------------|----------------|---------|-------------|
| **OpenAI GPT-4o** | £6,000/year | ✅ (text+image) | 128K tokens | <2s | ✅ (£500-£2K) |
| Anthropic Claude 3 Opus | £7,200/year | ✅ (text+image) | 200K tokens | <3s | ❌ (not public) |
| Meta Llama 3 (self-hosted) | £1,200/year | ❌ (text only) | 8K tokens | <1s | ✅ (free) |
| Google Gemini Pro | £4,800/year | ✅ (text+image+video) | 1M tokens | <3s | ✅ (free) |
| Azure OpenAI Service | £6,000/year | ✅ (text+image) | 128K tokens | <2s | ✅ (£500-£2K) |

### Pros
1. **🖼️ Multimodal**: Process text + images = analyze hyperspectral scans, photos, diagrams in one model
2. **📚 Large Context**: 128K tokens = can process entire HS2 specification documents in one prompt
3. **⚡ Low Latency**: <2 seconds for report generation = real-time user experience
4. **🔧 Fine-Tuning**: Can fine-tune on HS2-specific reports for better quality
5. **🏆 Industry Leader**: 70%+ LLM API market share = best performance, most reliable

### Cons
1. **⚠️ Cost**: £6,000/year for 12K reports = higher than self-hosted Llama 3
2. **⚠️ Data Privacy**: Data sent to OpenAI servers = potential GDPR/HS2 security concerns
3. **⚠️ Vendor Lock-In**: OpenAI-specific prompt engineering = migration risk

### Risk Mitigation
- **Cost**: Use gpt-4o-mini for simple tasks (80% cheaper) = reduce average cost to £2,400/year
- **Data Privacy**: Use Azure OpenAI Service (data stays in Azure UK South) = GDPR compliant
- **Vendor Lock-In**: Design prompts to be model-agnostic (fallback to Claude 3 if needed)

### Cost Analysis (12,000 reports/year, 7,000 tokens/report = 84M tokens/year)
- **GPT-4o**: £6,000/year (gpt-4o) or £1,200/year (gpt-4o-mini for 80% of reports)
- **Claude 3 Opus**: £7,200/year
- **Llama 3 (self-hosted)**: £1,200/year (GPU server) + £6,000 engineering time = £7,200/year
- **Net Cost**: £1,200-£6,000/year (GPT-4o-mini most cost-effective)

### Verdict
✅ **OpenAI GPT-4o (with gpt-4o-mini fallback) is the optimal choice**: Multimodal + low latency + fine-tuning justify cost.

**Risk Level**: 🟡 **MEDIUM** (Data privacy concerns mitigated by Azure OpenAI Service)

---

## 14. Monitoring: Grafana + Prometheus

### Decision
**Chosen**: Grafana (visualization) + Prometheus (metrics collection)

### Context
The platform requires monitoring for:
- **System Health**: CPU, memory, disk usage on backend servers
- **Application Metrics**: API latency, Celery task queue length, database query time
- **Business Metrics**: Reports generated/day, scans processed/day, user logins
- **Alerts**: PagerDuty integration for critical failures (database down, API 500 errors)

### Alternatives Considered

| Solution | Cost (10 servers) | Visualization | Alerting | Cloud-Native | Self-Hosted |
|----------|-------------------|---------------|----------|--------------|-------------|
| **Grafana + Prometheus** | £0 (open-source) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ (K8s native) | ✅ Yes |
| Datadog | £6,000/year | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | ❌ No |
| New Relic | £7,200/year | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | ❌ No |
| Dynatrace | £12,000/year | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | ❌ No |
| CloudWatch (AWS) | £1,200/year | ⭐⭐⭐ | ⭐⭐⭐ | ✅ (AWS only) | ❌ No (AWS) |

### Pros
1. **💰 Cost-Effective**: Open-source = £0 license cost (vs £6K-£12K/year for commercial)
2. **📊 Rich Visualization**: Grafana dashboards = customizable charts, graphs, heatmaps
3. **🔔 Powerful Alerting**: Prometheus Alertmanager = flexible alerting rules (CPU >80%, API latency >200ms)
4. **🔌 Extensible**: 150+ Grafana data sources (Prometheus, PostgreSQL, Elasticsearch)
5. **🏆 Industry Standard**: Used by Uber, DigitalOcean, GitLab for monitoring

### Cons
1. **⚠️ Operational Burden**: Need to manage Prometheus server, Grafana server, exporters
2. **⚠️ Long-Term Storage**: Prometheus designed for 15-day retention (need Thanos/Cortex for long-term)
3. **⚠️ Learning Curve**: PromQL query language = 1-2 week learning curve

### Risk Mitigation
- **Operational Burden**: Use Grafana Cloud (managed) for production (£600/year for 10 servers)
- **Long-Term Storage**: Use Prometheus + Thanos for 90-day retention (adequate for HS2)
- **Learning Curve**: Provide PromQL training + create dashboard templates

### Cost Analysis (10 servers)
- **Self-Hosted Grafana + Prometheus**: £0 license + £600/year infrastructure
- **Grafana Cloud (managed)**: £600/year (10 servers, 15-day retention)
- **Datadog**: £6,000/year (10 servers)
- **New Relic**: £7,200/year (10 servers)
- **Net Savings**: £5,400-£6,600/year vs commercial alternatives

### Verdict
✅ **Grafana + Prometheus is the optimal choice**: Cost savings + flexibility + industry standard justify operational burden.

**Risk Level**: 🟢 **LOW** (Mature projects with production use at scale)

---

## 15. CI/CD: GitHub Actions

### Decision
**Chosen**: GitHub Actions (CI/CD automation)

### Context
The platform requires automated workflows for:
- **Testing**: Run pytest, ESLint, type checking on every commit
- **Building**: Build Docker images for backend, frontend, Celery workers
- **Deployment**: Deploy to Azure Container Instances on merge to main branch
- **Security Scanning**: Snyk/Dependabot for vulnerability detection

### Alternatives Considered

| Solution | Cost (private repo) | GitHub Integration | Docker Support | Self-Hosted | Learning Curve |
|----------|---------------------|-------------------|----------------|-------------|----------------|
| **GitHub Actions** | £0 (2,000 min/month free) | ⭐⭐⭐⭐⭐ (native) | ⭐⭐⭐⭐⭐ | ✅ (runners) | 🟢 Low |
| GitLab CI | £0 (400 min/month free) | ⭐⭐⭐ (needs import) | ⭐⭐⭐⭐⭐ | ✅ | 🟢 Low |
| Jenkins | £0 (open-source) | ⭐⭐⭐ (plugin) | ⭐⭐⭐⭐ | ✅ (required) | 🔴 High |
| CircleCI | £1,200/year | ⭐⭐⭐⭐ (good) | ⭐⭐⭐⭐⭐ | ❌ No | 🟢 Low |
| Azure Pipelines | £0 (1,800 min/month free) | ⭐⭐⭐ (needs import) | ⭐⭐⭐⭐⭐ | ✅ | 🟡 Medium |

### Pros
1. **💰 Cost-Effective**: 2,000 minutes/month free (vs £1,200/year for CircleCI)
2. **🔗 GitHub Integration**: Native = no import/sync needed, auto-detects PRs
3. **📦 Docker Support**: Pre-built Docker actions = build/push images in 5 lines of YAML
4. **🔐 Secrets Management**: GitHub Secrets for API keys, credentials
5. **🌍 Marketplace**: 10,000+ pre-built actions (Snyk, Codecov, Slack notifications)

### Cons
1. **⚠️ Vendor Lock-In**: GitHub-specific YAML syntax = migration to GitLab CI requires rewrite
2. **⚠️ Limited Free Tier**: 2,000 minutes/month = ~10 builds/day (may need paid tier)
3. **⚠️ No Self-Hosted Dashboard**: GitHub Actions UI only (vs Jenkins' custom UI)

### Risk Mitigation
- **Vendor Lock-In**: Use standardized CI/CD patterns (test → build → deploy) for easier migration
- **Free Tier**: Optimize workflows (cache dependencies, parallel jobs) to stay under 2,000 min/month
- **Dashboard**: Use GitHub Projects + GitHub Actions logs for visibility

### Cost Analysis (100 builds/month)
- **GitHub Actions**: £0 (within free tier) or £360/year (3,000 min/month paid tier)
- **GitLab CI**: £0 (within free tier) or £480/year (10,000 min/month)
- **Jenkins (self-hosted)**: £0 license + £3,600/year infrastructure + £6,000/year maintenance = £9,600/year
- **CircleCI**: £1,200/year (Performance plan)
- **Net Savings**: £840-£9,240/year vs alternatives

### Verdict
✅ **GitHub Actions is the optimal choice**: Cost + GitHub integration + Docker support justify vendor lock-in risk.

**Risk Level**: 🟢 **LOW** (Widely adopted with proven reliability)

---

## 16. Documentation: Markdown + MkDocs

### Decision
**Chosen**: Markdown + MkDocs (static site generator)

### Context
The platform requires documentation for:
- **Technical Docs**: API reference, database schema, deployment guides
- **User Guides**: Field technician training, dashboard tutorials
- **Architecture Docs**: Technology decisions (this document!), system diagrams
- **Compliance Docs**: PAS 128 validation, HS2 security audits

### Alternatives Considered

| Solution | Cost (5 contributors) | Markdown Support | Version Control | Search | Hosting |
|----------|----------------------|------------------|-----------------|--------|---------|
| **Markdown + MkDocs** | £0 | ⭐⭐⭐⭐⭐ (native) | ✅ (Git) | ⭐⭐⭐⭐ (lunr.js) | £0 (GitHub Pages) |
| Confluence | £3,000/year | ⭐⭐⭐ (limited) | ⭐⭐ (versions) | ⭐⭐⭐⭐⭐ | ☁️ (Atlassian) |
| Notion | £600/year | ⭐⭐⭐⭐ (good) | ⭐⭐⭐ (history) | ⭐⭐⭐⭐ | ☁️ (Notion) |
| GitBook | £1,200/year | ⭐⭐⭐⭐⭐ (native) | ✅ (Git) | ⭐⭐⭐⭐⭐ | ☁️ (GitBook) |
| Read the Docs | £0 (open-source) | ⭐⭐⭐⭐⭐ (Sphinx/MkDocs) | ✅ (Git) | ⭐⭐⭐⭐ | £0 (RTD) |

### Pros
1. **💰 Cost-Effective**: Open-source + GitHub Pages hosting = £0
2. **📝 Developer-Friendly**: Markdown = easy to write, version control with Git
3. **🔍 Search**: Built-in search (lunr.js) = fast client-side search
4. **🎨 Customizable**: MkDocs themes (Material for MkDocs) = professional look
5. **🔗 Integration**: Embed diagrams (Mermaid), code snippets (syntax highlighting), API specs (OpenAPI)

### Cons
1. **⚠️ No Real-Time Collaboration**: Markdown = file-based, no Google Docs-style collaboration
2. **⚠️ Limited Rich Media**: No drag-and-drop images, videos (need manual uploads)
3. **⚠️ Learning Curve**: Non-technical users need Markdown training

### Risk Mitigation
- **Collaboration**: Use GitHub PRs for review, Markdown preview in VSCode
- **Rich Media**: Use Markdown image syntax + screenshot tools (Snagit, CloudApp)
- **Learning Curve**: Provide Markdown cheat sheet + VSCode extensions

### Cost Analysis
- **Markdown + MkDocs**: £0 license + £0 hosting (GitHub Pages)
- **Confluence**: £3,000/year (5 users, Standard plan)
- **Notion**: £600/year (5 users, Team plan)
- **GitBook**: £1,200/year (5 users, Team plan)
- **Net Savings**: £600-£3,000/year vs alternatives

### Verdict
✅ **Markdown + MkDocs is the optimal choice**: Cost + version control + developer-friendly justify lack of real-time collaboration.

**Risk Level**: 🟢 **LOW** (Industry standard for open-source project documentation)

---

## Summary: Why These Technologies?

### Core Principles Across All Decisions

1. **💰 Cost-Effectiveness**:
   - Total Annual Cost: £6,300-£115,800 (conservative to full purchase)
   - Customer ROI: £16M-£73M/year savings (100-site HS2 project) = 138-11,571x return
   - Open-source first: 11 of 16 technologies are open-source = reduce licensing costs

2. **🔓 Avoid Vendor Lock-In**:
   - S3-compatible storage (MinIO) = migrate to AWS S3/Wasabi without code changes
   - Standard protocols: OAuth 2.0, OpenAPI, IFC, PostgreSQL wire protocol
   - Exception: Azure for HS2 alignment (strategic decision)

3. **🤖 AI/ML-First Architecture**:
   - Python throughout backend = seamless NumPy/TensorFlow/PyTorch integration
   - GPU-optimized cloud (Azure NCv3) = fast hyperspectral inference
   - Vector database (PGVector) = RAG pipeline for regulatory compliance

4. **🏗️ HS2 Alignment**:
   - Microsoft Azure = HS2 standard cloud platform
   - Leica LiDAR = HS2 specified scanner
   - PAS 128 compliance built-in = UK construction industry standard

5. **⚖️ Risk Management**:
   - 🟢 13 Low-Risk technologies (mature, proven, industry standard)
   - 🟡 3 Medium-Risk technologies (high cost or newer tech, but proven ROI)
   - 🔴 0 High-Risk technologies (no experimental/unproven tech)

### Total Cost of Ownership (3-Year Projection)

| Category | Year 1 | Year 2 | Year 3 | 3-Year Total |
|----------|--------|--------|--------|--------------|
| **Cameras** (lease 4× Specim IQ) | £57,600 | £57,600 | £57,600 | £172,800 |
| **LiDAR** (lease 2× Leica RTC360) | £36,000 | £36,000 | £36,000 | £108,000 |
| **Cloud Infrastructure** (Azure) | £12,000 | £12,000 | £12,000 | £36,000 |
| **LLM API** (GPT-4o-mini) | £1,200 | £2,400 | £3,600 | £7,200 |
| **Storage** (MinIO self-hosted) | £3,600 | £3,600 | £3,600 | £10,800 |
| **Monitoring** (Grafana Cloud) | £600 | £600 | £600 | £1,800 |
| **All Other Tech** (open-source) | £0 | £0 | £0 | £0 |
| **TOTAL** | £111,000 | £112,200 | £113,400 | £336,600 |

**Customer Value**: £16M-£73M/year savings (100 sites) = **48-651x ROI**

---

## References

- [ARCHITECTURE.md](ARCHITECTURE.md) - Main architecture specification
- [DIAGRAMS.md](DIAGRAMS.md) - System architecture diagrams
- [BUSINESS_WORKFLOWS.md](../BUSINESS_WORKFLOWS.md) - Business process workflows
- [EXECUTIVE_SUMMARY.md](../EXECUTIVE_SUMMARY.md) - Commercial and investor summary
- [YC_PITCH_DECK.md](../YC_PITCH_DECK.md) - Y Combinator pitch deck

---

Last Updated: 2025-11-25
Document Version: 4.0 (Added Technology Decision Matrix)
