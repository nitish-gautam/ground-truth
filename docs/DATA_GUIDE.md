# Data Guide - Infrastructure Intelligence Platform

> Complete guide to datasets, data import, and data sources

---

## Current Data Status

### ✅ Imported Data (Ready to Use)

**University of Twente GPR Dataset**
- **Surveys Imported**: 10 of 125 available
- **GPR Scans**: 100+ SEG-Y radargram files
- **File Format**: SEG-Y (.sgy) - industry standard
- **Location**: Netherlands (University of Twente campus)
- **Ground Truth**: Known utility locations included
- **Metadata**: Environmental conditions, utility types, soil data

**Database Status**:
```sql
-- Check imported surveys
SELECT COUNT(*) FROM gpr_surveys WHERE survey_name LIKE 'Twente%';
-- Result: 10 surveys

-- View survey details
SELECT survey_name, location_id, survey_date, status
FROM gpr_surveys
WHERE survey_name LIKE 'Twente%'
ORDER BY created_at DESC;
```

**API Access**:
```bash
# Get all surveys
curl http://localhost:8002/api/v1/gpr/surveys | python3 -m json.tool

# Get specific survey
curl http://localhost:8002/api/v1/gpr/surveys/{survey_id}
```

---

## Available Datasets

### 1. University of Twente GPR Dataset ⭐ (READY)

**Overview**:
- **Total Surveys**: 125 locations
- **Total Scans**: ~1,500 radargrams
- **File Size**: ~400MB compressed
- **Format**: SEG-Y (.sgy), PNG (ground truth), CSV (metadata)

**Data Characteristics**:
| Attribute | Details |
|-----------|---------|
| **Location** | Netherlands (various land uses) |
| **Equipment** | GSSI SIR-4000, 400 MHz antenna |
| **Soil Types** | Sandy (majority), Clayey |
| **Utilities** | Electricity, telecom, water, gas, sewer |
| **Weather** | Dry, rainy conditions |
| **Ground Truth** | Annotated utility locations |

**Sample Survey Data**:
```
Survey: 01.1
- Scans: 12 radargrams (Path1.sgy through Path12.sgy)
- Utilities: 8 detected (water, sewer, telecom, gas)
- Ground: Sandy, 9.0 permittivity
- Weather: Dry
- Terrain: Flat, smooth
```

**How to Import**:
```bash
# Import 10 surveys (quick test)
docker compose exec backend python /app/scripts/import_twente_gpr_data.py

# Import ALL 125 surveys (edit script first)
# Edit: backend/scripts/import_twente_gpr_data.py
# Change: extract_limit=13, survey_limit=125
docker compose exec backend python /app/scripts/import_twente_gpr_data.py
```

**File Locations**:
- **Raw Data**: `/datasets/raw/twente_gpr/*.zip`
- **Metadata**: `/datasets/raw/twente_gpr/Metadata.csv`
- **Extracted**: `/datasets/processed/twente_gpr_extracted/`

---

### 2. Mojahid GPR Images Dataset (AVAILABLE)

**Overview**:
- **Total Images**: 2,239+ labeled images
- **Categories**: 6 classes
- **Use Case**: Image classification, CNN training
- **Format**: PNG/JPEG images

**Categories**:
1. Utility pipes
2. Background/soil
3. Hyperbola patterns
4. Rebar
5. Voids
6. Unknown objects

**Status**: Downloaded but not yet imported to database

**Location**: `/datasets/raw/mojahid_images/`

**How to Use**:
```bash
# List images
ls /datasets/raw/mojahid_images/

# Count by category
find /datasets/raw/mojahid_images -name "*.png" | wc -l
```

---

### 3. PAS 128 Compliance Documents (AVAILABLE)

**Overview**:
- **Standard**: PAS 128:2022 specification
- **Related**: CDM 2015 regulations
- **Format**: PDF, HTML, JSON

**Files Available**:
- `PAS128_Client_Specification_Guide_Sep22.pdf`
- `CDM2015_Regulations_HSE.html`
- `quality_levels_specification.json`
- `PAS128_Report_Template.md`

**Location**: `/datasets/raw/pas128_docs/`

**Usage**: Phase 2 RAG pipeline, compliance validation

---

### 4. Asset & Certification Data (Phase 1D) 🆕

**Overview**:
- **Scale**: Designed for 2M+ assets (HS2-scale projects)
- **Documents**: 100,000+ deliverables per major contract
- **Certificates**: PDFs, Excel spreadsheets, scanned documents
- **Formats**: PDF, XLSX, CSV, scanned images

**Data Categories**:

#### Asset Data
- **Asset Registers**: Excel/CSV with asset codes, types, locations
- **Installation Records**: Photos, commissioning docs, test certificates
- **As-Built Drawings**: CAD/PDF with final asset locations
- **Maintenance Records**: Service history, inspections

#### Certificate Data
- **Qualification Certificates**: PDF scans of worker certifications
- **Material Certificates**: Mill test certificates, compliance docs
- **Test Certificates**: PAT, pressure tests, commissioning tests
- **Inspection Reports**: Third-party inspection results

#### TAEM & IDP Data
- **TAEM Requirements**: Technical Assurance Evidence Model specs (JSON/Excel)
- **IDP Deliverables**: Information Delivery Plan tracking (Excel/CSV)
- **Compliance Matrices**: Requirements vs evidence mapping
- **Milestone Schedules**: Contract milestone tracking

**Sample Data Structure**:
```
datasets/raw/
├── assets/
│   ├── asset_register.xlsx          # 2M+ asset records
│   ├── installation_photos/         # Asset photos
│   └── commissioning_docs/          # Commissioning certificates
│
├── certificates/
│   ├── worker_qualifications/       # PDF certificates
│   ├── material_certs/              # Mill test certificates
│   └── test_reports/                # Inspection reports
│
├── taem_idp/
│   ├── taem_requirements.json       # TAEM model
│   ├── idp_deliverables.xlsx        # IDP tracker
│   └── compliance_matrix.xlsx       # Evidence mapping
│
└── enterprise_exports/
    ├── aims_export.json             # AIMS data export
    ├── bim360_docs.zip              # BIM 360 documents
    └── sharepoint_index.csv         # Document index
```

**Import Commands**:
```bash
# Import asset register
docker compose exec backend python /app/scripts/import_assets.py \
  --file /datasets/raw/assets/asset_register.xlsx

# Import certificates (batch)
docker compose exec backend python /app/scripts/import_certificates.py \
  --folder /datasets/raw/certificates/worker_qualifications/

# Import TAEM requirements
docker compose exec backend python /app/scripts/import_taem.py \
  --file /datasets/raw/taem_idp/taem_requirements.json

# Import IDP deliverables
docker compose exec backend python /app/scripts/import_idp.py \
  --file /datasets/raw/taem_idp/idp_deliverables.xlsx
```

---

## Download Additional Datasets

### Free / Public Datasets

#### 1. University of Twente (Already Downloaded)
- **Source**: https://data.4tu.nl/
- **Search**: "Ground Penetrating Radar" or "GPR"
- **License**: Open access (CC BY 4.0)
- **Status**: ✅ Downloaded (400MB)

#### 2. Zenodo GPR Datasets
- **Source**: https://zenodo.org/
- **Search**: "GPR utility detection"
- **Example**: https://zenodo.org/record/1234567
- **Format**: Various (SEG-Y, DZT, CSV)

**How to Download**:
```bash
# Using wget
cd /datasets/raw/
wget https://zenodo.org/record/XXXXXX/files/dataset.zip
unzip dataset.zip -d zenodo_gpr/
```

#### 3. OpenEI (US Department of Energy)
- **Source**: https://openei.org/
- **Search**: "utility detection" or "subsurface"
- **Format**: Various formats
- **License**: Public domain

#### 4. USGS Geophysical Data
- **Source**: https://www.usgs.gov/
- **Search**: "ground penetrating radar"
- **Format**: Government standard formats
- **License**: Public domain

---

### Commercial / Sample Datasets

#### 1. GSSI (Geophysical Survey Systems)
- **Source**: https://www.geophysical.com/sample-data
- **Format**: DZT (proprietary but readable)
- **Cost**: Free samples available
- **Note**: Registration required

**How to Request**:
1. Visit GSSI website
2. Create account
3. Request sample data
4. Download DZT files

#### 2. Sensors & Software
- **Source**: https://www.sensoft.ca/
- **Format**: DT1 (proprietary)
- **Cost**: Free samples
- **Equipment**: Noggin, pulseEKKO systems

#### 3. MALÅ GPR
- **Source**: https://www.guidelinegeo.com/
- **Format**: RD3, RAD files
- **Cost**: Demo data available

---

## Data Import Guide

### Import Twente GPR Data

**Step 1: Verify Data Exists**
```bash
# Check raw data
ls -lh /datasets/raw/twente_gpr/*.zip

# Expected: 13 ZIP files (01.zip through 013.zip)
```

**Step 2: Run Import Script**
```bash
docker compose exec backend python /app/scripts/import_twente_gpr_data.py
```

**Step 3: Verify Import**
```bash
# Check database
docker compose exec postgres psql -U gpr_user -d gpr_db -c \
  "SELECT COUNT(*) FROM gpr_surveys WHERE survey_name LIKE 'Twente%';"

# Check extracted files
ls -lh /datasets/processed/twente_gpr_extracted/
```

**Import Statistics**:
- **Time**: ~5 seconds for 10 surveys
- **Surveys Created**: 10 (configurable)
- **Files Extracted**: 100+ .sgy files
- **Database Records**: 10 survey entries

---

### Create Your Own GPR Data

#### Synthetic Data Generation (Coming Soon)

```python
# Generate synthetic GPR data for testing
from scripts.generate_synthetic_gpr import generate_survey

survey = generate_survey(
    utilities=['water', 'gas', 'electric'],
    soil_type='sandy',
    depth_range=(0.5, 3.0),  # meters
    trace_count=512,
    samples_per_trace=512
)
```

#### Import from Equipment

**Supported Formats**:
- SEG-Y (.sgy, .segy)
- GSSI DZT (.dzt)
- Sensors & Software DT1 (.dt1)
- MALÅ RD3 (.rd3)

**Import Workflow**:
```bash
# 1. Copy files to datasets/raw/
cp /path/to/scan.sgy /datasets/raw/custom_gpr/

# 2. Create import script (see import_twente_gpr_data.py as template)
# 3. Run import
docker compose exec backend python /app/scripts/import_custom_gpr.py
```

---

## Data Processing Pipeline

### Current Status (Phase 1A)

```
Raw GPR Files (.sgy)
    ↓
Extract Metadata ✅
    ↓
Store in Database ✅
    ↓
[Signal Processing] 🚧 (Pending)
    ↓
[Feature Extraction] 🚧 (Pending)
    ↓
[ML Classification] 🚧 (Phase 2)
```

### Future Processing (Planned)

**Phase 1A** (Current):
- ✅ File extraction
- ✅ Metadata parsing
- ✅ Database storage
- 🚧 SEG-Y signal parsing
- 🚧 B-scan image generation

**Phase 2** (LLM Integration):
- Vector embeddings
- Similarity search
- Pattern recognition
- Automated report generation

---

## Data Organization

### Directory Structure

```
datasets/
├── raw/                           # Original downloaded data
│   ├── twente_gpr/               # University of Twente (125 surveys)
│   │   ├── 01.zip through 013.zip
│   │   ├── Metadata.csv
│   │   └── Readme.txt
│   ├── mojahid_images/           # Labeled GPR images (2,239+)
│   ├── pas128_docs/              # Compliance documents
│   ├── bgs_data/                 # British Geological Survey
│   ├── uk_gas_networks/          # Utility company data
│   ├── usag_reports/             # Strike incident reports
│   │
│   ├── assets/                   # 🆕 Phase 1D: Asset data
│   │   ├── asset_register.xlsx   # 2M+ asset records
│   │   ├── installation_photos/  # Asset photos
│   │   ├── commissioning_docs/   # Commissioning certificates
│   │   └── as_built_drawings/    # Final asset locations
│   │
│   ├── certificates/             # 🆕 Phase 1D: Certificate data
│   │   ├── worker_qualifications/ # PDF certificates
│   │   ├── material_certs/       # Mill test certificates
│   │   ├── test_reports/         # Inspection reports
│   │   └── training_records/     # Training certificates
│   │
│   ├── taem_idp/                 # 🆕 Phase 1D: TAEM & IDP
│   │   ├── taem_requirements.json # TAEM model
│   │   ├── idp_deliverables.xlsx # IDP tracker (100k+ items)
│   │   ├── compliance_matrix.xlsx # Evidence mapping
│   │   └── milestone_schedule.xlsx # Contract milestones
│   │
│   └── enterprise_exports/       # 🆕 Phase 2B: Enterprise data
│       ├── aims_export.json      # AIMS data export
│       ├── bim360_docs.zip       # BIM 360 documents
│       ├── sharepoint_index.csv  # Document index
│       └── aconex_deliverables/  # Aconex exports
│
├── processed/                     # Extracted/processed data
│   ├── twente_gpr_extracted/     # Unzipped radargrams
│   │   ├── 01/01.1/Radargrams/  # 12 .sgy files
│   │   ├── 01/01.2/Radargrams/  # 12 .sgy files
│   │   └── ...
│   ├── synthetic/                # Generated test data
│   ├── certificate_ocr/          # 🆕 OCR results (JSON)
│   ├── asset_embeddings/         # 🆕 Vector embeddings
│   └── document_index/           # 🆕 Full-text search index
│
└── download_scripts/              # Automated download tools
    ├── download_all.py
    ├── download_twente.py
    ├── download_mojahid.py
    ├── download_pas128.py
    ├── export_from_aims.py       # 🆕 AIMS export script
    └── sync_from_cde.py          # 🆕 CDE sync script
```

---

## Data Statistics

### Twente GPR Dataset Breakdown

**By Survey Type**:
- Verify statutory records: 85 surveys
- Map free subsoil space: 25 surveys
- Locate specific utilities: 15 surveys

**By Land Use**:
- High density residential: 45 surveys
- Commercial/industrial: 30 surveys
- Public institutions: 25 surveys
- Rural residential: 15 surveys
- Other: 10 surveys

**By Utility Type Detected**:
- Electricity: 95 surveys
- Telecommunications: 88 surveys
- Water: 62 surveys
- Gas/Oil/Chemicals: 48 surveys
- Sewer: 35 surveys
- Unknown: 40 surveys

**Ground Conditions**:
- Sandy soil: 105 surveys (84%)
- Clayey soil: 20 surveys (16%)
- Permittivity range: 8.16 - 19.46

**Weather Conditions**:
- Dry: 95 surveys (76%)
- Rainy: 30 surveys (24%)

---

## Data Quality Metrics

### Ground Truth Validation (GPR Data)

Each Twente survey includes:
- ✅ **Ground truth images**: Annotated utility locations
- ✅ **Survey maps**: Site layout diagrams
- ✅ **Metadata**: Complete environmental data
- ✅ **Known utilities**: Material, diameter, depth information

**Accuracy Potential**:
- Training set: 80 surveys (10,000+ traces)
- Validation set: 25 surveys
- Test set: 20 surveys
- **Target**: >95% detection accuracy

### Certificate Quality Metrics (Phase 1D) 🆕

**OCR Accuracy Requirements**:
- **Target**: >98% character recognition accuracy
- **DPI Requirement**: 300+ DPI for scanned documents
- **Formats**: PDF (text or scanned), JPEG/PNG images, Excel/CSV

**Validation Checks**:
- ✅ **Issuer Validation**: Verify against approved certification bodies
- ✅ **Date Validation**: Check issue/expiry dates are logical
- ✅ **Qualification Validation**: Match against TAEM requirements
- ✅ **Signature Detection**: Verify authorized signatures present
- ✅ **Duplicate Detection**: Identify duplicate submissions

**Certificate Processing Pipeline**:
```
PDF/Image Upload
    ↓
OCR (Azure Document Intelligence / AWS Textract)
    ↓
NLP Qualification Extraction (spaCy + GPT-4)
    ↓
Validation Against Requirements
    ↓
Database Storage + Vector Embedding
    ↓
Real-time Assurance Scoring
```

**Quality Metrics**:
| Metric | Target | Purpose |
|--------|--------|---------|
| OCR Accuracy | >98% | Character recognition quality |
| Qualification Extraction | >95% | Correct field extraction (names, dates, skills) |
| Validation Accuracy | >99% | Correct pass/fail against requirements |
| Processing Time | <30 sec | Per certificate (OCR + NLP + validation) |
| False Positive Rate | <2% | Incorrectly approved certificates |
| False Negative Rate | <1% | Incorrectly rejected certificates |

**Example Certificate Data**:
```json
{
  "certificate_id": "CERT-12345",
  "certificate_type": "CSCS Card",
  "worker_name": "John Smith",
  "qualification": "Advanced Scaffolder",
  "issuer": "CITB",
  "issue_date": "2024-01-15",
  "expiry_date": "2029-01-15",
  "card_number": "ABC123456",
  "extracted_skills": [
    "Scaffold Inspection",
    "Temporary Works Coordinator",
    "Working at Height"
  ],
  "validation_status": "approved",
  "confidence_score": 0.97,
  "ocr_quality": "high"
}
```

---

## Data Security & Privacy

### Data Classification

**Public Data** (No Restrictions):
- PAS 128 specification documents
- Academic datasets (Twente, Mojahid)
- Synthetic generated data

**Confidential Data** (Project Isolation):
- Client survey data (when added)
- Proprietary utility records
- Strike incident details

### Storage Security

- **Encryption at Rest**: MinIO with AES-256
- **Encryption in Transit**: TLS 1.3
- **Access Control**: Role-based (RBAC)
- **Audit Logging**: All data access logged
- **Backup**: Automated daily backups (when configured)

---

## FAQ

### Q: How do I import more than 10 surveys?

Edit `backend/scripts/import_twente_gpr_data.py`:
```python
# Change line ~300
importer.import_dataset(
    extract_limit=13,   # All ZIP files
    survey_limit=125    # All surveys
)
```

### Q: Can I use my own GPR data?

Yes! Create a custom import script based on `import_twente_gpr_data.py` or manually upload via API:
```bash
POST /api/v1/gpr/surveys
POST /api/v1/gpr/scans
```

### Q: What data formats are supported?

Currently:
- ✅ SEG-Y (.sgy, .segy)
- 🚧 GSSI DZT (.dzt) - Planned
- 🚧 Sensors & Software DT1 (.dt1) - Planned
- 🚧 MALÅ RD3 (.rd3) - Planned

### Q: Where is the data physically stored?

- **Database**: PostgreSQL container volume
- **Files**: Docker volume `/datasets` → host machine
- **Object Storage**: MinIO container (S3-compatible)

### Q: How do I backup my data?

```bash
# Database backup
docker compose exec postgres pg_dump -U gpr_user gpr_db > backup.sql

# File backup
cp -r /datasets /backup/datasets_$(date +%Y%m%d)

# MinIO backup
mc mirror local-minio/gpr-data /backup/minio/
```

---

## Next Steps

1. **Import Full Dataset**: Run import for all 125 surveys
2. **Parse SEG-Y Files**: Extract trace data from .sgy files
3. **Generate Images**: Create B-scan visualizations
4. **Train Models**: Use data for ML utility classification
5. **Add Custom Data**: Import your own GPR surveys

---

**For more information, see**:
- [GETTING_STARTED.md](GETTING_STARTED.md) - Setup instructions
- [architecture/ARCHITECTURE.md](architecture/ARCHITECTURE.md) - System design
- Backend import script: `backend/scripts/import_twente_gpr_data.py`
