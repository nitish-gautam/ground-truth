# Data Dictionary - HS2 Assurance Intelligence

**Version**: 1.0
**Last Updated**: November 25, 2024
**Database**: PostgreSQL 16

---

## 📋 Overview

This document describes the complete database schema for the HS2 Assurance Intelligence Demonstrator, including table structures, relationships, indexes, and sample data.

**Total Schema**:
- **6 Core Tables**: Assets, Deliverables, Costs, Certificates, TAEM Rules, Evaluations
- **1 Materialized View**: Asset Readiness Summary
- **150+ Fields** across all tables
- **900+ Sample Records** in placeholder data

---

## 🗄️ Table Schemas

### 1. `hs2_assets`

**Purpose**: Core asset tracking for HS2 infrastructure (viaducts, bridges, tunnels, OLE masts)

**Row Count**: 50 in demo data

| Column | Type | Constraints | Description | Example |
|--------|------|-------------|-------------|---------|
| `id` | UUID | PK, Default uuid_generate_v4() | Unique internal identifier | `a1b2c3d4-...` |
| `asset_id` | VARCHAR(50) | UNIQUE, NOT NULL | External asset identifier from AIMS | `VA-001`, `BR-015` |
| `asset_type` | VARCHAR(50) | NOT NULL | Asset classification | `Viaduct`, `Bridge`, `Tunnel`, `OLE Mast` |
| `route_section` | VARCHAR(100) | | HS2 route section | `London-Euston`, `Old Oak Common` |
| `contractor` | VARCHAR(100) | | Joint venture responsible for construction | `JV-Alpha`, `JV-Bravo`, `JV-Charlie` |
| `readiness_status` | VARCHAR(50) | | Current readiness classification (computed) | `Ready`, `Not Ready`, `At Risk` |
| `taem_evaluation_score` | DECIMAL(5,2) | DEFAULT 0.00 | TAEM score 0-100 (computed) | `95.35`, `47.22` |
| `location_text` | VARCHAR(255) | | Human-readable location (no GPS in demo) | `Between Euston Station and Camden` |
| `metadata` | JSONB | | Flexible additional data | `{"height_m": 15, "span_m": 45}` |
| `created_at` | TIMESTAMPTZ | DEFAULT NOW() | Record creation timestamp | `2024-11-25 10:00:00+00` |
| `updated_at` | TIMESTAMPTZ | DEFAULT NOW() | Last update timestamp | `2024-11-25 14:30:00+00` |
| `created_by` | VARCHAR(100) | | User who created record | `system` |
| `updated_by` | VARCHAR(100) | | User who last updated record | `john.smith@hs2.org.uk` |

**Indexes**:
- `idx_hs2_assets_asset_id`: B-tree on `asset_id` (unique lookups)
- `idx_hs2_assets_type_status`: B-tree on `(asset_type, readiness_status)` (dashboard queries)
- `idx_hs2_assets_contractor`: B-tree on `contractor` (contractor reports)
- `idx_hs2_assets_metadata`: GIN on `metadata` (JSONB queries)

**Sample Data**:
```sql
INSERT INTO hs2_assets (asset_id, asset_type, route_section, contractor, location_text, metadata)
VALUES (
    'VA-001',
    'Viaduct',
    'London-Euston',
    'JV-Alpha',
    'Between Euston Station and Camden Town',
    '{"height_m": 15, "span_m": 45, "construction_start": "2023-01-15"}'
);
```

**Relationships**:
- One-to-many with `hs2_deliverables` (1 asset → N deliverables)
- One-to-many with `hs2_costs` (1 asset → N cost lines)
- One-to-many with `hs2_certificates` (1 asset → N certificates)
- One-to-many with `hs2_rule_evaluations` (1 asset → N evaluations)

---

### 2. `hs2_deliverables`

**Purpose**: Information Delivery Plan (IDP) tracking - documents required for asset handover

**Row Count**: 586 in demo data (5-15 per asset)

| Column | Type | Constraints | Description | Example |
|--------|------|-------------|-------------|---------|
| `id` | UUID | PK | Unique internal identifier | `b2c3d4e5-...` |
| `asset_id` | UUID | FK → hs2_assets(id), NOT NULL | Parent asset | `a1b2c3d4-...` |
| `deliverable_id` | VARCHAR(50) | UNIQUE, NOT NULL | External deliverable identifier | `DEL-VA-001-01` |
| `deliverable_name` | VARCHAR(255) | NOT NULL | Full deliverable name | `Design Certificate - Structural Integrity` |
| `deliverable_type` | VARCHAR(100) | | Category of deliverable | `Design Certificate`, `Test Report - Welding` |
| `due_date` | DATE | | Planned submission date | `2024-03-15` |
| `completion_status` | VARCHAR(50) | | Current status | `Complete`, `In Progress`, `Not Started`, `Overdue` |
| `submitted_date` | DATE | | Actual submission date (if complete) | `2024-03-10` |
| `document_reference` | VARCHAR(255) | | Link to document in CDE/SharePoint | `DOC-ABC123`, `https://bim360.com/...` |
| `metadata` | JSONB | | Additional deliverable data | `{"reviewer": "jane.doe", "version": 2}` |
| `created_at` | TIMESTAMPTZ | DEFAULT NOW() | Record creation timestamp | |
| `updated_at` | TIMESTAMPTZ | DEFAULT NOW() | Last update timestamp | |

**Indexes**:
- `idx_hs2_deliverables_asset_id`: B-tree on `asset_id` (get all deliverables for asset)
- `idx_hs2_deliverables_status`: B-tree on `completion_status` (filter by status)
- `idx_hs2_deliverables_type`: B-tree on `deliverable_type` (group by type)
- `idx_hs2_deliverables_due_date`: B-tree on `due_date` (find upcoming/overdue)

**Sample Data**:
```sql
INSERT INTO hs2_deliverables (asset_id, deliverable_id, deliverable_name, deliverable_type, due_date, completion_status)
VALUES (
    (SELECT id FROM hs2_assets WHERE asset_id = 'VA-001'),
    'DEL-VA-001-01',
    'Design Certificate - Structural Integrity',
    'Design Certificate',
    '2024-03-15',
    'Complete'
);
```

**Business Rules**:
- Critical deliverable types: `Design Certificate`, `Assurance Sign-off`, `Test Report - Concrete`, `Test Report - Welding`
- Status transition: `Not Started` → `In Progress` → `Complete` (or `Overdue` if past due_date)
- `Overdue` status automatically set if `completion_status != 'Complete'` AND `due_date < today`

---

### 3. `hs2_costs`

**Purpose**: Cost tracking and budget variance monitoring per asset

**Row Count**: 50 in demo data (1 per asset)

| Column | Type | Constraints | Description | Example |
|--------|------|-------------|-------------|---------|
| `id` | UUID | PK | Unique internal identifier | `c3d4e5f6-...` |
| `asset_id` | UUID | FK → hs2_assets(id), UNIQUE, NOT NULL | Parent asset (one cost line per asset) | `a1b2c3d4-...` |
| `cost_line_id` | VARCHAR(50) | UNIQUE, NOT NULL | External cost identifier | `COST-VA-001` |
| `budget` | DECIMAL(15,2) | NOT NULL | Original budgeted amount (£) | `5000000.00` |
| `cost_to_date` | DECIMAL(15,2) | NOT NULL | Actual cost incurred to date (£) | `6200000.00` |
| `variance_amount` | DECIMAL(15,2) | | Difference (actual - budget) | `1200000.00` |
| `variance_pct` | DECIMAL(5,2) | | Variance as percentage | `24.00` |
| `last_updated` | DATE | | Date of last cost update | `2024-11-20` |
| `metadata` | JSONB | | Additional cost data | `{"contract_id": "HS2-C1-123"}` |
| `created_at` | TIMESTAMPTZ | DEFAULT NOW() | Record creation timestamp | |
| `updated_at` | TIMESTAMPTZ | DEFAULT NOW() | Last update timestamp | |

**Indexes**:
- `idx_hs2_costs_asset_id`: B-tree on `asset_id` (get cost for asset)
- `idx_hs2_costs_variance`: B-tree on `variance_pct` (find over-budget assets)

**Computed Fields**:
```sql
-- variance_amount and variance_pct are computed:
variance_amount = cost_to_date - budget
variance_pct = ((cost_to_date - budget) / budget) * 100
```

**Sample Data**:
```sql
INSERT INTO hs2_costs (asset_id, cost_line_id, budget, cost_to_date, variance_amount, variance_pct)
VALUES (
    (SELECT id FROM hs2_assets WHERE asset_id = 'VA-001'),
    'COST-VA-001',
    5000000.00,
    6200000.00,
    1200000.00,
    24.00
);
```

**Business Rules**:
- TAEM Rule R002 evaluates variance_pct:
  - Pass: ≤10%
  - Warning: 10-20%
  - Fail: >20%
- Negative variance = under budget (favorable)
- Positive variance = over budget (unfavorable)

---

### 4. `hs2_certificates`

**Purpose**: Certificate metadata tracking (welding qualifications, test certificates, design approvals)

**Row Count**: 214 in demo data (2-5 per asset)

| Column | Type | Constraints | Description | Example |
|--------|------|-------------|-------------|---------|
| `id` | UUID | PK | Unique internal identifier | `d4e5f6g7-...` |
| `asset_id` | UUID | FK → hs2_assets(id), NOT NULL | Parent asset | `a1b2c3d4-...` |
| `certificate_id` | VARCHAR(50) | UNIQUE, NOT NULL | External certificate identifier | `CERT-VA-001-01` |
| `certificate_type` | VARCHAR(100) | NOT NULL | Type of certificate | `Welding Qualification`, `Design Certificate` |
| `qualification_text` | TEXT | | Extracted qualification details (from OCR) | `Qualified for structural steel welding per BS EN 1090` |
| `issue_date` | DATE | NOT NULL | Certificate issue date | `2023-06-15` |
| `expiry_date` | DATE | | Certificate expiration date | `2025-06-15` |
| `issuing_body` | VARCHAR(255) | | Certifying organization | `The Welding Institute`, `BSI Group` |
| `status` | VARCHAR(50) | | Certificate validity status | `Valid`, `Expired`, `Qualified` |
| `extracted_metadata` | JSONB | | Full OCR/NLP output | `{"confidence": 0.95, "source": "azure_doc_intel"}` |
| `created_at` | TIMESTAMPTZ | DEFAULT NOW() | Record creation timestamp | |
| `updated_at` | TIMESTAMPTZ | DEFAULT NOW() | Last update timestamp | |

**Indexes**:
- `idx_hs2_certificates_asset_id`: B-tree on `asset_id`
- `idx_hs2_certificates_status`: B-tree on `status`
- `idx_hs2_certificates_expiry`: B-tree on `expiry_date` (find expiring soon)
- `idx_hs2_certificates_type`: B-tree on `certificate_type`

**Sample Data**:
```sql
INSERT INTO hs2_certificates (asset_id, certificate_id, certificate_type, issue_date, expiry_date, issuing_body, status)
VALUES (
    (SELECT id FROM hs2_assets WHERE asset_id = 'VA-001'),
    'CERT-VA-001-01',
    'Welding Qualification',
    '2023-06-15',
    '2025-06-15',
    'The Welding Institute',
    'Valid'
);
```

**Business Rules**:
- Status determination:
  - `Valid`: expiry_date > today
  - `Expired`: expiry_date ≤ today
  - `Qualified`: Special status with conditions (see qualification_text)
- TAEM Rule R003 checks for expired certificates
- Warning issued if expiry within 30 days

---

### 5. `hs2_taem_rules`

**Purpose**: TAEM (Technical Assurance Evidence Model) rule definitions - configurable business rules

**Row Count**: 6 in demo data

| Column | Type | Constraints | Description | Example |
|--------|------|-------------|-------------|---------|
| `id` | UUID | PK | Unique internal identifier | `e5f6g7h8-...` |
| `rule_id` | VARCHAR(50) | UNIQUE, NOT NULL | External rule identifier | `R001`, `R002` |
| `rule_name` | VARCHAR(255) | NOT NULL | Human-readable rule name | `Critical Deliverables Present` |
| `description` | TEXT | | Detailed rule description | `Ensure all critical deliverables...` |
| `severity` | VARCHAR(50) | NOT NULL | Rule importance level | `Critical`, `Major`, `Minor` |
| `weight` | INTEGER | NOT NULL, CHECK (weight > 0) | Points awarded for passing | `50`, `25`, `10` |
| `config` | JSONB | | Configurable rule parameters (tinkerability) | `{"threshold_fail": 20}` |
| `is_active` | BOOLEAN | DEFAULT TRUE | Whether rule is currently enforced | `true`, `false` |
| `created_at` | TIMESTAMPTZ | DEFAULT NOW() | Record creation timestamp | |
| `updated_at` | TIMESTAMPTZ | DEFAULT NOW() | Last update timestamp | |

**Indexes**:
- `idx_hs2_taem_rules_rule_id`: B-tree on `rule_id` (unique lookups)
- `idx_hs2_taem_rules_severity`: B-tree on `severity` (filter by severity)
- `idx_hs2_taem_rules_active`: B-tree on `is_active` (get active rules only)

**Sample Data**:
```sql
INSERT INTO hs2_taem_rules (rule_id, rule_name, description, severity, weight, config)
VALUES (
    'R002',
    'Cost Variance Within Tolerance',
    'Ensure asset costs remain within acceptable budget variance thresholds',
    'Major',
    25,
    '{"threshold_warning": 10, "threshold_fail": 20}'::jsonb
);
```

**Severity Levels**:
- **Critical** (50 points): Any failure = "Not Ready" status
- **Major** (25-30 points): Multiple failures = "At Risk" status
- **Minor** (10 points): For warnings and nice-to-have checks

**Tinkerability**:
```sql
-- Adjust R002 cost variance threshold
UPDATE hs2_taem_rules
SET config = jsonb_set(config, '{threshold_fail}', '15')
WHERE rule_id = 'R002';
```

---

### 6. `hs2_rule_evaluations`

**Purpose**: Audit trail of all TAEM rule evaluations - full evidence for 7+ year compliance (CDM 2015)

**Row Count**: 300 in demo data (6 rules × 50 assets = 300 evaluations)

| Column | Type | Constraints | Description | Example |
|--------|------|-------------|-------------|---------|
| `id` | UUID | PK | Unique internal identifier | `f6g7h8i9-...` |
| `asset_id` | UUID | FK → hs2_assets(id), NOT NULL | Asset being evaluated | `a1b2c3d4-...` |
| `rule_id` | UUID | FK → hs2_taem_rules(id), NOT NULL | Rule applied | `e5f6g7h8-...` |
| `outcome` | VARCHAR(50) | NOT NULL | Evaluation result | `Pass`, `Fail`, `Warning`, `N/A` |
| `points_awarded` | INTEGER | DEFAULT 0 | Points earned (0 to rule.weight) | `50`, `25`, `0` |
| `evidence` | JSONB | NOT NULL | Detailed evidence supporting outcome | `{"missing": ["Design Certificate"]}` |
| `message` | TEXT | | Human-readable explanation | `Missing critical deliverables: Design Certificate` |
| `evaluated_at` | TIMESTAMPTZ | DEFAULT NOW() | Evaluation timestamp | `2024-11-25 14:30:00+00` |
| `evaluated_by` | VARCHAR(100) | | User or system that triggered evaluation | `system`, `john.smith@hs2.org.uk` |

**Indexes**:
- `idx_hs2_rule_evaluations_asset_id`: B-tree on `asset_id` (get all evaluations for asset)
- `idx_hs2_rule_evaluations_rule_id`: B-tree on `rule_id` (see all instances of rule)
- `idx_hs2_rule_evaluations_outcome`: B-tree on `outcome` (filter by Pass/Fail)
- `idx_hs2_rule_evaluations_evaluated_at`: B-tree on `evaluated_at` (time-based queries)
- `idx_hs2_rule_evaluations_evidence`: GIN on `evidence` (JSONB queries)

**Unique Constraint**:
```sql
UNIQUE (asset_id, rule_id, evaluated_at)
-- Allows re-evaluation of same rule over time
```

**Sample Data**:
```sql
INSERT INTO hs2_rule_evaluations (asset_id, rule_id, outcome, points_awarded, evidence, message)
VALUES (
    (SELECT id FROM hs2_assets WHERE asset_id = 'VA-007'),
    (SELECT id FROM hs2_taem_rules WHERE rule_id = 'R001'),
    'Fail',
    0,
    '{
        "required": ["Design Certificate", "Assurance Sign-off"],
        "found_complete": [],
        "missing": ["Design Certificate", "Assurance Sign-off"]
    }'::jsonb,
    'Missing critical deliverables: Design Certificate, Assurance Sign-off'
);
```

**Evidence Structure Examples**:

**R001 (Deliverable Completion)**:
```json
{
  "required": ["Design Certificate", "Assurance Sign-off"],
  "found_complete": ["Test Report - Concrete"],
  "missing": ["Design Certificate", "Assurance Sign-off"]
}
```

**R002 (Cost Variance)**:
```json
{
  "budget": 5000000.0,
  "cost_to_date": 6200000.0,
  "variance_pct": 24.0,
  "threshold": 20.0,
  "over_budget": true
}
```

**R003 (Certificate Validity)**:
```json
{
  "total_certificates": 5,
  "valid": 4,
  "expired": 1,
  "expired_list": [
    {
      "cert_id": "CERT-VA-007-02",
      "type": "Welding Qualification",
      "expiry_date": "2024-10-15",
      "days_expired": 41
    }
  ]
}
```

---

### 7. `hs2_asset_readiness_summary` (Materialized View)

**Purpose**: Pre-computed dashboard view for performance (avoids expensive joins on every query)

**Refresh Strategy**: Triggered after each batch evaluation

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `asset_id` | VARCHAR(50) | Asset identifier | `VA-001` |
| `asset_type` | VARCHAR(50) | Asset classification | `Viaduct` |
| `route_section` | VARCHAR(100) | Route section | `London-Euston` |
| `contractor` | VARCHAR(100) | JV contractor | `JV-Alpha` |
| `readiness_status` | VARCHAR(50) | Overall status | `Ready`, `Not Ready`, `At Risk` |
| `taem_evaluation_score` | DECIMAL(5,2) | Score 0-100 | `95.35` |
| `critical_fails` | BIGINT | Count of critical failures | `0` |
| `major_fails` | BIGINT | Count of major failures | `2` |
| `minor_fails` | BIGINT | Count of minor failures | `1` |
| `budget` | DECIMAL(15,2) | Budget amount | `5000000.00` |
| `cost_to_date` | DECIMAL(15,2) | Actual cost | `5250000.00` |
| `variance_pct` | DECIMAL(5,2) | Cost variance | `5.00` |

**Creation SQL**:
```sql
CREATE MATERIALIZED VIEW hs2_asset_readiness_summary AS
SELECT
    a.asset_id,
    a.asset_type,
    a.route_section,
    a.contractor,
    a.readiness_status,
    a.taem_evaluation_score,
    COUNT(CASE WHEN re.outcome = 'Fail' AND tr.severity = 'Critical' THEN 1 END) AS critical_fails,
    COUNT(CASE WHEN re.outcome = 'Fail' AND tr.severity = 'Major' THEN 1 END) AS major_fails,
    COUNT(CASE WHEN re.outcome = 'Fail' AND tr.severity = 'Minor' THEN 1 END) AS minor_fails,
    c.budget,
    c.cost_to_date,
    c.variance_pct
FROM hs2_assets a
LEFT JOIN hs2_rule_evaluations re ON a.id = re.asset_id
LEFT JOIN hs2_taem_rules tr ON re.rule_id = tr.id
LEFT JOIN hs2_costs c ON a.id = c.asset_id
GROUP BY a.id, a.asset_id, a.asset_type, a.route_section, a.contractor,
         a.readiness_status, a.taem_evaluation_score, c.budget, c.cost_to_date, c.variance_pct;
```

**Refresh Command**:
```sql
REFRESH MATERIALIZED VIEW hs2_asset_readiness_summary;
```

**Usage**:
```sql
-- Dashboard queries use the materialized view for speed
SELECT readiness_status, COUNT(*), AVG(taem_evaluation_score)
FROM hs2_asset_readiness_summary
GROUP BY readiness_status;

-- Fast lookup (no joins needed)
SELECT * FROM hs2_asset_readiness_summary WHERE asset_id = 'VA-007';
```

---

## 🔗 Entity Relationships

```
┌─────────────┐
│  hs2_assets │
└─────┬───────┘
      │
      ├──< hs2_deliverables (1:N)
      │
      ├──< hs2_costs (1:1)
      │
      ├──< hs2_certificates (1:N)
      │
      └──< hs2_rule_evaluations (1:N)
             │
             └──> hs2_taem_rules (N:1)
```

**Cascade Behavior**:
```sql
-- Deleting an asset cascades to all related records:
DELETE FROM hs2_assets WHERE asset_id = 'VA-999';
-- This also deletes:
--   - All deliverables for VA-999
--   - All costs for VA-999
--   - All certificates for VA-999
--   - All rule evaluations for VA-999
```

---

## 📊 Sample Queries

### Get Asset with Full Details

```sql
SELECT
    a.asset_id,
    a.asset_type,
    a.readiness_status,
    a.taem_evaluation_score,
    COUNT(DISTINCT d.id) as total_deliverables,
    COUNT(DISTINCT CASE WHEN d.completion_status = 'Complete' THEN d.id END) as complete_deliverables,
    COUNT(DISTINCT CASE WHEN d.completion_status = 'Overdue' THEN d.id END) as overdue_deliverables,
    COUNT(DISTINCT cert.id) as total_certificates,
    COUNT(DISTINCT CASE WHEN cert.status = 'Expired' THEN cert.id END) as expired_certificates,
    c.budget,
    c.cost_to_date,
    c.variance_pct
FROM hs2_assets a
LEFT JOIN hs2_deliverables d ON a.id = d.asset_id
LEFT JOIN hs2_certificates cert ON a.id = cert.asset_id
LEFT JOIN hs2_costs c ON a.id = c.asset_id
WHERE a.asset_id = 'VA-007'
GROUP BY a.asset_id, a.asset_type, a.readiness_status, a.taem_evaluation_score,
         c.budget, c.cost_to_date, c.variance_pct;
```

### Dashboard Summary Statistics

```sql
SELECT
    COUNT(*) as total_assets,
    COUNT(CASE WHEN readiness_status = 'Ready' THEN 1 END) as ready_count,
    COUNT(CASE WHEN readiness_status = 'Not Ready' THEN 1 END) as not_ready_count,
    COUNT(CASE WHEN readiness_status = 'At Risk' THEN 1 END) as at_risk_count,
    ROUND(AVG(taem_evaluation_score), 2) as avg_score,
    ROUND(AVG(CASE WHEN readiness_status = 'Ready' THEN taem_evaluation_score END), 2) as avg_score_ready,
    ROUND(AVG(CASE WHEN readiness_status = 'Not Ready' THEN taem_evaluation_score END), 2) as avg_score_not_ready
FROM hs2_assets;
```

### Find Assets with Cost Overruns

```sql
SELECT
    a.asset_id,
    a.contractor,
    c.budget,
    c.cost_to_date,
    c.variance_pct,
    a.readiness_status
FROM hs2_assets a
JOIN hs2_costs c ON a.id = c.asset_id
WHERE c.variance_pct > 20
ORDER BY c.variance_pct DESC;
```

### Audit Trail: All Evaluations for an Asset

```sql
SELECT
    re.evaluated_at,
    tr.rule_name,
    tr.severity,
    re.outcome,
    re.points_awarded,
    re.message,
    re.evidence
FROM hs2_rule_evaluations re
JOIN hs2_taem_rules tr ON re.rule_id = tr.id
JOIN hs2_assets a ON re.asset_id = a.id
WHERE a.asset_id = 'VA-007'
ORDER BY re.evaluated_at DESC, tr.severity, tr.rule_name;
```

### Find All Assets Failing a Specific Rule

```sql
SELECT
    a.asset_id,
    a.asset_type,
    a.contractor,
    re.outcome,
    re.message,
    re.evaluated_at
FROM hs2_rule_evaluations re
JOIN hs2_assets a ON re.asset_id = a.id
JOIN hs2_taem_rules tr ON re.rule_id = tr.id
WHERE tr.rule_id = 'R001'
  AND re.outcome = 'Fail'
ORDER BY a.asset_id;
```

### Certificates Expiring Soon (Next 30 Days)

```sql
SELECT
    a.asset_id,
    cert.certificate_id,
    cert.certificate_type,
    cert.expiry_date,
    cert.expiry_date - CURRENT_DATE as days_until_expiry
FROM hs2_certificates cert
JOIN hs2_assets a ON cert.asset_id = a.id
WHERE cert.expiry_date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '30 days'
  AND cert.status = 'Valid'
ORDER BY cert.expiry_date;
```

---

## 🗺️ Mapping to Real HS2 Data

### Data Source Mapping

| Our Table | Real HS2 Source | Integration Method |
|-----------|----------------|-------------------|
| `hs2_assets` | AIMS (Asset Information Management System) | REST API sync |
| `hs2_deliverables` | IDP/MIDP systems (Information Delivery Plans) | CDE connector (BIM 360, Aconex) |
| `hs2_costs` | ERP systems (SAP, Oracle) | Database replication |
| `hs2_certificates` | SharePoint, BIM 360, email attachments | OCR + Azure Document Intelligence |
| `hs2_taem_rules` | Internal HS2 TAEM framework | Manual configuration |
| `hs2_rule_evaluations` | Generated by this system | N/A (internal) |

### Field Mapping Examples

**AIMS → hs2_assets**:
```
AIMS.UAID                → hs2_assets.asset_id
AIMS.AssetType           → hs2_assets.asset_type
AIMS.RouteSection        → hs2_assets.route_section
AIMS.Contractor          → hs2_assets.contractor
AIMS.LocationDescription → hs2_assets.location_text
AIMS.CustomFields        → hs2_assets.metadata (JSONB)
```

**IDP System → hs2_deliverables**:
```
IDP.DeliverableID     → hs2_deliverables.deliverable_id
IDP.Title             → hs2_deliverables.deliverable_name
IDP.Category          → hs2_deliverables.deliverable_type
IDP.DueDate           → hs2_deliverables.due_date
IDP.Status            → hs2_deliverables.completion_status
IDP.SubmittedDate     → hs2_deliverables.submitted_date
IDP.DocumentLink      → hs2_deliverables.document_reference
```

**ERP → hs2_costs**:
```
ERP.CostCode          → hs2_costs.cost_line_id
ERP.BudgetAmount      → hs2_costs.budget
ERP.ActualCost        → hs2_costs.cost_to_date
ERP.Variance          → hs2_costs.variance_pct
ERP.LastUpdate        → hs2_costs.last_updated
```

---

## 🔄 Data Migration Strategy

### Phase 1: Placeholder to Production

**Current (Demo)**:
- 50 synthetic assets
- Deterministic random data
- No external dependencies

**Target (Production)**:
- 2M+ real assets from AIMS
- Real-time sync with AIMS API
- Historical data import (3+ years)

**Migration Steps**:
1. **ETL Pipeline Setup** (Airbyte, Fivetran, or custom)
2. **AIMS API Integration** (hourly sync)
3. **Data Validation** (reconcile counts, check referential integrity)
4. **Incremental Updates** (CDC - Change Data Capture)
5. **Backfill Historical Data** (3 years of evaluations)

### Phase 2: Microsoft Fabric Integration

**Target Architecture**:
```
AIMS/IDP/ERP → Microsoft Fabric Lakehouse → Fabric KQL DB → Power BI
                    ↓
              Our TAEM Engine (Python)
                    ↓
              OneLake (storage)
```

**Schema Changes**:
- Move from PostgreSQL to Fabric Lakehouse (Delta tables)
- Use Fabric KQL DB for real-time queries
- Keep same table structure (portable SQL)
- Leverage Fabric notebooks for TAEM evaluation (PySpark + Python)

---

## 📚 References

- **PostgreSQL Documentation**: https://postgresql.org/docs/
- **JSONB Data Type**: https://postgresql.org/docs/current/datatype-json.html
- **Materialized Views**: https://postgresql.org/docs/current/rules-materializedviews.html
- **PAS 128:2022**: Specification for underground utility detection
- **CDM 2015**: Construction (Design and Management) Regulations 2015

---

**Version History**:
- v1.0 (2024-11-25): Initial release

**Maintainer**: Infrastructure Intelligence Platform Team
**Related Docs**:
- [TAEM_RULES_CATALOG.md](./TAEM_RULES_CATALOG.md)
- [HS2_ORCHESTRATION_PLAN.md](../HS2_ORCHESTRATION_PLAN.md)
