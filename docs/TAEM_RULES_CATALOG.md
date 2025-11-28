# TAEM Rules Catalog - HS2 Assurance Intelligence

**Technical Assurance Evidence Model (TAEM)** - Explainable Rule Definitions

**Version**: 1.0
**Last Updated**: November 25, 2024
**Applicable To**: HS2 Viaducts, Bridges, Tunnels, OLE Masts

---

## 📋 Overview

The TAEM Rule Engine evaluates asset readiness based on **6 transparent, auditable rules** across four categories:

1. **Deliverable Completion** (Critical, 50 points)
2. **Cost Management** (Major, 25 points)
3. **Certificate Validity** (Major, 30 points)
4. **Schedule Adherence** (Major, 25 points)
5. **Documentation Completeness** (Minor, 10 points)
6. **Quality Inspections** (Minor, 10 points)

**Total Score**: 150 points (normalized to 0-100 scale)

**Status Classification**:
- **Ready**: Score ≥80 AND no Critical failures
- **At Risk**: Score 60-79 OR 1-2 Major failures
- **Not Ready**: Score <60 OR any Critical failure

---

## 🔍 Rule Definitions

### R001: Critical Deliverables Present

**Category**: Deliverable Completion
**Severity**: Critical
**Weight**: 50 points
**Applies To**: All asset types

#### Purpose
Ensure all critical deliverables required for asset handover are complete and approved.

#### Required Deliverables
- Design Certificate
- Assurance Sign-off
- Test Report - Concrete
- Test Report - Welding

#### Evaluation Logic

```python
def evaluate_R001(asset):
    required_types = [
        "Design Certificate",
        "Assurance Sign-off",
        "Test Report - Concrete",
        "Test Report - Welding"
    ]

    deliverables = get_deliverables(asset.id)

    found_complete = []
    missing = []

    for required_type in required_types:
        matching = [d for d in deliverables
                    if d.deliverable_type == required_type
                    and d.completion_status == "Complete"]

        if matching:
            found_complete.append(required_type)
        else:
            missing.append(required_type)

    if missing:
        return {
            "outcome": "Fail",
            "points": 0,
            "evidence": {
                "required": required_types,
                "found_complete": found_complete,
                "missing": missing
            },
            "message": f"Missing critical deliverables: {', '.join(missing)}"
        }

    return {
        "outcome": "Pass",
        "points": 50,
        "evidence": {
            "required": required_types,
            "found_complete": found_complete
        },
        "message": "All critical deliverables present and complete"
    }
```

#### Evidence Structure

**Pass Example**:
```json
{
  "rule_id": "R001",
  "outcome": "Pass",
  "points": 50,
  "evidence": {
    "required": [
      "Design Certificate",
      "Assurance Sign-off",
      "Test Report - Concrete",
      "Test Report - Welding"
    ],
    "found_complete": [
      "Design Certificate",
      "Assurance Sign-off",
      "Test Report - Concrete",
      "Test Report - Welding"
    ]
  },
  "message": "All critical deliverables present and complete"
}
```

**Fail Example**:
```json
{
  "rule_id": "R001",
  "outcome": "Fail",
  "points": 0,
  "evidence": {
    "required": ["Design Certificate", "Assurance Sign-off", ...],
    "found_complete": ["Test Report - Concrete", "Test Report - Welding"],
    "missing": ["Design Certificate", "Assurance Sign-off"]
  },
  "message": "Missing critical deliverables: Design Certificate, Assurance Sign-off"
}
```

#### Tinkerability

Adjust required deliverables in `config` JSONB:
```sql
UPDATE hs2_taem_rules
SET config = '{"required_types": ["Design Certificate", "Assurance Sign-off"]}'
WHERE rule_id = 'R001';
```

---

### R002: Cost Variance Within Tolerance

**Category**: Cost Management
**Severity**: Major
**Weight**: 25 points
**Applies To**: All asset types

#### Purpose
Ensure asset costs remain within acceptable budget variance thresholds.

#### Thresholds (Configurable)
- **Pass**: Variance ≤ 10%
- **Warning**: Variance 10-20%
- **Fail**: Variance > 20%

#### Evaluation Logic

```python
def evaluate_R002(asset):
    cost = get_cost_line(asset.id)

    if not cost:
        return {
            "outcome": "Warning",
            "points": 12,  # Half points
            "evidence": {"cost_data_available": False},
            "message": "No cost data available"
        }

    variance_pct = abs(cost.variance_pct)
    threshold_fail = 20.0
    threshold_warning = 10.0

    if variance_pct > threshold_fail:
        return {
            "outcome": "Fail",
            "points": 0,
            "evidence": {
                "budget": cost.budget,
                "cost_to_date": cost.cost_to_date,
                "variance_pct": cost.variance_pct,
                "threshold": threshold_fail,
                "over_budget": cost.variance_pct > 0
            },
            "message": f"Cost variance {variance_pct:.1f}% exceeds ±{threshold_fail}% tolerance"
        }
    elif variance_pct > threshold_warning:
        return {
            "outcome": "Warning",
            "points": 15,
            "evidence": {
                "budget": cost.budget,
                "cost_to_date": cost.cost_to_date,
                "variance_pct": cost.variance_pct,
                "threshold": threshold_warning
            },
            "message": f"Cost variance {variance_pct:.1f}% approaching limit"
        }

    return {
        "outcome": "Pass",
        "points": 25,
        "evidence": {
            "budget": cost.budget,
            "cost_to_date": cost.cost_to_date,
            "variance_pct": cost.variance_pct
        },
        "message": f"Cost variance {variance_pct:.1f}% within tolerance"
    }
```

#### Evidence Structure

**Fail Example**:
```json
{
  "rule_id": "R002",
  "outcome": "Fail",
  "points": 0,
  "evidence": {
    "budget": 5000000.0,
    "cost_to_date": 6200000.0,
    "variance_pct": 24.0,
    "threshold": 20.0,
    "over_budget": true
  },
  "message": "Cost variance 24.0% exceeds ±20% tolerance (Budget: £5.0M, Actual: £6.2M)"
}
```

#### Tinkerability

Adjust thresholds in `config` JSONB:
```sql
UPDATE hs2_taem_rules
SET config = '{
  "threshold_warning": 10,
  "threshold_fail": 15
}'
WHERE rule_id = 'R002';

-- After updating, re-evaluate all assets:
-- This will cause more assets to fail with stricter threshold
```

**Demo Scenario**: Change fail threshold from 20% to 15%, re-run evaluation, show 5 more assets become "Not Ready"

---

### R003: Certificate Validity

**Category**: Certificate Management
**Severity**: Major
**Weight**: 30 points
**Applies To**: Viaducts, Bridges (requires welding)

#### Purpose
Ensure all required certificates (welding qualifications, test certificates) are valid and not expired.

#### Evaluation Logic

```python
def evaluate_R003(asset):
    certificates = get_certificates(asset.id)

    if not certificates:
        return {
            "outcome": "Warning",
            "points": 15,
            "evidence": {"certificates_available": False},
            "message": "No certificate data available"
        }

    valid = []
    expired = []
    expiring_soon = []

    today = datetime.now().date()
    warning_window = timedelta(days=30)

    for cert in certificates:
        if cert.status == "Expired" or cert.expiry_date < today:
            expired.append({
                "cert_id": cert.certificate_id,
                "type": cert.certificate_type,
                "expiry_date": cert.expiry_date,
                "days_expired": (today - cert.expiry_date).days
            })
        elif cert.expiry_date < (today + warning_window):
            expiring_soon.append({
                "cert_id": cert.certificate_id,
                "type": cert.certificate_type,
                "expiry_date": cert.expiry_date,
                "days_until_expiry": (cert.expiry_date - today).days
            })
        else:
            valid.append(cert.certificate_id)

    if expired:
        return {
            "outcome": "Fail",
            "points": 0,
            "evidence": {
                "total_certificates": len(certificates),
                "valid": len(valid),
                "expired": len(expired),
                "expiring_soon": len(expiring_soon),
                "expired_list": expired
            },
            "message": f"{len(expired)} certificate(s) expired"
        }
    elif expiring_soon:
        return {
            "outcome": "Warning",
            "points": 20,
            "evidence": {
                "total_certificates": len(certificates),
                "valid": len(valid),
                "expiring_soon": len(expiring_soon),
                "expiring_list": expiring_soon
            },
            "message": f"{len(expiring_soon)} certificate(s) expiring within 30 days"
        }

    return {
        "outcome": "Pass",
        "points": 30,
        "evidence": {
            "total_certificates": len(certificates),
            "valid": len(valid)
        },
        "message": "All certificates valid"
    }
```

#### Evidence Structure

**Fail Example**:
```json
{
  "rule_id": "R003",
  "outcome": "Fail",
  "points": 0,
  "evidence": {
    "total_certificates": 5,
    "valid": 4,
    "expired": 1,
    "expiring_soon": 0,
    "expired_list": [
      {
        "cert_id": "CERT-VA-007-02",
        "type": "Welding Qualification",
        "expiry_date": "2024-10-15",
        "days_expired": 41
      }
    ]
  },
  "message": "1 certificate(s) expired"
}
```

#### Tinkerability

Adjust expiration warning window:
```sql
UPDATE hs2_taem_rules
SET config = '{"expiration_warning_days": 60}'
WHERE rule_id = 'R003';
```

---

### R004: Schedule Adherence

**Category**: Schedule Management
**Severity**: Major
**Weight**: 25 points
**Applies To**: All asset types

#### Purpose
Ensure deliverables are submitted on time with no overdue items.

#### Evaluation Logic

```python
def evaluate_R004(asset):
    deliverables = get_deliverables(asset.id)

    today = datetime.now().date()

    on_track = []
    delayed = []
    overdue = []

    for d in deliverables:
        if d.completion_status == "Overdue":
            days_overdue = (today - d.due_date).days
            overdue.append({
                "deliverable_id": d.deliverable_id,
                "deliverable_name": d.deliverable_name,
                "due_date": d.due_date,
                "days_overdue": days_overdue
            })
        elif d.completion_status == "In Progress" and d.due_date < (today + timedelta(days=7)):
            delayed.append({
                "deliverable_id": d.deliverable_id,
                "deliverable_name": d.deliverable_name,
                "due_date": d.due_date,
                "days_until_due": (d.due_date - today).days
            })
        else:
            on_track.append(d.deliverable_id)

    if overdue:
        return {
            "outcome": "Fail",
            "points": 0,
            "evidence": {
                "total_deliverables": len(deliverables),
                "on_track": len(on_track),
                "delayed": len(delayed),
                "overdue": len(overdue),
                "overdue_list": overdue
            },
            "message": f"{len(overdue)} deliverable(s) overdue"
        }
    elif delayed:
        return {
            "outcome": "Warning",
            "points": 15,
            "evidence": {
                "total_deliverables": len(deliverables),
                "on_track": len(on_track),
                "delayed": len(delayed),
                "delayed_list": delayed
            },
            "message": f"{len(delayed)} deliverable(s) at risk of delay"
        }

    return {
        "outcome": "Pass",
        "points": 25,
        "evidence": {
            "total_deliverables": len(deliverables),
            "on_track": len(on_track)
        },
        "message": "All deliverables on schedule"
    }
```

---

### R005: Documentation Completeness

**Category**: Documentation
**Severity**: Minor
**Weight**: 10 points
**Applies To**: All asset types

#### Purpose
Ensure all required documentation is submitted (not necessarily approved yet).

#### Evaluation Logic

```python
def evaluate_R005(asset):
    deliverables = get_deliverables(asset.id)

    total = len(deliverables)
    submitted = len([d for d in deliverables if d.submitted_date is not None])

    completeness_pct = (submitted / total * 100) if total > 0 else 0

    if completeness_pct >= 90:
        return {
            "outcome": "Pass",
            "points": 10,
            "evidence": {
                "total_deliverables": total,
                "submitted": submitted,
                "completeness_pct": completeness_pct
            },
            "message": f"Documentation {completeness_pct:.1f}% complete"
        }
    elif completeness_pct >= 70:
        return {
            "outcome": "Warning",
            "points": 5,
            "evidence": {
                "total_deliverables": total,
                "submitted": submitted,
                "completeness_pct": completeness_pct
            },
            "message": f"Documentation {completeness_pct:.1f}% complete (approaching threshold)"
        }

    return {
        "outcome": "Fail",
        "points": 0,
        "evidence": {
            "total_deliverables": total,
            "submitted": submitted,
            "completeness_pct": completeness_pct
        },
        "message": f"Documentation only {completeness_pct:.1f}% complete"
    }
```

---

### R006: Quality Inspections Complete

**Category**: Quality Assurance
**Severity**: Minor
**Weight**: 10 points
**Applies To**: All asset types

#### Purpose
Ensure all required quality inspections (QA, NDT, etc.) are complete.

#### Evaluation Logic

```python
def evaluate_R006(asset):
    deliverables = get_deliverables(asset.id)

    qa_deliverables = [d for d in deliverables
                       if "QA" in d.deliverable_type
                       or "Inspection" in d.deliverable_type]

    if not qa_deliverables:
        return {
            "outcome": "Warning",
            "points": 5,
            "evidence": {"qa_deliverables_found": False},
            "message": "No QA inspection records found"
        }

    total = len(qa_deliverables)
    complete = len([d for d in qa_deliverables if d.completion_status == "Complete"])

    completeness_pct = (complete / total * 100)

    if completeness_pct == 100:
        return {
            "outcome": "Pass",
            "points": 10,
            "evidence": {
                "total_qa_inspections": total,
                "complete": complete
            },
            "message": "All QA inspections complete"
        }

    return {
        "outcome": "Fail",
        "points": 0,
        "evidence": {
            "total_qa_inspections": total,
            "complete": complete,
            "completeness_pct": completeness_pct
        },
        "message": f"Only {complete}/{total} QA inspections complete"
    }
```

---

## 📊 Scoring Algorithm

### Weighted Score Calculation

```python
def calculate_weighted_score(rule_results):
    total_points_possible = 150  # Sum of all rule weights
    total_points_awarded = sum(r["points"] for r in rule_results)

    # Normalize to 0-100 scale
    normalized_score = (total_points_awarded / total_points_possible) * 100

    return round(normalized_score, 2)
```

### Status Classification

```python
def classify_readiness_status(score, rule_results):
    # Check for critical failures
    critical_fails = [r for r in rule_results
                      if r["severity"] == "Critical" and r["outcome"] == "Fail"]

    if critical_fails:
        return "Not Ready"  # Any critical failure = Not Ready

    # Check score thresholds
    if score >= 80:
        return "Ready"
    elif score >= 60:
        return "At Risk"
    else:
        return "Not Ready"
```

### Example Calculation

**Asset VA-023 (Ready)**:
```
R001: Pass  = 50 points (Critical Deliverables)
R002: Pass  = 25 points (Cost Variance 5%)
R003: Pass  = 30 points (All Certificates Valid)
R004: Pass  = 25 points (No Overdue)
R005: Pass  = 10 points (95% Documentation)
R006: Pass  = 10 points (All QA Complete)

Total: 150/150 points = 100% → Status: Ready
```

**Asset VA-007 (Not Ready)**:
```
R001: Fail    = 0 points  (Missing Design Certificate)
R002: Fail    = 0 points  (Cost Variance 24%)
R003: Fail    = 0 points  (1 Expired Certificate)
R004: Fail    = 0 points  (2 Overdue Deliverables)
R005: Warning = 5 points  (75% Documentation)
R006: Warning = 5 points  (3/5 QA Complete)

Total: 10/150 points = 6.7% → Status: Not Ready
```

---

## 🔧 Tinkerability Guide

### Modifying Rule Thresholds

Rules are stored in `hs2_taem_rules` table with configurable `config` JSONB column:

```sql
-- View current configuration
SELECT rule_id, rule_name, config
FROM hs2_taem_rules;

-- Adjust R002 cost variance threshold (20% → 15%)
UPDATE hs2_taem_rules
SET config = jsonb_set(
    COALESCE(config, '{}'::jsonb),
    '{threshold_fail}',
    '15'::jsonb
)
WHERE rule_id = 'R002';

-- Adjust R003 certificate expiration warning (30 days → 60 days)
UPDATE hs2_taem_rules
SET config = jsonb_set(
    COALESCE(config, '{}'::jsonb),
    '{expiration_warning_days}',
    '60'::jsonb
)
WHERE rule_id = 'R003';

-- Re-evaluate all assets with new thresholds
-- (Call via API: POST /api/v1/hs2/taem/evaluate-all)
```

### Adding Custom Rules

To add a new rule (e.g., R007: BIM Model Submission):

1. **Insert rule definition**:
```sql
INSERT INTO hs2_taem_rules (rule_id, rule_name, description, severity, weight, config)
VALUES (
    'R007',
    'BIM Model Submission',
    'Ensure IFC model is submitted and validated',
    'Critical',
    40,
    '{"required_lod": "LOD 300"}'::jsonb
);
```

2. **Implement rule handler** in `taem_engine.py`:
```python
async def _evaluate_R007(self, asset: Asset) -> Dict:
    """Rule R007: BIM Model Submission"""
    # Implementation logic here
    pass
```

3. **Re-register rule** in engine initialization

---

## 📈 Audit Trail

Every evaluation is stored in `hs2_rule_evaluations` table:

```sql
-- View evaluation history for an asset
SELECT
    re.evaluated_at,
    tr.rule_name,
    re.outcome,
    re.points_awarded,
    re.evidence
FROM hs2_rule_evaluations re
JOIN hs2_taem_rules tr ON re.rule_id = tr.id
WHERE re.asset_id = (SELECT id FROM hs2_assets WHERE asset_id = 'VA-007')
ORDER BY re.evaluated_at DESC;

-- Find all assets that failed R001 (Critical Deliverables)
SELECT DISTINCT a.asset_id, a.route_section, a.contractor
FROM hs2_rule_evaluations re
JOIN hs2_assets a ON re.asset_id = a.id
JOIN hs2_taem_rules tr ON re.rule_id = tr.id
WHERE tr.rule_id = 'R001' AND re.outcome = 'Fail'
ORDER BY a.asset_id;
```

---

## 🎯 Demo Scenarios Using Rules

### Scenario 1: Show Explainability

**Narrative**: "Why is VA-007 Not Ready?"

**Expected Output**:
```json
{
  "asset_id": "VA-007",
  "overall_status": "Not Ready",
  "taem_score": 6.7,
  "failed_rules": [
    {
      "rule_id": "R001",
      "rule_name": "Critical Deliverables Present",
      "severity": "Critical",
      "outcome": "Fail",
      "message": "Missing critical deliverables: Design Certificate, Assurance Sign-off",
      "evidence": {
        "missing": ["Design Certificate", "Assurance Sign-off"]
      }
    },
    {
      "rule_id": "R002",
      "rule_name": "Cost Variance Within Tolerance",
      "severity": "Major",
      "outcome": "Fail",
      "message": "Cost variance 24.0% exceeds ±20% tolerance",
      "evidence": {
        "budget": 5000000.0,
        "cost_to_date": 6200000.0,
        "variance_pct": 24.0
      }
    }
  ]
}
```

### Scenario 2: Demonstrate Tinkerability

**Steps**:
1. Show current threshold: R002 = 20%
2. Update: `PATCH /api/v1/hs2/taem/rules/R002` with `{"config": {"threshold_fail": 15}}`
3. Re-evaluate: `POST /api/v1/hs2/taem/evaluate-all`
4. Show results: 5 more assets now "Not Ready" due to stricter cost control

---

## 📚 References

- **PAS 128:2022**: Specification for underground utility detection, verification and location
- **CDM 2015**: Construction (Design and Management) Regulations 2015
- **HS2 TAEM Framework**: Technical Assurance Evidence Model (internal documentation)
- **ISO 9001**: Quality management systems requirements

---

**Version History**:
- v1.0 (2024-11-25): Initial release with 6 core rules
- v1.1 (planned): Add BIM validation rules, LiDAR progress tracking

**Maintainer**: Infrastructure Intelligence Platform Team
**Contact**: See [HS2_ORCHESTRATION_PLAN.md](../HS2_ORCHESTRATION_PLAN.md)
