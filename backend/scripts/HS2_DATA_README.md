# HS2 Assurance Intelligence Data Generation

This directory contains scripts for generating, seeding, and evaluating HS2 infrastructure asset data.

## Overview

The HS2 Assurance Intelligence Demonstrator uses a comprehensive data model with:
- **50 Assets** (viaducts, bridges, tunnels, OLE masts)
- **300-750 Deliverables** (design certificates, test reports, sign-offs)
- **50 Cost Records** (budget tracking with variance)
- **100-250 Certificates** (validity and expiry tracking)

### Status Distribution
- **Ready**: 10 assets (20%)
- **Not Ready**: 25 assets (50%)
- **At Risk**: 15 assets (30%)

## Scripts

### 1. `generate_placeholder_data.py`
Generates realistic synthetic data with coherent relationships.

**Usage:**
```bash
# From project root
cd backend
python scripts/generate_placeholder_data.py
```

**Output:**
- `placeholder_data/assets.json` - 50 assets
- `placeholder_data/deliverables.json` - 300-750 deliverables
- `placeholder_data/costs.json` - 50 cost records
- `placeholder_data/certificates.json` - 100-250 certificates

**Features:**
- Deterministic (uses random seed for reproducibility)
- Status-aware data generation (Ready assets have complete deliverables, etc.)
- Realistic HS2 terminology and values
- Coherent relationships between entities

### 2. `database/create_hs2_tables.py`
Creates HS2 database schema.

**Usage:**
```bash
# From Docker container
docker compose exec backend python scripts/database/create_hs2_tables.py

# Or locally
python backend/scripts/database/create_hs2_tables.py
```

**Tables Created:**
- `hs2_assets` - Infrastructure assets
- `hs2_deliverables` - Project deliverables
- `hs2_costs` - Cost tracking
- `hs2_certificates` - Certificate management
- `hs2_taem_rules` - TAEM evaluation rules
- `hs2_evaluations` - Evaluation history (audit trail)

### 3. `seed_database.py`
Loads JSON data into PostgreSQL database.

**Usage:**
```bash
# From Docker container
docker compose exec backend python scripts/seed_database.py

# Or locally
python backend/scripts/seed_database.py
```

**Features:**
- Idempotent (can run multiple times safely)
- Handles foreign key relationships
- Maps string asset IDs to UUIDs
- Comprehensive logging
- Data verification after seeding

### 4. `evaluate_all_assets.py`
Runs TAEM evaluation on all assets.

**Usage:**
```bash
# From Docker container
docker compose exec backend python scripts/evaluate_all_assets.py

# Force re-evaluation
docker compose exec backend python scripts/evaluate_all_assets.py --force

# Or locally
python backend/scripts/evaluate_all_assets.py
```

**Features:**
- Evaluates all assets against TAEM rules
- Stores evaluation results for audit trail
- Updates asset readiness status
- Comprehensive statistics and reporting
- Refreshes materialized views

## Complete Setup Workflow

### Step 1: Create Database Tables
```bash
docker compose exec backend python scripts/database/create_hs2_tables.py
```

### Step 2: Generate Placeholder Data
```bash
docker compose exec backend python scripts/generate_placeholder_data.py
```

### Step 3: Seed Database
```bash
docker compose exec backend python scripts/seed_database.py
```

### Step 4: Run TAEM Evaluation
```bash
docker compose exec backend python scripts/evaluate_all_assets.py
```

## Data Model

### Assets
```json
{
  "asset_id": "VA-001",
  "asset_type": "Viaduct",
  "route_section": "London-Euston",
  "contractor": "JV-Alpha",
  "readiness_status": "Not Ready",
  "taem_evaluation_score": 45.23,
  "metadata": {
    "height_m": 15.2,
    "span_m": 45.0,
    "construction_start": "2023-01-15"
  }
}
```

### Deliverables
```json
{
  "deliverable_id": "DEL-VA-001-01",
  "asset_id": "VA-001",
  "deliverable_name": "Design Certificate - Structural Integrity",
  "deliverable_type": "Design Certificate",
  "status": "Overdue",
  "priority": "Critical",
  "due_date": "2024-03-15T00:00:00",
  "days_overdue": 45
}
```

### Costs
```json
{
  "cost_line_id": "COST-VA-001",
  "asset_id": "VA-001",
  "budget_amount": 5000000.00,
  "actual_amount": 6200000.00,
  "variance_pct": 24.0,
  "status": "Over Budget"
}
```

### Certificates
```json
{
  "certificate_id": "CERT-VA-001-01",
  "asset_id": "VA-001",
  "certificate_type": "Welding Qualification",
  "status": "Expired",
  "expiry_date": "2024-10-15T00:00:00",
  "days_until_expiry": -15
}
```

## TAEM Evaluation Rules

### Default Rules
1. **TAEM-001** - Critical Deliverables Completeness (Weight: 30%)
   - All critical deliverables must be approved
   
2. **TAEM-002** - Major Deliverables Completeness (Weight: 15%)
   - At least 90% of major deliverables submitted
   
3. **TAEM-003** - No Overdue Critical Deliverables (Weight: 20%)
   - Zero overdue critical deliverables
   
4. **TAEM-004** - Cost Variance Within Tolerance (Weight: 15%)
   - Cost variance within ±10%
   
5. **TAEM-005** - All Certificates Valid (Weight: 15%)
   - 100% of certificates must be valid
   
6. **TAEM-006** - No Expiring Certificates (Weight: 5%)
   - No certificates expiring within 30 days

### Readiness Classification
- **Ready**: Score ≥ 85% AND no critical failures
- **At Risk**: Score ≥ 65% OR some major issues
- **Not Ready**: Score < 65% OR critical failures

## Data Coherence

The data generator ensures logical consistency:

### "Ready" Asset Profile
- All critical deliverables approved
- Budget variance: -10% to +5%
- All certificates valid, none expiring soon
- TAEM Score: 85-100

### "Not Ready" Asset Profile
- 1-2 critical deliverables missing or overdue
- Budget variance: +10% to +30% (cost overruns)
- 1-2 expired certificates
- TAEM Score: 30-60

### "At Risk" Asset Profile
- Critical deliverables delayed but on track
- Budget variance: -5% to +15%
- Valid certificates but some expiring within 30 days
- TAEM Score: 60-75

## Verification Queries

### Check Asset Distribution
```sql
SELECT readiness_status, COUNT(*) as count
FROM hs2_assets
GROUP BY readiness_status
ORDER BY readiness_status;
```

### Check Critical Deliverables
```sql
SELECT a.asset_id, a.readiness_status, 
       COUNT(*) as total_deliverables,
       SUM(CASE WHEN d.priority = 'Critical' THEN 1 ELSE 0 END) as critical_deliverables,
       SUM(CASE WHEN d.priority = 'Critical' AND d.status = 'Approved' THEN 1 ELSE 0 END) as approved_critical
FROM hs2_assets a
LEFT JOIN hs2_deliverables d ON a.id = d.asset_id
GROUP BY a.asset_id, a.readiness_status
ORDER BY a.asset_id;
```

### Check Evaluation Results
```sql
SELECT asset_id, evaluation_date, overall_score, readiness_status,
       rules_evaluated, rules_passed, rules_failed
FROM hs2_evaluations
ORDER BY evaluation_date DESC
LIMIT 10;
```

## Troubleshooting

### Issue: "File not found" error in seed_database.py
**Solution:** Run `generate_placeholder_data.py` first to create JSON files.

### Issue: "Table already exists" error
**Solution:** Tables are created with `checkfirst=True`, so this shouldn't happen. If it does, the table already exists and you can proceed.

### Issue: Foreign key constraint violation
**Solution:** Ensure you seed in the correct order:
1. Assets (parent table)
2. Deliverables, Costs, Certificates (child tables)

### Issue: Evaluation scores don't match expected distribution
**Solution:** Run with `--force` flag to recalculate:
```bash
docker compose exec backend python scripts/evaluate_all_assets.py --force
```

## API Integration

Once data is seeded, access via API:

### List Assets
```bash
curl http://localhost:8002/api/v1/hs2/assets?status=Not%20Ready&limit=10
```

### Get Asset Details
```bash
curl http://localhost:8002/api/v1/hs2/assets/{asset_id}
```

### Trigger Evaluation
```bash
curl -X POST http://localhost:8002/api/v1/hs2/taem/evaluate-all
```

### View Dashboard
```bash
curl http://localhost:8002/api/v1/hs2/dashboard/summary
```

## Development

### Modifying Data Distribution
Edit `generate_placeholder_data.py`:
```python
STATUS_DISTRIBUTION = {
    "Ready": 15,      # Change to 30%
    "Not Ready": 20,  # Change to 40%
    "At Risk": 15,    # Change to 30%
}
```

### Adding New TAEM Rules
Edit `app/services/taem_engine.py` in the `create_default_rules()` method.

### Changing Asset Types
Edit `ASSET_CONFIG` in `generate_placeholder_data.py`:
```python
ASSET_CONFIG = {
    "Viaduct": {"count": 35, "prefix": "VA", "budget_range": (3_000_000, 8_000_000)},
    # Add more types...
}
```

## Performance

### Generation Time
- Data generation: ~1-2 seconds
- Database seeding: ~3-5 seconds
- TAEM evaluation: ~5-10 seconds for 50 assets

### Database Size
- 50 assets + related data: ~500KB
- With evaluation history: ~1MB (grows over time)

## Production Considerations

1. **Random Seed**: Change `RANDOM_SEED` in production for different data
2. **Data Validation**: Add schema validation before seeding
3. **Backup**: Backup database before seeding
4. **Logging**: Configure loguru for production logging levels
5. **Error Handling**: Add retry logic for database operations
6. **Monitoring**: Track evaluation performance metrics

---

**Last Updated**: 2024-11-25
**Version**: 1.0.0
