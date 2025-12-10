# HS2 Assurance Intelligence Demonstrator - Demo Script

**Demo Duration**: 3-5 minutes
**Audience**: HS2 Leadership, Project Managers, Technical Teams
**Objective**: Demonstrate explainable asset readiness tracking with full audit trail

---

## 🎯 Demo Overview

This demonstration showcases an **AI-native assurance intelligence platform** that answers the critical question: **"Which assets are Safe, Complete, Compliant, and Ready for Handover?"**

**Key Messages**:
1. **Explainable**: Every decision has transparent evidence (no black box)
2. **Auditable**: Full 7-year compliance trail (CDM 2015)
3. **Tinkerable**: Business users can modify rules without IT involvement
4. **Read-Only**: Works with existing data, no process changes needed

---

## 🎬 Scenario 1: Dashboard Summary (30 seconds)

### Narrative

> "We're tracking 50 HS2 viaducts and bridges across 5 route sections and 3 joint ventures. This dashboard gives us a unified view of asset readiness across fragmented enterprise systems."

### Steps

1. **Navigate to Dashboard**
   ```
   URL: http://localhost:3003/hs2/dashboard
   or
   API: curl http://localhost:8002/api/v1/hs2/dashboard/summary | jq
   ```

2. **Show Summary Cards**
   - **Total Assets**: 50
   - **Ready**: 10 (20%) - Green card
   - **Not Ready**: 40 (80%) - Red card
   - **At Risk**: 0 (0%) - Orange card

3. **Show Breakdown Charts**
   - **By Contractor**:
     ```
     JV-Alpha:   17 assets (4 Ready, 13 Not Ready)
     JV-Bravo:   17 assets (3 Ready, 14 Not Ready)
     JV-Charlie: 16 assets (3 Ready, 13 Not Ready)
     ```

   - **By Asset Type**:
     ```
     Viaducts: 30 assets (6 Ready, 24 Not Ready)
     Bridges:  15 assets (3 Ready, 12 Not Ready)
     Tunnels:  3 assets (1 Ready, 2 Not Ready)
     OLE Masts: 2 assets (0 Ready, 2 Not Ready)
     ```

   - **By Route Section**:
     ```
     London-Euston: 10 assets (2 Ready, 8 Not Ready)
     Old Oak Common: 10 assets (2 Ready, 8 Not Ready)
     Acton: 10 assets (2 Ready, 8 Not Ready)
     Northolt: 10 assets (2 Ready, 8 Not Ready)
     Denham: 10 assets (2 Ready, 8 Not Ready)
     ```

4. **Highlight Key Insight**
   > "80% of assets are flagged as Not Ready - this gives leadership early visibility into delivery risks."

### Expected Questions & Answers

**Q**: "Where does this data come from?"
**A**: "The platform ingests data from existing HS2 systems: AIMS for asset data, IDP systems for deliverables, ERP for costs, and document management systems for certificates. It's read-only - no changes to current processes."

**Q**: "How often is this updated?"
**A**: "Currently, data is refreshed daily. In production, we can sync hourly or trigger updates based on events in source systems."

---

## 🎬 Scenario 2: Investigate "Not Ready" Asset (60 seconds)

### Narrative

> "Let's investigate why Asset VA-007 is flagged as Not Ready. The system shows exactly which requirements are blocking handover and provides full evidence."

### Steps

1. **Navigate to Asset VA-007**
   ```
   URL: http://localhost:3003/hs2/assets/VA-007
   or
   API: curl http://localhost:8002/api/v1/hs2/assets/VA-007 | jq
   ```

2. **Show Asset Header**
   - **Asset ID**: VA-007
   - **Type**: Viaduct
   - **Route**: London-Euston
   - **Contractor**: JV-Alpha
   - **Status**: 🔴 **Not Ready** (red badge)
   - **TAEM Score**: 6.7/100

3. **Show Explainability Panel**

   **"Why is VA-007 Not Ready?"**

   The system shows 4 rule failures:

   **❌ R001: Critical Deliverables Present (Critical)**
   - **Outcome**: Fail
   - **Message**: "Missing critical deliverables: Design Certificate, Assurance Sign-off"
   - **Evidence**:
     ```json
     {
       "required": ["Design Certificate", "Assurance Sign-off", "Test Report - Concrete", "Test Report - Welding"],
       "found_complete": ["Test Report - Concrete", "Test Report - Welding"],
       "missing": ["Design Certificate", "Assurance Sign-off"]
     }
     ```

   **❌ R002: Cost Variance Within Tolerance (Major)**
   - **Outcome**: Fail
   - **Message**: "Cost variance 24.0% exceeds ±20% tolerance"
   - **Evidence**:
     ```json
     {
       "budget": 5000000.0,
       "cost_to_date": 6200000.0,
       "variance_pct": 24.0,
       "threshold": 20.0,
       "over_budget": true
     }
     ```

   **❌ R003: Certificate Validity (Major)**
   - **Outcome**: Fail
   - **Message**: "1 certificate(s) expired"
   - **Evidence**:
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

   **❌ R004: Schedule Adherence (Major)**
   - **Outcome**: Fail
   - **Message**: "2 deliverable(s) overdue"
   - **Evidence**:
     ```json
     {
       "total_deliverables": 12,
       "on_track": 8,
       "overdue": 2,
       "overdue_list": [
         {
           "deliverable_id": "DEL-VA-007-02",
           "deliverable_name": "Assurance Sign-off - Structural Safety",
           "due_date": "2024-10-10",
           "days_overdue": 46
         }
       ]
     }
     ```

4. **Navigate Through Tabs**

   - **Readiness Tab**: Show all 6 TAEM rules with Pass/Fail status
   - **Deliverables Tab**: List of 12 deliverables (2 overdue, 3 in progress, 7 complete)
   - **Costs Tab**: Budget £5.0M, Actual £6.2M, Variance +24%
   - **Certificates Tab**: 5 certificates (1 expired, 4 valid)
   - **History Tab**: Evaluation audit trail with timestamps

5. **Highlight Key Insight**
   > "Every 'Not Ready' status has specific, actionable evidence. Project managers know exactly what to fix: get Design Certificate approved, renew welding qualification, submit overdue Assurance Sign-off."

### Expected Questions & Answers

**Q**: "What if we disagree with the rules?"
**A**: "That's the beauty of explainable AI - you can see exactly how decisions are made and modify the rules. Let me show you in the next scenario."

**Q**: "Is this data real-time?"
**A**: "This demo uses placeholder data. In production, data would sync from source systems hourly or daily, depending on HS2 requirements."

---

## 🎬 Scenario 3: Audit Trail (30 seconds)

### Narrative

> "CDM 2015 requires 7+ year retention of assurance records. Every evaluation in this system is stored with full evidence, timestamps, and traceability."

### Steps

1. **Show Evaluation History**
   ```
   API: curl http://localhost:8002/api/v1/hs2/taem/evaluations/VA-007 | jq
   ```

2. **Display Recent Evaluations Table**

   | Evaluated At | Rule | Outcome | Score | Evidence |
   |-------------|------|---------|-------|----------|
   | 2024-11-25 14:30 | R001: Critical Deliverables | Fail | 0/50 | {...} |
   | 2024-11-25 14:30 | R002: Cost Variance | Fail | 0/25 | {...} |
   | 2024-11-25 14:30 | R003: Certificate Validity | Fail | 0/30 | {...} |
   | 2024-11-25 14:30 | R004: Schedule Adherence | Fail | 0/25 | {...} |
   | 2024-11-25 14:30 | R005: Documentation | Warning | 5/10 | {...} |
   | 2024-11-25 14:30 | R006: Quality Inspections | Warning | 5/10 | {...} |

3. **Show Raw Evidence JSON**

   Click on any row to expand evidence:
   ```json
   {
     "eval_id": "f6g7h8i9-...",
     "asset_id": "VA-007",
     "rule_id": "R001",
     "outcome": "Fail",
     "points_awarded": 0,
     "evidence": {
       "required": ["Design Certificate", "Assurance Sign-off"],
       "found_complete": [],
       "missing": ["Design Certificate", "Assurance Sign-off"]
     },
     "message": "Missing critical deliverables: Design Certificate, Assurance Sign-off",
     "evaluated_at": "2024-11-25T14:30:00Z",
     "evaluated_by": "system"
   }
   ```

4. **Database Query (Optional)**

   Show raw database query for technical audiences:
   ```sql
   SELECT
     re.evaluated_at,
     tr.rule_name,
     re.outcome,
     re.points_awarded,
     re.evidence,
     re.message
   FROM hs2_rule_evaluations re
   JOIN hs2_taem_rules tr ON re.rule_id = tr.id
   WHERE re.asset_id = (SELECT id FROM hs2_assets WHERE asset_id = 'VA-007')
   ORDER BY re.evaluated_at DESC;
   ```

5. **Highlight Key Insight**
   > "Every evaluation is immutable and stored forever. If auditors ask 'Why was VA-007 approved for handover on March 15, 2026?', we can show exactly what data was used and how the decision was made."

### Expected Questions & Answers

**Q**: "How long do you retain this data?"
**A**: "Indefinitely. Storage is cheap, and CDM 2015 requires 7+ years. This evidence protects HS2 from liability if issues arise post-handover."

**Q**: "What if someone disputes a decision?"
**A**: "The evidence is tamper-proof in the database. We log who triggered the evaluation, what data was used, and the exact outcome. Full transparency."

---

## 🎬 Scenario 4: Tinkerability (45 seconds)

### Narrative

> "HS2 business users can experiment with rules in real-time without waiting for IT. Let's demonstrate by tightening the cost variance threshold."

### Steps

1. **Show Current Rule Configuration**
   ```
   GET /api/v1/hs2/taem/rules/R002
   ```

   **Response**:
   ```json
   {
     "rule_id": "R002",
     "rule_name": "Cost Variance Within Tolerance",
     "description": "Ensure asset costs remain within acceptable budget variance thresholds",
     "severity": "Major",
     "weight": 25,
     "config": {
       "threshold_warning": 10,
       "threshold_fail": 20
     },
     "is_active": true
   }
   ```

2. **Explain Current Threshold**
   > "Currently, assets fail if cost variance exceeds ±20%. Let's say HS2 wants tighter cost control - we can change this to 15% instantly."

3. **Update Rule Configuration**
   ```bash
   curl -X PATCH http://localhost:8002/api/v1/hs2/taem/rules/R002 \
     -H "Content-Type: application/json" \
     -d '{
       "config": {
         "threshold_warning": 10,
         "threshold_fail": 15
       }
     }' | jq
   ```

4. **Re-Evaluate All Assets**
   ```bash
   curl -X POST http://localhost:8002/api/v1/hs2/taem/evaluate-all | jq
   ```

5. **Show Updated Dashboard**

   **Before**:
   - Ready: 10 (20%)
   - Not Ready: 40 (80%)

   **After** (with stricter 15% threshold):
   - Ready: 5 (10%)
   - Not Ready: 45 (90%)

   > "5 more assets are now flagged as Not Ready because they exceed the new 15% cost variance threshold."

6. **Show Which Assets Changed**

   Query assets that now fail R002 but previously passed:
   ```sql
   SELECT asset_id, contractor, variance_pct
   FROM hs2_assets a
   JOIN hs2_costs c ON a.id = c.asset_id
   WHERE c.variance_pct BETWEEN 15 AND 20
   ORDER BY c.variance_pct DESC;
   ```

   **Results**:
   ```
   VA-012 | JV-Alpha   | 18.5%
   BR-003 | JV-Bravo   | 17.2%
   VA-025 | JV-Charlie | 16.8%
   BR-009 | JV-Alpha   | 15.9%
   VA-033 | JV-Bravo   | 15.3%
   ```

7. **Revert Change (If Desired)**
   ```bash
   curl -X PATCH http://localhost:8002/api/v1/hs2/taem/rules/R002 \
     -H "Content-Type: application/json" \
     -d '{"config": {"threshold_fail": 20}}' | jq
   ```

8. **Highlight Key Insight**
   > "Business users can experiment with different thresholds to see impact on readiness. No code deployment, no IT tickets, instant feedback. This lets HS2 tune rules based on project phase, risk tolerance, or lessons learned."

### Expected Questions & Answers

**Q**: "Who can modify these rules?"
**A**: "That's configurable. Typically, we'd restrict rule changes to senior project managers or assurance leads with appropriate authentication and authorization."

**Q**: "What prevents someone from making the rules too lenient?"
**A**: "All rule changes are logged in the audit trail with timestamps and user IDs. Leadership can review rule change history and, if needed, require approval workflows before changes take effect."

**Q**: "Can we add new rules?"
**A**: "Absolutely. New rules can be added through the API or database. For complex logic, we'd work with HS2 to implement custom rule handlers, but simple threshold-based rules can be configured without code changes."

---

## 🎯 Alternative Demo Flow (Technical Audience)

For technical teams (IT, data architects, developers), emphasize:

### 1. Architecture (2 minutes)

**Show System Diagram**:
```
┌─────────────────────────────────────────────────────┐
│  HS2 Enterprise Systems                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────┐ │
│  │  AIMS   │  │   IDP   │  │   ERP   │  │  CDEs  │ │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬───┘ │
└───────┼───────────┼───────────┼───────────┼────────┘
        │           │           │           │
        └───────────┴───────────┴───────────┘
                    ↓ API Sync (Read-Only)
        ┌───────────────────────────────────────┐
        │  Assurance Intelligence Platform      │
        │  ┌─────────────┐  ┌─────────────────┐ │
        │  │  PostgreSQL │  │  TAEM Engine    │ │
        │  │  (6 tables) │  │  (6 rules)      │ │
        │  └──────┬──────┘  └────────┬────────┘ │
        │         │                  │          │
        │         └──────────┬───────┘          │
        │                    ↓                  │
        │         ┌──────────────────┐          │
        │         │   REST API       │          │
        │         │   (24 endpoints) │          │
        │         └──────────┬───────┘          │
        └────────────────────┼──────────────────┘
                             ↓
                ┌────────────────────────┐
                │   React Dashboard      │
                │   (Asset Readiness)    │
                └────────────────────────┘
```

**Key Points**:
- FastAPI async backend (Python 3.11+)
- PostgreSQL 16 with JSONB for evidence storage
- Materialized views for dashboard performance
- Explainable rule engine (no ML black box)
- RESTful API with full OpenAPI documentation

### 2. Code Walkthrough (3 minutes)

**Show TAEM Rule Implementation**:
```python
# backend/app/services/taem_engine.py
async def _evaluate_R002(self, asset: Asset) -> Dict:
    """Rule R002: Cost variance must be within tolerance (±20%)"""

    # Get cost data
    cost = await self.db.get(CostLine, asset_id=asset.id)

    if not cost:
        return {
            "outcome": "Warning",
            "points": 12,
            "evidence": {"cost_data_available": False},
            "message": "No cost data available"
        }

    # Get threshold from rule config (tinkerable!)
    threshold = self.rule.config.get("threshold_fail", 20.0)
    variance_pct = abs(cost.variance_pct)

    # Evaluate
    if variance_pct > threshold:
        return {
            "outcome": "Fail",
            "points": 0,
            "evidence": {
                "budget": float(cost.budget),
                "cost_to_date": float(cost.cost_to_date),
                "variance_pct": float(cost.variance_pct),
                "threshold": threshold
            },
            "message": f"Cost variance {variance_pct:.1f}% exceeds ±{threshold}% tolerance"
        }

    return {
        "outcome": "Pass",
        "points": 25,
        "evidence": {"variance_pct": float(cost.variance_pct)},
        "message": f"Cost variance {variance_pct:.1f}% within tolerance"
    }
```

**Emphasize**:
- Transparent logic (no neural networks)
- Evidence structure is clearly defined
- Thresholds come from database config (tinkerable)
- Human-readable messages
- Full type hints and error handling

### 3. Data Model (2 minutes)

**Show Database Schema**:
```sql
-- Core relationship
hs2_assets (1) ----< (N) hs2_deliverables
           (1) ----< (N) hs2_costs
           (1) ----< (N) hs2_certificates
           (1) ----< (N) hs2_rule_evaluations

-- Evidence stored as JSONB
CREATE TABLE hs2_rule_evaluations (
    id UUID PRIMARY KEY,
    asset_id UUID REFERENCES hs2_assets(id),
    rule_id UUID REFERENCES hs2_taem_rules(id),
    outcome VARCHAR(50) NOT NULL,
    points_awarded INTEGER DEFAULT 0,
    evidence JSONB NOT NULL,  -- ← Full audit trail
    message TEXT,
    evaluated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Materialized view for performance
CREATE MATERIALIZED VIEW hs2_asset_readiness_summary AS
SELECT
    a.asset_id,
    a.taem_evaluation_score,
    COUNT(CASE WHEN re.outcome = 'Fail' AND tr.severity = 'Critical' THEN 1 END) as critical_fails,
    c.variance_pct
FROM hs2_assets a
LEFT JOIN hs2_rule_evaluations re ON a.id = re.asset_id
LEFT JOIN hs2_taem_rules tr ON re.rule_id = tr.id
LEFT JOIN hs2_costs c ON a.id = c.asset_id
GROUP BY a.asset_id, a.taem_evaluation_score, c.variance_pct;
```

**Emphasize**:
- JSONB for flexible evidence storage
- Foreign keys ensure referential integrity
- Materialized views for dashboard speed
- Scalable to millions of records

---

## 📝 Demo Preparation Checklist

### Before Demo

- [ ] Run `./scripts/demo.sh` to ensure environment is ready
- [ ] Verify all 50 assets are in database
- [ ] Test all API endpoints with curl commands
- [ ] Ensure frontend is accessible (if implemented)
- [ ] Prepare backup queries in case of network issues
- [ ] Have database query tool open (psql or DataGrip)
- [ ] Load this script on secondary monitor/tablet

### During Demo

- [ ] Speak clearly and at moderate pace
- [ ] Pause for questions after each scenario
- [ ] Show evidence JSON for at least one rule failure
- [ ] Emphasize "explainable" and "auditable" repeatedly
- [ ] If technical audience, show code and database
- [ ] If business audience, focus on outcomes and insights

### After Demo

- [ ] Collect feedback and questions
- [ ] Share documentation links
- [ ] Provide API documentation URL
- [ ] Schedule follow-up if requested
- [ ] Note any feature requests or concerns

---

## 🎤 Key Talking Points

### For Leadership

> "This platform reduces time-to-insight from weeks to seconds. Instead of manually reviewing thousands of documents, leadership gets instant visibility into which assets are ready for handover and exactly what's blocking the others."

### For Project Managers

> "You get actionable intelligence. Not just 'VA-007 is Not Ready' - you see it's missing Design Certificate, Welding Qualification expired, and Assurance Sign-off is 46 days overdue. You know exactly what to chase."

### For Assurance Teams

> "Every decision is explainable and auditable. If someone challenges why an asset was approved, you have timestamped evidence showing exactly what data was used and how the rules were applied."

### For IT/Technical Teams

> "It's API-first, cloud-native, and integrates with existing systems via read-only APIs. No complex data migration, no process changes. The TAEM engine is open-source Python - you can inspect, modify, and extend it."

---

## 🚨 Troubleshooting

### If Demo Environment Fails

**Backup Option 1**: Use API directly with curl
```bash
curl http://localhost:8002/api/v1/hs2/dashboard/summary | jq
curl http://localhost:8002/api/v1/hs2/assets/VA-007 | jq
```

**Backup Option 2**: Show database directly
```bash
docker compose exec postgres psql -U gpr_user -d gpr_db
SELECT * FROM hs2_assets WHERE readiness_status = 'Not Ready' LIMIT 5;
```

**Backup Option 3**: Show static screenshots
- Keep screenshots of dashboard, asset detail, audit trail in presentation

### If Questions Go Off-Topic

**Redirect Strategy**:
> "That's a great question about [topic]. Let me note that for follow-up, and for now let me show you [back to demo scenario]..."

**Common Off-Topic Questions**:
- **"Can this do X?"** → "Not in this demo, but the architecture supports it. Let's discuss after."
- **"How much does it cost?"** → "Let's focus on the value first, then we can discuss commercial models."
- **"When can we deploy?"** → "This is a demonstrator. Production deployment requires integration planning, which we can scope after this demo."

---

## 📊 Success Metrics for Demo

### Immediate Feedback (During Demo)

- ✅ Audience asks clarifying questions (engagement)
- ✅ Audience nods when showing explainability panel (understanding)
- ✅ Someone says "That's useful" or "We need this" (value recognition)
- ✅ Technical questions about integration (buy-in from IT)

### Post-Demo Outcomes (Within 1 Week)

- ✅ Request for follow-up meeting
- ✅ Request for access to demo environment
- ✅ Questions about production deployment timeline
- ✅ Introduction to other stakeholders
- ✅ Request for integration with specific HS2 systems

---

**Demo Script Version**: 1.0
**Last Updated**: November 25, 2024
**Estimated Duration**: 3-5 minutes (core scenarios) or 7-10 minutes (with technical deep-dive)

**Related Documents**:
- [HS2_ORCHESTRATION_PLAN.md](../HS2_ORCHESTRATION_PLAN.md)
- [TAEM_RULES_CATALOG.md](./TAEM_RULES_CATALOG.md)
- [DATA_DICTIONARY.md](./DATA_DICTIONARY.md)
