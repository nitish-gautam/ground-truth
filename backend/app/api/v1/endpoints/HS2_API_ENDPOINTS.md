# HS2 Assurance Intelligence Demonstrator - API Endpoints

## Overview

This document provides comprehensive documentation for the HS2 Assurance Intelligence Demonstrator API endpoints. The API enables management of infrastructure assets, deliverables, TAEM rule evaluation, and dashboard analytics.

## Base URL

```
http://localhost:8000/api/v1/hs2
```

## Authentication

*Note: Authentication implementation pending. All endpoints currently accessible without authentication.*

---

## Asset Management Endpoints

### 1. List All Assets

**Endpoint:** `GET /api/v1/hs2/assets`

**Description:** Retrieve a paginated list of HS2 infrastructure assets with optional filtering.

**Query Parameters:**
- `skip` (integer, optional): Number of items to skip for pagination (default: 0)
- `limit` (integer, optional): Number of items to return (default: 50, max: 100)
- `asset_type` (string, optional): Filter by asset type (e.g., "Viaduct", "Bridge", "Tunnel")
- `status` (string, optional): Filter by readiness status ("Ready", "Not Ready", "At Risk")
- `contractor` (string, optional): Filter by contractor name
- `route_section` (string, optional): Filter by route section (e.g., "Phase 1", "Phase 2a")

**Response:** `AssetPaginatedResponse`

**Example Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/hs2/assets?limit=10&status=At%20Risk&asset_type=Viaduct"
```

**Example Response:**
```json
{
  "total": 50,
  "skip": 0,
  "limit": 10,
  "items": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "asset_id": "BR001-N1-V01",
      "asset_name": "London-Birmingham Viaduct Section 1",
      "asset_type": "Viaduct",
      "route_section": "Phase 1",
      "contractor": "Balfour Beatty",
      "design_status": "Approved",
      "construction_status": "In Progress",
      "planned_completion_date": "2025-06-30T00:00:00Z",
      "readiness_status": "At Risk",
      "taem_evaluation_score": 68.5,
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-11-20T14:22:00Z"
    }
  ]
}
```

---

### 2. Get Asset Details

**Endpoint:** `GET /api/v1/hs2/assets/{asset_id}`

**Description:** Retrieve detailed information for a specific asset including readiness summary.

**Path Parameters:**
- `asset_id` (UUID, required): Asset unique identifier

**Response:** `AssetDetailResponse`

**Example Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/hs2/assets/123e4567-e89b-12d3-a456-426614174000"
```

**Example Response:**
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "asset_id": "BR001-N1-V01",
  "asset_name": "London-Birmingham Viaduct Section 1",
  "asset_type": "Viaduct",
  "route_section": "Phase 1",
  "contractor": "Balfour Beatty",
  "readiness_status": "At Risk",
  "taem_evaluation_score": 68.5,
  "readiness_summary": {
    "deliverables_submitted": 17,
    "deliverables_required": 20,
    "deliverables_completion_pct": 85.0,
    "certificates_issued": 8,
    "certificates_required": 10,
    "certificates_completion_pct": 80.0,
    "cost_variance_pct": 12.5,
    "schedule_variance_days": 14,
    "critical_risks": 2,
    "major_risks": 3,
    "minor_risks": 5,
    "overall_readiness": "At Risk",
    "taem_score": 68.5,
    "last_evaluation": "2024-11-20T14:22:00Z"
  }
}
```

---

### 3. Evaluate Asset (Trigger TAEM Evaluation)

**Endpoint:** `POST /api/v1/hs2/assets/{asset_id}/evaluate`

**Description:** Trigger TAEM rule evaluation for a specific asset. This evaluates all active rules and calculates overall readiness.

**Path Parameters:**
- `asset_id` (UUID, required): Asset unique identifier

**Request Body:** `AssetEvaluationRequest`
```json
{
  "force_refresh": false
}
```

**Response:** `AssetEvaluationResponse`

**Example Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/hs2/assets/123e4567-e89b-12d3-a456-426614174000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{"force_refresh": true}'
```

**Example Response:**
```json
{
  "id": "789e4567-e89b-12d3-a456-426614174999",
  "asset_id": "123e4567-e89b-12d3-a456-426614174000",
  "evaluation_date": "2024-11-25T10:30:00Z",
  "overall_score": 68.5,
  "readiness_status": "At Risk",
  "rules_evaluated": 12,
  "rules_passed": 8,
  "rules_failed": 4,
  "rule_results": [
    {
      "rule_code": "TAEM-001",
      "rule_name": "Deliverables Completeness Check",
      "status": "Pass",
      "score": 85.0,
      "weight": 0.25,
      "weighted_score": 21.25,
      "message": "17 of 20 deliverables submitted (85%)",
      "details": {
        "submitted": 17,
        "required": 20,
        "completion_pct": 85.0
      }
    },
    {
      "rule_code": "TAEM-002",
      "rule_name": "Cost Variance Check",
      "status": "Fail",
      "score": 45.0,
      "weight": 0.20,
      "weighted_score": 9.0,
      "message": "Cost variance 12.5% exceeds threshold of 10%",
      "details": {
        "variance_pct": 12.5,
        "threshold": 10.0
      }
    }
  ],
  "created_at": "2024-11-25T10:30:00Z"
}
```

---

### 4. Get Asset Deliverables

**Endpoint:** `GET /api/v1/hs2/assets/{asset_id}/deliverables`

**Description:** Retrieve all deliverables associated with a specific asset.

**Path Parameters:**
- `asset_id` (UUID, required): Asset unique identifier

**Response:** `List[DeliverableResponse]`

**Example Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/hs2/assets/123e4567-e89b-12d3-a456-426614174000/deliverables"
```

---

### 5. Get Asset Costs

**Endpoint:** `GET /api/v1/hs2/assets/{asset_id}/costs`

**Description:** Retrieve cost tracking information for a specific asset.

**Path Parameters:**
- `asset_id` (UUID, required): Asset unique identifier

**Response:** `List[CostResponse]`

**Example Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/hs2/assets/123e4567-e89b-12d3-a456-426614174000/costs"
```

---

### 6. Get Asset Certificates

**Endpoint:** `GET /api/v1/hs2/assets/{asset_id}/certificates`

**Description:** Retrieve all certificates for a specific asset.

**Path Parameters:**
- `asset_id` (UUID, required): Asset unique identifier

**Response:** `List[CertificateResponse]`

**Example Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/hs2/assets/123e4567-e89b-12d3-a456-426614174000/certificates"
```

---

### 7. Get Asset Evaluation History

**Endpoint:** `GET /api/v1/hs2/assets/{asset_id}/evaluations`

**Description:** Retrieve evaluation history for a specific asset (audit trail).

**Path Parameters:**
- `asset_id` (UUID, required): Asset unique identifier

**Query Parameters:**
- `limit` (integer, optional): Number of evaluations to return (default: 10, max: 100)

**Response:** `List[AssetEvaluationResponse]`

**Example Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/hs2/assets/123e4567-e89b-12d3-a456-426614174000/evaluations?limit=20"
```

---

## Deliverable Management Endpoints

### 8. List All Deliverables

**Endpoint:** `GET /api/v1/hs2/deliverables`

**Description:** Retrieve a paginated list of all deliverables with filtering options.

**Query Parameters:**
- `skip` (integer, optional): Pagination offset (default: 0)
- `limit` (integer, optional): Items per page (default: 50, max: 100)
- `status` (string, optional): Filter by status ("Submitted", "Pending", "Approved", "Rejected")
- `deliverable_type` (string, optional): Filter by deliverable type
- `asset_id` (UUID, optional): Filter by asset ID
- `overdue_only` (boolean, optional): Show only overdue deliverables (default: false)

**Response:** `DeliverablePaginatedResponse`

**Example Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/hs2/deliverables?overdue_only=true&limit=20"
```

---

### 9. Get Deliverable Details

**Endpoint:** `GET /api/v1/hs2/deliverables/{deliverable_id}`

**Description:** Retrieve detailed information for a specific deliverable.

**Path Parameters:**
- `deliverable_id` (UUID, required): Deliverable unique identifier

**Response:** `DeliverableResponse`

---

### 10. Create Deliverable

**Endpoint:** `POST /api/v1/hs2/deliverables`

**Description:** Create a new deliverable.

**Request Body:** `DeliverableCreate`
```json
{
  "asset_id": "123e4567-e89b-12d3-a456-426614174000",
  "deliverable_type": "Design Report",
  "deliverable_name": "Structural Design Report - Section 1",
  "required_by_date": "2025-03-31T00:00:00Z",
  "responsible_party": "Design Team Lead"
}
```

**Response:** `DeliverableResponse` (201 Created)

---

### 11. Update Deliverable

**Endpoint:** `PUT /api/v1/hs2/deliverables/{deliverable_id}`

**Description:** Update deliverable information (e.g., mark as submitted, update approval status).

**Path Parameters:**
- `deliverable_id` (UUID, required): Deliverable unique identifier

**Request Body:** `DeliverableUpdate` (all fields optional)
```json
{
  "status": "Submitted",
  "submission_date": "2024-11-25T10:00:00Z",
  "approval_status": "Pending"
}
```

**Response:** `DeliverableResponse`

---

### 12. Delete Deliverable

**Endpoint:** `DELETE /api/v1/hs2/deliverables/{deliverable_id}`

**Description:** Delete a deliverable (cannot be undone).

**Path Parameters:**
- `deliverable_id` (UUID, required): Deliverable unique identifier

**Response:** 204 No Content

---

### 13. Get Deliverables Statistics

**Endpoint:** `GET /api/v1/hs2/deliverables/statistics/summary`

**Description:** Get overall deliverables statistics.

**Response:**
```json
{
  "total_deliverables": 250,
  "status_distribution": {
    "Submitted": 180,
    "Pending": 50,
    "Approved": 150,
    "Rejected": 10
  },
  "overdue_count": 15,
  "completion_rate": 72.0,
  "last_updated": "2024-11-25T10:30:00Z"
}
```

---

## TAEM Rules Endpoints

### 14. List TAEM Rules

**Endpoint:** `GET /api/v1/hs2/taem/rules`

**Description:** Retrieve all TAEM rules with optional filtering.

**Query Parameters:**
- `category` (string, optional): Filter by category ("Deliverables", "Costs", "Certificates", "Schedule")
- `severity` (string, optional): Filter by severity ("Critical", "Major", "Minor")
- `active_only` (boolean, optional): Show only active rules (default: true)

**Response:** `List[TAEMRuleResponse]`

**Example Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/hs2/taem/rules?category=Costs&severity=Critical"
```

---

### 15. Get TAEM Rule Details

**Endpoint:** `GET /api/v1/hs2/taem/rules/{rule_id}`

**Description:** Retrieve detailed information for a specific TAEM rule.

**Path Parameters:**
- `rule_id` (UUID, required): Rule unique identifier

**Response:** `TAEMRuleResponse`

---

### 16. Update TAEM Rule (Tinkerability)

**Endpoint:** `PATCH /api/v1/hs2/taem/rules/{rule_id}`

**Description:** Update TAEM rule configuration (adjust weights, thresholds, enable/disable).

**Path Parameters:**
- `rule_id` (UUID, required): Rule unique identifier

**Request Body:** `TAEMRuleUpdate` (all fields optional)
```json
{
  "weight": 0.30,
  "is_active": true,
  "threshold_value": 85.0,
  "severity": "Major"
}
```

**Response:** `TAEMRuleResponse`

---

### 17. Evaluate All Assets

**Endpoint:** `POST /api/v1/hs2/taem/evaluate-all`

**Description:** Trigger TAEM evaluation for all assets (background task).

**Query Parameters:**
- `force_refresh` (boolean, optional): Force re-evaluation of all assets (default: false)

**Response:** 202 Accepted
```json
{
  "status": "accepted",
  "message": "Evaluation queued for 50 assets",
  "total_assets": 50,
  "estimated_completion": "2024-11-25T10:35:00Z",
  "job_id": null
}
```

---

### 18. Get Evaluation History (Audit Trail)

**Endpoint:** `GET /api/v1/hs2/taem/evaluations`

**Description:** Retrieve evaluation history across all assets for audit purposes.

**Query Parameters:**
- `skip` (integer, optional): Pagination offset (default: 0)
- `limit` (integer, optional): Items per page (default: 50, max: 100)
- `asset_id` (UUID, optional): Filter by asset ID
- `from_date` (datetime, optional): Filter from date (ISO 8601)
- `to_date` (datetime, optional): Filter to date (ISO 8601)

**Response:** `EvaluationPaginatedResponse`

---

### 19. Get Rules Statistics

**Endpoint:** `GET /api/v1/hs2/taem/rules/statistics/summary`

**Description:** Get TAEM rules statistics.

**Response:**
```json
{
  "total_rules": 25,
  "active_rules": 20,
  "inactive_rules": 5,
  "category_distribution": {
    "Deliverables": 8,
    "Costs": 6,
    "Certificates": 5,
    "Schedule": 6
  },
  "severity_distribution": {
    "Critical": 5,
    "Major": 10,
    "Minor": 10
  },
  "last_updated": "2024-11-25T10:30:00Z"
}
```

---

## Dashboard Endpoints

### 20. Get Dashboard Summary

**Endpoint:** `GET /api/v1/hs2/dashboard/summary`

**Description:** Get comprehensive dashboard summary with statistics across all dimensions.

**Response:** `DashboardSummary`

**Example Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/hs2/dashboard/summary"
```

**Example Response:**
```json
{
  "total_assets": 50,
  "ready": 10,
  "not_ready": 25,
  "at_risk": 15,
  "ready_pct": 20.0,
  "not_ready_pct": 50.0,
  "at_risk_pct": 30.0,
  "avg_taem_score": 68.5,
  "last_updated": "2024-11-25T10:30:00Z",
  "by_asset_type": [
    {
      "asset_type": "Viaduct",
      "total": 20,
      "ready": 4,
      "not_ready": 10,
      "at_risk": 6,
      "ready_pct": 20.0
    },
    {
      "asset_type": "Bridge",
      "total": 15,
      "ready": 3,
      "not_ready": 8,
      "at_risk": 4,
      "ready_pct": 20.0
    }
  ],
  "by_contractor": [
    {
      "contractor": "Balfour Beatty",
      "total": 25,
      "ready": 6,
      "not_ready": 12,
      "at_risk": 7,
      "ready_pct": 24.0,
      "avg_taem_score": 70.2
    }
  ],
  "by_route": [
    {
      "route_section": "Phase 1",
      "total": 30,
      "ready": 8,
      "not_ready": 15,
      "at_risk": 7,
      "ready_pct": 26.67
    }
  ]
}
```

---

### 21. Get Contractor Breakdown

**Endpoint:** `GET /api/v1/hs2/dashboard/by-contractor`

**Description:** Get detailed breakdown by contractor.

**Query Parameters:**
- `contractor` (string, optional): Filter by specific contractor

**Response:**
```json
{
  "contractors": [
    {
      "contractor": "Balfour Beatty",
      "total_assets": 25,
      "ready": 6,
      "not_ready": 12,
      "at_risk": 7,
      "ready_pct": 24.0,
      "avg_taem_score": 70.2,
      "min_taem_score": 45.0,
      "max_taem_score": 92.5
    }
  ],
  "last_updated": "2024-11-25T10:30:00Z"
}
```

---

### 22. Get Asset Type Breakdown

**Endpoint:** `GET /api/v1/hs2/dashboard/by-asset-type`

**Description:** Get detailed breakdown by asset type.

**Query Parameters:**
- `asset_type` (string, optional): Filter by specific asset type

---

### 23. Get Route Breakdown

**Endpoint:** `GET /api/v1/hs2/dashboard/by-route`

**Description:** Get detailed breakdown by route section.

**Query Parameters:**
- `route_section` (string, optional): Filter by specific route section

---

### 24. Get Readiness Trends

**Endpoint:** `GET /api/v1/hs2/dashboard/trends`

**Description:** Get readiness trends over time.

**Query Parameters:**
- `days` (integer, optional): Number of days to analyze (default: 30, max: 365)

**Response:**
```json
{
  "trends": [
    {
      "date": "2024-11-01T00:00:00Z",
      "evaluations_count": 45,
      "avg_taem_score": 65.2,
      "ready": 8,
      "not_ready": 22,
      "at_risk": 15
    },
    {
      "date": "2024-11-02T00:00:00Z",
      "evaluations_count": 48,
      "avg_taem_score": 66.8,
      "ready": 9,
      "not_ready": 21,
      "at_risk": 18
    }
  ],
  "period_days": 30,
  "start_date": "2024-10-26T00:00:00Z",
  "end_date": "2024-11-25T10:30:00Z"
}
```

---

## Error Responses

All endpoints return standard HTTP error responses:

### 404 Not Found
```json
{
  "detail": "Asset with ID 123e4567-e89b-12d3-a456-426614174000 not found"
}
```

### 422 Validation Error
```json
{
  "detail": [
    {
      "loc": ["body", "weight"],
      "msg": "ensure this value is less than or equal to 1.0",
      "type": "value_error.number.not_le"
    }
  ]
}
```

### 500 Internal Server Error
```json
{
  "detail": "Failed to retrieve assets: database connection error"
}
```

---

## OpenAPI Documentation

Interactive API documentation is available at:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
- **OpenAPI JSON:** `http://localhost:8000/openapi.json`

---

## Testing

### Using cURL

```bash
# List assets
curl -X GET "http://localhost:8000/api/v1/hs2/assets?limit=5"

# Get asset details
curl -X GET "http://localhost:8000/api/v1/hs2/assets/{asset_id}"

# Trigger evaluation
curl -X POST "http://localhost:8000/api/v1/hs2/assets/{asset_id}/evaluate" \
  -H "Content-Type: application/json" \
  -d '{"force_refresh": true}'

# Get dashboard summary
curl -X GET "http://localhost:8000/api/v1/hs2/dashboard/summary"
```

### Using Python

```python
import requests

BASE_URL = "http://localhost:8000/api/v1/hs2"

# List assets
response = requests.get(f"{BASE_URL}/assets", params={"limit": 10, "status": "At Risk"})
assets = response.json()

# Evaluate asset
asset_id = assets["items"][0]["id"]
response = requests.post(
    f"{BASE_URL}/assets/{asset_id}/evaluate",
    json={"force_refresh": True}
)
evaluation = response.json()

# Get dashboard
response = requests.get(f"{BASE_URL}/dashboard/summary")
dashboard = response.json()
```

---

## Performance Considerations

- **Pagination:** Always use appropriate `limit` values to avoid large responses
- **Caching:** Dashboard endpoints may implement caching for improved performance
- **Background Tasks:** Bulk operations (evaluate-all) run asynchronously
- **Database Indexes:** Ensure indexes on `asset_id`, `status`, `contractor`, `route_section`

---

## Future Enhancements

- [ ] Authentication & authorization (JWT tokens, role-based access)
- [ ] WebSocket support for real-time evaluation updates
- [ ] File upload endpoints for deliverables
- [ ] Export endpoints (CSV, Excel, PDF reports)
- [ ] Advanced filtering with complex query syntax
- [ ] Rate limiting for bulk operations
- [ ] Audit log endpoints for compliance

---

## Support

For questions or issues with the API:
- Email: support@hs2.infrastructure
- Documentation: Internal Wiki
- Issue Tracker: JIRA Project

---

**Last Updated:** 2024-11-25
**API Version:** 1.0.0
**Status:** Development
