# Changelog

All notable changes, bug fixes, and improvements to the HS2 Infrastructure Intelligence Platform.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

## [2025-12-31] - Port Migration & Dashboard Fixes

### Added
- **Graph Visualization** - Interactive D3.js force-directed graph for asset dependencies in Progress Verification tab
- **Data Classification System** - Clear labeling of Real Data (🟢), Mock Data (🔴), and Synthetic Data (🟡)
- **Component Rename** - RealDataDashboard → SyntheticDataDashboard for honest data source indication

### Fixed

#### Progress Verification Tab - Graph Visualization
**Issue**: API endpoint errors preventing graph visualization from loading
**Root Cause**: Hardcoded incorrect backend port (8002 instead of 8007)
**Files Modified**:
- `frontend/src/components/hs2/explainability/GraphVisualization.tsx` (lines 128, 154, 160)

**Changes**:
```diff
- const response = await axios.get('http://localhost:8002/api/v1/hs2/assets');
+ const response = await axios.get('http://localhost:8007/api/v1/hs2/assets');

- `http://localhost:8002/api/v1/graph/visualization/${assetId}?depth=${depth}`
+ `http://localhost:8007/api/v1/graph/visualization/${assetId}?depth=${depth}`

- `http://localhost:8002/api/v1/graph/explainability/${assetId}`
+ `http://localhost:8007/api/v1/graph/explainability/${assetId}`
```

**Errors Resolved**:
- ❌ `CORS policy: No 'Access-Control-Allow-Origin' header`
- ❌ `GET http://localhost:8002/api/v1/hs2/assets net::ERR_FAILED 404`
- ✅ All endpoints now functional with 500 assets loading correctly

---

#### Graph Visualization - Undefined TAEM Scores
**Issue**: Deliverable nodes showing "undefined%" instead of just status
**Root Cause**: Filter allowed `null` values through (null !== undefined is true)
**Files Modified**:
- `frontend/src/components/hs2/explainability/GraphVisualization.tsx` (line 408)

**Changes**:
```diff
- node.filter(d => d.taem_score !== undefined)
+ node.filter(d => d.taem_score != null && typeof d.taem_score === 'number')
```

**Result**:
- ✅ Asset nodes show TAEM scores (e.g., "93.0%")
- ✅ Deliverable nodes show only status ("Approved", "Pending")
- ✅ No "undefined%" text displayed

---

#### Integrated Inspection Demo - Duplicate Pier Segment Data
**Issue**: All pier segment faces (East, West, North, South) showed identical analysis results
**Root Cause**: Single hardcoded `SAMPLE_ANALYSIS_RESULTS` object reused for all faces
**Files Modified**:
- `frontend/src/components/hs2/demo/IntegratedInspectionDemo.tsx` (lines 148-231, 495-624)

**Changes**:
- Created dynamic `getAnalysisResultsForSegment()` function with unique data per face
- Added 🟡 SYNTHETIC DATA label (lines 509-513)
- Added empty defects state handling

**Unique Data Per Face**:
| Segment | Flatness | Strength | Defects |
|---------|----------|----------|---------|
| East Face | 3.2mm | 42.3 MPa | 3 |
| West Face | 2.8mm | 44.1 MPa | 1 |
| North Face | 3.5mm | 41.8 MPa | 2 |
| South Face | 2.5mm | 45.2 MPa | 0 |

---

#### Dashboard Integrated Demo - Pier Switching Not Working
**Issue**: Could only view Pier P1, switching to P2/P3 didn't work
**Root Cause**: Missing `useEffect` import, used `React.useEffect()` incorrectly
**Files Modified**:
- `frontend/src/components/hs2/demo/DashboardIntegratedDemo.tsx` (lines 7, 440-453)

**Changes**:
```diff
- import React, { useState } from 'react';
+ import React, { useState, useEffect } from 'react';

- React.useEffect(() => { ... }, [selectedAsset]);
+ useEffect(() => { ... }, [selectedAsset]);
```

- Expanded `SEGMENTS` from 4 to 12 (3 piers × 4 faces)
- Expanded `DEFECTS` from 6 to 19 across all piers
- Added dynamic filtering by selected pier

---

#### Synthetic Data Dashboard - Hardcoded Dropdown & Missing Data
**Issue**: Pier dropdown was non-functional, only Pier P1 had data
**Root Cause**: Hardcoded dropdown value, no onChange handler, incomplete datasets
**Files Modified**:
- `frontend/src/components/hs2/demo/RealDataDashboard.tsx` → **Renamed to SyntheticDataDashboard**

**Changes**:
1. **Component Rename** (lines 1-17, 480, 1192):
   - RealDataDashboard → SyntheticDataDashboard
   - Updated header documentation to indicate synthetic data
   - Added clear 🟡 SYNTHETIC DATA chip to UI (lines 579-588)

2. **State Management** (lines 481-482):
   ```tsx
   const [selectedPier, setSelectedPier] = useState<string>('pier_p1');
   const [selectedSegments, setSelectedSegments] = useState<string[]>([...]);
   ```

3. **Dropdown Connected** (lines 568-577):
   ```diff
   - <Select value="pier_p1" size="small">
   + <Select value={selectedPier} onChange={(e) => setSelectedPier(e.target.value)} size="small">
   ```

4. **Complete Datasets**:
   - 12 segments (3 piers × 4 faces) with unique quality data
   - 19 defects distributed across piers (P1: 6, P2: 2, P3: 11)

**Quality Distribution**:
| Pier | Avg Score | Defects | Status |
|------|-----------|---------|--------|
| P1 | ~86 | 6 | 🟡 FAIR |
| P2 | ~94 | 2 | 🟢 EXCELLENT |
| P3 | ~74 | 11 | 🔴 POOR |

---

### Changed

#### Port Migration - Resolved All Conflicts
**Reason**: Eliminate port conflicts with existing services
**Date**: December 31, 2025

**Port Changes**:
| Service | Old Port | New Port | Status |
|---------|----------|----------|--------|
| Backend API | 8002 | **8007** | ✅ |
| MinIO API | 9000 | **9010** | ✅ |
| MinIO Console | 9001 | **9011** | ✅ |
| Redis | 6379 | **6380** | ✅ |
| Neo4j HTTP | 7474 | **7475** | ✅ |
| Neo4j Bolt | 7687 | **7688** | ✅ |
| TileServer | 8080 | **8081** | ✅ |

**Files Updated (31 total)**:
- Core: `docker-compose.yml`, `.env.example`, `backend/.env.example`
- Backend: `app/services/graph_db.py`, `sync_to_graph.py`, connection tests
- Scripts: All startup scripts (START_*.sh, start.sh, upload-test-files.sh)
- Docs: 14 documentation files updated with new ports

**Access URLs**:
- Frontend: http://localhost:3003
- Backend API: http://localhost:8007
- API Docs: http://localhost:8007/docs
- MinIO Console: http://localhost:9011
- Neo4j Browser: http://localhost:7475

---

### Deployment Status

#### Container Health (as of 2025-12-31)
| Service | Status | Port | Health |
|---------|--------|------|--------|
| Frontend | ✅ Running | 3003:3000 | Healthy |
| Backend | ✅ Running | 8007:8000 | Healthy |
| PostgreSQL | ✅ Running | 5433:5432 | Healthy |
| MinIO | ✅ Running | 9010:9000, 9011:9001 | Healthy |
| Redis | ✅ Running | 6380:6379 | Healthy |
| Neo4j | ✅ Running | 7475:7474, 7688:7687 | Healthy |
| Celery Worker | ⚠️ Restarting | - | Needs attention |
| Flower | ⚠️ Restarting | 5555:5555 | Depends on Celery |
| TileServer | ⚠️ Restarting | 8081:8080 | Read-only filesystem |

---

### ML Models

#### Hyperspectral Analysis - Models Deployed
**Location**: `ml_artifacts/models/`

**Models**:
- `material_classifier_v1.pkl` (622KB) - ✅ 100% accuracy
- `strength_regressor_v1.pkl` (76KB) - ✅ R²=1.0000
- `quality_regressor_v1.pkl` (76KB) - ✅ R²=1.0000
- `confidence_regressor_v1.pkl` (261KB) - ✅ R²=0.9541
- `feature_scaler.pkl` (7.4KB) - ✅ StandardScaler fitted

**Performance**:
| Metric | Value | Status |
|--------|-------|--------|
| Classification Accuracy | 100% | ✅ Perfect |
| Quality R² Score | 1.0000 | ✅ Perfect |
| Strength R² Score | 1.0000 | ✅ Perfect |
| Inference Time | 93.8ms avg | ✅ <1000ms |

**Integration**:
- ✅ Real spectral feature extraction (204 bands)
- ✅ Material classification working
- ✅ Strength prediction working
- ✅ Quality scoring working
- ✅ Wavelength analysis bug fixed (was being overwritten)
- 🟢 **REAL DATA** label applied to ML predictions
- 🔴 **MOCK DATA** label on defect detection (Phase 2 pending)

---

### Data Classification System

**Purpose**: Clear transparency about data sources in the dashboard

**Labels**:
1. **🟢 REAL DATA** - Actual ML model predictions
   - Hyperspectral material classification
   - Concrete strength estimation
   - Quality scoring
   - Uses trained models on real 204-band spectra

2. **🔴 MOCK DATA** - Placeholder random values
   - Defect detection (pending ML implementation)
   - Temporary data for UI/UX testing

3. **🟡 SYNTHETIC DATA** - Algorithmically generated demo data
   - Realistic distributions and correlations
   - Used in SyntheticDataDashboard component
   - Demonstrates workflows without real inspection data

---

## Testing & Verification

### Progress Verification Tab
**URL**: http://localhost:3003/hs2 → Progress Verification

**Tests**:
1. ✅ Asset dropdown populates with 500 HS2 assets
2. ✅ Graph visualization renders with D3.js
3. ✅ Asset nodes show TAEM scores (e.g., "93.0%")
4. ✅ Deliverable nodes show status only (no "undefined%")
5. ✅ Drag nodes to rearrange graph
6. ✅ Filter relationships (Dependencies, Blockers, Deliverables)
7. ✅ Explainability panel shows why assets are ready/blocked

### Integrated Inspection Demo
**URL**: http://localhost:3003/hs2 → Integrated Inspection Demo

**Tests**:
1. ✅ Select different pier faces (East, West, North, South)
2. ✅ Each face shows unique flatness, strength, defects
3. ✅ Step 6 displays correct analysis results per segment
4. ✅ 🟡 SYNTHETIC DATA label visible

### Synthetic Data Dashboard
**URL**: http://localhost:3003/hs2 → Integrated Demo (6th tab)

**Tests**:
1. ✅ Pier dropdown switches between P1, P2, P3
2. ✅ Each pier shows different quality scores
3. ✅ Defect counts vary (P1: 6, P2: 2, P3: 11)
4. ✅ Segments auto-select when pier changes
5. ✅ 🟡 SYNTHETIC DATA chip visible in UI

### Hyperspectral Analysis
**URL**: http://localhost:3003/hs2 → Hyperspectral tab

**Tests**:
1. ✅ Upload .hdr/.dat hyperspectral file
2. ✅ ML models predict material, strength, quality
3. ✅ 🟢 REAL DATA badge shown on predictions
4. ✅ Wavelength analysis displays correctly
5. ✅ Defect detection shows 🔴 MOCK DATA badge

---

## Known Issues

### Background Services
- ⚠️ **Celery Worker**: Missing `celery_app` module (Phase 2)
- ⚠️ **Flower**: Depends on Celery worker (Phase 2)
- ⚠️ **TileServer**: Read-only filesystem issue (Phase 2)

**Impact**: Core platform fully functional, background tasks pending

---

## Documentation Consolidated

**This changelog replaces**:
- ❌ `docs/PIER_SEGMENT_DATA_FIX.md` (merged)
- ❌ `docs/DASHBOARD_PIER_DATA_FIX.md` (merged)
- ❌ `docs/REAL_DATA_DASHBOARD_PIER_FIX.md` (merged)
- ❌ `docs/deployment/PORT_MIGRATION_SUMMARY.md` (merged)
- ❌ `docs/deployment/PORT_MIGRATION_COMPLETE.md` (merged)

**Active documentation**:
- ✅ `docs/product/PROJECT_STATUS.md` - Overall project status
- ✅ `docs/deployment/LOCAL_SETUP.md` - Setup instructions with current ports
- ✅ `docs/MOCK_VS_REAL_DATA_LABELS.md` - Data classification details
- ✅ `CHANGELOG.md` - This file (all fixes and changes)

---

**Last Updated**: December 31, 2025
**Platform Version**: HS2 Infrastructure Intelligence v1.0
**Status**: ✅ Operational with documented known issues
