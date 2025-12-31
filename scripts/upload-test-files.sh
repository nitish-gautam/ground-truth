#!/bin/bash

###############################################################################
# Upload Test Files to MinIO
# Creates sample files and uploads them to demonstrate the platform
###############################################################################

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  MinIO Test Data Upload                                           ║"
echo "║  Creating sample files to demonstrate object storage              ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Create temporary directory for test files
TEST_DIR="/tmp/hs2-minio-test-data"
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"/{gpr,bim,lidar,documents,reports}

echo -e "${BLUE}Step 1: Creating test files...${NC}"

# ============================================
# GPR Test Files
# ============================================
echo "  Creating GPR data files..."

# Create a fake GPR file (SEG-Y format header simulation)
cat > "$TEST_DIR/gpr/VA-007_GPR_Survey_Line1.sgy" <<'EOF'
SEG-Y Format Ground Penetrating Radar Data
============================================
Asset: VA-007 Colne Valley Viaduct
Survey Date: 2024-11-15
Equipment: GSSI SIR-4000
Frequency: 400 MHz
Survey Line: Line 1 - North-South Traverse

Data Points: 10,000
Sampling Interval: 0.5ns
Time Window: 100ns
Distance: 250m

[BINARY DATA WOULD BE HERE]
This is a test file for demonstration purposes.
Real GPR data would be in binary SEG-Y format.
EOF

cat > "$TEST_DIR/gpr/BR-012_GPR_Survey_CrossSection.dzt" <<'EOF'
GSSI DZT Format - GPR Data
==========================
Asset: BR-012 Acton Bridge
Survey Date: 2024-12-01
Antenna: 900 MHz
Depth Range: 2m

[BINARY GPR DATA]
Test file - Real data would be binary
EOF

# Create GPR processing report
cat > "$TEST_DIR/reports/VA-007_GPR_Interpretation_Report.pdf" <<'EOF'
%PDF-1.4
GPR INTERPRETATION REPORT
=========================

Asset: VA-007 Colne Valley Viaduct
Report Date: 2024-11-20
Surveyor: John Smith, GPR Solutions Ltd

EXECUTIVE SUMMARY:
- Survey completed: 2024-11-15
- Total survey length: 1.2km
- Anomalies detected: 3 locations requiring investigation
- Quality Level: QL-B (as per PAS 128:2022)

FINDINGS:
1. Chainage 245m: Possible void beneath foundation (depth 1.2m)
2. Chainage 567m: Reinforcement bar detected (expected)
3. Chainage 891m: Moisture ingress detected

RECOMMENDATIONS:
- Conduct trial pit at chainage 245m
- Monitor moisture levels at chainage 891m
- No action required for chainage 567m

[This is a test PDF - real file would have proper formatting and images]
EOF

echo -e "    ${GREEN}✓${NC} Created 3 GPR-related files"

# ============================================
# BIM Test Files
# ============================================
echo "  Creating BIM model files..."

cat > "$TEST_DIR/bim/VA-007_Structural_Model_v3.ifc" <<'EOF'
ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('ViewDefinition [CoordinationView]'),'2;1');
FILE_NAME('VA-007_Structural_Model.ifc','2024-11-25T10:00:00',('Structural Engineer'),('HS2 Ltd'),'IFC Viewer','Revit 2024','');
FILE_SCHEMA(('IFC4'));
ENDSEC;

DATA;
/* HS2 Colne Valley Viaduct - Structural BIM Model */
/* Asset ID: VA-007 */
/* Model Version: v3.0 */
/* Last Updated: 2024-11-25 */

#1=IFCPROJECT('2a$dGxKDP0LQkqoO9O4Y0C',$,'Colne Valley Viaduct',$,$,$,$,(#20),#19);
#19=IFCUNITASSIGNMENT((#21,#22,#23));
#20=IFCGEOMETRICREPRESENTATIONCONTEXT($,'Model',3,1.E-05,#24,$);

/* Simplified IFC content - Real file would have thousands of entities */
/* This is a test file for demonstration */

ENDSEC;
END-ISO-10303-21;
EOF

cat > "$TEST_DIR/bim/BR-012_Architecture_Model_v2.ifc" <<'EOF'
ISO-10303-21;
/* BIM Model for BR-012 Acton Bridge */
/* Architectural Model */
/* Version: 2.0 */
/* Date: 2024-12-01 */

/* Test IFC file for demonstration */
/* Real model would contain detailed geometry and metadata */
END-ISO-10303-21;
EOF

echo -e "    ${GREEN}✓${NC} Created 2 BIM model files"

# ============================================
# LiDAR Test Files
# ============================================
echo "  Creating LiDAR scan files..."

cat > "$TEST_DIR/lidar/VA-007_LiDAR_Scan_20241110.las" <<'EOF'
LASF
LAS Format 1.4 - Point Cloud Data
==================================
Asset: VA-007 Colne Valley Viaduct
Scan Date: 2024-11-10
Scanner: Leica RTC360
Points: 145,000,000
Coverage Area: 2.5km²

Point Data:
X,Y,Z,Intensity,R,G,B,Classification
[BINARY LAS DATA WOULD BE HERE]

This is a test file.
Real LAS files are binary and contain millions of 3D points.
EOF

cat > "$TEST_DIR/lidar/Route_Section_Denham_Aerial_LiDAR.laz" <<'EOF'
LAZ Format (Compressed LAS)
===========================
Route Section: Denham
Survey Date: 2024-10-15
Aircraft: Fixed-wing drone
Point Density: 50 points/m²
Coverage: 12km²

[COMPRESSED BINARY DATA]
Test file for demonstration
EOF

echo -e "    ${GREEN}✓${NC} Created 2 LiDAR scan files"

# ============================================
# Document Test Files
# ============================================
echo "  Creating document files..."

cat > "$TEST_DIR/documents/VA-007_Design_Certificate.pdf" <<'EOF'
%PDF-1.4
DESIGN CERTIFICATE
==================

Asset: VA-007 Colne Valley Viaduct
Certificate Number: DC-VA007-2024-11
Issue Date: 2024-11-01
Valid Until: 2025-11-01

CERTIFICATION:
I hereby certify that the design of the above asset complies with:
- Eurocodes EN 1990, EN 1991, EN 1992, EN 1993
- Network Rail Standards NR/L3/CIV/006
- HS2 Technical Standards

Certified by:
Name: Dr. Sarah Johnson, CEng MIStructE
Organization: Arup
Date: 2024-11-01
Signature: [SIGNED]

[Test PDF for demonstration]
EOF

cat > "$TEST_DIR/documents/Health_Safety_Plan_JV-Bravo.pdf" <<'EOF'
%PDF-1.4
CONSTRUCTION HEALTH & SAFETY PLAN
==================================

Contractor: JV-Bravo
Route Section: Acton
Plan Reference: HSP-JVB-2024-003
Version: 3.0
Date: 2024-11-20

1. INTRODUCTION
This Health & Safety Plan covers construction activities for:
- BR-012 Acton Bridge
- TN-003 Northolt Tunnel approaches
- Various earthworks

2. RISK ASSESSMENTS
- Working at height
- Excavation works
- Traffic management
- Environmental protection

3. METHOD STATEMENTS
[Detailed method statements would follow]

[Test document for demonstration]
EOF

cat > "$TEST_DIR/documents/IDP_Deliverable_VA007_STR_001.pdf" <<'EOF'
%PDF-1.4
IDP DELIVERABLE SUBMISSION
==========================

Deliverable Code: IDP-STR-001
Deliverable Name: Structural Calculations Package
Asset: VA-007 Colne Valley Viaduct
Version: 1.0
Submission Date: 2024-11-18

CONTENTS:
1. Foundation design calculations
2. Pier design calculations
3. Deck design calculations
4. Connection details
5. Seismic analysis

Status: APPROVED
Approved by: Chief Engineer
Approval Date: 2024-11-22

[Test IDP deliverable for demonstration]
EOF

echo -e "    ${GREEN}✓${NC} Created 3 document files"

# ============================================
# Report Test Files
# ============================================
echo "  Creating report files..."

cat > "$TEST_DIR/reports/Monthly_Progress_Report_2024-11.pdf" <<'EOF'
%PDF-1.4
HS2 MONTHLY PROGRESS REPORT
============================
Period: November 2024

EXECUTIVE SUMMARY:
- Assets on track: 35 (70%)
- Assets delayed: 15 (30%)
- Critical issues: 3
- Budget variance: -2.5% (within tolerance)

CONTRACTOR PERFORMANCE:
JV-Alpha: 65% ready (target 70%) - BEHIND
JV-Bravo: 85% ready (target 80%) - AHEAD
JV-Charlie: 60% ready (target 75%) - BEHIND

KEY ACHIEVEMENTS:
- VA-007 Viaduct foundations completed
- BR-012 Bridge steelwork 80% complete
- TN-003 Tunnel boring at 45% completion

RISKS & ISSUES:
1. Material delivery delays (steel)
2. Weather impact on earthworks
3. Utility diversions slower than planned

[Test report for demonstration]
EOF

cat > "$TEST_DIR/reports/TAEM_Evaluation_Summary_2024-11-25.json" <<'EOF'
{
  "report_date": "2024-11-25T10:00:00Z",
  "total_assets_evaluated": 50,
  "summary": {
    "ready": 10,
    "not_ready": 40,
    "at_risk": 0,
    "average_taem_score": 56.85
  },
  "critical_issues": [
    {
      "asset_id": "VA-007",
      "issue": "Cost variance exceeds 15% threshold",
      "severity": "Major",
      "action_required": "Budget review meeting scheduled"
    },
    {
      "asset_id": "TN-003",
      "issue": "Safety certificate expired",
      "severity": "Critical",
      "action_required": "Work suspended pending recertification"
    }
  ],
  "recommendations": [
    "Increase monitoring frequency for at-risk assets",
    "Expedite deliverable approvals for JV-Alpha projects",
    "Review resource allocation for JV-Charlie"
  ]
}
EOF

echo -e "    ${GREEN}✓${NC} Created 2 report files"

# ============================================
# Upload to MinIO using mc (MinIO Client)
# ============================================
echo ""
echo -e "${BLUE}Step 2: Installing MinIO Client (mc)...${NC}"

# Check if mc is already installed
if command -v mc &> /dev/null; then
    echo "  MinIO Client already installed"
else
    echo "  Downloading MinIO Client..."
    curl -o /usr/local/bin/mc https://dl.min.io/client/mc/release/darwin-arm64/mc 2>/dev/null || \
    curl -o /usr/local/bin/mc https://dl.min.io/client/mc/release/linux-amd64/mc 2>/dev/null
    chmod +x /usr/local/bin/mc
fi

echo ""
echo -e "${BLUE}Step 3: Configuring MinIO connection...${NC}"

# Configure mc to connect to local MinIO
mc alias set local http://localhost:9010 minioadmin mD9E3_kgZJAPRjNvBWOxGQ 2>/dev/null || true

# Verify connection
if mc ls local > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} Connected to MinIO"
else
    echo -e "  ${YELLOW}⚠ Could not connect to MinIO. Trying alternative method...${NC}"

    # Alternative: Use docker exec with mc
    docker exec infrastructure-minio mc alias set myminio http://localhost:9000 minioadmin mD9E3_kgZJAPRjNvBWOxGQ
fi

echo ""
echo -e "${BLUE}Step 4: Uploading files to MinIO...${NC}"

# Upload GPR files
echo "  Uploading GPR data..."
mc cp "$TEST_DIR/gpr/"* local/gpr-data/ 2>/dev/null || \
    docker exec -i infrastructure-minio sh -c "cat > /tmp/gpr1.sgy" < "$TEST_DIR/gpr/VA-007_GPR_Survey_Line1.sgy" && \
    docker exec infrastructure-minio mc cp /tmp/gpr1.sgy /data/gpr-data/

echo -e "    ${GREEN}✓${NC} Uploaded GPR files"

# Upload BIM files
echo "  Uploading BIM models..."
mc cp "$TEST_DIR/bim/"* local/bim-models/ 2>/dev/null || \
    docker cp "$TEST_DIR/bim/." infrastructure-minio:/tmp/bim/ && \
    docker exec infrastructure-minio sh -c "mc cp /tmp/bim/* /data/bim-models/"

echo -e "    ${GREEN}✓${NC} Uploaded BIM files"

# Upload LiDAR files
echo "  Uploading LiDAR scans..."
mc cp "$TEST_DIR/lidar/"* local/lidar-scans/ 2>/dev/null || \
    docker cp "$TEST_DIR/lidar/." infrastructure-minio:/tmp/lidar/ && \
    docker exec infrastructure-minio sh -c "mc cp /tmp/lidar/* /data/lidar-scans/"

echo -e "    ${GREEN}✓${NC} Uploaded LiDAR files"

# Upload documents
echo "  Uploading documents..."
mc cp "$TEST_DIR/documents/"* local/documents/ 2>/dev/null || \
    docker cp "$TEST_DIR/documents/." infrastructure-minio:/tmp/docs/ && \
    docker exec infrastructure-minio sh -c "mc cp /tmp/docs/* /data/documents/"

echo -e "    ${GREEN}✓${NC} Uploaded documents"

# Upload reports
echo "  Uploading reports..."
mc cp "$TEST_DIR/reports/"* local/reports/ 2>/dev/null || \
    docker cp "$TEST_DIR/reports/." infrastructure-minio:/tmp/reports/ && \
    docker exec infrastructure-minio sh -c "mc cp /tmp/reports/* /data/reports/"

echo -e "    ${GREEN}✓${NC} Uploaded reports"

echo ""
echo -e "${BLUE}Step 5: Verifying uploads...${NC}"

# List files in each bucket
echo ""
echo "Files in buckets:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for bucket in gpr-data bim-models lidar-scans documents reports; do
    echo ""
    echo -e "${GREEN}$bucket:${NC}"
    mc ls local/$bucket 2>/dev/null || docker exec infrastructure-minio mc ls /data/$bucket/ || echo "  (empty)"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${GREEN}✓ Upload complete!${NC}"
echo ""
echo "Access MinIO Console: http://localhost:9011"
echo "Username: minioadmin"
echo "Password: mD9E3_kgZJAPRjNvBWOxGQ"
echo ""
echo "Buckets now contain test files for demonstration."
echo ""

# Cleanup temp directory
rm -rf "$TEST_DIR"
echo "Temporary files cleaned up."
