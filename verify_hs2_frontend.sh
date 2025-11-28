#!/bin/bash

echo "======================================================================"
echo "HS2 FRONTEND IMPLEMENTATION VERIFICATION"
echo "======================================================================"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check frontend directory
if [ -d "frontend/src" ]; then
    echo -e "${GREEN}✓${NC} Frontend directory exists"
else
    echo -e "${RED}✗${NC} Frontend directory not found"
    exit 1
fi

echo ""
echo "Checking created files..."
echo "---------------------------------------"

# Array of files to check
declare -a files=(
    "frontend/src/types/hs2Types.ts"
    "frontend/src/utils/formatting.ts"
    "frontend/src/api/hs2Client.ts"
    "frontend/src/components/HS2SummaryCards.tsx"
    "frontend/src/components/HS2AssetTable.tsx"
    "frontend/src/components/HS2ReadinessPanel.tsx"
    "frontend/src/components/HS2ExplainabilityPanel.tsx"
    "frontend/src/components/HS2DeliverablesTable.tsx"
    "frontend/src/components/HS2CostSummary.tsx"
    "frontend/src/components/HS2CertificatesTable.tsx"
    "frontend/src/pages/HS2Dashboard.tsx"
    "frontend/src/pages/HS2AssetList.tsx"
    "frontend/src/pages/HS2AssetDetail.tsx"
    "frontend/src/App.tsx"
    "frontend/package.json"
)

total_files=0
found_files=0

for file in "${files[@]}"; do
    total_files=$((total_files + 1))
    if [ -f "$file" ]; then
        lines=$(wc -l < "$file")
        echo -e "${GREEN}✓${NC} $file (${lines} lines)"
        found_files=$((found_files + 1))
    else
        echo -e "${RED}✗${NC} $file - NOT FOUND"
    fi
done

echo ""
echo "---------------------------------------"
echo "Files: $found_files/$total_files"
echo "---------------------------------------"

if [ $found_files -eq $total_files ]; then
    echo -e "${GREEN}✓ All files created successfully!${NC}"
else
    echo -e "${RED}✗ Some files are missing${NC}"
    exit 1
fi

echo ""
echo "Checking package.json dependencies..."
echo "---------------------------------------"

# Check if Material-UI is in package.json
if grep -q "@mui/material" frontend/package.json; then
    echo -e "${GREEN}✓${NC} Material-UI dependencies added"
else
    echo -e "${RED}✗${NC} Material-UI not found in package.json"
fi

if grep -q "@tanstack/react-query" frontend/package.json; then
    echo -e "${GREEN}✓${NC} React Query added"
else
    echo -e "${RED}✗${NC} React Query not found in package.json"
fi

if grep -q "recharts" frontend/package.json; then
    echo -e "${GREEN}✓${NC} Recharts added"
else
    echo -e "${RED}✗${NC} Recharts not found in package.json"
fi

echo ""
echo "Checking total lines of code..."
echo "---------------------------------------"

total_lines=$(find frontend/src -name "*.ts" -o -name "*.tsx" | xargs wc -l | tail -1 | awk '{print $1}')
echo "Total lines of TypeScript code: $total_lines"

echo ""
echo "======================================================================"
echo -e "${GREEN}✓ VERIFICATION COMPLETE${NC}"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "1. cd frontend"
echo "2. npm install"
echo "3. Create .env file (see QUICK_START_HS2_FRONTEND.md)"
echo "4. npm run dev"
echo ""
echo "Documentation:"
echo "- QUICK_START_HS2_FRONTEND.md"
echo "- HS2_FRONTEND_COMPLETE.md"
echo "- HS2_FRONTEND_IMPLEMENTATION_PLAN.md"
echo ""

