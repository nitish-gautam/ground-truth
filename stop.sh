#!/bin/bash

###############################################################################
# HS2 Infrastructure Intelligence Platform - Stop Script
# Stop all Docker containers
###############################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  Stopping HS2 Infrastructure Intelligence Platform                ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if user wants to remove volumes
if [ "$1" == "--clean" ] || [ "$1" == "-v" ]; then
    echo -e "${YELLOW}Stopping containers and removing volumes...${NC}"
    docker compose down -v
    echo -e "${GREEN}✓ Containers stopped and volumes removed${NC}"
else
    echo -e "${YELLOW}Stopping containers (keeping data)...${NC}"
    docker compose down
    echo -e "${GREEN}✓ Containers stopped${NC}"
    echo ""
    echo -e "${YELLOW}To remove volumes and start fresh, run:${NC}"
    echo "  ./stop.sh --clean"
fi

echo ""
