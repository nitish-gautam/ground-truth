#!/bin/bash

###############################################################################
# HS2 Infrastructure Intelligence Platform - Simple Startup Script
# One-command to build and run all Docker containers
###############################################################################

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  HS2 Infrastructure Intelligence Platform                          ║"
echo "║  Docker Compose Startup                                            ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if Docker is running
echo -e "${BLUE}Checking Docker...${NC}"
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}✗ Docker is not running. Please start Docker Desktop first.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker is running${NC}"
echo ""

# Stop any existing containers (optional - comment out if you want to keep them running)
# echo -e "${BLUE}Stopping existing containers...${NC}"
# docker compose down
# echo ""

# Build all containers
echo -e "${BLUE}Building Docker containers...${NC}"
echo "This may take 2-5 minutes on first run..."
docker compose build --parallel
echo -e "${GREEN}✓ Build complete${NC}"
echo ""

# Start all services
echo -e "${BLUE}Starting all services...${NC}"
docker compose up -d
echo -e "${GREEN}✓ All services started${NC}"
echo ""

# Wait a moment for services to initialize
echo -e "${BLUE}Waiting for services to initialize...${NC}"
sleep 5
echo ""

# Show running containers
echo -e "${BLUE}Running containers:${NC}"
docker compose ps
echo ""

# Show service URLs
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  🚀 Services are running!                                          ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}Frontend Dashboard:${NC}     http://localhost:3003"
echo -e "${GREEN}Backend API:${NC}            http://localhost:8002"
echo -e "${GREEN}API Documentation:${NC}      http://localhost:8002/docs"
echo -e "${GREEN}API Health Check:${NC}       http://localhost:8002/health"
echo -e "${GREEN}MinIO Console:${NC}          http://localhost:9001"
echo -e "${GREEN}Flower (Celery):${NC}        http://localhost:5555"
echo -e "${GREEN}TileServer:${NC}             http://localhost:8080"
echo ""
echo -e "${YELLOW}Useful commands:${NC}"
echo "  docker compose logs -f              # View all logs"
echo "  docker compose logs -f frontend     # View frontend logs only"
echo "  docker compose logs -f backend      # View backend logs only"
echo "  docker compose ps                   # Show running containers"
echo "  docker compose down                 # Stop all containers"
echo "  docker compose down -v              # Stop and remove volumes"
echo "  docker compose restart frontend     # Restart a specific service"
echo ""
echo -e "${GREEN}✓ Setup complete!${NC}"
