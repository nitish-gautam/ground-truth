#!/bin/bash
# HS2 Platform - Complete Startup Script
# =======================================
# Starts all services via Docker Compose

set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║    HS2 Infrastructure Intelligence Platform - STARTUP    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}▶${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker Desktop and try again."
    exit 1
fi
print_success "Docker is running"

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null 2>&1; then
    print_error "docker-compose not found. Please install Docker Compose."
    exit 1
fi
print_success "Docker Compose available"

# Determine docker compose command
if docker compose version &> /dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

echo ""
print_status "Starting HS2 Platform services..."
echo ""

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    print_status "Creating .env file with default configuration..."
    cat > .env << 'EOF'
# PostgreSQL
POSTGRES_USER=gpr_app_user
POSTGRES_PASSWORD=change_me_app_2024!
POSTGRES_DB=gpr_platform

# MinIO
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin123

# Application
SECRET_KEY=your-secret-key-change-in-production-12345
DEBUG=true

# Optional: LLM APIs (for Phase 2)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
EOF
    print_success ".env file created"
fi

# Start services
print_status "Starting Docker containers..."
echo ""

$DOCKER_COMPOSE up -d --build

echo ""
print_status "Waiting for services to be healthy..."
sleep 10

# Check service status
print_status "Checking service status..."
echo ""

SERVICES=(postgres redis neo4j minio backend frontend)
ALL_HEALTHY=true

for service in "${SERVICES[@]}"; do
    if $DOCKER_COMPOSE ps | grep -q "$service.*Up"; then
        print_success "$service is running"
    else
        print_warning "$service is not running"
        ALL_HEALTHY=false
    fi
done

echo ""

if [ "$ALL_HEALTHY" = true ]; then
    print_success "All core services are running!"
else
    print_warning "Some services may need more time to start"
    print_status "Run 'docker-compose ps' to check status"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                   PLATFORM IS READY                      ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

echo "${GREEN}WEB INTERFACES:${NC}"
echo ""
echo "  🌐 Frontend (HS2 Dashboard):  ${BLUE}http://localhost:3003${NC}"
echo "  📖 Backend API Docs:          ${BLUE}http://localhost:8007/docs${NC}"
echo "  🔍 Neo4j Browser:             ${BLUE}http://localhost:7475${NC}"
echo "     Username: neo4j / Password: hs2_graph_2024"
echo "  📦 MinIO Console:             ${BLUE}http://localhost:9011${NC}"
echo "     Username: minioadmin / Password: minioadmin123"
echo "  🌺 Celery Flower:             ${BLUE}http://localhost:5555${NC}"
echo ""

echo "${GREEN}DATABASE ACCESS:${NC}"
echo ""
echo "  PostgreSQL:   localhost:5433"
echo "  Username:     gpr_app_user"
echo "  Password:     change_me_app_2024!"
echo "  Database:     gpr_platform"
echo ""

echo "${GREEN}API ENDPOINTS:${NC}"
echo ""
echo "  LiDAR Analysis:      ${BLUE}http://localhost:8007/api/v1/lidar${NC}"
echo "  Hyperspectral:       ${BLUE}http://localhost:8007/api/v1/progress/hyperspectral${NC}"
echo "  BIM Models:          ${BLUE}http://localhost:8007/api/v1/bim${NC}"
echo "  Progress Tracking:   ${BLUE}http://localhost:8007/api/v1/progress${NC}"
echo ""

echo "${GREEN}QUICK TESTS:${NC}"
echo ""
echo "  # Test Backend Health"
echo "  ${YELLOW}curl http://localhost:8007/health${NC}"
echo ""
echo "  # Test LiDAR Endpoint"
echo "  ${YELLOW}curl -X POST http://localhost:8007/api/v1/lidar/elevation/point \\${NC}"
echo "  ${YELLOW}  -H 'Content-Type: application/json' \\${NC}"
echo "  ${YELLOW}  -d '{\"easting\": 423500, \"northing\": 338500}'${NC}"
echo ""
echo "  # View Logs"
echo "  ${YELLOW}docker-compose logs -f backend${NC}"
echo ""
echo "  # Stop All Services"
echo "  ${YELLOW}docker-compose down${NC}"
echo ""

echo "═══════════════════════════════════════════════════════════"
echo ""
echo "${GREEN}✨ Platform is running! Open http://localhost:3003/hs2 to get started${NC}"
echo ""
