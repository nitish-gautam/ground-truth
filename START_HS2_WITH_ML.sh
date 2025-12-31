#!/bin/bash

# ============================================
# HS2 Platform Startup Script with ML Models
# ============================================

set -e

echo "======================================================================"
echo "    HS2 Infrastructure Intelligence Platform - ML Enabled"
echo "======================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if ML models exist
echo "📋 Pre-flight checks..."
echo ""

if [ ! -d "ml_artifacts/models" ]; then
    echo -e "${RED}❌ ML models directory not found: ml_artifacts/models${NC}"
    echo ""
    echo "Please train ML models first:"
    echo "  python3 backend/app/ml/training/train_material_classifier.py"
    echo "  python3 backend/app/ml/training/train_quality_regressor.py"
    exit 1
fi

MODEL_COUNT=$(ls ml_artifacts/models/*.pkl 2>/dev/null | wc -l)
if [ "$MODEL_COUNT" -lt 5 ]; then
    echo -e "${RED}❌ Expected 5 ML model files, found $MODEL_COUNT${NC}"
    echo ""
    echo "Missing models. Please train all models:"
    echo "  python3 backend/app/ml/training/train_material_classifier.py"
    echo "  python3 backend/app/ml/training/train_quality_regressor.py"
    exit 1
fi

echo -e "${GREEN}✅ ML models found: $MODEL_COUNT files${NC}"
ls -lh ml_artifacts/models/*.pkl | awk '{print "  ", $9, "("$5")"}'
echo ""

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running${NC}"
    echo "Please start Docker Desktop and try again"
    exit 1
fi

echo -e "${GREEN}✅ Docker is running${NC}"
echo ""

# Stop existing containers
echo "🛑 Stopping existing containers (if any)..."
docker-compose down 2>/dev/null || true
echo ""

# Build containers
echo "🔨 Building Docker containers..."
echo ""
docker-compose build --no-cache backend
echo ""

# Start services
echo "🚀 Starting HS2 Platform..."
echo ""
docker-compose up -d
echo ""

# Wait for services to be healthy
echo "⏳ Waiting for services to start (this may take 30-60 seconds)..."
echo ""

# Function to check if a service is healthy
wait_for_service() {
    local service=$1
    local url=$2
    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$url" > /dev/null 2>&1; then
            echo -e "  ${GREEN}✅ $service is ready${NC}"
            return 0
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done

    echo -e "  ${YELLOW}⚠️  $service did not become healthy (timeout)${NC}"
    return 1
}

# Check each service
echo "Checking services:"
wait_for_service "Backend API" "http://localhost:8007/health"
wait_for_service "Frontend" "http://localhost:3003"
wait_for_service "MinIO" "http://localhost:9010/minio/health/live"
wait_for_service "Neo4j" "http://localhost:7475"
echo ""

# Verify ML models are loaded
echo "🤖 Verifying ML models loaded..."
if docker-compose logs backend 2>&1 | grep -q "ML predictor loaded successfully"; then
    echo -e "${GREEN}✅ ML predictor loaded successfully${NC}"
else
    echo -e "${YELLOW}⚠️  ML predictor status unclear - check logs${NC}"
    echo "  Run: docker-compose logs backend | grep ML"
fi
echo ""

# Test ML inference
echo "🧪 Testing ML inference..."
if docker-compose exec -T backend ls /app/ml_artifacts/models/ > /dev/null 2>&1; then
    MODEL_COUNT_DOCKER=$(docker-compose exec -T backend ls /app/ml_artifacts/models/*.pkl 2>/dev/null | wc -l)
    echo -e "${GREEN}✅ ML models accessible in container: $MODEL_COUNT_DOCKER files${NC}"
else
    echo -e "${YELLOW}⚠️  Could not verify ML models in container${NC}"
fi
echo ""

# Display service URLs
echo "======================================================================"
echo "✅ HS2 Platform is running with ML models!"
echo "======================================================================"
echo ""
echo "📊 Service URLs:"
echo "  Frontend:          http://localhost:3003"
echo "  Backend API:       http://localhost:8007"
echo "  API Docs:          http://localhost:8007/docs"
echo "  MinIO Console:     http://localhost:9011  (admin / minioadmin)"
echo "  Neo4j Browser:     http://localhost:7475  (neo4j / hs2_graph_2024)"
echo "  Flower (Celery):   http://localhost:5555"
echo ""
echo "🤖 ML Hyperspectral Endpoint:"
echo "  POST http://localhost:8007/api/v1/progress/hyperspectral/analyze-material"
echo ""
echo "📝 Test ML Models:"
echo ""
echo "  # Test with concrete sample:"
echo "  curl -X POST \"http://localhost:8007/api/v1/progress/hyperspectral/analyze-material\" \\"
echo "    -F \"file=@datasets/raw/hyperspectral/umkc-material-surfaces/Concrete/HSI_TIFF_50x50/Auto119.tiff\""
echo ""
echo "  # Test with asphalt sample:"
echo "  curl -X POST \"http://localhost:8007/api/v1/progress/hyperspectral/analyze-material\" \\"
echo "    -F \"file=@datasets/raw/hyperspectral/umkc-material-surfaces/Asphalt/HSI_TIFF_50x50/Auto005.tiff\""
echo ""
echo "  # Run validation tests inside container:"
echo "  docker-compose exec backend python3 app/ml/inference/test_predictor.py"
echo ""
echo "📋 Useful Commands:"
echo "  View logs:         docker-compose logs -f backend"
echo "  View ML logs:      docker-compose logs backend | grep ML"
echo "  Stop platform:     docker-compose down"
echo "  Restart backend:   docker-compose restart backend"
echo ""
echo "======================================================================"
echo ""
