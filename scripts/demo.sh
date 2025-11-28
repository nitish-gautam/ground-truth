#!/bin/bash

################################################################################
# HS2 Assurance Intelligence Demonstrator - One-Command Demo Setup
################################################################################
#
# This script sets up the complete HS2 demo environment:
# 1. Starts all Docker services
# 2. Creates database tables
# 3. Generates placeholder data
# 4. Seeds the database
# 5. Runs TAEM evaluation
# 6. Displays access URLs
#
# Usage: ./scripts/demo.sh [--reset]
#
# Options:
#   --reset    Drop existing HS2 tables and start fresh
#
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_step() {
    echo -e "${BLUE}==>${NC} $1"
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

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker Desktop and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Function to check if docker-compose is available
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null 2>&1; then
        print_error "docker-compose is not installed. Please install it and try again."
        exit 1
    fi
    print_success "docker-compose is available"
}

# Parse command-line arguments
RESET=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --reset)
            RESET=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./scripts/demo.sh [--reset]"
            exit 1
            ;;
    esac
done

# Clear screen for better visibility
clear

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  HS2 Assurance Intelligence Demonstrator - Demo Setup             ║"
echo "║  Infrastructure Intelligence Platform                              ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Pre-flight checks
print_step "Step 1/7: Pre-flight checks"
check_docker
check_docker_compose
echo ""

# Step 2: Build and start Docker services
print_step "Step 2/7: Building and starting Docker services"
echo "This may take 2-5 minutes if building for the first time..."
docker compose build --parallel
docker compose up -d

# Wait for services to be healthy
print_step "Waiting for services to be ready..."
sleep 10

# Check if backend is responding
MAX_RETRIES=30
RETRY_COUNT=0
while ! curl -s http://localhost:8002/health > /dev/null 2>&1; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        print_error "Backend failed to start after 30 retries"
        print_warning "Check logs with: docker compose logs backend"
        exit 1
    fi
    echo -n "."
    sleep 2
done
echo ""
print_success "All services are running"
echo ""

# Step 3: Create HS2 tables (or reset if requested)
if [ "$RESET" = true ]; then
    print_step "Step 3/7: Resetting HS2 database tables"
    print_warning "Dropping existing HS2 tables..."
    docker compose exec -T postgres psql -U gpr_user -d gpr_db <<-EOSQL
        DROP TABLE IF EXISTS hs2_rule_evaluations CASCADE;
        DROP TABLE IF EXISTS hs2_taem_rules CASCADE;
        DROP TABLE IF EXISTS hs2_certificates CASCADE;
        DROP TABLE IF EXISTS hs2_costs CASCADE;
        DROP TABLE IF EXISTS hs2_deliverables CASCADE;
        DROP TABLE IF EXISTS hs2_assets CASCADE;
        DROP MATERIALIZED VIEW IF EXISTS hs2_asset_readiness_summary CASCADE;
EOSQL
    print_success "Existing tables dropped"
fi

print_step "Step 3/7: Creating HS2 database tables"
docker compose exec backend python scripts/database/create_hs2_tables.py
print_success "Database tables created"
echo ""

# Step 4: Generate placeholder data
print_step "Step 4/7: Generating placeholder data (50 assets, 900+ records)"
docker compose exec backend python scripts/generate_placeholder_data.py
print_success "Placeholder data generated in placeholder_data/"
echo ""

# Step 5: Seed database
print_step "Step 5/7: Seeding database with placeholder data"
docker compose exec backend python scripts/seed_database.py
print_success "Database seeded successfully"
echo ""

# Step 6: Run TAEM evaluation
print_step "Step 6/7: Running TAEM evaluation on all assets"
docker compose exec backend python scripts/evaluate_all_assets.py
print_success "TAEM evaluation complete"
echo ""

# Step 7: Verify data
print_step "Step 7/7: Verifying data integrity"
ASSET_COUNT=$(docker compose exec -T postgres psql -U gpr_user -d gpr_db -t -c "SELECT COUNT(*) FROM hs2_assets;" | xargs)
DELIVERABLE_COUNT=$(docker compose exec -T postgres psql -U gpr_user -d gpr_db -t -c "SELECT COUNT(*) FROM hs2_deliverables;" | xargs)
EVALUATION_COUNT=$(docker compose exec -T postgres psql -U gpr_user -d gpr_db -t -c "SELECT COUNT(*) FROM hs2_rule_evaluations;" | xargs)

print_success "Assets: $ASSET_COUNT"
print_success "Deliverables: $DELIVERABLE_COUNT"
print_success "Evaluations: $EVALUATION_COUNT"
echo ""

# Display readiness summary
print_step "Asset Readiness Summary:"
docker compose exec -T postgres psql -U gpr_user -d gpr_db -c "
    SELECT
        readiness_status as \"Status\",
        COUNT(*) as \"Count\",
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as \"Percentage\",
        ROUND(AVG(taem_evaluation_score), 2) as \"Avg Score\"
    FROM hs2_assets
    GROUP BY readiness_status
    ORDER BY readiness_status;
"
echo ""

# Display access information
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  🎉 Demo Environment Ready!                                        ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Access the platform:"
echo ""
echo "  📊 Backend API:       http://localhost:8002"
echo "  📚 API Documentation: http://localhost:8002/docs"
echo "  🔄 API Redoc:         http://localhost:8002/redoc"
echo "  🌐 Frontend UI:       http://localhost:3003"
echo "  🗄️  MinIO Console:     http://localhost:9001"
echo ""
echo "Quick API Tests:"
echo ""
echo "  # List all assets"
echo "  curl http://localhost:8002/api/v1/hs2/assets | jq"
echo ""
echo "  # Get dashboard summary"
echo "  curl http://localhost:8002/api/v1/hs2/dashboard/summary | jq"
echo ""
echo "  # Get specific asset details"
echo "  curl http://localhost:8002/api/v1/hs2/assets/VA-007 | jq"
echo ""
echo "  # Get evaluation history"
echo "  curl http://localhost:8002/api/v1/hs2/taem/evaluations/VA-007 | jq"
echo ""
echo "Database Access:"
echo ""
echo "  docker compose exec postgres psql -U gpr_user -d gpr_db"
echo ""
echo "View Logs:"
echo ""
echo "  docker compose logs -f backend"
echo "  docker compose logs -f frontend"
echo ""
echo "Stop Demo:"
echo ""
echo "  docker compose down"
echo ""
echo "Reset Demo (fresh start):"
echo ""
echo "  ./scripts/demo.sh --reset"
echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  📖 Documentation: See HS2_ORCHESTRATION_PLAN.md                   ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
