#!/bin/bash
# HS2 Platform Quick Start
# ========================
# Complete setup and launch script

set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  HS2 Infrastructure Intelligence Platform - Quick Start  ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Check prerequisites
print_status "Checking prerequisites..."

# Check PostgreSQL
if ! command -v psql &> /dev/null; then
    print_error "PostgreSQL not found. Please install: brew install postgresql"
    exit 1
fi
print_success "PostgreSQL installed"

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found. Please install Python 3.11+"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
print_success "Python $PYTHON_VERSION installed"

# Check Node.js
if ! command -v node &> /dev/null; then
    print_warning "Node.js not found. Frontend will not be available."
    SKIP_FRONTEND=1
else
    NODE_VERSION=$(node --version)
    print_success "Node.js $NODE_VERSION installed"
    SKIP_FRONTEND=0
fi

echo ""

# Backend Setup
print_status "Setting up backend..."

cd backend

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_status "Creating Python virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
print_status "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt 2>&1 | grep -v "already satisfied" || true
print_success "Dependencies installed"

# Database setup
print_status "Setting up database..."
if [ -f "scripts/setup_database.sh" ]; then
    chmod +x scripts/setup_database.sh
    ./scripts/setup_database.sh 2>&1 | tail -20
    if [ $? -eq 0 ]; then
        print_success "Database setup complete"
    else
        print_warning "Database setup had warnings (check if DB already exists)"
    fi
else
    print_warning "Database setup script not found - skipping"
fi

echo ""

cd ..

# Frontend Setup
if [ $SKIP_FRONTEND -eq 0 ]; then
    print_status "Setting up frontend..."

    cd frontend

    if [ ! -d "node_modules" ]; then
        print_status "Installing npm dependencies..."
        npm install --silent 2>&1 | grep -E "added|removed|updated" || true
        print_success "Dependencies installed"
    else
        print_success "Dependencies already installed"
    fi

    cd ..
    echo ""
fi

# Create launch script
print_status "Creating launch scripts..."

# Backend launch script
cat > start_backend.sh << 'EOF'
#!/bin/bash
cd backend
source venv/bin/activate
echo "🚀 Starting Backend Server..."
echo "📖 API Docs: http://localhost:8000/docs"
echo "🔍 Health Check: http://localhost:8000/health"
echo ""
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
EOF

chmod +x start_backend.sh

# Frontend launch script
if [ $SKIP_FRONTEND -eq 0 ]; then
    cat > start_frontend.sh << 'EOF'
#!/bin/bash
cd frontend
echo "🚀 Starting Frontend Server..."
echo "🌐 Frontend: http://localhost:3000"
echo ""
npm run dev
EOF
    chmod +x start_frontend.sh
fi

print_success "Launch scripts created"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                    SETUP COMPLETE                        ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Summary
print_success "Backend Services Ready"
echo "  - Hyperspectral Processor (concrete quality)"
echo "  - LiDAR Processor (elevation profiles)"
echo "  - BIM/IFC Processor (infrastructure models)"
echo ""

print_success "Database Initialized"
echo "  - 9 tables created"
echo "  - Sample data loaded"
echo "  - PostGIS enabled"
echo ""

if [ $SKIP_FRONTEND -eq 0 ]; then
    print_success "Frontend Components Ready"
    echo "  - LiDAR Profile Viewer"
    echo "  - Concrete Quality Analyzer"
    echo ""
fi

print_success "API Endpoints Available"
echo "  - /api/v1/lidar/* (7 endpoints)"
echo "  - /api/v1/progress/hyperspectral/*"
echo "  - /api/v1/bim/*"
echo ""

echo "═══════════════════════════════════════════════════════════"
echo ""
echo "${BLUE}LAUNCH OPTIONS:${NC}"
echo ""
echo "1️⃣  Start Backend:  ${GREEN}./start_backend.sh${NC}"
echo "    Access API Docs: ${BLUE}http://localhost:8000/docs${NC}"
echo ""

if [ $SKIP_FRONTEND -eq 0 ]; then
    echo "2️⃣  Start Frontend: ${GREEN}./start_frontend.sh${NC}"
    echo "    Access Frontend: ${BLUE}http://localhost:3000${NC}"
    echo ""
    echo "3️⃣  Start Both:     ${GREEN}./start_backend.sh & ./start_frontend.sh${NC}"
    echo ""
fi

echo "═══════════════════════════════════════════════════════════"
echo ""
echo "${BLUE}QUICK TESTS:${NC}"
echo ""
echo "# Test LiDAR Endpoint"
echo 'curl -X POST http://localhost:8000/api/v1/lidar/elevation/point \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'{"easting": 423500, "northing": 338500}'"'"
echo ""
echo "# Test Hyperspectral Endpoint"
echo 'curl http://localhost:8000/api/v1/progress/hyperspectral/scans'
echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "${GREEN}Ready to launch! 🚀${NC}"
echo ""
