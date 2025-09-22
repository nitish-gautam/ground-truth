#!/bin/bash
# Underground Utility Detection Platform - Dataset Setup Script
# =============================================================

set -e  # Exit on any error

echo "🚀 Underground Utility Detection Platform - Dataset Setup"
echo "=========================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.11"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    print_warning "Python 3.11+ recommended (current: $PYTHON_VERSION)"
else
    print_success "Python version OK: $PYTHON_VERSION"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating Python virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Install basic requirements for dataset downloads
print_status "Installing dataset download dependencies..."
pip install --upgrade pip
pip install requests beautifulsoup4 lxml tqdm python-dotenv

print_success "Basic dependencies installed"

# Make download scripts executable
print_status "Making download scripts executable..."
chmod +x datasets/download_scripts/*.py

# Create directory structure
print_status "Creating dataset directory structure..."
mkdir -p datasets/raw/{twente_gpr,mojahid_images,usag_reports,bgs_data,uk_gas_networks,uk_power_networks,pas128_docs}
mkdir -p datasets/processed/{training_data,validation_data,embeddings,knowledge_base}
mkdir -p datasets/synthetic/{gprmax_models,synthetic_gpr,augmented_data}

print_success "Directory structure created"

# Check dataset status
print_status "Checking dataset availability..."
cd datasets/download_scripts
python3 download_all.py --status

echo ""
echo "🎯 QUICK START COMMANDS:"
echo "========================"
echo ""
echo "1. Download all automatic datasets:"
echo "   cd datasets/download_scripts"
echo "   python download_all.py --auto"
echo ""
echo "2. Check what was downloaded:"
echo "   python download_all.py --status"
echo ""
echo "3. Get detailed help:"
echo "   python download_all.py --help-detailed"
echo ""
echo "4. Install full requirements (for development):"
echo "   pip install -r ../../requirements.txt"
echo ""

print_success "Dataset setup completed!"
print_warning "Manual downloads required for some datasets - see individual README files"

echo ""
echo "📁 DATASET PRIORITIES:"
echo "====================="
echo "Priority 1 (Essential): Twente GPR, Mojahid Images, PAS 128"
echo "Priority 2 (Important): USAG Reports, UK Networks"
echo "Priority 3 (Optional):  BGS Geotechnical Data"
echo ""
echo "🤖 AUTOMATIC: USAG, PAS 128 (partial)"
echo "👤 MANUAL:    Twente, Mojahid, BGS, UK Networks"