#!/bin/bash
# Database Setup Script
# =====================
# Sets up database, loads schema, and populates sample data

set -e  # Exit on error

echo "=================================="
echo "Database Setup for HS2 Platform"
echo "=================================="
echo ""

# Configuration
DB_NAME="${DB_NAME:-gpr_platform}"
DB_USER="${DB_USER:-gpr_app_user}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"

echo "Database Configuration:"
echo "  Host: $DB_HOST"
echo "  Port: $DB_PORT"
echo "  Database: $DB_NAME"
echo "  User: $DB_USER"
echo ""

# Check if PostgreSQL is running
echo "1. Checking PostgreSQL connection..."
if ! pg_isready -h $DB_HOST -p $DB_PORT > /dev/null 2>&1; then
    echo "❌ PostgreSQL is not running or not accessible"
    echo "   Start PostgreSQL and try again"
    exit 1
fi
echo "✅ PostgreSQL is running"
echo ""

# Check if database exists
echo "2. Checking if database exists..."
if psql -h $DB_HOST -p $DB_PORT -U $DB_USER -lqt | cut -d \| -f 1 | grep -qw $DB_NAME; then
    echo "✅ Database '$DB_NAME' exists"
else
    echo "⚠️  Database '$DB_NAME' does not exist"
    echo "   Creating database..."
    createdb -h $DB_HOST -p $DB_PORT -U $DB_USER $DB_NAME || {
        echo "❌ Failed to create database"
        exit 1
    }
    echo "✅ Database created"
fi
echo ""

# Run schema initialization
echo "3. Initializing database schema..."
echo "   - Creating tables"
echo "   - Enabling PostGIS"
echo "   - Creating indexes"
echo ""

psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f scripts/init_database.sql

if [ $? -eq 0 ]; then
    echo "✅ Schema initialized successfully"
else
    echo "❌ Schema initialization failed"
    exit 1
fi
echo ""

# Load sample data
echo "4. Loading sample data..."
echo "   - LiDAR DTM tiles"
echo "   - Hyperspectral training samples"
echo "   - BIM test models"
echo ""

python scripts/load_sample_data.py

if [ $? -eq 0 ]; then
    echo "✅ Sample data loaded successfully"
else
    echo "❌ Sample data loading failed"
    exit 1
fi
echo ""

# Verify setup
echo "5. Verifying setup..."
TABLES=$(psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE';")
echo "   Tables created: $TABLES"

if [ "$TABLES" -ge 9 ]; then
    echo "✅ Database setup verified"
else
    echo "⚠️  Expected at least 9 tables, found $TABLES"
fi
echo ""

echo "=================================="
echo "✅ DATABASE SETUP COMPLETE"
echo "=================================="
echo ""
echo "Next Steps:"
echo "  1. Start the backend server: uvicorn app.main:app --reload"
echo "  2. Check API docs: http://localhost:8000/docs"
echo "  3. Test LiDAR endpoints: http://localhost:8000/api/v1/lidar/tiles"
echo ""
