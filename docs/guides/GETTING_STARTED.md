# Getting Started - Infrastructure Intelligence Platform

> Complete guide to set up and run the platform locally

---

## Quick Start (5 minutes)

### Prerequisites

- **Docker Desktop** (latest version)
- **8GB RAM** minimum (16GB recommended)
- **20GB free disk space**

### 1. Generate Environment Configuration

```bash
cd /Users/nitishgautam/Code/prototype/ground-truth

# Generate secure keys and create .env file
chmod +x setup_env.sh
./setup_env.sh
```

The script will prompt:
```
.env file already exists. Overwrite? (y/n):
```
- Type `y` if first time setup or to regenerate keys
- Type `n` to keep existing configuration

### 2. Start All Services

```bash
# Start all containers in detached mode
docker compose up -d

# Verify all services are healthy (wait ~30 seconds)
docker compose ps
```

**Expected Output**:
```
NAME                      STATUS
infrastructure-backend    Up (healthy)
infrastructure-frontend   Up (healthy)
infrastructure-postgres   Up (healthy)
infrastructure-redis      Up (healthy)
infrastructure-minio      Up (healthy)
```

### 3. Access the Platform

| Service | URL | Credentials |
|---------|-----|-------------|
| **Frontend UI** | http://localhost:3003 | N/A |
| **Backend API** | http://localhost:8002 | N/A |
| **API Docs** | http://localhost:8002/docs | N/A |
| **MinIO Console** | http://localhost:9011 | minioadmin / minioadmin |

### 4. Verify Installation

```bash
# Test backend health
curl http://localhost:8002/health

# Expected response:
# {"status":"healthy","service":"Infrastructure Intelligence Platform","version":"1.0.0"}

# Test database connection
docker compose exec postgres psql -U gpr_user -d gpr_db -c "SELECT COUNT(*) FROM gpr_surveys;"
```

---

## Database Setup

### Create Tables (First Time)

```bash
# Create all 17 database tables
docker compose exec backend python /app/scripts/database/create_tables.py
```

**Expected Output**:
```
Creating database tables...
Found 17 tables to create:
  - gpr_surveys
  - gpr_scans
  - environmental_data
  [... 14 more tables]
✅ All tables created successfully!
```

### Verify Tables

```bash
# List all tables
docker compose exec backend python /app/scripts/database/list_tables.py

# Or use psql directly
docker compose exec postgres psql -U gpr_user -d gpr_db -c "\dt"
```

### Reset Database (if needed)

```bash
# WARNING: This drops all data!
docker compose exec backend python /app/scripts/database/reset_database.py
```

---

## Import Sample Data

### Import GPR Surveys (University of Twente Dataset)

```bash
# Import 10 sample surveys (takes ~5 seconds)
docker compose exec backend python /app/scripts/import_twente_gpr_data.py
```

**What Gets Imported**:
- 10 real GPR surveys from Netherlands
- 100+ SEG-Y radargram files
- Environmental data (soil, weather, permittivity)
- Utility information (materials, depths, types)

**Verify Import**:
```bash
# Check surveys in database
curl http://localhost:8002/api/v1/gpr/surveys | python3 -m json.tool

# Count surveys
docker compose exec postgres psql -U gpr_user -d gpr_db -c \
  "SELECT COUNT(*) FROM gpr_surveys WHERE survey_name LIKE 'Twente%';"
```

**Import More Data**:
Edit the script limits in `/backend/scripts/import_twente_gpr_data.py`:
```python
importer.import_dataset(
    extract_limit=13,   # Extract all 13 ZIP files (default: 2)
    survey_limit=125    # Process all 125 surveys (default: 10)
)
```

---

## Troubleshooting

### Services Not Starting

**Problem**: Containers exit immediately or show unhealthy status

**Solutions**:
```bash
# Check logs for errors
docker compose logs backend
docker compose logs postgres

# Restart all services
docker compose restart

# Full reset (removes containers)
docker compose down
docker compose up -d
```

### Port Conflicts

**Problem**: `Bind for 0.0.0.0:8002 failed: port is already allocated`

**Ports Used**:
- 3003 - Frontend
- 8002 - Backend API
- 5433 - PostgreSQL
- 6379 - Redis
- 9000 - MinIO API
- 9001 - MinIO Console

**Solution**: Change ports in `docker-compose.yml`:
```yaml
ports:
  - "8003:8000"  # Changed from 8002 to 8003
```

### Database Connection Errors

**Problem**: `could not connect to server: Connection refused`

**Solutions**:
```bash
# Check postgres is running
docker compose ps postgres

# Restart postgres
docker compose restart postgres

# Check database exists
docker compose exec postgres psql -U gpr_user -l

# Recreate database
docker compose exec backend python /app/scripts/database/reset_database.py
```

### Frontend Shows 404

**Problem**: http://localhost:3003 returns 404 Not Found

**Solutions**:
```bash
# Rebuild frontend
docker compose build --no-cache frontend
docker compose up -d frontend

# Check frontend logs
docker compose logs frontend

# Verify files exist
docker compose exec frontend ls -la /app/src/
```

### Import Script Fails

**Problem**: `Data directory not found: /datasets/raw/twente_gpr`

**Solutions**:
```bash
# Check volume mount
docker compose exec backend ls -la /datasets/

# Verify datasets exist on host
ls -la /Users/nitishgautam/Code/prototype/ground-truth/datasets/raw/twente_gpr/

# Restart backend with volume mount
docker compose down backend
docker compose up -d backend
```

### Enterprise Integration Errors

**Problem**: `Failed to connect to AIMS API` or `401 Unauthorized` from CDEs

**Solutions**:
```bash
# Check environment variables are set
docker compose exec backend env | grep AIMS
docker compose exec backend env | grep BIM360

# Test API credentials manually
curl -H "Authorization: Bearer YOUR_TOKEN" https://aims.yourcompany.com/api/health

# Check integration status
curl http://localhost:8002/api/v1/integrations/status

# View sync logs
docker compose exec postgres psql -U gpr_user -d gpr_db -c \
  "SELECT * FROM sync_logs ORDER BY created_at DESC LIMIT 10;"
```

### Certificate Parsing Issues

**Problem**: OCR not extracting qualifications correctly

**Solutions**:
```bash
# Check OCR service is configured
docker compose exec backend env | grep AZURE_DOCUMENT_INTELLIGENCE_KEY

# Test certificate parsing manually
curl -X POST http://localhost:8002/api/v1/certificates/{cert_id}/parse

# View parsing logs
docker compose logs backend | grep "certificate_parser"

# Use higher quality scan (300+ DPI recommended)
# Re-upload certificate with better quality
```

---

## Development Workflow

### Backend Development

```bash
# Watch backend logs
docker compose logs -f backend

# Enter backend container
docker compose exec backend bash

# Run Python scripts
docker compose exec backend python /app/scripts/your_script.py

# Install new Python package
docker compose exec backend pip install package-name
# Then add to backend/requirements.txt
```

### Frontend Development

```bash
# Watch frontend logs
docker compose logs -f frontend

# Enter frontend container
docker compose exec frontend sh

# Install new NPM package
docker compose exec frontend npm install package-name

# Rebuild frontend
docker compose build frontend
docker compose up -d frontend
```

### Database Operations

```bash
# Connect to PostgreSQL
docker compose exec postgres psql -U gpr_user -d gpr_db

# Common SQL queries
SELECT COUNT(*) FROM gpr_surveys;
SELECT survey_name, location_id, status FROM gpr_surveys LIMIT 10;
SELECT table_name FROM information_schema.tables WHERE table_schema='public';

# Export data
docker compose exec postgres pg_dump -U gpr_user gpr_db > backup.sql

# Import data
docker compose exec -T postgres psql -U gpr_user -d gpr_db < backup.sql
```

---

## Enterprise Integration Setup (Phase 1D+)

### Connect to Enterprise Systems

**Supported Integrations**:
- **AIMS** (Asset Information Management System)
- **CDEs**: BIM 360, Aconex, ProjectWise, Viewpoint
- **Document Management**: SharePoint, Microsoft Teams
- **ERP Systems**: SAP, Oracle (via REST/GraphQL APIs)

**Configuration**:
```bash
# Edit .env file to add enterprise API credentials
AIMS_API_URL=https://aims.yourcompany.com/api
AIMS_API_KEY=your-api-key
BIM360_CLIENT_ID=your-client-id
BIM360_CLIENT_SECRET=your-secret
SHAREPOINT_TENANT=yourcompany.sharepoint.com
SHAREPOINT_CLIENT_ID=your-client-id
```

**Test Integration**:
```bash
# Test AIMS connection
curl http://localhost:8002/api/v1/integrations/status

# Trigger data sync
curl -X POST http://localhost:8002/api/v1/integrations/sync \
  -H "Content-Type: application/json" \
  -d '{"system": "aims", "sync_type": "full"}'
```

---

## Certificate Management (Phase 1D)

### Import Certificates

**Supported Formats**:
- PDF certificates (scanned or digital)
- Excel spreadsheets with qualification data
- CSV files with certificate metadata

**Import via API**:
```bash
# Upload certificate
curl -X POST http://localhost:8002/api/v1/certificates \
  -F "file=@certificate.pdf" \
  -F "asset_id=AST-001" \
  -F "certificate_type=qualification"

# Trigger OCR + NLP parsing
curl -X POST http://localhost:8002/api/v1/certificates/{cert_id}/parse

# Get extracted qualifications
curl http://localhost:8002/api/v1/certificates/{cert_id}/qualifications
```

**Bulk Import**:
```bash
# Import from folder
docker compose exec backend python /app/scripts/import_certificates.py \
  --folder /datasets/raw/certificates/ \
  --asset-type "railway_track"
```

**Certificate Validation**:
```bash
# Validate certificate against requirements
curl -X POST http://localhost:8002/api/v1/certificates/{cert_id}/validate

# Get expiring certificates (alerts)
curl http://localhost:8002/api/v1/certificates/expiring?days=30
```

---

## Asset Management (Phase 1D)

### Create Assets

```bash
# Create single asset
curl -X POST http://localhost:8002/api/v1/assets \
  -H "Content-Type: application/json" \
  -d '{
    "asset_code": "TRK-001-SEG-05",
    "asset_type": "railway_track_segment",
    "location": {"lat": 51.5074, "lon": -0.1278},
    "status": "installation",
    "metadata": {
      "length_m": 25.0,
      "material": "continuous_welded_rail",
      "installation_date": "2025-01-15"
    }
  }'

# Bulk import from Excel
docker compose exec backend python /app/scripts/import_assets.py \
  --file /datasets/raw/assets/railway_assets.xlsx \
  --sheet "Track_Segments"
```

### Track Asset Lifecycle

```bash
# Get asset lifecycle history
curl http://localhost:8002/api/v1/assets/{asset_id}/lifecycle

# Update asset status
curl -X PUT http://localhost:8002/api/v1/assets/{asset_id} \
  -H "Content-Type: application/json" \
  -d '{"status": "commissioned", "commissioned_date": "2025-02-01"}'
```

---

## Next Steps

### 1. Explore the API
- Visit http://localhost:8002/docs
- Try the interactive Swagger UI
- Test GPR survey endpoints

### 2. View Sample Data
- Open http://localhost:3003
- Check GPR surveys in the UI
- Explore environmental data

### 3. Import More Data
- Run import script for all 125 surveys
- See [DATA_GUIDE.md](DATA_GUIDE.md) for data sources

### 4. Start Development
- Review architecture in [architecture/](architecture/)
- Check API documentation
- Begin Phase 1B (BIM integration) or Phase 1C (LiDAR)

### 5. Explore Asset Intelligence (Phase 1D+)
- Import certificates and test OCR parsing
- Set up enterprise system integrations
- Create assets and track lifecycle
- Configure TAEM compliance requirements
- View real-time assurance dashboard

---

## Configuration Reference

### Environment Variables (.env)

```bash
# PostgreSQL
POSTGRES_USER=gpr_user
POSTGRES_PASSWORD=<generated>
POSTGRES_DB=gpr_db

# MinIO (S3-compatible storage)
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin

# Backend Security
SECRET_KEY=<generated>
DEBUG=true

# CORS (add your frontend URL)
CORS_ORIGINS=http://localhost:3003,http://localhost:8002

# API Keys (Phase 2)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

### Docker Compose Services

```yaml
# Core Services
services:
  backend:       # FastAPI application
  frontend:      # React UI
  postgres:      # PostgreSQL 16 + PGVector + PostGIS
  redis:         # Cache layer
  minio:         # S3-compatible storage
```

### Ports Summary

| Port | Service | Protocol |
|------|---------|----------|
| 3003 | Frontend | HTTP |
| 8002 | Backend API | HTTP |
| 5433 | PostgreSQL | TCP |
| 6379 | Redis | TCP |
| 9000 | MinIO API | HTTP |
| 9001 | MinIO Console | HTTP |

---

## Support

### Documentation
- **API Reference**: http://localhost:8002/docs
- **Data Sources**: [DATA_GUIDE.md](DATA_GUIDE.md)
- **Architecture**: [architecture/ARCHITECTURE.md](architecture/ARCHITECTURE.md)

### Logs
```bash
# All services
docker compose logs

# Specific service
docker compose logs backend
docker compose logs -f frontend  # Follow mode
docker compose logs --tail=100 postgres
```

### Health Checks
```bash
# Backend
curl http://localhost:8002/health

# Database
docker compose exec postgres pg_isready -U gpr_user

# Redis
docker compose exec redis redis-cli ping

# MinIO
curl http://localhost:9010/minio/health/live
```

---

**You're now ready to start developing on the Infrastructure Intelligence Platform!** 🚀
