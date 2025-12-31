# How to Build and Run the HS2 Infrastructure Intelligence Platform

## Quick Start (Recommended)

### Option 1: One-Command Startup with ML Models

```bash
# Start everything (backend, frontend, database, ML models)
./START_HS2_WITH_ML.sh
```

This script will:
- ✅ Check ML models exist
- ✅ Build Docker containers
- ✅ Start all services (Postgres, Redis, Neo4j, MinIO, Backend, Frontend)
- ✅ Wait for services to be healthy
- ✅ Verify ML models are loaded

**Access the application:**
- Frontend: http://localhost:3003
- HS2 Dashboard: http://localhost:3003/hs2
- Hyperspectral Viewer: http://localhost:3003/hs2/hyperspectral
- API Docs: http://localhost:8007/docs

---

## Option 2: Standard Docker Compose

### Build and Start

```bash
# Build all containers
docker-compose build

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### Stop Services

```bash
# Stop all containers
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v
```

---

## Option 3: Individual Services

### Backend Only

```bash
# Build backend
docker-compose build backend

# Start backend + dependencies
docker-compose up -d postgres redis neo4j minio backend

# View backend logs
docker-compose logs -f backend
```

### Frontend Only

```bash
# Build frontend
docker-compose build frontend

# Start frontend
docker-compose up -d frontend

# View frontend logs
docker-compose logs -f frontend
```

---

## ML Models Setup

### Train ML Models (First Time)

```bash
cd ml_artifacts
python3 train_models.py

# Expected output:
# ✅ 5 model files created in models/ directory
# - material_classifier_v1.pkl (622KB)
# - strength_regressor_v1.pkl (76KB)
# - quality_regressor_v1.pkl (76KB)
# - confidence_regressor_v1.pkl (261KB)
# - feature_scaler.pkl (7.4KB)
```

**Note**: ML models are already trained and included in the repository.

---

## Service URLs (After Startup)

| Service | URL | Credentials |
|---------|-----|-------------|
| **Frontend** | http://localhost:3003 | - |
| **HS2 Dashboard** | http://localhost:3003/hs2 | - |
| **Hyperspectral Viewer** | http://localhost:3003/hs2/hyperspectral | - |
| **Backend API** | http://localhost:8007 | - |
| **API Documentation** | http://localhost:8007/docs | - |
| **MinIO Console** | http://localhost:9011 | minioadmin / (from .env) |
| **Neo4j Browser** | http://localhost:7475 | neo4j / hs2_graph_2024 |
| **Flower (Celery)** | http://localhost:5555 | - |

---

## Environment Setup

### Create .env file

```bash
# Copy example configuration
cp .env.example .env

# Edit with your values (optional - defaults work)
nano .env
```

**Required variables** (already set in .env.example):
```bash
POSTGRES_USER=gpr_user
POSTGRES_PASSWORD=Lb1RcTOayzhQlwhU2E9dbA
POSTGRES_DB=gpr_db
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=mD9E3_kgZJAPRjNvBWOxGQ
SECRET_KEY=changeme_random_secret_key_at_least_32_characters_long
```

---

## Troubleshooting

### Port Conflicts

If you see "port already in use" errors:

```bash
# Check which ports are in use
lsof -ti:8007,3003,9010,7475,6380,5433

# Stop conflicting containers
docker-compose down

# Restart
docker-compose up -d
```

### Database Not Initialized

```bash
# Check database tables
docker exec infrastructure-postgres psql -U gpr_user -d gpr_db -c "\dt"

# If empty, check initialization logs
docker-compose logs postgres | grep -i "database\|init"
```

### ML Models Not Loading

```bash
# Check models exist
ls -lh ml_artifacts/models/*.pkl

# Check backend can access them
docker exec infrastructure-backend ls -la /app/ml_artifacts/models/

# View backend logs for ML loading
docker-compose logs backend | grep -i "ml\|model"
```

### Frontend Not Connecting to Backend

```bash
# Check backend is running
curl http://localhost:8007/health

# Check frontend environment
docker exec infrastructure-frontend env | grep VITE_API_URL

# Should show: VITE_API_URL=http://localhost:8007
```

---

## Rebuild Everything (Fresh Start)

```bash
# Stop and remove all containers + volumes
docker-compose down -v

# Remove all images
docker-compose down --rmi all

# Rebuild and start
docker-compose build --no-cache
docker-compose up -d

# Check status
docker-compose ps
```

---

## Development Mode

### Hot Reload (Backend)

Backend already has hot reload enabled via `--reload` flag in docker-compose.yml:

```yaml
command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Changes to Python files auto-reload!**

### Hot Reload (Frontend)

Frontend uses Vite with HMR (Hot Module Replacement):

```yaml
command: npm run dev
```

**Changes to React/TS files auto-reload!**

### View Live Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
```

---

## Testing

### Test Backend API

```bash
# Health check
curl http://localhost:8007/health

# Get assets
curl "http://localhost:8007/api/v1/hs2/assets?limit=5" | jq

# Get dashboard summary
curl http://localhost:8007/api/v1/hs2/dashboard/summary | jq

# Test ML hyperspectral analysis
curl -X POST "http://localhost:8007/api/v1/progress/hyperspectral/analyze-material" \
  -F "file=@datasets/raw/hyperspectral/umkc-material-surfaces/Concrete/HSI_TIFF_50x50/Auto119.tiff" \
  | jq
```

### Test Database

```bash
# Connect to database
docker exec -it infrastructure-postgres psql -U gpr_user -d gpr_db

# Run queries
SELECT COUNT(*) FROM hs2_assets;
SELECT readiness_status, COUNT(*) FROM hs2_assets GROUP BY readiness_status;

# Exit
\q
```

### Run ML Tests

```bash
# Standalone ML validation
python3 tests/ml/test_ml_standalone.py

# API integration test
python3 tests/ml/test_ml_api_response.py
```

---

## Production Deployment

### Build Production Images

```bash
# Build production backend
docker-compose -f docker-compose.yml -f docker-compose.prod.yml build backend

# Build production frontend
docker-compose -f docker-compose.yml -f docker-compose.prod.yml build frontend
```

### Environment Variables

```bash
# Set production variables
export DEBUG=false
export CORS_ORIGINS=https://yourdomain.com
export SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
```

---

## Summary of Commands

```bash
# 🚀 QUICKEST WAY TO START
./START_HS2_WITH_ML.sh

# 🔨 Manual Docker Compose
docker-compose build
docker-compose up -d

# 🧪 Check Everything
curl http://localhost:8007/health
curl http://localhost:3003

# 🔍 View Logs
docker-compose logs -f

# 🛑 Stop Everything
docker-compose down

# 🔄 Fresh Start
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

---

**Created**: December 31, 2025  
**Status**: ✅ Ready to use  
**All services running on conflict-free ports**

