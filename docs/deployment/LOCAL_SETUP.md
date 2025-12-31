# Local Development Setup Guide

Complete guide for setting up the Infrastructure Intelligence Platform on your local machine using Docker Compose.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start (5 Minutes)](#quick-start-5-minutes)
3. [Detailed Setup](#detailed-setup)
4. [Docker Compose Services](#docker-compose-services)
5. [Environment Configuration](#environment-configuration)
6. [Verifying the Setup](#verifying-the-setup)
7. [Sample Data Loading](#sample-data-loading)
8. [Development Workflow](#development-workflow)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

1. **Docker Desktop** (includes Docker Compose)
   - **Mac**: Download from https://www.docker.com/products/docker-desktop
   - **Windows**: Docker Desktop for Windows
   - **Linux**: Install Docker Engine + Docker Compose plugin

   ```bash
   # Verify installation
   docker --version  # Should be 20.10+
   docker compose version  # Should be 2.0+
   ```

2. **Git**
   ```bash
   git --version  # Should be 2.0+
   ```

3. **Code Editor** (Optional but recommended)
   - VS Code with Docker extension
   - PyCharm Professional
   - Any text editor

### System Requirements

- **RAM**: 8GB minimum, 16GB recommended
- **Disk Space**: 20GB free space
- **CPU**: 4 cores recommended
- **OS**: macOS, Windows 10/11, Linux (Ubuntu 20.04+)

---

## Quick Start (5 Minutes)

For experienced developers who just want to get started:

```bash
# 1. Clone repository
git clone https://github.com/your-org/infrastructure-intelligence-platform.git
cd infrastructure-intelligence-platform

# 2. Copy environment file
cp .env.example .env

# 3. Start all services
docker compose up -d

# 4. Wait for services to be healthy (~2 minutes)
docker compose ps

# 5. Initialize database
docker compose exec backend python -m app.db.init

# 6. Load sample data (optional)
docker compose exec backend python -m app.scripts.load_sample_data

# 7. Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000/docs
# MinIO Console: http://localhost:9011
# Flower (Celery): http://localhost:5555
```

---

## Detailed Setup

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/your-org/infrastructure-intelligence-platform.git
cd infrastructure-intelligence-platform

# Check current branch
git branch
# Should show: * main

# Verify directory structure
ls -la
# Should see: backend/, frontend/, database/, docker-compose.yml, etc.
```

### Step 2: Environment Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your preferred editor
nano .env  # or: code .env, vim .env
```

**Key Variables to Configure**:

```bash
# Database
POSTGRES_USER=gpr_user
POSTGRES_PASSWORD=changeme_secure_password
POSTGRES_DB=gpr_db

# MinIO (S3-compatible storage)
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=changeme_minio_password

# Backend
SECRET_KEY=changeme_random_secret_key_at_least_32_chars
DEBUG=true

# Frontend
VITE_API_URL=http://localhost:8000
VITE_TILE_SERVER_URL=http://localhost:8080

# Optional: OpenAI for LLM features (Phase 2)
OPENAI_API_KEY=sk-your-key-here  # Leave empty for now
```

**Generate Secure Secrets**:
```bash
# Generate secure random strings
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Step 3: Download Sample Data (Optional)

Before starting Docker, download sample data for testing:

```bash
# Create data directories
mkdir -p data/{gpr,bim,lidar,tiles}

# Download sample LiDAR (Environment Agency)
# Visit: https://environment.data.gov.uk/DefraDataDownload/
# Download a 1km² tile (25cm or 50cm resolution)
# Place in: data/lidar/sample.laz

# Download sample BIM model
wget https://raw.githubusercontent.com/buildingSMART/Sample-Test-Files/master/IFC%202x3/Duplex_A_20110907.ifc \\
  -O data/bim/sample.ifc

# Download UK basemap tiles (optional, ~2GB)
# Visit: https://openmaptiles.org/downloads/
# Download Great Britain MBTiles
# Place in: data/tiles/uk-map.mbtiles
```

### Step 4: Start Docker Compose

```bash
# Start all services in detached mode
docker compose up -d

# Watch logs (optional)
docker compose logs -f

# Or watch specific service
docker compose logs -f backend
```

**First Startup** (takes ~3-5 minutes):
- Downloads Docker images (~2GB)
- Creates volumes for data persistence
- Initializes PostgreSQL database
- Creates MinIO buckets
- Starts all services

### Step 5: Verify Services

```bash
# Check service status
docker compose ps

# Should see all services "Up" and "healthy"
NAME                  SERVICE    STATUS    PORTS
backend               backend    Up        0.0.0.0:8000->8000/tcp
frontend              frontend   Up        0.0.0.0:3000->3000/tcp
postgres              postgres   Up        0.0.0.0:5432->5432/tcp
minio                 minio      Up        0.0.0.0:9000-9001->9000-9001/tcp
redis                 redis      Up        0.0.0.0:6379->6379/tcp
celery_worker         celery     Up
flower                flower     Up        0.0.0.0:5555->5555/tcp
tileserver            tileserver Up        0.0.0.0:8080->8080/tcp
```

### Step 6: Initialize Database

```bash
# Run database migrations
docker compose exec backend alembic upgrade head

# Create initial admin user
docker compose exec backend python -m app.scripts.create_admin \\
  --email admin@example.com \\
  --password changeme

# Verify database connection
docker compose exec backend python -c "from app.core.database import test_connection; test_connection()"
```

### Step 7: Load Sample Data

```bash
# Load Twente GPR dataset (if available)
docker compose exec backend python -m app.scripts.load_twente_data

# Load Mojahid image dataset (if available)
docker compose exec backend python -m app.scripts.load_mojahid_data

# Or load minimal test data
docker compose exec backend python -m app.scripts.load_sample_data
```

---

## Docker Compose Services

### Service Overview

| Service | Port | Purpose | Access URL |
|---------|------|---------|------------|
| **frontend** | 3000 | React web app | http://localhost:3000 |
| **backend** | 8000 | FastAPI server | http://localhost:8000/docs |
| **postgres** | 5432 | Database | postgresql://localhost:5432/gpr_db |
| **minio** | 9000, 9001 | Object storage | http://localhost:9011 (console) |
| **redis** | 6379 | Cache + broker | redis://localhost:6380 |
| **celery_worker** | - | Background tasks | (monitored via Flower) |
| **flower** | 5555 | Celery monitoring | http://localhost:5555 |
| **tileserver** | 8080 | Map tiles | http://localhost:8080 |

### Service Dependencies

```
frontend → backend → postgres
                  → minio
                  → redis

celery_worker → redis
              → postgres
              → minio

tileserver → (independent, serves static tiles)
```

---

## Environment Configuration

### Full `.env` Template

```bash
# ============================================
# DATABASE CONFIGURATION
# ============================================
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_USER=gpr_user
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_DB=gpr_db
DATABASE_URL=postgresql+asyncpg://gpr_user:your_secure_password_here@postgres:5432/gpr_db

# ============================================
# BACKEND CONFIGURATION
# ============================================
SECRET_KEY=your_secret_key_min_32_characters_random
DEBUG=true
ALLOWED_HOSTS=localhost,127.0.0.1
CORS_ORIGINS=http://localhost:3000,http://localhost:8000

# ============================================
# MINIO (S3 STORAGE)
# ============================================
MINIO_ENDPOINT=minio:9000
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=your_minio_password_here
MINIO_SECURE=false  # true for HTTPS
MINIO_BUCKET_GPR=gpr-data
MINIO_BUCKET_BIM=bim-models
MINIO_BUCKET_LIDAR=lidar-scans
MINIO_BUCKET_DOCS=documents
MINIO_BUCKET_REPORTS=reports

# ============================================
# REDIS CONFIGURATION
# ============================================
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0
REDIS_URL=redis://redis:6379/0

# ============================================
# CELERY CONFIGURATION
# ============================================
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0

# ============================================
# FRONTEND CONFIGURATION
# ============================================
VITE_API_URL=http://localhost:8000
VITE_TILE_SERVER_URL=http://localhost:8080
VITE_MINIO_ENDPOINT=http://localhost:9010

# ============================================
# LLM CONFIGURATION (OPTIONAL - PHASE 2)
# ============================================
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o
ANTHROPIC_API_KEY=  # Leave empty for now

# ============================================
# LOGGING
# ============================================
LOG_LEVEL=INFO
LOG_FORMAT=json  # or: text
```

---

## Verifying the Setup

### 1. Check Backend API

```bash
# Health check
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","database":"connected","redis":"connected"}

# API documentation
open http://localhost:8000/docs  # Mac
xdg-open http://localhost:8000/docs  # Linux
start http://localhost:8000/docs  # Windows
```

### 2. Check Frontend

```bash
# Open frontend
open http://localhost:3000  # Mac
xdg-open http://localhost:3000  # Linux
start http://localhost:3000  # Windows

# Should see: Infrastructure Intelligence Platform login page
```

### 3. Check MinIO

```bash
# Open MinIO console
open http://localhost:9011

# Login with:
# Username: minioadmin (or your MINIO_ROOT_USER)
# Password: minioadmin (or your MINIO_ROOT_PASSWORD)

# Should see 5 buckets: gpr-data, bim-models, lidar-scans, documents, reports
```

### 4. Check Celery Workers

```bash
# Open Flower UI
open http://localhost:5555

# Should see:
# - 1+ active worker
# - Task queue status
# - Recent task executions
```

### 5. Check Database

```bash
# Connect to PostgreSQL
docker compose exec postgres psql -U gpr_user -d gpr_db

# Run test query
gpr_db=# \\dt  # List tables
gpr_db=# SELECT version();  # Check PostgreSQL version
gpr_db=# SELECT PostGIS_Version();  # Check PostGIS
gpr_db=# \\q  # Quit
```

---

## Sample Data Loading

### Option 1: Quick Test Data

```bash
# Load minimal test data (no external files needed)
docker compose exec backend python -m app.scripts.load_sample_data

# Creates:
# - 1 test project
# - 3 sample GPR surveys
# - 10 detected utilities
# - Sample environmental data
```

### Option 2: Real Twente GPR Dataset

```bash
# 1. Download Twente dataset (if not already downloaded)
# Place ZIP files in: /datasets/raw/twente_gpr/

# 2. Load via script
docker compose exec backend python -m app.scripts.load_twente_data

# 3. Verify
curl http://localhost:8000/api/v1/datasets/twente/status
```

### Option 3: Upload via API

```bash
# 1. Get JWT token
TOKEN=$(curl -X POST http://localhost:8000/api/v1/auth/login \\
  -H "Content-Type: application/json" \\
  -d '{"email":"admin@example.com","password":"changeme"}' \\
  | jq -r '.access_token')

# 2. Upload GPR file
curl -X POST http://localhost:8000/api/v1/gpr/upload \\
  -H "Authorization: Bearer $TOKEN" \\
  -F "file=@data/gpr/sample.sgy"

# 3. Upload BIM file (Phase 2)
curl -X POST http://localhost:8000/api/v1/bim/upload \\
  -H "Authorization: Bearer $TOKEN" \\
  -F "file=@data/bim/sample.ifc"
```

---

## Development Workflow

### Making Code Changes

**Backend (Python)**:
```bash
# Edit code in ./backend/app/
# FastAPI auto-reloads on file changes

# View logs
docker compose logs -f backend

# Run tests
docker compose exec backend pytest

# Format code
docker compose exec backend black app/
docker compose exec backend isort app/
```

**Frontend (React)**:
```bash
# Edit code in ./frontend/src/
# Vite auto-reloads on file changes

# View logs
docker compose logs -f frontend

# Run tests
docker compose exec frontend npm test

# Build production
docker compose exec frontend npm run build
```

### Database Migrations

```bash
# Create new migration
docker compose exec backend alembic revision --autogenerate -m "Add bim_models table"

# Apply migrations
docker compose exec backend alembic upgrade head

# Rollback migration
docker compose exec backend alembic downgrade -1

# View migration history
docker compose exec backend alembic history
```

### Restarting Services

```bash
# Restart specific service
docker compose restart backend

# Restart all services
docker compose restart

# Rebuild and restart (after dependency changes)
docker compose up -d --build

# Stop all services
docker compose down

# Stop and remove volumes (⚠️ deletes data)
docker compose down -v
```

---

## Troubleshooting

### Service Won't Start

```bash
# Check service logs
docker compose logs backend

# Check all logs
docker compose logs

# Restart service
docker compose restart backend

# Rebuild service
docker compose up -d --build backend
```

### Database Connection Issues

```bash
# Check PostgreSQL is running
docker compose ps postgres

# Test connection
docker compose exec postgres pg_isready -U gpr_user

# Check database logs
docker compose logs postgres

# Reset database (⚠️ deletes all data)
docker compose down -v
docker compose up -d postgres
docker compose exec backend alembic upgrade head
```

### MinIO Access Issues

```bash
# Check MinIO is running
docker compose ps minio

# Recreate buckets
docker compose exec minio_client sh -c "
  mc alias set myminio http://minio:9000 minioadmin minioadmin
  mc mb myminio/gpr-data || true
  mc mb myminio/bim-models || true
  mc mb myminio/lidar-scans || true
  mc mb myminio/documents || true
  mc mb myminio/reports || true
"
```

### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000  # Mac/Linux
netstat -ano | findstr :8000  # Windows

# Change port in docker-compose.yml
# Example: Change backend from 8000:8000 to 8001:8000
ports:
  - "8001:8000"  # host:container
```

### Out of Disk Space

```bash
# Clean up Docker
docker system prune -a --volumes

# Remove old images
docker image prune -a

# Check disk usage
docker system df
```

### Slow Performance

```bash
# Allocate more resources to Docker Desktop
# Docker Desktop → Preferences → Resources
# - CPUs: 4+
# - Memory: 8GB+
# - Swap: 2GB+

# Check container resource usage
docker stats
```

---

## Useful Commands

### Docker Compose Commands

```bash
# Start services
docker compose up -d

# Stop services
docker compose down

# View logs
docker compose logs -f

# Check status
docker compose ps

# Execute command in container
docker compose exec backend bash

# Restart service
docker compose restart backend

# Rebuild service
docker compose up -d --build backend

# Scale service (e.g., more workers)
docker compose up -d --scale celery_worker=3
```

### Database Commands

```bash
# Connect to database
docker compose exec postgres psql -U gpr_user -d gpr_db

# Backup database
docker compose exec postgres pg_dump -U gpr_user gpr_db > backup.sql

# Restore database
docker compose exec -T postgres psql -U gpr_user -d gpr_db < backup.sql

# Run SQL file
docker compose exec -T postgres psql -U gpr_user -d gpr_db < script.sql
```

### MinIO Commands

```bash
# List buckets
docker compose exec minio_client mc ls myminio

# Upload file
docker compose exec minio_client mc cp /data/file.pdf myminio/documents/

# Download file
docker compose exec minio_client mc cp myminio/documents/file.pdf /data/

# Remove file
docker compose exec minio_client mc rm myminio/documents/file.pdf
```

---

## Next Steps

After successful setup:

1. ✅ **Explore API Documentation**: http://localhost:8000/docs
2. ✅ **Load Sample Data**: Follow [Sample Data Loading](#sample-data-loading)
3. ✅ **Read Architecture Docs**: [/docs/architecture/ARCHITECTURE.md](../architecture/ARCHITECTURE.md)
4. ✅ **Check Development Guide**: [/docs/development/CONTRIBUTING.md](../development/CONTRIBUTING.md)
5. ✅ **Start Developing**: Make changes and see them live!

---

## Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [MinIO Documentation](https://min.io/docs/minio/linux/index.html)

---

Last Updated: 2025-11-24
