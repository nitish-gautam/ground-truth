# HS2 Infrastructure Intelligence Platform - Quick Start

## One-Command Setup

### Start Everything
```bash
./start.sh
```

This will:
- ✅ Build all Docker containers
- ✅ Start all services (frontend, backend, postgres, redis, minio, etc.)
- ✅ Show you the access URLs

### Stop Everything
```bash
./stop.sh              # Stop containers, keep data
./stop.sh --clean      # Stop containers and remove all data
```

## Alternative: Full Demo with Data Loading

For a complete demo with placeholder data:
```bash
./scripts/demo.sh
```

This includes:
- Everything from `start.sh`
- Database initialization
- 900+ placeholder records (50 assets, 586 deliverables, 214 certificates, 50 costs)
- Automated browser launch

## Access URLs

Once started, access:

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend** | http://localhost:3003 | React dashboard |
| **Backend API** | http://localhost:8002 | FastAPI endpoints |
| **API Docs** | http://localhost:8002/docs | Interactive Swagger UI |
| **Health Check** | http://localhost:8002/health | System status |
| **MinIO Console** | http://localhost:9001 | Object storage (admin/minioadmin) |
| **TileServer** | http://localhost:8080 | Map tiles (optional) |

**Note**: Celery worker and Flower are configured but not required for the HS2 demo. They can be started when background task processing is needed.

## Useful Commands

### View Logs
```bash
docker compose logs -f              # All services
docker compose logs -f frontend     # Frontend only
docker compose logs -f backend      # Backend only
```

### Check Status
```bash
docker compose ps                   # Running containers
docker ps                          # Same as above
```

### Restart a Service
```bash
docker compose restart frontend     # Restart frontend
docker compose restart backend      # Restart backend
```

### Rebuild After Code Changes
```bash
docker compose build frontend       # Rebuild frontend
docker compose build backend        # Rebuild backend
docker compose up -d frontend       # Start with new build
```

### Clean Restart
```bash
./stop.sh --clean                  # Remove everything
./start.sh                         # Start fresh
```

## Project Structure

```
ground-truth/
├── start.sh                    # ⭐ Simple startup script
├── stop.sh                     # ⭐ Simple stop script
├── scripts/demo.sh             # ⭐ Full demo with data
├── docker-compose.yml          # Docker services configuration
├── frontend/                   # React + TypeScript + Vite
├── backend/                    # FastAPI + Python 3.11
├── docs/                       # Documentation
└── datasets/                   # Sample data
```

## Troubleshooting

### Port Already in Use
```bash
# Find what's using port 3003
lsof -i :3003
# Kill it
kill -9 <PID>
```

### Frontend Shows Import Errors
```bash
# Rebuild frontend with fresh dependencies
docker compose stop frontend
docker compose rm -f frontend
cd frontend && npm install && cd ..
docker compose build frontend
docker compose up -d frontend
```

### Database Connection Error
```bash
# Check if postgres is healthy
docker compose ps postgres
# Restart postgres
docker compose restart postgres
```

### Start Fresh
```bash
./stop.sh --clean              # Remove all containers and data
./start.sh                     # Start from scratch
```

## Next Steps

1. **Explore the Dashboard**: http://localhost:3003
2. **Check API Docs**: http://localhost:8002/docs
3. **Load Demo Data**: Run `./scripts/demo.sh` if you haven't already
4. **Review Architecture**: See [docs/architecture/](docs/architecture/)
5. **Read TAEM Rules**: See [docs/TAEM_RULES_CATALOG.md](docs/TAEM_RULES_CATALOG.md)

## For HS2 Stakeholders

For a full demonstration:
1. Run `./scripts/demo.sh`
2. Follow the demo script: [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md)
3. Review the presentation: [docs/HS2_STAKEHOLDER_PRESENTATION.md](docs/HS2_STAKEHOLDER_PRESENTATION.md)

## Documentation

- **Quick Start**: [QUICKSTART.md](QUICKSTART.md) (this file)
- **HS2 Demo**: [HS2_QUICK_START.md](HS2_QUICK_START.md)
- **Architecture**: [docs/architecture/](docs/architecture/)
- **API Reference**: http://localhost:8002/docs (when running)
- **TAEM Rules**: [docs/TAEM_RULES_CATALOG.md](docs/TAEM_RULES_CATALOG.md)
- **Data Dictionary**: [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md)
- **Deployment Guide**: [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)
