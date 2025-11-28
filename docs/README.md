# Documentation Index

**Infrastructure Intelligence Platform (HS2)**
**Last Updated:** November 27, 2025

---

## 📚 Core Documentation (Start Here)

### Essential Guides

| Document | Purpose | When to Read |
|----------|---------|--------------|
| [**GETTING_STARTED.md**](GETTING_STARTED.md) | Complete setup guide | **Start here** - First-time setup |
| [**CREDENTIALS.md**](CREDENTIALS.md) | Quick credentials reference | Need database/MinIO passwords |
| [**IMPLEMENTATION_COMPLETE.md**](IMPLEMENTATION_COMPLETE.md) | Implementation status | Check what's built & working |
| [**DESIGN_AND_UI.md**](DESIGN_AND_UI.md) | UI/UX guidelines & patterns | Building/styling components |

### Reference Documentation

| Document | Purpose |
|----------|---------|
| [**DATA_GUIDE.md**](DATA_GUIDE.md) | Data sources & structure |
| [**DATA_DICTIONARY.md**](DATA_DICTIONARY.md) | Complete schema reference |
| [**DEPLOYMENT_GUIDE.md**](DEPLOYMENT_GUIDE.md) | Production deployment |
| [**TAEM_RULES_CATALOG.md**](TAEM_RULES_CATALOG.md) | TAEM assessment rules |
| [**PROJECT_STATUS.md**](PROJECT_STATUS.md) | Status & roadmap |
| [**DEMO_SCRIPT.md**](DEMO_SCRIPT.md) | Demonstration guide |

---

## 🚀 Quick Start (5 Minutes)

```bash
# 1. Start services
docker compose up -d

# 2. Install frontend dependencies
cd frontend && npm install

# 3. Start development server
npm run dev

# 4. Access application
open http://localhost:3003/hs2
```

**See [GETTING_STARTED.md](GETTING_STARTED.md) for detailed setup instructions.**

---

## 🔑 Quick Access

### URLs
```
Frontend:      http://localhost:3003/hs2
API Docs:      http://localhost:8002/docs
MinIO Console: http://localhost:9001
```

### Credentials
**See [CREDENTIALS.md](CREDENTIALS.md) for all passwords**

```
PostgreSQL:  gpr_user / Lb1RcTOayzhQlwhU2E9dbA (port 5433)
MinIO:       minioadmin / mD9E3_kgZJAPRjNvBWOxGQ
Redis:       No password (port 6379)
```

---

## 📊 System Status

**Current Version:** 1.0.0 (Production Ready)
**Design Compliance:** 85%
**Test Coverage:** 70%

### What's Working ✅
- ✅ Backend API (12 endpoints, 500+ assets)
- ✅ Frontend Dashboard (4 tabs with real data)
- ✅ GIS Integration (8 HS2 shapefiles)
- ✅ BIM Viewer (45 IFC models)
- ✅ Progress Tracking (EVM metrics)
- ✅ Point Cloud Upload (demo mode)

### Known Limitations ⚠️
- ⚠️ Multilingual support not implemented (English only)
- ⚠️ Mobile optimization needed
- ⚠️ Full WCAG AA audit pending

**See [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md) for complete details.**

---

## 🗂️ Documentation Organization

### Before Consolidation (17 scattered docs)
- HS2_QUICK_START, QUICK_START_HS2_FRONTEND, POINT_CLOUD_QUICK_START
- DESIGN_REVIEW, DESIGN_FIXES, UI_UX_IMPROVEMENTS
- HS2_COMPLETION, HS2_FRONTEND_COMPLETE, HS2_IMPLEMENTATION_STATUS
- FILE_UPLOAD_GUIDE, MINIO_GUIDE, PROGRESS_VERIFICATION
- ... and 5 more overlapping files

### After Consolidation (4 core docs + existing guides)
1. **IMPLEMENTATION_COMPLETE.md** - All implementation/status docs merged
2. **DESIGN_AND_UI.md** - All design/UI/UX docs merged
3. **CREDENTIALS.md** - Simplified quick reference
4. Existing guides preserved (GETTING_STARTED, DATA_GUIDE, etc.)

**Result:** Simpler, clearer, no duplication

---

## 🛠️ Development Commands

### Docker Services
```bash
docker compose ps                    # Check status
docker compose logs -f backend      # View logs
docker compose restart frontend     # Restart service
docker compose down                 # Stop all services
```

### Database
```bash
# Connect to PostgreSQL
docker exec -it infrastructure-postgres psql -U postgres -d infrastructure_db

# Run migrations
docker exec infrastructure-backend alembic upgrade head
```

### Frontend
```bash
cd frontend
npm install                # Install dependencies
npm run dev                # Start dev server
npm run build              # Production build
npm test                   # Run tests
```

---

## 📖 Documentation Principles

1. **One Source of Truth** - No duplicate information
2. **Always Current** - Update docs with code changes
3. **Searchable** - Clear headings and keywords
4. **Actionable** - Include examples and commands
5. **Concise** - Merged overlapping content

---

## 🤝 Contributing

### Adding Documentation
1. Use Markdown format (.md)
2. Follow existing structure
3. Update this README
4. Keep synced with code

### Documentation Review
- Review frequency: After major feature changes
- Maintainer: Frontend/Backend Teams
- Approval: Tech Lead

---

**Need Help?** Start with [GETTING_STARTED.md](GETTING_STARTED.md)
**Have Questions?** Check [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md) for system details
