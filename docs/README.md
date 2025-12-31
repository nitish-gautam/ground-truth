# 📚 Documentation Navigator

**Infrastructure Intelligence Platform - Complete Documentation Index**

*Your central hub for navigating all project documentation. Everything organized by category for easy access.*

Last Updated: December 31, 2025

---

## 🚀 Quick Start (New Users Start Here!)

### First Time Setup
1. 📖 **[Getting Started Guide](guides/GETTING_STARTED.md)** - Complete setup instructions (5-10 minutes)
2. 🔑 **[Credentials Reference](CREDENTIALS.md)** - Database passwords, API keys, service credentials
3. 🎮 **[HS2 Demo Quick Start](guides/HS2_DEMO_QUICKSTART.md)** - Run HS2 demo in 2 minutes
4. 📋 **[CHANGELOG](../CHANGELOG.md)** - Latest fixes and improvements (Dec 2025)

### Quick Access
```bash
# URLs (Updated December 2025)
Frontend:      http://localhost:3003/hs2
API Docs:      http://localhost:8007/docs
MinIO Console: http://localhost:9011
Neo4j Browser: http://localhost:7475

# Start Everything
docker compose up -d
cd frontend && npm install && npm run dev
```

---

## 📂 Documentation Categories

### 🏢 Business & Commercial
*For investors, stakeholders, procurement teams, and executives*

| Document | Description | Audience | Pages |
|----------|-------------|----------|-------|
| **[HS2 Business Strategy](HS2_BUSINESS_STRATEGY.md)** ⭐ **CONSOLIDATED** | Response to HS2 Accelerator feedback (11.0/15 → 14.1/15) | HS2 Ltd, Tier 1 contractors, Investors | 1065 lines |
| **[Executive Summary](business/EXECUTIVE_SUMMARY.md)** | One-page to full commercial overview | Investors, C-Suite | 22 |
| **[YC Pitch Deck](business/HS2_STAKEHOLDER_PRESENTATION.md)** | Y Combinator pitch (15 slides + appendix) | Investors, Accelerators | 41 |
| **[HS2 Deck PDF](business/HS2%20Deck.pdf)** | Visual presentation deck | Stakeholders | PDF |

**Key Topics**: Programme scale economics (£16M → £2.7M), scalability through replication, Tier 1 augmentation strategy, distributed local deployment, Market size (£2.5B TAM), ROI (7-43x), customer savings (£16M-£73M/year), competitive landscape, financial projections

**Business Strategy Note**: This comprehensive document consolidates PITCH_DECK_UPDATE, PROGRAMME_SCALE_ECONOMICS, TIER1_INTEGRATION_STRATEGY, HS2_ACCELERATOR_RESPONSE, and RESPONSE_TO_HS2_FEEDBACK into one complete strategy guide.

---

### 🔧 Technical Documentation
*For developers, architects, and technical due diligence*

| Document | Description | Use Case | Pages |
|----------|-------------|----------|-------|
| **[ML Technical Guide](ML_TECHNICAL_GUIDE.md)** ⭐ **CONSOLIDATED** | Complete ML implementation - training, deployment, troubleshooting | ML development, all ML topics | 1065 lines |
| **[HS2 Implementation Guide](technical/HS2_PROGRESS_ASSURANCE_IMPLEMENTATION_GUIDE.md)** | Complete HS2 system implementation + workflows | Build/deploy HS2 module | 51 |
| **[HS2 Technical FAQ](technical/HS2_TECHNICAL_FAQ.md)** | 40+ Q&A on hyperspectral, AI/ML, security | Due diligence, technical validation | 39 |
| **[API Reference](technical/API_IMPLEMENTATION_COMPLETE.md)** | API endpoint specifications | Backend development | 10 |
| **[Data Sources](technical/DATA_SOURCES.md)** | Available datasets and references | Data engineering | 5 |

**Key Topics**: UMKC hyperspectral ML (100% accuracy), 292 spectral features, Random Forest models, Docker deployment, Hyperspectral imaging (Specim IQ, 204 bands, R²=0.89), AI/ML architecture, Azure deployment, security (GDPR, ISO 27001)

**ML Guide Note**: This single comprehensive document consolidates ML_MODEL_TRAINING_PLAN, ML_IMPLEMENTATION_COMPLETE, ML_INTEGRATION_GUIDE, and ML_MODELS_READY_FOR_PRODUCTION into one complete reference.

---

### 🏗️ Architecture & Design
*For system architects and technical leads*

| Document | Description | Use Case | Pages |
|----------|-------------|----------|-------|
| **[System Architecture](architecture/ARCHITECTURE.md)** | Complete architecture + technology decisions | Understand system design | 73 |
| **[Architecture Diagrams](architecture/DIAGRAMS.md)** | 17 Mermaid diagrams (data flow, system layers) | Visual architecture reference | 45 |
| **[Implementation Milestones](architecture/IMPLEMENTATION_MILESTONES.md)** | Development roadmap and phase plan | Project planning | 8 |

**Key Topics**: 4-layer architecture, 3-phase data pipeline, technology stack (FastAPI, React, PostgreSQL, MinIO, Azure), 16 technology justifications

---

### 📊 Database & Data
*For data engineers and database administrators*

| Document | Description | Use Case | Pages |
|----------|-------------|----------|-------|
| **[Data Dictionary](database/DATA_DICTIONARY.md)** | Complete schema reference (57 tables) | Database development | 24 |
| **[Data Guide](database/DATA_GUIDE.md)** | Data sources and organization | Data management | 18 |
| **[Demo Data Strategy](database/DEMO_DATA_STRATEGY.md)** | Demo data approach and samples | Testing and demos | 24 |

**Key Topics**: PostgreSQL + PostGIS + PGVector, 17 deployed tables, 40 planned tables, GPR/BIM/LiDAR/Hyperspectral data models

---

### 📖 User Guides
*For end users, trainers, and field teams*

| Document | Description | Use Case | Pages |
|----------|-------------|----------|-------|
| **[Getting Started](guides/GETTING_STARTED.md)** | Setup and first-time configuration | New developer onboarding | 13 |
| **[Demo Script](guides/DEMO_SCRIPT.md)** | Demonstration walkthrough | Live demos and presentations | 22 |
| **[HS2 Demo Quick Start](guides/HS2_DEMO_QUICKSTART.md)** | 2-minute HS2 demo | Quick HS2 showcase | 11 |
| **[Deployment Guide](guides/DEPLOYMENT_GUIDE.md)** | Production deployment steps | DevOps and deployment | 31 |

**Key Topics**: Docker setup, service configuration, demo scenarios, production deployment (Azure)

---

### 📦 Product & Implementation
*For product managers and implementation teams*

| Document | Description | Use Case | Pages |
|----------|-------------|----------|-------|
| **[Implementation Status](product/IMPLEMENTATION_COMPLETE.md)** | Current implementation status | Track progress | 11 |
| **[Implementation Summary](product/IMPLEMENTATION_COMPLETE_SUMMARY.md)** | Concise status summary | Quick status check | 14 |
| **[Project Status](product/PROJECT_STATUS.md)** | Overall project status and roadmap | Project management | 31 |
| **[Design & UI Guidelines](product/DESIGN_AND_UI.md)** | UI/UX patterns and standards | Frontend development | 7 |
| **[TAEM Rules Catalog](product/TAEM_RULES_CATALOG.md)** | Technical assurance rules | HS2 compliance | 21 |

**Key Topics**: Feature status, design system, TAEM compliance, roadmap phases

---

## 🎯 Documentation by Use Case

### "I want to understand the business case"
1. Start: [Executive Summary](business/EXECUTIVE_SUMMARY.md) (one-page overview)
2. Deep dive: [YC Pitch Deck](business/HS2_STAKEHOLDER_PRESENTATION.md) (full pitch)
3. Workflows: [HS2 Implementation Guide](technical/HS2_PROGRESS_ASSURANCE_IMPLEMENTATION_GUIDE.md) (business workflows section)

### "I want to validate the technology"
1. Start: [HS2 Technical FAQ](technical/HS2_TECHNICAL_FAQ.md) (40+ Q&A)
2. Deep dive: [System Architecture](architecture/ARCHITECTURE.md) (technology decisions)
3. Visual: [Architecture Diagrams](architecture/DIAGRAMS.md) (17 diagrams)

### "I want to build/deploy the platform"
1. Start: [Getting Started](guides/GETTING_STARTED.md) (setup)
2. API Reference: [API Implementation](technical/API_IMPLEMENTATION_COMPLETE.md)
3. Database: [Data Dictionary](database/DATA_DICTIONARY.md) (schema)
4. Deploy: [Deployment Guide](guides/DEPLOYMENT_GUIDE.md) (production)

### "I want to demo the platform"
1. Quick: [HS2 Demo Quick Start](guides/HS2_DEMO_QUICKSTART.md) (2 minutes)
2. Full: [Demo Script](guides/DEMO_SCRIPT.md) (complete walkthrough)
3. Data: [Demo Data Strategy](database/DEMO_DATA_STRATEGY.md) (sample data)

### "I want to understand hyperspectral technology"
1. Overview: [HS2 Implementation Guide](technical/HS2_PROGRESS_ASSURANCE_IMPLEMENTATION_GUIDE.md) (hyperspectral section)
2. Validation: [HS2 Technical FAQ](technical/HS2_TECHNICAL_FAQ.md) (R²=0.89 lab, R²=0.82 field)
3. Architecture: [System Architecture](architecture/ARCHITECTURE.md) (Layer 2 - AI/ML processing)

---

## 📈 Key Metrics Quick Reference

### Business Metrics
- **Market Size**: £2.5B TAM (UK construction monitoring)
- **Customer Savings**: £16M-£73M/year (100-site HS2 deployment)
- **Customer ROI**: 7-43x return on investment
- **Time Reduction**: 95% (8 hours → 10 minutes for progress reports)

### Technical Metrics
- **Hyperspectral**: Specim IQ, 204 spectral bands, 400-1000nm, ~3nm resolution
- **Lab Accuracy**: R²=0.89, MAE=3.2 MPa (500 samples, Dec 2024)
- **Field Accuracy**: R²=0.82, MAE=4.2 MPa (150 samples, A14 bridge, Nov 2024)
- **LiDAR**: Leica RTC360, 2M points/sec, ±1mm accuracy

### Implementation Metrics
- **Database**: 57 tables (17 deployed, 40 planned)
- **API Endpoints**: 93+ (30 deployed, 63 planned)
- **Architecture**: 4 layers, 3-phase data pipeline
- **Platform Cost**: £720K-£1.44M/year (100 sites)

---

## 🔗 Cross-Document Navigation

### Technology Stack Flow
```
[Architecture](architecture/ARCHITECTURE.md)
  ↓ Technology Decisions (16 justified)
  ↓
[API Reference](technical/API_IMPLEMENTATION_COMPLETE.md)
  ↓ Endpoints & Schemas
  ↓
[Data Dictionary](database/DATA_DICTIONARY.md)
  ↓ Database Schema
  ↓
[Implementation Status](product/IMPLEMENTATION_COMPLETE.md)
```

### Business to Technical Flow
```
[Executive Summary](business/EXECUTIVE_SUMMARY.md)
  ↓ Problem & Solution
  ↓
[HS2 Implementation Guide](technical/HS2_PROGRESS_ASSURANCE_IMPLEMENTATION_GUIDE.md)
  ↓ Workflows & ROI
  ↓
[HS2 Technical FAQ](technical/HS2_TECHNICAL_FAQ.md)
  ↓ Validation & Due Diligence
  ↓
[Architecture](architecture/ARCHITECTURE.md)
```

### Setup to Deployment Flow
```
[Getting Started](guides/GETTING_STARTED.md)
  ↓ Local Setup
  ↓
[Demo Script](guides/DEMO_SCRIPT.md)
  ↓ Test Functionality
  ↓
[Deployment Guide](guides/DEPLOYMENT_GUIDE.md)
  ↓ Production Deployment
  ↓
[Project Status](product/PROJECT_STATUS.md)
```

---

## 🗂️ Folder Structure

```
docs/
├── README.md                          # 📍 You are here - Navigation hub
├── CREDENTIALS.md                     # 🔑 Quick credentials reference
│
├── architecture/                      # 🏗️ System architecture
│   ├── ARCHITECTURE.md               # Complete system design + tech decisions
│   ├── DIAGRAMS.md                   # 17 Mermaid diagrams
│   └── IMPLEMENTATION_MILESTONES.md  # Development roadmap
│
├── business/                          # 💼 Business & commercial
│   ├── EXECUTIVE_SUMMARY.md          # Investor/executive summary
│   ├── HS2_STAKEHOLDER_PRESENTATION.md  # YC pitch deck
│   └── HS2 Deck.pdf                  # Visual presentation
│
├── database/                          # 📊 Database & data
│   ├── DATA_DICTIONARY.md            # Schema reference (57 tables)
│   ├── DATA_GUIDE.md                 # Data sources and structure
│   └── DEMO_DATA_STRATEGY.md         # Demo data approach
│
├── guides/                            # 📖 User guides
│   ├── GETTING_STARTED.md            # Setup guide
│   ├── DEMO_SCRIPT.md                # Demo walkthrough
│   ├── HS2_DEMO_QUICKSTART.md        # Quick HS2 demo
│   └── DEPLOYMENT_GUIDE.md           # Production deployment
│
├── product/                           # 📦 Product & implementation
│   ├── IMPLEMENTATION_COMPLETE.md     # Implementation status
│   ├── IMPLEMENTATION_COMPLETE_SUMMARY.md  # Status summary
│   ├── PROJECT_STATUS.md             # Project roadmap
│   ├── DESIGN_AND_UI.md              # UI/UX guidelines
│   └── TAEM_RULES_CATALOG.md         # TAEM compliance rules
│
└── technical/                         # 🔧 Technical documentation
    ├── HS2_PROGRESS_ASSURANCE_IMPLEMENTATION_GUIDE.md  # HS2 implementation
    ├── HS2_TECHNICAL_FAQ.md          # Technical Q&A (40+ questions)
    ├── API_IMPLEMENTATION_COMPLETE.md  # API reference
    └── DATA_SOURCES.md               # Dataset references
```

---

## 🎓 Learning Paths

### Path 1: Investor/Executive (30 minutes)
1. [Executive Summary](business/EXECUTIVE_SUMMARY.md) (10 min) - Problem, solution, market
2. [YC Pitch Deck](business/HS2_STAKEHOLDER_PRESENTATION.md) (15 min) - Financials, traction
3. [HS2 Technical FAQ](technical/HS2_TECHNICAL_FAQ.md) (5 min) - Skim validation section

### Path 2: Technical Due Diligence (2 hours)
1. [System Architecture](architecture/ARCHITECTURE.md) (30 min) - Complete architecture
2. [HS2 Technical FAQ](technical/HS2_TECHNICAL_FAQ.md) (45 min) - All Q&A
3. [Architecture Diagrams](architecture/DIAGRAMS.md) (30 min) - Visual architecture
4. [Data Dictionary](database/DATA_DICTIONARY.md) (15 min) - Database schema

### Path 3: Developer Onboarding (3 hours)
1. [Getting Started](guides/GETTING_STARTED.md) (30 min) - Setup environment
2. [HS2 Demo Quick Start](guides/HS2_DEMO_QUICKSTART.md) (10 min) - Run demo
3. [API Reference](technical/API_IMPLEMENTATION_COMPLETE.md) (30 min) - API endpoints
4. [Data Dictionary](database/DATA_DICTIONARY.md) (30 min) - Database schema
5. [Design & UI](product/DESIGN_AND_UI.md) (20 min) - Frontend patterns
6. [Implementation Status](product/IMPLEMENTATION_COMPLETE.md) (20 min) - Current state
7. Hands-on: Build a feature (1+ hour)

### Path 4: Product Manager (1 hour)
1. [Executive Summary](business/EXECUTIVE_SUMMARY.md) (10 min) - Business overview
2. [Implementation Status](product/IMPLEMENTATION_COMPLETE.md) (20 min) - What's built
3. [Project Status](product/PROJECT_STATUS.md) (20 min) - Roadmap
4. [Demo Script](guides/DEMO_SCRIPT.md) (10 min) - Demo scenarios

---

## 🔍 Search Tips

### Find Information By Keyword
- **Hyperspectral**: [HS2 Technical FAQ](technical/HS2_TECHNICAL_FAQ.md), [Architecture](architecture/ARCHITECTURE.md)
- **ROI/Financials**: [Executive Summary](business/EXECUTIVE_SUMMARY.md), [YC Pitch Deck](business/HS2_STAKEHOLDER_PRESENTATION.md)
- **API Endpoints**: [API Reference](technical/API_IMPLEMENTATION_COMPLETE.md)
- **Database Schema**: [Data Dictionary](database/DATA_DICTIONARY.md)
- **Setup/Installation**: [Getting Started](guides/GETTING_STARTED.md)
- **HS2 Specific**: All `HS2_*` documents
- **Technology Choices**: [Architecture](architecture/ARCHITECTURE.md) - Technology Decision Matrix section

### Find Information By Stakeholder
- **Investors**: [business/](business/)
- **Developers**: [technical/](technical/), [guides/](guides/)
- **Architects**: [architecture/](architecture/)
- **Product Managers**: [product/](product/)
- **Data Engineers**: [database/](database/)

---

## 📝 Document Status

| Category | Total Docs | Status | Last Updated |
|----------|-----------|--------|--------------|
| **Business** | 3 | ✅ Complete | Dec 10, 2025 |
| **Technical** | 4 | ✅ Complete | Dec 10, 2025 |
| **Architecture** | 3 | ✅ Complete | Dec 10, 2025 |
| **Database** | 3 | ✅ Complete | Dec 9, 2025 |
| **Guides** | 4 | ✅ Complete | Dec 9, 2025 |
| **Product** | 5 | ✅ Complete | Dec 10, 2025 |
| **Total** | **22 docs** | **Organized** | **Current** |

---

## 🆘 Getting Help

### Common Questions
- **Q: Where do I start?**
  A: New user? → [Getting Started](guides/GETTING_STARTED.md). Investor? → [Executive Summary](business/EXECUTIVE_SUMMARY.md)

- **Q: How do I run a demo?**
  A: [HS2 Demo Quick Start](guides/HS2_DEMO_QUICKSTART.md) (2 minutes)

- **Q: What's the hyperspectral accuracy?**
  A: R²=0.89 (lab), R²=0.82 (field). See [HS2 Technical FAQ](technical/HS2_TECHNICAL_FAQ.md)

- **Q: What's the ROI?**
  A: 7-43x customer ROI, £16M-£73M/year savings. See [Executive Summary](business/EXECUTIVE_SUMMARY.md)

- **Q: What's implemented?**
  A: [Implementation Status](product/IMPLEMENTATION_COMPLETE.md) - 30+ API endpoints, 17 database tables, full HS2 frontend

### Still Can't Find It?
1. Check the [Project Status](product/PROJECT_STATUS.md) for roadmap
2. Search within specific category folders
3. Use Ctrl+F (Cmd+F) within documents - all docs are markdown with searchable text

---

## 🎯 Next Steps

### For New Users
1. ✅ Read [Getting Started](guides/GETTING_STARTED.md)
2. ✅ Set up environment (Docker, Node.js)
3. ✅ Run [HS2 Demo](guides/HS2_DEMO_QUICKSTART.md)
4. ✅ Explore [Implementation Status](product/IMPLEMENTATION_COMPLETE.md)

### For Investors
1. ✅ Read [Executive Summary](business/EXECUTIVE_SUMMARY.md)
2. ✅ Review [YC Pitch Deck](business/HS2_STAKEHOLDER_PRESENTATION.md)
3. ✅ Validate with [HS2 Technical FAQ](technical/HS2_TECHNICAL_FAQ.md)

### For Developers
1. ✅ Set up environment: [Getting Started](guides/GETTING_STARTED.md)
2. ✅ Review architecture: [Architecture](architecture/ARCHITECTURE.md)
3. ✅ Explore API: [API Reference](technical/API_IMPLEMENTATION_COMPLETE.md)
4. ✅ Check database: [Data Dictionary](database/DATA_DICTIONARY.md)

---

**Document Version**: 2.0 (Reorganized and Enhanced)
**Last Updated**: December 10, 2025
**Maintained By**: Infrastructure Intelligence Platform Team
**Total Documentation**: 22 organized documents across 6 categories

---

**Quick Links**:
- 🏠 [Project Root](../)
- 📖 [Main README](../README.md)
- 🔑 [Credentials](CREDENTIALS.md)
- 🎯 [Getting Started](guides/GETTING_STARTED.md)
