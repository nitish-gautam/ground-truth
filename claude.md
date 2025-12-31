# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains documentation and architecture specifications for an **Underground Utility Detection Platform** - an AI-native solution for PAS 128 compliance and utility strike prevention. The platform aims to reduce utility strikes by 60% and generate compliant reports in 10 minutes vs 8 hours manual work.

### Technology Stack (As Specified)
- **Backend**: FastAPI (Python 3.11+) - Async, ML-friendly architecture
- **Frontend**: React 18 + TypeScript - PWA capable for mobile field use
- **Vector Database**: Pinecone - <100ms latency, managed service
- **LLM**: GPT-4o - For complex regulatory reasoning
- **Embeddings**: text-embedding-3-small - Cost/performance optimized
- **Cloud Platform**: AWS - GPU availability, enterprise ready
- **Cache**: Redis - Query and embedding caching
- **Database**: PostgreSQL - Structured data storage

## Architecture Overview

The platform uses a microservices architecture with these core components:

### RAG Pipeline Architecture
- **Query Processor**: Intent understanding and query expansion
- **Embedding Service**: Multi-modal document processing (GPR, PDF, CAD)
- **Vector Search**: Hybrid dense/sparse retrieval from Pinecone
- **Context Builder**: PAS 128 compliance-aware context assembly
- **LLM Router**: Multi-model selection based on task complexity

### Data Processing Pipeline
```
GPR Files → Signal Processing → ML Interpretation →
Utility Records → OCR + Parsing → Spatial Correlation →
CAD Drawings → Layer Extraction → Conflict Detection →
→ Risk Scoring → Report Generation → PAS 128 Compliance Validation
```

### Core Data Types
1. **Regulatory Documents**: PAS 128:2022, CDM 2015 (semantic chunking)
2. **GPR Data**: SEG-Y, GSSI DZT formats (10,000+ training files needed)
3. **Utility Records**: PDF, CAD, GIS (OCR + georeferencing required)
4. **Incident Database**: HSE RIDDOR reports (15,000+ incidents for training)

## Development Guidelines

### PAS 128 Compliance Requirements
- **Quality Levels**: Implement QL-A through QL-D classification algorithms
- **Accuracy Standards**: >95% accuracy vs manual interpretation required
- **Documentation**: Full audit trail for 7 years (CDM requirement)
- **Security**: Project-level data isolation, encryption at rest/transit

### Data Security & Privacy
- **GDPR Compliance**: Automated PII detection and redaction
- **Project Isolation**: Strict data separation between clients
- **Audit Logging**: Immutable logs for all operations
- **No Hardcoded Credentials**: Use AWS Parameter Store/Secrets Manager

### Performance Targets
- **Report Generation**: <10 minutes end-to-end
- **API Latency**: <200ms P95 for search operations
- **Vector Search**: <100ms query response from Pinecone
- **Uptime**: 99.9% availability target

### ML/AI Guidelines
- **No Hallucinations**: Implement citation tracking for all LLM outputs
- **Multi-Model Fallback**: GPT-4o primary, Claude-3/Llama fallbacks
- **Embedding Consistency**: Use text-embedding-3-small across all text
- **Risk Scoring**: 90% AUC-ROC target for strike prediction model

## Common Development Commands

### Python Environment Setup
```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (when available)
pip install -r requirements.txt

# Run development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Testing Commands
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_rag_pipeline.py -v
```

### Database Operations
```bash
# Run migrations (when implemented)
alembic upgrade head

# Create new migration
alembic revision --autogenerate -m "description"
```

### Code Quality
```bash
# Format code
black .
isort .

# Type checking
mypy app/

# Linting
flake8 app/
```

## File Organization Patterns

### Expected Project Structure
```
app/
├── main.py                     # FastAPI application entry
├── core/
│   ├── config.py              # Environment configuration
│   ├── security.py            # Authentication/authorization
│   └── dependencies.py        # Dependency injection
├── api/
│   ├── v1/                    # API versioning
│   │   ├── endpoints/         # Route handlers
│   │   └── dependencies.py    # Route-specific dependencies
├── services/
│   ├── rag/                   # RAG pipeline services
│   ├── gpr/                   # GPR data processing
│   ├── compliance/            # PAS 128 validation
│   └── reporting/             # Report generation
├── models/
│   ├── database.py            # SQLAlchemy models
│   └── schemas.py             # Pydantic models
├── utils/
│   ├── embeddings.py          # Embedding utilities
│   ├── vector_store.py        # Pinecone operations
│   └── file_processing.py     # Document processing
└── tests/
    ├── unit/                  # Unit tests
    ├── integration/           # Integration tests
    └── fixtures/              # Test data
```

## Domain-Specific Guidelines

### GPR Data Processing
- **File Formats**: Support SEG-Y, GSSI DZT, Sensors & Software DT1
- **Signal Processing**: Implement time-zero correction, background removal
- **ML Interpretation**: Train on 10,000+ labeled radargrams
- **Validation**: Cross-validate with trench verification data

### Regulatory Compliance (PAS 128)
- **Quality Classification**: Automate QL-A/B/C/D assignment
- **Report Sections**: Site description, methodology, results, limitations
- **Drawing Standards**: CAD layer conventions for utility types
- **Certification**: Digital signatures for qualified surveyors

### Risk Assessment Engine
- **Strike Probability**: Historical incident correlation
- **Spatial Analysis**: Buffer zones around known utilities
- **Confidence Scoring**: Uncertainty quantification for predictions
- **Real-time Updates**: Dynamic risk recalculation

## Integration Requirements

### External Services
- **Pinecone**: Vector database for RAG retrieval
- **OpenAI API**: GPT-4o for report generation
- **AWS S3**: File storage for GPR data, PDFs, CAD files
- **PostgreSQL**: Structured data (projects, users, incidents)

### Data Sources (Future Integration)
- **Utility Companies**: API access to utility records
- **Ordnance Survey**: Mapping data integration
- **HSE Database**: Real-time incident data feeds
- **Equipment APIs**: Direct GPR/EMI device integration

## Security Requirements

### Data Classification
- **Public**: Marketing materials, general documentation
- **Internal**: Business processes, non-client data
- **Confidential**: Client project data, utility locations
- **Restricted**: Strike incident data, regulatory compliance records

### Access Controls
- **Authentication**: Enterprise SSO (SAML/OAuth2)
- **Authorization**: Role-based access (Admin, Surveyor, Viewer)
- **Data Isolation**: Client-specific data segregation
- **Audit Trail**: Full activity logging for compliance

## Deployment Guidelines

### Environment Configuration
```bash
# Required environment variables
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=us-west1-gcp
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://localhost:6380
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

### Health Checks
- **API Health**: `/health` endpoint with dependency checks
- **Database**: Connection pool status
- **Vector Store**: Pinecone connectivity
- **LLM Services**: OpenAI API availability

# CLAUDE CODE GENERATION RULES - MANDATORY COMPLIANCE

## CRITICAL SECURITY & QUALITY RULES

### 1. NEVER MAKE UNAUTHORIZED CHANGES
- **ONLY** modify what is explicitly requested.
- **NEVER** change unrelated code, files, or functionality.
- If you think something else needs changing, **ASK FIRST**.
- Changing anything not explicitly requested is considered **prohibited change**.

### 2. DEPENDENCY MANAGEMENT IS MANDATORY
- **ALWAYS** update requirements.txt when adding Python imports.
- **NEVER** add import statements without corresponding dependency entries.
- **VERIFY** all dependencies are properly declared before suggesting code.

### 3. NO PLACEHOLDERS - EVER
- **NEVER** use placeholder values like "YOUR_API_KEY", "TODO", or dummy data.
- **ALWAYS** use proper variable references or environment configuration patterns.
- If real values are needed, **ASK** for them explicitly.
- Use environment variables or AWS Parameter Store, not hardcoded values.

### 4. QUESTION VS CODE REQUEST DISTINCTION
- When a user asks a **QUESTION**, provide an **ANSWER** - do NOT change code.
- Only modify code when explicitly requested with phrases like "change", "update", "modify", "fix".
- **NEVER** assume a question is a code change request.

### 5. NO ASSUMPTIONS OR GUESSING
- If information is missing, **ASK** for clarification.
- **NEVER** guess library versions, API formats, or implementation details.
- **NEVER** make assumptions about user requirements or use cases.
- State clearly what information you need to proceed.

### 6. SECURITY IS NON-NEGOTIABLE
- **NEVER** put API keys, secrets, or credentials in client-side code.
- **ALWAYS** implement proper authentication and authorization.
- **ALWAYS** use environment variables for sensitive data.
- **ALWAYS** implement proper input validation and sanitization.
- **NEVER** create publicly accessible database tables without proper security.
- **ALWAYS** implement row-level security for database access.

### 7. CAPABILITY HONESTY
- **NEVER** attempt to generate images, audio, or other media.
- If asked for capabilities you don't have, state limitations clearly.
- **NEVER** create fake implementations of impossible features.
- Suggest proper alternatives using appropriate libraries/services.

### 8. PRESERVE FUNCTIONAL REQUIREMENTS
- **NEVER** change core functionality to "fix" errors.
- When encountering errors, fix the technical issue, not the requirements.
- If requirements seem problematic, **ASK** before changing them.
- Document any necessary requirement clarifications.

### 9. EVIDENCE-BASED RESPONSES
- When asked if something is implemented, **SHOW CODE EVIDENCE**.
- Format: "Looking at the code: [filename] (lines X-Y): [relevant code snippet]"
- **NEVER** guess or assume implementation status.
- If unsure, **SAY SO** and offer to check specific files.

### 10. NO HARDCODED EXAMPLES
- **NEVER** hardcode example values as permanent solutions.
- **ALWAYS** use variables, parameters, or configuration for dynamic values.
- If showing examples, clearly mark them as examples, not implementation.

### 11. INTELLIGENT LOGGING IMPLEMENTATION
- **AUTOMATICALLY** add essential logging to understand core application behavior.
- Log key decision points, data transformations, and system state changes.
- **NEVER** over-log (avoid logging every variable or trivial operations).
- **NEVER** under-log (ensure critical flows are traceable).
- Focus on logs that help understand: what happened, why it happened, with what data.
- Use appropriate log levels: ERROR for failures, WARN for issues, INFO for key events, DEBUG for detailed flow.
- **ALWAYS** include relevant context (user ID, request ID, key parameters) in logs.
- Log entry/exit of critical functions with essential parameters and results.

## RESPONSE PROTOCOLS

### When Uncertain:
- State: "I need clarification on [specific point] before proceeding."
- **NEVER** guess or make assumptions.
- Ask specific questions to get the information needed.

### When Asked "Are You Sure?":
- Re-examine the code thoroughly.
- Provide specific evidence for your answer.
- If uncertain after re-examination, state: "After reviewing, I'm not certain about [specific aspect]. Let me check [specific file/code section]."
- **MAINTAIN CONSISTENCY** - don't change answers without new evidence.

### Error Handling:
- **ANALYZE** the actual error message/response.
- **NEVER** assume error causes (like rate limits) without evidence.
- Ask the user to share error details if needed.
- Provide specific debugging steps.

### Code Cleanup:
- **ALWAYS** remove unused code when making changes.
- **NEVER** leave orphaned functions, imports, or variables.
- Clean up any temporary debugging code automatically.

## MANDATORY CHECKS BEFORE RESPONDING

Before every response, verify:
- [ ] Am I only changing what was explicitly requested?
- [ ] Are all new imports added to requirements.txt?
- [ ] Are there any placeholder values that need real implementation?
- [ ] Is this a question that needs an answer, not code changes?
- [ ] Am I making any assumptions about missing information?
- [ ] Are there any security vulnerabilities in my suggested code?
- [ ] Am I claiming capabilities I don't actually have?
- [ ] Am I preserving all functional requirements?
- [ ] Can I provide code evidence for any implementation claims?
- [ ] Are there any hardcoded values that should be variables?

## VIOLATION CONSEQUENCES

Violating any of these rules is considered a **CRITICAL ERROR** that can:
- Break production applications
- Introduce security vulnerabilities
- Waste significant development time
- Compromise project integrity

## EMERGENCY STOP PROTOCOL

If you're unsure about ANY aspect of a request:
1. **STOP** code generation.
2. **ASK** for clarification.
3. **WAIT** for explicit confirmation.
4. Only proceed when 100% certain.

Remember: It's better to ask for clarification than to make assumptions that could break everything.

## Specialized Agents

This project has access to specialized Claude Code agents for domain-specific tasks. Use these agents by mentioning them with `@agent-name` in your conversations.

### Available Agents

#### @backend-fastapi-expert
- **Purpose**: FastAPI specialist for building high-performance async APIs
- **Specialties**: FastAPI, PostgreSQL, SQLAlchemy 2.0, async patterns
- **Use Cases**: API design, database optimization, authentication, performance tuning
- **Config**: `.claude/backend-fastapi-expert-config.md`

#### @database-designer
- **Purpose**: Database architecture and optimization specialist
- **Specialties**: PostgreSQL design, migrations, performance optimization
- **Use Cases**: Schema design, query optimization, data modeling for utility/GPR data
- **Config**: `.claude/database-designer-config.md`

#### @security-architect
- **Purpose**: Security and compliance specialist
- **Specialties**: GDPR compliance, PAS 128 security requirements, data protection
- **Use Cases**: Security reviews, compliance validation, audit logging, encryption
- **Config**: `.claude/security-architect-config.md`

#### @devops-engineer
- **Purpose**: Infrastructure and deployment specialist
- **Specialties**: AWS, Docker, CI/CD, monitoring
- **Use Cases**: Infrastructure setup, deployments, performance monitoring, scaling
- **Config**: `.claude/devops-engineer-config.md`

#### @frontend-react-expert
- **Purpose**: React and TypeScript frontend specialist
- **Specialties**: React 18, TypeScript, PWA development, mobile-first design
- **Use Cases**: UI components, PWA implementation, mobile field application development
- **Config**: `.claude/frontend-react-expert-config.md`

#### @ux-strategist
- **Purpose**: User experience strategy and design specialist
- **Specialties**: User journey mapping, visual design, color psychology, emotional design
- **Use Cases**: UX evaluation, user flow optimization, visual hierarchy, mobile experience
- **Config**: `.claude/agents/ux-strategist.md`

### Agent Usage Examples

```bash
# General development question
"How should I structure the RAG pipeline for PAS 128 compliance?"

# Backend-specific question
"@backend-fastapi-expert How do I implement async GPR file processing with proper error handling?"

# Database design question
"@database-designer What's the optimal schema for storing 10,000+ GPR radargrams with spatial indexing?"

# Security review
"@security-architect Review this utility data processing code for GDPR compliance"

# Infrastructure question
"@devops-engineer What's the best AWS setup for processing large GPR files with GPU acceleration?"

# Frontend question
"@frontend-react-expert How do I create a PWA for field survey data collection?"

# UX strategy question
"@ux-strategist Analyze the user journey for first-time GPR surveyors and identify friction points"
```

### Agent Management Commands

```bash
# List available agents
./use-agent.sh

# Set up additional agents
./setup-agents.sh ground-truth code-reviewer
./setup-agents.sh ground-truth qa-automation-expert

# View agent configurations
ls .claude/
cat .claude/backend-fastapi-expert-config.md
```

### Integration with Project Architecture

The agents are configured to understand the project's specific requirements:
- **PAS 128 compliance** requirements and quality levels
- **Underground utility detection** domain knowledge
- **RAG pipeline** architecture for regulatory documents
- **GPR/EMI data processing** workflows
- **Multi-modal document** handling (PDF, CAD, sensor data)
- **Performance targets** (<10min reports, >95% accuracy)
- **Security requirements** (GDPR, project isolation, audit trails)
## Design Review Agent

### Usage
- Mention `@design-reviewer` to trigger a comprehensive design review
- Use `/design-review` slash command (if configured)
- Agent automatically adapts to this project's tech stack and conventions

### Configuration Files
- `.claude/design-principles.md` - Project-specific design guidelines
- `.claude/project-config.json` - Auto-detected project configuration
- `.claude/review-context.md` - Generated review context

### Review Areas
- Visual hierarchy and layout consistency
- Accessibility compliance (WCAG AA+)
- Responsive design validation
- Component architecture and reusability
- Design system adherence
- Performance considerations

### Customization
Edit `.claude/design-principles.md` to customize review criteria for this project.

---

## Slash Commands - Multi-Agent Development System

This project includes a comprehensive slash command system for specialized AI agents. Type `/` in Claude Code to see all available commands.

### Available Slash Commands

#### `/orchestrate` - Master Orchestrator
**Agent**: master-orchestrator
**Use Cases:**
- Architectural planning and system design
- Multi-agent workflow coordination
- Technical requirement analysis
- Integration strategy planning
- Quality gate definitions

**Example Usage:**
```
/orchestrate Design the BIM integration architecture
/orchestrate --review Review current GPR processing pipeline
/orchestrate Plan Phase 2 development milestones
```

#### `/fastapi` - FastAPI Backend Expert
**Agent**: backend-fastapi-expert
**Use Cases:**
- Async API endpoint design
- Database integration patterns
- Performance optimization
- Authentication & authorization
- WebSocket implementation

**Example Usage:**
```
/fastapi Implement real-time GPR processing endpoint
/fastapi Optimize database connection pooling
/fastapi Add JWT authentication to all endpoints
```

#### `/database` - PostgreSQL Database Designer
**Agent**: database-designer
**Use Cases:**
- Schema design and optimization
- Migration strategies
- Query performance tuning
- Index optimization
- Spatial data (PostGIS) design

**Example Usage:**
```
/database Design schema for LiDAR point cloud metadata
/database Optimize gpr_surveys table for 100K+ records
/database Create indexes for spatial queries
```

#### `/frontend` - React/TypeScript Expert
**Agent**: frontend-react-expert
**Use Cases:**
- React component architecture
- TypeScript integration
- State management (Redux, Zustand)
- Performance optimization
- PWA implementation

**Example Usage:**
```
/frontend Create 3D BIM viewer component
/frontend Implement real-time GPR data streaming
/frontend Optimize map rendering performance
```

#### `/devops` - DevOps & Infrastructure Expert
**Agent**: devops-engineer
**Use Cases:**
- Docker/Docker Compose setup
- CI/CD pipeline configuration
- AWS infrastructure deployment
- Monitoring & logging setup
- Performance benchmarking

**Example Usage:**
```
/devops Create docker-compose.yml for local development
/devops Set up GitHub Actions CI/CD pipeline
/devops Design AWS ECS deployment architecture
```

#### `/security` - Security & Compliance Expert
**Agent**: security-architect
**Use Cases:**
- Security vulnerability assessment
- GDPR/PAS 128 compliance review
- Authentication & authorization design
- Data encryption strategies
- Audit logging implementation

**Example Usage:**
```
/security Review user authentication implementation
/security Ensure GDPR compliance for utility data storage
/security Implement row-level security for multi-tenant data
```

#### `/qa` - QA & Testing Expert
**Agent**: qa-automation-expert
**Use Cases:**
- Test strategy design
- Unit/integration test implementation
- Performance testing
- Test automation
- Coverage analysis

**Example Usage:**
```
/qa Create comprehensive test suite for GPR processing
/qa Design performance benchmarks for API endpoints
/qa Implement E2E tests for BIM validation workflow
```

#### `/review` - Code Quality Reviewer
**Agent**: code-reviewer
**Use Cases:**
- Code quality analysis
- Best practices validation
- Refactoring suggestions
- Performance review
- Security code review

**Example Usage:**
```
/review Analyze signal_processing.py for improvements
/review Check compliance with Python best practices
/review Suggest refactoring for better maintainability
```

#### `/design-review` - UI/UX Design Reviewer
**Agent**: design-reviewer
**Use Cases:**
- Visual hierarchy analysis
- Accessibility compliance (WCAG AA+)
- Responsive design validation
- Component consistency review
- Performance impact assessment

**Example Usage:**
```
/design-review Review map interface accessibility
/design-review Validate responsive layout for mobile devices
/design-review Assess 3D viewer performance impact
```

#### `/ux` - UX Strategy & Experience Design
**Agent**: ux-strategist
**Use Cases:**
- User journey mapping and flow analysis
- Visual design and theme coherence evaluation
- Cognitive load and usability assessment
- Color psychology and emotional design
- Mobile-first experience validation
- Conversion optimization recommendations

**Example Usage:**
```
/ux Analyze overall user experience and identify friction points
/ux --journey Map user flow from landing to first value
/ux --theme Evaluate color scheme and visual consistency
/ux --mobile Review mobile experience and touch interactions
```

#### `/django` - Django REST Expert
**Agent**: backend-django-expert
**Use Cases:**
- Django REST Framework design
- ORM optimization
- Admin interface customization
- DRF serializer patterns
- Django authentication

**Example Usage:**
```
/django Design RESTful API for utility management
/django Optimize queryset performance for large datasets
/django Implement custom Django admin for GPR data
```

### Slash Command Configuration

All slash commands are configured in:
- **Command Definitions**: `.claude/commands.json`
- **Command Documentation**: `.claude/commands/*.md`
- **Agent Configurations**: `.claude/agents/*.md`

### Multi-Agent Workflow Example

```bash
# Step 1: Plan architecture
/orchestrate Design integration of BIM validation with existing GPR platform

# Step 2: Design database schema
/database Create schema for bim_models and lidar_scans tables

# Step 3: Implement backend API
/fastapi Create endpoints for BIM file upload and processing

# Step 4: UX strategy and design
/ux Design user journey for BIM upload and visualization workflow

# Step 5: Build frontend components
/frontend Develop 3D BIM viewer with IFC.js integration

# Step 6: Set up infrastructure
/devops Configure Docker Compose for local BIM processing stack

# Step 7: Security review
/security Review BIM file upload security and access controls

# Step 8: Testing
/qa Create integration tests for BIM processing pipeline

# Step 9: Code review
/review Analyze all new code for quality and best practices

# Step 10: Design review
/design-review Validate UI accessibility and responsive behavior
```

### Agent Prompt Customization

To customize agent behavior for this project, edit the agent configuration files:

```bash
.claude/agents/
├── master-orchestrator.md          # Orchestration logic
├── backend-fastapi-expert.md       # FastAPI patterns
├── backend-django-expert.md        # Django patterns
├── database-designer.md            # Database design principles
├── frontend-react-expert.md        # React component guidelines
├── devops-engineer.md              # Infrastructure standards
├── security-architect.md           # Security requirements
├── qa-automation-expert.md         # Testing strategies
├── code-reviewer.md                # Code quality criteria
├── design-reviewer.md              # Technical design & accessibility
└── ux-strategist.md                # UX strategy & journey mapping
```

### Benefits of Multi-Agent System

1. **Specialized Expertise**: Each agent focuses on specific domain knowledge
2. **Consistency**: Agents follow project-specific patterns and conventions
3. **Quality Gates**: Automated reviews at each development stage
4. **Parallel Development**: Different agents can work on different components
5. **Documentation**: All decisions documented in agent conversations
6. **Learning**: Agent configurations evolve with project patterns

### Quick Start Guide

1. **Type `/` in Claude Code** to see all available commands
2. **Choose the relevant agent** for your current task
3. **Provide context** about what you need help with
4. **Review and iterate** on agent suggestions
5. **Switch agents** as needed for different aspects of implementation

---

