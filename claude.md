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
REDIS_URL=redis://localhost:6379
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

