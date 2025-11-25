# Underground Utility Detection Platform
## Consolidated Product Requirements Document

---

## Product Overview

### Vision Statement
To become the leading AI-native platform for underground utility detection and PAS 128 compliance, preventing utility strikes and transforming manual surveying workflows into automated, intelligent processes.

### Mission
Deliver a comprehensive solution that reduces utility strike incidents by 60%, cuts report generation time from 8 hours to 10 minutes, and ensures 100% PAS 128:2022 compliance while providing predictive risk assessment capabilities.

---

## Business Objectives

### Primary Goals
1. **Market Leadership**: Establish platform as the standard for AI-native utility surveying
2. **Revenue Growth**: Achieve £750K ARR in Year 1, scaling to £6.8M by Year 3
3. **Customer Success**: Demonstrate 75% time savings and 95% accuracy vs manual processes
4. **Risk Reduction**: Prevent 500+ utility strikes annually through predictive analytics

### Success Metrics
- **Financial**: £750K ARR (30 customers) by end of Year 1
- **Product**: <10 minute report generation, >95% accuracy
- **Market**: 50% market awareness in UK utility surveying sector
- **Customer**: NPS >70, <5% annual churn rate

---

## Target Market & Users

### Primary Market Segments

#### 1. Tier 2 Contractors (£50-200M revenue)
**Profile**:
- 500+ companies in UK market
- 50-200 projects annually requiring PAS 128
- £20-100K annual software budgets
- Pain: Manual reporting bottleneck, strike liability

**Use Cases**:
- Large infrastructure projects
- Highway and rail construction
- Commercial development
- Utilities installation and maintenance

#### 2. Specialist Survey Firms
**Profile**:
- 200+ PAS 128 certified firms
- 10-50 surveyors per firm
- £10-50K software budgets
- Pain: Skilled staff shortage, competition pressure

**Use Cases**:
- Commissioned utility surveys
- Due diligence surveys
- Damage investigation
- Compliance verification

#### 3. Utility Companies
**Profile**:
- 50+ major UK operators
- 500+ excavations monthly
- £100K+ software budgets
- Pain: Asset protection, liability management

**Use Cases**:
- Asset location verification
- Maintenance planning
- Third-party works oversight
- Risk assessment

### User Personas

#### Primary: Survey Manager
- **Role**: Manages survey operations and compliance
- **Experience**: 10+ years in utility surveying
- **Goals**: Ensure compliance, reduce costs, prevent strikes
- **Frustrations**: Time-consuming reports, staff shortages
- **Tech Comfort**: Moderate, prefers simple interfaces

#### Secondary: Field Surveyor
- **Role**: Conducts on-site surveys and data collection
- **Experience**: 5+ years GPR/EMI experience
- **Goals**: Accurate detection, efficient workflows
- **Frustrations**: Complex equipment, weather delays
- **Tech Comfort**: High, mobile-first preference

#### Tertiary: Project Manager
- **Role**: Oversees construction projects requiring surveys
- **Experience**: 15+ years construction management
- **Goals**: Avoid delays, prevent incidents, manage costs
- **Frustrations**: Survey delays, unclear reports
- **Tech Comfort**: Low, needs simple dashboards

---

## Product Requirements

### Functional Requirements

#### 1. Data Ingestion & Processing

**FR1.1: Multi-Format Data Import**
- **Priority**: MUST HAVE
- **Description**: System shall support import of GPR files (SEG-Y, DZT, DT1), utility records (PDF, Excel), CAD drawings (DWG, DXF), and site photos (JPEG with GPS)
- **Acceptance Criteria**:
  - Process 95% of common file formats without errors
  - Validate file integrity and format compliance
  - Extract metadata automatically
  - Handle file sizes up to 2GB

**FR1.2: Real-time Data Validation**
- **Priority**: MUST HAVE
- **Description**: System shall validate uploaded data for completeness, accuracy, and compliance requirements
- **Acceptance Criteria**:
  - Flag missing mandatory fields
  - Validate coordinate systems (OSGB36, WGS84)
  - Check data quality scores
  - Provide validation reports with specific issues

**FR1.3: Automated Data Correlation**
- **Priority**: MUST HAVE
- **Description**: System shall correlate data from multiple sources to create unified utility positions
- **Acceptance Criteria**:
  - Spatially align GPR, EMI, and record data
  - Resolve conflicts between sources
  - Assign confidence weights
  - Flag discrepancies for review

#### 2. AI-Powered Analysis

**FR2.1: GPR Interpretation**
- **Priority**: MUST HAVE
- **Description**: System shall automatically interpret GPR data to detect buried utilities
- **Acceptance Criteria**:
  - Detect hyperbola patterns in radargrams
  - Classify utility types (gas, electric, water, telecom, sewer)
  - Estimate depths with ±15% accuracy
  - Provide confidence scores (0-100)

**FR2.2: Risk Assessment**
- **Priority**: MUST HAVE
- **Description**: System shall assess strike risk based on detection confidence, historical incidents, and construction plans
- **Acceptance Criteria**:
  - Generate risk scores (1-10 scale)
  - Identify high-risk zones
  - Provide mitigation recommendations
  - Reference historical incident data

**FR2.3: PAS 128 Compliance Analysis**
- **Priority**: MUST HAVE
- **Description**: System shall ensure all outputs comply with PAS 128:2022 requirements
- **Acceptance Criteria**:
  - Assign quality levels (QL-A to QL-D)
  - Validate compliance against 500+ requirements
  - Cite relevant PAS 128 clauses
  - Flag non-compliance issues

#### 3. Report Generation

**FR3.1: Automated Report Creation**
- **Priority**: MUST HAVE
- **Description**: System shall generate PAS 128-compliant reports automatically
- **Acceptance Criteria**:
  - Complete report generation in <10 minutes
  - Include all mandatory sections per PAS 128
  - Embed automatic citations and references
  - Support custom client branding

**FR3.2: Multi-Format Export**
- **Priority**: MUST HAVE
- **Description**: System shall export reports in multiple formats for different stakeholders
- **Acceptance Criteria**:
  - PDF with client branding
  - Editable Word documents
  - Excel data tables
  - CAD/GIS integration files
  - Email-ready summaries

**FR3.3: Dynamic Content Generation**
- **Priority**: SHOULD HAVE
- **Description**: System shall adapt report content based on project type and requirements
- **Acceptance Criteria**:
  - Project-specific templates
  - Stakeholder-appropriate language
  - Configurable detail levels
  - Automated quality level justification

#### 4. Field Data Collection

**FR4.1: Mobile Application**
- **Priority**: MUST HAVE
- **Description**: System shall provide mobile app for field data collection
- **Acceptance Criteria**:
  - Offline capability for remote sites
  - GPS-tagged photo capture
  - Voice note recording
  - Real-time data sync when connected
  - Works on iOS and Android

**FR4.2: Equipment Integration**
- **Priority**: SHOULD HAVE
- **Description**: System shall integrate with common GPR/EMI equipment
- **Acceptance Criteria**:
  - Direct data import from Radiodetection devices
  - Support for GSSI and Sensors & Software
  - Bluetooth/WiFi connectivity
  - Real-time data streaming

#### 5. Collaboration & Workflow

**FR5.1: Project Management**
- **Priority**: MUST HAVE
- **Description**: System shall support multi-user project collaboration
- **Acceptance Criteria**:
  - Role-based access control
  - Project sharing capabilities
  - Comment and review workflows
  - Version control for reports

**FR5.2: Customer Portal**
- **Priority**: SHOULD HAVE
- **Description**: System shall provide customer access to project reports and status
- **Acceptance Criteria**:
  - Secure customer login
  - Project status dashboards
  - Report download capabilities
  - Notification system

### Non-Functional Requirements

#### Performance Requirements

**NFR1: Response Time**
- **Priority**: MUST HAVE
- **Requirement**: System shall respond to user actions within specified timeframes
- **Acceptance Criteria**:
  - Web page load: <3 seconds
  - API responses: <200ms P95
  - Report generation: <10 minutes
  - Search queries: <100ms

**NFR2: Throughput**
- **Priority**: MUST HAVE
- **Requirement**: System shall handle specified concurrent load
- **Acceptance Criteria**:
  - 50 concurrent users (Year 1)
  - 500 concurrent users (Year 2)
  - 2000 concurrent users (Year 3)
  - 1000 reports generated per day

**NFR3: Scalability**
- **Priority**: MUST HAVE
- **Requirement**: System shall scale to support business growth
- **Acceptance Criteria**:
  - Horizontal scaling capability
  - Auto-scaling based on load
  - Database partitioning support
  - CDN for global distribution

#### Reliability Requirements

**NFR4: Availability**
- **Priority**: MUST HAVE
- **Requirement**: System shall maintain high availability
- **Acceptance Criteria**:
  - 99.9% uptime (8.76 hours downtime per year)
  - Planned maintenance <4 hours monthly
  - Automatic failover within 5 minutes
  - Disaster recovery RTO <4 hours

**NFR5: Data Integrity**
- **Priority**: MUST HAVE
- **Requirement**: System shall ensure data accuracy and consistency
- **Acceptance Criteria**:
  - Zero data loss
  - ACID compliance for transactions
  - Checksums for file integrity
  - Audit trail for all changes

#### Security Requirements

**NFR6: Authentication & Authorization**
- **Priority**: MUST HAVE
- **Requirement**: System shall secure access to resources
- **Acceptance Criteria**:
  - Multi-factor authentication
  - Role-based access control
  - Session management
  - Password policy enforcement

**NFR7: Data Protection**
- **Priority**: MUST HAVE
- **Requirement**: System shall protect sensitive data
- **Acceptance Criteria**:
  - Encryption at rest (AES-256)
  - Encryption in transit (TLS 1.3)
  - PII detection and redaction
  - GDPR compliance

#### Usability Requirements

**NFR8: User Experience**
- **Priority**: MUST HAVE
- **Requirement**: System shall provide intuitive user experience
- **Acceptance Criteria**:
  - <80% users complete key tasks without training
  - <3 clicks to access common features
  - Responsive design for all devices
  - Accessibility compliance (WCAG 2.1)

**NFR9: Learning Curve**
- **Priority**: SHOULD HAVE
- **Requirement**: System shall minimize learning time for new users
- **Acceptance Criteria**:
  - <30 minutes to complete first report
  - In-app guidance and tutorials
  - Context-sensitive help
  - Video training materials

---

## MVP Feature Prioritization

### Must Have (MVP)
1. **Data Import**: GPR, PDF, CAD file processing
2. **AI Analysis**: Basic GPR interpretation and risk scoring
3. **Report Generation**: PAS 128-compliant PDF reports
4. **Web Interface**: Project management and report viewing
5. **Compliance**: Basic PAS 128 validation
6. **Mobile App**: Photo capture and GPS tagging

### Should Have (Post-MVP)
1. **Equipment Integration**: Direct device connectivity
2. **Advanced Analytics**: Machine learning improvements
3. **Customer Portal**: External stakeholder access
4. **API**: Third-party integrations
5. **Advanced Reports**: Custom templates and branding

### Could Have (Future Versions)
1. **Offline Processing**: Complete offline capability
2. **AR Visualization**: Augmented reality utility display
3. **Predictive Maintenance**: Asset condition monitoring
4. **International Standards**: ASCE 38-22 support
5. **White Label**: Partner platform options

### Won't Have (This Release)
1. **CAD Design Tools**: Full CAD editing capabilities
2. **Financial Management**: Invoicing and billing
3. **HR Management**: Team scheduling and payroll
4. **Asset Management**: Comprehensive asset tracking

---

## User Stories & Acceptance Criteria

### Epic 1: Survey Data Processing

**US1.1**: As a Survey Manager, I want to upload multiple data sources so that I can generate comprehensive reports
**Acceptance Criteria**:
- Given I have GPR files, utility records, and CAD drawings
- When I upload them to a new project
- Then the system processes all files and shows validation results
- And I can proceed to report generation

**US1.2**: As a Field Surveyor, I want to capture site photos with GPS locations so that I can document field conditions
**Acceptance Criteria**:
- Given I am using the mobile app on site
- When I take photos of utility markers or excavations
- Then GPS coordinates and timestamp are automatically recorded
- And photos are uploaded when connectivity is available

### Epic 2: AI Analysis & Risk Assessment

**US2.1**: As a Survey Manager, I want automated GPR interpretation so that I can reduce analysis time
**Acceptance Criteria**:
- Given I have uploaded GPR radargram files
- When the AI processes the data
- Then detected utilities are marked with confidence scores
- And depth estimates are provided with accuracy indicators

**US2.2**: As a Project Manager, I want risk assessment reports so that I can plan excavation safely
**Acceptance Criteria**:
- Given the system has processed all survey data
- When I request a risk assessment
- Then I receive risk scores for all planned excavation areas
- And specific mitigation recommendations are provided

### Epic 3: Report Generation & Compliance

**US3.1**: As a Survey Manager, I want automated PAS 128 reports so that I can deliver compliant documentation quickly
**Acceptance Criteria**:
- Given I have completed survey processing
- When I generate a PAS 128 report
- Then the report is created in <10 minutes
- And all mandatory sections are included with proper citations

**US3.2**: As a Client, I want to receive reports in multiple formats so that I can use them in different contexts
**Acceptance Criteria**:
- Given a completed survey report
- When I request the deliverables
- Then I receive PDF, Word, Excel, and CAD versions
- And all formats contain consistent information

### Epic 4: Collaboration & Project Management

**US4.1**: As a Survey Manager, I want to track project progress so that I can manage deliverables effectively
**Acceptance Criteria**:
- Given I have multiple active projects
- When I view the project dashboard
- Then I see status updates for each project phase
- And I can identify any blocked or delayed projects

**US4.2**: As a Team Member, I want to collaborate on reports so that we can leverage collective expertise
**Acceptance Criteria**:
- Given I am working on a survey report
- When I invite colleagues to review
- Then they can add comments and suggestions
- And I can track all changes and approvals

---

## Technical Constraints

### Platform Constraints
- **Web Browser Support**: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
- **Mobile Support**: iOS 14+, Android 10+
- **File Size Limits**: 2GB maximum per file upload
- **Concurrent Users**: 50 (MVP), scaling to 2000 (Year 3)

### Integration Constraints
- **API Rate Limits**: OpenAI API limits, Pinecone query limits
- **Data Formats**: Must support legacy utility record formats
- **Geographic Scope**: UK initially, US expansion planned
- **Regulatory Standards**: PAS 128:2022 primary, ASCE 38-22 future

### Performance Constraints
- **Report Generation**: <10 minutes for standard surveys
- **Data Processing**: <30 seconds for 100MB GPR files
- **Search Response**: <100ms for vector database queries
- **Uptime**: 99.9% availability requirement

---

## Compliance & Regulatory Requirements

### UK Regulatory Compliance

**PAS 128:2022 Compliance**
- All reports must conform to PAS 128:2022 specification
- Quality level assignments (QL-A to QL-D) required
- Proper citation of relevant clauses
- Confidence levels for all detections
- Survey methodology documentation

**CDM Regulations 2015**
- 7-year data retention requirement
- Safety risk documentation
- Principal designer coordination
- Worker safety considerations

**GDPR Compliance**
- Personal data protection
- Right to deletion
- Data portability
- Consent management
- Privacy by design

### Data Security Standards
- **ISO 27001**: Information security management
- **SOC 2 Type II**: Service organization controls
- **Cyber Essentials Plus**: UK government scheme
- **OWASP Top 10**: Web application security

---

## Success Criteria & KPIs

### Product Success Metrics

**Functionality KPIs**:
- Report generation time: <10 minutes (vs 8 hours manual)
- Accuracy rate: >95% vs manual interpretation
- Compliance rate: 100% PAS 128 conformance
- User task completion: >80% success rate

**Performance KPIs**:
- System uptime: >99.9%
- API response time: <200ms P95
- Page load time: <3 seconds
- Mobile app responsiveness: <1 second

**User Experience KPIs**:
- Net Promoter Score: >70
- User adoption rate: >80% within 30 days
- Support ticket volume: <5% of active users
- Training completion: <30 minutes for new users

### Business Success Metrics

**Revenue KPIs**:
- Annual Recurring Revenue: £750K (Year 1)
- Customer count: 30 (Year 1) → 150 (Year 3)
- Average contract value: £25K → £45K
- Customer lifetime value: £200K-400K

**Market KPIs**:
- Market awareness: 50% of target segment
- Lead conversion rate: >25%
- Customer acquisition cost: <£25K
- Sales cycle length: <4 months

**Impact KPIs**:
- Utility strikes prevented: 500+ annually
- Time savings delivered: 10,000+ hours annually
- Customer ROI: >300%
- Industry recognition: Awards and certifications

---

## Risk Assessment & Mitigation

### Product Risks

**Technical Risks**:
- **AI Accuracy**: Risk of false positives/negatives in utility detection
  - Mitigation: Human-in-loop validation, confidence thresholds
- **Data Quality**: Poor input data affecting results
  - Mitigation: Robust validation, multiple source correlation
- **Performance**: System cannot handle scale requirements
  - Mitigation: Load testing, performance optimization

**User Adoption Risks**:
- **Learning Curve**: Users struggle with new technology
  - Mitigation: Intuitive UX design, comprehensive training
- **Integration**: Difficulty integrating with existing workflows
  - Mitigation: API-first design, flexible configuration
- **Resistance**: Industry resistance to AI-based solutions
  - Mitigation: Pilot programs, proven ROI demonstration

### Business Risks

**Market Risks**:
- **Competition**: Established players launching competing solutions
  - Mitigation: Fast execution, proprietary data advantages
- **Regulation**: Changes to PAS 128 or other standards
  - Mitigation: Modular compliance engine, industry relationships
- **Economy**: Economic downturn affecting construction spending
  - Mitigation: Focus on cost savings and ROI

---

## Future Roadmap

### Version 2.0 (Year 2)
- Advanced ML models with improved accuracy
- Real-time equipment integration
- Customer portal and API access
- Infrastructure inspection module
- US market localization (ASCE 38-22)

### Version 3.0 (Year 3)
- Predictive maintenance capabilities
- AR visualization for field workers
- Advanced analytics and reporting
- White-label platform options
- International expansion (EU markets)

### Version 4.0+ (Future)
- IoT sensor integration
- Blockchain for compliance verification
- Advanced AI including computer vision
- Drone integration for site surveys
- Full BIM integration

---

*This consolidated product requirements document provides the complete specification for building the Underground Utility Detection Platform, incorporating all functional and non-functional requirements, user stories, technical constraints, and success criteria necessary for successful product development and market launch.*