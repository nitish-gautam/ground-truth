/**
 * Glossary of Technical Terms
 * Provides definitions for tooltips and help text
 */

export const GLOSSARY = {
  TAEM: 'Technical, Asset, Engineering, Management - A composite health score measuring asset readiness across four key dimensions',
  PAS_128: 'Publicly Available Specification 128:2022 - UK standard for utility detection and verification surveys',
  QUALITY_LEVEL: 'PAS 128 quality classification (QL-A through QL-D) indicating survey accuracy and methodology',
  BIM: 'Building Information Modeling - 3D digital representation of infrastructure with embedded data',
  GIS: 'Geographic Information System - Spatial data platform for mapping and analysis',
  LIDAR: 'Light Detection and Ranging - Laser scanning technology for precise 3D terrain mapping',
  GPR: 'Ground Penetrating Radar - Technology for detecting underground utilities and features',
  CDM: 'Construction Design and Management Regulations 2015 - UK health and safety legislation',
  RIDDOR: 'Reporting of Injuries, Diseases and Dangerous Occurrences Regulations - UK safety reporting framework',
  IFC: 'Industry Foundation Classes - Open standard file format for BIM data exchange',
  SYNTHETIC_DATA: 'Computer-generated data for demonstration purposes - not from real-world measurements',
  REAL_DATA: 'Actual data collected from HS2 project sources including sensors, surveys, and documentation',
  COMPLIANCE_CHECK: 'Automated verification that assets and processes meet required standards and regulations',
  READINESS_STATUS: 'Overall assessment of whether an asset is prepared for the next project phase',
  AT_RISK: 'Assets with potential compliance, schedule, or quality issues requiring immediate attention',
};

export type GlossaryKey = keyof typeof GLOSSARY;

export const getTooltip = (key: GlossaryKey): string => {
  return GLOSSARY[key] || '';
};
