/**
 * HS2 Assurance Intelligence Platform - TypeScript Type Definitions
 * 
 * These types match the backend Pydantic schemas for type-safe API integration.
 */

// ============================================================================
// Core Asset Types
// ============================================================================

export type AssetStatus = 'Ready' | 'Not Ready' | 'At Risk';

export type AssetType = 
  | 'Bridge' 
  | 'Tunnel' 
  | 'Viaduct' 
  | 'Station' 
  | 'Track' 
  | 'Signaling' 
  | 'Earthworks';

export type RuleSeverity = 'Critical' | 'Major' | 'Minor';

export interface HS2Asset {
  asset_id: string;
  asset_name: string;
  asset_type: AssetType;
  contractor: string;
  route_section: string;
  status: AssetStatus;
  taem_score: number;
  last_evaluated: string; // ISO datetime
  metadata?: Record<string, any>;
}

// ============================================================================
// Rule Evaluation Types
// ============================================================================

export interface RuleEvaluation {
  rule_id: string;
  rule_name: string;
  passed: boolean;
  severity: RuleSeverity;
  message: string;
  evidence: Record<string, any>;
  evaluated_at: string; // ISO datetime
}

export interface ReadinessReport {
  asset_id: string;
  status: AssetStatus;
  taem_score: number;
  rules_passed: number;
  rules_failed: number;
  critical_failures: number;
  major_failures: number;
  minor_failures: number;
  evaluations: RuleEvaluation[];
  evaluated_at: string; // ISO datetime
  evaluated_by?: string;
}

// ============================================================================
// Deliverable Types
// ============================================================================

export type DeliverableStatus = 
  | 'Not Started' 
  | 'In Progress' 
  | 'Submitted' 
  | 'Approved' 
  | 'Rejected';

export interface Deliverable {
  deliverable_id: string;
  asset_id: string;
  deliverable_name: string;
  deliverable_code: string;
  status: DeliverableStatus;
  due_date?: string; // ISO date
  submitted_date?: string; // ISO date
  approved_date?: string; // ISO date
  version: string;
  file_path?: string;
  metadata?: Record<string, any>;
}

// ============================================================================
// Cost Types
// ============================================================================

export interface CostRecord {
  cost_id: string;
  asset_id: string;
  category: string;
  budget_amount: number;
  actual_amount: number;
  variance: number;
  variance_percentage: number;
  currency: string;
  reporting_period: string;
  notes?: string;
}

export interface CostSummary {
  asset_id: string;
  total_budget: number;
  total_actual: number;
  total_variance: number;
  variance_percentage: number;
  currency: string;
  categories: CostRecord[];
}

// ============================================================================
// Certificate Types
// ============================================================================

export type CertificateStatus = 'Valid' | 'Expired' | 'Expiring Soon';

export interface Certificate {
  certificate_id: string;
  asset_id: string;
  certificate_name: string;
  certificate_type: string;
  issuer: string;
  issue_date: string; // ISO date
  expiry_date?: string; // ISO date
  status: CertificateStatus;
  file_path?: string;
  metadata?: Record<string, any>;
}

// ============================================================================
// History Types
// ============================================================================

export type ChangeType = 
  | 'status_change' 
  | 'evaluation' 
  | 'deliverable_update' 
  | 'cost_update' 
  | 'certificate_added';

export interface HistoryRecord {
  history_id: string;
  asset_id: string;
  change_type: ChangeType;
  changed_by: string;
  changed_at: string; // ISO datetime
  old_value?: any;
  new_value?: any;
  description: string;
}

// ============================================================================
// Dashboard Summary Types
// ============================================================================

export interface DashboardSummary {
  total_assets: number;
  ready_count: number;
  not_ready_count: number;
  at_risk_count: number;
  ready_percentage: number;
  average_taem_score: number;
  critical_issues: number;
  status_breakdown: {
    status: AssetStatus;
    count: number;
    percentage: number;
  }[];
  readiness_by_contractor: {
    contractor: string;
    ready: number;
    not_ready: number;
    at_risk: number;
  }[];
  readiness_by_type: {
    asset_type: AssetType;
    ready: number;
    not_ready: number;
    at_risk: number;
  }[];
}

// ============================================================================
// Analytics Types
// ============================================================================

export interface AssetAnalytics {
  asset_id: string;
  trend_data: {
    date: string;
    taem_score: number;
    status: AssetStatus;
  }[];
  rule_compliance_rate: number;
  deliverable_completion_rate: number;
  cost_variance_trend: number[];
}

export interface ContractorPerformance {
  contractor: string;
  total_assets: number;
  ready_percentage: number;
  average_taem_score: number;
  on_time_deliverables: number;
  cost_variance_percentage: number;
}

// ============================================================================
// API Response Types
// ============================================================================

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface ApiError {
  detail: string;
  error_code?: string;
  timestamp?: string;
}

// ============================================================================
// Filter and Query Types
// ============================================================================

export interface AssetFilters {
  asset_type?: AssetType;
  status?: AssetStatus;
  contractor?: string;
  route_section?: string;
  search?: string;
  min_taem_score?: number;
  max_taem_score?: number;
}

export interface AssetSortOptions {
  field: 'asset_id' | 'asset_name' | 'taem_score' | 'status' | 'last_evaluated';
  direction: 'asc' | 'desc';
}

export interface PaginationOptions {
  page: number;
  page_size: number;
}

// ============================================================================
// Form Types
// ============================================================================

export interface EvaluateAssetRequest {
  asset_id: string;
  evaluated_by?: string;
  notes?: string;
}

export interface UpdateDeliverableRequest {
  status: DeliverableStatus;
  submitted_date?: string;
  notes?: string;
}

// ============================================================================
// UI State Types
// ============================================================================

export interface LoadingState {
  isLoading: boolean;
  message?: string;
}

export interface ErrorState {
  hasError: boolean;
  message: string;
  code?: string;
}

export interface TabValue {
  index: number;
  label: string;
}

// ============================================================================
// Constants
// ============================================================================

export const ASSET_TYPES: AssetType[] = [
  'Bridge',
  'Tunnel',
  'Viaduct',
  'Station',
  'Track',
  'Signaling',
  'Earthworks'
];

export const ASSET_STATUSES: AssetStatus[] = [
  'Ready',
  'Not Ready',
  'At Risk'
];

export const RULE_SEVERITIES: RuleSeverity[] = [
  'Critical',
  'Major',
  'Minor'
];

export const DELIVERABLE_STATUSES: DeliverableStatus[] = [
  'Not Started',
  'In Progress',
  'Submitted',
  'Approved',
  'Rejected'
];
