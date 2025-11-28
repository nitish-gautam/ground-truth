/**
 * HS2 Assurance Intelligence Platform - Formatting Utilities
 * 
 * Provides consistent formatting for dates, currency, numbers, and status displays.
 */

import { AssetStatus, RuleSeverity, DeliverableStatus, CertificateStatus } from '../types/hs2Types';

// ============================================================================
// Date Formatting
// ============================================================================

/**
 * Format ISO datetime string to human-readable format
 * @example "2024-11-25T14:30:00Z" -> "Nov 25, 2024 2:30 PM"
 */
export const formatDateTime = (isoString: string): string => {
  try {
    const date = new Date(isoString);
    return new Intl.DateTimeFormat('en-GB', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }).format(date);
  } catch {
    return 'Invalid date';
  }
};

/**
 * Format ISO date string to short date format
 * @example "2024-11-25" -> "Nov 25, 2024"
 */
export const formatDate = (isoString: string): string => {
  try {
    const date = new Date(isoString);
    return new Intl.DateTimeFormat('en-GB', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    }).format(date);
  } catch {
    return 'Invalid date';
  }
};

/**
 * Format relative time from now
 * @example "2024-11-20" -> "5 days ago"
 */
export const formatRelativeTime = (isoString: string): string => {
  try {
    const date = new Date(isoString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;
    if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;
    if (diffDays < 365) return `${Math.floor(diffDays / 30)} months ago`;
    return `${Math.floor(diffDays / 365)} years ago`;
  } catch {
    return 'Unknown';
  }
};

/**
 * Check if date is within X days from now
 * @param isoString ISO date string
 * @param days Number of days threshold
 */
export const isWithinDays = (isoString: string, days: number): boolean => {
  try {
    const date = new Date(isoString);
    const now = new Date();
    const diffMs = date.getTime() - now.getTime();
    const diffDays = Math.ceil(diffMs / (1000 * 60 * 60 * 24));
    return diffDays >= 0 && diffDays <= days;
  } catch {
    return false;
  }
};

/**
 * Check if date has passed
 */
export const isPastDate = (isoString: string): boolean => {
  try {
    const date = new Date(isoString);
    const now = new Date();
    return date < now;
  } catch {
    return false;
  }
};

// ============================================================================
// Currency Formatting
// ============================================================================

/**
 * Format number as currency
 * @example 1234567.89 -> "£1,234,567.89"
 */
export const formatCurrency = (
  amount: number,
  currency: string = 'GBP'
): string => {
  try {
    return new Intl.NumberFormat('en-GB', {
      style: 'currency',
      currency: currency,
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(amount);
  } catch {
    return `£${amount.toFixed(2)}`;
  }
};

/**
 * Format variance with +/- sign and color indication
 * @example -15000 -> "-£15,000.00" (negative)
 */
export const formatVariance = (
  variance: number,
  currency: string = 'GBP'
): { text: string; isPositive: boolean } => {
  const formatted = formatCurrency(Math.abs(variance), currency);
  return {
    text: variance >= 0 ? `+${formatted}` : `-${formatted}`,
    isPositive: variance >= 0
  };
};

// ============================================================================
// Number Formatting
// ============================================================================

/**
 * Format percentage
 * @example 0.8567 -> "85.67%"
 */
export const formatPercentage = (value: number): string => {
  return `${(value * 100).toFixed(2)}%`;
};

/**
 * Format TAEM score with color coding
 * @param score TAEM score (0-100)
 */
export const formatTAEMScore = (score: number): { 
  text: string; 
  color: 'success' | 'warning' | 'error' 
} => {
  const rounded = score.toFixed(1);
  let color: 'success' | 'warning' | 'error';
  
  if (score >= 80) color = 'success';
  else if (score >= 60) color = 'warning';
  else color = 'error';

  return { text: rounded, color };
};

/**
 * Format large numbers with K/M suffixes
 * @example 1500 -> "1.5K", 2500000 -> "2.5M"
 */
export const formatCompactNumber = (value: number): string => {
  if (value >= 1000000) {
    return `${(value / 1000000).toFixed(1)}M`;
  }
  if (value >= 1000) {
    return `${(value / 1000).toFixed(1)}K`;
  }
  return value.toString();
};

// ============================================================================
// Status Formatting
// ============================================================================

/**
 * Get color for asset status
 */
export const getAssetStatusColor = (status: AssetStatus): string => {
  const colorMap: Record<AssetStatus, string> = {
    'Ready': '#4caf50',
    'Not Ready': '#f44336',
    'At Risk': '#ff9800'
  };
  return colorMap[status] || '#757575';
};

/**
 * Get MUI color variant for asset status
 */
export const getAssetStatusVariant = (
  status: AssetStatus
): 'success' | 'error' | 'warning' => {
  const variantMap: Record<AssetStatus, 'success' | 'error' | 'warning'> = {
    'Ready': 'success',
    'Not Ready': 'error',
    'At Risk': 'warning'
  };
  return variantMap[status] || 'error';
};

/**
 * Get color for rule severity
 */
export const getSeverityColor = (severity: RuleSeverity): string => {
  const colorMap: Record<RuleSeverity, string> = {
    'Critical': '#d32f2f',
    'Major': '#f57c00',
    'Minor': '#fbc02d'
  };
  return colorMap[severity] || '#757575';
};

/**
 * Get MUI color variant for severity
 */
export const getSeverityVariant = (
  severity: RuleSeverity
): 'error' | 'warning' | 'info' => {
  const variantMap: Record<RuleSeverity, 'error' | 'warning' | 'info'> = {
    'Critical': 'error',
    'Major': 'warning',
    'Minor': 'info'
  };
  return variantMap[severity] || 'info';
};

/**
 * Get color for deliverable status
 */
export const getDeliverableStatusColor = (status: DeliverableStatus): string => {
  const colorMap: Record<DeliverableStatus, string> = {
    'Not Started': '#9e9e9e',
    'In Progress': '#2196f3',
    'Submitted': '#ff9800',
    'Approved': '#4caf50',
    'Rejected': '#f44336'
  };
  return colorMap[status] || '#757575';
};

/**
 * Get color for certificate status
 */
export const getCertificateStatusColor = (status: CertificateStatus): string => {
  const colorMap: Record<CertificateStatus, string> = {
    'Valid': '#4caf50',
    'Expired': '#f44336',
    'Expiring Soon': '#ff9800'
  };
  return colorMap[status] || '#757575';
};

// ============================================================================
// Text Formatting
// ============================================================================

/**
 * Truncate text to specified length
 */
export const truncateText = (text: string, maxLength: number): string => {
  if (text.length <= maxLength) return text;
  return `${text.substring(0, maxLength - 3)}...`;
};

/**
 * Convert snake_case to Title Case
 * @example "asset_type" -> "Asset Type"
 */
export const toTitleCase = (text: string): string => {
  return text
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

/**
 * Pluralize word based on count
 */
export const pluralize = (count: number, singular: string, plural?: string): string => {
  if (count === 1) return singular;
  return plural || `${singular}s`;
};

// ============================================================================
// JSON Formatting
// ============================================================================

/**
 * Pretty print JSON with syntax highlighting class hints
 */
export const formatJSON = (obj: any): string => {
  try {
    return JSON.stringify(obj, null, 2);
  } catch {
    return 'Invalid JSON';
  }
};

/**
 * Safe parse JSON with fallback
 */
export const safeParseJSON = <T = any>(jsonString: string, fallback: T): T => {
  try {
    return JSON.parse(jsonString);
  } catch {
    return fallback;
  }
};

// ============================================================================
// Validation Helpers
// ============================================================================

/**
 * Check if value is empty (null, undefined, empty string)
 */
export const isEmpty = (value: any): boolean => {
  return value === null || value === undefined || value === '';
};

/**
 * Get display value or fallback
 */
export const getDisplayValue = (value: any, fallback: string = 'N/A'): string => {
  return isEmpty(value) ? fallback : String(value);
};
