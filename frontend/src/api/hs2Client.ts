/**
 * HS2 Assurance Intelligence Platform - API Client
 * 
 * Axios-based API client with React Query integration for type-safe backend communication.
 */

import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';
import {
  HS2Asset,
  ReadinessReport,
  Deliverable,
  CostRecord,
  CostSummary,
  Certificate,
  HistoryRecord,
  DashboardSummary,
  AssetAnalytics,
  ContractorPerformance,
  PaginatedResponse,
  AssetFilters,
  EvaluateAssetRequest
} from '../types/hs2Types';

// ============================================================================
// API Configuration
// ============================================================================

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8007';
const API_TIMEOUT = Number(import.meta.env.VITE_API_TIMEOUT) || 30000;

/**
 * Axios instance with default configuration
 */
const apiClient: AxiosInstance = axios.create({
  baseURL: `${API_BASE_URL}/api/v1/hs2`,
  timeout: API_TIMEOUT,
  headers: {
    'Content-Type': 'application/json'
  }
});

// ============================================================================
// Request/Response Interceptors
// ============================================================================

/**
 * Request interceptor: Add auth tokens, logging, etc.
 */
apiClient.interceptors.request.use(
  (config) => {
    // Add timestamp for request tracking
    config.metadata = { startTime: new Date().getTime() };
    
    // TODO: Add auth token when authentication is implemented
    // const token = getAuthToken();
    // if (token) {
    //   config.headers.Authorization = `Bearer ${token}`;
    // }
    
    console.log(`[API Request] ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('[API Request Error]', error);
    return Promise.reject(error);
  }
);

/**
 * Response interceptor: Handle errors, logging, etc.
 */
apiClient.interceptors.response.use(
  (response) => {
    const duration = new Date().getTime() - (response.config.metadata?.startTime || 0);
    console.log(
      `[API Response] ${response.config.method?.toUpperCase()} ${response.config.url} - ${duration}ms`
    );
    return response;
  },
  (error) => {
    if (error.response) {
      // Server responded with error status
      console.error(
        `[API Error] ${error.response.status} - ${error.response.data?.detail || 'Unknown error'}`
      );
    } else if (error.request) {
      // Request made but no response received
      console.error('[API Error] No response from server');
    } else {
      // Error in request setup
      console.error('[API Error]', error.message);
    }
    return Promise.reject(error);
  }
);

// ============================================================================
// Dashboard API
// ============================================================================

export const dashboardAPI = {
  /**
   * Get dashboard summary statistics
   */
  getSummary: async (): Promise<DashboardSummary> => {
    const response = await apiClient.get('/dashboard/summary');
    const data = response.data;

    // Transform API response to match DashboardSummary interface
    return {
      total_assets: data.total_assets,
      ready_count: data.ready,
      not_ready_count: data.not_ready,
      at_risk_count: data.at_risk,
      ready_percentage: data.ready_pct,
      average_taem_score: data.avg_taem_score,
      critical_issues: 0, // TODO: Add to backend response
      status_breakdown: [
        {
          status: 'Ready' as const,
          count: data.ready,
          percentage: data.ready_pct
        },
        {
          status: 'Not Ready' as const,
          count: data.not_ready,
          percentage: data.not_ready_pct
        },
        {
          status: 'At Risk' as const,
          count: data.at_risk,
          percentage: data.at_risk_pct
        }
      ],
      readiness_by_contractor: data.by_contractor || [],
      readiness_by_type: data.by_asset_type || []
    };
  },

  /**
   * Get readiness breakdown by contractor
   */
  getReadinessByContractor: async () => {
    const response = await apiClient.get('/dashboard/readiness-by-contractor');
    return response.data;
  },

  /**
   * Get readiness breakdown by asset type
   */
  getReadinessByType: async () => {
    const response = await apiClient.get('/dashboard/readiness-by-type');
    return response.data;
  }
};

// ============================================================================
// Assets API
// ============================================================================

export const assetsAPI = {
  /**
   * Get list of assets with optional filters
   */
  getAssets: async (
    filters?: AssetFilters,
    page: number = 1,
    pageSize: number = 25
  ): Promise<PaginatedResponse<HS2Asset>> => {
    const params = new URLSearchParams();
    
    if (filters?.asset_type) params.append('asset_type', filters.asset_type);
    if (filters?.status) params.append('status', filters.status);
    if (filters?.contractor) params.append('contractor', filters.contractor);
    if (filters?.route_section) params.append('route_section', filters.route_section);
    if (filters?.search) params.append('search', filters.search);
    if (filters?.min_taem_score !== undefined) params.append('min_taem_score', String(filters.min_taem_score));
    if (filters?.max_taem_score !== undefined) params.append('max_taem_score', String(filters.max_taem_score));
    
    params.append('page', String(page));
    params.append('page_size', String(pageSize));

    const response = await apiClient.get<PaginatedResponse<HS2Asset>>(
      `/assets?${params.toString()}`
    );
    return response.data;
  },

  /**
   * Get single asset by ID
   */
  getAsset: async (assetId: string): Promise<HS2Asset> => {
    const response = await apiClient.get<HS2Asset>(`/assets/${assetId}`);
    return response.data;
  },

  /**
   * Trigger asset evaluation
   */
  evaluateAsset: async (assetId: string, request?: EvaluateAssetRequest): Promise<ReadinessReport> => {
    const response = await apiClient.post<ReadinessReport>(
      `/assets/${assetId}/evaluate`,
      request || {}
    );
    return response.data;
  },

  /**
   * Get unique contractors (for filter dropdown)
   */
  getContractors: async (): Promise<string[]> => {
    const response = await apiClient.get<string[]>('/assets/contractors');
    return response.data;
  },

  /**
   * Get unique route sections (for filter dropdown)
   */
  getRouteSections: async (): Promise<string[]> => {
    const response = await apiClient.get<string[]>('/assets/route-sections');
    return response.data;
  }
};

// ============================================================================
// Readiness API
// ============================================================================

export const readinessAPI = {
  /**
   * Get readiness report for an asset
   */
  getReport: async (assetId: string): Promise<ReadinessReport> => {
    const response = await apiClient.get<ReadinessReport>(`/readiness/${assetId}`);
    return response.data;
  },

  /**
   * Get explainability for "Why Not Ready?"
   */
  getExplain: async (assetId: string) => {
    const response = await apiClient.get(`/readiness/${assetId}/explain`);
    return response.data;
  }
};

// ============================================================================
// Deliverables API
// ============================================================================

export const deliverablesAPI = {
  /**
   * Get deliverables for an asset
   */
  getByAsset: async (assetId: string): Promise<Deliverable[]> => {
    const response = await apiClient.get<Deliverable[]>(`/deliverables?asset_id=${assetId}`);
    return response.data;
  },

  /**
   * Get single deliverable by ID
   */
  getDeliverable: async (deliverableId: string): Promise<Deliverable> => {
    const response = await apiClient.get<Deliverable>(`/deliverables/${deliverableId}`);
    return response.data;
  }
};

// ============================================================================
// Costs API
// ============================================================================

export const costsAPI = {
  /**
   * Get cost records for an asset
   */
  getByAsset: async (assetId: string): Promise<CostRecord[]> => {
    const response = await apiClient.get<CostRecord[]>(`/costs?asset_id=${assetId}`);
    return response.data;
  },

  /**
   * Get cost summary for an asset
   */
  getSummary: async (assetId: string): Promise<CostSummary> => {
    const response = await apiClient.get<CostSummary>(`/costs/summary?asset_id=${assetId}`);
    return response.data;
  }
};

// ============================================================================
// Certificates API
// ============================================================================

export const certificatesAPI = {
  /**
   * Get certificates for an asset
   */
  getByAsset: async (assetId: string): Promise<Certificate[]> => {
    const response = await apiClient.get<Certificate[]>(`/certificates?asset_id=${assetId}`);
    return response.data;
  },

  /**
   * Get expiring certificates across all assets
   */
  getExpiring: async (days: number = 30): Promise<Certificate[]> => {
    const response = await apiClient.get<Certificate[]>(`/certificates/expiring?days=${days}`);
    return response.data;
  }
};

// ============================================================================
// History API
// ============================================================================

export const historyAPI = {
  /**
   * Get history for an asset
   */
  getByAsset: async (assetId: string, limit: number = 50): Promise<HistoryRecord[]> => {
    const response = await apiClient.get<HistoryRecord[]>(
      `/history?asset_id=${assetId}&limit=${limit}`
    );
    return response.data;
  }
};

// ============================================================================
// Analytics API
// ============================================================================

export const analyticsAPI = {
  /**
   * Get analytics for an asset
   */
  getAssetAnalytics: async (assetId: string): Promise<AssetAnalytics> => {
    const response = await apiClient.get<AssetAnalytics>(`/analytics/asset/${assetId}`);
    return response.data;
  },

  /**
   * Get contractor performance metrics
   */
  getContractorPerformance: async (): Promise<ContractorPerformance[]> => {
    const response = await apiClient.get<ContractorPerformance[]>('/analytics/contractor-performance');
    return response.data;
  }
};

// ============================================================================
// Unified Export
// ============================================================================

const hs2Client = {
  dashboard: dashboardAPI,
  assets: assetsAPI,
  readiness: readinessAPI,
  deliverables: deliverablesAPI,
  costs: costsAPI,
  certificates: certificatesAPI,
  history: historyAPI,
  analytics: analyticsAPI
};

export default hs2Client;
