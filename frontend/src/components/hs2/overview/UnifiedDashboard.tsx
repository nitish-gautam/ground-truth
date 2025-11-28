/**
 * Unified Dashboard Component
 * ===========================
 *
 * Comprehensive customer showcase dashboard combining:
 * - Asset readiness summary (500 synthetic assets)
 * - Deliverable tracking (2,000 synthetic deliverables)
 * - Real dataset statistics
 * - TAEM compliance metrics
 * - Contractor performance comparison
 */

import React, { useState } from 'react';
import {
  Grid,
  Card,
  CardHeader,
  CardContent,
  Typography,
  Box,
  Chip,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Alert,
  Divider,
  CircularProgress,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText
} from '@mui/material';
import {
  CheckCircle,
  Warning,
  Error,
  Schedule,
  TrendingUp,
  Business,
  Assessment,
  DataUsage,
  PlayArrow,
  Close
} from '@mui/icons-material';
import { useQuery, useMutation } from '@tanstack/react-query';
import axios from 'axios';

interface UnifiedDashboardData {
  assets: {
    total: number;
    ready: number;
    not_ready: number;
    at_risk: number;
    avg_taem_score: number;
    is_synthetic: boolean;
  };
  assets_by_type: Array<{ asset_type: string; count: number }>;
  assets_by_contractor: Array<{
    contractor: string;
    count: number;
    avg_taem_score: number;
  }>;
  deliverables: {
    total: number;
    approved: number;
    pending: number;
    not_started: number;
    overdue: number;
    is_synthetic: boolean;
  };
  deliverables_by_priority: Array<{ priority: string; count: number }>;
  overdue_deliverables: Array<{
    name: string;
    due_date: string;
    days_overdue: number;
    priority: string;
    responsible_party: string;
  }>;
  datasets: {
    bim_models: { count: number; is_real: boolean };
    gis_shapefiles: { count: number; is_real: boolean };
    ecology_surveys: { shapefiles: number; csv_files: number; is_real: boolean };
    legal_injunctions: { zones: number; is_real: boolean };
    property_compensation: { shapefiles: number; is_real: boolean };
    noise_monitoring: { months: number; measurements: number; is_real: boolean };
  };
  metadata: {
    generated_at: string;
    generation_time_ms: number;
    data_sources: {
      synthetic: string[];
      real: string[];
    };
  };
}

export const UnifiedDashboard: React.FC = () => {
  const [complianceDialogOpen, setComplianceDialogOpen] = useState(false);
  const [complianceReport, setComplianceReport] = useState<any>(null);

  // Fetch unified dashboard data
  const { data, isLoading, error } = useQuery<UnifiedDashboardData>({
    queryKey: ['unified-dashboard'],
    queryFn: async () => {
      const response = await axios.get('/api/v1/hs2/dashboard/unified-summary');
      return response.data;
    },
    refetchInterval: 60000 // Refresh every minute
  });

  // TAEM Compliance Check mutation
  const complianceCheckMutation = useMutation({
    mutationFn: async () => {
      const response = await axios.post('/api/v1/hs2/dashboard/taem-compliance-check');
      return response.data;
    },
    onSuccess: (data) => {
      setComplianceReport(data);
      setComplianceDialogOpen(true);
    }
  });

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress size={60} />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error">
        Failed to load dashboard data. Please try again later.
      </Alert>
    );
  }

  if (!data) return null;

  const assetsReadyPct = (data.assets.ready / data.assets.total) * 100;
  const deliverablesApprovedPct = (data.deliverables.approved / data.deliverables.total) * 100;

  return (
    <Box sx={{ maxWidth: '1600px', margin: '0 auto', p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" fontWeight={700} gutterBottom sx={{ color: 'primary.main' }}>
          HS2 Infrastructure Intelligence Platform
        </Typography>
        <Typography variant="body1" color="text.secondary" gutterBottom>
          Unified dashboard combining real-time project data with predictive analytics
        </Typography>
        <Box display="flex" gap={2} alignItems="center" sx={{ mt: 2 }}>
          <Chip
            label="SYNTHETIC DATA"
            color="warning"
            size="small"
            sx={{ fontWeight: 600 }}
          />
          <Typography variant="caption" color="text.secondary">
            {data.assets.total} assets, {data.deliverables.total} deliverables
          </Typography>
          <Chip
            label="REAL DATA"
            color="success"
            size="small"
            sx={{ fontWeight: 600 }}
          />
          <Typography variant="caption" color="text.secondary">
            {data.datasets.bim_models.count} BIM models, {data.datasets.gis_shapefiles.count}+ GIS layers, {(data.datasets.noise_monitoring.measurements / 1000000).toFixed(1)}M+ measurements
          </Typography>
          <Box sx={{ flexGrow: 1 }} />
          <Button
            variant="contained"
            color="primary"
            startIcon={complianceCheckMutation.isPending ? <CircularProgress size={20} color="inherit" /> : <PlayArrow />}
            onClick={() => complianceCheckMutation.mutate()}
            disabled={complianceCheckMutation.isPending}
            size="large"
          >
            {complianceCheckMutation.isPending ? 'Running...' : 'TAEM Compliance Check'}
          </Button>
        </Box>
      </Box>

      {/* TAEM Compliance Check Dialog */}
      <Dialog
        open={complianceDialogOpen}
        onClose={() => setComplianceDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle sx={{ bgcolor: complianceReport?.status_color === 'success' ? 'success.main' : complianceReport?.status_color === 'warning' ? 'warning.main' : 'error.main', color: 'white' }}>
          <Box display="flex" alignItems="center" justifyContent="space-between">
            <Typography variant="h6">TAEM Compliance Check Results</Typography>
            <Chip
              label={complianceReport?.overall_status || 'UNKNOWN'}
              color={complianceReport?.status_color as any}
              sx={{ fontWeight: 700, fontSize: '1rem' }}
            />
          </Box>
        </DialogTitle>
        <DialogContent dividers>
          {complianceReport && (
            <Box>
              {/* Summary Statistics */}
              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={3}>
                  <Paper elevation={1} sx={{ p: 2, textAlign: 'center', bgcolor: 'success.50' }}>
                    <Typography variant="h3" color="success.main">{complianceReport.summary.compliant}</Typography>
                    <Typography variant="body2">Compliant</Typography>
                  </Paper>
                </Grid>
                <Grid item xs={3}>
                  <Paper elevation={1} sx={{ p: 2, textAlign: 'center', bgcolor: 'grey.200' }}>
                    <Typography variant="h3" color="text.secondary">{complianceReport.summary.non_compliant}</Typography>
                    <Typography variant="body2">Non-Compliant</Typography>
                  </Paper>
                </Grid>
                <Grid item xs={3}>
                  <Paper elevation={1} sx={{ p: 2, textAlign: 'center', bgcolor: 'error.50' }}>
                    <Typography variant="h3" color="error.main">{complianceReport.summary.at_risk}</Typography>
                    <Typography variant="body2">At Risk</Typography>
                  </Paper>
                </Grid>
                <Grid item xs={3}>
                  <Paper elevation={1} sx={{ p: 2, textAlign: 'center', bgcolor: 'info.50' }}>
                    <Typography variant="h3" color="info.main">{complianceReport.overall_compliance_rate}%</Typography>
                    <Typography variant="body2">Compliance Rate</Typography>
                  </Paper>
                </Grid>
              </Grid>

              {/* Compliance Criteria */}
              <Typography variant="h6" gutterBottom>Compliance Criteria:</Typography>
              <Box sx={{ mb: 3, p: 2, bgcolor: 'grey.100', borderRadius: 1 }}>
                <Typography variant="body2">• Minimum TAEM Score: {complianceReport.compliance_criteria.min_taem_score}</Typography>
                <Typography variant="body2">• Minimum Completion Rate: {complianceReport.compliance_criteria.min_completion_rate}%</Typography>
                <Typography variant="body2">• Maximum Overdue Deliverables: {complianceReport.compliance_criteria.max_overdue_deliverables}</Typography>
                <Typography variant="body2">• Required Readiness Status: {complianceReport.compliance_criteria.required_readiness_status}</Typography>
              </Box>

              {/* Recommendations */}
              {complianceReport.recommendations.length > 0 && (
                <>
                  <Typography variant="h6" gutterBottom>Recommendations:</Typography>
                  <List>
                    {complianceReport.recommendations.map((rec: any, index: number) => (
                      <ListItem key={index} sx={{ bgcolor: 'warning.50', mb: 1, borderRadius: 1 }}>
                        <ListItemText
                          primary={
                            <Box display="flex" alignItems="center" gap={1}>
                              <Chip label={rec.priority} color={rec.priority === 'Critical' ? 'error' : 'warning'} size="small" />
                              <Typography variant="body2" fontWeight={600}>{rec.issue}</Typography>
                            </Box>
                          }
                          secondary={
                            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                              <strong>Action:</strong> {rec.action}
                            </Typography>
                          }
                        />
                      </ListItem>
                    ))}
                  </List>
                </>
              )}

              {/* At Risk Assets */}
              {complianceReport.at_risk_assets.length > 0 && (
                <>
                  <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Top At-Risk Assets:</Typography>
                  <TableContainer component={Paper} elevation={1}>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell><strong>Asset</strong></TableCell>
                          <TableCell><strong>Contractor</strong></TableCell>
                          <TableCell align="center"><strong>TAEM Score</strong></TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {complianceReport.at_risk_assets.slice(0, 5).map((asset: any, index: number) => (
                          <TableRow key={index}>
                            <TableCell>{asset.asset_name}</TableCell>
                            <TableCell>{asset.contractor}</TableCell>
                            <TableCell align="center">
                              <Chip label={asset.taem_score} color="error" size="small" />
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </>
              )}

              {/* Metadata */}
              <Box sx={{ mt: 3, p: 2, bgcolor: 'grey.100', borderRadius: 1 }}>
                <Typography variant="caption" color="text.secondary">
                  <strong>Check Completed:</strong> {new Date(complianceReport.metadata.check_timestamp).toLocaleString()}
                  {' | '}
                  <strong>Execution Time:</strong> {complianceReport.metadata.execution_time_ms}ms
                  {' | '}
                  <Chip label="SIMULATED CHECK" size="small" color="info" sx={{ ml: 1 }} />
                </Typography>
              </Box>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setComplianceDialogOpen(false)} startIcon={<Close />}>
            Close
          </Button>
        </DialogActions>
      </Dialog>

      <Grid container spacing={3}>
        {/* ==================== ASSETS OVERVIEW ==================== */}
        <Grid item xs={12}>
          <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
            <Assessment sx={{ color: 'primary.main' }} />
            <Typography variant="h5" fontWeight={600}>Asset Readiness Summary</Typography>
            <Chip label="SYNTHETIC DATA" color="warning" size="small" sx={{ ml: 'auto' }} />
          </Box>
          <Grid container spacing={2}>
            {/* Total Assets */}
            <Grid item xs={12} md={3}>
              <Paper elevation={2} sx={{ p: 3, textAlign: 'center', bgcolor: 'primary.50', height: '100%' }}>
                <Assessment sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
                <Typography variant="h3" color="primary" fontWeight={700}>
                  {data.assets.total}
                </Typography>
                <Typography variant="body1" fontWeight={600} color="text.primary">
                  Total Assets
                </Typography>
                <Divider sx={{ my: 1 }} />
                <Typography variant="caption" color="text.secondary">
                  Avg TAEM: {data.assets.avg_taem_score}
                </Typography>
              </Paper>
            </Grid>

            {/* Ready */}
            <Grid item xs={12} md={3}>
              <Paper elevation={2} sx={{ p: 3, textAlign: 'center', bgcolor: 'success.50', height: '100%' }}>
                <CheckCircle sx={{ fontSize: 40, color: 'success.main', mb: 1 }} />
                <Typography variant="h3" color="success.main" fontWeight={700}>
                  {data.assets.ready}
                </Typography>
                <Typography variant="body1" fontWeight={600} color="text.primary">
                  Ready
                </Typography>
                <Typography variant="caption" color="text.secondary" display="block">
                  {assetsReadyPct.toFixed(1)}% of total
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={assetsReadyPct}
                  color="success"
                  sx={{ mt: 1, height: 6, borderRadius: 3 }}
                />
              </Paper>
            </Grid>

            {/* Not Ready */}
            <Grid item xs={12} md={3}>
              <Paper elevation={2} sx={{ p: 3, textAlign: 'center', bgcolor: 'error.50', height: '100%' }}>
                <Error sx={{ fontSize: 40, color: 'error.main', mb: 1 }} />
                <Typography variant="h3" color="error.main" fontWeight={700}>
                  {data.assets.not_ready}
                </Typography>
                <Typography variant="body1" fontWeight={600} color="text.primary">
                  Not Ready
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {((data.assets.not_ready / data.assets.total) * 100).toFixed(1)}% of total
                </Typography>
              </Paper>
            </Grid>

            {/* At Risk */}
            <Grid item xs={12} md={3}>
              <Paper elevation={2} sx={{ p: 3, textAlign: 'center', bgcolor: 'warning.50', height: '100%' }}>
                <Warning sx={{ fontSize: 40, color: 'warning.main', mb: 1 }} />
                <Typography variant="h3" color="warning.main" fontWeight={700}>
                  {data.assets.at_risk}
                </Typography>
                <Typography variant="body1" fontWeight={600} color="text.primary">
                  At Risk
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {((data.assets.at_risk / data.assets.total) * 100).toFixed(1)}% of total
                </Typography>
              </Paper>
            </Grid>
          </Grid>

          {/* Assets by Type */}
          <Box sx={{ mt: 4 }}>
            <Typography variant="h6" gutterBottom fontWeight={600}>
              Distribution by Asset Type
            </Typography>
            <Grid container spacing={2}>
              {data.assets_by_type.map((item) => (
                <Grid item xs={6} md={2} key={item.asset_type}>
                  <Paper elevation={1} sx={{ p: 2, textAlign: 'center', bgcolor: 'grey.50' }}>
                    <Typography variant="h5" color="primary" fontWeight={600}>
                      {item.count}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {item.asset_type}
                    </Typography>
                  </Paper>
                </Grid>
              ))}
            </Grid>
          </Box>
        </Grid>

        {/* ==================== CONTRACTOR PERFORMANCE ==================== */}
        <Grid item xs={12} md={6}>
          <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
            <Business sx={{ color: 'secondary.main' }} />
            <Typography variant="h5" fontWeight={600}>Contractor Performance</Typography>
            <Chip label="SYNTHETIC DATA" color="warning" size="small" sx={{ ml: 'auto' }} />
          </Box>
          <Paper elevation={2} sx={{ p: 2 }}>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell><strong>Contractor</strong></TableCell>
                      <TableCell align="center"><strong>Assets</strong></TableCell>
                      <TableCell align="center"><strong>Avg TAEM</strong></TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {data.assets_by_contractor.map((contractor) => (
                      <TableRow key={contractor.contractor}>
                        <TableCell>{contractor.contractor}</TableCell>
                        <TableCell align="center">
                          <Chip label={contractor.count} color="primary" size="small" />
                        </TableCell>
                        <TableCell align="center">
                          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
                            <Typography
                              variant="body2"
                              fontWeight={600}
                              color={
                                contractor.avg_taem_score >= 85
                                  ? 'success.main'
                                  : contractor.avg_taem_score >= 70
                                  ? 'warning.main'
                                  : 'error.main'
                              }
                            >
                              {contractor.avg_taem_score}
                            </Typography>
                            <LinearProgress
                              variant="determinate"
                              value={contractor.avg_taem_score}
                              color={
                                contractor.avg_taem_score >= 85
                                  ? 'success'
                                  : contractor.avg_taem_score >= 70
                                  ? 'warning'
                                  : 'error'
                              }
                              sx={{ width: 80, height: 6, borderRadius: 3 }}
                            />
                          </Box>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
          </Paper>
        </Grid>

        {/* ==================== DELIVERABLES SUMMARY ==================== */}
        <Grid item xs={12} md={6}>
          <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
            <DataUsage sx={{ color: 'success.main' }} />
            <Typography variant="h5" fontWeight={600}>Deliverables Summary</Typography>
            <Chip label="SYNTHETIC DATA" color="warning" size="small" sx={{ ml: 'auto' }} />
          </Box>
          <Paper elevation={2} sx={{ p: 2 }}>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Box sx={{ p: 2, bgcolor: 'success.50', borderRadius: 1 }}>
                    <Typography variant="h4" color="success.dark">
                      {data.deliverables.approved}
                    </Typography>
                    <Typography variant="body2" fontWeight={600}>Approved</Typography>
                    <LinearProgress
                      variant="determinate"
                      value={deliverablesApprovedPct}
                      color="success"
                      sx={{ mt: 1, height: 6, borderRadius: 3 }}
                    />
                  </Box>
                </Grid>

                <Grid item xs={6}>
                  <Box sx={{ p: 2, bgcolor: 'info.50', borderRadius: 1 }}>
                    <Typography variant="h4" color="info.dark">
                      {data.deliverables.pending}
                    </Typography>
                    <Typography variant="body2" fontWeight={600}>Pending</Typography>
                  </Box>
                </Grid>

                <Grid item xs={6}>
                  <Box sx={{ p: 2, bgcolor: 'grey.200', borderRadius: 1 }}>
                    <Typography variant="h4" color="text.secondary">
                      {data.deliverables.not_started}
                    </Typography>
                    <Typography variant="body2" fontWeight={600}>Not Started</Typography>
                  </Box>
                </Grid>

                <Grid item xs={6}>
                  <Box sx={{ p: 2, bgcolor: 'error.50', borderRadius: 1 }}>
                    <Typography variant="h4" color="error.dark">
                      {data.deliverables.overdue}
                    </Typography>
                    <Typography variant="body2" fontWeight={600}>Overdue</Typography>
                  </Box>
                </Grid>
              </Grid>

              <Divider sx={{ my: 2 }} />

              <Box sx={{ px: 1 }}>
                <Typography variant="body2" fontWeight={600} gutterBottom>
                  By Priority:
                </Typography>
                {data.deliverables_by_priority.map((item) => (
                  <Box key={item.priority} sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">{item.priority}</Typography>
                    <Chip
                      label={item.count}
                      size="small"
                      color={
                        item.priority === 'Critical'
                          ? 'error'
                          : item.priority === 'Major'
                          ? 'warning'
                          : 'default'
                      }
                    />
                  </Box>
                ))}
              </Box>
          </Paper>
        </Grid>

        {/* ==================== OVERDUE DELIVERABLES ALERT ==================== */}
        {data.deliverables.overdue > 0 && (
          <Grid item xs={12}>
            <Alert severity="error" sx={{ mb: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                <Schedule />
                <Typography variant="h6" fontWeight={600}>
                  {data.deliverables.overdue} Overdue Deliverables - Action Required!
                </Typography>
              </Box>
            </Alert>
            <Paper elevation={2}>
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell><strong>Deliverable</strong></TableCell>
                        <TableCell><strong>Days Overdue</strong></TableCell>
                        <TableCell><strong>Priority</strong></TableCell>
                        <TableCell><strong>Contractor</strong></TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {data.overdue_deliverables.slice(0, 5).map((item, index) => (
                        <TableRow key={index}>
                          <TableCell>{item.name}</TableCell>
                          <TableCell>
                            <Chip
                              label={`${item.days_overdue} days`}
                              color="error"
                              size="small"
                              icon={<Warning />}
                            />
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={item.priority}
                              size="small"
                              color={item.priority === 'Critical' ? 'error' : 'warning'}
                            />
                          </TableCell>
                          <TableCell>{item.responsible_party}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
                {data.overdue_deliverables.length > 5 && (
                  <Box sx={{ p: 2, bgcolor: 'grey.50' }}>
                    <Typography variant="caption" color="text.secondary">
                      Showing top 5 of {data.overdue_deliverables.length} overdue deliverables
                    </Typography>
                  </Box>
                )}
            </Paper>
          </Grid>
        )}

        {/* ==================== REAL DATASETS SUMMARY ==================== */}
        <Grid item xs={12}>
          <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1, mt: 2 }}>
            <TrendingUp sx={{ color: 'success.main' }} />
            <Typography variant="h5" fontWeight={600}>Available HS2 Datasets</Typography>
            <Chip label="REAL DATA" color="success" size="small" sx={{ ml: 'auto' }} />
          </Box>
          <Grid container spacing={2}>
            <Grid item xs={6} md={2}>
              <Paper elevation={2} sx={{ p: 3, textAlign: 'center', bgcolor: 'primary.50' }}>
                <Typography variant="h4" color="primary" fontWeight={600}>
                  {data.datasets.bim_models.count}
                </Typography>
                <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
                  BIM Models
                </Typography>
                <Chip label="IFC 4.3.x" size="small" />
              </Paper>
            </Grid>

            <Grid item xs={6} md={2}>
              <Paper elevation={2} sx={{ p: 3, textAlign: 'center', bgcolor: 'secondary.50' }}>
                <Typography variant="h4" color="secondary" fontWeight={600}>
                  {data.datasets.gis_shapefiles.count}+
                </Typography>
                <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
                  GIS Shapefiles
                </Typography>
                <Chip label="Multi-layer" size="small" color="secondary" />
              </Paper>
            </Grid>

            <Grid item xs={6} md={2}>
              <Paper elevation={2} sx={{ p: 3, textAlign: 'center', bgcolor: 'success.50' }}>
                <Typography variant="h4" color="success.main" fontWeight={600}>
                  {data.datasets.ecology_surveys.csv_files + data.datasets.ecology_surveys.shapefiles}
                </Typography>
                <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
                  Ecology Files
                </Typography>
                <Chip label="Nov 2024" size="small" color="success" />
              </Paper>
            </Grid>

            <Grid item xs={6} md={2}>
              <Paper elevation={2} sx={{ p: 3, textAlign: 'center', bgcolor: 'error.50' }}>
                <Typography variant="h4" color="error.main" fontWeight={600}>
                  {(data.datasets.legal_injunctions.zones / 1000).toFixed(1)}K
                </Typography>
                <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
                  Legal Zones
                </Typography>
                <Chip label="GeoJSON" size="small" color="error" />
              </Paper>
            </Grid>

            <Grid item xs={6} md={2}>
              <Paper elevation={2} sx={{ p: 3, textAlign: 'center', bgcolor: 'warning.50' }}>
                <Typography variant="h4" color="warning.main" fontWeight={600}>
                  {data.datasets.property_compensation.shapefiles}
                </Typography>
                <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
                  Property Data
                </Typography>
                <Chip label="Historical" size="small" color="warning" />
              </Paper>
            </Grid>

            <Grid item xs={6} md={2}>
              <Paper elevation={2} sx={{ p: 3, textAlign: 'center', bgcolor: 'info.50' }}>
                <Typography variant="h4" color="info.main" fontWeight={600}>
                  {(data.datasets.noise_monitoring.measurements / 1000000).toFixed(1)}M
                </Typography>
                <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
                  Noise Data
                </Typography>
                <Chip
                  label={`${data.datasets.noise_monitoring.months} months`}
                  size="small"
                  color="info"
                />
              </Paper>
            </Grid>
          </Grid>
        </Grid>

        {/* ==================== METADATA FOOTER ==================== */}
        <Grid item xs={12}>
          <Paper elevation={0} sx={{ p: 2, bgcolor: 'grey.100' }}>
            <Typography variant="caption" color="text.secondary">
              <strong>Dashboard Generated:</strong> {new Date(data.metadata.generated_at).toLocaleString()}
              {' | '}
              <strong>Generation Time:</strong> {data.metadata.generation_time_ms.toFixed(2)}ms
              {' | '}
              <strong>Data Sources:</strong> Synthetic ({data.metadata.data_sources.synthetic.join(', ')}) +
              Real ({data.metadata.data_sources.real.join(', ')})
            </Typography>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default UnifiedDashboard;
