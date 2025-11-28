/**
 * HS2 Dashboard Page
 * Main dashboard showing summary metrics and charts
 */

import React from 'react';
import { Box, Container, Typography, Paper, Grid, Alert, Card, CardContent, CardActionArea } from '@mui/material';
import { useQuery } from '@tanstack/react-query';
import { BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { useNavigate } from 'react-router-dom';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import BarChartIcon from '@mui/icons-material/BarChart';
import hs2Client from '../api/hs2Client';
import HS2SummaryCards from '../components/HS2SummaryCards';

const COLORS = {
  Ready: '#4caf50',
  'Not Ready': '#f44336',
  'At Risk': '#ff9800'
};

const HS2Dashboard: React.FC = () => {
  const navigate = useNavigate();

  // Fetch dashboard summary with auto-refresh every 30 seconds
  const {
    data: summary,
    isLoading,
    error,
    refetch
  } = useQuery({
    queryKey: ['hs2-dashboard-summary'],
    queryFn: () => hs2Client.dashboard.getSummary(),
    refetchInterval: 30000 // Auto-refresh every 30s
  });

  if (error) {
    return (
      <Container maxWidth="xl" sx={{ mt: 4 }}>
        <Alert severity="error">
          Failed to load dashboard data. Please try again.
        </Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      {/* Page Header */}
      <Box mb={4}>
        <Typography variant="h4" gutterBottom>
          HS2 Assurance Dashboard
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Real-time overview of asset readiness and compliance
        </Typography>
      </Box>

      {/* Summary Cards */}
      <Box mb={4}>
        <HS2SummaryCards summary={summary} isLoading={isLoading} />
      </Box>

      {/* Quick Access Cards */}
      <Grid container spacing={3} mb={4}>
        <Grid item xs={12} sm={6}>
          <Card>
            <CardActionArea onClick={() => navigate('/hs2/monitoring')}>
              <CardContent>
                <Box display="flex" alignItems="center" mb={2}>
                  <BarChartIcon sx={{ fontSize: 40, color: 'primary.main', mr: 2 }} />
                  <Typography variant="h5" component="div">
                    Noise Monitoring
                  </Typography>
                </Box>
                <Typography variant="body2" color="text.secondary">
                  View real-time noise monitoring data from 170 Excel files across 10 months.
                  Analyze compliance violations and geographic distribution.
                </Typography>
              </CardContent>
            </CardActionArea>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6}>
          <Card>
            <CardActionArea onClick={() => navigate('/hs2/upload')}>
              <CardContent>
                <Box display="flex" alignItems="center" mb={2}>
                  <UploadFileIcon sx={{ fontSize: 40, color: 'primary.main', mr: 2 }} />
                  <Typography variant="h5" component="div">
                    Upload Files
                  </Typography>
                </Box>
                <Typography variant="body2" color="text.secondary">
                  Upload GPR surveys, BIM models, LiDAR scans, monitoring data, and reports.
                  Track processing status and view analytics.
                </Typography>
              </CardContent>
            </CardActionArea>
          </Card>
        </Grid>
      </Grid>

      {/* Charts Grid */}
      <Grid container spacing={3}>
        {/* Readiness by Contractor */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Readiness by Contractor
            </Typography>
            {summary && summary.readiness_by_contractor.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={summary.readiness_by_contractor}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="contractor" angle={-45} textAnchor="end" height={100} />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="ready" fill={COLORS.Ready} name="Ready" />
                  <Bar dataKey="not_ready" fill={COLORS['Not Ready']} name="Not Ready" />
                  <Bar dataKey="at_risk" fill={COLORS['At Risk']} name="At Risk" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <Box height={300} display="flex" alignItems="center" justifyContent="center">
                <Typography color="text.secondary">No contractor data available</Typography>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Readiness by Asset Type */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Readiness by Asset Type
            </Typography>
            {summary && summary.readiness_by_type.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={summary.readiness_by_type}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="asset_type" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="ready" fill={COLORS.Ready} name="Ready" />
                  <Bar dataKey="not_ready" fill={COLORS['Not Ready']} name="Not Ready" />
                  <Bar dataKey="at_risk" fill={COLORS['At Risk']} name="At Risk" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <Box height={300} display="flex" alignItems="center" justifyContent="center">
                <Typography color="text.secondary">No asset type data available</Typography>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Status Distribution Pie Chart */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Overall Status Distribution
            </Typography>
            {summary && summary.status_breakdown.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={summary.status_breakdown}
                    dataKey="count"
                    nameKey="status"
                    cx="50%"
                    cy="50%"
                    outerRadius={100}
                    label={({ status, percentage }) => `${status}: ${percentage.toFixed(1)}%`}
                  >
                    {summary.status_breakdown.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[entry.status]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <Box height={300} display="flex" alignItems="center" justifyContent="center">
                <Typography color="text.secondary">No status data available</Typography>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Key Metrics */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Key Metrics
            </Typography>
            {summary ? (
              <Box>
                <Box display="flex" justifyContent="space-between" mb={2}>
                  <Typography color="text.secondary">Ready Percentage</Typography>
                  <Typography variant="h6" color={summary.ready_percentage >= 80 ? 'success.main' : 'error.main'}>
                    {summary.ready_percentage.toFixed(1)}%
                  </Typography>
                </Box>
                <Box display="flex" justifyContent="space-between" mb={2}>
                  <Typography color="text.secondary">Average TAEM Score</Typography>
                  <Typography variant="h6">
                    {summary.average_taem_score.toFixed(1)}
                  </Typography>
                </Box>
                <Box display="flex" justifyContent="space-between" mb={2}>
                  <Typography color="text.secondary">Critical Issues</Typography>
                  <Typography variant="h6" color={summary.critical_issues > 0 ? 'error.main' : 'success.main'}>
                    {summary.critical_issues}
                  </Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography color="text.secondary">Total Assets</Typography>
                  <Typography variant="h6">
                    {summary.total_assets}
                  </Typography>
                </Box>
              </Box>
            ) : (
              <Box height={300} display="flex" alignItems="center" justifyContent="center">
                <Typography color="text.secondary">Loading metrics...</Typography>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default HS2Dashboard;
