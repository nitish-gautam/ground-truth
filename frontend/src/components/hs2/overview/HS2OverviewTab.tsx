/**
 * HS2 Overview & Assurance Tab
 * Structured view of project overview, pain points, and proposed features
 */

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Grid,
  Card,
  CardHeader,
  CardContent,
  Typography,
  Box,
  Chip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Alert,
  Button,
  CardActions,
  Tabs,
  Tab,
  Paper
} from '@mui/material';
import {
  CheckCircle,
  Warning,
  TrendingUp,
  Assessment,
  Storage,
  Speed,
  VolumeUp,
  ArrowForward,
  Dashboard as DashboardIcon,
  Info
} from '@mui/icons-material';
import HS2SummaryCards from '../../HS2SummaryCards';
import UnifiedDashboard from './UnifiedDashboard';
import { useQuery } from '@tanstack/react-query';
import hs2Client from '../../../api/hs2Client';
import axios from 'axios';

export const HS2OverviewTab: React.FC = () => {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState(0);

  // Fetch dashboard summary
  const { data: summary, isLoading } = useQuery({
    queryKey: ['hs2-dashboard-summary'],
    queryFn: () => hs2Client.dashboard.getSummary(),
    refetchInterval: 30000
  });

  // Fetch real dataset statistics
  const { data: datasetStats, isLoading: isLoadingStats } = useQuery({
    queryKey: ['hs2-dataset-stats'],
    queryFn: async () => {
      const response = await axios.get('/api/v1/hs2/dashboard/dataset-stats');
      return response.data;
    },
    refetchInterval: 60000
  });

  return (
    <Box>
      {/* Tabs Navigation */}
      <Paper elevation={2} sx={{ mb: 3 }}>
        <Tabs
          value={activeTab}
          onChange={(_, newValue) => setActiveTab(newValue)}
          indicatorColor="primary"
          textColor="primary"
          variant="fullWidth"
        >
          <Tab
            icon={<Info />}
            label="Project Overview"
            iconPosition="start"
          />
          <Tab
            icon={<DashboardIcon />}
            label="Unified Dashboard (Customer Showcase)"
            iconPosition="start"
          />
        </Tabs>
      </Paper>

      {/* Tab Content */}
      {activeTab === 1 ? (
        // Unified Dashboard (New!)
        <UnifiedDashboard />
      ) : (
        // Original Overview Content
        <Box>
          {/* Summary Cards */}
          <Box sx={{ mb: 4 }}>
            <HS2SummaryCards summary={summary} isLoading={isLoading} />
          </Box>

      {/* Quick Access to Features */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h5" fontWeight={600} gutterBottom sx={{ mb: 2 }}>
          Quick Access
        </Typography>
        <Grid container spacing={3}>
          {/* Noise Monitoring Card */}
          <Grid item xs={12} md={4}>
            <Card elevation={2} sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              <CardContent sx={{ flexGrow: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <VolumeUp sx={{ fontSize: 36, color: 'success.main', mr: 1.5 }} />
                  <Typography variant="h6" fontWeight={600}>Noise Monitoring</Typography>
                </Box>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  {datasetStats?.noise_monitoring?.measurements ?
                    `${(datasetStats.noise_monitoring.measurements / 1000000).toFixed(1)}M+ measurements • ${datasetStats.noise_monitoring.months} months` :
                    'Real-time monitoring data'
                  }
                </Typography>
                <Chip label="Live Data" color="success" size="small" />
              </CardContent>
              <CardActions sx={{ p: 2, pt: 0 }}>
                <Button
                  variant="contained"
                  color="success"
                  endIcon={<ArrowForward />}
                  onClick={() => navigate('/hs2/monitoring')}
                  fullWidth
                  aria-label="Open noise monitoring dashboard"
                >
                  Open Dashboard
                </Button>
              </CardActions>
            </Card>
          </Grid>

          {/* GIS Map Card */}
          <Grid item xs={12} md={4}>
            <Card elevation={2} sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              <CardContent sx={{ flexGrow: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Storage sx={{ fontSize: 36, color: 'secondary.main', mr: 1.5 }} />
                  <Typography variant="h6" fontWeight={600}>GIS Route Map</Typography>
                </Box>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  Interactive map • {datasetStats?.gis_shapefiles?.count || 94} layers • Ecology & legal data
                </Typography>
                <Chip label="Live Map" color="secondary" size="small" />
              </CardContent>
              <CardActions sx={{ p: 2, pt: 0 }}>
                <Button
                  variant="contained"
                  color="secondary"
                  endIcon={<ArrowForward />}
                  onClick={() => {
                    const event = new CustomEvent('navigate-to-tab', { detail: { tab: 1 } });
                    window.dispatchEvent(event);
                  }}
                  fullWidth
                  aria-label="View GIS route map"
                >
                  View Map
                </Button>
              </CardActions>
            </Card>
          </Grid>

          {/* BIM Viewer Card */}
          <Grid item xs={12} md={4}>
            <Card elevation={2} sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              <CardContent sx={{ flexGrow: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Assessment sx={{ fontSize: 36, color: 'info.main', mr: 1.5 }} />
                  <Typography variant="h6" fontWeight={600}>BIM Viewer</Typography>
                </Box>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  3D visualization • {datasetStats?.bim_models?.count || 45} IFC models • Three.js renderer
                </Typography>
                <Chip label="3D Viewer" color="info" size="small" />
              </CardContent>
              <CardActions sx={{ p: 2, pt: 0 }}>
                <Button
                  variant="contained"
                  color="info"
                  endIcon={<ArrowForward />}
                  onClick={() => {
                    const event = new CustomEvent('navigate-to-tab', { detail: { tab: 2 } });
                    window.dispatchEvent(event);
                  }}
                  fullWidth
                  aria-label="View 3D BIM models"
                >
                  View 3D Models
                </Button>
              </CardActions>
            </Card>
          </Grid>
        </Grid>
      </Box>

      <Grid container spacing={3}>
        {/* Key Facts / Problem Summary */}
        <Grid item xs={12} md={6}>
          <Card elevation={2}>
            <CardHeader
              title="Key Facts & Project Overview"
              sx={{
                bgcolor: 'primary.main',
                '& .MuiCardHeader-title': {
                  color: 'primary.contrastText',
                  fontWeight: 600
                }
              }}
            />
            <CardContent>
              <List>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircle color="success" />
                  </ListItemIcon>
                  <ListItemText
                    primary="High Speed Two (HS2)"
                    secondary="225-mile high-speed railway connecting London, Birmingham, Manchester, and Leeds"
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <TrendingUp color="primary" />
                  </ListItemIcon>
                  <ListItemText
                    primary="£96 Billion Project"
                    secondary="Largest infrastructure project in Europe"
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <Assessment color="secondary" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Phase 1: London - Birmingham"
                    secondary="Opening planned for 2033"
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Current Data & Pain Points */}
        <Grid item xs={12} md={6}>
          <Card elevation={2}>
            <CardHeader
              title="Current Data & Pain Points"
              avatar={<Warning color="warning" />}
              sx={{
                bgcolor: 'primary.main',
                '& .MuiCardHeader-title': {
                  color: 'primary.contrastText',
                  fontWeight: 600
                },
                '& .MuiCardHeader-avatar': {
                  color: 'warning.main'
                }
              }}
            />
            <CardContent>
              <List>
                <ListItem>
                  <ListItemIcon>
                    <Warning color="error" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Manual Report Generation"
                    secondary="8 hours to generate PAS 128 compliance reports manually"
                  />
                </ListItem>
                <Divider component="li" />
                <ListItem>
                  <ListItemIcon>
                    <Warning color="error" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Data Fragmentation"
                    secondary="Multiple data sources: BIM, GIS, monitoring data in silos"
                  />
                </ListItem>
                <Divider component="li" />
                <ListItem>
                  <ListItemIcon>
                    <Warning color="error" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Reactive Compliance"
                    secondary="No proactive risk detection or predictive analytics"
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Proposed Assurance Intelligence Features */}
        <Grid item xs={12}>
          <Card elevation={2}>
            <CardHeader
              title="Proposed Assurance Intelligence Features"
              avatar={<Speed color="success" />}
              sx={{
                bgcolor: 'primary.main',
                '& .MuiCardHeader-title': {
                  color: 'primary.contrastText',
                  fontWeight: 600
                },
                '& .MuiCardHeader-avatar': {
                  color: 'success.main'
                }
              }}
            />
            <CardContent>
              <Grid container spacing={2}>
                <Grid item xs={12} md={4}>
                  <Box sx={{ p: 2, border: 1, borderColor: 'divider', borderRadius: 2 }}>
                    <Typography variant="h6" color="primary" gutterBottom>
                      AI-Powered Compliance
                    </Typography>
                    <Typography variant="body2" paragraph>
                      Auto-generate PAS 128 compliant reports in <strong>&lt;10 minutes</strong> vs 8 hours manual work
                    </Typography>
                    <Chip label="95% Time Reduction" color="success" size="small" />
                  </Box>
                </Grid>

                <Grid item xs={12} md={4}>
                  <Box sx={{ p: 2, border: 1, borderColor: 'divider', borderRadius: 2 }}>
                    <Typography variant="h6" color="primary" gutterBottom>
                      Multi-Modal RAG Pipeline
                    </Typography>
                    <Typography variant="body2" paragraph>
                      Process GPR, BIM, LiDAR, and utility records with intelligent context retrieval
                    </Typography>
                    <Chip label="&gt;95% Accuracy" color="success" size="small" />
                  </Box>
                </Grid>

                <Grid item xs={12} md={4}>
                  <Box sx={{ p: 2, border: 1, borderColor: 'divider', borderRadius: 2 }}>
                    <Typography variant="h6" color="primary" gutterBottom>
                      Predictive Risk Scoring
                    </Typography>
                    <Typography variant="body2" paragraph>
                      Predict utility strike probability with 90% AUC-ROC target
                    </Typography>
                    <Chip label="60% Strike Reduction" color="success" size="small" />
                  </Box>
                </Grid>
              </Grid>

              <Alert severity="info" sx={{ mt: 3 }}>
                <Typography variant="body2">
                  <strong>Platform Goal:</strong> Reduce utility strikes by 60% and generate compliant reports in 10 minutes vs 8 hours manual work
                </Typography>
              </Alert>
            </CardContent>
          </Card>
        </Grid>

        {/* Datasets & Sample Data */}
        <Grid item xs={12}>
          <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
            <Storage sx={{ color: 'success.main' }} />
            <Typography variant="h5" fontWeight={600}>Available Datasets</Typography>
            <Chip label="REAL DATA" color="success" size="small" sx={{ ml: 'auto' }} />
          </Box>

          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={3}>
              <Paper elevation={2} sx={{ p: 3, textAlign: 'center', bgcolor: 'primary.50' }}>
                <Typography variant="h3" color="primary" fontWeight={600}>
                  {isLoadingStats ? '...' : datasetStats?.bim_models?.count || 45}
                </Typography>
                <Typography variant="body1" fontWeight={600}>BIM Models</Typography>
                <Typography variant="caption" color="text.secondary" display="block">
                  {datasetStats?.bim_models?.format || 'IFC 4.3.x'}
                </Typography>
              </Paper>
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <Paper elevation={2} sx={{ p: 3, textAlign: 'center', bgcolor: 'secondary.50' }}>
                <Typography variant="h3" color="secondary" fontWeight={600}>
                  {isLoadingStats ? '...' : datasetStats?.gis_shapefiles?.count || 94}
                </Typography>
                <Typography variant="body1" fontWeight={600}>GIS Layers</Typography>
                <Typography variant="caption" color="text.secondary" display="block">
                  Phase 2a route data
                </Typography>
              </Paper>
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <Paper elevation={2} sx={{ p: 3, textAlign: 'center', bgcolor: 'success.50' }}>
                <Typography variant="h3" color="success.main" fontWeight={600}>
                  {datasetStats?.noise_monitoring?.measurements ?
                    `${(datasetStats.noise_monitoring.measurements / 1000000).toFixed(1)}M` :
                    '1.9M'
                  }
                </Typography>
                <Typography variant="body1" fontWeight={600}>Measurements</Typography>
                <Typography variant="caption" color="text.secondary" display="block">
                  {datasetStats?.noise_monitoring?.months || 10} months monitoring
                </Typography>
              </Paper>
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <Paper elevation={2} sx={{ p: 3, textAlign: 'center', bgcolor: 'info.50' }}>
                <Typography variant="h3" color="info.main" fontWeight={600}>
                  {isLoadingStats ? '...' : datasetStats?.total_files || 364}
                </Typography>
                <Typography variant="body1" fontWeight={600}>Total Files</Typography>
                <Typography variant="caption" color="text.secondary" display="block">
                  {datasetStats?.total_size_gb ? `${datasetStats.total_size_gb.toFixed(1)}GB` : '2.7GB'} organized
                </Typography>
              </Paper>
            </Grid>

            {/* New Datasets Row */}
            <Grid item xs={12} sm={6} md={3}>
              <Paper elevation={1} sx={{ p: 2.5, textAlign: 'center', bgcolor: 'success.50', border: '2px solid', borderColor: 'success.main' }}>
                <Typography variant="h4" color="success.dark" fontWeight={600}>
                  {isLoadingStats ? '...' : datasetStats?.ecology_surveys?.total_files || 63}
                </Typography>
                <Typography variant="body2" fontWeight={600}>Ecology Surveys</Typography>
                <Typography variant="caption" color="text.secondary" display="block">
                  Nov 2024 • CSV + shapefiles
                </Typography>
              </Paper>
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <Paper elevation={1} sx={{ p: 2.5, textAlign: 'center', bgcolor: 'error.50', border: '2px solid', borderColor: 'error.main' }}>
                <Typography variant="h4" color="error.dark" fontWeight={600}>
                  6.9K
                </Typography>
                <Typography variant="body2" fontWeight={600}>Legal Zones</Typography>
                <Typography variant="caption" color="text.secondary" display="block">
                  Court injunctions • GeoJSON
                </Typography>
              </Paper>
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <Paper elevation={1} sx={{ p: 2.5, textAlign: 'center', bgcolor: 'warning.50' }}>
                <Typography variant="h4" color="warning.dark" fontWeight={600}>
                  {isLoadingStats ? '...' : datasetStats?.property_compensation?.shapefiles || 1}
                </Typography>
                <Typography variant="body2" fontWeight={600}>Property Data</Typography>
                <Typography variant="caption" color="text.secondary" display="block">
                  July 2014 consultation
                </Typography>
              </Paper>
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <Paper elevation={1} sx={{ p: 2.5, textAlign: 'center', bgcolor: 'grey.100' }}>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  <strong>Multi-Modal Data</strong>
                </Typography>
                <Typography variant="caption" display="block">
                  BIM • GIS • Noise • Ecology
                </Typography>
                <Typography variant="caption" display="block">
                  Legal • Property
                </Typography>
              </Paper>
            </Grid>
          </Grid>

          <Alert severity="info" sx={{ mt: 3 }}>
            <Typography variant="body2">
              <strong>Data Location:</strong> <code>/datasets/hs2/organized/</code> • 180 new files added (~575MB)
            </Typography>
          </Alert>
        </Grid>
      </Grid>
        </Box>
      )}
    </Box>
  );
};

export default HS2OverviewTab;
