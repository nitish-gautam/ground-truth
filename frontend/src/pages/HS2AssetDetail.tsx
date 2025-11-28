/**
 * HS2 Asset Detail Page
 * Comprehensive asset view with tabs for different data sections
 */

import React, { useState } from 'react';
import {
  Box,
  Container,
  Typography,
  Paper,
  Chip,
  Tabs,
  Tab,
  Button,
  Breadcrumbs,
  Link,
  Grid,
  Card,
  CardContent,
  CircularProgress,
  Alert
} from '@mui/material';
import {
  Home,
  ArrowBack,
  Refresh as RefreshIcon,
  Assessment
} from '@mui/icons-material';
import { useParams, useNavigate, Link as RouterLink } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import hs2Client from '../api/hs2Client';
import HS2ReadinessPanel from '../components/HS2ReadinessPanel';
import HS2ExplainabilityPanel from '../components/HS2ExplainabilityPanel';
import HS2DeliverablesTable from '../components/HS2DeliverablesTable';
import HS2CostSummary from '../components/HS2CostSummary';
import HS2CertificatesTable from '../components/HS2CertificatesTable';
import { formatDateTime, getAssetStatusVariant, formatTAEMScore } from '../utils/formatting';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => {
  return (
    <div role="tabpanel" hidden={value !== index}>
      {value === index && <Box sx={{ pt: 3 }}>{children}</Box>}
    </div>
  );
};

const HS2AssetDetail: React.FC = () => {
  const { assetId } = useParams<{ assetId: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState(0);

  // Fetch asset details
  const {
    data: asset,
    isLoading: assetLoading,
    error: assetError
  } = useQuery({
    queryKey: ['hs2-asset', assetId],
    queryFn: () => hs2Client.assets.getAsset(assetId!),
    enabled: !!assetId
  });

  // Fetch readiness report
  const {
    data: readinessReport,
    isLoading: readinessLoading
  } = useQuery({
    queryKey: ['hs2-readiness', assetId],
    queryFn: () => hs2Client.readiness.getReport(assetId!),
    enabled: !!assetId
  });

  // Fetch deliverables
  const {
    data: deliverables,
    isLoading: deliverablesLoading
  } = useQuery({
    queryKey: ['hs2-deliverables', assetId],
    queryFn: () => hs2Client.deliverables.getByAsset(assetId!),
    enabled: !!assetId && activeTab === 1
  });

  // Fetch cost summary
  const {
    data: costSummary,
    isLoading: costsLoading
  } = useQuery({
    queryKey: ['hs2-costs', assetId],
    queryFn: () => hs2Client.costs.getSummary(assetId!),
    enabled: !!assetId && activeTab === 2
  });

  // Fetch certificates
  const {
    data: certificates,
    isLoading: certificatesLoading
  } = useQuery({
    queryKey: ['hs2-certificates', assetId],
    queryFn: () => hs2Client.certificates.getByAsset(assetId!),
    enabled: !!assetId && activeTab === 3
  });

  // Fetch history
  const {
    data: history,
    isLoading: historyLoading
  } = useQuery({
    queryKey: ['hs2-history', assetId],
    queryFn: () => hs2Client.history.getByAsset(assetId!),
    enabled: !!assetId && activeTab === 4
  });

  // Re-evaluate mutation
  const evaluateMutation = useMutation({
    mutationFn: (assetId: string) => hs2Client.assets.evaluateAsset(assetId),
    onSuccess: () => {
      // Invalidate relevant queries to refetch data
      queryClient.invalidateQueries(['hs2-asset', assetId]);
      queryClient.invalidateQueries(['hs2-readiness', assetId]);
    }
  });

  const handleEvaluate = () => {
    if (assetId) {
      evaluateMutation.mutate(assetId);
    }
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  if (assetLoading) {
    return (
      <Container maxWidth="xl" sx={{ mt: 4, display: 'flex', justifyContent: 'center' }}>
        <CircularProgress />
      </Container>
    );
  }

  if (assetError || !asset) {
    return (
      <Container maxWidth="xl" sx={{ mt: 4 }}>
        <Alert severity="error">
          Failed to load asset details. Asset may not exist.
        </Alert>
        <Button onClick={() => navigate('/hs2/assets')} sx={{ mt: 2 }}>
          Back to Asset List
        </Button>
      </Container>
    );
  }

  const taemDisplay = formatTAEMScore(asset.taem_score);

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      {/* Breadcrumbs */}
      <Breadcrumbs sx={{ mb: 2 }}>
        <Link
          component={RouterLink}
          to="/hs2"
          color="inherit"
          sx={{ display: 'flex', alignItems: 'center' }}
        >
          <Home sx={{ mr: 0.5 }} fontSize="small" />
          Dashboard
        </Link>
        <Link component={RouterLink} to="/hs2/assets" color="inherit">
          Assets
        </Link>
        <Typography color="text.primary">{asset.asset_id}</Typography>
      </Breadcrumbs>

      {/* Asset Header */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Box display="flex" justifyContent="space-between" alignItems="flex-start">
          <Box flex={1}>
            <Box display="flex" alignItems="center" gap={2} mb={2}>
              <Typography variant="h4">{asset.asset_name}</Typography>
              <Chip
                label={asset.status}
                color={getAssetStatusVariant(asset.status)}
              />
              <Chip
                label={`TAEM: ${taemDisplay.text}`}
                color={taemDisplay.color}
                variant="outlined"
              />
            </Box>
            
            <Grid container spacing={3}>
              <Grid item xs={12} sm={6} md={3}>
                <Typography variant="caption" color="text.secondary">Asset ID</Typography>
                <Typography variant="body1" fontWeight="bold">{asset.asset_id}</Typography>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Typography variant="caption" color="text.secondary">Type</Typography>
                <Typography variant="body1">{asset.asset_type}</Typography>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Typography variant="caption" color="text.secondary">Contractor</Typography>
                <Typography variant="body1">{asset.contractor}</Typography>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Typography variant="caption" color="text.secondary">Route Section</Typography>
                <Typography variant="body1">{asset.route_section}</Typography>
              </Grid>
              <Grid item xs={12}>
                <Typography variant="caption" color="text.secondary">Last Evaluated</Typography>
                <Typography variant="body2">{formatDateTime(asset.last_evaluated)}</Typography>
              </Grid>
            </Grid>
          </Box>

          <Box display="flex" flexDirection="column" gap={1}>
            <Button
              variant="outlined"
              startIcon={<ArrowBack />}
              onClick={() => navigate('/hs2/assets')}
            >
              Back
            </Button>
            <Button
              variant="contained"
              startIcon={evaluateMutation.isLoading ? <CircularProgress size={16} /> : <Assessment />}
              onClick={handleEvaluate}
              disabled={evaluateMutation.isLoading}
            >
              Re-Evaluate
            </Button>
          </Box>
        </Box>
      </Paper>

      {/* Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs value={activeTab} onChange={handleTabChange}>
          <Tab label="Readiness" />
          <Tab label="Deliverables" />
          <Tab label="Costs" />
          <Tab label="Certificates" />
          <Tab label="History" />
        </Tabs>
      </Paper>

      {/* Tab Content */}
      <TabPanel value={activeTab} index={0}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <HS2ReadinessPanel report={readinessReport} isLoading={readinessLoading} />
          </Grid>
          <Grid item xs={12} md={6}>
            <HS2ExplainabilityPanel report={readinessReport} isLoading={readinessLoading} />
          </Grid>
        </Grid>
      </TabPanel>

      <TabPanel value={activeTab} index={1}>
        <HS2DeliverablesTable
          deliverables={deliverables || []}
          isLoading={deliverablesLoading}
        />
      </TabPanel>

      <TabPanel value={activeTab} index={2}>
        <HS2CostSummary costSummary={costSummary} isLoading={costsLoading} />
      </TabPanel>

      <TabPanel value={activeTab} index={3}>
        <HS2CertificatesTable
          certificates={certificates || []}
          isLoading={certificatesLoading}
        />
      </TabPanel>

      <TabPanel value={activeTab} index={4}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>Change History</Typography>
            {historyLoading ? (
              <Box display="flex" justifyContent="center" py={4}>
                <CircularProgress />
              </Box>
            ) : history && history.length > 0 ? (
              <Box>
                {history.map((record) => (
                  <Box key={record.history_id} sx={{ mb: 2, pb: 2, borderBottom: '1px solid #eee' }}>
                    <Typography variant="body2">
                      <strong>{record.change_type}</strong> - {record.description}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {formatDateTime(record.changed_at)} by {record.changed_by}
                    </Typography>
                  </Box>
                ))}
              </Box>
            ) : (
              <Typography color="text.secondary">No history available</Typography>
            )}
          </CardContent>
        </Card>
      </TabPanel>
    </Container>
  );
};

export default HS2AssetDetail;
