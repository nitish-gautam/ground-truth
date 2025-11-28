/**
 * HS2 Noise Monitoring Dashboard
 *
 * Visualizes 10 months of real HS2 noise monitoring data (Dec 2024 - Sept 2025)
 * Data sources: 170 Excel files from organized/monitoring/by-month/
 */

import React, { useState } from 'react';
import {
  Container,
  Typography,
  Box,
  Grid,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Card,
  CardContent,
  Alert,
  SelectChangeEvent
} from '@mui/material';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ZAxis
} from 'recharts';
import { Warning, CheckCircle, Error as ErrorIcon } from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';

// Available months in the dataset
const AVAILABLE_MONTHS = [
  { value: 'December_2024', label: 'December 2024' },
  { value: 'January_2025', label: 'January 2025' },
  { value: 'February_2025', label: 'February 2025' },
  { value: 'March_2025', label: 'March 2025' },
  { value: 'April_2025', label: 'April 2025' },
  { value: 'May_2025', label: 'May 2025' },
  { value: 'June_2025', label: 'June 2025' },
  { value: 'July_2025', label: 'July 2025' },
  { value: 'August_2025', label: 'August 2025' },
  { value: 'Sept_2025', label: 'September 2025' }
];

const GEOGRAPHIC_AREAS = ['Area North', 'Area Central', 'Area South'];

const COUNCILS = [
  'Birmingham', 'Solihull', 'Warwick', 'Stratford-On-Avon', 'North Warwickshire',
  'Lichfield', 'West Northamptonshire', 'Cherwell', 'Buckinghamshire', 'Three Rivers',
  'Hillingdon', 'Ealing', 'Brent', 'Camden'
];

export const HS2NoiseMonitoring: React.FC = () => {
  const [selectedMonth, setSelectedMonth] = useState('December_2024');
  const [selectedArea, setSelectedArea] = useState<string>('all');
  const [selectedCouncil, setSelectedCouncil] = useState<string>('all');

  // Fetch noise data from backend (real data from PostgreSQL)
  const { data: noiseData, isLoading, error } = useQuery({
    queryKey: ['noise-monitoring', selectedMonth, selectedArea, selectedCouncil],
    queryFn: async () => {
      console.log('🔍 Fetching noise data:', { selectedMonth, selectedArea, selectedCouncil });
      const response = await axios.get(`/api/v1/monitoring/noise`, {
        params: {
          month: selectedMonth,
          area: selectedArea !== 'all' ? selectedArea : undefined,
          council: selectedCouncil !== 'all' ? selectedCouncil : undefined
        }
      });
      console.log('✅ Noise data received:', response.data);
      return response.data;
    },
    retry: 2
  });

  // Log data state
  console.log('📊 Component state:', { noiseData, isLoading, error });

  if (error) {
    console.error('❌ Error fetching noise data:', error);
  }

  const handleMonthChange = (event: SelectChangeEvent) => {
    setSelectedMonth(event.target.value);
  };

  const handleAreaChange = (event: SelectChangeEvent) => {
    setSelectedArea(event.target.value);
  };

  const handleCouncilChange = (event: SelectChangeEvent) => {
    setSelectedCouncil(event.target.value);
  };

  if (isLoading) {
    return <Container><Typography>Loading noise data...</Typography></Container>;
  }

  const stats = noiseData?.summary || {
    total_measurements: 0,
    avg_noise: 0,
    compliance_rate: 0,
    violations: 0
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          HS2 Noise Monitoring Dashboard
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Real-time noise monitoring across HS2 construction sites (10 months of data)
        </Typography>
      </Box>

      {/* Filters */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={2}>
          <Grid item xs={12} md={4}>
            <FormControl fullWidth>
              <InputLabel>Month</InputLabel>
              <Select value={selectedMonth} onChange={handleMonthChange} label="Month">
                {AVAILABLE_MONTHS.map(month => (
                  <MenuItem key={month.value} value={month.value}>
                    {month.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={4}>
            <FormControl fullWidth>
              <InputLabel>Geographic Area</InputLabel>
              <Select value={selectedArea} onChange={handleAreaChange} label="Geographic Area">
                <MenuItem value="all">All Areas</MenuItem>
                {GEOGRAPHIC_AREAS.map(area => (
                  <MenuItem key={area} value={area}>{area}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={4}>
            <FormControl fullWidth>
              <InputLabel>Council</InputLabel>
              <Select value={selectedCouncil} onChange={handleCouncilChange} label="Council">
                <MenuItem value="all">All Councils</MenuItem>
                {COUNCILS.map(council => (
                  <MenuItem key={council} value={council}>{council}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
        </Grid>
      </Paper>

      {/* KPI Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Total Measurements
              </Typography>
              <Typography variant="h4">{stats.total_measurements.toLocaleString()}</Typography>
              <Typography variant="caption" color="text.secondary">
                Across all sites
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Average Noise Level
              </Typography>
              <Typography variant="h4">{stats.avg_noise.toFixed(1)} dB</Typography>
              <Typography variant="caption" color="text.secondary">
                All locations
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Compliance Rate
              </Typography>
              <Typography variant="h4" color={stats.compliance_rate >= 90 ? 'success.main' : 'warning.main'}>
                {stats.compliance_rate.toFixed(1)}%
              </Typography>
              <Chip
                icon={stats.compliance_rate >= 90 ? <CheckCircle /> : <Warning />}
                label={stats.compliance_rate >= 90 ? 'Good' : 'Needs Attention'}
                color={stats.compliance_rate >= 90 ? 'success' : 'warning'}
                size="small"
                sx={{ mt: 1 }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Violations
              </Typography>
              <Typography variant="h4" color={stats.violations > 0 ? 'error.main' : 'success.main'}>
                {stats.violations}
              </Typography>
              <Chip
                icon={stats.violations > 0 ? <ErrorIcon /> : <CheckCircle />}
                label={stats.violations > 0 ? 'Action Required' : 'Compliant'}
                color={stats.violations > 0 ? 'error' : 'success'}
                size="small"
                sx={{ mt: 1 }}
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Compliance Alert */}
      {stats.violations > 0 && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          <strong>{stats.violations} noise violations</strong> detected in {selectedMonth.replace('_', ' ')}.
          Immediate action required for non-compliant sites.
        </Alert>
      )}

      {/* Time Series Chart */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Noise Levels Over Time
        </Typography>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={noiseData?.time_series || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis label={{ value: 'Noise Level (dB)', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="avg_noise" stroke="#012A39" strokeWidth={2} name="Average Noise" />
            <Line type="monotone" dataKey="max_noise" stroke="#FF8500" strokeWidth={2} name="Max Noise" />
            <Line type="monotone" dataKey="limit" stroke="#FF0000" strokeWidth={2} strokeDasharray="5 5" name="Limit (75 dB)" />
          </LineChart>
        </ResponsiveContainer>
      </Paper>

      {/* Geographic Heatmap & Bar Chart */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {/* Noise by Location (Scatter/Heatmap simulation) */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Geographic Distribution (Heatmap)
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart>
                <CartesianGrid />
                <XAxis dataKey="x" name="X" />
                <YAxis dataKey="y" name="Y" />
                <ZAxis dataKey="noise_level" range={[50, 400]} name="Noise Level" />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Legend />
                <Scatter
                  name="Noise Measurements"
                  data={noiseData?.geographic || []}
                  fill="#019C4B"
                />
              </ScatterChart>
            </ResponsiveContainer>
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
              Bubble size represents noise level intensity
            </Typography>
          </Paper>
        </Grid>

        {/* Noise by Council */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Average Noise by Council
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={noiseData?.by_council || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="council" angle={-45} textAnchor="end" height={100} />
                <YAxis label={{ value: 'Noise Level (dB)', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="avg_noise" fill="#019C4B" name="Average Noise" />
                <Bar dataKey="max_noise" fill="#FF8500" name="Max Noise" />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
      </Grid>

      {/* Violations Table */}
      {stats.violations > 0 && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Noise Violations (Exceeds 75 dB Limit)
          </Typography>
          <Box sx={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #ddd' }}>
                  <th style={{ padding: '12px', textAlign: 'left' }}>Date/Time</th>
                  <th style={{ padding: '12px', textAlign: 'left' }}>Location</th>
                  <th style={{ padding: '12px', textAlign: 'left' }}>Council</th>
                  <th style={{ padding: '12px', textAlign: 'right' }}>Noise Level (dB)</th>
                  <th style={{ padding: '12px', textAlign: 'right' }}>Excess (dB)</th>
                  <th style={{ padding: '12px', textAlign: 'left' }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {(noiseData?.violations || []).map((violation, idx) => (
                  <tr key={idx} style={{ borderBottom: '1px solid #eee' }}>
                    <td style={{ padding: '12px' }}>{violation.timestamp}</td>
                    <td style={{ padding: '12px' }}>{violation.location}</td>
                    <td style={{ padding: '12px' }}>{violation.council}</td>
                    <td style={{ padding: '12px', textAlign: 'right', fontWeight: 'bold', color: '#d32f2f' }}>
                      {violation.noiseLevel} dB
                    </td>
                    <td style={{ padding: '12px', textAlign: 'right' }}>
                      +{violation.excess} dB
                    </td>
                    <td style={{ padding: '12px' }}>
                      <Chip label={violation.status} color="error" size="small" />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Box>
        </Paper>
      )}

      {/* Data Source Info */}
      <Alert severity="info" sx={{ mt: 3 }}>
        <Typography variant="body2">
          <strong>Data Source:</strong> Real HS2 monitoring data from{' '}
          <code>datasets/hs2/organized/monitoring/by-month/{selectedMonth}/</code>
          <br />
          <strong>Coverage:</strong> 10 months (Dec 2024 - Sept 2025), 170 Excel files, 15+ councils
        </Typography>
      </Alert>
    </Container>
  );
};

// Mock data removed - now using real data from PostgreSQL database loaded from Excel files
