/**
 * LiDAR Elevation Profile Viewer
 * ===============================
 *
 * Displays elevation profiles from LiDAR DTM data along railway alignments.
 *
 * Features:
 * - Interactive profile chart
 * - Elevation statistics
 * - Gradient analysis
 * - Export profile data
 */

import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  Grid,
  Alert,
  CircularProgress,
  Chip,
  Paper
} from '@mui/material';
import {
  Terrain as Mountain,
  TrendingUp,
  TrendingDown,
  Straighten as Ruler,
  Download,
  Refresh as RefreshCw
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  AreaChart
} from 'recharts';

interface ElevationPoint {
  distance: number;
  easting: number;
  northing: number;
  elevation: number;
}

interface ElevationProfile {
  profile_id: string | null;
  profile_name: string | null;
  profile_length_m: number;
  num_samples: number;
  start_point: [number, number];  // API returns array [easting, northing]
  end_point: [number, number];  // API returns array [easting, northing]
  min_elevation: number;  // Flat fields from API
  max_elevation: number;
  elevation_gain: number;
  elevation_loss: number;
  profile_data: ElevationPoint[];  // API returns profile_data not elevation_data
  source_tiles: string[];
  created_at: string;
}

export function LidarProfileViewer() {
  const [startEasting, setStartEasting] = useState('426000');
  const [startNorthing, setStartNorthing] = useState('337000');
  const [endEasting, setEndEasting] = useState('427000');
  const [endNorthing, setEndNorthing] = useState('338000');
  const [numSamples, setNumSamples] = useState('100');
  const [loading, setLoading] = useState(false);
  const [profile, setProfile] = useState<ElevationProfile | null>(null);
  const [error, setError] = useState<string | null>(null);

  const generateProfile = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/v1/lidar/elevation/profile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          start_point: [parseFloat(startEasting), parseFloat(startNorthing)],
          end_point: [parseFloat(endEasting), parseFloat(endNorthing)],
          num_samples: parseInt(numSamples),
          save_profile: false,  // Don't save to database (table not created yet)
          profile_name: `Profile_${new Date().toISOString().split('T')[0]}`
        })
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }

      const data = await response.json();
      setProfile(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate profile');
    } finally {
      setLoading(false);
    }
  };

  const exportProfile = () => {
    if (!profile) return;

    const csv = [
      'Distance (m),Easting,Northing,Elevation (m)',
      ...profile.profile_data.map(p =>
        `${p.distance},${p.easting},${p.northing},${p.elevation}`
      )
    ].join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `elevation_profile_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Calculate maximum gradient
  const maxGradient = profile ? (() => {
    let max = 0;
    for (let i = 1; i < profile.profile_data.length; i++) {
      const rise = Math.abs(
        profile.profile_data[i].elevation - profile.profile_data[i - 1].elevation
      );
      const run = profile.profile_data[i].distance - profile.profile_data[i - 1].distance;
      const gradient = (rise / run) * 100;
      if (gradient > max) max = gradient;
    }
    return max;
  })() : 0;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      {/* Input Form */}
      <Card elevation={2}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Mountain color="primary" />
              <Typography variant="h6" fontWeight={600}>
                Generate Elevation Profile
              </Typography>
            </Box>
            <Chip
              label="🟢 REAL DATA (UK LiDAR DTM 2022)"
              size="small"
              sx={{ bgcolor: 'rgb(220, 252, 231)', color: 'rgb(22, 101, 52)', fontWeight: 600 }}
            />
          </Box>

          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                Start Point (British National Grid)
              </Typography>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <TextField
                  fullWidth
                  size="small"
                  label="Easting"
                  value={startEasting}
                  onChange={(e) => setStartEasting(e.target.value)}
                />
                <TextField
                  fullWidth
                  size="small"
                  label="Northing"
                  value={startNorthing}
                  onChange={(e) => setStartNorthing(e.target.value)}
                />
              </Box>
            </Grid>

            <Grid item xs={12} md={6}>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                End Point (British National Grid)
              </Typography>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <TextField
                  fullWidth
                  size="small"
                  label="Easting"
                  value={endEasting}
                  onChange={(e) => setEndEasting(e.target.value)}
                />
                <TextField
                  fullWidth
                  size="small"
                  label="Northing"
                  value={endNorthing}
                  onChange={(e) => setEndNorthing(e.target.value)}
                />
              </Box>
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                size="small"
                type="number"
                label="Number of Samples"
                value={numSamples}
                onChange={(e) => setNumSamples(e.target.value)}
                inputProps={{ min: 10, max: 1000 }}
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <Button
                fullWidth
                variant="contained"
                onClick={generateProfile}
                disabled={loading}
                startIcon={loading ? <CircularProgress size={20} /> : null}
              >
                {loading ? 'Generating...' : 'Generate Profile'}
              </Button>
            </Grid>
          </Grid>

          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Profile Statistics */}
      {profile && (
        <>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={3}>
              <Card elevation={2}>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Profile Length
                      </Typography>
                      <Typography variant="h4" fontWeight={600}>
                        {profile.profile_length_m.toFixed(0)}m
                      </Typography>
                    </Box>
                    <Ruler sx={{ fontSize: 40, color: 'primary.main' }} />
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <Card elevation={2}>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Elevation Range
                      </Typography>
                      <Typography variant="h4" fontWeight={600}>
                        {(profile.max_elevation - profile.min_elevation).toFixed(1)}m
                      </Typography>
                    </Box>
                    <Mountain sx={{ fontSize: 40, color: 'success.main' }} />
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <Card elevation={2}>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Total Climb
                      </Typography>
                      <Typography variant="h4" fontWeight={600}>
                        {profile.elevation_gain.toFixed(1)}m
                      </Typography>
                    </Box>
                    <TrendingUp sx={{ fontSize: 40, color: 'warning.main' }} />
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <Card elevation={2}>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Total Descent
                      </Typography>
                      <Typography variant="h4" fontWeight={600}>
                        {profile.elevation_loss.toFixed(1)}m
                      </Typography>
                    </Box>
                    <TrendingDown sx={{ fontSize: 40, color: 'secondary.main' }} />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Elevation Chart */}
          <Card elevation={2}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                <Typography variant="h6" fontWeight={600}>
                  Elevation Profile
                </Typography>
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={<Download />}
                  onClick={exportProfile}
                >
                  Export CSV
                </Button>
              </Box>

              <ResponsiveContainer width="100%" height={400}>
                <AreaChart data={profile.profile_data}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="distance"
                    label={{ value: 'Distance (m)', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis
                    label={{ value: 'Elevation (m ODN)', angle: -90, position: 'insideLeft' }}
                    domain={['dataMin - 5', 'dataMax + 5']}
                  />
                  <Tooltip
                    formatter={(value: any) => [`${value.toFixed(2)} m`, 'Elevation']}
                    labelFormatter={(label: any) => `Distance: ${label} m`}
                  />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="elevation"
                    stroke="#3b82f6"
                    fill="#93c5fd"
                    name="Elevation (m)"
                  />
                </AreaChart>
              </ResponsiveContainer>

              {/* Gradient Warning */}
              {maxGradient > 2.5 && (
                <Alert severity="warning" sx={{ mt: 2 }}>
                  <strong>Maximum gradient: {maxGradient.toFixed(2)}%</strong>
                  {' - Exceeds HS2 limit (2.5%)'}
                </Alert>
              )}

              {/* Additional Details */}
              <Grid container spacing={2} sx={{ mt: 2 }}>
                <Grid item xs={6} sm={3}>
                  <Typography variant="body2" color="text.secondary">
                    Min Elevation
                  </Typography>
                  <Typography variant="body1" fontWeight={600}>
                    {profile.min_elevation.toFixed(2)} m ODN
                  </Typography>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Typography variant="body2" color="text.secondary">
                    Max Elevation
                  </Typography>
                  <Typography variant="body1" fontWeight={600}>
                    {profile.max_elevation.toFixed(2)} m ODN
                  </Typography>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Typography variant="body2" color="text.secondary">
                    Samples
                  </Typography>
                  <Typography variant="body1" fontWeight={600}>
                    {profile.num_samples}
                  </Typography>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Typography variant="body2" color="text.secondary">
                    Max Gradient
                  </Typography>
                  <Typography variant="body1" fontWeight={600}>
                    {maxGradient.toFixed(2)}%
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </>
      )}
    </Box>
  );
}
