/**
 * HS2 Dashboard-Style Integrated Inspection Demo
 * Enterprise UX: All inputs, visualizations, and outputs visible simultaneously
 * 3-Column Layout: Inputs (Left) | Visualization (Center) | Analysis (Right)
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Card,
  CardContent,
  CardHeader,
  Grid,
  Chip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Checkbox,
  Button,
  ButtonGroup,
  Tabs,
  Tab,
  AppBar,
  Toolbar,
  Select,
  MenuItem,
  IconButton,
  CircularProgress,
  LinearProgress,
  Avatar,
  Divider,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
  Badge
} from '@mui/material';
import {
  Terrain,
  Colorize,
  ViewInAr,
  CameraAlt,
  CheckCircle,
  Schedule,
  Settings,
  Refresh,
  Straighten,
  Science,
  Visibility,
  Assessment,
  Warning,
  ZoomIn,
  Info,
  TrendingUp,
  Error as ErrorIcon
} from '@mui/icons-material';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';

// Sample data
const ASSETS = [
  { id: 'pier_p1', label: 'Pier P1' },
  { id: 'pier_p2', label: 'Pier P2' },
  { id: 'pier_p3', label: 'Pier P3' }
];

const ALL_SEGMENTS = [
  // Pier P1 - Mixed quality (has some defects)
  {
    id: 'Pier_P1_East_Face',
    pier: 'pier_p1',
    face: 'East Face',
    area_sqm: 45.2,
    lidar_points: 125000,
    hsi_coverage: 95,
    images_360: 8,
    quality_score: 78,
    flatness_mm: 3.2,
    verticality_mm: 2.1,
    strength_mpa: 42.3,
    moisture_pct: 3.8,
    defects: 3
  },
  {
    id: 'Pier_P1_West_Face',
    pier: 'pier_p1',
    face: 'West Face',
    area_sqm: 45.2,
    lidar_points: 118000,
    hsi_coverage: 92,
    images_360: 7,
    quality_score: 89,
    flatness_mm: 2.8,
    verticality_mm: 1.9,
    strength_mpa: 44.1,
    moisture_pct: 3.5,
    defects: 1
  },
  {
    id: 'Pier_P1_North_Face',
    pier: 'pier_p1',
    face: 'North Face',
    area_sqm: 22.5,
    lidar_points: 62000,
    hsi_coverage: 88,
    images_360: 4,
    quality_score: 82,
    flatness_mm: 4.1,
    verticality_mm: 2.5,
    strength_mpa: 39.8,
    moisture_pct: 4.2,
    defects: 2
  },
  {
    id: 'Pier_P1_South_Face',
    pier: 'pier_p1',
    face: 'South Face',
    area_sqm: 22.5,
    lidar_points: 65000,
    hsi_coverage: 90,
    images_360: 4,
    quality_score: 94,
    flatness_mm: 2.3,
    verticality_mm: 1.7,
    strength_mpa: 46.2,
    moisture_pct: 3.2,
    defects: 0
  },

  // Pier P2 - Better quality (fewer defects, better scores)
  {
    id: 'Pier_P2_East_Face',
    pier: 'pier_p2',
    face: 'East Face',
    area_sqm: 45.2,
    lidar_points: 130000,
    hsi_coverage: 97,
    images_360: 8,
    quality_score: 92,
    flatness_mm: 2.1,
    verticality_mm: 1.5,
    strength_mpa: 45.8,
    moisture_pct: 3.1,
    defects: 1
  },
  {
    id: 'Pier_P2_West_Face',
    pier: 'pier_p2',
    face: 'West Face',
    area_sqm: 45.2,
    lidar_points: 128000,
    hsi_coverage: 96,
    images_360: 8,
    quality_score: 95,
    flatness_mm: 1.9,
    verticality_mm: 1.3,
    strength_mpa: 47.2,
    moisture_pct: 2.9,
    defects: 0
  },
  {
    id: 'Pier_P2_North_Face',
    pier: 'pier_p2',
    face: 'North Face',
    area_sqm: 22.5,
    lidar_points: 68000,
    hsi_coverage: 93,
    images_360: 4,
    quality_score: 90,
    flatness_mm: 2.5,
    verticality_mm: 1.8,
    strength_mpa: 44.5,
    moisture_pct: 3.3,
    defects: 1
  },
  {
    id: 'Pier_P2_South_Face',
    pier: 'pier_p2',
    face: 'South Face',
    area_sqm: 22.5,
    lidar_points: 70000,
    hsi_coverage: 94,
    images_360: 4,
    quality_score: 97,
    flatness_mm: 1.7,
    verticality_mm: 1.2,
    strength_mpa: 48.1,
    moisture_pct: 2.8,
    defects: 0
  },

  // Pier P3 - Poorest quality (more defects, lower scores)
  {
    id: 'Pier_P3_East_Face',
    pier: 'pier_p3',
    face: 'East Face',
    area_sqm: 45.2,
    lidar_points: 115000,
    hsi_coverage: 89,
    images_360: 7,
    quality_score: 68,
    flatness_mm: 4.3,
    verticality_mm: 2.7,
    strength_mpa: 38.9,
    moisture_pct: 4.5,
    defects: 4
  },
  {
    id: 'Pier_P3_West_Face',
    pier: 'pier_p3',
    face: 'West Face',
    area_sqm: 45.2,
    lidar_points: 112000,
    hsi_coverage: 87,
    images_360: 7,
    quality_score: 72,
    flatness_mm: 3.9,
    verticality_mm: 2.4,
    strength_mpa: 40.2,
    moisture_pct: 4.1,
    defects: 2
  },
  {
    id: 'Pier_P3_North_Face',
    pier: 'pier_p3',
    face: 'North Face',
    area_sqm: 22.5,
    lidar_points: 58000,
    hsi_coverage: 85,
    images_360: 4,
    quality_score: 75,
    flatness_mm: 4.5,
    verticality_mm: 2.8,
    strength_mpa: 37.5,
    moisture_pct: 4.7,
    defects: 3
  },
  {
    id: 'Pier_P3_South_Face',
    pier: 'pier_p3',
    face: 'South Face',
    area_sqm: 22.5,
    lidar_points: 60000,
    hsi_coverage: 86,
    images_360: 4,
    quality_score: 80,
    flatness_mm: 3.5,
    verticality_mm: 2.2,
    strength_mpa: 41.3,
    moisture_pct: 3.9,
    defects: 2
  }
];

const ALL_DEFECTS = [
  // Pier P1 defects (6 total)
  {
    id: 1,
    type: 'Crack',
    severity: 'Minor',
    location: [12.3, 5.6],
    length_mm: 45,
    segment: 'Pier_P1_East_Face'
  },
  {
    id: 2,
    type: 'Spalling',
    severity: 'Moderate',
    location: [18.2, 8.1],
    area_sqcm: 12,
    segment: 'Pier_P1_East_Face'
  },
  {
    id: 3,
    type: 'Discoloration',
    severity: 'Minor',
    location: [15.7, 12.3],
    area_sqcm: 28,
    segment: 'Pier_P1_East_Face'
  },
  {
    id: 4,
    type: 'Crack',
    severity: 'Minor',
    location: [8.5, 6.2],
    length_mm: 32,
    segment: 'Pier_P1_West_Face'
  },
  {
    id: 5,
    type: 'Spalling',
    severity: 'Minor',
    location: [5.3, 4.1],
    area_sqcm: 8,
    segment: 'Pier_P1_North_Face'
  },
  {
    id: 6,
    type: 'Crack',
    severity: 'Minor',
    location: [10.2, 7.8],
    length_mm: 28,
    segment: 'Pier_P1_North_Face'
  },

  // Pier P2 defects (2 total - better quality)
  {
    id: 7,
    type: 'Minor Surface Wear',
    severity: 'Minor',
    location: [14.5, 6.8],
    area_sqcm: 5,
    segment: 'Pier_P2_East_Face'
  },
  {
    id: 8,
    type: 'Discoloration',
    severity: 'Minor',
    location: [7.2, 9.3],
    area_sqcm: 15,
    segment: 'Pier_P2_North_Face'
  },

  // Pier P3 defects (11 total - poorest quality)
  {
    id: 9,
    type: 'Crack',
    severity: 'Moderate',
    location: [11.5, 7.2],
    length_mm: 68,
    segment: 'Pier_P3_East_Face'
  },
  {
    id: 10,
    type: 'Spalling',
    severity: 'Moderate',
    location: [16.8, 9.4],
    area_sqcm: 18,
    segment: 'Pier_P3_East_Face'
  },
  {
    id: 11,
    type: 'Discoloration',
    severity: 'Minor',
    location: [19.3, 11.2],
    area_sqcm: 32,
    segment: 'Pier_P3_East_Face'
  },
  {
    id: 12,
    type: 'Crack',
    severity: 'Minor',
    location: [6.8, 4.5],
    length_mm: 42,
    segment: 'Pier_P3_East_Face'
  },
  {
    id: 13,
    type: 'Crack',
    severity: 'Moderate',
    location: [9.7, 8.1],
    length_mm: 55,
    segment: 'Pier_P3_West_Face'
  },
  {
    id: 14,
    type: 'Spalling',
    severity: 'Minor',
    location: [13.2, 6.3],
    area_sqcm: 10,
    segment: 'Pier_P3_West_Face'
  },
  {
    id: 15,
    type: 'Crack',
    severity: 'Minor',
    location: [4.2, 3.8],
    length_mm: 38,
    segment: 'Pier_P3_North_Face'
  },
  {
    id: 16,
    type: 'Spalling',
    severity: 'Moderate',
    location: [7.5, 5.9],
    area_sqcm: 14,
    segment: 'Pier_P3_North_Face'
  },
  {
    id: 17,
    type: 'Discoloration',
    severity: 'Minor',
    location: [11.8, 8.7],
    area_sqcm: 22,
    segment: 'Pier_P3_North_Face'
  },
  {
    id: 18,
    type: 'Crack',
    severity: 'Minor',
    location: [8.3, 6.1],
    length_mm: 35,
    segment: 'Pier_P3_South_Face'
  },
  {
    id: 19,
    type: 'Spalling',
    severity: 'Minor',
    location: [12.1, 9.2],
    area_sqcm: 9,
    segment: 'Pier_P3_South_Face'
  }
];

// Spectral data for material analysis
const SPECTRAL_DATA = Array.from({ length: 20 }, (_, i) => ({
  wavelength: 400 + i * 30,
  reflectance: 0.3 + Math.random() * 0.4
}));

export const DashboardIntegratedDemo: React.FC = () => {
  const [selectedAsset, setSelectedAsset] = useState('pier_p1');
  const [selectedSegments, setSelectedSegments] = useState<string[]>(['Pier_P1_East_Face']);
  const [viewMode, setViewMode] = useState('3d');
  const [analysisTab, setAnalysisTab] = useState(0);
  const [overlays, setOverlays] = useState<string[]>(['lidar']);

  // Filter segments and defects based on selected asset
  const SEGMENTS = ALL_SEGMENTS.filter(s => s.pier === selectedAsset);
  const DEFECTS = ALL_DEFECTS.filter(d => {
    // Extract pier ID from segment name (e.g., 'Pier_P1_East_Face' -> 'pier_p1')
    const segmentPier = d.segment.split('_').slice(0, 2).join('_').toLowerCase();
    return segmentPier === selectedAsset;
  });

  // Reset selected segments when asset changes
  useEffect(() => {
    const currentPierSegments = ALL_SEGMENTS.filter(s => s.pier === selectedAsset);
    if (currentPierSegments.length > 0) {
      setSelectedSegments([currentPierSegments[0].id]);
    }
  }, [selectedAsset]);

  const toggleSegment = (segmentId: string) => {
    setSelectedSegments(prev =>
      prev.includes(segmentId)
        ? prev.filter(id => id !== segmentId)
        : [...prev, segmentId]
    );
  };

  const selectAllSegments = () => {
    setSelectedSegments(SEGMENTS.map(s => s.id));
  };

  const clearAllSegments = () => {
    setSelectedSegments([]);
  };

  const getScoreColor = (score: number) => {
    if (score >= 90) return 'success';
    if (score >= 75) return 'primary';
    if (score >= 60) return 'warning';
    return 'error';
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'Critical':
        return 'error.main';
      case 'Moderate':
        return 'warning.main';
      case 'Minor':
        return 'info.main';
      default:
        return 'default';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'Critical':
        return <ErrorIcon />;
      case 'Moderate':
        return <Warning />;
      case 'Minor':
        return <Info />;
      default:
        return <Info />;
    }
  };

  // Calculate average quality score
  const avgQualityScore = selectedSegments.length > 0
    ? Math.round(
        SEGMENTS.filter(s => selectedSegments.includes(s.id))
          .reduce((sum, s) => sum + s.quality_score, 0) / selectedSegments.length
      )
    : 85;

  // Get comparison data for selected segments
  const comparisonData = SEGMENTS.filter(s => selectedSegments.includes(s.id));

  return (
    <Box sx={{ width: '100%', height: 'calc(100vh - 150px)', display: 'flex', flexDirection: 'column' }}>
      {/* Top Control Bar */}
      <AppBar position="static" color="default" elevation={1} sx={{ width: '100%' }}>
        <Toolbar>
          {/* Asset Selector */}
          <Select
            value={selectedAsset}
            onChange={(e) => setSelectedAsset(e.target.value)}
            size="small"
            sx={{ minWidth: 150, mr: 2 }}
          >
            {ASSETS.map(asset => (
              <MenuItem key={asset.id} value={asset.id}>{asset.label}</MenuItem>
            ))}
          </Select>

          {/* Data Status Chips */}
          <Chip
            icon={<Terrain />}
            label="LiDAR: 370K pts"
            color="success"
            size="small"
            sx={{ mr: 1 }}
          />
          <Chip
            icon={<Colorize />}
            label="HSI: 4 segments"
            color="success"
            size="small"
            sx={{ mr: 1 }}
          />
          <Chip
            icon={<ViewInAr />}
            label="BIM: Ready"
            color="success"
            size="small"
            sx={{ mr: 1 }}
          />
          <Chip
            icon={<CameraAlt />}
            label="360°: Pending"
            size="small"
            sx={{ mr: 1 }}
          />

          {/* Quality Score - Large Display */}
          <Box ml="auto" display="flex" alignItems="center" gap={1}>
            <Box position="relative" display="inline-flex">
              <CircularProgress
                variant="determinate"
                value={avgQualityScore}
                size={60}
                thickness={5}
                color={getScoreColor(avgQualityScore)}
              />
              <Box
                sx={{
                  top: 0,
                  left: 0,
                  bottom: 0,
                  right: 0,
                  position: 'absolute',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <Typography variant="caption" component="div" fontWeight={600}>
                  {avgQualityScore}
                </Typography>
              </Box>
            </Box>
            <Box>
              <Typography variant="h5" fontWeight={600}>
                {avgQualityScore}/100
              </Typography>
              <Chip
                label={avgQualityScore >= 85 ? 'GOOD' : avgQualityScore >= 70 ? 'FAIR' : 'POOR'}
                color={getScoreColor(avgQualityScore)}
                size="small"
              />
            </Box>
          </Box>

          {/* Actions */}
          <IconButton sx={{ ml: 2 }}>
            <Settings />
          </IconButton>
          <IconButton>
            <Refresh />
          </IconButton>
        </Toolbar>
      </AppBar>

      {/* Main 3-Column Dashboard */}
      <Box sx={{ flex: 1, display: 'flex', overflow: 'hidden', gap: 2, width: '100%', px: 2, py: 1 }}>
        {/* LEFT PANEL: Inputs & Controls */}
        <Box sx={{ width: '300px', minWidth: '300px', maxWidth: '300px', overflow: 'auto' }}>
          {/* Asset Overview */}
          <Card sx={{ mb: 2 }}>
            <CardHeader
              title="Asset Overview"
              titleTypographyProps={{ variant: 'subtitle1', fontWeight: 600 }}
            />
            <CardContent>
              <Grid container spacing={1}>
                <Grid item xs={12}>
                  <Typography variant="caption" color="text.secondary">Type</Typography>
                  <Typography variant="body2" fontWeight={500}>Concrete Bridge Pier</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">Risk Level</Typography>
                  <Chip label="Medium" color="warning" size="small" sx={{ mt: 0.5 }} />
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">Phase</Typography>
                  <Typography variant="body2">Construction QA</Typography>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="caption" color="text.secondary">Inspection Date</Typography>
                  <Typography variant="body2">2024-12-10</Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>

          {/* Data Collection Status */}
          <Card sx={{ mb: 2 }}>
            <CardHeader
              title="Data Collection"
              titleTypographyProps={{ variant: 'subtitle1', fontWeight: 600 }}
            />
            <List dense>
              <ListItem>
                <ListItemIcon>
                  <Terrain color="success" />
                </ListItemIcon>
                <ListItemText
                  primary={<Typography variant="body2" fontWeight={500}>LiDAR Scan</Typography>}
                  secondary={<Typography variant="caption">370,000 points • 1m resolution</Typography>}
                />
                <CheckCircle color="success" fontSize="small" />
              </ListItem>

              <ListItem>
                <ListItemIcon>
                  <Colorize color="success" />
                </ListItemIcon>
                <ListItemText
                  primary={<Typography variant="body2" fontWeight={500}>Hyperspectral Imaging</Typography>}
                  secondary={<Typography variant="caption">204 bands • 4 segments</Typography>}
                />
                <CheckCircle color="success" fontSize="small" />
              </ListItem>

              <ListItem>
                <ListItemIcon>
                  <ViewInAr color="success" />
                </ListItemIcon>
                <ListItemText
                  primary={<Typography variant="body2" fontWeight={500}>BIM Model</Typography>}
                  secondary={<Typography variant="caption">IFC format • Ready</Typography>}
                />
                <CheckCircle color="success" fontSize="small" />
              </ListItem>

              <ListItem>
                <ListItemIcon>
                  <CameraAlt />
                </ListItemIcon>
                <ListItemText
                  primary={<Typography variant="body2" fontWeight={500}>360° Imagery</Typography>}
                  secondary={<Typography variant="caption">23 images planned</Typography>}
                />
                <Schedule fontSize="small" />
              </ListItem>
            </List>
          </Card>

          {/* Segment Selector */}
          <Card>
            <CardHeader
              title="Segments"
              titleTypographyProps={{ variant: 'subtitle1', fontWeight: 600 }}
              action={
                <ButtonGroup size="small" variant="text">
                  <Button onClick={selectAllSegments}>All</Button>
                  <Button onClick={clearAllSegments}>Clear</Button>
                </ButtonGroup>
              }
            />
            <List dense>
              {SEGMENTS.map(segment => (
                <ListItem
                  key={segment.id}
                  button
                  selected={selectedSegments.includes(segment.id)}
                  onClick={() => toggleSegment(segment.id)}
                >
                  <ListItemIcon>
                    <Checkbox
                      edge="start"
                      checked={selectedSegments.includes(segment.id)}
                      size="small"
                    />
                  </ListItemIcon>
                  <ListItemText
                    primary={<Typography variant="body2" fontWeight={500}>{segment.face}</Typography>}
                    secondary={
                      <Typography variant="caption">
                        {(segment.lidar_points / 1000).toFixed(0)}K pts • {segment.hsi_coverage}% HSI
                      </Typography>
                    }
                  />
                  <Chip
                    label={segment.quality_score}
                    size="small"
                    color={getScoreColor(segment.quality_score)}
                  />
                </ListItem>
              ))}
            </List>
          </Card>
        </Box>

        {/* CENTER PANEL: Visualization */}
        <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <Paper elevation={2} sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
            {/* Visualization Mode Tabs */}
            <Tabs
              value={viewMode}
              onChange={(_, v) => setViewMode(v)}
              variant="fullWidth"
              sx={{
                borderBottom: 1,
                borderColor: 'divider',
                bgcolor: 'background.default'
              }}
            >
              <Tab
                label="3D Model"
                value="3d"
                icon={<ViewInAr />}
                iconPosition="start"
                sx={{ minHeight: 64, fontSize: '0.875rem', fontWeight: 600 }}
              />
              <Tab
                label="Point Cloud"
                value="pointcloud"
                icon={<Terrain />}
                iconPosition="start"
                sx={{ minHeight: 64, fontSize: '0.875rem', fontWeight: 600 }}
              />
              <Tab
                label="Heatmap"
                value="heatmap"
                icon={<Colorize />}
                iconPosition="start"
                sx={{ minHeight: 64, fontSize: '0.875rem', fontWeight: 600 }}
              />
              <Tab
                label="Defects"
                value="defects"
                icon={<Warning />}
                iconPosition="start"
                sx={{ minHeight: 64, fontSize: '0.875rem', fontWeight: 600 }}
              />
            </Tabs>

            {/* Main Viewport */}
            <Box sx={{ flex: 1, position: 'relative', bgcolor: 'grey.100', p: 2 }}>
              {/* Placeholder visualization */}
              <Box
                sx={{
                  height: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  border: '2px dashed',
                  borderColor: 'primary.main',
                  borderRadius: 2,
                  bgcolor: 'background.paper',
                  backgroundImage: 'linear-gradient(45deg, #f5f5f5 25%, transparent 25%, transparent 75%, #f5f5f5 75%, #f5f5f5), linear-gradient(45deg, #f5f5f5 25%, transparent 25%, transparent 75%, #f5f5f5 75%, #f5f5f5)',
                  backgroundSize: '20px 20px',
                  backgroundPosition: '0 0, 10px 10px'
                }}
              >
                <Box textAlign="center" sx={{ bgcolor: 'background.paper', p: 4, borderRadius: 2, boxShadow: 2 }}>
                  {viewMode === '3d' && <ViewInAr sx={{ fontSize: 100, color: 'primary.main' }} />}
                  {viewMode === 'pointcloud' && <Terrain sx={{ fontSize: 100, color: 'success.main' }} />}
                  {viewMode === 'heatmap' && <Colorize sx={{ fontSize: 100, color: 'warning.main' }} />}
                  {viewMode === 'defects' && <Warning sx={{ fontSize: 100, color: 'error.main' }} />}
                  <Typography variant="h5" fontWeight={600} mt={2} mb={1}>
                    {viewMode === '3d' && 'BIM 3D Model Viewer'}
                    {viewMode === 'pointcloud' && 'LiDAR Point Cloud Viewer'}
                    {viewMode === 'heatmap' && 'Hyperspectral Heatmap'}
                    {viewMode === 'defects' && 'Defect Detection Overlay'}
                  </Typography>

                  <Chip
                    label="🟢 REAL DATA Available"
                    color="success"
                    sx={{ mb: 2, fontWeight: 600 }}
                  />

                  <Typography variant="body2" color="text.secondary" mb={2}>
                    {viewMode === '3d' && 'Ready to integrate: Three.js / BabylonJS for BIM IFC rendering'}
                    {viewMode === 'pointcloud' && 'Ready to integrate: Potree for 370K LiDAR points visualization'}
                    {viewMode === 'heatmap' && 'Ready to integrate: D3.js heatmap overlay on 204-band HSI data'}
                    {viewMode === 'defects' && 'Ready to integrate: OpenCV/TensorFlow defect detection visualization'}
                  </Typography>

                  {selectedSegments.length > 0 ? (
                    <Box>
                      <Divider sx={{ my: 2 }} />
                      <Typography variant="body2" fontWeight={600} mb={1}>
                        Currently Selected:
                      </Typography>
                      <Typography variant="body2" color="primary.main">
                        {SEGMENTS.filter(s => selectedSegments.includes(s.id)).map(s => s.face).join(', ')}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Total: {SEGMENTS.filter(s => selectedSegments.includes(s.id))
                          .reduce((sum, s) => sum + s.lidar_points, 0)
                          .toLocaleString()} LiDAR points
                      </Typography>
                    </Box>
                  ) : (
                    <Box>
                      <Divider sx={{ my: 2 }} />
                      <Typography variant="body2" color="text.secondary">
                        ← Select segments from left panel to view data
                      </Typography>
                    </Box>
                  )}
                </Box>
              </Box>

              {/* Overlay Controls */}
              <Box sx={{ position: 'absolute', top: 16, right: 16 }}>
                <Paper elevation={3}>
                  <ToggleButtonGroup
                    size="small"
                    value={overlays}
                    onChange={(_, v) => setOverlays(v)}
                  >
                    <ToggleButton value="lidar">
                      <Tooltip title="LiDAR Overlay">
                        <Terrain fontSize="small" />
                      </Tooltip>
                    </ToggleButton>
                    <ToggleButton value="hsi">
                      <Tooltip title="HSI Overlay">
                        <Colorize fontSize="small" />
                      </Tooltip>
                    </ToggleButton>
                    <ToggleButton value="defects">
                      <Tooltip title="Defects Overlay">
                        <Warning fontSize="small" />
                      </Tooltip>
                    </ToggleButton>
                  </ToggleButtonGroup>
                </Paper>
              </Box>

              {/* Info Badge */}
              {selectedSegments.length > 0 && (
                <Box sx={{ position: 'absolute', bottom: 16, left: 16 }}>
                  <Paper elevation={2} sx={{ p: 1 }}>
                    <Typography variant="caption" color="text.secondary">
                      Selected: {selectedSegments.length} segment(s) • Total Area: {' '}
                      {SEGMENTS.filter(s => selectedSegments.includes(s.id))
                        .reduce((sum, s) => sum + s.area_sqm, 0)
                        .toFixed(1)} m²
                    </Typography>
                  </Paper>
                </Box>
              )}
            </Box>
          </Paper>
        </Box>

        {/* RIGHT PANEL: Analysis Results */}
        <Box sx={{ width: '360px', minWidth: '360px', maxWidth: '360px', overflow: 'auto' }}>
          <Paper elevation={2} sx={{ height: '100%' }}>
            {/* Analysis Tabs */}
            <Tabs
              value={analysisTab}
              onChange={(_, v) => setAnalysisTab(v)}
              variant="fullWidth"
              sx={{ borderBottom: 1, borderColor: 'divider' }}
            >
              <Tab label="Geometric" icon={<Straighten fontSize="small" />} />
              <Tab label="Material" icon={<Science fontSize="small" />} />
              <Tab label="Visual" icon={<Visibility fontSize="small" />} />
              <Tab label="Summary" icon={<Assessment fontSize="small" />} />
            </Tabs>

            {/* Tab Content */}
            <Box sx={{ p: 2 }}>
              {/* Geometric Analysis Tab */}
              {analysisTab === 0 && selectedSegments.length > 0 && (
                <Box>
                  {comparisonData.length === 1 ? (
                    // Single segment detailed view
                    <List dense>
                      <ListItem>
                        <ListItemText
                          primary={<Typography variant="body2" fontWeight={600}>Flatness</Typography>}
                          secondary={<Typography variant="caption">Tolerance: ±5mm</Typography>}
                        />
                        <Box display="flex" alignItems="center">
                          <Typography variant="h6" color="success.main" fontWeight={600}>
                            {comparisonData[0].flatness_mm} mm
                          </Typography>
                          <CheckCircle color="success" fontSize="small" sx={{ ml: 0.5 }} />
                        </Box>
                      </ListItem>

                      <ListItem>
                        <ListItemText
                          primary={<Typography variant="body2" fontWeight={600}>Verticality</Typography>}
                          secondary={<Typography variant="caption">Tolerance: ±3mm</Typography>}
                        />
                        <Box display="flex" alignItems="center">
                          <Typography variant="h6" color="success.main" fontWeight={600}>
                            {comparisonData[0].verticality_mm} mm
                          </Typography>
                          <CheckCircle color="success" fontSize="small" sx={{ ml: 0.5 }} />
                        </Box>
                      </ListItem>

                      <Divider sx={{ my: 1 }} />

                      <ListItem>
                        <Box width="100%">
                          <Typography variant="caption" color="text.secondary">
                            As-Built Deviation
                          </Typography>
                          <LinearProgress
                            variant="determinate"
                            value={(comparisonData[0].flatness_mm / 10) * 100}
                            color="success"
                            sx={{ my: 1 }}
                          />
                          <Typography variant="caption">
                            {comparisonData[0].flatness_mm}mm / 10mm max tolerance
                          </Typography>
                        </Box>
                      </ListItem>
                    </List>
                  ) : (
                    // Multi-segment comparison view
                    <Box>
                      <Typography variant="subtitle2" gutterBottom fontWeight={600}>
                        Multi-Segment Comparison
                      </Typography>
                      <Typography variant="caption" color="text.secondary" display="block" mb={2}>
                        Comparing {comparisonData.length} segments
                      </Typography>

                      {/* Flatness Comparison */}
                      <Card variant="outlined" sx={{ mb: 2 }}>
                        <CardContent>
                          <Typography variant="caption" fontWeight={600} display="block" mb={1}>
                            Flatness (mm)
                          </Typography>
                          <ResponsiveContainer width="100%" height={120}>
                            <BarChart data={comparisonData}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="face" tick={{ fontSize: 10 }} />
                              <YAxis domain={[0, 5]} />
                              <RechartsTooltip />
                              <ReferenceLine y={5} stroke="red" strokeDasharray="3 3" label="Max" />
                              <Bar dataKey="flatness_mm" fill="#4caf50" />
                            </BarChart>
                          </ResponsiveContainer>
                        </CardContent>
                      </Card>

                      {/* Verticality Comparison */}
                      <Card variant="outlined">
                        <CardContent>
                          <Typography variant="caption" fontWeight={600} display="block" mb={1}>
                            Verticality (mm)
                          </Typography>
                          <ResponsiveContainer width="100%" height={120}>
                            <BarChart data={comparisonData}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="face" tick={{ fontSize: 10 }} />
                              <YAxis domain={[0, 3]} />
                              <RechartsTooltip />
                              <ReferenceLine y={3} stroke="red" strokeDasharray="3 3" label="Max" />
                              <Bar dataKey="verticality_mm" fill="#2196f3" />
                            </BarChart>
                          </ResponsiveContainer>
                        </CardContent>
                      </Card>
                    </Box>
                  )}
                </Box>
              )}

              {/* Material Analysis Tab */}
              {analysisTab === 1 && selectedSegments.length > 0 && (
                <Box>
                  {comparisonData.length === 1 ? (
                    // Single segment detailed view
                    <>
                      <Card variant="outlined" sx={{ mb: 2 }}>
                        <CardContent>
                          <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                            Concrete Strength
                          </Typography>
                          <Box display="flex" justifyContent="center" my={2}>
                            <Box position="relative" display="inline-flex">
                              <CircularProgress
                                variant="determinate"
                                value={(comparisonData[0].strength_mpa / 60) * 100}
                                size={120}
                                thickness={4}
                                color={comparisonData[0].strength_mpa >= 40 ? 'success' : 'warning'}
                              />
                              <Box
                                sx={{
                                  top: 0,
                                  left: 0,
                                  bottom: 0,
                                  right: 0,
                                  position: 'absolute',
                                  display: 'flex',
                                  flexDirection: 'column',
                                  alignItems: 'center',
                                  justifyContent: 'center',
                                }}
                              >
                                <Typography variant="h4" fontWeight={600}>
                                  {comparisonData[0].strength_mpa}
                                </Typography>
                                <Typography variant="caption">MPa</Typography>
                              </Box>
                            </Box>
                          </Box>
                          <Chip
                            label={comparisonData[0].strength_mpa >= 40 ? 'Meets C40 Spec (≥40 MPa)' : 'Below C40 Spec'}
                            color={comparisonData[0].strength_mpa >= 40 ? 'success' : 'warning'}
                            size="small"
                            sx={{ width: '100%' }}
                          />
                        </CardContent>
                      </Card>

                      <List dense>
                        <ListItem>
                          <ListItemText primary="Moisture Content" />
                          <Typography variant="body2" fontWeight={600}>
                            {comparisonData[0].moisture_pct}%
                          </Typography>
                        </ListItem>
                        <ListItem>
                          <ListItemText primary="Aggregate Quality" />
                          <Chip label="Good" color="success" size="small" />
                        </ListItem>
                        <ListItem>
                          <ListItemText primary="Carbonation Depth" />
                          <Typography variant="body2" fontWeight={600}>1.2 mm</Typography>
                        </ListItem>
                        <ListItem>
                          <ListItemText primary="Chloride Content" />
                          <Chip label="Low" color="success" size="small" />
                        </ListItem>
                      </List>

                      <Card variant="outlined" sx={{ mt: 2 }}>
                        <CardContent>
                          <Typography variant="caption" fontWeight={600} display="block" mb={1}>
                            Spectral Signature
                          </Typography>
                          <ResponsiveContainer width="100%" height={100}>
                            <LineChart data={SPECTRAL_DATA}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="wavelength" tick={{ fontSize: 8 }} />
                              <YAxis domain={[0, 1]} tick={{ fontSize: 8 }} />
                              <RechartsTooltip />
                              <Line type="monotone" dataKey="reflectance" stroke="#ff9800" strokeWidth={2} dot={false} />
                              <ReferenceLine y={0.5} stroke="green" strokeDasharray="3 3" label={{ value: 'Baseline', fontSize: 8 }} />
                            </LineChart>
                          </ResponsiveContainer>
                        </CardContent>
                      </Card>
                    </>
                  ) : (
                    // Multi-segment comparison
                    <Box>
                      <Typography variant="subtitle2" gutterBottom fontWeight={600}>
                        Strength Comparison
                      </Typography>
                      <Card variant="outlined">
                        <CardContent>
                          <ResponsiveContainer width="100%" height={150}>
                            <BarChart data={comparisonData}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="face" tick={{ fontSize: 10 }} />
                              <YAxis domain={[0, 60]} />
                              <RechartsTooltip />
                              <ReferenceLine y={40} stroke="green" strokeDasharray="3 3" label="C40 Spec" />
                              <Bar dataKey="strength_mpa" fill="#ff9800" />
                            </BarChart>
                          </ResponsiveContainer>
                        </CardContent>
                      </Card>
                    </Box>
                  )}
                </Box>
              )}

              {/* Visual Defects Tab */}
              {analysisTab === 2 && (
                <Box>
                  <Typography variant="subtitle2" gutterBottom fontWeight={600}>
                    Detected Defects
                  </Typography>
                  {DEFECTS.filter(d => selectedSegments.includes(d.segment)).length > 0 ? (
                    <List>
                      {DEFECTS.filter(d => selectedSegments.includes(d.segment)).map(defect => (
                        <ListItem key={defect.id} button>
                          <ListItemIcon>
                            <Avatar sx={{ bgcolor: getSeverityColor(defect.severity), width: 32, height: 32 }}>
                              {defect.severity[0]}
                            </Avatar>
                          </ListItemIcon>
                          <ListItemText
                            primary={
                              <Box display="flex" alignItems="center" gap={1}>
                                <Typography variant="body2" fontWeight={600}>{defect.type}</Typography>
                                <Chip label={defect.severity} size="small" />
                              </Box>
                            }
                            secondary={
                              <Typography variant="caption">
                                {defect.area_sqcm ? `${defect.area_sqcm}cm²` : `${defect.length_mm}mm`}
                                {' • '}Location: ({defect.location[0]}, {defect.location[1]})
                              </Typography>
                            }
                          />
                          <IconButton size="small">
                            <ZoomIn fontSize="small" />
                          </IconButton>
                        </ListItem>
                      ))}
                    </List>
                  ) : (
                    <Box textAlign="center" py={4}>
                      <CheckCircle sx={{ fontSize: 60, color: 'success.main' }} />
                      <Typography variant="body2" color="text.secondary" mt={1}>
                        No defects detected in selected segments
                      </Typography>
                    </Box>
                  )}
                </Box>
              )}

              {/* Summary Tab */}
              {analysisTab === 3 && selectedSegments.length > 0 && (
                <Box>
                  <Typography variant="subtitle2" gutterBottom fontWeight={600}>
                    Inspection Summary
                  </Typography>

                  <Card variant="outlined" sx={{ mb: 2 }}>
                    <CardContent>
                      <Typography variant="caption" color="text.secondary">Overall Quality Score</Typography>
                      <Box display="flex" alignItems="center" my={1}>
                        <LinearProgress
                          variant="determinate"
                          value={avgQualityScore}
                          sx={{ flex: 1, height: 10, borderRadius: 1, mr: 2 }}
                          color={getScoreColor(avgQualityScore)}
                        />
                        <Typography variant="h5" fontWeight={600}>
                          {avgQualityScore}/100
                        </Typography>
                      </Box>
                      <Chip
                        label={avgQualityScore >= 85 ? 'GOOD QUALITY' : avgQualityScore >= 70 ? 'FAIR QUALITY' : 'POOR QUALITY'}
                        color={getScoreColor(avgQualityScore)}
                        size="small"
                      />
                    </CardContent>
                  </Card>

                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Card variant="outlined">
                        <CardContent>
                          <Typography variant="caption" color="text.secondary">Segments</Typography>
                          <Typography variant="h4" fontWeight={600}>{selectedSegments.length}</Typography>
                          <Typography variant="caption">of 4 total</Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                    <Grid item xs={6}>
                      <Card variant="outlined">
                        <CardContent>
                          <Typography variant="caption" color="text.secondary">Defects</Typography>
                          <Typography variant="h4" fontWeight={600}>
                            {DEFECTS.filter(d => selectedSegments.includes(d.segment)).length}
                          </Typography>
                          <Typography variant="caption">detected</Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                    <Grid item xs={6}>
                      <Card variant="outlined">
                        <CardContent>
                          <Typography variant="caption" color="text.secondary">Avg Strength</Typography>
                          <Typography variant="h4" fontWeight={600}>
                            {(comparisonData.reduce((sum, s) => sum + s.strength_mpa, 0) / comparisonData.length).toFixed(1)}
                          </Typography>
                          <Typography variant="caption">MPa</Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                    <Grid item xs={6}>
                      <Card variant="outlined">
                        <CardContent>
                          <Typography variant="caption" color="text.secondary">Coverage</Typography>
                          <Typography variant="h4" fontWeight={600}>
                            {Math.round(comparisonData.reduce((sum, s) => sum + s.hsi_coverage, 0) / comparisonData.length)}%
                          </Typography>
                          <Typography variant="caption">HSI</Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                  </Grid>

                  <Card variant="outlined" sx={{ mt: 2 }}>
                    <CardContent>
                      <Typography variant="caption" fontWeight={600} display="block" mb={1}>
                        Quality Trend
                      </Typography>
                      <ResponsiveContainer width="100%" height={100}>
                        <LineChart data={comparisonData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="face" tick={{ fontSize: 8 }} />
                          <YAxis domain={[0, 100]} tick={{ fontSize: 8 }} />
                          <RechartsTooltip />
                          <Line type="monotone" dataKey="quality_score" stroke="#2196f3" strokeWidth={2} />
                        </LineChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>
                </Box>
              )}

              {selectedSegments.length === 0 && (
                <Box textAlign="center" py={4}>
                  <Info sx={{ fontSize: 60, color: 'info.main' }} />
                  <Typography variant="body2" color="text.secondary" mt={2}>
                    Select segments from the left panel to view analysis results
                  </Typography>
                </Box>
              )}
            </Box>
          </Paper>
        </Box>
      </Box>
    </Box>
  );
};

export default DashboardIntegratedDemo;
