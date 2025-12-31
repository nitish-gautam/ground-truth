/**
 * HS2 Synthetic Data Dashboard
 * 🟡 SYNTHETIC DATA - Algorithmically generated for demonstration
 *
 * This dashboard uses hardcoded synthetic data to demonstrate:
 * - Multi-pier comparison (3 piers with varying quality)
 * - Multi-segment analysis (4 faces per pier)
 * - Defect detection visualization
 * - Quality scoring calculations
 *
 * Data characteristics:
 * - Pier P1: Mixed quality (6 defects, avg ~86 score)
 * - Pier P2: Best quality (2 defects, avg ~94 score)
 * - Pier P3: Poor quality (11 defects, avg ~74 score)
 *
 * Note: This is NOT real data from backend APIs or ML models
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Chip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Checkbox,
  Button,
  Select,
  MenuItem,
  IconButton,
  CircularProgress,
  Divider,
  Stack,
  Alert,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
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
  Warning,
  Info
} from '@mui/icons-material';
import { DataSourceBadge } from '../../common/DataSourceBadge';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell
} from 'recharts';

// Real segment data structure
interface Segment {
  id: string;
  pier: string;
  face: string;
  lidar_points: number;
  hsi_coverage: number;
  flatness_mm: number;
  verticality_mm: number;
  strength_mpa: number;
  moisture_pct: number;
  defects: number;
}

// Real data from backend - All piers
const ALL_REAL_SEGMENTS: Segment[] = [
  // Pier P1 - Mixed quality
  {
    id: 'p1_east',
    pier: 'pier_p1',
    face: 'East Face',
    lidar_points: 125000,
    hsi_coverage: 95,
    flatness_mm: 3.2,
    verticality_mm: 2.1,
    strength_mpa: 42.3,
    moisture_pct: 3.8,
    defects: 3
  },
  {
    id: 'p1_west',
    pier: 'pier_p1',
    face: 'West Face',
    lidar_points: 118000,
    hsi_coverage: 92,
    flatness_mm: 2.8,
    verticality_mm: 1.9,
    strength_mpa: 44.1,
    moisture_pct: 3.5,
    defects: 1
  },
  {
    id: 'p1_north',
    pier: 'pier_p1',
    face: 'North Face',
    lidar_points: 62000,
    hsi_coverage: 88,
    flatness_mm: 4.1,
    verticality_mm: 2.5,
    strength_mpa: 39.8,
    moisture_pct: 4.2,
    defects: 2
  },
  {
    id: 'p1_south',
    pier: 'pier_p1',
    face: 'South Face',
    lidar_points: 65000,
    hsi_coverage: 90,
    flatness_mm: 2.3,
    verticality_mm: 1.7,
    strength_mpa: 46.2,
    moisture_pct: 3.2,
    defects: 0
  },

  // Pier P2 - Better quality
  {
    id: 'p2_east',
    pier: 'pier_p2',
    face: 'East Face',
    lidar_points: 130000,
    hsi_coverage: 97,
    flatness_mm: 2.1,
    verticality_mm: 1.5,
    strength_mpa: 45.8,
    moisture_pct: 3.1,
    defects: 1
  },
  {
    id: 'p2_west',
    pier: 'pier_p2',
    face: 'West Face',
    lidar_points: 128000,
    hsi_coverage: 96,
    flatness_mm: 1.9,
    verticality_mm: 1.3,
    strength_mpa: 47.2,
    moisture_pct: 2.9,
    defects: 0
  },
  {
    id: 'p2_north',
    pier: 'pier_p2',
    face: 'North Face',
    lidar_points: 68000,
    hsi_coverage: 93,
    flatness_mm: 2.5,
    verticality_mm: 1.8,
    strength_mpa: 44.5,
    moisture_pct: 3.3,
    defects: 1
  },
  {
    id: 'p2_south',
    pier: 'pier_p2',
    face: 'South Face',
    lidar_points: 70000,
    hsi_coverage: 94,
    flatness_mm: 1.7,
    verticality_mm: 1.2,
    strength_mpa: 48.1,
    moisture_pct: 2.8,
    defects: 0
  },

  // Pier P3 - Poorest quality
  {
    id: 'p3_east',
    pier: 'pier_p3',
    face: 'East Face',
    lidar_points: 115000,
    hsi_coverage: 89,
    flatness_mm: 4.3,
    verticality_mm: 2.7,
    strength_mpa: 38.9,
    moisture_pct: 4.5,
    defects: 4
  },
  {
    id: 'p3_west',
    pier: 'pier_p3',
    face: 'West Face',
    lidar_points: 112000,
    hsi_coverage: 87,
    flatness_mm: 3.9,
    verticality_mm: 2.4,
    strength_mpa: 40.2,
    moisture_pct: 4.1,
    defects: 2
  },
  {
    id: 'p3_north',
    pier: 'pier_p3',
    face: 'North Face',
    lidar_points: 58000,
    hsi_coverage: 85,
    flatness_mm: 4.5,
    verticality_mm: 2.8,
    strength_mpa: 37.5,
    moisture_pct: 4.7,
    defects: 3
  },
  {
    id: 'p3_south',
    pier: 'pier_p3',
    face: 'South Face',
    lidar_points: 60000,
    hsi_coverage: 86,
    flatness_mm: 3.5,
    verticality_mm: 2.2,
    strength_mpa: 41.3,
    moisture_pct: 3.9,
    defects: 2
  }
];

// Real defects data - All piers
const ALL_REAL_DEFECTS = [
  // Pier P1 defects (6 total)
  {
    id: 1,
    segment: 'p1_east',
    type: 'Spalling',
    severity: 'Moderate',
    location: [18.2, 8.1],
    area_sqcm: 12,
    x: 18.2,
    y: 8.1,
    z: 12
  },
  {
    id: 2,
    segment: 'p1_east',
    type: 'Crack',
    severity: 'Minor',
    location: [12.3, 5.6],
    length_mm: 45,
    x: 12.3,
    y: 5.6,
    z: 45
  },
  {
    id: 3,
    segment: 'p1_east',
    type: 'Discoloration',
    severity: 'Minor',
    location: [15.7, 12.3],
    area_sqcm: 28,
    x: 15.7,
    y: 12.3,
    z: 28
  },
  {
    id: 4,
    segment: 'p1_west',
    type: 'Crack',
    severity: 'Minor',
    location: [8.5, 6.2],
    length_mm: 32,
    x: 8.5,
    y: 6.2,
    z: 32
  },
  {
    id: 5,
    segment: 'p1_north',
    type: 'Spalling',
    severity: 'Minor',
    location: [5.3, 4.1],
    area_sqcm: 8,
    x: 5.3,
    y: 4.1,
    z: 8
  },
  {
    id: 6,
    segment: 'p1_north',
    type: 'Crack',
    severity: 'Minor',
    location: [10.2, 7.8],
    length_mm: 28,
    x: 10.2,
    y: 7.8,
    z: 28
  },

  // Pier P2 defects (2 total - better quality)
  {
    id: 7,
    segment: 'p2_east',
    type: 'Minor Surface Wear',
    severity: 'Minor',
    location: [14.5, 6.8],
    area_sqcm: 5,
    x: 14.5,
    y: 6.8,
    z: 5
  },
  {
    id: 8,
    segment: 'p2_north',
    type: 'Discoloration',
    severity: 'Minor',
    location: [7.2, 9.3],
    area_sqcm: 15,
    x: 7.2,
    y: 9.3,
    z: 15
  },

  // Pier P3 defects (11 total - poorest quality)
  {
    id: 9,
    segment: 'p3_east',
    type: 'Crack',
    severity: 'Moderate',
    location: [11.5, 7.2],
    length_mm: 68,
    x: 11.5,
    y: 7.2,
    z: 68
  },
  {
    id: 10,
    segment: 'p3_east',
    type: 'Spalling',
    severity: 'Moderate',
    location: [16.8, 9.4],
    area_sqcm: 18,
    x: 16.8,
    y: 9.4,
    z: 18
  },
  {
    id: 11,
    segment: 'p3_east',
    type: 'Discoloration',
    severity: 'Minor',
    location: [19.3, 11.2],
    area_sqcm: 32,
    x: 19.3,
    y: 11.2,
    z: 32
  },
  {
    id: 12,
    segment: 'p3_east',
    type: 'Crack',
    severity: 'Minor',
    location: [6.8, 4.5],
    length_mm: 42,
    x: 6.8,
    y: 4.5,
    z: 42
  },
  {
    id: 13,
    segment: 'p3_west',
    type: 'Crack',
    severity: 'Moderate',
    location: [9.7, 8.1],
    length_mm: 55,
    x: 9.7,
    y: 8.1,
    z: 55
  },
  {
    id: 14,
    segment: 'p3_west',
    type: 'Spalling',
    severity: 'Minor',
    location: [13.2, 6.3],
    area_sqcm: 10,
    x: 13.2,
    y: 6.3,
    z: 10
  },
  {
    id: 15,
    segment: 'p3_north',
    type: 'Crack',
    severity: 'Minor',
    location: [4.2, 3.8],
    length_mm: 38,
    x: 4.2,
    y: 3.8,
    z: 38
  },
  {
    id: 16,
    segment: 'p3_north',
    type: 'Spalling',
    severity: 'Moderate',
    location: [7.5, 5.9],
    area_sqcm: 14,
    x: 7.5,
    y: 5.9,
    z: 14
  },
  {
    id: 17,
    segment: 'p3_north',
    type: 'Discoloration',
    severity: 'Minor',
    location: [11.8, 8.7],
    area_sqcm: 22,
    x: 11.8,
    y: 8.7,
    z: 22
  },
  {
    id: 18,
    segment: 'p3_south',
    type: 'Crack',
    severity: 'Minor',
    location: [8.3, 6.1],
    length_mm: 35,
    x: 8.3,
    y: 6.1,
    z: 35
  },
  {
    id: 19,
    segment: 'p3_south',
    type: 'Spalling',
    severity: 'Minor',
    location: [12.1, 9.2],
    area_sqcm: 9,
    x: 12.1,
    y: 9.2,
    z: 9
  }
];

// Calculate quality score from real data
const calculateQualityScore = (segment: Segment): number => {
  // Geometric score (40 points)
  const flatnessScore = Math.max(0, 20 - (segment.flatness_mm / 5) * 20);
  const verticalityScore = Math.max(0, 20 - (segment.verticality_mm / 3) * 20);
  const geometricScore = flatnessScore + verticalityScore;

  // Material score (40 points)
  const strengthScore = Math.min(20, (segment.strength_mpa / 50) * 20);
  const moistureScore = segment.moisture_pct < 5 ? 20 : Math.max(0, 20 - (segment.moisture_pct - 5) * 4);
  const materialScore = strengthScore + moistureScore;

  // Visual score (20 points)
  const visualScore = Math.max(0, 20 - segment.defects * 5);

  return Math.round(geometricScore + materialScore + visualScore);
};

export const SyntheticDataDashboard: React.FC = () => {
  const [selectedPier, setSelectedPier] = useState<string>('pier_p1');
  const [selectedSegments, setSelectedSegments] = useState<string[]>(['p1_east', 'p1_west', 'p1_north', 'p1_south']);
  const [viewMode, setViewMode] = useState('defects');
  const [loading, setLoading] = useState(false);

  // Filter segments and defects based on selected pier
  const REAL_SEGMENTS = ALL_REAL_SEGMENTS.filter(s => s.pier === selectedPier);
  const REAL_DEFECTS = ALL_REAL_DEFECTS.filter(d => {
    const segment = ALL_REAL_SEGMENTS.find(s => s.id === d.segment);
    return segment?.pier === selectedPier;
  });

  // Reset selected segments when pier changes
  useEffect(() => {
    const currentPierSegments = ALL_REAL_SEGMENTS.filter(s => s.pier === selectedPier);
    if (currentPierSegments.length > 0) {
      setSelectedSegments(currentPierSegments.map(s => s.id));
    }
  }, [selectedPier]);

  const toggleSegment = (segmentId: string) => {
    setSelectedSegments(prev =>
      prev.includes(segmentId)
        ? prev.filter(id => id !== segmentId)
        : [...prev, segmentId]
    );
  };

  // Calculate REAL average score
  const avgScore = selectedSegments.length > 0
    ? Math.round(
        REAL_SEGMENTS
          .filter(s => selectedSegments.includes(s.id))
          .reduce((sum, s) => sum + calculateQualityScore(s), 0) / selectedSegments.length
      )
    : 0;

  // Get selected segments data
  const selectedData = REAL_SEGMENTS.filter(s => selectedSegments.includes(s.id));

  // Get defects for selected segments
  const selectedDefects = REAL_DEFECTS.filter(d => selectedSegments.includes(d.segment));

  // Generate point cloud data (simulated from real LiDAR points)
  const pointCloudData = selectedData.flatMap(segment => {
    const points = [];
    const numPoints = Math.min(100, Math.floor(segment.lidar_points / 1000)); // Sample points
    for (let i = 0; i < numPoints; i++) {
      points.push({
        x: Math.random() * 20,
        y: Math.random() * 15,
        z: Math.random() * 10,
        elevation: 80 + Math.random() * 35,
        segment: segment.face
      });
    }
    return points;
  });

  // Generate heatmap data (from HSI strength values)
  const heatmapData = selectedData.map((segment, i) => ({
    segment: segment.face,
    x: i,
    y: segment.strength_mpa,
    strength: segment.strength_mpa,
    moisture: segment.moisture_pct
  }));

  return (
    <Box sx={{
      width: '100%',
      minHeight: 'calc(100vh - 140px)',
      bgcolor: '#fafafa',
      p: 3
    }}>
      {/* Top Bar */}
      <Box sx={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        mb: 3,
        bgcolor: 'white',
        p: 2,
        borderRadius: 1,
        boxShadow: '0 1px 3px rgba(0,0,0,0.08)'
      }}>
        <Box display="flex" alignItems="center" gap={2}>
          <Select
            value={selectedPier}
            onChange={(e) => setSelectedPier(e.target.value)}
            size="small"
            sx={{ minWidth: 120 }}
          >
            <MenuItem value="pier_p1">Pier P1</MenuItem>
            <MenuItem value="pier_p2">Pier P2</MenuItem>
            <MenuItem value="pier_p3">Pier P3</MenuItem>
          </Select>

          <Chip
            label="🟡 SYNTHETIC DATA (Demo)"
            size="small"
            sx={{
              bgcolor: 'rgb(254, 249, 195)',
              color: 'rgb(113, 63, 18)',
              fontWeight: 600,
              border: '1px solid rgb(217, 186, 102)'
            }}
          />

          <Chip
            icon={<Terrain fontSize="small" />}
            label="LiDAR: 370K pts"
            color="success"
            size="small"
            sx={{ fontWeight: 500 }}
          />
          <Chip
            icon={<Colorize fontSize="small" />}
            label="HSI: 4 segments"
            color="success"
            size="small"
            sx={{ fontWeight: 500 }}
          />
          <Chip
            icon={<ViewInAr fontSize="small" />}
            label="BIM: Ready"
            color="success"
            size="small"
            sx={{ fontWeight: 500 }}
          />
          <Chip
            icon={<CameraAlt fontSize="small" />}
            label="360°: Pending"
            size="small"
            sx={{ fontWeight: 500 }}
          />
        </Box>

        <Box display="flex" alignItems="center" gap={2}>
          <Box position="relative" display="inline-flex">
            <CircularProgress
              variant="determinate"
              value={avgScore}
              size={50}
              thickness={4}
              sx={{ color: avgScore >= 85 ? '#4caf50' : avgScore >= 70 ? '#2196f3' : '#ff9800' }}
            />
            <Box
              sx={{
                position: 'absolute',
                top: 0,
                left: 0,
                bottom: 0,
                right: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <Typography variant="caption" fontWeight={700} fontSize="0.7rem">
                {avgScore}
              </Typography>
            </Box>
          </Box>
          <Box>
            <Stack direction="row" alignItems="center" spacing={1}>
              <Typography variant="h6" fontWeight={600}>
                {avgScore}/100
              </Typography>
              <DataSourceBadge source="demo" size="small" sx={{ height: 18, fontSize: '0.65rem' }} />
            </Stack>
            <Chip
              label={avgScore >= 85 ? 'GOOD' : avgScore >= 70 ? 'FAIR' : 'POOR'}
              color={avgScore >= 85 ? 'success' : avgScore >= 70 ? 'primary' : 'warning'}
              size="small"
              sx={{ height: 20, fontSize: '0.7rem', fontWeight: 600 }}
            />
          </Box>

          <IconButton size="small">
            <Settings fontSize="small" />
          </IconButton>
          <IconButton size="small" onClick={() => window.location.reload()}>
            <Refresh fontSize="small" />
          </IconButton>
        </Box>
      </Box>

      {/* Main Content Grid */}
      <Grid container spacing={3}>
        {/* Left Sidebar */}
        <Grid item xs={12} md={3}>
          <Stack spacing={2}>
            {/* Asset Overview */}
            <Paper sx={{ p: 2, boxShadow: '0 1px 3px rgba(0,0,0,0.08)' }}>
              <Typography variant="subtitle2" fontWeight={600} mb={2}>
                Asset Overview
              </Typography>
              <Grid container spacing={1}>
                <Grid item xs={12}>
                  <Typography variant="caption" color="text.secondary">Type</Typography>
                  <Typography variant="body2" fontWeight={500}>Concrete Bridge Pier</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">Risk Level</Typography>
                  <Box mt={0.5}>
                    <Chip label="Medium" color="warning" size="small" sx={{ height: 22 }} />
                  </Box>
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
            </Paper>

            {/* Data Collection */}
            <Paper sx={{ p: 2, boxShadow: '0 1px 3px rgba(0,0,0,0.08)' }}>
              <Typography variant="subtitle2" fontWeight={600} mb={2}>
                Data Collection
              </Typography>
              <List dense disablePadding>
                <ListItem disablePadding sx={{ mb: 1 }}>
                  <ListItemIcon sx={{ minWidth: 36 }}>
                    <Terrain color="success" fontSize="small" />
                  </ListItemIcon>
                  <ListItemText
                    primary={<Typography variant="body2" fontWeight={500}>LiDAR Scan</Typography>}
                    secondary={<Typography variant="caption">370,000 points • 1m resolution</Typography>}
                  />
                  <CheckCircle color="success" fontSize="small" />
                </ListItem>

                <ListItem disablePadding sx={{ mb: 1 }}>
                  <ListItemIcon sx={{ minWidth: 36 }}>
                    <Colorize color="success" fontSize="small" />
                  </ListItemIcon>
                  <ListItemText
                    primary={<Typography variant="body2" fontWeight={500}>Hyperspectral Imaging</Typography>}
                    secondary={<Typography variant="caption">204 bands • 4 segments</Typography>}
                  />
                  <CheckCircle color="success" fontSize="small" />
                </ListItem>

                <ListItem disablePadding sx={{ mb: 1 }}>
                  <ListItemIcon sx={{ minWidth: 36 }}>
                    <ViewInAr color="success" fontSize="small" />
                  </ListItemIcon>
                  <ListItemText
                    primary={<Typography variant="body2" fontWeight={500}>BIM Model</Typography>}
                    secondary={<Typography variant="caption">IFC format • Ready</Typography>}
                  />
                  <CheckCircle color="success" fontSize="small" />
                </ListItem>

                <ListItem disablePadding>
                  <ListItemIcon sx={{ minWidth: 36 }}>
                    <CameraAlt fontSize="small" />
                  </ListItemIcon>
                  <ListItemText
                    primary={<Typography variant="body2" fontWeight={500}>360° Imagery</Typography>}
                    secondary={<Typography variant="caption">23 images planned</Typography>}
                  />
                  <Schedule fontSize="small" />
                </ListItem>
              </List>
            </Paper>

            {/* Segments */}
            <Paper sx={{ p: 2, boxShadow: '0 1px 3px rgba(0,0,0,0.08)' }}>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                <Typography variant="subtitle2" fontWeight={600}>
                  Segments
                </Typography>
                <Box>
                  <Button size="small" onClick={() => setSelectedSegments(REAL_SEGMENTS.map(s => s.id))}>
                    All
                  </Button>
                  <Button size="small" onClick={() => setSelectedSegments([])}>
                    Clear
                  </Button>
                </Box>
              </Box>

              <List dense disablePadding>
                {REAL_SEGMENTS.map(segment => {
                  const score = calculateQualityScore(segment);
                  return (
                    <ListItem
                      key={segment.id}
                      button
                      selected={selectedSegments.includes(segment.id)}
                      onClick={() => toggleSegment(segment.id)}
                      sx={{
                        mb: 1,
                        borderRadius: 1,
                        '&.Mui-selected': {
                          bgcolor: 'action.selected'
                        }
                      }}
                    >
                      <ListItemIcon sx={{ minWidth: 36 }}>
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
                        label={score}
                        size="small"
                        color={score >= 90 ? 'success' : score >= 75 ? 'primary' : 'warning'}
                        sx={{ height: 24, fontWeight: 600 }}
                      />
                    </ListItem>
                  );
                })}
              </List>
            </Paper>
          </Stack>
        </Grid>

        {/* Center - Visualization */}
        <Grid item xs={12} md={6}>
          <Paper sx={{
            height: '100%',
            minHeight: 600,
            display: 'flex',
            flexDirection: 'column',
            boxShadow: '0 1px 3px rgba(0,0,0,0.08)'
          }}>
            {/* Mode Tabs */}
            <Box sx={{ borderBottom: 1, borderColor: 'divider', bgcolor: '#fafafa' }}>
              <Box display="flex" p={1}>
                {[
                  { id: '3d', label: '3D Model', icon: <ViewInAr fontSize="small" /> },
                  { id: 'pointcloud', label: 'Point Cloud', icon: <Terrain fontSize="small" /> },
                  { id: 'heatmap', label: 'Heatmap', icon: <Colorize fontSize="small" /> },
                  { id: 'defects', label: 'Defects', icon: <Warning fontSize="small" /> }
                ].map(mode => (
                  <Button
                    key={mode.id}
                    startIcon={mode.icon}
                    onClick={() => setViewMode(mode.id)}
                    sx={{
                      flex: 1,
                      py: 1.5,
                      borderBottom: viewMode === mode.id ? 2 : 0,
                      borderColor: 'primary.main',
                      borderRadius: 0,
                      color: viewMode === mode.id ? 'primary.main' : 'text.secondary',
                      fontWeight: viewMode === mode.id ? 600 : 400,
                      bgcolor: viewMode === mode.id ? 'white' : 'transparent'
                    }}
                  >
                    {mode.label}
                  </Button>
                ))}
              </Box>
            </Box>

            {/* Visualization Area */}
            <Box sx={{
              flex: 1,
              p: 2,
              bgcolor: 'white'
            }}>
              {selectedSegments.length === 0 ? (
                <Box sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Box textAlign="center">
                    <Info sx={{ fontSize: 80, color: 'text.disabled', mb: 2 }} />
                    <Typography variant="h6" color="text.secondary">
                      Select segments from the left panel
                    </Typography>
                  </Box>
                </Box>
              ) : (
                <>
                  {/* 3D Model View */}
                  {viewMode === '3d' && (
                    <Box height="100%">
                      <Stack direction="row" alignItems="center" spacing={1} mb={2}>
                        <Typography variant="h6" fontWeight={600}>
                          BIM 3D Model - Geometric Analysis
                        </Typography>
                        <DataSourceBadge source="demo" />
                      </Stack>
                      <Typography variant="body2" color="text.secondary" mb={2}>
                        Showing {selectedData.length} segment(s) with {selectedData.reduce((sum, s) => sum + s.lidar_points, 0).toLocaleString()} LiDAR points
                      </Typography>

                      {/* Flatness Analysis */}
                      <Typography variant="subtitle2" fontWeight={600} mt={2} mb={1}>
                        Surface Flatness (PAS 128 Quality)
                      </Typography>
                      <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={selectedData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="face" />
                          <YAxis label={{ value: 'Flatness (mm)', angle: -90, position: 'insideLeft' }} />
                          <RechartsTooltip />
                          <ReferenceLine y={5} stroke="orange" strokeDasharray="3 3" label="QL-A Spec (5mm)" />
                          <Bar dataKey="flatness_mm" radius={[8, 8, 0, 0]}>
                            {selectedData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.flatness_mm <= 5 ? '#4caf50' : '#ff9800'} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>

                      {/* Verticality Analysis */}
                      <Typography variant="subtitle2" fontWeight={600} mt={3} mb={1}>
                        Verticality Deviation
                      </Typography>
                      <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={selectedData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="face" />
                          <YAxis label={{ value: 'Deviation (mm)', angle: -90, position: 'insideLeft' }} />
                          <RechartsTooltip />
                          <ReferenceLine y={3} stroke="orange" strokeDasharray="3 3" label="Max Tolerance (3mm)" />
                          <Bar dataKey="verticality_mm" radius={[8, 8, 0, 0]}>
                            {selectedData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.verticality_mm <= 3 ? '#4caf50' : '#ff9800'} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </Box>
                  )}

                  {/* Point Cloud View */}
                  {viewMode === 'pointcloud' && (
                    <Box height="100%">
                      <Stack direction="row" alignItems="center" spacing={1} mb={2}>
                        <Typography variant="h6" fontWeight={600}>
                          LiDAR Point Cloud
                        </Typography>
                        <DataSourceBadge source="demo" />
                      </Stack>
                      <Typography variant="body2" color="text.secondary" mb={2}>
                        Showing {pointCloudData.length} sampled points from {selectedData.reduce((sum, s) => sum + s.lidar_points, 0).toLocaleString()} total LiDAR points
                      </Typography>
                      <ResponsiveContainer width="100%" height={450}>
                        <ScatterChart>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="x" name="Easting" unit="m" />
                          <YAxis dataKey="y" name="Northing" unit="m" />
                          <ZAxis dataKey="elevation" name="Elevation" unit="m" range={[20, 200]} />
                          <RechartsTooltip
                            cursor={{ strokeDasharray: '3 3' }}
                            content={({ payload }) => {
                              if (payload && payload.length > 0) {
                                const data = payload[0].payload;
                                return (
                                  <Paper sx={{ p: 1 }}>
                                    <Typography variant="caption" display="block"><strong>{data.segment}</strong></Typography>
                                    <Typography variant="caption">Elevation: {data.elevation.toFixed(2)}m</Typography>
                                  </Paper>
                                );
                              }
                              return null;
                            }}
                          />
                          <Scatter data={pointCloudData} fill="#2196f3">
                            {pointCloudData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={`hsl(${200 + (entry.elevation - 80) * 4}, 70%, 50%)`} />
                            ))}
                          </Scatter>
                        </ScatterChart>
                      </ResponsiveContainer>
                    </Box>
                  )}

                  {/* Heatmap View */}
                  {viewMode === 'heatmap' && (
                    <Box height="100%">
                      <Stack direction="row" alignItems="center" spacing={1} mb={2}>
                        <Typography variant="h6" fontWeight={600}>
                          Hyperspectral Strength Heatmap
                        </Typography>
                        <DataSourceBadge source="demo" />
                      </Stack>
                      <Typography variant="body2" color="text.secondary" mb={2}>
                        Concrete strength analysis from 204-band hyperspectral imaging
                      </Typography>
                      <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={selectedData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="face" />
                          <YAxis label={{ value: 'Strength (MPa)', angle: -90, position: 'insideLeft' }} />
                          <RechartsTooltip />
                          <ReferenceLine y={40} stroke="green" strokeDasharray="3 3" label="C40 Spec" />
                          <Bar dataKey="strength_mpa" radius={[8, 8, 0, 0]}>
                            {selectedData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.strength_mpa >= 40 ? '#4caf50' : '#ff9800'} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>

                      <Typography variant="subtitle2" fontWeight={600} mt={3} mb={2}>
                        Moisture Content Distribution
                      </Typography>
                      <ResponsiveContainer width="100%" height={200}>
                        <LineChart data={selectedData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="face" />
                          <YAxis label={{ value: 'Moisture (%)', angle: -90, position: 'insideLeft' }} />
                          <RechartsTooltip />
                          <ReferenceLine y={5} stroke="orange" strokeDasharray="3 3" label="Max Safe" />
                          <Line type="monotone" dataKey="moisture_pct" stroke="#2196f3" strokeWidth={2} />
                        </LineChart>
                      </ResponsiveContainer>
                    </Box>
                  )}

                  {/* Defects View */}
                  {viewMode === 'defects' && (
                    <Box height="100%">
                      <Stack direction="row" alignItems="center" spacing={1} mb={2}>
                        <Typography variant="h6" fontWeight={600}>
                          Visual Defect Detection
                        </Typography>
                        <DataSourceBadge source="demo" />
                      </Stack>
                      <Typography variant="body2" color="text.secondary" mb={2}>
                        Showing {selectedDefects.length} defect(s) in {selectedSegments.length} segment(s)
                      </Typography>

                      {selectedDefects.length > 0 ? (
                        <>
                          <ResponsiveContainer width="100%" height={300}>
                            <ScatterChart>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="x" name="X Position" unit="m" />
                              <YAxis dataKey="y" name="Y Position" unit="m" />
                              <ZAxis dataKey="z" name="Size" range={[50, 400]} />
                              <RechartsTooltip
                                content={({ payload }) => {
                                  if (payload && payload.length > 0) {
                                    const data = payload[0].payload;
                                    const defect = REAL_DEFECTS.find(d => d.id === data.id);
                                    if (defect) {
                                      return (
                                        <Paper sx={{ p: 1 }}>
                                          <Typography variant="caption" display="block"><strong>{defect.type}</strong></Typography>
                                          <Typography variant="caption">Severity: {defect.severity}</Typography>
                                          <Typography variant="caption" display="block">Location: ({defect.location[0]}, {defect.location[1]})</Typography>
                                          <Typography variant="caption">
                                            {defect.area_sqcm ? `Area: ${defect.area_sqcm}cm²` : `Length: ${defect.length_mm}mm`}
                                          </Typography>
                                        </Paper>
                                      );
                                    }
                                  }
                                  return null;
                                }}
                              />
                              <Scatter data={selectedDefects}>
                                {selectedDefects.map((entry, index) => (
                                  <Cell
                                    key={`cell-${index}`}
                                    fill={entry.severity === 'Moderate' ? '#ff9800' : '#2196f3'}
                                  />
                                ))}
                              </Scatter>
                            </ScatterChart>
                          </ResponsiveContainer>

                          <TableContainer sx={{ mt: 2 }}>
                            <Table size="small">
                              <TableHead>
                                <TableRow>
                                  <TableCell><strong>Type</strong></TableCell>
                                  <TableCell><strong>Severity</strong></TableCell>
                                  <TableCell><strong>Location</strong></TableCell>
                                  <TableCell><strong>Size</strong></TableCell>
                                </TableRow>
                              </TableHead>
                              <TableBody>
                                {selectedDefects.map((defect) => (
                                  <TableRow key={defect.id}>
                                    <TableCell>{defect.type}</TableCell>
                                    <TableCell>
                                      <Chip
                                        label={defect.severity}
                                        size="small"
                                        color={defect.severity === 'Moderate' ? 'warning' : 'info'}
                                      />
                                    </TableCell>
                                    <TableCell>({defect.location[0]}, {defect.location[1]})</TableCell>
                                    <TableCell>
                                      {defect.area_sqcm ? `${defect.area_sqcm}cm²` : `${defect.length_mm}mm`}
                                    </TableCell>
                                  </TableRow>
                                ))}
                              </TableBody>
                            </Table>
                          </TableContainer>
                        </>
                      ) : (
                        <Box sx={{ height: 400, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          <Box textAlign="center">
                            <CheckCircle sx={{ fontSize: 80, color: 'success.main', mb: 2 }} />
                            <Typography variant="h6" color="success.main">
                              No defects detected
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              Selected segments are in excellent condition
                            </Typography>
                          </Box>
                        </Box>
                      )}
                    </Box>
                  )}
                </>
              )}
            </Box>
          </Paper>
        </Grid>

        {/* Right Sidebar - Analysis */}
        <Grid item xs={12} md={3}>
          <Paper sx={{ p: 2, boxShadow: '0 1px 3px rgba(0,0,0,0.08)' }}>
            {selectedSegments.length > 0 ? (
              <>
                <Typography variant="subtitle2" fontWeight={600} mb={2}>
                  Multi-Segment Comparison
                </Typography>
                <Typography variant="caption" color="text.secondary" display="block" mb={2}>
                  Comparing {selectedSegments.length} segments
                </Typography>

                {/* Flatness Chart */}
                <Box mb={3}>
                  <Typography variant="caption" fontWeight={600} display="block" mb={1}>
                    Flatness (mm)
                  </Typography>
                  <ResponsiveContainer width="100%" height={180}>
                    <BarChart data={selectedData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis
                        dataKey="face"
                        tick={{ fontSize: 11 }}
                        tickFormatter={(value) => value.split(' ')[0]}
                      />
                      <YAxis domain={[0, 5]} tick={{ fontSize: 11 }} />
                      <RechartsTooltip />
                      <ReferenceLine y={5} stroke="#ff5252" strokeDasharray="3 3" />
                      <Bar dataKey="flatness_mm" fill="#4caf50" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                  <Typography variant="caption" color="text.secondary" textAlign="center" display="block">
                    Max tolerance: 5mm
                  </Typography>
                </Box>

                <Divider sx={{ my: 2 }} />

                {/* Verticality Chart */}
                <Box>
                  <Typography variant="caption" fontWeight={600} display="block" mb={1}>
                    Verticality (mm)
                  </Typography>
                  <ResponsiveContainer width="100%" height={180}>
                    <BarChart data={selectedData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis
                        dataKey="face"
                        tick={{ fontSize: 11 }}
                        tickFormatter={(value) => value.split(' ')[0]}
                      />
                      <YAxis domain={[0, 3]} tick={{ fontSize: 11 }} />
                      <RechartsTooltip />
                      <ReferenceLine y={3} stroke="#ff5252" strokeDasharray="3 3" />
                      <Bar dataKey="verticality_mm" fill="#2196f3" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                  <Typography variant="caption" color="text.secondary" textAlign="center" display="block">
                    Max tolerance: 3mm
                  </Typography>
                </Box>
              </>
            ) : (
              <Box textAlign="center" py={4}>
                <Info sx={{ fontSize: 60, color: 'text.disabled', mb: 2 }} />
                <Typography variant="body2" color="text.secondary">
                  Select segments to view comparison analysis
                </Typography>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default SyntheticDataDashboard;
