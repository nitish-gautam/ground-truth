/**
 * HS2 Clean Dashboard - Professional Enterprise Design
 * Matches the clean, spacious aesthetic of the production mockup
 */

import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Card,
  CardContent,
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
  Stack
} from '@mui/material';
import {
  Terrain,
  Colorize,
  ViewInAr,
  CameraAlt,
  CheckCircle,
  Schedule,
  Settings,
  Refresh
} from '@mui/icons-material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';

// Sample data
const SEGMENTS = [
  {
    id: 'east',
    face: 'East Face',
    points: '125K pts',
    coverage: '95% HSI',
    score: 78
  },
  {
    id: 'west',
    face: 'West Face',
    points: '118K pts',
    coverage: '92% HSI',
    score: 89
  },
  {
    id: 'north',
    face: 'North Face',
    points: '62K pts',
    coverage: '88% HSI',
    score: 82
  },
  {
    id: 'south',
    face: 'South Face',
    points: '65K pts',
    coverage: '90% HSI',
    score: 94
  }
];

const CHART_DATA = [
  { segment: 'East Face', flatness: 3.2, verticality: 2.1 },
  { segment: 'West Face', flatness: 2.8, verticality: 1.9 },
  { segment: 'North Face', flatness: 4.1, verticality: 2.5 },
  { segment: 'South Face', flatness: 2.3, verticality: 1.7 }
];

export const CleanDashboardDemo: React.FC = () => {
  const [selectedSegments, setSelectedSegments] = useState<string[]>(['east', 'west', 'north', 'south']);
  const [viewMode, setViewMode] = useState('defects');

  const toggleSegment = (segmentId: string) => {
    setSelectedSegments(prev =>
      prev.includes(segmentId)
        ? prev.filter(id => id !== segmentId)
        : [...prev, segmentId]
    );
  };

  const avgScore = selectedSegments.length > 0
    ? Math.round(
        SEGMENTS.filter(s => selectedSegments.includes(s.id))
          .reduce((sum, s) => sum + s.score, 0) / selectedSegments.length
      )
    : 86;

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
          <Select value="pier_p1" size="small" sx={{ minWidth: 120 }}>
            <MenuItem value="pier_p1">Pier P1</MenuItem>
            <MenuItem value="pier_p2">Pier P2</MenuItem>
            <MenuItem value="pier_p3">Pier P3</MenuItem>
          </Select>

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
              sx={{ color: avgScore >= 85 ? '#4caf50' : '#2196f3' }}
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
            <Typography variant="h6" fontWeight={600}>
              {avgScore}/100
            </Typography>
            <Chip
              label="GOOD"
              color="success"
              size="small"
              sx={{ height: 20, fontSize: '0.7rem', fontWeight: 600 }}
            />
          </Box>

          <IconButton size="small">
            <Settings fontSize="small" />
          </IconButton>
          <IconButton size="small">
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
                  <Button size="small" onClick={() => setSelectedSegments(SEGMENTS.map(s => s.id))}>
                    All
                  </Button>
                  <Button size="small" onClick={() => setSelectedSegments([])}>
                    Clear
                  </Button>
                </Box>
              </Box>

              <List dense disablePadding>
                {SEGMENTS.map(segment => (
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
                      secondary={<Typography variant="caption">{segment.points} • {segment.coverage}</Typography>}
                    />
                    <Chip
                      label={segment.score}
                      size="small"
                      color={segment.score >= 90 ? 'success' : segment.score >= 75 ? 'primary' : 'warning'}
                      sx={{ height: 24, fontWeight: 600 }}
                    />
                  </ListItem>
                ))}
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
                  { id: 'defects', label: 'Defects', icon: <CheckCircle fontSize="small" /> }
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
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              bgcolor: '#fafafa',
              p: 4
            }}>
              <Box textAlign="center">
                <CheckCircle sx={{ fontSize: 120, color: 'text.disabled', mb: 2 }} />
                <Typography variant="h5" fontWeight={600} mb={1}>
                  Visual Defect Detection Overlay
                </Typography>
                <Typography variant="body2" color="text.secondary" mb={2}>
                  Showing {selectedSegments.length} segment(s)
                </Typography>
                <Chip
                  label="🟢 REAL DATA Available"
                  color="success"
                  sx={{ fontWeight: 600 }}
                />
              </Box>
            </Box>
          </Paper>
        </Grid>

        {/* Right Sidebar - Analysis */}
        <Grid item xs={12} md={3}>
          <Paper sx={{ p: 2, boxShadow: '0 1px 3px rgba(0,0,0,0.08)' }}>
            <Typography variant="subtitle2" fontWeight={600} mb={2}>
              Multi-Segment Comparison
            </Typography>

            {selectedSegments.length > 0 && (
              <>
                <Typography variant="caption" color="text.secondary" display="block" mb={1}>
                  Comparing {selectedSegments.length} segments
                </Typography>

                {/* Flatness Chart */}
                <Box mb={3}>
                  <Typography variant="caption" fontWeight={600} display="block" mb={1}>
                    Flatness (mm)
                  </Typography>
                  <ResponsiveContainer width="100%" height={180}>
                    <BarChart data={CHART_DATA.filter((_, i) => selectedSegments.includes(SEGMENTS[i].id))}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis
                        dataKey="segment"
                        tick={{ fontSize: 11 }}
                        tickFormatter={(value) => value.split(' ')[0]}
                      />
                      <YAxis domain={[0, 5]} tick={{ fontSize: 11 }} />
                      <RechartsTooltip />
                      <ReferenceLine y={5} stroke="#ff5252" strokeDasharray="3 3" />
                      <Bar dataKey="flatness" fill="#4caf50" radius={[4, 4, 0, 0]} />
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
                    <BarChart data={CHART_DATA.filter((_, i) => selectedSegments.includes(SEGMENTS[i].id))}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis
                        dataKey="segment"
                        tick={{ fontSize: 11 }}
                        tickFormatter={(value) => value.split(' ')[0]}
                      />
                      <YAxis domain={[0, 3]} tick={{ fontSize: 11 }} />
                      <RechartsTooltip />
                      <ReferenceLine y={3} stroke="#ff5252" strokeDasharray="3 3" />
                      <Bar dataKey="verticality" fill="#2196f3" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                  <Typography variant="caption" color="text.secondary" textAlign="center" display="block">
                    Max tolerance: 3mm
                  </Typography>
                </Box>
              </>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default CleanDashboardDemo;
