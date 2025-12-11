/**
 * LiDAR Viewer Tab
 * Dedicated tab for LiDAR terrain analysis and visualization
 * Uses real UK LiDAR DTM 2022 data (1m resolution)
 */

import React from 'react';
import {
  Box,
  Paper,
  Typography,
  Chip
} from '@mui/material';
import { Terrain } from '@mui/icons-material';
import { LidarProfileViewer } from './LidarProfileViewer';

export const LidarViewerTab: React.FC = () => {
  return (
    <Box>
      {/* Tab Header */}
      <Box mb={4}>
        <Box display="flex" alignItems="center" gap={2} mb={2}>
          <Terrain sx={{ fontSize: 40, color: 'primary.main' }} />
          <Box>
            <Typography variant="h4" component="h2" gutterBottom>
              LiDAR Terrain Analysis
            </Typography>
            <Typography variant="body1" color="text.secondary">
              High-resolution elevation data from UK Environment Agency LiDAR surveys
            </Typography>
          </Box>
        </Box>

        <Box display="flex" gap={1} flexWrap="wrap">
          <Chip
            label="🟢 REAL DATA"
            size="small"
            sx={{ bgcolor: 'rgb(220, 252, 231)', color: 'rgb(22, 101, 52)', fontWeight: 600 }}
          />
          <Chip
            label="1m Resolution DTM"
            size="small"
            variant="outlined"
            color="primary"
          />
          <Chip
            label="British National Grid (EPSG:27700)"
            size="small"
            variant="outlined"
            color="primary"
          />
          <Chip
            label="ODN Vertical Datum"
            size="small"
            variant="outlined"
            color="primary"
          />
        </Box>
      </Box>

      {/* LiDAR Profile Viewer */}
      <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
        <LidarProfileViewer />
      </Paper>

      {/* Data Coverage Information */}
      <Paper elevation={2} sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom fontWeight={600}>
          LiDAR Data Coverage
        </Typography>
        <Typography variant="body2" color="text.secondary" paragraph>
          The platform currently has access to <strong>17 high-resolution DTM tiles</strong> covering
          strategic locations along transport infrastructure corridors. Each tile covers a 5km × 5km area
          with 1-meter horizontal resolution.
        </Typography>

        <Box mt={3}>
          <Typography variant="subtitle2" fontWeight={600} gutterBottom>
            Available Tile Coverage Areas:
          </Typography>
          <Box display="grid" gridTemplateColumns="repeat(auto-fill, minmax(200px, 1fr))" gap={1} mt={1}>
            {[
              'SK23ne: E 425-430km, N 335-340km',
              'SK24ne: E 425-430km, N 345-350km',
              'SK24se: E 425-430km, N 340-345km',
              'SK33ne: E 435-440km, N 335-340km',
              'SK33nw: E 430-435km, N 335-340km',
              'SK33se: E 435-440km, N 330-335km',
              'SK33sw: E 430-435km, N 330-335km',
              'SK34ne: E 435-440km, N 345-350km',
              'SK34nw: E 430-435km, N 345-350km',
              'SK43sw: E 440-445km, N 330-335km',
              'SK44ne: E 445-450km, N 345-350km',
              'SK44nw: E 440-445km, N 345-350km',
              'SK44se: E 445-450km, N 340-345km',
              'SK44sw: E 440-445km, N 340-345km',
              'SK54ne: E 455-460km, N 345-350km',
              'SK54nw: E 450-455km, N 345-350km',
              'SK54sw: E 450-455km, N 340-345km'
            ].map((tile) => (
              <Chip
                key={tile}
                label={tile}
                size="small"
                variant="outlined"
                sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}
              />
            ))}
          </Box>
        </Box>

        <Box mt={3}>
          <Typography variant="subtitle2" fontWeight={600} gutterBottom>
            Data Source:
          </Typography>
          <Typography variant="body2" color="text.secondary">
            UK Environment Agency LiDAR Composite DTM 2022 - Airborne laser scanning survey data
            processed to 1m resolution Digital Terrain Models. Vertical accuracy: ±10cm.
          </Typography>
        </Box>

        <Box mt={3}>
          <Typography variant="subtitle2" fontWeight={600} gutterBottom>
            Coordinate Systems:
          </Typography>
          <Box component="ul" sx={{ mt: 1, pl: 2 }}>
            <Typography component="li" variant="body2" color="text.secondary">
              <strong>Horizontal:</strong> British National Grid (OSGB36 / EPSG:27700)
            </Typography>
            <Typography component="li" variant="body2" color="text.secondary">
              <strong>Vertical:</strong> Ordnance Datum Newlyn (ODN) - UK mean sea level
            </Typography>
          </Box>
        </Box>
      </Paper>
    </Box>
  );
};

export default LidarViewerTab;
