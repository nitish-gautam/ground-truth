/**
 * Data Viewers Tab
 * Consolidated view for BIM, LiDAR, and Hyperspectral viewers with subtabs
 */

import React, { useState } from 'react';
import { Box, Tabs, Tab, Paper } from '@mui/material';
import { ViewInAr, Terrain, Colorize } from '@mui/icons-material';
import { HS2BIMTab } from '../bim/HS2BIMTab';
import { LidarViewerTab } from '../lidar/LidarViewerTab';
import { HyperspectralViewerTab } from '../hyperspectral/HyperspectralViewerTab';

export const DataViewersTab: React.FC = () => {
  const [activeViewer, setActiveViewer] = useState(0);

  return (
    <Box>
      {/* Viewer Type Selection */}
      <Paper elevation={2} sx={{ mb: 3 }}>
        <Tabs
          value={activeViewer}
          onChange={(_, newValue) => setActiveViewer(newValue)}
          indicatorColor="secondary"
          textColor="primary"
          variant="fullWidth"
          sx={{
            '& .MuiTab-root': {
              fontWeight: 600,
              fontSize: '0.95rem'
            }
          }}
        >
          <Tab
            icon={<ViewInAr />}
            iconPosition="start"
            label="BIM 3D Models"
          />
          <Tab
            icon={<Terrain />}
            iconPosition="start"
            label="LiDAR Point Clouds"
          />
          <Tab
            icon={<Colorize />}
            iconPosition="start"
            label="Hyperspectral Analysis"
          />
        </Tabs>
      </Paper>

      {/* Viewer Content */}
      <Box>
        {activeViewer === 0 && <HS2BIMTab />}
        {activeViewer === 1 && <LidarViewerTab />}
        {activeViewer === 2 && <HyperspectralViewerTab />}
      </Box>
    </Box>
  );
};

export default DataViewersTab;
