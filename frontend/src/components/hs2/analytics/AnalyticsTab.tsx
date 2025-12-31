/**
 * Analytics & Reports Tab
 * Consolidated view for Integrated Demo and Progress Verification
 */

import React, { useState } from 'react';
import { Box, Tabs, Tab, Paper } from '@mui/material';
import { Assessment, Timeline } from '@mui/icons-material';
import { SyntheticDataDashboard } from '../demo/RealDataDashboard';
import { ProgressVerificationTab } from '../progress/ProgressVerificationTab';

export const AnalyticsTab: React.FC = () => {
  const [activeView, setActiveView] = useState(0);

  return (
    <Box>
      {/* Analytics Type Selection */}
      <Paper elevation={2} sx={{ mb: 3 }}>
        <Tabs
          value={activeView}
          onChange={(_, newValue) => setActiveView(newValue)}
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
            icon={<Assessment />}
            iconPosition="start"
            label="Demo Showcase"
          />
          <Tab
            icon={<Timeline />}
            iconPosition="start"
            label="Progress Verification"
          />
        </Tabs>
      </Paper>

      {/* Analytics Content */}
      <Box>
        {activeView === 0 && <SyntheticDataDashboard />}
        {activeView === 1 && <ProgressVerificationTab />}
      </Box>
    </Box>
  );
};

export default AnalyticsTab;
