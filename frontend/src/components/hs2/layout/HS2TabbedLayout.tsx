/**
 * HS2 Tabbed Layout Component
 * Main layout wrapper with tab navigation for the HS2 workspace
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Tabs,
  Tab,
  Container,
  Typography,
  Paper
} from '@mui/material';
import { Dashboard, Map, ViewInAr, Analytics } from '@mui/icons-material';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`hs2-tabpanel-${index}`}
      aria-labelledby={`hs2-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ py: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function a11yProps(index: number) {
  return {
    id: `hs2-tab-${index}`,
    'aria-controls': `hs2-tabpanel-${index}`,
  };
}

interface HS2TabbedLayoutProps {
  children: [React.ReactNode, React.ReactNode, React.ReactNode, React.ReactNode]; // [Overview, GIS, DataViewers, Analytics]
}

export const HS2TabbedLayout: React.FC<HS2TabbedLayoutProps> = ({ children }) => {
  const [currentTab, setCurrentTab] = useState(0);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
  };

  // Listen for custom navigation events from Quick Access cards
  useEffect(() => {
    const handleNavigateToTab = (event: Event) => {
      const customEvent = event as CustomEvent<{ tab: number }>;
      if (customEvent.detail && typeof customEvent.detail.tab === 'number') {
        setCurrentTab(customEvent.detail.tab);
        // Scroll to top smoothly
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }
    };

    window.addEventListener('navigate-to-tab', handleNavigateToTab);
    return () => {
      window.removeEventListener('navigate-to-tab', handleNavigateToTab);
    };
  }, []);

  return (
    <Box sx={{ width: '100%', bgcolor: 'background.default', minHeight: '100vh' }}>
      {/* Header */}
      <Paper
        elevation={0}
        sx={{
          borderBottom: 1,
          borderColor: 'divider',
          bgcolor: 'white',
          position: 'sticky',
          top: 0,
          zIndex: 1100
        }}
      >
        <Container maxWidth="xl">
          <Box sx={{ py: 3 }}>
            <Typography variant="h4" component="h1" gutterBottom color="primary">
              HS2 Data & Assurance Workspace
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Infrastructure Intelligence Platform - Prototype Development
            </Typography>
          </Box>

          {/* Tabs */}
          <Tabs
            value={currentTab}
            onChange={handleTabChange}
            variant="fullWidth"
            aria-label="HS2 workspace navigation tabs"
            sx={{
              '& .MuiTab-root': {
                textTransform: 'none',
                fontWeight: 600,
                fontSize: '1rem',
                minHeight: 64,
                color: 'text.secondary',
                '&.Mui-selected': {
                  color: 'primary.main'
                }
              },
              '& .MuiTabs-indicator': {
                backgroundColor: 'secondary.main',
                height: 3
              }
            }}
          >
            <Tab
              icon={<Dashboard />}
              iconPosition="start"
              label="Dashboard"
              {...a11yProps(0)}
            />
            <Tab
              icon={<Map />}
              iconPosition="start"
              label="GIS Map"
              {...a11yProps(1)}
            />
            <Tab
              icon={<ViewInAr />}
              iconPosition="start"
              label="Data Viewers"
              {...a11yProps(2)}
            />
            <Tab
              icon={<Analytics />}
              iconPosition="start"
              label="Analytics & Reports"
              {...a11yProps(3)}
            />
          </Tabs>
        </Container>
      </Paper>

      {/* Tab Content */}
      <Container maxWidth="xl">
        <TabPanel value={currentTab} index={0}>
          {children[0]}
        </TabPanel>
        <TabPanel value={currentTab} index={1}>
          {children[1]}
        </TabPanel>
        <TabPanel value={currentTab} index={2}>
          {children[2]}
        </TabPanel>
        <TabPanel value={currentTab} index={3}>
          {children[3]}
        </TabPanel>
      </Container>
    </Box>
  );
};

export default HS2TabbedLayout;
