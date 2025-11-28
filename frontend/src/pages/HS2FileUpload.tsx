/**
 * HS2 File Upload Page
 *
 * Allows uploading GPR, BIM, LiDAR, Excel files
 * Shows processing status and analytics
 */

import React, { useState } from 'react';
import {
  Container,
  Typography,
  Box,
  Tabs,
  Tab,
  Paper
} from '@mui/material';
import { FileUpload } from '../components/FileUpload';
import { FileProcessingStatus } from '../components/FileProcessingStatus';
import { FileAnalyticsChart } from '../components/FileAnalyticsChart';

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
      id={`file-upload-tabpanel-${index}`}
      aria-labelledby={`file-upload-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ pt: 3 }}>{children}</Box>}
    </div>
  );
}

export const HS2FileUpload: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [lastUploadedFileId, setLastUploadedFileId] = useState<string | null>(null);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handleFileUploaded = (fileId: string) => {
    setLastUploadedFileId(fileId);
    // Auto-switch to processing tab
    setActiveTab(1);
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          HS2 File Upload & Processing
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Upload GPR surveys, BIM models, LiDAR scans, monitoring data, and reports.
          View processing status and analytics.
        </Typography>
      </Box>

      {/* Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs value={activeTab} onChange={handleTabChange} aria-label="file upload tabs">
          <Tab label="Upload Files" />
          <Tab label="Processing Status" />
          <Tab label="Analytics" />
        </Tabs>
      </Paper>

      {/* Tab Panels */}
      <TabPanel value={activeTab} index={0}>
        <FileUpload onFileUploaded={handleFileUploaded} />
      </TabPanel>

      <TabPanel value={activeTab} index={1}>
        {lastUploadedFileId ? (
          <FileProcessingStatus fileId={lastUploadedFileId} />
        ) : (
          <Paper sx={{ p: 4, textAlign: 'center' }}>
            <Typography variant="body1" color="text.secondary">
              Upload a file to see processing status
            </Typography>
          </Paper>
        )}
      </TabPanel>

      <TabPanel value={activeTab} index={2}>
        <FileAnalyticsChart />
      </TabPanel>
    </Container>
  );
};
