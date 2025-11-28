/**
 * FileProcessingStatus Component
 * Shows real-time processing status of uploaded files
 */

import React, { useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  LinearProgress,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Alert,
  Chip,
  List,
  ListItem,
  ListItemText,
  CircularProgress
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import HourglassEmptyIcon from '@mui/icons-material/HourglassEmpty';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';

interface ProcessingStep {
  name: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  message?: string;
  started_at?: string;
  completed_at?: string;
}

interface FileProcessingData {
  file_id: string;
  filename: string;
  file_type: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress_percentage: number;
  steps: ProcessingStep[];
  issues?: string[];
  metadata?: {
    total_segments?: number;
    detected_issues?: number;
    file_size_mb?: number;
  };
  created_at: string;
  updated_at: string;
}

interface FileProcessingStatusProps {
  fileId?: string;
  autoRefresh?: boolean;
  refreshInterval?: number; // milliseconds
}

export const FileProcessingStatus: React.FC<FileProcessingStatusProps> = ({
  fileId,
  autoRefresh = true,
  refreshInterval = 2000 // 2 seconds
}) => {
  // Fetch processing status
  const {
    data: processingData,
    isLoading,
    error,
    refetch
  } = useQuery<FileProcessingData>({
    queryKey: ['file-processing', fileId],
    queryFn: async () => {
      if (!fileId) throw new Error('No file ID provided');
      const response = await axios.get(`/api/v1/files/process/${fileId}`);
      return response.data;
    },
    enabled: !!fileId,
    refetchInterval: autoRefresh ? refreshInterval : false
  });

  // Get status color
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'failed':
        return 'error';
      case 'processing':
      case 'in_progress':
        return 'primary';
      default:
        return 'default';
    }
  };

  // Get step icon
  const getStepIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircleIcon color="success" />;
      case 'failed':
        return <ErrorIcon color="error" />;
      case 'in_progress':
        return <CircularProgress size={24} />;
      default:
        return <HourglassEmptyIcon color="disabled" />;
    }
  };

  if (!fileId) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight={300}>
        <Typography color="text.secondary">
          No file selected. Upload a file to see processing status.
        </Typography>
      </Box>
    );
  }

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight={300}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error">
        Failed to load processing status. Please try again.
      </Alert>
    );
  }

  if (!processingData) {
    return (
      <Alert severity="info">
        No processing data available for this file.
      </Alert>
    );
  }

  return (
    <Box>
      {/* File Info Header */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6">
            {processingData.filename}
          </Typography>
          <Chip
            label={processingData.status}
            color={getStatusColor(processingData.status) as any}
            icon={getStepIcon(processingData.status)}
          />
        </Box>

        {/* Progress Bar */}
        <Box sx={{ mb: 2 }}>
          <Box display="flex" justifyContent="space-between" mb={1}>
            <Typography variant="body2" color="text.secondary">
              Processing Progress
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {processingData.progress_percentage}%
            </Typography>
          </Box>
          <LinearProgress
            variant="determinate"
            value={processingData.progress_percentage}
            color={getStatusColor(processingData.status) as any}
          />
        </Box>

        {/* Metadata */}
        {processingData.metadata && (
          <Box display="flex" gap={2} flexWrap="wrap">
            {processingData.metadata.file_size_mb && (
              <Typography variant="body2" color="text.secondary">
                Size: {processingData.metadata.file_size_mb.toFixed(2)} MB
              </Typography>
            )}
            {processingData.metadata.total_segments && (
              <Typography variant="body2" color="text.secondary">
                Segments: {processingData.metadata.total_segments}
              </Typography>
            )}
            {processingData.metadata.detected_issues !== undefined && (
              <Typography
                variant="body2"
                color={processingData.metadata.detected_issues > 0 ? 'error.main' : 'success.main'}
              >
                Issues: {processingData.metadata.detected_issues}
              </Typography>
            )}
          </Box>
        )}
      </Paper>

      {/* Processing Steps */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Processing Steps
        </Typography>

        <Stepper orientation="vertical">
          {processingData.steps.map((step, index) => {
            const isActive = step.status === 'in_progress';
            const isCompleted = step.status === 'completed';
            const isFailed = step.status === 'failed';

            return (
              <Step key={index} active={isActive} completed={isCompleted}>
                <StepLabel
                  error={isFailed}
                  icon={getStepIcon(step.status)}
                >
                  <Box display="flex" alignItems="center" gap={1}>
                    <Typography variant="body1">
                      {step.name}
                    </Typography>
                    <Chip
                      label={step.status}
                      size="small"
                      color={getStatusColor(step.status) as any}
                    />
                  </Box>
                </StepLabel>
                <StepContent>
                  {step.message && (
                    <Typography variant="body2" color="text.secondary">
                      {step.message}
                    </Typography>
                  )}
                  {step.started_at && (
                    <Typography variant="caption" color="text.secondary" display="block">
                      Started: {new Date(step.started_at).toLocaleTimeString()}
                    </Typography>
                  )}
                  {step.completed_at && (
                    <Typography variant="caption" color="text.secondary" display="block">
                      Completed: {new Date(step.completed_at).toLocaleTimeString()}
                    </Typography>
                  )}
                </StepContent>
              </Step>
            );
          })}
        </Stepper>
      </Paper>

      {/* Issues Detected */}
      {processingData.issues && processingData.issues.length > 0 && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom color="error">
            Issues Detected ({processingData.issues.length})
          </Typography>
          <List>
            {processingData.issues.map((issue, index) => (
              <ListItem key={index}>
                <ErrorIcon color="error" sx={{ mr: 2 }} />
                <ListItemText
                  primary={issue}
                  primaryTypographyProps={{ variant: 'body2' }}
                />
              </ListItem>
            ))}
          </List>
        </Paper>
      )}

      {/* Success Message */}
      {processingData.status === 'completed' && (
        <Alert severity="success" sx={{ mt: 2 }}>
          File processing completed successfully!
          {processingData.metadata?.total_segments && (
            <> {processingData.metadata.total_segments} segments created.</>
          )}
        </Alert>
      )}

      {/* Error Message */}
      {processingData.status === 'failed' && (
        <Alert severity="error" sx={{ mt: 2 }}>
          File processing failed. Please check the issues above or try uploading again.
        </Alert>
      )}
    </Box>
  );
};
