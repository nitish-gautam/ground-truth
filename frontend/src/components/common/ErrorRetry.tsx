/**
 * ErrorRetry Component
 * Reusable error display with retry functionality
 */

import React from 'react';
import { Alert, Button, Box, Typography } from '@mui/material';
import { Refresh, ErrorOutline } from '@mui/icons-material';

export interface ErrorRetryProps {
  error: string | Error;
  onRetry?: () => void;
  severity?: 'error' | 'warning';
  showDetails?: boolean;
}

export const ErrorRetry: React.FC<ErrorRetryProps> = ({
  error,
  onRetry,
  severity = 'error',
  showDetails = false
}) => {
  const errorMessage = typeof error === 'string' ? error : error.message;
  const errorStack = typeof error !== 'string' && showDetails ? error.stack : null;

  return (
    <Alert
      severity={severity}
      icon={<ErrorOutline />}
      action={
        onRetry && (
          <Button
            color="inherit"
            size="small"
            startIcon={<Refresh />}
            onClick={onRetry}
            sx={{
              fontWeight: 600,
              '&:hover': {
                backgroundColor: 'rgba(255, 255, 255, 0.2)'
              }
            }}
          >
            Retry
          </Button>
        )
      }
      sx={{
        '& .MuiAlert-message': {
          width: '100%'
        }
      }}
    >
      <Typography variant="body2" fontWeight={600} gutterBottom>
        {errorMessage}
      </Typography>
      {errorStack && (
        <Box
          component="pre"
          sx={{
            mt: 1,
            p: 1,
            bgcolor: 'rgba(0, 0, 0, 0.05)',
            borderRadius: 1,
            fontSize: '0.75rem',
            overflow: 'auto',
            maxHeight: '150px'
          }}
        >
          {errorStack}
        </Box>
      )}
    </Alert>
  );
};

export default ErrorRetry;
