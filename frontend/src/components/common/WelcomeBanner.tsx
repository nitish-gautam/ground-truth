/**
 * Welcome Banner Component
 * Displays a dismissible welcome message for first-time users with tour CTA
 */

import React, { useState, useEffect } from 'react';
import { Alert, AlertTitle, Button, Box, Collapse } from '@mui/material';
import { School, Close } from '@mui/icons-material';

interface WelcomeBannerProps {
  onStartTour?: () => void;
}

export const WelcomeBanner: React.FC<WelcomeBannerProps> = ({ onStartTour }) => {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    // Check if user has dismissed the welcome banner before
    const dismissed = localStorage.getItem('hs2_welcome_dismissed');
    if (!dismissed) {
      setVisible(true);
    }
  }, []);

  const handleDismiss = () => {
    setVisible(false);
    localStorage.setItem('hs2_welcome_dismissed', 'true');
  };

  const handleStartTour = () => {
    if (onStartTour) {
      onStartTour();
    }
    handleDismiss();
  };

  return (
    <Collapse in={visible}>
      <Alert
        severity="info"
        icon={<School fontSize="large" />}
        onClose={handleDismiss}
        action={
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
            <Button
              size="small"
              variant="contained"
              color="primary"
              onClick={handleStartTour}
              sx={{ fontWeight: 600 }}
            >
              Take 2-Min Tour
            </Button>
            <Button
              size="small"
              variant="text"
              onClick={handleDismiss}
              startIcon={<Close />}
              sx={{ color: 'text.secondary' }}
            >
              Dismiss
            </Button>
          </Box>
        }
        sx={{
          mb: 4,
          '& .MuiAlert-message': {
            width: '100%'
          },
          '& .MuiAlert-action': {
            paddingTop: 0,
            alignItems: 'center'
          }
        }}
      >
        <AlertTitle sx={{ fontWeight: 700, fontSize: '1.1rem' }}>
          👋 Welcome to HS2 Infrastructure Intelligence Platform!
        </AlertTitle>
        <Box sx={{ mt: 1 }}>
          Track <strong>500+ assets</strong>, run instant <strong>TAEM compliance checks</strong>, and visualize infrastructure data
          across BIM models, GIS maps, and real-time monitoring. New here? Take our quick tour to get started!
        </Box>
      </Alert>
    </Collapse>
  );
};

export default WelcomeBanner;
