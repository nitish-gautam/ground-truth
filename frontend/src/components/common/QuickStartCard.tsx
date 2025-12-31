/**
 * Quick Start Card Component
 * Provides guided first steps for new users
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Box,
  IconButton,
  Collapse
} from '@mui/material';
import {
  LooksOne,
  LooksTwo,
  Looks3,
  Close,
  CheckCircle
} from '@mui/icons-material';

export const QuickStartCard: React.FC = () => {
  const [visible, setVisible] = useState(false);
  const [completedSteps, setCompletedSteps] = useState<number[]>([]);

  useEffect(() => {
    // Check if user has dismissed the quick start card before
    const dismissed = localStorage.getItem('hs2_quickstart_dismissed');
    const completed = localStorage.getItem('hs2_quickstart_completed');

    if (!dismissed) {
      setVisible(true);
    }

    if (completed) {
      setCompletedSteps(JSON.parse(completed));
    }
  }, []);

  const handleDismiss = () => {
    setVisible(false);
    localStorage.setItem('hs2_quickstart_dismissed', 'true');
  };

  const handleStepClick = (stepNumber: number) => {
    if (!completedSteps.includes(stepNumber)) {
      const newCompleted = [...completedSteps, stepNumber];
      setCompletedSteps(newCompleted);
      localStorage.setItem('hs2_quickstart_completed', JSON.stringify(newCompleted));
    }
  };

  const steps = [
    {
      number: 1,
      icon: <LooksOne sx={{ color: 'white' }} />,
      title: 'Review Asset Readiness Summary',
      description: 'Check the summary cards below to see overall project health and status'
    },
    {
      number: 2,
      icon: <LooksTwo sx={{ color: 'white' }} />,
      title: 'Explore GIS Map',
      description: 'Navigate to the "GIS Route Map" tab to visualize infrastructure spatially'
    },
    {
      number: 3,
      icon: <Looks3 sx={{ color: 'white' }} />,
      title: 'Run TAEM Compliance Check',
      description: 'Click the "TAEM Compliance Check" button to audit asset readiness'
    }
  ];

  if (!visible) return null;

  return (
    <Collapse in={visible}>
      <Card
        elevation={3}
        sx={{
          mb: 4,
          bgcolor: 'primary.main',
          color: 'white',
          position: 'relative'
        }}
      >
        <IconButton
          onClick={handleDismiss}
          sx={{
            position: 'absolute',
            top: 8,
            right: 8,
            color: 'white'
          }}
          size="small"
        >
          <Close />
        </IconButton>

        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ fontWeight: 700, fontSize: '1.1rem', pr: 4 }}>
            👋 New here? Start with these 3 steps:
          </Typography>

          <List sx={{ mt: 2 }}>
            {steps.map((step) => (
              <ListItem
                key={step.number}
                onClick={() => handleStepClick(step.number)}
                sx={{
                  cursor: 'pointer',
                  borderRadius: 1,
                  mb: 1,
                  bgcolor: 'rgba(255, 255, 255, 0.1)',
                  '&:hover': {
                    bgcolor: 'rgba(255, 255, 255, 0.15)'
                  },
                  opacity: completedSteps.includes(step.number) ? 0.7 : 1,
                  transition: 'all 0.2s'
                }}
              >
                <ListItemIcon>
                  {completedSteps.includes(step.number) ? (
                    <CheckCircle sx={{ color: 'success.light' }} />
                  ) : (
                    <Box
                      sx={{
                        width: 32,
                        height: 32,
                        borderRadius: '50%',
                        bgcolor: 'secondary.main',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}
                    >
                      {step.icon}
                    </Box>
                  )}
                </ListItemIcon>
                <ListItemText
                  primary={
                    <Typography
                      variant="body1"
                      sx={{
                        fontWeight: 600,
                        textDecoration: completedSteps.includes(step.number) ? 'line-through' : 'none'
                      }}
                    >
                      {step.title}
                    </Typography>
                  }
                  secondary={
                    <Typography
                      variant="body2"
                      sx={{
                        color: 'rgba(255, 255, 255, 0.8)',
                        mt: 0.5
                      }}
                    >
                      {step.description}
                    </Typography>
                  }
                />
              </ListItem>
            ))}
          </List>

          {completedSteps.length === steps.length && (
            <Box sx={{ mt: 2, p: 2, bgcolor: 'success.main', borderRadius: 1, textAlign: 'center' }}>
              <Typography variant="body2" sx={{ fontWeight: 600 }}>
                🎉 Great job! You've completed the quick start guide.
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>
    </Collapse>
  );
};

export default QuickStartCard;
