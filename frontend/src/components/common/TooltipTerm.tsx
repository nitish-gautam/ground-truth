/**
 * Tooltip Term Component
 * Wraps technical terms with helpful tooltip explanations
 */

import React from 'react';
import { Tooltip, Typography, Box } from '@mui/material';
import { HelpOutline } from '@mui/icons-material';
import { GLOSSARY, GlossaryKey } from '../../utils/glossary';

interface TooltipTermProps {
  term: GlossaryKey;
  children: React.ReactNode;
  showIcon?: boolean;
  variant?: 'inherit' | 'body1' | 'body2' | 'caption' | 'h1' | 'h2' | 'h3' | 'h4' | 'h5' | 'h6';
}

export const TooltipTerm: React.FC<TooltipTermProps> = ({
  term,
  children,
  showIcon = true,
  variant = 'inherit'
}) => {
  const tooltipText = GLOSSARY[term];

  return (
    <Tooltip
      title={tooltipText}
      arrow
      placement="top"
      enterDelay={200}
      enterNextDelay={200}
      sx={{
        '& .MuiTooltip-tooltip': {
          fontSize: '0.875rem',
          maxWidth: 320,
          padding: 2,
        }
      }}
    >
      <Box
        component="span"
        sx={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: 0.5,
          cursor: 'help',
          borderBottom: '1px dotted',
          borderColor: 'text.secondary'
        }}
      >
        <Typography component="span" variant={variant}>
          {children}
        </Typography>
        {showIcon && (
          <HelpOutline
            sx={{
              fontSize: '0.875rem',
              color: 'info.main',
              opacity: 0.7
            }}
          />
        )}
      </Box>
    </Tooltip>
  );
};

export default TooltipTerm;
