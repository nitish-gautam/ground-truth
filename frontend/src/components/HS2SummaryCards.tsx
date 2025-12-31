/**
 * HS2 Summary Cards Component
 * 
 * Displays key dashboard metrics in card format with icons and color coding.
 */

import React from 'react';
import { Box, Card, CardContent, Typography, Grid, Skeleton, Chip } from '@mui/material';
import {
  CheckCircleOutline,
  ErrorOutline,
  WarningAmber,
  Assessment
} from '@mui/icons-material';
import { DashboardSummary } from '../types/hs2Types';
import { formatPercentage, formatCompactNumber } from '../utils/formatting';

interface HS2SummaryCardsProps {
  summary: DashboardSummary | undefined;
  isLoading: boolean;
}

interface MetricCardProps {
  title: string;
  value: number | string;
  subtitle?: string;
  icon: React.ReactNode;
  color: string;
  isLoading: boolean;
  isSynthetic?: boolean;
}

/**
 * Individual metric card component
 */
const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtitle,
  icon,
  color,
  isLoading,
  isSynthetic = false
}) => {
  if (isLoading) {
    return (
      <Card sx={{ height: '100%', minHeight: 120 }}>
        <CardContent>
          <Skeleton variant="text" width="60%" height={24} />
          <Skeleton variant="text" width="40%" height={48} sx={{ my: 1 }} />
          <Skeleton variant="text" width="50%" height={20} />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card
      sx={{
        height: '100%',
        minHeight: 120,
        borderLeft: `4px solid ${color}`,
        border: isSynthetic ? 2 : undefined,
        borderColor: isSynthetic ? 'warning.main' : undefined,
        transition: 'transform 0.2s, box-shadow 0.2s',
        '&:hover': {
          transform: 'translateY(-4px)',
          boxShadow: 4
        }
      }}
    >
      <CardContent>
        {isSynthetic && (
          <Chip
            label="SYNTHETIC DATA"
            color="warning"
            size="small"
            sx={{ mb: 1, fontWeight: 700, fontSize: '0.7rem' }}
          />
        )}
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Box flex={1}>
            <Typography
              variant="subtitle2"
              color="text.secondary"
              gutterBottom
              sx={{ fontWeight: 500 }}
            >
              {title}
            </Typography>
            <Typography variant="h3" component="div" sx={{ fontWeight: 600, color }}>
              {value}
            </Typography>
            {subtitle && (
              <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                {subtitle}
              </Typography>
            )}
          </Box>
          <Box
            sx={{
              width: 56,
              height: 56,
              borderRadius: 2,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              backgroundColor: `${color}15`,
              color: color
            }}
          >
            {icon}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

/**
 * Main summary cards component
 */
const HS2SummaryCards: React.FC<HS2SummaryCardsProps> = ({ summary, isLoading }) => {
  const cards = [
    {
      title: 'Total Assets',
      value: summary ? formatCompactNumber(summary.total_assets) : '0',
      subtitle: 'All assets tracked',
      icon: <Assessment fontSize="large" />,
      color: '#1976d2',
      isSynthetic: true
    },
    {
      title: 'Ready',
      value: summary ? formatCompactNumber(summary.ready_count) : '0',
      subtitle: summary ? formatPercentage(summary.ready_percentage / 100) : '0%',
      icon: <CheckCircleOutline fontSize="large" />,
      color: '#4caf50',
      isSynthetic: true
    },
    {
      title: 'Not Ready',
      value: summary ? formatCompactNumber(summary.not_ready_count) : '0',
      subtitle: summary
        ? formatPercentage((summary.not_ready_count / summary.total_assets) || 0)
        : '0%',
      icon: <ErrorOutline fontSize="large" />,
      color: '#f44336',
      isSynthetic: true
    },
    {
      title: 'At Risk',
      value: summary ? formatCompactNumber(summary.at_risk_count) : '0',
      subtitle: summary
        ? formatPercentage((summary.at_risk_count / summary.total_assets) || 0)
        : '0%',
      icon: <WarningAmber fontSize="large" />,
      color: '#ff9800',
      isSynthetic: true
    }
  ];

  return (
    <Grid container spacing={4}>
      {cards.map((card, index) => (
        <Grid item xs={12} sm={6} md={3} key={index}>
          <MetricCard {...card} isLoading={isLoading} />
        </Grid>
      ))}
    </Grid>
  );
};

export default HS2SummaryCards;
