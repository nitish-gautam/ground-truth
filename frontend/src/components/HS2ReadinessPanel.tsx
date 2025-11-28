/**
 * HS2 Readiness Panel Component
 * 
 * Displays rule evaluation results with severity breakdown
 */

import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Chip,
  Grid,
  List,
  ListItem,
  ListItemText,
  LinearProgress,
  Divider,
  Skeleton
} from '@mui/material';
import { CheckCircle, Cancel, Warning } from '@mui/icons-material';
import { ReadinessReport } from '../types/hs2Types';
import { getSeverityVariant, formatDateTime } from '../utils/formatting';

interface HS2ReadinessPanelProps {
  report: ReadinessReport | undefined;
  isLoading: boolean;
}

const HS2ReadinessPanel: React.FC<HS2ReadinessPanelProps> = ({
  report,
  isLoading
}) => {
  if (isLoading) {
    return (
      <Card>
        <CardContent>
          <Skeleton variant="text" width="60%" height={32} />
          <Skeleton variant="rectangular" height={100} sx={{ my: 2 }} />
          <Skeleton variant="text" width="80%" />
        </CardContent>
      </Card>
    );
  }

  if (!report) {
    return (
      <Card>
        <CardContent>
          <Typography color="text.secondary">No readiness data available</Typography>
        </CardContent>
      </Card>
    );
  }

  const passRate = (report.rules_passed / (report.rules_passed + report.rules_failed)) * 100;

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Readiness Evaluation Results
        </Typography>
        
        {/* Summary Stats */}
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={12} sm={3}>
            <Box textAlign="center">
              <Typography variant="h4" color="success.main">
                {report.rules_passed}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Passed
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={12} sm={3}>
            <Box textAlign="center">
              <Typography variant="h4" color="error.main">
                {report.rules_failed}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Failed
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={12} sm={3}>
            <Box textAlign="center">
              <Typography variant="h4" color="warning.main">
                {report.major_failures}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Major Issues
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={12} sm={3}>
            <Box textAlign="center">
              <Typography variant="h4" color="error.dark">
                {report.critical_failures}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Critical
              </Typography>
            </Box>
          </Grid>
        </Grid>

        {/* Pass Rate */}
        <Box sx={{ mb: 3 }}>
          <Box display="flex" justifyContent="space-between" mb={1}>
            <Typography variant="body2">Pass Rate</Typography>
            <Typography variant="body2" fontWeight="bold">
              {passRate.toFixed(1)}%
            </Typography>
          </Box>
          <LinearProgress
            variant="determinate"
            value={passRate}
            color={passRate >= 80 ? 'success' : passRate >= 60 ? 'warning' : 'error'}
            sx={{ height: 8, borderRadius: 4 }}
          />
        </Box>

        <Divider sx={{ my: 2 }} />

        {/* Rule Evaluations */}
        <Typography variant="subtitle2" gutterBottom>
          Rule Evaluations
        </Typography>
        <List dense>
          {report.evaluations.slice(0, 10).map((evaluation, idx) => (
            <ListItem key={idx} divider>
              <ListItemText
                primary={
                  <Box display="flex" alignItems="center" gap={1}>
                    {evaluation.passed ? (
                      <CheckCircle color="success" fontSize="small" />
                    ) : (
                      <Cancel color="error" fontSize="small" />
                    )}
                    <Typography variant="body2">{evaluation.rule_name}</Typography>
                  </Box>
                }
                secondary={
                  <Box display="flex" gap={1} mt={0.5}>
                    <Chip
                      label={evaluation.severity}
                      size="small"
                      color={getSeverityVariant(evaluation.severity)}
                    />
                    <Typography variant="caption" color="text.secondary">
                      {evaluation.message}
                    </Typography>
                  </Box>
                }
              />
            </ListItem>
          ))}
        </List>

        <Typography variant="caption" color="text.secondary" display="block" mt={2}>
          Evaluated {formatDateTime(report.evaluated_at)}
          {report.evaluated_by && ` by ${report.evaluated_by}`}
        </Typography>
      </CardContent>
    </Card>
  );
};

export default HS2ReadinessPanel;
