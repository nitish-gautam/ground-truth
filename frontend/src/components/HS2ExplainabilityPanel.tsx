/**
 * HS2 Explainability Panel - "Why Not Ready?"
 * Shows detailed explanation of why an asset is not ready
 */

import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Alert,
  Skeleton
} from '@mui/material';
import { ExpandMore, Error, Warning, Info } from '@mui/icons-material';
import { ReadinessReport, RuleSeverity } from '../types/hs2Types';
import { getSeverityVariant, formatJSON } from '../utils/formatting';

interface HS2ExplainabilityPanelProps {
  report: ReadinessReport | undefined;
  isLoading: boolean;
}

const HS2ExplainabilityPanel: React.FC<HS2ExplainabilityPanelProps> = ({
  report,
  isLoading
}) => {
  const [expandedRule, setExpandedRule] = useState<string | false>(false);

  if (isLoading) {
    return (
      <Card>
        <CardContent>
          <Skeleton variant="text" width="70%" height={32} />
          <Skeleton variant="rectangular" height={150} sx={{ my: 2 }} />
        </CardContent>
      </Card>
    );
  }

  if (!report) {
    return (
      <Card>
        <CardContent>
          <Typography color="text.secondary">No evaluation data available</Typography>
        </CardContent>
      </Card>
    );
  }

  const failedRules = report.evaluations.filter(e => !e.passed);
  
  if (failedRules.length === 0) {
    return (
      <Card>
        <CardContent>
          <Alert severity="success">
            All rules passed! This asset is ready.
          </Alert>
        </CardContent>
      </Card>
    );
  }

  const getSeverityIcon = (severity: RuleSeverity) => {
    switch (severity) {
      case 'Critical': return <Error color="error" />;
      case 'Major': return <Warning color="warning" />;
      case 'Minor': return <Info color="info" />;
    }
  };

  const groupedBySeverity = {
    Critical: failedRules.filter(r => r.severity === 'Critical'),
    Major: failedRules.filter(r => r.severity === 'Major'),
    Minor: failedRules.filter(r => r.severity === 'Minor')
  };

  const handleAccordionChange = (ruleId: string) => (
    event: React.SyntheticEvent,
    isExpanded: boolean
  ) => {
    setExpandedRule(isExpanded ? ruleId : false);
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Why is {report.asset_id} {report.status}?
        </Typography>

        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          {failedRules.length} rule{failedRules.length !== 1 ? 's' : ''} failed:
          {report.critical_failures > 0 && ` ${report.critical_failures} Critical`}
          {report.major_failures > 0 && `, ${report.major_failures} Major`}
          {report.minor_failures > 0 && `, ${report.minor_failures} Minor`}
        </Typography>

        {(['Critical', 'Major', 'Minor'] as RuleSeverity[]).map(severity => {
          const rules = groupedBySeverity[severity];
          if (rules.length === 0) return null;

          return (
            <Box key={severity} sx={{ mb: 2 }}>
              <Typography
                variant="subtitle2"
                sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}
              >
                {getSeverityIcon(severity)}
                {severity} Issues ({rules.length})
              </Typography>

              {rules.map((rule) => (
                <Accordion
                  key={rule.rule_id}
                  expanded={expandedRule === rule.rule_id}
                  onChange={handleAccordionChange(rule.rule_id)}
                  sx={{ mb: 1 }}
                >
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Box display="flex" alignItems="center" gap={2} width="100%">
                      <Chip
                        label={rule.severity}
                        size="small"
                        color={getSeverityVariant(rule.severity)}
                      />
                      <Typography variant="body2" flex={1}>
                        {rule.rule_name}
                      </Typography>
                    </Box>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Box>
                      <Typography variant="body2" gutterBottom>
                        <strong>Message:</strong> {rule.message}
                      </Typography>
                      
                      {Object.keys(rule.evidence).length > 0 && (
                        <Box sx={{ mt: 2 }}>
                          <Typography variant="caption" color="text.secondary" gutterBottom>
                            Evidence:
                          </Typography>
                          <Box
                            component="pre"
                            sx={{
                              p: 2,
                              backgroundColor: '#f5f5f5',
                              borderRadius: 1,
                              overflow: 'auto',
                              fontSize: '0.75rem',
                              maxHeight: 300
                            }}
                          >
                            {formatJSON(rule.evidence)}
                          </Box>
                        </Box>
                      )}
                    </Box>
                  </AccordionDetails>
                </Accordion>
              ))}
            </Box>
          );
        })}
      </CardContent>
    </Card>
  );
};

export default HS2ExplainabilityPanel;
