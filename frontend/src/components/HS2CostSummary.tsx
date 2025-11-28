/**
 * HS2 Cost Summary Component
 */

import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Grid,
  Skeleton
} from '@mui/material';
import { TrendingUp, TrendingDown } from '@mui/icons-material';
import { CostSummary } from '../types/hs2Types';
import { formatCurrency, formatVariance } from '../utils/formatting';

interface HS2CostSummaryProps {
  costSummary: CostSummary | undefined;
  isLoading: boolean;
}

const HS2CostSummary: React.FC<HS2CostSummaryProps> = ({
  costSummary,
  isLoading
}) => {
  if (isLoading) {
    return (
      <Box>
        <Grid container spacing={2} mb={3}>
          {[...Array(3)].map((_, i) => (
            <Grid item xs={12} sm={4} key={i}>
              <Skeleton variant="rectangular" height={100} />
            </Grid>
          ))}
        </Grid>
        <Skeleton variant="rectangular" height={200} />
      </Box>
    );
  }

  if (!costSummary) {
    return (
      <Card>
        <CardContent>
          <Typography color="text.secondary">No cost data available</Typography>
        </CardContent>
      </Card>
    );
  }

  const variance = formatVariance(costSummary.total_variance, costSummary.currency);
  const isOverBudget = costSummary.total_variance < 0;

  return (
    <Box>
      {/* Summary Cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={4}>
          <Card>
            <CardContent>
              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                Total Budget
              </Typography>
              <Typography variant="h5">
                {formatCurrency(costSummary.total_budget, costSummary.currency)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={4}>
          <Card>
            <CardContent>
              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                Actual Spend
              </Typography>
              <Typography variant="h5">
                {formatCurrency(costSummary.total_actual, costSummary.currency)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={4}>
          <Card sx={{ borderLeft: `4px solid ${isOverBudget ? '#f44336' : '#4caf50'}` }}>
            <CardContent>
              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                Variance
              </Typography>
              <Box display="flex" alignItems="center" gap={1}>
                {isOverBudget ? (
                  <TrendingDown color="error" />
                ) : (
                  <TrendingUp color="success" />
                )}
                <Typography
                  variant="h5"
                  color={isOverBudget ? 'error.main' : 'success.main'}
                >
                  {variance.text}
                </Typography>
              </Box>
              <Typography variant="caption" color="text.secondary">
                {costSummary.variance_percentage.toFixed(1)}% variance
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Category Breakdown */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Cost Breakdown by Category
          </Typography>
          
          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell><strong>Category</strong></TableCell>
                  <TableCell align="right"><strong>Budget</strong></TableCell>
                  <TableCell align="right"><strong>Actual</strong></TableCell>
                  <TableCell align="right"><strong>Variance</strong></TableCell>
                  <TableCell align="right"><strong>%</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {costSummary.categories.map((cost) => {
                  const catVariance = formatVariance(cost.variance, cost.currency);
                  return (
                    <TableRow key={cost.cost_id}>
                      <TableCell>{cost.category}</TableCell>
                      <TableCell align="right">
                        {formatCurrency(cost.budget_amount, cost.currency)}
                      </TableCell>
                      <TableCell align="right">
                        {formatCurrency(cost.actual_amount, cost.currency)}
                      </TableCell>
                      <TableCell
                        align="right"
                        sx={{
                          color: catVariance.isPositive ? 'success.main' : 'error.main',
                          fontWeight: 'bold'
                        }}
                      >
                        {catVariance.text}
                      </TableCell>
                      <TableCell align="right">
                        {cost.variance_percentage.toFixed(1)}%
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>
    </Box>
  );
};

export default HS2CostSummary;
