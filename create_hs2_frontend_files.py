#!/usr/bin/env python3
"""
Script to create all HS2 Frontend files
This creates production-ready React components with complete implementations
"""

import os

BASE_DIR = "frontend/src"

def write_file(relative_path, content):
    """Write content to file, creating directories as needed"""
    full_path = os.path.join(BASE_DIR, relative_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, 'w') as f:
        f.write(content)
    print(f"✅ Created: {relative_path}")

# ============================================================================
# Component: HS2AssetTable
# ============================================================================

HS2_ASSET_TABLE = '''/**
 * HS2 Asset Table Component
 * 
 * Filterable, sortable table for displaying asset list
 */

import React, { useState } from 'react';
import {
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Paper,
  Chip,
  IconButton,
  Tooltip,
  TextField,
  MenuItem,
  Grid,
  Skeleton
} from '@mui/material';
import { Visibility, Refresh } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { HS2Asset, AssetStatus, AssetType, ASSET_TYPES, ASSET_STATUSES } from '../types/hs2Types';
import {
  formatDateTime,
  formatTAEMScore,
  getAssetStatusVariant
} from '../utils/formatting';

interface HS2AssetTableProps {
  assets: HS2Asset[];
  isLoading: boolean;
  onRefresh?: () => void;
}

const HS2AssetTable: React.FC<HS2AssetTableProps> = ({
  assets,
  isLoading,
  onRefresh
}) => {
  const navigate = useNavigate();
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(25);
  const [filterStatus, setFilterStatus] = useState<AssetStatus | ''>('');
  const [filterType, setFilterType] = useState<AssetType | ''>('');
  const [searchTerm, setSearchTerm] = useState('');

  // Filter assets
  const filteredAssets = assets.filter(asset => {
    if (filterStatus && asset.status !== filterStatus) return false;
    if (filterType && asset.asset_type !== filterType) return false;
    if (searchTerm && !asset.asset_id.toLowerCase().includes(searchTerm.toLowerCase()) &&
        !asset.asset_name.toLowerCase().includes(searchTerm.toLowerCase())) return false;
    return true;
  });

  const handleChangePage = (event: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const handleRowClick = (assetId: string) => {
    navigate(`/hs2/assets/${assetId}`);
  };

  if (isLoading) {
    return (
      <Box>
        {[...Array(5)].map((_, i) => (
          <Skeleton key={i} variant="rectangular" height={60} sx={{ my: 1 }} />
        ))}
      </Box>
    );
  }

  return (
    <Box>
      {/* Filters */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={4}>
          <TextField
            fullWidth
            size="small"
            label="Search Asset"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Asset ID or Name"
          />
        </Grid>
        <Grid item xs={12} sm={3}>
          <TextField
            fullWidth
            select
            size="small"
            label="Status"
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value as AssetStatus | '')}
          >
            <MenuItem value="">All Statuses</MenuItem>
            {ASSET_STATUSES.map(status => (
              <MenuItem key={status} value={status}>{status}</MenuItem>
            ))}
          </TextField>
        </Grid>
        <Grid item xs={12} sm={3}>
          <TextField
            fullWidth
            select
            size="small"
            label="Asset Type"
            value={filterType}
            onChange={(e) => setFilterType(e.target.value as AssetType | '')}
          >
            <MenuItem value="">All Types</MenuItem>
            {ASSET_TYPES.map(type => (
              <MenuItem key={type} value={type}>{type}</MenuItem>
            ))}
          </TextField>
        </Grid>
        <Grid item xs={12} sm={2}>
          {onRefresh && (
            <Tooltip title="Refresh">
              <IconButton onClick={onRefresh} color="primary">
                <Refresh />
              </IconButton>
            </Tooltip>
          )}
        </Grid>
      </Grid>

      {/* Table */}
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell><strong>Asset ID</strong></TableCell>
              <TableCell><strong>Asset Name</strong></TableCell>
              <TableCell><strong>Type</strong></TableCell>
              <TableCell><strong>Contractor</strong></TableCell>
              <TableCell><strong>Status</strong></TableCell>
              <TableCell><strong>TAEM Score</strong></TableCell>
              <TableCell><strong>Last Evaluated</strong></TableCell>
              <TableCell align="center"><strong>Actions</strong></TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {filteredAssets
              .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
              .map((asset) => {
                const taemDisplay = formatTAEMScore(asset.taem_score);
                return (
                  <TableRow
                    key={asset.asset_id}
                    hover
                    onClick={() => handleRowClick(asset.asset_id)}
                    sx={{ cursor: 'pointer' }}
                  >
                    <TableCell>{asset.asset_id}</TableCell>
                    <TableCell>{asset.asset_name}</TableCell>
                    <TableCell>{asset.asset_type}</TableCell>
                    <TableCell>{asset.contractor}</TableCell>
                    <TableCell>
                      <Chip
                        label={asset.status}
                        color={getAssetStatusVariant(asset.status)}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={taemDisplay.text}
                        color={taemDisplay.color}
                        size="small"
                        variant="outlined"
                      />
                    </TableCell>
                    <TableCell>{formatDateTime(asset.last_evaluated)}</TableCell>
                    <TableCell align="center">
                      <Tooltip title="View Details">
                        <IconButton size="small" color="primary">
                          <Visibility />
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                );
              })}
          </TableBody>
        </Table>
        <TablePagination
          component="div"
          count={filteredAssets.length}
          page={page}
          onPageChange={handleChangePage}
          rowsPerPage={rowsPerPage}
          onRowsPerPageChange={handleChangeRowsPerPage}
          rowsPerPageOptions={[10, 25, 50, 100]}
        />
      </TableContainer>
    </Box>
  );
};

export default HS2AssetTable;
'''

write_file('components/HS2AssetTable.tsx', HS2_ASSET_TABLE)

# ============================================================================
# Component: HS2ReadinessPanel
# ============================================================================

HS2_READINESS_PANEL = '''/**
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
'''

write_file('components/HS2ReadinessPanel.tsx', HS2_READINESS_PANEL)

print("\n✨ Component files created successfully!")
print("Continuing with more components...")

