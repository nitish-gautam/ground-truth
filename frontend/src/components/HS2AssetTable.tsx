/**
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
