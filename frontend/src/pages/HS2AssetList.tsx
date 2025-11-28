/**
 * HS2 Asset List Page
 * Filterable list of all assets
 */

import React, { useState } from 'react';
import { Box, Container, Typography, Button } from '@mui/material';
import { Add, Refresh } from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import hs2Client from '../api/hs2Client';
import HS2AssetTable from '../components/HS2AssetTable';
import { AssetFilters } from '../types/hs2Types';

const HS2AssetList: React.FC = () => {
  const navigate = useNavigate();
  const [filters, setFilters] = useState<AssetFilters>({});
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);

  const {
    data: assetsResponse,
    isLoading,
    error,
    refetch
  } = useQuery({
    queryKey: ['hs2-assets', filters, page, pageSize],
    queryFn: () => hs2Client.assets.getAssets(filters, page, pageSize)
  });

  const handleRefresh = () => {
    refetch();
  };

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      {/* Page Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={4}>
        <Box>
          <Typography variant="h4" gutterBottom>
            Assets
          </Typography>
          <Typography variant="body1" color="text.secondary">
            {assetsResponse ? `${assetsResponse.total} assets found` : 'Loading...'}
          </Typography>
        </Box>
        <Box display="flex" gap={2}>
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={handleRefresh}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={() => navigate('/hs2/assets/new')}
          >
            Add Asset
          </Button>
        </Box>
      </Box>

      {/* Asset Table */}
      <HS2AssetTable
        assets={assetsResponse?.items || []}
        isLoading={isLoading}
        onRefresh={handleRefresh}
      />
    </Container>
  );
};

export default HS2AssetList;
