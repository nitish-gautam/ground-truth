/**
 * HS2 Certificates Table Component
 */

import React from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Typography,
  Box,
  Skeleton,
  Alert
} from '@mui/material';
import { Warning } from '@mui/icons-material';
import { Certificate } from '../types/hs2Types';
import { formatDate, getCertificateStatusColor, isWithinDays } from '../utils/formatting';

interface HS2CertificatesTableProps {
  certificates: Certificate[];
  isLoading: boolean;
}

const HS2CertificatesTable: React.FC<HS2CertificatesTableProps> = ({
  certificates,
  isLoading
}) => {
  if (isLoading) {
    return (
      <Box>
        {[...Array(3)].map((_, i) => (
          <Skeleton key={i} variant="rectangular" height={50} sx={{ my: 1 }} />
        ))}
      </Box>
    );
  }

  if (certificates.length === 0) {
    return (
      <Box textAlign="center" py={4}>
        <Typography color="text.secondary">No certificates found</Typography>
      </Box>
    );
  }

  const expiringSoon = certificates.filter(
    c => c.expiry_date && isWithinDays(c.expiry_date, 30) && c.status !== 'Expired'
  );

  return (
    <Box>
      {expiringSoon.length > 0 && (
        <Alert severity="warning" icon={<Warning />} sx={{ mb: 2 }}>
          {expiringSoon.length} certificate{expiringSoon.length !== 1 ? 's' : ''} expiring within 30 days
        </Alert>
      )}

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell><strong>Certificate Name</strong></TableCell>
              <TableCell><strong>Type</strong></TableCell>
              <TableCell><strong>Issuer</strong></TableCell>
              <TableCell><strong>Issue Date</strong></TableCell>
              <TableCell><strong>Expiry Date</strong></TableCell>
              <TableCell><strong>Status</strong></TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {certificates.map((cert) => (
              <TableRow key={cert.certificate_id}>
                <TableCell>{cert.certificate_name}</TableCell>
                <TableCell>{cert.certificate_type}</TableCell>
                <TableCell>{cert.issuer}</TableCell>
                <TableCell>{formatDate(cert.issue_date)}</TableCell>
                <TableCell>
                  {cert.expiry_date ? formatDate(cert.expiry_date) : 'N/A'}
                </TableCell>
                <TableCell>
                  <Chip
                    label={cert.status}
                    size="small"
                    sx={{
                      backgroundColor: getCertificateStatusColor(cert.status),
                      color: 'white'
                    }}
                  />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
};

export default HS2CertificatesTable;
