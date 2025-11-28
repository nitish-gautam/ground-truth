/**
 * HS2 Deliverables Table Component
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
  Skeleton
} from '@mui/material';
import { Deliverable } from '../types/hs2Types';
import { formatDate, getDeliverableStatusColor, isPastDate } from '../utils/formatting';

interface HS2DeliverablesTableProps {
  deliverables: Deliverable[];
  isLoading: boolean;
}

const HS2DeliverablesTable: React.FC<HS2DeliverablesTableProps> = ({
  deliverables,
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

  if (deliverables.length === 0) {
    return (
      <Box textAlign="center" py={4}>
        <Typography color="text.secondary">No deliverables found</Typography>
      </Box>
    );
  }

  return (
    <TableContainer component={Paper}>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell><strong>Code</strong></TableCell>
            <TableCell><strong>Deliverable Name</strong></TableCell>
            <TableCell><strong>Status</strong></TableCell>
            <TableCell><strong>Version</strong></TableCell>
            <TableCell><strong>Due Date</strong></TableCell>
            <TableCell><strong>Submitted</strong></TableCell>
            <TableCell><strong>Approved</strong></TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {deliverables.map((deliverable) => {
            const isOverdue = deliverable.due_date && 
                             deliverable.status !== 'Approved' && 
                             isPastDate(deliverable.due_date);
            
            return (
              <TableRow key={deliverable.deliverable_id}>
                <TableCell>{deliverable.deliverable_code}</TableCell>
                <TableCell>{deliverable.deliverable_name}</TableCell>
                <TableCell>
                  <Chip
                    label={deliverable.status}
                    size="small"
                    sx={{
                      backgroundColor: getDeliverableStatusColor(deliverable.status),
                      color: 'white'
                    }}
                  />
                </TableCell>
                <TableCell>{deliverable.version}</TableCell>
                <TableCell>
                  <Box>
                    {deliverable.due_date ? formatDate(deliverable.due_date) : 'N/A'}
                    {isOverdue && (
                      <Chip label="Overdue" size="small" color="error" sx={{ ml: 1 }} />
                    )}
                  </Box>
                </TableCell>
                <TableCell>
                  {deliverable.submitted_date ? formatDate(deliverable.submitted_date) : '-'}
                </TableCell>
                <TableCell>
                  {deliverable.approved_date ? formatDate(deliverable.approved_date) : '-'}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

export default HS2DeliverablesTable;
