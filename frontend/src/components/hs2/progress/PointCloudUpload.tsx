/**
 * Point Cloud Upload and Comparison Component
 * Handles file upload, processing, and visualization of point cloud comparison results
 */

import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  Alert,
  CircularProgress,
  LinearProgress,
  Chip,
  Stack,
  TextField,
  FormControlLabel,
  Switch,
  Divider,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableRow,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  DialogContentText
} from '@mui/material';
import {
  CloudUpload,
  CompareArrows,
  Assessment,
  CheckCircle,
  Error,
  Warning
} from '@mui/icons-material';
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8002';

interface ComparisonResult {
  id: string;
  asset_id: string;
  baseline_file_path: string;
  current_file_path: string;
  comparison_date: string;

  // Deviation metrics
  surface_deviation_avg: number;
  surface_deviation_max: number;
  surface_deviation_min: number;
  surface_deviation_std: number;

  // Volume metrics
  volume_planned_m3: number;
  volume_actual_m3: number;
  volume_difference_m3: number;

  // Completion metrics
  completion_percentage?: number;
  points_within_tolerance_pct: number;
  tolerance_threshold_mm: number;

  // Processing metadata
  processing_time_seconds: number;
  point_count_baseline: number;
  point_count_current: number;
  confidence_score: number;

  // Quality
  quality_flags: Array<{
    type: string;
    severity: string;
    message: string;
    recommendation?: string;
  }>;

  // Hotspots
  hotspots: Array<{
    x: number;
    y: number;
    z: number;
    deviation_mm: number;
  }>;
}

interface PointCloudUploadProps {
  assetId?: string;
}

export const PointCloudUpload: React.FC<PointCloudUploadProps> = ({ assetId }) => {
  const [baselineFile, setBaselineFile] = useState<File | null>(null);
  const [currentFile, setCurrentFile] = useState<File | null>(null);
  const [toleranceMm, setToleranceMm] = useState<number>(50);
  const [voxelSize, setVoxelSize] = useState<number>(0.05);
  const [removeOutliers, setRemoveOutliers] = useState<boolean>(true);

  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [result, setResult] = useState<ComparisonResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Deletion confirmation state
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [fileToDelete, setFileToDelete] = useState<'baseline' | 'current' | null>(null);

  const handleBaselineFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setBaselineFile(event.target.files[0]);
      setError(null);
    }
  };

  const handleCurrentFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setCurrentFile(event.target.files[0]);
      setError(null);
    }
  };

  const handleDeleteFileClick = (fileType: 'baseline' | 'current') => {
    setFileToDelete(fileType);
    setDeleteConfirmOpen(true);
  };

  const handleDeleteConfirm = () => {
    if (fileToDelete === 'baseline') {
      setBaselineFile(null);
    } else if (fileToDelete === 'current') {
      setCurrentFile(null);
    }
    setDeleteConfirmOpen(false);
    setFileToDelete(null);
  };

  const handleDeleteCancel = () => {
    setDeleteConfirmOpen(false);
    setFileToDelete(null);
  };

  const handleUploadAndCompare = async () => {
    if (!baselineFile || !currentFile) {
      setError('Please select both baseline and current scan files');
      return;
    }

    if (!assetId) {
      setError('No asset selected. Please select an asset first.');
      return;
    }

    try {
      setUploading(true);
      setUploadProgress(0);
      setError(null);

      const formData = new FormData();
      formData.append('asset_id', assetId);
      formData.append('baseline_file', baselineFile);
      formData.append('current_file', currentFile);
      formData.append('tolerance_mm', toleranceMm.toString());
      formData.append('downsample_voxel_size', voxelSize.toString());
      formData.append('remove_outliers', removeOutliers.toString());

      const response = await axios.post(
        `${API_BASE_URL}/api/v1/progress/point-cloud/upload-and-compare`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          onUploadProgress: (progressEvent) => {
            if (progressEvent.total) {
              const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
              setUploadProgress(progress);
            }
          },
        }
      );

      setResult(response.data);
      setUploading(false);
      setUploadProgress(100);
    } catch (err: any) {
      console.error('Upload error:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to process point cloud comparison');
      setUploading(false);
      setUploadProgress(0);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'critical': return 'error';
      case 'high': return 'error';
      case 'medium': return 'warning';
      case 'low': return 'info';
      default: return 'default';
    }
  };

  return (
    <Box>
      <Card>
        <CardContent>
          <Box display="flex" alignItems="center" mb={2}>
            <CloudUpload sx={{ mr: 1, fontSize: 30 }} color="primary" />
            <Typography variant="h5">Point Cloud Upload & Comparison</Typography>
          </Box>

          <Typography variant="body2" color="text.secondary" mb={3}>
            Upload baseline (BIM model or initial scan) and current site scan for automated comparison.
            Supports LAS, LAZ, and PLY formats.
          </Typography>

          <Grid container spacing={3}>
            {/* File Upload Section */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 2, border: '1px dashed', borderColor: 'divider' }}>
                <Typography variant="subtitle2" gutterBottom>
                  Baseline Point Cloud (BIM/Initial Scan)
                </Typography>
                <input
                  accept=".las,.laz,.ply"
                  style={{ display: 'none' }}
                  id="baseline-file-upload"
                  type="file"
                  onChange={handleBaselineFileChange}
                  aria-label="Upload baseline point cloud file (LAS, LAZ, or PLY format)"
                />
                <label htmlFor="baseline-file-upload">
                  <Button
                    variant="contained"
                    component="span"
                    fullWidth
                    endIcon={<CloudUpload />}
                    aria-label="Select baseline point cloud file"
                    sx={{
                      bgcolor: 'secondary.dark',
                      color: 'secondary.contrastText',
                      '&:hover': {
                        bgcolor: 'secondary.main'
                      }
                    }}
                  >
                    Select Baseline File
                  </Button>
                </label>
                {baselineFile && (
                  <Chip
                    label={baselineFile.name}
                    onDelete={() => handleDeleteFileClick('baseline')}
                    size="small"
                    sx={{
                      mt: 1,
                      bgcolor: 'secondary.main',
                      color: 'secondary.contrastText',
                      '&:hover': {
                        bgcolor: 'secondary.dark'
                      },
                      '& .MuiChip-deleteIcon': {
                        color: 'secondary.contrastText',
                        '&:hover': {
                          color: 'error.light'
                        }
                      }
                    }}
                  />
                )}
              </Paper>
            </Grid>

            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 2, border: '1px dashed', borderColor: 'divider' }}>
                <Typography variant="subtitle2" gutterBottom>
                  Current Site Scan (LiDAR/Photogrammetry)
                </Typography>
                <input
                  accept=".las,.laz,.ply"
                  style={{ display: 'none' }}
                  id="current-file-upload"
                  type="file"
                  onChange={handleCurrentFileChange}
                  aria-label="Upload current site scan file (LAS, LAZ, or PLY format)"
                />
                <label htmlFor="current-file-upload">
                  <Button
                    variant="contained"
                    component="span"
                    fullWidth
                    endIcon={<CloudUpload />}
                    aria-label="Select current site scan file"
                    sx={{
                      bgcolor: 'secondary.dark',
                      color: 'secondary.contrastText',
                      '&:hover': {
                        bgcolor: 'secondary.main'
                      }
                    }}
                  >
                    Select Current Scan
                  </Button>
                </label>
                {currentFile && (
                  <Chip
                    label={currentFile.name}
                    onDelete={() => handleDeleteFileClick('current')}
                    size="small"
                    sx={{
                      mt: 1,
                      bgcolor: 'secondary.main',
                      color: 'secondary.contrastText',
                      '&:hover': {
                        bgcolor: 'secondary.dark'
                      },
                      '& .MuiChip-deleteIcon': {
                        color: 'secondary.contrastText',
                        '&:hover': {
                          color: 'error.light'
                        }
                      }
                    }}
                  />
                )}
              </Paper>
            </Grid>

            {/* Processing Options */}
            <Grid item xs={12}>
              <Divider sx={{ my: 2 }} />
              <Typography variant="subtitle2" gutterBottom>
                Processing Options
              </Typography>
            </Grid>

            <Grid item xs={12} md={4}>
              <TextField
                label="Tolerance (mm)"
                type="number"
                value={toleranceMm}
                onChange={(e) => setToleranceMm(Number(e.target.value))}
                fullWidth
                helperText="Acceptable deviation threshold"
                inputProps={{ min: 1, max: 500, step: 1 }}
              />
            </Grid>

            <Grid item xs={12} md={4}>
              <TextField
                label="Voxel Size (m)"
                type="number"
                value={voxelSize}
                onChange={(e) => setVoxelSize(Number(e.target.value))}
                fullWidth
                helperText="Downsampling resolution"
                inputProps={{ min: 0.01, max: 1.0, step: 0.01 }}
              />
            </Grid>

            <Grid item xs={12} md={4}>
              <FormControlLabel
                control={
                  <Switch
                    checked={removeOutliers}
                    onChange={(e) => setRemoveOutliers(e.target.checked)}
                  />
                }
                label="Remove Outliers"
              />
            </Grid>

            {/* Upload Button */}
            <Grid item xs={12}>
              <Button
                variant="contained"
                size="large"
                fullWidth
                onClick={handleUploadAndCompare}
                disabled={uploading || !baselineFile || !currentFile}
                endIcon={uploading ? <CircularProgress size={20} /> : <CompareArrows />}
                sx={{
                  bgcolor: 'secondary.dark',
                  color: 'secondary.contrastText',
                  '&:hover': {
                    bgcolor: 'secondary.main'
                  },
                  '&:disabled': {
                    bgcolor: 'action.disabledBackground',
                    color: 'action.disabled'
                  }
                }}
              >
                {uploading ? 'Processing...' : 'Upload and Compare'}
              </Button>
            </Grid>

            {/* Progress Bar */}
            {uploading && (
              <Grid item xs={12}>
                <Box sx={{ width: '100%' }}>
                  <LinearProgress
                    variant="determinate"
                    value={uploadProgress}
                    aria-label={`Upload progress: ${uploadProgress}%`}
                    aria-valuenow={uploadProgress}
                    aria-valuemin={0}
                    aria-valuemax={100}
                  />
                  <Typography variant="caption" color="text.secondary" align="center" display="block" mt={1}>
                    {uploadProgress}% - {uploadProgress < 50 ? 'Uploading files...' : 'Processing point clouds...'}
                  </Typography>
                </Box>
              </Grid>
            )}

            {/* Error Alert */}
            {error && (
              <Grid item xs={12}>
                <Alert severity="error" onClose={() => setError(null)}>
                  {error}
                </Alert>
              </Grid>
            )}

            {/* Results Section */}
            {result && (
              <>
                <Grid item xs={12}>
                  <Divider sx={{ my: 2 }} />
                  <Box display="flex" alignItems="center" mb={2}>
                    <Assessment sx={{ mr: 1, fontSize: 28 }} color="success" />
                    <Typography variant="h6">Comparison Results</Typography>
                  </Box>
                </Grid>

                {/* Summary Cards */}
                <Grid item xs={12} md={4}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                        Average Deviation
                      </Typography>
                      <Typography variant="h4">
                        {result.surface_deviation_avg.toFixed(1)} mm
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Max: {result.surface_deviation_max.toFixed(1)} mm
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>

                <Grid item xs={12} md={4}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                        Within Tolerance
                      </Typography>
                      <Typography variant="h4" color={result.points_within_tolerance_pct >= 80 ? 'success.main' : 'error.main'}>
                        {result.points_within_tolerance_pct.toFixed(1)}%
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Tolerance: {result.tolerance_threshold_mm} mm
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>

                <Grid item xs={12} md={4}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                        Volume Difference
                      </Typography>
                      <Typography variant="h4">
                        {Math.abs(result.volume_difference_m3).toFixed(2)} m³
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {result.volume_difference_m3 >= 0 ? 'Over budget' : 'Under budget'}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>

                {/* Detailed Metrics */}
                <Grid item xs={12}>
                  <TableContainer component={Paper} variant="outlined">
                    <Table size="small">
                      <TableBody>
                        <TableRow>
                          <TableCell><strong>Processing Time</strong></TableCell>
                          <TableCell>{result.processing_time_seconds}s</TableCell>
                          <TableCell><strong>Baseline Points</strong></TableCell>
                          <TableCell>{result.point_count_baseline.toLocaleString()}</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell><strong>Confidence Score</strong></TableCell>
                          <TableCell>{result.confidence_score.toFixed(1)}%</TableCell>
                          <TableCell><strong>Current Points</strong></TableCell>
                          <TableCell>{result.point_count_current.toLocaleString()}</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell><strong>Std Deviation</strong></TableCell>
                          <TableCell>{result.surface_deviation_std.toFixed(2)} mm</TableCell>
                          <TableCell><strong>Hotspots Detected</strong></TableCell>
                          <TableCell>{result.hotspots.length}</TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Grid>

                {/* Quality Flags */}
                {result.quality_flags.length > 0 && (
                  <Grid item xs={12}>
                    <Alert severity="warning" icon={<Warning />}>
                      <Typography variant="subtitle2" gutterBottom>
                        Quality Issues Detected
                      </Typography>
                      <Stack spacing={1}>
                        {result.quality_flags.map((flag, index) => (
                          <Box key={index}>
                            <Chip
                              label={flag.type}
                              size="small"
                              color={getSeverityColor(flag.severity) as any}
                              sx={{ mr: 1 }}
                            />
                            <Typography variant="body2" component="span">
                              {flag.message}
                            </Typography>
                            {flag.recommendation && (
                              <Typography variant="caption" display="block" color="text.secondary">
                                → {flag.recommendation}
                              </Typography>
                            )}
                          </Box>
                        ))}
                      </Stack>
                    </Alert>
                  </Grid>
                )}

                {/* Success Message */}
                <Grid item xs={12}>
                  <Alert severity="success" icon={<CheckCircle />}>
                    Point cloud comparison completed successfully! Results have been saved to the database.
                  </Alert>
                </Grid>
              </>
            )}
          </Grid>
        </CardContent>
      </Card>

      {/* Deletion Confirmation Dialog */}
      <Dialog
        open={deleteConfirmOpen}
        onClose={handleDeleteCancel}
        aria-labelledby="delete-dialog-title"
        aria-describedby="delete-dialog-description"
      >
        <DialogTitle
          id="delete-dialog-title"
          sx={{
            bgcolor: 'primary.main',
            color: 'primary.contrastText'
          }}
        >
          Confirm File Removal
        </DialogTitle>
        <DialogContent sx={{ mt: 2 }}>
          <DialogContentText id="delete-dialog-description">
            Are you sure you want to remove the selected {fileToDelete} file?
            You will need to select a new file to continue with the upload.
          </DialogContentText>
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button
            onClick={handleDeleteCancel}
            variant="outlined"
            sx={{
              color: 'text.secondary',
              borderColor: 'divider'
            }}
          >
            Cancel
          </Button>
          <Button
            onClick={handleDeleteConfirm}
            variant="contained"
            color="error"
            autoFocus
            endIcon={<Error />}
          >
            Remove File
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};
