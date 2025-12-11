/**
 * HS2 Integrated Inspection Demo
 * Demonstrates complete 8-step multi-modal inspection workflow
 * Combines LiDAR, Hyperspectral, 360° Imagery, and BIM data
 */

import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Button,
  Grid,
  Card,
  CardContent,
  Chip,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Alert,
  Divider
} from '@mui/material';
import {
  CheckCircle,
  RadioButtonUnchecked,
  Cloud,
  GridOn,
  QueryStats,
  Assessment,
  TrendingUp,
  Warning
} from '@mui/icons-material';

const WORKFLOW_STEPS = [
  {
    id: 1,
    label: 'Planning & Asset Selection',
    description: 'Define inspection scope and select target assets',
    status: 'implemented',
    icon: <GridOn />
  },
  {
    id: 2,
    label: 'Site Data Collection',
    description: 'Capture LiDAR, 360° imagery, and hyperspectral data',
    status: 'implemented',
    icon: <Cloud />
  },
  {
    id: 3,
    label: 'Upload & Preprocessing',
    description: 'Ingest multi-modal data and validate quality',
    status: 'implemented',
    icon: <CheckCircle />
  },
  {
    id: 4,
    label: 'Spatial Alignment (ICP)',
    description: 'Register point clouds with BIM reference model',
    status: 'planned',
    icon: <RadioButtonUnchecked />
  },
  {
    id: 5,
    label: 'Segmentation',
    description: 'Divide structure into inspectable patches',
    status: 'planned',
    icon: <RadioButtonUnchecked />
  },
  {
    id: 6,
    label: 'Multi-Modal Analysis',
    description: 'Analyze geometry, visual defects, and material properties',
    status: 'partial',
    icon: <QueryStats />
  },
  {
    id: 7,
    label: 'Scoring & Roll-up',
    description: 'Aggregate scores from patch to element to site level',
    status: 'planned',
    icon: <Assessment />
  },
  {
    id: 8,
    label: 'Temporal Tracking',
    description: 'Compare inspections over time to detect changes',
    status: 'planned',
    icon: <TrendingUp />
  }
];

const SAMPLE_SEGMENTS = [
  {
    id: 'Pier_P1_East_Face',
    element: 'Pier P1',
    face: 'East',
    area_sqm: 45.2,
    lidar_points: 125000,
    hsi_coverage: 95,
    images_360: 8,
    defects_detected: 3,
    quality_score: 78
  },
  {
    id: 'Pier_P1_West_Face',
    element: 'Pier P1',
    face: 'West',
    area_sqm: 45.2,
    lidar_points: 118000,
    hsi_coverage: 92,
    images_360: 7,
    defects_detected: 1,
    quality_score: 89
  },
  {
    id: 'Pier_P1_North_Face',
    element: 'Pier P1',
    face: 'North',
    area_sqm: 22.5,
    lidar_points: 62000,
    hsi_coverage: 88,
    images_360: 4,
    defects_detected: 2,
    quality_score: 82
  },
  {
    id: 'Pier_P1_South_Face',
    element: 'Pier P1',
    face: 'South',
    area_sqm: 22.5,
    lidar_points: 65000,
    hsi_coverage: 90,
    images_360: 4,
    defects_detected: 0,
    quality_score: 94
  }
];

const SAMPLE_ANALYSIS_RESULTS = {
  geometric_analysis: {
    flatness_mm: 3.2,
    verticality_mm: 2.1,
    as_built_deviation_mm: 4.5,
    surface_roughness: 'Acceptable',
    meets_tolerance: true
  },
  material_analysis: {
    concrete_strength_mpa: 42.3,
    moisture_content_pct: 3.8,
    aggregate_quality: 'Good',
    carbonation_depth_mm: 1.2,
    chloride_content: 'Low'
  },
  visual_defects: [
    { type: 'Crack', severity: 'Minor', location: [12.3, 5.6], length_mm: 45 },
    { type: 'Spalling', severity: 'Moderate', location: [18.2, 8.1], area_sqcm: 12 },
    { type: 'Discoloration', severity: 'Minor', location: [15.7, 12.3], area_sqcm: 28 }
  ]
};

export const IntegratedInspectionDemo: React.FC = () => {
  const [activeStep, setActiveStep] = useState(0);
  const [selectedSegment, setSelectedSegment] = useState<string | null>(null);

  const handleNext = () => {
    setActiveStep((prevActiveStep) => prevActiveStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  const handleReset = () => {
    setActiveStep(0);
    setSelectedSegment(null);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'implemented':
        return 'success';
      case 'partial':
        return 'warning';
      case 'planned':
        return 'default';
      default:
        return 'default';
    }
  };

  const getStatusLabel = (status: string) => {
    switch (status) {
      case 'implemented':
        return '✅ Implemented';
      case 'partial':
        return '🟡 Partial';
      case 'planned':
        return '📋 Planned';
      default:
        return status;
    }
  };

  const renderStepContent = (stepId: number) => {
    switch (stepId) {
      case 1:
        return (
          <Box>
            <Typography variant="body2" paragraph>
              Select assets for inspection and define scope based on project phase, risk assessment,
              and regulatory requirements.
            </Typography>
            <Card variant="outlined" sx={{ mt: 2 }}>
              <CardContent>
                <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                  Selected Asset: Pier P1
                </Typography>
                <Grid container spacing={2} mt={1}>
                  <Grid item xs={6}>
                    <Typography variant="caption" color="text.secondary">Asset Type</Typography>
                    <Typography variant="body2">Concrete Bridge Pier</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="caption" color="text.secondary">Inspection Date</Typography>
                    <Typography variant="body2">2024-12-10</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="caption" color="text.secondary">Project Phase</Typography>
                    <Typography variant="body2">Construction QA</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="caption" color="text.secondary">Risk Level</Typography>
                    <Chip label="Medium" size="small" color="warning" />
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Box>
        );

      case 2:
        return (
          <Box>
            <Typography variant="body2" paragraph>
              Multi-modal data collection using LiDAR scanners, hyperspectral cameras, and 360° imaging systems.
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={4}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2" fontWeight={600}>LiDAR Scan</Typography>
                    <Typography variant="caption" color="text.secondary" display="block" mb={1}>
                      1m resolution DTM
                    </Typography>
                    <Chip label="✅ Captured" size="small" color="success" />
                    <Typography variant="body2" mt={1}>370,000 points</Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} sm={4}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2" fontWeight={600}>Hyperspectral Imaging</Typography>
                    <Typography variant="caption" color="text.secondary" display="block" mb={1}>
                      204 bands (400-1000nm)
                    </Typography>
                    <Chip label="✅ Captured" size="small" color="success" />
                    <Typography variant="body2" mt={1}>4 segments scanned</Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} sm={4}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2" fontWeight={600}>360° Imagery</Typography>
                    <Typography variant="caption" color="text.secondary" display="block" mb={1}>
                      4K resolution
                    </Typography>
                    <Chip label="📋 Planned" size="small" />
                    <Typography variant="body2" mt={1}>23 images planned</Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Box>
        );

      case 3:
        return (
          <Box>
            <Typography variant="body2" paragraph>
              Upload collected data, validate quality, and preprocess for analysis.
            </Typography>
            <TableContainer component={Paper} variant="outlined" sx={{ mt: 2 }}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell><strong>Data Type</strong></TableCell>
                    <TableCell><strong>Files</strong></TableCell>
                    <TableCell><strong>Size</strong></TableCell>
                    <TableCell><strong>Status</strong></TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  <TableRow>
                    <TableCell>LiDAR Point Clouds</TableCell>
                    <TableCell>17 tiles</TableCell>
                    <TableCell>748 MB</TableCell>
                    <TableCell><Chip label="✅ Ready" size="small" color="success" /></TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Hyperspectral Images</TableCell>
                    <TableCell>50 samples</TableCell>
                    <TableCell>125 MB</TableCell>
                    <TableCell><Chip label="✅ Ready" size="small" color="success" /></TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>BIM Reference Model</TableCell>
                    <TableCell>1 IFC file</TableCell>
                    <TableCell>45 MB</TableCell>
                    <TableCell><Chip label="✅ Ready" size="small" color="success" /></TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>360° Imagery</TableCell>
                    <TableCell>0 images</TableCell>
                    <TableCell>0 MB</TableCell>
                    <TableCell><Chip label="Pending" size="small" /></TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </TableContainer>
          </Box>
        );

      case 4:
        return (
          <Box>
            <Typography variant="body2" paragraph>
              Register LiDAR point clouds with BIM reference model using Iterative Closest Point (ICP) algorithm
              to establish spatial correspondence.
            </Typography>
            <Alert severity="info" sx={{ mt: 2 }}>
              <strong>Status:</strong> Planned for future implementation. This step will enable precise
              as-built vs design comparison by aligning captured point clouds with BIM geometry.
            </Alert>
            <Card variant="outlined" sx={{ mt: 2, bgcolor: 'grey.50' }}>
              <CardContent>
                <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                  ICP Registration Parameters (Planned)
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="caption" color="text.secondary">Max Iterations</Typography>
                    <Typography variant="body2">50</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="caption" color="text.secondary">Convergence Threshold</Typography>
                    <Typography variant="body2">0.001m</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="caption" color="text.secondary">RANSAC Inlier Distance</Typography>
                    <Typography variant="body2">0.05m</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="caption" color="text.secondary">Subsampling Voxel Size</Typography>
                    <Typography variant="body2">0.01m</Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Box>
        );

      case 5:
        return (
          <Box>
            <Typography variant="body2" paragraph>
              Divide structure into inspectable segments (patches) for detailed analysis. Each segment
              aggregates data from all modalities.
            </Typography>
            <Alert severity="info" sx={{ mt: 2, mb: 2 }}>
              <strong>Status:</strong> Automatic segmentation planned. Currently showing manual segment definition.
            </Alert>
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell><strong>Segment ID</strong></TableCell>
                    <TableCell><strong>Element</strong></TableCell>
                    <TableCell><strong>Face</strong></TableCell>
                    <TableCell align="right"><strong>Area (m²)</strong></TableCell>
                    <TableCell align="right"><strong>LiDAR Points</strong></TableCell>
                    <TableCell align="right"><strong>HSI Coverage</strong></TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {SAMPLE_SEGMENTS.map((segment) => (
                    <TableRow
                      key={segment.id}
                      hover
                      onClick={() => setSelectedSegment(segment.id)}
                      sx={{
                        cursor: 'pointer',
                        bgcolor: selectedSegment === segment.id ? 'action.selected' : 'inherit'
                      }}
                    >
                      <TableCell>
                        <Typography variant="body2" fontFamily="monospace">{segment.id}</Typography>
                      </TableCell>
                      <TableCell>{segment.element}</TableCell>
                      <TableCell>{segment.face}</TableCell>
                      <TableCell align="right">{segment.area_sqm.toFixed(1)}</TableCell>
                      <TableCell align="right">{segment.lidar_points.toLocaleString()}</TableCell>
                      <TableCell align="right">{segment.hsi_coverage}%</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Box>
        );

      case 6:
        return (
          <Box>
            <Typography variant="body2" paragraph>
              Analyze each segment using geometric analysis (LiDAR), material properties (HSI),
              and visual defects (360° imagery).
            </Typography>

            {selectedSegment ? (
              <Box>
                <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                  Analysis Results: {selectedSegment}
                </Typography>

                <Grid container spacing={2} mt={1}>
                  {/* Geometric Analysis */}
                  <Grid item xs={12} md={4}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle2" fontWeight={600} color="primary" gutterBottom>
                          Geometric Analysis (LiDAR)
                        </Typography>
                        <Divider sx={{ my: 1 }} />
                        <Box>
                          <Box display="flex" justifyContent="space-between" mb={1}>
                            <Typography variant="caption" color="text.secondary">Flatness</Typography>
                            <Typography variant="body2">{SAMPLE_ANALYSIS_RESULTS.geometric_analysis.flatness_mm} mm</Typography>
                          </Box>
                          <Box display="flex" justifyContent="space-between" mb={1}>
                            <Typography variant="caption" color="text.secondary">Verticality</Typography>
                            <Typography variant="body2">{SAMPLE_ANALYSIS_RESULTS.geometric_analysis.verticality_mm} mm</Typography>
                          </Box>
                          <Box display="flex" justifyContent="space-between" mb={1}>
                            <Typography variant="caption" color="text.secondary">As-Built Deviation</Typography>
                            <Typography variant="body2">{SAMPLE_ANALYSIS_RESULTS.geometric_analysis.as_built_deviation_mm} mm</Typography>
                          </Box>
                          <Box display="flex" justifyContent="space-between" mb={1}>
                            <Typography variant="caption" color="text.secondary">Surface Roughness</Typography>
                            <Typography variant="body2">{SAMPLE_ANALYSIS_RESULTS.geometric_analysis.surface_roughness}</Typography>
                          </Box>
                          <Box display="flex" justifyContent="space-between">
                            <Typography variant="caption" color="text.secondary">Meets Tolerance</Typography>
                            <Chip
                              label={SAMPLE_ANALYSIS_RESULTS.geometric_analysis.meets_tolerance ? 'Yes' : 'No'}
                              size="small"
                              color={SAMPLE_ANALYSIS_RESULTS.geometric_analysis.meets_tolerance ? 'success' : 'error'}
                            />
                          </Box>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>

                  {/* Material Analysis */}
                  <Grid item xs={12} md={4}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle2" fontWeight={600} color="primary" gutterBottom>
                          Material Analysis (HSI)
                        </Typography>
                        <Divider sx={{ my: 1 }} />
                        <Box>
                          <Box display="flex" justifyContent="space-between" mb={1}>
                            <Typography variant="caption" color="text.secondary">Concrete Strength</Typography>
                            <Typography variant="body2">{SAMPLE_ANALYSIS_RESULTS.material_analysis.concrete_strength_mpa} MPa</Typography>
                          </Box>
                          <Box display="flex" justifyContent="space-between" mb={1}>
                            <Typography variant="caption" color="text.secondary">Moisture Content</Typography>
                            <Typography variant="body2">{SAMPLE_ANALYSIS_RESULTS.material_analysis.moisture_content_pct}%</Typography>
                          </Box>
                          <Box display="flex" justifyContent="space-between" mb={1}>
                            <Typography variant="caption" color="text.secondary">Aggregate Quality</Typography>
                            <Typography variant="body2">{SAMPLE_ANALYSIS_RESULTS.material_analysis.aggregate_quality}</Typography>
                          </Box>
                          <Box display="flex" justifyContent="space-between" mb={1}>
                            <Typography variant="caption" color="text.secondary">Carbonation Depth</Typography>
                            <Typography variant="body2">{SAMPLE_ANALYSIS_RESULTS.material_analysis.carbonation_depth_mm} mm</Typography>
                          </Box>
                          <Box display="flex" justifyContent="space-between">
                            <Typography variant="caption" color="text.secondary">Chloride Content</Typography>
                            <Chip label={SAMPLE_ANALYSIS_RESULTS.material_analysis.chloride_content} size="small" color="success" />
                          </Box>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>

                  {/* Visual Defects */}
                  <Grid item xs={12} md={4}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle2" fontWeight={600} color="primary" gutterBottom>
                          Visual Defects
                        </Typography>
                        <Divider sx={{ my: 1 }} />
                        {SAMPLE_ANALYSIS_RESULTS.visual_defects.map((defect, index) => (
                          <Card key={index} variant="outlined" sx={{ mb: 1, p: 1 }}>
                            <Box display="flex" justifyContent="space-between" alignItems="center" mb={0.5}>
                              <Typography variant="caption" fontWeight={600}>{defect.type}</Typography>
                              <Chip
                                label={defect.severity}
                                size="small"
                                color={defect.severity === 'Minor' ? 'success' : 'warning'}
                              />
                            </Box>
                            <Typography variant="caption" color="text.secondary" display="block">
                              Location: ({defect.location[0]}, {defect.location[1]})
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {defect.length_mm ? `Length: ${defect.length_mm}mm` : `Area: ${defect.area_sqcm}cm²`}
                            </Typography>
                          </Card>
                        ))}
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>
              </Box>
            ) : (
              <Alert severity="info" sx={{ mt: 2 }}>
                Select a segment from Step 5 to view detailed analysis results.
              </Alert>
            )}
          </Box>
        );

      case 7:
        return (
          <Box>
            <Typography variant="body2" paragraph>
              Aggregate individual analysis scores from patch level → element level → site level
              for comprehensive quality assessment.
            </Typography>
            <Alert severity="info" sx={{ mt: 2, mb: 2 }}>
              <strong>Status:</strong> Scoring algorithm planned. Sample scores shown below.
            </Alert>

            {/* Element-Level Scores */}
            <Paper variant="outlined" sx={{ p: 2, mb: 2 }}>
              <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                Element-Level Quality Scores
              </Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell><strong>Element</strong></TableCell>
                      <TableCell align="right"><strong>Geometric</strong></TableCell>
                      <TableCell align="right"><strong>Material</strong></TableCell>
                      <TableCell align="right"><strong>Visual</strong></TableCell>
                      <TableCell align="right"><strong>Overall Score</strong></TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    <TableRow>
                      <TableCell>Pier P1</TableCell>
                      <TableCell align="right">92/100</TableCell>
                      <TableCell align="right">88/100</TableCell>
                      <TableCell align="right">76/100</TableCell>
                      <TableCell align="right">
                        <Chip label="85/100" color="success" size="small" />
                      </TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </TableContainer>
            </Paper>

            {/* Site-Level Score */}
            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6" fontWeight={600} gutterBottom>
                  Site-Level Quality Score
                </Typography>
                <Box display="flex" alignItems="center" gap={2} mt={2}>
                  <Box flex={1}>
                    <LinearProgress
                      variant="determinate"
                      value={85}
                      sx={{ height: 10, borderRadius: 1 }}
                      color="success"
                    />
                  </Box>
                  <Typography variant="h5" fontWeight={600} color="success.main">
                    85/100
                  </Typography>
                </Box>
                <Typography variant="caption" color="text.secondary" display="block" mt={1}>
                  Based on 4 segments, 1 element analyzed
                </Typography>
              </CardContent>
            </Card>
          </Box>
        );

      case 8:
        return (
          <Box>
            <Typography variant="body2" paragraph>
              Track quality scores over time to detect trends, predict maintenance needs, and validate
              construction progress.
            </Typography>
            <Alert severity="info" sx={{ mt: 2, mb: 2 }}>
              <strong>Status:</strong> Temporal tracking planned. Requires multiple inspection cycles.
            </Alert>

            <Card variant="outlined">
              <CardContent>
                <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                  Inspection History (Simulated)
                </Typography>
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell><strong>Date</strong></TableCell>
                        <TableCell><strong>Inspector</strong></TableCell>
                        <TableCell align="right"><strong>Score</strong></TableCell>
                        <TableCell align="right"><strong>Trend</strong></TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      <TableRow>
                        <TableCell>2024-12-10</TableCell>
                        <TableCell>Current Inspection</TableCell>
                        <TableCell align="right">85/100</TableCell>
                        <TableCell align="right">
                          <Chip label="Baseline" size="small" />
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>2024-12-17 (Planned)</TableCell>
                        <TableCell>Follow-up</TableCell>
                        <TableCell align="right">-</TableCell>
                        <TableCell align="right">
                          <Chip label="Pending" size="small" />
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </Box>
        );

      default:
        return null;
    }
  };

  return (
    <Box>
      {/* Header */}
      <Box mb={4}>
        <Box display="flex" alignItems="center" gap={2} mb={2}>
          <Assessment sx={{ fontSize: 40, color: 'primary.main' }} />
          <Box>
            <Typography variant="h4" component="h2" gutterBottom>
              HS2 Integrated Inspection Demo
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Complete 8-step multi-modal inspection workflow demonstration
            </Typography>
          </Box>
        </Box>

        <Box display="flex" gap={1} flexWrap="wrap">
          <Chip label="🟢 Real LiDAR Data" size="small" sx={{ bgcolor: 'rgb(220, 252, 231)', color: 'rgb(22, 101, 52)', fontWeight: 600 }} />
          <Chip label="🟢 Real HSI Data" size="small" sx={{ bgcolor: 'rgb(220, 252, 231)', color: 'rgb(22, 101, 52)', fontWeight: 600 }} />
          <Chip label="🟢 Real BIM Model" size="small" sx={{ bgcolor: 'rgb(220, 252, 231)', color: 'rgb(22, 101, 52)', fontWeight: 600 }} />
          <Chip label="Demo Mode" size="small" variant="outlined" />
        </Box>
      </Box>

      {/* Workflow Status Overview */}
      <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom fontWeight={600}>
          Workflow Status Overview
        </Typography>
        <Grid container spacing={2} mt={1}>
          {WORKFLOW_STEPS.map((step) => (
            <Grid item xs={12} sm={6} md={3} key={step.id}>
              <Card variant="outlined" sx={{ height: '100%' }}>
                <CardContent>
                  <Box display="flex" alignItems="center" gap={1} mb={1}>
                    {step.icon}
                    <Typography variant="subtitle2" fontWeight={600}>
                      Step {step.id}
                    </Typography>
                  </Box>
                  <Typography variant="caption" display="block" mb={1}>
                    {step.label}
                  </Typography>
                  <Chip
                    label={getStatusLabel(step.status)}
                    size="small"
                    color={getStatusColor(step.status)}
                  />
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Paper>

      {/* Interactive Stepper */}
      <Paper elevation={2} sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom fontWeight={600}>
          Interactive Workflow Walkthrough
        </Typography>

        <Stepper activeStep={activeStep} orientation="vertical" sx={{ mt: 3 }}>
          {WORKFLOW_STEPS.map((step, index) => (
            <Step key={step.id}>
              <StepLabel
                optional={
                  <Chip
                    label={getStatusLabel(step.status)}
                    size="small"
                    color={getStatusColor(step.status)}
                  />
                }
              >
                <Typography variant="subtitle1" fontWeight={600}>
                  {step.label}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {step.description}
                </Typography>
              </StepLabel>
              <StepContent>
                {renderStepContent(step.id)}

                <Box sx={{ mb: 2, mt: 2 }}>
                  <div>
                    <Button
                      variant="contained"
                      onClick={handleNext}
                      sx={{ mt: 1, mr: 1 }}
                    >
                      {index === WORKFLOW_STEPS.length - 1 ? 'Finish' : 'Continue'}
                    </Button>
                    <Button
                      disabled={index === 0}
                      onClick={handleBack}
                      sx={{ mt: 1, mr: 1 }}
                    >
                      Back
                    </Button>
                  </div>
                </Box>
              </StepContent>
            </Step>
          ))}
        </Stepper>

        {activeStep === WORKFLOW_STEPS.length && (
          <Paper square elevation={0} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Workflow Complete
            </Typography>
            <Typography variant="body2" paragraph>
              You've reviewed all 8 steps of the integrated inspection workflow. This demonstration
              shows how LiDAR, hyperspectral imaging, 360° photography, and BIM data combine to provide
              comprehensive asset quality assessment.
            </Typography>
            <Alert severity="success" sx={{ mt: 2, mb: 2 }}>
              <strong>Next Steps:</strong> Visit the LiDAR Viewer and Hyperspectral Viewer tabs to
              explore real data analysis capabilities in detail.
            </Alert>
            <Button onClick={handleReset} sx={{ mt: 1, mr: 1 }}>
              Reset
            </Button>
          </Paper>
        )}
      </Paper>

      {/* Implementation Roadmap */}
      <Paper elevation={2} sx={{ p: 3, mt: 3 }}>
        <Typography variant="h6" gutterBottom fontWeight={600}>
          <Warning sx={{ mr: 1, verticalAlign: 'middle' }} />
          Implementation Roadmap
        </Typography>
        <Typography variant="body2" color="text.secondary" paragraph>
          This demo shows the complete vision for multi-modal inspection. Current implementation status:
        </Typography>

        <Grid container spacing={2} mt={1}>
          <Grid item xs={12} md={6}>
            <Card variant="outlined" sx={{ bgcolor: 'rgb(220, 252, 231)' }}>
              <CardContent>
                <Typography variant="subtitle2" fontWeight={600} color="success.dark" gutterBottom>
                  ✅ Currently Implemented
                </Typography>
                <Box component="ul" sx={{ pl: 2, mt: 1 }}>
                  <Typography component="li" variant="body2">LiDAR elevation profile generation</Typography>
                  <Typography component="li" variant="body2">Hyperspectral material classification</Typography>
                  <Typography component="li" variant="body2">BIM model viewing (IFC format)</Typography>
                  <Typography component="li" variant="body2">Individual data type analysis</Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card variant="outlined" sx={{ bgcolor: 'rgb(254, 243, 199)' }}>
              <CardContent>
                <Typography variant="subtitle2" fontWeight={600} color="warning.dark" gutterBottom>
                  📋 Planned Features
                </Typography>
                <Box component="ul" sx={{ pl: 2, mt: 1 }}>
                  <Typography component="li" variant="body2">ICP point cloud registration</Typography>
                  <Typography component="li" variant="body2">Automatic spatial segmentation</Typography>
                  <Typography component="li" variant="body2">Multi-modal data fusion</Typography>
                  <Typography component="li" variant="body2">Temporal change tracking</Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Paper>
    </Box>
  );
};

export default IntegratedInspectionDemo;
