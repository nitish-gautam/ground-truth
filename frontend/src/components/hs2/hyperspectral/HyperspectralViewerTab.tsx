/**
 * Hyperspectral Viewer Tab
 * Dedicated tab for hyperspectral imaging analysis
 * Uses Specim IQ camera data for material classification and quality assessment
 */

import React from 'react';
import {
  Box,
  Paper,
  Typography,
  Chip,
  Grid,
  Card,
  CardContent
} from '@mui/material';
import { Colorize, Science } from '@mui/icons-material';
import { ConcreteQualityAnalyzer } from './ConcreteQualityAnalyzer';

export const HyperspectralViewerTab: React.FC = () => {
  return (
    <Box>
      {/* Tab Header */}
      <Box mb={4}>
        <Box display="flex" alignItems="center" gap={2} mb={2}>
          <Colorize sx={{ fontSize: 40, color: 'primary.main' }} />
          <Box>
            <Typography variant="h4" component="h2" gutterBottom>
              Hyperspectral Imaging Analysis
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Advanced material classification and quality assessment using spectral signatures
            </Typography>
          </Box>
        </Box>

        <Box display="flex" gap={1} flexWrap="wrap">
          <Chip
            label="🟢 REAL DATA"
            size="small"
            sx={{ bgcolor: 'rgb(220, 252, 231)', color: 'rgb(22, 101, 52)', fontWeight: 600 }}
          />
          <Chip
            label="Specim IQ Camera"
            size="small"
            variant="outlined"
            color="primary"
          />
          <Chip
            label="400-1000nm Range"
            size="small"
            variant="outlined"
            color="primary"
          />
          <Chip
            label="204 Spectral Bands"
            size="small"
            variant="outlined"
            color="primary"
          />
        </Box>
      </Box>

      {/* Hyperspectral Analyzer */}
      <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
        <ConcreteQualityAnalyzer />
      </Paper>

      {/* Technology Information */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper elevation={2} sx={{ p: 3, height: '100%' }}>
            <Box display="flex" alignItems="center" gap={1} mb={2}>
              <Science color="primary" />
              <Typography variant="h6" fontWeight={600}>
                Hyperspectral Technology
              </Typography>
            </Box>

            <Typography variant="body2" color="text.secondary" paragraph>
              Hyperspectral imaging captures data across hundreds of narrow spectral bands,
              revealing material properties invisible to conventional cameras. Each pixel contains
              a complete spectral signature enabling precise material identification and quality assessment.
            </Typography>

            <Box mt={2}>
              <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                Key Capabilities:
              </Typography>
              <Box component="ul" sx={{ mt: 1, pl: 2 }}>
                <Typography component="li" variant="body2" color="text.secondary">
                  Material classification (concrete, asphalt, steel, etc.)
                </Typography>
                <Typography component="li" variant="body2" color="text.secondary">
                  Concrete strength prediction (R² = 0.89 target accuracy)
                </Typography>
                <Typography component="li" variant="body2" color="text.secondary">
                  Moisture content detection (critical for curing)
                </Typography>
                <Typography component="li" variant="body2" color="text.secondary">
                  Defect identification (cracks, voids, delamination) - Phase 2
                </Typography>
                <Typography component="li" variant="body2" color="text.secondary">
                  Aggregate quality assessment
                </Typography>
              </Box>
            </Box>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper elevation={2} sx={{ p: 3, height: '100%' }}>
            <Box display="flex" alignItems="center" gap={1} mb={2}>
              <Colorize color="primary" />
              <Typography variant="h6" fontWeight={600}>
                Data Sources
              </Typography>
            </Box>

            <Typography variant="body2" color="text.secondary" paragraph>
              The platform integrates hyperspectral data from multiple sources for comprehensive
              material analysis across HS2 construction sites.
            </Typography>

            <Box mt={2}>
              <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                Available Datasets:
              </Typography>
              <Box mt={1}>
                <Card variant="outlined" sx={{ mb: 1 }}>
                  <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
                    <Typography variant="subtitle2" fontWeight={600}>
                      UMKC Material Surfaces Dataset
                    </Typography>
                    <Typography variant="caption" color="text.secondary" display="block">
                      150 calibrated HSI samples (75 concrete, 75 asphalt)
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      50×50 pixels per sample, full spectral resolution
                    </Typography>
                  </CardContent>
                </Card>

                <Card variant="outlined" sx={{ mb: 1 }}>
                  <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
                    <Typography variant="subtitle2" fontWeight={600}>
                      Field Survey Data
                    </Typography>
                    <Typography variant="caption" color="text.secondary" display="block">
                      Specim IQ camera captures from HS2 sites
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Real-world conditions with environmental variations
                    </Typography>
                  </CardContent>
                </Card>
              </Box>
            </Box>
          </Paper>
        </Grid>

        <Grid item xs={12}>
          <Paper elevation={2} sx={{ p: 3 }}>
            <Typography variant="h6" fontWeight={600} gutterBottom>
              Spectral Analysis Workflow
            </Typography>

            <Grid container spacing={2} mt={1}>
              <Grid item xs={12} sm={6} md={3}>
                <Box textAlign="center">
                  <Box
                    sx={{
                      width: 60,
                      height: 60,
                      borderRadius: '50%',
                      bgcolor: 'primary.light',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      margin: '0 auto',
                      mb: 1
                    }}
                  >
                    <Typography variant="h6" fontWeight={600} color="white">
                      1
                    </Typography>
                  </Box>
                  <Typography variant="subtitle2" fontWeight={600}>
                    Image Capture
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Upload hyperspectral image (TIFF/HSI format)
                  </Typography>
                </Box>
              </Grid>

              <Grid item xs={12} sm={6} md={3}>
                <Box textAlign="center">
                  <Box
                    sx={{
                      width: 60,
                      height: 60,
                      borderRadius: '50%',
                      bgcolor: 'primary.light',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      margin: '0 auto',
                      mb: 1
                    }}
                  >
                    <Typography variant="h6" fontWeight={600} color="white">
                      2
                    </Typography>
                  </Box>
                  <Typography variant="subtitle2" fontWeight={600}>
                    Spectral Extraction
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Extract 204 bands (400-1000nm) per pixel
                  </Typography>
                </Box>
              </Grid>

              <Grid item xs={12} sm={6} md={3}>
                <Box textAlign="center">
                  <Box
                    sx={{
                      width: 60,
                      height: 60,
                      borderRadius: '50%',
                      bgcolor: 'primary.light',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      margin: '0 auto',
                      mb: 1
                    }}
                  >
                    <Typography variant="h6" fontWeight={600} color="white">
                      3
                    </Typography>
                  </Box>
                  <Typography variant="subtitle2" fontWeight={600}>
                    ML Classification
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Random Forest model identifies materials
                  </Typography>
                </Box>
              </Grid>

              <Grid item xs={12} sm={6} md={3}>
                <Box textAlign="center">
                  <Box
                    sx={{
                      width: 60,
                      height: 60,
                      borderRadius: '50%',
                      bgcolor: 'primary.light',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      margin: '0 auto',
                      mb: 1
                    }}
                  >
                    <Typography variant="h6" fontWeight={600} color="white">
                      4
                    </Typography>
                  </Box>
                  <Typography variant="subtitle2" fontWeight={600}>
                    Quality Assessment
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Predict strength, detect defects, score quality
                  </Typography>
                </Box>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default HyperspectralViewerTab;
