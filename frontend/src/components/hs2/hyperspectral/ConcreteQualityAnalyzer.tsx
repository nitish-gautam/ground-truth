/**
 * Hyperspectral Concrete Quality Analyzer
 * ========================================
 *
 * CRITICAL COMPONENT for HS2 concrete quality assessment using Specim IQ camera.
 *
 * Features:
 * - Image upload for hyperspectral analysis
 * - Concrete strength prediction (Target: R²=0.89)
 * - Defect detection and visualization
 * - Spectral signature display
 * - Quality score dashboard
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
  Chip,
  Paper,
  LinearProgress,
  Divider
} from '@mui/material';
import {
  CloudUpload as Upload,
  CheckCircle as CheckCircle2,
  Cancel as XCircle,
  Warning as AlertTriangle,
  ShowChart as Activity,
  Opacity as Droplet,
  Layers
} from '@mui/icons-material';

interface AnalysisResult {
  analysis_id: string;
  image_metadata: {
    width: number;
    height: number;
    analyzed_at: string;
  };
  material_classification?: {
    material_type: string;
    confidence: number;
  };
  concrete_strength?: {
    predicted_strength_mpa: number;
    confidence: number;
    strength_range_min: number;
    strength_range_max: number;
    model_r_squared: number;
    key_wavelength_values: {
      cement_hydration_500_600: number;
      moisture_content_700_850: number;
      aggregate_quality_900_1000: number;
    };
  };
  defects?: {
    defects_detected: Array<{
      defect_type: string;
      location_x: number;
      location_y: number;
      confidence: number;
      severity: string;
    }>;
    num_defects: number;
    overall_severity: string;
  };
}

export function ConcreteQualityAnalyzer() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreviewUrl(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
    }
  };

  const analyzeImage = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/v1/progress/hyperspectral/analyze-material', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to analyze image');
    } finally {
      setLoading(false);
    }
  };

  const clearAnalysis = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  // Determine pass/fail based on C40/50 specification (40 MPa minimum)
  const isPass = result?.concrete_strength
    ? result.concrete_strength.predicted_strength_mpa >= 40
    : null;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      {/* Upload Section */}
      <Card elevation={2}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Activity color="primary" />
              <Typography variant="h6" fontWeight={600}>
                Hyperspectral Concrete Analysis
              </Typography>
            </Box>
            <Chip
              label="🟢 REAL DATA (UMKC Dataset)"
              size="small"
              sx={{ bgcolor: 'rgb(220, 252, 231)', color: 'rgb(22, 101, 52)', fontWeight: 600 }}
            />
          </Box>

          <Box
            sx={{
              border: '2px dashed',
              borderColor: 'grey.300',
              borderRadius: 2,
              p: 4,
              textAlign: 'center',
              bgcolor: 'grey.50',
              mb: 2
            }}
          >
            <Upload sx={{ fontSize: 48, color: 'grey.400', mb: 2 }} />
            <Typography variant="body1" gutterBottom>
              Upload hyperspectral image from Specim IQ camera
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Supported formats: .hdr, .img, .tif (204-band hyperspectral)
            </Typography>
            <input
              accept=".hdr,.img,.tif,.tiff,image/*"
              style={{ display: 'none' }}
              id="hyperspectral-file-upload"
              type="file"
              onChange={handleFileChange}
            />
            <label htmlFor="hyperspectral-file-upload">
              <Button variant="contained" component="span">
                Choose File
              </Button>
            </label>
            {file && (
              <Typography variant="body2" sx={{ mt: 2 }} color="primary">
                Selected: {file.name}
              </Typography>
            )}
          </Box>

          {previewUrl && (
            <Box sx={{ mb: 2, textAlign: 'center' }}>
              <img
                src={previewUrl}
                alt="Preview"
                style={{ maxWidth: '100%', maxHeight: '300px', borderRadius: '8px' }}
              />
            </Box>
          )}

          <Box sx={{ display: 'flex', gap: 2 }}>
            <Button
              variant="contained"
              fullWidth
              onClick={analyzeImage}
              disabled={!file || loading}
              startIcon={loading ? <CircularProgress size={20} /> : null}
            >
              {loading ? 'Analyzing...' : 'Analyze Concrete'}
            </Button>
            {(file || result) && (
              <Button variant="outlined" onClick={clearAnalysis} disabled={loading}>
                Clear
              </Button>
            )}
          </Box>

          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Results Section */}
      {result && (
        <>
          {/* Pass/Fail Banner */}
          {result.concrete_strength && (
            <Alert
              severity={isPass ? 'success' : 'error'}
              icon={isPass ? <CheckCircle2 /> : <XCircle />}
              sx={{ fontSize: '1.1rem' }}
            >
              <Typography variant="h6" fontWeight={600}>
                {isPass ? 'PASS - Specification Met' : 'FAIL - Below Specification'}
              </Typography>
              <Typography variant="body2">
                {isPass
                  ? `Concrete strength meets C40/50 specification (≥40 MPa)`
                  : `Concrete strength below C40/50 specification (minimum 40 MPa required)`}
              </Typography>
            </Alert>
          )}

          {/* Strength Prediction */}
          {result.concrete_strength && (
            <Card elevation={2}>
              <CardContent>
                <Typography variant="h6" gutterBottom fontWeight={600}>
                  Concrete Strength Prediction
                </Typography>
                <Divider sx={{ mb: 2 }} />

                <Grid container spacing={3}>
                  <Grid item xs={12} md={4}>
                    <Paper elevation={1} sx={{ p: 2, bgcolor: isPass ? 'success.50' : 'error.50' }}>
                      <Typography variant="body2" color="text.secondary">
                        Predicted Strength
                      </Typography>
                      <Typography variant="h3" fontWeight={600} color={isPass ? 'success.main' : 'error.main'}>
                        {result.concrete_strength.predicted_strength_mpa.toFixed(1)} MPa
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Range: {result.concrete_strength.strength_range_min.toFixed(1)} - {result.concrete_strength.strength_range_max.toFixed(1)} MPa
                      </Typography>
                    </Paper>
                  </Grid>

                  <Grid item xs={12} md={4}>
                    <Paper elevation={1} sx={{ p: 2 }}>
                      <Typography variant="body2" color="text.secondary">
                        Model Confidence
                      </Typography>
                      <Typography variant="h3" fontWeight={600}>
                        {result.concrete_strength.confidence.toFixed(1)}%
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={result.concrete_strength.confidence}
                        sx={{ mt: 1 }}
                      />
                    </Paper>
                  </Grid>

                  <Grid item xs={12} md={4}>
                    <Paper elevation={1} sx={{ p: 2 }}>
                      <Typography variant="body2" color="text.secondary">
                        Model R² Score
                      </Typography>
                      <Typography variant="h3" fontWeight={600}>
                        {result.concrete_strength.model_r_squared.toFixed(3)}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Target: ≥0.89 (Lab), ≥0.82 (Field)
                      </Typography>
                    </Paper>
                  </Grid>
                </Grid>

                <Typography variant="subtitle1" sx={{ mt: 3, mb: 1 }} fontWeight={600}>
                  Key Wavelength Analysis
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={4}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
                      <Layers sx={{ color: 'primary.main' }} />
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          Cement Hydration (500-600nm)
                        </Typography>
                        <Typography variant="h6" fontWeight={600}>
                          {result.concrete_strength.key_wavelength_values.cement_hydration_500_600.toFixed(3)}
                        </Typography>
                      </Box>
                    </Box>
                  </Grid>

                  <Grid item xs={12} sm={4}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
                      <Droplet sx={{ color: 'info.main' }} />
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          Moisture Content (700-850nm)
                        </Typography>
                        <Typography variant="h6" fontWeight={600}>
                          {result.concrete_strength.key_wavelength_values.moisture_content_700_850.toFixed(3)}
                        </Typography>
                      </Box>
                    </Box>
                  </Grid>

                  <Grid item xs={12} sm={4}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
                      <Layers sx={{ color: 'secondary.main' }} />
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          Aggregate Quality (900-1000nm)
                        </Typography>
                        <Typography variant="h6" fontWeight={600}>
                          {result.concrete_strength.key_wavelength_values.aggregate_quality_900_1000.toFixed(3)}
                        </Typography>
                      </Box>
                    </Box>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          )}

          {/* Defect Detection */}
          {result.defects && result.defects.num_defects > 0 && (
            <Card elevation={2}>
              <CardContent>
                <Typography variant="h6" gutterBottom fontWeight={600}>
                  Defect Detection
                </Typography>
                <Divider sx={{ mb: 2 }} />

                <Alert severity="warning" sx={{ mb: 2 }}>
                  <Typography variant="body1" fontWeight={600}>
                    {result.defects.num_defects} defect(s) detected
                  </Typography>
                  <Typography variant="body2">
                    Overall Severity: {result.defects.overall_severity}
                  </Typography>
                </Alert>

                <Grid container spacing={2}>
                  {result.defects.defects_detected.slice(0, 6).map((defect, idx) => (
                    <Grid item xs={12} sm={6} key={idx}>
                      <Paper elevation={1} sx={{ p: 2 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                          <AlertTriangle
                            sx={{
                              color:
                                defect.severity === 'high'
                                  ? 'error.main'
                                  : defect.severity === 'medium'
                                  ? 'warning.main'
                                  : 'info.main',
                            }}
                          />
                          <Typography variant="body1" fontWeight={600}>
                            {defect.defect_type}
                          </Typography>
                        </Box>
                        <Typography variant="body2" color="text.secondary">
                          Location: ({defect.location_x}, {defect.location_y})
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Confidence: {(defect.confidence * 100).toFixed(1)}%
                        </Typography>
                        <Chip
                          label={defect.severity.toUpperCase()}
                          size="small"
                          color={
                            defect.severity === 'high'
                              ? 'error'
                              : defect.severity === 'medium'
                              ? 'warning'
                              : 'info'
                          }
                          sx={{ mt: 1 }}
                        />
                      </Paper>
                    </Grid>
                  ))}
                </Grid>
              </CardContent>
            </Card>
          )}

          {/* Material Classification */}
          {result.material_classification && (
            <Card elevation={2}>
              <CardContent>
                <Typography variant="h6" gutterBottom fontWeight={600}>
                  Material Classification
                </Typography>
                <Divider sx={{ mb: 2 }} />
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Typography variant="body1">
                    Material Type: <strong>{result.material_classification.material_type}</strong>
                  </Typography>
                  <Chip
                    label={`${result.material_classification.confidence.toFixed(1)}% confidence`}
                    color="primary"
                  />
                </Box>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </Box>
  );
}
