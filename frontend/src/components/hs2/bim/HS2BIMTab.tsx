/**
 * HS2 BIM Model Viewer Tab
 * 3D visualization of IFC BIM models using Three.js
 * NOW LOADING REAL IFC MODEL DATA FROM BACKEND API
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  Grid,
  Card,
  CardHeader,
  CardContent,
  Typography,
  Box,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Chip,
  Paper,
  Alert,
  Collapse,
  IconButton,
  CircularProgress
} from '@mui/material';
import {
  InsertDriveFile,
  ExpandMore,
  ExpandLess,
  ViewInAr,
  Folder,
  Architecture
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

interface BIMModel {
  id: string;
  name: string;
  filename: string;
  category: string;
  size_mb: number;
  size_kb: number;
  path: string;
}

export const HS2BIMTab: React.FC = () => {
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [expandedCategories, setExpandedCategories] = useState<string[]>([]);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);

  // Fetch real BIM models from backend API
  const { data: bimData, isLoading: modelsLoading } = useQuery({
    queryKey: ['bim-models'],
    queryFn: async () => {
      const response = await axios.get('/api/v1/bim/models');
      return response.data;
    }
  });

  // Get real model list and categories from API
  const bimModels: BIMModel[] = bimData?.models || [];

  // Fetch categories separately
  const { data: categoriesData } = useQuery({
    queryKey: ['bim-categories'],
    queryFn: async () => {
      const response = await axios.get('/api/v1/bim/categories');
      return response.data;
    }
  });

  const modelCategories = categoriesData?.categories || [];
  const totalModels = bimData?.total_count || 0;

  const toggleCategory = (category: string) => {
    setExpandedCategories(prev =>
      prev.includes(category)
        ? prev.filter(c => c !== category)
        : [...prev, category]
    );
  };

  const handleModelSelect = (modelId: string) => {
    setSelectedModel(modelId);
    // In a real implementation, this would load the actual IFC file
    createSampleGeometry(modelId);
  };

  const createSampleGeometry = (modelId: string) => {
    if (!sceneRef.current) return;

    // Clear existing geometry
    while (sceneRef.current.children.length > 3) {
      sceneRef.current.remove(sceneRef.current.children[3]);
    }

    // Create sample geometry based on model type
    let geometry: THREE.BufferGeometry;
    let material = new THREE.MeshPhongMaterial({
      color: 0x019C4B,
      specular: 0x111111,
      shininess: 30,
      side: THREE.DoubleSide
    });

    if (modelId.includes('beam')) {
      // I-Beam shape
      geometry = new THREE.BoxGeometry(10, 0.5, 0.5);
      const geometry2 = new THREE.BoxGeometry(0.5, 2, 0.5);
      const mesh1 = new THREE.Mesh(geometry, material);
      const mesh2 = new THREE.Mesh(geometry2, material);
      mesh1.position.y = 1;
      mesh2.position.y = 0;
      sceneRef.current.add(mesh1);
      sceneRef.current.add(mesh2);
      return;
    } else if (modelId.includes('column')) {
      geometry = new THREE.BoxGeometry(0.5, 5, 0.5);
    } else if (modelId.includes('basin') || modelId.includes('brep')) {
      geometry = new THREE.TorusKnotGeometry(1.5, 0.5, 100, 16);
    } else if (modelId.includes('wall')) {
      geometry = new THREE.BoxGeometry(10, 3, 0.3);
    } else if (modelId.includes('slab')) {
      geometry = new THREE.BoxGeometry(10, 0.3, 8);
    } else if (modelId.includes('alignment')) {
      // Create a curved railway track path
      const curve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(-5, 0, 0),
        new THREE.Vector3(-2, 0.5, 2),
        new THREE.Vector3(0, 0, 4),
        new THREE.Vector3(2, -0.5, 6),
        new THREE.Vector3(5, 0, 8)
      ]);
      geometry = new THREE.TubeGeometry(curve, 50, 0.3, 8, false);
      material = new THREE.MeshPhongMaterial({ color: 0x012A39, side: THREE.DoubleSide });
    } else {
      geometry = new THREE.BoxGeometry(2, 2, 2);
    }

    const mesh = new THREE.Mesh(geometry, material);
    sceneRef.current.add(mesh);
  };

  useEffect(() => {
    if (!canvasRef.current) return;

    // Initialize Three.js scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf5f5f5);
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(
      75,
      canvasRef.current.clientWidth / canvasRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.set(5, 5, 10);
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({
      canvas: canvasRef.current,
      antialias: true
    });
    renderer.setSize(canvasRef.current.clientWidth, canvasRef.current.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    rendererRef.current = renderer;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 10);
    scene.add(directionalLight);

    // Grid
    const gridHelper = new THREE.GridHelper(20, 20, 0x012A39, 0xcccccc);
    scene.add(gridHelper);

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controlsRef.current = controls;

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Handle resize
    const handleResize = () => {
      if (!canvasRef.current || !camera || !renderer) return;
      camera.aspect = canvasRef.current.clientWidth / canvasRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(canvasRef.current.clientWidth, canvasRef.current.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    // Load default model
    createSampleGeometry('beam-curved-i-shape');
    setSelectedModel('beam-curved-i-shape');

    return () => {
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
    };
  }, []);

  const getSelectedModelInfo = () => {
    return bimModels.find(m => m.id === selectedModel);
  };

  return (
    <Box>
      <Alert severity={bimModels.length > 0 ? "success" : "info"} sx={{ mb: 3 }}>
        <Typography variant="body2">
          <strong>3D BIM Model Viewer</strong> - Browse and visualize IFC 4.3.x models
          <br />
          {modelsLoading ? (
            <>Loading real IFC model data... <CircularProgress size={16} sx={{ ml: 1 }} /></>
          ) : bimModels.length > 0 ? (
            <>✅ Loaded <strong>{totalModels} real IFC models</strong> from datasets (Sample 3D geometry shown - full IFC parsing coming soon)</>
          ) : (
            <>No models available - check backend connection</>
          )}
        </Typography>
      </Alert>

      <Grid container spacing={3}>
        {/* Model File Selector */}
        <Grid item xs={12} md={4}>
          <Card elevation={3}>
            <CardHeader
              title="BIM Model Files"
              subheader={`${bimModels.length} models available`}
              sx={{
                bgcolor: 'primary.main',
                '& .MuiCardHeader-title': {
                  color: 'primary.contrastText',
                  fontWeight: 600
                },
                '& .MuiCardHeader-subheader': {
                  color: 'rgba(255,255,255,0.7)'
                }
              }}
            />
            <CardContent sx={{ p: 0, maxHeight: 600, overflow: 'auto' }}>
              <List>
                {modelCategories.map((category: any) => (
                  <Box key={category.name}>
                    <ListItemButton onClick={() => toggleCategory(category.name)} sx={{ bgcolor: 'grey.100' }}>
                      <ListItemIcon>
                        <Architecture color="primary" />
                      </ListItemIcon>
                      <ListItemText
                        primary={category.name}
                        secondary={`${category.model_count} models • ${category.total_size_mb.toFixed(2)} MB`}
                      />
                      {expandedCategories.includes(category.name) ? <ExpandLess /> : <ExpandMore />}
                    </ListItemButton>
                    <Collapse in={expandedCategories.includes(category.name)}>
                      {bimModels
                        .filter(model => model.category === category.name)
                        .map(model => (
                          <ListItemButton
                            key={model.id}
                            selected={selectedModel === model.id}
                            onClick={() => handleModelSelect(model.id)}
                            sx={{ pl: 4, borderBottom: 1, borderColor: 'divider' }}
                          >
                            <ListItemIcon>
                              <InsertDriveFile fontSize="small" />
                            </ListItemIcon>
                            <ListItemText
                              primary={model.name}
                              secondary={`${model.size_kb} KB${model.filename ? ` • ${model.filename}` : ''}`}
                              primaryTypographyProps={{ variant: 'body2' }}
                              secondaryTypographyProps={{ variant: 'caption' }}
                            />
                          </ListItemButton>
                        ))}
                    </Collapse>
                  </Box>
                ))}
              </List>

              <Box sx={{ p: 2, bgcolor: 'grey.100', borderTop: 1, borderColor: 'divider' }}>
                <Typography variant="caption" color="text.secondary">
                  <strong>Format:</strong> IFC 4.3.x
                  <br />
                  <strong>Source:</strong> buildingSMART Sample Models
                  <br />
                  <strong>Total Size:</strong> ~4.2 MB
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* 3D Viewer */}
        <Grid item xs={12} md={8}>
          <Paper elevation={3} sx={{ borderRadius: 2, overflow: 'hidden' }}>
            <Box sx={{ bgcolor: 'primary.main', color: 'white', p: 2 }}>
              <Typography variant="h6">3D Model Viewer</Typography>
              <Typography variant="caption">
                {getSelectedModelInfo()?.name || 'Select a model to view'}
              </Typography>
            </Box>
            <Box sx={{ position: 'relative', height: 500, bgcolor: '#f5f5f5' }}>
              <canvas
                ref={canvasRef}
                style={{
                  width: '100%',
                  height: '100%',
                  display: 'block'
                }}
              />
              <Box
                sx={{
                  position: 'absolute',
                  bottom: 16,
                  right: 16,
                  bgcolor: 'white',
                  p: 1,
                  borderRadius: 1,
                  boxShadow: 2
                }}
              >
                <Typography variant="caption" color="text.secondary">
                  Left Click + Drag: Rotate | Right Click + Drag: Pan | Scroll: Zoom
                </Typography>
              </Box>
            </Box>

            {/* Model Information Panel */}
            {getSelectedModelInfo() && (
              <Box sx={{ p: 2, bgcolor: 'grey.50', borderTop: 1, borderColor: 'divider' }}>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <Typography variant="body2" fontWeight={600}>Model Information</Typography>
                    <Typography variant="caption" display="block" color="text.secondary">
                      <strong>Name:</strong> {getSelectedModelInfo()?.name}
                    </Typography>
                    <Typography variant="caption" display="block" color="text.secondary">
                      <strong>Category:</strong> {getSelectedModelInfo()?.category}
                    </Typography>
                    <Typography variant="caption" display="block" color="text.secondary">
                      <strong>Size:</strong> {getSelectedModelInfo()?.size}
                    </Typography>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Typography variant="body2" fontWeight={600}>Description</Typography>
                    <Typography variant="caption" color="text.secondary">
                      {getSelectedModelInfo()?.description}
                    </Typography>
                    <Box sx={{ mt: 1 }}>
                      <Chip label="IFC 4.3.x" size="small" sx={{ mr: 1 }} />
                      <Chip label="Sample Model" color="info" size="small" />
                    </Box>
                  </Grid>
                </Grid>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default HS2BIMTab;
