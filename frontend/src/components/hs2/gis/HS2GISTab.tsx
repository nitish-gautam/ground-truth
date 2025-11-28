/**
 * HS2 GIS Route Map Tab
 * Interactive map with HS2 route and construction data layers
 * NOW LOADING REAL SHAPEFILE DATA FROM BACKEND
 */

import React, { useState } from 'react';
import {
  Grid,
  Card,
  CardHeader,
  CardContent,
  Typography,
  Box,
  List,
  ListItemButton,
  Checkbox,
  Chip,
  Alert,
  Paper,
  CircularProgress
} from '@mui/material';
import { MapContainer, TileLayer, Polyline, CircleMarker, Popup, LayersControl, GeoJSON, useMap } from 'react-leaflet';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// HS2 Phase 1 Route (London to Birmingham) - Approximate coordinates
const hs2RouteCoordinates: [number, number][] = [
  [51.5308, -0.1238], // London Euston
  [51.5500, -0.2000], // Old Oak Common
  [51.6500, -0.4000], // Ruislip
  [51.7500, -0.5500], // Harefield
  [51.8000, -0.6500], // Chalfont St Peter
  [51.8500, -0.7500], // Amersham
  [51.9000, -0.8500], // Great Missenden
  [51.9500, -0.9500], // Wendover
  [52.0000, -1.0000], // Aylesbury
  [52.1000, -1.1000], // Calvert
  [52.2000, -1.2000], // Brackley
  [52.3000, -1.3000], // Kings Sutton
  [52.4000, -1.4000], // Chipping Warden
  [52.4800, -1.4800], // Birmingham Interchange (NEC)
  [52.5000, -1.5000], // Birmingham Curzon Street
];

// Sample construction sites
const constructionSites = [
  { lat: 51.5308, lng: -0.1238, name: 'Euston Station', status: 'In Progress', activity: 'Demolition & Site Prep' },
  { lat: 51.5500, lng: -0.2000, name: 'Old Oak Common', status: 'Advanced', activity: 'Platform Construction' },
  { lat: 52.4800, lng: -1.4800, name: 'Birmingham Interchange', status: 'Planning', activity: 'Ground Investigation' },
  { lat: 52.5000, lng: -1.5000, name: 'Curzon Street', status: 'In Progress', activity: 'Foundation Work' },
];

// Environmental monitoring points - Positioned along HS2 route for realism (synthetic data)
const environmentalPoints = [
  // London to Old Oak Common
  { lat: 51.5308, lng: -0.1238, name: 'Euston Air Quality', type: 'Air Quality', status: 'Monitoring' },
  { lat: 51.5400, lng: -0.1600, name: 'Camden Noise Monitor', type: 'Noise', status: 'Active' },
  { lat: 51.5500, lng: -0.2000, name: 'Old Oak Common Wildlife', type: 'Ecology', status: 'Protected' },

  // Ruislip to Chalfont
  { lat: 51.6000, lng: -0.3000, name: 'Northolt Biodiversity', type: 'Ecology', status: 'Survey Complete' },
  { lat: 51.6500, lng: -0.4000, name: 'Ruislip Woodland', type: 'Ecology', status: 'Protected' },
  { lat: 51.7000, lng: -0.5000, name: 'Harefield Water Quality', type: 'Water Quality', status: 'Monitoring' },
  { lat: 51.7500, lng: -0.5500, name: 'Denham Green Space', type: 'Landscape', status: 'Mitigation Active' },
  { lat: 51.8000, lng: -0.6500, name: 'Chalfont Viaduct Monitor', type: 'Vibration', status: 'Active' },

  // Chiltern Hills AONB
  { lat: 51.8500, lng: -0.7500, name: 'Chiltern Hills AONB', type: 'Landscape', status: 'Protected Area' },
  { lat: 51.8700, lng: -0.7800, name: 'Amersham Heritage Site', type: 'Cultural', status: 'Monitoring' },
  { lat: 51.9000, lng: -0.8500, name: 'Great Missenden Ecology', type: 'Ecology', status: 'Survey Active' },
  { lat: 51.9300, lng: -0.9000, name: 'Tunnel Portal Air Quality', type: 'Air Quality', status: 'Monitoring' },
  { lat: 51.9500, lng: -0.9500, name: 'Wendover Wildlife Corridor', type: 'Ecology', status: 'Mitigation Active' },

  // Central Section
  { lat: 52.0000, lng: -1.0000, name: 'Aylesbury Vale', type: 'Water Quality', status: 'Monitoring' },
  { lat: 52.0500, lng: -1.0500, name: 'Stoke Mandeville Wetlands', type: 'Ecology', status: 'Protected' },
  { lat: 52.1000, lng: -1.1000, name: 'Calvert Lake', type: 'Water Quality', status: 'Monitoring' },
  { lat: 52.1500, lng: -1.1500, name: 'Buckingham Floodplain', type: 'Hydrology', status: 'Survey Complete' },
  { lat: 52.2000, lng: -1.2000, name: 'Brackley Ancient Woodland', type: 'Ecology', status: 'Protected' },

  // Northamptonshire to Birmingham
  { lat: 52.2500, lng: -1.2500, name: 'Silverstone Noise Monitor', type: 'Noise', status: 'Active' },
  { lat: 52.3000, lng: -1.3000, name: 'Kings Sutton Heritage', type: 'Cultural', status: 'Monitoring' },
  { lat: 52.3500, lng: -1.3500, name: 'Chipping Warden Vibration', type: 'Vibration', status: 'Active' },
  { lat: 52.4000, lng: -1.4000, name: 'Warwickshire Farmland', type: 'Ecology', status: 'Survey Active' },
  { lat: 52.4400, lng: -1.4400, name: 'Birmingham Approach Air', type: 'Air Quality', status: 'Monitoring' },
  { lat: 52.4800, lng: -1.4800, name: 'NEC Interchange Noise', type: 'Noise', status: 'Active' },
  { lat: 52.4900, lng: -1.4900, name: 'Solihull Green Belt', type: 'Landscape', status: 'Monitoring' },
  { lat: 52.5000, lng: -1.5000, name: 'Curzon Street Urban Ecology', type: 'Ecology', status: 'Survey Complete' },
];

interface GISLayer {
  id: string;
  name: string;
  count: number;
  color: string;
  description: string;
  isRealData: boolean;
}

// Component to auto-fit map bounds to GeoJSON data
const FitBounds: React.FC<{ bounds: [number, number, number, number] | null }> = ({ bounds }) => {
  const map = useMap();

  React.useEffect(() => {
    if (bounds && bounds.length === 4) {
      // Convert [minLng, minLat, maxLng, maxLat] to Leaflet bounds format
      const leafletBounds = L.latLngBounds(
        [bounds[1], bounds[0]], // southwest corner
        [bounds[3], bounds[2]]  // northeast corner
      );
      console.log('🗺️ Fitting map to bounds:', leafletBounds);
      map.fitBounds(leafletBounds, { padding: [50, 50] });
    }
  }, [bounds, map]);

  return null;
};

const gisLayers: GISLayer[] = [
  { id: 'route', name: 'HS2 Route Line', count: 65, color: '#FF0000', description: '65 railway formation polygons', isRealData: true },
  { id: 'construction', name: 'Construction Compounds', count: 46, color: '#019C4B', description: '46 satellite construction sites', isRealData: true },
  { id: 'assets', name: 'Asset Locations', count: 500, color: '#2196F3', description: '500 assets color-coded by readiness (synthetic visualization)', isRealData: false },
  { id: 'environmental', name: 'Environmental Monitoring', count: 27, color: '#FF8500', description: 'Distributed along route (synthetic for demo)', isRealData: false },
  { id: 'landscape', name: 'Landscape Character Areas', count: 62, color: '#9C27B0', description: '62 landscape zones', isRealData: true },
  { id: 'ecology', name: 'Ecological Assets', count: 500, color: '#4CAF50', description: 'Ecological assets (limited to 500 for performance)', isRealData: true },
  { id: 'injunctions', name: 'Legal Injunctions', count: 6886, color: '#E91E63', description: 'Court-ordered restriction zones', isRealData: true },
  { id: 'property', name: 'Property Compensation', count: 1, color: '#FFC107', description: 'July 2014 compensation zones', isRealData: true },
];

export const HS2GISTab: React.FC = () => {
  const [selectedLayers, setSelectedLayers] = useState<string[]>(['route', 'construction']);
  const [loadedLayersData, setLoadedLayersData] = useState<Record<string, any>>({});

  // Fetch available GIS layers from backend
  const { data: gisLayersData, isLoading: layersLoading } = useQuery({
    queryKey: ['gis-layers'],
    queryFn: async () => {
      const response = await axios.get('/api/v1/gis/layers');
      return response.data;
    }
  });

  // Fetch GeoJSON for selected route layer
  const { data: routeGeoJSON, isLoading: routeLoading } = useQuery({
    queryKey: ['gis-route-geojson'],
    queryFn: async () => {
      console.log('🗺️ Fetching HS2 route shapefile GeoJSON...');
      const response = await axios.get('/api/v1/gis/layer/HS2_RE_RailAlignmentFormation_Ply_CT05_WDEIA', {
        params: { simplify: 0.0001 }
      });
      console.log('✅ Route GeoJSON loaded:', response.data.metadata?.feature_count, 'features');
      return response.data;
    },
    enabled: selectedLayers.includes('route')
  });

  // Fetch GeoJSON for construction compounds
  const { data: constructionGeoJSON } = useQuery({
    queryKey: ['gis-construction-geojson'],
    queryFn: async () => {
      console.log('🏗️ Fetching construction compounds GeoJSON...');
      const response = await axios.get('/api/v1/gis/layer/CON_CN_SatelliteConstructionCompounds_Ply_CT05_WDEIA', {
        params: { simplify: 0.0001 }
      });
      console.log('✅ Construction GeoJSON loaded:', response.data.metadata?.feature_count, 'features');
      return response.data;
    },
    enabled: selectedLayers.includes('construction')
  });

  // Fetch GeoJSON for landscape character areas
  const { data: landscapeGeoJSON } = useQuery({
    queryKey: ['gis-landscape-geojson'],
    queryFn: async () => {
      console.log('🌳 Fetching landscape character areas GeoJSON...');
      const response = await axios.get('/api/v1/gis/layer/ENV_ARP_C861_LV_LandscapeCharacterAreas_Ply_WDEIA', {
        params: { simplify: 0.0001 }
      });
      console.log('✅ Landscape GeoJSON loaded:', response.data.metadata?.feature_count, 'features');
      return response.data;
    },
    enabled: selectedLayers.includes('landscape')
  });

  // Fetch GeoJSON for ecology surveys (NEW)
  const { data: ecologyGeoJSON, error: ecologyError } = useQuery({
    queryKey: ['gis-ecology-geojson'],
    queryFn: async () => {
      console.log('🌿 Fetching ecology survey GeoJSON...');
      const response = await axios.get('/api/v1/gis/ecology', {
        params: { limit: 500 } // Limit to 500 features for performance
      });
      console.log('✅ Ecology GeoJSON loaded:', response.data.metadata?.feature_count, 'features');
      return response.data;
    },
    enabled: selectedLayers.includes('ecology'),
    retry: 1
  });

  // Fetch GeoJSON for injunctions (NEW)
  const { data: injectionsGeoJSON } = useQuery({
    queryKey: ['gis-injunctions-geojson'],
    queryFn: async () => {
      console.log('⚖️ Fetching injunctions GeoJSON...');
      const response = await axios.get('/api/v1/gis/injunctions', {
        params: { simplify: 0.0005, limit: 1000 } // Limit for performance (6,886 features total)
      });
      console.log('✅ Injunctions GeoJSON loaded:', response.data.metadata?.feature_count, 'features');
      return response.data;
    },
    enabled: selectedLayers.includes('injunctions')
  });

  // Fetch GeoJSON for property compensation (NEW)
  const { data: propertyGeoJSON } = useQuery({
    queryKey: ['gis-property-geojson'],
    queryFn: async () => {
      console.log('🏘️ Fetching property compensation GeoJSON...');
      const response = await axios.get('/api/v1/gis/property-compensation', {
        params: { simplify: 0.0001 }
      });
      console.log('✅ Property GeoJSON loaded:', response.data.metadata?.feature_count, 'features');
      return response.data;
    },
    enabled: selectedLayers.includes('property')
  });

  // Fetch GeoJSON for asset locations (NEW!)
  const { data: assetsGeoJSON } = useQuery({
    queryKey: ['gis-assets-geojson'],
    queryFn: async () => {
      console.log('🏗️ Fetching asset locations GeoJSON...');
      const response = await axios.get('/api/v1/gis/assets-locations');
      console.log('✅ Assets GeoJSON loaded:', response.data.metadata?.total_assets, 'assets');
      return response.data;
    },
    enabled: selectedLayers.includes('assets')
  });

  // Debug log
  React.useEffect(() => {
    if (routeGeoJSON) {
      console.log('📍 GeoJSON ready for rendering:', {
        featureCount: routeGeoJSON.metadata?.feature_count,
        geometryTypes: routeGeoJSON.metadata?.geometry_types,
        bounds: routeGeoJSON.metadata?.bounds
      });
    }
  }, [routeGeoJSON]);

  const handleToggleLayer = (layerId: string) => {
    setSelectedLayers(prev =>
      prev.includes(layerId)
        ? prev.filter(id => id !== layerId)
        : [...prev, layerId]
    );
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'In Progress': return '#019C4B';
      case 'Advanced': return '#FF8500';
      case 'Planning': return '#012A39';
      default: return '#666';
    }
  };

  return (
    <Box>
      <Alert severity={routeGeoJSON ? "success" : "info"} sx={{ mb: 3 }}>
        <Typography variant="body2">
          <strong>Interactive HS2 Route Map</strong> - Phase 2a + Ecology & Legal Data
          <br />
          {routeGeoJSON ? (
            <>
              ✅ Displaying <strong>real shapefile data</strong>: {routeGeoJSON.metadata?.feature_count || 0} route polygons
              {constructionGeoJSON && `, ${constructionGeoJSON.metadata?.feature_count || 0} construction compounds`}
              {landscapeGeoJSON && `, ${landscapeGeoJSON.metadata?.feature_count || 0} landscape areas`}
              {ecologyGeoJSON && `, ${ecologyGeoJSON.metadata?.feature_count || 0} ecology features`}
              {injectionsGeoJSON && `, ${injectionsGeoJSON.metadata?.feature_count || 0} injunction zones`}
              {propertyGeoJSON && `, ${propertyGeoJSON.metadata?.feature_count || 0} property compensation zones`}
              {assetsGeoJSON && `, ${assetsGeoJSON.metadata?.total_assets || 0} asset locations`}
              {layersLoading && <CircularProgress size={16} sx={{ ml: 1 }} />}
            </>
          ) : (
            <>Loading {gisLayersData?.total_count || 420} shapefiles from HS2 datasets...</>
          )}
        </Typography>
      </Alert>

      <Box sx={{ display: 'flex', gap: 3, width: '100vw', position: 'relative', left: '50%', right: '50%', marginLeft: '-50vw', marginRight: '-50vw' }}>
        {/* Layer Selector */}
        <Box sx={{ width: 320, flexShrink: 0, ml: 3 }}>
          <Card elevation={3}>
            <CardHeader
              title="Map Layers"
              sx={{
                bgcolor: 'primary.main',
                '& .MuiCardHeader-title': {
                  color: 'primary.contrastText',
                  fontWeight: 600
                }
              }}
            />
            <CardContent sx={{ p: 0 }}>
              <List>
                {gisLayers.map(layer => (
                  <ListItemButton
                    key={layer.id}
                    onClick={() => handleToggleLayer(layer.id)}
                    sx={{
                      borderBottom: 1,
                      borderColor: 'divider',
                      borderLeft: layer.isRealData ? 4 : undefined,
                      borderLeftColor: layer.isRealData ? 'success.main' : undefined,
                      '&:hover': { bgcolor: 'action.hover' }
                    }}
                  >
                    <Checkbox
                      edge="start"
                      checked={selectedLayers.includes(layer.id)}
                      tabIndex={-1}
                      disableRipple
                    />
                    <Box sx={{ flex: 1, ml: 1 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5, gap: 1 }}>
                        <Box
                          sx={{
                            width: 16,
                            height: 16,
                            bgcolor: layer.color,
                            mr: 1,
                            borderRadius: '50%'
                          }}
                        />
                        <Typography variant="body2" fontWeight={600}>
                          {layer.name}
                        </Typography>
                        {layer.isRealData ? (
                          <Chip
                            label="REAL DATA"
                            color="success"
                            size="small"
                            sx={{ fontSize: '0.6rem', height: 18, fontWeight: 700 }}
                          />
                        ) : (
                          <Chip
                            label="SYNTHETIC"
                            color="warning"
                            size="small"
                            sx={{ fontSize: '0.6rem', height: 18 }}
                          />
                        )}
                      </Box>
                      <Typography variant="caption" color="text.secondary">
                        {layer.description}
                      </Typography>
                      <Box sx={{ mt: 0.5 }}>
                        <Chip
                          label={`${layer.count} items`}
                          size="small"
                          sx={{ fontSize: '0.7rem', height: 20 }}
                        />
                      </Box>
                    </Box>
                  </ListItemButton>
                ))}
              </List>

              <Box sx={{ p: 2, bgcolor: 'grey.100' }}>
                <Typography variant="caption" color="text.secondary">
                  <strong>Data Source:</strong> HS2 Ltd Open Data
                  <br />
                  <strong>Total Shapefiles:</strong> {gisLayersData?.total_count || '...'} files loaded
                  {layersLoading && ' (loading...)'}
                  <br />
                  <strong>Coverage:</strong> Phase 2a Route (Midlands)
                  <br />
                  {gisLayersData && (
                    <Chip
                      label={`${gisLayersData.layers.filter((l: any) => l.status === 'loaded').length} layers ready`}
                      size="small"
                      color="success"
                      sx={{ mt: 1, fontSize: '0.65rem', height: 18 }}
                    />
                  )}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Box>

        {/* Interactive Map */}
        <Box sx={{ flex: 1, mr: 3 }}>
          <Paper elevation={3} sx={{ height: 1000, overflow: 'hidden', borderRadius: 2 }}>
            <MapContainer
              center={[52.9, -2.12]}
              zoom={9}
              style={{ height: '100%', width: '100%' }}
            >
              {/* Base Map Layers Control - Street vs Satellite */}
              <LayersControl position="topright">
                <LayersControl.BaseLayer checked name="Street Map">
                  <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  />
                </LayersControl.BaseLayer>

                <LayersControl.BaseLayer name="Satellite View">
                  <TileLayer
                    attribution='Imagery &copy; <a href="https://www.esri.com/">Esri</a>'
                    url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                    maxZoom={19}
                  />
                </LayersControl.BaseLayer>
              </LayersControl>

              {/* Auto-fit map to shapefile bounds */}
              {routeGeoJSON?.metadata?.bounds && (
                <FitBounds bounds={routeGeoJSON.metadata.bounds as [number, number, number, number]} />
              )}

              {/* HS2 Route Line - REAL DATA from shapefile */}
              {selectedLayers.includes('route') && routeGeoJSON?.features && (
                <GeoJSON
                  key={`route-geojson-${routeGeoJSON.metadata?.feature_count}`}
                  data={{
                    type: "FeatureCollection",
                    features: routeGeoJSON.features
                  }}
                  style={{
                    color: '#FF0000',        // Bright red outline for visibility
                    weight: 4,                // Thicker line
                    opacity: 1,               // Fully opaque
                    fillColor: '#FF8500',     // Orange fill
                    fillOpacity: 0.5          // More visible fill
                  }}
                  onEachFeature={(feature, layer) => {
                    const props = feature.properties || {};
                    layer.bindPopup(`
                      <strong>HS2 Railway Formation</strong><br/>
                      Model: ${props.Model_Num || 'N/A'}<br/>
                      Area: ${props.Shape_Area ? props.Shape_Area.toFixed(0) : 'N/A'} m²<br/>
                      <em>Click polygon to see details</em>
                    `);
                  }}
                />
              )}

              {/* Construction Compounds - REAL DATA from shapefile */}
              {selectedLayers.includes('construction') && constructionGeoJSON?.features && (
                <GeoJSON
                  key={`construction-geojson-${constructionGeoJSON.metadata?.feature_count}`}
                  data={{
                    type: "FeatureCollection",
                    features: constructionGeoJSON.features
                  }}
                  style={{
                    color: '#019C4B',        // Green outline
                    weight: 3,
                    opacity: 1,
                    fillColor: '#4CAF50',    // Light green fill
                    fillOpacity: 0.4
                  }}
                  onEachFeature={(feature, layer) => {
                    const props = feature.properties || {};
                    layer.bindPopup(`
                      <strong>Construction Compound</strong><br/>
                      Model: ${props.Model_Num || 'N/A'}<br/>
                      Area: ${props.Shape_Area ? props.Shape_Area.toFixed(0) : 'N/A'} m²
                    `);
                  }}
                />
              )}

              {/* Landscape Character Areas - REAL DATA from shapefile */}
              {selectedLayers.includes('landscape') && landscapeGeoJSON?.features && (
                <GeoJSON
                  key={`landscape-geojson-${landscapeGeoJSON.metadata?.feature_count}`}
                  data={{
                    type: "FeatureCollection",
                    features: landscapeGeoJSON.features
                  }}
                  style={{
                    color: '#9C27B0',        // Purple outline
                    weight: 2,
                    opacity: 0.8,
                    fillColor: '#BA68C8',    // Light purple fill
                    fillOpacity: 0.2
                  }}
                  onEachFeature={(feature, layer) => {
                    const props = feature.properties || {};
                    layer.bindPopup(`
                      <strong>Landscape Character Area</strong><br/>
                      Area: ${props.Area_Name || 'N/A'}<br/>
                      Character Type: ${props.Char_Type || 'N/A'}<br/>
                      Significance: ${props.SIG_LCA || 'N/A'}
                    `);
                  }}
                />
              )}

              {/* Ecology Surveys - NEW REAL DATA from organized folder */}
              {selectedLayers.includes('ecology') && ecologyGeoJSON?.features && (
                <GeoJSON
                  key={`ecology-geojson-${ecologyGeoJSON.metadata?.feature_count}`}
                  data={{
                    type: "FeatureCollection",
                    features: ecologyGeoJSON.features
                  }}
                  style={{
                    color: '#4CAF50',        // Green outline
                    weight: 2,
                    opacity: 0.8,
                    fillColor: '#81C784',    // Light green fill
                    fillOpacity: 0.3
                  }}
                  onEachFeature={(feature, layer) => {
                    const props = feature.properties || {};
                    layer.bindPopup(`
                      <strong>Ecology Survey Feature</strong><br/>
                      Type: ${props.ecology_type || 'N/A'}<br/>
                      Survey Date: November 2024<br/>
                      <em>Click for ecological assessment details</em>
                    `);
                  }}
                />
              )}

              {/* Legal Injunctions - NEW REAL DATA from organized folder */}
              {selectedLayers.includes('injunctions') && injectionsGeoJSON?.features && (
                <GeoJSON
                  key={`injunctions-geojson-${injectionsGeoJSON.metadata?.feature_count}`}
                  data={{
                    type: "FeatureCollection",
                    features: injectionsGeoJSON.features
                  }}
                  style={{
                    color: '#E91E63',        // Pink/Red outline
                    weight: 2,
                    opacity: 0.8,
                    fillColor: '#F48FB1',    // Light pink fill
                    fillOpacity: 0.2
                  }}
                  onEachFeature={(feature, layer) => {
                    const props = feature.properties || {};
                    layer.bindPopup(`
                      <strong>Legal Injunction Zone</strong><br/>
                      Court-ordered restriction<br/>
                      <em>Access and activities restricted</em>
                    `);
                  }}
                />
              )}

              {/* Property Compensation - NEW REAL DATA from organized folder */}
              {selectedLayers.includes('property') && propertyGeoJSON?.features && (
                <GeoJSON
                  key={`property-geojson-${propertyGeoJSON.metadata?.feature_count}`}
                  data={{
                    type: "FeatureCollection",
                    features: propertyGeoJSON.features
                  }}
                  style={{
                    color: '#FFC107',        // Orange/Yellow outline
                    weight: 2,
                    opacity: 0.8,
                    fillColor: '#FFD54F',    // Light yellow fill
                    fillOpacity: 0.2
                  }}
                  onEachFeature={(feature, layer) => {
                    const props = feature.properties || {};
                    layer.bindPopup(`
                      <strong>Property Compensation Zone</strong><br/>
                      Consultation Date: July 2014<br/>
                      Zone Type: ${props.ZONE_TYPE || 'N/A'}<br/>
                      <em>Property affected by HS2 Phase Two</em>
                    `);
                  }}
                />
              )}

              {/* Mock Construction Sites (kept as sample points) */}
              {selectedLayers.includes('construction') && !constructionGeoJSON && constructionSites.map((site, idx) => (
                <CircleMarker
                  key={`const-${idx}`}
                  center={[site.lat, site.lng]}
                  radius={10}
                  pathOptions={{
                    fillColor: getStatusColor(site.status),
                    color: '#fff',
                    weight: 2,
                    opacity: 1,
                    fillOpacity: 0.8
                  }}
                >
                  <Popup>
                    <Box sx={{ minWidth: 200 }}>
                      <Typography variant="body2" fontWeight={600}>{site.name}</Typography>
                      <Typography variant="caption" display="block">
                        <strong>Status:</strong> {site.status}
                      </Typography>
                      <Typography variant="caption" display="block">
                        <strong>Activity:</strong> {site.activity}
                      </Typography>
                      <Chip
                        label={site.status}
                        size="small"
                        sx={{
                          mt: 1,
                          bgcolor: getStatusColor(site.status),
                          color: 'white'
                        }}
                      />
                    </Box>
                  </Popup>
                </CircleMarker>
              ))}

              {/* Environmental Monitoring */}
              {selectedLayers.includes('environmental') && environmentalPoints.map((point, idx) => (
                <CircleMarker
                  key={`env-${idx}`}
                  center={[point.lat, point.lng]}
                  radius={8}
                  pathOptions={{
                    fillColor: '#FF8500',
                    color: '#fff',
                    weight: 2,
                    opacity: 1,
                    fillOpacity: 0.7
                  }}
                >
                  <Popup>
                    <Box sx={{ minWidth: 180 }}>
                      <Typography variant="body2" fontWeight={600}>{point.name}</Typography>
                      <Typography variant="caption" display="block">
                        <strong>Type:</strong> {point.type}
                      </Typography>
                      <Typography variant="caption" display="block">
                        <strong>Status:</strong> {point.status}
                      </Typography>
                    </Box>
                  </Popup>
                </CircleMarker>
              ))}

              {/* Asset Locations (NEW!) */}
              {selectedLayers.includes('assets') && assetsGeoJSON && assetsGeoJSON.features && (
                <GeoJSON
                  key="assets-layer"
                  data={assetsGeoJSON}
                  pointToLayer={(feature, latlng) => {
                    const color = feature.properties.color || '#2196F3';
                    return L.circleMarker(latlng, {
                      radius: 6,
                      fillColor: color,
                      color: '#fff',
                      weight: 1,
                      opacity: 1,
                      fillOpacity: 0.8
                    });
                  }}
                  onEachFeature={(feature, layer) => {
                    if (feature.properties) {
                      const props = feature.properties;
                      layer.bindPopup(`
                        <div style="min-width: 200px;">
                          <strong style="font-size: 14px;">${props.asset_name}</strong><br/>
                          <span style="background-color: ${props.color}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 11px; font-weight: 600;">${props.readiness_status}</span><br/><br/>
                          <strong>ID:</strong> ${props.asset_id}<br/>
                          <strong>Type:</strong> ${props.asset_type}<br/>
                          <strong>Contractor:</strong> ${props.contractor}<br/>
                          <strong>TAEM Score:</strong> ${props.taem_score}<br/>
                          <strong>Route:</strong> ${props.route_section}
                        </div>
                      `);
                    }
                  }}
                />
              )}
            </MapContainer>
          </Paper>

          {/* Map Legend - Dynamic based on selected layers */}
          <Paper elevation={2} sx={{ mt: 2, p: 2 }}>
            <Typography variant="body2" fontWeight={600} gutterBottom>
              Map Legend ({selectedLayers.length} layer{selectedLayers.length !== 1 ? 's' : ''} visible)
            </Typography>
            <Grid container spacing={2}>
              {selectedLayers.length === 0 && (
                <Grid item xs={12}>
                  <Typography variant="caption" color="text.secondary" fontStyle="italic">
                    No layers selected. Choose layers from the panel on the left.
                  </Typography>
                </Grid>
              )}
              {gisLayers
                .filter(layer => selectedLayers.includes(layer.id))
                .map(layer => (
                  <Grid item xs={6} md={3} key={layer.id}>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Box
                        sx={{
                          width: layer.id === 'route' ? 20 : 12,
                          height: layer.id === 'route' ? 4 : 12,
                          bgcolor: layer.color,
                          borderRadius: layer.id === 'route' ? 0 : '50%',
                          mr: 1
                        }}
                      />
                      <Typography variant="caption">
                        {layer.name} ({layer.count})
                      </Typography>
                    </Box>
                  </Grid>
                ))}
            </Grid>
          </Paper>
        </Box>
      </Box>
    </Box>
  );
};

export default HS2GISTab;
