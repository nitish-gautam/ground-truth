/**
 * GraphVisualization Component
 * ============================
 *
 * Interactive force-directed graph visualization for HS2 asset relationships
 * using D3.js. Shows dependencies, blockers, and impact analysis.
 *
 * Features:
 * - Force-directed graph layout
 * - Interactive node dragging
 * - Relationship filtering
 * - Color-coded by node type
 * - HS2 design compliance
 */

import React, { useEffect, useRef, useState } from 'react';
import {
  Box,
  Card,
  CardHeader,
  CardContent,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Typography,
  Chip,
  Stack,
  Alert,
  CircularProgress,
  FormGroup,
  FormControlLabel,
  Checkbox,
  Paper
} from '@mui/material';
import * as d3 from 'd3';
import axios from 'axios';

// ==================== Types ====================

interface GraphNode {
  id: string;
  label: string;
  type: string;
  status?: string;
  taem_score?: number;
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
}

interface GraphLink {
  source: string | GraphNode;
  target: string | GraphNode;
  type: string;
  label: string;
}

interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
  center_node_id: string;
}

interface BlockerInfo {
  id: string;
  type: string;
  name?: string;
  status?: string;
  reason: string;
}

interface ExplainabilityData {
  asset_id: string;
  asset_name: string;
  status: string;
  taem_score: number;
  blockers: BlockerInfo[];
  blocker_count: number;
  ready: boolean;
}

// ==================== Node Colors (HS2 Theme) ====================

const NODE_COLORS = {
  asset: '#012A39',          // Primary dark blue
  deliverable: '#009C4A',    // Success green
  certificate: '#0066B3',    // Info blue
  payment: '#FF8500',        // Warning orange
  blocker: '#E31E24',        // Error red
  default: '#6B7280'
};

const STATUS_COLORS = {
  Ready: '#009C4A',
  'In Progress': '#FF8500',
  'Not Ready': '#E31E24',
  Blocked: '#E31E24',
  Approved: '#009C4A',
  Pending: '#FF8500',
  Overdue: '#E31E24',
  default: '#6B7280'
};

// ==================== Component ====================

export const GraphVisualization: React.FC = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const simulationRef = useRef<any>(null);
  const [assetId, setAssetId] = useState<string>('');
  const [depth, setDepth] = useState<number>(2);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [explainability, setExplainability] = useState<ExplainabilityData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [availableAssets, setAvailableAssets] = useState<string[]>([]);

  // Relationship type filters
  const [showDependencies, setShowDependencies] = useState(true);
  const [showBlockers, setShowBlockers] = useState(true);
  const [showDeliverables, setShowDeliverables] = useState(true);

  // Fetch available assets on mount
  useEffect(() => {
    const fetchAssets = async () => {
      try {
        const response = await axios.get('http://localhost:8002/api/v1/hs2/assets');
        // Handle paginated response structure: { total, skip, limit, items }
        const items = response.data.items || response.data;
        const assets = (Array.isArray(items) ? items : []).slice(0, 50).map((a: any) => a.asset_id);
        setAvailableAssets(assets);
        if (assets.length > 0) {
          setAssetId(assets[0]); // Default to first asset
        }
      } catch (err) {
        console.error('Failed to fetch assets:', err);
      }
    };
    fetchAssets();
  }, []);

  // Fetch graph data when asset changes
  useEffect(() => {
    if (!assetId) return;

    const fetchGraphData = async () => {
      setLoading(true);
      setError(null);

      try {
        // Fetch visualization data
        const vizResponse = await axios.get(
          `http://localhost:8002/api/v1/graph/visualization/${assetId}?depth=${depth}`
        );
        setGraphData(vizResponse.data);

        // Fetch explainability data
        const explainResponse = await axios.get(
          `http://localhost:8002/api/v1/graph/explainability/${assetId}`
        );
        setExplainability(explainResponse.data);

      } catch (err: any) {
        setError(err.response?.data?.detail || 'Failed to load graph data');
        console.error('Graph fetch error:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchGraphData();
  }, [assetId, depth]);

  // Render D3 graph
  useEffect(() => {
    if (!graphData) {
      console.log('⚠️ No graph data');
      return;
    }

    // Retry logic to wait for SVG element to mount
    const attemptRender = (attempt = 0) => {
      if (!svgRef.current) {
        if (attempt < 5) {
          console.log(`⏳ Waiting for SVG to mount (attempt ${attempt + 1}/5)...`);
          setTimeout(() => attemptRender(attempt + 1), 100);
        } else {
          console.error('❌ SVG element never mounted after 5 attempts');
        }
        return;
      }

      console.log('🎨 Rendering graph:', {
        nodes: graphData.nodes.length,
        links: graphData.links.length,
        centerNode: graphData.center_node_id
      });

      renderGraph();
    };

    attemptRender();

    // Cleanup
    return () => {
      if (simulationRef.current) {
        simulationRef.current.stop();
      }
    };
  }, [graphData, showDependencies, showBlockers, showDeliverables]);

  const renderGraph = () => {
    if (!graphData || !svgRef.current) return;

    // Filter links based on relationship toggles
    const filteredLinks = graphData.links.filter(link => {
      if (!showDependencies && link.type === 'DEPENDS_ON') return false;
      if (!showBlockers && link.type === 'BLOCKS') return false;
      if (!showDeliverables && link.type === 'HAS_DELIVERABLE') return false;
      return true;
    });

    // Get nodes that are connected by filtered links
    const connectedNodeIds = new Set<string>();
    filteredLinks.forEach(link => {
      const source = typeof link.source === 'string' ? link.source : link.source.id;
      const target = typeof link.target === 'string' ? link.target : link.target.id;
      connectedNodeIds.add(source);
      connectedNodeIds.add(target);
    });

    // Always include center node
    connectedNodeIds.add(graphData.center_node_id);

    const filteredNodes = graphData.nodes.filter(node => connectedNodeIds.has(node.id));

    // Clear previous graph
    d3.select(svgRef.current).selectAll('*').remove();

    const width = 1200;
    const height = 800;

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', `0 0 ${width} ${height}`)
      .style('background', 'linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)');

    // Create arrow markers for directed edges
    svg.append('defs').selectAll('marker')
      .data(['DEPENDS_ON', 'BLOCKS', 'HAS_DELIVERABLE'])
      .enter().append('marker')
      .attr('id', d => `arrow-${d}`)
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 20)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', d => d === 'BLOCKS' ? '#FF0000' : '#666');

    // Stop previous simulation if exists
    if (simulationRef.current) {
      simulationRef.current.stop();
    }

    // Create force simulation with better spacing
    const simulation = d3.forceSimulation<GraphNode>(filteredNodes)
      .force('link', d3.forceLink<GraphNode, GraphLink>(filteredLinks)
        .id(d => d.id)
        .distance(200)
        .strength(0.5))
      .force('charge', d3.forceManyBody().strength(-800))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(60))
      .force('x', d3.forceX(width / 2).strength(0.05))
      .force('y', d3.forceY(height / 2).strength(0.05));

    simulationRef.current = simulation;

    // Create link elements with better styling
    const link = svg.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(filteredLinks)
      .enter().append('line')
      .attr('stroke', d => {
        if (d.type === 'BLOCKS') return '#E31E24';
        if (d.type === 'HAS_DELIVERABLE') return '#009C4A';
        return '#64748B';
      })
      .attr('stroke-width', d => d.type === 'BLOCKS' ? 3 : 2)
      .attr('stroke-opacity', 0.4)
      .attr('marker-end', d => `url(#arrow-${d.type})`)
      .style('transition', 'all 0.3s ease');

    // Create link labels with background for better readability
    const linkLabelGroup = svg.append('g')
      .attr('class', 'link-labels')
      .selectAll('g')
      .data(filteredLinks)
      .enter().append('g');

    linkLabelGroup.append('rect')
      .attr('fill', 'white')
      .attr('opacity', 0.9)
      .attr('rx', 3)
      .attr('ry', 3);

    const linkLabel = linkLabelGroup.append('text')
      .attr('font-size', 10)
      .attr('font-weight', 500)
      .attr('fill', '#475569')
      .attr('text-anchor', 'middle')
      .text(d => d.type.replace(/_/g, ' '));

    // Position and size the background rectangles
    linkLabel.each(function(d) {
      const bbox = (this as SVGTextElement).getBBox();
      d3.select((this as any).parentNode).select('rect')
        .attr('x', bbox.x - 4)
        .attr('y', bbox.y - 2)
        .attr('width', bbox.width + 8)
        .attr('height', bbox.height + 4);
    });

    // Create node groups with hover effects
    const node = svg.append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(filteredNodes)
      .enter().append('g')
      .attr('class', 'node')
      .style('cursor', 'grab')
      .call(d3.drag<SVGGElement, GraphNode>()
        .on('start', dragStarted)
        .on('drag', dragged)
        .on('end', dragEnded));

    // Add shadow/glow effect for center node
    node.filter(d => d.id === graphData.center_node_id)
      .append('circle')
      .attr('r', 35)
      .attr('fill', 'none')
      .attr('stroke', '#012A39')
      .attr('stroke-width', 2)
      .attr('stroke-opacity', 0.2)
      .attr('class', 'node-glow');

    // Add circles to nodes with gradient and better styling
    node.append('circle')
      .attr('r', d => d.id === graphData.center_node_id ? 30 : 18)
      .attr('fill', d => {
        if (d.status) {
          return STATUS_COLORS[d.status as keyof typeof STATUS_COLORS] || STATUS_COLORS.default;
        }
        return NODE_COLORS[d.type as keyof typeof NODE_COLORS] || NODE_COLORS.default;
      })
      .attr('stroke', '#fff')
      .attr('stroke-width', d => d.id === graphData.center_node_id ? 4 : 3)
      .attr('filter', 'drop-shadow(0px 4px 6px rgba(0, 0, 0, 0.15))')
      .style('transition', 'all 0.3s ease');

    // Add hover effects
    node.on('mouseenter', function(event, d) {
      d3.select(this).select('circle')
        .transition()
        .duration(200)
        .attr('r', d.id === graphData.center_node_id ? 35 : 22)
        .attr('filter', 'drop-shadow(0px 6px 12px rgba(0, 0, 0, 0.25))');
    })
    .on('mouseleave', function(event, d) {
      d3.select(this).select('circle')
        .transition()
        .duration(200)
        .attr('r', d.id === graphData.center_node_id ? 30 : 18)
        .attr('filter', 'drop-shadow(0px 4px 6px rgba(0, 0, 0, 0.15))');
    });

    // Add node labels with better styling
    node.append('text')
      .attr('dy', d => d.id === graphData.center_node_id ? -42 : -28)
      .attr('text-anchor', 'middle')
      .attr('font-size', d => d.id === graphData.center_node_id ? 16 : 12)
      .attr('font-weight', d => d.id === graphData.center_node_id ? 700 : 600)
      .attr('fill', '#1e293b')
      .attr('filter', 'drop-shadow(0px 2px 4px rgba(255, 255, 255, 0.9))')
      .text(d => {
        // Truncate long labels
        const maxLength = d.id === graphData.center_node_id ? 40 : 25;
        return d.label.length > maxLength ? d.label.substring(0, maxLength) + '...' : d.label;
      });

    // Add status badges for deliverables
    node.filter(d => d.status && d.type === 'deliverable')
      .append('text')
      .attr('dy', 28)
      .attr('text-anchor', 'middle')
      .attr('font-size', 9)
      .attr('font-weight', 600)
      .attr('fill', d => STATUS_COLORS[d.status as keyof typeof STATUS_COLORS] || STATUS_COLORS.default)
      .text(d => d.status);

    // Add TAEM score badges for assets
    node.filter(d => d.taem_score !== undefined)
      .append('text')
      .attr('dy', 40)
      .attr('text-anchor', 'middle')
      .attr('font-size', 11)
      .attr('font-weight', 700)
      .attr('fill', d => {
        const score = d.taem_score || 0;
        if (score >= 90) return '#009C4A';
        if (score >= 70) return '#FF8500';
        return '#E31E24';
      })
      .text(d => `${d.taem_score?.toFixed(1)}%`);

    // Update positions on simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => (d.source as GraphNode).x!)
        .attr('y1', d => (d.source as GraphNode).y!)
        .attr('x2', d => (d.target as GraphNode).x!)
        .attr('y2', d => (d.target as GraphNode).y!);

      linkLabelGroup.attr('transform', d => {
        const x = ((d.source as GraphNode).x! + (d.target as GraphNode).x!) / 2;
        const y = ((d.source as GraphNode).y! + (d.target as GraphNode).y!) / 2;
        return `translate(${x},${y})`;
      });

      node.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    // Drag functions with cursor feedback
    function dragStarted(event: any, d: GraphNode) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
      d3.select(event.sourceEvent.target.parentNode).style('cursor', 'grabbing');
    }

    function dragged(event: any, d: GraphNode) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragEnded(event: any, d: GraphNode) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
      d3.select(event.sourceEvent.target.parentNode).style('cursor', 'grab');
    }

    console.log('✅ Graph rendered successfully:', {
      filteredNodes: filteredNodes.length,
      filteredLinks: filteredLinks.length,
      svgWidth: width,
      svgHeight: height
    });
  };

  return (
    <Box>
      {/* Graph Visualization Card */}
      <Card sx={{ mb: 3 }}>
        <CardHeader
          title="Asset Relationship Graph"
          sx={{
            bgcolor: 'primary.main',
            '& .MuiCardHeader-title': {
              color: 'primary.contrastText',
              fontWeight: 600
            }
          }}
        />
        <CardContent>
          {/* Controls */}
          <Box sx={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: 2,
            alignItems: 'center',
            p: 2.5,
            mb: 2,
            bgcolor: '#f8fafc',
            borderRadius: 2,
            border: '1px solid',
            borderColor: 'divider'
          }}>
            <FormControl sx={{ minWidth: 220, bgcolor: 'white', borderRadius: 1 }}>
              <InputLabel size="small">Select Asset</InputLabel>
              <Select
                value={assetId}
                onChange={(e) => setAssetId(e.target.value)}
                label="Select Asset"
                size="small"
              >
                {availableAssets.map((id) => (
                  <MenuItem key={id} value={id}>{id}</MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControl sx={{ minWidth: 160, bgcolor: 'white', borderRadius: 1 }}>
              <InputLabel size="small">Graph Depth</InputLabel>
              <Select
                value={depth}
                onChange={(e) => setDepth(e.target.value as number)}
                label="Graph Depth"
                size="small"
              >
                <MenuItem value={1}>1 Level</MenuItem>
                <MenuItem value={2}>2 Levels</MenuItem>
                <MenuItem value={3}>3 Levels</MenuItem>
                <MenuItem value={4}>4 Levels</MenuItem>
                <MenuItem value={5}>5 Levels</MenuItem>
              </Select>
            </FormControl>

            <Box sx={{ flex: 1 }} />

            <FormGroup row sx={{ gap: 1 }}>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={showDependencies}
                    onChange={(e) => setShowDependencies(e.target.checked)}
                    size="small"
                  />
                }
                label={<Typography variant="body2" fontWeight={500}>Dependencies</Typography>}
              />
              <FormControlLabel
                control={
                  <Checkbox
                    checked={showBlockers}
                    onChange={(e) => setShowBlockers(e.target.checked)}
                    size="small"
                  />
                }
                label={<Typography variant="body2" fontWeight={500}>Blockers</Typography>}
              />
              <FormControlLabel
                control={
                  <Checkbox
                    checked={showDeliverables}
                    onChange={(e) => setShowDeliverables(e.target.checked)}
                    size="small"
                  />
                }
                label={<Typography variant="body2" fontWeight={500}>Deliverables</Typography>}
              />
            </FormGroup>

            {graphData && (
              <Chip
                label={`${graphData.nodes.length} nodes, ${graphData.links.length} links`}
                size="small"
                sx={{
                  bgcolor: 'primary.main',
                  color: 'white',
                  fontWeight: 600
                }}
              />
            )}
          </Box>

          {/* Error Alert */}
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {/* Loading State */}
          {loading && (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
              <CircularProgress />
            </Box>
          )}

          {/* Graph SVG */}
          {!loading && graphData && (
            <Paper
              elevation={3}
              sx={{
                p: 0,
                bgcolor: 'transparent',
                overflow: 'hidden',
                borderRadius: 2,
                border: '1px solid',
                borderColor: 'divider'
              }}
            >
              <Box sx={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                bgcolor: '#f8fafc'
              }}>
                <svg
                  ref={svgRef}
                  style={{
                    width: '100%',
                    maxWidth: '1200px',
                    height: '800px',
                    display: 'block'
                  }}
                ></svg>
              </Box>
            </Paper>
          )}

          {/* Legend */}
          {!loading && graphData && (
            <Paper
              elevation={2}
              sx={{
                mt: 3,
                p: 2,
                bgcolor: '#f8fafc',
                border: '1px solid',
                borderColor: 'divider'
              }}
            >
              <Typography variant="subtitle2" fontWeight={600} color="text.secondary" sx={{ mb: 1.5 }}>
                Legend
              </Typography>
              <Stack direction="row" spacing={2} flexWrap="wrap" gap={1}>
                <Chip
                  icon={<Box sx={{ width: 12, height: 12, borderRadius: '50%', bgcolor: NODE_COLORS.asset, ml: 1 }} />}
                  label="Asset"
                  size="medium"
                  variant="outlined"
                  sx={{ fontWeight: 500 }}
                />
                <Chip
                  icon={<Box sx={{ width: 12, height: 12, borderRadius: '50%', bgcolor: NODE_COLORS.deliverable, ml: 1 }} />}
                  label="Deliverable"
                  size="medium"
                  variant="outlined"
                  sx={{ fontWeight: 500 }}
                />
                <Chip
                  icon={<Box sx={{ width: 12, height: 12, borderRadius: '50%', bgcolor: NODE_COLORS.certificate, ml: 1 }} />}
                  label="Certificate"
                  size="medium"
                  variant="outlined"
                  sx={{ fontWeight: 500 }}
                />
                <Chip
                  icon={<Box sx={{ width: 12, height: 12, borderRadius: '50%', bgcolor: NODE_COLORS.blocker, ml: 1 }} />}
                  label="Blocker"
                  size="medium"
                  variant="outlined"
                  sx={{ fontWeight: 500 }}
                />
                <Box sx={{ width: '100%', height: 1, bgcolor: 'divider', my: 1 }} />
                <Typography variant="caption" color="text.secondary" sx={{ fontStyle: 'italic' }}>
                  💡 Tip: Drag nodes to rearrange • Hover to highlight • Uncheck filters to hide relationships
                </Typography>
              </Stack>
            </Paper>
          )}
        </CardContent>
      </Card>

      {/* Explainability Panel */}
      {explainability && (
        <Card>
          <CardHeader
            title={`Why Not Ready? - ${explainability.asset_name}`}
            sx={{
              bgcolor: explainability.ready ? 'success.main' : 'error.main',
              '& .MuiCardHeader-title': {
                color: 'white',
                fontWeight: 600
              }
            }}
          />
          <CardContent>
            <Stack spacing={2}>
              {/* Asset Status */}
              <Box>
                <Typography variant="body2" color="text.secondary">
                  Status
                </Typography>
                <Chip
                  label={explainability.status}
                  size="small"
                  sx={{
                    bgcolor: STATUS_COLORS[explainability.status as keyof typeof STATUS_COLORS] || STATUS_COLORS.default,
                    color: 'white'
                  }}
                />
              </Box>

              {/* TAEM Score */}
              <Box>
                <Typography variant="body2" color="text.secondary">
                  TAEM Score
                </Typography>
                <Typography variant="h5" color={explainability.taem_score > 80 ? 'success.main' : 'error.main'}>
                  {explainability.taem_score.toFixed(1)}%
                </Typography>
              </Box>

              {/* Blockers */}
              <Box>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  Blockers ({explainability.blocker_count})
                </Typography>
                {explainability.blockers.length === 0 ? (
                  <Alert severity="success">No blockers - Asset is ready!</Alert>
                ) : (
                  <Stack spacing={1}>
                    {explainability.blockers.map((blocker) => (
                      <Paper key={blocker.id} elevation={1} sx={{ p: 2 }}>
                        <Stack direction="row" spacing={2} alignItems="center">
                          <Chip
                            label={blocker.type}
                            size="small"
                            color="error"
                          />
                          <Box sx={{ flex: 1 }}>
                            {blocker.name && (
                              <Typography variant="subtitle2">{blocker.name}</Typography>
                            )}
                            <Typography variant="body2" color="text.secondary">
                              {blocker.reason}
                            </Typography>
                          </Box>
                        </Stack>
                      </Paper>
                    ))}
                  </Stack>
                )}
              </Box>
            </Stack>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};
