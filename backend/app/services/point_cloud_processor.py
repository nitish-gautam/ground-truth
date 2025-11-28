"""
HS2 Point Cloud Processing Service
====================================

Provides point cloud processing capabilities for progress verification:
- LAS/LAZ file loading and parsing
- Point cloud-to-BIM comparison algorithms
- Surface deviation analysis
- Volume difference calculations
- Heatmap generation for visualization
- Quality assessment and anomaly detection

Supports industry-standard formats:
- LAS 1.2, 1.3, 1.4 (LiDAR data)
- LAZ (compressed LAS)
- PLY (mesh and point cloud)
- IFC (via existing BIM service integration)
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
from datetime import datetime
from decimal import Decimal
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from open3d.geometry import PointCloud
else:
    PointCloud = Any

# Point cloud libraries
try:
    import open3d as o3d
    import laspy
    from plyfile import PlyData
    POINT_CLOUD_AVAILABLE = True
except ImportError:
    # Create dummy modules for type hints when libraries aren't installed
    POINT_CLOUD_AVAILABLE = False
    o3d = None  # type: ignore
    laspy = None  # type: ignore
    PlyData = None  # type: ignore
    logging.warning("Point cloud libraries not installed. Install with: pip install open3d laspy plyfile")

logger = logging.getLogger(__name__)


class PointCloudProcessor:
    """
    Service for processing point cloud data for progress verification.

    Key Features:
    - Load LAS/LAZ/PLY files
    - Align point clouds (ICP algorithm)
    - Compute surface deviations
    - Calculate volume differences
    - Generate heatmaps for visualization
    - Detect anomalies and quality issues
    """

    def __init__(self, tolerance_mm: float = 50.0):
        """
        Initialize point cloud processor.

        Args:
            tolerance_mm: Tolerance threshold in millimeters for compliance
        """
        if not POINT_CLOUD_AVAILABLE:
            raise ImportError("Point cloud libraries not available. Install: pip install open3d laspy plyfile")

        self.tolerance_mm = tolerance_mm
        self.executor = ThreadPoolExecutor(max_workers=4)

        logger.info(f"PointCloudProcessor initialized with {tolerance_mm}mm tolerance")

    # ==================== File Loading ====================

    async def load_las_file(self, file_path: str) -> "PointCloud":
        """
        Load LAS/LAZ file asynchronously.

        Args:
            file_path: Path to LAS or LAZ file

        Returns:
            Open3D PointCloud object

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        logger.info(f"Loading LAS/LAZ file: {file_path}")

        def _load_las():
            if not Path(file_path).exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Read LAS file
            las = laspy.read(file_path)

            # Extract XYZ coordinates
            points = np.vstack((las.x, las.y, las.z)).transpose()

            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # Extract colors if available (RGB)
            if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
                colors = np.vstack((
                    las.red / 65535.0,
                    las.green / 65535.0,
                    las.blue / 65535.0
                )).transpose()
                pcd.colors = o3d.utility.Vector3dVector(colors)

            logger.info(f"Loaded {len(points)} points from {file_path}")
            return pcd

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _load_las)

    async def load_ply_file(self, file_path: str) -> "PointCloud":
        """
        Load PLY file asynchronously.

        Args:
            file_path: Path to PLY file

        Returns:
            Open3D PointCloud object
        """
        logger.info(f"Loading PLY file: {file_path}")

        def _load_ply():
            pcd = o3d.io.read_point_cloud(file_path)
            logger.info(f"Loaded {len(pcd.points)} points from {file_path}")
            return pcd

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _load_ply)

    # ==================== Preprocessing ====================

    async def downsample_point_cloud(
        self,
        pcd: "PointCloud",
        voxel_size: float = 0.05
    ) -> "PointCloud":
        """
        Downsample point cloud for faster processing.

        Args:
            pcd: Input point cloud
            voxel_size: Voxel size in meters (default 5cm)

        Returns:
            Downsampled point cloud
        """
        logger.info(f"Downsampling point cloud with voxel_size={voxel_size}m")

        def _downsample():
            downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
            logger.info(f"Downsampled from {len(pcd.points)} to {len(downsampled.points)} points")
            return downsampled

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _downsample)

    async def remove_outliers(
        self,
        pcd: "PointCloud",
        nb_neighbors: int = 20,
        std_ratio: float = 2.0
    ) -> "PointCloud":
        """
        Remove statistical outliers from point cloud.

        Args:
            pcd: Input point cloud
            nb_neighbors: Number of neighbors to analyze
            std_ratio: Standard deviation ratio threshold

        Returns:
            Cleaned point cloud
        """
        logger.info(f"Removing outliers (nb_neighbors={nb_neighbors}, std_ratio={std_ratio})")

        def _remove_outliers():
            cleaned, _ = pcd.remove_statistical_outlier(
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio
            )
            logger.info(f"Removed {len(pcd.points) - len(cleaned.points)} outlier points")
            return cleaned

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _remove_outliers)

    # ==================== Alignment ====================

    async def align_point_clouds(
        self,
        source: "PointCloud",
        target: "PointCloud",
        threshold: float = 0.05,
        max_iterations: int = 50
    ) -> Tuple["PointCloud", NDArray[np.float64], Dict[str, Any]]:
        """
        Align two point clouds using ICP (Iterative Closest Point) algorithm.

        Args:
            source: Source point cloud (to be aligned)
            target: Target point cloud (reference)
            threshold: Distance threshold in meters
            max_iterations: Maximum ICP iterations

        Returns:
            Tuple of (aligned_source, transformation_matrix, alignment_metrics)
        """
        logger.info(f"Aligning point clouds with threshold={threshold}m, max_iter={max_iterations}")

        def _align():
            # Estimate normals for better alignment
            source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

            # Run ICP registration
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source, target, threshold,
                np.eye(4),  # Initial transformation
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
            )

            # Apply transformation
            source_aligned = source.transform(reg_p2p.transformation)

            metrics = {
                "fitness": float(reg_p2p.fitness),
                "inlier_rmse": float(reg_p2p.inlier_rmse),
                "iterations": int(max_iterations),
                "correspondence_set_size": len(reg_p2p.correspondence_set)
            }

            logger.info(f"Alignment complete: fitness={metrics['fitness']:.4f}, RMSE={metrics['inlier_rmse']:.4f}m")
            return source_aligned, reg_p2p.transformation, metrics

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _align)

    # ==================== Deviation Analysis ====================

    async def compute_point_to_point_distances(
        self,
        source: "PointCloud",
        target: "PointCloud"
    ) -> NDArray[np.float64]:
        """
        Compute point-to-point distances between two aligned point clouds.

        Args:
            source: Source point cloud
            target: Target point cloud (reference)

        Returns:
            Array of distances in meters for each point
        """
        logger.info("Computing point-to-point distances")

        def _compute_distances():
            distances = source.compute_point_cloud_distance(target)
            distances_array = np.asarray(distances)
            logger.info(f"Computed {len(distances_array)} point distances")
            return distances_array

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _compute_distances)

    async def analyze_surface_deviation(
        self,
        baseline_pcd: "PointCloud",
        current_pcd: "PointCloud"
    ) -> Dict[str, Any]:
        """
        Comprehensive surface deviation analysis between baseline and current scans.

        Args:
            baseline_pcd: Baseline point cloud (BIM model or initial scan)
            current_pcd: Current site scan

        Returns:
            Dictionary with deviation statistics and heatmap data
        """
        logger.info("Starting surface deviation analysis")

        # Align point clouds
        aligned_current, transformation, alignment_metrics = await self.align_point_clouds(
            current_pcd, baseline_pcd
        )

        # Compute distances (meters)
        distances_m = await self.compute_point_to_point_distances(aligned_current, baseline_pcd)

        # Convert to millimeters for reporting
        distances_mm = distances_m * 1000.0

        # Statistical analysis
        avg_deviation_mm = float(np.mean(distances_mm))
        max_deviation_mm = float(np.max(distances_mm))
        min_deviation_mm = float(np.min(distances_mm))
        std_deviation_mm = float(np.std(distances_mm))

        # Tolerance analysis
        within_tolerance = np.sum(distances_mm <= self.tolerance_mm)
        within_tolerance_pct = (within_tolerance / len(distances_mm)) * 100.0

        # Identify hotspots (high deviation areas)
        hotspot_threshold_mm = self.tolerance_mm * 2.0  # 2x tolerance
        hotspot_indices = np.where(distances_mm > hotspot_threshold_mm)[0]

        hotspots = []
        if len(hotspot_indices) > 0:
            current_points = np.asarray(aligned_current.points)
            for idx in hotspot_indices[:100]:  # Limit to top 100 hotspots
                hotspots.append({
                    "x": float(current_points[idx][0]),
                    "y": float(current_points[idx][1]),
                    "z": float(current_points[idx][2]),
                    "deviation_mm": float(distances_mm[idx])
                })

        # Generate heatmap data (simplified for API response)
        heatmap_data = self._generate_heatmap_summary(
            np.asarray(aligned_current.points),
            distances_mm
        )

        logger.info(f"Deviation analysis complete: avg={avg_deviation_mm:.2f}mm, max={max_deviation_mm:.2f}mm")

        return {
            "surface_deviation_avg": Decimal(str(round(avg_deviation_mm, 2))),
            "surface_deviation_max": Decimal(str(round(max_deviation_mm, 2))),
            "surface_deviation_min": Decimal(str(round(min_deviation_mm, 2))),
            "surface_deviation_std": Decimal(str(round(std_deviation_mm, 2))),
            "points_within_tolerance_pct": Decimal(str(round(within_tolerance_pct, 2))),
            "tolerance_threshold_mm": Decimal(str(self.tolerance_mm)),
            "heatmap_data": heatmap_data,
            "hotspots": hotspots,
            "alignment_metrics": alignment_metrics,
            "transformation_matrix": transformation.tolist(),
            "point_count_baseline": len(baseline_pcd.points),
            "point_count_current": len(current_pcd.points)
        }

    def _generate_heatmap_summary(
        self,
        points: NDArray[np.float64],
        deviations_mm: NDArray[np.float64],
        grid_size: int = 50
    ) -> Dict[str, Any]:
        """
        Generate simplified heatmap data for frontend visualization.

        Args:
            points: Point coordinates (Nx3 array)
            deviations_mm: Deviation values in mm (N array)
            grid_size: Grid resolution for heatmap

        Returns:
            Heatmap summary data
        """
        # Compute bounding box
        min_bounds = points.min(axis=0)
        max_bounds = points.max(axis=0)

        # Create 2D grid (XY plane)
        x_edges = np.linspace(min_bounds[0], max_bounds[0], grid_size + 1)
        y_edges = np.linspace(min_bounds[1], max_bounds[1], grid_size + 1)

        # Bin points into grid and compute average deviation per cell
        heatmap_grid = np.zeros((grid_size, grid_size))
        count_grid = np.zeros((grid_size, grid_size))

        x_indices = np.digitize(points[:, 0], x_edges) - 1
        y_indices = np.digitize(points[:, 1], y_edges) - 1

        # Clip to valid range
        x_indices = np.clip(x_indices, 0, grid_size - 1)
        y_indices = np.clip(y_indices, 0, grid_size - 1)

        for i in range(len(points)):
            heatmap_grid[x_indices[i], y_indices[i]] += deviations_mm[i]
            count_grid[x_indices[i], y_indices[i]] += 1

        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            heatmap_grid = np.where(count_grid > 0, heatmap_grid / count_grid, 0)

        return {
            "grid_size": grid_size,
            "x_min": float(min_bounds[0]),
            "x_max": float(max_bounds[0]),
            "y_min": float(min_bounds[1]),
            "y_max": float(max_bounds[1]),
            "z_min": float(min_bounds[2]),
            "z_max": float(max_bounds[2]),
            "heatmap_values": heatmap_grid.tolist(),
            "description": "Average deviation (mm) per grid cell"
        }

    # ==================== Volume Analysis ====================

    async def compute_volume_difference(
        self,
        baseline_pcd: "PointCloud",
        current_pcd: "PointCloud"
    ) -> Dict[str, Decimal]:
        """
        Compute volume difference between baseline and current point clouds.

        Uses Alpha Shape algorithm to estimate volumes from point clouds.

        Args:
            baseline_pcd: Baseline point cloud
            current_pcd: Current point cloud

        Returns:
            Dictionary with volume metrics in cubic meters
        """
        logger.info("Computing volume difference")

        def _compute_volume(pcd: "PointCloud") -> float:
            """Estimate volume using convex hull."""
            try:
                hull, _ = pcd.compute_convex_hull()
                volume = hull.get_volume()
                return volume
            except Exception as e:
                logger.warning(f"Volume computation failed: {e}")
                return 0.0

        loop = asyncio.get_event_loop()

        # Compute volumes in parallel
        volume_baseline, volume_current = await asyncio.gather(
            loop.run_in_executor(self.executor, _compute_volume, baseline_pcd),
            loop.run_in_executor(self.executor, _compute_volume, current_pcd)
        )

        volume_diff = volume_current - volume_baseline

        logger.info(f"Volume analysis: baseline={volume_baseline:.2f}m³, current={volume_current:.2f}m³, diff={volume_diff:.2f}m³")

        return {
            "volume_planned_m3": Decimal(str(round(volume_baseline, 2))),
            "volume_actual_m3": Decimal(str(round(volume_current, 2))),
            "volume_difference_m3": Decimal(str(round(volume_diff, 2)))
        }

    # ==================== Quality Assessment ====================

    async def assess_scan_quality(
        self,
        pcd: "PointCloud"
    ) -> List[Dict[str, Any]]:
        """
        Assess point cloud scan quality and identify issues.

        Args:
            pcd: Point cloud to assess

        Returns:
            List of quality flags/warnings
        """
        logger.info("Assessing scan quality")

        quality_flags = []

        # Check point density
        points = np.asarray(pcd.points)
        bbox = pcd.get_axis_aligned_bounding_box()
        volume = bbox.volume()

        if volume > 0:
            density = len(points) / volume
            if density < 1000:  # Less than 1000 points per m³
                quality_flags.append({
                    "type": "low_density",
                    "severity": "medium",
                    "message": f"Low point density: {density:.0f} points/m³",
                    "recommendation": "Consider rescanning with higher resolution"
                })

        # Check for gaps (using nearest neighbor distances)
        def _check_gaps():
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            distances = []

            # Sample 1000 points for performance
            sample_indices = np.random.choice(len(points), min(1000, len(points)), replace=False)

            for idx in sample_indices:
                [_, idx_neighbors, _] = pcd_tree.search_knn_vector_3d(pcd.points[idx], 2)
                if len(idx_neighbors) > 1:
                    neighbor_point = points[idx_neighbors[1]]
                    dist = np.linalg.norm(points[idx] - neighbor_point)
                    distances.append(dist)

            return np.array(distances)

        loop = asyncio.get_event_loop()
        nn_distances = await loop.run_in_executor(self.executor, _check_gaps)

        if len(nn_distances) > 0:
            avg_spacing = float(np.mean(nn_distances))
            max_spacing = float(np.max(nn_distances))

            if max_spacing > 0.5:  # Gaps larger than 0.5m
                quality_flags.append({
                    "type": "large_gaps",
                    "severity": "high",
                    "message": f"Large gaps detected: max spacing {max_spacing:.2f}m",
                    "recommendation": "Review scan coverage and rescan missing areas"
                })

        logger.info(f"Quality assessment complete: {len(quality_flags)} flags raised")
        return quality_flags

    # ==================== High-Level Processing ====================

    async def process_comparison(
        self,
        baseline_file_path: str,
        current_file_path: str,
        downsample_voxel_size: float = 0.05,
        remove_outliers_flag: bool = True
    ) -> Dict[str, Any]:
        """
        Complete point cloud comparison workflow.

        Args:
            baseline_file_path: Path to baseline point cloud (BIM or initial scan)
            current_file_path: Path to current site scan
            downsample_voxel_size: Voxel size for downsampling (meters)
            remove_outliers_flag: Whether to remove outliers

        Returns:
            Complete comparison results with all metrics
        """
        logger.info(f"Starting point cloud comparison: baseline={baseline_file_path}, current={current_file_path}")

        start_time = datetime.now()

        # Load point clouds
        baseline_pcd = await self.load_las_file(baseline_file_path)
        current_pcd = await self.load_las_file(current_file_path)

        # Preprocessing
        if downsample_voxel_size > 0:
            baseline_pcd = await self.downsample_point_cloud(baseline_pcd, downsample_voxel_size)
            current_pcd = await self.downsample_point_cloud(current_pcd, downsample_voxel_size)

        if remove_outliers_flag:
            baseline_pcd = await self.remove_outliers(baseline_pcd)
            current_pcd = await self.remove_outliers(current_pcd)

        # Run analyses in parallel
        deviation_analysis, volume_analysis, quality_flags = await asyncio.gather(
            self.analyze_surface_deviation(baseline_pcd, current_pcd),
            self.compute_volume_difference(baseline_pcd, current_pcd),
            self.assess_scan_quality(current_pcd)
        )

        # Combine results
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        results = {
            **deviation_analysis,
            **volume_analysis,
            "quality_flags": quality_flags,
            "processing_time_seconds": int(processing_time),
            "algorithm_version": "1.0.0",
            "processed_by": "PointCloudProcessor"
        }

        logger.info(f"Point cloud comparison complete in {processing_time:.2f}s")
        return results


# Singleton instance
_processor_instance: Optional[PointCloudProcessor] = None


def get_point_cloud_processor(tolerance_mm: float = 50.0) -> PointCloudProcessor:
    """
    Get or create singleton PointCloudProcessor instance.

    Args:
        tolerance_mm: Tolerance threshold in millimeters

    Returns:
        PointCloudProcessor instance
    """
    global _processor_instance

    if _processor_instance is None:
        _processor_instance = PointCloudProcessor(tolerance_mm=tolerance_mm)

    return _processor_instance
