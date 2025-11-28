"""
GIS API Endpoints
Serve GIS shapefile data as GeoJSON
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
import geopandas as gpd
import json
from pathlib import Path
import os

router = APIRouter()

# Base paths to GIS data
GIS_BASE_PATH = Path("/datasets/hs2/rawdata/Phase_2a_WDEIAR_September_2016/Shapefiles")
ORGANIZED_PATH = Path("/datasets/hs2/organized")

# Map of available layer categories from rawdata
GIS_CATEGORIES = {
    "construction": "Construction_Proposed_Scheme",
    "environmental": "Environmental_Topics",
    "parliamentary": "Hybrid_Bill_Parliamentary_Plans_Sections"
}

# Map of organized data categories (NEW)
ORGANIZED_CATEGORIES = {
    "ecology": "ecology/surveys/november-2024/shapefiles",
    "property": "property/compensation/july-2014/shapefiles",
    "legal": "gis-shapefiles/legal/injunctions"
}


@router.get("/layers")
async def get_available_layers():
    """
    Get list of all available GIS layers (from both rawdata and organized folders)
    """
    layers = []

    # Load layers from rawdata (original shapefiles)
    for category_id, category_path in GIS_CATEGORIES.items():
        category_full_path = GIS_BASE_PATH / category_path

        if not category_full_path.exists():
            continue

        # Find all shapefiles in this category
        shapefiles = list(category_full_path.glob("*.shp"))

        for shp_file in shapefiles:
            try:
                # Read shapefile using pyogrio engine (faster, more reliable than Fiona)
                gdf = gpd.read_file(shp_file, engine='pyogrio')

                layers.append({
                    "id": shp_file.stem,
                    "name": shp_file.stem.replace("_", " "),
                    "category": category_id,
                    "feature_count": len(gdf),
                    "geometry_type": gdf.geometry.type.unique().tolist(),
                    "bounds": gdf.total_bounds.tolist() if len(gdf) > 0 else None,
                    "status": "loaded",
                    "source": "rawdata"
                })
            except Exception as e:
                print(f"Error reading {shp_file}: {e}")
                continue

    # Load layers from organized data (NEW)
    for category_id, category_path in ORGANIZED_CATEGORIES.items():
        category_full_path = ORGANIZED_PATH / category_path

        if not category_full_path.exists():
            continue

        # Find all shapefiles in this category
        shapefiles = list(category_full_path.glob("*.shp"))

        for shp_file in shapefiles:
            try:
                gdf = gpd.read_file(shp_file, engine='pyogrio')

                layers.append({
                    "id": shp_file.stem,
                    "name": shp_file.stem.replace("_", " "),
                    "category": category_id,
                    "feature_count": len(gdf),
                    "geometry_type": gdf.geometry.type.unique().tolist(),
                    "bounds": gdf.total_bounds.tolist() if len(gdf) > 0 else None,
                    "status": "loaded",
                    "source": "organized"
                })
            except Exception as e:
                print(f"Error reading {shp_file}: {e}")
                continue

    return {
        "layers": layers,
        "total_count": len(layers),
        "categories": list(GIS_CATEGORIES.keys()) + list(ORGANIZED_CATEGORIES.keys())
    }


@router.get("/layer/{layer_id}")
async def get_layer_geojson(
    layer_id: str,
    simplify: Optional[float] = Query(None, description="Simplification tolerance in degrees"),
    limit: Optional[int] = Query(None, description="Limit number of features returned")
):
    """
    Get GeoJSON data for a specific layer

    Args:
        layer_id: Shapefile name (without .shp extension)
        simplify: Optional simplification tolerance (recommended: 0.0001 for web display)
        limit: Optional limit on number of features
    """
    # Search for shapefile across all categories in rawdata
    shapefile_path = None

    for category_path in GIS_CATEGORIES.values():
        category_full_path = GIS_BASE_PATH / category_path
        potential_path = category_full_path / f"{layer_id}.shp"

        if potential_path.exists():
            shapefile_path = potential_path
            break

    # If not found, search in organized data (NEW)
    if not shapefile_path:
        for category_path in ORGANIZED_CATEGORIES.values():
            category_full_path = ORGANIZED_PATH / category_path
            potential_path = category_full_path / f"{layer_id}.shp"

            if potential_path.exists():
                shapefile_path = potential_path
                break

    # Also check for GeoJSON files in organized legal folder (injunctions)
    if not shapefile_path and layer_id == "HS2_Injunctions":
        geojson_path = ORGANIZED_PATH / "gis-shapefiles/legal/injunctions/HS2_Injunctions.geojson"
        if geojson_path.exists():
            # Read GeoJSON directly
            gdf = gpd.read_file(geojson_path, engine='pyogrio')

            # Ensure WGS84 projection
            if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")

            # Simplify if requested
            if simplify:
                gdf['geometry'] = gdf['geometry'].simplify(tolerance=simplify, preserve_topology=True)

            # Limit if requested
            if limit:
                gdf = gdf.head(limit)

            # Convert datetime columns
            for col in gdf.columns:
                if gdf[col].dtype == 'datetime64[ns]' or str(gdf[col].dtype).startswith('datetime'):
                    gdf[col] = gdf[col].astype(str)

            geojson = json.loads(gdf.to_json())

            return {
                "type": "FeatureCollection",
                "name": layer_id,
                "crs": {
                    "type": "name",
                    "properties": {"name": "EPSG:4326"}
                },
                "features": geojson["features"],
                "metadata": {
                    "feature_count": len(gdf),
                    "bounds": gdf.total_bounds.tolist(),
                    "geometry_types": gdf.geometry.type.unique().tolist(),
                    "columns": gdf.columns.tolist()
                }
            }

    if not shapefile_path:
        raise HTTPException(status_code=404, detail=f"Layer '{layer_id}' not found")

    try:
        # Read shapefile using pyogrio engine
        gdf = gpd.read_file(shapefile_path, engine='pyogrio')

        # Ensure WGS84 projection for web maps
        if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        # Simplify geometry if requested
        if simplify:
            gdf['geometry'] = gdf['geometry'].simplify(tolerance=simplify, preserve_topology=True)

        # Limit features if requested
        if limit:
            gdf = gdf.head(limit)

        # Convert datetime columns to strings for JSON serialization
        for col in gdf.columns:
            if gdf[col].dtype == 'datetime64[ns]' or str(gdf[col].dtype).startswith('datetime'):
                gdf[col] = gdf[col].astype(str)

        # Convert to GeoJSON
        geojson = json.loads(gdf.to_json())

        return {
            "type": "FeatureCollection",
            "name": layer_id,
            "crs": {
                "type": "name",
                "properties": {"name": "EPSG:4326"}
            },
            "features": geojson["features"],
            "metadata": {
                "feature_count": len(gdf),
                "bounds": gdf.total_bounds.tolist(),
                "geometry_types": gdf.geometry.type.unique().tolist(),
                "columns": gdf.columns.tolist()
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing shapefile: {str(e)}")


@router.get("/route")
async def get_hs2_route():
    """
    Get the main HS2 railway alignment route
    """
    # Find railway alignment shapefile
    alignment_files = [
        "HS2_RE_RailAlignmentFormation_Ply_CT05_WDEIA",
        "HS2_RE_RailAlignmentFormation_Ply_CT06_WDEIA"
    ]

    all_features = []

    for layer_id in alignment_files:
        for category_path in GIS_CATEGORIES.values():
            category_full_path = GIS_BASE_PATH / category_path
            shapefile_path = category_full_path / f"{layer_id}.shp"

            if shapefile_path.exists():
                try:
                    gdf = gpd.read_file(shapefile_path, engine='pyogrio')

                    # Convert to WGS84
                    if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
                        gdf = gdf.to_crs("EPSG:4326")

                    # Simplify for web display
                    gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.0001, preserve_topology=True)

                    # Convert datetime columns to strings
                    for col in gdf.columns:
                        if gdf[col].dtype == 'datetime64[ns]' or str(gdf[col].dtype).startswith('datetime'):
                            gdf[col] = gdf[col].astype(str)

                    geojson = json.loads(gdf.to_json())
                    all_features.extend(geojson["features"])

                except Exception as e:
                    print(f"Error reading {shapefile_path}: {e}")
                    continue

    if not all_features:
        raise HTTPException(status_code=404, detail="HS2 route alignment not found")

    return {
        "type": "FeatureCollection",
        "name": "HS2_Route_Alignment",
        "features": all_features,
        "metadata": {
            "feature_count": len(all_features),
            "description": "HS2 Phase 2a Railway Alignment"
        }
    }


@router.get("/construction-sites")
async def get_construction_sites():
    """
    Get construction sites and compounds
    """
    compound_files = [
        "CON_CN_SatelliteConstructionCompounds_Ply_CT05_WDEIA",
        "CON_CN_SatelliteConstructionCompounds_Ply_CT06_WDEIA",
        "CON_CN_MainConstructionCompounds_Ply_CT05_WDEIA",
        "CON_CN_MainConstructionCompounds_Ply_CT06_WDEIA"
    ]

    all_features = []

    for layer_id in compound_files:
        category_full_path = GIS_BASE_PATH / "Construction_Proposed_Scheme"
        shapefile_path = category_full_path / f"{layer_id}.shp"

        if shapefile_path.exists():
            try:
                gdf = gpd.read_file(shapefile_path)

                # Convert to WGS84
                if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
                    gdf = gdf.to_crs("EPSG:4326")

                geojson = json.loads(gdf.to_json())

                # Add site type to properties
                site_type = "Satellite" if "Satellite" in layer_id else "Main"
                for feature in geojson["features"]:
                    feature["properties"]["site_type"] = site_type

                all_features.extend(geojson["features"])

            except Exception as e:
                print(f"Error reading {shapefile_path}: {e}")
                continue

    if not all_features:
        raise HTTPException(status_code=404, detail="Construction sites not found")

    return {
        "type": "FeatureCollection",
        "name": "HS2_Construction_Sites",
        "features": all_features,
        "metadata": {
            "feature_count": len(all_features),
            "description": "HS2 Construction Compounds and Sites"
        }
    }


@router.get("/environmental")
async def get_environmental_features():
    """
    Get environmental features (noise barriers, wetlands, planting areas)
    """
    env_files = [
        "BEN_AG_NoiseBarriers_Ln_CT06_WDEIA",
        "ENV_LD_PlantingWetlands_Ply_CT06_WDEIA",
        "ENV_LD_PlantingWoodland_Ply_CT06_WDEIA"
    ]

    all_features = []

    for layer_id in env_files:
        category_full_path = GIS_BASE_PATH / "Construction_Proposed_Scheme"
        shapefile_path = category_full_path / f"{layer_id}.shp"

        if shapefile_path.exists():
            try:
                gdf = gpd.read_file(shapefile_path)

                # Convert to WGS84
                if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
                    gdf = gdf.to_crs("EPSG:4326")

                geojson = json.loads(gdf.to_json())

                # Add feature type to properties
                if "NoiseBarriers" in layer_id:
                    feature_type = "Noise Barrier"
                elif "Wetlands" in layer_id:
                    feature_type = "Wetland Planting"
                else:
                    feature_type = "Woodland Planting"

                for feature in geojson["features"]:
                    feature["properties"]["env_type"] = feature_type

                all_features.extend(geojson["features"])

            except Exception as e:
                print(f"Error reading {shapefile_path}: {e}")
                continue

    return {
        "type": "FeatureCollection",
        "name": "HS2_Environmental_Features",
        "features": all_features,
        "metadata": {
            "feature_count": len(all_features),
            "description": "HS2 Environmental Mitigation Features"
        }
    }


@router.get("/ecology")
async def get_ecology_features(
    limit: Optional[int] = Query(500, description="Maximum number of features to return")
):
    """
    Get ecology survey features (NEW - from organized data)
    Limited to prevent browser overload
    """
    ecology_path = ORGANIZED_PATH / "ecology/surveys/november-2024/shapefiles"

    # Load only the main ecological assets shapefile (not all shapefiles)
    shapefile_path = ecology_path / "HS2_Ecological_Assets.shp"

    if not shapefile_path.exists():
        raise HTTPException(status_code=404, detail="Ecology survey data not found")

    try:
        # Read shapefile
        gdf = gpd.read_file(shapefile_path, engine='pyogrio')

        # Filter out invalid geometries
        gdf = gdf[gdf.geometry.is_valid]

        # Limit features for performance
        if limit and len(gdf) > limit:
            gdf = gdf.head(limit)

        # Convert to WGS84
        if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        # Simplify for web display
        gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.001, preserve_topology=True)

        # Convert datetime columns
        for col in gdf.columns:
            if gdf[col].dtype == 'datetime64[ns]' or str(gdf[col].dtype).startswith('datetime'):
                gdf[col] = gdf[col].astype(str)

        # Convert to GeoJSON
        geojson = json.loads(gdf.to_json())

        return {
            "type": "FeatureCollection",
            "name": "HS2_Ecological_Assets",
            "features": geojson["features"],
            "metadata": {
                "feature_count": len(gdf),
                "description": "HS2 Ecological Assets - November 2024 (Limited for Performance)",
                "survey_date": "November 2024",
                "total_available": len(gdf)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing ecology data: {str(e)}")


@router.get("/injunctions")
async def get_injunctions():
    """
    Get HS2 legal injunction boundaries (NEW - from organized data)
    """
    geojson_path = ORGANIZED_PATH / "gis-shapefiles/legal/injunctions/HS2_Injunctions.geojson"

    if not geojson_path.exists():
        raise HTTPException(status_code=404, detail="Injunctions data not found")

    try:
        # Read GeoJSON directly
        gdf = gpd.read_file(geojson_path, engine='pyogrio')

        # Ensure WGS84 projection
        if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        # Simplify for web display (injunctions are large dataset - 6,886 features)
        gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.0005, preserve_topology=True)

        # Convert datetime columns
        for col in gdf.columns:
            if gdf[col].dtype == 'datetime64[ns]' or str(gdf[col].dtype).startswith('datetime'):
                gdf[col] = gdf[col].astype(str)

        geojson = json.loads(gdf.to_json())

        return {
            "type": "FeatureCollection",
            "name": "HS2_Injunctions",
            "features": geojson["features"],
            "metadata": {
                "feature_count": len(gdf),
                "description": "HS2 Court-Ordered Injunction Boundaries",
                "bounds": gdf.total_bounds.tolist()
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing injunctions: {str(e)}")


@router.get("/property-compensation")
async def get_property_compensation():
    """
    Get property compensation zones (NEW - from organized data)
    """
    property_path = ORGANIZED_PATH / "property/compensation/july-2014/shapefiles"
    shapefile_path = property_path / "HS2_HSTWO_Property_Compensation_Zones_July_2014.shp"

    if not shapefile_path.exists():
        raise HTTPException(status_code=404, detail="Property compensation data not found")

    try:
        gdf = gpd.read_file(shapefile_path, engine='pyogrio')

        # Convert to WGS84
        if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        # Simplify for web display
        gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.0001, preserve_topology=True)

        # Convert datetime columns
        for col in gdf.columns:
            if gdf[col].dtype == 'datetime64[ns]' or str(gdf[col].dtype).startswith('datetime'):
                gdf[col] = gdf[col].astype(str)

        geojson = json.loads(gdf.to_json())

        return {
            "type": "FeatureCollection",
            "name": "HS2_Property_Compensation",
            "features": geojson["features"],
            "metadata": {
                "feature_count": len(gdf),
                "description": "HS2 Property Compensation Zones - July 2014 Consultation",
                "consultation_date": "July 2014"
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing property data: {str(e)}")


@router.get("/assets-locations")
async def get_assets_locations():
    """
    Get locations of all 500 HS2 assets for GIS map display

    Returns GeoJSON with asset locations along HS2 route, color-coded by readiness status:
    - Ready (green)
    - Not Ready (red)
    - At Risk (orange)
    """
    from sqlalchemy import select
    from app.core.database import get_sync_db
    from app.models.hs2 import HS2Asset
    import random

    try:
        # Get all assets from database
        db = get_sync_db()
        result = db.execute(select(HS2Asset))
        assets = result.scalars().all()
        db.close()

        # HS2 Phase 1 Route coordinates (London to Birmingham) - Actual route points
        route_sections = {
            "London Euston - Old Oak Common": {
                "points": [
                    [51.5308, -0.1238],  # London Euston
                    [51.5400, -0.1500],
                    [51.5500, -0.2000]   # Old Oak Common
                ]
            },
            "Old Oak Common - Ruislip": {
                "points": [
                    [51.5500, -0.2000],  # Old Oak Common
                    [51.6000, -0.3000],
                    [51.6500, -0.4000]   # Ruislip
                ]
            },
            "Ruislip - Chalfont St Peter": {
                "points": [
                    [51.6500, -0.4000],  # Ruislip
                    [51.7000, -0.5000],
                    [51.7500, -0.5500],  # Harefield
                    [51.8000, -0.6500],  # Chalfont St Peter
                ]
            },
            "Chalfont St Peter - Wendover": {
                "points": [
                    [51.8000, -0.6500],  # Chalfont St Peter
                    [51.8500, -0.7500],  # Amersham
                    [51.9000, -0.8500],  # Great Missenden
                    [51.9500, -0.9500]   # Wendover
                ]
            },
            "Wendover - Calvert": {
                "points": [
                    [51.9500, -0.9500],  # Wendover
                    [52.0000, -1.0000],  # Aylesbury
                    [52.1000, -1.1000]   # Calvert
                ]
            },
            "Calvert - Brackley": {
                "points": [
                    [52.1000, -1.1000],  # Calvert
                    [52.1500, -1.1500],
                    [52.2000, -1.2000]   # Brackley
                ]
            },
            "Brackley - Birmingham Interchange": {
                "points": [
                    [52.2000, -1.2000],  # Brackley
                    [52.3000, -1.3000],  # Kings Sutton
                    [52.4000, -1.4000],  # Chipping Warden
                    [52.4800, -1.4800]   # Birmingham Interchange (NEC)
                ]
            },
            "Birmingham Interchange - Curzon Street": {
                "points": [
                    [52.4800, -1.4800],  # Birmingham Interchange
                    [52.4900, -1.4900],
                    [52.5000, -1.5000]   # Birmingham Curzon Street
                ]
            },
            "West Midlands - Crewe (Phase 2a)": {
                "points": [
                    [52.5000, -1.5000],  # Birmingham
                    [52.6000, -1.7000],
                    [52.7000, -2.1000],  # West Midlands
                    [52.9000, -2.3000]   # Towards Crewe
                ]
            }
        }

        # Generate GeoJSON features for each asset
        features = []

        for asset in assets:
            # Get route section points
            section_info = route_sections.get(asset.route_section)
            if not section_info:
                # Default to center if section not found
                lat = 52.0
                lng = -1.0
            else:
                # Interpolate along the route points
                points = section_info["points"]
                # Pick a random segment
                if len(points) > 1:
                    segment_idx = random.randint(0, len(points) - 2)
                    start_point = points[segment_idx]
                    end_point = points[segment_idx + 1]

                    # Interpolate between start and end with small offset
                    t = random.uniform(0.0, 1.0)
                    lat = start_point[0] + t * (end_point[0] - start_point[0])
                    lng = start_point[1] + t * (end_point[1] - start_point[1])

                    # Add small perpendicular offset (max 0.01 degrees ~1km)
                    offset = random.uniform(-0.01, 0.01)
                    lat += offset
                    lng += offset * 0.7  # Adjust for longitude at UK latitude
                else:
                    lat = points[0][0]
                    lng = points[0][1]

            # Determine color based on readiness status
            if asset.readiness_status == "Ready":
                color = "#4CAF50"  # Green
                marker_color = "green"
            elif asset.readiness_status == "At Risk":
                color = "#FF9800"  # Orange
                marker_color = "orange"
            else:  # Not Ready
                color = "#F44336"  # Red
                marker_color = "red"

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lng, lat]
                },
                "properties": {
                    "asset_id": asset.asset_id,
                    "asset_name": asset.asset_name,
                    "asset_type": asset.asset_type,
                    "contractor": asset.contractor,
                    "readiness_status": asset.readiness_status,
                    "taem_score": float(asset.taem_evaluation_score) if asset.taem_evaluation_score else 0.0,
                    "route_section": asset.route_section,
                    "color": color,
                    "marker_color": marker_color
                }
            }
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "name": "HS2_Assets",
            "features": features,
            "metadata": {
                "total_assets": len(features),
                "ready": sum(1 for f in features if f["properties"]["readiness_status"] == "Ready"),
                "not_ready": sum(1 for f in features if f["properties"]["readiness_status"] == "Not Ready"),
                "at_risk": sum(1 for f in features if f["properties"]["readiness_status"] == "At Risk"),
                "description": "500 synthetic HS2 assets with locations along Phase 1 route",
                "is_synthetic": True
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating asset locations: {str(e)}")
