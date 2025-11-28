"""
API endpoints for noise monitoring data
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Optional
from datetime import datetime
import logging

from app.core.database import get_db
from app.models.noise_monitoring import NoiseMonitoringMeasurement, NoiseMonitoringLocation

logger = logging.getLogger(__name__)

router = APIRouter()

NOISE_LIMIT = 75.0  # dB threshold for violations

@router.get("/noise")
async def get_noise_monitoring_data(
    month: str = Query(..., description="Month name (e.g., December_2024)"),
    area: Optional[str] = Query(None, description="Geographic area (North/Central/South)"),
    council: Optional[str] = Query(None, description="Council name"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get noise monitoring data for a specific month

    Returns aggregated statistics, time series, and geographic distribution
    """
    try:
        # Build base query
        query = select(NoiseMonitoringMeasurement).where(
            NoiseMonitoringMeasurement.month == month
        )

        # Apply filters
        if area and area.lower() != 'all':
            query = query.where(NoiseMonitoringMeasurement.area == area)

        if council and council.lower() != 'all':
            query = query.where(NoiseMonitoringMeasurement.council == council)

        # Execute query
        result = await db.execute(query)
        measurements = result.scalars().all()

        if not measurements:
            return {
                'month': month,
                'summary': {
                    'total_measurements': 0,
                    'avg_noise': 0.0,
                    'compliance_rate': 0.0,
                    'violations': 0
                },
                'time_series': [],
                'geographic': [],
                'by_council': [],
                'violations': [],
                'filters': {
                    'areas': [],
                    'councils': []
                }
            }

        # Calculate summary statistics
        total_measurements = len(measurements)
        avg_noise = sum(m.avg_noise_db for m in measurements) / total_measurements
        violations = [m for m in measurements if m.is_violation == 1]
        violation_count = len(violations)
        compliance_rate = ((total_measurements - violation_count) / total_measurements * 100) if total_measurements > 0 else 0.0

        # Time series data (aggregate by hour)
        time_series_dict = {}
        for m in measurements:
            hour_key = m.timestamp.replace(minute=0, second=0, microsecond=0)
            if hour_key not in time_series_dict:
                time_series_dict[hour_key] = {
                    'time': hour_key.strftime('%Y-%m-%d %H:%M'),
                    'avg_noise_values': [],
                    'max_noise_values': []
                }
            time_series_dict[hour_key]['avg_noise_values'].append(m.avg_noise_db)
            time_series_dict[hour_key]['max_noise_values'].append(m.max_noise_db)

        time_series = [
            {
                'time': v['time'],
                'avg_noise': round(sum(v['avg_noise_values']) / len(v['avg_noise_values']), 2),
                'max_noise': round(max(v['max_noise_values']), 2),
                'limit': NOISE_LIMIT
            }
            for k, v in sorted(time_series_dict.items())
        ]

        # Geographic distribution (by location)
        geo_dict = {}
        for m in measurements:
            if m.location_id not in geo_dict:
                geo_dict[m.location_id] = {
                    'location': m.location_id,
                    'council': m.council,
                    'area': m.area,
                    'noise_values': [],
                    'max_values': []
                }
            geo_dict[m.location_id]['noise_values'].append(m.avg_noise_db)
            geo_dict[m.location_id]['max_values'].append(m.max_noise_db)

        geographic = [
            {
                'location': k,
                'council': v['council'],
                'area': v['area'],
                'noise_level': round(sum(v['noise_values']) / len(v['noise_values']), 2),
                'max_noise': round(max(v['max_values']), 2),
                'x': hash(k) % 100,  # Mock coordinates for visualization
                'y': hash(v['council']) % 100
            }
            for k, v in geo_dict.items()
        ]

        # By council statistics
        council_dict = {}
        for m in measurements:
            if m.council not in council_dict:
                council_dict[m.council] = {
                    'council': m.council,
                    'noise_values': [],
                    'max_values': []
                }
            council_dict[m.council]['noise_values'].append(m.avg_noise_db)
            council_dict[m.council]['max_values'].append(m.max_noise_db)

        by_council = [
            {
                'council': k,
                'avg_noise': round(sum(v['noise_values']) / len(v['noise_values']), 2),
                'max_noise': round(max(v['max_values']), 2)
            }
            for k, v in council_dict.items()
        ]

        # Violations list (limit to 20)
        violations_list = [
            {
                'timestamp': v.timestamp.strftime('%Y-%m-%d %H:%M'),
                'location': v.location_id,
                'council': v.council,
                'noise_level': round(v.avg_noise_db, 2),
                'limit': NOISE_LIMIT,
                'excess': round(v.avg_noise_db - NOISE_LIMIT, 2)
            }
            for v in violations[:20]
        ]

        # Get unique areas and councils for filters
        unique_areas = sorted(set(m.area for m in measurements))
        unique_councils = sorted(set(m.council for m in measurements))

        return {
            'month': month,
            'summary': {
                'total_measurements': total_measurements,
                'avg_noise': round(avg_noise, 2),
                'compliance_rate': round(compliance_rate, 2),
                'violations': violation_count
            },
            'time_series': time_series,
            'geographic': geographic,
            'by_council': by_council,
            'violations': violations_list,
            'filters': {
                'areas': unique_areas,
                'councils': unique_councils
            }
        }

    except Exception as e:
        logger.error(f"Error fetching noise monitoring data: {e}", exc_info=True)
        raise


@router.get("/available-months")
async def get_available_months(db: AsyncSession = Depends(get_db)):
    """Get list of available months with data"""
    try:
        query = select(NoiseMonitoringMeasurement.month).distinct()
        result = await db.execute(query)
        months = [row[0] for row in result.fetchall()]
        return {"months": sorted(months)}
    except Exception as e:
        logger.error(f"Error fetching available months: {e}", exc_info=True)
        raise


@router.get("/locations")
async def get_monitoring_locations(db: AsyncSession = Depends(get_db)):
    """Get all monitoring locations"""
    try:
        query = select(NoiseMonitoringLocation)
        result = await db.execute(query)
        locations = result.scalars().all()

        return {
            "locations": [
                {
                    "location_id": loc.location_id,
                    "area": loc.area,
                    "council": loc.council,
                    "latitude": loc.latitude,
                    "longitude": loc.longitude
                }
                for loc in locations
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching locations: {e}", exc_info=True)
        raise
