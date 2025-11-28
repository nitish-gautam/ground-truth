"""
Noise Monitoring Data Service
Loads and processes HS2 noise monitoring Excel files
"""
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class NoiseMonitoringService:
    """Service to load and process HS2 noise monitoring data"""

    def __init__(self, data_dir: str = "/datasets/hs2/rawdata"):
        self.data_dir = Path(data_dir)
        self.cache = {}  # Simple in-memory cache

    def get_available_months(self) -> List[str]:
        """Get list of available months"""
        months = []
        for folder in self.data_dir.glob("hs2_monthly_monitoring_data_*"):
            month_name = folder.name.replace("hs2_monthly_monitoring_data_", "")
            months.append(month_name)
        return sorted(months)

    def get_month_data(
        self,
        month: str,
        area: Optional[str] = None,
        council: Optional[str] = None
    ) -> Dict:
        """
        Load noise monitoring data for a specific month

        Args:
            month: Month name (e.g., "December_2024")
            area: Geographic area filter (e.g., "North", "Central", "South")
            council: Council filter (e.g., "Birmingham", "Camden")

        Returns:
            Dictionary with aggregated noise data
        """
        cache_key = f"{month}_{area}_{council}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        month_folder = self.data_dir / f"hs2_monthly_monitoring_data_{month}"

        if not month_folder.exists():
            logger.warning(f"Month folder not found: {month_folder}")
            return self._empty_response()

        # Get all Excel files for this month
        excel_files = list(month_folder.glob("*.xlsx"))

        if not excel_files:
            logger.warning(f"No Excel files found in {month_folder}")
            return self._empty_response()

        # Filter files by area and council
        if area:
            area_lower = area.lower()
            excel_files = [f for f in excel_files if area_lower in f.name.lower()]

        if council:
            council_lower = council.lower()
            excel_files = [f for f in excel_files if council_lower in f.name.lower()]

        # Load and aggregate data from all files
        all_measurements = []
        all_locations = []

        for excel_file in excel_files:
            try:
                measurements, locations = self._load_excel_file(excel_file)
                all_measurements.extend(measurements)
                all_locations.extend(locations)
            except Exception as e:
                logger.error(f"Error loading {excel_file}: {e}")
                continue

        if not all_measurements:
            return self._empty_response()

        # Convert to DataFrame for easier aggregation
        df = pd.DataFrame(all_measurements)

        # Calculate aggregated statistics
        result = self._calculate_statistics(df, all_locations, month)

        # Cache the result
        self.cache[cache_key] = result

        return result

    def _load_excel_file(self, file_path: Path) -> tuple:
        """
        Load data from a single Excel file

        Returns:
            Tuple of (measurements list, locations list)
        """
        measurements = []
        locations = []

        # Extract metadata from filename
        filename = file_path.name.lower()

        # Determine area
        if 'areanorth' in filename:
            area = 'North'
        elif 'areacentral' in filename:
            area = 'Central'
        elif 'areasouth' in filename:
            area = 'South'
        else:
            area = 'Unknown'

        # Extract council name (between area and month)
        parts = filename.replace('.xlsx', '').split('_')
        council = None
        for i, part in enumerate(parts):
            if 'area' in part:
                if i + 1 < len(parts) and parts[i+1] not in ['december', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november']:
                    council = ' '.join(parts[i+1:]).split('_')[0]
                    council = council.replace('_', ' ').title()
                    break

        if not council:
            # Try alternative pattern
            for part in parts:
                if part not in ['hs2', 'noise', 'data', 'areanorth', 'areacentral', 'areasouth', 'december', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', '2024', '2025']:
                    council = part.replace('_', ' ').title()
                    break

        if not council:
            council = 'Unknown'

        # Read Excel file
        xl = pd.ExcelFile(file_path)

        # Process each sheet (except Metadata)
        for sheet_name in xl.sheet_names:
            if sheet_name.lower() == 'metadata':
                continue

            try:
                # Read sheet with header at row 2 (0-indexed)
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=2)

                # Skip if no data
                if len(df) == 0:
                    continue

                # Get column names from first row
                if df.iloc[0, 0] == 'Date/Time':
                    # First row contains headers, use it
                    df.columns = df.iloc[0].values
                    df = df.iloc[1:]  # Remove header row

                # Rename columns
                df.columns = ['timestamp', 'period', 'avg_noise', 'max_noise', 'background_noise']

                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

                # Convert noise levels to float
                df['avg_noise'] = pd.to_numeric(df['avg_noise'], errors='coerce')
                df['max_noise'] = pd.to_numeric(df['max_noise'], errors='coerce')
                df['background_noise'] = pd.to_numeric(df['background_noise'], errors='coerce')

                # Filter out invalid rows
                df = df.dropna(subset=['timestamp', 'avg_noise'])

                # Add metadata
                df['location_id'] = sheet_name
                df['area'] = area
                df['council'] = council

                # Add to measurements
                measurements.extend(df.to_dict('records'))

                # Add to locations (unique)
                location_entry = {
                    'location_id': sheet_name,
                    'area': area,
                    'council': council
                }
                if location_entry not in locations:
                    locations.append(location_entry)

            except Exception as e:
                logger.error(f"Error processing sheet {sheet_name} in {file_path}: {e}")
                continue

        return measurements, locations

    def _calculate_statistics(self, df: pd.DataFrame, locations: List[Dict], month: str) -> Dict:
        """Calculate aggregated statistics from measurements DataFrame"""

        # Compliance threshold (75 dB for HS2)
        NOISE_LIMIT = 75.0

        # Total measurements
        total_measurements = len(df)

        # Average noise level across all measurements
        avg_noise = df['avg_noise'].mean() if len(df) > 0 else 0.0

        # Violations (measurements above limit)
        violations = df[df['avg_noise'] > NOISE_LIMIT]
        violation_count = len(violations)

        # Compliance rate
        compliance_rate = ((total_measurements - violation_count) / total_measurements * 100) if total_measurements > 0 else 0.0

        # Time series data (hourly averages)
        df['hour'] = df['timestamp'].dt.floor('H')
        time_series = df.groupby('hour').agg({
            'avg_noise': 'mean',
            'max_noise': 'max'
        }).reset_index()

        time_series_data = [
            {
                'time': row['hour'].strftime('%Y-%m-%d %H:%M'),
                'avg_noise': round(row['avg_noise'], 2),
                'max_noise': round(row['max_noise'], 2),
                'limit': NOISE_LIMIT
            }
            for _, row in time_series.iterrows()
        ]

        # Geographic distribution (by location)
        geo_data = df.groupby('location_id').agg({
            'avg_noise': 'mean',
            'max_noise': 'max',
            'council': 'first',
            'area': 'first'
        }).reset_index()

        geographic_data = [
            {
                'location': row['location_id'],
                'council': row['council'],
                'area': row['area'],
                'noise_level': round(row['avg_noise'], 2),
                'max_noise': round(row['max_noise'], 2),
                'x': hash(row['location_id']) % 100,  # Mock coordinates
                'y': hash(row['council']) % 100
            }
            for _, row in geo_data.iterrows()
        ]

        # Noise by council
        council_data = df.groupby('council').agg({
            'avg_noise': 'mean',
            'max_noise': 'max'
        }).reset_index()

        council_stats = [
            {
                'council': row['council'],
                'avg_noise': round(row['avg_noise'], 2),
                'max_noise': round(row['max_noise'], 2)
            }
            for _, row in council_data.iterrows()
        ]

        # Violations list
        violations_list = [
            {
                'timestamp': row['timestamp'].strftime('%Y-%m-%d %H:%M'),
                'location': row['location_id'],
                'council': row['council'],
                'noise_level': round(row['avg_noise'], 2),
                'limit': NOISE_LIMIT,
                'excess': round(row['avg_noise'] - NOISE_LIMIT, 2)
            }
            for _, row in violations.head(20).iterrows()  # Limit to 20 violations
        ]

        # Available filters
        areas = df['area'].unique().tolist()
        councils = df['council'].unique().tolist()

        return {
            'month': month,
            'summary': {
                'total_measurements': int(total_measurements),
                'avg_noise': round(avg_noise, 2),
                'compliance_rate': round(compliance_rate, 2),
                'violations': int(violation_count)
            },
            'time_series': time_series_data,
            'geographic': geographic_data,
            'by_council': council_stats,
            'violations': violations_list,
            'filters': {
                'areas': areas,
                'councils': councils
            }
        }

    def _empty_response(self) -> Dict:
        """Return empty response structure"""
        return {
            'month': '',
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

# Global instance
noise_service = NoiseMonitoringService()
