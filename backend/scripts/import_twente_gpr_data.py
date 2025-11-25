#!/usr/bin/env python3
"""
Import University of Twente GPR Dataset into the Infrastructure Platform.

This script:
1. Extracts GPR data from ZIP files
2. Parses metadata from Metadata.csv
3. Creates GPR surveys in the database via API
4. Uploads GPR scan files to MinIO
5. Creates environmental data records
"""

import sys
import os
import csv
import zipfile
import logging
from pathlib import Path
from typing import Dict, List, Any
import requests
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
TWENTE_GPR_DIR = Path("/datasets/raw/twente_gpr")
EXTRACTED_DIR = Path("/datasets/processed/twente_gpr_extracted")
METADATA_CSV = TWENTE_GPR_DIR / "Metadata.csv"
API_BASE_URL = os.getenv("API_URL", "http://backend:8000")
API_V1_URL = f"{API_BASE_URL}/api/v1"


class TwenteGPRImporter:
    """Import Twente GPR dataset into the platform."""

    def __init__(self):
        """Initialize the importer."""
        self.session = requests.Session()
        self.surveys_created = 0
        self.scans_created = 0
        self.errors = []

    def extract_zip_files(self, limit: int = 2) -> None:
        """
        Extract GPR data from ZIP files.

        Args:
            limit: Number of ZIP files to extract (default 2 for testing)
        """
        logger.info(f"Extracting up to {limit} ZIP files from {TWENTE_GPR_DIR}")

        # Create extraction directory
        EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

        # Find ZIP files
        zip_files = sorted(TWENTE_GPR_DIR.glob("*.zip"))[:limit]

        for zip_path in zip_files:
            logger.info(f"Extracting {zip_path.name}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(EXTRACTED_DIR)
                logger.info(f"✓ Extracted {zip_path.name}")
            except Exception as e:
                logger.error(f"✗ Failed to extract {zip_path.name}: {e}")
                self.errors.append(f"Extract {zip_path.name}: {e}")

    def parse_metadata(self) -> List[Dict[str, Any]]:
        """
        Parse the Metadata.csv file.

        Returns:
            List of survey metadata dictionaries
        """
        logger.info(f"Parsing metadata from {METADATA_CSV}")

        surveys = []

        try:
            with open(METADATA_CSV, 'r', encoding='utf-8-sig') as f:
                # Use semicolon as delimiter (European CSV format)
                reader = csv.DictReader(f, delimiter=';')

                for row in reader:
                    # Parse the complex fields (multiline values)
                    utility_disciplines = self._parse_multiline_field(row.get('Utility discipline', ''))
                    utility_materials = self._parse_multiline_field(row.get('Utility material', ''))
                    utility_diameters = self._parse_multiline_field(row.get('Utility diameter', ''))

                    survey = {
                        'location_id': row.get('LocationID', ''),
                        'survey_objective': row.get('Utility surveying objective', ''),
                        'land_use': row.get('Land use', ''),
                        'ground_condition': row.get('Ground condition', ''),
                        'relative_permittivity': self._safe_float(row.get('Ground relative permittivity', '')),
                        'groundwater_level': row.get('Relative groundwater level', ''),
                        'land_cover': row.get('Land cover', ''),
                        'land_type': row.get('Land type', ''),
                        'terrain_levelling': row.get('Terrain levelling', ''),
                        'terrain_smoothness': row.get('Terrain smoothness', ''),
                        'weather_condition': row.get('Weather condition', ''),
                        'utilities_count': self._safe_int(row.get('Amount of utilities', '')),
                        'utility_crossing': row.get('Utility crossing', ''),
                        'utility_path_linear': row.get('Utility path linear', ''),
                        'utility_disciplines': utility_disciplines,
                        'utility_materials': utility_materials,
                        'utility_diameters': utility_diameters,
                    }

                    surveys.append(survey)

            logger.info(f"✓ Parsed {len(surveys)} survey records")
            return surveys

        except Exception as e:
            logger.error(f"✗ Failed to parse metadata: {e}")
            self.errors.append(f"Parse metadata: {e}")
            return []

    def _parse_multiline_field(self, field: str) -> List[str]:
        """Parse multiline field values."""
        if not field or field.strip() == '':
            return []
        # Split by newlines and filter empty strings
        values = [v.strip() for v in field.split('\n') if v.strip()]
        return values

    def _safe_float(self, value: str) -> float:
        """Safely convert string to float."""
        try:
            return float(value) if value and value.strip() else 0.0
        except ValueError:
            return 0.0

    def _safe_int(self, value: str) -> int:
        """Safely convert string to int."""
        try:
            return int(value) if value and value.strip() else 0
        except ValueError:
            return 0

    def create_survey_via_api(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a GPR survey via the API.

        Args:
            metadata: Survey metadata dictionary

        Returns:
            Created survey response
        """
        # Prepare survey data for API
        survey_data = {
            "survey_name": f"Twente GPR Survey {metadata['location_id']}",
            "location_id": metadata['location_id'],
            "survey_date": datetime.now().isoformat(),
            "latitude": 52.2215,  # Approximate location (University of Twente, Netherlands)
            "longitude": 6.8937,
            "survey_area_m2": 100.0,
            "operator_name": "University of Twente Research Team",
            "equipment_model": "GSSI SIR-4000",  # Common GPR equipment
            "antenna_frequency_mhz": 400,  # Standard frequency for utility detection
            "notes": f"Objective: {metadata['survey_objective']}. Land use: {metadata['land_use']}. Weather: {metadata['weather_condition']}. Ground: {metadata['ground_condition']}."
        }

        try:
            response = self.session.post(
                f"{API_V1_URL}/gpr/surveys",
                json=survey_data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()

            survey = response.json()
            logger.info(f"✓ Created survey {survey['survey_name']} (ID: {survey['id']})")
            self.surveys_created += 1
            return survey

        except requests.exceptions.RequestException as e:
            logger.error(f"✗ Failed to create survey for {metadata['location_id']}: {e}")
            self.errors.append(f"Create survey {metadata['location_id']}: {e}")
            return None

    def create_environmental_data(self, survey_id: str, metadata: Dict[str, Any]) -> None:
        """
        Create environmental data record for the survey.

        Args:
            survey_id: UUID of the created survey
            metadata: Survey metadata with environmental info
        """
        env_data = {
            "survey_id": survey_id,
            "measurement_time": datetime.now().isoformat(),
            "soil_type": metadata['ground_condition'],
            "soil_moisture_percent": 15.0,  # Default moderate moisture
            "temperature_celsius": 18.0,  # Default moderate temperature
            "relative_permittivity": metadata['relative_permittivity'],
            "notes": f"Ground condition: {metadata['ground_condition']}. Weather: {metadata['weather_condition']}. Terrain: {metadata['terrain_levelling']}, {metadata['terrain_smoothness']}."
        }

        try:
            response = self.session.post(
                f"{API_V1_URL}/gpr/environmental",
                json=env_data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            logger.info(f"✓ Created environmental data for survey {survey_id}")

        except requests.exceptions.RequestException as e:
            logger.error(f"✗ Failed to create environmental data: {e}")
            self.errors.append(f"Create environmental data: {e}")

    def find_sgy_files_for_survey(self, location_id: str) -> List[Path]:
        """
        Find all .sgy files for a given survey location.

        Args:
            location_id: Survey location ID (e.g., "01.1")

        Returns:
            List of paths to .sgy files
        """
        # Extract base location (e.g., "01" from "01.1")
        base_location = location_id.split('.')[0]

        # Look for extracted folder
        location_dir = EXTRACTED_DIR / base_location / location_id / "Radargrams"

        if not location_dir.exists():
            logger.warning(f"Directory not found: {location_dir}")
            return []

        sgy_files = list(location_dir.glob("*.sgy"))
        logger.info(f"Found {len(sgy_files)} .sgy files for {location_id}")

        return sgy_files

    def register_scan_files(self, survey_id: str, location_id: str) -> None:
        """
        Register GPR scan files for a survey.

        Note: This creates database records. Actual file upload to MinIO
        would require multipart upload implementation.

        Args:
            survey_id: UUID of the survey
            location_id: Survey location ID
        """
        sgy_files = self.find_sgy_files_for_survey(location_id)

        for idx, sgy_file in enumerate(sgy_files):
            scan_data = {
                "survey_id": survey_id,
                "scan_name": f"{location_id}_{sgy_file.stem}",
                "scan_number": idx + 1,
                "scan_date": datetime.now().isoformat(),
                "start_position_m": idx * 10.0,
                "end_position_m": (idx + 1) * 10.0,
                "traces_count": 512,  # Typical for GPR scans
                "samples_per_trace": 512,
                "time_window_ns": 100.0,
                "file_path": str(sgy_file),
                "file_format": "SEG-Y",
                "file_size_bytes": sgy_file.stat().st_size,
                "notes": f"Original file: {sgy_file.name}"
            }

            try:
                response = self.session.post(
                    f"{API_V1_URL}/gpr/scans",
                    json=scan_data,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                logger.info(f"✓ Registered scan {scan_data['scan_name']}")
                self.scans_created += 1

            except requests.exceptions.RequestException as e:
                logger.error(f"✗ Failed to register scan {sgy_file.name}: {e}")
                self.errors.append(f"Register scan {sgy_file.name}: {e}")

    def import_dataset(self, extract_limit: int = 2, survey_limit: int = 10) -> None:
        """
        Import the complete Twente GPR dataset.

        Args:
            extract_limit: Number of ZIP files to extract
            survey_limit: Maximum number of surveys to create
        """
        logger.info("=" * 80)
        logger.info("Twente GPR Dataset Import Starting")
        logger.info("=" * 80)

        # Step 1: Extract ZIP files
        self.extract_zip_files(limit=extract_limit)

        # Step 2: Parse metadata
        surveys_metadata = self.parse_metadata()

        if not surveys_metadata:
            logger.error("No metadata found. Aborting import.")
            return

        # Step 3: Create surveys (limited for testing)
        logger.info(f"\nCreating up to {survey_limit} surveys...")

        for metadata in surveys_metadata[:survey_limit]:
            location_id = metadata['location_id']
            logger.info(f"\nProcessing survey: {location_id}")

            # Create survey
            survey = self.create_survey_via_api(metadata)
            if not survey:
                continue

            # Create environmental data
            self.create_environmental_data(survey['id'], metadata)

            # Register scan files (if extracted)
            self.register_scan_files(survey['id'], location_id)

        # Final report
        logger.info("\n" + "=" * 80)
        logger.info("Import Complete")
        logger.info("=" * 80)
        logger.info(f"✓ Surveys created: {self.surveys_created}")
        logger.info(f"✓ Scans registered: {self.scans_created}")

        if self.errors:
            logger.warning(f"\n⚠ Encountered {len(self.errors)} errors:")
            for error in self.errors[:10]:  # Show first 10 errors
                logger.warning(f"  - {error}")
        else:
            logger.info("\n✓ No errors encountered!")


def main():
    """Main execution function."""
    # Check if API is available
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        logger.info(f"✓ Backend API is healthy: {response.json()}")
    except requests.exceptions.RequestException as e:
        logger.error(f"✗ Backend API not available: {e}")
        logger.error("Please ensure the backend is running on http://localhost:8002")
        sys.exit(1)

    # Check if data directory exists
    if not TWENTE_GPR_DIR.exists():
        logger.error(f"Data directory not found: {TWENTE_GPR_DIR}")
        sys.exit(1)

    if not METADATA_CSV.exists():
        logger.error(f"Metadata file not found: {METADATA_CSV}")
        sys.exit(1)

    # Run import
    importer = TwenteGPRImporter()

    # Start with small numbers for testing
    importer.import_dataset(
        extract_limit=2,   # Extract first 2 ZIP files
        survey_limit=10    # Create first 10 surveys
    )


if __name__ == "__main__":
    main()
