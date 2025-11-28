"""
Script to load HS2 noise monitoring data from Excel files into PostgreSQL database
"""
import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.models.noise_monitoring import Base, NoiseMonitoringMeasurement, NoiseMonitoringLocation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database connection (use psycopg2 for sync operations)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/hs2_db")
# Replace postgresql+asyncpg with postgresql for sync operations
if "+asyncpg" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("+asyncpg", "")
NOISE_LIMIT = 75.0  # dB threshold for violations

def create_tables(engine):
    """Create database tables"""
    Base.metadata.create_all(engine)
    logger.info("Database tables created")

def extract_metadata_from_filename(filename: str) -> dict:
    """Extract area and council from filename"""
    filename_lower = filename.lower()

    # Determine area
    if 'areanorth' in filename_lower:
        area = 'North'
    elif 'areacentral' in filename_lower:
        area = 'Central'
    elif 'areasouth' in filename_lower:
        area = 'South'
    else:
        area = 'Unknown'

    # Extract council name
    parts = filename_lower.replace('.xlsx', '').split('_')
    council = None

    for i, part in enumerate(parts):
        if 'area' in part:
            # Get the next part(s) that aren't month names
            remaining = parts[i+1:]
            council_parts = []
            for p in remaining:
                if p not in ['december', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', '2024', '2025']:
                    council_parts.append(p)
            if council_parts:
                council = ' '.join(council_parts).title()
                break

    if not council:
        council = 'Unknown'

    return {'area': area, 'council': council}

def load_excel_file(file_path: Path, month: str, session) -> tuple:
    """
    Load noise data from a single Excel file

    Returns:
        Tuple of (measurements_loaded, locations_added)
    """
    logger.info(f"Loading {file_path.name}...")

    # Extract metadata
    metadata = extract_metadata_from_filename(file_path.name)
    area = metadata['area']
    council = metadata['council']

    measurements_loaded = 0
    locations_added = set()

    try:
        # Read Excel file
        xl = pd.ExcelFile(file_path)

        # Process each sheet (skip Metadata)
        for sheet_name in xl.sheet_names:
            if sheet_name.lower() == 'metadata':
                continue

            try:
                # Read sheet with header at row 2
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=2)

                if len(df) == 0:
                    continue

                # Check if first row contains headers
                if df.iloc[0, 0] == 'Date/Time':
                    df.columns = df.iloc[0].values
                    df = df.iloc[1:]

                # Rename columns to match our model
                df.columns = ['timestamp', 'period', 'avg_noise', 'max_noise', 'background_noise']

                # Convert data types
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df['avg_noise'] = pd.to_numeric(df['avg_noise'], errors='coerce')
                df['max_noise'] = pd.to_numeric(df['max_noise'], errors='coerce')
                df['background_noise'] = pd.to_numeric(df['background_noise'], errors='coerce')

                # Drop rows with invalid data
                df = df.dropna(subset=['timestamp', 'avg_noise'])

                if len(df) == 0:
                    continue

                # Add location if not exists
                location_id = sheet_name
                location = session.query(NoiseMonitoringLocation).filter_by(
                    location_id=location_id
                ).first()

                if not location:
                    location = NoiseMonitoringLocation(
                        location_id=location_id,
                        area=area,
                        council=council
                    )
                    session.add(location)
                    locations_added.add(location_id)

                # Bulk insert measurements
                measurements = []
                for _, row in df.iterrows():
                    measurement = NoiseMonitoringMeasurement(
                        timestamp=row['timestamp'],
                        month=month,
                        location_id=location_id,
                        area=area,
                        council=council,
                        period_hours=1.0,  # Assuming 1 hour periods
                        avg_noise_db=row['avg_noise'],
                        max_noise_db=row['max_noise'],
                        background_noise_db=row['background_noise'],
                        is_violation=1 if row['avg_noise'] > NOISE_LIMIT else 0,
                        source_file=file_path.name
                    )
                    measurements.append(measurement)

                session.bulk_save_objects(measurements)
                measurements_loaded += len(measurements)

                logger.info(f"  Sheet {sheet_name}: {len(measurements)} measurements")

            except Exception as e:
                logger.error(f"  Error processing sheet {sheet_name}: {e}")
                continue

        # Commit after each file
        session.commit()

    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        session.rollback()
        return 0, 0

    return measurements_loaded, len(locations_added)

def load_all_months(data_dir: str, session, months_to_load=None):
    """
    Load data from all available months

    Args:
        data_dir: Path to HS2 rawdata directory
        session: Database session
        months_to_load: List of specific months to load (None = all)
    """
    data_path = Path(data_dir)

    # Find all month folders
    month_folders = sorted(data_path.glob("hs2_monthly_monitoring_data_*"))

    if not month_folders:
        logger.error(f"No month folders found in {data_dir}")
        return

    logger.info(f"Found {len(month_folders)} month folders")

    total_measurements = 0
    total_locations = 0

    for month_folder in month_folders:
        # Extract month name
        month_name = month_folder.name.replace("hs2_monthly_monitoring_data_", "")

        # Check if we should load this month
        if months_to_load and month_name not in months_to_load:
            logger.info(f"Skipping {month_name} (not in specified months)")
            continue

        logger.info(f"\n{'='*80}")
        logger.info(f"Processing month: {month_name}")
        logger.info(f"{'='*80}")

        # Check if month already loaded
        existing = session.query(NoiseMonitoringMeasurement).filter_by(month=month_name).count()
        if existing > 0:
            logger.info(f"Month {month_name} already has {existing} measurements. Skipping.")
            continue

        # Get all Excel files
        excel_files = list(month_folder.glob("*.xlsx"))
        logger.info(f"Found {len(excel_files)} Excel files")

        month_measurements = 0
        month_locations = 0

        for excel_file in excel_files:
            measurements, locations = load_excel_file(excel_file, month_name, session)
            month_measurements += measurements
            month_locations += locations

        logger.info(f"Month {month_name} total: {month_measurements} measurements, {month_locations} new locations")
        total_measurements += month_measurements
        total_locations += month_locations

    logger.info(f"\n{'='*80}")
    logger.info(f"FINAL TOTALS")
    logger.info(f"{'='*80}")
    logger.info(f"Total measurements loaded: {total_measurements}")
    logger.info(f"Total locations added: {total_locations}")

def main():
    """Main entry point"""
    logger.info("Starting HS2 Noise Monitoring Data Loader")

    # Get data directory from environment or use default
    data_dir = os.getenv("HS2_DATA_DIR", "/datasets/hs2/rawdata")

    # Create database engine
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Create tables
        logger.info("Creating database tables...")
        create_tables(engine)

        # Load specific months or all
        months_to_load = None  # Set to ['December_2024', 'January_2025'] to load specific months

        # Load data
        load_all_months(data_dir, session, months_to_load)

        logger.info("\nData loading completed successfully!")

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    main()
