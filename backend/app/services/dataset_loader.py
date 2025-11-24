"""
Dataset loading services
========================

Services for loading and processing GPR datasets including
Twente GPR data and Mojahid images with comprehensive error handling.
"""

import asyncio
import csv
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, AsyncGenerator
import aiofiles
import pandas as pd

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from loguru import logger

from ..core.config import settings
from ..core.logging_config import LoggerMixin
from ..models.gpr_data import GPRSurvey, GPRScan
from ..models.environmental import EnvironmentalData
from ..models.validation import GroundTruthData
from ..schemas.dataset import TwenteDatasetStatus, MojahidDatasetStatus, DatasetStatus


class TwenteDatasetLoader(LoggerMixin):
    """Service for loading University of Twente GPR dataset."""

    def __init__(self):
        super().__init__()
        self.dataset_path = settings.GPR_TWENTE_PATH
        self.metadata_file = self.dataset_path / "Metadata.csv"

    async def get_processing_status(self, db: AsyncSession) -> TwenteDatasetStatus:
        """Get current processing status of Twente dataset."""
        self.log_operation_start("get_twente_status")

        try:
            # Count ZIP files
            zip_files = settings.get_twente_zip_files()
            total_zip_files = len(zip_files)

            # Count processed surveys and scans
            surveys_result = await db.execute(
                select(func.count(GPRSurvey.id)).where(
                    GPRSurvey.survey_name.like("Twente_%")
                )
            )
            processed_surveys = surveys_result.scalar() or 0

            scans_result = await db.execute(
                select(func.count(GPRScan.id)).join(GPRSurvey).where(
                    GPRSurvey.survey_name.like("Twente_%")
                )
            )
            processed_scans = scans_result.scalar() or 0

            # Count environmental data records
            env_result = await db.execute(
                select(func.count(EnvironmentalData.id)).join(GPRSurvey).where(
                    GPRSurvey.survey_name.like("Twente_%")
                )
            )
            env_records = env_result.scalar() or 0

            # Check metadata loading status
            metadata_loaded = self.metadata_file.exists()
            metadata_records = 0
            if metadata_loaded:
                try:
                    df = pd.read_csv(self.metadata_file, sep=';')
                    metadata_records = len(df)
                except Exception as e:
                    self.logger.warning(f"Could not read metadata file: {e}")
                    metadata_loaded = False

            # Determine overall status
            if processed_surveys == 0:
                status = DatasetStatus.AVAILABLE
            elif processed_surveys < total_zip_files:
                status = DatasetStatus.LOADING
            else:
                status = DatasetStatus.COMPLETED

            # Calculate progress
            progress = (processed_surveys / total_zip_files * 100) if total_zip_files > 0 else 0

            self.log_operation_complete("get_twente_status", 0)

            return TwenteDatasetStatus(
                total_zip_files=total_zip_files,
                processed_zip_files=processed_surveys,
                total_scans=processed_scans,
                processed_scans=processed_scans,
                total_processed=processed_surveys + processed_scans + env_records,
                metadata_loaded=metadata_loaded,
                metadata_records=metadata_records,
                processing_status=status,
                error_count=0,  # TODO: Track errors in database
                last_processed_at=None,  # TODO: Get from database
                progress_percentage=progress,
                estimated_completion=None  # TODO: Calculate based on processing rate
            )

        except Exception as e:
            self.log_operation_error("get_twente_status", e)
            raise

    async def load_metadata_csv(self, db: AsyncSession) -> int:
        """Load metadata from Twente CSV file into environmental data."""
        self.log_operation_start("load_twente_metadata")

        try:
            if not self.metadata_file.exists():
                raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")

            # Read CSV with proper delimiter
            df = pd.read_csv(self.metadata_file, sep=';')
            self.log_data_processing("twente_metadata", len(df))

            records_created = 0

            for _, row in df.iterrows():
                try:
                    # Create or get survey
                    location_id = str(row.get('LocationID', '')).strip()
                    if not location_id:
                        continue

                    # Check if survey already exists
                    survey_result = await db.execute(
                        select(GPRSurvey).where(GPRSurvey.location_id == location_id)
                    )
                    survey = survey_result.scalar_one_or_none()

                    if not survey:
                        survey = GPRSurvey(
                            survey_name=f"Twente_{location_id}",
                            location_id=location_id,
                            survey_objective=str(row.get('Utility surveying objective', '')),
                            status="completed"
                        )
                        db.add(survey)
                        await db.flush()  # Get the ID

                    # Create environmental data
                    env_data = EnvironmentalData(
                        survey_id=survey.id,
                        land_use=str(row.get('Land use', '')),
                        land_cover=str(row.get('Land cover', '')),
                        land_type=str(row.get('Land type', '')),
                        ground_condition=str(row.get('Ground condition', '')),
                        ground_relative_permittivity=self._safe_float(row.get('Ground relative permittivity')),
                        relative_groundwater_level=str(row.get('Relative groundwater level', '')),
                        terrain_levelling=str(row.get('Terrain levelling', '')),
                        terrain_smoothness=str(row.get('Terrain smoothness', '')),
                        rubble_presence=self._safe_bool(row.get('Rubble presence')),
                        tree_roots_presence=self._safe_bool(row.get('Tree roots presence')),
                        polluted_soil_presence=self._safe_bool(row.get('Polluted soil presence')),
                        blast_furnace_slag_presence=self._safe_bool(row.get('Blast-furnace slag presence')),
                        weather_condition=str(row.get('Weather condition', '')),
                        amount_of_utilities=self._safe_int(row.get('Amount of utilities')),
                        utility_crossing=self._safe_bool(row.get('Utility crossing')),
                        utility_path_linear=self._safe_bool(row.get('Utility path linear')),
                        construction_workers=str(row.get('Construction workers', '')),
                        complementary_works=str(row.get('Complementary works', '')),
                        exact_location_accuracy_required=self._safe_bool(row.get('Exact location accuracy required')),
                        data_source="twente_metadata_csv"
                    )

                    db.add(env_data)
                    records_created += 1

                    # Create ground truth data if utility information exists
                    utility_disciplines = str(row.get('Utility discipline', ''))
                    utility_materials = str(row.get('Utility material', ''))
                    utility_diameters = str(row.get('Utility diameter', ''))

                    if utility_disciplines and utility_disciplines.strip():
                        # Parse multi-line utility data
                        disciplines = [d.strip() for d in utility_disciplines.split('\n') if d.strip()]
                        materials = [m.strip() for m in utility_materials.split('\n') if m.strip()]
                        diameters = [d.strip() for d in utility_diameters.split('\n') if d.strip()]

                        for i, discipline in enumerate(disciplines):
                            if discipline and discipline != 'unknown':
                                material = materials[i] if i < len(materials) else None
                                diameter_str = diameters[i] if i < len(diameters) else None
                                diameter = self._safe_float(diameter_str)

                                ground_truth = GroundTruthData(
                                    survey_id=survey.id,
                                    utility_id=f"{location_id}_{i+1}",
                                    utility_discipline=discipline,
                                    utility_material=material,
                                    utility_diameter=diameter,
                                    data_source="twente_metadata_csv",
                                    confidence_level="medium"
                                )
                                db.add(ground_truth)

                except Exception as e:
                    self.logger.error(f"Error processing metadata row {location_id}: {e}")
                    continue

            await db.commit()
            self.log_operation_complete("load_twente_metadata", 0, records_created=records_created)

            return records_created

        except Exception as e:
            await db.rollback()
            self.log_operation_error("load_twente_metadata", e)
            raise

    async def load_dataset_async(
        self,
        db: AsyncSession,
        batch_size: int = 5,
        force_reload: bool = False
    ) -> None:
        """Load Twente dataset asynchronously."""
        self.log_operation_start("load_twente_dataset", batch_size=batch_size, force_reload=force_reload)

        try:
            # First load metadata
            metadata_records = await self.load_metadata_csv(db)
            self.logger.info(f"Loaded {metadata_records} metadata records")

            # Get ZIP files to process
            zip_files = settings.get_twente_zip_files()
            self.log_data_processing("twente_zip_files", len(zip_files))

            # Process ZIP files in batches
            for i in range(0, len(zip_files), batch_size):
                batch = zip_files[i:i + batch_size]
                await self._process_zip_batch(db, batch, force_reload)

                self.log_batch_processing(
                    batch_id=f"twente_batch_{i//batch_size + 1}",
                    total_items=len(zip_files),
                    processed_items=min(i + batch_size, len(zip_files))
                )

            self.log_operation_complete("load_twente_dataset", 0)

        except Exception as e:
            self.log_operation_error("load_twente_dataset", e)
            raise

    async def _process_zip_batch(
        self,
        db: AsyncSession,
        zip_files: List[Path],
        force_reload: bool
    ) -> None:
        """Process a batch of ZIP files."""
        for zip_file in zip_files:
            try:
                await self._process_single_zip(db, zip_file, force_reload)
            except Exception as e:
                self.logger.error(f"Error processing {zip_file.name}: {e}")
                continue

    async def _process_single_zip(
        self,
        db: AsyncSession,
        zip_file: Path,
        force_reload: bool
    ) -> None:
        """Process a single ZIP file from Twente dataset."""
        self.log_file_processing(str(zip_file), zip_file.stat().st_size, 0)

        try:
            location_id = zip_file.stem  # e.g., "01" from "01.zip"

            # Check if already processed
            if not force_reload:
                existing = await db.execute(
                    select(GPRSurvey).where(GPRSurvey.location_id == location_id)
                )
                if existing.scalar_one_or_none():
                    self.logger.info(f"Skipping {zip_file.name} - already processed")
                    return

            # Extract and process ZIP contents
            with zipfile.ZipFile(zip_file, 'r') as zf:
                # List all files in the ZIP
                file_list = zf.namelist()
                gpr_files = [f for f in file_list if f.lower().endswith(('.dt1', '.hd', '.gps'))]

                self.logger.info(f"Processing {zip_file.name} with {len(gpr_files)} GPR files")

                # Create survey record
                survey = GPRSurvey(
                    survey_name=f"Twente_{location_id}",
                    location_id=location_id,
                    survey_objective="University of Twente GPR dataset",
                    status="completed"
                )
                db.add(survey)
                await db.flush()

                # Process each GPR file
                for gpr_file in gpr_files:
                    await self._process_gpr_file_from_zip(db, zf, gpr_file, survey.id)

                await db.commit()

        except Exception as e:
            await db.rollback()
            self.log_operation_error("process_single_zip", e, zip_file=str(zip_file))
            raise

    async def _process_gpr_file_from_zip(
        self,
        db: AsyncSession,
        zip_file: zipfile.ZipFile,
        file_path: str,
        survey_id: str
    ) -> None:
        """Process a GPR file from within a ZIP archive."""
        try:
            file_info = zip_file.getinfo(file_path)

            # Create scan record
            scan = GPRScan(
                survey_id=survey_id,
                scan_number=1,  # TODO: Extract from filename
                scan_name=Path(file_path).name,
                file_path=file_path,
                file_size_bytes=file_info.file_size,
                data_format=Path(file_path).suffix.upper(),
                is_processed=False,
                processing_status="extracted"
            )

            db.add(scan)

        except Exception as e:
            self.logger.error(f"Error processing GPR file {file_path}: {e}")
            raise

    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert value to float."""
        try:
            if pd.isna(value) or value == '' or value is None:
                return None
            return float(value)
        except (ValueError, TypeError):
            return None

    def _safe_int(self, value: Any) -> Optional[int]:
        """Safely convert value to int."""
        try:
            if pd.isna(value) or value == '' or value is None:
                return None
            return int(float(value))
        except (ValueError, TypeError):
            return None

    def _safe_bool(self, value: Any) -> Optional[bool]:
        """Safely convert value to bool."""
        try:
            if pd.isna(value) or value == '' or value is None:
                return None
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ('yes', 'true', '1', 'y')
            return bool(value)
        except (ValueError, TypeError):
            return None


class MojahidDatasetLoader(LoggerMixin):
    """Service for loading Mojahid GPR images dataset."""

    def __init__(self):
        super().__init__()
        self.dataset_path = settings.GPR_MOJAHID_PATH / "GPR_data"

    async def get_processing_status(self, db: AsyncSession) -> MojahidDatasetStatus:
        """Get current processing status of Mojahid dataset."""
        self.log_operation_start("get_mojahid_status")

        try:
            categories = settings.get_mojahid_categories()
            category_counts = {}
            total_images = 0

            # Count images in each category
            for category in categories:
                category_path = self.dataset_path / category
                if category_path.exists():
                    image_files = list(category_path.glob("*.jpg")) + list(category_path.glob("*.png"))
                    count = len(image_files)
                    category_counts[category] = count
                    total_images += count
                else:
                    category_counts[category] = 0

            # TODO: Query database for processed images
            processed_images = 0

            # Determine status
            if processed_images == 0:
                status = DatasetStatus.AVAILABLE
            elif processed_images < total_images:
                status = DatasetStatus.LOADING
            else:
                status = DatasetStatus.COMPLETED

            progress = (processed_images / total_images * 100) if total_images > 0 else 0

            self.log_operation_complete("get_mojahid_status", 0)

            return MojahidDatasetStatus(
                categories=categories,
                category_counts=category_counts,
                total_images=total_images,
                processed_images=processed_images,
                processing_status=status,
                error_count=0,
                last_processed_at=None,
                progress_percentage=progress,
                augmented_data_included=any("augmented" in cat for cat in categories)
            )

        except Exception as e:
            self.log_operation_error("get_mojahid_status", e)
            raise

    async def load_dataset_async(
        self,
        db: AsyncSession,
        categories: List[str],
        max_images_per_category: Optional[int] = None,
        force_reload: bool = False
    ) -> None:
        """Load Mojahid dataset asynchronously."""
        self.log_operation_start("load_mojahid_dataset", categories=categories)

        try:
            for category in categories:
                await self._process_category(db, category, max_images_per_category, force_reload)

            self.log_operation_complete("load_mojahid_dataset", 0)

        except Exception as e:
            self.log_operation_error("load_mojahid_dataset", e)
            raise

    async def _process_category(
        self,
        db: AsyncSession,
        category: str,
        max_images: Optional[int],
        force_reload: bool
    ) -> None:
        """Process images from a specific category."""
        category_path = self.dataset_path / category

        if not category_path.exists():
            self.logger.warning(f"Category path does not exist: {category_path}")
            return

        # Get image files
        image_files = list(category_path.glob("*.jpg")) + list(category_path.glob("*.png"))

        if max_images:
            image_files = image_files[:max_images]

        self.log_data_processing(f"mojahid_{category}", len(image_files))

        # TODO: Process each image file
        # This would involve:
        # 1. Creating image records in database
        # 2. Extracting features if needed
        # 3. Creating ground truth labels based on category

        for image_file in image_files:
            # Placeholder for image processing
            pass