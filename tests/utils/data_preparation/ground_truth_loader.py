"""
Ground Truth Data Loader for University of Twente GPR Dataset.

This module provides functionality to load, parse, and standardize the ground truth
data from the University of Twente GPR dataset for validation testing.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re


logger = logging.getLogger(__name__)


class UtilityDiscipline(Enum):
    """Utility discipline classifications."""
    WATER = "water"
    SEWER = "sewer"
    ELECTRICITY = "electricity"
    TELECOMMUNICATIONS = "telecommunications"
    OIL_GAS_CHEMICALS = "oilGasChemicals"
    UNKNOWN = "unknown"


class UtilityMaterial(Enum):
    """Utility material classifications."""
    STEEL = "steel"
    POLYVINYLCHLORIDE = "polyVinylChloride"
    ASBESTOS_CEMENT = "asbestosCement"
    HIGH_DENSITY_POLYETHYLENE = "highDensityPolyEthylene"
    POLYETHYLENE = "polyEthylene"
    CONCRETE = "concrete"
    CAST_IRON = "castIron"
    PAPER_INSULATED_LEAD_COVERED = "paperInsulatedLeadCovered"
    FIBER_REINFORCED_PLASTIC = "fiberReinforcedPlastic"
    UNKNOWN = "unknown"


class WeatherCondition(Enum):
    """Weather condition classifications."""
    DRY = "Dry"
    RAINY = "Rainy"


class GroundCondition(Enum):
    """Ground condition classifications."""
    SANDY = "Sandy"
    CLAYEY = "Clayey"


@dataclass
class UtilityInfo:
    """Information about a single utility."""
    discipline: str
    material: Optional[str] = None
    diameter: Optional[float] = None
    additional_info: Optional[str] = None

    def __post_init__(self):
        """Post-initialization processing."""
        # Standardize discipline
        if self.discipline:
            self.discipline = self.discipline.strip().lower()

        # Standardize material
        if self.material:
            self.material = self.material.strip()

        # Convert diameter to float if possible
        if self.diameter and isinstance(self.diameter, str):
            try:
                self.diameter = float(self.diameter)
            except (ValueError, TypeError):
                self.diameter = None


@dataclass
class EnvironmentalConditions:
    """Environmental conditions for a survey location."""
    weather: str
    ground_condition: str
    ground_permittivity: float
    land_cover: str
    land_use: str
    terrain_levelling: str
    terrain_smoothness: str
    rubble_presence: bool = False
    tree_roots_presence: bool = False
    polluted_soil_presence: bool = False
    blast_furnace_slag_presence: bool = False

    def __post_init__(self):
        """Post-initialization processing."""
        # Convert string booleans to actual booleans
        bool_fields = ['rubble_presence', 'tree_roots_presence',
                      'polluted_soil_presence', 'blast_furnace_slag_presence']

        for field_name in bool_fields:
            value = getattr(self, field_name)
            if isinstance(value, str):
                setattr(self, field_name, value.lower() in ['yes', 'true', '1'])


@dataclass
class SurveyMetadata:
    """Survey metadata and context."""
    objective: str
    construction_workers: str
    accuracy_required: str
    complementary_works: str
    amount_of_utilities: int
    utility_crossing: bool
    utility_path_linear: bool

    def __post_init__(self):
        """Post-initialization processing."""
        # Convert utility crossing and path linear to boolean
        if isinstance(self.utility_crossing, str):
            self.utility_crossing = self.utility_crossing.lower() in ['yes', 'true']
        if isinstance(self.utility_path_linear, str):
            self.utility_path_linear = self.utility_path_linear.lower() in ['yes', 'true']


@dataclass
class GroundTruthLocation:
    """Complete ground truth data for a survey location."""
    location_id: str
    utilities: List[UtilityInfo] = field(default_factory=list)
    environmental_conditions: Optional[EnvironmentalConditions] = None
    survey_metadata: Optional[SurveyMetadata] = None

    @property
    def utility_count(self) -> int:
        """Number of utilities at this location."""
        return len(self.utilities)

    @property
    def has_utilities(self) -> bool:
        """Whether this location has any utilities."""
        return self.utility_count > 0

    @property
    def utility_disciplines(self) -> List[str]:
        """List of unique utility disciplines at this location."""
        return list(set(util.discipline for util in self.utilities if util.discipline))

    @property
    def utility_materials(self) -> List[str]:
        """List of unique utility materials at this location."""
        return list(set(util.material for util in self.utilities if util.material))


class TwenteDataLoader:
    """Loader for University of Twente GPR ground truth dataset."""

    def __init__(self, metadata_path: Path):
        """
        Initialize the data loader.

        Args:
            metadata_path: Path to the Twente metadata CSV file
        """
        self.metadata_path = Path(metadata_path)
        self.raw_data: Optional[pd.DataFrame] = None
        self.locations: Dict[str, GroundTruthLocation] = {}

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

    def load_data(self) -> Dict[str, GroundTruthLocation]:
        """
        Load and parse the ground truth data.

        Returns:
            Dictionary mapping location IDs to GroundTruthLocation objects
        """
        logger.info(f"Loading Twente ground truth data from {self.metadata_path}")

        try:
            # Read CSV with semicolon separator
            self.raw_data = pd.read_csv(self.metadata_path, sep=';', encoding='utf-8-sig')
            logger.info(f"Loaded {len(self.raw_data)} records from metadata file")

            # Process each location
            for _, row in self.raw_data.iterrows():
                location = self._parse_location_record(row)
                if location:
                    self.locations[location.location_id] = location

            logger.info(f"Successfully parsed {len(self.locations)} locations")
            return self.locations

        except Exception as e:
            logger.error(f"Error loading Twente data: {e}")
            raise

    def _parse_location_record(self, row: pd.Series) -> Optional[GroundTruthLocation]:
        """
        Parse a single location record from the CSV.

        Args:
            row: Pandas Series representing a CSV row

        Returns:
            GroundTruthLocation object or None if parsing fails
        """
        try:
            location_id = str(row.get('LocationID', '')).strip()
            if not location_id:
                return None

            # Parse utilities
            utilities = self._parse_utilities(row)

            # Parse environmental conditions
            environmental_conditions = self._parse_environmental_conditions(row)

            # Parse survey metadata
            survey_metadata = self._parse_survey_metadata(row)

            return GroundTruthLocation(
                location_id=location_id,
                utilities=utilities,
                environmental_conditions=environmental_conditions,
                survey_metadata=survey_metadata
            )

        except Exception as e:
            logger.warning(f"Error parsing location record: {e}")
            return None

    def _parse_utilities(self, row: pd.Series) -> List[UtilityInfo]:
        """Parse utility information from a row."""
        utilities = []

        # Get utility disciplines (can be multi-line)
        disciplines_raw = str(row.get('Utility discipline', '')).strip()
        materials_raw = str(row.get('Utility material', '')).strip()
        diameters_raw = str(row.get('Utility diameter', '')).strip()
        additional_info_raw = str(row.get('Additional utility information', '')).strip()

        # Split multi-line values
        disciplines = self._split_multiline_value(disciplines_raw)
        materials = self._split_multiline_value(materials_raw)
        diameters = self._split_multiline_value(diameters_raw)
        additional_infos = self._split_multiline_value(additional_info_raw)

        # Create utility objects
        max_length = max(len(disciplines), len(materials), len(diameters), len(additional_infos))

        for i in range(max_length):
            discipline = disciplines[i] if i < len(disciplines) else None
            material = materials[i] if i < len(materials) else None
            diameter = diameters[i] if i < len(diameters) else None
            additional_info = additional_infos[i] if i < len(additional_infos) else None

            # Skip empty entries
            if not discipline and not material and not diameter:
                continue

            # Convert diameter to float
            diameter_float = None
            if diameter and diameter.strip():
                try:
                    diameter_float = float(diameter.strip())
                except (ValueError, TypeError):
                    pass

            utilities.append(UtilityInfo(
                discipline=discipline or 'unknown',
                material=material,
                diameter=diameter_float,
                additional_info=additional_info
            ))

        return utilities

    def _parse_environmental_conditions(self, row: pd.Series) -> EnvironmentalConditions:
        """Parse environmental conditions from a row."""
        return EnvironmentalConditions(
            weather=str(row.get('Weather condition', '')).strip(),
            ground_condition=str(row.get('Ground condition', '')).strip(),
            ground_permittivity=self._safe_float_convert(row.get('Ground relative permittivity')),
            land_cover=str(row.get('Land cover', '')).strip(),
            land_use=str(row.get('Land use', '')).strip(),
            terrain_levelling=str(row.get('Terrain levelling', '')).strip(),
            terrain_smoothness=str(row.get('Terrain smoothness', '')).strip(),
            rubble_presence=self._safe_bool_convert(row.get('Rubble presence')),
            tree_roots_presence=self._safe_bool_convert(row.get('Tree roots presence')),
            polluted_soil_presence=self._safe_bool_convert(row.get('Polluted soil presence')),
            blast_furnace_slag_presence=self._safe_bool_convert(row.get('Blast-furnace slag presence'))
        )

    def _parse_survey_metadata(self, row: pd.Series) -> SurveyMetadata:
        """Parse survey metadata from a row."""
        return SurveyMetadata(
            objective=str(row.get('Utility surveying objective', '')).strip(),
            construction_workers=str(row.get('Construction workers', '')).strip(),
            accuracy_required=str(row.get('Exact location accuracy required', '')).strip(),
            complementary_works=str(row.get('Complementary works', '')).strip(),
            amount_of_utilities=self._safe_int_convert(row.get('Amount of utilities')),
            utility_crossing=self._safe_bool_convert(row.get('Utility crossing')),
            utility_path_linear=self._safe_bool_convert(row.get('Utility path linear'))
        )

    def _split_multiline_value(self, value: str) -> List[str]:
        """Split a multi-line value into individual items."""
        if not value or value in ['nan', 'None', '']:
            return []

        # Split by newlines and clean up
        items = [item.strip() for item in value.split('\n') if item.strip()]
        return [item for item in items if item and item != 'nan']

    def _safe_float_convert(self, value: Any) -> float:
        """Safely convert a value to float."""
        if pd.isna(value) or value == '' or value is None:
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def _safe_int_convert(self, value: Any) -> int:
        """Safely convert a value to int."""
        if pd.isna(value) or value == '' or value is None:
            return 0
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return 0

    def _safe_bool_convert(self, value: Any) -> bool:
        """Safely convert a value to bool."""
        if pd.isna(value) or value == '' or value is None:
            return False
        if isinstance(value, str):
            return value.strip().lower() in ['yes', 'true', '1']
        return bool(value)

    def get_location(self, location_id: str) -> Optional[GroundTruthLocation]:
        """Get ground truth data for a specific location."""
        return self.locations.get(location_id)

    def get_locations_by_criteria(self, **criteria) -> List[GroundTruthLocation]:
        """
        Get locations matching specific criteria.

        Args:
            **criteria: Filtering criteria (e.g., weather='Dry', ground_condition='Sandy')

        Returns:
            List of matching GroundTruthLocation objects
        """
        matching_locations = []

        for location in self.locations.values():
            matches = True

            for key, value in criteria.items():
                if hasattr(location.environmental_conditions, key):
                    if getattr(location.environmental_conditions, key) != value:
                        matches = False
                        break
                elif hasattr(location.survey_metadata, key):
                    if getattr(location.survey_metadata, key) != value:
                        matches = False
                        break

            if matches:
                matching_locations.append(location)

        return matching_locations

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded ground truth data."""
        if not self.locations:
            return {}

        stats = {
            'total_locations': len(self.locations),
            'locations_with_utilities': sum(1 for loc in self.locations.values() if loc.has_utilities),
            'total_utilities': sum(loc.utility_count for loc in self.locations.values()),
            'weather_conditions': {},
            'ground_conditions': {},
            'utility_disciplines': {},
            'utility_materials': {},
        }

        # Count occurrences
        for location in self.locations.values():
            if location.environmental_conditions:
                weather = location.environmental_conditions.weather
                ground = location.environmental_conditions.ground_condition

                stats['weather_conditions'][weather] = stats['weather_conditions'].get(weather, 0) + 1
                stats['ground_conditions'][ground] = stats['ground_conditions'].get(ground, 0) + 1

            for discipline in location.utility_disciplines:
                stats['utility_disciplines'][discipline] = stats['utility_disciplines'].get(discipline, 0) + 1

            for material in location.utility_materials:
                stats['utility_materials'][material] = stats['utility_materials'].get(material, 0) + 1

        return stats

    def export_processed_data(self, output_path: Path) -> None:
        """Export processed ground truth data to JSON format."""
        if not self.locations:
            raise ValueError("No data loaded to export")

        export_data = {}
        for location_id, location in self.locations.items():
            export_data[location_id] = {
                'location_id': location.location_id,
                'utilities': [
                    {
                        'discipline': util.discipline,
                        'material': util.material,
                        'diameter': util.diameter,
                        'additional_info': util.additional_info
                    }
                    for util in location.utilities
                ],
                'environmental_conditions': {
                    'weather': location.environmental_conditions.weather,
                    'ground_condition': location.environmental_conditions.ground_condition,
                    'ground_permittivity': location.environmental_conditions.ground_permittivity,
                    'land_cover': location.environmental_conditions.land_cover,
                    'land_use': location.environmental_conditions.land_use,
                    'terrain_levelling': location.environmental_conditions.terrain_levelling,
                    'terrain_smoothness': location.environmental_conditions.terrain_smoothness,
                    'rubble_presence': location.environmental_conditions.rubble_presence,
                    'tree_roots_presence': location.environmental_conditions.tree_roots_presence,
                    'polluted_soil_presence': location.environmental_conditions.polluted_soil_presence,
                    'blast_furnace_slag_presence': location.environmental_conditions.blast_furnace_slag_presence
                } if location.environmental_conditions else None,
                'survey_metadata': {
                    'objective': location.survey_metadata.objective,
                    'construction_workers': location.survey_metadata.construction_workers,
                    'accuracy_required': location.survey_metadata.accuracy_required,
                    'complementary_works': location.survey_metadata.complementary_works,
                    'amount_of_utilities': location.survey_metadata.amount_of_utilities,
                    'utility_crossing': location.survey_metadata.utility_crossing,
                    'utility_path_linear': location.survey_metadata.utility_path_linear
                } if location.survey_metadata else None
            }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported ground truth data to {output_path}")


def create_ground_truth_loader(metadata_path: str) -> TwenteDataLoader:
    """
    Factory function to create a ground truth data loader.

    Args:
        metadata_path: Path to the Twente metadata CSV file

    Returns:
        Configured TwenteDataLoader instance
    """
    return TwenteDataLoader(Path(metadata_path))