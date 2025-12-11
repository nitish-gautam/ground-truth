"""
BIM/IFC Processing Service
==========================

Service for importing, validating, and extracting data from IFC (Industry Foundation Classes) files.

Capabilities:
- Import IFC 4.0 and IFC 4.3 files
- Validate IFC schema compliance
- Extract building elements and properties
- Check IFC 4.3 infrastructure features (critical for HS2)
- Calculate quantities from BIM models
- Parse architectural scan data

IFC 4.3 Infrastructure Support (HS2):
- IfcAlignment (railway alignment)
- IfcBridge (viaducts)
- IfcTunnel (tunnel sections)
- IfcEarthworksCut/Fill (excavations)
"""

import logging
import json
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID, uuid4
from sqlalchemy.orm import Session
from datetime import datetime
from pathlib import Path
import base64
import io

from app.models.bim import (
    BIMTestModel,
    ArchitecturalScan,
    BIMElement,
    IFCVersion,
    BIMPurpose,
    ComplexityLevel
)

logger = logging.getLogger(__name__)

# Try to import ifcopenshell for IFC parsing
try:
    import ifcopenshell
    import ifcopenshell.util.element
    import ifcopenshell.util.shape
    IFC_AVAILABLE = True
    logger.info("ifcopenshell available for IFC processing")
except ImportError:
    IFC_AVAILABLE = False
    logger.warning("ifcopenshell not available - IFC processing will be limited")


class BIMProcessor:
    """
    Main BIM/IFC processing service.

    Handles IFC file import, validation, and element extraction.
    """

    def __init__(self, db: Session):
        self.db = db
        logger.info("BIMProcessor initialized")

    async def import_ifc_file(
        self,
        file_path: Optional[str] = None,
        file_data: Optional[str] = None,
        model_name: Optional[str] = None,
        extract_geometry: bool = True,
        extract_properties: bool = True,
        validate_schema: bool = True
    ) -> Dict[str, Any]:
        """
        Import IFC file and extract metadata.

        Args:
            file_path: Path to IFC file
            file_data: Base64 encoded IFC file content
            model_name: Name for the imported model
            extract_geometry: Extract geometric data
            extract_properties: Extract element properties
            validate_schema: Validate IFC schema compliance

        Returns:
            Dictionary with import results and model metadata

        Raises:
            ValueError: If neither file_path nor file_data provided
            RuntimeError: If IFC parsing fails
        """
        logger.info(
            f"Importing IFC file: path={file_path}, "
            f"geometry={extract_geometry}, properties={extract_properties}, "
            f"validate={validate_schema}"
        )

        if not IFC_AVAILABLE:
            logger.error("ifcopenshell not available - cannot import IFC file")
            raise RuntimeError("ifcopenshell not available - IFC import not supported")

        # Load IFC file
        if file_path:
            ifc_file = ifcopenshell.open(file_path)
            logger.info(f"IFC file loaded from path: {file_path}")
        elif file_data:
            # Decode base64 and load
            file_bytes = base64.b64decode(file_data)
            ifc_file = ifcopenshell.file.from_string(file_bytes.decode('utf-8'))
            logger.info("IFC file loaded from base64 data")
        else:
            raise ValueError("Either file_path or file_data must be provided")

        # Extract IFC metadata
        schema_version = ifc_file.schema
        logger.info(f"IFC schema version: {schema_version}")

        # Map schema version to enum
        if "4.3" in schema_version:
            ifc_version = IFCVersion.IFC_4_3
        elif "4" in schema_version:
            ifc_version = IFCVersion.IFC_4
        else:
            ifc_version = IFCVersion.IFC_2X3

        # Get project information
        project = ifc_file.by_type("IfcProject")
        project_name = project[0].Name if project else "Unknown"
        project_description = project[0].Description if project and project[0].Description else None

        logger.info(f"IFC Project: {project_name}")

        # Count elements by type
        element_counts = {}
        total_elements = 0

        for ifc_type in ["IfcWall", "IfcSlab", "IfcColumn", "IfcBeam", "IfcDoor",
                         "IfcWindow", "IfcRoof", "IfcStair", "IfcSpace"]:
            elements = ifc_file.by_type(ifc_type)
            if elements:
                element_counts[ifc_type] = len(elements)
                total_elements += len(elements)

        logger.info(f"Total elements: {total_elements}")
        logger.debug(f"Element counts: {element_counts}")

        # Check for IFC 4.3 infrastructure features (CRITICAL for HS2)
        has_alignment = len(ifc_file.by_type("IfcAlignment")) > 0
        has_bridge = len(ifc_file.by_type("IfcBridge")) > 0
        has_tunnel = len(ifc_file.by_type("IfcTunnel")) > 0
        has_earthworks = (
            len(ifc_file.by_type("IfcEarthworksCut")) > 0 or
            len(ifc_file.by_type("IfcEarthworksFill")) > 0
        )

        logger.info(
            f"IFC 4.3 Infrastructure features: alignment={has_alignment}, "
            f"bridge={has_bridge}, tunnel={has_tunnel}, earthworks={has_earthworks}"
        )

        # Validate schema if requested
        validation_errors = []
        validation_warnings = []

        if validate_schema:
            logger.info("Validating IFC schema")
            validation_result = await self.validate_ifc_schema(ifc_file)
            validation_errors = validation_result.get("errors", [])
            validation_warnings = validation_result.get("warnings", [])

        # Create BIMTestModel record
        model_record = BIMTestModel(
            model_name=model_name or project_name or f"IFC_Import_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            description=project_description,
            file_path=file_path,
            ifc_version=ifc_version,
            num_elements=total_elements,
            file_size_kb=Path(file_path).stat().st_size / 1024 if file_path else None,
            has_alignment=has_alignment,
            has_bridge=has_bridge,
            has_tunnel=has_tunnel,
            has_earthworks=has_earthworks,
            element_counts=element_counts,
            import_date=datetime.utcnow(),
            metadata={
                "project_name": project_name,
                "schema_version": schema_version,
                "validation_errors": validation_errors,
                "validation_warnings": validation_warnings
            }
        )

        self.db.add(model_record)
        self.db.commit()
        self.db.refresh(model_record)

        logger.info(f"BIM model imported: {model_record.id}")

        result = {
            "model_id": str(model_record.id),
            "model_name": model_record.model_name,
            "ifc_version": ifc_version.value,
            "schema_version": schema_version,
            "project_name": project_name,
            "total_elements": total_elements,
            "element_counts": element_counts,
            "ifc_43_features": {
                "has_alignment": has_alignment,
                "has_bridge": has_bridge,
                "has_tunnel": has_tunnel,
                "has_earthworks": has_earthworks
            },
            "validation": {
                "errors": validation_errors,
                "warnings": validation_warnings,
                "is_valid": len(validation_errors) == 0
            }
        }

        # Extract elements if requested
        if extract_properties:
            logger.info("Extracting element properties")
            # This could be done in background for large models
            # For now, just indicate that extraction is available
            result["properties_extracted"] = False
            result["note"] = "Use extract_elements endpoint for detailed element extraction"

        return result

    async def validate_ifc_schema(
        self,
        ifc_file: Any
    ) -> Dict[str, Any]:
        """
        Validate IFC schema compliance.

        Args:
            ifc_file: ifcopenshell IFC file object

        Returns:
            Dictionary with validation results (errors and warnings)
        """
        logger.info("Validating IFC schema compliance")

        errors = []
        warnings = []

        # Basic validation checks
        try:
            # Check for required project elements
            projects = ifc_file.by_type("IfcProject")
            if not projects:
                errors.append("Missing required IfcProject element")
            elif len(projects) > 1:
                warnings.append(f"Multiple IfcProject elements found ({len(projects)})")

            # Check for geometric representation contexts
            contexts = ifc_file.by_type("IfcGeometricRepresentationContext")
            if not contexts:
                warnings.append("No geometric representation contexts found")

            # Check for units
            units = ifc_file.by_type("IfcUnitAssignment")
            if not units:
                warnings.append("No unit assignments found")

            # Check for orphaned elements (elements without spatial structure)
            all_products = ifc_file.by_type("IfcProduct")
            spatial_structure = ifc_file.by_type("IfcSpatialStructureElement")

            if all_products and not spatial_structure:
                warnings.append(
                    f"Found {len(all_products)} products but no spatial structure"
                )

            logger.info(
                f"Validation complete: {len(errors)} errors, {len(warnings)} warnings"
            )

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            errors.append(f"Validation exception: {str(e)}")

        return {
            "errors": errors,
            "warnings": warnings,
            "is_valid": len(errors) == 0
        }

    async def extract_elements(
        self,
        bim_model_id: UUID,
        ifc_types: Optional[List[str]] = None,
        building_storey: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Extract BIM elements from model.

        Args:
            bim_model_id: UUID of BIM model
            ifc_types: Filter by IFC types (e.g., ["IfcWall", "IfcSlab"])
            building_storey: Filter by building storey
            limit: Maximum number of elements to return

        Returns:
            List of element dictionaries

        Raises:
            ValueError: If model not found
            RuntimeError: If IFC file not accessible
        """
        logger.info(
            f"Extracting elements from model {bim_model_id}, "
            f"types={ifc_types}, storey={building_storey}, limit={limit}"
        )

        # Load model from database
        model = self.db.query(BIMTestModel).filter(
            BIMTestModel.id == bim_model_id
        ).first()

        if not model:
            logger.error(f"BIM model not found: {bim_model_id}")
            raise ValueError(f"BIM model not found: {bim_model_id}")

        if not model.file_path or not Path(model.file_path).exists():
            logger.error(f"IFC file not accessible: {model.file_path}")
            raise RuntimeError(f"IFC file not accessible: {model.file_path}")

        if not IFC_AVAILABLE:
            logger.error("ifcopenshell not available")
            raise RuntimeError("ifcopenshell not available")

        # Load IFC file
        ifc_file = ifcopenshell.open(model.file_path)
        logger.info(f"IFC file loaded: {model.file_path}")

        # Query elements
        elements = []

        # Get element types to extract
        if ifc_types:
            target_types = ifc_types
        else:
            # Default to common building elements
            target_types = ["IfcWall", "IfcSlab", "IfcColumn", "IfcBeam",
                           "IfcDoor", "IfcWindow"]

        for ifc_type in target_types:
            ifc_elements = ifc_file.by_type(ifc_type)

            for elem in ifc_elements[:limit]:  # Apply limit
                # Extract basic properties
                element_data = {
                    "ifc_type": ifc_type,
                    "global_id": elem.GlobalId,
                    "name": elem.Name if hasattr(elem, 'Name') else None,
                    "description": elem.Description if hasattr(elem, 'Description') else None,
                    "tag": elem.Tag if hasattr(elem, 'Tag') else None
                }

                # Extract properties
                if hasattr(elem, 'IsDefinedBy'):
                    properties = self._extract_element_properties(elem)
                    element_data["properties"] = properties

                # Extract quantities
                if hasattr(elem, 'IsDefinedBy'):
                    quantities = self._extract_element_quantities(elem)
                    element_data["quantities"] = quantities

                elements.append(element_data)

                if len(elements) >= limit:
                    break

            if len(elements) >= limit:
                break

        logger.info(f"Extracted {len(elements)} elements from model")

        return elements

    async def check_ifc_43_features(
        self,
        bim_model_id: UUID
    ) -> Dict[str, Any]:
        """
        Check for IFC 4.3 infrastructure features (critical for HS2).

        Args:
            bim_model_id: UUID of BIM model

        Returns:
            Dictionary with infrastructure feature availability

        Raises:
            ValueError: If model not found
        """
        logger.info(f"Checking IFC 4.3 features for model {bim_model_id}")

        # Load model from database
        model = self.db.query(BIMTestModel).filter(
            BIMTestModel.id == bim_model_id
        ).first()

        if not model:
            logger.error(f"BIM model not found: {bim_model_id}")
            raise ValueError(f"BIM model not found: {bim_model_id}")

        # Check if IFC 4.3
        is_ifc_43 = model.ifc_version == IFCVersion.IFC_4_3

        result = {
            "model_id": str(model.id),
            "model_name": model.model_name,
            "ifc_version": model.ifc_version.value,
            "is_ifc_43": is_ifc_43,
            "infrastructure_features": {
                "alignment": {
                    "available": model.has_alignment,
                    "description": "Railway/road alignment geometry (IfcAlignment)"
                },
                "bridge": {
                    "available": model.has_bridge,
                    "description": "Bridge/viaduct structures (IfcBridge)"
                },
                "tunnel": {
                    "available": model.has_tunnel,
                    "description": "Tunnel sections (IfcTunnel)"
                },
                "earthworks": {
                    "available": model.has_earthworks,
                    "description": "Excavation/fill operations (IfcEarthworksCut/Fill)"
                }
            },
            "feature_summary": {
                "total_features": sum([
                    model.has_alignment,
                    model.has_bridge,
                    model.has_tunnel,
                    model.has_earthworks
                ]),
                "is_infrastructure_model": any([
                    model.has_alignment,
                    model.has_bridge,
                    model.has_tunnel,
                    model.has_earthworks
                ])
            }
        }

        logger.info(
            f"IFC 4.3 feature check complete: "
            f"{result['feature_summary']['total_features']} features found"
        )

        return result

    async def calculate_quantities(
        self,
        bim_model_id: UUID,
        element_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculate quantities from BIM model.

        Args:
            bim_model_id: UUID of BIM model
            element_types: Filter by element types

        Returns:
            Dictionary with quantity takeoffs

        Raises:
            ValueError: If model not found
        """
        logger.info(f"Calculating quantities for model {bim_model_id}")

        # Load model
        model = self.db.query(BIMTestModel).filter(
            BIMTestModel.id == bim_model_id
        ).first()

        if not model:
            logger.error(f"BIM model not found: {bim_model_id}")
            raise ValueError(f"BIM model not found: {bim_model_id}")

        if not model.file_path or not Path(model.file_path).exists():
            logger.error(f"IFC file not accessible: {model.file_path}")
            raise RuntimeError(f"IFC file not accessible: {model.file_path}")

        if not IFC_AVAILABLE:
            logger.error("ifcopenshell not available")
            raise RuntimeError("ifcopenshell not available")

        # Load IFC file
        ifc_file = ifcopenshell.open(model.file_path)

        # Calculate quantities by element type
        quantities = {}
        target_types = element_types or ["IfcWall", "IfcSlab", "IfcColumn", "IfcBeam"]

        for ifc_type in target_types:
            elements = ifc_file.by_type(ifc_type)

            type_quantities = {
                "count": len(elements),
                "total_volume": 0.0,
                "total_area": 0.0,
                "total_length": 0.0
            }

            for elem in elements:
                # Extract quantities from property sets
                elem_quantities = self._extract_element_quantities(elem)

                if "Volume" in elem_quantities:
                    type_quantities["total_volume"] += elem_quantities["Volume"]
                if "GrossArea" in elem_quantities:
                    type_quantities["total_area"] += elem_quantities["GrossArea"]
                if "Length" in elem_quantities:
                    type_quantities["total_length"] += elem_quantities["Length"]

            quantities[ifc_type] = type_quantities

        logger.info(f"Quantities calculated for {len(quantities)} element types")

        return {
            "model_id": str(model.id),
            "model_name": model.model_name,
            "quantities_by_type": quantities,
            "summary": {
                "total_elements": sum(q["count"] for q in quantities.values()),
                "total_volume_m3": sum(q["total_volume"] for q in quantities.values()),
                "total_area_m2": sum(q["total_area"] for q in quantities.values()),
                "total_length_m": sum(q["total_length"] for q in quantities.values())
            }
        }

    async def parse_architectural_scan(
        self,
        scan_id: UUID
    ) -> Dict[str, Any]:
        """
        Parse architectural scan data from ArchScanLib.

        Args:
            scan_id: UUID of architectural scan

        Returns:
            Dictionary with parsed scan data

        Raises:
            ValueError: If scan not found
        """
        logger.info(f"Parsing architectural scan: {scan_id}")

        # Load scan from database
        scan = self.db.query(ArchitecturalScan).filter(
            ArchitecturalScan.id == scan_id
        ).first()

        if not scan:
            logger.error(f"Architectural scan not found: {scan_id}")
            raise ValueError(f"Architectural scan not found: {scan_id}")

        logger.info(f"Scan loaded: {scan.scan_name}")

        # Parse scan data based on file type
        result = {
            "scan_id": str(scan.id),
            "scan_name": scan.scan_name,
            "building_type": scan.building_type,
            "complexity_level": scan.complexity_level.value,
            "num_floors": scan.num_floors,
            "total_area_m2": scan.total_area_m2,
            "capture_method": scan.capture_method,
            "point_count": scan.point_count,
            "has_textures": scan.has_textures,
            "metadata": scan.metadata
        }

        # If point cloud file exists, parse it
        if scan.point_cloud_path and Path(scan.point_cloud_path).exists():
            logger.info(f"Point cloud file found: {scan.point_cloud_path}")
            # TODO: Parse point cloud data (E57, LAS, etc.)
            result["point_cloud_available"] = True
        else:
            result["point_cloud_available"] = False

        logger.info(f"Architectural scan parsed: {scan.scan_name}")

        return result

    # Helper methods

    def _extract_element_properties(self, element: Any) -> Dict[str, Any]:
        """Extract property sets from IFC element."""
        properties = {}

        try:
            if hasattr(element, 'IsDefinedBy'):
                for definition in element.IsDefinedBy:
                    if definition.is_a('IfcRelDefinesByProperties'):
                        prop_set = definition.RelatingPropertyDefinition

                        if prop_set.is_a('IfcPropertySet'):
                            for prop in prop_set.HasProperties:
                                if prop.is_a('IfcPropertySingleValue'):
                                    prop_name = prop.Name
                                    prop_value = prop.NominalValue.wrappedValue if prop.NominalValue else None
                                    properties[prop_name] = prop_value

        except Exception as e:
            logger.warning(f"Failed to extract properties: {e}")

        return properties

    def _extract_element_quantities(self, element: Any) -> Dict[str, float]:
        """Extract quantities from IFC element."""
        quantities = {}

        try:
            if hasattr(element, 'IsDefinedBy'):
                for definition in element.IsDefinedBy:
                    if definition.is_a('IfcRelDefinesByProperties'):
                        quantity_set = definition.RelatingPropertyDefinition

                        if quantity_set.is_a('IfcElementQuantity'):
                            for quantity in quantity_set.Quantities:
                                q_name = quantity.Name

                                if quantity.is_a('IfcQuantityLength'):
                                    quantities[q_name] = float(quantity.LengthValue)
                                elif quantity.is_a('IfcQuantityArea'):
                                    quantities[q_name] = float(quantity.AreaValue)
                                elif quantity.is_a('IfcQuantityVolume'):
                                    quantities[q_name] = float(quantity.VolumeValue)
                                elif quantity.is_a('IfcQuantityCount'):
                                    quantities[q_name] = float(quantity.CountValue)
                                elif quantity.is_a('IfcQuantityWeight'):
                                    quantities[q_name] = float(quantity.WeightValue)

        except Exception as e:
            logger.warning(f"Failed to extract quantities: {e}")

        return quantities
