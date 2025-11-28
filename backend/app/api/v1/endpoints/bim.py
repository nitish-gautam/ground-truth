"""
BIM API Endpoints
Serve BIM model files and metadata
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import List, Dict, Optional
from pathlib import Path
import os

router = APIRouter()

# Base path to BIM data
BIM_BASE_PATH = Path("/datasets/hs2/rawdata/IFC4.3.x-sample-models-main/models")


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in MB"""
    if file_path.exists():
        return round(file_path.stat().st_size / (1024 * 1024), 2)
    return 0.0


@router.get("/models")
async def get_available_models():
    """
    Get list of all available BIM models with metadata
    """
    models = []

    if not BIM_BASE_PATH.exists():
        return {"models": [], "total_count": 0}

    # Find all IFC files
    ifc_files = list(BIM_BASE_PATH.rglob("*.ifc"))

    for ifc_file in ifc_files:
        # Get category from parent directory
        category = ifc_file.parent.parent.name if ifc_file.parent.parent != BIM_BASE_PATH else "Unknown"

        # Clean up category name
        category_clean = category.replace("-", " ").title()

        models.append({
            "id": ifc_file.stem,
            "name": ifc_file.stem.replace("-", " ").title(),
            "filename": ifc_file.name,
            "category": category_clean,
            "size_mb": get_file_size_mb(ifc_file),
            "size_kb": round(get_file_size_mb(ifc_file) * 1024, 0),
            "path": str(ifc_file.relative_to(BIM_BASE_PATH.parent))
        })

    # Sort by category, then name
    models.sort(key=lambda x: (x["category"], x["name"]))

    # Group by category for summary
    categories = {}
    for model in models:
        cat = model["category"]
        if cat not in categories:
            categories[cat] = {"count": 0, "total_size_mb": 0}
        categories[cat]["count"] += 1
        categories[cat]["total_size_mb"] += model["size_mb"]

    return {
        "models": models,
        "total_count": len(models),
        "categories": categories,
        "total_size_mb": sum(m["size_mb"] for m in models)
    }


@router.get("/model/{model_id}")
async def get_model_metadata(model_id: str):
    """
    Get metadata for a specific BIM model
    """
    # Search for the IFC file
    ifc_files = list(BIM_BASE_PATH.rglob(f"{model_id}.ifc"))

    if not ifc_files:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    ifc_file = ifc_files[0]

    # Get category
    category = ifc_file.parent.parent.name if ifc_file.parent.parent != BIM_BASE_PATH else "Unknown"
    category_clean = category.replace("-", " ").title()

    return {
        "id": ifc_file.stem,
        "name": ifc_file.stem.replace("-", " ").title(),
        "filename": ifc_file.name,
        "category": category_clean,
        "size_mb": get_file_size_mb(ifc_file),
        "size_bytes": ifc_file.stat().st_size,
        "path": str(ifc_file.relative_to(BIM_BASE_PATH.parent)),
        "full_path": str(ifc_file),
        "exists": ifc_file.exists()
    }


@router.get("/model/{model_id}/download")
async def download_model(model_id: str):
    """
    Download a specific IFC model file
    """
    # Search for the IFC file
    ifc_files = list(BIM_BASE_PATH.rglob(f"{model_id}.ifc"))

    if not ifc_files:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    ifc_file = ifc_files[0]

    if not ifc_file.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found on disk")

    return FileResponse(
        path=str(ifc_file),
        filename=ifc_file.name,
        media_type="application/ifc"
    )


@router.get("/categories")
async def get_model_categories():
    """
    Get list of all model categories with counts
    """
    if not BIM_BASE_PATH.exists():
        return {"categories": [], "total_count": 0}

    # Get all subdirectories (categories)
    categories = []

    for category_dir in BIM_BASE_PATH.iterdir():
        if category_dir.is_dir() and not category_dir.name.startswith('.'):
            # Count IFC files in this category
            ifc_files = list(category_dir.rglob("*.ifc"))

            if ifc_files:
                category_name = category_dir.name.replace("-", " ").title()
                total_size_mb = sum(get_file_size_mb(f) for f in ifc_files)

                categories.append({
                    "id": category_dir.name,
                    "name": category_name,
                    "model_count": len(ifc_files),
                    "total_size_mb": round(total_size_mb, 2)
                })

    categories.sort(key=lambda x: x["name"])

    return {
        "categories": categories,
        "total_count": len(categories),
        "total_models": sum(c["model_count"] for c in categories)
    }


@router.get("/stats")
async def get_bim_stats():
    """
    Get overall BIM dataset statistics
    """
    if not BIM_BASE_PATH.exists():
        return {
            "total_models": 0,
            "total_categories": 0,
            "total_size_mb": 0,
            "avg_model_size_mb": 0,
            "largest_model": None,
            "smallest_model": None
        }

    ifc_files = list(BIM_BASE_PATH.rglob("*.ifc"))

    if not ifc_files:
        return {
            "total_models": 0,
            "total_categories": 0,
            "total_size_mb": 0,
            "avg_model_size_mb": 0,
            "largest_model": None,
            "smallest_model": None
        }

    # Calculate stats
    sizes = [(f, get_file_size_mb(f)) for f in ifc_files]
    total_size = sum(s[1] for s in sizes)
    largest = max(sizes, key=lambda x: x[1])
    smallest = min(sizes, key=lambda x: x[1])

    # Count categories
    categories = set()
    for ifc_file in ifc_files:
        category = ifc_file.parent.parent.name if ifc_file.parent.parent != BIM_BASE_PATH else "Unknown"
        categories.add(category)

    return {
        "total_models": len(ifc_files),
        "total_categories": len(categories),
        "total_size_mb": round(total_size, 2),
        "avg_model_size_mb": round(total_size / len(ifc_files), 2),
        "largest_model": {
            "name": largest[0].stem,
            "size_mb": largest[1]
        },
        "smallest_model": {
            "name": smallest[0].stem,
            "size_mb": smallest[1]
        }
    }
