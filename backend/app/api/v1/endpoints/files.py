"""
File Upload and Processing Endpoints
Handles file uploads, MinIO storage, processing, and visualization
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import io
from datetime import datetime
import mimetypes
from pathlib import Path

from app.core.database import get_db
from app.services.file_processor import FileProcessorService
from app.services.minio_client import MinIOService
from app.models.hs2 import FileUpload, FileProcessingStatus
from pydantic import BaseModel

router = APIRouter(prefix="/files", tags=["File Management"])

# ============================================================================
# Pydantic Models
# ============================================================================

class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    file_type: str
    file_size: int
    bucket: str
    object_name: str
    upload_status: str
    processing_status: str
    uploaded_at: datetime
    asset_id: Optional[str] = None

class FileProcessingResult(BaseModel):
    file_id: str
    filename: str
    processing_status: str
    segments_count: int
    missing_data: List[str]
    metadata: dict
    errors: List[str]
    warnings: List[str]
    processed_at: datetime

class FileAnalytics(BaseModel):
    total_files: int
    total_size_mb: float
    files_by_type: dict
    files_by_status: dict
    recent_uploads: List[FileUploadResponse]
    processing_timeline: List[dict]

# ============================================================================
# Dependencies
# ============================================================================

minio_service = MinIOService()
file_processor = FileProcessorService()

# ============================================================================
# Upload Endpoints
# ============================================================================

@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    asset_id: Optional[str] = None,
    bucket: str = "documents",
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a single file to MinIO and trigger processing

    Supported file types:
    - GPR: .sgy, .dzt, .dt1
    - BIM: .ifc, .rvt
    - LiDAR: .las, .laz, .e57
    - Documents: .pdf, .docx, .xlsx
    - CAD: .dwg, .dxf
    - Images: .jpg, .png, .tif
    """
    try:
        # Validate file size (max 1GB)
        contents = await file.read()
        file_size = len(contents)

        if file_size > 1024 * 1024 * 1024:  # 1GB
            raise HTTPException(400, "File too large. Maximum size is 1GB")

        # Detect file type
        file_ext = Path(file.filename).suffix.lower()
        content_type = file.content_type or mimetypes.guess_type(file.filename)[0] or "application/octet-stream"

        # Determine bucket based on file type
        if bucket == "auto":
            bucket = _determine_bucket(file_ext)

        # Generate unique object name
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        object_name = f"{timestamp}_{file.filename}"

        # Upload to MinIO
        upload_result = await minio_service.upload_file(
            bucket_name=bucket,
            object_name=object_name,
            file_data=io.BytesIO(contents),
            content_type=content_type,
            metadata={
                "original_filename": file.filename,
                "asset_id": asset_id or "unassigned",
                "upload_date": datetime.utcnow().isoformat()
            }
        )

        # Create database record
        file_record = FileUpload(
            filename=file.filename,
            original_filename=file.filename,
            file_type=file_ext,
            content_type=content_type,
            file_size=file_size,
            bucket=bucket,
            object_name=object_name,
            asset_id=asset_id,
            upload_status="completed",
            processing_status="pending",
            uploaded_at=datetime.utcnow()
        )

        db.add(file_record)
        await db.commit()
        await db.refresh(file_record)

        # Trigger background processing
        if background_tasks:
            background_tasks.add_task(
                process_file_background,
                file_id=str(file_record.id),
                bucket=bucket,
                object_name=object_name,
                file_type=file_ext
            )

        return FileUploadResponse(
            file_id=str(file_record.id),
            filename=file.filename,
            file_type=file_ext,
            file_size=file_size,
            bucket=bucket,
            object_name=object_name,
            upload_status="completed",
            processing_status="pending",
            uploaded_at=file_record.uploaded_at,
            asset_id=asset_id
        )

    except Exception as e:
        raise HTTPException(500, f"Upload failed: {str(e)}")


@router.post("/upload-multiple", response_model=List[FileUploadResponse])
async def upload_multiple_files(
    files: List[UploadFile] = File(...),
    asset_id: Optional[str] = None,
    bucket: str = "documents",
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Upload multiple files at once
    """
    results = []

    for file in files:
        try:
            result = await upload_file(
                file=file,
                asset_id=asset_id,
                bucket=bucket,
                background_tasks=background_tasks,
                db=db
            )
            results.append(result)
        except Exception as e:
            # Log error but continue with other files
            results.append(FileUploadResponse(
                file_id="error",
                filename=file.filename,
                file_type="unknown",
                file_size=0,
                bucket=bucket,
                object_name="",
                upload_status="failed",
                processing_status="error",
                uploaded_at=datetime.utcnow(),
                asset_id=asset_id
            ))

    return results


# ============================================================================
# Processing Endpoints
# ============================================================================

@router.get("/process/{file_id}", response_model=FileProcessingResult)
async def get_processing_status(
    file_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get processing status and results for a file
    """
    # Get file record
    file_record = await db.get(FileUpload, file_id)
    if not file_record:
        raise HTTPException(404, "File not found")

    # Get processing status
    processing_status = await file_processor.get_processing_status(file_id)

    return FileProcessingResult(
        file_id=file_id,
        filename=file_record.filename,
        processing_status=processing_status.get("status", "pending"),
        segments_count=processing_status.get("segments_count", 0),
        missing_data=processing_status.get("missing_data", []),
        metadata=processing_status.get("metadata", {}),
        errors=processing_status.get("errors", []),
        warnings=processing_status.get("warnings", []),
        processed_at=processing_status.get("processed_at", datetime.utcnow())
    )


@router.post("/process/{file_id}/retry")
async def retry_processing(
    file_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Retry processing a failed file
    """
    file_record = await db.get(FileUpload, file_id)
    if not file_record:
        raise HTTPException(404, "File not found")

    # Reset processing status
    file_record.processing_status = "pending"
    await db.commit()

    # Trigger reprocessing
    background_tasks.add_task(
        process_file_background,
        file_id=file_id,
        bucket=file_record.bucket,
        object_name=file_record.object_name,
        file_type=file_record.file_type
    )

    return {"status": "reprocessing", "file_id": file_id}


# ============================================================================
# Analytics Endpoints
# ============================================================================

@router.get("/analytics", response_model=FileAnalytics)
async def get_file_analytics(
    asset_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Get file upload and processing analytics
    """
    from sqlalchemy import select, func

    # Base query
    query = select(FileUpload)
    if asset_id:
        query = query.where(FileUpload.asset_id == asset_id)

    result = await db.execute(query)
    files = result.scalars().all()

    # Calculate statistics
    total_files = len(files)
    total_size_mb = sum(f.file_size for f in files) / (1024 * 1024)

    # Group by type
    files_by_type = {}
    for f in files:
        files_by_type[f.file_type] = files_by_type.get(f.file_type, 0) + 1

    # Group by status
    files_by_status = {}
    for f in files:
        files_by_status[f.processing_status] = files_by_status.get(f.processing_status, 0) + 1

    # Recent uploads (last 10)
    recent_uploads = sorted(files, key=lambda x: x.uploaded_at, reverse=True)[:10]

    # Processing timeline (last 24 hours)
    timeline = []
    for f in files:
        timeline.append({
            "timestamp": f.uploaded_at.isoformat(),
            "filename": f.filename,
            "status": f.processing_status,
            "file_type": f.file_type
        })

    return FileAnalytics(
        total_files=total_files,
        total_size_mb=round(total_size_mb, 2),
        files_by_type=files_by_type,
        files_by_status=files_by_status,
        recent_uploads=[
            FileUploadResponse(
                file_id=str(f.id),
                filename=f.filename,
                file_type=f.file_type,
                file_size=f.file_size,
                bucket=f.bucket,
                object_name=f.object_name,
                upload_status=f.upload_status,
                processing_status=f.processing_status,
                uploaded_at=f.uploaded_at,
                asset_id=f.asset_id
            )
            for f in recent_uploads
        ],
        processing_timeline=timeline
    )


@router.get("/list")
async def list_files(
    asset_id: Optional[str] = None,
    bucket: Optional[str] = None,
    file_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
):
    """
    List uploaded files with filters
    """
    from sqlalchemy import select

    query = select(FileUpload)

    if asset_id:
        query = query.where(FileUpload.asset_id == asset_id)
    if bucket:
        query = query.where(FileUpload.bucket == bucket)
    if file_type:
        query = query.where(FileUpload.file_type == file_type)

    query = query.order_by(FileUpload.uploaded_at.desc())
    query = query.limit(limit).offset(offset)

    result = await db.execute(query)
    files = result.scalars().all()

    return {
        "total": len(files),
        "limit": limit,
        "offset": offset,
        "files": [
            {
                "file_id": str(f.id),
                "filename": f.filename,
                "file_type": f.file_type,
                "file_size": f.file_size,
                "bucket": f.bucket,
                "processing_status": f.processing_status,
                "uploaded_at": f.uploaded_at.isoformat(),
                "asset_id": f.asset_id
            }
            for f in files
        ]
    }


@router.get("/download/{file_id}")
async def download_file(
    file_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Download a file from MinIO
    """
    file_record = await db.get(FileUpload, file_id)
    if not file_record:
        raise HTTPException(404, "File not found")

    # Get file from MinIO
    file_data = await minio_service.get_file(
        bucket_name=file_record.bucket,
        object_name=file_record.object_name
    )

    return StreamingResponse(
        io.BytesIO(file_data),
        media_type=file_record.content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{file_record.filename}"'
        }
    )


# ============================================================================
# Helper Functions
# ============================================================================

def _determine_bucket(file_ext: str) -> str:
    """Determine the appropriate bucket based on file extension"""
    gpr_extensions = [".sgy", ".dzt", ".dt1", ".seg", ".gssi"]
    bim_extensions = [".ifc", ".rvt", ".rfa"]
    lidar_extensions = [".las", ".laz", ".e57", ".ply", ".xyz"]
    cad_extensions = [".dwg", ".dxf"]

    if file_ext in gpr_extensions:
        return "gpr-data"
    elif file_ext in bim_extensions:
        return "bim-models"
    elif file_ext in lidar_extensions:
        return "lidar-scans"
    elif file_ext in cad_extensions:
        return "documents"
    else:
        return "documents"


async def process_file_background(
    file_id: str,
    bucket: str,
    object_name: str,
    file_type: str
):
    """
    Background task to process uploaded files
    """
    try:
        # Download file from MinIO
        file_data = await minio_service.get_file(bucket, object_name)

        # Process based on file type
        result = await file_processor.process_file(
            file_id=file_id,
            file_data=file_data,
            file_type=file_type
        )

        # Update status in database
        # (would need database connection here)
        print(f"Processing completed for {file_id}: {result}")

    except Exception as e:
        print(f"Processing failed for {file_id}: {str(e)}")
