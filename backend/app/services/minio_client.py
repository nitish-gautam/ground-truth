"""
MinIO Client Service
Handles all MinIO object storage operations
"""

from minio import Minio
from minio.error import S3Error
from io import BytesIO
from typing import Optional, Dict
import os
from datetime import timedelta

class MinIOService:
    """MinIO client for file storage operations"""

    def __init__(self):
        self.endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
        self.access_key = os.getenv("MINIO_ROOT_USER", "minioadmin")
        self.secret_key = os.getenv("MINIO_ROOT_PASSWORD", "mD9E3_kgZJAPRjNvBWOxGQ")
        self.secure = os.getenv("MINIO_SECURE", "false").lower() == "true"

        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure
        )

        # Ensure buckets exist
        self._ensure_buckets()

    def _ensure_buckets(self):
        """Create default buckets if they don't exist"""
        buckets = [
            "gpr-data",
            "bim-models",
            "lidar-scans",
            "documents",
            "reports"
        ]

        for bucket in buckets:
            try:
                if not self.client.bucket_exists(bucket):
                    self.client.make_bucket(bucket)
                    print(f"Created bucket: {bucket}")
            except S3Error as e:
                print(f"Error creating bucket {bucket}: {e}")

    async def upload_file(
        self,
        bucket_name: str,
        object_name: str,
        file_data: BytesIO,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Upload a file to MinIO

        Args:
            bucket_name: Target bucket
            object_name: Object name in bucket
            file_data: File data as BytesIO
            content_type: MIME type
            metadata: Optional metadata dict

        Returns:
            Dict with upload result
        """
        try:
            file_data.seek(0)  # Reset file pointer
            file_size = file_data.getbuffer().nbytes

            result = self.client.put_object(
                bucket_name=bucket_name,
                object_name=object_name,
                data=file_data,
                length=file_size,
                content_type=content_type,
                metadata=metadata or {}
            )

            return {
                "success": True,
                "bucket": bucket_name,
                "object_name": object_name,
                "etag": result.etag,
                "version_id": result.version_id
            }

        except S3Error as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def get_file(
        self,
        bucket_name: str,
        object_name: str
    ) -> bytes:
        """
        Download a file from MinIO

        Args:
            bucket_name: Source bucket
            object_name: Object name

        Returns:
            File data as bytes
        """
        try:
            response = self.client.get_object(bucket_name, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            return data

        except S3Error as e:
            raise Exception(f"Failed to get file: {str(e)}")

    async def delete_file(
        self,
        bucket_name: str,
        object_name: str
    ) -> bool:
        """Delete a file from MinIO"""
        try:
            self.client.remove_object(bucket_name, object_name)
            return True
        except S3Error as e:
            print(f"Error deleting file: {e}")
            return False

    async def get_file_url(
        self,
        bucket_name: str,
        object_name: str,
        expiry: int = 3600
    ) -> str:
        """
        Get a presigned URL for file access

        Args:
            bucket_name: Bucket name
            object_name: Object name
            expiry: URL expiry in seconds (default 1 hour)

        Returns:
            Presigned URL
        """
        try:
            url = self.client.presigned_get_object(
                bucket_name,
                object_name,
                expires=timedelta(seconds=expiry)
            )
            return url
        except S3Error as e:
            raise Exception(f"Failed to generate URL: {str(e)}")

    async def list_files(
        self,
        bucket_name: str,
        prefix: Optional[str] = None
    ) -> list:
        """List all files in a bucket"""
        try:
            objects = self.client.list_objects(
                bucket_name,
                prefix=prefix,
                recursive=True
            )

            files = []
            for obj in objects:
                files.append({
                    "object_name": obj.object_name,
                    "size": obj.size,
                    "last_modified": obj.last_modified,
                    "etag": obj.etag
                })

            return files

        except S3Error as e:
            print(f"Error listing files: {e}")
            return []

    async def get_file_stats(
        self,
        bucket_name: str,
        object_name: str
    ) -> Dict:
        """Get file statistics"""
        try:
            stat = self.client.stat_object(bucket_name, object_name)
            return {
                "size": stat.size,
                "etag": stat.etag,
                "content_type": stat.content_type,
                "last_modified": stat.last_modified,
                "metadata": stat.metadata
            }
        except S3Error as e:
            raise Exception(f"Failed to get stats: {str(e)}")
