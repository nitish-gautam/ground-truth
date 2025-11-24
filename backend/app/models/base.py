"""
Base model with common functionality
===================================

Shared base class for all database models with audit trails,
timestamps, and common fields.
"""

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from sqlalchemy import Column, DateTime, String, text
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.ext.declarative import as_declarative, declared_attr
from sqlalchemy.orm import Mapped, mapped_column


@as_declarative()
class Base:
    """Base class for all database models."""

    id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=text("gen_random_uuid()"),
        unique=True,
        nullable=False
    )

    # Timestamp fields
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("now()"),
        nullable=False
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("now()"),
        onupdate=datetime.utcnow,
        nullable=False
    )

    # Audit fields
    created_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        server_default=text("current_user"),
        nullable=True
    )

    updated_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        server_default=text("current_user"),
        nullable=True
    )

    @declared_attr
    def __tablename__(cls) -> str:
        """Generate table name from class name."""
        # Convert CamelCase to snake_case
        name = cls.__name__
        result = []
        for i, c in enumerate(name):
            if c.isupper() and i > 0:
                result.append('_')
            result.append(c.lower())
        return ''.join(result)

    def to_dict(self, exclude: Optional[set] = None) -> Dict[str, Any]:
        """Convert model instance to dictionary."""
        exclude = exclude or set()
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
            if column.name not in exclude
        }

    def update_from_dict(self, data: Dict[str, Any], exclude: Optional[set] = None) -> None:
        """Update model instance from dictionary."""
        exclude = exclude or {'id', 'created_at', 'created_by'}
        for key, value in data.items():
            if key not in exclude and hasattr(self, key):
                setattr(self, key, value)

        # Update the updated_at timestamp
        self.updated_at = datetime.utcnow()

    def __repr__(self) -> str:
        """String representation of model instance."""
        return f"<{self.__class__.__name__}(id={self.id})>"


class BaseModel(Base):
    """Enhanced base model with additional functionality."""

    __abstract__ = True

    def __init__(self, **kwargs):
        """Initialize model with keyword arguments."""
        super().__init__()
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @classmethod
    def create(cls, **kwargs):
        """Factory method to create new instance."""
        return cls(**kwargs)

    def refresh_updated_at(self) -> None:
        """Manually refresh the updated_at timestamp."""
        self.updated_at = datetime.utcnow()

    def get_changes(self, original_dict: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Get changes compared to original state."""
        current_dict = self.to_dict()
        changes = {}

        for key, current_value in current_dict.items():
            original_value = original_dict.get(key)
            if original_value != current_value:
                changes[key] = {
                    'from': original_value,
                    'to': current_value
                }

        return changes

    @property
    def is_new(self) -> bool:
        """Check if this is a new instance (not yet persisted)."""
        return self.created_at is None

    @property
    def age_seconds(self) -> float:
        """Get age of record in seconds."""
        if self.created_at:
            return (datetime.utcnow() - self.created_at.replace(tzinfo=None)).total_seconds()
        return 0.0

    @property
    def last_modified_seconds(self) -> float:
        """Get seconds since last modification."""
        if self.updated_at:
            return (datetime.utcnow() - self.updated_at.replace(tzinfo=None)).total_seconds()
        return 0.0