"""Pydantic models for YAML configuration validation."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Fallback for when pydantic is not installed
    class BaseModel:
        pass

    def Field(*args, **kwargs):
        """Dummy Field function for when pydantic is not installed."""
        return None

    def validator(*args, **kwargs):
        """Dummy validator decorator for when pydantic is not installed."""
        def decorator(func):
            return func
        return decorator


# Datasources configuration
class OpenAlexSourceConfig(BaseModel):
    """OpenAlex API source configuration."""
    kind: str = Field(..., pattern="^openalex$")
    entity: str = Field(default="works", pattern="^(works|authors|institutions|topics)$")
    mailto: Optional[str] = Field(default=None, description="Contact email for API compliance")
    filters: Optional[Dict[str, Any]] = None
    search: Optional[str] = None
    select: Optional[List[str]] = None
    sort: Optional[str] = None
    per_page: int = Field(default=200, ge=1, le=200)
    max_records: Optional[int] = Field(default=None, ge=1)


class CSVSourceConfig(BaseModel):
    """CSV file source configuration."""
    kind: str = Field(..., pattern="^csv$")
    path: str = Field(..., description="Path to CSV file")
    dtypes: Optional[Dict[str, str]] = None
    date_cols: Optional[List[str]] = None


class ParquetSourceConfig(BaseModel):
    """Parquet file source configuration."""
    kind: str = Field(..., pattern="^parquet$")
    path: str = Field(..., description="Path to Parquet file")


class DataSourcesConfig(BaseModel):
    """Root datasources configuration."""
    sources: Dict[str, Dict[str, Any]] = Field(..., description="Data source definitions")

    @validator("sources")
    def validate_primary_exists(cls, v):
        if "primary" not in v:
            raise ValueError("sources.primary is required")
        return v


# Schema configuration
class SchemaConstraints(BaseModel):
    """Schema validation constraints."""
    non_null: Optional[List[str]] = Field(default_factory=list)
    unique: Optional[List[str]] = Field(default_factory=list)


class SchemaConfig(BaseModel):
    """Root schema configuration."""
    coerce_types: Optional[Dict[str, str]] = None
    constraints: Optional[SchemaConstraints] = None
    required_columns: Optional[List[str]] = None


# Slices configuration
class SliceSpec(BaseModel):
    """Individual slice specification."""
    expr: Optional[str] = Field(default=None, description="Pandas query expression")
    groupby: Optional[str | List[str]] = Field(default=None, description="Column(s) to group by")


class SlicesConfig(BaseModel):
    """Root slices configuration."""
    slices: Dict[str, SliceSpec] = Field(default_factory=dict, description="Slice definitions")


def validate_datasources_yaml(path: Path) -> DataSourcesConfig:
    """
    Validate datasources.yaml configuration file.

    Args:
        path: Path to datasources.yaml

    Returns:
        Validated DataSourcesConfig object

    Raises:
        ValueError: If validation fails
        FileNotFoundError: If file doesn't exist
    """
    if not PYDANTIC_AVAILABLE:
        raise ImportError("pydantic is required for config validation. Install with: pip install pydantic")

    if not path.exists():
        raise FileNotFoundError(f"Datasources file not found: {path}")

    import yaml
    data = yaml.safe_load(path.read_text())
    try:
        return DataSourcesConfig(**data)
    except Exception as e:
        raise ValueError(f"Invalid datasources configuration in {path}: {e}")


def validate_schema_yaml(path: Path) -> SchemaConfig:
    """
    Validate schema.yaml configuration file.

    Args:
        path: Path to schema.yaml

    Returns:
        Validated SchemaConfig object

    Raises:
        ValueError: If validation fails
        FileNotFoundError: If file doesn't exist
    """
    if not PYDANTIC_AVAILABLE:
        raise ImportError("pydantic is required for config validation. Install with: pip install pydantic")

    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")

    import yaml
    data = yaml.safe_load(path.read_text())
    try:
        return SchemaConfig(**data)
    except Exception as e:
        raise ValueError(f"Invalid schema configuration in {path}: {e}")


def validate_slices_yaml(path: Path) -> SlicesConfig:
    """
    Validate slices.yaml configuration file.

    Args:
        path: Path to slices.yaml

    Returns:
        Validated SlicesConfig object

    Raises:
        ValueError: If validation fails
        FileNotFoundError: If file doesn't exist
    """
    if not PYDANTIC_AVAILABLE:
        raise ImportError("pydantic is required for config validation. Install with: pip install pydantic")

    if not path.exists():
        raise FileNotFoundError(f"Slices file not found: {path}")

    import yaml
    data = yaml.safe_load(path.read_text())
    try:
        return SlicesConfig(**data)
    except Exception as e:
        raise ValueError(f"Invalid slices configuration in {path}: {e}")


def validate_all_configs(datasources_path: Path, schema_path: Path, slices_path: Path) -> bool:
    """
    Validate all configuration files at startup.

    Args:
        datasources_path: Path to datasources.yaml
        schema_path: Path to schema.yaml
        slices_path: Path to slices.yaml

    Returns:
        True if all configs are valid

    Raises:
        ValueError: If any validation fails
    """
    if not PYDANTIC_AVAILABLE:
        # Gracefully skip validation if pydantic not installed
        return True

    validate_datasources_yaml(datasources_path)
    validate_schema_yaml(schema_path)
    validate_slices_yaml(slices_path)
    return True
