"""
models.py

A bunch of Pydantic stuff that Claude generated.
"""
import os
from typing import Optional, Any, ClassVar, Union
from pydantic import BaseModel, Field, field_validator
from typing_extensions import Annotated

class ScrapedData(BaseModel):
    """Schema for a single processed data entry."""
    name: str
    data: str  # The blueprint string.
    source_file: str = ""
    source_row: Optional[int] = None
    date: Optional[str] = None
    tags: list = Field(default_factory=list)

    expected_extension: ClassVar[Optional[str]] = None
    
    @field_validator('data')
    @classmethod
    def validate_data_string(cls, v):
        """Basic validation that the data string is not empty."""
        if not v:
            raise ValueError("Data string cannot be empty")
        return v

    @field_validator('date')
    @classmethod
    def validate_date(cls, v):
        """ Check that this works, at least: """
        try:
            if v is None:
                return v
            # i guess.
            assert( len(v.split('/')) == 3 )
        except Exception:
            raise Exception(f"Failure in casting date string: {v}")

    @field_validator('source_file')
    @classmethod
    def validate_source_file(cls, v: str) -> str:
        """Generic validator for file extensions based on class property."""
        if not v:
            raise ValueError("Expected a source file but not provided.")
        elif not cls.expected_extension:
            return v
            
        _, extension = os.path.splitext(v.lower())
        if extension != cls.expected_extension:
            raise ValueError(f"Expected {cls.expected_extension} file, got '{extension}' extension")
        return v

class ScrapedCSVRow(ScrapedData):
    source_row: int

    expected_extension: ClassVar[str] = '.csv'
    
    @field_validator('source_row')
    @classmethod
    def validate_source_row(cls, v: int) -> int:
        """Ensure source_row is a positive integer."""
        if v < 0:
            raise ValueError(f"Row index must be non-negative, got {v}")
        return v


class ScrapedJSONFile(ScrapedData):
    source_row: None = None

    expected_extension: ClassVar[str] = '.json'
    
    @field_validator('source_row')
    @classmethod
    def validate_source_row_none(cls, v: None) -> None:
        """Ensure source_row is None for JSON files."""
        if v is not None:
            raise ValueError("source_row must be None for JSON files")
        return v


class ManifestEntry(BaseModel):
    """Schema for an entry in the data manifest."""
    filename: str
    name: str
    source_file: str
    # Any additional metadata fields you want to include


class DataManifest(BaseModel):
    """Schema for the data manifest file."""
    processed_time: str
    entry_count: int
    entries: list[ManifestEntry]


class BlueprintMetaData(BaseModel):
    """
    Represents an individual blueprint.
    Created by splitting a blueprint book into its component blueprints.
    """
    # Basic identification
    name: str
    data: str  # The blueprint string
    
    # Provenance information
    parent_name: str  # Name of the parent blueprint book
    parent_file: Optional[str] = None  # Original file the ScrapedData came from.
    parent_row: Optional[int] = None  #  Row, if it had one.
    
    # Enrichment data
    entities: dict[str, int] = Field(default_factory=dict)  # Counts of entities by type
    tech_level: Optional[int] = None  # Technology level required
    
    # Metadata and metrics
    area: Optional[dict[str, int]] = None  # Width and height
    creation_time: Optional[str] = None  # When this split was performed
    
    # Tags and categorization
    category: Optional[str] = None  # E.g., "production", "logistics", etc.
    tags: list[str] = Field(default_factory=list)  # User-defined or auto-generated tags
    

    def __init__(self, **data):
        super().__init__(**data)
        # Auto-calculate some fields if not provided
        if self.entities and self.entity_count is None:
            self.entity_count = sum(self.entities.values())
            
        # Add creation time if not present
        if self.creation_time is None:
            from datetime import datetime
            self.creation_time = datetime.now().isoformat()


class BlueprintData(BaseModel):
    # Data ex
    label: str
    description: str = ""