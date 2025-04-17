"""
models.py

A bunch of Pydantic stuff that Claude generated.
"""
from typing import Optional, Any, Optional
from pydantic import BaseModel, Field, field_validator
from typing_extensions import Annotated

class ScrapedData(BaseModel):
    """Schema for a single processed data entry."""
    name: str
    data: str  # The raw data string
    metadata: dict = Field(default_factory=dict)
    label: str = ""
    source_csv: str = ""
    source_row: int = 0
    date: Optional[str] = None
    original_name: Optional[str] = None
    
    # Replace Config class with model_config dict
    model_config = {
        "extra": "allow"  # Allow additional fields from CSV
    }
    
    @field_validator('data')
    @classmethod  # Add this decorator
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


class ManifestEntry(BaseModel):
    """Schema for an entry in the data manifest."""
    filename: str
    name: str
    source_csv: str
    # Any additional metadata fields you want to include


class DataManifest(BaseModel):
    """Schema for the data manifest file."""
    processed_time: str
    entry_count: int
    entries: list[ManifestEntry]


class BlueprintData(BaseModel):
    """
    Represents an individual blueprint with enriched data.
    Created by splitting a blueprint book into its component blueprints.
    """
    # Basic identification
    name: str
    data: str  # The blueprint string
    
    # Provenance information
    source_file: str  # The ScrapedData file this was split from
    parent_name: str  # Name of the parent blueprint book
    source_csv: Optional[str] = None  # Original CSV source
    source_row: Optional[int] = None  # Original row in CSV
    
    # Enrichment data
    entities: dict[str, int] = Field(default_factory=dict)  # Counts of entities by type
    tech_level: Optional[int] = None  # Technology level required
    
    # Matrix representation (could be a separate complex type)
    matrix_repr: Optional[List[List[Any]]] = None
    
    # Metadata and metrics
    entity_count: Optional[int] = None  # Total number of entities
    area: Optional[dict[str, int]] = None  # Width and height
    connections: Optional[int] = None  # Number of circuit connections
    creation_time: Optional[str] = None  # When this split was performed
    
    # Tags and categorization
    category: Optional[str] = None  # E.g., "production", "logistics", etc.
    tags: List[str] = Field(default_factory=list)  # User-defined or auto-generated tags
    
    # Performance metrics (if applicable)
    throughput: Optional[dict[str, float]] = None  # Estimated item throughput
    power_consumption: Optional[float] = None  # Estimated power usage
    
    def __init__(self, **data):
        super().__init__(**data)
        # Auto-calculate some fields if not provided
        if self.entities and self.entity_count is None:
            self.entity_count = sum(self.entities.values())
            
        # Add creation time if not present
        if self.creation_time is None:
            from datetime import datetime
            self.creation_time = datetime.now().isoformat()
