"""
models.py

A bunch of Pydantic stuff that Claude generated.
"""
from typing import Optional
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
