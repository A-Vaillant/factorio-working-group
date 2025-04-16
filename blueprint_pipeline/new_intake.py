"""
factorio_blueprint_pipeline.py

An idempotent pipeline for processing Factorio blueprint CSVs
using Luigi for orchestration and the existing intake.py code.

Author: AV, Claude Opus 3.7 (April 2025)
"""
import sys
import csv
import hashlib
import json
import luigi
import os
from collections import Counter
from pathlib import Path
from typing import List, Dict, Callable, Any, Optional
from datetime import datetime

# Assuming imports from draftsman would work
from draftsman.blueprintable import Blueprintable, Blueprint
from draftsman.utils import string_to_JSON

from pydantic import BaseModel, Field, field_validator
from typing_extensions import Annotated


csv.field_size_limit(sys.maxsize)


class ScrapedData(BaseModel):
    """Schema for a single processed data entry."""
    name: str
    data: str  # The raw data string
    metadata: Dict[str, Any] = Field(default_factory=dict)
    label: str = ""
    source_csv: str = ""
    source_row: int = 0
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


# ==================== Configuration Parameters 
class PipelineConfig(luigi.Config):
    source_directory = luigi.Parameter(default="../data/raw")
    output_directory = luigi.Parameter(default="../data/processed")


# ==================== Etc. 
def calculate_file_hash(filename):
    """Calculate MD5 hash of file for idempotency checking."""
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def hash_bp_row(bp_string):
    """Hash a blueprint string to generate a unique filename."""
    return hashlib.md5(bp_string.encode()).hexdigest()[:10]


def recursive_blueprint_book_parse(bp_book: Blueprintable) -> List[Blueprint]:
    """Extract all blueprints from a blueprint book recursively."""
    # Reached a leaf node
    if isinstance(bp_book, Blueprint):
        return [bp_book]

    blueprints = []
    for bp_node in bp_book.blueprints:
        blueprints += recursive_blueprint_book_parse(bp_node)
    return blueprints


def quantify_entities(bp: Blueprint, granularity: int = 5) -> Dict[str, int]:
    """Count entities in a blueprint with specified granularity."""
    # granularity: How specific we want. 5 will be everything, lower ones
    # will group together entities into classes (conveyor belts, assemblers)
    if granularity == 5:
        grain = lambda x: x
    else:
        # Add more granularity levels as needed
        grain = lambda x: x  

    # Create counter of entity names
    c = Counter(grain(entity.name) for entity in bp.entities)
    return dict(c)


# TODO: Populate with actual tech level mapping
tech_map = {}
def quantify_tech_level(bp: Blueprint) -> int:
    """Determine the highest tech level required for a blueprint."""
    # Using Factorio 1.0. 0 is initial, 1 is red science.
    if not bp.entities:
        return 0
        
    tech_levels = {}
    for e in bp.entities:
        tech_levels[e.name] = tech_map.get(e.name, 0)
    
    return max(tech_levels.values()) if tech_levels else 0

def parse_csv_row(row: Dict[str, Any], 
                 row_num: int,
                 csv_file: str="",
                 filter_func: Callable = lambda x: True) -> List[ScrapedData]:
    """Process a single CSV row and convert it to our data model.
    
    Instead, it just creates a data model from the CSV row.
    """
    data_string = row.get('data', '')
    if not data_string:
        return []
        
    # Create a name for the data entry
    name = row.get('name', f"data_{row_num}")
    
    processed_data = ScrapedData(
        name=name,
        data=data_string,
        source_csv=csv_file,
        source_row=row_num,
        # Include all other fields from the CSV
        **{k: v for k, v in row.items() if k not in ['data', 'name']}
    )
    
    if filter_func(processed_data):
        return [processed_data]
    else:
        return []


class InputCSVFile(luigi.ExternalTask):
    """Represents a single input CSV file."""
    file_path = luigi.Parameter()
    
    def output(self):
        return luigi.LocalTarget(self.file_path)


class ProcessCSVFile(luigi.Task):
    """Process a single CSV file containing blueprint data."""
    file_path = luigi.Parameter()
    output_dir = luigi.Parameter()
    
    def requires(self):
        return InputCSVFile(file_path=self.file_path)
    
    def output(self):
        # Output marker indicating this file was processed
        file_hash = calculate_file_hash(self.file_path)
        return luigi.LocalTarget(os.path.join(self.output_dir, f".processed_{file_hash}"))
    
    def run(self):
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Process the CSV file
        with self.input().open('r') as infile:
            reader = csv.DictReader(infile)
            
            for ix, row in enumerate(reader):
                # Process each row into potentially multiple blueprints
                row_data = parse_csv_row(row, row_num=ix,
                                         csv_file=str(self.input()))


                for blueprint in row_data:
                    # Generate a unique filename for each blueprint
                    bp_hash = hash_bp_row(blueprint.data)
                    bp_name = blueprint.name or 'unnamed'
                    bp_name = bp_name.replace(' ', '_').replace('/', '_')
                    output_file = os.path.join(self.output_dir, f"{bp_name}_{bp_hash}.json")
                    
                    # Write blueprint data to JSON file
                    with open(output_file, 'w') as outfile:
                        json.dump(blueprint.model_dump(), outfile, indent=2)
        
        # Create marker file to indicate successful processing
        with self.output().open('w') as outfile:
            outfile.write(f"Processed on: {datetime.now().isoformat()}")


class ProcessAllCSVs(luigi.WrapperTask):
    """Process all CSV files in the source directory."""
    source_directory = luigi.Parameter(default=PipelineConfig().source_directory)
    output_directory = luigi.Parameter(default=PipelineConfig().output_directory)
    
    def requires(self):
        source_dir = Path(self.source_directory)
        tasks = []
        
        for csv_file in source_dir.glob("*.csv"):
            tasks.append(
                ProcessCSVFile(
                    file_path=str(csv_file),
                    output_dir=self.output_directory
                )
            )
        
        return tasks


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
    entries: List[ManifestEntry]


class GenerateManifest(luigi.Task):
    """Generate a manifest file with metadata about all processed data entries."""
    source_directory = luigi.Parameter(default=PipelineConfig().source_directory)
    output_directory = luigi.Parameter(default=PipelineConfig().output_directory)
    
    def requires(self):
        return ProcessAllCSVs(
            source_directory=self.source_directory,
            output_directory=self.output_directory
        )
    
    def output(self):
        return luigi.LocalTarget(os.path.join(self.output_directory, "manifest.json"))
    
    def run(self):
        output_dir = Path(self.output_directory)
        json_files = list(output_dir.glob("*.json"))
        
        entries = []
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            entry = ManifestEntry(
                filename=json_file.name,
                name=data.name,
                source_csv=data.source_csv,
            )
            
            entries.append(entry)
        
        # Create the manifest
        manifest = DataManifest(
            processed_time=str(datetime.now().isoformat()),
            entry_count=len(entries),
            entries=entries
        )
        
        # Write manifest to file
        with self.output().open('w') as outfile:
            outfile.write(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    luigi.run()
