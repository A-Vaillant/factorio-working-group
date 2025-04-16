"""
pipeline.py

An idempotent pipeline for processing Factorio blueprint CSVs
using Luigi for orchestration.

Author: A. Vaillant, Claude Opus 3.7 (April 2025)
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
from datetime import datetime, date

# Assuming imports from draftsman would work
from draftsman.blueprintable import Blueprintable, Blueprint
from draftsman.utils import string_to_JSON


# enrichment
from enrichment import *
from models import ScrapedData, ManifestEntry, DataManifest
from utils import calculate_file_hash, hash_bp_row


csv.field_size_limit(sys.maxsize)


# ==================== Configuration Parameters 
class PipelineConfig(luigi.Config):
    source_directory = luigi.Parameter(default="../../data/raw")
    output_directory = luigi.Parameter(default="../../data/processed")


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
    
    # One of them is DD-MM-YY.
    processed_data = ScrapedData(
        name=name,
        data=data_string,
        source_csv=os.path.basename(csv_file),
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
                name=data['name'],
                source_csv=data['source_csv'],
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
