""" Claude 3.7 fixing some data stuff. """

import os
import json
import re
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('json_converter')

root_dir = Path("./json/")
input_dir = root_dir / 'txt'
output_dir = root_dir / 'factorio-tech'
# Create output directory if it doesn't exist
os.makedirs(str(output_dir), exist_ok=True)

for filename in input_dir.iterdir():
    if filename.suffix == ('.txt'):
        input_path = filename
        output_path = output_dir / (filename.with_suffix('.json')).name
        
        logger.info(f"Processing {filename}")
        
        with input_path.open('r') as f:
            content = f.read()
        
        # First attempt: try to evaluate as Python dict and convert to JSON
        try:
            # This safely evaluates the Python dictionary literal
            data_dict = eval(content)
            # Convert to properly formatted JSON
            with output_path.open('w') as f:
                json.dump(data_dict, f, indent=2)
            logger.info(f"Successfully converted {filename} using eval method")
        except Exception as e:
            logger.warning(f"Eval failed for {filename}: {str(e)}")
            logger.info(f"Falling back to regex replacement for {filename}")
            
            # Fallback: regex replacement for common JSON-like patterns
            # Fix keys
            content = re.sub(r"(\{|\,)\s*'([^']+)'\s*:", r'\1 "\2":', content)
            # Fix values that are strings
            content = re.sub(r":\s*'([^']+)'", r': "\1"', content)
            
            try:
                with output_path.open('w') as f:
                    f.write(content)
                logger.info(f"Wrote regex-fixed content to {output_path}")
            except Exception as e:
                logger.error(f"Failed to write output file {output_path}: {str(e)}")