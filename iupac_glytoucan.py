from pathlib import Path
from typing import Dict, Optional
import logging
import json

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_json_data(file_path: Path) -> dict:
    """Load JSON data from file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary containing JSON data
    """
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        logger.error(f"Failed to load JSON from {file_path}: {str(e)}")
        raise

def create_iupac_glytoucan_mapping(data: dict) -> Dict[str, Optional[str]]:
    """Create mapping between IUPAC names and GlyTouCan IDs.
    
    Args:
        data: Dictionary containing glycan data
        
    Returns:
        Dictionary mapping IUPAC names to GlyTouCan IDs
    """
    mapping = {}
    for key, value in data.items():
        iupac_name = value.get('iupac', None)
        glytoucan_id = value.get('glytoucan_id', None)
        
        if iupac_name:
            mapping[iupac_name] = glytoucan_id if glytoucan_id else None
    
    return mapping

def save_json_data(data: dict, file_path: Path) -> None:
    """Save dictionary as JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Output file path
    """
    try:
        with open(file_path, 'w') as output_file:
            json.dump(data, output_file, indent=4)
        logger.info(f"Saved mapping to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {file_path}: {str(e)}")
        raise

def main() -> None:
    """Main execution function."""
    # Setup paths
    input_file = Path(config.output_path) / "GLYCOSHAPE.json"
    output_file = Path(config.output_path) / "iupac_glytoucan_mapping.json"

    logger.info(f"Processing GLYCOSHAPE data from {input_file}")

    try:
        # Load and process data
        data = load_json_data(input_file)
        mapping = create_iupac_glytoucan_mapping(data)
        save_json_data(mapping, output_file)
        
    except Exception as e:
        logger.error(f"Failed to process GLYCOSHAPE data: {str(e)}")
        raise

if __name__ == "__main__":
    main()