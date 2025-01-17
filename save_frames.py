from pathlib import Path
from typing import Optional
import logging
import shutil
import json

import numpy as np
from lib import flip, pdb
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_frames(pdb_path: Path, output_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Save frames from PDB file.
    
    Args:
        pdb_path: Path to PDB file
        output_path: Path to save frames
        
    Returns:
        Tuple of (pdb_data, frame data)
    """
    pdb_data, frames = pdb.multi(str(pdb_path))
    np.save(output_path, frames)
    logger.info(f"Saved frames to {output_path}")
    return pdb_data, frames

def process_molecule(folder_path: Path) -> None:
    """Process a single molecule directory.
    
    Args:
        folder_path: Path to molecule directory
    """
    molecule_name = folder_path.name
    pdb_file = folder_path / f"{molecule_name}.pdb"
    pack_dir = folder_path / "clusters" / "pack"
    alpha_file = pack_dir / "alpha.npy"
    beta_file = pack_dir / "beta.npy"

    if not pdb_file.exists():
        logger.warning(f"No PDB file found in {folder_path}")
        return

    if alpha_file.exists() and beta_file.exists():
        logger.info(f"Skipping {molecule_name} - output files already exist")
        return

    try:
        # Create output directory
        pack_dir.mkdir(parents=True, exist_ok=True)

        # Determine molecule type and output paths
        if len(folder_path.name) == 8:
            json_file = folder_path / f"{folder_path.name}.json"
            with open(json_file, 'r') as f:
                data = json.load(f)
            glycam = data.get("indexOrderedSequence", "output")
            is_alpha = flip.is_alpha(glycam)
        else:
            is_alpha = flip.is_alpha(molecule_name)
        output_pdb = folder_path / "output" / ("beta.pdb" if is_alpha else "alpha.pdb")
        
        # Generate flipped structure
        logger.info(f"Processing {pdb_file}")
        flip.flip_alpha_beta_multi(str(pdb_file), str(output_pdb), step_size=10)
        logger.info(f"Generated flipped structure: {output_pdb}")

        # Save frames for original structure
        if is_alpha:
            save_frames(pdb_file, alpha_file)
            save_frames(output_pdb, beta_file)
        else:
            save_frames(pdb_file, beta_file)
            save_frames(output_pdb, alpha_file)

        # Cleanup
        output_pdb.unlink()
        logger.info(f"Removed temporary file: {output_pdb}")

    except Exception as e:
        logger.error(f"Failed to process {molecule_name}: {str(e)}")

def main() -> None:
    """Main execution function."""
    data_dir = Path(config.data_dir)
    logger.info(f"Processing molecules in: {data_dir}")
    
    for folder in data_dir.iterdir():
        if folder.is_dir():
            if folder.name.endswith(('a2-OH', 'b2-OH')):
                logger.info(f"Skipping {folder.name} - a2-OH/b2-OH molecule")
                continue
            process_molecule(folder)

if __name__ == "__main__":
    main()
