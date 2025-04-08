from pathlib import Path
from typing import List, Tuple, Optional
import logging
import traceback
import os
import sys
import gc
import time

import numpy as np
import pandas as pd
from lib import pdb, dihedral, clustering, tfindr
import config

# Enhanced logging configuration
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"glycan_analysis_{time.strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def extract_first_frame(input_path: Path, output_path: Path) -> None:
    """Extract the first model frame from a PDB file.

    Args:
        input_path: Path to input PDB file
        output_path: Path to save the extracted frame
    """
    try:
        with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
            model_started = False
            for line in infile:
                if line.startswith('MODEL'):
                    if not model_started:
                        model_started = True
                    else:
                        break
                if model_started:
                    outfile.write(line)

            if not model_started:
                logger.warning(f"No model found in {input_path}")
                return
        logger.info(f"First frame extracted and saved as {output_path}")
    except Exception as e:
        logger.error(f"Error extracting first frame: {e}")
        raise

def get_subdirectories(folder_path: Path) -> List[Path]:
    """List all subdirectories in the given path.

    Args:
        folder_path: Path to parent directory

    Returns:
        List of subdirectory paths
    """
    try:
        subdirs = [d for d in folder_path.iterdir() if d.is_dir()]
        logger.info(f"Found {len(subdirs)} subdirectories in {folder_path}")
        return subdirs
    except Exception as e:
        logger.error(f"Error getting subdirectories: {e}")
        return []

def process_molecule(name: str) -> bool:
    """Process molecular calculations for the given structure.
    
    Args:
        name: Name of the molecule/directory to process
        
    Returns:
        bool: True if processing succeeded, False otherwise
    """
    start_time = time.time()
    logger.info(f"Starting processing of molecule: {name}")
    
    try:
        data_dir = Path(config.data_dir)
        input_file = data_dir / name / f"{name}.pdb"
        output_dir = data_dir / name / "output"
        
        if not input_file.exists():
            logger.error(f"Input file does not exist: {input_file}")
            return False
            
        output_paths = {
            'structure': output_dir / "structure.pdb",
            'pca': output_dir / "pca.csv",
            'torsions': output_dir / "torsions.csv",
            'info': output_dir / "info.txt",
            'clusters': output_dir / "cluster_default/" 
        }

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        if not output_dir.exists():
            logger.error(f"Failed to create output directory: {output_dir}")
            return False
            
        logger.info(f"Extracting first frame from {input_file}")
        # Extract first frame
        extract_first_frame(input_file, output_paths['structure'])

        logger.info(f"Loading and preparing data from {input_file}")
        # Load and prepare data
        pdb_data, frames = pdb.multi(str(input_file))
        df = pdb.to_DF(pdb_data)
        idx_noH = df.loc[df['Element'] != "H", 'Number'] - 1
        logger.info(f"Loaded {len(frames)} frames with {len(df)} atoms")

        # PCA and clustering analysis
        success_pca = False
        try:
            logger.info("Starting PCA and clustering analysis")
            pcaG, n_dim = clustering.pcawithG(frames, idx_noH, config.number_of_dimensions, name)
            pcaG.to_csv(output_paths['pca'], index_label="i")

            n_clus, s_scores = clustering.plot_Silhouette(pcaG, name, n_dim)
            
            # Cluster processing
            pca_df = pd.read_csv(output_paths['pca'])
            selected_columns = [str(i) for i in range(1, n_dim + 1)]
            clustering_labels, _ = clustering.best_clustering(n_clus, pca_df[selected_columns])
            pca_df.insert(1, "cluster", clustering_labels)
            pca_df["cluster"] = pca_df["cluster"].astype(str)
            
            popp = clustering.kde_c(n_clus, pca_df, selected_columns)
            cluster_dir = output_paths['clusters']
            cluster_dir.mkdir(parents=True, exist_ok=True)
            pdb.exportframeidPDB(str(input_file), popp, str(cluster_dir)+"/")

            # Save analysis info
            with open(output_paths['info'], 'w') as f:
                f.write(f"n_clus = {n_clus}\n"
                       f"n_dim = {n_dim}\n"
                       f"popp = {list(popp)}\n"
                       f"s_scores = {list(s_scores)}\n")
                       
            success_pca = True
            logger.info(f"PCA analysis completed successfully for {name}")
            
        except MemoryError:
            logger.error(f"Memory error during PCA for {name}. File may be too large.")
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            logger.error(f"PCA failed for {name}: {e}")
            logger.error(traceback.format_exc())
            # Force garbage collection
            gc.collect()

        # Torsion analysis
        try:
            logger.info(f"Starting torsion analysis for {name}")
            pairs, external, internal = tfindr.torsionspairs(pdb_data, name)
            torsion_names = dihedral.pairtoname(external, df)
            ext_DF = dihedral.pairstotorsion(external, frames, torsion_names)
            
            torsion_names.extend(["internal"] * len(internal))
            torsion_data_DF = dihedral.pairstotorsion(np.asarray(pairs), frames, torsion_names)
            torsion_data_DF.to_csv(output_paths['torsions'], index_label="i")

            with open(output_paths['info'], 'a' if success_pca else 'w') as f:
                f.write(f"flexibility = {len(external)}\n")
                
            logger.info(f"Torsion analysis completed for {name}")
            
        except MemoryError:
            logger.error(f"Memory error during torsion analysis for {name}. File may be too large.")
            # Force garbage collection
            gc.collect()
            return False
            
        except Exception as e:
            logger.error(f"Torsion analysis failed for {name}: {e}")
            logger.error(traceback.format_exc())
            # Force garbage collection
            gc.collect()
            return False

        elapsed_time = time.time() - start_time
        logger.info(f"Processing of {name} completed in {elapsed_time:.2f} seconds")
        return True
        
    except Exception as e:
        logger.error(f"Unexpected error during processing of {name}: {e}")
        logger.error(traceback.format_exc())
        return False

def main() -> None:
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("Glycan Analysis Pipeline starting")
    logger.info("=" * 80)
    
    try:
        # Check data directory existence and permissions
        data_dir = Path(config.data_dir)
        if not data_dir.exists():
            logger.error(f"Data directory does not exist: {data_dir}")
            return
            
        if not os.access(data_dir, os.R_OK | os.W_OK):
            logger.error(f"Insufficient permissions for data directory: {data_dir}")
            return
            
        logger.info(f"Processing directories in: {data_dir}")
        directories = get_subdirectories(data_dir)
        
        if not directories:
            logger.warning(f"No subdirectories found in {data_dir}")
            return
            
        total_dirs = len(directories)
        processed = 0
        skipped = 0
        failed = 0
        
        for i, directory in enumerate(directories, 1):
            output_dir = directory / 'output'
            logger.info(f"Processing directory {i}/{total_dirs}: {directory.name}")
            
            if output_dir.exists():
                logger.info(f"Skipping: {directory.name} (output directory already exists)")
                skipped += 1
                continue
            
            try:
                success = process_molecule(directory.name)
                if success:
                    processed += 1
                    logger.info(f"Successfully processed: {directory.name}")
                else:
                    failed += 1
                    logger.warning(f"Processing incomplete for: {directory.name}")
            except Exception as e:
                failed += 1
                logger.error(f"Failed to process {directory.name}: {e}")
                logger.error(traceback.format_exc())
            
            # Force garbage collection after each molecule
            gc.collect()
                
        logger.info("=" * 80)
        logger.info(f"Processing complete: {processed} succeeded, {failed} failed, {skipped} skipped")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()