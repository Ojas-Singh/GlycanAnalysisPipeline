from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
import shutil
import os
import traceback
import time

import pandas as pd
import numpy as np

from lib import flip, pdb, clustering, align
import lib.config as config

# Enhanced logging configuration
logger = logging.getLogger(__name__)

def process_pca_clustering(pca_file: Path, n_dim: int, n_clus: int) -> tuple[pd.DataFrame, List[List[Any]]]:
    """Process PCA and clustering for molecule."""
    logger.debug(f"Processing PCA clustering with dimensions={n_dim}, clusters={n_clus}")
    try:
        pca_df = pd.read_csv(pca_file)
        logger.debug(f"Loaded PCA data with shape {pca_df.shape}")
        
        selected_columns = [str(i) for i in range(1, n_dim + 1)]
        logger.debug(f"Selected {len(selected_columns)} columns for clustering: {selected_columns}")
        
        clustering_labels, silhouette = clustering.best_clustering(n_clus, pca_df[selected_columns])
        # Handle silhouette value that might be a numpy array of multiple values
        silhouette_value = silhouette.item() if hasattr(silhouette, 'item') and silhouette.size == 1 else (
            float(silhouette) if isinstance(silhouette, (int, float)) else float(np.mean(silhouette)))
        logger.debug(f"Clustering complete. Silhouette score: {silhouette_value:.4f}")
        
        pca_df.insert(1, "cluster", clustering_labels, False)
        pca_df["cluster"] = pca_df["cluster"].astype(str)
        
        popp = clustering.kde_c(n_clus, pca_df, selected_columns)
        logger.debug(f"KDE clustering generated {len(popp)} representative points")
        
        return pca_df, popp
    except Exception as e:
        logger.error(f"PCA clustering failed: {str(e)}")
        logger.debug(traceback.format_exc())
        raise

def save_cluster_info(output_path: Path, data: Dict[str, Any]) -> None:
    """Save cluster information to JSON."""
    try:
        with open(output_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
        logger.debug(f"Cluster information saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save cluster info to {output_path}: {str(e)}")
        raise

def save_mol2_files(directory: Path, mol2_file: Path) -> None:
    """Save mol2 files to directory."""
    try:
        if not mol2_file.exists():
            logger.warning(f"MOL2 file not found: {mol2_file}")
            return
        
        shutil.copy(mol2_file, directory / f"structure.mol2")
        logger.debug(f"MOL2 file copied to {directory / 'structure.mol2'}")
    except Exception as e:
        logger.error(f"Failed to copy MOL2 file {mol2_file}: {str(e)}")
        logger.debug(traceback.format_exc())

def process_torsions(input_file: Path, clustering_labels: List[int], output_file: Path) -> None:
    """Process and save torsion data."""
    try:
        if not input_file.exists():
            logger.warning(f"Torsion input file not found: {input_file}")
            return
            
        df = pd.read_csv(input_file)
        logger.debug(f"Loaded torsion data with shape {df.shape}")
        
        df.insert(1, "cluster", clustering_labels, False)
        
        cols_to_drop = [col for col in df.columns if "internal" in col]
        logger.debug(f"Dropping {len(cols_to_drop)} internal columns")
        df = df.drop(columns=cols_to_drop)
        
        df.to_csv(output_file, index=False)
        logger.debug(f"Processed torsion data saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to process torsion data: {str(e)}")
        logger.debug(traceback.format_exc())
        raise

def align_cluster_files(temp_dir: Path, final_dir: Path) -> None:
    """Align PDB files and save to final directory."""
    try:
        filenames = list(temp_dir.glob("*.pdb"))
        if not filenames:
            logger.warning(f"No PDB files found in {temp_dir} for alignment")
            return
            
        logger.debug(f"Found {len(filenames)} PDB files for alignment")
        sorted_files = sorted(filenames, key=lambda x: float(x.stem.split("_")[1]), reverse=True)
        
        reference_file = sorted_files[0]
        logger.debug(f"Using reference file for alignment: {reference_file.name}")
        
        source_files = [str(f) for f in sorted_files]
        target_files = [str(final_dir / f.name) for f in sorted_files]
        
        align.align_pdb_files(str(reference_file), source_files, target_files)
        logger.debug(f"Alignment complete. {len(target_files)} files saved to {final_dir}")
    except Exception as e:
        logger.error(f"Failed to align cluster files: {str(e)}")
        logger.debug(traceback.format_exc())
        raise

def process_molecule(directory: Path) -> bool:
    """Process a single molecule directory.
    
    Args:
        directory: Directory containing molecule data
        
    Returns:
        bool: True if processing succeeded, False otherwise
    """
    start_time = time.time()
    clusters_dir = directory / "clusters"
    
    if clusters_dir.exists():
        logger.info(f"Skipping existing directory: {directory.name}")
        return True
        
    logger.info(f"{'='*20} Processing molecule: {directory.name} {'='*20}")
    
    try:
        # Create directory structure
        logger.debug(f"Creating directory structure for {directory.name}")
        for subdir in ["pack", "alpha", "beta"]:
            (clusters_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Read configuration
        logger.debug(f"Reading configuration from info.txt")
        info_file = directory / "output/info.txt"
        if not info_file.exists():
            logger.error(f"Info file not found: {info_file}")
            shutil.rmtree(clusters_dir)  # Clean up on failure
            logger.debug(f"Removed incomplete clusters directory: {clusters_dir}")
            return False
            
        with open(info_file) as f:
            config_vars = {}
            for line in f:
                exec(line, None, config_vars)
                
        logger.debug(f"Configuration loaded: dims={config_vars.get('n_dim')}, clusters={config_vars.get('n_clus')}")

        # Copy files
        logger.debug("Copying analysis files to pack directory")
        for file in ["torparts.npz", "PCA_variance.png", "Silhouette_Score.png"]:
            source_file = directory / f"output/{file}"
            if source_file.exists():
                shutil.copy(source_file, clusters_dir / f"pack/{file}")
                logger.debug(f"Copied {file}")
            else:
                logger.warning(f"File not found: {source_file}")
            
        # Determine molecule type and set paths
        logger.debug(f"Determining molecule type for {directory.name}")
        if len(directory.name) == 8:  # GlyTouCan ID
            logger.debug(f"Processing as GlyTouCan ID: {directory.name}")
            json_file = directory / f"{directory.name}.json"
            if not json_file.exists():
                logger.warning(f"JSON file not found: {json_file}")
                is_alpha = False  # Default to beta if can't determine
            else:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                glycam = data.get("indexOrderedSequence", "output")
                is_alpha = flip.is_alpha(glycam)
                logger.debug(f"Molecule determined to be {'alpha' if is_alpha else 'beta'} anomer")
        else:
            is_alpha = flip.is_alpha(directory.name)
            logger.debug(f"Based on name, molecule is {'alpha' if is_alpha else 'beta'} anomer")
            
        temp_dir = clusters_dir / ("alpha_temp" if is_alpha else "beta_temp")
        final_dir = clusters_dir / ("alpha" if is_alpha else "beta")
        flip_dir = clusters_dir / ("beta" if is_alpha else "alpha")
        temp_dir.mkdir(exist_ok=True)
        
        # Process PCA and clustering
        logger.info(f"Processing PCA and clustering for {directory.name}")
        pca_file = directory / "output/pca.csv"
        if not pca_file.exists():
            logger.error(f"PCA file not found: {pca_file}")
            shutil.rmtree(clusters_dir)  # Clean up on failure
            logger.debug(f"Removed incomplete clusters directory: {clusters_dir}")
            return False
            
        try:
            pca_df, popp = process_pca_clustering(
                pca_file,
                config_vars.get("n_dim", 3),
                config_vars.get("n_clus", 3)
            )
            logger.info(f"PCA clustering complete with {len(popp)} representatives")
        except Exception as e:
            logger.error(f"PCA clustering failed: {str(e)}")
            logger.debug(traceback.format_exc())
            shutil.rmtree(clusters_dir)  # Clean up on failure
            logger.debug(f"Removed incomplete clusters directory: {clusters_dir}")
            return False

        # Export cluster representatives
        logger.info(f"Exporting cluster representative structures")
        pdb_file = directory / f"{directory.name}.pdb"
        if not pdb_file.exists():
            logger.error(f"PDB file not found: {pdb_file}")
            shutil.rmtree(clusters_dir)  # Clean up on failure
            logger.debug(f"Removed incomplete clusters directory: {clusters_dir}")
            return False
            
        try:
            pdb.exportframeidPDB(str(pdb_file), popp, str(temp_dir)+"/")
            logger.debug(f"Exported {len(popp)} representative structures to {temp_dir}")
        except Exception as e:
            logger.error(f"Failed to export PDB structures: {str(e)}")
            shutil.rmtree(clusters_dir)  # Clean up on failure
            logger.debug(f"Removed incomplete clusters directory: {clusters_dir}")
            return False

        # Save cluster info
        logger.info(f"Saving cluster information")
        try:
            sorted_popp = sorted(popp, key=lambda x: int(x[1]))
            data = {
                "n_clus": config_vars.get("n_clus", 3),
                "n_dim": config_vars.get("n_dim", 3),
                "popp": [int(item[0]) for item in sorted_popp],
                "s_scores": [float(s) for s in config_vars.get("s_scores", [])],
                "flexibility": int(config_vars.get("flexibility", 0))
            }
            save_cluster_info(clusters_dir / "pack/info.json", data)
            logger.debug(f"Cluster info saved with {len(sorted_popp)} clusters")
        except Exception as e:
            logger.error(f"Failed to save cluster info: {str(e)}")
            shutil.rmtree(clusters_dir)  # Clean up on failure
            logger.debug(f"Removed incomplete clusters directory: {clusters_dir}")
            return False
            
        # Save MOL2 files
        logger.info(f"Copying MOL2 files")
        mol2_file = directory / f"{directory.name}.mol2"
        save_mol2_files(clusters_dir / "pack", mol2_file)

        # Process torsions
        logger.info(f"Processing torsion data")
        try:
            process_torsions(
                directory / "output/torsions.csv",
                pca_df["cluster"],
                clusters_dir / "pack/torsions.csv"
            )
            logger.debug(f"Torsion processing complete")
        except Exception as e:
            logger.error(f"Failed to process torsion data: {str(e)}")
            # Continue despite torsion errors - not critical for basic operation

        # Save simplified PCA data
        logger.info(f"Saving simplified PCA data")
        try:
            columns = ["0", "1", "2", "i", "cluster"]
            available_cols = [col for col in columns if col in pca_df.columns]
            if len(available_cols) < 3:
                logger.warning(f"Insufficient columns for PCA data: {available_cols}")
            else:
                pca_df[available_cols].to_csv(clusters_dir / "pack/pca.csv", index=False)
                logger.debug(f"Simplified PCA data saved with columns: {available_cols}")
        except Exception as e:
            logger.error(f"Failed to save simplified PCA data: {str(e)}")
            # Continue despite PCA errors - not critical for basic operation

        # Align cluster files
        logger.info(f"Aligning cluster structures")
        try:
            if not list(temp_dir.glob("*.pdb")):
                logger.warning(f"No PDB files found in {temp_dir} for alignment")
                shutil.rmtree(clusters_dir)  # Clean up on failure - critical step
                logger.debug(f"Removed incomplete clusters directory: {clusters_dir}")
                return False
            else:
                align_cluster_files(temp_dir, final_dir)
                logger.debug(f"Structure alignment complete")
                shutil.rmtree(temp_dir)
                logger.debug(f"Removed temporary directory: {temp_dir}")
        except Exception as e:
            logger.error(f"Failed to align structures: {str(e)}")
            shutil.rmtree(clusters_dir)  # Clean up on failure
            logger.debug(f"Removed incomplete clusters directory: {clusters_dir}")
            return False

        # Flip structures
        logger.info(f"Flipping {'alpha' if is_alpha else 'beta'} structures")
        try:
            file_count = 0
            for pdb_file in final_dir.glob("*.pdb"):
                flip.flip_alpha_beta(str(pdb_file), str(flip_dir / pdb_file.name))
                file_count += 1
                
            if file_count == 0:
                logger.warning(f"No structures were flipped - no files in {final_dir}")
                # This is concerning but not fatal - we continue
            else:
                logger.info(f"Structure flipping completed successfully for {file_count} files")
        except Exception as e:
            logger.error(f"Failed to flip structures: {str(e)}")
            logger.debug(traceback.format_exc())
            # We don't delete the cluster dir for flipping failures
            # as this is not critical - the primary conformer is still available

        elapsed_time = time.time() - start_time
        logger.info(f"{'='*20} Completed {directory.name} in {elapsed_time:.2f} seconds {'='*20}")
        return True
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"{'X'*20} Failed to process {directory.name} after {elapsed_time:.2f} seconds: {str(e)}")
        logger.debug(traceback.format_exc())
        
        # Clean up on any unexpected failure
        if clusters_dir.exists():
            try:
                shutil.rmtree(clusters_dir)
                logger.debug(f"Removed incomplete clusters directory: {clusters_dir}")
            except Exception as cleanup_error:
                logger.error(f"Failed to clean up clusters directory: {cleanup_error}")
                
        return False

def main() -> None:
    """Main execution function."""
    start_time = time.time()
    data_dir = Path(config.data_dir)
    
    # Check if data directory exists
    if not data_dir.exists() or not data_dir.is_dir():
        logger.error(f"Data directory does not exist or is not a directory: {data_dir}")
        return
        
    logger.info(f"{'='*30} Reclustering Analysis Started {'='*30}")
    logger.info(f"Processing directories in: {data_dir}")
    
    directories = [d for d in data_dir.iterdir() if d.is_dir()]
    total_dirs = len(directories)
    
    if total_dirs == 0:
        logger.warning(f"No directories found in {data_dir}")
        return
        
    logger.info(f"Found {total_dirs} directories to process")
    
    successful = 0
    failed = 0
    skipped = 0
    
    for i, directory in enumerate(directories, 1):
        logger.info(f"Processing directory {i}/{total_dirs}: {directory.name}")
        
        if (directory / "clusters").exists():
            logger.info(f"Skipping: {directory.name} (already processed)")
            skipped += 1
            continue
            
        success = process_molecule(directory)
        if success:
            successful += 1
        else:
            failed += 1
    
    elapsed_time = time.time() - start_time
    logger.info(f"{'='*30} Reclustering Analysis Complete {'='*30}")
    logger.info(f"Total time: {elapsed_time:.2f} seconds")
    logger.info(f"Directories: {total_dirs} total, {successful} successful, {failed} failed, {skipped} skipped")

if __name__ == "__main__":
    main()