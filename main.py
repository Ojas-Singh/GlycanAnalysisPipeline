from pathlib import Path
from typing import List, Tuple, Optional
import logging
import traceback

import numpy as np
import pandas as pd
from lib import pdb, dihedral, clustering, tfindr
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_first_frame(input_path: Path, output_path: Path) -> None:
    """Extract the first model frame from a PDB file.

    Args:
        input_path: Path to input PDB file
        output_path: Path to save the extracted frame
    """
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

def get_subdirectories(folder_path: Path) -> List[Path]:
    """List all subdirectories in the given path.

    Args:
        folder_path: Path to parent directory

    Returns:
        List of subdirectory paths
    """
    return [d for d in folder_path.iterdir() if d.is_dir()]

def process_molecule(name: str) -> None:
    """Process molecular calculations for the given structure.
    
    Args:
        name: Name of the molecule/directory to process
    """
    data_dir = Path(config.data_dir)
    input_file = data_dir / name / f"{name}.pdb"
    output_dir = data_dir / name / "output"
    output_paths = {
        'structure': output_dir / "structure.pdb",
        'pca': output_dir / "pca.csv",
        'torsions': output_dir / "torsions.csv",
        'info': output_dir / "info.txt",
        'clusters': output_dir / "cluster_default/" 
    }

    # Extract first frame
    extract_first_frame(input_file, output_paths['structure'])

    # Load and prepare data
    pdb_data, frames = pdb.multi(str(input_file))
    df = pdb.to_DF(pdb_data)
    idx_noH = df.loc[df['Element'] != "H", 'Number'] - 1

    try:
        # PCA and clustering analysis
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

    except Exception as e:
        logger.error(f"PCA failed: {traceback.format_exc()}")

    # Torsion analysis
    pairs, external, internal = tfindr.torsionspairs(pdb_data, name)
    torsion_names = dihedral.pairtoname(external, df)
    ext_DF = dihedral.pairstotorsion(external, frames, torsion_names)
    
    torsion_names.extend(["internal"] * len(internal))
    torsion_data_DF = dihedral.pairstotorsion(np.asarray(pairs), frames, torsion_names)
    torsion_data_DF.to_csv(output_paths['torsions'], index_label="i")

    with open(output_paths['info'], 'a') as f:
        f.write(f"flexibility = {len(external)}\n")

def main() -> None:
    """Main execution function."""
    data_dir = Path(config.data_dir)
    directories = get_subdirectories(data_dir)
    
    logger.info(f"Processing directories in: {data_dir}")
    
    for directory in directories:
        output_dir = directory / 'output'
        if output_dir.exists():
            logger.info(f"Skipping: {directory.name}")
            continue
            
        logger.info(f"Processing: {directory.name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            process_molecule(directory.name)
            logger.info("Success!")
        except Exception:
            logger.error(f"Failed: {traceback.format_exc()}")

if __name__ == "__main__":
    main()