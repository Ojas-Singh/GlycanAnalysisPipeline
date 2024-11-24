from pathlib import Path
from typing import List, Dict, Any
import logging
import json
import shutil

import pandas as pd

from lib import flip, pdb, clustering, align
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_pca_clustering(pca_file: Path, n_dim: int, n_clus: int) -> tuple[pd.DataFrame, List[List[Any]]]:
    """Process PCA and clustering for molecule."""
    pca_df = pd.read_csv(pca_file)
    selected_columns = [str(i) for i in range(1, n_dim + 1)]
    clustering_labels, _ = clustering.best_clustering(n_clus, pca_df[selected_columns])
    pca_df.insert(1, "cluster", clustering_labels, False)
    pca_df["cluster"] = pca_df["cluster"].astype(str)
    popp = clustering.kde_c(n_clus, pca_df, selected_columns)
    return pca_df, popp

def save_cluster_info(output_path: Path, data: Dict[str, Any]) -> None:
    """Save cluster information to JSON."""
    with open(output_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

def process_torsions(input_file: Path, clustering_labels: List[int], output_file: Path) -> None:
    """Process and save torsion data."""
    df = pd.read_csv(input_file)
    df.insert(1, "cluster", clustering_labels, False)
    cols_to_drop = [col for col in df.columns if "internal" in col]
    df = df.drop(columns=cols_to_drop)
    df.to_csv(output_file, index=False)

def align_cluster_files(temp_dir: Path, final_dir: Path) -> None:
    """Align PDB files and save to final directory."""
    filenames = [f for f in temp_dir.glob("*.pdb")]
    sorted_files = sorted(filenames, key=lambda x: float(x.stem.split("_")[1]), reverse=True)
    
    if not sorted_files:
        raise ValueError("No PDB files found for alignment")
        
    reference_file = sorted_files[0]
    align.align_pdb_files(str(reference_file), [str(f) for f in sorted_files], 
                         [str(final_dir / f.name) for f in sorted_files])

def process_molecule(directory: Path) -> None:
    """Process a single molecule directory."""
    clusters_dir = directory / "clusters"
    if clusters_dir.exists():
        logger.info(f"Skipping existing directory: {directory.name}")
        return

    logger.info(f"Processing directory: {directory.name}")
    try:
        # Create directory structure
        for subdir in ["pack", "alpha", "beta"]:
            (clusters_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Read configuration
        with open(directory / "output/info.txt") as f:
            config_vars = {}
            for line in f:
                exec(line, None, config_vars)

        # Determine molecule type and set paths
        is_alpha = flip.is_alpha(directory.name)
        temp_dir = clusters_dir / ("alpha_temp" if is_alpha else "beta_temp")
        final_dir = clusters_dir / ("alpha" if is_alpha else "beta")
        flip_dir = clusters_dir / ("beta" if is_alpha else "alpha")
        temp_dir.mkdir(exist_ok=True)

        # Copy files
        for file in ["torparts.npz", "PCA_variance.png", "Silhouette_Score.png"]:
            shutil.copy(directory / f"output/{file}", clusters_dir / f"pack/{file}")

        # Process PCA and clustering
        pca_df, popp = process_pca_clustering(
            directory / "output/pca.csv",
            config_vars["n_dim"],
            config_vars["n_clus"]
        )

        # Export cluster representatives
        pdb.exportframeidPDB(str(directory / f"{directory.name}.pdb"), popp, str(temp_dir)+"/")

        # Save cluster info
        sorted_popp = sorted(popp, key=lambda x: int(x[1]))
        data = {
            "n_clus": config_vars["n_clus"],
            "n_dim": config_vars["n_dim"],
            "popp": [int(item[0]) for item in sorted_popp],
            "s_scores": [float(s) for s in config_vars["s_scores"]],
            "flexibility": int(config_vars["flexibility"])
        }
        save_cluster_info(clusters_dir / "pack/info.json", data)

        # Process torsions
        process_torsions(
            directory / "output/torsions.csv",
            pca_df["cluster"],
            clusters_dir / "pack/torsions.csv"
        )

        # Save simplified PCA data
        pca_df[["0", "1", "2", "i", "cluster"]].to_csv(
            clusters_dir / "pack/pca.csv", index=False
        )

        # Align cluster files
        align_cluster_files(temp_dir, final_dir)
        shutil.rmtree(temp_dir)

        # Flip structures
        try:
            for pdb_file in final_dir.glob("*.pdb"):
                flip.flip_alpha_beta(str(pdb_file), str(flip_dir / pdb_file.name))
            logger.info("Structure flipping completed successfully")
        except Exception as e:
            logger.error(f"Failed to flip structures: {e}")
            # try:
            #     for pdb_file in final_dir.glob("*.pdb"):
            #         flip.flip_c2_connected_atoms(str(pdb_file), str(flip_dir / pdb_file.name))
            #     logger.info("Structure flipping completed successfully using c2")
            # except Exception as e:
            #     logger.error(f"Failed to flip structures with c2: {e}")
    except Exception as e:
        logger.error(f"Failed to process {directory.name}: {e}")
        pass

def main() -> None:
    """Main execution function."""
    data_dir = Path(config.data_dir)
    logger.info(f"Processing directories in: {data_dir}")
    
    for directory in data_dir.iterdir():
        if directory.is_dir():
            process_molecule(directory)

if __name__ == "__main__":
    main()