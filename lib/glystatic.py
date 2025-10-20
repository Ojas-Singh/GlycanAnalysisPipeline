#!/usr/bin/env python3

import json
import shutil
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
import re

from glycowork.motif.processing import canonicalize_iupac
from glycowork.motif.tokenization import glycan_to_composition
from glycowork.motif.draw import GlycoDraw

import logging
from lib import glypdbio, config, name_utils
from lib.storage import get_storage_manager
import warnings
from urllib3.exceptions import InsecureRequestWarning

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress only the specific InsecureRequestWarning
warnings.filterwarnings("ignore", category=InsecureRequestWarning)


def get_glycan_metadata_from_inventory(glycam_name: str) -> dict:
    """
    Get glycan metadata from the inventory CSV file.
    
    Args:
        glycam_name: GLYCAM name of the glycan
    
    Returns:
        Dictionary containing metadata from the inventory
    """
    try:
        # Read the inventory CSV (works for local and Oracle PAR)
        storage_manager = get_storage_manager()
        inv_path = str(config.inventory_path)
        if not storage_manager.exists(inv_path):
            # Fallback: try to locate the inventory CSV anywhere in the bucket
            candidates = []
            try:
                # Root listing (Oracle backend ignores pattern and uses prefix)
                for obj in storage_manager.list_files("", "*"):
                    name = obj.as_posix() if hasattr(obj, 'as_posix') else str(obj)
                    if name.lower().endswith("glycoshape_inventory.csv"):
                        candidates.append(name)
            except Exception:
                pass
            if candidates:
                # Prefer shortest path (closest to root)
                candidates.sort(key=lambda s: len(s))
                inv_path = candidates[0]
            else:
                raise FileNotFoundError(f"Inventory file not found: {inv_path}")
        # Read via storage layer to avoid direct FS coupling
        csv_bytes = storage_manager.read_binary(inv_path)
        from io import BytesIO
        df = pd.read_csv(BytesIO(csv_bytes))
        
        # Search for the glycam name in the inventory
        # The column name is 'Full GLYCAM name of glycan being submitted.'
        glycam_col = 'Full GLYCAM name of glycan being submitted.'
        
        # Find matching row
        matching_rows = df[df[glycam_col] == glycam_name]
        
        if matching_rows.empty:
            raise ValueError(f"Glycam name '{glycam_name}' not found in inventory")
        
        if len(matching_rows) > 1:
            logger.warning(f"Multiple entries found for {glycam_name}, using the first one")
        
        # Get the first matching row
        row = matching_rows.iloc[0]
        
        # Extract metadata
        metadata = {
            "ID": row['ID'],
            "timestamp": row['Timestamp'],
            "email": row['Email address'],
            "glycam_name": row[glycam_col],
            "transfer_method": row['How will the data be transferred?'],
            "length": row['What is the aggregated length of the simulations?'],
            "package": row['What MD package was used for the simulations?'],
            "forcefield": row['What force field was used for the simulations?'],
            "temperature": row['What temperature target was used for the simulations? '],
            "pressure": row['What pressure target was used for the simulations?'],
            "salt": row['What NaCl concentration was used for the simulations?'],
            "comments": row['Any comments that should be noted with the submission?'],
            "glytoucan_id": row['What is the GlyTouCan ID of the glycan?']
        }
        
        logger.info(f"Found metadata for {glycam_name}: ID={metadata['ID']}")
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to get metadata from inventory for {glycam_name}: {e}")
        raise


def _emit_multimodel_pdbs(dest_level_dir: str, source_level_dir: str, cluster_data: dict, level_name: str, anomer_meta: dict | None = None) -> None:
    """Delegate multi-model emission to glypdbio with remarks support."""
    try:
        glypdbio.emit_multimodel_pdbs(
            source_level_dir=source_level_dir,
            dest_level_dir=dest_level_dir,
            cluster_data=cluster_data,
            level_name=level_name,
            anomer_meta=anomer_meta,
            formats=["GLYCAM", "PDB"],
        )
    except Exception as e:
        logger.warning(f"Failed to emit multi-model PDBs for {level_name}: {e}")


def generate_glycan_id(glycam_name: str) -> str:
    """Generate a simple ID from glycam name as fallback."""
    # Simple ID generation - you can make this more sophisticated if needed
    import hashlib
    # Create a hash of the glycam name for a consistent ID
    hash_obj = hashlib.md5(glycam_name.encode())
    return f"GS{hash_obj.hexdigest()[:6].upper()}"


def load_info_json(folder_path) -> dict:
    """Load and parse the info.json file from the output directory."""
    storage = get_storage_manager()
    info_path = f"{folder_path}/output/info.json"
    if not storage.exists(info_path):
        raise FileNotFoundError(f"info.json not found at {info_path}")
    
    with storage.open(info_path, 'r') as f:
        return json.load(f)


def extract_cluster_data(info_data: dict, folder_path: str) -> dict:
    """Extract cluster data from info.json and organize by levels."""
    storage = get_storage_manager()
    cluster_data = {
        "entropy_nats": info_data.get("entropy_nats"),
        "torsions": info_data.get("torsions", {}),
        "levels": {},
        "pca": info_data.get("pca", {})
    }
    
    # Process each level
    for level_info in info_data.get("levels", []):
        level_num = level_info["level"]
        
        # Extract cluster information for this level
        clusters = []
        for cluster in level_info.get("clusters", []):
            cluster_info = {
                "cluster_id": cluster["cluster_id"],
                "cluster_size_pct": cluster["cluster_size_pct"],
                "representative_idx": cluster["representative_idx"],
                "members_count": cluster["members_count"],
                "torsion_stats": cluster.get("torsion_stats", {})
            }
            
            # Add PDB file paths for alpha and beta anomers
            level_dir = f"{folder_path}/output/level_{level_num}"
            alpha_dir = f"{level_dir}/alpha"
            beta_dir = f"{level_dir}/beta"
            
            cluster_info["pdb_files"] = {
                "alpha": None,
                "beta": None
            }
            
            # Find corresponding PDB files
            if storage.exists(alpha_dir):
                alpha_files = [p for p in storage.list_files(alpha_dir, f"cluster_{cluster['cluster_id']}_*.pdb")]
                if alpha_files:
                    cluster_info["pdb_files"]["alpha"] = str(alpha_files[0])
                    
            if storage.exists(beta_dir):
                beta_files = [p for p in storage.list_files(beta_dir, f"cluster_{cluster['cluster_id']}_*.pdb")]
                if beta_files:
                    cluster_info["pdb_files"]["beta"] = str(beta_files[0])
            
            clusters.append(cluster_info)
        
        cluster_data["levels"][f"level_{level_num}"] = {
            "n_clusters": level_info["n_clusters"],
            "clusters": clusters
        }
    
    return cluster_data


def copy_and_convert_pdbs(
    folder_path: str, 
    output_dir: str, 
    glycam_name: str,
    archetype_glytoucan: str = None,
    archetype_iupac: str = None,
    alpha_glytoucan: str = None,
    alpha_iupac: str = None,
    beta_glytoucan: str = None,
    beta_iupac: str = None,
    cluster_data: dict = None
):
    """Copy PDB files and convert them to different formats using glypdbio.
    
    Parameters:
        folder_path: Path to the source folder containing PDB files
        output_dir: Path to the output directory
        glycam_name: GLYCAM name of the glycan
        archetype_glytoucan: GlyTouCan ID for the archetype form
        archetype_iupac: IUPAC name for the archetype form
        alpha_glytoucan: GlyTouCan ID for the alpha anomer
        alpha_iupac: IUPAC name for the alpha anomer
        beta_glytoucan: GlyTouCan ID for the beta anomer
        beta_iupac: IUPAC name for the beta anomer
        cluster_data: Dictionary containing cluster information from info.json
    """
    storage = get_storage_manager()
    source_output = Path(folder_path) / "output"

    # Determine levels from info.json cluster_data to be backend-agnostic
    level_names: list[str] = []
    if isinstance(cluster_data, dict) and "levels" in cluster_data and isinstance(cluster_data["levels"], dict):
        level_names = sorted([k for k in cluster_data["levels"].keys() if k.startswith("level_")])
    # Fallback to filesystem probing only if necessary
    if not level_names:
        try:
            # Best-effort: collect unique names under output/ that look like level_* prefixes
            all_objs = storage.list_files(source_output, "*")
            seen = set()
            for obj in all_objs:
                parts = Path(obj).as_posix().split('/')
                try:
                    idx = parts.index("output")
                    name = parts[idx+1]
                    if name.startswith("level_"):
                        seen.add(name)
                except Exception:
                    continue
            level_names = sorted(seen)
        except Exception:
            level_names = []

    for level_name in level_names:
        dest_level_dir = Path(output_dir) / "output" / level_name
        storage.mkdir(dest_level_dir)

        # Copy dist.svg if it exists
        dist_file = source_output / level_name / "dist.svg"
        if storage.exists(dist_file):
            # Use storage abstraction to read and write
            content = storage.read_binary(dist_file)
            storage.write_binary(dest_level_dir / "dist.svg", content)

        # Build multi-model PDBs for both level_1 and level_2; do not emit per-cluster converted PDBs
        try:
            anomer_meta = {
                "alpha": {"glytoucan": alpha_glytoucan or archetype_glytoucan, "iupac": alpha_iupac or archetype_iupac},
                "beta": {"glytoucan": beta_glytoucan or archetype_glytoucan, "iupac": beta_iupac or archetype_iupac},
            }
            _emit_multimodel_pdbs(
                dest_level_dir=dest_level_dir,
                source_level_dir=source_output / level_name,
                cluster_data=cluster_data,
                level_name=level_name,
                anomer_meta=anomer_meta,
            )
        except Exception as e:
            logger.warning(f"Failed to build multi-model PDBs for {level_name}: {e}")


def make_serializable(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    else:
        return obj


def save_glycan_data(glycan_data: dict, output_dir, filename: str) -> None:
    """Save glycan data as JSON file."""
    try:
        storage = get_storage_manager()
        storage.mkdir(output_dir)
        output_file = f"{output_dir}/{filename}.json"
        serializable_data = make_serializable(glycan_data)
        with storage.open(output_file, 'w') as f:
            json.dump(serializable_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved glycan data to {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to save glycan data: {str(e)}")
        raise


def process_glycan(folder_path: str, glycam_name: str, output_static_dir: str) -> None:
    """
    Process a single glycan folder and generate static data.
    
    Args:
        folder_path: Path to the glycan folder containing output/ subdirectory
        glycam_name: GLYCAM name of the glycan
        output_static_dir: Directory where static data will be created
    """
    from lib.storage import get_storage_manager
    
    folder_path = Path(folder_path)
    output_dir = Path(output_static_dir)
    storage = get_storage_manager()
    
    logger.info(f"Processing glycan: {glycam_name} from {folder_path}")
    
    # Use storage abstraction layer for validation
    storage = get_storage_manager()
    
    # Validate input using storage abstraction
    if not storage.exists(folder_path):
        raise FileNotFoundError(f"Folder path does not exist: {folder_path}")
    
    output_subdir = folder_path / "output"
    if not storage.exists(output_subdir):
        raise FileNotFoundError(f"Output subdirectory not found: {output_subdir}")
    
    # Load analysis data from info.json
    try:
        info_data = load_info_json(folder_path)
        cluster_data = extract_cluster_data(info_data, folder_path)
    except Exception as e:
        logger.error(f"Failed to load info.json: {e}")
        raise
    
    # Get metadata from inventory CSV
    try:
        metadata = get_glycan_metadata_from_inventory(glycam_name)
        ID = metadata["ID"]
        length = metadata["length"]
        package = metadata["package"]
        forcefield = metadata["forcefield"]
        temperature = metadata["temperature"]
        pressure = metadata["pressure"]
        salt = metadata["salt"]
        comments = metadata["comments"]
        inventory_glytoucan = metadata["glytoucan_id"]
    except Exception as e:
        logger.warning(f"Failed to get metadata from inventory: {e}")
        logger.info("Using fallback ID generation")
        ID = generate_glycan_id(glycam_name)
        length = package = forcefield = temperature = pressure = salt = comments = inventory_glytoucan = None
    
    logger.info(f"Using ID: {ID}")
    
    # Create output directory
    final_output_dir = f"{output_dir}/{ID}"
    # Remove existing directory if it exists (for both local and Oracle)
    if storage.exists(final_output_dir):
        if hasattr(storage, '_local_backend') and storage._local_backend.exists(final_output_dir):
            # For local backend, remove the directory
            import shutil
            shutil.rmtree(storage._local_backend._resolve_path(final_output_dir))
        # For Oracle, individual files will be overwritten
    storage.mkdir(final_output_dir)
    
    # Get basic glycan info
    glycam_tidy, end_pos, end_link = name_utils.glycam_info(glycam_name)
    iupac = canonicalize_iupac(glycam_name)
    
    # Generate SNFG diagram
    try:
        # For Oracle storage, generate to temp file first then upload
        if hasattr(storage, '_is_oracle') and storage._is_oracle:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as tmp_file:
                temp_path = tmp_file.name
            snfg = GlycoDraw(iupac, filepath=temp_path, show_linkage=True)
            # Read the temp file and upload to storage
            with open(temp_path, 'rb') as f:
                svg_content = f.read()
            storage.write_binary(f"{final_output_dir}/snfg.svg", svg_content)
            # Clean up temp file
            import os
            os.unlink(temp_path)
            logger.info(f"Generated SNFG diagram: {final_output_dir}/snfg.svg")
        else:
            # Local storage - write directly
            snfg_svg_path = storage._local_backend._resolve_path(f"{final_output_dir}/snfg.svg")
            snfg = GlycoDraw(iupac, filepath=str(snfg_svg_path), show_linkage=True)
            logger.info(f"Generated SNFG diagram: {snfg_svg_path}")
    except Exception as e:
        logger.warning(f"Could not generate SNFG diagram: {e}")
    
    # Process glycan chemical data
    try:
        
        iupac_alpha = f"{iupac}(a{end_pos}-"
        iupac_beta = f"{iupac}(b{end_pos}-"
        glycam_alpha = f"{glycam_tidy}a{end_pos}-OH"
        glycam_beta = f"{glycam_tidy}b{end_pos}-OH"
        
        logger.info(f"Processing glycan: {glycam_name}, IUPAC: {iupac}, GlyCAM: {glycam_tidy}")
        
        # Get chemical properties
        try:
            mass, tpsa, rot_bonds, hbond_donor, hbond_acceptor = name_utils.iupac2properties(iupac)
        except Exception as e:
            logger.warning(f"Could not retrieve properties for {iupac}: {e}")
            mass, tpsa, rot_bonds, hbond_donor, hbond_acceptor = None, None, None, None, None

        try:
            composition_glycowork = glycan_to_composition(iupac)
        except Exception:
            composition_glycowork = None

        # Get additional data
        smiles = name_utils.iupac2smiles(iupac)
        glycoct = name_utils.iupac2glycoct(iupac)
        composition = name_utils.iupac2composition(iupac)
        oxford = name_utils.get_oxford(iupac)
        
        try:
            termini = name_utils.iupac2termini(iupac)
        except Exception as e:
            logger.warning(f"Could not retrieve termini for {iupac}: {e}")
            termini = None
        
        # Get WURCS and GlyTouCan IDs
        glytoucan, wurcs = name_utils.iupac2wurcs_glytoucan(iupac)
        glytoucan_alpha, wurcs_alpha = name_utils.iupac2wurcs_glytoucan(iupac_alpha)
        glytoucan_beta, wurcs_beta = name_utils.iupac2wurcs_glytoucan(iupac_beta)

        # If wurcs is None, try to get it from Mol2WURCS using smiles
        if wurcs is None:
            try:
                logger.info("WURCS is None, trying to get it from Mol2WURCS")
                wurcs = name_utils.smiles2wurcs(smiles)
                wurcs, wurcs_alpha, wurcs_beta = name_utils.get_wurcs_variants(wurcs)
                glytoucan = name_utils.wurcs2glytoucan(wurcs)
                glytoucan_alpha = name_utils.wurcs2glytoucan(wurcs_alpha)
                glytoucan_beta = name_utils.wurcs2glytoucan(wurcs_beta)
            except Exception as e:
                logger.warning(f"Could not retrieve WURCS from Mol2WURCS for {iupac}: {e}")
                wurcs = wurcs_alpha = wurcs_beta = None
                glytoucan = glytoucan_alpha = glytoucan_beta = None

        # Get motifs
        try: 
            motifs = name_utils.glytoucan2motif(glytoucan) if glytoucan else None
        except Exception as e:
            logger.warning(f"Could not retrieve motifs for {iupac}: {e}")
            motifs = None
        
        # Calculate molecular weight
        mol_weight = None
        if smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    mol_weight = round(Descriptors.ExactMolWt(mol), 2)
            except Exception as e:
                logger.warning(f"Could not calculate molecular weight: {e}")
        
        # Extract level 1 (main) and level 2 (coverage) cluster data
        level_1_main_clusters = {}
        level_2_coverage_clusters = {}
        
        # Extract PCA clustering metadata (silhouette scores, coverage mapping)
        silhouette_scores = {}
        coverage_clusters_per_main = {}
        pca_variance = {}
        
        if "pca" in cluster_data:
            pca_data = cluster_data["pca"]
            if "main_clustering" in pca_data:
                silhouette_scores = pca_data["main_clustering"].get("silhouette_scores", {})
                coverage_clusters_per_main = pca_data["main_clustering"].get("coverage_clusters_per_main", {})
            
            # Extract PCA variance information
            if "pca_analysis" in pca_data:
                pca_analysis = pca_data["pca_analysis"]
                pca_variance = {
                    "explained_variance_ratio": pca_analysis.get("explained_variance_ratio", []),
                    "cumulative_variance": pca_analysis.get("cumulative_variance", []),
                    "n_components": pca_analysis.get("n_components", 10)
                }
        
        if "levels" in cluster_data:
            for level_key, level_data in cluster_data["levels"].items():
                if level_key == "level_1":  # Main clustering (5 clusters)
                    for cluster in level_data.get("clusters", []):
                        cluster_id = cluster["cluster_id"]
                        cluster_name = f"Cluster {cluster_id}"
                        level_1_main_clusters[cluster_name] = round(cluster["cluster_size_pct"], 2)
                elif level_key == "level_2":  # Coverage clustering
                    for cluster in level_data.get("clusters", []):
                        cluster_id = cluster["cluster_id"]
                        cluster_name = f"Cluster {cluster_id}"
                        level_2_coverage_clusters[cluster_name] = round(cluster["cluster_size_pct"], 4)
        
        # Build glycan data structure
        glycan_data = {
            "archetype": {
                # General info
                "ID": ID,
                "name": glycam_name,
                "glycam": glycam_tidy,
                "iupac": iupac,
                "iupac_extended": name_utils.wurcs2extendediupac(wurcs),
                "glytoucan": glytoucan,
                "wurcs": wurcs,
                "glycoct": glycoct,
                "smiles": smiles,
                "oxford": oxford,

                # Chemical properties
                "mass": mol_weight,
                "motifs": motifs,
                "termini": termini,
                "components": composition,
                "composition": composition_glycowork,
                "rot_bonds": rot_bonds, 
                "hbond_donor": hbond_donor, 
                "hbond_acceptor": hbond_acceptor,
                
                # cluster analysis data 
                "entropy": cluster_data.get("entropy_nats"),
                "clusters": level_1_main_clusters,
                "coverage_clusters": level_2_coverage_clusters,
                "silhouette_scores": silhouette_scores,
                "coverage_clusters_per_main": coverage_clusters_per_main,
                "pca_variance": pca_variance,
                
                # Molecular Dynamics info from inventory
                "length": length,
                "package": package,
                "forcefield": forcefield,
                "temperature": temperature,
                "pressure": pressure,
                "salt": salt,
            },

            "alpha": {
                # General info
                "ID": ID,
                "name": glycam_name,
                "glycam": glycam_alpha,
                "iupac": iupac_alpha,
                "iupac_extended": name_utils.wurcs2extendediupac(wurcs_alpha),
                "glytoucan": glytoucan_alpha,
                "wurcs": wurcs_alpha,
                
            },
            
            "beta": {
                # General info  
                "ID": ID,
                "name": glycam_name,
                "glycam": glycam_beta,
                "iupac": iupac_beta,
                "iupac_extended": name_utils.wurcs2extendediupac(wurcs_beta),
                "glytoucan": glytoucan_beta,
                "wurcs": wurcs_beta,
                
                
            },

            "search_meta": {
                "common_names": [],
                "description": "",
                "keywords": [],
            }
        }

        # Save data files
        save_glycan_data(glycan_data, final_output_dir, "data")
        save_glycan_data(name_utils.glytoucan2glygen(glytoucan), final_output_dir, "glygen")
        save_glycan_data(name_utils.glytoucan2glycosmos(glytoucan), final_output_dir, "glycosmos")
        
        # Copy and convert PDB files
        copy_and_convert_pdbs(
            folder_path, 
            final_output_dir, 
            glycam_name,
            archetype_glytoucan=glytoucan,
            archetype_iupac=iupac,
            alpha_glytoucan=glytoucan_alpha,
            alpha_iupac=iupac_alpha,
            beta_glytoucan=glytoucan_beta,
            beta_iupac=iupac_beta,
            cluster_data=cluster_data
        )
        
        # Copy additional analysis files to output subdirectory
        output_subdir = f"{final_output_dir}/output"
        storage.mkdir(output_subdir)
        
        for file_name in ["pca.csv", "torsion_glycosidic.csv", "torparts.npz"]:
            src_file = f"{folder_path}/output/{file_name}"
            if storage.exists(src_file):
                # Use storage abstraction to copy files
                content = storage.read_binary(src_file)
                storage.write_binary(f"{output_subdir}/{file_name}", content)
        
        logger.info(f"Successfully processed {glycam_name} -> {final_output_dir}")
        
    except Exception as e:
        logger.error(f"Failed to process glycan data for {glycam_name}: {e}")
        raise


def main():
    """Example usage of the process_glycan function."""
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python glystatic.py <folder_path> <glycam_name> <output_static_dir>")
        print("Example: python glystatic.py ./DManpa1-2DManpa1-OH DManpa1-2DManpa1-OH ./static_output")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    glycam_name = sys.argv[2] 
    output_static_dir = sys.argv[3]
    
    try:
        process_glycan(folder_path, glycam_name, output_static_dir)
        print(f"Successfully processed {glycam_name}")
    except Exception as e:
        print(f"Error processing {glycam_name}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

