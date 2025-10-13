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
        # Read the inventory CSV
        df = pd.read_csv(config.inventory_path)
        
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


def _emit_multimodel_pdbs(dest_level_dir: Path, cluster_data: dict, level_name: str) -> None:
    """Create multi-model PDBs by concatenating per-cluster PDBs.

    This scans the converted output directories under dest_level_dir, groups
    PDBs by anomer and format, and writes a single multi-model PDB per group
    using MODEL/ENDMDL blocks. Files are ordered by cluster_id found in
    filenames (cluster_<id>_...).

    Args:
        dest_level_dir: Destination level directory under output (e.g., .../output/level_2)
        cluster_data: Parsed cluster info dict from info.json for annotating REMARKs
        level_name: Name of the level (e.g., "level_2") used in output filenames
    """
    # Build a map of cluster_id -> population for REMARKs if available
    cluster_pop: dict[str, float] = {}
    try:
        if cluster_data and "levels" in cluster_data and level_name in cluster_data["levels"]:
            for c in cluster_data["levels"][level_name].get("clusters", []):
                cid = str(c.get("cluster_id"))
                pct = c.get("cluster_size_pct")
                if cid is not None and pct is not None:
                    try:
                        cluster_pop[cid] = round(float(pct), 4)
                    except Exception:
                        pass
    except Exception as e:
        logger.debug(f"Could not map cluster populations for {level_name}: {e}")

    def _cluster_id_from_name(p: Path) -> int:
        """Extract numeric cluster id from a filename; default large to push unknowns last."""
        m = re.search(r"cluster_(\d+)", p.stem)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return 10**9
        return 10**9

    def _write_multimodel(out_path: Path, files: list[Path], anomer: str) -> None:
        if not files:
            return
        out_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing multi-model PDB: {out_path}")
        with open(out_path, "w", newline="\n") as out_f:
            out_f.write(f"REMARK   GLYCAN MULTI-MODEL PDB  LEVEL {level_name.upper()}  ANOMER {anomer.upper()}\n")
            for i, fpath in enumerate(files, start=1):
                cid_match = re.search(r"cluster_(\d+)", fpath.stem)
                cid = cid_match.group(1) if cid_match else None
                pop = cluster_pop.get(str(cid)) if cid else None
                out_f.write(f"MODEL     {i:4d}\n")
                if cid is not None:
                    out_f.write(f"REMARK   CLUSTER_ID {cid}\n")
                if pop is not None:
                    out_f.write(f"REMARK   CLUSTER_POPULATION_PCT {pop}\n")
                # Include atom records; strip headers and any existing MODEL/ENDMDL/END
                try:
                    with open(fpath, "r") as in_f:
                        for line in in_f:
                            if line.startswith(("ATOM", "HETATM", "ANISOU", "TER")):
                                out_f.write(line.rstrip("\n") + "\n")
                except Exception as e:
                    logger.warning(f"Skipping {fpath} while building multi-model: {e}")
                out_f.write("ENDMDL\n")
            out_f.write("END\n")

    # For each format/anomer pair, create a combined file
    for fmt in ["GLYCAM", "PDB"]:
        for anomer in ["alpha", "beta"]:
            src_dir = dest_level_dir / fmt / anomer
            if not src_dir.exists():
                continue
            files = sorted([p for p in src_dir.glob("*.pdb")], key=_cluster_id_from_name)
            if not files:
                continue
            out_path = dest_level_dir / fmt / f"{anomer}.pdb"
            _write_multimodel(out_path, files, anomer)


def generate_glycan_id(glycam_name: str) -> str:
    """Generate a simple ID from glycam name as fallback."""
    # Simple ID generation - you can make this more sophisticated if needed
    import hashlib
    # Create a hash of the glycam name for a consistent ID
    hash_obj = hashlib.md5(glycam_name.encode())
    return f"GS{hash_obj.hexdigest()[:6].upper()}"


def load_info_json(folder_path: Path) -> dict:
    """Load and parse the info.json file from the output directory."""
    info_path = folder_path / "output" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"info.json not found at {info_path}")
    
    with open(info_path, 'r') as f:
        return json.load(f)


def extract_cluster_data(info_data: dict, folder_path: Path) -> dict:
    """Extract cluster data from info.json and organize by levels."""
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
            level_dir = folder_path / "output" / f"level_{level_num}"
            alpha_dir = level_dir / "alpha"
            beta_dir = level_dir / "beta"
            
            cluster_info["pdb_files"] = {
                "alpha": None,
                "beta": None
            }
            
            # Find corresponding PDB files
            if alpha_dir.exists():
                alpha_files = list(alpha_dir.glob(f"cluster_{cluster['cluster_id']}_*.pdb"))
                if alpha_files:
                    cluster_info["pdb_files"]["alpha"] = str(alpha_files[0])
                    
            if beta_dir.exists():
                beta_files = list(beta_dir.glob(f"cluster_{cluster['cluster_id']}_*.pdb"))
                if beta_files:
                    cluster_info["pdb_files"]["beta"] = str(beta_files[0])
            
            clusters.append(cluster_info)
        
        cluster_data["levels"][f"level_{level_num}"] = {
            "n_clusters": level_info["n_clusters"],
            "clusters": clusters
        }
    
    return cluster_data


def copy_and_convert_pdbs(
    folder_path: Path, 
    output_dir: Path, 
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
    source_output = folder_path / "output"
    
    # Process each level directory
    for level_dir in source_output.glob("level_*"):
        if not level_dir.is_dir():
            continue
            
        level_name = level_dir.name
        dest_level_dir = output_dir / "output" / level_name
        dest_level_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy dist.svg if it exists
        dist_file = level_dir / "dist.svg"
        if dist_file.exists():
            shutil.copy2(dist_file, dest_level_dir / "dist.svg")
        
        # Create format directories within each level
        pdb_formats = ["GLYCAM", "PDB"]
        
        for format_name in pdb_formats:
            format_dir = dest_level_dir / format_name
            format_dir.mkdir(parents=True, exist_ok=True)
            
            # Create alpha and beta subdirectories
            alpha_dir = format_dir / "alpha"
            beta_dir = format_dir / "beta"
            alpha_dir.mkdir(parents=True, exist_ok=True)
            beta_dir.mkdir(parents=True, exist_ok=True)
        
        # Process alpha and beta directories
        for anomer in ["alpha", "beta"]:
            anomer_dir = level_dir / anomer
            if not anomer_dir.exists():
                continue
                
            # Process each PDB file
            for pdb_file in anomer_dir.glob("*.pdb"):
                base_name = pdb_file.stem
                
                # Extract cluster information from filename
                # Expected format: cluster_{cluster_id}_{something}
                cluster_id = None
                cluster_population = None
                cluster_members = None
                
                if "cluster_" in base_name:
                    try:
                        # Extract cluster ID from filename
                        parts = base_name.split("_")
                        if len(parts) >= 2 and parts[0] == "cluster":
                            cluster_id = parts[1]
                            
                            # Look up cluster data if available
                            if cluster_data and "levels" in cluster_data:
                                level_name = level_dir.name  # e.g., "level_1", "level_2"
                                if level_name in cluster_data["levels"]:
                                    level_info = cluster_data["levels"][level_name]
                                    for cluster in level_info.get("clusters", []):
                                        if str(cluster["cluster_id"]) == cluster_id:
                                            cluster_population = round(cluster["cluster_size_pct"], 2)
                                            break
                    except Exception as e:
                        logger.warning(f"Could not extract cluster info from {base_name}: {e}")
                
                # Convert to each format
                for format_name in pdb_formats:
                    output_format_dir = dest_level_dir / format_name / anomer
                    output_file = output_format_dir / f"{base_name}.{format_name}.pdb"
                    
                    try:
                        # First, clean the PDB file by removing TITLE and MODEL lines
                        with open(pdb_file, 'r') as f:
                            lines = f.readlines()
                        
                        # Filter out TITLE and MODEL lines
                        cleaned_lines = []
                        for line in lines:
                            if not (line.startswith("TITLE") or line.startswith("MODEL")):
                                cleaned_lines.append(line)
                        
                        # Write cleaned content to a temporary file
                        temp_file = output_file.with_suffix('.tmp')
                        with open(temp_file, 'w') as f:
                            f.writelines(cleaned_lines)
                        
                        # Use glypdbio.convert_pdb to convert formats
                        glypdbio.convert_pdb(str(temp_file), str(output_file), format_name)
                        
                        # Remove temporary file
                        temp_file.unlink()
                        
                        # Add remarks with appropriate GlyTouCan ID and IUPAC name
                        if anomer == "alpha":
                            current_glytoucan = alpha_glytoucan if alpha_glytoucan else archetype_glytoucan
                            current_iupac = alpha_iupac if alpha_iupac else archetype_iupac
                        elif anomer == "beta":
                            current_glytoucan = beta_glytoucan if beta_glytoucan else archetype_glytoucan
                            current_iupac = beta_iupac if beta_iupac else archetype_iupac
                        else:
                            current_glytoucan = archetype_glytoucan
                            current_iupac = archetype_iupac
                            
                        # Add remarks to the final PDB file including cluster information
                        glypdbio._add_pdb_remarks(
                            str(output_file), 
                            current_glytoucan, 
                            current_iupac,
                            cluster_id=cluster_id,
                            cluster_population=cluster_population,
                        )
                        
                    except Exception as e:
                        logger.warning(f"Failed to convert {pdb_file} to {format_name}: {e}")
                        # Clean up temporary file if it exists
                        temp_file = output_file.with_suffix('.tmp')
                        if temp_file.exists():
                            temp_file.unlink()

        # After converting individual files, optionally emit multi-model PDBs for level_2
        try:
            if level_name == "level_2":
                _emit_multimodel_pdbs(dest_level_dir, cluster_data, level_name)
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


def save_glycan_data(glycan_data: dict, output_dir: Path, filename: str) -> None:
    """Save glycan data as JSON file."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{filename}.json"
        serializable_data = make_serializable(glycan_data)
        with open(output_file, 'w') as f:
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
    folder_path = Path(folder_path)
    output_dir = Path(output_static_dir)
    
    logger.info(f"Processing glycan: {glycam_name} from {folder_path}")
    
    # Validate input
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder path does not exist: {folder_path}")
    
    output_subdir = folder_path / "output"
    if not output_subdir.exists():
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
    final_output_dir = output_dir / str(ID)
    if final_output_dir.exists():
        shutil.rmtree(final_output_dir)
    final_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get basic glycan info
    glycam_tidy, end_pos, end_link = name_utils.glycam_info(glycam_name)
    iupac = canonicalize_iupac(glycam_name)
    
    # Generate SNFG diagram
    try:
        snfg_svg_path = final_output_dir / "snfg.svg"
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
        output_subdir = final_output_dir / "output"
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        for file_name in ["pca.csv", "torsion_glycosidic.csv", "torparts.npz"]:
            src_file = folder_path / "output" / file_name
            if src_file.exists():
                shutil.copy2(src_file, output_subdir / file_name)
        
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

