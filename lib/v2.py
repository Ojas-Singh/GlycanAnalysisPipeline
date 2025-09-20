"""Stepwise pipeline wrapper for the logic in the exploratory `test2.py` script.

This module splits the original monolithic flow into discrete steps that can be
run idempotently. Each step will skip if its expected output already exists,
    # Load clustering results

Steps implemented:
  1. store: run `store_from_pdb` to create `frame_data_dir` content (atoms.parquet, coords.zarr, ...)
  2. pca: compute conformation landscape and PCA; write `embedding/pca_conformation_landscape.parquet`
  3. torsions: calculate torsion angles for all frames using batch processing and save torsions.csv and torparts.npz
  4. cluster: compute entropy, fit hierarchical clustering, save dendrogram and clustering outputs
  5. cluster_torsion_stats: compute torsion statistics per cluster and write info.json
  6. save_representative_structures: save PDB structures of cluster representatives with alpha/beta flipping
  7. plot_torsion_distributions: create torsion distribution plots for each clustering level
  8. export_analysis_data: export pca.csv (3 components), torsion_glycosidic.csv, and copy key files

Usage: run this script directly. Example:
  python test2.py --name "MyName" --update
"""

from pathlib import Path
import argparse
import os
import sys
import logging
import shutil
from typing import Dict, List, Tuple, Optional
import polars as pl
import numpy as np
import json

from lib import config
from lib.glystore import store_from_pdb
from lib.glypdbio import parse_mol2_bonds
from lib.glypdbio import get_conformation_landscape
from lib.embedding import (
    pca_conformation_landscape,
    knn_conformational_entropy,
    save_clustering_results,
    save_clustering_results_parquet,
    load_clustering_results_parquet,
    _read_csv_robust,
    gmm_optimize_silhouette,
    gmm_optimize_glycosidic_deviation,
    kcenter_coverage_clustering,
    create_cluster_info,
)
from lib import glypdbio
from lib.torsion import get_glycan_torsions_enhanced, get_torsion_values, get_torsion_values_batch, classify_glycosidic_from_name, circular_stats, save_torparts_npz, plot_torsion_distribution

try:  # RDKit is optional; torsion steps will be skipped if absent
    from rdkit import Chem
    from rdkit.Geometry import Point3D  # type: ignore
except Exception:  # pragma: no cover
    Chem = None  # type: ignore
    Point3D = None  # type: ignore

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def _read_parquet_robust(path: str):
    """Try pyarrow first (avoids glob issues), then fall back to Polars."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        import pyarrow.parquet as pq  # type: ignore
        table = pq.read_table(path)
        return pl.from_arrow(table)
    except Exception:
        try:
            import glob
            return pl.read_parquet(glob.escape(path))
        except Exception:
            return pl.read_parquet(path)


def step_store(name: str, frame_data_dir: str, pdb_path: str, mol2_path: str, force: bool = False) -> None:
    """Run store_from_pdb unless frame_data_dir already contains data and force==False."""
    out_p = Path(frame_data_dir)
    atoms_p = out_p / "atoms.parquet"
    if atoms_p.exists() and not force:
        logger.info("Store step: atoms.parquet exists; skipping (use --update to force)")
        return
    logger.info("Running store_from_pdb -> this may take a while")
    os.makedirs(frame_data_dir, exist_ok=True)
    store_from_pdb(
        pdb_path,
        frame_data_dir,
        get_connectivity=parse_mol2_bonds(mol2_path),
        connectivity_kind="serial",
        frames_buffer=1024,
        frame_chunk=1024,
    )


def step_pca(frame_data_dir: str, embedding_dir: str, force: bool = False, per_pair_scale: bool = False) -> str:
    """Compute conformation landscape and PCA, save parquet. Returns pca file path."""
    os.makedirs(embedding_dir, exist_ok=True)
    pca_path = os.path.join(embedding_dir, "pca_conformation_landscape.parquet")
    if os.path.exists(pca_path) and not force:
        logger.info("PCA step: pca file exists; skipping (use --update to force)")
        return pca_path

    logger.info("Computing conformation landscape (may be memory intensive)")
    landscape = get_conformation_landscape(frame_data_dir, method="noH_fast", sample_size=200, seed=0)
    logger.info("Running PCA on conformation landscape")
    pca_df = pca_conformation_landscape(landscape, n_components=10, scale=True)
    pca_df.write_parquet(pca_path)
    logger.info(f"Wrote PCA parquet: {pca_path}")
    return pca_path


def step_clustering_gmm(embedding_dir: str, pca_path: str, force: bool = False) -> float:
    """Run GMM clustering with level-specific constraints.
    
    Returns the conformational entropy H (nats).
    """
    pca_df = _read_parquet_robust(pca_path)
    selected_cols = [f"PC{i}" for i in range(1, 4)]

    logger.info("Estimating conformational entropy (kNN)")
    H = float(knn_conformational_entropy(pca_df, k=5, n_components=3))
    _ = knn_conformational_entropy(pca_df, k=5, n_components=3, return_details=True, boltzmann=True, temperature=300)  # side calculations
    logger.info(f"Entropy (nats): {H}")

    cluster_out_json = os.path.join(embedding_dir, "clustering_results.json")
    if os.path.exists(cluster_out_json) and not force:
        logger.info("GMM Clustering step: outputs exist; skipping")
        return H

    # Load glycosidic info and torsion data
    glycosidic_path = os.path.join(embedding_dir, "glycosidic_info.json")
    torsion_csv = os.path.join(embedding_dir, "torsions.csv")
    
    with open(glycosidic_path, 'r') as f:
        glycosidic_info = json.load(f)
    
    torsion_df = _read_csv_robust(torsion_csv)
    
    # Extract PCA data
    pca_data = pca_df.select(selected_cols).to_numpy()
    
    # Level 1: Single cluster (entire dataset)
    labels_level1 = np.zeros(len(pca_data), dtype=int)

    # Level 2: GMM with silhouette optimization (3-10 clusters)
    labels_level2, n_clusters_level2 = gmm_optimize_silhouette(pca_data, range_clusters=(3, 10))

    # Level 3: K-center coverage clustering (≤128 representatives)
    labels_level3, n_clusters_level3 = kcenter_coverage_clustering(
        pca_data,
        max_clusters=128,
        coverage_threshold=0.95,
        min_coverage_improvement=0.01
    )
    
    # Format results and save
    all_levels = {
        1: create_cluster_info(labels_level1, pca_df, 1),
        2: create_cluster_info(labels_level2, pca_df, n_clusters_level2),
        3: create_cluster_info(labels_level3, pca_df, n_clusters_level3)
    }
    
    save_clustering_results(all_levels, output_path=cluster_out_json)
    save_clustering_results_parquet(all_levels, output_dir=embedding_dir)
    
    logger.info(f"Saved clustering results to {cluster_out_json} and parquet in {embedding_dir}")
    logger.info(f"Level 1: {1} cluster, Level 2: {n_clusters_level2} clusters, Level 3: {n_clusters_level3} clusters (k-center)")
    
    # Quick read-back and simple diagnostics
    cr = load_clustering_results_parquet(embedding_dir)
    logger.info(f"Cluster reader levels: {cr.levels()}")
    return H


def _build_rdkit_mol(frame_data_dir: str):
    """Construct an RDKit Mol with a single conformer from stored atoms/bonds/coords."""
    if Chem is None:
        raise ImportError("RDKit not available; cannot build molecule for torsions.")
    atoms_df = glypdbio.load_atoms(frame_data_dir)
    bonds_df = glypdbio.load_bonds(frame_data_dir)
    coords = glypdbio.load_coords(frame_data_dir)
    if coords.shape[0] == 0:
        raise ValueError("No frames available in coords store")
    mol = Chem.RWMol()
    elements = atoms_df["element"].to_list()
    res_ids = atoms_df["res_id"].to_list() if "res_id" in atoms_df.columns else [1] * len(elements)
    atom_names = atoms_df["name"].to_list() if "name" in atoms_df.columns else [f"{el}{i}" for i, el in enumerate(elements)]
    
    for i, el in enumerate(elements):
        a = Chem.Atom(str(el))
        mol.AddAtom(a)
        # Store residue ID and atom name as properties
        atom = mol.GetAtomWithIdx(i)
        atom.SetProp("res_id", str(res_ids[i]))
        atom.SetProp("name", str(atom_names[i]))
        
    for a, b in bonds_df.iter_rows():
        try:
            mol.AddBond(int(a), int(b), Chem.BondType.SINGLE)
        except Exception:
            continue  # skip malformed bond indices
    m = mol.GetMol()
    conf = Chem.Conformer(len(elements))
    xyz0 = np.asarray(coords[0], dtype=float)
    for i in range(len(elements)):
        x, y, z = xyz0[i]
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    m.AddConformer(conf, assignId=True)
    return m


def step_torsions(frame_data_dir: str, embedding_dir: str, force: bool = False) -> tuple[list[tuple[int,int,int,int]], list[str]]:
    """Compute torsion angles for all frames and write torsions.csv in embedding_dir.
    Also saves torparts.npz with torsion parts structure.

    Returns (torsion_list, torsion_labels). If RDKit not available, returns ([],[]).
    """
    os.makedirs(embedding_dir, exist_ok=True)
    torsion_csv = os.path.join(embedding_dir, "torsions.csv")
    torparts_path = os.path.join(embedding_dir, "torparts.npz")
    
    if os.path.exists(torsion_csv) and os.path.exists(torparts_path) and not force:
        logger.info("Torsion step: torsions.csv and torparts.npz exist; skipping (use --update to force)")
        # Attempt to recover labels by reading header
        try:
            import csv
            with open(torsion_csv, "r", newline="", encoding="utf-8") as fh:
                r = csv.reader(fh)
                header = next(r)
            labels = header[1:]
        except Exception:
            labels = []
        return [], labels
    if Chem is None:
        logger.warning("RDKit not installed; skipping torsion calculation")
        return [], []
    
    logger.info("Extracting glycan torsions using graph-based method")
    atoms_df = glypdbio.load_atoms(frame_data_dir)
    bonds_df = glypdbio.load_bonds(frame_data_dir)
    torsion_list = get_glycan_torsions_enhanced(atoms_df, bonds_df)
    
    # Build RDKit molecule for torsion value calculation
    mol = _build_rdkit_mol(frame_data_dir)
    
    # Create labels using atom_name_residue_number format
    elements = atoms_df["element"].to_list()
    atom_names = atoms_df["name"].to_list() if "name" in atoms_df.columns else [f"{elements[i]}{i}" for i in range(len(elements))]
    res_ids = atoms_df["res_id"].to_list() if "res_id" in atoms_df.columns else [1] * len(elements)
    
    labels = []
    for torsion in torsion_list:
        # Use atom_name_residue_number format (e.g., C1_2, O2_3, C3_3, C4_2)
        torsion_name = "-".join(f"{atom_names[i]}_{res_ids[i]}" for i in torsion)
        labels.append(torsion_name)
    
    if not torsion_list:
        logger.info("No rotatable torsions detected; skipping file write")
        return torsion_list, labels
    
    # Save torparts.npz with same structure as tfindr.py
    logger.info("Generating torsion parts for torparts.npz")
    save_torparts_npz(torsion_list, bonds_df, atoms_df, torparts_path)
    logger.info(f"Saved torsion parts to {torparts_path}")
    
    coords = glypdbio.load_coords(frame_data_dir)
    n_frames = int(coords.shape[0])
    logger.info(f"Computing torsions for {n_frames} frames ({len(torsion_list)} torsions) using batch processing")
    
    # Convert coords to numpy array for batch processing
    # coords.shape should be (n_frames, n_atoms, 3)
    coords_array = np.asarray(coords, dtype=float)
    
    # Use batch processing for much faster computation
    logger.info("Running batch torsion calculation...")
    torsion_angles = get_torsion_values_batch(coords_array, torsion_list)
    
    # Write results to CSV
    logger.info(f"Writing {n_frames} x {len(torsion_list)} torsion values to CSV")
    with open(torsion_csv, "w", encoding="utf-8") as fh:
        fh.write("frame," + ",".join(labels) + "\n")
        for f in range(n_frames):
            vals = torsion_angles[f, :]
            fh.write(str(f) + "," + ",".join(f"{v:.3f}" for v in vals) + "\n")
            if (f+1) % 1000 == 0:
                logger.info(f"  written {f+1} frames")
    
    logger.info(f"Wrote torsion angles to {torsion_csv}")
    
    # Save glycosidic linkage info
    glycosidic_path = os.path.join(embedding_dir, "glycosidic_info.json")
    glycosidic_torsions = []
    for i, label in enumerate(labels):
        glycosidic_name = classify_glycosidic_from_name(label)
        if '_' in glycosidic_name and any(t in glycosidic_name for t in ['phi', 'psi', 'omega']):
            glycosidic_torsions.append({
                'index': i,
                'original_name': label,
                'glycosidic_name': glycosidic_name,
                'torsion_indices': torsion_list[i]
            })
    
    with open(glycosidic_path, 'w', encoding='utf-8') as f:
        json.dump(glycosidic_torsions, f, indent=2)
    logger.info(f"Saved glycosidic info to {glycosidic_path}")
    
    return torsion_list, labels


def step_cluster_torsion_stats(frame_data_dir: str, embedding_dir: str, entropy: float, force: bool = False) -> None:
    """Compute torsion stats per cluster and write info.json in embedding_dir.

    Expects torsions.csv (skip silently if missing) and clustering parquet outputs.
    Reads torsion names directly from CSV and performs glycosidic classification.
    """
    info_path = os.path.join(embedding_dir, "info.json")
    if os.path.exists(info_path) and not force:
        logger.info("Info step: info.json exists; skipping (use --update to force)")
        return
    torsion_csv = os.path.join(embedding_dir, "torsions.csv")
    if not os.path.exists(torsion_csv):
        logger.warning("torsions.csv not found; skipping cluster torsion stats")
        return
    
    # Load torsions table and extract column names
    tors_df = _read_csv_robust(torsion_csv)
    if 'frame' not in tors_df.columns:
        logger.warning("torsions.csv missing 'frame' column; aborting stats step")
        return
    
    # Get torsion labels from CSV columns (excluding 'frame')
    torsion_labels = [c for c in tors_df.columns if c != 'frame']
    
    # Perform glycosidic classification based on torsion names
    name_mapping = {}
    glycosidic_torsions = []
    other_torsions = []
    
    for i, label in enumerate(torsion_labels):
        # Parse atom_name_residue_number format (e.g., "C1_2-O3_2-C4_3-C5_3")
        glycosidic_name = classify_glycosidic_from_name(label)
        name_mapping[label] = glycosidic_name
        
        if '_' in glycosidic_name and any(t in glycosidic_name for t in ['phi', 'psi', 'omega']):
            # This is a glycosidic torsion
            parts = glycosidic_name.split('_')
            if len(parts) == 3 and parts[2] in ['phi', 'psi', 'omega']:
                glycosidic_torsions.append({
                    'index': i,
                    'original_name': label,
                    'glycosidic_name': glycosidic_name,
                    'res1': int(parts[0]) if parts[0].isdigit() else parts[0],
                    'res2': int(parts[1]) if parts[1].isdigit() else parts[1],
                    'type': parts[2]
                })
                continue
        
        other_torsions.append({
            'index': i,
            'original_name': label,
            'alternate_name': glycosidic_name,
            'type': 'other'
        })
    
    # Create glycosidic_info summary
    glycosidic_info = {
        'summary': {
            'total_torsions': len(torsion_labels),
            'glycosidic_torsions': len(glycosidic_torsions),
            'other_torsions': len(other_torsions),
            'phi_count': len([g for g in glycosidic_torsions if g['type'] == 'phi']),
            'psi_count': len([g for g in glycosidic_torsions if g['type'] == 'psi']),
            'omega_count': len([g for g in glycosidic_torsions if g['type'] == 'omega'])
        }
    }
    # Load clustering results
    try:
        cr = load_clustering_results_parquet(embedding_dir)
    except Exception as exc:
        logger.warning(f"Could not load clustering results: {exc}; skipping stats")
        return
    levels_summary = []
    angle_cols = [c for c in tors_df.columns if c != 'frame']
    frame_to_row = {int(f): i for i, f in enumerate(tors_df['frame'].to_list())}

    # Try to load PCA dataframe for entropy calculations
    pca_df = None
    try:
        pca_path = os.path.join(embedding_dir, "pca_conformation_landscape.parquet")
        if os.path.exists(pca_path):
            pca_df = _read_parquet_robust(pca_path)
        else:
            logger.debug(f"PCA parquet not found at {pca_path}; per-cluster entropies will be skipped")
    except Exception as exc:
        logger.debug(f"Failed to read PCA parquet for entropy calc: {exc}")

    # Choose PCA component columns consistently with global entropy calc (first 3 PCs)
    entropy_pc_cols: List[str] = []
    if pca_df is not None:
        preferred = ["PC1", "PC2", "PC3"]
        entropy_pc_cols = [c for c in preferred if c in pca_df.columns]
        if not entropy_pc_cols:
            # fallback: any PC columns matching pattern, sorted by index
            try:
                pc_nums = []
                for c in pca_df.columns:
                    if c.upper().startswith("PC"):
                        try:
                            pc_nums.append((int(c[2:]), c))
                        except Exception:
                            pass
                pc_nums.sort()
                entropy_pc_cols = [c for _, c in pc_nums[:3]]
            except Exception:
                entropy_pc_cols = []

    level_entropy_map = {}
    for level in cr.levels():
        n_clusters = cr.get_n_clusters(level)
        reps_df = cr.representatives(level)
        clusters_out = []
        cluster_entropies = []
        for row in reps_df.iter_rows(named=True):
            cid = int(row['cluster_id'])
            rep_idx = row.get('representative_idx')
            pct = float(row.get('cluster_size_pct')) if row.get('cluster_size_pct') is not None else None
            members = []
            try:
                members = cr.members(level, cid)
            except Exception:
                pass
            cluster_stats = {}
            if members:
                member_rows = tors_df.filter(pl.col('frame').is_in(members))
                for col in angle_cols:
                    vals = member_rows[col].to_numpy().astype(float)
                    cluster_stats[col] = circular_stats(vals)

            # Compute per-cluster conformational entropy (nats) using PCA rows for member frames
            cluster_entropy = None
            if pca_df is not None and members and entropy_pc_cols:
                try:
                    # Select PCA rows where 'frame' in members
                    cluster_pca = pca_df.filter(pl.col('frame').is_in(members))
                    if cluster_pca.height > 1:
                        # Use same target k as global (5), clipped to N-1
                        k_eff = min(5, max(1, cluster_pca.height - 1))
                        h_val = knn_conformational_entropy(
                            cluster_pca,
                            component_cols=entropy_pc_cols,
                            k=k_eff,
                        )
                        if isinstance(h_val, dict):
                            cluster_entropy = float(h_val.get('H_nats', float('nan')))
                        else:
                            cluster_entropy = float(h_val)
                        cluster_entropies.append(cluster_entropy)
                except Exception as exc:
                    logger.debug(f"Could not compute entropy for level {level} cluster {cid}: {exc}")

            # Representative torsions
            rep_angles = {}
            if rep_idx is not None and rep_idx in frame_to_row:
                rep_row = tors_df.filter(pl.col('frame') == int(rep_idx)).to_dicts()
                if rep_row:
                    rep_row = rep_row[0]
                    for col in angle_cols:
                        if col not in cluster_stats:
                            cluster_stats[col] = circular_stats(np.array([], dtype=float))
                        cluster_stats[col]['representative_angle_deg'] = float(rep_row[col])

            clusters_out.append({
                'cluster_id': cid,
                'cluster_size_pct': pct,
                'representative_idx': int(rep_idx) if rep_idx is not None else None,
                'members_count': len(members),
                'entropy_nats': None if cluster_entropy is None else float(cluster_entropy),
                'torsion_stats': cluster_stats,
            })

        # compute average entropy for this level
        avg_entropy = None
        if cluster_entropies:
            try:
                avg_entropy = float(sum(cluster_entropies) / len(cluster_entropies))
            except Exception:
                avg_entropy = None
        level_entropy_map[int(level)] = avg_entropy

        levels_summary.append({
            'level': int(level),
            'n_clusters': int(n_clusters) if n_clusters is not None else None,
            'clusters': clusters_out,
            'average_entropy_nats': None if avg_entropy is None else float(avg_entropy),
        })
    
    # Assemble JSON
    info = {
        'entropy_nats': float(entropy),
        'torsions': {
            'labels': torsion_labels,  # Original atom-based names (used in CSV headers)
            'count': len(torsion_labels),
            'name_mapping': name_mapping,  # Mapping from original to glycosidic names
            'glycosidic': glycosidic_torsions,
            'other': other_torsions,
            'glycosidic_summary': glycosidic_info.get('summary', {})
        },
        'levels': levels_summary,
        'schema_version': 1,
    }
    import json
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
    logger.info(f"Wrote cluster torsion stats to {info_path}")
    logger.info(f"Found {len(glycosidic_torsions)} glycosidic torsions and {len(other_torsions)} other torsions")

    # Update embedding/metadata.json with average entropy per level
    try:
        meta_path = os.path.join(embedding_dir, 'metadata.json')
        meta = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as mf:
                    meta = json.load(mf)
            except Exception:
                meta = {}
        # add or replace mapping
        meta['level_average_entropy'] = {str(k): (None if v is None else float(v)) for k, v in level_entropy_map.items()} if 'level_entropy_map' in locals() or 'level_entropy_map' in globals() else {}
        with open(meta_path, 'w', encoding='utf-8') as mf:
            json.dump(meta, mf, indent=2)
        logger.info(f"Updated metadata.json with level average entropies at {meta_path}")
    except Exception as exc:
        logger.warning(f"Failed to update metadata.json with entropy info: {exc}")


def step_save_representative_structures(
    frame_data_dir: str, 
    embedding_dir: str, 
    out_dir: str, 
    glycam_name: str, 
    force: bool = False
) -> None:
    """Save PDB structures of cluster representatives with alpha/beta flipping.
    
    This step reads clustering results and saves PDB files for each representative frame.
    The structures are saved with both original anomer and flipped anomer versions.
    
    Parameters:
        frame_data_dir: Directory containing trajectory data (atoms.parquet, coords.zarr)
        embedding_dir: Directory containing clustering results
        out_dir: Output directory where structures will be saved directly
        glycam_name: GLYCAM sequence name for anomer detection
        force: Whether to overwrite existing structures
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Check if structures already exist (simple check for any PDB files)
    existing_pdbs = []
    if os.path.exists(out_dir):
        for root, dirs, files in os.walk(out_dir):
            existing_pdbs.extend([f for f in files if f.endswith('.pdb')])
    
    if existing_pdbs and not force:
        logger.info("Representative structures exist; skipping (use --update to force)")
        return
    
    # Load clustering results
    try:
        cr = load_clustering_results_parquet(embedding_dir)
    except Exception as exc:
        logger.warning(f"Could not load clustering results: {exc}; skipping structure generation")
        return
    
    # Determine current anomer type
    current_anomer = glypdbio.get_anomer(glycam_name)
    if current_anomer == "unknown":
        logger.warning(f"Could not determine anomer type for {glycam_name}; using 'alpha' as default")
        current_anomer = "alpha"
    
    logger.info(f"Current structure anomer: {current_anomer}")
    
    # Process each clustering level
    total_structures = 0
    for level in cr.levels():
        reps_df = cr.representatives(level)
        level_dir = os.path.join(out_dir, f"level_{level}")
        os.makedirs(level_dir, exist_ok=True)
        
        # Create alpha and beta subdirectories
        alpha_dir = os.path.join(level_dir, "alpha")
        beta_dir = os.path.join(level_dir, "beta")
        os.makedirs(alpha_dir, exist_ok=True)
        os.makedirs(beta_dir, exist_ok=True)
        
        for row in reps_df.iter_rows(named=True):
            cid = int(row['cluster_id'])
            rep_idx = row.get('representative_idx')
            cluster_size_pct = row.get('cluster_size_pct', 0.0)
            
            if rep_idx is None:
                logger.warning(f"No representative index for cluster {cid} at level {level}")
                continue
            
            rep_frame = int(rep_idx)
            
            # Format cluster percentage for display
            pct_str = f"{cluster_size_pct:.1f}%" if cluster_size_pct is not None else "0.0%"
            
            # Save original anomer structure
            original_filename = f"cluster_{cid}_rep_{rep_frame}_{pct_str}.pdb"
            if current_anomer == "alpha":
                original_path = os.path.join(alpha_dir, original_filename)
            else:
                original_path = os.path.join(beta_dir, original_filename)
            
            try:
                title = f"Level {level} Cluster {cid} Representative {rep_frame} ({current_anomer}) - {pct_str}"
                glypdbio.write_pdb_for_frame(
                    out_dir=frame_data_dir,
                    frame_idx=rep_frame,
                    path=original_path,
                    title=title,
                    flip=False
                )
                total_structures += 1
                logger.debug(f"Saved {original_filename}")
            except Exception as exc:
                logger.warning(f"Failed to save original structure for cluster {cid}: {exc}")
                continue
            
            # Save flipped anomer structure
            flipped_anomer = "beta" if current_anomer == "alpha" else "alpha"
            flipped_filename = f"cluster_{cid}_rep_{rep_frame}_{pct_str}.pdb"
            if flipped_anomer == "alpha":
                flipped_path = os.path.join(alpha_dir, flipped_filename)
            else:
                flipped_path = os.path.join(beta_dir, flipped_filename)
            
            try:
                title = f"Level {level} Cluster {cid} Representative {rep_frame} ({flipped_anomer}) - {pct_str}"
                glypdbio.write_pdb_for_frame(
                    out_dir=frame_data_dir,
                    frame_idx=rep_frame,
                    path=flipped_path,
                    title=title,
                    flip=True
                )
                total_structures += 1
                logger.debug(f"Saved {flipped_filename}")
            except Exception as exc:
                logger.warning(f"Failed to save flipped structure for cluster {cid}: {exc}")
                continue
    
    logger.info(f"Saved {total_structures} representative structures in {out_dir}")
    logger.info(f"Structures organized by level and anomer type (alpha/beta subdirectories)")


def step_plot_torsion_distributions(
    embedding_dir: str,
    out_dir: str,
    force: bool = False
) -> None:
    """Create torsion distribution plots for each clustering level.
    
    This step reads clustering results and torsions.csv to generate distribution plots
    for each level, highlighting cluster representatives.
    
    Parameters:
        embedding_dir: Directory containing clustering results and torsions.csv
        out_dir: Output directory where level folders exist
        force: Whether to overwrite existing distribution plots
    """
    # Check for required files
    torsion_csv_path = os.path.join(embedding_dir, "torsions.csv")
    if not os.path.exists(torsion_csv_path):
        logger.warning("torsions.csv not found; skipping distribution plots")
        return
    
    # Load clustering results
    try:
        cr = load_clustering_results_parquet(embedding_dir)
    except Exception as exc:
        logger.warning(f"Could not load clustering results: {exc}; skipping distribution plots")
        return
    
    # Process each clustering level
    plots_created = 0
    plots_skipped = 0
    
    for level in cr.levels():
        level_dir = os.path.join(out_dir, f"level_{level}")
        if not os.path.exists(level_dir):
            logger.warning(f"Level directory {level_dir} does not exist; skipping distribution plot")
            continue
        
        dist_plot_path = os.path.join(level_dir, "dist.svg")
        
        # Check if plot already exists and force is not set
        if os.path.exists(dist_plot_path) and not force:
            logger.info(f"Distribution plot exists for level {level}; skipping (use --update to force)")
            plots_skipped += 1
            continue
        
        # Get representative frames for this level
        reps_df = cr.representatives(level)
        representative_frames = []
        
        for row in reps_df.iter_rows(named=True):
            rep_idx = row.get('representative_idx')
            if rep_idx is not None:
                representative_frames.append(int(rep_idx))
        
        if not representative_frames:
            logger.warning(f"No representative frames found for level {level}; skipping distribution plot")
            continue
        
        # Create distribution plot
        try:
            plot_torsion_distribution(
                torsion_csv_path=torsion_csv_path,
                output_path=dist_plot_path,
                cluster_representatives=representative_frames,
                level=level
            )
            logger.info(f"Saved torsion distribution plot: {dist_plot_path}")
            plots_created += 1
        except Exception as exc:
            logger.warning(f"Failed to create torsion distribution plot for level {level}: {exc}")
    
    logger.info(f"Distribution plots: {plots_created} created, {plots_skipped} skipped")


def step_export_analysis_data(
    embedding_dir: str,
    out_dir: str,
    force: bool = False
) -> None:
    """Export key analysis data files to output directory.
    
    This step exports:
    - pca.csv: PCA data with first 3 components only
    - torsion_glycosidic.csv: Only glycosidic torsions from torsions.csv
    - Copies torparts.npz and info.json to output directory
    
    Parameters:
        embedding_dir: Directory containing analysis results
        out_dir: Output directory where files will be exported
        force: Whether to overwrite existing files
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Define output paths
    pca_out_path = os.path.join(out_dir, "pca.csv")
    torsion_glycosidic_out_path = os.path.join(out_dir, "torsion_glycosidic.csv")
    torparts_out_path = os.path.join(out_dir, "torparts.npz")
    info_out_path = os.path.join(out_dir, "info.json")
    
    files_created = 0
    files_skipped = 0
    
    # 1. Export PCA data with first 3 components
    pca_source_path = os.path.join(embedding_dir, "pca_conformation_landscape.parquet")
    if os.path.exists(pca_source_path):
        if os.path.exists(pca_out_path) and not force:
            logger.info("pca.csv exists; skipping (use --update to force)")
            files_skipped += 1
        else:
            try:
                pca_df = _read_parquet_robust(pca_source_path)
                # Select only first 3 PCA components
                pca_cols = ["PC1", "PC2", "PC3"]
                available_cols = [col for col in pca_cols if col in pca_df.columns]
                
                if available_cols:
                    pca_export_df = pca_df.select(available_cols)
                    pca_export_df.write_csv(pca_out_path)
                    logger.info(f"Exported PCA data with {len(available_cols)} components to {pca_out_path}")
                    files_created += 1
                else:
                    logger.warning("No PCA components found in parquet file")
            except Exception as exc:
                logger.warning(f"Failed to export PCA data: {exc}")
    else:
        logger.warning(f"PCA source file not found: {pca_source_path}")
    
    # 2. Export glycosidic torsions only
    torsion_source_path = os.path.join(embedding_dir, "torsions.csv")
    info_source_path = os.path.join(embedding_dir, "info.json")
    
    if os.path.exists(torsion_source_path) and os.path.exists(info_source_path):
        if os.path.exists(torsion_glycosidic_out_path) and not force:
            logger.info("torsion_glycosidic.csv exists; skipping (use --update to force)")
            files_skipped += 1
        else:
            try:
                # Load info.json to get glycosidic torsion information
                with open(info_source_path, 'r', encoding='utf-8') as f:
                    info_data = json.load(f)

                glycosidic_torsions = info_data.get('torsions', {}).get('glycosidic', [])

                if glycosidic_torsions:
                    # Map original_name -> glycosidic_name (fallback to name_mapping)
                    name_mapping = info_data.get('torsions', {}).get('name_mapping', {})

                    # Instead of loading the entire CSV with Polars (which can fail
                    # on very wide files or cause memory/parse issues), stream the
                    # CSV: read the header, find column indices, then write a new
                    # CSV with only the requested columns.
                    import csv

                    with open(torsion_source_path, 'r', encoding='utf-8', newline='') as fin:
                        reader = csv.reader(fin)
                        header = next(reader)
                        col_to_idx = {col: i for i, col in enumerate(header)}

                        # Determine which glycosidic columns are present
                        selected_indices = []
                        out_header = ['frame']
                        for tor in glycosidic_torsions:
                            original_name = tor.get('original_name')
                            gly_name = tor.get('glycosidic_name') or name_mapping.get(original_name)
                            if original_name in col_to_idx:
                                selected_indices.append(col_to_idx[original_name])
                                out_header.append(gly_name)

                        if selected_indices:
                            # Stream rows and write selected columns to output CSV
                            with open(torsion_glycosidic_out_path, 'w', encoding='utf-8', newline='') as fout:
                                writer = csv.writer(fout)
                                writer.writerow(out_header)
                                for row in reader:
                                    # frame is always first column (index 0)
                                    out_row = [row[0]] + [row[idx] for idx in selected_indices]
                                    writer.writerow(out_row)

                            logger.info(f"Exported {len(selected_indices)} glycosidic torsions with glycosidic names to {torsion_glycosidic_out_path}")
                            files_created += 1
                        else:
                            logger.warning("No glycosidic torsions found to export (none present in CSV header)")
                else:
                    logger.warning("No glycosidic torsions identified in info.json")
            except Exception as exc:
                logger.warning(f"Failed to export glycosidic torsions: {exc}")
    else:
        if not os.path.exists(torsion_source_path):
            logger.warning(f"Torsions source file not found: {torsion_source_path}")
        if not os.path.exists(info_source_path):
            logger.warning(f"Info source file not found: {info_source_path}")
    
    # 3. Copy torparts.npz
    torparts_source_path = os.path.join(embedding_dir, "torparts.npz")
    if os.path.exists(torparts_source_path):
        if os.path.exists(torparts_out_path) and not force:
            logger.info("torparts.npz exists; skipping (use --update to force)")
            files_skipped += 1
        else:
            try:
                shutil.copy2(torparts_source_path, torparts_out_path)
                logger.info(f"Copied torparts.npz to {torparts_out_path}")
                files_created += 1
            except Exception as exc:
                logger.warning(f"Failed to copy torparts.npz: {exc}")
    else:
        logger.warning(f"torparts.npz source file not found: {torparts_source_path}")
    
    # 4. Copy info.json
    if os.path.exists(info_source_path):
        if os.path.exists(info_out_path) and not force:
            logger.info("info.json exists; skipping (use --update to force)")
            files_skipped += 1
        else:
            try:
                shutil.copy2(info_source_path, info_out_path)
                logger.info(f"Copied info.json to {info_out_path}")
                files_created += 1
            except Exception as exc:
                logger.warning(f"Failed to copy info.json: {exc}")
    else:
        logger.warning(f"info.json source file not found: {info_source_path}")
    
    logger.info(f"Data export: {files_created} files created, {files_skipped} files skipped")


class GlycanAnalysisPipeline:
    """Class-based interface for running the v2 glycan analysis pipeline."""
    
    def __init__(self, data_dir: str = None, process_dir: str = None):
        """Initialize the pipeline.
        
        Args:
            data_dir: Base directory containing glycan input data (defaults to config.data_dir)
            process_dir: Base directory for processing outputs (defaults to config.process_dir)
        """
        self.data_dir = data_dir or str(config.data_dir)
        self.process_dir = process_dir or str(config.process_dir)
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def get_glycan_paths(self, glycan_name: str) -> Dict[str, str]:
        """Get all relevant paths for a glycan.
        
        Args:
            glycan_name: Name of the glycan
            
        Returns:
            Dictionary containing all paths
        """
        # Input paths from data_dir (read-only)
        input_glycan_dir = os.path.join(self.data_dir, glycan_name)
        
        # Processing paths in process_dir
        process_glycan_dir = os.path.join(self.process_dir, glycan_name)
        
        # Determine GLYCAM name
        if glycan_name.endswith("-OH"):
            glycam_name = glycan_name
        else:
            json_path = os.path.join(input_glycan_dir, "name.json")
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                glycam_name = data['indexOrderedSequence']
            except (FileNotFoundError, KeyError, json.JSONDecodeError):
                self.logger.warning(f"Could not read GLYCAM name from {json_path}, using directory name")
                glycam_name = glycan_name
        
        return {
            'input_glycan_dir': input_glycan_dir,  # For reading input files
            'process_glycan_dir': process_glycan_dir,  # For storing processing outputs
            'glycam_name': glycam_name,
            'pdb_path': os.path.join(input_glycan_dir, f"{glycan_name}.pdb"),
            'mol2_path': os.path.join(input_glycan_dir, f"{glycan_name}.mol2"),
            'frame_data_dir': os.path.join(process_glycan_dir, "data"),
            'out_dir': os.path.join(process_glycan_dir, "output"),
            'embedding_dir': os.path.join(process_glycan_dir, "embedding")
        }
    
    def validate_inputs(self, paths: Dict[str, str]) -> bool:
        """Validate that required input files exist.
        
        Args:
            paths: Dictionary of paths from get_glycan_paths
            
        Returns:
            True if all required files exist, False otherwise
        """
        required_files = ['pdb_path', 'mol2_path']
        
        for file_key in required_files:
            file_path = paths[file_key]
            if not os.path.exists(file_path):
                self.logger.error(f"Required file not found: {file_path}")
                return False
                
        return True
    
    def run_analysis(self, glycan_name: str, force_update: bool = False, per_pair_scale: bool = False) -> Dict[str, any]:
        """Run the complete v2 analysis pipeline for a glycan.
        
        Args:
            glycan_name: Name of the glycan to analyze
            force_update: Whether to force recomputation of existing results
            per_pair_scale: Whether to normalize each atom pair distance independently in PCA
            
        Returns:
            Dictionary with analysis results and status
        """
        results = {
            'glycan_name': glycan_name,
            'success': False,
            'steps_completed': [],
            'steps_failed': [],
            'error_message': None,
            'paths': {}
        }
        
        try:
            # Get paths
            paths = self.get_glycan_paths(glycan_name)
            results['paths'] = paths
            
            # Validate inputs
            if not self.validate_inputs(paths):
                results['error_message'] = "Input validation failed"
                return results
            
            self.logger.info(f"Starting v2 analysis pipeline for: {glycan_name}")
            
            # Step 1: Store trajectory data
            try:
                self.logger.info("Step 1: Processing trajectory data")
                step_store(
                    glycan_name, 
                    paths['frame_data_dir'], 
                    paths['pdb_path'], 
                    paths['mol2_path'], 
                    force=force_update
                )
                results['steps_completed'].append('store')
            except Exception as e:
                self.logger.error(f"Store step failed: {str(e)}")
                results['steps_failed'].append('store')
                results['error_message'] = f"Store step failed: {str(e)}"
                return results

            # Step 2: PCA analysis
            try:
                self.logger.info("Step 2: PCA analysis")
                pca_path = step_pca(
                    paths['frame_data_dir'], 
                    paths['embedding_dir'], 
                    force=force_update,
                    per_pair_scale=per_pair_scale
                )
                results['steps_completed'].append('pca')
            except Exception as e:
                self.logger.error(f"PCA step failed: {str(e)}")
                results['steps_failed'].append('pca')
                results['error_message'] = f"PCA step failed: {str(e)}"
                return results

            # Step 3: Torsion analysis (moved before clustering since it's independent)
            try:
                self.logger.info("Step 3: Torsion analysis")
                torsion_list, torsion_labels = step_torsions(
                    paths['frame_data_dir'], 
                    paths['embedding_dir'], 
                    force=force_update
                )
                results['steps_completed'].append('torsions')
            except Exception as e:
                self.logger.error(f"Torsion step failed: {str(e)}")
                results['steps_failed'].append('torsions')
                results['error_message'] = f"Torsion step failed: {str(e)}"
                return results

            # Step 4: Clustering analysis
            try:
                self.logger.info("Step 4: Clustering analysis")
                entropy = step_clustering_gmm(
                    paths['embedding_dir'], 
                    pca_path, 
                    force=force_update
                )
                results['steps_completed'].append('clustering')
            except Exception as e:
                self.logger.error(f"Clustering step failed: {str(e)}")
                results['steps_failed'].append('clustering')
                results['error_message'] = f"Clustering step failed: {str(e)}"
                return results

            # Step 5: Cluster torsion statistics
            # Step 5: Cluster torsion statistics
            try:
                self.logger.info("Step 5: Cluster torsion statistics")
                step_cluster_torsion_stats(
                    paths['frame_data_dir'], 
                    paths['embedding_dir'], 
                    entropy, 
                    force=force_update
                )
                results['steps_completed'].append('cluster_stats')
            except Exception as e:
                self.logger.error(f"Cluster stats step failed: {str(e)}")
                results['steps_failed'].append('cluster_stats')
                results['error_message'] = f"Cluster stats step failed: {str(e)}"
                return results

            # Step 6: Save representative structures
            try:
                self.logger.info("Step 6: Saving representative structures")
                step_save_representative_structures(
                    paths['frame_data_dir'], 
                    paths['embedding_dir'], 
                    paths['out_dir'], 
                    paths['glycam_name'], 
                    force=force_update
                )
                results['steps_completed'].append('structures')
            except Exception as e:
                self.logger.error(f"Structure saving step failed: {str(e)}")
                results['steps_failed'].append('structures')
                results['error_message'] = f"Structure saving step failed: {str(e)}"
                return results

            # Step 7: Create torsion distribution plots
            try:
                self.logger.info("Step 7: Creating distribution plots")
                step_plot_torsion_distributions(
                    paths['embedding_dir'], 
                    paths['out_dir'], 
                    force=force_update
                )
                results['steps_completed'].append('plots')
            except Exception as e:
                self.logger.error(f"Plotting step failed: {str(e)}")
                results['steps_failed'].append('plots')
                results['error_message'] = f"Plotting step failed: {str(e)}"
                return results

            # Step 8: Export analysis data
            try:
                self.logger.info("Step 8: Exporting analysis data")
                step_export_analysis_data(
                    paths['embedding_dir'], 
                    paths['out_dir'], 
                    force=force_update
                )
                results['steps_completed'].append('export')
            except Exception as e:
                self.logger.error(f"Export step failed: {str(e)}")
                results['steps_failed'].append('export')
                results['error_message'] = f"Export step failed: {str(e)}"
                return results

            # All steps completed successfully
            results['success'] = True
            self.logger.info(f"Successfully completed v2 analysis for {glycan_name}")
            
        except Exception as e:
            self.logger.error(f"Unexpected error in v2 analysis: {str(e)}")
            results['error_message'] = f"Unexpected error: {str(e)}"
            
        return results
    
    def get_analysis_status(self, glycan_name: str) -> Dict[str, any]:
        """Check the status of analysis for a glycan.
        
        Args:
            glycan_name: Name of the glycan
            
        Returns:
            Dictionary with status information
        """
        paths = self.get_glycan_paths(glycan_name)
        status = {
            'glycan_name': glycan_name,
            'paths': paths,
            'input_files_exist': self.validate_inputs(paths),
            'steps_status': {}
        }
        
        # Check each step's output
        steps_to_check = {
            'store': os.path.join(paths['frame_data_dir'], 'atoms.parquet'),
            'pca': os.path.join(paths['embedding_dir'], 'pca_conformation_landscape.parquet'),
            'clustering': os.path.join(paths['embedding_dir'], 'clustering_results.json'),
            'torsions': os.path.join(paths['embedding_dir'], 'torsions.csv'),
            'cluster_stats': os.path.join(paths['embedding_dir'], 'info.json'),
            'structures': paths['out_dir'],  # Check if output directory exists
            'export': os.path.join(paths['out_dir'], 'pca.csv')
        }
        
        for step, check_path in steps_to_check.items():
            status['steps_status'][step] = os.path.exists(check_path)
            
        return status


def run_glycan_analysis(glycan_name: str, data_dir: str = None, process_dir: str = None, force_update: bool = False, per_pair_scale: bool = False) -> Dict[str, any]:
    """Convenience function to run v2 analysis for a single glycan.
    
    Args:
        glycan_name: Name of the glycan to analyze
        data_dir: Base directory containing glycan input data (defaults to config.data_dir)
        process_dir: Base directory for processing outputs (defaults to config.process_dir)
        force_update: Whether to force recomputation of existing results
        per_pair_scale: Whether to normalize each atom pair distance independently in PCA
        
    Returns:
        Dictionary with analysis results and status
    """
    pipeline = GlycanAnalysisPipeline(data_dir, process_dir)
    return pipeline.run_analysis(glycan_name, force_update, per_pair_scale)


def get_glycan_status(glycan_name: str, data_dir: str = None, process_dir: str = None) -> Dict[str, any]:
    """Convenience function to check analysis status for a glycan.
    
    Args:
        glycan_name: Name of the glycan
        data_dir: Base directory containing glycan input data (defaults to config.data_dir)
        process_dir: Base directory for processing outputs (defaults to config.process_dir)
        
    Returns:
        Dictionary with status information
    """
    pipeline = GlycanAnalysisPipeline(data_dir, process_dir)
    return pipeline.get_analysis_status(glycan_name)


def main():
    parser = argparse.ArgumentParser(description="Stepwise pipeline for storing, PCA and clustering")
    parser.add_argument('--name', type=str, default=None, help='Molecule/directory name (overrides default config)')
    parser.add_argument('--update', action='store_true', help='Force recompute all steps (overwrite existing outputs)')
    parser.add_argument('--status', action='store_true', help='Check analysis status instead of running')
    parser.add_argument('--per-pair-scale', action='store_true', help='Normalize each atom pair distance independently in PCA')
    args = parser.parse_args()

    # Default name from the original file if not provided
    default_name = "DManpa1-2DManpa1-OH"
    name = args.name or default_name

    if args.status:
        # Just check status
        status = get_glycan_status(name)
        print(f"Analysis status for {name}:")
        print(f"Input files exist: {status['input_files_exist']}")
        print("Step completion status:")
        for step, completed in status['steps_status'].items():
            status_symbol = "✓" if completed else "✗"
            print(f"  {step}: {status_symbol}")
        return

    # Run analysis using the new API
    results = run_glycan_analysis(name, force_update=args.update, per_pair_scale=args.per_pair_scale)
    
    if results['success']:
        logger.info(f"Analysis completed successfully for {name}")
        print(f"✓ Analysis completed for {name}")
        print(f"Steps completed: {', '.join(results['steps_completed'])}")
    else:
        logger.error(f"Analysis failed for {name}: {results['error_message']}")
        print(f"✗ Analysis failed for {name}")
        print(f"Error: {results['error_message']}")
        if results['steps_completed']:
            print(f"Steps completed before failure: {', '.join(results['steps_completed'])}")
        if results['steps_failed']:
            print(f"Failed steps: {', '.join(results['steps_failed'])}")
        sys.exit(1)


if __name__ == '__main__':
    main()