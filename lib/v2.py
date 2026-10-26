"""Stepwise pipeline wrapper for the logic in the exploratory `test2.py` script.

This module splits the original monolithic flow into discrete steps that can be
run idempotently. Each step will skip if its expected output already exists,
    # Load clustering results

Steps implemented:
  1. store: run `store_from_pdb` to create `frame_data_dir` content (atoms.parquet, coords.zarr, ...)
  2. pca: compute ROH-excluded conformation landscape and PCA; write
     `embedding/pca_conformation_landscape.parquet` plus frozen projection artifacts
     in `embedding/landscape/`
  3. torsions: calculate torsion angles for all frames using batch processing and save torsions.parquet and torparts.npz
  4. cluster: compute entropy, fit hierarchical clustering, save dendrogram and clustering outputs
  5. cluster_torsion_stats: compute torsion statistics per cluster and write info.json
  6. save_representative_structures: save PDB structures of cluster representatives with alpha/beta flipping
  7. plot_torsion_distributions: create torsion distribution plots for each clustering level
  8. export_analysis_data: export Parquet analysis tables for Hugging Face and copy key static-ready files

Usage: run this script directly. Example:
  python test2.py --name "MyName" --update
"""

from pathlib import Path
import argparse
import copy
import io
import os
import sys
import logging
import shutil
import time
from typing import Dict, List, Optional, Sequence, Tuple
import polars as pl
import numpy as np
import json
import joblib  # type: ignore

from lib import config
from lib.storage import get_storage_manager
from lib import archive
from lib.glystore import store_trajectory
from lib.glypdbio import parse_mol2_bonds
from lib.frame_quality import (
    FrameQualityResult,
    filter_broken_frames,
    quality_manifest_is_current,
)
from lib.glypdbio import get_conformation_landscape
from lib.embedding import (
    PCA_RANDOM_STATE,
    pca_conformation_landscape,
    knn_conformational_entropy,
    save_clustering_results,
    save_clustering_results_parquet,
    load_clustering_results_parquet,
    _read_csv_robust,
    StandardScaler,
    gmm_optimize_silhouette,
    gmm_optimize_glycosidic_deviation,
    silhouette_scores_shared,
    kcenter_coverage_clustering,
    create_cluster_info,
    create_cluster_info_fast,
    create_cluster_info_fast_big
)
from lib import glypdbio
from lib.torsion import (
    LINKAGE_FAMILIES,
    TORSION_CONVENTIONS,
    build_torsion_metadata,
    classify_glycosidic_from_name,
    circular_stats,
    get_glycan_torsions_enhanced,
    get_torsion_values,
    get_torsion_values_batch,
    plot_torsion_distribution,
    save_torparts_npz,
    summarize_torsion_metadata,
)

try:  # RDKit is optional; torsion steps will be skipped if absent
    from rdkit import Chem
    from rdkit.Geometry import Point3D  # type: ignore
except Exception:  # pragma: no cover
    Chem = None  # type: ignore
    Point3D = None  # type: ignore

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def _path_exists(path: str) -> bool:
    """Return True when a path exists locally or via the active storage backend."""
    if os.path.exists(path):
        return True
    storage = get_storage_manager()
    return storage.exists(path)


def _read_parquet_robust(path: str):
    """Try pyarrow first (avoids glob issues), then fall back to Polars."""
    storage = get_storage_manager()
    if os.path.exists(path):
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
    if not storage.exists(path):
        raise FileNotFoundError(path)
    raw = storage.read_binary(path)
    try:
        import pyarrow.parquet as pq  # type: ignore
        table = pq.read_table(io.BytesIO(raw))
        return pl.from_arrow(table)
    except Exception:
        return pl.read_parquet(io.BytesIO(raw))


def _write_parquet(df: pl.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.write_parquet(path, compression="zstd", statistics=True)


def _read_table_robust(path: str) -> pl.DataFrame:
    if path.endswith(".parquet"):
        return _read_parquet_robust(path)
    return _read_csv_robust(path)


def _first_existing_path(*paths: str) -> Optional[str]:
    for path in paths:
        if _path_exists(path):
            return path
    return None


def _torsions_parquet_path(embedding_dir: str) -> str:
    return os.path.join(embedding_dir, "torsions.parquet")


def _torsions_csv_path(embedding_dir: str) -> str:
    return os.path.join(embedding_dir, "torsions.csv")


def _torsions_source_path(embedding_dir: str) -> Optional[str]:
    return _first_existing_path(_torsions_parquet_path(embedding_dir), _torsions_csv_path(embedding_dir))


def _read_torsions_table(embedding_dir: str) -> pl.DataFrame:
    source = _torsions_source_path(embedding_dir)
    if source is None:
        raise FileNotFoundError(f"No torsions.parquet or torsions.csv found in {embedding_dir}")
    return _read_table_robust(source)


def _unique_column_names(names: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for raw_name in names:
        name = str(raw_name)
        count = seen.get(name, 0)
        if count:
            unique = f"{name}__{count + 1}"
        else:
            unique = name
        seen[name] = count + 1
        out.append(unique)
    return out


def _update_pca_json(embedding_dir: str, updates: Dict) -> None:
    """Merge/update a pca.json file in the embedding_dir with the provided updates."""
    try:
        storage = get_storage_manager()
        p = f"{embedding_dir}/pca.json"
        base = {}
        if storage.exists(p):
            with storage.open(p, "r") as fh:
                base = json.load(fh) or {}
        # deep merge: preserve nested mappings, replace everything else
        def _deep_merge(a: dict, b: dict) -> dict:
            """Return a new dict that's a deep merge of a and b (b overrides a).

            - dict values are merged recursively
            - list and scalar values: b replaces a.  A list is the result of
              the computation that writes it (explained variance ratios, the
              Level 2 clusters of a Level 1 cluster); merging it with the one
              of an earlier run put several runs into one list.
            """
            out = dict(a)
            for k, v in b.items():
                if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                    out[k] = _deep_merge(out[k], v)
                else:
                    out[k] = v
            return out

        base = _deep_merge(base, updates)
        with storage.open(p, "w") as fh:
            json.dump(base, fh, indent=2)
        logger.info(f"Updated {p} with keys: {list(updates.keys())}")
    except Exception as exc:
        logger.warning(f"Failed to update pca.json: {exc}")


def clusters_from_entropy(
    H,
    H_min=5.0,   # ~mono
    H_max=21.0,  # your biggest glycan
    n_min=5,     # clusters for mono
    n_max=128,   # clusters for H≈21
    gamma=1.0    # <1 -> even more top-heavy; >1 -> flatter
):
    if H < 5:
        return 5
    # clamp H to [H_min, H_max]
    Hc = max(H_min, min(H, H_max))
    t = (Hc - H_min) / (H_max - H_min)  # 0..1
    # geometric interpolation; raise t to gamma to tune steepness
    n = n_min * ((n_max / n_min) ** (t ** gamma))
    return int(round(n))


def _load_json_if_exists(path: str, default):
    storage = get_storage_manager()
    if not storage.exists(path):
        return default
    try:
        with storage.open(path, "r") as fh:
            return json.load(fh)
    except Exception:
        return default


def _write_json(path: str, payload) -> None:
    storage = get_storage_manager()
    with storage.open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


def _fallback_torsion_metadata_from_labels(labels: List[str]) -> List[Dict]:
    metadata: List[Dict] = []
    for index, label in enumerate(labels):
        canonical_name = classify_glycosidic_from_name(label)
        is_glycosidic = isinstance(canonical_name, str) and not canonical_name.startswith("other_")
        subtype = canonical_name.rsplit("_", 1)[-1] if is_glycosidic else "unknown"
        metadata.append({
            "index": int(index),
            "original_name": label,
            "canonical_name": canonical_name if is_glycosidic else label.replace("-", "__"),
            "family": "glycosidic" if is_glycosidic else "unknown",
            "subtype": subtype,
            "common_name": subtype if is_glycosidic else None,
            "atom_indices": [],
            "atom_names": [],
            "residue_ids": [],
            "residue_names": [],
            "central_bond_atoms": [],
            "central_bond_residues": [],
            **({"glycosidic_name": canonical_name} if is_glycosidic else {}),
        })
    return metadata


def _write_torsion_annotation_files(
    embedding_dir: str,
    metadata: List[Dict],
    torsion_list: List[Tuple[int, int, int, int]],
) -> None:
    metadata_path = os.path.join(embedding_dir, "torsion_metadata.json")
    _write_json(metadata_path, metadata)

    glycosidic_torsions = []
    for entry in metadata:
        if entry.get("family") == "glycosidic":
            glycosidic_torsions.append({
                "index": int(entry["index"]),
                "original_name": entry["original_name"],
                "glycosidic_name": entry.get("glycosidic_name") or entry["canonical_name"],
                "torsion_indices": torsion_list[int(entry["index"])],
                "canonical_name": entry["canonical_name"],
                "family": entry.get("family"),
                "subtype": entry.get("subtype"),
                "common_name": entry.get("common_name"),
                "atom_indices": entry.get("atom_indices", []),
                "atom_names": entry.get("atom_names", []),
                "residue_ids": entry.get("residue_ids", []),
                "residue_names": entry.get("residue_names", []),
                "central_bond_atoms": entry.get("central_bond_atoms", []),
                "central_bond_residues": entry.get("central_bond_residues", []),
            })
    glycosidic_path = os.path.join(embedding_dir, "glycosidic_info.json")
    _write_json(glycosidic_path, glycosidic_torsions)


def _persist_torsion_annotation_files(
    frame_data_dir: str,
    embedding_dir: str,
    torsion_list: List[Tuple[int, int, int, int]],
    labels: List[str],
) -> List[Dict]:
    atoms_df = glypdbio.load_atoms(frame_data_dir)
    bonds_df = glypdbio.load_bonds(frame_data_dir)
    metadata = build_torsion_metadata(
        torsion_list,
        atoms_df=atoms_df,
        original_names=labels,
        bonds_df=bonds_df,
    )
    _write_torsion_annotation_files(embedding_dir, metadata, torsion_list)
    return metadata


def _torsion_labels(torsion_list: List[Tuple[int, int, int, int]], atoms_df: pl.DataFrame) -> List[str]:
    """Column names of a torsion table: the four atoms as <atom name>_<residue id>, made unique."""
    elements = atoms_df["element"].to_list()
    atom_names = atoms_df["name"].to_list() if "name" in atoms_df.columns else [f"{elements[i]}{i}" for i in range(len(elements))]
    res_ids = atoms_df["res_id"].to_list() if "res_id" in atoms_df.columns else [1] * len(elements)
    return _unique_column_names([
        "-".join(f"{atom_names[i]}_{res_ids[i]}" for i in torsion)
        for torsion in torsion_list
    ])


def _stored_torsion_labels(embedding_dir: str) -> Optional[List[str]]:
    """The torsion columns of the stored table (without ``frame``), or None when there is none."""
    source = _torsions_source_path(embedding_dir)
    if source is None:
        return None
    header = _read_parquet_header(source) if source.endswith(".parquet") else _read_csv_header(source)
    if not header:
        return None
    return [column for column in header if column != "frame"]


def torsion_table_is_current(frame_data_dir: str, embedding_dir: str) -> bool:
    """Whether the stored torsion table has the columns the current definitions give.

    A table computed with earlier torsion definitions (``TORSION_CONVENTIONS``
    version) has other atoms in some columns (psi with C(x+1), omega
    C4-C5-C6-O6); it must be computed again, and what is derived from it
    with it.  When the structure cannot be read the table is taken as current.
    """
    stored = _stored_torsion_labels(embedding_dir)
    if stored is None:
        return False
    try:
        atoms_df = glypdbio.load_atoms(frame_data_dir)
        bonds_df = glypdbio.load_bonds(frame_data_dir)
        if bonds_df.is_empty():
            return True
        current = _torsion_labels(get_glycan_torsions_enhanced(atoms_df, bonds_df), atoms_df)
    except Exception:
        return True
    return list(stored) == list(current)


def _load_or_build_torsion_metadata(
    frame_data_dir: str,
    embedding_dir: str,
    torsion_labels: List[str],
    torsion_list: Optional[List[Tuple[int, int, int, int]]] = None,
) -> List[Dict]:
    """Metadata of the columns of the stored torsion table, from the structure.

    The columns must be those the current definitions give: metadata of the
    current torsions over values measured on other atoms would be wrong, so a
    table computed with earlier definitions raises (``torsion_table_is_current``
    is what the analysis checks first).  Without the structure (atoms, bonds,
    RDKit) the labels are classified by name.
    """
    metadata_path = os.path.join(embedding_dir, "torsion_metadata.json")
    try:
        atoms_df = glypdbio.load_atoms(frame_data_dir)
        bonds_df = glypdbio.load_bonds(frame_data_dir)
        if bonds_df.is_empty():
            raise ValueError("no bonds stored")
        if torsion_list is None:
            torsion_list = get_glycan_torsions_enhanced(atoms_df, bonds_df)
        current_labels = _torsion_labels(torsion_list, atoms_df)
    except Exception as exc:
        logger.warning(f"Could not rebuild torsion metadata from structure; using label fallback ({exc})")
        metadata = _fallback_torsion_metadata_from_labels(torsion_labels)
        _write_json(metadata_path, metadata)
        return metadata

    if list(current_labels) != list(torsion_labels):
        raise ValueError(
            f"The torsion table in {embedding_dir} has other columns than the torsion definitions "
            f"(version {TORSION_CONVENTIONS['version']}) give: it was computed with earlier definitions. "
            "Compute the torsions again (--update)."
        )
    metadata = build_torsion_metadata(
        torsion_list,
        atoms_df=atoms_df,
        original_names=list(torsion_labels),
        bonds_df=bonds_df,
    )
    if _load_json_if_exists(metadata_path, None) != metadata:
        _write_torsion_annotation_files(embedding_dir, metadata, torsion_list)
    return metadata


def _canonical_torsion_export_names(torsion_metadata: List[Dict]) -> List[str]:
    counts: Dict[str, int] = {}
    for entry in torsion_metadata:
        base_name = str(entry.get("canonical_name") or entry.get("original_name") or f"torsion_{entry.get('index', 0)}")
        counts[base_name] = counts.get(base_name, 0) + 1

    names: List[str] = []
    for entry in torsion_metadata:
        base_name = str(entry.get("canonical_name") or entry.get("original_name") or f"torsion_{entry.get('index', 0)}")
        if counts.get(base_name, 0) > 1:
            names.append(f"{base_name}__{int(entry.get('index', len(names)))}")
        else:
            names.append(base_name)
    return names


def _align_torsion_metadata_to_labels(torsion_labels: List[str], torsion_metadata: List[Dict]) -> List[Dict]:
    if len(torsion_metadata) == len(torsion_labels) and all(
        torsion_metadata[i].get("original_name") == torsion_labels[i] for i in range(len(torsion_labels))
    ):
        return torsion_metadata
    return _fallback_torsion_metadata_from_labels(torsion_labels)


def _level_summary_from_metadata(metadata: List[Dict]) -> Dict:
    family_summary = summarize_torsion_metadata(metadata)
    glycosidic = [entry for entry in metadata if entry.get("family") == "glycosidic"]
    # The omega of a sulfate, phosphate, methyl or acetyl group on an
    # exocyclic oxygen (6-O-sulfate ...): a block of its own, not glycosidic.
    substituent = [entry for entry in metadata if entry.get("family") == "substituent"]
    non_glycosidic = [entry for entry in metadata if entry.get("family") not in LINKAGE_FAMILIES]
    return {
        "metadata": metadata,
        # The definitions of phi, psi and omega (IUPAC; the atoms of every case).
        "conventions": copy.deepcopy(TORSION_CONVENTIONS),
        "name_mapping": {entry["original_name"]: entry["canonical_name"] for entry in metadata},
        "glycosidic": glycosidic,
        "substituent": substituent,
        "non_glycosidic": non_glycosidic,
        "family_summary": family_summary,
        # Compatibility aliases for older consumers.
        "other": non_glycosidic,
        "glycosidic_summary": {
            "total_torsions": len(metadata),
            "glycosidic_torsions": len(glycosidic),
            "substituent_torsions": len(substituent),
            "other_torsions": len(non_glycosidic),
            "phi_count": sum(1 for entry in glycosidic if entry.get("subtype") == "phi"),
            "psi_count": sum(1 for entry in glycosidic if entry.get("subtype") == "psi"),
            "omega_count": sum(1 for entry in glycosidic if entry.get("subtype") == "omega"),
        },
    }


def step_store(
    name: str,
    frame_data_dir: str,
    pdb_path: str,
    mol2_path: str,
    force: bool = False,
) -> FrameQualityResult:
    """Atomically store, recover topology, and remove truly broken frames."""
    out_p = Path(frame_data_dir)
    atoms_p = out_p / "atoms.parquet"
    threshold = config.max_covalent_bond_angstrom
    if (
        atoms_p.exists()
        and not force
        and quality_manifest_is_current(out_p, threshold, excluded=config.excluded_source_frames)
    ):
        manifest = _load_json_if_exists(str(out_p / "frame_filter.json"), {})
        logger.info(
            "Store/quality step is current: retained %s/%s source frames",
            manifest.get("retained_frames"),
            manifest.get("source_frames"),
        )
        return FrameQualityResult(
            source_frames=int(manifest.get("source_frames", 0)),
            retained_frames=int(manifest.get("retained_frames", 0)),
            rejected_frames=int(manifest.get("rejected_frames", 0)),
            input_bonds=int(manifest.get("input_bonds", 0)),
            retained_input_bonds=int(manifest.get("retained_input_bonds", 0)),
            inferred_bonds=int(manifest.get("inferred_bonds", 0)),
            topology_repaired=bool(manifest.get("topology_repaired", False)),
            changed=False,
        )

    logger.info("Running atomic trajectory store and bond-quality filtering")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_p.parent / f".{out_p.name}.store.tmp.{os.getpid()}"
    backup = out_p.parent / f".{out_p.name}.store.backup.{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    try:
        store_trajectory(
            pdb_path,
            str(temporary),
            get_connectivity=parse_mol2_bonds(mol2_path),
            connectivity_kind="serial",
            frames_buffer=config.store_frames_buffer,
            frame_chunk=config.store_frame_chunk,
        )
        result = filter_broken_frames(
            temporary,
            max_bond_distance=threshold,
            block_frames=config.frame_quality_block_frames,
            frame_chunk=config.store_frame_chunk,
            excluded=config.excluded_source_frames,
        )

        if backup.exists():
            shutil.rmtree(backup)
        if out_p.exists():
            out_p.rename(backup)
        temporary.rename(out_p)
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if not out_p.exists() and backup.exists():
            backup.rename(out_p)
        raise

    logger.info(
        "Frame quality retained %d/%d source frames; rejected=%d, inferred_bonds=%d",
        result.retained_frames,
        result.source_frames,
        result.rejected_frames,
        result.inferred_bonds,
    )
    return result


def _build_atom_table_for_landscape(frame_data_dir: str, chosen_atoms: np.ndarray) -> List[Dict]:
    atoms = glypdbio.load_atoms(frame_data_dir)
    names = atoms["name"].to_list()
    res_names = atoms["res_name"].to_list()
    res_ids = atoms["res_id"].to_list()
    elements = atoms["element"].to_list()
    serials = atoms["serial"].to_list()

    atom_table = []
    for idx in np.asarray(chosen_atoms, dtype=int):
        i = int(idx)
        atom_table.append({
            "index": i,
            "serial": int(serials[i]),
            "key": f"{str(res_names[i])}:{str(names[i])}:{int(res_ids[i])}",
            "name": str(names[i]),
            "res_name": str(res_names[i]),
            "res_id": int(res_ids[i]),
            "element": str(elements[i]),
        })
    return atom_table


def _run_in_threads(function, arguments: List, n_threads: int) -> None:
    if n_threads > 1 and len(arguments) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            list(pool.map(function, arguments))
    else:
        for argument in arguments:
            function(argument)


def _column_blocks(n_rows: int, n_columns: int, block_bytes: int = 64 * 2**20) -> List[Tuple[int, int]]:
    """Column ranges of at least two columns holding about block_bytes of float64 each.

    Two columns at least: NumPy sums a single strided column pairwise, but
    the columns of a wider block row after row, as it does for the whole
    matrix. That order is what keeps the block results equal to X.mean(axis=0)
    and X.std(axis=0) bit for bit.
    """
    width = max(2, block_bytes // (8 * max(1, n_rows)))
    starts = list(range(0, n_columns, width))
    blocks = [(start, min(start + width, n_columns)) for start in starts]
    if len(blocks) > 1 and blocks[-1][1] - blocks[-1][0] < 2:
        last = blocks.pop()
        blocks[-1] = (blocks[-1][0], last[1])
    return blocks


def _column_mean(X: np.ndarray, n_threads: int = 1) -> np.ndarray:
    """X.mean(axis=0), computed block of columns by block of columns, in threads."""
    out = np.empty(X.shape[1], dtype=np.float64)

    def block_mean(block: Tuple[int, int]) -> None:
        out[block[0]:block[1]] = X[:, block[0]:block[1]].mean(axis=0)

    _run_in_threads(block_mean, _column_blocks(*X.shape), n_threads)
    return out


def _column_std(X: np.ndarray, n_threads: int = 1) -> np.ndarray:
    """X.std(axis=0, ddof=1) without the temporary of the size of X that np.std makes."""
    out = np.empty(X.shape[1], dtype=np.float64)

    def block_std(block: Tuple[int, int]) -> None:
        out[block[0]:block[1]] = X[:, block[0]:block[1]].std(axis=0, ddof=1)

    _run_in_threads(block_std, _column_blocks(*X.shape), n_threads)
    return out


def _scale_rows(X: np.ndarray, operation, row: np.ndarray, n_threads: int = 1) -> None:
    """X = operation(X, row) in place (X -= row, X /= row), blocks of rows in threads."""
    n_rows = X.shape[0]
    step = max(1, -(-n_rows // max(1, 4 * n_threads)))

    def apply(start: int) -> None:
        operation(X[start:start + step], row, out=X[start:start + step])

    _run_in_threads(apply, list(range(0, n_rows, step)), n_threads)


def _fit_pca_and_write_landscape(
    frame_data_dir: str,
    embedding_dir: str,
    landscape: np.ndarray,
    chosen_atoms: np.ndarray,
    *,
    per_pair_scale: bool = False,
    method: str = "noH_noROH_fast",
    sample_size: int = 300,
    seed: int = 0,
    overwrite_landscape: bool = False,
) -> tuple[pl.DataFrame, List[float], List[float]]:
    """Fit PCA once and persist both frame scores and reusable projection artifacts.

    The float64 matrix of the fit is the largest object of the pipeline
    (frames x pairs x 8 bytes: 18.8 GB for 159,940 frames and 14,706 pairs).
    It exists once: with ``overwrite_landscape`` a float64 landscape is
    centred in place (and, with ``per_pair_scale``, divided by the standard
    deviation of each pair, taken block of columns by block of columns: NumPy's
    std otherwise allocates a second matrix), and scikit-learn centres the
    matrix itself instead of a copy of it.

    The distances are centred but not divided by their standard deviations:
    every pair keeps its variance in square angstroms, so the components follow
    the large motions of the glycan rather than giving the fluctuation of a
    bonded distance the same weight as an antenna moving.
    ``per_pair_scale`` divides each pair by its standard deviation first (the
    correlation-matrix PCA of the earlier database).
    """
    if np.ndim(landscape) != 2:
        raise ValueError("landscape must be 2D (n_frames, n_features)")

    n_frames, n_pairs = np.shape(landscape)
    if n_frames == 0:
        return pl.DataFrame({"frame": []}), [], []
    if n_pairs == 0:
        raise ValueError("No atom pairs available for PCA landscape")

    if (
        overwrite_landscape
        and isinstance(landscape, np.ndarray)
        and landscape.dtype == np.float64
        and landscape.flags.c_contiguous
        and landscape.flags.writeable
    ):
        X_proc = landscape
    else:
        X_proc = np.array(landscape, dtype=float, order="C")
    del landscape
    n_threads = max(1, int(config.max_workers))
    pair_stds = None
    if per_pair_scale:
        pair_stds = _column_std(X_proc, n_threads)
        pair_stds = np.nan_to_num(pair_stds, nan=1.0, posinf=1.0, neginf=1.0)
        pair_stds[pair_stds == 0] = 1.0
        _scale_rows(X_proc, np.divide, pair_stds, n_threads)

    feature_means = _column_mean(X_proc, n_threads)
    _scale_rows(X_proc, np.subtract, feature_means, n_threads)

    n_components = min(10, n_frames, n_pairs)
    from sklearn.decomposition import PCA as _SkPCA
    # svd_solver="auto" takes the randomized solver for most glycans (more
    # than 1,000 atom pairs, or fewer than ten frames per pair); its draws are
    # seeded so that the same landscape always gives the same projection.
    pca = _SkPCA(n_components=n_components, svd_solver="auto", random_state=PCA_RANDOM_STATE)
    # copy=False only makes the fit centre X_proc itself rather than a copy;
    # X_proc is not used afterwards. The stored model keeps the default.
    pca.copy = False
    scores = pca.fit_transform(X_proc)
    pca.copy = True
    del X_proc
    evr_arr = np.nan_to_num(pca.explained_variance_ratio_, nan=0.0, posinf=0.0, neginf=0.0)
    evr = [float(x) for x in evr_arr]
    cumulative = [float(sum(evr[: i + 1])) for i in range(len(evr))]

    data = {"frame": np.arange(n_frames, dtype=int)}
    for idx in range(n_components):
        data[f"PC{idx + 1}"] = scores[:, idx]
    data["explained_variance_ratio"] = [evr] * n_frames
    pca_df = pl.DataFrame(data)

    landscape_dir = os.path.join(embedding_dir, "landscape")
    os.makedirs(landscape_dir, exist_ok=True)
    model_path = os.path.join(landscape_dir, "pca_model.joblib")
    meta_path = os.path.join(landscape_dir, "landscape_metadata.json")
    preproc_path = os.path.join(landscape_dir, "preprocessing.npz")

    joblib.dump(pca, model_path)
    logger.info(f"Saved PCA model: {model_path}")

    chosen_sorted = np.asarray(chosen_atoms, dtype=int)
    pair_i, pair_j = np.tril_indices(int(chosen_sorted.size), k=-1)
    metadata = {
        "method": method,
        "sample_size": int(sample_size),
        "seed": int(seed),
        "n_frames": int(n_frames),
        "n_atoms": int(chosen_sorted.size),
        "n_pairs": int(n_pairs),
        "n_components": int(n_components),
        "center": True,
        "scale": bool(per_pair_scale),
        "per_pair_scale": bool(per_pair_scale),
        "explained_variance_ratio": evr,
        "cumulative_variance": cumulative,
        "atoms": _build_atom_table_for_landscape(frame_data_dir, chosen_sorted),
    }

    storage = get_storage_manager()
    with storage.open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved landscape metadata: {meta_path}")

    npz_data = dict(
        feature_means=feature_means.astype(np.float64),
        pair_i=pair_i.astype(np.int32),
        pair_j=pair_j.astype(np.int32),
        atom_indices=chosen_sorted.astype(np.int32),
    )
    if pair_stds is not None:
        npz_data["pair_stds"] = pair_stds.astype(np.float64)
    np.savez_compressed(preproc_path, **npz_data)
    logger.info(f"Saved preprocessing data: {preproc_path}")

    return pca_df, evr, cumulative


def step_pca(frame_data_dir: str, embedding_dir: str, force: bool = False, per_pair_scale: bool = False) -> str:
    """Compute ROH-excluded conformation landscape and PCA, save parquet and model artifacts."""
    os.makedirs(embedding_dir, exist_ok=True)
    pca_path = os.path.join(embedding_dir, "pca_conformation_landscape.parquet")
    model_path = os.path.join(embedding_dir, "landscape", "pca_model.joblib")
    meta_path = os.path.join(embedding_dir, "landscape", "landscape_metadata.json")
    preproc_path = os.path.join(embedding_dir, "landscape", "preprocessing.npz")
    if _path_exists(pca_path) and _path_exists(model_path) and _path_exists(meta_path) and _path_exists(preproc_path) and not force:
        logger.info("PCA step: pca file exists; skipping (use --update to force)")
        return pca_path

    logger.info("Computing ROH-excluded conformation landscape (may be memory intensive)")
    # The landscape is asked for as float64 (float32 distances, widened) and
    # handed over to the fit, which centres it in place: one matrix
    # instead of the float32 one, its float64 copy and the working copy.
    landscape, chosen_atoms = get_conformation_landscape(
        frame_data_dir,
        method="noH_noROH_fast",
        sample_size=300,
        seed=0,
        return_atom_indices=True,
        dtype=np.float64,
        n_threads=max(1, int(config.max_workers)),
    )
    logger.info("Running PCA on conformation landscape and saving frozen projection artifacts")
    pca_df, evr_list, cumulative = _fit_pca_and_write_landscape(
        frame_data_dir,
        embedding_dir,
        landscape,
        chosen_atoms,
        per_pair_scale=per_pair_scale,
        method="noH_noROH_fast",
        sample_size=300,
        seed=0,
        overwrite_landscape=True,
    )
    del landscape
    _write_parquet(pca_df, pca_path)
    logger.info(f"Wrote PCA parquet: {pca_path}")

    # Save explained variance ratios to pca.json (robust extraction)
    try:
        _update_pca_json(
            embedding_dir,
            {
                "pca_analysis": {
                    "explained_variance_ratio": evr_list,
                    "cumulative_variance": cumulative,
                    "n_components": len(evr_list),
                    "params": {
                        "center": True,
                        "scale": bool(per_pair_scale),
                        "per_pair_scale": per_pair_scale,
                        "method": "noH_noROH_fast",
                        "sample_size": 300,
                        "seed": 0,
                    },
                }
            },
        )
    except Exception as exc:
        logger.warning(f"Could not extract/save PCA variance info: {exc}")

    return pca_path


def _build_cluster_member_lookup(labels: np.ndarray, n_clusters: int) -> Dict[int, np.ndarray]:
    return {
        int(cluster_id): np.flatnonzero(labels == cluster_id)
        for cluster_id in range(int(n_clusters))
    }


def _build_frame_row_lookup(df: pl.DataFrame, frame_col: str = "frame") -> tuple[np.ndarray, Dict[int, int]]:
    frames = np.asarray(df[frame_col].to_list(), dtype=int)
    return frames, {int(frame): idx for idx, frame in enumerate(frames.tolist())}


def _required_cluster_levels() -> Tuple[int, ...]:
    """Levels that are computed, exported and required for completeness."""
    return tuple(int(level) for level in config.cluster_levels)


def _level_assignment_columns() -> List[str]:
    """Per-frame cluster and center columns of torsion_all for the configured levels."""
    return [
        f"level_{level}_{kind}"
        for level in _required_cluster_levels()
        for kind in ("cluster", "center")
    ]


def _levels_to_process(cr) -> List[int]:
    """Stored levels that are configured; an older folder may hold more."""
    wanted = set(_required_cluster_levels())
    return [int(level) for level in cr.levels() if int(level) in wanted]


def coverage_clusters_per_main(
    level_1: List[Dict],
    level_2: List[Dict],
    n_frames: int,
    n_main: Optional[int] = None,
) -> Dict[int, List[int]]:
    """The Level 2 clusters of every Level 1 cluster, both in the published numbering.

    ``level_1`` and ``level_2`` are the cluster lists that
    ``create_cluster_info_fast`` returns and the clustering results store:
    sorted by decreasing size, ``cluster_id`` being the number under which a
    cluster is published (``Cluster 0`` the largest) and ``members`` its
    frames.  A Level 2 cluster belongs to the Level 1 cluster that holds most
    of its frames (the lower number on a tie).  The keys are the Level 1
    clusters ``0 .. n_main - 1``; the Level 2 clusters of each are in
    increasing order.  (The labels of the clustering algorithms themselves
    are numbered before this sorting and must not be used here.)
    """
    main_of_frame = np.full(int(n_frames), -1, dtype=int)
    for cluster in level_1:
        main_of_frame[np.asarray(cluster["members"], dtype=int)] = int(cluster["cluster_id"])
    keys = range(int(n_main)) if n_main is not None else sorted(int(c["cluster_id"]) for c in level_1)
    mapping: Dict[int, List[int]] = {int(key): [] for key in keys}
    for cluster in sorted(level_2, key=lambda c: int(c["cluster_id"])):
        mains = main_of_frame[np.asarray(cluster["members"], dtype=int)]
        mains = mains[mains >= 0]
        if mains.size == 0:
            continue
        unique, counts = np.unique(mains, return_counts=True)
        mapping.setdefault(int(unique[np.argmax(counts)]), []).append(int(cluster["cluster_id"]))
    return mapping


def step_clustering_gmm(embedding_dir: str, pca_path: str, force: bool = False, return_context: bool = False):
    """Run two-level clustering:
       - main: H_nats-guided target (1–8) with local silhouette refinement; save silhouettes to pca.json
       - coverage: linear H_nats -> [10..128] via k-center.
       - level 3 (k-center, up to 512 centers) only when config.cluster_levels includes it.
       
       Returns the conformational entropy H (nats). With return_context the
       context says whether the clustering was computed ("computed") or reused.
    """
    levels = _required_cluster_levels()
    pca_df = _read_parquet_robust(pca_path)
    selected_cols = [f"PC{i}" for i in range(1, 4)]
    all_pc_cols = [col for col in pca_df.columns if col.startswith("PC") and col[2:].isdigit()]
    pca_data_scaled = None
    if all_pc_cols:
        try:
            pca_all = pca_df.select(all_pc_cols).to_numpy().astype(float)
            if StandardScaler is not None:
                pca_data_scaled = StandardScaler().fit_transform(pca_all)
            else:
                mean = pca_all.mean(axis=0, keepdims=True)
                std = pca_all.std(axis=0, keepdims=True)
                std[std == 0] = 1.0
                pca_data_scaled = (pca_all - mean) / std
        except Exception as exc:
            logger.warning(f"Could not pre-standardize PCA data for representatives: {exc}")

    logger.info("Estimating conformational entropy (kNN)")
    H = float(knn_conformational_entropy(pca_df, k=5, n_components=3))
    logger.info(f"Entropy (nats): {H}")

    cluster_out_json = os.path.join(embedding_dir, "clustering_results.json")
    if _path_exists(cluster_out_json) and not force:
        try:
            cr_existing = load_clustering_results_parquet(embedding_dir)
            existing_levels = set(cr_existing.levels())
        except Exception:
            existing_levels = set()

        if set(levels).issubset(existing_levels):
            logger.info(f"Clustering step: outputs exist and include levels {list(levels)}; skipping")
            pca_updates: Dict = {"main_clustering": {"H_nats": H}}
            if 3 in levels:
                pca_updates["level_3_clustering"] = {
                    "selected_n_clusters": int(cr_existing.get_n_clusters(3) or 0),
                }
            _update_pca_json(embedding_dir, pca_updates)
            if return_context:
                return H, {"cr": cr_existing, "pca_df": pca_df, "computed": False}
            return H

        logger.info(f"Clustering step: existing outputs do not include levels {list(levels)}; recomputing")

    # Extract PCA data
    pca_data = pca_df.select(selected_cols).to_numpy()

    # MAIN: Fixed 5 clusters, but compute silhouette scores for n=2..8 for diagnostics
    n_main = 5
    
    # Compute silhouette scores for n in [2..8]
    silhouette_scores: Dict[str, Optional[float]] = {}
    try:
        from sklearn.mixture import GaussianMixture
        from sklearn.metrics import silhouette_score as _silhouette_score
        
        labels_by_n: Dict[int, Optional[np.ndarray]] = {}
        for n in range(2, 9):
            try:
                gmm = GaussianMixture(n_components=n, random_state=42)
                labels_by_n[n] = gmm.fit_predict(pca_data)
            except Exception:
                labels_by_n[n] = None

        # The silhouette of every n needs all pairwise distances of the same
        # points: compute them once for the seven labelings, on all workers.
        fitted = [n for n in range(2, 9) if labels_by_n[n] is not None]
        shared_scores: Optional[Dict[int, Optional[float]]] = None
        try:
            shared_scores = dict(zip(
                fitted,
                silhouette_scores_shared(pca_data, [labels_by_n[n] for n in fitted]),
            ))
        except Exception as exc:
            logger.warning(f"Shared silhouette computation failed ({exc}); computing one labeling at a time")

        for n in range(2, 9):
            try:
                labels_n = labels_by_n[n]
                if labels_n is None:
                    raise ValueError(f"GMM with {n} components was not fitted")
                if len(np.unique(labels_n)) > 1:
                    if shared_scores is not None:
                        score = shared_scores[n]
                        if score is None:
                            raise ValueError("silhouette is not defined for this labeling")
                    else:
                        score = float(_silhouette_score(pca_data, labels_n))
                    silhouette_scores[str(n)] = score
                else:
                    silhouette_scores[str(n)] = None
            except Exception:
                silhouette_scores[str(n)] = None
        
        # Main clustering with 5 clusters: the fit of n=5 above (same data,
        # same random_state); fitted here only if that one failed.
        if labels_by_n.get(n_main) is not None:
            labels_main = labels_by_n[n_main]
        else:
            gmm_main = GaussianMixture(n_components=n_main, random_state=42)
            labels_main = gmm_main.fit_predict(pca_data)

    except Exception as exc:
        logger.warning(f"sklearn unavailable or failed ({exc}); defaulting main clustering via single cluster")
        silhouette_scores = {}
        n_main = 1
        labels_main = np.zeros(len(pca_data), dtype=int)

    # COVERAGE: exponential scaling H in [4,21] -> clusters in [5,128]
    n_cov_target = clusters_from_entropy(H)

    labels_cov, n_clusters_cov = kcenter_coverage_clustering(
        pca_data,
        max_clusters=n_cov_target,
        coverage_threshold=0.95,
        min_coverage_improvement=0.01
    )

    # LEVEL 3: k-center with up to 512 centers, only when configured. The
    # k-center clustering draws nothing from a random generator, so levels 1
    # and 2 do not depend on whether this one runs.
    n_frames = int(len(pca_data))
    n_clusters_level_3 = None
    if 3 in levels:
        n_level_3_target = 512 if n_frames >= 512 else n_frames
        labels_level_3, n_clusters_level_3 = kcenter_coverage_clustering(
            pca_data,
            max_clusters=n_level_3_target,
            coverage_threshold=0.95,
            min_coverage_improvement=0.01,
        )

    main_member_lookup = _build_cluster_member_lookup(labels_main, 1 if n_main == 1 else n_main)
    cov_member_lookup = _build_cluster_member_lookup(labels_cov, n_clusters_cov)

    # Format results and save
    all_levels = {
        1: create_cluster_info_fast(
            labels_main,
            pca_df,
            1 if n_main == 1 else n_main,
            pca_data_scaled=pca_data_scaled,
            cluster_member_indices=main_member_lookup,
        ),
        2: create_cluster_info_fast(
            labels_cov,
            pca_df,
            n_clusters_cov,
            pca_data_scaled=pca_data_scaled,
            cluster_member_indices=cov_member_lookup,
        ),
    }
    if n_clusters_level_3 is not None:
        level3_member_lookup = _build_cluster_member_lookup(labels_level_3, n_clusters_level_3)
        all_levels[3] = create_cluster_info_fast(
            labels_level_3,
            pca_df,
            n_clusters_level_3,
            pca_data_scaled=pca_data_scaled,
            cluster_member_indices=level3_member_lookup,
        )

    save_clustering_results(all_levels, output_path=cluster_out_json)
    save_clustering_results_parquet(all_levels, output_dir=embedding_dir)

    # Map coverage clusters to main clusters, in the published numbering of
    # both levels (the clusters as create_cluster_info_fast sorted them).
    try:
        main_to_coverage_map = coverage_clusters_per_main(
            all_levels[1], all_levels[2], n_frames=n_frames, n_main=1 if n_main == 1 else n_main
        )
    except Exception as exc:
        logger.warning(f"Failed to map coverage clusters to main clusters: {exc}")
        main_to_coverage_map = {i: [] for i in range(n_main)}

    # Update pca.json with main silhouettes and selections
    pca_updates = {
        "main_clustering": {
            "H_nats": H,
            "silhouette_scores": silhouette_scores,
            "selected_n_clusters": int(1 if n_main == 1 else n_main),
            "coverage_clusters_per_main": {str(k): v for k, v in sorted(main_to_coverage_map.items())},
        },
        "coverage_clustering": {
            "selected_n_clusters": int(n_clusters_cov),
        },
    }
    if n_clusters_level_3 is not None:
        pca_updates["level_3_clustering"] = {
            "selected_n_clusters": int(n_clusters_level_3),
        }
    _update_pca_json(embedding_dir, pca_updates)

    logger.info(f"Saved clustering results to {cluster_out_json} and parquet in {embedding_dir}")
    logger.info(
        f"Main clusters: {1 if n_main == 1 else n_main}; "
        f"Coverage clusters: {n_clusters_cov}"
        + (f"; Level 3 coverage clusters: {n_clusters_level_3}" if n_clusters_level_3 is not None else "")
    )
    # Quick read-back and simple diagnostics
    cr = load_clustering_results_parquet(embedding_dir)
    logger.info(f"Cluster reader levels: {cr.levels()}")
    if return_context:
        return H, {"cr": cr, "pca_df": pca_df, "computed": True}
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
    """Compute torsion angles for all frames and write torsions.parquet in embedding_dir.
    Also saves torparts.npz with torsion parts structure.

    Returns (torsion_list, torsion_labels). If RDKit not available, returns ([],[]).
    """
    storage = get_storage_manager()
    storage.mkdir(embedding_dir)
    torsion_parquet = _torsions_parquet_path(embedding_dir)
    torsion_csv = _torsions_csv_path(embedding_dir)
    torparts_path = f"{embedding_dir}/torparts.npz"
    
    torsion_existing = _first_existing_path(torsion_parquet, torsion_csv)
    if torsion_existing and storage.exists(torparts_path) and not force:
        if not torsion_table_is_current(frame_data_dir, embedding_dir):
            logger.info(
                "Torsion step: the stored table was computed with earlier torsion definitions "
                f"(now version {TORSION_CONVENTIONS['version']}); computing it again"
            )
            return step_torsions(frame_data_dir, embedding_dir, force=True)
        logger.info("Torsion step: torsion table and torparts.npz exist; skipping (use --update to force)")
        # Attempt to recover labels by reading header
        try:
            header = _read_table_robust(torsion_existing).columns
            labels = header[1:]
        except Exception:
            labels = []
        if labels:
            _load_or_build_torsion_metadata(frame_data_dir, embedding_dir, labels)
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
    
    # atom_name_residue_number format (e.g., C1_2, O2_3, C3_3, C4_2)
    labels = _torsion_labels(torsion_list, atoms_df)
    
    if not torsion_list:
        logger.info("No rotatable torsions detected; writing frame-only torsions.parquet")
        coords = glypdbio.load_coords(frame_data_dir)
        n_frames = int(coords.shape[0])
        _write_parquet(pl.DataFrame({"frame": np.arange(n_frames, dtype=np.int32)}), torsion_parquet)
        save_torparts_npz([], bonds_df, atoms_df, torparts_path)
        _write_json(os.path.join(embedding_dir, "torsion_metadata.json"), [])
        _write_json(os.path.join(embedding_dir, "glycosidic_info.json"), [])
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
    
    logger.info(f"Writing {n_frames} x {len(torsion_list)} torsion values to Parquet")
    torsion_data = {"frame": np.arange(n_frames, dtype=np.int32)}
    for idx, label in enumerate(labels):
        torsion_data[label] = torsion_angles[:, idx].astype(np.float32)
    _write_parquet(pl.DataFrame(torsion_data), torsion_parquet)
    logger.info(f"Wrote torsion angles to {torsion_parquet}")

    metadata = _persist_torsion_annotation_files(frame_data_dir, embedding_dir, torsion_list, labels)
    logger.info(f"Saved torsion metadata for {len(metadata)} torsions to {embedding_dir}/torsion_metadata.json")
    
    return torsion_list, labels


def step_cluster_torsion_stats(
    frame_data_dir: str,
    embedding_dir: str,
    force: bool = False,
    *,
    cr=None,
    tors_df: Optional[pl.DataFrame] = None,
    pca_df: Optional[pl.DataFrame] = None,
) -> None:
    """Compute torsion stats per cluster and write torsion_stats.json in embedding_dir.

    Expects torsions.parquet or legacy torsions.csv (skip silently if missing)
    and clustering parquet outputs.
    Reads torsion names directly from the table and performs glycosidic classification.
    """
    stats_path = os.path.join(embedding_dir, "torsion_stats.json")
    # If file exists but appears incomplete (empty torsions/levels), recompute
    if _path_exists(stats_path) and not force:
        try:
            storage = get_storage_manager()
            with storage.open(stats_path, 'r') as f:
                existing = json.load(f)
            torsions_block = existing.get('torsions', {}) if isinstance(existing, dict) else {}
            tors_ok = (
                isinstance(existing, dict)
                and torsions_block.get('labels')
                and torsions_block.get('metadata')
                and torsions_block.get('family_summary')
            )
            level_keys = {
                int(level.get('level'))
                for level in existing.get('levels', [])
                if isinstance(level, dict) and level.get('level') is not None
            } if isinstance(existing, dict) else set()
            levels_ok = set(_required_cluster_levels()).issubset(level_keys)
            if tors_ok and levels_ok:
                logger.info("Torsion stats step: torsion_stats.json complete; skipping (use --update to force)")
                return
            else:
                logger.info("Torsion stats step: existing file incomplete; recomputing")
        except Exception:
            # Fall through to recompute
            pass
    torsion_source = _torsions_source_path(embedding_dir)
    if torsion_source is None:
        logger.warning("No torsions.parquet or torsions.csv found; skipping cluster torsion stats")
        # Create an empty file to mark step as complete
        storage = get_storage_manager()
        with storage.open(stats_path, 'w') as f:
            json.dump({'levels': [], 'torsions': {}}, f)
        return
    
    # Load torsions table and extract column names
    if tors_df is None:
        tors_df = _read_table_robust(torsion_source)
    if 'frame' not in tors_df.columns:
        logger.warning("Torsions table missing 'frame' column; aborting stats step")
        return
    
    # Get torsion labels from CSV columns (excluding 'frame')
    torsion_labels = [c for c in tors_df.columns if c != 'frame']

    torsion_metadata = _load_or_build_torsion_metadata(frame_data_dir, embedding_dir, torsion_labels)
    torsion_summary = _level_summary_from_metadata(torsion_metadata)
    # Load clustering results
    if cr is None:
        try:
            cr = load_clustering_results_parquet(embedding_dir)
        except Exception as exc:
            logger.warning(f"Could not load clustering results: {exc}; skipping stats")
            return
    levels_summary = []
    angle_cols = [c for c in tors_df.columns if c != 'frame']
    torsion_matrix = tors_df.select(angle_cols).to_numpy().astype(float) if angle_cols else np.empty((tors_df.height, 0), dtype=float)
    _, frame_to_row = _build_frame_row_lookup(tors_df)

    # Try to load PCA dataframe for entropy calculations
    if pca_df is None:
        try:
            pca_path = os.path.join(embedding_dir, "pca_conformation_landscape.parquet")
            if _path_exists(pca_path):
                pca_df = _read_parquet_robust(pca_path)
            else:
                logger.debug(f"PCA parquet not found at {pca_path}; per-cluster entropies will be skipped")
        except Exception as exc:
            logger.debug(f"Failed to read PCA parquet for entropy calc: {exc}")

    # Choose PCA component columns consistently with global entropy calc (first 3 PCs)
    entropy_pc_cols: List[str] = []
    pca_entropy_matrix: Optional[np.ndarray] = None
    pca_frame_to_row: Dict[int, int] = {}
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
        if entropy_pc_cols and "frame" in pca_df.columns:
            try:
                pca_entropy_matrix = pca_df.select(entropy_pc_cols).to_numpy().astype(float)
                _, pca_frame_to_row = _build_frame_row_lookup(pca_df)
            except Exception as exc:
                logger.debug(f"Failed to prepare PCA matrix for entropy calc: {exc}")
                pca_entropy_matrix = None
                pca_frame_to_row = {}

    level_entropy_map = {}
    for level in _levels_to_process(cr):
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
                member_row_indices = [frame_to_row[m] for m in members if m in frame_to_row]
                if member_row_indices:
                    member_angles = torsion_matrix[np.asarray(member_row_indices, dtype=int)]
                    for col_idx, col in enumerate(angle_cols):
                        cluster_stats[col] = circular_stats(member_angles[:, col_idx].astype(float, copy=False))

            # Compute per-cluster conformational entropy (nats) using PCA rows for member frames
            cluster_entropy = None
            if pca_entropy_matrix is not None and members and entropy_pc_cols:
                try:
                    cluster_pca_indices = [pca_frame_to_row[m] for m in members if m in pca_frame_to_row]
                    if len(cluster_pca_indices) > 1:
                        cluster_pca_np = pca_entropy_matrix[np.asarray(cluster_pca_indices, dtype=int)]
                        cluster_pca = pl.DataFrame(
                            {col: cluster_pca_np[:, idx] for idx, col in enumerate(entropy_pc_cols)}
                        )
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
            if rep_idx is not None and rep_idx in frame_to_row:
                rep_values = torsion_matrix[frame_to_row[int(rep_idx)]]
                for col_idx, col in enumerate(angle_cols):
                    if col not in cluster_stats:
                        cluster_stats[col] = circular_stats(np.array([], dtype=float))
                    cluster_stats[col]['representative_angle_deg'] = float(rep_values[col_idx])

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

    # Assemble JSON for torsion stats
    torsion_stats_data = {
        'torsions': {
            'labels': torsion_labels,
            'count': len(torsion_labels),
            **torsion_summary,
        },
        'levels': levels_summary,
    }

    storage = get_storage_manager()
    with storage.open(stats_path, 'w') as f:
        json.dump(torsion_stats_data, f, indent=2)
    logger.info(f"Wrote cluster torsion stats to {stats_path}")

    # Update embedding/metadata.json with average entropy per level
    try:
        meta_path = os.path.join(embedding_dir, 'metadata.json')
        meta = {}
        if _path_exists(meta_path):
            try:
                storage = get_storage_manager()
                with storage.open(meta_path, 'r') as mf:
                    meta = json.load(mf)
            except Exception:
                meta = {}
        # add or replace mapping
        meta['level_average_entropy'] = {str(k): (None if v is None else float(v)) for k, v in level_entropy_map.items()}
        with storage.open(meta_path, 'w') as mf:
            json.dump(meta, mf, indent=2)
        logger.info(f"Updated metadata.json with level average entropies at {meta_path}")
    except Exception as exc:
        logger.warning(f"Failed to update metadata.json with entropy info: {exc}")


def step_create_info_json(embedding_dir: str, entropy: float, force: bool = False) -> None:
    """Create info.json by consolidating data from pca.json and torsion_stats.json."""
    info_path = os.path.join(embedding_dir, "info.json")
    storage = get_storage_manager()
    if _path_exists(info_path) and not force:
        try:
            with storage.open(info_path, 'r') as existing_f:
                existing = json.load(existing_f)
            existing_levels = {int(level.get('level')) for level in existing.get('levels', []) if isinstance(level, dict) and level.get('level') is not None}
            torsions_block = existing.get('torsions', {}) if isinstance(existing, dict) else {}
            if (
                int(existing.get('schema_version', 0)) >= 3
                and set(_required_cluster_levels()).issubset(existing_levels)
                and torsions_block.get('metadata')
            ):
                logger.info("Create info.json step: info.json exists and matches schema v3; skipping")
                return
            logger.info("Create info.json step: existing info.json is outdated; regenerating")
        except Exception:
            logger.info("Create info.json step: existing info.json unreadable; regenerating")

    # Load pca.json data
    pca_json_data = {}
    pca_json_path = os.path.join(embedding_dir, 'pca.json')
    if _path_exists(pca_json_path):
        try:
            with storage.open(pca_json_path, 'r') as pf:
                pca_json_data = json.load(pf)
            logger.info(f"Loaded pca.json data for info.json")
        except Exception as exc:
            logger.warning(f"Failed to load pca.json for info.json: {exc}")

    # Load torsion_stats.json data
    torsion_stats_data = {}
    torsion_stats_path = os.path.join(embedding_dir, 'torsion_stats.json')
    if _path_exists(torsion_stats_path):
        try:
            with storage.open(torsion_stats_path, 'r') as tsf:
                torsion_stats_data = json.load(tsf)
            logger.info(f"Loaded torsion_stats.json data for info.json")
        except Exception as exc:
            logger.warning(f"Failed to load torsion_stats.json: {exc}")

    # Assemble final info.json
    info = {
        'entropy_nats': float(entropy),
        'torsions': torsion_stats_data.get('torsions', {}),
        'levels': torsion_stats_data.get('levels', []),
        'pca': pca_json_data,
        'schema_version': 3,
    }

    with storage.open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    logger.info(f"Wrote consolidated data to {info_path}")

def _load_frames(coords, frame_indices: List[int]) -> Dict[int, np.ndarray]:
    """Frames of the coordinate store by index, each chunk decompressed once."""
    n_frames = int(coords.shape[0])
    wanted = sorted({int(index) for index in frame_indices if 0 <= int(index) < n_frames})
    if not wanted:
        return {}
    block = np.asarray(coords.oindex[wanted, :, :])
    return {index: block[position] for position, index in enumerate(wanted)}


def _aligned_frame_xyz_cached(
    coords,
    atoms_df: pl.DataFrame,
    reference_xyz: np.ndarray,
    frame_idx: int,
    cache: Dict[int, np.ndarray],
    residue_range: tuple[int, int] = (1, 5),
) -> np.ndarray:
    cached = cache.get(int(frame_idx))
    if cached is not None:
        return cached.copy()
    xyz = np.asarray(coords[int(frame_idx)], dtype=float).copy()
    if int(frame_idx) != 0:
        xyz, _ = glypdbio.align_structures_by_residues(
            reference_xyz,
            xyz,
            atoms_df,
            atoms_df,
            residue_range,
            True,
        )
        xyz = np.asarray(xyz, dtype=float)
    cache[int(frame_idx)] = xyz
    return xyz.copy()


def _flip_anomer_model(
    xyz: np.ndarray,
    atom_names: List[str],
    residue_ids: List[int],
    base_res_names: List[str],
    *,
    target_anomer: Optional[str] = None,
    bonds: Optional[Sequence[Tuple[int, int]]] = None,
) -> tuple[np.ndarray, List[str]]:
    """Flip reducing-end geometry and return residue names for the target anomer.

    ``target_anomer`` is preferred over toggling the source code because input
    PDBs are not always internally consistent with their sequence metadata.
    Names are normalized before the geometry work so a missing/degenerate OH
    geometry cannot leave the generated counterpart with the source anomer's
    residue code.

    An aldose (``C1`` with ``H1`` in residue 2) exchanges its anomeric
    hydrogen and the oxygen of the aglycone ``ROH``.  A 2-ketose (sialic
    acids, Kdn, Kdo, Fru, Psi, Tag, Sor: the ``ROH`` oxygen is bonded to
    ``C2``, which has no hydrogen) exchanges its two exocyclic groups at
    ``C2``: the ``ROH`` group and the ``C1`` branch (the carboxylate of a
    ulosonic acid, the hydroxymethyl of a ketohexose with whatever is linked
    to it, which ``bonds``, pairs of atom indices, tell).  Both are turned
    half a turn about the bisector of the two bonds, which puts each where
    the other was with its bond length kept and inverts the configuration at
    ``C2``; the ring and the rest of the glycan do not move.
    """
    xyz_out = np.asarray(xyz, dtype=float).copy()
    res_names = list(base_res_names)
    if target_anomer is None:
        res_names = glypdbio.flip_anomeric_residue_names(res_names, residue_ids)
    else:
        res_names = glypdbio.normalize_anomeric_residue_names(
            res_names,
            residue_ids,
            target_anomer,
        )
    try:
        idx_C1 = next((i for i, (name, res_id) in enumerate(zip(atom_names, residue_ids)) if name == "C1" and res_id == 2), None)
        idx_H1 = next((i for i, (name, res_id) in enumerate(zip(atom_names, residue_ids)) if name == "H1" and res_id == 2), None)
        idx_O1 = next((i for i, (name, res_id) in enumerate(zip(atom_names, residue_ids)) if name == "O1" and res_id == 1), None)
        idx_HO1 = next((i for i, (name, res_id) in enumerate(zip(atom_names, residue_ids)) if name == "HO1" and res_id == 1), None)
        if None not in (idx_C1, idx_H1, idx_O1, idx_HO1):
            C1 = xyz_out[idx_C1]
            H1 = xyz_out[idx_H1]
            O1 = xyz_out[idx_O1]
            HO1 = xyz_out[idx_HO1]

            CO1_vec = O1 - C1
            CH1_vec = H1 - C1
            HO1O1_vec = HO1 - O1

            len_C1O1 = float(np.linalg.norm(CO1_vec)) if np.linalg.norm(CO1_vec) > 0 else 1.43
            len_C1H1 = float(np.linalg.norm(CH1_vec)) if np.linalg.norm(CH1_vec) > 0 else 1.09
            if np.linalg.norm(CH1_vec) > 0 and np.linalg.norm(CO1_vec) > 0:
                new_O1 = C1 + (CH1_vec / np.linalg.norm(CH1_vec)) * len_C1O1
                new_H1 = C1 + (CO1_vec / np.linalg.norm(CO1_vec)) * len_C1H1
                new_HO1 = new_O1 + HO1O1_vec
                xyz_out[idx_O1] = new_O1
                xyz_out[idx_H1] = new_H1
                xyz_out[idx_HO1] = new_HO1
        elif idx_H1 is None and idx_O1 is not None:
            xyz_out = _flip_ketose_reducing_end(xyz_out, atom_names, residue_ids, idx_O1, bonds)

    except Exception:
        pass
    return xyz_out, res_names


#: Atoms of the C1 branch of a 2-ketose by name, when no bond table is given.
_KETOSE_C1_BRANCH_NAMES = frozenset({"C1", "O1", "O1A", "O1B", "H11", "H12", "H1O", "HO1"})


def _flip_ketose_reducing_end(
    xyz: np.ndarray,
    atom_names: List[str],
    residue_ids: List[int],
    idx_o_aglycone: int,
    bonds: Optional[Sequence[Tuple[int, int]]] = None,
) -> np.ndarray:
    """The other anomer of a 2-ketose at the reducing end (see :func:`_flip_anomer_model`).

    Returns ``xyz`` unchanged when the reducing residue is not a 2-ketose
    bonded to the aglycone at ``C2``, or when the geometry is degenerate.
    """
    xyz_out = np.asarray(xyz, dtype=float).copy()
    atoms_of = {(name, int(res_id)): index for index, (name, res_id) in enumerate(zip(atom_names, residue_ids))}
    idx_c2, idx_c1 = atoms_of.get(("C2", 2)), atoms_of.get(("C1", 2))
    if idx_c2 is None or idx_c1 is None:
        return xyz_out
    oxygen = xyz_out[idx_o_aglycone]
    # The aglycone oxygen has to be the one bonded to C2 (a 2-ketose), not to C1.
    if np.linalg.norm(oxygen - xyz_out[idx_c2]) >= np.linalg.norm(oxygen - xyz_out[idx_c1]):
        return xyz_out
    aglycone = [index for index, res_id in enumerate(residue_ids) if int(res_id) == 1]
    if bonds is not None and len(bonds):
        neighbours: Dict[int, List[int]] = {}
        for a, b in bonds:
            neighbours.setdefault(int(a), []).append(int(b))
            neighbours.setdefault(int(b), []).append(int(a))
        branch, frontier = {idx_c1}, [idx_c1]
        while frontier:
            atom = frontier.pop()
            for other in neighbours.get(atom, ()):
                if other == idx_c2 and atom == idx_c1:
                    continue
                if other == idx_c2 or other in aglycone:
                    return xyz_out                       # C1 is in a ring with C2: not a 2-ketose
                if other not in branch:
                    branch.add(other)
                    frontier.append(other)
    else:
        branch = {index for index, (name, res_id) in enumerate(zip(atom_names, residue_ids))
                  if int(res_id) == 2 and name in _KETOSE_C1_BRANCH_NAMES}
    centre = xyz_out[idx_c2]
    to_c1, to_oxygen = xyz_out[idx_c1] - centre, oxygen - centre
    if np.linalg.norm(to_c1) == 0 or np.linalg.norm(to_oxygen) == 0:
        return xyz_out
    axis = to_c1 / np.linalg.norm(to_c1) + to_oxygen / np.linalg.norm(to_oxygen)
    if np.linalg.norm(axis) < 1e-6:
        return xyz_out
    axis = axis / np.linalg.norm(axis)
    half_turn = 2.0 * np.outer(axis, axis) - np.eye(3)        # rotation by 180 degrees about the axis
    moved = sorted(branch | set(aglycone))
    xyz_out[moved] = centre + (xyz_out[moved] - centre) @ half_turn.T
    return xyz_out


def _write_representative_model(
    fh,
    model_index: int,
    cluster_id: int,
    cluster_size_pct: Optional[float],
    xyz: np.ndarray,
    serials: np.ndarray,
    atom_names: List[str],
    res_names: List[str],
    chain_u8: np.ndarray,
    residue_ids_np: np.ndarray,
    elements: List[str],
    conect_block: str,
) -> None:
    fh.write(f"MODEL     {model_index:4d}\n")
    fh.write(f"REMARK   CLUSTER_ID {cluster_id}\n")
    try:
        fh.write(f"REMARK   CLUSTER_POPULATION_PCT {float(cluster_size_pct):.4f}\n")
    except Exception:
        pass
    for atom_idx in range(len(serials)):
        x, y, z = xyz[atom_idx]
        fh.write(glypdbio._pdb_atom_line(
            serial=int(serials[atom_idx]),
            name=atom_names[atom_idx],
            res_name=res_names[atom_idx],
            chain_u8=int(chain_u8[atom_idx]),
            res_id=int(residue_ids_np[atom_idx]),
            x=float(x), y=float(y), z=float(z),
            element=elements[atom_idx],
        ))
    if conect_block:
        fh.write(conect_block)
    fh.write("ENDMDL\n")


def step_save_representative_structures(
    frame_data_dir: str, 
    embedding_dir: str, 
    out_dir: str, 
    glycam_name: str, 
    force: bool = False,
    *,
    cr=None,
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
    
    # Load clustering results
    if cr is None:
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

    try:
        atoms = glypdbio.load_atoms(frame_data_dir)
        bonds = glypdbio.load_bonds(frame_data_dir)
        coords = glypdbio.load_coords(frame_data_dir)
    except Exception as exc:
        logger.warning(f"Could not load trajectory data for representative structures: {exc}")
        return

    serials = atoms["serial"].to_numpy()
    atom_names = atoms["name"].to_list()
    source_res_names = atoms["res_name"].to_list()
    chain_u8 = atoms["chain_id_u8"].to_numpy()
    residue_ids_np = atoms["res_id"].to_numpy()
    residue_ids = [int(x) for x in residue_ids_np.tolist()]
    elements = atoms["element"].to_list()
    base_res_names = glypdbio.normalize_anomeric_residue_names(
        source_res_names,
        residue_ids,
        current_anomer,
    )
    opposite_anomer = "beta" if current_anomer == "alpha" else "alpha"
    levels = _levels_to_process(cr)
    # Read the reference frame and every representative frame in one pass:
    # with one Zarr chunk per trajectory, each single-frame read decompresses
    # all frames.
    coords = _load_frames(
        coords,
        [0] + [
            int(rep_idx)
            for level in levels
            for rep_idx in cr.representatives(level)["representative_idx"].to_list()
            if rep_idx is not None
        ],
    )
    reference_xyz = np.asarray(coords[0], dtype=float).copy()
    conect_block = glypdbio._pdb_conect_block(serials, bonds, atoms, reference_xyz)
    # what moves with the C1 branch when the other anomer of a 2-ketose is made
    bond_pairs = list(zip(bonds["a"].to_list(), bonds["b"].to_list())) if {"a", "b"} <= set(bonds.columns) else []
    aligned_cache: Dict[int, np.ndarray] = {0: reference_xyz.copy()}
    
    # Process each clustering level
    storage = get_storage_manager()
    total_structures = 0
    for level in levels:
        reps_df = cr.representatives(level)
        level_dir = os.path.join(out_dir, f"level_{level}")
        os.makedirs(level_dir, exist_ok=True)
        # Prepare aggregated alpha/beta files
        alpha_out = os.path.join(level_dir, "alpha.pdb")
        beta_out = os.path.join(level_dir, "beta.pdb")
        if os.path.exists(alpha_out) and os.path.exists(beta_out) and not force:
            logger.info(f"Representative structures already exist for level {level}; skipping")
            continue
        rows = list(reps_df.iter_rows(named=True))
        alpha_model_idx = 0
        beta_model_idx = 0
        with storage.open(alpha_out, "w") as alpha_fh, storage.open(beta_out, "w") as beta_fh:
            alpha_fh.write(f"REMARK   MULTI-MODEL REPRESENTATIVES LEVEL {level} ANOMER ALPHA\n")
            beta_fh.write(f"REMARK   MULTI-MODEL REPRESENTATIVES LEVEL {level} ANOMER BETA\n")

            for row in rows:
                cid = int(row['cluster_id'])
                rep_idx = row.get('representative_idx')
                cluster_size_pct = row.get('cluster_size_pct', 0.0)

                if rep_idx is None:
                    logger.warning(f"No representative index for cluster {cid} at level {level}")
                    continue

                rep_frame = int(rep_idx)
                try:
                    aligned_xyz = _aligned_frame_xyz_cached(
                        coords,
                        atoms,
                        reference_xyz,
                        rep_frame,
                        aligned_cache,
                    )
                    original_xyz = aligned_xyz.copy()
                    flipped_xyz, flipped_res_names = _flip_anomer_model(
                        aligned_xyz,
                        atom_names,
                        residue_ids,
                        base_res_names,
                        target_anomer=opposite_anomer,
                        bonds=bond_pairs,
                    )

                    if current_anomer == 'alpha':
                        alpha_model_idx += 1
                        _write_representative_model(
                            alpha_fh,
                            alpha_model_idx,
                            cid,
                            cluster_size_pct,
                            original_xyz,
                            serials,
                            atom_names,
                            base_res_names,
                            chain_u8,
                            residue_ids_np,
                            elements,
                            conect_block,
                        )
                        beta_model_idx += 1
                        _write_representative_model(
                            beta_fh,
                            beta_model_idx,
                            cid,
                            cluster_size_pct,
                            flipped_xyz,
                            serials,
                            atom_names,
                            flipped_res_names,
                            chain_u8,
                            residue_ids_np,
                            elements,
                            conect_block,
                        )
                    else:
                        beta_model_idx += 1
                        _write_representative_model(
                            beta_fh,
                            beta_model_idx,
                            cid,
                            cluster_size_pct,
                            original_xyz,
                            serials,
                            atom_names,
                            base_res_names,
                            chain_u8,
                            residue_ids_np,
                            elements,
                            conect_block,
                        )
                        alpha_model_idx += 1
                        _write_representative_model(
                            alpha_fh,
                            alpha_model_idx,
                            cid,
                            cluster_size_pct,
                            flipped_xyz,
                            serials,
                            atom_names,
                            flipped_res_names,
                            chain_u8,
                            residue_ids_np,
                            elements,
                            conect_block,
                        )
                    total_structures += 2
                except Exception as exc:
                    logger.warning(f"Failed to write representative structure for cluster {cid} at level {level}: {exc}")
                    continue

            alpha_fh.write("END\n")
            beta_fh.write("END\n")
    
    logger.info(f"Saved {total_structures} representative models across alpha/beta multi-model files in {out_dir}")


def _plot_level(torsion_table_path: str, dist_plot_path: str, representative_frames: List[int], level: int) -> Optional[str]:
    """Draw the distribution plot of one level; returns None, or the error as text."""
    try:
        plot_torsion_distribution(
            torsion_csv_path=torsion_table_path,
            output_path=dist_plot_path,
            cluster_representatives=representative_frames,
            level=level
        )
        return None
    except Exception as exc:
        return str(exc)


def step_plot_torsion_distributions(
    embedding_dir: str,
    out_dir: str,
    force: bool = False,
    *,
    cr=None,
) -> None:
    """Create torsion distribution plots for each clustering level.
    
    This step reads clustering results and torsion tables to generate distribution plots
    for each level, highlighting cluster representatives.
    
    Parameters:
        embedding_dir: Directory containing clustering results and torsions.parquet
        out_dir: Output directory where level folders exist
        force: Whether to overwrite existing distribution plots
    """
    # Check for required files
    torsion_table_path = _torsions_source_path(embedding_dir)
    if torsion_table_path is None:
        logger.warning("No torsions.parquet or torsions.csv found; skipping distribution plots")
        return
    
    # A monosaccharide has a frame-only torsion table: it has valid conformers
    # but no glycosidic torsion to plot. Emit a small explicit placeholder so
    # static-only validation does not mistake it for an incomplete process run.
    torsion_columns = []
    try:
        torsion_columns = [column for column in _read_torsions_table(embedding_dir).columns if column != "frame"]
    except Exception as exc:
        logger.warning(f"Could not inspect torsion columns: {exc}")

    # Load clustering results
    if cr is None:
        try:
            cr = load_clustering_results_parquet(embedding_dir)
        except Exception as exc:
            logger.warning(f"Could not load clustering results: {exc}; skipping distribution plots")
            return
    
    # Process each clustering level
    plots_created = 0
    plots_skipped = 0
    pending: List[Tuple[int, str, List[int]]] = []
    
    for level in _levels_to_process(cr):
        level_dir = os.path.join(out_dir, f"level_{level}")
        if not _path_exists(level_dir):
            logger.warning(f"Level directory {level_dir} does not exist; skipping distribution plot")
            continue
        
        dist_plot_path = os.path.join(level_dir, "dist.svg")
        
        # Check if plot already exists and force is not set
        if _path_exists(dist_plot_path) and not force:
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

        if not torsion_columns:
            with open(dist_plot_path, "w", encoding="utf-8") as handle:
                handle.write(
                    '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="120" '
                    'viewBox="0 0 720 120" role="img" aria-label="No glycosidic torsions">'
                    '<rect width="720" height="120" fill="white"/>'
                    '<text x="24" y="68" font-family="sans-serif" font-size="20" fill="#444">'
                    'No glycosidic torsions for this monosaccharide.'
                    '</text></svg>\n'
                )
            logger.info(f"Wrote no-torsion placeholder plot: {dist_plot_path}")
            plots_created += 1
            continue

        pending.append((level, dist_plot_path, representative_frames))

    # Create distribution plots. The figures of the levels do not depend on
    # each other (same distributions, other representative lines), so with
    # more than one worker each level is drawn in its own process.
    if len(pending) > 1 and int(config.max_workers) > 1:
        from joblib import Parallel, delayed

        errors = Parallel(n_jobs=min(len(pending), int(config.max_workers)))(
            delayed(_plot_level)(torsion_table_path, dist_plot_path, representative_frames, level)
            for level, dist_plot_path, representative_frames in pending
        )
    else:
        errors = [
            _plot_level(torsion_table_path, dist_plot_path, representative_frames, level)
            for level, dist_plot_path, representative_frames in pending
        ]
    for (level, dist_plot_path, _), error in zip(pending, errors):
        if error is None:
            logger.info(f"Saved torsion distribution plot: {dist_plot_path}")
            plots_created += 1
        else:
            logger.warning(f"Failed to create torsion distribution plot for level {level}: {error}")
    
    logger.info(f"Distribution plots: {plots_created} created, {plots_skipped} skipped")


def step_export_analysis_data(
    embedding_dir: str,
    out_dir: str,
    force: bool = False,
    *,
    frame_data_dir: Optional[str] = None,
    cr=None,
    pca_df: Optional[pl.DataFrame] = None,
) -> None:
    """Export Parquet analysis tables plus static-ready non-tabular files."""
    os.makedirs(out_dir, exist_ok=True)
    exports_dir = os.path.join(embedding_dir, "exports")
    os.makedirs(exports_dir, exist_ok=True)
    storage = get_storage_manager()

    pca_export_path = os.path.join(exports_dir, "pca.parquet")
    torsion_glycosidic_export_path = os.path.join(exports_dir, "torsion_glycosidic.parquet")
    torsion_all_export_path = os.path.join(exports_dir, "torsion_all.parquet")
    torparts_out_path = os.path.join(out_dir, "torparts.npz")
    info_out_path = os.path.join(out_dir, "info.json")

    files_created = 0
    files_skipped = 0

    def add_source_frame(table: pl.DataFrame) -> pl.DataFrame:
        if "frame" not in table.columns or "source_frame" in table.columns:
            return table
        resolved_data_dir = frame_data_dir or os.path.join(
            os.path.dirname(embedding_dir), "data"
        )
        mapping_path = os.path.join(resolved_data_dir, "frame_map.parquet")
        if not _path_exists(mapping_path):
            raise FileNotFoundError(
                f"Missing frame provenance required for export: {mapping_path}"
            )
        mapping = _read_parquet_robust(mapping_path).select(
            pl.col("frame").cast(pl.Int64),
            pl.col("source_frame").cast(pl.Int64),
        )
        return (
            table.with_columns(pl.col("frame").cast(pl.Int64))
            .join(mapping, on="frame", how="left")
        )

    pca_source_path = os.path.join(embedding_dir, "pca_conformation_landscape.parquet")
    torsion_source_path = _torsions_source_path(embedding_dir)
    info_source_path = os.path.join(embedding_dir, "info.json")
    torparts_source_path = os.path.join(embedding_dir, "torparts.npz")

    if cr is None:
        try:
            cr = load_clustering_results_parquet(embedding_dir)
        except Exception as exc:
            logger.warning(f"Could not load clustering results for export: {exc}")

    # 1. Export PCA data with first 3 components and level_1 cluster assignments.
    if _path_exists(pca_source_path):
        if _has_required_parquet_headers(pca_export_path, {"frame", "source_frame", "PC1", "PC2", "PC3", "cluster"}) and not force:
            logger.info("pca.parquet exists; skipping (use --update to force)")
            files_skipped += 1
        else:
            try:
                if pca_df is None:
                    pca_df = _read_parquet_robust(pca_source_path)
                pca_df = add_source_frame(pca_df)
                pca_cols = ["frame", "PC1", "PC2", "PC3"]
                available_cols = [col for col in pca_cols if col in pca_df.columns]
                if available_cols:
                    pca_export_df = pca_df.select(
                        [*available_cols, "source_frame"]
                    )
                    if cr is not None:
                        frame_to_cluster = {}
                        try:
                            for cluster_id in cr.clusters(1):
                                for frame in cr.members(1, int(cluster_id)):
                                    frame_to_cluster[int(frame)] = int(cluster_id)
                            pca_export_df = pca_export_df.with_columns(
                                pl.Series("cluster", [frame_to_cluster.get(int(f), -1) for f in pca_export_df["frame"].to_list()])
                            )
                        except Exception as exc:
                            logger.warning(f"Could not add level_1 cluster column to PCA export: {exc}")
                    _write_parquet(pca_export_df, pca_export_path)
                    logger.info(f"Exported PCA data with {len(available_cols)} components to {pca_export_path}")
                    files_created += 1
                else:
                    logger.warning("No PCA components found in parquet file")
            except Exception as exc:
                logger.warning(f"Failed to export PCA data: {exc}")
    else:
        logger.warning(f"PCA source file not found: {pca_source_path}")

    # 2. Export torsion tables.
    if torsion_source_path and _path_exists(torsion_source_path):
        try:
            info_data = _load_json_if_exists(info_source_path, {})
            torsions_block = info_data.get("torsions", {}) if isinstance(info_data, dict) else {}

            tors_df = _read_table_robust(torsion_source_path)
            if "frame" not in tors_df.columns:
                raise ValueError("torsion table is missing the frame column")
            tors_df = add_source_frame(tors_df)

            torsion_labels = [
                c for c in tors_df.columns if c not in ("frame", "source_frame")
            ]
            raw_metadata = torsions_block.get("metadata") or _load_json_if_exists(
                os.path.join(embedding_dir, "torsion_metadata.json"),
                [],
            )
            torsion_metadata = _align_torsion_metadata_to_labels(torsion_labels, raw_metadata)
            export_names = _unique_column_names(_canonical_torsion_export_names(torsion_metadata))

            levels_to_export = list(_required_cluster_levels())
            frame_to_cluster_by_level: Dict[int, Dict[int, int]] = {}
            representative_by_level: Dict[int, Dict[int, int]] = {}
            reps_rows_by_level: Dict[int, List[Dict]] = {}

            if cr is not None:
                for level in levels_to_export:
                    try:
                        reps_df = cr.representatives(level)
                    except Exception:
                        reps_df = None
                    if reps_df is None or reps_df.height == 0:
                        continue

                    level_map: Dict[int, int] = {}
                    try:
                        for cluster_id in cr.clusters(level):
                            for frame in cr.members(level, int(cluster_id)):
                                level_map[int(frame)] = int(cluster_id)
                    except Exception as exc:
                        logger.warning(f"Could not read members for level {level}: {exc}")
                    frame_to_cluster_by_level[level] = level_map

                    rep_map: Dict[int, int] = {}
                    rep_rows: List[Dict] = []
                    for row in reps_df.iter_rows(named=True):
                        rep_rows.append(row)
                        rep_idx = row.get("representative_idx")
                        if rep_idx is not None:
                            rep_map[int(row["cluster_id"])] = int(rep_idx)
                    representative_by_level[level] = rep_map
                    reps_rows_by_level[level] = rep_rows

            rename_map = {old: new for old, new in zip(torsion_labels, export_names) if old != new}
            all_df = tors_df.rename(rename_map) if rename_map else tors_df.clone()
            frames = [int(f) for f in all_df["frame"].to_list()]
            for level in levels_to_export:
                cluster_values = [frame_to_cluster_by_level.get(level, {}).get(frame, -1) for frame in frames]
                center_values = []
                for frame, cluster_id in zip(frames, cluster_values):
                    is_center = representative_by_level.get(level, {}).get(cluster_id) == frame
                    center_values.append(cluster_id if is_center and cluster_id != -1 else None)
                all_df = all_df.with_columns(
                    pl.Series(f"level_{level}_cluster", cluster_values, dtype=pl.Int32),
                    pl.Series(f"level_{level}_center", center_values, dtype=pl.Int32),
                )

            if force or not _has_required_parquet_headers(torsion_all_export_path, {"frame", "source_frame", *_level_assignment_columns()}):
                _write_parquet(all_df, torsion_all_export_path)
                logger.info(f"Exported all torsions with cluster assignments to {torsion_all_export_path}")
                files_created += 1
            else:
                logger.info("torsion_all.parquet exists; skipping (use --update to force)")
                files_skipped += 1

            glycosidic_indices: List[int] = [
                int(entry["index"]) for entry in torsion_metadata if entry.get("family") == "glycosidic"
            ]
            glycosidic_names: List[str] = [export_names[idx] for idx in glycosidic_indices if idx < len(export_names)]
            gly_cols = ["frame", "source_frame", *glycosidic_names]
            gly_df = all_df.select([col for col in gly_cols if col in all_df.columns])
            level_1_clusters = [frame_to_cluster_by_level.get(1, {}).get(frame, -1) for frame in frames]
            level_1_centers = []
            for frame, cluster_id in zip(frames, level_1_clusters):
                is_center = representative_by_level.get(1, {}).get(cluster_id) == frame
                level_1_centers.append(cluster_id if is_center and cluster_id != -1 else None)
            gly_df = gly_df.with_columns(
                pl.Series("cluster", level_1_clusters, dtype=pl.Int32),
                pl.Series("center", level_1_centers, dtype=pl.Int32),
            )
            if force or not _has_required_parquet_headers(torsion_glycosidic_export_path, {"frame", "source_frame", "cluster", "center"}):
                _write_parquet(gly_df, torsion_glycosidic_export_path)
                logger.info(f"Exported glycosidic torsions to {torsion_glycosidic_export_path}")
                files_created += 1
            else:
                logger.info("torsion_glycosidic.parquet exists; skipping (use --update to force)")
                files_skipped += 1
            if not glycosidic_indices:
                logger.warning("No glycosidic torsions identified for torsion_glycosidic.parquet")

            # Only the rows of representative frames are looked up below.
            representative_frames = sorted({
                int(rep["representative_idx"])
                for rep_rows in reps_rows_by_level.values()
                for rep in rep_rows
                if rep.get("representative_idx") is not None
            })
            frame_rows = {
                int(row["frame"]): row
                for row in all_df.select(
                    ["frame", "source_frame", *export_names]
                ).filter(pl.col("frame").is_in(representative_frames)).iter_rows(named=True)
            }
            for level in levels_to_export:
                reps_out_path = os.path.join(exports_dir, f"torsion_level_{level}_reps.parquet")
                rep_rows = reps_rows_by_level.get(level, [])
                if not rep_rows:
                    continue
                if _has_required_parquet_headers(reps_out_path, {"frame", "source_frame", "cluster_id", "cluster_size_pct"}) and not force:
                    logger.info(f"torsion_level_{level}_reps.parquet exists; skipping (use --update to force)")
                    files_skipped += 1
                    continue

                rows = []
                for rep in rep_rows:
                    rep_idx = rep.get("representative_idx")
                    if rep_idx is None:
                        continue
                    frame_num = int(rep_idx)
                    torsion_values = frame_rows.get(frame_num, {})
                    row = {
                        "frame": frame_num,
                        "source_frame": int(
                            torsion_values.get("source_frame", frame_num)
                        ),
                        "cluster_id": int(rep["cluster_id"]),
                        "cluster_size_pct": float(rep.get("cluster_size_pct") or 0.0),
                    }
                    for name in export_names:
                        row[name] = torsion_values.get(name)
                    rows.append(row)
                if rows:
                    _write_parquet(pl.DataFrame(rows), reps_out_path)
                    logger.info(f"Exported representative torsions for level {level} to {reps_out_path}")
                    files_created += 1
        except Exception as exc:
            logger.warning(f"Failed to export torsion tables: {exc}")
    else:
        logger.warning("Torsions source file not found")

    # 3. Copy torparts.npz
    if _path_exists(torparts_source_path):
        if _path_exists(torparts_out_path) and not force:
            logger.info("torparts.npz exists; skipping (use --update to force)")
            files_skipped += 1
        else:
            try:
                storage.write_binary(torparts_out_path, storage.read_binary(torparts_source_path))
                logger.info(f"Copied torparts.npz to {torparts_out_path}")
                files_created += 1
            except Exception as exc:
                logger.warning(f"Failed to copy torparts.npz: {exc}")
    else:
        logger.warning(f"torparts.npz source file not found: {torparts_source_path}")

    # 4. Copy info.json
    if _path_exists(info_source_path):
        if _has_current_output_info_json(info_out_path) and not force:
            logger.info("info.json exists; skipping (use --update to force)")
            files_skipped += 1
        else:
            try:
                storage.write_binary(info_out_path, storage.read_binary(info_source_path))
                logger.info(f"Copied info.json to {info_out_path}")
                files_created += 1
            except Exception as exc:
                logger.warning(f"Failed to copy info.json: {exc}")
    else:
        logger.warning(f"info.json source file not found: {info_source_path}")

    logger.info(f"Data export: {files_created} files created, {files_skipped} files skipped")


def _read_csv_header(path: str) -> List[str]:
    import csv

    if not _path_exists(path):
        return []
    storage = get_storage_manager()
    try:
        with storage.open(path, "r") as fh:
            reader = csv.reader(fh)
            return [str(value) for value in next(reader, [])]
    except Exception:
        return []


def _has_required_csv_headers(path: str, required_headers: set[str]) -> bool:
    header = set(_read_csv_header(path))
    return bool(header) and required_headers.issubset(header)


def _read_parquet_header(path: str) -> List[str]:
    if not _path_exists(path):
        return []
    try:
        if os.path.exists(path):
            # Column names from the footer; the table itself is not needed.
            import pyarrow.parquet as pq  # type: ignore
            return [str(value) for value in pq.read_schema(path).names]
        return [str(value) for value in _read_parquet_robust(path).columns]
    except Exception:
        return []


def _has_required_parquet_headers(path: str, required_headers: set[str]) -> bool:
    header = set(_read_parquet_header(path))
    return bool(header) and required_headers.issubset(header)


def _has_complete_clustering_outputs(embedding_dir: str) -> bool:
    storage = get_storage_manager()
    metadata_path = os.path.join(embedding_dir, "metadata.json")
    summary_path = os.path.join(embedding_dir, "cluster_summary.parquet")
    members_path = os.path.join(embedding_dir, "cluster_members.parquet")

    if not storage.exists(metadata_path) or not storage.exists(summary_path) or not storage.exists(members_path):
        return False

    metadata = _load_json_if_exists(metadata_path, {})
    levels = set()
    for level in metadata.get("levels", []) if isinstance(metadata, dict) else []:
        try:
            levels.add(int(level))
        except Exception:
            continue

    return set(_required_cluster_levels()).issubset(levels)


def _has_complete_torsion_stats(embedding_dir: str) -> bool:
    stats_path = os.path.join(embedding_dir, "torsion_stats.json")
    stats = _load_json_if_exists(stats_path, {})
    torsions_block = stats.get("torsions", {}) if isinstance(stats, dict) else {}
    level_keys = {
        int(level.get("level"))
        for level in stats.get("levels", [])
        if isinstance(level, dict) and level.get("level") is not None
    } if isinstance(stats, dict) else set()

    return bool(
        torsions_block.get("labels")
        and torsions_block.get("metadata")
        and torsions_block.get("family_summary")
        and set(_required_cluster_levels()).issubset(level_keys)
    )


def _has_current_info_json(embedding_dir: str) -> bool:
    info_path = os.path.join(embedding_dir, "info.json")
    return _has_current_output_info_json(info_path)


def _has_current_output_info_json(info_path: str) -> bool:
    info = _load_json_if_exists(info_path, {})
    levels = {
        int(level.get("level"))
        for level in info.get("levels", [])
        if isinstance(level, dict) and level.get("level") is not None
    } if isinstance(info, dict) else set()
    torsions_block = info.get("torsions", {}) if isinstance(info, dict) else {}

    return bool(
        int(info.get("schema_version", 0)) >= 3
        and set(_required_cluster_levels()).issubset(levels)
        and torsions_block.get("metadata")
    )


def _has_complete_structure_outputs(out_dir: str) -> bool:
    for level in _required_cluster_levels():
        for file_name in ("alpha.pdb", "beta.pdb"):
            if not _path_exists(os.path.join(out_dir, f"level_{level}", file_name)):
                return False
    return True


def _has_complete_plot_outputs(out_dir: str) -> bool:
    for level in _required_cluster_levels():
        if not _path_exists(os.path.join(out_dir, f"level_{level}", "dist.svg")):
            return False
    return True


def _has_complete_export_outputs(embedding_dir: str, out_dir: str) -> bool:
    exports_dir = os.path.join(embedding_dir, "exports")

    pca_out_path = os.path.join(exports_dir, "pca.parquet")
    if not _has_required_parquet_headers(pca_out_path, {"frame", "PC1", "PC2", "PC3", "cluster"}):
        return False

    info_out_path = os.path.join(out_dir, "info.json")
    if not _has_current_output_info_json(info_out_path):
        return False

    torsion_source_path = _torsions_source_path(embedding_dir)
    if torsion_source_path is None:
        return True

    torsion_all_out_path = os.path.join(exports_dir, "torsion_all.parquet")
    torsion_glycosidic_out_path = os.path.join(exports_dir, "torsion_glycosidic.parquet")
    torparts_out_path = os.path.join(out_dir, "torparts.npz")

    required_paths = [
        torsion_all_out_path,
        torsion_glycosidic_out_path,
        torparts_out_path,
    ]
    required_paths.extend(os.path.join(exports_dir, f"torsion_level_{level}_reps.parquet") for level in _required_cluster_levels())

    if not all(_path_exists(path) for path in required_paths):
        return False

    if not _has_required_parquet_headers(
        torsion_all_out_path,
        {"frame", *_level_assignment_columns()},
    ):
        return False

    if not _has_required_parquet_headers(torsion_glycosidic_out_path, {"frame", "cluster", "center"}):
        return False

    return True


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
        
        # The source directory is an opaque storage key and is not necessarily a
        # GLYCAM sequence (for example, legacy GlyTouCan-style identifiers).
        # Always prefer its explicit name.json metadata, even if the directory
        # itself happens to end in "-OH".
        json_path = os.path.join(input_glycan_dir, "name.json")
        # An archive entry (lib/archive.py) carries its GLYCAM name in the
        # manifest and its frames in trajectory.xtc; the store step reads both.
        archive_inputs = archive.entry_inputs(input_glycan_dir)
        try:
            storage = get_storage_manager()
            with storage.open(json_path, 'r') as f:
                data = json.load(f)
            glycam_name = str(data["indexOrderedSequence"]).strip()
            if not glycam_name:
                raise ValueError("indexOrderedSequence is empty")
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
            if archive_inputs is None:
                self.logger.warning(f"Could not read a GLYCAM name from {json_path}, using directory name")
            glycam_name = glycan_name
        
        # Resolve input file names, supporting alternate names used in Oracle bucket
        storage = get_storage_manager()
        default_pdb = os.path.join(input_glycan_dir, f"{glycan_name}.pdb")
        default_mol2 = os.path.join(input_glycan_dir, f"{glycan_name}.mol2")
        alt_pdb = os.path.join(input_glycan_dir, "simulation.pdb")
        alt_mol2 = os.path.join(input_glycan_dir, "structure.mol2")

        pdb_path = default_pdb if storage.exists(default_pdb) else (alt_pdb if storage.exists(alt_pdb) else default_pdb)
        mol2_path = default_mol2 if storage.exists(default_mol2) else (alt_mol2 if storage.exists(alt_mol2) else default_mol2)

        if archive_inputs is not None:
            glycam_name = archive_inputs["glycam_name"]
            pdb_path = archive_inputs["trajectory_path"]
            mol2_path = archive_inputs["mol2_path"]

        return {
            'input_glycan_dir': input_glycan_dir,  # For reading input files
            'process_glycan_dir': process_glycan_dir,  # For storing processing outputs
            'glycam_name': glycam_name,
            'pdb_path': pdb_path,
            'mol2_path': mol2_path,
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
        storage = get_storage_manager()
        # Already resolved preferred paths in get_glycan_paths
        if not storage.exists(paths['pdb_path']):
            self.logger.error(f"Required file not found: {paths['pdb_path']}")
            return False
        if not storage.exists(paths['mol2_path']):
            self.logger.error(f"Required file not found: {paths['mol2_path']}")
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
            'paths': {},
            'step_durations_sec': {},
        }
        
        try:
            # Get paths
            paths = self.get_glycan_paths(glycan_name)
            results['paths'] = paths
            clustering_context: Dict[str, object] = {}
            
            # Validate inputs
            if not self.validate_inputs(paths):
                results['error_message'] = "Input validation failed"
                return results
            
            self.logger.info(f"Starting v2 analysis pipeline for: {glycan_name}")
            
            # Step 1: Store trajectory data
            try:
                self.logger.info("Step 1: Processing trajectory data")
                step_started_at = time.perf_counter()
                store_result = step_store(
                    glycan_name, 
                    paths['frame_data_dir'], 
                    paths['pdb_path'], 
                    paths['mol2_path'], 
                    force=force_update
                )
                results['step_durations_sec']['store'] = round(time.perf_counter() - step_started_at, 3)
                self.logger.info(f"Step 1 completed in {results['step_durations_sec']['store']:.3f}s")
                results['steps_completed'].append('store')
            except Exception as e:
                self.logger.error(f"Store step failed: {str(e)}")
                results['steps_failed'].append('store')
                results['error_message'] = f"Store step failed: {str(e)}"
                return results

            # A newly stored or newly filtered coordinate set invalidates every
            # downstream frame-indexed artifact, even in incremental mode.
            downstream_force = bool(force_update or store_result.changed)

            # Step 2: PCA analysis
            try:
                self.logger.info("Step 2: PCA analysis")
                step_started_at = time.perf_counter()
                pca_path = step_pca(
                    paths['frame_data_dir'], 
                    paths['embedding_dir'], 
                    force=downstream_force,
                    per_pair_scale=per_pair_scale
                )
                results['step_durations_sec']['pca'] = round(time.perf_counter() - step_started_at, 3)
                self.logger.info(f"Step 2 completed in {results['step_durations_sec']['pca']:.3f}s")
                results['steps_completed'].append('pca')
            except Exception as e:
                self.logger.error(f"PCA step failed: {str(e)}")
                results['steps_failed'].append('pca')
                results['error_message'] = f"PCA step failed: {str(e)}"
                return results

            # Step 3: Torsion analysis (moved before clustering since it's independent)
            try:
                self.logger.info("Step 3: Torsion analysis")
                step_started_at = time.perf_counter()
                # A torsion table from earlier torsion definitions is computed
                # again, and with it what is derived from it (steps 5, 6, 8, 9).
                torsions_stale = not downstream_force and not torsion_table_is_current(
                    paths['frame_data_dir'], paths['embedding_dir']
                )
                torsion_list, torsion_labels = step_torsions(
                    paths['frame_data_dir'], 
                    paths['embedding_dir'], 
                    force=downstream_force
                )
                results['step_durations_sec']['torsions'] = round(time.perf_counter() - step_started_at, 3)
                self.logger.info(f"Step 3 completed in {results['step_durations_sec']['torsions']:.3f}s")
                results['steps_completed'].append('torsions')
            except Exception as e:
                self.logger.error(f"Torsion step failed: {str(e)}")
                results['steps_failed'].append('torsions')
                results['error_message'] = f"Torsion step failed: {str(e)}"
                return results

            # Step 4: Clustering analysis
            try:
                self.logger.info("Step 4: Clustering analysis")
                step_started_at = time.perf_counter()
                entropy, clustering_context = step_clustering_gmm(
                    paths['embedding_dir'], 
                    pca_path, 
                    force=downstream_force,
                    return_context=True,
                )
                results['step_durations_sec']['clustering'] = round(time.perf_counter() - step_started_at, 3)
                self.logger.info(f"Step 4 completed in {results['step_durations_sec']['clustering']:.3f}s")
                results['steps_completed'].append('clustering')
            except Exception as e:
                self.logger.error(f"Clustering step failed: {str(e)}")
                results['steps_failed'].append('clustering')
                results['error_message'] = f"Clustering step failed: {str(e)}"
                return results

            # A clustering computed in this run (first run, forced, or a level
            # added to existing outputs) invalidates what is indexed by its
            # clusters: statistics, info.json, structures, plots, exports.
            cluster_force = bool(downstream_force or clustering_context.get('computed'))
            torsion_force = bool(cluster_force or torsions_stale)

            # Step 5: Cluster torsion statistics
            try:
                self.logger.info("Step 5: Torsion statistics")
                step_started_at = time.perf_counter()
                step_cluster_torsion_stats(
                    paths['frame_data_dir'], 
                    paths['embedding_dir'], 
                    force=torsion_force,
                    cr=clustering_context.get('cr'),
                    pca_df=clustering_context.get('pca_df'),
                )
                results['step_durations_sec']['torsion_stats'] = round(time.perf_counter() - step_started_at, 3)
                self.logger.info(f"Step 5 completed in {results['step_durations_sec']['torsion_stats']:.3f}s")
                results['steps_completed'].append('torsion_stats')
            except Exception as e:
                self.logger.error(f"Torsion stats step failed: {str(e)}")
                results['steps_failed'].append('torsion_stats')
                results['error_message'] = f"Torsion stats step failed: {str(e)}"
                return results

            # Step 6: Create info.json
            try:
                self.logger.info("Step 6: Creating info.json")
                step_started_at = time.perf_counter()
                step_create_info_json(
                    paths['embedding_dir'], 
                    entropy, 
                    force=torsion_force
                )
                results['step_durations_sec']['create_info'] = round(time.perf_counter() - step_started_at, 3)
                self.logger.info(f"Step 6 completed in {results['step_durations_sec']['create_info']:.3f}s")
                results['steps_completed'].append('create_info')
            except Exception as e:
                self.logger.error(f"Info.json creation step failed: {str(e)}")
                results['steps_failed'].append('create_info')
                results['error_message'] = f"Info.json creation step failed: {str(e)}"
                return results

            # Step 7: Save representative structures
            try:
                self.logger.info("Step 7: Saving representative structures")
                step_started_at = time.perf_counter()
                step_save_representative_structures(
                    paths['frame_data_dir'], 
                    paths['embedding_dir'], 
                    paths['out_dir'], 
                    paths['glycam_name'], 
                    force=cluster_force,
                    cr=clustering_context.get('cr'),
                )
                results['step_durations_sec']['structures'] = round(time.perf_counter() - step_started_at, 3)
                self.logger.info(f"Step 7 completed in {results['step_durations_sec']['structures']:.3f}s")
                results['steps_completed'].append('structures')
            except Exception as e:
                self.logger.error(f"Structure saving step failed: {str(e)}")
                results['steps_failed'].append('structures')
                results['error_message'] = f"Structure saving step failed: {str(e)}"
                return results

            # Step 8: Create torsion distribution plots
            try:
                self.logger.info("Step 8: Creating distribution plots")
                step_started_at = time.perf_counter()
                step_plot_torsion_distributions(
                    paths['embedding_dir'], 
                    paths['out_dir'], 
                    force=torsion_force,
                    cr=clustering_context.get('cr'),
                )
                results['step_durations_sec']['plots'] = round(time.perf_counter() - step_started_at, 3)
                self.logger.info(f"Step 8 completed in {results['step_durations_sec']['plots']:.3f}s")
                results['steps_completed'].append('plots')
            except Exception as e:
                self.logger.error(f"Plotting step failed: {str(e)}")
                results['steps_failed'].append('plots')
                results['error_message'] = f"Plotting step failed: {str(e)}"
                return results

            # Step 9: Export analysis data
            try:
                self.logger.info("Step 9: Exporting analysis data")
                step_started_at = time.perf_counter()
                step_export_analysis_data(
                    paths['embedding_dir'], 
                    paths['out_dir'], 
                    force=torsion_force,
                    frame_data_dir=paths['frame_data_dir'],
                    cr=clustering_context.get('cr'),
                    pca_df=clustering_context.get('pca_df'),
                )
                results['step_durations_sec']['export'] = round(time.perf_counter() - step_started_at, 3)
                self.logger.info(f"Step 9 completed in {results['step_durations_sec']['export']:.3f}s")
                results['steps_completed'].append('export')
            except Exception as e:
                self.logger.error(f"Export step failed: {str(e)}")
                results['steps_failed'].append('export')
                results['error_message'] = f"Export step failed: {str(e)}"
                return results

            # All steps completed successfully
            results['success'] = True
            self.logger.info(f"Successfully completed v2 analysis for {glycan_name}")
            if results['step_durations_sec']:
                self.logger.info(f"Step timing summary (s): {results['step_durations_sec']}")
            
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
        storage = get_storage_manager()
        steps_to_check = {
            'store': os.path.join(paths['frame_data_dir'], 'atoms.parquet'),
        }
        for step, check_path in steps_to_check.items():
            status['steps_status'][step] = storage.exists(check_path)
        status['steps_status']['store'] = (
            status['steps_status']['store']
            and quality_manifest_is_current(
                paths['frame_data_dir'],
                config.max_covalent_bond_angstrom,
            )
        )
        status['steps_status']['pca'] = (
            storage.exists(os.path.join(paths['embedding_dir'], 'pca_conformation_landscape.parquet'))
            and storage.exists(os.path.join(paths['embedding_dir'], 'landscape', 'pca_model.joblib'))
            and storage.exists(os.path.join(paths['embedding_dir'], 'landscape', 'landscape_metadata.json'))
            and storage.exists(os.path.join(paths['embedding_dir'], 'landscape', 'preprocessing.npz'))
        )

        status['steps_status']['torsions'] = bool(
            _torsions_source_path(paths['embedding_dir'])
            and storage.exists(os.path.join(paths['embedding_dir'], 'torparts.npz'))
            # computed with the current torsion definitions
            and torsion_table_is_current(paths['frame_data_dir'], paths['embedding_dir'])
        )
        status['steps_status']['clustering'] = _has_complete_clustering_outputs(paths['embedding_dir'])
        status['steps_status']['cluster_stats'] = _has_complete_torsion_stats(paths['embedding_dir'])
        status['steps_status']['create_info'] = _has_current_info_json(paths['embedding_dir'])
        status['steps_status']['structures'] = _has_complete_structure_outputs(paths['out_dir'])
        status['steps_status']['plots'] = _has_complete_plot_outputs(paths['out_dir'])
        status['steps_status']['export'] = _has_complete_export_outputs(paths['embedding_dir'], paths['out_dir'])
            
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
