# glypdbio.py
from __future__ import annotations
import os
import io
import re
from typing import Optional
from lib.storage import get_storage_manager


import numpy as np
import polars as pl
import zarr

from pathlib import Path
from typing import List, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Loaders
# -------------------------
def load_atoms(out_dir: str) -> pl.DataFrame:
    path = os.path.join(out_dir, "atoms.parquet")
    # Some parquet readers (and Polars internals) may treat the path as a
    # glob pattern which breaks when the path contains characters like
    # '[' and ']'. To be robust, try reading via pyarrow first (no globbing),
    # and fall back to Polars. If pyarrow is missing, escape glob meta-chars
    # before calling Polars.
    if os.path.exists(path):
        try:
            # Prefer pyarrow if available (atomic file read, avoids glob interpretation)
            import pyarrow.parquet as pq  # type: ignore
            table = pq.read_table(path)
            return pl.from_arrow(table)
        except Exception:
            # Fallback: escape glob characters and use polars reader
            try:
                import glob
                escaped = glob.escape(path)
                return pl.read_parquet(escaped)
            except Exception:
                # Last resort: call pl.read_parquet on the raw path
                return pl.read_parquet(path)
    # Remote read fallback (e.g., Oracle PAR)
    try:
        storage = get_storage_manager()
        import pyarrow.parquet as pq  # type: ignore
        data = storage.read_binary(path)
        table = pq.read_table(io.BytesIO(data))
        return pl.from_arrow(table)
    except Exception as e:
        raise FileNotFoundError(f"atoms.parquet not found locally or remotely at {path}: {e}")

def load_bonds(out_dir: str) -> pl.DataFrame:
    path = os.path.join(out_dir, "bonds.parquet")
    if os.path.exists(path):
        # Use pyarrow first to avoid any glob interpretation by underlying readers
        try:
            import pyarrow.parquet as pq  # type: ignore
            table = pq.read_table(path)
            return pl.from_arrow(table)
        except Exception:
            try:
                import glob
                escaped = glob.escape(path)
                return pl.read_parquet(escaped)
            except Exception:
                return pl.read_parquet(path)
    # Remote read fallback
    try:
        storage = get_storage_manager()
        import pyarrow.parquet as pq  # type: ignore
        data = storage.read_binary(path)
        table = pq.read_table(io.BytesIO(data))
        return pl.from_arrow(table)
    except Exception:
        return pl.DataFrame({"a": [], "b": []})

def load_coords(out_dir: str):
    """Open coords.zarr array; if missing locally in Oracle mode, fetch it first.

    In Oracle/PAR mode, materializes the remote zarr prefix to a local mirror under
    out_dir/coords.zarr when not present locally.
    """
    zarr_path = os.path.join(out_dir, "coords.zarr")
    if not os.path.exists(zarr_path):
        try:
            storage = get_storage_manager()
            if getattr(storage, "_is_oracle", False):
                remote_prefix = f"{out_dir}/coords.zarr".replace("\\", "/").rstrip('/')
                objs = storage.backend.list_files(remote_prefix)
                if objs:
                    for obj in objs:
                        key = obj.as_posix() if hasattr(obj, 'as_posix') else str(obj)
                        if not key.startswith(remote_prefix + '/'):
                            continue
                        rel = key[len(remote_prefix)+1:]
                        local_file = Path(zarr_path) / rel
                        local_file.parent.mkdir(parents=True, exist_ok=True)
                        data = storage.backend.read_binary(key)
                        with open(local_file, 'wb') as fh:
                            fh.write(data)
        except Exception:
            pass
    root = zarr.open(zarr_path, mode="r")
    return root["xyz"]  # zarr array: shape (n_frames, n_atoms, 3), float32

def load_frame_xyz(out_dir: str, frame_idx: int) -> np.ndarray:
    z = load_coords(out_dir)
    if frame_idx < 0 or frame_idx >= z.shape[0]:
        raise IndexError(f"frame_idx {frame_idx} out of range [0, {z.shape[0]-1}]")
    return z[frame_idx, :, :]  # (n_atoms, 3)


def normalize_anomeric_residue_name(residue_name: str, target_anomer: str) -> str:
    """Return a GLYCAM residue code with the requested alpha/beta suffix.

    GLYCAM reducing-residue codes normally encode the anomer in their final
    letter (``4YA``/``4YB`` alpha/beta pyranose, ``1CD``/``1CU`` alpha/beta
    furanose; ``lib.residue_templates``).  A few source structures contain
    the opposite code even though the sequence/name says otherwise.  Only an
    explicit final ``A``/``B`` (or, for a three-character code, ``D``/``U``)
    is changed; the ulosonic acids whose two-letter code ends in another
    letter (``GL``, ``KN``, ``KO``) and modified codes such as ``4YS`` are
    preserved because their anomeric counterpart cannot be inferred safely
    from the code alone.
    """
    name = str(residue_name)
    target = str(target_anomer).strip().lower()
    if target not in {"alpha", "beta"} or not name:
        return name
    if name[-1] in {"A", "B"}:
        return name[:-1] + ("A" if target == "alpha" else "B")
    if len(name) == 3 and name[-1] in {"D", "U"} and name[1:] not in {"GL", "KN", "KO"}:
        return name[:-1] + ("D" if target == "alpha" else "U")
    return name


def normalize_anomeric_residue_names(
    residue_names: List[str],
    residue_ids: List[int],
    target_anomer: str,
    reducing_residue_id: int = 2,
) -> List[str]:
    """Normalize all atoms in the reducing residue to ``target_anomer``.

    The same residue name must be used for every atom in a PDB residue.  This
    helper is intentionally conservative: it changes only reducing-residue
    names whose GLYCAM code has an unambiguous A/B suffix.
    """
    names = list(residue_names)
    root_indices = [
        index for index, residue_id in enumerate(residue_ids)
        if int(residue_id) == int(reducing_residue_id)
    ]
    if not root_indices:
        return names

    root_name = names[root_indices[0]]
    normalized = normalize_anomeric_residue_name(root_name, target_anomer)
    if normalized == root_name:
        return names
    for index in root_indices:
        names[index] = normalized
    return names


def flip_anomeric_residue_names(
    residue_names: List[str],
    residue_ids: List[int],
    reducing_residue_id: int = 2,
) -> List[str]:
    """Toggle an unambiguous A/B reducing-residue code.

    This is kept as the backwards-compatible operation for callers that only
    request a geometric flip.  Callers that know the target output should use
    :func:`normalize_anomeric_residue_names` instead.
    """
    names = list(residue_names)
    root_indices = [
        index for index, residue_id in enumerate(residue_ids)
        if int(residue_id) == int(reducing_residue_id)
    ]
    if not root_indices:
        return names
    root_name = str(names[root_indices[0]])
    if root_name and normalize_anomeric_residue_name(root_name, "alpha") != normalize_anomeric_residue_name(root_name, "beta"):
        target = "beta" if root_name[-1] in {"A", "D"} else "alpha"
        return normalize_anomeric_residue_names(
            names,
            residue_ids,
            target,
            reducing_residue_id=reducing_residue_id,
        )
    return names


# -------------------------
# Distance utilities
# -------------------------
def _lower_triangle_flat_distances(xyz: np.ndarray) -> np.ndarray:
    """Compute pairwise distances for coordinates and return the flattened lower-triangle (i>j).

    Returns an array of shape (n*(n-1)/2,) with distances for all unique unordered atom pairs.
    
    Optimized version using efficient memory access patterns and float32 arithmetic.
    """
    if xyz is None:
        return np.array([], dtype=np.float32)
    xyz = np.asarray(xyz, dtype=np.float32)  # Use float32 for better memory efficiency
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must be an (n_atoms, 3) array")
    n = xyz.shape[0]
    if n < 2:
        return np.array([], dtype=np.float32)
    
    # For small numbers of atoms, use the full distance matrix approach
    if n <= 1000:  # Threshold for memory vs computation tradeoff
        # pairwise distances: using broadcasting
        diffs = xyz[:, None, :] - xyz[None, :, :]
        dists = np.sqrt((diffs * diffs).sum(axis=-1))
        i, j = np.tril_indices(n, k=-1)
        return dists[i, j].astype(np.float32)
    else:
        # For large numbers of atoms, compute distances directly for lower triangle
        # to save memory (avoids creating full n x n matrix)
        n_pairs = n * (n - 1) // 2
        distances = np.empty(n_pairs, dtype=np.float32)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                diff = xyz[j] - xyz[i]
                distances[idx] = np.sqrt(np.sum(diff * diff))
                idx += 1
        return distances

def get_distance_array(out_dir: str, frame_idx: int) -> np.ndarray:
    """Return flattened lower-triangle distances for the specified frame.

    - out_dir: directory containing coords.zarr (and atoms.parquet)
    - frame_idx: zero-based frame index

    Returns a numpy array of distances for all unique atom pairs (i>j).
    """
    xyz = load_frame_xyz(out_dir, frame_idx)
    return _lower_triangle_flat_distances(xyz)

def get_distance_array_noH(out_dir: str, frame_idx: int) -> np.ndarray:
    """Same as get_distance_array but excludes atoms whose element looks like hydrogen.

    The heuristic used: an atom is considered hydrogen if its element symbol (string)
    stripped and upper-cased begins with 'H' (covers 'H', 'H1', etc.).
    
    Optimized version that loads data efficiently and uses vectorized operations.
    """
    atoms = load_atoms(out_dir)
    # element column expected; be robust to missing column
    if "element" not in atoms.columns:
        # no element information available -> fall back to full distance array
        return get_distance_array(out_dir, frame_idx)
    
    # Vectorized element processing - avoid list comprehension
    elems = atoms["element"].to_numpy()
    # Create mask for non-hydrogen atoms more efficiently
    non_h_mask = np.array([not str(elem).strip().upper().startswith("H") for elem in elems], dtype=bool)
    indices = np.nonzero(non_h_mask)[0]
    
    if indices.size < 2:
        return np.array([], dtype=float)
    
    # Load coordinates only once and slice efficiently
    xyz_all = load_frame_xyz(out_dir, frame_idx)
    xyz = xyz_all[indices, :]  # This creates a view when possible
    return _lower_triangle_flat_distances(xyz)


def get_distance_array_noH_fast(out_dir: str, frame_idx: int, sample_size: int = 200, seed: int = 0) -> np.ndarray:
    """Return flattened lower-triangle distances for non-H atoms, but if the number
    of non-H atoms is greater than `sample_size` randomly sample `sample_size`
    atoms (without replacement) using a fixed `seed` for reproducibility.

    - out_dir: directory containing atoms.parquet and coords.zarr
    - frame_idx: zero-based frame index
    - sample_size: maximum number of non-H atoms to include (default 200)
    - seed: RNG seed for reproducible sampling
    
    Optimized version with improved vectorization and reduced data loading.
    """
    atoms = load_atoms(out_dir)
    if "element" not in atoms.columns:
        # fallback to full distances if element info missing
        return get_distance_array(out_dir, frame_idx)

    # Vectorized element processing - avoid list comprehension
    elems = atoms["element"].to_numpy()
    # Create mask for non-hydrogen atoms more efficiently  
    non_h_mask = np.array([not str(elem).strip().upper().startswith("H") for elem in elems], dtype=bool)
    indices = np.nonzero(non_h_mask)[0]
    
    if indices.size < 2:
        return np.array([], dtype=float)

    # Load coordinates only once
    xyz_all = load_frame_xyz(out_dir, frame_idx)

    # If small enough, compute full distances for all non-H atoms
    if indices.size <= int(sample_size):
        xyz = xyz_all[indices, :]  # Create view when possible
        return _lower_triangle_flat_distances(xyz)

    # Otherwise, sample `sample_size` indices reproducibly and compute distances
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(indices, size=int(sample_size), replace=False)
    xyz = xyz_all[chosen, :]  # Create view when possible
    return _lower_triangle_flat_distances(xyz)


def get_conformation_landscape(
    out_dir: str,
    method: str = "noH_fast",
    sample_size: int = 200,
    seed: int = 0,
    chunk_size: int = 1000,
    return_atom_indices: bool = False,
    dtype=np.float32,
    n_threads: int = 1,
):
    """Fast conformational landscape matrix.

    Returns a 2D array with shape (n_frames, n_distances) where each row is the
    flattened lower-triangle distance vector for a subset of atoms based on the
    specified method. This is an optimized version that:

    - Loads atom table once.
    - Selects / samples atom indices once (reproducibly via `seed` for sampling methods).
    - Extracts the coordinates subset across all frames in one slice.
    - Computes all pairwise distances for every frame in a single vectorized
      operation avoiding Python loops per frame.

    Parameters:
        out_dir: Directory containing atoms.parquet and coords.zarr.
        method: Distance calculation method:
            - "noH_fast": Use non-H atoms, sample up to `sample_size` if too many (default)
            - "noH": Use all non-H atoms without sampling
            - "all": Use all atoms (including hydrogens)
            - "noH_noROH_fast": Use non-H atoms excluding ROH residue, sample up to `sample_size`
            - "noH_noROH": Use all non-H atoms excluding ROH residue, without sampling
        sample_size: Maximum number of atoms to include for sampling methods (default 200).
                    Only used when method is "noH_fast" or "noH_noROH_fast" and there are
                    more eligible atoms than this limit.
        seed: RNG seed for reproducible sampling (only used with sampling methods).
        chunk_size: Number of frames to process in each memory-efficient chunk (default 1000).
                    Reduces peak memory usage for large trajectories by processing frames in batches.
        return_atom_indices: If True, return (landscape, chosen_atom_indices) tuple instead
                    of just the landscape array.
        dtype: dtype of the returned matrix. The distances are always computed in float32;
                    np.float64 stores those float32 values widened, which is what a later
                    conversion of the float32 matrix gives, without holding both matrices.
        n_threads: number of threads that compute frame chunks (the result does not depend on it).

    Notes:
        - If <2 eligible atoms, returns an array of shape (n_frames, 0).
        - Uses vectorized operations for optimal performance.
    """
    if method not in ["noH_fast", "noH", "all", "noH_noROH_fast", "noH_noROH"]:
        raise ValueError(f"method must be one of ['noH_fast', 'noH', 'all', 'noH_noROH_fast', 'noH_noROH'], got '{method}'")

    z = load_coords(out_dir)  # expected shape (n_frames, n_atoms, 3)
    # Ensure we have a numpy array with float32 dtype to avoid large float64
    z = np.asarray(z)
    # If axes were accidentally swapped (some code may have written (n_atoms,n_frames,3)),
    # detect and transpose: prefer z.shape[1] == atom_count when atoms available.
    n_frames = int(z.shape[0])
    if n_frames == 0:
        return np.zeros((0, 0), dtype=float)

    atoms = load_atoms(out_dir)
    # If atoms table length matches z.shape[0], it's likely axes were swapped -> transpose
    try:
        atom_count = len(atoms)
    except Exception:
        atom_count = int(z.shape[1]) if z.ndim >= 2 else 0
    if z.ndim == 3 and z.shape[1] != atom_count and z.shape[0] == atom_count:
        # transpose from (n_atoms, n_frames, 3) -> (n_frames, n_atoms, 3)
        z = np.transpose(z, (1, 0, 2))
    n_frames = int(z.shape[0])
    
    # Determine atom indices based on method
    if method == "all":
        # Use all atoms
        all_indices = np.arange(z.shape[1], dtype=int)
    elif method in ["noH", "noH_fast"]:
        # Use non-hydrogen atoms
        if "element" not in atoms.columns:
            # Fall back to all atoms if no element information
            all_indices = np.arange(z.shape[1], dtype=int)
        else:
            elems = [str(x).strip().upper() for x in atoms["element"].to_list()]
            non_h_mask = np.array([not e.startswith("H") for e in elems], dtype=bool)
            all_indices = np.nonzero(non_h_mask)[0]
    elif method in ["noH_noROH_fast", "noH_noROH"]:
        # Use non-hydrogen atoms excluding ROH residue (anomer-independent)
        if "element" not in atoms.columns:
            all_indices = np.arange(z.shape[1], dtype=int)
        else:
            elems = [str(x).strip().upper() for x in atoms["element"].to_list()]
            res_names = [str(x) for x in atoms["res_name"].to_list()]
            mask = np.array([
                not e.startswith("H") and r != "ROH"
                for e, r in zip(elems, res_names)
            ], dtype=bool)
            all_indices = np.nonzero(mask)[0]
    else:
        raise ValueError(f"Unsupported method: {method}")

    if all_indices.size < 2:
        if return_atom_indices:
            return np.zeros((n_frames, 0), dtype=float), np.array([], dtype=int)
        return np.zeros((n_frames, 0), dtype=float)

    # Apply sampling if method is a *_fast variant and we have too many atoms
    is_fast = method in ("noH_fast", "noH_noROH_fast")
    if is_fast and all_indices.size > int(sample_size):
        rng = np.random.default_rng(int(seed))
        chosen = rng.choice(all_indices, size=int(sample_size), replace=False)
        chosen.sort()  # keep deterministic ordering for reproducibility of pair index mapping
    else:
        # For non-fast method or fast with few atoms, use all available indices
        chosen = all_indices

    m = chosen.size
    # Number of pair distances per frame
    n_pairs = m * (m - 1) // 2
    if n_pairs == 0:
        if return_atom_indices:
            return np.zeros((n_frames, 0), dtype=float), chosen
        return np.zeros((n_frames, 0), dtype=float)

    # Gather coordinates subset: shape (n_frames, m, 3)
    # zarr supports advanced indexing with a list/ndarray of indices.
    coords_sub = z[:, chosen, :]
    # ensure numpy ndarray and use float32 to reduce memory
    coords_sub = np.asarray(coords_sub, dtype=np.float32)

    # Precompute pair indices once (lower triangle i>j)
    i_idx, j_idx = np.tril_indices(m, k=-1)
    
    # Chunked distance computation to reduce memory usage. Every frame is
    # computed on its own, so the chunk length and the number of threads do
    # not change the values; the chunks are written straight into the result
    # (no list of chunks next to the stacked matrix). A chunk is kept to
    # about 64 MiB of temporaries per thread.
    chunk_size = max(1, min(chunk_size, n_frames, (64 * 2**20) // (n_pairs * 48) or 1))
    landscape = np.empty((n_frames, n_pairs), dtype=dtype)

    def fill_chunk(start: int) -> None:
        end = min(start + chunk_size, n_frames)
        chunk_coords = coords_sub[start:end]  # Shape: (chunk_frames, m, 3)
        chunk_diffs = chunk_coords[:, i_idx, :] - chunk_coords[:, j_idx, :]  # Shape: (chunk_frames, n_pairs, 3)
        chunk_dist = np.sqrt(np.sum(chunk_diffs * chunk_diffs, axis=-1)).astype(np.float32)  # Shape: (chunk_frames, n_pairs)
        landscape[start:end] = chunk_dist

    starts = range(0, n_frames, chunk_size)
    if int(n_threads) > 1 and len(starts) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=int(n_threads)) as pool:
            list(pool.map(fill_chunk, starts))
    else:
        for start in starts:
            fill_chunk(start)
    if return_atom_indices:
        return landscape, chosen
    return landscape

# -------------------------
# PDB writer
# -------------------------
def _u8_to_char(u8: int) -> str:
    return " " if u8 == 0 else chr(int(u8))

def _pdb_atom_line(
    serial: int, name: str, res_name: str, chain_u8: int, res_id: int,
    x: float, y: float, z: float, element: str
) -> str:
    # Always write "ATOM  "
    # PDB fixed width fields
    chain_id = _u8_to_char(chain_u8)
    return (
        f"ATOM  "
        f"{serial:5d} "
        f"{name:>4s}"
        f" "
        f"{res_name:>3s} "
        f"{chain_id:1s}"
        f"{res_id:4d}"
        f"    "  # icode + 3 spaces
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"  1.00  0.00"  # occupancy, bfactor (placeholder to keep common readers happy)
        f"          "
        f"{element:>2s}"
        f"\n"
    )

def _complete_bond_pairs_from_geometry(
    atoms_df: pl.DataFrame,
    bonds_df: pl.DataFrame,
    xyz: np.ndarray,
) -> np.ndarray:
    """Return stored bonds supplemented by unambiguous covalent-distance bonds.

    MOL2 is authoritative. Geometry is only used to restore omitted bonds, most
    commonly hydrogen or carbohydrate ring/substituent bonds. Conservative
    covalent radii and valence limits prevent nearby non-bonded atoms from being
    connected.
    """
    n_atoms = len(atoms_df)
    existing: set[tuple[int, int]] = set()
    if not bonds_df.is_empty():
        for a, b in bonds_df.select(["a", "b"]).iter_rows():
            a_i, b_i = int(a), int(b)
            if not (0 <= a_i < n_atoms and 0 <= b_i < n_atoms):
                raise ValueError(
                    "Cannot complete PDB connectivity: stored bond references atom "
                    f"index outside 0..{n_atoms - 1}: {(a_i, b_i)}"
                )
            if a_i != b_i:
                existing.add(tuple(sorted((a_i, b_i))))

    coords = np.asarray(xyz, dtype=float)
    if coords.shape != (n_atoms, 3):
        return np.asarray(sorted(existing), dtype=np.int32).reshape((-1, 2))

    elements = [str(value).strip().upper() for value in atoms_df["element"].to_list()]
    radii = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
             "P": 1.07, "S": 1.05, "CL": 1.02, "BR": 1.20, "I": 1.39}
    max_degree = {"H": 1, "C": 4, "N": 4, "O": 2, "F": 1,
                  "P": 6, "S": 6, "CL": 1, "BR": 1, "I": 1}
    degree = [0] * n_atoms
    for a, b in existing:
        degree[a] += 1
        degree[b] += 1

    candidates: list[tuple[float, int, int]] = []
    for i in range(n_atoms):
        elem_i = elements[i]
        radius_i = radii.get(elem_i)
        if radius_i is None:
            continue
        for j in range(i + 1, n_atoms):
            if (i, j) in existing:
                continue
            elem_j = elements[j]
            radius_j = radii.get(elem_j)
            if radius_j is None or (elem_i == "H" and elem_j == "H"):
                continue
            distance = float(np.linalg.norm(coords[i] - coords[j]))
            # 0.30 A tolerance handles rounded PDB coordinates without making
            # hydrogen bonds or close non-bonded contacts look covalent.
            if 0.40 < distance <= radius_i + radius_j + 0.30:
                candidates.append((distance, i, j))

    for _, i, j in sorted(candidates):
        if degree[i] >= max_degree.get(elements[i], 4):
            continue
        if degree[j] >= max_degree.get(elements[j], 4):
            continue
        existing.add((i, j))
        degree[i] += 1
        degree[j] += 1

    return np.asarray(sorted(existing), dtype=np.int32).reshape((-1, 2))


def _pdb_conect_block(
    serials_np: np.ndarray,
    bonds_df: pl.DataFrame,
    atoms_df: Optional[pl.DataFrame] = None,
    xyz: Optional[np.ndarray] = None,
) -> str:
    if atoms_df is not None and xyz is not None:
        pairs = _complete_bond_pairs_from_geometry(atoms_df, bonds_df, xyz)
    elif bonds_df.is_empty():
        return ""
    else:
        pairs = np.c_[bonds_df["a"].to_numpy(), bonds_df["b"].to_numpy()]
    # map atom_index -> serial
    s = serials_np
    n_atoms = s.shape[0]
    adjacency: dict[int, set[int]] = {int(serial): set() for serial in s}
    # Stored bonds are atom indices.  Silently dropping an invalid pair creates a
    # chemically incomplete PDB, so make the provenance error actionable.
    valid_mask = (pairs[:, 0] >= 0) & (pairs[:, 0] < n_atoms) & (pairs[:, 1] >= 0) & (pairs[:, 1] < n_atoms)
    if not np.all(valid_mask):
        bad = np.nonzero(~valid_mask)[0]
        examples = pairs[bad[:5]].tolist()
        raise ValueError(
            f"Cannot emit PDB CONECT records: {bad.size} bond(s) reference atom "
            f"indices outside 0..{n_atoms - 1}; examples={examples}"
        )

    for i in range(pairs.shape[0]):
        a = int(s[int(pairs[i, 0])])
        b = int(s[int(pairs[i, 1])])
        if a == b:
            continue
        adjacency[a].add(b)
        adjacency[b].add(a)

    isolated = sorted(serial for serial, neighbours in adjacency.items() if not neighbours)
    if isolated and len(adjacency) > 1:
        raise ValueError(
            "Cannot emit complete PDB CONECT records: atom serial(s) have no "
            f"covalent neighbours after MOL2 and geometry repair: {isolated[:10]}"
            + (" ..." if len(isolated) > 10 else "")
        )

    # Emit the complete adjacency for each atom. PDB permits four neighbours
    # per CONECT line; repeat the source serial when more are present.
    out = io.StringIO()
    for source in sorted(adjacency):
        targets = sorted(adjacency[source])
        for start in range(0, len(targets), 4):
            out.write(f"CONECT{source:5d}")
            for target in targets[start:start + 4]:
                out.write(f"{target:5d}")
            out.write("\n")
    return out.getvalue()

def pdb_string_for_frame(out_dir: str, frame_idx: int, title: Optional[str] = None) -> str:
    atoms = load_atoms(out_dir)
    bonds = load_bonds(out_dir)
    xyz = load_frame_xyz(out_dir, frame_idx)

    serials = atoms["serial"].to_numpy()
    names = atoms["name"].to_list()
    resn = atoms["res_name"].to_list()
    chain_u8 = atoms["chain_id_u8"].to_numpy()
    resid = atoms["res_id"].to_numpy()
    elem = atoms["element"].to_list()

    out = io.StringIO()
    if title:
        out.write(f"TITLE     {title}\n")
    out.write(f"MODEL     {frame_idx+1}\n")

    for i in range(len(serials)):
        x, y, z = xyz[i]
        out.write(_pdb_atom_line(
            serial=int(serials[i]),
            name=names[i],
            res_name=resn[i],
            chain_u8=int(chain_u8[i]),
            res_id=int(resid[i]),
            x=float(x), y=float(y), z=float(z),
            element=elem[i],
        ))

    conect = _pdb_conect_block(serials, bonds, atoms, xyz)
    if conect:
        out.write(conect)

    out.write("ENDMDL\n")
    return out.getvalue()

def write_pdb_for_frame(
    out_dir: str,
    frame_idx: int,
    path: str,
    title: Optional[str] = None,
    flip: bool = False,
) -> None:
    """Write a PDB for one frame.

    Parameters:
        out_dir: directory containing atoms.parquet, bonds.parquet (optional) and coords.zarr
        frame_idx: zero-based frame index
        path: output PDB path
        title: optional TITLE record
        flip: if True, perform alpha/beta OH flip using logic analogous to flip.flip_alpha_beta
              but preserving the original C1-O1, C1-H1 and O1-HO1 bond lengths present in the
              current frame instead of using hard‑coded values (1.43, 1.09).

    Flip details:
        - Identifies atoms: (res_id==2 & name in {C1,H1}) and (res_id==1 & name in {O1,HO1}).
        - Swaps orientation by placing O1 along the original C1->H1 direction at the original
          C1-O1 bond length; places H1 along the original C1->O1 direction at the original
          C1-H1 bond length; translates HO1 by the same vector applied to O1 (preserving
          O1-HO1 vector). If residue 2's name ends with A/B the letter is toggled.
        - If any required atom is missing, flip is skipped with a silent fallback.
    """
    atoms = load_atoms(out_dir)
    bonds = load_bonds(out_dir)
    xyz = load_frame_xyz(out_dir, frame_idx).copy()  # (n_atoms,3) mutable copy
    res_names = atoms["res_name"].to_list()

    if flip:
        # Locate indices
        names = atoms["name"].to_list()
        res_ids = atoms["res_id"].to_list()
        # The geometric operation creates the opposite anomer, so its reducing
        # residue name must change even when coordinates cannot be flipped.
        res_names = flip_anomeric_residue_names(res_names, res_ids)

        idx_C1 = next((i for i,(n,r) in enumerate(zip(names, res_ids)) if n == "C1" and r == 2), None)
        idx_H1 = next((i for i,(n,r) in enumerate(zip(names, res_ids)) if n == "H1" and r == 2), None)
        idx_O1 = next((i for i,(n,r) in enumerate(zip(names, res_ids)) if n == "O1" and r == 1), None)
        idx_HO1 = next((i for i,(n,r) in enumerate(zip(names, res_ids)) if n == "HO1" and r == 1), None)

        if None not in (idx_C1, idx_H1, idx_O1, idx_HO1):
            C1 = xyz[idx_C1]
            H1 = xyz[idx_H1]
            O1 = xyz[idx_O1]
            HO1 = xyz[idx_HO1]

            CO1_vec = O1 - C1
            CH1_vec = H1 - C1
            HO1O1_vec = HO1 - O1

            # Existing bond lengths
            len_C1O1 = float(np.linalg.norm(CO1_vec)) if np.linalg.norm(CO1_vec) > 0 else 1.43
            len_C1H1 = float(np.linalg.norm(CH1_vec)) if np.linalg.norm(CH1_vec) > 0 else 1.09

            # New coordinates using normalized swapped directions & preserved lengths
            if np.linalg.norm(CH1_vec) > 0 and np.linalg.norm(CO1_vec) > 0:
                new_O1 = C1 + (CH1_vec / np.linalg.norm(CH1_vec)) * len_C1O1
                new_H1 = C1 + (CO1_vec / np.linalg.norm(CO1_vec)) * len_C1H1
                new_HO1 = new_O1 + HO1O1_vec  # preserve relative vector

                xyz[idx_O1] = new_O1
                xyz[idx_H1] = new_H1
                xyz[idx_HO1] = new_HO1

        # else: silently ignore if atoms missing (fallback to original coordinates)

    # Prepare for writing (reuse existing helper formatting routines)
    serials = atoms["serial"].to_numpy()
    names = atoms["name"].to_list()
    chain_u8 = atoms["chain_id_u8"].to_numpy()
    resid = atoms["res_id"].to_numpy()
    elem = atoms["element"].to_list()

    storage = get_storage_manager()
    with storage.open(path, "w") as fh:
        if title:
            fh.write(f"TITLE     {title}\n")
        fh.write(f"MODEL     {frame_idx+1}\n")
        for i in range(len(serials)):
            x,y,z = xyz[i]
            fh.write(_pdb_atom_line(
                serial=int(serials[i]),
                name=names[i],
                res_name=res_names[i],
                chain_u8=int(chain_u8[i]),
                res_id=int(resid[i]),
                x=float(x), y=float(y), z=float(z),
                element=elem[i],
            ))
        conect = _pdb_conect_block(serials, bonds, atoms, xyz)
        if conect:
            fh.write(conect)
        fh.write("ENDMDL\n")


def get_mol_from_frame(
    out_dir: str,
    frame_idx: int,
    flip: bool = False,
    sanitize: bool = True,
) -> "object":  # return type is rdkit.Chem.Mol when RDKit is installed
    """Return an RDKit Mol for a given trajectory frame with coordinates and (optional) alpha/beta flip.

    This mirrors the data written by ``write_pdb_for_frame`` but constructs the
    molecule directly in‑memory for downstream torsion / conformer analysis.

    Atom properties set (when present in the atoms table):
        - serial (int)
        - name (string)
        - res_name (string)
        - res_id (int)
        - chain_id (single character derived from chain_id_u8)

    Parameters
    ----------
    out_dir : str
        Directory containing ``atoms.parquet`` / ``bonds.parquet`` / ``coords.zarr``.
    frame_idx : int
        Zero-based frame index.
    flip : bool, default False
        Apply the same optional alpha/beta OH flip logic as ``write_pdb_for_frame``.
    sanitize : bool, default True
        Run ``Chem.SanitizeMol`` after construction (errors are caught & ignored if sanitize=True).

    Returns
    -------
    rdkit.Chem.Mol
        A molecule with a single conformer (ID 0) containing the frame's coordinates.
    """
    # Local imports so that importing glypdbio does not mandate RDKit availability
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore
    atoms = load_atoms(out_dir)
    bonds = load_bonds(out_dir)
    xyz = load_frame_xyz(out_dir, frame_idx).copy()
    res_names = atoms["res_name"].to_list()

    # Optional flip duplicates the coordinate transformation in write_pdb_for_frame
    if flip:
        try:
            names = atoms["name"].to_list()
            res_ids = atoms["res_id"].to_list()
            # Keep metadata in sync with the generated opposite anomer even if
            # the coordinate fallback below cannot find all OH atoms.
            res_names = flip_anomeric_residue_names(res_names, res_ids)
            idx_C1 = next((i for i,(n,r) in enumerate(zip(names, res_ids)) if n == "C1" and r == 2), None)
            idx_H1 = next((i for i,(n,r) in enumerate(zip(names, res_ids)) if n == "H1" and r == 2), None)
            idx_O1 = next((i for i,(n,r) in enumerate(zip(names, res_ids)) if n == "O1" and r == 1), None)
            idx_HO1 = next((i for i,(n,r) in enumerate(zip(names, res_ids)) if n == "HO1" and r == 1), None)
            if None not in (idx_C1, idx_H1, idx_O1, idx_HO1):
                C1 = xyz[idx_C1]; H1 = xyz[idx_H1]; O1 = xyz[idx_O1]; HO1 = xyz[idx_HO1]
                CO1_vec = O1 - C1; CH1_vec = H1 - C1; HO1O1_vec = HO1 - O1
                len_C1O1 = float(np.linalg.norm(CO1_vec)) or 1.43
                len_C1H1 = float(np.linalg.norm(CH1_vec)) or 1.09
                if np.linalg.norm(CH1_vec) > 0 and np.linalg.norm(CO1_vec) > 0:
                    new_O1 = C1 + (CH1_vec / np.linalg.norm(CH1_vec)) * len_C1O1
                    new_H1 = C1 + (CO1_vec / np.linalg.norm(CO1_vec)) * len_C1H1
                    new_HO1 = new_O1 + HO1O1_vec
                    xyz[idx_O1] = new_O1; xyz[idx_H1] = new_H1; xyz[idx_HO1] = new_HO1
        except Exception:
            pass  # silent fallback

    # Build RDKit molecule
    rw = Chem.RWMol()
    serials = atoms["serial"].to_numpy()
    elem_col = atoms["element"].to_list()
    names_col = atoms["name"].to_list()
    resn_col = res_names
    resid_col = atoms["res_id"].to_list()
    try:
        chain_ids_u8 = atoms["chain_id_u8"].to_numpy()
    except Exception:
        chain_ids_u8 = np.zeros(len(serials), dtype=int)

    for i in range(len(serials)):
        sym = str(elem_col[i]).strip() or "C"
        try:
            atom = Chem.Atom(sym)
        except Exception:
            atom = Chem.Atom("C")  # fallback
        # Annotate useful metadata
        atom.SetProp("serial", str(int(serials[i])))
        atom.SetProp("name", str(names_col[i]))
        atom.SetProp("res_name", str(resn_col[i]))
        atom.SetProp("res_id", str(int(resid_col[i])))
        chain_char = _u8_to_char(int(chain_ids_u8[i])) if len(str(chain_ids_u8[i])) else " "
        atom.SetProp("chain_id", chain_char)
        rw.AddAtom(atom)

    # Add bonds if present
    try:
        if not bonds.is_empty():
            a_arr = bonds["a"].to_numpy(); b_arr = bonds["b"].to_numpy()
            for a,b in zip(a_arr, b_arr):
                a_i = int(a); b_i = int(b)
                if 0 <= a_i < rw.GetNumAtoms() and 0 <= b_i < rw.GetNumAtoms():
                    # Avoid duplicates; RDKit ignores duplicate add attempts but we can check
                    if rw.GetBondBetweenAtoms(a_i, b_i) is None:
                        rw.AddBond(a_i, b_i, Chem.BondType.SINGLE)
    except Exception:
        pass

    mol = rw.GetMol()
    # Create conformer
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        x,y,z = map(float, xyz[i])
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(x,y,z))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)

    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
    # Ensure we have a 3D flag to keep downstream code happy
    try:
        from rdkit.Chem import AllChem  # noqa: F401
        mol.SetProp("_Name", f"frame_{frame_idx}")
    except Exception:
        pass
    return mol


# -------------------------
# PDB conversion utilities
# -------------------------
def convert_pdb(input_pdb_path: str, output_pdb_path: str, naming_format: str) -> None:
    """
    Convert a PDB file from GLYCAM format to the specified naming format.
    
    Parameters:
        input_pdb_path: Path to the input PDB file (assumed to be in GLYCAM format)
        output_pdb_path: Path where the converted PDB file will be saved
        naming_format: Target naming format ('GLYCAM', 'PDB', or 'CHARMM')
    
    Raises:
        ValueError: If naming_format is not one of the supported formats
        FileNotFoundError: If input_pdb_path does not exist
    """
    import re
    import os
    
    # Validate inputs
    if naming_format not in ['GLYCAM', 'PDB', 'CHARMM']:
        raise ValueError(f"naming_format must be one of ['GLYCAM', 'PDB', 'CHARMM'], got '{naming_format}'")
    
    if not os.path.exists(input_pdb_path):
        raise FileNotFoundError(f"Input PDB file not found: {input_pdb_path}")
    
    # Read the input PDB file
    storage = get_storage_manager()
    with storage.open(input_pdb_path, 'r') as file:
        filedata = file.read()

    # Apply naming format conversions
    filedata = _convert_pdb_text_naming(filedata, naming_format)
    
    # Write the converted data to a temporary file first
    import tempfile
    import shutil
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as temp_file:
        temp_file.write(filedata)
        temp_filename = temp_file.name
    
    try:
        
        # Move the final file to the desired output path
        os.makedirs(os.path.dirname(output_pdb_path), exist_ok=True)
        shutil.move(temp_filename, output_pdb_path)
        
    except Exception as e:
        # Clean up temporary file if something goes wrong
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)
        raise e


def _add_pdb_remarks(
    filename: str, 
    glytoucan_id: str = None, 
    iupac: str = None,
    cluster_id: str = None,
    cluster_population: float = None,
) -> None:
    """
    Add GlycoShape remarks to a PDB file.
    Based on the pdb_remark_adder function from pdb.py.
    
    Parameters:
        filename: Path to the PDB file to modify
        glytoucan_id: GlyTouCan ID to add as a remark (optional)
        iupac: IUPAC name to add as a remark (optional)
        cluster_id: Cluster ID for this structure (optional)
        cluster_population: Cluster population percentage (optional)
    """
    storage = get_storage_manager()
    with storage.open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write("REMARK     GENERATED BY GlycanAnalysisPipeline from GlycoShape     \n")
        f.write("REMARK        ____ _                ____  _                       \n")
        f.write("REMARK       / ___| |_   _  ___ ___/ ___|| |__   __ _ _ __   ___  \n")
        f.write("REMARK      | |  _| | | | |/ __/ _ \___ \| '_ \ / _` | '_ \ / _ \ \n")
        f.write("REMARK      | |_| | | |_| | (_| (_) |__) | | | | (_| | |_) |  __/ \n")
        f.write("REMARK       \____|_|\__, |\___\___/____/|_| |_|\__,_| .__/ \___| \n")
        f.write("REMARK               |___/                           |_|          \n")
        f.write("REMARK                   https://GlycoShape.org                   \n")
        # Add GlyTouCan ID and IUPAC remarks if provided
        if glytoucan_id:
            f.write(f"REMARK    GlyTouCan ID: {glytoucan_id}\n")
        if iupac:
            f.write(f"REMARK    IUPAC Name: {iupac}\n")
        
        # Add cluster information if provided
        if cluster_id is not None:
            f.write(f"REMARK    Cluster ID: {cluster_id}\n")
        if cluster_population is not None:
            f.write(f"REMARK    Cluster Population: {cluster_population}%\n")
        
        f.write("REMARK    Cite:   Restoring protein glycosylation with GlycoShape.\n")
        f.write("REMARK    Nat Methods 21, 2117–2127 (2024).  https://doi.org/10.1038/s41592-024-02464-7 \n")
        f.write("REMARK    Callum M. Ives* and Ojas Singh*, Silvia D'Andrea, Carl A. Fogarty, \n")
        f.write("REMARK    Aoife M. Harbison, Akash Satheesan, Beatrice Tropea, Elisa Fadda\n")
        f.write("REMARK    Data available under CC BY-NC-ND 4.0 for academic use only.\n")
        f.write("REMARK    Contact elisa.fadda@soton.ac.uk for commercial licence.\n")
        
        
        # Add MODEL line if cluster_id is provided
        if cluster_id is not None:
            f.write(f"MODEL     {cluster_id}\n")

        f.write(content)
        
        

_PDB_ANOMER_RESIDUE_PAIRS = {
    "NDG": ("NDG", "NAG"),  # GlcNAc alpha/beta
    "NAG": ("NDG", "NAG"),
    "A2G": ("A2G", "NGA"),  # GalNAc alpha/beta
    "NGA": ("A2G", "NGA"),
    "GLC": ("GLC", "BGC"),  # Glc alpha/beta
    "BGC": ("GLC", "BGC"),
    "GLA": ("GLA", "GAL"),  # Gal alpha/beta
    "GAL": ("GLA", "GAL"),
    "FUC": ("FUC", "FUL"),  # L-Fuc alpha/beta
    "FUL": ("FUC", "FUL"),
    "MAN": ("MAN", "BMA"),  # Man alpha/beta
    "BMA": ("MAN", "BMA"),
    "GCU": ("GCU", "BDP"),  # GlcA alpha/beta
    "BDP": ("GCU", "BDP"),
    "XYS": ("XYS", "XYP"),  # Xyl alpha/beta
    "XYP": ("XYS", "XYP"),
    "RAM": ("RAM", "RHM"),  # Rha alpha/beta
    "RHM": ("RAM", "RHM"),
    "RIB": ("RIB", "BDR"),  # Rib alpha/beta
    "BDR": ("RIB", "BDR"),
    "ARA": ("ARA", "ARB"),  # Ara alpha/beta
    "ARB": ("ARA", "ARB"),
}


def normalize_anomeric_pdb_text(filedata: str, target_anomer: str) -> str:
    """Normalize the reducing-residue name in fixed-width PDB atom records.

    This is used only for legacy/fallback text paths where the atom table is
    unavailable.  It understands both GLYCAM A/B codes and the corresponding
    three-letter PDB aliases, and leaves all non-reducing residues untouched.
    """
    target = str(target_anomer).strip().lower()
    if target not in {"alpha", "beta"}:
        return filedata
    target_index = 0 if target == "alpha" else 1
    converted_lines = []
    for line in filedata.splitlines(keepends=True):
        if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 26:
            try:
                is_reducing_residue = int(line[22:26].strip()) == 2
            except (TypeError, ValueError):
                is_reducing_residue = False
            if is_reducing_residue:
                current = line[17:20].strip()
                pair = _PDB_ANOMER_RESIDUE_PAIRS.get(current)
                desired = pair[target_index] if pair else normalize_anomeric_residue_name(current, target)
                if desired and desired != current:
                    line = f"{line[:17]}{desired:>3s}{line[20:]}"
        converted_lines.append(line)
    return "".join(converted_lines)


def _convert_pdb_text_naming(filedata: str, naming_format: str) -> str:
    """Convert PDB residue naming in text for the specified target format.

    Supported naming_format values: 'GLYCAM', 'PDB', 'CHARMM'.
    GLYCAM returns input unchanged.
    """
    if naming_format == 'GLYCAM':
        return filedata
    if naming_format == 'PDB':
        # GLYCAM names have a linkage-position prefix followed by a two-character
        # residue code (for example ``0VB``).  Convert each atom record exactly
        # once by editing its fixed-width resName field (PDB columns 18--20).
        # Whole-text sequential substitutions used to remap their own output:
        # ``0VB -> NGA -> GLC`` and ``0MB -> BMA -> MAN``.
        pdb_residue_names = {
            "YA": "NDG",  # GlcNAc alpha
            "YB": "NAG",  # GlcNAc beta
            "VA": "A2G",  # GalNAc alpha
            "VB": "NGA",  # GalNAc beta
            "GA": "GLC",  # Glc alpha
            "GB": "BGC",  # Glc beta
            "GL": "NGC",  # Neu5Gc alpha
            "LA": "GLA",  # Gal alpha
            "LB": "GAL",  # Gal beta
            "fA": "FUC",  # L-Fuc alpha
            "fB": "FUL",  # L-Fuc beta
            "MB": "BMA",  # Man beta
            "MA": "MAN",  # Man alpha
            "SA": "SIA",  # Neu5Ac alpha; beta source code is not defined here
            "ZA": "GCU",  # GlcA alpha
            "ZB": "BDP",  # GlcA beta
            "XA": "XYS",  # Xyl alpha
            "XB": "XYP",  # Xyl beta
            "uA": "IDR",  # IdoA alpha
            "hA": "RAM",  # Rha alpha
            "hB": "RHM",  # Rha beta
            "RA": "RIB",  # Rib alpha
            "RB": "BDR",  # Rib beta
            "AA": "ARA",  # Ara alpha
            "AB": "ARB",  # Ara beta
        }

        converted_lines = []
        for line in filedata.splitlines(keepends=True):
            if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 20:
                glycam_residue = line[17:20]
                pdb_residue = pdb_residue_names.get(glycam_residue[-2:])
                if pdb_residue is not None:
                    line = f"{line[:17]}{pdb_residue:>3s}{line[20:]}"
            converted_lines.append(line)
        return "".join(converted_lines)
    if naming_format == 'CHARMM':
        # Convert GLYCAM to CHARMM naming
        filedata = re.sub(r"\s\wYA ", " AGLC", filedata)  # GlcNAc alpha
        filedata = re.sub(r"\s\wYB ", " BGLC", filedata)  # GlcNAc beta
        filedata = re.sub(r"\s\wVA ", " AGAL", filedata)  # GalNAc alpha
        filedata = re.sub(r"\s\wVB ", " BGAL", filedata)  # GalNAc beta
        filedata = re.sub(r"\s\wGA ", " AGLC", filedata)  # Glc alpha
        filedata = re.sub(r"\s\wGB ", " BGLC", filedata)  # Glc beta
        filedata = re.sub(r"\s\wLA ", " AGAL", filedata)  # Gal alpha
        filedata = re.sub(r"\s\wLB ", " BGAL", filedata)  # Gal beta
        filedata = re.sub(r"\s\wf[A|B]", " FUC", filedata)  # Fuc alpha and beta
        filedata = re.sub(r"\s\wMA ", " AMAN", filedata)  # Man alpha
        filedata = re.sub(r"\s\wMB ", " BMAN", filedata)  # Man beta
        filedata = re.sub(r"\s\wSA ", " ANE5", filedata)  # Neu5Ac alpha
        filedata = re.sub(r"\s\wGL ", " ANE5", filedata)  # Neu5Gc
        filedata = re.sub(r"\s\wXA ", " AXYL", filedata)  # Xyl alpha
        filedata = re.sub(r"\s\wXB ", " BXYL", filedata)  # Xyl beta
        filedata = re.sub(r"\s\wuA ", " AIDO", filedata)  # IdoA alpha
        filedata = re.sub(r"\s\wZA ", " AGLC", filedata)  # GlcA alpha
        filedata = re.sub(r"\s\wZB ", " BGLC", filedata)  # GlcA beta
        filedata = re.sub(r"\s\whA ", " ARHM", filedata)  # Rha alpha
        filedata = re.sub(r"\s\whB ", " BRHM", filedata)  # Rha beta
        filedata = re.sub(r"\s\wAA ", " AARB", filedata)  # Ara alpha
        filedata = re.sub(r"\s\wAB ", " BARB", filedata)  # Ara beta
        filedata = re.sub(r"\s\wRA ", " ARIB", filedata)  # Rib alpha
        filedata = re.sub(r"\s\wRB ", " BRIB", filedata)  # Rib beta
        return filedata
    raise ValueError(f"naming_format must be one of ['GLYCAM','PDB','CHARMM'], got '{naming_format}'")


def emit_multimodel_pdbs(
    source_level_dir,
    dest_level_dir,
    cluster_data: dict,
    level_name: str,
    anomer_meta: dict | None = None,
    formats: list[str] | None = None,
) -> None:
    """Create multi-model PDBs (alpha/beta) for a level into GLYCAM and PDB formats.

    Preferred path: rebuild the multi-models from the underlying atoms/bonds/coords
    to ensure proper CONECT records and alignment across models using the first
    5 residues as the alignment core. If underlying trajectory data is not
    accessible, fall back to concatenating per‑cluster PDB files.

    Outputs one multi-model file per anomer under dest_level_dir/<FORMAT>/<anomer>.pdb.

    Additionally, GlycoShape remarks (glytoucan/iupac) are added at the top of the
    resulting files. Per-model REMARKs include cluster ID and population percent
    when available in cluster_data.
    """
    from pathlib import Path
    storage = get_storage_manager()

    if formats is None:
        formats = ["GLYCAM", "PDB"]

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
    except Exception:
        pass

    def _cluster_id_from_name(p: Path) -> int:
        m = re.search(r"cluster_(\d+)", p.stem)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return 10**9
        return 10**9

    source_level_dir = Path(source_level_dir)
    dest_level_dir = Path(dest_level_dir)

    # ---------- Helpers for rebuild-from-trajectory path ----------
    def _select_align_indices(atoms_df: pl.DataFrame, n_residues: int = 5) -> np.ndarray:
        """Return atom indices for alignment using heavy atoms of first n residues.

        Select the lowest n unique res_id values and include atoms whose element
        does not start with 'H'. Returns an int numpy array (possibly empty).
        """
        try:
            res_ids = atoms_df["res_id"].to_numpy()
        except Exception:
            return np.array([], dtype=int)
        try:
            elems = atoms_df["element"].to_numpy()
        except Exception:
            elems = np.array(["C"] * len(res_ids))
        # first n unique residue ids (sorted)
        uniq = []
        for r in res_ids:
            if r not in uniq:
                uniq.append(int(r))
            if len(uniq) >= int(n_residues):
                break
        uniq_set = set(uniq)
        mask = np.array([(rid in uniq_set) and (str(el).strip().upper()[:1] != "H") for rid, el in zip(res_ids, elems)], dtype=bool)
        idx = np.nonzero(mask)[0]
        return idx.astype(int)

    def _kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute optimal rotation (R) and translation (t) to map P->Q.

        P, Q: (m,3) arrays. Returns R (3x3) and t (3,) such that Q ≈ P@R + t.
        """
        Pc = P.mean(axis=0)
        Qc = Q.mean(axis=0)
        P0 = P - Pc
        Q0 = Q - Qc
        H = P0.T @ Q0
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = Qc - Pc @ R
        return R, t

    def _aligned_xyzs(data_dir: Path, frame_indices: list[int]) -> list[np.ndarray]:
        """Load xyz for frames and align each to the first using first 5 residues.

        Returns list of (n_atoms,3) float32 arrays.
        """
        atoms_df = load_atoms(str(data_dir))
        align_idx = _select_align_indices(atoms_df, n_residues=5)
        # Fallback: if insufficient atoms, skip alignment
        do_align = align_idx.size >= 3
        xyz0 = load_frame_xyz(str(data_dir), int(frame_indices[0]))
        out = [np.asarray(xyz0, dtype=np.float32)]
        if not do_align:
            for fi in frame_indices[1:]:
                out.append(np.asarray(load_frame_xyz(str(data_dir), int(fi)), dtype=np.float32))
            return out
        ref = np.asarray(xyz0[align_idx, :], dtype=np.float64)
        for fi in frame_indices[1:]:
            xyz = np.asarray(load_frame_xyz(str(data_dir), int(fi)), dtype=np.float64)
            P = xyz[align_idx, :]
            try:
                R, t = _kabsch(P, ref)
                aligned = (xyz @ R) + t
                out.append(aligned.astype(np.float32))
            except Exception:
                out.append(np.asarray(xyz, dtype=np.float32))
        return out

    def _write_multimodel_with_connect(
        data_dir: Path,
        out_path: Path,
        frame_indices: list[int],
        title_header: str | None,
        per_model_meta: list[tuple[str | None, float | None]] | None,
        target_anomer: str,
    ) -> None:
        """Write a GLYCAM multi-model PDB with CONECT records per model.

        per_model_meta: list of (cluster_id, population_pct) parallel to frame_indices
        """
        atoms = load_atoms(str(data_dir))
        bonds = load_bonds(str(data_dir))
        # Build aligned coordinates
        aligned_xyzs = _aligned_xyzs(data_dir, frame_indices)

        serials = atoms["serial"].to_numpy()
        names = atoms["name"].to_list()
        resn = normalize_anomeric_residue_names(
            atoms["res_name"].to_list(),
            atoms["res_id"].to_list(),
            target_anomer,
        )
        chain_u8 = atoms["chain_id_u8"].to_numpy()
        resid = atoms["res_id"].to_numpy()
        elem = atoms["element"].to_list()
        # Covalent topology is invariant across conformers. Geometry repair is
        # O(n_atoms^2), so infer the complete CONECT block once from the first
        # aligned model instead of repeating it for every model.
        conect = (
            _pdb_conect_block(serials, bonds, atoms, aligned_xyzs[0])
            if aligned_xyzs
            else ""
        )

        with storage.open(out_path, "w") as fh:
            if title_header:
                fh.write(title_header + "\n")
            for i, xyz in enumerate(aligned_xyzs, start=1):
                fh.write(f"MODEL     {i:4d}\n")
                # Per-model REMARKs
                if per_model_meta and i-1 < len(per_model_meta):
                    cid, pop = per_model_meta[i-1]
                    if cid is not None:
                        fh.write(f"REMARK   CLUSTER_ID {cid}\n")
                    if pop is not None:
                        fh.write(f"REMARK   CLUSTER_POPULATION_PCT {pop}\n")
                for j in range(len(serials)):
                    x, y, z = map(float, xyz[j])
                    fh.write(_pdb_atom_line(
                        serial=int(serials[j]),
                        name=names[j],
                        res_name=resn[j],
                        chain_u8=int(chain_u8[j]),
                        res_id=int(resid[j]),
                        x=x, y=y, z=z,
                        element=elem[j],
                    ))
                if conect:
                    fh.write(conect)
                fh.write("ENDMDL\n")
            fh.write("END\n")

    def _parse_multimodel_xyzs_from_pdb(pdb_text: str, n_atoms: int) -> list[np.ndarray]:
        """Parse a multi-model PDB text into a list of (n_atoms,3) arrays.

        Assumes models are contiguous blocks of ATOM/HETATM ending with ENDMDL.
        Coordinates are mapped in the encountered order. If a model has fewer
        or more atoms than n_atoms, it will be skipped to avoid mismatches.
        """
        models: list[np.ndarray] = []
        cur: list[tuple[float, float, float]] = []
        for line in pdb_text.splitlines():
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    # PDB fixed columns: x[30:38], y[38:46], z[46:54]
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    cur.append((x, y, z))
                except Exception:
                    continue
            elif line.startswith("ENDMDL"):
                if len(cur) == n_atoms:
                    models.append(np.asarray(cur, dtype=np.float32))
                cur = []
        # Handle trailing model if file lacks ENDMDL
        if cur:
            if len(cur) == n_atoms:
                models.append(np.asarray(cur, dtype=np.float32))
        return models

    for fmt in formats:
        out_fmt_dir = dest_level_dir / fmt
        storage.mkdir(out_fmt_dir)
        for anomer in ["alpha", "beta"]:
            src_dir = Path(source_level_dir) / anomer
            out_path = out_fmt_dir / f"{anomer}.pdb"
            wrote = False
            # First preference: rebuild from pre-aggregated alpha/beta.pdb (fast, no zarr download)
            if not wrote:
                # If a pre-aggregated alpha.pdb/beta.pdb exists at the source level, rebuild it
                preagg = Path(source_level_dir) / f"{anomer}.pdb"
                if storage.exists(preagg):
                    try:
                        # We prefer to rebuild to inject CONECT + alignment; use atoms/bonds
                        # If atoms/bonds unavailable, fall back to simple conversion
                        # Determine data_dir as above
                        glycan_root = source_level_dir.parent
                        glycan_dir = glycan_root.parent
                        data_dir = glycan_dir / "data"
                        atoms_df = None
                        bonds_df = None
                        try:
                            atoms_df = load_atoms(str(data_dir))
                            bonds_df = load_bonds(str(data_dir))
                        except Exception:
                            atoms_df = None
                        raw = storage.read_text(preagg)
                        if atoms_df is not None:
                            n_atoms = len(atoms_df)
                            xyzs = _parse_multimodel_xyzs_from_pdb(raw, n_atoms=n_atoms)
                            if xyzs:
                                # Align using first-5-residues heavy-atom indices
                                align_idx = _select_align_indices(atoms_df, n_residues=5)
                                do_align = align_idx.size >= 3
                                if do_align:
                                    ref = np.asarray(xyzs[0][align_idx, :], dtype=np.float64)
                                    aligned = []
                                    for i, arr in enumerate(xyzs):
                                        if i == 0:
                                            aligned.append(arr)
                                            continue
                                        P = np.asarray(arr[align_idx, :], dtype=np.float64)
                                        try:
                                            R, t = _kabsch(P, ref)
                                            aa = (np.asarray(arr, dtype=np.float64) @ R) + t
                                            aligned.append(aa.astype(np.float32))
                                        except Exception:
                                            aligned.append(arr)
                                    xyzs = aligned
                                serials = atoms_df["serial"].to_numpy()
                                names = atoms_df["name"].to_list()
                                resn = normalize_anomeric_residue_names(
                                    atoms_df["res_name"].to_list(),
                                    atoms_df["res_id"].to_list(),
                                    anomer,
                                )
                                chain_u8 = atoms_df["chain_id_u8"].to_numpy()
                                resid = atoms_df["res_id"].to_numpy()
                                elem = atoms_df["element"].to_list()
                                conect = (
                                    _pdb_conect_block(serials, bonds_df, atoms_df, xyzs[0])
                                    if xyzs
                                    else ""
                                )
                                # Write GLYCAM then convert if needed.
                                if fmt == "GLYCAM":
                                    with storage.open(out_path, "w") as fh:
                                        fh.write(
                                            f"REMARK   GLYCAN MULTI-MODEL PDB  LEVEL {level_name.upper()}  ANOMER {anomer.upper()}\n"
                                        )
                                        for i, xyz in enumerate(xyzs, start=1):
                                            fh.write(f"MODEL     {i:4d}\n")
                                            for j in range(len(serials)):
                                                x, y, z = map(float, xyz[j])
                                                fh.write(_pdb_atom_line(
                                                    serial=int(serials[j]),
                                                    name=names[j],
                                                    res_name=resn[j],
                                                    chain_u8=int(chain_u8[j]),
                                                    res_id=int(resid[j]),
                                                    x=x, y=y, z=z,
                                                    element=elem[j],
                                                ))
                                            if conect:
                                                fh.write(conect)
                                            fh.write("ENDMDL\n")
                                        fh.write("END\n")
                                    wrote = True
                                else:
                                    # Write to temp GLYCAM then convert
                                    import tempfile, os
                                    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
                                        tmp_path = tmp.name
                                    try:
                                        with open(tmp_path, "w", encoding="utf-8") as fh:
                                            fh.write(
                                                f"REMARK   GLYCAN MULTI-MODEL PDB  LEVEL {level_name.upper()}  ANOMER {anomer.upper()}\n"
                                            )
                                            for i, xyz in enumerate(xyzs, start=1):
                                                fh.write(f"MODEL     {i:4d}\n")
                                                for j in range(len(serials)):
                                                    x, y, z = map(float, xyz[j])
                                                    fh.write(_pdb_atom_line(
                                                        serial=int(serials[j]),
                                                        name=names[j],
                                                        res_name=resn[j],
                                                        chain_u8=int(chain_u8[j]),
                                                        res_id=int(resid[j]),
                                                        x=x, y=y, z=z,
                                                        element=elem[j],
                                                    ))
                                                if conect:
                                                    fh.write(conect)
                                                fh.write("ENDMDL\n")
                                            fh.write("END\n")
                                        with open(tmp_path, "r", encoding="utf-8") as _fh:
                                            raw2 = _fh.read()
                                        converted = _convert_pdb_text_naming(raw2, fmt)
                                        storage.write_text(out_path, converted)
                                        wrote = True
                                    finally:
                                        try:
                                            os.unlink(tmp_path)
                                        except Exception:
                                            pass
                            else:
                                # No parseable models -> simple conversion
                                normalized_raw = normalize_anomeric_pdb_text(raw, anomer)
                                converted = _convert_pdb_text_naming(normalized_raw, fmt)
                                storage.write_text(out_path, converted)
                                wrote = True
                        else:
                            # atoms/bonds not available -> simple conversion
                            normalized_raw = normalize_anomeric_pdb_text(raw, anomer)
                            converted = _convert_pdb_text_naming(normalized_raw, fmt)
                            storage.write_text(out_path, converted)
                            wrote = True
                    except Exception:
                        wrote = False

            # Second preference: rebuild from trajectory frames (coords.zarr) if available
            if not wrote:
                try:
                    # Identify the data directory containing coords.zarr
                    glycan_root = source_level_dir.parent  # .../output
                    glycan_dir = glycan_root.parent        # .../<glycan>
                    data_dir = glycan_dir / "data"
                    if storage.exists(data_dir / "coords.zarr") and isinstance(cluster_data, dict):
                        # Collect frame indices for this anomer using representative_idx
                        level = cluster_data.get("levels", {}).get(level_name, {})
                        frames: list[int] = []
                        per_meta: list[tuple[str | None, float | None]] = []
                        for c in level.get("clusters", []) or []:
                            rep = c.get("representative_idx")
                            if rep is not None:
                                try:
                                    frames.append(int(rep))
                                    cid = str(c.get("cluster_id")) if c.get("cluster_id") is not None else None
                                    per_meta.append((cid, cluster_pop.get(str(c.get("cluster_id"))) ))
                                except Exception:
                                    continue
                        order = list(range(len(frames)))
                        try:
                            order = sorted(order, key=lambda i: int(per_meta[i][0]) if per_meta[i][0] is not None else 10**9)
                        except Exception:
                            pass
                        if order:
                            frames = [frames[i] for i in order]
                            per_meta = [per_meta[i] for i in order]
                        if frames:
                            if fmt == "GLYCAM":
                                _write_multimodel_with_connect(
                                    data_dir=data_dir,
                                    out_path=out_path,
                                    frame_indices=frames,
                                    title_header=f"REMARK   GLYCAN MULTI-MODEL PDB  LEVEL {level_name.upper()}  ANOMER {anomer.upper()}",
                                    per_model_meta=per_meta,
                                    target_anomer=anomer,
                                )
                                wrote = True
                            else:
                                import tempfile, os
                                with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
                                    tmp_path = tmp.name
                                try:
                                    _write_multimodel_with_connect(
                                        data_dir=data_dir,
                                        out_path=Path(tmp_path),
                                        frame_indices=frames,
                                        title_header=f"REMARK   GLYCAN MULTI-MODEL PDB  LEVEL {level_name.upper()}  ANOMER {anomer.upper()}",
                                        per_model_meta=per_meta,
                                        target_anomer=anomer,
                                    )
                                    with open(tmp_path, "r", encoding="utf-8") as _fh:
                                        raw = _fh.read()
                                    converted = _convert_pdb_text_naming(raw, fmt)
                                    storage.write_text(out_path, converted)
                                    wrote = True
                                finally:
                                    try:
                                        os.unlink(tmp_path)
                                    except Exception:
                                        pass
                except Exception:
                    wrote = False

            if not wrote:
                # Build from per-cluster files (no align / no CONECT beyond what exists)
                if not storage.exists(src_dir):
                    continue
                listed = storage.list_files(src_dir, "*.pdb")
                files = []
                for p in listed:
                    pp = Path(p)
                    if str(pp).lower().endswith(".pdb"):
                        files.append(pp)
                files = sorted(files, key=_cluster_id_from_name)
                if not files:
                    continue
                with storage.open(out_path, "w") as out_f:
                    out_f.write(
                        f"REMARK   GLYCAN MULTI-MODEL PDB  LEVEL {level_name.upper()}  ANOMER {anomer.upper()}\n"
                    )
                    for i, fpath in enumerate(files, start=1):
                        try:
                            cid_match = re.search(r"cluster_(\d+)", fpath.stem)
                            cid = cid_match.group(1) if cid_match else None
                            pop = cluster_pop.get(str(cid)) if cid else None
                            out_f.write(f"MODEL     {i:4d}\n")
                            if cid is not None:
                                out_f.write(f"REMARK   CLUSTER_ID {cid}\n")
                            if pop is not None:
                                out_f.write(f"REMARK   CLUSTER_POPULATION_PCT {pop}\n")
                            with storage.open(fpath, "r") as in_f:
                                raw = in_f.read()
                            normalized_raw = normalize_anomeric_pdb_text(raw, anomer)
                            converted = _convert_pdb_text_naming(normalized_raw, fmt)
                            for line in converted.splitlines():
                                if line.startswith(("ATOM", "HETATM", "ANISOU", "TER")):
                                    out_f.write(line + "\n")
                            out_f.write("ENDMDL\n")
                        except Exception as e:
                            out_f.write(f"REMARK   SKIPPED {getattr(fpath, 'name', str(fpath))}: {e}\n")
                    out_f.write("END\n")

            # Add GlycoShape remarks at file top (no cluster-specific MODEL injection)
            if anomer_meta:
                meta = anomer_meta.get(anomer, {}) if isinstance(anomer_meta, dict) else {}
                try:
                    _add_pdb_remarks(
                        str(out_path),
                        glytoucan_id=meta.get("glytoucan"),
                        iupac=meta.get("iupac"),
                        cluster_id=None,
                        cluster_population=None,
                    )
                except Exception:
                    pass


# -------------------------
# Anomer detection utilities
# -------------------------
def get_glycan_sequence_from_filename(filename: str) -> list[str]:
    """Extract glycan sequence from filename using regex pattern."""
    pattern = re.compile(r'[A-Za-z0-9]+-?OH?')
    sequence = pattern.findall(filename)
    return sequence


def find_alpha_or_beta_oh_from_filename(filename: str) -> str:
    """Find alpha or beta OH designation from filename."""
    glycan_sequence = get_glycan_sequence_from_filename(filename)
    if not glycan_sequence:
        return "OH not found"
    
    last_residue = glycan_sequence[-1]
    
    if "a1-OH" in last_residue or "a2-OH" in last_residue:
        return "Alpha OH"
    elif "b1-OH" in last_residue or "b2-OH" in last_residue:
        return "Beta OH"
    else:
        return "OH not found"


def get_anomer(glycam_name: str) -> str:
    """
    Determine if the glycam structure is alpha or beta based on GLYCAM naming logic.
    
    This function uses the same logic as flip.is_alpha but returns the actual anomer type.
    
    Parameters:
        glycam_name: GLYCAM sequence name (e.g., "DManpa1-2DManpa1-OH")
    
    Returns:
        "alpha" if the structure is alpha, "beta" if beta, "unknown" if cannot determine
    """
    if find_alpha_or_beta_oh_from_filename(glycam_name) == "Alpha OH":
        return "alpha"
    elif find_alpha_or_beta_oh_from_filename(glycam_name) == "Beta OH":
        return "beta"
    else:
        return "unknown"


def align_structures_by_residues(
    reference_coords: np.ndarray,
    target_coords: np.ndarray,
    reference_atoms: pl.DataFrame,
    target_atoms: pl.DataFrame,
    residue_range: tuple[int, int] = (1, 3),
    use_rdkit: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align two structures by overlaying specified residues.
    
    Parameters:
        reference_coords: Reference structure coordinates (n_atoms, 3)
        target_coords: Target structure coordinates to be aligned (n_atoms, 3)
        reference_atoms: Reference atoms DataFrame with res_id and name columns
        target_atoms: Target atoms DataFrame with res_id and name columns
        residue_range: Tuple of (first_res_id, last_res_id) to use for alignment
        use_rdkit: Whether to use RDKit for alignment (if available)
    
    Returns:
        Tuple of (aligned_target_coords, transformation_matrix)
        If alignment fails, returns (target_coords, identity_matrix)
    """
    try:
        # Find atoms in the specified residue range for alignment
        min_res, max_res = residue_range
        
        # Get reference alignment atoms
        ref_res_ids = reference_atoms["res_id"].to_list()
        ref_names = reference_atoms["name"].to_list()
        ref_align_indices = []
        ref_align_names = []
        
        for i, (res_id, name) in enumerate(zip(ref_res_ids, ref_names)):
            if min_res <= res_id <= max_res:
                ref_align_indices.append(i)
                ref_align_names.append(f"{name}_{res_id}")
        
        # Get target alignment atoms (matching by name and residue)
        tgt_res_ids = target_atoms["res_id"].to_list()
        tgt_names = target_atoms["name"].to_list()
        tgt_align_indices = []
        
        for ref_name in ref_align_names:
            name, res_id_str = ref_name.rsplit('_', 1)
            res_id = int(res_id_str)
            
            for i, (t_res_id, t_name) in enumerate(zip(tgt_res_ids, tgt_names)):
                if t_res_id == res_id and t_name == name:
                    tgt_align_indices.append(i)
                    break
        
        if len(ref_align_indices) < 3 or len(tgt_align_indices) < 3:
            print(f"Warning: Not enough matching atoms for alignment (ref: {len(ref_align_indices)}, tgt: {len(tgt_align_indices)})")
            return target_coords.copy(), np.eye(4)
        
        if len(ref_align_indices) != len(tgt_align_indices):
            # Take common subset
            common_len = min(len(ref_align_indices), len(tgt_align_indices))
            ref_align_indices = ref_align_indices[:common_len]
            tgt_align_indices = tgt_align_indices[:common_len]
        
        if use_rdkit:
            try:
                # Use RDKit alignment if available
                from rdkit import Chem  # type: ignore
                from rdkit.Chem import AllChem, rdMolAlign  # type: ignore
                
                # Create temporary RDKit molecules for alignment
                ref_mol = Chem.RWMol()
                tgt_mol = Chem.RWMol()
                
                # Add atoms (simplified - just use Carbon for alignment points)
                for _ in ref_align_indices:
                    ref_mol.AddAtom(Chem.Atom(6))  # Carbon
                for _ in tgt_align_indices:
                    tgt_mol.AddAtom(Chem.Atom(6))  # Carbon
                
                ref_mol = ref_mol.GetMol()
                tgt_mol = tgt_mol.GetMol()
                
                # Create conformers
                ref_conf = Chem.Conformer(len(ref_align_indices))
                tgt_conf = Chem.Conformer(len(tgt_align_indices))
                
                for i, idx in enumerate(ref_align_indices):
                    x, y, z = reference_coords[idx]
                    ref_conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(float(x), float(y), float(z)))
                
                for i, idx in enumerate(tgt_align_indices):
                    x, y, z = target_coords[idx]
                    tgt_conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(float(x), float(y), float(z)))
                
                ref_mol.AddConformer(ref_conf, assignId=True)
                tgt_mol.AddConformer(tgt_conf, assignId=True)
                
                # Perform alignment (mutates tgt_mol in place)
                atom_map = [(i, i) for i in range(len(ref_align_indices))]
                rmsd = rdMolAlign.AlignMol(tgt_mol, ref_mol, atomMap=atom_map)

                # Extract the rigid transform actually applied to the alignment
                # subset by reading back the (now-moved) tgt conformer and
                # comparing against the original subset of target_coords. This
                # avoids the prior bug where the transform was discarded and
                # unaligned target coordinates were returned.
                post_tgt = np.asarray(
                    [
                        [
                            tgt_mol.GetConformer().GetAtomPosition(i).x,
                            tgt_mol.GetConformer().GetAtomPosition(i).y,
                            tgt_mol.GetConformer().GetAtomPosition(i).z,
                        ]
                        for i in range(len(tgt_align_indices))
                    ],
                    dtype=float,
                )
                pre_tgt = np.asarray(target_coords[tgt_align_indices], dtype=float)
                # Recover R, t from the moved subset via Kabsch; this is
                # numerically equivalent to RDKit's transform on this subset.
                pre_centroid = pre_tgt.mean(axis=0)
                post_centroid = post_tgt.mean(axis=0)
                H = (pre_tgt - pre_centroid).T @ (post_tgt - post_centroid)
                U, S, Vt = np.linalg.svd(H)
                R = Vt.T @ U.T
                if np.linalg.det(R) < 0:
                    Vt[-1, :] *= -1
                    R = Vt.T @ U.T
                t = post_centroid - R @ pre_centroid

                # Apply the recovered transform to the full target_coords
                aligned_coords = (R @ target_coords.T).T + t
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = R
                transform_matrix[:3, 3] = t
                return aligned_coords, transform_matrix

            except ImportError:
                use_rdkit = False
            except Exception as e:
                print(f"RDKit alignment failed: {e}, falling back to Kabsch algorithm")
                use_rdkit = False
        
        if not use_rdkit:
            # Use Kabsch algorithm for alignment
            ref_points = reference_coords[ref_align_indices]
            tgt_points = target_coords[tgt_align_indices]
            
            # Center the points
            ref_centroid = np.mean(ref_points, axis=0)
            tgt_centroid = np.mean(tgt_points, axis=0)
            
            ref_centered = ref_points - ref_centroid
            tgt_centered = tgt_points - tgt_centroid
            
            # Compute optimal rotation matrix using SVD
            H = tgt_centered.T @ ref_centered
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            # Ensure proper rotation (det(R) = 1)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            # Compute translation
            t = ref_centroid - R @ tgt_centroid
            
            # Apply transformation to all coordinates
            aligned_coords = (R @ target_coords.T).T + t
            
            # Build transformation matrix
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = R
            transform_matrix[:3, 3] = t
            
            return aligned_coords, transform_matrix
    
    except Exception as e:
        print(f"Alignment failed: {e}")
        return target_coords.copy(), np.eye(4)
    
    return target_coords.copy(), np.eye(4)


def align_frame_to_reference(
    frame_data_dir: str,
    reference_frame_idx: int,
    target_frame_idx: int,
    residue_range: tuple[int, int] = (1, 3),
    use_rdkit: bool = True
) -> np.ndarray:
    """
    Align a target frame to a reference frame based on specified residues.
    
    Parameters:
        frame_data_dir: Directory containing trajectory data
        reference_frame_idx: Index of the reference frame
        target_frame_idx: Index of the target frame to align
        residue_range: Tuple of (first_res_id, last_res_id) for alignment
        use_rdkit: Whether to use RDKit for alignment
    
    Returns:
        Aligned coordinates for the target frame (n_atoms, 3)
    """
    atoms = load_atoms(frame_data_dir)
    ref_coords = load_frame_xyz(frame_data_dir, reference_frame_idx)
    tgt_coords = load_frame_xyz(frame_data_dir, target_frame_idx)
    
    aligned_coords, _ = align_structures_by_residues(
        ref_coords, tgt_coords, atoms, atoms, residue_range, use_rdkit
    )
    
    return aligned_coords


def write_aligned_pdb_for_frame(
    frame_data_dir: str,
    frame_idx: int,
    reference_frame_idx: int,
    path: str,
    title: Optional[str] = None,
    flip: bool = False,
    residue_range: tuple[int, int] = (1, 5),
    use_rdkit: bool = True,
) -> None:
    """Write a PDB for one frame aligned to a reference frame.

    Parameters:
        frame_data_dir: directory containing atoms.parquet, bonds.parquet and coords.zarr
        frame_idx: zero-based frame index to write
        reference_frame_idx: zero-based reference frame index for alignment
        path: output PDB path
        title: optional TITLE record
        flip: if True, perform alpha/beta OH flip
        residue_range: tuple of (first_res_id, last_res_id) for alignment
        use_rdkit: whether to use RDKit for alignment
    """
    atoms = load_atoms(frame_data_dir)
    bonds = load_bonds(frame_data_dir)
    res_names = atoms["res_name"].to_list()
    
    # Get aligned coordinates
    if frame_idx == reference_frame_idx:
        # No need to align if it's the same frame
        xyz = load_frame_xyz(frame_data_dir, frame_idx).copy()
    else:
        xyz = align_frame_to_reference(
            frame_data_dir, reference_frame_idx, frame_idx, residue_range, use_rdkit
        )

    if flip:
        # Apply flip logic (same as in write_pdb_for_frame)
        try:
            names = atoms["name"].to_list()
            res_ids = atoms["res_id"].to_list()
            # The output is the opposite anomer; keep its residue metadata
            # consistent even if the coordinate flip falls back.
            res_names = flip_anomeric_residue_names(res_names, res_ids)

            idx_C1 = next((i for i,(n,r) in enumerate(zip(names, res_ids)) if n == "C1" and r == 2), None)
            idx_H1 = next((i for i,(n,r) in enumerate(zip(names, res_ids)) if n == "H1" and r == 2), None)
            idx_O1 = next((i for i,(n,r) in enumerate(zip(names, res_ids)) if n == "O1" and r == 1), None)
            idx_HO1 = next((i for i,(n,r) in enumerate(zip(names, res_ids)) if n == "HO1" and r == 1), None)

            if None not in (idx_C1, idx_H1, idx_O1, idx_HO1):
                C1 = xyz[idx_C1]
                H1 = xyz[idx_H1]
                O1 = xyz[idx_O1]
                HO1 = xyz[idx_HO1]

                CO1_vec = O1 - C1
                CH1_vec = H1 - C1
                HO1O1_vec = HO1 - O1

                # Existing bond lengths
                len_C1O1 = float(np.linalg.norm(CO1_vec)) if np.linalg.norm(CO1_vec) > 0 else 1.43
                len_C1H1 = float(np.linalg.norm(CH1_vec)) if np.linalg.norm(CH1_vec) > 0 else 1.09

                # New coordinates using normalized swapped directions & preserved lengths
                if np.linalg.norm(CH1_vec) > 0 and np.linalg.norm(CO1_vec) > 0:
                    new_O1 = C1 + (CH1_vec / np.linalg.norm(CH1_vec)) * len_C1O1
                    new_H1 = C1 + (CO1_vec / np.linalg.norm(CO1_vec)) * len_C1H1
                    new_HO1 = new_O1 + HO1O1_vec  # preserve relative vector

                    xyz[idx_O1] = new_O1
                    xyz[idx_H1] = new_H1
                    xyz[idx_HO1] = new_HO1

        except Exception:
            pass  # silent fallback

    # Write PDB with the aligned coordinates
    serials = atoms["serial"].to_numpy()
    names = atoms["name"].to_list()
    resn = res_names
    chain_u8 = atoms["chain_id_u8"].to_numpy()
    resid = atoms["res_id"].to_numpy()
    elem = atoms["element"].to_list()

    storage = get_storage_manager()
    with storage.open(path, 'w') as out:
        if title:
            out.write(f"TITLE     {title}\n")
        out.write(f"MODEL     {frame_idx+1}\n")

        for i in range(len(serials)):
            x, y, z = xyz[i]
            out.write(_pdb_atom_line(
                serial=int(serials[i]),
                name=names[i],
                res_name=resn[i],
                chain_u8=int(chain_u8[i]),
                res_id=int(resid[i]),
                x=float(x), y=float(y), z=float(z),
                element=elem[i],
            ))

        conect = _pdb_conect_block(serials, bonds, atoms, xyz)
        if conect:
            out.write(conect)

        out.write("ENDMDL\n")


def parse_mol2_bonds(file_path: Path) -> List[Tuple[int, int]]:
    """Parse MOL2 bonds and restore only unambiguous residue-local bonds.
    
    Args:
        file_path: Path to MOL2 file
        
    Returns:
        List of bonded atom pairs. Missing C1--C2 bonds are restored for
        pyranose residues identified by C1, C2, and O5. Disconnected components
        are deliberately left disconnected here: trajectory-aware recovery in
        ``lib.frame_quality`` is the only safe place to infer missing linkages.
    """
    try:
        storage = get_storage_manager()
        with storage.open(file_path, 'r') as file:
            lines = file.readlines()

        # Parse atom coordinates
        atom_start = -1
        atom_end = -1
        for i, line in enumerate(lines):
            if line.strip() == "@<TRIPOS>ATOM":
                atom_start = i + 1
            elif atom_start != -1 and line.strip().startswith("@<TRIPOS>"):
                atom_end = i
                break

        atom_coords = {}
        # MOL2 atom columns 7/8 identify the residue/substructure. Preserve
        # them so we can validate glycan ring topology rather than relying
        # solely on whole-graph connectedness.
        atom_residues = {}
        if atom_start != -1:
            if atom_end == -1:
                atom_end = len(lines)
            
            for line in lines[atom_start:atom_end]:
                try:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        atom_id = int(parts[0])
                        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                        atom_coords[atom_id] = (x, y, z)
                        if len(parts) >= 8:
                            atom_residues[atom_id] = (parts[1], parts[6], parts[7])
                except (ValueError, IndexError):
                    continue

        # Parse bonds
        bond_start = -1
        bond_end = -1
        for i, line in enumerate(lines):
            if line.strip() == "@<TRIPOS>BOND":
                bond_start = i + 1
            elif bond_start != -1 and line.strip().startswith("@<TRIPOS>"):
                bond_end = i
                break

        if bond_start == -1:
            raise ValueError("No @<TRIPOS>BOND section found in the mol2 file.")

        if bond_end == -1:
            bond_end = len(lines)

        bonded_atoms = []
        for line in lines[bond_start:bond_end]:
            try:
                _, atom1, atom2, _ = line.strip().split()
                bonded_atoms.append((int(atom1), int(atom2)))
                
            except ValueError:
                logger.warning(f"Skipping invalid bond line: {line.strip()}")
                continue

        # A missing C1--C2 bond does not split a pyranose residue: the remaining
        # C1--O5--C5--C4--C3--C2 path still forms one connected component. That
        # allowed malformed MOL2 topologies (for example 4YB residue 6 in
        # G45661AN) to pass the older disconnected-component repair unchanged.
        # Restore only the chemically unambiguous ring closure for residues that
        # explicitly contain C1, C2, and O5.
        residue_atoms = {}
        for atom_id, (atom_name, residue_id, residue_name) in atom_residues.items():
            residue_atoms.setdefault((residue_id, residue_name), {})[atom_name] = atom_id

        known_bonds = {tuple(sorted((a, b))) for a, b in bonded_atoms}
        for (residue_id, residue_name), names in residue_atoms.items():
            if not {"C1", "C2", "O5"}.issubset(names):
                continue
            ring_closure = tuple(sorted((names["C1"], names["C2"])))
            if ring_closure not in known_bonds:
                bonded_atoms.append(ring_closure)
                known_bonds.add(ring_closure)
                logger.warning(
                    "Added missing pyranose ring-closure bond %s-%s for residue %s %s",
                    ring_closure[0], ring_closure[1], residue_name, residue_id,
                )

        return bonded_atoms
    except Exception as e:
        logger.error(f"Failed to parse MOL2 file {file_path}: {str(e)}")
        raise
