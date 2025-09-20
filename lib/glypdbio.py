# glypdbio.py
from __future__ import annotations
import os
import io
import re
from typing import Optional


import numpy as np
import networkx as nx
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
    if not os.path.exists(path):
        raise FileNotFoundError(path)
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

def load_bonds(out_dir: str) -> pl.DataFrame:
    path = os.path.join(out_dir, "bonds.parquet")
    if not os.path.exists(path):
        return pl.DataFrame({"a": [], "b": []})
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

def load_coords(out_dir: str):
    root = zarr.open(os.path.join(out_dir, "coords.zarr"), mode="r")
    return root["xyz"]  # zarr array: shape (n_frames, n_atoms, 3), float32

def load_frame_xyz(out_dir: str, frame_idx: int) -> np.ndarray:
    z = load_coords(out_dir)
    if frame_idx < 0 or frame_idx >= z.shape[0]:
        raise IndexError(f"frame_idx {frame_idx} out of range [0, {z.shape[0]-1}]")
    return z[frame_idx, :, :]  # (n_atoms, 3)


# -------------------------
# Distance utilities
# -------------------------
def _lower_triangle_flat_distances(xyz: np.ndarray) -> np.ndarray:
    """Compute pairwise distances for coordinates and return the flattened lower-triangle (i>j).

    Returns an array of shape (n*(n-1)/2,) with distances for all unique unordered atom pairs.
    """
    if xyz is None:
        return np.array([], dtype=float)
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must be an (n_atoms, 3) array")
    n = xyz.shape[0]
    if n < 2:
        return np.array([], dtype=float)
    # pairwise distances: using broadcasting
    diffs = xyz[:, None, :] - xyz[None, :, :]
    dists = np.sqrt((diffs * diffs).sum(axis=-1))
    i, j = np.tril_indices(n, k=-1)
    return dists[i, j]

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
    """
    atoms = load_atoms(out_dir)
    # element column expected; be robust to missing column
    if "element" not in atoms.columns:
        # no element information available -> fall back to full distance array
        return get_distance_array(out_dir, frame_idx)
    elems = [str(x).strip().upper() for x in atoms["element"].to_list()]
    non_h_mask = np.array([not (e.startswith("H")) for e in elems], dtype=bool)
    indices = np.nonzero(non_h_mask)[0]
    if indices.size < 2:
        return np.array([], dtype=float)
    xyz_all = load_frame_xyz(out_dir, frame_idx)
    xyz = xyz_all[indices, :]
    return _lower_triangle_flat_distances(xyz)


def get_distance_array_noH_fast(out_dir: str, frame_idx: int, sample_size: int = 200, seed: int = 0) -> np.ndarray:
    """Return flattened lower-triangle distances for non-H atoms, but if the number
    of non-H atoms is greater than `sample_size` randomly sample `sample_size`
    atoms (without replacement) using a fixed `seed` for reproducibility.

    - out_dir: directory containing atoms.parquet and coords.zarr
    - frame_idx: zero-based frame index
    - sample_size: maximum number of non-H atoms to include (default 200)
    - seed: RNG seed for reproducible sampling
    """
    atoms = load_atoms(out_dir)
    if "element" not in atoms.columns:
        # fallback to full distances if element info missing
        return get_distance_array(out_dir, frame_idx)

    elems = [str(x).strip().upper() for x in atoms["element"].to_list()]
    non_h_mask = np.array([not (e.startswith("H")) for e in elems], dtype=bool)
    indices = np.nonzero(non_h_mask)[0]
    if indices.size < 2:
        return np.array([], dtype=float)

    # If small enough, compute full distances for all non-H atoms
    if indices.size <= int(sample_size):
        xyz_all = load_frame_xyz(out_dir, frame_idx)
        xyz = xyz_all[indices, :]
        return _lower_triangle_flat_distances(xyz)

    # Otherwise, sample `sample_size` indices reproducibly and compute distances
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(indices, size=int(sample_size), replace=False)
    xyz_all = load_frame_xyz(out_dir, frame_idx)
    xyz = xyz_all[chosen, :]
    return _lower_triangle_flat_distances(xyz)


def get_conformation_landscape(
    out_dir: str,
    method: str = "noH_fast",
    sample_size: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """Fast conformational landscape matrix.

    Returns a 2D array with shape (n_frames, n_distances) where each row is the
    flattened lower-triangle distance vector for a *fixed* subset of (at most
    `sample_size`) non-hydrogen atoms. This is an optimized version that:

    - Loads atom table once.
    - Selects / samples non-H atom indices once (reproducibly via `seed`).
    - Extracts the coordinates subset across all frames in one slice.
    - Computes all pairwise distances for every frame in a single vectorized
      operation avoiding Python loops per frame.

    Parameters:
        out_dir: Directory containing atoms.parquet and coords.zarr.
        method: (kept for backward compatibility; any value other than the
                default 'noH_fast' is ignored and a ValueError is raised to
                prevent silent misuse.)
        sample_size: Maximum number of non-H atoms to include (default 200).
        seed: RNG seed for reproducible sampling.

    Notes:
        - If <2 eligible atoms, returns an array of shape (n_frames, 0).
        - Always uses the "noH_fast" strategy internally now for speed.
    """
    if method != "noH_fast":
        raise ValueError("get_conformation_landscape now only supports method='noH_fast' for speed.")

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
    if "element" not in atoms.columns:
        # Fall back to all atoms; still sample if needed
        all_indices = np.arange(z.shape[1], dtype=int)
    else:
        elems = [str(x).strip().upper() for x in atoms["element"].to_list()]
        non_h_mask = np.array([not e.startswith("H") for e in elems], dtype=bool)
        all_indices = np.nonzero(non_h_mask)[0]

    if all_indices.size < 2:
        return np.zeros((n_frames, 0), dtype=float)

    if all_indices.size > int(sample_size):
        rng = np.random.default_rng(int(seed))
        chosen = rng.choice(all_indices, size=int(sample_size), replace=False)
        chosen.sort()  # keep deterministic ordering for reproducibility of pair index mapping
    else:
        chosen = all_indices

    m = chosen.size
    # Number of pair distances per frame
    n_pairs = m * (m - 1) // 2
    if n_pairs == 0:
        return np.zeros((n_frames, 0), dtype=float)

    # Gather coordinates subset: shape (n_frames, m, 3)
    # zarr supports advanced indexing with a list/ndarray of indices.
    coords_sub = z[:, chosen, :]
    # ensure numpy ndarray and use float32 to reduce memory
    coords_sub = np.asarray(coords_sub, dtype=np.float32)

    # Precompute pair indices once (lower triangle i>j)
    i_idx, j_idx = np.tril_indices(m, k=-1)
    # Vectorized distance computation across frames:
    # coords_sub[:, i_idx, :] -> (n_frames, n_pairs, 3)
    # compute pairwise diffs across frames; keep float32 arithmetic
    diffs = coords_sub[:, i_idx, :] - coords_sub[:, j_idx, :]
    # distances shape (n_frames, n_pairs) computed in float32
    dist_mat = np.sqrt(np.sum(diffs * diffs, axis=-1)).astype(np.float32)
    # Guarantee C-contiguous output
    return np.ascontiguousarray(dist_mat)

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

def _pdb_conect_block(serials_np: np.ndarray, bonds_df: pl.DataFrame) -> str:
    if bonds_df.is_empty():
        return ""
    pairs = np.c_[bonds_df["a"].to_numpy(), bonds_df["b"].to_numpy()]
    # map atom_index -> serial
    s = serials_np
    n_atoms = s.shape[0]
    out = io.StringIO()
    # skip any pair with indices outside [0, n_atoms-1]
    valid_mask = (pairs[:, 0] >= 0) & (pairs[:, 0] < n_atoms) & (pairs[:, 1] >= 0) & (pairs[:, 1] < n_atoms)
    if not np.all(valid_mask):
        # emit a diagnostic to stderr but continue
        import sys

        bad = np.nonzero(~valid_mask)[0]
        sys.stderr.write(f"Warning: {_pdb_conect_block.__name__}: skipping {bad.size} out-of-range bond(s)\n")
        pairs = pairs[valid_mask]

    for i in range(pairs.shape[0]):
        a = int(s[pairs[i, 0]])
        b = int(s[pairs[i, 1]])
        out.write(f"CONECT{a:5d}{b:5d}\n")
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

    conect = _pdb_conect_block(serials, bonds)
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

    if flip:
        # Locate indices
        names = atoms["name"].to_list()
        res_ids = atoms["res_id"].to_list()
        res_names = atoms["res_name"].to_list()

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

                # Toggle residue name A<->B for residue id 2 if last char matches
                # Find all atoms with res_id==2 and update their res_name consistently
                try:
                    res2_name = next(res_names[i] for i,r in enumerate(res_ids) if r == 2)
                    if res2_name and res2_name[-1] in ("A","B"):
                        flipped_res2 = res2_name[:-1] + ("B" if res2_name[-1] == "A" else "A")
                        # Apply change to local list (used later when writing)
                        for i,r in enumerate(res_ids):
                            if r == 2:
                                res_names[i] = flipped_res2
                except StopIteration:
                    pass
        # else: silently ignore if atoms missing (fallback to original coordinates)

    # Prepare for writing (reuse existing helper formatting routines)
    serials = atoms["serial"].to_numpy()
    names = atoms["name"].to_list()
    # res_names maybe updated if flip performed
    try:
        res_names  # type: ignore
    except NameError:  # flip False path
        res_names = atoms["res_name"].to_list()
    chain_u8 = atoms["chain_id_u8"].to_numpy()
    resid = atoms["res_id"].to_numpy()
    elem = atoms["element"].to_list()

    with open(path, "w", newline="\n") as fh:
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
        conect = _pdb_conect_block(serials, bonds)
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

    # Optional flip duplicates the coordinate transformation in write_pdb_for_frame
    if flip:
        try:
            names = atoms["name"].to_list()
            res_ids = atoms["res_id"].to_list()
            res_names = atoms["res_name"].to_list()
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
                    # Toggle residue 2 name A<->B if last char matches
                    try:
                        res2_name = next(res_names[i] for i,r in enumerate(res_ids) if r == 2)
                        if res2_name and res2_name[-1] in ("A","B"):
                            flipped_res2 = res2_name[:-1] + ("B" if res2_name[-1]=="A" else "A")
                            for i,r in enumerate(res_ids):
                                if r == 2:
                                    res_names[i] = flipped_res2
                    except StopIteration:
                        pass
        except Exception:
            pass  # silent fallback

    # Build RDKit molecule
    rw = Chem.RWMol()
    serials = atoms["serial"].to_numpy()
    elem_col = atoms["element"].to_list()
    names_col = atoms["name"].to_list()
    resn_col = atoms["res_name"].to_list()
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
    with open(input_pdb_path, 'r') as file:
        filedata = file.read()
    
    # Apply naming format conversions based on the logic from pdb.py convert_pdbs function
    if naming_format == 'PDB':
        # Convert GLYCAM to PDB naming
        filedata = re.sub(r"\s\wYA", " NDG", filedata)  # GlcNAc alpha
        filedata = re.sub(r"\s\wYB", " NAG", filedata)  # GlcNAc beta
        filedata = re.sub(r"\s\wVA", " A2G", filedata)  # GalNAc alpha
        filedata = re.sub(r"\s\wVB", " NGA", filedata)  # GalNAc beta
        filedata = re.sub(r"\s\wGA", " GLC", filedata)  # Glc alpha
        filedata = re.sub(r"\s\wGB", " BGC", filedata)  # Glc beta
        filedata = re.sub(r"\s\wGL", " NGC", filedata)  # Neu5Gc alpha
        filedata = re.sub(r"\s\wLA", " GLA", filedata)  # Gal alpha
        filedata = re.sub(r"\s\wLB", " GAL", filedata)  # Gal beta
        filedata = re.sub(r"\s\wfA", " FUC", filedata)  # L-Fuc alpha
        filedata = re.sub(r"\s\wfB", " FUL", filedata)  # L-Fuc beta
        filedata = re.sub(r"\s\wMB", " BMA", filedata)  # Man beta
        filedata = re.sub(r"\s\wMA", " MAN", filedata)  # Man alpha
        filedata = re.sub(r"\s\wSA", " SIA", filedata)  # Neu5Ac alpha
        filedata = re.sub(r"\s\wSA", " SLB", filedata)  # Neu5Ac beta
        filedata = re.sub(r"\s\wZA", " GCU", filedata)  # GlcA alpha
        filedata = re.sub(r"\s\wZB", " BDP", filedata)  # GlcA beta
        filedata = re.sub(r"\s\wXA", " XYS", filedata)  # Xyl alpha
        filedata = re.sub(r"\s\wXB", " XYP", filedata)  # Xyl beta
        filedata = re.sub(r"\s\wuA", " IDR", filedata)  # IdoA alpha
        filedata = re.sub(r"\s\whA", " RAM", filedata)  # Rha alpha
        filedata = re.sub(r"\s\whB", " RHM", filedata)  # Rha beta
        filedata = re.sub(r"\s\wRA", " RIB", filedata)  # Rib alpha
        filedata = re.sub(r"\s\wRB", " BDR", filedata)  # Rib beta
        filedata = re.sub(r"\s\wAA", " ARA", filedata)  # Ara alpha
        filedata = re.sub(r"\s\wAB", " ARB", filedata)  # Ara beta
        
    elif naming_format == 'CHARMM':
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
    
    # For GLYCAM format, no conversion needed as input is already GLYCAM
    
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
    with open(filename, 'r+') as f:
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
    residue_range: tuple[int, int] = (1, 5),
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
                
                # Perform alignment
                atom_map = [(i, i) for i in range(len(ref_align_indices))]
                rmsd = rdMolAlign.AlignMol(tgt_mol, ref_mol, atomMap=atom_map)
                
                # Extract transformation matrix
                # Get aligned coordinates
                aligned_conf = tgt_mol.GetConformer()
                aligned_coords = target_coords.copy()
                
                # Apply the same transformation to all target coordinates
                # This is a simplified approach - we'll use the Kabsch algorithm as fallback
                
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
    residue_range: tuple[int, int] = (1, 5),
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
            res_names = atoms["res_name"].to_list()

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

                    # Toggle residue 2 name A<->B if last char matches
                    try:
                        res2_name = next(res_names[i] for i,r in enumerate(res_ids) if r == 2)
                        if res2_name and res2_name[-1] in ("A","B"):
                            flipped_res2 = res2_name[:-1] + ("B" if res2_name[-1]=="A" else "A")
                            for i,r in enumerate(res_ids):
                                if r == 2:
                                    res_names[i] = flipped_res2
                    except StopIteration:
                        pass
        except Exception:
            pass  # silent fallback

    # Write PDB with the aligned coordinates
    serials = atoms["serial"].to_numpy()
    names = atoms["name"].to_list()
    resn = atoms["res_name"].to_list()
    chain_u8 = atoms["chain_id_u8"].to_numpy()
    resid = atoms["res_id"].to_numpy()
    elem = atoms["element"].to_list()

    with open(path, 'w') as out:
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

        conect = _pdb_conect_block(serials, bonds)
        if conect:
            out.write(conect)

        out.write("ENDMDL\n")


def parse_mol2_bonds(file_path: Path) -> List[Tuple[int, int]]:
    """Parse bond information from MOL2 file and ensure graph connectivity.
    
    Args:
        file_path: Path to MOL2 file
        
    Returns:
        List of tuples containing bonded atom pairs with missing connections added
    """
    try:
        with open(file_path, 'r') as file:
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
        if atom_start != -1:
            if atom_end == -1:
                atom_end = len(lines)
            
            for line in lines[atom_start:atom_end]:
                try:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        atom_id = int(parts[0])
                        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                        atom_coords[atom_id] = (x, y, z)
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
        
        # Check connectivity and add missing bonds if needed
        G = nx.Graph()
        G.add_edges_from(bonded_atoms)
        
        # Find all connected components
        components = list(nx.connected_components(G))
        
        if len(components) > 1:
            logger.warning(f"Graph not fully connected. Found {len(components)} components. Adding missing bonds.")
            
            # Connect components by adding bonds between closest atoms by coordinate distance
            while len(components) > 1:
                min_distance = float('inf')
                best_connection = None
                
                for i in range(len(components)):
                    for j in range(i + 1, len(components)):
                        comp1_nodes = list(components[i])
                        comp2_nodes = list(components[j])
                        
                        # Find closest atoms between components by 3D distance
                        for node1 in comp1_nodes:
                            for node2 in comp2_nodes:
                                if node1 in atom_coords and node2 in atom_coords:
                                    coord1 = atom_coords[node1]
                                    coord2 = atom_coords[node2]
                                    distance = np.sqrt(sum((c1 - c2)**2 for c1, c2 in zip(coord1, coord2)))
                                    if distance < min_distance:
                                        min_distance = distance
                                        best_connection = (node1, node2)
                
                if best_connection:
                    node1, node2 = best_connection
                    bonded_atoms.append((node1, node2))
                    logger.info(f"Added missing bond: {node1} - {node2} (distance: {min_distance:.3f})")
                    
                    # Update the graph with the new bond and recalculate components
                    G.add_edge(node1, node2)
                    components = list(nx.connected_components(G))
                else:
                    logger.warning("Could not find atoms with coordinates to connect components")
                    break

        return bonded_atoms
    except Exception as e:
        logger.error(f"Failed to parse MOL2 file {file_path}: {str(e)}")
        raise