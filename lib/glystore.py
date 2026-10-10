# glystore.py
# Dependencies: numpy, polars, zarr, numcodecs
from __future__ import annotations
import os
import io
from typing import Callable, Iterable, Tuple, Optional, List

import numpy as np
import polars as pl
import zarr
from lib.storage import get_storage_manager
from numcodecs import Blosc

# -------------------------
# Schema helpers
# -------------------------
def _build_atoms_df(
    serial: np.ndarray,             # int32
    name: List[str],                # str list
    element: List[str],             # str list
    res_name: List[str],            # str list
    res_id: np.ndarray,             # int16 preferred
    chain_id_u8: np.ndarray,        # uint8 ASCII of chain id (0 if blank)
) -> pl.DataFrame:
    n = len(serial)
    # ensure dtypes
    serial = serial.astype(np.int32, copy=False)
    res_id = res_id.astype(np.int16, copy=False)
    chain_id_u8 = chain_id_u8.astype(np.uint8, copy=False)

    df = pl.DataFrame({
        "atom_index": np.arange(n, dtype=np.int32),
        "serial": serial,
        "name": pl.Series(name, dtype=pl.Categorical),
        "element": pl.Series(element, dtype=pl.Categorical),
        "res_name": pl.Series(res_name, dtype=pl.Categorical),
        "res_id": res_id,
        "chain_id_u8": chain_id_u8,  # 0 for blank, else ord('A') etc.
    })
    return df

def _build_bonds_df(pairs_indexed: np.ndarray) -> pl.DataFrame:
    # pairs_indexed shape (m,2), int32 atom_index pairs (a<b)
    if pairs_indexed.size == 0:
        return pl.DataFrame({"a": np.array([], dtype=np.int32), "b": np.array([], dtype=np.int32)})

    a = pairs_indexed[:, 0].astype(np.int32, copy=False)
    b = pairs_indexed[:, 1].astype(np.int32, copy=False)
    # enforce a<b to keep a single undirected edge
    swap_mask = a > b
    if np.any(swap_mask):
        a[swap_mask], b[swap_mask] = b[swap_mask], a[swap_mask]
    # unique edges
    pairs = np.unique(np.stack([a, b], axis=1), axis=0)
    return pl.DataFrame({"a": pairs[:, 0], "b": pairs[:, 1]})

# -------------------------
# Filesystem helpers
# -------------------------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _parquet_write(df: pl.DataFrame, path: str):
    df.write_parquet(path, compression="zstd", statistics=True)

# -------------------------
# Zarr coordinate store
# -------------------------
def _create_coords_store(
    frame_data_dir: str,
    n_atoms: int,
    chunks: Optional[Tuple[int, int, int]] = None,
    frame_chunk: Optional[int] = None,
):
    """Create (or overwrite) a coords.zarr directory store.

    frame_chunk: if provided, overrides chunks[0]. A non-positive value means
      "one chunk for the complete trajectory"; callers that know the trajectory
      length should pass that length. This keeps process data to one coordinate
      chunk instead of thousands of small chunk files.
    """
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    store = zarr.DirectoryStore(os.path.join(frame_data_dir, "coords.zarr"))
    root = zarr.group(store=store, overwrite=True)
    if chunks is None:
        fc = frame_chunk if frame_chunk is not None else 1
        chunks = (fc, n_atoms, 3)
    else:
        if frame_chunk is not None:
            # replace first dimension with requested frame_chunk
            chunks = (frame_chunk, chunks[1], chunks[2])
    z = root.create_dataset(
        "xyz",
        shape=(0, n_atoms, 3),
        chunks=chunks,
        dtype="f4",
        compressor=compressor,
        overwrite=True,
        fill_value=np.nan,
    )
    return root, z

def _append_frame(zarr_arr, xyz_frame: np.ndarray):
    # xyz_frame shape (n_atoms,3), dtype float32
    fcount = zarr_arr.shape[0]
    # zarr.Array.resize expects the new full shape (tuple), not an axis kwarg
    new_shape = list(zarr_arr.shape)
    new_shape[0] = fcount + 1
    zarr_arr.resize(tuple(new_shape))
    zarr_arr[fcount, :, :] = xyz_frame.astype(np.float32, copy=False)

def _append_frames(zarr_arr, xyz_frames: np.ndarray):
    """Append multiple frames in one resize (xyz_frames shape (k, n_atoms, 3))."""
    if xyz_frames.ndim != 3:
        raise ValueError("xyz_frames must have shape (k, n_atoms, 3)")
    if xyz_frames.size == 0:
        return
    k = xyz_frames.shape[0]
    fcount = zarr_arr.shape[0]
    new_shape = list(zarr_arr.shape)
    new_shape[0] = fcount + k
    zarr_arr.resize(tuple(new_shape))
    zarr_arr[fcount:fcount + k, :, :] = xyz_frames.astype(np.float32, copy=False)

# -------------------------
# PDB streaming parser (MODEL/ENDMDL)
# Minimal fields: serial, name, res_name, res_seq (res_id), chain_id, element, x,y,z
# -------------------------
def _parse_pdb_atom_line(line: str):
    # PDB fixed columns (1-based):
    # record[1:6], serial[7:11], name[13:16], altLoc[17], resName[18:20],
    # chainID[22], resSeq[23:26], iCode[27], x[31:38], y[39:46], z[47:54], element[77:78]
    serial = int(line[6:11])
    name = line[12:16].strip()
    res_name = line[17:20].strip()
    chain_id = line[21:22]
    chain_u8 = ord(chain_id) if chain_id and chain_id != " " else 0
    # resSeq can be negative or have spaces; handle robustly
    res_seq = line[22:26].strip()
    res_id = int(res_seq) if res_seq else 0
    x = float(line[30:38])
    y = float(line[38:46])
    z = float(line[46:54])
    element = line[76:78].strip()
    if not element:
        # try to derive from name
        element = "".join([c for c in name if c.isalpha()])[:1].upper()
    return serial, name, element, res_name, res_id, chain_u8, x, y, z

def _iter_models(pdb_path: str):
    """Yield lists of ATOM/HETATM lines per MODEL. If no MODEL/ENDMDL found, yield one model of all atoms."""
    has_model = False
    current: List[str] = []
    storage = get_storage_manager()
    with storage.open(pdb_path, "r") as fh:
        for line in fh:
            rec = line[:6]
            if rec.startswith("MODEL"):
                has_model = True
                current = []
            elif rec.startswith("ENDMDL"):
                if current:
                    yield current
                current = []
            elif rec.startswith("ATOM  ") or rec.startswith("HETATM"):
                current.append(line)
        # file end
        if not has_model:
            # treat entire file as single model
            if current:
                yield current


def _count_pdb_frames(pdb_path: str) -> int:
    """Count trajectory models without retaining their coordinates."""
    storage = get_storage_manager()
    model_count = 0
    saw_model = False
    saw_atoms = False
    with storage.open(pdb_path, "r") as fh:
        for line in fh:
            rec = line[:6]
            if rec.startswith("MODEL"):
                saw_model = True
                model_count += 1
            elif rec.startswith("ATOM  ") or rec.startswith("HETATM"):
                saw_atoms = True
    if saw_model:
        return max(1, model_count)
    return 1 if saw_atoms else 0

# -------------------------
# Public API
# -------------------------
def store_from_pdb(
    pdb_path: str,
    frame_data_dir: str,
    get_connectivity: Optional[Callable[[pl.DataFrame], Iterable[Tuple[int, int]]]] = None,
    connectivity_kind: str = "serial",  # "serial" or "index"
    frames_buffer: int = 1,              # >=1; collect this many frames before writing
    frame_chunk: Optional[int] = None,   # optional chunk size along frame dim
) -> None:
    """
    Stream-parse multi-frame PDB, write:
      - atoms.parquet (topology)
      - bonds.parquet (connectivity from get_connectivity)
      - coords.zarr/xyz (float32)
    """
    _ensure_dir(frame_data_dir)

    frame_iter = _iter_models(pdb_path)

    # Parse first model to construct topology
    try:
        first_model = next(frame_iter)
    except StopIteration:
        raise ValueError("No ATOM/HETATM records found in PDB.")

    # First model: collect topology arrays and xyz
    serials: List[int] = []
    names: List[str] = []
    elements: List[str] = []
    res_names: List[str] = []
    res_ids: List[int] = []
    chain_u8s: List[int] = []
    xyz_list: List[Tuple[float, float, float]] = []

    for L in first_model:
        s, n, e, rn, rid, cu8, x, y, z = _parse_pdb_atom_line(L)
        serials.append(s)
        names.append(n)
        elements.append(e)
        res_names.append(rn)
        res_ids.append(rid)
        chain_u8s.append(cu8)
        xyz_list.append((x, y, z))

    serial_np = np.asarray(serials, dtype=np.int32)
    name_ls = names
    element_ls = elements
    res_name_ls = res_names
    res_id_np = np.asarray(res_ids, dtype=np.int16)
    chain_u8_np = np.asarray(chain_u8s, dtype=np.uint8)
    xyz0 = np.asarray(xyz_list, dtype=np.float32)
    n_atoms = xyz0.shape[0]

    # Make atoms dataframe and save
    atoms_df = _build_atoms_df(
        serial=serial_np,
        name=name_ls,
        element=element_ls,
        res_name=res_name_ls,
        res_id=res_id_np,
        chain_id_u8=chain_u8_np,
    )
    _parquet_write(atoms_df, os.path.join(frame_data_dir, "atoms.parquet"))

    # Build connectivity (once)
    if get_connectivity is not None:
        pairs_in = get_connectivity
        pairs_np = np.asarray(pairs_in, dtype=np.int64)  # allow big serials
        if pairs_np.size == 0:
            bonds_df = _build_bonds_df(np.empty((0, 2), dtype=np.int32))
        else:
            if connectivity_kind == "serial":
                # map serial -> atom_index
                serial_to_index = {int(s): i for i, s in enumerate(serial_np.tolist())}
                missing_serials = sorted(
                    {
                        int(serial)
                        for serial in pairs_np.ravel()
                        if int(serial) not in serial_to_index
                    }
                )
                if missing_serials:
                    raise ValueError(
                        "Connectivity provenance failure: MOL2 bond atom IDs do not "
                        "match any PDB atom serial. Missing serial(s): "
                        f"{missing_serials[:10]}"
                        + (" ..." if len(missing_serials) > 10 else "")
                    )
                idx_pairs = np.zeros_like(pairs_np, dtype=np.int32)
                for i in range(pairs_np.shape[0]):
                    a_s, b_s = int(pairs_np[i, 0]), int(pairs_np[i, 1])
                    idx_pairs[i, 0] = serial_to_index[a_s]
                    idx_pairs[i, 1] = serial_to_index[b_s]
                bonds_df = _build_bonds_df(idx_pairs)

                # Check that serial-to-index conversion retained every unique,
                # non-directional input bond before persisting it.
                expected = {tuple(sorted((int(a), int(b)))) for a, b in pairs_np}
                actual = {
                    tuple(sorted((int(serial_np[a]), int(serial_np[b]))))
                    for a, b in bonds_df.iter_rows()
                }
                if expected != actual:
                    missing = sorted(expected - actual)
                    unexpected = sorted(actual - expected)
                    raise ValueError(
                        "Connectivity provenance failure while storing bonds; "
                        f"missing={missing[:5]}, unexpected={unexpected[:5]}"
                    )
            elif connectivity_kind == "index":
                bonds_df = _build_bonds_df(pairs_np.astype(np.int32, copy=False))
            else:
                raise ValueError("connectivity_kind must be 'serial' or 'index'")
    else:
        bonds_df = _build_bonds_df(np.empty((0, 2), dtype=np.int32))

    _parquet_write(bonds_df, os.path.join(frame_data_dir, "bonds.parquet"))

    # Create coords store. A non-positive/unspecified frame chunk requests a
    # single chunk spanning the trajectory, reducing filesystem metadata files.
    if frame_chunk is None or frame_chunk <= 0:
        frame_chunk = _count_pdb_frames(pdb_path)
        if frame_chunk < 1:
            raise ValueError("No trajectory frames found in PDB")
    _, z = _create_coords_store(frame_data_dir, n_atoms=n_atoms, frame_chunk=frame_chunk)

    # Buffered append logic
    buffer: List[np.ndarray] = [xyz0]
    frame_count = 1
    if frames_buffer < 1:
        raise ValueError("frames_buffer must be >=1")
    for model in frame_iter:
        if len(model) != n_atoms:
            raise ValueError(f"Frame {frame_count}: atom count changed ({len(model)} vs {n_atoms}).")
        xyz_list = [_parse_pdb_atom_line(L)[6:9] for L in model]  # x,y,z only
        xyz = np.asarray(xyz_list, dtype=np.float32)
        buffer.append(xyz)
        frame_count += 1
        if len(buffer) >= frames_buffer:
            _append_frames(z, np.stack(buffer, axis=0))
            buffer.clear()
    # flush remainder
    if buffer:
        _append_frames(z, np.stack(buffer, axis=0))

    # Write a tiny frames.parquet with just count (handy for sanity checks)
    pl.DataFrame({"n_frames": [frame_count]}).write_parquet(
        os.path.join(frame_data_dir, "frames.parquet"),
        compression="zstd",
        statistics=True,
    )

# -------------------------
# Convenience: store from memory (coords already available)
# -------------------------
def store_from_memory(
    frame_data_dir: str,
    atoms: dict,
    coords: np.ndarray,  # shape (n_frames, n_atoms, 3), float32
    connectivity_pairs: Optional[Iterable[Tuple[int, int]]] = None,
    connectivity_kind: str = "serial",
    frame_chunk: Optional[int] = None,
):
    """
    Directly store given topology/coords, bypassing PDB parsing.
    'atoms' keys: serial(List[int]|np.ndarray), name(List[str]),
                  element(List[str]), res_name(List[str]),
                  res_id(np.ndarray int16), chain_id_u8(np.ndarray uint8)
    """
    _ensure_dir(frame_data_dir)
    atoms_df = _build_atoms_df(
        serial=np.asarray(atoms["serial"], dtype=np.int32),
        name=list(atoms["name"]),
        element=list(atoms["element"]),
        res_name=list(atoms["res_name"]),
        res_id=np.asarray(atoms["res_id"], dtype=np.int16),
        chain_id_u8=np.asarray(atoms["chain_id_u8"], dtype=np.uint8),
    )
    _parquet_write(atoms_df, os.path.join(frame_data_dir, "atoms.parquet"))

    if connectivity_pairs is not None:
        pairs_np = np.asarray(list(connectivity_pairs), dtype=np.int64)
        if connectivity_kind == "serial":
            serial_np = atoms_df["serial"].to_numpy()
            s2i = {int(s): i for i, s in enumerate(serial_np.tolist())}
            idx_pairs = np.zeros_like(pairs_np, dtype=np.int32)
            for i in range(pairs_np.shape[0]):
                idx_pairs[i, 0] = s2i[int(pairs_np[i, 0])]
                idx_pairs[i, 1] = s2i[int(pairs_np[i, 1])]
            bonds_df = _build_bonds_df(idx_pairs)
        else:
            bonds_df = _build_bonds_df(pairs_np.astype(np.int32, copy=False))
    else:
        bonds_df = _build_bonds_df(np.empty((0, 2), dtype=np.int32))
    _parquet_write(bonds_df, os.path.join(frame_data_dir, "bonds.parquet"))

    n_frames, n_atoms, _ = coords.shape
    if frame_chunk is None or frame_chunk <= 0:
        frame_chunk = int(n_frames)
    _, z = _create_coords_store(frame_data_dir, n_atoms=n_atoms, frame_chunk=frame_chunk)
    # single resize append all frames
    _append_frames(z, coords.astype(np.float32, copy=False))

    pl.DataFrame({"n_frames": [n_frames]}).write_parquet(
        os.path.join(frame_data_dir, "frames.parquet"),
        compression="zstd",
        statistics=True,
    )

# -------------------------
# Archive entries (lib/archive.py): trajectory.xtc + topology.pdb
# -------------------------
def store_from_archive(
    entry_path: str,
    frame_data_dir: str,
    get_connectivity: Optional[Callable[[pl.DataFrame], Iterable[Tuple[int, int]]]] = None,
    connectivity_kind: str = "serial",
    frames_buffer: int = 1,
    frame_chunk: Optional[int] = None,
) -> None:
    """
    Same outputs as store_from_pdb for an archive entry (its folder, or its
    trajectory.xtc). Frames are streamed from the XTC; nothing is read as text
    except topology.pdb, the first frame of the source.
    """
    from lib.archive import ArchiveEntry

    if frames_buffer < 1:
        raise ValueError("frames_buffer must be >=1")
    entry = ArchiveEntry(entry_path)
    n_frames = entry.n_frames
    if frame_chunk is None or frame_chunk <= 0:
        frame_chunk = n_frames

    # atoms.parquet and bonds.parquet: the PDB code path on the first frame.
    store_from_pdb(
        str(entry.topology_path),
        frame_data_dir,
        get_connectivity=get_connectivity,
        connectivity_kind=connectivity_kind,
        frames_buffer=1,
        frame_chunk=1,
    )

    # Coordinates: every frame from the XTC, appended in whole Zarr chunks so
    # that no chunk is decompressed and rewritten.
    _, z = _create_coords_store(frame_data_dir, n_atoms=entry.n_atoms, frame_chunk=frame_chunk)
    pending: List[np.ndarray] = []
    pending_frames = 0
    for block in entry.iter_frames(chunk=max(int(frames_buffer), 1024)):
        pending.append(block)
        pending_frames += block.shape[0]
        if pending_frames >= frame_chunk:
            ready = np.concatenate(pending, axis=0) if len(pending) > 1 else pending[0]
            whole = (pending_frames // frame_chunk) * frame_chunk
            _append_frames(z, ready[:whole])
            pending = [ready[whole:]] if whole < pending_frames else []
            pending_frames -= whole
    if pending_frames:
        _append_frames(z, np.concatenate(pending, axis=0))
    if z.shape[0] != n_frames:
        raise ValueError(f"Archive entry yielded {z.shape[0]} frames, manifest says {n_frames}.")

    pl.DataFrame({"n_frames": [n_frames]}).write_parquet(
        os.path.join(frame_data_dir, "frames.parquet"),
        compression="zstd",
        statistics=True,
    )


def store_trajectory(trajectory_path: str, frame_data_dir: str, **kwargs) -> None:
    """Store step for either input: an archive entry or a multi-frame PDB."""
    from lib import archive

    path = str(trajectory_path)
    entry_dir = os.path.dirname(path) if os.path.basename(path) == archive.TRAJECTORY_FILE else path
    if archive.is_archive_entry(entry_dir):
        store_from_archive(entry_dir, frame_data_dir, **kwargs)
    else:
        store_from_pdb(path, frame_data_dir, **kwargs)
