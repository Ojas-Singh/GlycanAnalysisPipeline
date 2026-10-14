"""Covalent-topology recovery and broken-frame filtering.

This module intentionally does not use inter-frame displacement. A frame is
invalid only when coordinates are non-finite or a recovered covalent bond is
longer than the configured threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Iterable

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import zarr
from numcodecs import Blosc


FILTER_VERSION = 1
DEFAULT_MAX_BOND_ANGSTROM = 3.0

_COVALENT_RADII = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "P": 1.07, "S": 1.05, "CL": 1.02, "BR": 1.20, "I": 1.39,
}
_MAX_DEGREE = {
    "H": 1, "C": 4, "N": 4, "O": 2, "F": 1,
    "P": 6, "S": 6, "CL": 1, "BR": 1, "I": 1,
}
_ALLOWED_ELEMENT_PAIRS = {
    frozenset(("H", "C")), frozenset(("H", "N")),
    frozenset(("H", "O")), frozenset(("H", "S")),
    frozenset(("C", "C")), frozenset(("C", "N")),
    frozenset(("C", "O")), frozenset(("C", "S")),
    frozenset(("N", "N")), frozenset(("N", "O")),
    frozenset(("N", "S")), frozenset(("O", "P")),
    frozenset(("O", "S")), frozenset(("P", "S")),
    frozenset(("S", "S")),
}


@dataclass(frozen=True)
class FrameQualityResult:
    source_frames: int
    retained_frames: int
    rejected_frames: int
    input_bonds: int
    retained_input_bonds: int
    inferred_bonds: int
    topology_repaired: bool
    changed: bool


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> bool:
        a, b = self.find(left), self.find(right)
        if a == b:
            return False
        self.parent[b] = a
        return True

    def component_count(self) -> int:
        return len({self.find(i) for i in range(len(self.parent))})


def _normalise_element(value: object) -> str:
    text = str(value).strip().upper()
    if text in _COVALENT_RADII:
        return text
    letters = "".join(char for char in text if char.isalpha())
    if letters[:2] in ("CL", "BR"):
        return letters[:2]
    return letters[:1]


def _chemically_allowed(left: str, right: str) -> bool:
    if not left or not right or left == right == "H":
        return False
    return frozenset((left, right)) in _ALLOWED_ELEMENT_PAIRS


def _covalent_cutoff(left: str, right: str) -> float:
    return _COVALENT_RADII[left] + _COVALENT_RADII[right] + 0.35


def normalise_exclusions(excluded) -> list:
    """``[[first_frame, frames, name], ...]`` sorted, from what a caller gives (None: nothing excluded)."""
    if not excluded:
        return []
    rows = []
    for item in excluded:
        first, count, name = (list(item) + [None, None, None])[:3]
        first, count = int(first), int(count)
        if first < 0 or count < 1:
            raise ValueError(f"an excluded frame range must start at 0 or later and hold frames: {item!r}")
        rows.append([first, count, None if name is None else str(name)])
    rows.sort(key=lambda row: row[0])
    for left, right in zip(rows, rows[1:]):
        if left[0] + left[1] > right[0]:
            raise ValueError(f"excluded frame ranges overlap: {left!r} and {right!r}")
    return rows


def quality_manifest_is_current(
    frame_data_dir: str | Path,
    max_bond_distance: float = DEFAULT_MAX_BOND_ANGSTROM,
    excluded=None,
) -> bool:
    path = Path(frame_data_dir) / "frame_filter.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return (
            int(data.get("filter_version", -1)) == FILTER_VERSION
            and float(data.get("max_bond_distance_A")) == float(max_bond_distance)
            and (data.get("excluded_source_frames") or []) == normalise_exclusions(excluded)
            and (Path(frame_data_dir) / "frame_quality.parquet").exists()
            and (Path(frame_data_dir) / "frame_map.parquet").exists()
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _sanitise_bonds(
    n_atoms: int,
    elements: list[str],
    raw_pairs: Iterable[tuple[int, int]],
) -> tuple[list[tuple[int, int]], int]:
    """Remove impossible edges while retaining valid declared connectivity."""
    candidates: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    raw_count = 0
    for raw_a, raw_b in raw_pairs:
        raw_count += 1
        a, b = int(raw_a), int(raw_b)
        if a > b:
            a, b = b, a
        pair = (a, b)
        if (
            a == b
            or a < 0
            or b >= n_atoms
            or pair in seen
            or not _chemically_allowed(elements[a], elements[b])
        ):
            continue
        seen.add(pair)
        candidates.append(pair)

    # Prefer heavy-heavy and residue/topology order before hydrogen edges. This
    # prevents a malformed hydrogen edge from consuming an atom's sole valence.
    candidates.sort(key=lambda pair: (
        elements[pair[0]] == "H" or elements[pair[1]] == "H",
        pair,
    ))
    degree = [0] * n_atoms
    accepted: list[tuple[int, int]] = []
    for a, b in candidates:
        if degree[a] >= _MAX_DEGREE.get(elements[a], 4):
            continue
        if degree[b] >= _MAX_DEGREE.get(elements[b], 4):
            continue
        accepted.append((a, b))
        degree[a] += 1
        degree[b] += 1
    return accepted, raw_count


def _coordinate_array(xyz) -> np.ndarray:
    """Materialise single-chunk trajectories once; otherwise return a view copy."""
    return np.asarray(xyz[:], dtype=np.float32)


def _read_parquet_exact(path: Path) -> pl.DataFrame:
    """Read one literal path without treating glycan brackets as glob syntax."""
    return pl.from_arrow(pq.read_table(path))


def _recover_topology(
    coords: np.ndarray,
    atoms: pl.DataFrame,
    raw_pairs: Iterable[tuple[int, int]],
    block_frames: int,
) -> tuple[list[tuple[int, int]], int, int]:
    n_frames, n_atoms, _ = coords.shape
    elements = [_normalise_element(value) for value in atoms["element"].to_list()]
    bonds, input_count = _sanitise_bonds(n_atoms, elements, raw_pairs)
    retained_input_count = len(bonds)

    graph = _UnionFind(n_atoms)
    degree = [0] * n_atoms
    for a, b in bonds:
        graph.union(a, b)
        degree[a] += 1
        degree[b] += 1

    if graph.component_count() == 1:
        return bonds, input_count, retained_input_count

    candidates: list[tuple[int, int, float]] = []
    for a in range(n_atoms):
        if degree[a] >= _MAX_DEGREE.get(elements[a], 4):
            continue
        for b in range(a + 1, n_atoms):
            if graph.find(a) == graph.find(b):
                continue
            if degree[b] >= _MAX_DEGREE.get(elements[b], 4):
                continue
            if not _chemically_allowed(elements[a], elements[b]):
                continue
            candidates.append((a, b, _covalent_cutoff(elements[a], elements[b])))

    if not candidates:
        raise ValueError(
            f"Topology recovery failed: {graph.component_count()} disconnected "
            "components and no chemically plausible missing bonds"
        )

    candidate_a = np.asarray([item[0] for item in candidates], dtype=np.int32)
    candidate_b = np.asarray([item[1] for item in candidates], dtype=np.int32)
    cutoffs = np.asarray([item[2] for item in candidates], dtype=np.float32)
    minimum = np.full(len(candidates), np.inf, dtype=np.float32)
    close_count = np.zeros(len(candidates), dtype=np.int64)

    for start in range(0, n_frames, block_frames):
        block = coords[start:min(start + block_frames, n_frames)]
        distances = np.linalg.norm(
            block[:, candidate_a, :] - block[:, candidate_b, :], axis=2
        )
        finite = np.isfinite(distances)
        safe = np.where(finite, distances, np.inf)
        minimum = np.minimum(minimum, safe.min(axis=0))
        close_count += (safe <= cutoffs[None, :]).sum(axis=0)

    observed = [
        (float(minimum[i]), -int(close_count[i]), int(candidate_a[i]), int(candidate_b[i]))
        for i in range(len(candidates))
        if close_count[i] > 0
    ]
    observed.sort()

    inferred: list[tuple[int, int]] = []
    for _, _, a, b in observed:
        if graph.find(a) == graph.find(b):
            continue
        if degree[a] >= _MAX_DEGREE.get(elements[a], 4):
            continue
        if degree[b] >= _MAX_DEGREE.get(elements[b], 4):
            continue
        graph.union(a, b)
        degree[a] += 1
        degree[b] += 1
        inferred.append((a, b))
        if graph.component_count() == 1:
            break

    if graph.component_count() != 1:
        raise ValueError(
            "Topology recovery failed: trajectory evidence could not connect "
            f"{graph.component_count()} remaining molecular components"
        )
    return sorted(bonds + inferred), input_count, retained_input_count


def _write_compacted_coords(
    frame_data_dir: Path,
    coords: np.ndarray,
    accepted: np.ndarray,
    frame_chunk: int,
) -> None:
    destination = frame_data_dir / "coords.zarr"
    temporary = frame_data_dir / f".coords.zarr.quality.tmp.{os.getpid()}"
    backup = frame_data_dir / f".coords.zarr.quality.backup.{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)

    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    root = zarr.group(store=zarr.DirectoryStore(str(temporary)), overwrite=True)
    retained = int(accepted.sum())
    chunk = max(1, min(int(frame_chunk), retained))
    output = root.create_dataset(
        "xyz",
        shape=(retained, coords.shape[1], 3),
        chunks=(chunk, coords.shape[1], 3),
        dtype="f4",
        compressor=compressor,
        overwrite=True,
        fill_value=np.nan,
    )
    output[:] = coords[accepted].astype(np.float32, copy=False)

    try:
        destination.rename(backup)
        temporary.rename(destination)
        shutil.rmtree(backup)
    except Exception:
        if not destination.exists() and backup.exists():
            backup.rename(destination)
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def filter_broken_frames(
    frame_data_dir: str | Path,
    *,
    max_bond_distance: float = DEFAULT_MAX_BOND_ANGSTROM,
    block_frames: int = 1024,
    frame_chunk: int = 0,
    excluded=None,
) -> FrameQualityResult:
    """Recover topology, remove truly broken frames, and persist provenance.

    ``excluded`` (``[[first_frame, frames, name], ...]`` in source frames)
    leaves frames out on purpose, for example a replica that an
    administrator excluded from the analysis.  They are not used for the
    topology either, are counted as ``excluded_frames`` (not as rejected),
    and ``frame_quality.parquet`` gets an ``excluded`` column; without
    exclusions every file is written exactly as before.
    """
    excluded = normalise_exclusions(excluded)
    data_dir = Path(frame_data_dir)
    atoms = _read_parquet_exact(data_dir / "atoms.parquet")
    raw_bonds = _read_parquet_exact(data_dir / "bonds.parquet")
    xyz = zarr.open(str(data_dir / "coords.zarr"), mode="r")["xyz"]
    coords = _coordinate_array(xyz)
    if coords.ndim != 3 or coords.shape[2] != 3 or coords.shape[1] != len(atoms):
        raise ValueError(
            f"Unexpected coordinates shape {coords.shape} for {len(atoms)} atoms"
        )
    n_source_frames = int(coords.shape[0])
    if n_source_frames == 0:
        raise ValueError("No trajectory frames available for quality filtering")
    selected = np.ones(n_source_frames, dtype=bool)
    for first, count, name in excluded:
        if first + count > n_source_frames:
            raise ValueError(
                f"excluded frames {first} to {first + count - 1} ({name}) lie beyond the "
                f"{n_source_frames} frames of the trajectory"
            )
        selected[first:first + count] = False
    if not selected.any():
        raise ValueError("every frame of the trajectory is excluded")

    raw_pairs = [
        (int(a), int(b))
        for a, b in raw_bonds.select(["a", "b"]).iter_rows()
    ]
    bonds, input_count, retained_input_count = _recover_topology(
        coords if selected.all() else coords[selected], atoms, raw_pairs, max(1, int(block_frames))
    )
    bond_a = np.asarray([pair[0] for pair in bonds], dtype=np.int32)
    bond_b = np.asarray([pair[1] for pair in bonds], dtype=np.int32)
    if bond_a.size == 0 and len(atoms) > 1:
        raise ValueError("Recovered topology has no covalent bonds")

    max_distance = np.full(n_source_frames, np.nan, dtype=np.float32)
    offending_a = np.full(n_source_frames, -1, dtype=np.int32)
    offending_b = np.full(n_source_frames, -1, dtype=np.int32)
    finite_frame = np.isfinite(coords).all(axis=(1, 2))
    if bond_a.size == 0:
        max_distance.fill(0.0)
    else:
        for start in range(0, n_source_frames, max(1, int(block_frames))):
            stop = min(start + max(1, int(block_frames)), n_source_frames)
            distances = np.linalg.norm(
                coords[start:stop, bond_a, :] - coords[start:stop, bond_b, :],
                axis=2,
            )
            safe = np.where(np.isfinite(distances), distances, np.inf)
            peak_index = safe.argmax(axis=1)
            rows = np.arange(stop - start)
            max_distance[start:stop] = safe[rows, peak_index]
            offending_a[start:stop] = bond_a[peak_index]
            offending_b[start:stop] = bond_b[peak_index]

    accepted = finite_frame & np.isfinite(max_distance) & (
        max_distance <= float(max_bond_distance)
    )
    excluded_count = int((~selected).sum())
    rejected = int((selected & ~accepted).sum())
    accepted = accepted & selected
    retained = int(accepted.sum())
    if retained == 0:
        raise ValueError(
            "All trajectory frames are broken or non-finite; original process "
            "coordinates were left unchanged"
        )

    if rejected or excluded_count:
        requested_chunk = (
            retained if int(frame_chunk) <= 0 else int(frame_chunk)
        )
        _write_compacted_coords(data_dir, coords, accepted, requested_chunk)

    repaired_bonds = pl.DataFrame({
        "a": bond_a,
        "b": bond_b,
    })
    repaired_bonds.write_parquet(
        data_dir / "bonds.parquet", compression="zstd", statistics=True
    )

    source_frames = np.arange(n_source_frames, dtype=np.int32)
    quality_columns = {
        "source_frame": source_frames,
        "accepted": accepted,
        "max_bond_distance_A": max_distance,
        "offending_bond_a": offending_a,
        "offending_bond_b": offending_b,
    }
    if excluded:
        quality_columns["excluded"] = ~selected
    pl.DataFrame(quality_columns).write_parquet(
        data_dir / "frame_quality.parquet", compression="zstd", statistics=True
    )
    retained_sources = source_frames[accepted]
    pl.DataFrame({
        "frame": np.arange(retained, dtype=np.int32),
        "source_frame": retained_sources,
    }).write_parquet(
        data_dir / "frame_map.parquet", compression="zstd", statistics=True
    )
    pl.DataFrame({
        "n_frames": [retained],
        "source_frames": [n_source_frames],
        "retained_frames": [retained],
        "rejected_frames": [rejected],
    }).write_parquet(
        data_dir / "frames.parquet", compression="zstd", statistics=True
    )

    manifest = {
        "filter_version": FILTER_VERSION,
        "max_bond_distance_A": float(max_bond_distance),
        "source_frames": n_source_frames,
        "retained_frames": retained,
        "rejected_frames": rejected,
        "input_bonds": input_count,
        "retained_input_bonds": retained_input_count,
        "inferred_bonds": len(bonds) - retained_input_count,
        "topology_repaired": (
            input_count != retained_input_count
            or len(bonds) != retained_input_count
        ),
        "exclusion_rule": "nonfinite_or_covalent_bond_over_threshold",
    }
    if excluded:
        manifest["excluded_frames"] = excluded_count
        manifest["excluded_source_frames"] = excluded
    temporary_manifest = data_dir / f".frame_filter.json.tmp.{os.getpid()}"
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(data_dir / "frame_filter.json")

    return FrameQualityResult(
        source_frames=n_source_frames,
        retained_frames=retained,
        rejected_frames=rejected,
        input_bonds=input_count,
        retained_input_bonds=retained_input_count,
        inferred_bonds=len(bonds) - retained_input_count,
        topology_repaired=bool(manifest["topology_repaired"]),
        changed=True,
    )
