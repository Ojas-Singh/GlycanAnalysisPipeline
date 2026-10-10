#!/usr/bin/env python3
"""Audit broken bonds and report informational trajectory displacement.

The processed coordinates are in Angstrom. A trajectory is marked suspect only
when it contains an implausibly long covalent bond. Large internal movement and
whole-glycan translation remain available as diagnostics, but never classify a
frame as broken and never control pipeline frame inclusion.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq
import zarr


@dataclass(frozen=True)
class Settings:
    max_bond_distance: float
    jump_distance: float
    translation_distance: float
    min_jumping_atoms: int
    block_frames: int


def _ranges(mask: np.ndarray, values: np.ndarray) -> list[dict[str, int | float]]:
    """Convert a boolean frame mask to compact inclusive frame ranges."""
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) != 1) + 1
    groups = np.split(indices, breaks)
    result = []
    for group in groups:
        group_values = values[group]
        peak_offset = int(np.nanargmax(group_values))
        result.append(
            {
                "start_frame": int(group[0]),
                "end_frame": int(group[-1]),
                "frame_count": int(group.size),
                "peak_frame": int(group[peak_offset]),
                "max_value_A": float(group_values[peak_offset]),
            }
        )
    return result


def _empty_result(glycan: str, status: str, error: str = "") -> dict:
    return {
        "glycan": glycan,
        "status": status,
        "suspect": False,
        "n_frames": 0,
        "n_atoms": 0,
        "broken_bond_frames": 0,
        "partial_jump_frames": 0,
        "whole_jump_frames": 0,
        "max_bond_distance_A": "",
        "max_internal_jump_A": "",
        "max_centroid_jump_A": "",
        "first_suspect_frame": "",
        "error": error,
        "events": [],
    }


def inspect_glycan(args: tuple[str, str, Settings]) -> dict:
    glycan, data_dir_string, settings = args
    data_dir = Path(data_dir_string)
    try:
        coords_path = data_dir / "coords.zarr"
        atoms_path = data_dir / "atoms.parquet"
        bonds_path = data_dir / "bonds.parquet"
        if not coords_path.exists() or not atoms_path.exists():
            return _empty_result(glycan, "missing_coordinates")

        xyz = zarr.open(str(coords_path), mode="r")["xyz"]
        if len(xyz.shape) != 3 or xyz.shape[2] != 3:
            raise ValueError(f"unexpected coordinate shape {xyz.shape}")
        n_frames, n_atoms, _ = map(int, xyz.shape)
        if n_frames == 0 or n_atoms == 0:
            return _empty_result(glycan, "empty_trajectory")

        atoms = pq.read_table(atoms_path, columns=["element"]).to_pydict()
        elements = np.asarray(
            [str(value).strip().upper() for value in atoms["element"]]
        )
        if elements.size != n_atoms:
            raise ValueError(
                f"atoms table has {elements.size} rows but coordinates have {n_atoms}"
            )
        heavy_indices = np.flatnonzero((elements != "H") & (elements != "D"))
        # A malformed/missing element column should not silently disable checks.
        if heavy_indices.size == 0:
            heavy_indices = np.arange(n_atoms)

        bonds_a = np.empty(0, dtype=np.int64)
        bonds_b = np.empty(0, dtype=np.int64)
        if bonds_path.exists():
            bonds = pq.read_table(bonds_path, columns=["a", "b"]).to_pydict()
            bonds_a = np.asarray(bonds["a"], dtype=np.int64)
            bonds_b = np.asarray(bonds["b"], dtype=np.int64)
            valid = (
                (bonds_a >= 0)
                & (bonds_b >= 0)
                & (bonds_a < n_atoms)
                & (bonds_b < n_atoms)
            )
            bonds_a, bonds_b = bonds_a[valid], bonds_b[valid]

        bond_max = np.full(n_frames, np.nan, dtype=np.float32)
        internal_jump = np.full(n_frames, np.nan, dtype=np.float32)
        centroid_jump = np.full(n_frames, np.nan, dtype=np.float32)
        previous = None
        # The pipeline normally stores one compressed Zarr chunk for the whole
        # trajectory. Re-reading that chunk for every logical block would
        # repeatedly decompress tens of GB, so materialize it once per worker.
        # Multi-chunk stores remain streamed.
        in_memory = (
            np.asarray(xyz[:], dtype=np.float32)
            if int(xyz.chunks[0]) >= n_frames
            else None
        )

        for start in range(0, n_frames, settings.block_frames):
            stop = min(start + settings.block_frames, n_frames)
            block = (
                in_memory[start:stop]
                if in_memory is not None
                else np.asarray(xyz[start:stop], dtype=np.float32)
            )
            if not np.isfinite(block).all():
                raise ValueError("trajectory contains NaN or infinite coordinates")

            if bonds_a.size:
                distances = np.linalg.norm(
                    block[:, bonds_a, :] - block[:, bonds_b, :], axis=2
                )
                bond_max[start:stop] = distances.max(axis=1)

            if previous is not None:
                joined = np.concatenate((previous[None, :, :], block), axis=0)
                offset = start
            else:
                joined = block
                offset = start + 1
            if joined.shape[0] > 1:
                delta = np.diff(joined[:, heavy_indices, :], axis=0)
                # Median atom displacement estimates a coherent translation
                # robustly even when only part of a molecule crosses the box.
                translation = np.median(delta, axis=1, keepdims=True)
                residual = delta - translation
                count = delta.shape[0]
                residual_norm = np.linalg.norm(residual, axis=2)
                enough_atoms = (
                    residual_norm > settings.jump_distance
                ).sum(axis=1) >= settings.min_jumping_atoms
                # Store zero for isolated heavy-atom excursions so they do not
                # become candidates; retain the peak for coordinated events.
                peaks = residual_norm.max(axis=1)
                internal_jump[offset : offset + count] = np.where(
                    enough_atoms, peaks, 0.0
                )
                centroid_jump[offset : offset + count] = np.linalg.norm(
                    delta.mean(axis=1), axis=1
                )
            previous = block[-1].copy()

        broken = np.nan_to_num(bond_max, nan=-np.inf) > settings.max_bond_distance
        partial = (
            np.nan_to_num(internal_jump, nan=-np.inf) > settings.jump_distance
        )
        whole = (
            np.nan_to_num(centroid_jump, nan=-np.inf)
            > settings.translation_distance
        ) & ~partial
        # Only a long validated covalent bond means the molecule is broken.
        # Displacement diagnostics are intentionally non-excluding.
        suspect_mask = broken

        events = []
        for event_type, mask, values, threshold in (
            ("broken_bond", broken, bond_max, settings.max_bond_distance),
            ("partial_molecule_jump", partial, internal_jump, settings.jump_distance),
            (
                "whole_molecule_translation",
                whole,
                centroid_jump,
                settings.translation_distance,
            ),
        ):
            for event_range in _ranges(mask, values):
                events.append(
                    {
                        "event_type": event_type,
                        "affects_processing": event_type == "broken_bond",
                        "threshold_A": threshold,
                        **event_range,
                    }
                )

        suspect_indices = np.flatnonzero(suspect_mask)
        return {
            "glycan": glycan,
            "status": "ok",
            "suspect": bool(suspect_indices.size),
            "n_frames": n_frames,
            "n_atoms": n_atoms,
            "broken_bond_frames": int(broken.sum()),
            "partial_jump_frames": int(partial.sum()),
            "whole_jump_frames": int(whole.sum()),
            "max_bond_distance_A": (
                float(np.nanmax(bond_max)) if bonds_a.size else ""
            ),
            "max_internal_jump_A": float(np.nanmax(internal_jump)),
            "max_centroid_jump_A": float(np.nanmax(centroid_jump)),
            "first_suspect_frame": (
                int(suspect_indices[0]) if suspect_indices.size else ""
            ),
            "error": "",
            "events": events,
        }
    except Exception as exc:
        return _empty_result(glycan, "error", f"{type(exc).__name__}: {exc}")


def _write_csv(path: Path, rows: Iterable[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--process-dir", type=Path, default=Path("process"))
    parser.add_argument("--output-dir", type=Path, default=Path("pbc_jump_report"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--glycan", action="append", default=[])
    parser.add_argument("--glycan-file", type=Path)
    parser.add_argument("--max-bond-distance", type=float, default=3.0)
    parser.add_argument("--jump-distance", type=float, default=10.0)
    parser.add_argument("--translation-distance", type=float, default=10.0)
    parser.add_argument("--min-jumping-atoms", type=int, default=3)
    parser.add_argument("--block-frames", type=int, default=2048)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.block_frames < 2 or args.min_jumping_atoms < 1:
        raise SystemExit(
            "--workers and --min-jumping-atoms must be >= 1; "
            "--block-frames must be >= 2"
        )
    settings = Settings(
        args.max_bond_distance,
        args.jump_distance,
        args.translation_distance,
        args.min_jumping_atoms,
        args.block_frames,
    )

    requested = set(args.glycan)
    if args.glycan_file:
        requested.update(
            line.strip()
            for line in args.glycan_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if requested:
        glycan_dirs = [
            args.process_dir / name for name in sorted(requested)
        ]
    else:
        glycan_dirs = sorted(
            (path for path in args.process_dir.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        )
    work = [(path.name, str(path / "data"), settings) for path in glycan_dirs]
    print(f"Checking {len(work)} glycans with {args.workers} worker(s)", flush=True)

    if args.workers == 1:
        results = [inspect_glycan(item) for item in work]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(inspect_glycan, work, chunksize=1))
    results.sort(key=lambda row: row["glycan"])

    summary_fields = [
        "glycan",
        "status",
        "suspect",
        "n_frames",
        "n_atoms",
        "broken_bond_frames",
        "partial_jump_frames",
        "whole_jump_frames",
        "max_bond_distance_A",
        "max_internal_jump_A",
        "max_centroid_jump_A",
        "first_suspect_frame",
        "error",
    ]
    _write_csv(args.output_dir / "glycan_summary.csv", results, summary_fields)

    event_rows = []
    for result in results:
        for event in result["events"]:
            event_rows.append({"glycan": result["glycan"], **event})
    event_fields = [
        "glycan",
        "event_type",
        "start_frame",
        "end_frame",
        "frame_count",
        "peak_frame",
        "max_value_A",
        "threshold_A",
        "affects_processing",
    ]
    _write_csv(args.output_dir / "jump_events.csv", event_rows, event_fields)

    counts = {
        "glycans_checked": len(results),
        "suspect_glycans": sum(bool(row["suspect"]) for row in results),
        "clean_glycans": sum(
            row["status"] == "ok" and not row["suspect"] for row in results
        ),
        "failed_or_missing": sum(row["status"] != "ok" for row in results),
        "event_ranges": len(event_rows),
        "thresholds_A": {
            "max_bond_distance": settings.max_bond_distance,
            "internal_jump": settings.jump_distance,
            "minimum_jumping_heavy_atoms": settings.min_jumping_atoms,
            "whole_molecule_translation": settings.translation_distance,
        },
        "classification_rule": "broken_covalent_bond_only",
    }
    with (args.output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(counts, handle, indent=2)
        handle.write("\n")
    print(json.dumps(counts, indent=2), flush=True)
    return 1 if counts["failed_or_missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
