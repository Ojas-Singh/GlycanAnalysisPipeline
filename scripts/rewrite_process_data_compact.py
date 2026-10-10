#!/usr/bin/env python3
"""Rewrite only process/<glycan>/data with one coordinate Zarr chunk."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# This maintenance script is intentionally local-only. Set this before importing
# pipeline modules, because lib.config loads .env on import using setdefault().
os.environ["GLYCOSHAPE_ORACLE_PAR_URL"] = ""
os.environ["GLYCOSHAPE_STORAGE"] = "local"   # the same for S3 storage
os.environ.setdefault("GLYCOSHAPE_MAX_WORKERS", "1")

import numpy as np
import polars as pl
import zarr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.glypdbio import parse_mol2_bonds
from lib import archive
from lib.glystore import store_trajectory
from lib.storage import reset_storage_manager


DEFAULT_FRAME_CHUNK = 0  # one Zarr coordinate chunk per trajectory
DEFAULT_FRAMES_BUFFER = 4096


def _count_process_data_files(process_dir: Path) -> int:
    if not process_dir.exists():
        return 0
    return sum(1 for _ in process_dir.glob("*/data/**/*") if _.is_file())


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")
            fh.flush()


def _discover_glycans(data_dir: Path) -> List[str]:
    return sorted(p.name for p in data_dir.iterdir() if p.is_dir())


def _read_glycan_list(path: Path) -> List[str]:
    glycans: List[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if line and not line.startswith("#"):
                glycans.append(line)
    return glycans


def _zarr_array(data_dir: Path):
    zarr_path = data_dir / "coords.zarr"
    if not zarr_path.exists():
        raise FileNotFoundError(f"missing {zarr_path}")
    root = zarr.open(str(zarr_path), mode="r")
    return root["xyz"]


def _validate_data_dir(data_dir: Path, min_frame_chunk: int) -> Dict[str, Any]:
    atoms_path = data_dir / "atoms.parquet"
    bonds_path = data_dir / "bonds.parquet"
    frames_path = data_dir / "frames.parquet"
    for required in (atoms_path, bonds_path, frames_path):
        if not required.exists():
            raise FileNotFoundError(f"missing {required}")

    atoms = _read_parquet(atoms_path)
    if atoms.height <= 0:
        raise ValueError(f"{atoms_path} has no atoms")

    frames = _read_parquet(frames_path)
    if "n_frames" not in frames.columns or frames.height <= 0:
        raise ValueError(f"{frames_path} does not contain n_frames")
    n_frames = int(frames["n_frames"][0])
    if n_frames <= 0:
        raise ValueError(f"{frames_path} has invalid n_frames={n_frames}")

    coords = _zarr_array(data_dir)
    if len(coords.shape) != 3:
        raise ValueError(f"coords shape must be 3D, got {coords.shape}")
    if coords.shape != (n_frames, atoms.height, 3):
        raise ValueError(
            f"coords shape {coords.shape} does not match "
            f"(n_frames={n_frames}, n_atoms={atoms.height}, 3)"
        )

    first = np.asarray(coords[0, :, :])
    last = np.asarray(coords[n_frames - 1, :, :])
    if not np.isfinite(first).all():
        raise ValueError("first coordinate frame contains non-finite values")
    if not np.isfinite(last).all():
        raise ValueError("last coordinate frame contains non-finite values")

    frame_chunk = int(coords.chunks[0])
    expected_chunk = n_frames if min_frame_chunk <= 0 else min(min_frame_chunk, n_frames)
    if frame_chunk < expected_chunk:
        raise ValueError(
            f"coords frame chunk {frame_chunk} is smaller than expected "
            f"{expected_chunk}"
        )

    return {
        "n_atoms": atoms.height,
        "n_frames": n_frames,
        "coords_shape": tuple(int(v) for v in coords.shape),
        "coords_chunks": tuple(int(v) for v in coords.chunks),
    }


def _read_parquet(path: Path) -> pl.DataFrame:
    try:
        import pyarrow.parquet as pq  # type: ignore

        return pl.from_arrow(pq.read_table(path))
    except Exception:
        import glob

        return pl.read_parquet(glob.escape(str(path)))


def _is_compact_valid(data_dir: Path, frame_chunk: int) -> Optional[Dict[str, Any]]:
    try:
        return _validate_data_dir(data_dir, frame_chunk)
    except Exception:
        return None


def _replace_data_dir(tmp_dir: Path, data_dir: Path) -> None:
    parent = data_dir.parent
    old_dir = parent / f".{data_dir.name}.old.{os.getpid()}.{time.time_ns()}"
    had_old = data_dir.exists()
    if had_old:
        data_dir.rename(old_dir)
    try:
        tmp_dir.rename(data_dir)
    except Exception:
        if had_old and old_dir.exists() and not data_dir.exists():
            old_dir.rename(data_dir)
        raise
    if had_old and old_dir.exists():
        shutil.rmtree(old_dir)


def _rewrite_one(
    glycan: str,
    data_dir: str,
    process_dir: str,
    frame_chunk: int,
    frames_buffer: int,
    force: bool,
) -> Dict[str, Any]:
    reset_storage_manager()
    start = time.time()
    data_root = Path(data_dir)
    process_root = Path(process_dir)
    source_dir = data_root / glycan
    process_glycan_dir = process_root / glycan
    target_data_dir = process_glycan_dir / "data"
    tmp_dir = process_glycan_dir / f".data.tmp.{os.getpid()}.{time.time_ns()}"

    result: Dict[str, Any] = {
        "glycan": glycan,
        "status": "unknown",
        "elapsed_sec": None,
        "frame_chunk": frame_chunk,
        "frames_buffer": frames_buffer,
    }
    try:
        pdb_path = source_dir / "simulation.pdb"
        mol2_path = source_dir / "structure.mol2"
        archive_inputs = archive.entry_inputs(source_dir)
        if archive_inputs is not None:
            pdb_path = Path(archive_inputs["trajectory_path"])
            mol2_path = Path(archive_inputs["mol2_path"])
        if not pdb_path.exists():
            raise FileNotFoundError(f"missing {pdb_path}")
        if not mol2_path.exists():
            raise FileNotFoundError(f"missing {mol2_path}")

        existing = None if force else _is_compact_valid(target_data_dir, frame_chunk)
        if existing is not None:
            result.update(existing)
            result["status"] = "skipped"
            result["reason"] = "existing data already satisfies compact chunk policy"
            return result

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)
        process_glycan_dir.mkdir(parents=True, exist_ok=True)

        store_trajectory(
            str(pdb_path),
            str(tmp_dir),
            get_connectivity=parse_mol2_bonds(mol2_path),
            connectivity_kind="serial",
            frames_buffer=frames_buffer,
            frame_chunk=frame_chunk,
        )
        validation = _validate_data_dir(tmp_dir, frame_chunk)
        _replace_data_dir(tmp_dir, target_data_dir)
        result.update(validation)
        result["status"] = "rewritten"
        return result
    except Exception as exc:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        result["status"] = "failed"
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        return result
    finally:
        result["elapsed_sec"] = round(time.time() - start, 3)


def _select_shard(glycans: List[str], shard_index: int, shard_count: int) -> List[str]:
    if shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--shard-index must be in [0, shard_count)")
    return [g for i, g in enumerate(glycans) if i % shard_count == shard_index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite only process/<glycan>/data with one coordinate Zarr chunk."
    )
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--process-dir", default="process", type=Path)
    parser.add_argument("--glycan", action="append", default=[])
    parser.add_argument("--glycan-list", type=Path)
    parser.add_argument("--frame-chunk", type=int, default=DEFAULT_FRAME_CHUNK)
    parser.add_argument("--frames-buffer", type=int, default=DEFAULT_FRAMES_BUFFER)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--log-dir", type=Path, default=Path("logs/compact_process_data"))
    parser.add_argument("--count-files", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frame_chunk < 0:
        raise SystemExit("--frame-chunk must be >= 0 (0: one chunk for the whole trajectory, the default)")
    if args.frames_buffer < 1:
        raise SystemExit("--frames-buffer must be >= 1")
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")

    if args.glycan:
        glycans = sorted(set(args.glycan))
    elif args.glycan_list:
        glycans = sorted(set(_read_glycan_list(args.glycan_list)))
    else:
        glycans = _discover_glycans(args.data_dir)
    glycans = _select_shard(glycans, args.shard_index, args.shard_count)

    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_shard{args.shard_index}-of-{args.shard_count}_{os.getpid()}"
    log_path = args.log_dir / f"rewrite_{run_id}.jsonl"
    summary_path = args.log_dir / f"summary_{run_id}.json"
    args.log_dir.mkdir(parents=True, exist_ok=True)

    before_count = _count_process_data_files(args.process_dir) if args.count_files else None
    header = {
        "event": "start",
        "run_id": run_id,
        "glycan_count": len(glycans),
        "data_dir": args.data_dir,
        "process_dir": args.process_dir,
        "frame_chunk": args.frame_chunk,
        "frames_buffer": args.frames_buffer,
        "jobs": args.jobs,
        "force": args.force,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "before_process_data_files": before_count,
    }
    _write_jsonl(log_path, [header])
    print(json.dumps(header, default=_json_default), flush=True)

    counts = {"rewritten": 0, "skipped": 0, "failed": 0}
    started = time.time()
    worker_args = [
        (
            glycan,
            str(args.data_dir),
            str(args.process_dir),
            args.frame_chunk,
            args.frames_buffer,
            args.force,
        )
        for glycan in glycans
    ]

    if args.jobs == 1:
        for item in worker_args:
            row = _rewrite_one(*item)
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            _write_jsonl(log_path, [row])
            print(json.dumps(row, default=_json_default), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(_rewrite_one, *item) for item in worker_args]
            for future in as_completed(futures):
                row = future.result()
                counts[row["status"]] = counts.get(row["status"], 0) + 1
                _write_jsonl(log_path, [row])
                print(json.dumps(row, default=_json_default), flush=True)

    after_count = _count_process_data_files(args.process_dir) if args.count_files else None
    summary = {
        "event": "summary",
        "run_id": run_id,
        "glycan_count": len(glycans),
        "counts": counts,
        "elapsed_sec": round(time.time() - started, 3),
        "before_process_data_files": before_count,
        "after_process_data_files": after_count,
        "log_path": log_path,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n")
    _write_jsonl(log_path, [summary])
    print(json.dumps(summary, indent=2, default=_json_default), flush=True)
    return 1 if counts.get("failed", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
