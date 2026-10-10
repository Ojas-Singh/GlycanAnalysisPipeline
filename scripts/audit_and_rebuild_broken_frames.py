#!/usr/bin/env python3
"""Audit old glycan inputs and selectively rebuild frame-quality-aware results."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# This maintenance workflow audits staged local data on compute nodes. Prevent
# a project .env file from routing absolute local paths through Oracle storage,
# which is unavailable from AMD nodes.
os.environ["GLYCOSHAPE_ORACLE_PAR_URL"] = ""

from lib import config
from lib.frame_quality import filter_broken_frames, quality_manifest_is_current
from lib.glypdbio import parse_mol2_bonds
from lib.glystore import store_from_pdb
from lib.v2 import run_glycan_analysis


def _input_paths(glycan_dir: Path) -> tuple[Path, Path]:
    name = glycan_dir.name
    named = (glycan_dir / f"{name}.pdb", glycan_dir / f"{name}.mol2")
    alternate = (glycan_dir / "simulation.pdb", glycan_dir / "structure.mol2")
    for pdb_path, mol2_path in (named, alternate):
        if pdb_path.is_file() and mol2_path.is_file():
            return pdb_path, mol2_path
    raise FileNotFoundError(
        f"{glycan_dir}: missing named PDB/MOL2 or simulation.pdb/structure.mol2"
    )


def _requested_names(args: argparse.Namespace) -> set[str]:
    names = set(args.glycan)
    if args.glycan_file:
        names.update(
            line.strip()
            for line in args.glycan_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if args.bond_report:
        with args.bond_report.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if int(row.get("frames_with_abnormal_bond") or 0) > 0:
                    names.add(str(row["glycan"]).strip())
    if args.retry_errors:
        report = json.loads(args.retry_errors.read_text(encoding="utf-8"))
        names.update(
            str(row["glycan"]).strip()
            for row in report.get("glycans", [])
            if row.get("status") == "error"
        )
    return names


def _audit_one(
    glycan_dir: Path,
    threshold: float,
    block_frames: int,
) -> dict:
    pdb_path, mol2_path = _input_paths(glycan_dir)
    # Glycan sequences can exceed the filesystem's 255-byte component limit.
    # The temporary path does not need to contain the sequence.
    with tempfile.TemporaryDirectory(prefix="gap-frame-audit-") as temporary:
        stored = Path(temporary) / "data"
        store_from_pdb(
            str(pdb_path),
            str(stored),
            get_connectivity=parse_mol2_bonds(mol2_path),
            connectivity_kind="serial",
            frames_buffer=config.store_frames_buffer,
            frame_chunk=config.store_frame_chunk,
        )
        result = filter_broken_frames(
            stored,
            max_bond_distance=threshold,
            block_frames=block_frames,
            frame_chunk=config.store_frame_chunk,
        )
    return {
        "source_frames": result.source_frames,
        "retained_frames": result.retained_frames,
        "rejected_frames": result.rejected_frames,
        "input_bonds": result.input_bonds,
        "retained_input_bonds": result.retained_input_bonds,
        "inferred_bonds": result.inferred_bonds,
        "topology_repaired": result.topology_repaired,
    }


def _rebuild_one(name: str, data_dir: Path, process_dir: Path) -> dict:
    return run_glycan_analysis(
        name,
        data_dir=str(data_dir),
        process_dir=str(process_dir),
        force_update=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(config.data_dir))
    parser.add_argument("--process-dir", type=Path, default=Path(config.process_dir))
    parser.add_argument("--output-dir", type=Path, default=Path(config.output_dir))
    parser.add_argument("--report", type=Path, default=Path("frame_quality_audit.json"))
    parser.add_argument("--glycan", action="append", default=[])
    parser.add_argument("--glycan-file", type=Path)
    parser.add_argument(
        "--bond-report",
        type=Path,
        help="Target only glycans with positive frames_with_abnormal_bond rows",
    )
    parser.add_argument(
        "--retry-errors",
        type=Path,
        help="Target only glycans marked error in a previous JSON audit report",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of glycans to audit concurrently",
    )
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--publish",
        action="store_true",
        help="After successful rebuild, run the normal full pipeline for publication",
    )
    parser.add_argument(
        "--max-bond-distance",
        type=float,
        default=config.max_covalent_bond_angstrom,
    )
    parser.add_argument(
        "--block-frames",
        type=int,
        default=config.frame_quality_block_frames,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.publish and not args.rebuild:
        raise SystemExit("--publish requires --rebuild")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    requested = _requested_names(args)
    glycan_dirs = sorted(
        path
        for path in args.data_dir.iterdir()
        if path.is_dir() and (not requested or path.name in requested)
    )
    rows = []
    failures = 0

    audits: dict[str, dict] = {}
    audit_errors: dict[str, Exception] = {}
    if args.workers == 1:
        for glycan_dir in glycan_dirs:
            try:
                audits[glycan_dir.name] = _audit_one(
                    glycan_dir, args.max_bond_distance, args.block_frames
                )
            except Exception as exc:
                audit_errors[glycan_dir.name] = exc
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            pending = {
                executor.submit(
                    _audit_one,
                    glycan_dir,
                    args.max_bond_distance,
                    args.block_frames,
                ): glycan_dir.name
                for glycan_dir in glycan_dirs
            }
            for future in as_completed(pending):
                name = pending[future]
                try:
                    audits[name] = future.result()
                except Exception as exc:
                    audit_errors[name] = exc

    for glycan_dir in glycan_dirs:
        name = glycan_dir.name
        process_data = args.process_dir / name / "data"
        current = quality_manifest_is_current(
            process_data, args.max_bond_distance
        )
        row = {
            "glycan": name,
            "status": "pending",
            "quality_metadata_current": current,
            "affected": False,
            "rebuilt": False,
            "published": False,
        }
        try:
            if name in audit_errors:
                raise audit_errors[name]
            audit = audits[name]
            row.update(audit)
            # Missing/stale quality metadata is intentionally affected: old
            # downstream artifacts cannot prove which source frames they used.
            row["affected"] = bool(
                not current
                or audit["rejected_frames"]
                or audit["topology_repaired"]
            )
            row["status"] = "affected" if row["affected"] else "clean"

        except Exception as exc:
            failures += 1
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    if args.rebuild:
        row_by_name = {row["glycan"]: row for row in rows}
        rebuild_names = [
            row["glycan"]
            for row in rows
            if row.get("affected") and row.get("status") != "error"
        ]
        rebuild_results: dict[str, dict] = {}
        rebuild_errors: dict[str, Exception] = {}
        if args.workers == 1:
            for name in rebuild_names:
                try:
                    rebuild_results[name] = _rebuild_one(
                        name, args.data_dir, args.process_dir
                    )
                except Exception as exc:
                    rebuild_errors[name] = exc
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                pending = {
                    executor.submit(
                        _rebuild_one, name, args.data_dir, args.process_dir
                    ): name
                    for name in rebuild_names
                }
                for future in as_completed(pending):
                    name = pending[future]
                    try:
                        rebuild_results[name] = future.result()
                    except Exception as exc:
                        rebuild_errors[name] = exc

        for name in rebuild_names:
            row = row_by_name[name]
            try:
                if name in rebuild_errors:
                    raise rebuild_errors[name]
                result = rebuild_results[name]
                if not result.get("success"):
                    raise RuntimeError(result.get("error_message") or "rebuild failed")
                row["rebuilt"] = True
                row["status"] = "rebuilt"

                if args.publish:
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve().parents[1] / "main.py"),
                        "--glycan", name,
                        "--update",
                        "--data-dir", str(args.data_dir),
                        "--process-dir", str(args.process_dir),
                        "--output-dir", str(args.output_dir),
                    ]
                    subprocess.run(command, check=True)
                    row["published"] = True
                    row["status"] = "published"
            except Exception as exc:
                failures += 1
                row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc}"
            print(json.dumps(row, sort_keys=True), flush=True)

    summary = {
        "filter_rule": "nonfinite_or_covalent_bond_over_threshold",
        "max_bond_distance_A": args.max_bond_distance,
        "glycans_checked": len(rows),
        "affected": sum(bool(row.get("affected")) for row in rows),
        "rebuilt": sum(bool(row.get("rebuilt")) for row in rows),
        "published": sum(bool(row.get("published")) for row in rows),
        "failures": failures,
        "glycans": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
