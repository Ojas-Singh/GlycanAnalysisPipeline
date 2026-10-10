#!/usr/bin/env python3
"""Convert the raw data directory into the trajectory archive (lib/archive.py).

Every source folder (``simulation.pdb``, ``structure.mol2`` and optionally
``info.json`` / ``name.json``) becomes one archive entry named by its GLYCAM
name.  The source directory is only read.  Each entry is written to a staging
folder inside the archive directory, read back, compared with the source text
and renamed to its final name when that comparison passes.  A rerun skips
every entry whose manifest still verifies, so an interrupted run is resumed by
running the same command again.

Written at the end, in the archive directory:

* ``index.json`` - one record per entry of the whole archive;
* ``conversion_report.json`` - what this run did, with the entries that
  failed and the reason under ``failed``.

Run one conversion per archive directory at a time.  ``--verify-only``
re-checks existing entries instead (against the source when ``--data-dir`` is
given) and writes ``verification_report.json``.

``--repair`` archives a source that is damaged in one of a few known ways
instead of refusing it: a frame that the file cuts short (it is dropped; every
complete frame is kept, in order), frames without a MODEL record, a MODEL
record without a frame.  The manifest of such an entry says what was wrong
under ``source_irregularities``; ``index.json`` and the report list these
entries under ``repaired``.  A source that needs no repair is converted to the
same entry with and without it.

``--report FILE`` writes the report of this run there instead of
``conversion_report.json`` (``verification_report.json``) of the archive
directory, which every run replaces: use it to add entries to an archive
whose report is to stay.

``--names FILE`` names source folders from outside: a JSON object of source
folder name to GLYCAM name.  It takes precedence over ``name.json``,
``info.json`` and the folder name, and the manifest records ``names-file`` as
the source of the name.  An entry that the same source folder was given under
another name before stays where it is and is reported;
``--remove-superseded`` deletes it once the new entry is in place.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# The conversion is single-threaded NumPy; keep BLAS/OpenMP pools from starting
# in every worker.  This has to happen before NumPy is imported.
for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib import archive

REPORT_FILE = "conversion_report.json"
VERIFICATION_REPORT_FILE = "verification_report.json"


def _peak_rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def _convert_one(
    source_dir: str, archive_dir: str, options: Dict[str, Any], name_override: Optional[str] = None
) -> Dict[str, Any]:
    """Worker: convert one source folder and describe the outcome."""
    row: Dict[str, Any] = {"source_folder": Path(source_dir).name, "status": "failed"}
    try:
        result = archive.convert_entry(
            source_dir,
            archive_dir,
            force=options["force"],
            check_frames=options["check_frames"],
            block_bytes=options["block_bytes"],
            name_override=name_override,
            repair=options["repair"],
        )
        manifest = result["manifest"]
        archive_size = sum(item["size"] for item in manifest["files"].values())
        archive_size += (Path(result["entry"]) / archive.MANIFEST_FILE).stat().st_size
        row.update(
            status=result["status"],
            glycam_name=manifest["glycam_name"],
            glycam_name_from=manifest["source"]["glycam_name_from"],
            folder=manifest["folder"],
            atoms=manifest["atoms"],
            frames=manifest["frames"],
            source_size=manifest["source"]["size"],
            archive_size=archive_size,
            warnings=manifest.get("warnings", []),
        )
        repaired = archive.repair_summary(manifest)
        if repaired is not None:   # also for an entry that is skipped: it was repaired when it was converted
            row["repaired"] = repaired
        timing = result["timing"]
        if timing:
            megabytes = manifest["source"]["size"] / 1e6
            row.update(
                seconds=round(timing["seconds"], 3),
                cpu_seconds=round(timing["cpu_seconds"], 3),
                parse_and_write_seconds=round(timing["parse_and_write_seconds"], 3),
                verify_seconds=round(timing["verify_seconds"], 3),
                mb_per_second=round(megabytes / timing["seconds"], 1),
                mb_per_cpu_second=round(megabytes / max(timing["cpu_seconds"], 1e-9), 1),
                max_abs_difference_angstrom=manifest["verification"]["max_abs_difference_angstrom"],
            )
    except archive.ArchiveError as exc:
        row.update(error_type=type(exc).__name__, reason=str(exc))
    except Exception as exc:  # unexpected: keep the traceback for the report
        row.update(
            error_type=type(exc).__name__,
            reason=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
    row["peak_rss_mb"] = _peak_rss_mb()
    return row


def _verify_one(
    entry_dir: str, source_dir: Optional[str], check_frames: int, name_override: Optional[str] = None
) -> Dict[str, Any]:
    """Worker: verify one existing entry, against its source when given."""
    started = time.perf_counter()
    try:
        report = archive.verify_entry(
            entry_dir, source_dir, deep=True, check_frames=check_frames, name_override=name_override)
    except Exception as exc:
        report = {"ok": False, "errors": [f"{type(exc).__name__}: {exc}"], "entry": entry_dir}
    report["folder"] = Path(entry_dir).name
    report["against_source"] = source_dir is not None
    report["seconds"] = round(time.perf_counter() - started, 3)
    return report


def _read_names(path: Path) -> List[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _load_names(path: Optional[Path]) -> Dict[str, str]:
    """The ``--names`` file: a JSON object of source folder name to GLYCAM name."""
    if path is None:
        return {}
    try:
        names = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"--names {path}: {exc}")
    if not isinstance(names, dict) or not all(
        isinstance(folder, str) and folder and isinstance(name, str) and name for folder, name in names.items()
    ):
        raise SystemExit(f"--names {path}: expected a JSON object of source folder name to GLYCAM name")
    return names


def _superseded(archive_dir: Path, rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Entries left under another name by source folders that the names file named in this run."""
    renamed = {
        row["source_folder"]: row["glycam_name"]
        for row in rows
        if row["status"] in ("converted", "skipped") and row.get("glycam_name_from") == archive.NAMES_FILE_ORIGIN
    }
    found: List[Dict[str, str]] = []
    if not renamed:
        return found
    for entry_dir in archive.iter_entry_dirs(archive_dir):
        try:
            manifest = archive.load_manifest(entry_dir)
            source_folder, glycam_name = manifest["source"]["folder"], manifest["glycam_name"]
        except (archive.ArchiveError, KeyError, TypeError):
            continue
        if source_folder in renamed and glycam_name != renamed[source_folder]:
            found.append({"source_folder": source_folder, "folder": entry_dir.name, "glycam_name": glycam_name})
    return found


def _plan(
    data_dir: Path, requested: List[str], names: Optional[Dict[str, str]] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Source folders to convert (largest first) and those that cannot be planned."""
    names = names or {}
    folders = sorted(p for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    unknown = sorted(set(names) - {p.name for p in folders})
    if unknown:
        raise SystemExit(f"--names lists folders that are not under {data_dir}: {', '.join(unknown)}")
    if requested:
        present = {p.name for p in folders}
        missing = sorted(set(requested) - present)
        if missing:
            raise SystemExit(f"No such source folder under {data_dir}: {', '.join(missing)}")
        folders = [p for p in folders if p.name in set(requested)]

    tasks: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    by_target: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for folder in folders:
        try:
            inputs = archive.source_inputs(folder)
            glycam_name, _, _ = archive.resolve_glycam_name(folder, names.get(folder.name))
            task = {
                "source_dir": str(folder),
                "source_folder": folder.name,
                "name_override": names.get(folder.name),
                "glycam_name": glycam_name,
                "folder": archive.entry_folder_name(glycam_name),
                "size": inputs["pdb"].stat().st_size,
            }
        except (archive.ArchiveError, OSError) as exc:
            failed.append({
                "source_folder": folder.name,
                "status": "failed",
                "error_type": type(exc).__name__,
                "reason": str(exc),
            })
            continue
        tasks.append(task)
        by_target[task["folder"]].append(task)

    # Two source folders with one GLYCAM name would overwrite each other.
    for target, group in by_target.items():
        if len(group) > 1:
            names = ", ".join(task["source_folder"] for task in group)
            for task in group:
                tasks.remove(task)
                failed.append({
                    "source_folder": task["source_folder"],
                    "glycam_name": task["glycam_name"],
                    "status": "failed",
                    "error_type": "NamingError",
                    "reason": f"source folders {names} have the same GLYCAM name",
                })
    tasks.sort(key=lambda task: (-task["size"], task["source_folder"]))
    return tasks, failed


def _progress(done: int, total: int, row: Dict[str, Any]) -> None:
    label = row.get("glycam_name") or row["source_folder"]
    if len(label) > 70:
        label = label[:67] + "..."
    if row["status"] == "converted":
        detail = (
            f"{row['frames']} frames x {row['atoms']} atoms, "
            f"{row['source_size'] / 1e6:.1f} MB -> {row['archive_size'] / 1e6:.1f} MB "
            f"({row['source_size'] / row['archive_size']:.1f}x), "
            f"{row['seconds']:.2f} s, {row['mb_per_cpu_second']:.0f} MB/s per core"
        )
    elif row["status"] == "skipped":
        detail = "manifest verifies"
    else:
        detail = row.get("reason", "")
    if "repaired" in row:
        repaired = row["repaired"]
        detail += (
            f"; REPAIRED: {row['frames']} of {repaired['frames_in_source']} frames of the source kept, "
            + ", ".join(f"{count} {kind}" for kind, count in repaired["counts"].items())
        )
    print(f"[{done}/{total}] {row['status']:9s} {label}: {detail}", flush=True)


def _run(function, argument_lists: List[tuple], workers: int):
    """Yield results of ``function(*arguments)``; in this process when workers == 1."""
    if workers == 1:
        for arguments in argument_lists:
            yield arguments, function(*arguments)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(function, *arguments): arguments for arguments in argument_lists}
        for future in as_completed(futures):
            try:
                yield futures[future], future.result()
            except Exception as exc:  # a worker died (for example out of memory)
                yield futures[future], exc


def _convert(args: argparse.Namespace) -> int:
    data_dir: Path = args.data_dir.resolve()
    archive_dir: Path = args.archive_dir.resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"--data-dir does not exist: {data_dir}")
    if archive_dir == data_dir or data_dir in archive_dir.parents or archive_dir in data_dir.parents:
        raise SystemExit("--archive-dir must be outside --data-dir (the source is never written to).")
    report_path = args.report.resolve() if args.report else archive_dir / REPORT_FILE
    if data_dir in report_path.parents:
        raise SystemExit("--report must be outside --data-dir (the source is never written to).")
    archive_dir.mkdir(parents=True, exist_ok=True)

    requested = list(args.glycan)
    if args.glycan_file:
        requested.extend(_read_names(args.glycan_file))
    tasks, rows = _plan(data_dir, requested, _load_names(args.names))
    total = len(tasks) + len(rows)
    for done, row in enumerate(rows, start=1):
        _progress(done, total, row)

    options = {
        "force": args.force,
        "check_frames": args.check_frames,
        "block_bytes": args.block_mb << 20,
        "repair": args.repair,
    }
    started = time.perf_counter()
    argument_lists = [(task["source_dir"], str(archive_dir), options, task["name_override"]) for task in tasks]
    for arguments, row in _run(_convert_one, argument_lists, args.workers):
        if isinstance(row, Exception):
            row = {
                "source_folder": Path(arguments[0]).name,
                "status": "failed",
                "error_type": type(row).__name__,
                "reason": f"worker process failed: {type(row).__name__}: {row}",
            }
        rows.append(row)
        _progress(len(rows), total, row)
    elapsed = time.perf_counter() - started

    shutil.rmtree(archive_dir / archive.STAGING_DIR, ignore_errors=True)
    superseded = _superseded(archive_dir, rows)
    for old in superseded:
        old["removed"] = bool(args.remove_superseded)
        if args.remove_superseded:
            shutil.rmtree(archive_dir / old["folder"])
        print(
            f"SUPERSEDED {old['source_folder']}: entry {old['folder']} holds the same source under another name; "
            + ("removed" if args.remove_superseded else "rerun with --remove-superseded to delete it"),
            file=sys.stderr,
        )
    index, problems = archive.write_index(archive_dir)

    converted = sorted((r for r in rows if r["status"] == "converted"), key=lambda r: r["source_folder"])
    skipped = sorted((r for r in rows if r["status"] == "skipped"), key=lambda r: r["source_folder"])
    failed = sorted((r for r in rows if r["status"] == "failed"), key=lambda r: r["source_folder"])
    repaired = [
        {
            "source_folder": r["source_folder"],
            "glycam_name": r["glycam_name"],
            "folder": r["folder"],
            "status": r["status"],
            "frames": r["frames"],
            **r["repaired"],
        }
        for r in converted + skipped if "repaired" in r
    ]
    repaired.sort(key=lambda r: r["source_folder"])
    source_bytes = sum(r["source_size"] for r in converted)
    cpu_seconds = sum(r["cpu_seconds"] for r in converted)
    report = {
        "generated": index["generated"],
        "converter_version": archive.CONVERTER_VERSION,
        "data_dir": str(data_dir),
        "archive_dir": str(archive_dir),
        "names_file": str(args.names.resolve()) if args.names else None,
        "repair": bool(args.repair),
        "superseded": superseded,
        "workers": args.workers,
        "wall_seconds": round(elapsed, 3),
        "counts": {"converted": len(converted), "skipped": len(skipped), "failed": len(failed)},
        "totals": {
            "source_bytes_converted": source_bytes,
            "archive_bytes_converted": sum(r["archive_size"] for r in converted),
            "mb_per_second": round(source_bytes / 1e6 / elapsed, 1) if converted else None,
            "mb_per_cpu_second": round(source_bytes / 1e6 / cpu_seconds, 1) if cpu_seconds else None,
            "peak_rss_mb_largest_worker": max((r.get("peak_rss_mb", 0) for r in rows), default=0),
        },
        "index_entries": index["entries"],
        "index_problems": problems,
        "failed": failed,
        "repaired": repaired,
        "converted": converted,
        "skipped": [r["source_folder"] for r in skipped],
    }
    temporary = report_path.with_name(report_path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(report_path)

    print(
        f"converted={len(converted)} skipped={len(skipped)} failed={len(failed)} "
        f"in {elapsed:.1f} s; index.json lists {index['entries']} entries"
        + (f", {len(index['repaired'])} of them repaired" if index["repaired"] else "")
        + f"; report: {report_path}",
        flush=True,
    )
    for row in repaired:
        print(
            f"REPAIRED {row['source_folder']} ({row['status']}): {row['frames']} of {row['frames_in_source']} "
            f"frames of the source kept, {row['bytes_ignored']} bytes not archived; "
            + ", ".join(f"{count} {kind}" for kind, count in row["counts"].items()),
            file=sys.stderr,
        )
    for row in failed:
        print(f"FAILED {row['source_folder']}: {row.get('reason', '')}", file=sys.stderr)
    for problem in problems:
        print(f"INDEX {problem}", file=sys.stderr)
    return 1 if failed or problems else 0


def _verify(args: argparse.Namespace) -> int:
    archive_dir: Path = args.archive_dir.resolve()
    data_dir: Optional[Path] = args.data_dir.resolve() if args.data_dir else None
    requested = set(args.glycan)
    if args.glycan_file:
        requested.update(_read_names(args.glycan_file))
    names = _load_names(args.names)

    argument_lists = []
    for entry_dir in archive.iter_entry_dirs(archive_dir):
        source_dir = None
        try:
            source_folder = archive.load_manifest(entry_dir)["source"]["folder"]
        except (archive.ArchiveError, KeyError, TypeError):
            source_folder = None
        if requested and source_folder not in requested and entry_dir.name not in requested:
            continue
        if data_dir is not None and source_folder:
            source_dir = str(data_dir / source_folder)
        argument_lists.append((str(entry_dir), source_dir, args.check_frames, names.get(source_folder)))

    started = time.perf_counter()
    reports = []
    for arguments, report in _run(_verify_one, argument_lists, args.workers):
        if isinstance(report, Exception):
            report = {"ok": False, "errors": [f"worker process failed: {report}"], "folder": Path(arguments[0]).name}
        reports.append(report)
        state = "ok" if report["ok"] else "FAILED: " + "; ".join(report["errors"])
        print(f"[{len(reports)}/{len(argument_lists)}] {report['folder'][:70]}: {state}", flush=True)
    reports.sort(key=lambda r: r["folder"])
    bad = [r for r in reports if not r["ok"]]
    summary = {
        "archive_dir": str(archive_dir),
        "data_dir": str(data_dir) if data_dir else None,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "entries": len(reports),
        "ok": len(reports) - len(bad),
        "failed": bad,
        "reports": reports,
    }
    report_path = args.report.resolve() if args.report else archive_dir / VERIFICATION_REPORT_FILE
    temporary = report_path.with_name(report_path.name + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    print(f"verified={len(reports) - len(bad)} failed={len(bad)}; report: {report_path}", flush=True)
    return 1 if bad else 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, help="Source directory, one folder per glycan (only read).")
    parser.add_argument("--archive-dir", type=Path, required=True, help="Archive directory to write.")
    parser.add_argument("--workers", type=int, default=1, help="Glycans converted at the same time (default 1).")
    parser.add_argument("--glycan", action="append", default=[], help="Restrict to this source folder; repeatable.")
    parser.add_argument("--glycan-file", type=Path, help="One source folder name per line.")
    parser.add_argument(
        "--names",
        type=Path,
        help="JSON object of source folder name to GLYCAM name; takes precedence over name.json, "
        "info.json and the folder name.",
    )
    parser.add_argument(
        "--remove-superseded",
        action="store_true",
        help="Delete an entry that a source folder named by --names left under another GLYCAM name.",
    )
    parser.add_argument("--force", action="store_true", help="Convert again even when the entry verifies.")
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Archive a source with a frame that the file cuts short (the frame is dropped), with frames "
        "without a MODEL record or with a MODEL record without a frame, and record it in the manifest "
        "(source_irregularities). Without it such a source fails.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Write the report of this run to this file instead of conversion_report.json "
        "(verification_report.json with --verify-only) in --archive-dir.",
    )
    parser.add_argument(
        "--check-frames",
        type=int,
        default=archive.DEFAULT_CHECK_FRAMES,
        help="Frames per entry re-read from the source text with float() as an independent check "
        "(default %(default)s; -1 for every frame).",
    )
    parser.add_argument("--block-mb", type=int, default=archive.DEFAULT_BLOCK_BYTES >> 20, help="Read block in MiB.")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify the entries of --archive-dir instead of converting; with --data-dir also against the source.",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.block_mb < 1:
        parser.error("--block-mb must be at least 1")
    if not args.verify_only and args.data_dir is None:
        parser.error("--data-dir is required unless --verify-only is given")
    if args.report is not None and not args.report.resolve().parent.is_dir():
        parser.error(f"--report: no such directory: {args.report.resolve().parent}")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return _verify(args) if args.verify_only else _convert(args)


if __name__ == "__main__":
    raise SystemExit(main())
