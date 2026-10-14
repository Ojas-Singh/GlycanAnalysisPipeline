#!/usr/bin/env python3
"""replicas.json for published entries, from the static database and the raw archive; nothing is clustered again.

    python scripts/replica_diagnostics.py --static-dir STATIC --archive-dir ARCHIVE --out OUT \\
        [--glycan GS00760 ...] [--workers N] [--check-divisions]

For every ``GSxxxxx/`` folder of STATIC (or the ones named with --glycan) the
Level 1 labels, principal components and glycosidic torsions are read from
``output/pca.csv``, ``output/torsion_glycosidic.csv`` and ``output/info.json``,
the replicas from the manifest of its raw archive entry (found by the GLYCAM
name of ``data.json``, as ``raw/index.json`` lists it), and the time per frame
from the simulation length of ``data.json`` where the replicas do not record
it.  See lib/replicas.py for the measures, the replica boundaries and the
verdict.  Both folders are only read.

Written into OUT:

    GSxxxxx.json    the replicas.json of the entry (the content that becomes
                    static/GSxxxxx/output/replicas.json when it is published, or
                    publish-state/static/replicas/GSxxxxx.json for the administrators)
    summary.json    one line per entry (verdict, reasons, basis, main numbers), the
                    counts by verdict and by basis, the archive entries without a
                    published entry, the failures, the time taken; with
                    --check-divisions also the class test of the equal runs of the
                    old entries without a run record, against lib.replicas.ASSUMED_DIVISIONS

Exit status: 0 every entry was written, 1 some failed (listed in summary.json
and on standard error), 2 the folders cannot be used.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from lib import replicas  # noqa: E402


def archive_folders(archive_dir: Path) -> Dict[str, str]:
    """Archive folder by GLYCAM name and by source folder (the name some old entries were published under)."""
    index = json.loads((archive_dir / "index.json").read_text(encoding="utf-8"))
    by: Dict[str, str] = {}
    for record in index["records"]:
        by[record["glycam_name"]] = record["folder"]
    for record in index["records"]:
        by.setdefault(record.get("source_folder") or record["glycam_name"], record["folder"])
    return by


def _job(task) -> Dict[str, Any]:
    glycan_id, static_dir, archive_dir, folder, out_dir, check = task
    started = time.perf_counter()
    try:
        data = json.loads((Path(static_dir) / glycan_id / "data.json").read_text(encoding="utf-8"))
        archetype = data.get("archetype") or {}
        manifest = None
        if folder is not None:
            manifest = json.loads((Path(archive_dir) / folder / "manifest.json").read_text(encoding="utf-8"))
        length = archetype.get("length")
        length = float(length) if isinstance(length, (int, float)) and not isinstance(length, bool) else None
        tables = replicas.tables_from_static(Path(static_dir) / glycan_id)
        document = replicas.diagnose_static(None, manifest, tables=tables, glycoshape_id=glycan_id, length_us=length,
                                            glycam_name=(manifest or {}).get("glycam_name") or archetype.get("name"))
        frames = document["frames"]["trajectory"]
        path = replicas.write(document, Path(out_dir) / f"{glycan_id}.json")
        line = replicas.summary(document)
        line.update({"archive_folder": folder, "glytoucan": archetype.get("glytoucan"),
                     "bytes": path.stat().st_size, "seconds": round(time.perf_counter() - started, 3)})
        percentile = None
        if check and manifest is not None and not manifest.get("segments"):
            kind, _ = replicas.runs_recorded_by(manifest.get("source") or {}, frames)
            single = kind == "one_block" and frames == replicas.Parameters().single_run_frames
            if kind not in ("records", "model_numbers") and not single:
                ranks = replicas.step_percentile(tables)
                percentile = None if ranks is None else ranks.astype(np.float32)
                line["source_record"] = kind
        return {"line": line, "percentile": percentile}
    except Exception as exc:                                   # reported per entry; the others go on
        return {"line": {"glycoshape_id": glycan_id, "archive_folder": folder, "error": f"{type(exc).__name__}: {exc}",
                         "seconds": round(time.perf_counter() - started, 3)}, "percentile": None}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--static-dir", required=True, help="published static database (GSxxxxx/ folders)")
    parser.add_argument("--archive-dir", required=True, help="raw archive (index.json, one folder per entry)")
    parser.add_argument("--out", required=True, help="folder for GSxxxxx.json and summary.json")
    parser.add_argument("--glycan", action="append", default=[], help="GlycoShape ID; repeatable (default: all)")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--check-divisions", action="store_true",
                        help="run the class test of equal runs over the entries without a run record and compare "
                             "it with lib.replicas.ASSUMED_DIVISIONS")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    static_dir, archive_dir, out_dir = Path(args.static_dir), Path(args.archive_dir), Path(args.out)
    if not static_dir.is_dir() or not (archive_dir / "index.json").is_file():
        print("replica_diagnostics: error: --static-dir must be a folder and --archive-dir an archive with index.json",
              file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    folders = archive_folders(archive_dir)
    ids = args.glycan or sorted(p.name for p in static_dir.iterdir() if p.is_dir() and p.name.startswith("GS"))
    tasks, published_names = [], set()
    for glycan_id in ids:
        try:
            name = json.loads((static_dir / glycan_id / "data.json").read_text(encoding="utf-8"))["archetype"]["name"]
        except (OSError, ValueError, KeyError):
            name = None
        folder = folders.get(name) if name else None
        if folder:
            published_names.add(folder)
        tasks.append((glycan_id, str(static_dir), str(archive_dir), folder, str(out_dir), args.check_divisions))
    with ProcessPoolExecutor(max(1, args.workers)) as pool:
        results = list(pool.map(_job, tasks, chunksize=2))

    lines = [result["line"] for result in results]
    failed = [line for line in lines if "error" in line]
    done = [line for line in lines if "error" not in line]
    by_verdict = Counter(line["verdict"] for line in done)
    by_basis = defaultdict(Counter)
    for line in done:
        by_basis[line["basis"]][line["verdict"]] += 1
    reasons = Counter(code for line in done for code in line["reasons"])
    report: Dict[str, Any] = {
        "what": "replicas.json of published entries, from the static database and the raw archive",
        "static_dir": str(static_dir), "archive_dir": str(archive_dir),
        "parameters": replicas.Parameters().describe(),
        "entries": len(lines), "written": len(done), "failed": len(failed),
        "verdicts": {verdict: by_verdict.get(verdict, 0) for verdict in replicas.VERDICTS},
        "verdicts_by_basis": {basis: {v: counts.get(v, 0) for v in replicas.VERDICTS} for basis, counts in sorted(by_basis.items())},
        "reasons": dict(reasons.most_common()),
        "without_archive_entry": sorted(line["glycoshape_id"] for line in done if not line.get("archive_folder")),
        "archive_entries_not_published": sorted(set(folders.values()) - published_names) if not args.glycan else None,
        "bytes": {"total": sum(line["bytes"] for line in done),
                  "median": int(np.median([line["bytes"] for line in done])) if done else 0,
                  "max": max((line["bytes"] for line in done), default=0)},
        "failures": failed,
    }
    if args.check_divisions:
        groups, percentiles = defaultdict(list), {}
        for result in results:
            line = result["line"]
            if result["percentile"] is not None:
                groups[line["frames"]].append(line["glycoshape_id"])
                percentiles[line["glycoshape_id"]] = result["percentile"].astype(np.float64)
        tested = replicas.class_test(groups, percentiles)
        accepted = {frames: test["accepted"]["run_frames"] for frames, test in tested.items() if test["accepted"]}
        report["divisions"] = {
            "accepted": {str(k): v for k, v in sorted(accepted.items())},
            "agrees_with_ASSUMED_DIVISIONS": accepted == replicas.ASSUMED_DIVISIONS,
            "only_here": {str(k): v for k, v in accepted.items() if replicas.ASSUMED_DIVISIONS.get(k) != v},
            "only_in_table": {str(k): v for k, v in replicas.ASSUMED_DIVISIONS.items() if accepted.get(k) != v},
            "classes": {str(k): {key: test[key] for key in ("entries", "tested", "accepted")} for k, test in tested.items()},
        }
    report["seconds"] = round(time.perf_counter() - started, 1)
    report["workers"] = args.workers
    report["lines"] = lines
    (out_dir / "summary.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    counts = ", ".join(f"{verdict} {count}" for verdict, count in report["verdicts"].items())
    print(f"replica_diagnostics: {len(done)} written, {len(failed)} failed in {report['seconds']} s ({counts}); "
          f"{out_dir / 'summary.json'}")
    for line in failed:
        print(f"replica_diagnostics: FAILED {line['glycoshape_id']}: {line['error']}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
