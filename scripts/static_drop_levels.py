#!/usr/bin/env python3
"""Write a static database that was built with Level 3 as the one without it (levels 1,2); nothing is recomputed.

    python scripts/static_drop_levels.py --source STATIC_DIR --dest NEW_DIR [--levels 1,2] [--workers N]

STATIC_DIR is a published static database (GSxxxxx/ folders, GLYCOSHAPE.json,
GlycoShape.zip, ...) that was built with Levels 1, 2 and 3; it is only read.
NEW_DIR, a new folder, becomes the database the pipeline writes with levels
1,2 from the same process data:

    per glycan   files that do not depend on the levels are hard-linked
                 (copied on another file system, or with --copy);
                 output/info.json, output/torsion_all.csv and data.json are
                 rewritten with the functions of the static stage;
                 output/level_3/ and output/torsion_level_3_reps.csv are left
                 out (a source without them gives the same result).
    database     GLYCOSHAPE.json, every GSxxxxx.zip, GlycoShape.zip and, last,
                 sync_manifest.json are written again with the functions of
                 the finalization; GLYCOSHAPE_RDF.ttl and the missing_* lists
                 do not depend on the levels and are copied.

Glycans are done in parallel; one that is in NEW_DIR is finished and is
skipped when the command is run again.  A glycan that fails is reported and
left out, the others are done, the root files are not written and the exit
status is 1.  Nothing but the database is written into NEW_DIR: the work
folder (default NEW_DIR.work, same file system) holds the staging folders,
the records and report.json (per glycan: files kept, rewritten and dropped,
bytes before and after).  See lib/static_levels.py and the README section
"Static database without Level 3".
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _say(message: str) -> None:
    print(message, flush=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="The static database to reduce (only read).")
    parser.add_argument("--dest", type=Path, required=True, help="The new static database (a new folder).")
    parser.add_argument("--levels", default="1,2", help="Clustering levels to keep (default 1,2).")
    parser.add_argument(
        "--workers", type=int, default=min(8, os.cpu_count() or 1),
        help="Glycans in work at the same time (default 8, or the number of cores).")
    parser.add_argument(
        "--work-dir", type=Path,
        help="Staging folders, records and the report (default: DEST.work; on the file system of DEST).")
    parser.add_argument("--report", type=Path, help="Write the report to this JSON file (default: WORK_DIR/report.json).")
    parser.add_argument(
        "--copy", action="store_true",
        help="Copy the unchanged files of the glycan folders instead of hard-linking them, "
        "so that the two databases share nothing.")
    parser.add_argument(
        "--skip-finalize", action="store_true",
        help="Stop after the glycan folders: no GSxxxxx.zip, no root files, no sync_manifest.json.")
    parser.add_argument("--verbose", action="store_true", help="Show the log of the pipeline functions.")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    """Exit status 0 done, 1 a glycan failed or the database was not completed, 2 unusable folders."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)   # before the pipeline sets it up

    from lib import static_levels

    try:
        report = static_levels.drop_levels(
            args.source,
            args.dest,
            levels=args.levels,
            workers=args.workers,
            work_dir=args.work_dir,
            report_path=args.report,
            copy=args.copy,
            finalize=not args.skip_finalize,
            log=_say,
        )
    except static_levels.StaticLevelsError as exc:
        print(f"{Path(__file__).name}: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(f"{Path(__file__).name}: interrupted; run the command again to continue", file=sys.stderr)
        return 130
    totals = report["totals"]
    for warning in report["warnings"][:20]:
        print(f"WARNING {warning}", file=sys.stderr)
    if len(report["warnings"]) > 20:
        print(f"WARNING ... and {len(report['warnings']) - 20} more, see the report", file=sys.stderr)
    for glycan_id, record in report["glycans"].items():
        if record["status"] == "failed":
            print(f"ERROR {glycan_id}: {record['error']}", file=sys.stderr)
    print(
        f"{report['dest']}: {totals['done']} glycans done, {totals['skipped']} already there, {totals['failed']} failed; "
        f"files kept={totals['files_kept']} (hard-linked {totals['files_linked']}, copied {totals['files_copied']}) "
        f"rewritten={totals['files_rewritten']} dropped={totals['files_dropped']}; "
        f"{totals['bytes_before'] / 1e9:.2f} GB -> {totals['bytes_after'] / 1e9:.2f} GB in the glycan folders; "
        f"root files {'written' if report['finalized'] else 'not written'}; "
        f"{report['seconds']['total']:.0f} s; report {report['report']}",
        flush=True,
    )
    if not report["ok"]:
        print(f"{Path(__file__).name}: {report['error']}; run the command again to continue", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
