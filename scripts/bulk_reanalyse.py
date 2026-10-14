#!/usr/bin/env python3
"""Re-analyse the whole raw archive into a new static database, check it and publish it (docs/bulk-reanalysis.md).

    bulk_reanalyse.py sync-published --published-dir P                                       (reads R2)
    bulk_reanalyse.py plan       --archive-dir A --published-dir P --run-dir R [--shards 4] [--fixes F]
    bulk_reanalyse.py analyse    --run-dir R --shard I [--archive-dir A] [--jobs 4 --workers 16]  (compute nodes)
    bulk_reanalyse.py complete   --run-dir R [--jobs 4] [--offline] [--curation-dir C]           (network)
    bulk_reanalyse.py finalize   --run-dir R [--keep-published ID|all] [--offline]
    bulk_reanalyse.py compare    --run-dir R [--intended-rename ID]
    bulk_reanalyse.py summary    --run-dir R
    bulk_reanalyse.py check-base --run-dir R [--archive-dir A]                                   (reads R2)
    bulk_reanalyse.py publish    --run-dir R --expect-base HASH [--workers 16] [--dry-run]        (writes R2)

Every stage can be run again: what is done is skipped.  Exit status: 0 done,
1 something failed or a check did not pass (the summary says what), 2 the
stage cannot run (for example no plan, or a missing published folder).
check-base and publish read GLYCOSHAPE_S3_* like scripts/publish_entry.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")


def _say(message: str) -> None:
    print(message, flush=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("sync-published", "plan", "analyse", "complete", "finalize", "compare", "summary", "check-base",
                 "publish"):
        command = commands.add_parser(name)
        if name != "sync-published":
            command.add_argument("--run-dir", type=Path, required=True, help="The folder of this run.")
        if name in ("sync-published", "plan", "complete", "finalize", "compare"):
            command.add_argument("--published-dir", type=Path, required=(name == "sync-published"),
                                 help="The published static database (default: the one of the plan).")
        if name in ("plan", "check-base", "analyse"):
            command.add_argument("--archive-dir", type=Path, required=(name == "plan"),
                                 help="The raw archive (an archive directory with index.json); for analyse, where the "
                                      "archive of the plan is on this machine.")
    plan = commands.choices["plan"]
    plan.add_argument("--shards", type=int, default=4, help="Shards of about equal time (Slurm array tasks; default 4).")
    plan.add_argument("--fixes", type=Path, help='JSON {"GSxxxxx" or GLYCAM name: {"exclude": [replica names]}}.')
    plan.add_argument("--rename-to-raw-names", action="store_true",
                      help="Complete the entries published under a GlyTouCan accession under their GLYCAM name.")
    plan.add_argument("--workers", type=int, default=16, help="Workers per entry assumed by the memory estimate.")
    analyse = commands.choices["analyse"]
    analyse.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID") or 1))
    analyse.add_argument("--shards", type=int, help="Check that the plan has this many shards.")
    analyse.add_argument("--jobs", type=int, default=4, help="Entries at the same time (default 4).")
    analyse.add_argument("--workers", type=int, default=16, help="Cores per entry (default 16).")
    analyse.add_argument("--memory-gb", type=float, default=230.0,
                         help="Memory of the node for the analyses; entries wait until theirs fits (default 230).")
    analyse.add_argument("--keep-process", action="store_true", help="Keep result/process/ (removed by default).")
    analyse.add_argument("--entry", action="append", default=[], metavar="ID|NAME",
                         help="Only these entries, whatever their shard (repeatable).")
    analyse.add_argument("--force", action="store_true", help="Analyse again even what is done.")
    complete = commands.choices["complete"]
    complete.add_argument("--jobs", type=int, default=4, help="Entries at the same time (processes; default 4).")
    complete.add_argument("--offline", action="store_true", help="No web lookups (accessions are then missing).")
    complete.add_argument("--curation-dir", type=Path,
                          help="Records of the incremental publisher (publish-state/static/entries/*.json).")
    complete.add_argument("--lookup-timeout", type=float)
    complete.add_argument("--lookup-budget", type=float, default=300.0)
    complete.add_argument("--refresh", action="store_true", help="Complete again even what is done.")
    finalize = commands.choices["finalize"]
    finalize.add_argument("--keep-published", action="append", default=[], metavar="ID|all",
                          help="Use the published folder of this ID as it is (no raw entry, or its analysis failed).")
    finalize.add_argument("--offline", action="store_true", help="The missing reports without services.")
    compare = commands.choices["compare"]
    compare.add_argument("--intended-rename", action="append", default=[], metavar="ID",
                         help="An ID whose name may change (a fix); repeatable.")
    for name in ("sync-published", "check-base", "publish"):
        command = commands.choices[name]
        command.add_argument("--static-bucket")
        command.add_argument("--raw-bucket")
        command.add_argument("--static-prefix")
        command.add_argument("--raw-prefix")
        command.add_argument("--report", type=Path)
    commands.choices["sync-published"].add_argument("--workers", type=int, default=8)
    commands.choices["sync-published"].add_argument(
        "--records-dir", type=Path, help="Also download the records of the entries (for complete --curation-dir).")
    publish = commands.choices["publish"]
    publish.add_argument("--expect-base", required=True,
                         help="global_hash of the published database the run was compared with (plan.json).")
    publish.add_argument("--workers", type=int, default=16)
    publish.add_argument("--dry-run", action="store_true")
    publish.add_argument("--delete-stale", action="store_true",
                         help="Delete objects under the prefix that the new manifest no longer lists.")
    return parser.parse_args(argv)


def _target(args: argparse.Namespace):
    from lib import entry_publish
    from lib.storage import S3Settings, create_s3_backend

    settings = S3Settings.from_env()
    workers = getattr(args, "workers", 8)
    connections = workers * settings.transfer_concurrency + 8
    return entry_publish.Target(
        static=create_s3_backend(settings, bucket=args.static_bucket or settings.static_bucket, max_connections=connections),
        raw=create_s3_backend(settings, bucket=args.raw_bucket or settings.raw_bucket, max_connections=connections),
        static_prefix=args.static_prefix or settings.static_prefix,
        raw_prefix=args.raw_prefix or settings.raw_prefix,
        staging_prefix=settings.staging_prefix,
        workers=workers,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    from lib import bulk_reanalysis as bulk

    try:
        if args.command == "sync-published":
            result = bulk.sync_published(_target(args), args.published_dir, workers=args.workers,
                                         records_dir=args.records_dir, log=_say)
            if args.report:
                args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            return 0
        if args.command == "plan":
            fixes = json.loads(args.fixes.read_text(encoding="utf-8")) if args.fixes else None
            plan = bulk.make_plan(args.archive_dir, args.published_dir, args.run_dir, shards=args.shards, fixes=fixes,
                                  rename_to_raw_names=args.rename_to_raw_names, workers=args.workers, log=_say)
            for row in plan["entries"]:
                if row["skip"]:
                    _say(f"skipped {row['glycam_name'][:80]}: {row['skip']}")
            return 0
        if args.command == "analyse":
            result = bulk.analyse(args.run_dir, shard=args.shard, shards=args.shards, jobs=args.jobs,
                                  workers=args.workers, memory_gb=args.memory_gb, keep_process=args.keep_process,
                                  only=args.entry, force=args.force, archive_dir=args.archive_dir, log=_say)
            return 0 if not result["failed"] else 1
        if args.command == "complete":
            result = bulk.complete(args.run_dir, args.published_dir, jobs=args.jobs, offline=args.offline,
                                   curation_dir=args.curation_dir, lookup_timeout=args.lookup_timeout,
                                   lookup_budget=args.lookup_budget, refresh=args.refresh, log=_say)
            return 0 if not (result["failed"] or result["incomplete"] or result["not_analysed"]) else 1
        if args.command == "finalize":
            bulk.finalize(args.run_dir, args.published_dir, keep_published=args.keep_published,
                          online=not args.offline, log=_say)
            return 0
        if args.command == "compare":
            return 0 if bulk.compare(args.run_dir, args.published_dir, args.intended_rename, log=_say)["ok"] else 1
        if args.command == "summary":
            result = bulk.summary(args.run_dir)
            _say(json.dumps({key: result[key] for key in ("counts", "finalized", "compared")}, indent=2))
            for item in result["failed"][:50]:
                _say(f"FAILED {item}")
            return 0 if not result["failed"] else 1
        plan = bulk.load_plan(args.run_dir)
        if args.command == "check-base":
            index = (args.archive_dir / "index.json") if args.archive_dir else None
            result = bulk.check_base(_target(args), plan["base_global_hash"], index)
            if args.report:
                args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            _say(f"stored {str(result['stored_global_hash'])[:16]}..., base {plan['base_global_hash'][:16]}...; "
                 + ("a bulk publication is safe" if result["safe"] else "NOT safe: " + "; ".join(result["problems"])))
            return 0 if result["safe"] else 1
        if args.command == "publish":
            result = bulk.publish(args.run_dir, _target(args), args.expect_base, workers=args.workers,
                                  dry_run=args.dry_run, delete_stale=args.delete_stale, log=_say)
            if args.report:
                args.report.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
            return 0
    except bulk.BulkError as exc:
        print(f"bulk_reanalyse.py {args.command}: {exc}", file=sys.stderr, flush=True)
        return 2
    except Exception as exc:
        from lib import entry_publish, static_publish

        if isinstance(exc, entry_publish.LockHeld):
            print(f"bulk_reanalyse.py {args.command}: {exc}", file=sys.stderr, flush=True)
            return 3
        if isinstance(exc, (static_publish.PublishError, entry_publish.PublishEntryError, OSError)):
            print(f"bulk_reanalyse.py {args.command}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return 1
        raise
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
