#!/usr/bin/env python3
"""Publish, replace or withdraw one entry of the database in S3 storage (Cloudflare R2).

    publish   the entry of a bundle as a new glycan
    replace   the entry of a bundle in place of the published one with the same ID
    withdraw  take a published glycan out (its raw data stays, its ID is never given to another glycan)
    status    what is stored: the database, the lock, an interrupted publication, one entry
    recover   finish a publication that was interrupted while its files were being switched
    update-metadata   publish a published entry again with other curated metadata, no new analysis
    reanalyse         analyse the stored raw entry of a published glycan again and publish it in place

``publish`` and ``replace`` take an extracted bundle directory (bundle.json,
archive/, static/, result.json, run.log) and a submission record::

    publish_entry.py publish --bundle run-17/ --submission run-17.submission.json --dry-run
    publish_entry.py publish --bundle run-17/ --submission run-17.submission.json --expect-hash <hash of the dry run>

The submission record (JSON) names the GlycoShape ID, the GLYCAM name, the
simulation metadata with their units and the curated search fields; see
lib/entry_complete.py.  ``--id GS01067`` instead of ``--submission`` takes the
metadata from ``source`` of bundle.json.

The glycan folder is completed in the work folder (default
``<bundle>.publish``), the root files are derived from the stored ones, and
everything is sent in the order that keeps the published database valid for
the website: new files, staged replacements, the switch, sync_manifest.json
last.  After any interruption run the same command again.  ``--dry-run``
does all of this locally, reads the storage and writes nothing to it; its
report is what an administrator approves.

``update-metadata`` takes a changes record (schema glycoshape-entry-changes,
lib/entry_ops.py); ``reanalyse`` the ID and what to do differently::

    publish_entry.py update-metadata --changes GS00020.changes.json --work-dir w/ --dry-run --report r.json
    publish_entry.py reanalyse --id GS00020 --simulated-anomer alpha --exclude-replica run_3 \
        --analysis-workers 12 --memory-limit-gb 30 --work-dir w/ --dry-run --report r.json

Both are published like ``replace`` (same ID), keep a record of the version in
publish-state/<static>/history/, and take --expect-hash like ``publish``.

Exit status: 0 done (or a dry run that would be published), 1 stopped (run
again), 2 refused by a check (nothing was written), 3 the database is locked
by another publisher.

The connection is read from GLYCOSHAPE_S3_ENDPOINT, GLYCOSHAPE_S3_REGION,
GLYCOSHAPE_S3_ACCESS_KEY_ID and GLYCOSHAPE_S3_SECRET_ACCESS_KEY, the places
from GLYCOSHAPE_S3_STATIC_BUCKET/_PREFIX, GLYCOSHAPE_S3_RAW_BUCKET/_PREFIX
and GLYCOSHAPE_S3_STAGING_PREFIX (environment or the .env file of the
repository).  The secret is handed to the client and never printed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")

EXIT_DONE, EXIT_STOPPED, EXIT_REFUSED, EXIT_LOCKED = 0, 1, 2, 3


def _say(message: str) -> None:
    print(message, flush=True)


def _levels(text: str):
    from lib.config import parse_cluster_levels

    return parse_cluster_levels(text)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("publish", "replace", "withdraw", "status", "recover", "update-metadata", "reanalyse"):
        command = commands.add_parser(name)
        command.add_argument("--static-bucket", help="Database bucket (default: GLYCOSHAPE_S3_STATIC_BUCKET).")
        command.add_argument("--raw-bucket", help="Archive bucket (default: GLYCOSHAPE_S3_RAW_BUCKET).")
        command.add_argument("--static-prefix", help="Published prefix (default: GLYCOSHAPE_S3_STATIC_PREFIX, static).")
        command.add_argument("--raw-prefix", help="Raw prefix (default: GLYCOSHAPE_S3_RAW_PREFIX, raw).")
        command.add_argument("--workers", type=int, default=8,
                             help="Parallel requests (default 8); also how many copies of the switch are made at the "
                                  "same time, which is what its duration depends on.")
        command.add_argument("--report", type=Path, help="Write the report of the run to this JSON file.")
        if name in ("publish", "replace", "withdraw", "recover", "update-metadata", "reanalyse"):
            command.add_argument("--lock-ttl", type=float, default=3600.0,
                                 help="Seconds after which the lock of a run that died may be taken over (default 3600).")
            command.add_argument("--break-lock", action="store_true",
                                 help="Take the lock although another run holds it: only when that run is known to be gone.")
        if name in ("publish", "replace", "withdraw", "update-metadata", "reanalyse"):
            command.add_argument("--work-dir", type=Path,
                                 help="Work folder with the journal (default: <bundle>.publish; required for withdraw).")
            command.add_argument("--cache-dir", type=Path,
                                 help="Folder that keeps the root files of the last publication, so the next one "
                                      "does not download them again.")
            command.add_argument("--dry-run", action="store_true",
                                 help="Complete, check and plan; read the storage; write nothing to it.")
            command.add_argument("--offline", action="store_true",
                                 help="Make no web lookup (GlyCosmos, GlyTouCan, GlyGen, PubChem).")
        if name in ("publish", "replace"):
            command.add_argument("--bundle", type=Path, required=True, help="Extracted bundle directory.")
            source = command.add_mutually_exclusive_group(required=True)
            source.add_argument("--submission", type=Path, help="Submission record (JSON file).")
            source.add_argument("--id", help="GlycoShape ID; the metadata is taken from source of bundle.json.")
            command.add_argument("--source-type", default="", help="With --id: where the data came from (keyword Source:<type>).")
        if name == "update-metadata":
            command.add_argument("--changes", type=Path, required=True,
                                 help="Changes record (JSON, schema glycoshape-entry-changes; lib/entry_ops.py).")
        if name == "reanalyse":
            command.add_argument("--id", required=True, help="GlycoShape ID of the published entry.")
            command.add_argument("--exclude-replica", action="append", default=[], metavar="NAME",
                                 help="Leave this replica out of the analysis (repeatable): a segment name of the raw "
                                      "manifest or a replica name of replicas.json (run_2, stored_run_1, ...).")
            command.add_argument("--simulated-anomer", choices=("alpha", "beta", "auto"),
                                 help="The anomer that was simulated; when it is not the one of the name, the entry is "
                                      "renamed to it and its files follow.  auto: what the residue code says.")
            command.add_argument("--note", default="", help="Why (for example the GAP fix this picks up); kept in the history.")
            command.add_argument("--author", default="", help="Who asked; kept in the history.")
            command.add_argument("--submission", type=Path,
                                 help="Curated metadata (submission record); default: those of the stored entry.")
            command.add_argument("--analysis-workers", type=int, default=8, help="Cores of the analysis (default 8).")
            command.add_argument("--memory-limit-gb", type=float,
                                 help="Refuse an entry whose analysis is estimated to need more memory than this.")
        if name in ("publish", "replace", "update-metadata", "reanalyse"):
            command.add_argument("--expect-hash", help="Publish only when the completed folder has this hash (of a dry run).")
            command.add_argument("--refresh-metadata", action="store_true",
                                 help="Complete the folder again even when the work folder has it (new lookups).")
            command.add_argument("--lookup-timeout", type=float, help="Seconds per web request (default: GLYCOSHAPE_LOOKUP_TIMEOUT, 30).")
            command.add_argument("--lookup-budget", type=float, default=300.0, help="Seconds for all web lookups of the entry (default 300).")
            command.add_argument("--levels", type=_levels, help="Clustering levels of the database (default: GLYCOSHAPE_CLUSTER_LEVELS, 1,2).")
            command.add_argument("--accept-pipeline-version", action="append", default=[], metavar="PATTERN",
                                 help="Accepted pipeline.version of a bundle (repeatable, fnmatch pattern; default: any).")
            command.add_argument("--accept-pipeline-commit", action="append", default=[], metavar="PATTERN",
                                 help="Accepted pipeline.commit of a bundle (repeatable; default: any).")
            command.add_argument("--require-clean", action="store_true",
                                 help="Refuse a bundle made from a checkout with uncommitted changes.")
            command.add_argument("--allow-incomplete", action="store_true",
                                 help="Publish an entry without WURCS and accession, or without its SNFG image.")
        if name in ("publish", "replace"):
            command.add_argument("--overwrite-raw", action="store_true",
                                 help="Replace a stored raw entry by one that does not extend it.")
            command.add_argument("--keep-previous-raw", choices=("auto", "manifest", "all"), default="auto",
                                 help="What of a replaced raw entry is copied to raw-history/<raw prefix>/ first (default auto: "
                                      "its manifest when the new entry extends it, every file otherwise).")
        if name == "withdraw":
            command.add_argument("--id", required=True, help="GlycoShape ID to withdraw.")
            command.add_argument("--glycam-name", help="Withdraw only when the ID is published for this GLYCAM name.")
            command.add_argument("--reason", default="", help="Kept with the record of the withdrawal.")
        if name == "status":
            command.add_argument("--id", help="GlycoShape ID to report on.")
            command.add_argument("--glycam-name", help="GLYCAM name to report on.")
            command.add_argument("--check", action="store_true",
                                 help="Compare size and SHA-256 metadata of the root files and of the files of the entry with the manifest.")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.command in ("withdraw", "update-metadata", "reanalyse") and args.work_dir is None:
        parser.error(f"{args.command} needs --work-dir")
    if args.command == "reanalyse" and args.analysis_workers < 1:
        parser.error("--analysis-workers must be at least 1")
    return args


def _target(args: argparse.Namespace) -> Any:
    from lib import entry_publish
    from lib.storage import S3Settings, create_s3_backend

    settings = S3Settings.from_env()
    connections = args.workers * settings.transfer_concurrency + 8
    return entry_publish.Target(
        static=create_s3_backend(settings, bucket=args.static_bucket or settings.static_bucket, max_connections=connections),
        raw=create_s3_backend(settings, bucket=args.raw_bucket or settings.raw_bucket, max_connections=connections),
        static_prefix=args.static_prefix or settings.static_prefix,
        raw_prefix=args.raw_prefix or settings.raw_prefix,
        staging_prefix=settings.staging_prefix,
        workers=args.workers,
    )


def _megabytes(count: int) -> str:
    return f"{count / 1e6:.1f} MB"


#: The simulation fields of an entry as they are printed: key of the report, unit, what it is.
_SIMULATION_LINE = (
    ("length", "us", "length"), ("package", "", "MD package"), ("forcefield", "", "force field"),
    ("temperature", "K", "temperature"), ("pressure", "bar", "pressure"), ("salt", "mM", "salt concentration"),
)


def _simulation_line(simulation: dict) -> str:
    """The simulation of an entry in one line; a value the submission does not have is said to be not given."""
    return ", ".join(
        f"{what} not given" if simulation.get(key) in (None, "") else f"{simulation[key]} {unit}".rstrip()
        for key, unit, what in _SIMULATION_LINE)


def _print_entry(report: dict) -> None:
    entry = report.get("entry")
    if not entry:
        return
    identity = entry["identity"]["archetype"]
    _say(f"entry {report['glycoshape_id']} {report['glycam_name'][:90]}")
    _say(f"  folder: {entry['file_count']} files, {_megabytes(entry['total_size'])}, hash {entry['hash']}")
    _say(f"  IUPAC {identity.get('iupac')}; GlyTouCan {identity.get('glytoucan')}; "
         f"WURCS {'yes' if identity.get('wurcs') else 'no'}")
    _say(f"  simulation: {_simulation_line(entry['simulation'])}")
    meta = entry.get("search_meta") or {}
    _say(f"  keywords {meta.get('keywords')}; common names {meta.get('common_names')}")
    lookups = entry["lookups"]
    _say(f"  lookups: {'offline' if lookups['offline'] else 'online'}, {lookups['counts']}; "
         f"unanswered {lookups['unanswered'] or 'none'}; empty fields {entry['gaps'] or 'none'}")


def _print_plan(report: dict) -> None:
    if "database" in report:
        database = report["database"]
        _say(f"database: {database['glycans']} glycans, global_hash {database['global_hash'][:16]}...")
    raw = report.get("raw")
    if raw and "to_key_prefix" in raw:
        # The raw entry moves with a new name (entry_ops.describe_raw_move).
        _say(f"raw {raw['from_key_prefix']} -> {raw['to_key_prefix']}: {raw['relation']}; "
             f"copy {len(raw['copy'])} file(s) ({_megabytes(raw['copy_bytes'])}) inside the storage, "
             f"upload {raw['upload'] or 'nothing'} ({_megabytes(raw['upload_bytes'])}); "
             f"delete {len(raw['delete'])} old object(s) afterwards; index {raw['index']}; "
             f"kept in history: {len(raw['history'])} object(s)")
    elif raw:
        _say(f"raw {raw['key_prefix']}: {raw['relation']}; upload {raw['upload'] or 'nothing'} "
             f"({_megabytes(raw['upload_bytes'])}); index {raw['index']}"
             + (f"; kept in history: {len(raw['history'])} object(s)" if raw["history"] else ""))
    static = report.get("static")
    if static and "upload_new" in static:
        _say(f"static: {static['upload_new']} new file(s) ({_megabytes(static['upload_new_bytes'])}), "
             f"{len(static['stage_and_switch'])} staged and switched ({_megabytes(static['stage_bytes'])}), "
             f"{static['unchanged']} unchanged, {static['delete_afterwards']} deleted afterwards; "
             f"global_hash -> {static['target_global_hash'][:16]}...")
    transfers = report.get("transfers")
    if transfers:
        _say(f"transfers: up {_megabytes(transfers['uploaded_bytes'])}, down {_megabytes(transfers['downloaded_bytes'])}, "
             f"copied in the storage {_megabytes(transfers['copied_bytes'])}; requests {transfers['requests']}")


def _finish(args: argparse.Namespace, report: dict) -> int:
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for warning in report.get("warnings") or []:
        _say(f"WARNING {warning}")
    for reason in report.get("refused") or []:
        print(f"REFUSED {reason}", file=sys.stderr, flush=True)
    return EXIT_REFUSED if report.get("refused") else EXIT_DONE


def _publish(args: argparse.Namespace) -> int:
    from lib import entry_complete, entry_publish

    if not (args.bundle / entry_complete.BUNDLE_FILE).is_file():
        raise SystemExit(f"not a bundle directory (no {entry_complete.BUNDLE_FILE}): {args.bundle}")
    if args.submission is not None:
        try:
            submission = json.loads(args.submission.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SystemExit(f"cannot read the submission record {args.submission}: {exc}")
    else:
        submission = entry_complete.submission_from_bundle(
            entry_complete.load_bundle(args.bundle, verify_hashes=False), args.id, source_type=args.source_type)
    policy = entry_publish.Policy(
        levels=args.levels,
        pipeline_versions=tuple(args.accept_pipeline_version),
        pipeline_commits=tuple(args.accept_pipeline_commit),
        require_clean=args.require_clean,
        allow_incomplete=args.allow_incomplete,
        overwrite_raw=args.overwrite_raw,
        keep_previous_raw=args.keep_previous_raw,
    )
    try:
        report = entry_publish.publish_entry(
            args.command, args.bundle, submission, _target(args), work_dir=args.work_dir, policy=policy,
            dry_run=args.dry_run, offline=True if args.offline else None, lookup_timeout=args.lookup_timeout,
            lookup_budget=args.lookup_budget, refresh=args.refresh_metadata, expect_hash=args.expect_hash,
            lock_ttl=args.lock_ttl, break_lock=args.break_lock, cache_dir=args.cache_dir, log=_say)
    except entry_publish.Refused as exc:
        report = exc.report or {"refused": [str(exc)]}
    _print_entry(report)
    _print_plan(report)
    status = _finish(args, report)
    if status == EXIT_DONE:
        _say("dry run: nothing was written to the storage; this entry would be published" if args.dry_run
             else f"{args.command}: {report['status']} {report['glycoshape_id']} "
                  f"(global_hash {report['static']['target_global_hash'][:16]}...) in {report['seconds']} s")
    return status


def _withdraw(args: argparse.Namespace) -> int:
    from lib import entry_publish

    try:
        report = entry_publish.withdraw_entry(
            args.id, _target(args), args.work_dir, glycam_name=args.glycam_name, reason=args.reason,
            dry_run=args.dry_run, offline=True if args.offline else None, lock_ttl=args.lock_ttl,
            break_lock=args.break_lock, cache_dir=args.cache_dir, log=_say)
    except entry_publish.Refused as exc:
        report = exc.report or {"refused": [str(exc)]}
    _print_plan(report)
    status = _finish(args, report)
    if status == EXIT_DONE:
        _say("dry run: nothing was written to the storage" if args.dry_run
             else f"withdraw: {report['status']} {report['glycoshape_id']} ({str(report.get('glycam_name'))[:80]})")
    return status


def _status(args: argparse.Namespace) -> int:
    from lib import entry_publish

    report = entry_publish.status(_target(args), args.id, args.glycam_name, check=args.check)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    database = report.get("database")
    if database:
        _say(f"database: {database['glycans']} glycans, global_hash {database['global_hash']}, "
             f"generated {database['generated_at']}; withdrawn IDs: {len(report['withdrawn'])}")
    lock = report.get("lock")
    _say("lock: none" if lock is None else
         f"lock: {lock['owner']} for {lock['action']} {lock['glycoshape_id']} until {lock['expires']}"
         + (" (expired)" if lock["expired"] else ""))
    pending = report.get("pending")
    _say("pending publication: none" if pending is None else
         f"pending publication: {pending['action']} of {pending['glycoshape_id']} by {pending['owner']} since {pending['created']}")
    entry = report.get("entry")
    if entry:
        _say(f"{entry['glycoshape_id']}: {entry['state']}; {entry['stored_objects']} object(s) stored"
             + (f"; {entry['file_count']} files, hash {entry['hash']}" if "hash" in entry else "")
             + (f"; raw entry {'stored' if entry.get('raw', {}).get('stored') else 'not stored'}" if "raw" in entry else ""))
        unanswered = ((entry.get("record") or {}).get("lookups") or {}).get("unanswered")
        if unanswered:
            _say(f"  lookups without an answer at publication: {', '.join(unanswered)}")
    for problem in report["problems"]:
        print(f"PROBLEM {problem}", file=sys.stderr, flush=True)
    return EXIT_DONE if report["ok"] else EXIT_STOPPED


def _print_operation(report: dict) -> None:
    """The lines an administrator reads of an update-metadata or reanalyse report."""
    rename = report.get("rename")
    if rename:
        raw = rename.get("raw") or {}
        _say(f"rename: {str(rename['from'])[:80]} -> {str(rename['to'])[:80]}; raw entry: {raw.get('relation')}")
    anomer = report.get("anomer")
    if anomer:
        _say(f"anomer: named {anomer['named']}, declared {anomer['declared']}, residue code "
             f"{anomer['residue_code']['residue']} ({anomer['residue_code']['anomer']})")
    analysis = report.get("analysis")
    if analysis:
        excluded = ", ".join(row["name"] for row in analysis.get("excluded_replicas") or []) or "none"
        _say(f"analysis: {analysis['seconds']} s{' (reused)' if analysis.get('reused') else ''}; "
             f"excluded replicas: {excluded}")
    memory = report.get("memory")
    if memory:
        _say(f"memory: about {memory['estimate_gb']} GB estimated, limit {memory['limit_gb']}")
    diff = report.get("diff")
    if diff:
        for name, change in list(diff["names"].items())[:12]:
            _say(f"  {name}: {str(change['before'])[:60]} -> {str(change['after'])[:60]}")
        for name, change in diff["simulation"].items():
            _say(f"  simulation {name}: {change['before']} -> {change['after']}")
        for name, change in diff["search_meta"].items():
            if "added" in change:
                _say(f"  {name}: +{change['added']} -{change['removed']}")
            else:
                _say(f"  {name} changed")
        for level, row in diff["analysis"]["levels"].items():
            reps = row.get("representative_source_frames") or {}
            _say(f"  {level}: population TV {row['total_variation']}"
                 + ("" if not reps else "; representatives " + ("unchanged" if reps.get("same") else "changed")))
        files = diff["files"]
        _say(f"  files: {len(files['added'])} added, {len(files['removed'])} removed, {len(files['changed'])} changed, "
             f"{files['unchanged']} unchanged; {_megabytes(files['bytes_before'])} -> {_megabytes(files['bytes_after'])}")


def _operation(args: argparse.Namespace) -> int:
    from lib import entry_complete, entry_ops, entry_publish

    policy = entry_publish.Policy(
        levels=args.levels,
        pipeline_versions=tuple(args.accept_pipeline_version),
        pipeline_commits=tuple(args.accept_pipeline_commit),
        require_clean=args.require_clean,
        allow_incomplete=args.allow_incomplete,
    )
    common = dict(policy=policy, dry_run=args.dry_run, offline=True if args.offline else None,
                  lookup_timeout=args.lookup_timeout, lookup_budget=args.lookup_budget, refresh=args.refresh_metadata,
                  expect_hash=args.expect_hash, lock_ttl=args.lock_ttl, break_lock=args.break_lock,
                  cache_dir=args.cache_dir, log=_say)
    try:
        if args.command == "update-metadata":
            try:
                changes = json.loads(args.changes.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise SystemExit(f"cannot read the changes record {args.changes}: {exc}")
            report = entry_ops.update_metadata(changes, _target(args), args.work_dir, **common)
        else:
            submission = None
            if args.submission is not None:
                try:
                    submission = json.loads(args.submission.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    raise SystemExit(f"cannot read the submission record {args.submission}: {exc}")
            report = entry_ops.reanalyse(
                args.id, _target(args), args.work_dir, exclude=args.exclude_replica, simulated=args.simulated_anomer,
                note=args.note, author=args.author, submission_record=submission, workers=args.analysis_workers,
                memory_limit_gb=args.memory_limit_gb, **common)
    except entry_publish.Refused as exc:
        report = exc.report or {"refused": [str(exc)]}
    except entry_complete.SubmissionError as exc:
        report = {"refused": [str(exc)]}
    _print_entry(report)
    _print_operation(report)
    _print_plan(report)
    status = _finish(args, report)
    if status == EXIT_DONE:
        _say("dry run: nothing was written to the storage; this version would be published" if args.dry_run
             else f"{args.command}: {report['status']} {report['glycoshape_id']} "
                  f"(global_hash {report['static']['target_global_hash'][:16]}...) in {report['seconds']} s")
    return status


def _recover(args: argparse.Namespace) -> int:
    from lib import entry_publish

    report = entry_publish.recover(_target(args), log=_say, lock_ttl=args.lock_ttl, break_lock=args.break_lock)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _say("nothing to recover: no publication is pending" if report is None else
         f"recovered: {report['action']} of {report['glycoshape_id']} is published (global_hash {report['global_hash'][:16]}...)")
    return EXIT_DONE


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    from lib import entry_complete, entry_publish, static_delta, static_publish

    handlers = {"publish": _publish, "replace": _publish, "withdraw": _withdraw, "status": _status, "recover": _recover,
                "update-metadata": _operation, "reanalyse": _operation}
    try:
        return handlers[args.command](args)
    except entry_publish.LockHeld as exc:
        print(f"{Path(__file__).name} {args.command}: {exc}", file=sys.stderr, flush=True)
        return EXIT_LOCKED
    except (entry_complete.SubmissionError, entry_complete.BundleError, entry_publish.Refused) as exc:
        print(f"{Path(__file__).name} {args.command}: REFUSED {exc}", file=sys.stderr, flush=True)
        return EXIT_REFUSED
    except Exception as exc:
        expected = (ValueError, RuntimeError, OSError, entry_publish.PublishEntryError, entry_complete.CompletionError,
                    static_delta.DeltaError, static_publish.PublishError)
        if isinstance(exc, expected) or type(exc).__module__.startswith(("botocore", "boto3", "s3transfer")):
            print(f"{Path(__file__).name} {args.command}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return EXIT_STOPPED
        raise


if __name__ == "__main__":
    raise SystemExit(main())
