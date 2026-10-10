#!/usr/bin/env python3
"""Move the raw-trajectory archive to and from S3 storage (Cloudflare R2).

An entry is stored under ``<prefix>/<GLYCAM name>/`` (``raw/`` by default) in
the raw bucket, keyed by the GLYCAM name of its ``manifest.json``; the index
of the archive is ``<prefix>/index.json``.

    upload    send the entries of a local archive directory.  An object whose
              size and SHA-256 metadata match the manifest is skipped; the
              manifest of an entry goes last; afterwards the listing is
              compared with index.json, which is uploaded when all is there.
    download  fetch entries by GLYCAM name (--glycan), or all entries of the
              stored index, into a local directory.  Every file is compared
              with the manifest; the entry folder appears when it verified
              and can be processed by ``main.py --data-dir <directory>``.
    compare   compare the listing, and the stored manifest of every entry,
              with the index.json of a local archive.

Run the same command again to resume.  The connection is read from
GLYCOSHAPE_S3_ENDPOINT, GLYCOSHAPE_S3_REGION, GLYCOSHAPE_S3_ACCESS_KEY_ID and
GLYCOSHAPE_S3_SECRET_ACCESS_KEY, the place from GLYCOSHAPE_S3_RAW_BUCKET and
GLYCOSHAPE_S3_RAW_PREFIX (environment or the .env file of the repository).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib import archive, archive_transfer
from lib.storage import S3Settings, create_s3_backend


def _names(args: argparse.Namespace) -> List[str]:
    names = [name.strip() for name in args.glycan if name.strip()]
    if args.glycan_file:
        names.extend(
            line.strip()
            for line in args.glycan_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return names


def _finish(args: argparse.Namespace, report: Dict[str, Any]) -> int:
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for row in report.get("failed", []):
        print(f"FAILED {row['glycam_name']}: {row.get('error', '')}", file=sys.stderr)
    return 0 if report["ok"] else 1


def _print_comparison(comparison: Dict[str, Any], prefix: str) -> None:
    print(
        f"listing of {prefix}/ against index.json: {comparison['entries']} entries, {comparison['objects']} objects; "
        f"missing={len(comparison['missing'])} other-size={len(comparison['size_mismatch'])} "
        f"other-manifest={len(comparison['stale'])} not-in-index={len(comparison['extra'])}",
        flush=True,
    )
    for label, keys in (("MISSING", comparison["missing"]), ("OTHER SIZE", comparison["size_mismatch"]),
                        ("OTHER MANIFEST", comparison["stale"]), ("NOT IN INDEX", comparison["extra"])):
        for key in keys[:20]:
            print(f"{label} {key}", file=sys.stderr)
        if len(keys) > 20:
            print(f"{label} ... and {len(keys) - 20} more", file=sys.stderr)


def _upload(args: argparse.Namespace, backend: Any, prefix: str) -> int:
    report = archive_transfer.upload_archive(
        args.archive_dir, backend, prefix, _names(args), args.workers, args.dry_run, _say)
    counts = report["counts"]
    verb = "would upload" if args.dry_run else "uploaded"
    print(
        f"{verb}={counts.get('uploaded', 0)} skipped={counts.get('skipped', 0)} failed={counts.get('failed', 0)} "
        f"({report['bytes'] / 1e6:.1f} MB) to {prefix}/",
        flush=True,
    )
    if report["comparison"] is not None:
        _print_comparison(report["comparison"], prefix)
    return _finish(args, report)


def _download(args: argparse.Namespace, backend: Any, prefix: str) -> int:
    names = _names(args)
    report = archive_transfer.download_archive(
        backend, args.archive_dir, prefix, names, args.workers, args.trust_local, args.dry_run, _say)
    counts = report["counts"]
    print(
        f"downloaded={counts.get('downloaded', 0)} would-download={counts.get('download', 0)} "
        f"skipped={counts.get('skipped', 0)} failed={counts.get('failed', 0)} "
        f"({report['bytes'] / 1e6:.1f} MB) into {args.archive_dir}",
        flush=True,
    )
    if names and not args.dry_run:
        for row in report["entries"]:
            if row["status"] != "failed":   # the folder is what main.py --glycan takes
                print(f"entry folder: {args.archive_dir / row['folder']}", flush=True)
    return _finish(args, report)


def _compare(args: argparse.Namespace, backend: Any, prefix: str) -> int:
    comparison = archive_transfer.compare_with_index(backend, prefix, archive_transfer.load_index(args.archive_dir))
    _print_comparison(comparison, prefix)
    return _finish(args, comparison)


def _say(message: str) -> None:
    print(message, flush=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (("upload", _upload), ("download", _download), ("compare", _compare)):
        command = commands.add_parser(name)
        command.set_defaults(function=function)
        command.add_argument(
            "--archive-dir", "--data-dir", dest="archive_dir", type=Path, required=True,
            help="Local archive directory (one folder per entry).")
        command.add_argument("--bucket", help="Bucket (default: GLYCOSHAPE_S3_RAW_BUCKET).")
        command.add_argument("--prefix", help="Key prefix of the archive (default: GLYCOSHAPE_S3_RAW_PREFIX, raw).")
        command.add_argument("--report", type=Path, help="Write the report of the run to this JSON file.")
        if name == "compare":
            continue
        command.add_argument(
            "--glycan", action="append", default=[],
            help="Restrict to this entry; repeatable. Download: the GLYCAM name. "
            "Upload: the GLYCAM name, the entry folder or the source folder.")
        command.add_argument("--glycan-file", type=Path, help="One such name per line.")
        command.add_argument("--workers", type=int, default=4, help="Entries transferred at the same time (default 4).")
        command.add_argument("--dry-run", action="store_true", help="Show what would be transferred; change nothing.")
        if name == "download":
            command.add_argument(
                "--trust-local", action="store_true",
                help="Do not hash the files of an entry that is already there with the stored manifest; compare sizes.")
    args = parser.parse_args(argv)
    if getattr(args, "workers", 1) < 1:
        parser.error("--workers must be at least 1")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        settings = S3Settings.from_env()
        workers = getattr(args, "workers", 1)
        backend = create_s3_backend(
            settings, bucket=args.bucket or settings.raw_bucket,
            max_connections=workers * settings.transfer_concurrency + 8)
        return args.function(args, backend, (args.prefix or settings.raw_prefix).strip("/"))
    except Exception as exc:
        expected = (ValueError, RuntimeError, OSError, archive_transfer.TransferError, archive.ArchiveError)
        if isinstance(exc, expected) or type(exc).__module__.startswith(("botocore", "boto3", "s3transfer")):
            raise SystemExit(f"{Path(__file__).name} {args.command}: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
