#!/usr/bin/env python3
"""Publish the static database to S3 storage (Cloudflare R2), and check what is published.

    publish   send what changed of a local static directory to
              ``<prefix>/`` of the static bucket.  Only the files that
              sync_manifest.json lists are published.  New files are sent to
              their keys, changed files to a staging prefix and copied over
              the published ones inside the storage at the end, and
              sync_manifest.json is written last.  An interrupted upload
              leaves the published prefix as it was for the website; run the
              command again to continue.
    check     read the published prefix back the way the website does: fetch
              sync_manifest.json, validate it, download every file it lists
              and compare size and SHA-256.  --quick compares the SHA-256
              metadata of the objects instead of downloading them.

Run ``main.py --mode static-only`` first; it writes sync_manifest.json.  The
connection is read from GLYCOSHAPE_S3_ENDPOINT, GLYCOSHAPE_S3_REGION,
GLYCOSHAPE_S3_ACCESS_KEY_ID and GLYCOSHAPE_S3_SECRET_ACCESS_KEY, the place
from GLYCOSHAPE_S3_STATIC_BUCKET, GLYCOSHAPE_S3_STATIC_PREFIX and
GLYCOSHAPE_S3_STAGING_PREFIX (environment or the .env file of the repository).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib import static_publish
from lib.storage import S3Settings, create_s3_backend


def _say(message: str) -> None:
    print(message, flush=True)


def _publish(args: argparse.Namespace, backend: Any, settings: S3Settings, prefix: str) -> int:
    if not args.static_dir.is_dir():
        raise SystemExit(f"Static directory does not exist: {args.static_dir}")
    report = static_publish.publish(
        args.static_dir,
        backend,
        prefix,
        staging_prefix=args.staging_prefix or settings.staging_prefix,
        workers=args.workers,
        delete_stale=args.delete_stale,
        dry_run=args.dry_run,
        log=_say,
    )
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print("dry run: nothing was sent", flush=True)
        return 0
    print(
        f"S3 static publish complete: new={report['new']} changed={report['changed']} "
        f"unchanged={report['unchanged']} manifest_written={report['manifest_written']} remote={prefix}/",
        flush=True,
    )
    if "cleanup_error" in report:   # published, but staging or stale objects are still there
        print(f"clean-up failed: {report['cleanup_error']}; run the command again", file=sys.stderr)
        return 1
    return 0


def _check(args: argparse.Namespace, backend: Any, settings: S3Settings, prefix: str) -> int:
    report = static_publish.check_published(backend, prefix, quick=args.quick, workers=args.workers, log=_say)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for error in report["errors"][:50]:
        print(f"ERROR {error}", file=sys.stderr)
    if len(report["errors"]) > 50:
        print(f"ERROR ... and {len(report['errors']) - 50} more", file=sys.stderr)
    how = "size and SHA-256 metadata" if args.quick else "downloaded, size and SHA-256"
    print(
        f"{prefix}/: {'valid' if report['ok'] else 'NOT VALID'}; {report['files']} files "
        f"({report['bytes'] / 1e6:.1f} MB) compared with sync_manifest.json ({how}); "
        f"errors={len(report['errors'])} not-in-manifest={len(report.get('unlisted', []))}",
        flush=True,
    )
    return 0 if report["ok"] else 1


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (("publish", _publish), ("check", _check)):
        command = commands.add_parser(name)
        command.set_defaults(function=function)
        command.add_argument("--bucket", help="Bucket (default: GLYCOSHAPE_S3_STATIC_BUCKET).")
        command.add_argument("--prefix", help="Published key prefix (default: GLYCOSHAPE_S3_STATIC_PREFIX, static).")
        command.add_argument("--workers", type=int, default=16, help="Parallel requests (default 16).")
        command.add_argument("--report", type=Path, help="Write the report of the run to this JSON file.")
        if name == "publish":
            command.add_argument("--static-dir", type=Path, default=Path("static"))
            command.add_argument(
                "--staging-prefix",
                help="Key prefix for changed files before they replace the published ones "
                "(default: GLYCOSHAPE_S3_STAGING_PREFIX, publish-staging).")
            command.add_argument(
                "--delete-stale", action="store_true",
                help="After the manifest is written, delete objects below the prefix that it does not list.")
            command.add_argument("--dry-run", action="store_true", help="Show what would be sent; change nothing.")
        else:
            command.add_argument(
                "--quick", action="store_true",
                help="Compare size and SHA-256 metadata of each object instead of downloading it.")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        settings = S3Settings.from_env()
        backend = create_s3_backend(
            settings, bucket=args.bucket or settings.static_bucket,
            max_connections=args.workers * settings.transfer_concurrency + 8)
        return args.function(args, backend, settings, (args.prefix or settings.static_prefix).strip("/"))
    except Exception as exc:
        expected = (ValueError, RuntimeError, OSError, static_publish.PublishError)
        if isinstance(exc, expected) or type(exc).__module__.startswith(("botocore", "boto3", "s3transfer")):
            raise SystemExit(f"{Path(__file__).name} {args.command}: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
