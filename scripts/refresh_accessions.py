#!/usr/bin/env python3
"""Look the GlyTouCan accessions of published entries up again (lib/accessions.py).

Which entries: the IDs given (``--id``, all three variants of each), else
every entry that lacks an archetype, alpha or beta accession (``--all``:
every entry, to find accessions that differ from what GlyTouCan has).

Where they are read from: a copy of the static database (``--static-dir``,
its ``GLYCOSHAPE.json``, else its ``GSxxxxx/data.json``), ``data.json`` files
(``--entry``, e.g. downloaded from ``/api/glycan/<ID>`` of the site; they
take precedence for their IDs), or else the storage of the environment
(``GLYCOSHAPE_S3_*``: ``GLYCOSHAPE.json`` of the published database, or the
``data.json`` of each ``--id``). Only read.

A dry run by default: the lookups are made and the report says what would be
set; nothing is written but ``--report``. ``--write-changes DIR`` writes one
``update-metadata`` changes record per entry that would change
(``DIR/GSxxxxx.changes.json``), the input of::

    scripts/publish_entry.py update-metadata --changes DIR/GSxxxxx.changes.json --dry-run

which shows what the new version of the entry would be, and publishes it
without ``--dry-run`` (with ``--expect-hash`` of that dry run).

Examples::

    python scripts/refresh_accessions.py --id GS00094 --report accessions.json
    python scripts/refresh_accessions.py --static-dir ~/gap-work/static-final --entry GS00094.json --report dry-run.json
    python scripts/refresh_accessions.py --write-changes changes/ --author "the curator"

Exit status: 0 done; 1 the entries could not be read; 2 bad arguments.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib import accessions  # noqa: E402

_ID = re.compile(r"GS\d{5}\Z")
INVENTORY = "GLYCOSHAPE.json"


def _id_of(path: Path, data: dict) -> str:
    archetype = data.get("archetype") if isinstance(data.get("archetype"), dict) else {}
    for candidate in (archetype.get("ID"), path.stem.split(".")[0], path.parent.name):
        if isinstance(candidate, str) and _ID.match(candidate.strip().upper()):
            return candidate.strip().upper()
    raise ValueError(f"{path}: no GlycoShape ID (archetype.ID, or a file named GSxxxxx.json)")


def from_static_dir(folder: Path) -> dict:
    """The entries of a copy of the static database: its inventory, else its data.json files."""
    inventory = folder / INVENTORY
    if inventory.is_file():
        document = json.loads(inventory.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError(f"{inventory} is not an inventory")
        return {gid: entry for gid, entry in document.items() if _ID.match(gid) and isinstance(entry, dict)}
    entries = {}
    for path in sorted(folder.glob("GS*/data.json")):
        if _ID.match(path.parent.name):
            entries[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
    if not entries:
        raise ValueError(f"{folder} has neither {INVENTORY} nor GSxxxxx/data.json")
    return entries


def from_storage(ids) -> dict:
    """The entries of the published database in the storage: the data.json of each ID, else the inventory."""
    from lib import entry_publish
    from lib.storage import S3Settings

    target = entry_publish.Target.from_settings(S3Settings.from_env(), workers=2)
    if ids:
        entries = {}
        for gid in ids:
            try:
                entries[gid] = json.loads(target.static.read_binary(target.key(f"{gid}/data.json")))
            except FileNotFoundError:
                continue
        return entries
    document = json.loads(target.static.read_binary(target.key(INVENTORY)))
    return {gid: entry for gid, entry in document.items() if _ID.match(gid) and isinstance(entry, dict)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--id", dest="ids", action="append", default=[], metavar="GSxxxxx",
                        help="an entry to look up (repeatable); default: every entry that lacks an accession")
    parser.add_argument("--all", action="store_true", help="every entry, also those with all three accessions")
    parser.add_argument("--only-missing", action="store_true",
                        help="never change a published accession, also where GlyTouCan has another one for the WURCS")
    parser.add_argument("--static-dir", type=Path, help="a copy of the static database (GLYCOSHAPE.json or GSxxxxx/data.json)")
    parser.add_argument("--entry", action="append", type=Path, default=[], metavar="FILE",
                        help="a data.json of an entry (repeatable; its ID from archetype.ID or the file name)")
    parser.add_argument("--report", type=Path, help="write the report (schema glycoshape-accession-refresh) here")
    parser.add_argument("--write-changes", type=Path, metavar="DIR",
                        help="write the changes record of each entry that would change to DIR/GSxxxxx.changes.json")
    parser.add_argument("--author", default="", help="who asks (the author of the changes records)")
    parser.add_argument("--offline", action="store_true", help="make no request (every lookup fails as 'no answer')")
    parser.add_argument("--timeout", type=float, default=None, help="cap of the timeout of every request, seconds")
    args = parser.parse_args(argv)

    ids = []
    for item in args.ids:
        gid = item.strip().upper()
        if not _ID.match(gid):
            parser.error(f"not a GlycoShape ID: {item!r}")
        ids.append(gid)
    if ids and args.all:
        parser.error("--id and --all exclude each other")

    try:
        if args.static_dir:
            entries, source = from_static_dir(args.static_dir), str(args.static_dir)
        elif args.entry:
            entries, source = {}, "data.json files"
        else:
            entries, source = from_storage(ids), "the storage"
        for path in args.entry:
            data = json.loads(path.read_text(encoding="utf-8"))
            entries[_id_of(path, data)] = data
        if args.entry and args.static_dir:
            source += f" and {len(args.entry)} data.json file(s)"
    except Exception as error:   # the storage's errors name the operation and the key, never a credential
        print(f"the entries could not be read: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    report = accessions.refresh(entries, ids=ids or None, every=args.all, only_missing=args.only_missing,
                                author=args.author, source=source, offline=True if args.offline else None,
                                timeout=args.timeout)
    written = []
    if args.write_changes:
        args.write_changes.mkdir(parents=True, exist_ok=True)
        for row in report["entries"]:
            if row["changes"]:
                path = args.write_changes / f"{row['glycoshape_id']}.changes.json"
                path.write_text(json.dumps(row["changes"], ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
                written.append(str(path))
        report["dry_run"] = False
    report["written"] = written
    for line in accessions.describe(report):
        print(line)
    if written:
        print(f"wrote {len(written)} changes record(s) to {args.write_changes}")
    elif not args.write_changes and report["summary"]["entries_with_changes"]:
        print("dry run: no changes record written (--write-changes DIR writes them)")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_bytes(accessions.encode(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
