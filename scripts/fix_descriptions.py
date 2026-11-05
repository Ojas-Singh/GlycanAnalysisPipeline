#!/usr/bin/env python3
"""Take the submission comments, a 0 K temperature and wrong Oxford names out of the published metadata.

Until November 2026 the generated description of a glycan (``search_meta.description``,
lib/glymeta.py) ended with the comment of its submission ("Auto upload from GOTW",
"Ojas code", a batch name such as "Donegal") and said "0.0K" when the temperature
was missing; lib/oxford.json gave a few glycans an Oxford name that names a GalNAc or a
sulfate they lack ("A1GalNAc1" for A1).  lib/glymeta and lib/name_utils no longer do;
this script repairs the published database in one publication:

    fix_descriptions.py prepare --tree TREE --report prepare.json
    fix_descriptions.py publish --tree TREE --base-hash <global_hash of the stored manifest> [--dry-run]

TREE is an exact copy of the published database (``glycoshape-ce sync-gap`` into it
first).  ``prepare`` keeps the stored ``sync_manifest.json`` as
``sync_manifest.stored.json``, rewrites the description in ``data.json`` of every
glycan whose generated description has a comment or a 0 K temperature (the rest of the
file stays the same bytes; a curated description is never touched), builds
``GSxxxxx.zip`` of those glycans and the root files again with the finalization
(``GLYCOSHAPE.json``, ``GlycoShape.zip``, ``GLYCOSHAPE_RDF.ttl``) and writes a new
``sync_manifest.json``.  It then checks that nothing else changed: every other glycan
and root file has the hashes of the stored manifest.

``publish`` takes the publishers' lock (lib.entry_publish), refuses when the stored
manifest is no longer the one TREE was copied from, and sends what changed
(lib.static_publish: new files, staged replacements, the switch, the manifest last).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MANIFEST = "sync_manifest.json"
STORED = "sync_manifest.stored.json"
ROOT_FILES_REBUILT = {"GLYCOSHAPE.json", "GlycoShape.zip", "GLYCOSHAPE_RDF.ttl"}
_COMPOSED = re.compile(r"Composed of [^.]*\.")
_ZERO_TEMPERATURE = re.compile(r", 0(?:\.0+)?K\.")
_ONLY_ZERO_TEMPERATURE = re.compile(r" ?Simulated with 0(?:\.0+)?K\.")
_ALSO_KNOWN = re.compile(r" Also known as: [^.]*\.")


def repaired(description: str, glycan_id: str) -> Optional[str]:
    """The generated description without the comment after it and without a 0 K temperature.

    None when there is nothing to repair, or when the description is a curated one
    (it does not start the way every generated one does).
    """
    if not description.startswith(f"{glycan_id} is a glycan with IUPAC notation "):
        return None
    text = description
    composed = _COMPOSED.search(text)
    if composed:
        text = text[: composed.end()]
    text = _ZERO_TEMPERATURE.sub(".", text)
    text = _ONLY_ZERO_TEMPERATURE.sub("", text)
    return text if text != description else None


def _data_bytes(data: Dict[str, Any]) -> bytes:
    """``data.json`` as the static stage writes it."""
    return json.dumps(data, indent=4, ensure_ascii=False).encode("utf-8")


def _comparable(entry: Dict[str, Any]) -> Dict[str, Any]:
    """A manifest entry without the modification times (no hash covers them)."""
    files = {path: {k: v for k, v in info.items() if k != "mtime"} for path, info in (entry.get("files") or {}).items()}
    return {"hash": entry.get("hash"), "files": files}


def prepare(tree: Path, report_path: Optional[Path]) -> int:
    from lib.glybake import GlycoShapeBaker
    from lib.glymeta import GlycanMetadataProcessor
    from lib.name_utils import oxford_fits
    from lib.static_delta import write_rdf
    from lib.storage import local_storage

    manifest_path = tree / MANIFEST
    stored_path = tree / STORED
    if not stored_path.exists():
        shutil.copyfile(manifest_path, stored_path)
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    processor = GlycanMetadataProcessor(".", pb_client=None)
    changed: Dict[str, Dict[str, str]] = {}
    left: List[str] = []
    for glycan_id in sorted(stored["glycans"]):
        path = tree / glycan_id / "data.json"
        raw = path.read_bytes()
        data = json.loads(raw)
        if _data_bytes(data) != raw:
            raise SystemExit(f"{glycan_id}/data.json is not written the way this script writes it: stopped")
        search = data.get("search_meta") or {}
        archetype = data.get("archetype") or {}
        old = str(search.get("description") or "")
        new = repaired(old, glycan_id)
        # An Oxford name that names a GalNAc or a sulfate the glycan lacks (lib/oxford.json) is dropped,
        # from the names and from the "Also known as" of a generated description.
        dropped = archetype.get("oxford")
        if dropped and not oxford_fits(dropped, archetype.get("iupac")):
            archetype["oxford"] = None
            search["common_names"] = [name for name in search.get("common_names") or [] if name != dropped]
            if old.startswith(f"{glycan_id} is a glycan with IUPAC notation "):
                names = search["common_names"]
                new = _ALSO_KNOWN.sub(f" Also known as: {', '.join(names)}." if names else "", new or old)
        else:
            dropped = None
        if new is None and dropped is None:
            if " 0.0K" in old or (old.startswith(f"{glycan_id} is a glycan") and not _COMPOSED.search(old)):
                left.append(glycan_id)
            continue
        search["description"] = new if new is not None else old
        path.write_bytes(_data_bytes(data))
        generated = processor._generate_description(data, {})
        changed[glycan_id] = {"before": old, "after": search["description"],
                              "as_generated_now": str(generated == search["description"]).lower(),
                              **({"oxford_dropped": dropped} if dropped else {})}
    print(f"{len(changed)} descriptions repaired; {len(left)} generated ones not understood: {left[:10]}", flush=True)

    with local_storage():
        baker = GlycoShapeBaker(tree)
        for glycan_id in changed:
            if not baker.create_individual_glycan_archive(tree / glycan_id):
                raise SystemExit(f"{glycan_id}.zip could not be written")
        baker.create_consolidated_json()
        baker.create_master_archive()
        write_rdf(tree / "GLYCOSHAPE.json", tree / "GLYCOSHAPE_RDF.ttl")
        # The copy of the stored manifest is not a file of the database.
        held = stored_path.rename(tree.parent / f"{tree.name}.{STORED}")
        try:
            baker.create_sync_manifest()
        finally:
            held.rename(stored_path)
    new_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Nothing else may have changed.
    problems: List[str] = []
    if set(new_manifest["glycans"]) != set(stored["glycans"]):
        problems.append("the glycans of the new manifest are not those of the stored one")
    for glycan_id, entry in stored["glycans"].items():
        after = new_manifest["glycans"].get(glycan_id)
        if after is None:
            continue
        before_files, after_files = _comparable(entry)["files"], _comparable(after)["files"]
        differing = sorted(p for p in set(before_files) | set(after_files) if before_files.get(p) != after_files.get(p))
        # Paths are relative to the folder of the glycan.
        expected = {"data.json", f"{glycan_id}.zip"} if glycan_id in changed else set()
        if set(differing) - expected:
            problems.append(f"{glycan_id}: unexpected changes {sorted(set(differing) - expected)}")
        if glycan_id in changed and not differing:
            problems.append(f"{glycan_id}: repaired but its files did not change")
    root_before = {name: {k: v for k, v in info.items() if k != "mtime"} for name, info in stored["root_files"].items()}
    root_after = {name: {k: v for k, v in info.items() if k != "mtime"} for name, info in new_manifest["root_files"].items()}
    root_changed = sorted(n for n in set(root_before) | set(root_after) if root_before.get(n) != root_after.get(n))
    if set(root_changed) - ROOT_FILES_REBUILT:
        problems.append(f"unexpected root file changes: {sorted(set(root_changed) - ROOT_FILES_REBUILT)}")

    report = {
        "stored_global_hash": stored["global_hash"], "new_global_hash": new_manifest["global_hash"],
        "repaired": len(changed), "root_files_changed": root_changed, "not_understood": left,
        "problems": problems, "descriptions": changed,
    }
    if report_path:
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"stored global_hash {stored['global_hash']}\nnew global_hash    {new_manifest['global_hash']}\n"
          f"root files changed: {root_changed}", flush=True)
    if problems:
        print("REFUSED:\n  " + "\n  ".join(problems[:20]), flush=True)
        return 2
    print("ready to publish with --base-hash " + stored["global_hash"], flush=True)
    return 0


def publish(tree: Path, base_hash: str, dry_run: bool, workers: int) -> int:
    from lib import entry_publish, static_publish

    target = entry_publish.Target.from_settings(workers=workers)
    store = entry_publish.Store(target.static, entry_publish.new_stats(), dry_run=dry_run)
    lock = None if dry_run else entry_publish.acquire_lock(
        target, store, entry_publish.new_owner(), "fix-descriptions", "all")
    try:
        stored = json.loads(store.read(target.key(MANIFEST)))
        if stored.get("global_hash") != base_hash:
            print(f"REFUSED: the stored database is {stored.get('global_hash')}, not {base_hash}: sync and prepare again",
                  flush=True)
            return 2
        if entry_publish.read_pending(target, store) is not None:
            print("REFUSED: an entry publication is pending (publish_entry.py recover)", flush=True)
            return 2
        report = static_publish.publish(tree, target.static, target.static_prefix,
                                        staging_prefix=target.staging_prefix, workers=workers, dry_run=dry_run,
                                        log=lambda m: print(m, flush=True))
        print(json.dumps({k: v for k, v in report.items() if not isinstance(v, (list, dict))}), flush=True)
        return 0 if dry_run or "cleanup_error" not in report else 1
    finally:
        entry_publish.release_lock(target, store, lock)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("prepare")
    one.add_argument("--tree", type=Path, required=True)
    one.add_argument("--report", type=Path)
    two = sub.add_parser("publish")
    two.add_argument("--tree", type=Path, required=True)
    two.add_argument("--base-hash", required=True)
    two.add_argument("--dry-run", action="store_true")
    two.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        return prepare(args.tree.resolve(), args.report)
    return publish(args.tree.resolve(), args.base_hash, args.dry_run, args.workers)


if __name__ == "__main__":
    sys.exit(main())
