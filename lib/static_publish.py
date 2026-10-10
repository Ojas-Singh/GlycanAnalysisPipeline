"""Publish the static database to S3 storage under the ``sync_manifest.json`` contract.

The website synchronises its database from ``<prefix>/sync_manifest.json``
(reader: ``glycoshape-api/src/gap_sync.rs``).  It fetches the manifest, and
unless the ``global_hash`` is the one it already has, downloads every root
file and every file of each glycan whose ``hash`` changed, and compares size
and SHA-256 of each download with the manifest.  One mismatch or one missing
object fails its whole run (its own database is not touched then).

So the published prefix is valid when every file the published manifest lists
is there with the listed content.  :func:`publish` keeps that true for as long
as it can and writes the manifest last:

1. *Plan.*  The local manifest is validated with the rules of the reader and
   against the local files.  Every listed file is compared with the storage:
   **same** (size and SHA-256 metadata match), **new** (no such object) or
   **changed** (an object with other content).
2. *Upload.*  New files go to their final keys: no published manifest lists
   them, so no reader fetches them.  Changed files go to a staging prefix
   outside the published prefix.  Until here the published prefix is exactly
   as valid as before; an interruption changes nothing a reader can see.
3. *Switch.*  Changed files are copied from the staging prefix onto their
   final keys inside the storage (no data passes through this machine), so
   the time in which the old manifest describes replaced files is the time of
   those copies, not of the upload.
4. *Manifest.*  The listing is compared with the manifest, then the manifest
   is written, as one object, last.
5. *Clean up.*  The staging objects are deleted, and with ``delete_stale``
   the objects below the prefix that the manifest no longer lists.

Two objects cannot be replaced at once, so step 3 cannot be made invisible:
between the first copy and the manifest a reader that is behind fails its
verification and retries later; a reader that already has the published
``global_hash`` does nothing and is not affected.  Running :func:`publish`
again finishes an interrupted run from where it stopped.

:func:`check_published` reads a published prefix back the way the reader does.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from lib.storage import S3StorageBackend, file_sha256, parallel_map

MANIFEST_NAME = "sync_manifest.json"
#: The reader fails a synchronisation that ends without this file.
REQUIRED_ROOT_FILE = "GLYCOSHAPE.json"
DEFAULT_STAGING_PREFIX = "publish-staging"

Log = Callable[[str], None]


class PublishError(Exception):
    """The static directory cannot be published, or publishing stopped."""


# ---------------------------------------------------------------------------
# The manifest contract (validate_manifest of gap_sync.rs)
# ---------------------------------------------------------------------------
def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdefABCDEF" for c in value)


def _is_count(value: Any) -> bool:
    """A JSON integer that the reader's unsigned fields take (no float, no boolean)."""
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 2 ** 64


def _path_parts(value: Any) -> Optional[List[str]]:
    """Components of a manifest path, or None when the reader refuses it.

    The reader refuses empty and absolute paths, ``..`` and every component
    that starts with a dot.  This is stricter in one way: the path has to be
    written as the object key is, without surrounding blanks, ``\\`` or empty
    components, which the reader would rewrite before using it.
    """
    if not isinstance(value, str) or not value or value != value.strip() or "\\" in value:
        return None
    parts = value.split("/")
    if any(not part or part.startswith(".") for part in parts):
        return None
    return parts


def _file_errors(label: str, entry: Any) -> Tuple[Optional[List[str]], List[str]]:
    if not isinstance(entry, dict):
        return None, [f"{label} is not an object"]
    errors = []
    parts = _path_parts(entry.get("path"))
    if parts is None:
        errors.append(f"unsafe object path {entry.get('path')!r} ({label})")
    if not _is_sha256(entry.get("sha256")):
        errors.append(f"invalid SHA-256 for {entry.get('path')!r}")
    if not _is_count(entry.get("size")):
        errors.append(f"invalid size for {entry.get('path')!r}")
    return parts, errors


def validate_manifest(manifest: Any) -> List[str]:
    """Reasons for which the reader would refuse this manifest; empty when it is valid.

    The rules are those of ``validate_manifest`` and of the manifest types in
    ``gap_sync.rs``: schema version 1 with SHA-256 hashes, a ``global_hash``,
    at least one glycan, root files at the root, glycan identifiers that
    start with ``GS``, a ``file_count`` that is the number of files, and
    every file of a glycan below the folder of that glycan.  Three more rules
    follow from what the reader does with a valid manifest: a path is listed
    once, no listed file is a folder of another, and ``GLYCOSHAPE.json`` is a
    root file.
    """
    if not isinstance(manifest, dict):
        return ["the manifest is not a JSON object"]
    errors: List[str] = []
    version = manifest.get("schema_version")
    if not _is_count(version) or version != 1 or manifest.get("hash_algorithm") != "sha256":
        errors.append("expected schema_version 1 with sha256 hashes")
    root_files, glycans = manifest.get("root_files"), manifest.get("glycans")
    if not isinstance(root_files, dict) or not isinstance(glycans, dict):
        return errors + ["root_files and glycans must be objects"]
    if not _is_sha256(manifest.get("global_hash")) or not glycans:
        errors.append("missing global hash or glycan entries")

    seen: Dict[str, str] = {}

    def note(parts: Optional[List[str]], label: str) -> None:
        if parts is not None:
            path = "/".join(parts)
            if path in seen:
                errors.append(f"{path} is listed twice ({seen[path]} and {label})")
            seen[path] = label

    for name, entry in root_files.items():
        parts, problems = _file_errors(f"root file {name!r}", entry)
        errors.extend(problems)
        if parts is not None and len(parts) != 1:
            errors.append(f"root file is not at the database root: {entry.get('path')}")
        note(parts, f"root file {name!r}")
    for glycan_id, glycan in glycans.items():
        files = glycan.get("files") if isinstance(glycan, dict) else None
        if (
            not isinstance(files, dict)
            or not glycan_id.startswith("GS")
            or "/" in glycan_id
            or not _is_sha256(glycan.get("hash"))
            or not _is_count(glycan.get("file_count"))
            or glycan["file_count"] != len(files)
            or not files
        ):
            errors.append(f"invalid glycan manifest entry {glycan_id!r}")
            continue
        for name, entry in files.items():
            parts, problems = _file_errors(f"{glycan_id} file {name!r}", entry)
            errors.extend(problems)
            if parts is not None and parts[0] != glycan_id:
                errors.append(f"{entry.get('path')} is outside {glycan_id}")
            note(parts, f"{glycan_id} file {name!r}")
    for path in seen:   # the reader writes every path into one directory tree
        parts = path.split("/")
        for depth in range(1, len(parts)):
            if "/".join(parts[:depth]) in seen:
                errors.append(f"{'/'.join(parts[:depth])} is listed as a file and is a folder of {path}")
    if REQUIRED_ROOT_FILE not in seen:
        errors.append(f"{REQUIRED_ROOT_FILE} is not a root file; the reader fails a synchronisation without it")
    return errors


def manifest_files(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Path -> ``{"size", "sha256"}`` of every file a valid manifest lists (SHA-256 in lower case)."""
    entries = list(manifest["root_files"].values())
    for glycan in manifest["glycans"].values():
        entries.extend(glycan["files"].values())
    return {entry["path"]: {"size": entry["size"], "sha256": entry["sha256"].lower()} for entry in entries}


def _strict_json(payload: bytes) -> Any:
    """Parse JSON the way the reader's parser does, which is stricter than ``json.loads`` on bytes.

    UTF-8 without a byte order mark; no ``NaN`` or ``Infinity``; no key twice
    in one object (the reader refuses a repeated field); no lone surrogate.
    """
    def no_constant(name: str) -> Any:
        raise ValueError(f"{name} is not JSON")

    def no_duplicates(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("an object has the same key twice")
        return result

    text = payload.decode("utf-8")
    if text.startswith("﻿"):
        raise ValueError("starts with a byte order mark")
    value = json.loads(text, parse_constant=no_constant, object_pairs_hook=no_duplicates)
    json.dumps(value, ensure_ascii=False).encode("utf-8")   # a lone surrogate cannot be encoded
    return value


def parse_manifest(payload: bytes, where: str) -> Dict[str, Any]:
    """A valid manifest from its bytes; raises :class:`PublishError` listing what the reader would refuse."""
    try:
        manifest = _strict_json(payload)
    except ValueError as exc:
        raise PublishError(f"{where} is not JSON: {exc}") from exc
    errors = validate_manifest(manifest)
    if errors:
        shown = "; ".join(errors[:5]) + (f"; and {len(errors) - 5} more" if len(errors) > 5 else "")
        raise PublishError(f"{where} would be refused by the reader: {shown}")
    return manifest


# ---------------------------------------------------------------------------
# Publish
# ---------------------------------------------------------------------------
def _check_local_files(static_dir: Path, files: Dict[str, Dict[str, Any]]) -> None:
    wrong = [
        path for path, recorded in files.items()
        if not (static_dir / path).is_file() or (static_dir / path).stat().st_size != recorded["size"]
    ]
    if wrong:
        shown = ", ".join(sorted(wrong)[:5]) + (f" and {len(wrong) - 5} more" if len(wrong) > 5 else "")
        raise PublishError(
            f"{len(wrong)} file(s) of {MANIFEST_NAME} are missing in {static_dir} or have another size ({shown}); "
            "rebuild the static output (main.py --mode static-only) before publishing"
        )


def _send(backend: S3StorageBackend, local_path: Path, key: str, recorded: Dict[str, Any]) -> None:
    """Upload one file after confirming that it is the content the manifest describes."""
    if file_sha256(local_path) != (recorded["size"], recorded["sha256"]):
        raise PublishError(
            f"{local_path} does not match the size and SHA-256 of {MANIFEST_NAME}; "
            "rebuild the static output (main.py --mode static-only) before publishing"
        )
    backend.upload_file(local_path, key, sha256=recorded["sha256"])


def _stored_as_recorded(backend: S3StorageBackend, key: str, recorded: Dict[str, Any]) -> bool:
    head = backend.head(key)
    return head is not None and head["size"] == recorded["size"] and head["sha256"] == recorded["sha256"]


def publish(
    static_dir: Path,
    backend: S3StorageBackend,
    prefix: str = "static",
    staging_prefix: str = DEFAULT_STAGING_PREFIX,
    workers: int = 8,
    delete_stale: bool = False,
    dry_run: bool = False,
    log: Log = print,
) -> Dict[str, Any]:
    """Publish ``static_dir`` under ``<prefix>/``; see the module documentation for the order.

    Only what the local ``sync_manifest.json`` lists is published, and of
    that only what the storage does not already hold.  Raises
    :class:`PublishError` before anything is sent when the manifest would be
    refused by the reader or a listed file is missing or has another size.
    A file is hashed when it is sent: one whose content is not that of the
    manifest stops the upload (new files sent before it stay under keys that
    no published manifest lists).  Also raises when a later step fails;
    errors of the storage while the plan is made are raised as they are.  A
    dry run only reads.  Returns the counts of the plan and what was done;
    ``cleanup_error`` is set when the last step failed after the manifest
    was written.
    """
    static_dir = Path(static_dir)
    prefix = prefix.strip("/")
    staging_prefix = staging_prefix.strip("/")
    if not prefix or not staging_prefix:
        raise PublishError("the published prefix and the staging prefix must not be empty")
    if (staging_prefix + "/").startswith(prefix + "/") or (prefix + "/").startswith(staging_prefix + "/"):
        raise PublishError("the staging prefix must be outside the published prefix")
    manifest_path = static_dir / MANIFEST_NAME
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise PublishError(f"cannot read {manifest_path}: {exc}") from exc
    manifest = parse_manifest(manifest_bytes, str(manifest_path))
    files = manifest_files(manifest)
    _check_local_files(static_dir, files)

    def key_of(path: str) -> str:
        return f"{prefix}/{path}"

    def staging_key(recorded: Dict[str, Any]) -> str:
        return f"{staging_prefix}/{prefix}/{recorded['sha256']}"

    manifest_key = key_of(MANIFEST_NAME)
    listing = {item["key"]: item["size"] for item in backend.list_objects(prefix + "/")}
    # Without a published manifest no reader can be reading: replace in place.
    in_place = manifest_key not in listing

    def classify(path: str) -> Tuple[str, str]:
        key, recorded = key_of(path), files[path]
        if key not in listing:
            return path, "new"
        if listing[key] == recorded["size"] and _stored_as_recorded(backend, key, recorded):
            return path, "same"
        return path, "new" if in_place else "changed"

    state = dict(parallel_map(classify, sorted(files), workers))
    new = sorted(path for path, kind in state.items() if kind == "new")
    changed = sorted(path for path, kind in state.items() if kind == "changed")
    stale = sorted(set(listing) - {key_of(path) for path in files} - {manifest_key})
    manifest_same = manifest_key in listing and backend.read_binary(manifest_key) == manifest_bytes
    report: Dict[str, Any] = {
        "dry_run": dry_run,
        "prefix": prefix,
        "glycans": len(manifest["glycans"]),
        "files": len(files),
        "new": len(new),
        "changed": len(changed),
        "unchanged": len(files) - len(new) - len(changed),
        "bytes_to_upload": sum(files[path]["size"] for path in new + changed),
        "stale": len(stale),
        "stale_deleted": 0,
        "manifest_written": False,
    }
    log(
        f"{prefix}/: {report['files']} files of {report['glycans']} glycans; {report['new']} new, "
        f"{report['changed']} changed, {report['unchanged']} unchanged, "
        f"{report['bytes_to_upload'] / 1e6:.1f} MB to upload; {report['stale']} stored objects are not in the manifest"
    )
    if dry_run:
        return report

    def upload_new(path: str) -> None:
        _send(backend, static_dir / path, key_of(path), files[path])

    def upload_changed(path: str) -> None:
        if not _stored_as_recorded(backend, staging_key(files[path]), files[path]):   # kept from an interrupted run
            _send(backend, static_dir / path, staging_key(files[path]), files[path])

    def switch(path: str) -> None:
        recorded = files[path]
        backend.copy(staging_key(recorded), key_of(path), sha256=recorded["sha256"], size=recorded["size"])

    if not new and not changed and manifest_same:
        log(f"{manifest_key} is already the published manifest")
    else:
        # 2. Upload.  Nothing that a reader of the published manifest fetches is touched.
        # Files with the same content share a staging object; it is sent once.
        to_stage = list({files[path]["sha256"]: path for path in changed}.values())
        try:
            for done, _ in enumerate(parallel_map(upload_new, new, workers), start=1):
                if done % 500 == 0 or done == len(new):
                    log(f"uploaded {done}/{len(new)} new files")
            for done, _ in enumerate(parallel_map(upload_changed, to_stage, workers), start=1):
                if done % 500 == 0 or done == len(to_stage):
                    log(f"staged {done}/{len(to_stage)} changed files")
        except PublishError as exc:   # a local file is not what the manifest says
            raise PublishError(f"{exc}; the published prefix has not changed for readers") from exc
        except (Exception, KeyboardInterrupt) as exc:
            raise PublishError(
                f"upload stopped ({type(exc).__name__}: {exc}); the published prefix has not changed for readers; "
                "run the command again to continue"
            ) from exc

        # 3. Switch, 4. manifest.
        try:
            for _ in parallel_map(switch, changed, workers):
                pass
            if changed:
                log(f"replaced {len(changed)} changed files from the staging prefix")
            listing = {item["key"]: item["size"] for item in backend.list_objects(prefix + "/")}
            wrong = sorted(path for path in files if listing.get(key_of(path)) != files[path]["size"])
            if wrong:
                raise PublishError(f"{len(wrong)} file(s) are not stored with the size of the manifest, first {wrong[0]}")
            backend.write_binary(manifest_key, manifest_bytes)
        except (Exception, KeyboardInterrupt) as exc:
            raise PublishError(
                f"publishing stopped before the manifest was written ({type(exc).__name__}: {exc}); "
                + (f"{len(changed)} changed files were being replaced, so the published manifest may describe "
                   "content that has been replaced: run the command again to finish" if changed else
                   "the published prefix has not changed for readers; run the command again to continue")
            ) from exc
        report["manifest_written"] = True
        log(f"wrote {manifest_key} (global_hash {manifest['global_hash'][:16]}...)")

    # 5. Clean up.  A failure here leaves only objects that the manifest does not list.
    try:
        doomed = [item["key"] for item in backend.list_objects(f"{staging_prefix}/{prefix}/")]
        doomed += stale if delete_stale else []
        for _ in parallel_map(backend.unlink, doomed, workers):
            pass
        if delete_stale:
            report["stale_deleted"] = len(stale)
            log(f"deleted {len(stale)} stored objects that the manifest does not list")
    except Exception as exc:
        report["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        log(f"clean-up did not finish ({report['cleanup_error']}); the published prefix is complete, run again to retry")
    return report


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------
def check_published(
    backend: S3StorageBackend,
    prefix: str = "static",
    quick: bool = False,
    workers: int = 8,
    log: Log = print,
) -> Dict[str, Any]:
    """Read a published prefix back and validate it against its manifest as the reader does.

    The manifest is fetched and validated (:func:`validate_manifest`); then
    every file it lists is downloaded as a stream and its size and SHA-256
    are compared with the manifest, which is what the reader does with each
    file it fetches.  ``quick`` compares the size and the SHA-256 metadata of
    each object instead of downloading it.  Returns ``{"ok", "errors", ...}``.
    """
    prefix = prefix.strip("/")
    manifest_key = f"{prefix}/{MANIFEST_NAME}"
    report: Dict[str, Any] = {"ok": False, "errors": [], "prefix": prefix, "quick": quick, "files": 0, "bytes": 0}
    errors: List[str] = report["errors"]
    try:
        manifest = parse_manifest(backend.read_binary(manifest_key), manifest_key)
    except FileNotFoundError:
        errors.append(f"{manifest_key} does not exist")
        return report
    except PublishError as exc:
        errors.append(str(exc))
        return report
    files = manifest_files(manifest)
    report.update(glycans=len(manifest["glycans"]), global_hash=manifest["global_hash"])

    def verify(path: str) -> Optional[str]:
        recorded, key = files[path], f"{prefix}/{path}"
        if quick:
            head = backend.head(key)
            if head is None:
                return f"{path} is missing"
            if head["sha256"] is None:
                return f"{path} has no SHA-256 metadata; it can only be checked by downloading (without --quick)"
            found = (head["size"], head["sha256"])
        else:
            try:
                found = backend.stream_sha256(key)
            except FileNotFoundError:
                return f"{path} is missing"
        if found != (recorded["size"], recorded["sha256"]):
            return f"download verification failed for {path}"
        return None

    for done, problem in enumerate(parallel_map(verify, sorted(files), workers), start=1):
        if problem:
            errors.append(problem)
        if done % 1000 == 0:
            log(f"checked {done}/{len(files)} files")
    errors.sort()
    expected = {f"{prefix}/{path}" for path in files} | {manifest_key}
    report.update(
        files=len(files),
        bytes=sum(recorded["size"] for recorded in files.values()),
        unlisted=sorted(item["key"] for item in backend.list_objects(prefix + "/") if item["key"] not in expected),
        ok=not errors,
    )
    return report
