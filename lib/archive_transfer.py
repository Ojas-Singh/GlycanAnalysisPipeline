"""Transfer of the raw-trajectory archive (lib/archive.py) to and from S3 storage.

An archive entry lives under ``<prefix>/<GLYCAM name>/`` (``raw/...`` by
default) and the index of the whole archive at ``<prefix>/index.json``.  The
key is built from the GLYCAM name in the entry's ``manifest.json``, never from
the name of the local folder, which is shortened for long names.

Upload (:func:`upload_archive`)
    * an object is skipped when it exists with the size of the manifest and
      its SHA-256 metadata equals the SHA-256 of the manifest;
    * a file is hashed before it is sent and must match its manifest;
    * the files of an entry go first and its ``manifest.json`` last, and the
      stored manifest of an entry is deleted before any of its files is
      replaced, so an entry that has a manifest in the storage is complete;
    * afterwards the listing of the prefix is compared with ``index.json``
      (keys and sizes) and the stored manifest of every entry with the
      SHA-256 the index records, and ``index.json`` is uploaded only when
      every entry it lists is there.

Download (:func:`download_archive`)
    * entries are named by GLYCAM name, or all entries of the stored index
      (which then pins the manifest of each);
    * every file is written to a staging folder, hashed and compared with the
      manifest; the entry folder (``archive.entry_folder_name``) appears only
      when all of it verified, so a folder with a manifest is complete and
      the pipeline can process it (``main.py --data-dir <dir>``).

Both are resumable: run the same command again.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from lib import archive
from lib.storage import S3_MAX_KEY_BYTES, S3StorageBackend, file_sha256, parallel_map

#: Files an entry may hold besides its manifest.
ENTRY_FILES = (
    archive.TRAJECTORY_FILE,
    archive.TOPOLOGY_FILE,
    archive.MOL2_FILE,
    archive.INFO_FILE,
    archive.NAME_FILE,
)
REQUIRED_ENTRY_FILES = (archive.TRAJECTORY_FILE, archive.TOPOLOGY_FILE, archive.MOL2_FILE)
PART_SUFFIX = ".part"

Log = Callable[[str], None]


class TransferError(Exception):
    """An entry cannot be transferred, or does not verify."""


def entry_key_prefix(prefix: str, glycam_name: str) -> str:
    """Object key prefix of an entry: ``<prefix>/<GLYCAM name>/``."""
    archive.object_key_prefix(glycam_name)   # refuses names that cannot be a key
    key_prefix = f"{prefix.strip('/')}/{glycam_name}/"
    longest = max(len(name) for name in ENTRY_FILES + (archive.MANIFEST_FILE,))
    if len(key_prefix.encode("utf-8")) + longest > S3_MAX_KEY_BYTES:
        raise archive.NamingError(f"object key for {glycam_name[:60]}... exceeds {S3_MAX_KEY_BYTES} bytes")
    return key_prefix


def _run(function: Callable[[Any], Dict[str, Any]], items: Sequence[Any], workers: int, log: Log) -> List[Dict[str, Any]]:
    """``function(item)`` for every item on ``workers`` threads; one progress line each."""
    rows: List[Dict[str, Any]] = []
    for row in parallel_map(function, items, workers):
        rows.append(row)
        label = row["glycam_name"] if len(row["glycam_name"]) <= 70 else row["glycam_name"][:67] + "..."
        detail = row.get("error") or f"{len(row['transferred'])} file(s), {row['bytes'] / 1e6:.1f} MB"
        log(f"[{len(rows)}/{len(items)}] {row['status']:10s} {label}: {detail}")
    rows.sort(key=lambda row: row["glycam_name"])
    return rows


def _write_file(path: Path, payload: bytes) -> None:
    with open(path, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter(row["status"] for row in rows)
    return {
        "counts": dict(sorted(counts.items())),
        "bytes": sum(row["bytes"] for row in rows),
        "failed": [row for row in rows if row["status"] == "failed"],
        "entries": rows,
    }


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------
def parse_index(payload: bytes, where: str) -> Dict[str, Any]:
    """``index.json`` as written by ``archive.write_index``; refuses anything else."""
    try:
        index = json.loads(payload)
    except ValueError as exc:
        raise TransferError(f"{where} is not JSON: {exc}") from exc
    if not isinstance(index, dict) or index.get("schema") != archive.INDEX_SCHEMA:
        raise TransferError(f"{where} is not a {archive.INDEX_SCHEMA} file")
    if index.get("schema_version") != archive.SCHEMA_VERSION or not isinstance(index.get("records"), list):
        raise TransferError(f"{where} has schema version {index.get('schema_version')!r}, expected {archive.SCHEMA_VERSION}")
    for record in index["records"]:
        if not isinstance(record, dict) or not isinstance(record.get("files"), dict) or not all(
            isinstance(record.get(field), str) for field in ("glycam_name", "folder", "source_folder")
        ):
            raise TransferError(f"{where} has a record without glycam_name, folder, source_folder and files")
        if archive.MANIFEST_FILE not in record["files"] or not all(
            isinstance(item, dict) and isinstance(item.get("size"), int) and isinstance(item.get("sha256"), str)
            for item in record["files"].values()
        ):
            raise TransferError(f"{where} has a record without size and SHA-256 of its files and its manifest")
    for field in ("glycam_name", "source_folder"):
        repeated = sorted(value for value, count in Counter(r[field] for r in index["records"]).items() if count > 1)
        if repeated:
            raise TransferError(
                f"{where} lists more than one entry with {field} {', '.join(repeated)}; "
                "remove the superseded entry and rebuild the index (scripts/convert_data_to_archive.py)"
            )
    return index


def read_index(archive_dir: Path) -> Tuple[Dict[str, Any], bytes]:
    """The index of a local archive and the bytes it was read from."""
    path = Path(archive_dir) / archive.INDEX_FILE
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise TransferError(f"cannot read {path}: {exc}") from exc
    return parse_index(payload, str(path)), payload


def load_index(archive_dir: Path) -> Dict[str, Any]:
    return read_index(archive_dir)[0]


def select_records(records: List[Dict[str, Any]], requested: Optional[Iterable[str]]) -> List[Dict[str, Any]]:
    """Records named by GLYCAM name, entry folder or source folder; all when nothing is requested."""
    wanted = [name for name in (requested or []) if name]
    if not wanted:
        return list(records)
    selected = [
        record for record in records
        if {record["glycam_name"], record["folder"], record["source_folder"]} & set(wanted)
    ]
    known = {name for record in selected for name in (record["glycam_name"], record["folder"], record["source_folder"])}
    unknown = sorted(set(wanted) - known)
    if unknown:
        raise TransferError("not in index.json: " + ", ".join(unknown))
    return selected


def _expected_objects(index: Dict[str, Any], prefix: str) -> Dict[str, Tuple[str, int]]:
    """Object key -> (GLYCAM name, size) for every file of every entry of the index."""
    expected: Dict[str, Tuple[str, int]] = {}
    for record in index["records"]:
        key_prefix = entry_key_prefix(prefix, record["glycam_name"])
        for name, recorded in record["files"].items():
            expected[key_prefix + name] = (record["glycam_name"], int(recorded["size"]))
    return expected


def compare_with_index(
    backend: S3StorageBackend, prefix: str, index: Dict[str, Any], workers: int = 8
) -> Dict[str, Any]:
    """Compare what is stored under ``<prefix>/`` with an index.

    The listing gives keys and sizes: ``missing`` and ``size_mismatch`` name
    objects of the index that are not stored as recorded, ``extra`` objects
    below the prefix that the index does not list.  The manifest of every
    entry is then asked for its SHA-256 metadata: ``stale`` names the entries
    whose stored ``manifest.json`` is not the one the index records (an
    entry that was converted again and not uploaded since).  A manifest is
    stored after the files it describes, so the stored manifest of the index
    stands for the entry.  ``incomplete`` names every entry with a finding.
    """
    prefix = prefix.strip("/")
    expected = _expected_objects(index, prefix)
    listing = {item["key"]: item["size"] for item in backend.list_objects(prefix + "/")}
    missing = sorted(key for key in expected if key not in listing)
    size_mismatch = sorted(key for key, (_, size) in expected.items() if key in listing and listing[key] != size)
    extra = sorted(set(listing) - set(expected) - {f"{prefix}/{archive.INDEX_FILE}"})

    def manifest_differs(record: Dict[str, Any]) -> Optional[str]:
        pinned = record["files"][archive.MANIFEST_FILE]
        key = entry_key_prefix(prefix, record["glycam_name"]) + archive.MANIFEST_FILE
        stored = _stored_as_recorded(backend, key, pinned["size"], pinned["sha256"], listing)
        return None if stored or listing.get(key) != pinned["size"] else record["glycam_name"]

    stale = sorted(name for name in parallel_map(manifest_differs, index["records"], workers) if name)
    return {
        "entries": len(index["records"]),
        "objects": len(expected),
        "missing": missing,
        "size_mismatch": size_mismatch,
        "stale": stale,
        "incomplete": sorted({expected[key][0] for key in missing + size_mismatch} | set(stale)),
        "extra": extra,
        "ok": not missing and not size_mismatch and not stale,
    }


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
def _local_entry_files(archive_dir: Path, record: Dict[str, Any]) -> List[Tuple[str, Path, int, str]]:
    """``(name, path, size, sha256)`` of an entry in upload order, the manifest last.

    The GLYCAM name and the files come from the entry's ``manifest.json``;
    the index record has to describe the same entry.
    """
    entry_dir = Path(archive_dir) / record["folder"]
    manifest = archive.load_manifest(entry_dir)
    manifest_size, manifest_sha256 = file_sha256(entry_dir / archive.MANIFEST_FILE)
    described = {name: {"size": item["size"], "sha256": item["sha256"]} for name, item in manifest["files"].items()}
    described[archive.MANIFEST_FILE] = {"size": manifest_size, "sha256": manifest_sha256}
    if manifest["glycam_name"] != record["glycam_name"] or described != record["files"]:
        raise TransferError(
            "index.json does not describe this entry any more; rebuild it with scripts/convert_data_to_archive.py"
        )
    files = []
    for name in sorted(manifest["files"]):
        if name not in ENTRY_FILES:
            raise TransferError(f"the manifest lists an unexpected file: {name!r}")
        path = entry_dir / name
        if not path.is_file() or path.stat().st_size != described[name]["size"]:
            raise TransferError(f"{name} is missing or does not have the size of the manifest")
        files.append((name, path, described[name]["size"], described[name]["sha256"]))
    files.append((archive.MANIFEST_FILE, entry_dir / archive.MANIFEST_FILE, manifest_size, manifest_sha256))
    return files


def _stored_as_recorded(
    backend: S3StorageBackend, key: str, size: int, sha256: str, listing: Optional[Dict[str, int]] = None
) -> bool:
    """True when the object exists with this size and this SHA-256 in its metadata."""
    if listing is not None and listing.get(key) != size:
        return False   # absent, or another size: no request needed
    head = backend.head(key)
    return head is not None and head["size"] == size and head["sha256"] == sha256


def upload_archive(
    archive_dir: Path,
    backend: S3StorageBackend,
    prefix: str = archive.OBJECT_KEY_ROOT,
    glycans: Optional[Iterable[str]] = None,
    workers: int = 4,
    dry_run: bool = False,
    log: Log = print,
) -> Dict[str, Any]:
    """Upload the entries of a local archive directory; see the module documentation.

    Returns a report with one row per entry (``status`` is ``uploaded``,
    ``skipped`` or ``failed``), the comparison of the listing with
    ``index.json`` and whether ``index.json`` was uploaded.  ``ok`` is true
    when no entry failed and every object of the selected entries is in the
    storage with its size.  A dry run sends nothing.
    """
    archive_dir = Path(archive_dir)
    prefix = prefix.strip("/")
    index, index_bytes = read_index(archive_dir)
    selected = select_records(index["records"], glycans)
    listing = {item["key"]: item["size"] for item in backend.list_objects(prefix + "/")}

    def upload_entry(record: Dict[str, Any]) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "glycam_name": record["glycam_name"], "folder": record["folder"],
            "status": "failed", "transferred": [], "unchanged": [], "bytes": 0,
        }
        try:
            key_prefix = entry_key_prefix(prefix, record["glycam_name"])
            manifest_key = key_prefix + archive.MANIFEST_FILE
            files = _local_entry_files(archive_dir, record)
            stored = {
                name: _stored_as_recorded(backend, key_prefix + name, size, sha256, listing)
                for name, _, size, sha256 in files
            }
            # A stored manifest must never describe files that are being replaced: when a file of
            # the entry is sent, the stored manifest goes first and comes back last.
            if manifest_key in listing and not all(stored[name] for name, *_ in files[:-1]):
                stored[archive.MANIFEST_FILE] = False
                if not dry_run:
                    backend.unlink(manifest_key)
            for name, path, size, sha256 in files:
                if stored[name]:
                    row["unchanged"].append(name)
                    continue
                if not dry_run:
                    if file_sha256(path) != (size, sha256):
                        raise TransferError(f"{name} does not match the size and SHA-256 of its manifest")
                    backend.upload_file(path, key_prefix + name, sha256=sha256)
                row["transferred"].append(name)
                row["bytes"] += size
            row["status"] = "uploaded" if row["transferred"] else "skipped"
        except Exception as exc:   # one entry failing must not stop the others
            row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    # largest first, so the long transfers do not all end up at the tail
    selected.sort(key=lambda record: -int(record.get("archive_size", 0)))
    report = _summary(_run(upload_entry, selected, workers, log))
    report.update(dry_run=dry_run, prefix=prefix, index_uploaded=False, comparison=None)
    if dry_run:
        report["ok"] = not report["failed"]
        return report

    comparison = compare_with_index(backend, prefix, index, workers)
    report["comparison"] = comparison
    selected_names = {record["glycam_name"] for record in selected}
    report["ok"] = not report["failed"] and not selected_names & set(comparison["incomplete"])
    if comparison["ok"] and not report["failed"]:
        # the index that was compared, not what index.json may have become since
        index_key = f"{prefix}/{archive.INDEX_FILE}"
        if not _stored_as_recorded(backend, index_key, len(index_bytes), hashlib.sha256(index_bytes).hexdigest()):
            backend.write_binary(index_key, index_bytes)
            report["index_uploaded"] = True
        log(f"{index_key} lists {comparison['entries']} entries; all {comparison['objects']} objects are in the storage")
    else:
        log(
            f"{prefix}/{archive.INDEX_FILE} not uploaded: {len(comparison['incomplete'])} of "
            f"{comparison['entries']} entries of index.json are not complete in the storage"
        )
    return report


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def _parse_entry_manifest(payload: bytes, glycam_name: str) -> Dict[str, Any]:
    try:
        manifest = json.loads(payload)
    except ValueError as exc:
        raise TransferError(f"stored {archive.MANIFEST_FILE} is not JSON: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != archive.ARCHIVE_SCHEMA:
        raise TransferError(f"stored {archive.MANIFEST_FILE} is not a {archive.ARCHIVE_SCHEMA} manifest")
    if manifest.get("schema_version") != archive.SCHEMA_VERSION:
        raise TransferError(f"stored {archive.MANIFEST_FILE} has schema version {manifest.get('schema_version')!r}")
    if manifest.get("glycam_name") != glycam_name:
        raise TransferError(f"stored {archive.MANIFEST_FILE} is of another entry: {str(manifest.get('glycam_name'))[:80]}")
    if manifest.get("folder") != archive.entry_folder_name(glycam_name):
        raise TransferError(f"stored {archive.MANIFEST_FILE} names a folder that is not the folder of its GLYCAM name")
    files = manifest.get("files")
    if not isinstance(files, dict) or any(name not in files for name in REQUIRED_ENTRY_FILES):
        raise TransferError(f"stored {archive.MANIFEST_FILE} does not list the files of an entry")
    for name, recorded in files.items():
        if name not in ENTRY_FILES:   # also keeps every download inside the entry folder
            raise TransferError(f"stored {archive.MANIFEST_FILE} lists an unexpected file: {name!r}")
        if not isinstance(recorded, dict) or not isinstance(recorded.get("size"), int) or not isinstance(recorded.get("sha256"), str):
            raise TransferError(f"stored {archive.MANIFEST_FILE} has no size and SHA-256 for {name}")
    return manifest


def _matches(path: Path, recorded: Dict[str, Any], hash_content: bool = True) -> bool:
    try:
        if not path.is_file() or path.stat().st_size != recorded["size"]:
            return False
        return not hash_content or file_sha256(path)[1] == recorded["sha256"]
    except OSError:
        return False


def download_entry(
    backend: S3StorageBackend,
    prefix: str,
    glycam_name: str,
    dest_dir: Path,
    record: Optional[Dict[str, Any]] = None,
    trust_local: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Download one entry, named by its GLYCAM name, into ``dest_dir/<entry folder>``.

    Every file is hashed and compared with the stored manifest; ``record`` is
    the entry's record of the stored index, when known, and pins the manifest
    itself.  An entry that is already there with the same manifest and whose
    files verify is left alone (``trust_local`` compares sizes only for
    that).  Raises :class:`TransferError` when something does not verify.
    """
    dest_dir = Path(dest_dir)
    key_prefix = entry_key_prefix(prefix, glycam_name)
    row: Dict[str, Any] = {
        "glycam_name": glycam_name, "folder": archive.entry_folder_name(glycam_name),
        "status": "failed", "transferred": [], "unchanged": [], "bytes": 0,
    }
    try:
        payload = backend.read_binary(key_prefix + archive.MANIFEST_FILE)
    except FileNotFoundError:
        raise TransferError(f"no entry in the storage under {key_prefix[:120]}") from None
    if record is not None:
        pinned = record["files"][archive.MANIFEST_FILE]
        if (len(payload), hashlib.sha256(payload).hexdigest()) != (pinned["size"], pinned["sha256"]):
            raise TransferError(
                f"stored {archive.MANIFEST_FILE} does not match the stored {archive.INDEX_FILE} "
                "(the entry was uploaded again and the index not yet; download it by name)"
            )
    files = _parse_entry_manifest(payload, glycam_name)["files"]

    final = dest_dir / row["folder"]
    final_manifest = final / archive.MANIFEST_FILE
    if final.exists() and not final_manifest.is_file():
        raise TransferError(f"{final} exists and is not an archive entry; it is left untouched")
    current = final_manifest.is_file() and final_manifest.read_bytes() == payload
    if current and all(_matches(final / name, files[name], not trust_local) for name in files):
        row.update(status="skipped", unchanged=sorted(files))
        return row

    staging = dest_dir / archive.STAGING_DIR / hashlib.sha256(glycam_name.encode("utf-8")).hexdigest()[:32]
    replaced = staging.with_name(staging.name + ".old")   # where the entry being replaced goes for a moment
    kept, reused = [], []   # verified in the staging folder (an interrupted run); in the entry being replaced
    for name in sorted(files):
        if _matches(staging / name, files[name]):
            kept.append(name)
        elif _matches(final / name, files[name]):
            reused.append(name)
        else:
            row["transferred"].append(name)
            row["bytes"] += files[name]["size"]
    row.update(status="download", unchanged=sorted(kept + reused))
    if dry_run:
        return row

    staging.mkdir(parents=True, exist_ok=True)
    for name in reused:
        shutil.copyfile(final / name, staging / name)
    for name in row["transferred"]:
        recorded, target = files[name], staging / name
        part = staging / (name + PART_SUFFIX)
        backend.download_file(key_prefix + name, part)
        if file_sha256(part) != (recorded["size"], recorded["sha256"]):
            part.unlink()
            raise TransferError(f"downloaded {name} does not match the size and SHA-256 of the manifest")
        os.replace(part, target)
    for leftover in staging.iterdir():   # nothing but the verified files may become the entry
        if leftover.name not in files:
            leftover.unlink()
    _write_file(staging / archive.MANIFEST_FILE, payload)
    # Two renames, so the entry folder is the old entry or the new one and never part of either.
    shutil.rmtree(replaced, ignore_errors=True)
    if final.exists():
        os.rename(final, replaced)
    os.rename(staging, final)
    shutil.rmtree(replaced, ignore_errors=True)
    row["status"] = "downloaded"
    return row


def download_archive(
    backend: S3StorageBackend,
    dest_dir: Path,
    prefix: str = archive.OBJECT_KEY_ROOT,
    glycans: Optional[Iterable[str]] = None,
    workers: int = 4,
    trust_local: bool = False,
    dry_run: bool = False,
    log: Log = print,
) -> Dict[str, Any]:
    """Download entries by GLYCAM name, or every entry of the stored ``index.json``.

    With names, only those entries are needed in the storage: each is
    verified against its own manifest, and the stored index, which may be
    older than an entry that was uploaded since, is not consulted.  Without
    names, the stored index says what the archive holds and pins every
    manifest, and it is written to ``dest_dir`` when all entries are there.
    Returns a report as :func:`upload_archive` does; ``ok`` is true when no
    entry failed.
    """
    dest_dir = Path(dest_dir)
    prefix = prefix.strip("/")
    names = [name for name in (glycans or []) if name]
    index_payload: Optional[bytes] = None
    records: Dict[str, Dict[str, Any]] = {}
    if not names:
        try:
            index_payload = backend.read_binary(f"{prefix}/{archive.INDEX_FILE}")
        except FileNotFoundError:
            raise TransferError(
                f"no {prefix}/{archive.INDEX_FILE} in the storage (the upload of the archive has not finished); "
                "name the entries to download"
            ) from None
        index = parse_index(index_payload, f"stored {archive.INDEX_FILE}")
        records = {record["glycam_name"]: record for record in index["records"]}
    wanted = list(dict.fromkeys(names)) if names else sorted(records)

    def one(glycam_name: str) -> Dict[str, Any]:
        try:
            return download_entry(
                backend, prefix, glycam_name, dest_dir, records.get(glycam_name), trust_local, dry_run)
        except Exception as exc:   # one entry failing must not stop the others
            return {
                "glycam_name": glycam_name, "status": "failed", "transferred": [], "unchanged": [], "bytes": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }

    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)
    report = _summary(_run(one, wanted, workers, log))
    report.update(dry_run=dry_run, prefix=prefix, index_written=False, ok=not report["failed"])
    if dry_run:
        return report
    staging_root = dest_dir / archive.STAGING_DIR
    if staging_root.is_dir() and not any(staging_root.iterdir()):
        staging_root.rmdir()
    if not names and report["ok"] and index_payload is not None:
        _write_file(dest_dir / archive.INDEX_FILE, index_payload)
        report["index_written"] = True
    return report
