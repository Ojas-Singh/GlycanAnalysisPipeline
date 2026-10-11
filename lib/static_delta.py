"""Add, replace or withdraw one glycan of a static database, from its stored root files.

The finalization of the pipeline (``lib.glybake``, ``lib.json2rdf``) builds
the root files from every glycan folder.  Here they are derived for a change
of *one* glycan from the root files that are stored and the folder of that
glycan; no other glycan folder is read:

``GLYCOSHAPE.json``
    the stored one with one key inserted, replaced or removed, in the same
    order and serialisation (``glybake.inventory_bytes``).
``GlycoShape.zip``
    the members of the stored one plus or minus one, written as the
    finalization writes it (``glybake.write_archive``).  Each stored member
    is checked against the ``zip_sha256`` the stored manifest records for
    its glycan.
``GLYCOSHAPE_RDF.ttl``
    converted from the new ``GLYCOSHAPE.json`` (``lib.json2rdf``; the
    conversion is a function of the JSON).
``missing_glytoucan.txt``, ``missing_wurcs.txt``
    the stored reports with the lines of the changed glycan taken out and
    those of its new entry put in (:func:`change_missing_reports`); no
    service is needed for the other glycans.
``sync_manifest.json``
    the stored manifest with one glycan entry changed, the entries of the
    root files above replaced where their content changed, and
    ``global_hash`` computed as the finalization computes it.  The entries
    of all other files are the stored ones, untouched.

For a database whose root files were written by the finalization, the result
is byte for byte what the finalization writes for the changed tree, apart
from the times in the manifest (``generated_at`` and the ``mtime`` of the
files that changed), which are not part of any hash.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from lib import glybake, static_publish

ADD, REPLACE, WITHDRAW = "add", "replace", "withdraw"
ACTIONS = (ADD, REPLACE, WITHDRAW)
INVENTORY = "GLYCOSHAPE.json"
MASTER_ARCHIVE = "GlycoShape.zip"
RDF = "GLYCOSHAPE_RDF.ttl"
MISSING_GLYTOUCAN = "missing_glytoucan.txt"
MISSING_WURCS = "missing_wurcs.txt"
MANIFEST = static_publish.MANIFEST_NAME
#: The root files a change writes again, in the order they are derived.
DERIVED_ROOT_FILES = (INVENTORY, MASTER_ARCHIVE, RDF, MISSING_GLYTOUCAN, MISSING_WURCS)
#: The stored root files a change is derived from; the two reports only when the manifest lists them.
NEEDED_ROOT_FILES = (INVENTORY, MASTER_ARCHIVE)
OPTIONAL_ROOT_FILES = (MISSING_GLYTOUCAN, MISSING_WURCS)

FetchArchive = Callable[[str], Optional[bytes]]


class DeltaError(Exception):
    """The change cannot be derived from what is stored."""


def _sha256(path: Path) -> Tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def _check_action(action: str, glycan_id: str, present: bool, where: str) -> None:
    if action not in ACTIONS:
        raise DeltaError(f"unknown action {action!r}")
    if action == ADD and present:
        raise DeltaError(f"{glycan_id} is already in {where}")
    if action in (REPLACE, WITHDRAW) and not present:
        raise DeltaError(f"{glycan_id} is not in {where}")


# ---------------------------------------------------------------------------
# One root file at a time
# ---------------------------------------------------------------------------
def folder_entry(glycan_dir: Path) -> Dict[str, Any]:
    """The manifest entry of a local glycan folder (named by its GlycoShape ID)."""
    glycan_dir = Path(glycan_dir)
    return glybake.manifest_glycan_entry(glycan_dir.name, glybake.GlycoShapeBaker.glycan_folder_files(glycan_dir))


def change_inventory(inventory: Dict[str, Any], action: str, glycan_id: str, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The inventory with the entry of one glycan inserted, replaced or removed (a new mapping, in the order of the IDs)."""
    _check_action(action, glycan_id, glycan_id in inventory, INVENTORY)
    changed = dict(inventory)
    if action == WITHDRAW:
        del changed[glycan_id]
        if not changed:
            raise DeltaError("the last glycan of a database cannot be withdrawn: the website refuses an empty manifest")
    else:
        if not isinstance(data, dict) or (data.get("archetype") or {}).get("ID") != glycan_id:
            raise DeltaError(f"data.json is not the one of {glycan_id}")
        changed[glycan_id] = data
    return dict(sorted(changed.items()))


def change_master_archive(
    stored: Path,
    target: Path,
    action: str,
    glycan_id: str,
    glycan_archive: Optional[bytes],
    expected: Dict[str, Optional[str]],
    fetch: Optional[FetchArchive] = None,
) -> Dict[str, Any]:
    """``GlycoShape.zip`` with the archive of one glycan added, replaced or removed.

    ``expected`` maps every *other* glycan of the database to the SHA-256 of
    its ``GSxxxxx.zip`` (``zip_sha256`` of the stored manifest).  A stored
    member that is not that archive, or is missing, is asked from
    ``fetch(glycan_id)`` (one small object of the storage); without it the
    change is refused.  Members of glycans that are not in the database are
    left out.  Returns ``{"members", "repaired", "dropped"}``.
    """
    if action not in ACTIONS:
        raise DeltaError(f"unknown action {action!r}")
    if action != WITHDRAW and not glycan_archive:
        raise DeltaError(f"the archive of {glycan_id} is missing")
    report: Dict[str, Any] = {"members": 0, "repaired": [], "dropped": []}
    try:
        archive = zipfile.ZipFile(stored)
    except (OSError, zipfile.BadZipFile) as exc:
        raise DeltaError(f"the stored {MASTER_ARCHIVE} cannot be read: {exc}") from exc
    with archive:
        stored_names = archive.namelist()
        if len(set(stored_names)) != len(stored_names):
            raise DeltaError(f"the stored {MASTER_ARCHIVE} names a member twice")
        report["dropped"] = sorted(
            name for name in stored_names
            if name[:-4] not in expected and name != f"{glycan_id}.zip" or not name.endswith(".zip"))

        def member(other_id: str) -> Callable[[], bytes]:
            def read() -> bytes:
                name = f"{other_id}.zip"
                content = archive.read(name) if name in stored_names else None
                if content is None or hashlib.sha256(content).hexdigest() != expected[other_id]:
                    content = fetch(other_id) if fetch is not None else None
                    if content is None or hashlib.sha256(content).hexdigest() != expected[other_id]:
                        raise DeltaError(
                            f"the stored {MASTER_ARCHIVE} does not hold the archive that the stored manifest "
                            f"records for {other_id}" + ("" if fetch is None else ", and neither does its folder"))
                    report["repaired"].append(other_id)
                return content
            return read

        members: List[Tuple[str, Any]] = []
        for other_id, digest in expected.items():
            if other_id == glycan_id:
                continue
            if not digest:
                raise DeltaError(f"the stored manifest records no archive for {other_id}; bake the database once")
            members.append((f"{other_id}.zip", member(other_id)))
        if action != WITHDRAW:
            members.append((f"{glycan_id}.zip", glycan_archive))
        report["members"] = len(members)
        glybake.write_archive(target, members)
    report["repaired"].sort()
    return report


def _own_reports(entry: Dict[str, Any], glycan_id: str) -> Tuple[List[str], List[str]]:
    missing, rows = glybake.missing_reports({glycan_id: entry})
    return missing, [_row_line(row) for row in rows]


def _row_line(row: Dict[str, str]) -> str:
    return "\t".join([row["glycoshape_id"], row["variant"], row["iupac"], row["glycam"]])


def change_missing_reports(
    stored_glytoucan: Optional[str],
    stored_wurcs: Optional[str],
    inventory_before: Dict[str, Any],
    inventory_after: Dict[str, Any],
    glycan_id: str,
    calculate: Optional[Callable[[str, str, str], Tuple[Optional[str], Optional[str]]]] = None,
    resolve: Optional[Callable[[List[str]], Dict[str, Optional[str]]]] = None,
) -> Tuple[str, str]:
    """``missing_glytoucan.txt`` and ``missing_wurcs.txt`` after the change of one glycan.

    The finalization decides for every variant of every glycan, with two
    services, whether it is listed (``glybake.missing_reports``).  Here its
    decisions for the *other* glycans are read from the stored reports: a
    WURCS of theirs is listed when the stored report lists it.  Only the
    changed glycan is decided again, with ``calculate`` and ``resolve`` when
    they are given, and from its entry alone without them.  Lines come in
    the order of the inventory, as the finalization writes them; a stored
    line that no entry of the inventory explains (a WURCS that an earlier
    finalization calculated for a variant that stores none) is kept at the
    end.  Without a stored report the inventory decides alone.
    """
    before_missing, before_rows = glybake.missing_reports(inventory_before)
    stored_lines = before_missing if stored_glytoucan is None else [
        line for line in stored_glytoucan.splitlines() if line]
    stored_rows = {_row_line(row) for row in before_rows} if stored_wurcs is None else {
        line for line in stored_wurcs.splitlines()[1:] if line}
    listed = set(stored_lines)
    own_missing: List[str] = []
    own_rows: List[str] = []
    if glycan_id in inventory_after:
        own_missing, rows = glybake.missing_reports(
            {glycan_id: inventory_after[glycan_id]}, calculate=calculate, resolve=resolve)
        own_rows = [_row_line(row) for row in rows]

    missing: List[str] = []
    rows_out: List[str] = []
    for other_id, entry in inventory_after.items():
        if other_id == glycan_id:
            candidates, candidate_rows = own_missing, own_rows
        else:
            pure_missing, pure_rows = _own_reports(entry, other_id)
            candidates = [wurcs for wurcs in pure_missing if wurcs in listed]
            candidate_rows = [row for row in pure_rows if row in stored_rows]
        for wurcs in candidates:
            if wurcs not in missing:
                missing.append(wurcs)
        rows_out.extend(candidate_rows)
    explained = set(before_missing) | set(missing)
    missing.extend(line for line in stored_lines if line not in explained)
    return (
        glybake.missing_glytoucan_text(missing),
        glybake.MISSING_WURCS_HEADER + "".join(row + "\n" for row in rows_out),
    )


def write_rdf(inventory_path: Path, target: Path) -> None:
    """``GLYCOSHAPE_RDF.ttl`` for an inventory file."""
    from lib.json2rdf import convert_glycoshape_to_rdf
    from lib.storage import local_storage

    with local_storage():
        if convert_glycoshape_to_rdf(str(inventory_path), str(target)) is None:
            raise DeltaError(f"{RDF} could not be written from {inventory_path}")


def change_manifest(
    stored_manifest: Dict[str, Any],
    action: str,
    glycan_id: str,
    entry: Optional[Dict[str, Any]],
    root_files: Dict[str, Path],
    generated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """The sync manifest after the change of one glycan and of the root files given.

    ``entry`` is the manifest entry of the new folder (:func:`folder_entry`).
    A root file whose size and SHA-256 are the stored ones keeps its stored
    entry; every entry that is not named here is the stored one.
    """
    glycans = copy.deepcopy(stored_manifest["glycans"])
    _check_action(action, glycan_id, glycan_id in glycans, MANIFEST)
    if action == WITHDRAW:
        del glycans[glycan_id]
        if not glycans:
            raise DeltaError("the last glycan of a database cannot be withdrawn: the website refuses an empty manifest")
    else:
        if not isinstance(entry, dict) or not entry.get("files"):
            raise DeltaError(f"the folder of {glycan_id} holds no files")
        glycans[glycan_id] = entry
    roots = copy.deepcopy(stored_manifest["root_files"])
    for name, path in root_files.items():
        size, digest = _sha256(Path(path))
        stored = roots.get(name)
        if isinstance(stored, dict) and stored.get("sha256") == digest and stored.get("size") == size:
            continue
        roots[name] = {
            "path": name, "sha256": digest, "size": size, "mtime": glybake.utc_timestamp(Path(path).stat().st_mtime)}
    return glybake.build_manifest(roots, dict(sorted(glycans.items())), MANIFEST, generated_at)


# ---------------------------------------------------------------------------
# The whole change
# ---------------------------------------------------------------------------
def needed_root_files(stored_manifest: Dict[str, Any]) -> List[str]:
    """The stored root files a change is derived from, for this manifest."""
    listed = stored_manifest.get("root_files", {})
    missing = [name for name in NEEDED_ROOT_FILES if name not in listed]
    if missing:
        raise DeltaError(f"the stored manifest does not list {', '.join(missing)}")
    return list(NEEDED_ROOT_FILES) + [name for name in OPTIONAL_ROOT_FILES if name in listed]


def apply_change(
    action: str,
    glycan_id: str,
    stored_dir: Path,
    out_dir: Path,
    glycan_dir: Optional[Path] = None,
    fetch_archive: Optional[FetchArchive] = None,
    calculate: Optional[Callable[[str, str, str], Tuple[Optional[str], Optional[str]]]] = None,
    resolve: Optional[Callable[[List[str]], Dict[str, Optional[str]]]] = None,
    generated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Derive the root files of the database after the change of one glycan.

    ``stored_dir`` holds local copies of the stored ``sync_manifest.json``
    and of the root files :func:`needed_root_files` names; they are checked
    against the manifest first, so a change is never derived from root files
    that are not the ones the manifest describes.  ``glycan_dir`` is the new
    folder, named by the GlycoShape ID (not for a withdrawal).  The new root
    files and the new ``sync_manifest.json`` are written into ``out_dir``;
    nothing else is written.  ``fetch_archive``, ``calculate`` and
    ``resolve`` are those of :func:`change_master_archive` and
    :func:`change_missing_reports`.

    Returns a report: the stored and the new ``global_hash``, the new
    manifest, per root file whether it changed, what was done to the master
    archive, and the seconds per step.  Raises :class:`DeltaError`.
    """
    stored_dir, out_dir = Path(stored_dir), Path(out_dir)
    seconds: Dict[str, float] = {}
    started = time.perf_counter()

    def timed(name: str) -> None:
        nonlocal started
        now = time.perf_counter()
        seconds[name] = round(now - started, 3)
        started = now

    if action not in ACTIONS:
        raise DeltaError(f"unknown action {action!r}")
    try:
        stored_manifest = static_publish.parse_manifest((stored_dir / MANIFEST).read_bytes(), f"the stored {MANIFEST}")
    except OSError as exc:
        raise DeltaError(f"the stored {MANIFEST} cannot be read: {exc}") from exc
    except static_publish.PublishError as exc:
        raise DeltaError(str(exc)) from exc
    for name in needed_root_files(stored_manifest):
        recorded = stored_manifest["root_files"][name]
        try:
            found = _sha256(stored_dir / name)
        except OSError as exc:
            raise DeltaError(f"the stored {name} cannot be read: {exc}") from exc
        if found != (recorded["size"], recorded["sha256"].lower()):
            raise DeltaError(
                f"the stored {name} is not the file the stored manifest describes "
                "(a publication is in progress, or was interrupted)")
    try:
        inventory = json.loads((stored_dir / INVENTORY).read_bytes())
    except ValueError as exc:
        raise DeltaError(f"the stored {INVENTORY} is not JSON: {exc}") from exc
    if not isinstance(inventory, dict) or sorted(inventory) != sorted(stored_manifest["glycans"]):
        raise DeltaError(f"the stored {INVENTORY} and the stored manifest do not list the same glycans")
    _check_action(action, glycan_id, glycan_id in inventory, f"the stored {INVENTORY}")

    data: Optional[Dict[str, Any]] = None
    entry: Optional[Dict[str, Any]] = None
    glycan_archive: Optional[bytes] = None
    if action != WITHDRAW:
        if glycan_dir is None or Path(glycan_dir).name != glycan_id or not Path(glycan_dir).is_dir():
            raise DeltaError(f"the folder of {glycan_id} is needed, named {glycan_id}")
        glycan_dir = Path(glycan_dir)
        try:
            data = json.loads((glycan_dir / "data.json").read_bytes())
            glycan_archive = (glycan_dir / f"{glycan_id}.zip").read_bytes()
        except (OSError, ValueError) as exc:
            raise DeltaError(f"the folder of {glycan_id} is not complete: {exc}") from exc
        entry = folder_entry(glycan_dir)
    timed("read")

    out_dir.mkdir(parents=True, exist_ok=True)
    changed_inventory = change_inventory(inventory, action, glycan_id, data)
    (out_dir / INVENTORY).write_bytes(glybake.inventory_bytes(changed_inventory))
    timed("inventory")

    expected = {other: item.get("zip_sha256") for other, item in stored_manifest["glycans"].items()}
    archive_report = change_master_archive(
        stored_dir / MASTER_ARCHIVE, out_dir / MASTER_ARCHIVE, action, glycan_id, glycan_archive, expected,
        fetch_archive)
    timed("master_archive")

    write_rdf(out_dir / INVENTORY, out_dir / RDF)
    timed("rdf")

    def stored_text(name: str) -> Optional[str]:
        return (stored_dir / name).read_text(encoding="utf-8") if name in stored_manifest["root_files"] else None

    missing_text, wurcs_text = change_missing_reports(
        stored_text(MISSING_GLYTOUCAN), stored_text(MISSING_WURCS), inventory, changed_inventory, glycan_id,
        calculate=calculate, resolve=resolve)
    (out_dir / MISSING_GLYTOUCAN).write_bytes(missing_text.encode("utf-8"))
    (out_dir / MISSING_WURCS).write_bytes(wurcs_text.encode("utf-8"))
    timed("missing_reports")

    manifest = change_manifest(
        stored_manifest, action, glycan_id, entry, {name: out_dir / name for name in DERIVED_ROOT_FILES}, generated_at)
    payload = glybake.manifest_bytes(manifest)
    try:
        static_publish.parse_manifest(payload, f"the new {MANIFEST}")
    except static_publish.PublishError as exc:
        raise DeltaError(str(exc)) from exc
    (out_dir / MANIFEST).write_bytes(payload)
    timed("manifest")

    stored_roots = stored_manifest["root_files"]
    return {
        "action": action,
        "glycoshape_id": glycan_id,
        "base_global_hash": stored_manifest["global_hash"],
        "global_hash": manifest["global_hash"],
        "glycans": len(manifest["glycans"]),
        "manifest": manifest,
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "manifest_size": len(payload),
        "root_files": {
            name: {
                "sha256": manifest["root_files"][name]["sha256"],
                "size": manifest["root_files"][name]["size"],
                "changed": (stored_roots.get(name) or {}).get("sha256") != manifest["root_files"][name]["sha256"],
            }
            for name in DERIVED_ROOT_FILES
        },
        "entry": None if entry is None else {key: entry[key] for key in ("hash", "file_count", "total_size")},
        "master_archive": archive_report,
        "seconds": seconds,
    }
