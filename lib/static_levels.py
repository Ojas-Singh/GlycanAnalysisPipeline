"""Reduce a published static database to fewer clustering levels, without recomputing anything.

    python scripts/static_drop_levels.py --source STATIC_DIR --dest NEW_DIR [--levels 1,2] [--workers N]

``STATIC_DIR`` is a static database that was built with Levels 1, 2 and 3
(``GSxxxxx/`` folders, ``GLYCOSHAPE.json``, ``GlycoShape.zip``, ...).
``NEW_DIR`` becomes the database that the pipeline writes with levels 1,2: the
static stage per glycan, then the finalization.  The source is only read.

**One glycan folder** (in parallel, :func:`reduce_glycan`).  Every file is one of:

* *kept*: hard-linked into the new folder (copied when that is not possible,
  or with ``copy``).  Everything that does not depend on the levels, and the
  folders ``output/level_<n>/`` and tables ``torsion_level_<n>_reps.csv`` of
  the levels that stay.  A file the static stage does not write is kept too
  and listed as *extra*.
* *rewritten*, with the functions the static stage uses, so that the bytes
  are those of a static build with these levels from the same process folder:
  ``output/info.json`` (:func:`lib.static_ready.info_json_for_levels`), the
  tables that carry ``level_<n>_cluster`` / ``level_<n>_center`` columns of
  other levels, which is ``output/torsion_all.csv``
  (:func:`lib.static_ready.csv_for_levels`), and ``data.json``, which loses
  ``archetype.cluster_levels.level_<n>`` and is written by
  :func:`lib.static_ready.save_glycan_data`, the writer of the static stage.
  A file that needs no change is kept.
* *dropped*: ``output/level_<n>/`` and ``output/torsion_level_<n>_reps.csv``
  of the other levels.  A source from which they were already removed gives
  the same result.
* left out: hidden files (the website refuses a manifest that lists one) and
  ``GSxxxxx.zip``, which holds ``data.json`` and is built again below.

A folder is built in a staging folder next to the destination and renamed
into it when it is complete and has every file the pipeline requires for
these levels, so the destination only ever holds finished folders; a folder
that is there is skipped by the next run.

**The database** (:func:`finalize_database`, once every folder is there), in
the order of the finalization of ``main.py`` and with its functions:
``GLYCOSHAPE.json`` (``create_consolidated_json``), every ``GSxxxxx.zip`` and
``GlycoShape.zip`` (``create_individual_glycan_archive``,
``create_master_archive``) and, last, ``sync_manifest.json``
(``create_sync_manifest``).  ``GLYCOSHAPE_RDF.ttl`` and the ``missing_*``
lists do not depend on the levels and are kept as they are (the RDF takes
``clusters`` and ``coverage_clusters``, never ``cluster_levels``; converting
again gives the same graph in another order of lines).  Other root files are
kept and listed.  The new manifest has to pass the validation of the
publisher and to list exactly the files of the destination.  When the source
has a ``sync_manifest.json``, every file that was taken from it (kept, or read
to be rewritten) is compared with the SHA-256 recorded there, and the new
``GLYCOSHAPE.json`` with the one of the source; a difference is a warning.

**Hard links.**  A kept file of a glycan folder is one file with two names
afterwards, so a program that later rewrites it in place in one database
changes the other.  Nothing here does (every file that is written is a new
file), and the files that the pipeline itself rewrites in place later are
never links: the root files and a kept ``data.json`` are copied.  ``copy``
shares nothing.

Nothing but the database is written into the destination: staging folders,
the records of finished glycans, the lock and the report live in the work
folder (default ``<dest>.work``).
"""
from __future__ import annotations

import contextlib
import datetime
import fcntl
import hashlib
import json
import logging
import multiprocessing
import os
import re
import shutil
import signal
import time
import zipfile
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from lib import config, static_ready
from lib.storage import parallel_map

MANIFEST_NAME = "sync_manifest.json"
INVENTORY_NAME = "GLYCOSHAPE.json"
MASTER_ARCHIVE_NAME = "GlycoShape.zip"
#: Root files that are written again, in this order, by :func:`finalize_database`.
REGENERATED_ROOT_FILES = (INVENTORY_NAME, MASTER_ARCHIVE_NAME, MANIFEST_NAME)
#: Every root name that the functions of the finalization write (the manifest by way of a temporary file).
_WRITTEN_ROOT_FILES = REGENERATED_ROOT_FILES + (MANIFEST_NAME + ".tmp",)
#: Root files of the finalization that do not depend on the levels.
LEVEL_INDEPENDENT_ROOT_FILES = (
    "GLYCOSHAPE_RDF.ttl",
    "missing_glytoucan.txt",
    "missing_wurcs.txt",
    "missing_glytoucan_no_wurcs_gags.tsv",
)
#: Files of a level folder (``main.py``: ``REQUIRED_GLYSTATIC_LEVEL_FILES``).
LEVEL_FILES = ("dist.svg", "PDB/alpha.pdb", "PDB/beta.pdb", "GLYCAM/alpha.pdb", "GLYCAM/beta.pdb")
#: Files of a glycan folder that do not depend on the levels.
_KEPT_FILES = frozenset(
    ["glygen.json", "glycosmos.json", "snfg.svg", "output/torparts.npz"]
    + [f"landscape/{name}" for name in static_ready.LANDSCAPE_FILES]
)
_TABLES = frozenset(f"output/{name}" for name in static_ready.ANALYSIS_TABLES)
_LEVEL_FOLDER = re.compile(r"output/level_(\d+)/(.+)")
_LEVEL_TABLE = re.compile(r"output/torsion_level_(\d+)_reps\.csv")
_LEVEL_KEY = re.compile(r"level_(\d+)")
_HEADER_BLOCK = 1 << 20

Log = Callable[[str], None]


class StaticLevelsError(Exception):
    """The source or the destination cannot be used, or the database could not be completed."""


# ---------------------------------------------------------------------------
# What a static folder is made of
# ---------------------------------------------------------------------------
def required_files(levels: Iterable[int]) -> List[str]:
    """Files that a complete static folder has for these levels.

    ``data.json`` and the list of ``_get_static_only_required_files`` in
    ``main.py`` (a test compares the two).
    """
    names = [
        "data.json",
        "output/info.json",
        "output/pca.csv",
        "output/torparts.npz",
        "output/torsion_all.csv",
        "output/torsion_glycosidic.csv",
    ]
    for level_name in static_ready.level_names(levels):
        names.append(f"output/torsion_{level_name}_reps.csv")
        names.extend(f"output/{level_name}/{name}" for name in LEVEL_FILES)
    names.extend(f"landscape/{name}" for name in static_ready.LANDSCAPE_FILES)
    return names


def classify(relative: str, glycan_id: str, levels: Iterable[int]) -> str:
    """What becomes of the file ``relative`` of the folder of ``glycan_id``.

    ``keep``, ``extra`` (kept, not a file of the static stage), ``table``,
    ``info``, ``data`` (rewritten when they describe other levels),
    ``archive`` (built again by the finalization), ``drop`` or ``hidden``.
    """
    keep = {int(level) for level in levels}
    if any(part.startswith(".") for part in relative.split("/")):
        return "hidden"
    if relative == f"{glycan_id}.zip":
        return "archive"
    match = _LEVEL_FOLDER.fullmatch(relative)
    if match:
        if int(match.group(1)) not in keep:
            return "drop"
        return "keep" if match.group(2) in LEVEL_FILES else "extra"
    match = _LEVEL_TABLE.fullmatch(relative)
    if match:
        return "table" if int(match.group(1)) in keep else "drop"
    if relative == "data.json":
        return "data"
    if relative == "output/info.json":
        return "info"
    if relative in _TABLES:
        return "table"
    return "keep" if relative in _KEPT_FILES else "extra"


def data_without_other_levels(data: Any, levels: Iterable[int]) -> Tuple[Any, List[str]]:
    """``data.json`` content for the given levels; returns ``(data, removed keys)``.

    The static stage writes one entry ``archetype.cluster_levels.level_<n>``
    per level and nothing else that depends on the set of levels (``clusters``
    is Level 1 and ``coverage_clusters`` Level 2, under names of their own).
    The entries of other levels are left out; the input is not changed.
    """
    keep = {int(level) for level in levels}
    archetype = data.get("archetype") if isinstance(data, dict) else None
    cluster_levels = archetype.get("cluster_levels") if isinstance(archetype, dict) else None
    if not isinstance(cluster_levels, dict):
        return data, []
    removed = [
        key for key in cluster_levels
        if (match := _LEVEL_KEY.fullmatch(str(key))) and int(match.group(1)) not in keep
    ]
    if not removed:
        return data, []
    reduced = dict(data)
    reduced["archetype"] = dict(archetype)
    reduced["archetype"]["cluster_levels"] = {
        key: value for key, value in cluster_levels.items() if key not in removed
    }
    return reduced, removed


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _local_storage() -> Iterator[None]:
    """Local files for the pipeline functions that go through ``lib.storage``.

    ``save_glycan_data`` and ``create_consolidated_json`` open their files
    with the storage manager of the process.  Here that has to be the local
    file system whatever ``GLYCOSHAPE_STORAGE`` or a ``.env`` file selects;
    the manager that was there is put back afterwards.
    """
    from lib import storage

    previous = storage._storage_manager
    storage._storage_manager = storage.StorageManager(storage.LocalStorageBackend())
    try:
        yield
    finally:
        storage._storage_manager = previous


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _files_below(folder: Path) -> List[str]:
    """Relative paths of the files below ``folder``, hidden ones included, sorted."""
    return sorted(path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file())


def _folder_bytes(folder: Path) -> int:
    return sum((folder / name).stat().st_size for name in _files_below(folder))


def _glycan_ids(root: Path) -> List[str]:
    """Names of the glycan folders of a static database (``GS*``, as the baker finds them)."""
    return sorted(path.name for path in root.glob("GS*") if path.is_dir())


def _place(source: Path, target: Path, copy: bool) -> str:
    """Put an unchanged file into the new tree: ``"linked"`` or ``"copied"``."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if not copy and not source.is_symlink():
        try:
            os.link(source, target)
            return "linked"
        except OSError:   # another file system, or one without hard links
            pass
    shutil.copy2(source, target)
    return "copied"


def _write(target: Path, content: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)


def _is_inside(path: Path, folder: Path) -> bool:
    return path == folder or folder in path.parents


def _fingerprint(path: Path) -> Tuple[int, int, str]:
    """Size, CRC-32 and SHA-256 of a small file."""
    content = path.read_bytes()
    return len(content), zlib.crc32(content), hashlib.sha256(content).hexdigest()


# ---------------------------------------------------------------------------
# One glycan
# ---------------------------------------------------------------------------
def _info_levels(content: bytes) -> List[int]:
    info = json.loads(content)
    if not isinstance(info, dict) or not isinstance(info.get("levels"), list):
        raise ValueError("output/info.json has no list of levels")
    return [int(entry["level"]) for entry in info["levels"] if isinstance(entry, dict) and entry.get("level") is not None]


def _reduce_data_json(content: bytes, target: Path, glycan_id: str, levels: Tuple[int, ...]) -> Tuple[bool, List[str]]:
    """Write ``data.json`` without the entries of other levels; ``(written, notes)``.

    Not written (the caller keeps the file) when it has no such entry.
    """
    data = json.loads(content)
    if not isinstance(data, dict) or not isinstance(data.get("archetype"), dict):
        raise ValueError("data.json has no archetype entry")
    notes = []
    identifier = str(data["archetype"].get("ID") or "")
    if identifier != glycan_id:
        notes.append(f"data.json names the glycan {identifier!r}, the folder is {glycan_id}")
    reduced, removed = data_without_other_levels(data, levels)
    if not removed:
        return False, notes
    # Written unchanged first: when that returns the bytes of the source, the
    # new file differs from it by the removed entries and by nothing else.
    check = target.with_name(target.name + ".check")
    static_ready.save_glycan_data(data, str(check), "data")
    same_format = (check / "data.json").read_bytes() == content
    shutil.rmtree(check)
    if not same_format:
        notes.append("data.json was not in the format of the static stage; it is now (same content)")
    static_ready.save_glycan_data(reduced, str(target.parent), "data")
    return True, notes


def reduce_glycan(source: Path, target: Path, levels: Iterable[int], copy: bool = False) -> Dict[str, Any]:
    """Write the static folder ``source`` for ``levels`` into the new folder ``target``.

    Returns what was done with every file.  Raises when a file that has to be
    read is not what the static stage writes, or when the result lacks a file
    that the pipeline requires for these levels; ``target`` is then incomplete
    and the caller removes it.  ``source`` is only read.
    """
    source, target = Path(source), Path(target)
    levels = tuple(int(level) for level in levels)
    glycan_id = source.name
    record: Dict[str, Any] = {
        "kept": [], "rewritten": [], "dropped": [], "extra": [], "hidden_skipped": [],
        "archive_bytes_before": None, "files_before": 0, "bytes_before": 0, "bytes_staged": 0,
        "linked": 0, "copied": 0, "notes": [], "rewritten_from_sha256": {},
    }
    target.mkdir(parents=True)

    def keep(relative: str, link: bool = True) -> None:
        record[_place(source / relative, target / relative, copy or not link)] += 1
        record["kept"].append(relative)

    def rewritten(relative: str, read: bytes) -> None:
        # What was read is recorded, so that it can be compared with the manifest of the source.
        record["rewritten"].append(relative)
        record["rewritten_from_sha256"][relative] = hashlib.sha256(read).hexdigest()

    for relative in _files_below(source):
        path = source / relative
        size = path.stat().st_size
        record["files_before"] += 1
        record["bytes_before"] += size
        kind = classify(relative, glycan_id, levels)
        if kind == "hidden":
            record["hidden_skipped"].append(relative)
        elif kind == "archive":
            record["archive_bytes_before"] = size
        elif kind == "drop":
            record["dropped"].append(relative)
        elif kind == "table":
            with open(path, "rb") as handle:
                head = handle.read(_HEADER_BLOCK)
            if static_ready.csv_other_level_columns(head, levels):
                content = path.read_bytes()
                _write(target / relative, static_ready.csv_for_levels(content, levels))
                rewritten(relative, content)
            else:
                keep(relative)
        elif kind == "info":
            content = path.read_bytes()
            try:
                reduced = static_ready.info_json_for_levels(content, levels)
                described = _info_levels(reduced)
            except ValueError as exc:
                raise ValueError(f"output/info.json cannot be used: {exc}") from exc
            absent = [level for level in levels if level not in described]
            if absent:
                raise ValueError(f"output/info.json does not describe level {', '.join(map(str, absent))}")
            if reduced == content:
                keep(relative)
            else:
                _write(target / relative, reduced)
                rewritten(relative, content)
        elif kind == "data":
            content = path.read_bytes()
            try:
                written, notes = _reduce_data_json(content, target / relative, glycan_id, levels)
            except ValueError as exc:
                raise ValueError(f"data.json cannot be used: {exc}") from exc
            record["notes"].extend(notes)
            if written:
                rewritten(relative, content)
            else:
                keep(relative, link=False)      # glymeta rewrites data.json in place: never a link
        else:
            keep(relative)
            if kind == "extra":
                record["extra"].append(relative)

    missing = [name for name in required_files(levels) if not (target / name).is_file()]
    if missing:
        raise ValueError(
            f"incomplete for levels {','.join(map(str, levels))}: no {', '.join(missing[:6])}"
            + (f" and {len(missing) - 6} more" if len(missing) > 6 else "")
        )
    record["bytes_staged"] = _folder_bytes(target)
    return record


def _initialize_worker(log_level: int) -> None:
    """A worker process: Ctrl-C is handled by the parent, and it logs what the parent logs.

    The modules of the pipeline set the root logger to INFO when they are
    imported, which a new process does before it gets here.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    logging.basicConfig()
    logging.getLogger().setLevel(log_level)


def _run_glycan(task: Dict[str, Any]) -> Dict[str, Any]:
    """One glycan from the source into the destination by way of the staging folder."""
    glycan_id = task["glycan_id"]
    staged = Path(task["staging"]) / glycan_id
    record_path = Path(task["records"]) / f"{glycan_id}.json"
    started = time.perf_counter()
    try:
        if staged.exists():
            shutil.rmtree(staged)
        with _local_storage():
            record = reduce_glycan(Path(task["source"]) / glycan_id, staged, task["levels"], task["copy"])
        record = {"status": "done", "error": None, **record, "seconds": round(time.perf_counter() - started, 3)}
        _write_json(record_path, record)            # before the folder appears: a folder there has a record
        os.rename(staged, Path(task["dest"]) / glycan_id)
        return record
    except Exception as exc:
        shutil.rmtree(staged, ignore_errors=True)
        record_path.unlink(missing_ok=True)
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "seconds": round(time.perf_counter() - started, 3),
        }


# ---------------------------------------------------------------------------
# The database
# ---------------------------------------------------------------------------
def _compare_inventories(source_path: Path, dest_path: Path, levels: Tuple[int, ...]) -> Dict[str, Any]:
    """The new ``GLYCOSHAPE.json`` against the one of the source without its entries of other levels."""
    old, new = json.loads(source_path.read_bytes()), json.loads(dest_path.read_bytes())

    def text(entry: Any) -> str:
        return json.dumps(entry, ensure_ascii=False)

    different = sorted(
        glycan_id for glycan_id in old.keys() & new.keys()
        if text(data_without_other_levels(old[glycan_id], levels)[0]) != text(new[glycan_id])
    )
    return {
        "entries_in_source": len(old),
        "entries": len(new),
        "only_in_source": sorted(old.keys() - new.keys()),
        "only_in_destination": sorted(new.keys() - old.keys()),
        "different": different,
        "same": len(old.keys() & new.keys()) - len(different),
    }


def _compare_with_source_manifest(
    source: Path, manifest: Dict[str, Any], records: Dict[str, Dict[str, Any]], root_kept: List[str]
) -> Dict[str, Any]:
    """The files of the source that went into the new database against the ``sync_manifest.json`` of the source.

    A kept file has the SHA-256 that the new manifest records for it; of a
    rewritten file the SHA-256 of what was read is in the record of its
    glycan.  Both have to be what the manifest of the source says, or the
    source is not the database that manifest describes (an incomplete copy,
    a file changed afterwards).
    """
    try:
        old = json.loads((source / MANIFEST_NAME).read_bytes())
        old_root, old_glycans = old["root_files"], old["glycans"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"compared": False, "reason": f"{type(exc).__name__}: {exc}"}
    same, unlisted, different = 0, 0, []

    def compare(path: str, recorded: Any, sha256: str) -> None:
        nonlocal same, unlisted
        if not isinstance(recorded, dict) or "sha256" not in recorded:
            unlisted += 1
        elif str(recorded["sha256"]).lower() == sha256:
            same += 1
        else:
            different.append(path)

    for name in root_kept:
        compare(name, old_root.get(name), manifest["root_files"][name]["sha256"])
    without_record = 0
    for glycan_id, record in records.items():
        if record.get("kept") is None:
            without_record += 1
            continue
        recorded = (old_glycans.get(glycan_id) or {}).get("files") or {}
        for name in record["kept"]:
            compare(f"{glycan_id}/{name}", recorded.get(name), manifest["glycans"][glycan_id]["files"][name]["sha256"])
        for name, sha256 in (record.get("rewritten_from_sha256") or {}).items():
            compare(f"{glycan_id}/{name}", recorded.get(name), sha256)
    return {
        "compared": True,
        "source_manifest_generated_at": old.get("generated_at"),
        "same": same,
        "not_in_source_manifest": unlisted,
        "different": len(different),
        "different_first": sorted(different)[:50],
        "glycans_without_record": without_record,
    }


def finalize_database(
    source: Path,
    dest: Path,
    levels: Iterable[int],
    workers: int = 1,
    records: Optional[Dict[str, Dict[str, Any]]] = None,
    log: Log = print,
) -> Dict[str, Any]:
    """Root files of the new database; ``sync_manifest.json`` is written last.

    ``dest`` holds the finished glycan folders.  With ``records`` (glycan ->
    what :func:`reduce_glycan` returned) the files that were taken from the
    source are compared with its manifest, when it has one.  Raises
    :class:`StaticLevelsError` when a step fails or the result is not what
    the publisher accepts; the destination has no manifest then.
    """
    from lib import static_publish
    from lib.glybake import GlycoShapeBaker

    source, dest = Path(source), Path(dest)
    levels = tuple(int(level) for level in levels)
    glycan_ids = _glycan_ids(dest)
    if not glycan_ids or glycan_ids != _glycan_ids(source):
        raise StaticLevelsError(f"{dest} does not hold the glycan folders of {source}")
    timings: Dict[str, float] = {}
    report: Dict[str, Any] = {
        "regenerated": [], "kept": [], "unknown_kept": [], "left_out": [], "hidden_skipped": [],
        "ignored_directories": [], "bytes_before": 0, "bytes_after": 0, "checks": {},
        "warnings": [], "seconds": timings,
    }

    def timed(name: str, started: float) -> None:
        timings[name] = round(time.perf_counter() - started, 3)

    # The functions below write these names in place.  None of them may be a
    # link to a file of the source, and a database without a manifest cannot
    # be published: all of them go first.
    for name in _WRITTEN_ROOT_FILES:
        (dest / name).unlink(missing_ok=True)

    # 1. Root files of the source that stay as they are.
    started = time.perf_counter()
    for entry in sorted(source.iterdir()):
        if entry.name.startswith("."):
            report["hidden_skipped"].append(entry.name)
        elif entry.is_dir():
            if not entry.name.startswith("GS"):
                report["ignored_directories"].append(entry.name)
        elif entry.is_file():
            report["bytes_before"] += entry.stat().st_size
            if entry.name in REGENERATED_ROOT_FILES:
                continue
            if entry.name in _WRITTEN_ROOT_FILES:      # the temporary file of an interrupted manifest
                report["left_out"].append(entry.name)
                continue
            target = dest / entry.name
            target.unlink(missing_ok=True)
            # Copied, not linked: a later finalization in the new database writes these in place.
            _place(entry, target, copy=True)
            report["kept" if entry.name in LEVEL_INDEPENDENT_ROOT_FILES else "unknown_kept"].append(entry.name)
    expected = set(report["kept"]) | set(report["unknown_kept"]) | set(glycan_ids)
    unexpected = sorted(path.name for path in dest.iterdir() if path.name not in expected)
    if unexpected:
        raise StaticLevelsError(f"{dest} holds something that is not of this database: {', '.join(unexpected[:5])}")
    timed("root_files", started)

    baker = GlycoShapeBaker(dest)
    with _local_storage():
        # 2. GLYCOSHAPE.json from the data.json files.
        started = time.perf_counter()
        baker.create_consolidated_json()
        inventory = json.loads((dest / INVENTORY_NAME).read_bytes())
        if sorted(inventory) != glycan_ids:
            absent = sorted(set(glycan_ids) - set(inventory))
            raise StaticLevelsError(f"{INVENTORY_NAME} lacks {len(absent)} glycan(s), first {absent[:5]}")
        report["regenerated"].append(INVENTORY_NAME)
        timed("inventory", started)

        # 3. The archives.  data.json is in every GSxxxxx.zip, so all are built again, the way the
        # baker builds them.  create_master_archive builds them once more itself (and raises BakeError
        # when one fails); the first pass in parallel and the comparison stay as the check of that.
        started = time.perf_counter()

        def archive(glycan_id: str) -> Tuple[str, bool]:
            return glycan_id, baker.create_individual_glycan_archive(dest / glycan_id)

        failed = sorted(glycan_id for glycan_id, built in parallel_map(archive, glycan_ids, workers) if not built)
        if failed:
            raise StaticLevelsError(f"the archive of {len(failed)} glycan(s) could not be built, first {failed[:5]}")
        first = {glycan_id: _fingerprint(dest / glycan_id / f"{glycan_id}.zip") for glycan_id in glycan_ids}
        timed("glycan_archives", started)
        started = time.perf_counter()
        baker.create_master_archive()
        changed = sorted(
            glycan_id for glycan_id in glycan_ids
            if _fingerprint(dest / glycan_id / f"{glycan_id}.zip") != first[glycan_id]
        )
        if changed:
            raise StaticLevelsError(
                f"{len(changed)} archive(s) came out differently when {MASTER_ARCHIVE_NAME} was built, first {changed[:5]}"
            )
        with zipfile.ZipFile(dest / MASTER_ARCHIVE_NAME) as master:
            members = {info.filename: (info.file_size, info.CRC) for info in master.infolist()}
            damaged = master.testzip()
        if damaged or members != {f"{glycan_id}.zip": first[glycan_id][:2] for glycan_id in glycan_ids}:
            raise StaticLevelsError(f"{MASTER_ARCHIVE_NAME} does not hold the archive of every glycan")
        report["regenerated"].append(MASTER_ARCHIVE_NAME)
        timed("master_archive", started)

    # 4. The manifest, last.
    started = time.perf_counter()
    manifest_path = baker.create_sync_manifest()
    report["regenerated"].append(MANIFEST_NAME)
    timed("manifest", started)

    # 5. What the publisher and the website require of it.
    started = time.perf_counter()
    try:
        manifest = static_publish.parse_manifest(manifest_path.read_bytes(), str(manifest_path))
    except static_publish.PublishError as exc:
        manifest_path.unlink(missing_ok=True)
        raise StaticLevelsError(str(exc)) from exc
    listed = static_publish.manifest_files(manifest)
    present = {name: (dest / name).stat().st_size for name in _files_below(dest) if name != MANIFEST_NAME}
    if present != {name: entry["size"] for name, entry in listed.items()}:
        odd = sorted(set(present) ^ set(listed))[:5]
        manifest_path.unlink(missing_ok=True)
        raise StaticLevelsError(f"{MANIFEST_NAME} does not list exactly the files of {dest} (for example {odd})")
    report["checks"]["manifest"] = {
        "accepted_by_the_publisher": True,
        "lists_every_file": True,
        "files": len(listed),
        "glycans": len(manifest["glycans"]),
        "global_hash": manifest["global_hash"],
    }
    if (source / INVENTORY_NAME).is_file():
        try:
            compared = _compare_inventories(source / INVENTORY_NAME, dest / INVENTORY_NAME, levels)
        except (ValueError, AttributeError) as exc:
            compared = {"compared": False, "reason": f"{type(exc).__name__}: {exc}"}
        report["checks"]["inventory_against_source"] = compared
        if compared.get("different") or compared.get("only_in_source") or compared.get("only_in_destination"):
            report["warnings"].append(
                f"{INVENTORY_NAME} is not the one of the source without the other levels: "
                f"{len(compared['different'])} entries differ, {len(compared['only_in_source'])} only in the source, "
                f"{len(compared['only_in_destination'])} only in the destination (it is built from the data.json files)"
            )
    if (source / MANIFEST_NAME).is_file() and records is not None:
        compared = _compare_with_source_manifest(source, manifest, records, report["kept"] + report["unknown_kept"])
        report["checks"]["source_files_against_source_manifest"] = compared
        if compared.get("different"):
            report["warnings"].append(
                f"{compared['different']} file(s) of the source do not have the content that its {MANIFEST_NAME} "
                f"records, first {compared['different_first'][:3]}"
            )
    report["bytes_after"] = sum(size for name, size in present.items() if "/" not in name)
    report["bytes_after"] += manifest_path.stat().st_size
    timed("checks", started)
    for name in REGENERATED_ROOT_FILES:
        log(f"wrote {name}")
    return report


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
def _prepare(source: Path, dest: Path, work: Path, report_path: Path, levels: Tuple[int, ...]) -> bool:
    """Check the folders and claim the work folder; returns whether this continues an earlier run."""
    if not source.is_dir():
        raise StaticLevelsError(f"the source is not a folder: {source}")
    if _is_inside(dest, source) or _is_inside(source, dest):
        raise StaticLevelsError("the destination must be a new folder outside the source")
    for path, what in ((work, "work folder"), (report_path, "report")):
        if _is_inside(path, source) or _is_inside(path, dest):
            raise StaticLevelsError(f"the {what} must be outside the source and the destination: {path}")
    if not _glycan_ids(source):
        raise StaticLevelsError(f"no glycan folder (GS*) in {source}")
    if dest.exists() and not dest.is_dir():
        raise StaticLevelsError(f"the destination is not a folder: {dest}")
    state_path = work / "state.json"
    state = {"tool": "static_drop_levels", "source": str(source), "dest": str(dest), "levels": list(levels)}
    started_before = dest.is_dir() and any(dest.iterdir())
    if started_before:
        try:
            previous = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise StaticLevelsError(
                f"{dest} is not empty and {state_path} does not say that this tool started it; "
                "choose a new destination"
            ) from None
        if previous != state:
            raise StaticLevelsError(
                f"{dest} was started from {previous.get('source')} with levels {previous.get('levels')}; "
                "choose a new destination"
            )
    else:   # a new run: nothing of an earlier one applies
        for stale in (work / "glycans", work / "staging"):
            if stale.exists():
                shutil.rmtree(stale)
    for folder in (dest, work / "glycans", work / "staging", report_path.parent):
        folder.mkdir(parents=True, exist_ok=True)
    if os.stat(work).st_dev != os.stat(dest).st_dev:
        raise StaticLevelsError(f"the work folder must be on the file system of the destination: {work}")
    _write_json(state_path, state)
    return started_before


def drop_levels(
    source: Path,
    dest: Path,
    levels: Iterable[int] = (1, 2),
    workers: int = 1,
    work_dir: Optional[Path] = None,
    report_path: Optional[Path] = None,
    copy: bool = False,
    finalize: bool = True,
    log: Log = print,
) -> Dict[str, Any]:
    """Write the static database ``source`` reduced to ``levels`` into ``dest``; returns the report.

    See the module documentation.  ``report["ok"]`` is false when a glycan
    failed (the others are done, the root files are not written) or the
    database could not be completed; run again to continue.  Raises
    :class:`StaticLevelsError` when the folders cannot be used.
    """
    started, started_at = time.perf_counter(), _utc_now()
    source, dest = Path(source).resolve(), Path(dest).resolve()
    work = Path(work_dir).resolve() if work_dir else dest.with_name(dest.name + ".work")
    report_path = Path(report_path).resolve() if report_path else work / "report.json"
    try:
        levels = config.parse_cluster_levels(levels if isinstance(levels, str) else list(levels))
    except (TypeError, ValueError) as exc:
        raise StaticLevelsError(str(exc)) from exc
    workers = max(1, int(workers))
    resumed = _prepare(source, dest, work, report_path, levels)

    lock = open(work / "lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        raise StaticLevelsError(f"another run is using {work}") from None
    report: Dict[str, Any] = {
        "tool": "static_drop_levels",
        "ok": False,
        "source": str(source), "dest": str(dest), "work_dir": str(work), "report": str(report_path),
        "levels": list(levels), "workers": workers, "copy": bool(copy), "resumed": resumed,
        "started_at": started_at, "finished_at": None, "seconds": {},
        "finalized": False, "error": None, "warnings": [], "totals": {}, "root": None, "glycans": {},
    }
    records: Dict[str, Dict[str, Any]] = {}
    interrupted: Optional[BaseException] = None
    try:
        glycan_ids = _glycan_ids(source)
        todo = []
        for glycan_id in glycan_ids:
            if (dest / glycan_id).is_dir():
                try:
                    earlier = json.loads((work / "glycans" / f"{glycan_id}.json").read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    earlier = {"notes": ["finished by an earlier run; its record is not in the work folder"]}
                records[glycan_id] = {**earlier, "status": "skipped", "error": None}
            else:
                todo.append(glycan_id)
        log(
            f"{len(glycan_ids)} glycans in {source}: {len(todo)} to do, {len(glycan_ids) - len(todo)} already in {dest}; "
            f"levels {','.join(map(str, levels))}, {min(workers, max(1, len(todo)))} worker(s)"
        )
        tasks = [
            {"glycan_id": glycan_id, "source": str(source), "dest": str(dest), "staging": str(work / "staging"),
             "records": str(work / "glycans"), "levels": levels, "copy": bool(copy)}
            for glycan_id in todo
        ]
        stage_started = time.perf_counter()

        def note(glycan_id: str, record: Dict[str, Any]) -> None:
            records[glycan_id] = record
            done = sum(1 for name in todo if name in records)
            if record["status"] == "failed":
                log(f"FAILED {glycan_id}: {record['error']}")
            if done % 50 == 0 or done == len(todo):
                failed = sum(1 for name in todo if records.get(name, {}).get("status") == "failed")
                log(f"{done}/{len(todo)} glycans, {failed} failed, {time.perf_counter() - stage_started:.0f} s")

        try:
            if workers == 1 or len(tasks) <= 1:
                for task in tasks:
                    note(task["glycan_id"], _run_glycan(task))
            else:
                pool = ProcessPoolExecutor(
                    max_workers=min(workers, len(tasks)),
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_initialize_worker,
                    initargs=(logging.getLogger().getEffectiveLevel(),),
                )
                try:
                    futures = {pool.submit(_run_glycan, task): task["glycan_id"] for task in tasks}
                    for future in as_completed(futures):
                        try:
                            record = future.result()
                        except Exception as exc:   # the worker process died
                            record = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "seconds": None}
                        note(futures[future], record)
                finally:
                    pool.shutdown(wait=True, cancel_futures=True)
        except KeyboardInterrupt as exc:
            interrupted = exc
            for glycan_id in todo:      # finished by a worker while the run was being stopped
                if glycan_id not in records and (dest / glycan_id).is_dir():
                    try:
                        records[glycan_id] = json.loads((work / "glycans" / f"{glycan_id}.json").read_text("utf-8"))
                    except (OSError, ValueError):
                        pass
        report["seconds"]["glycans"] = round(time.perf_counter() - stage_started, 3)
        shutil.rmtree(work / "staging", ignore_errors=True)

        failed = sorted(name for name, record in records.items() if record["status"] == "failed")
        pending = [name for name in glycan_ids if name not in records]
        if interrupted is None and not failed and not pending and finalize:
            built = any(record["status"] == "done" for record in records.values())
            if built or not (dest / MANIFEST_NAME).is_file():
                finalize_started = time.perf_counter()
                try:
                    report["root"] = finalize_database(source, dest, levels, workers=workers, records=records, log=log)
                    report["finalized"] = True
                    report["warnings"].extend(report["root"]["warnings"])
                except KeyboardInterrupt as exc:
                    interrupted = exc
                except Exception as exc:
                    report["error"] = f"the database was not completed: {type(exc).__name__}: {exc}"
                    log(f"FAILED {report['error']}")
                report["seconds"]["finalize"] = round(time.perf_counter() - finalize_started, 3)
            else:
                report["finalized"] = True
                log(f"{dest} is complete ({MANIFEST_NAME} is there); delete that file to write the root files again")

        # The report: per glycan what was done, with the archive of the finalization.
        for glycan_id in glycan_ids:
            record = dict(records.get(glycan_id) or {"status": "pending", "error": None})
            if record["status"] in ("done", "skipped"):
                if "bytes_staged" not in record:   # no record of the run that made it
                    record["bytes_before"] = _folder_bytes(source / glycan_id)
                archive = dest / glycan_id / f"{glycan_id}.zip"
                record["archive_bytes_after"] = archive.stat().st_size if archive.is_file() else None
                record["bytes_after"] = _folder_bytes(dest / glycan_id)
                for note_text in record.get("notes", []):
                    report["warnings"].append(f"{glycan_id}: {note_text}")
            report["glycans"][glycan_id] = record
        finished = [record for record in report["glycans"].values() if record["status"] in ("done", "skipped")]
        report["totals"] = {
            "glycans": len(glycan_ids),
            "done": sum(1 for record in records.values() if record["status"] == "done"),
            "skipped": sum(1 for record in records.values() if record["status"] == "skipped"),
            "failed": len(failed),
            "pending": len(pending),
            "files_kept": sum(len(record.get("kept") or []) for record in finished),
            "files_rewritten": sum(len(record.get("rewritten") or []) for record in finished),
            "files_dropped": sum(len(record.get("dropped") or []) for record in finished),
            "files_extra": sum(len(record.get("extra") or []) for record in finished),
            "files_hidden_skipped": sum(len(record.get("hidden_skipped") or []) for record in finished),
            "files_linked": sum(record.get("linked") or 0 for record in finished),
            "files_copied": sum(record.get("copied") or 0 for record in finished),
            "bytes_before": sum(record.get("bytes_before") or 0 for record in finished),
            "bytes_after": sum(record.get("bytes_after") or 0 for record in finished),
        }
        if interrupted is not None:
            report["error"] = "interrupted"
        elif failed:
            report["error"] = f"{len(failed)} glycan(s) failed: {', '.join(failed[:10])}" + (
                " ..." if len(failed) > 10 else "")
        report["ok"] = report["error"] is None and not pending and (report["finalized"] or not finalize)
    finally:
        report["finished_at"] = _utc_now()
        report["seconds"]["total"] = round(time.perf_counter() - started, 3)
        _write_json(report_path, report)
        lock.close()
    if interrupted is not None:
        raise interrupted
    return report
