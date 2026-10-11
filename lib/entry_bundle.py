"""The entry bundle: one file that carries a new or extended database entry from the node to publishing.

A bundle is a plain ``tar`` file (ustar, not compressed) whose members are, in
this order::

    bundle.json     what the bundle is: schema, GLYCAM name, creation time, pipeline, levels, size and
                    SHA-256 of every other member, what it was made from, the entry it extends
    archive/...     the raw archive entry as lib/archive.py and lib/archive_segments.py write it:
                    trajectory.xtc, topology.pdb, structure.mol2, manifest.json, info.json, name.json
    static/...      the static-ready folder as lib/entry_run.py writes it (output/..., landscape/...)
    result.json     the result of the analysis, as lib/entry_run.py writes it
    run.log         the log of the analysis

``bundle.json``::

    {
      "schema": "glycoshape-entry-bundle",
      "schema_version": 1,
      "glycam_name": "...",
      "created": "2026-10-11T20:00:00Z",                         UTC
      "pipeline": {"commit", "uncommitted_changes", "version", "version_of", "converter_version",
                   "python", "packages"},                        version is the SHA-256 of the sources
      "levels": [1, 2],
      "files": {"archive/trajectory.xtc": {"size": 1, "sha256": "..."}, ...},   every other member, in order
      "source": {...},                                           what it was made from (lib/entry_make.py)
      "extends": null | {"glycam_name", "manifest_sha256", "frames"}
    }

:func:`write_bundle` writes one from an archive entry and a result folder.
:func:`open_bundle` unpacks one into a new folder with exactly this layout
and returns it only when everything checks: a bundle is made under the
account of a member on a cluster and unpacked on a server, so it is read as
untrusted input.  The tar reader here is deliberately small: it accepts
regular files with plain ustar headers and nothing else (no links, devices,
directories, sparse files, extended or long-name headers), in the order and
with the sizes ``bundle.json`` announces.

Only the standard library is used, so the checks run wherever a bundle is
received.  ``lib.archive.verify_entry(<folder>/archive, check_folder=False)``
additionally decodes the trajectory (it needs MDTraj).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import math
import os
import re
import secrets
import shutil
import tarfile
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterator, List, Optional, Tuple, Union

PathLike = Union[str, "os.PathLike[str]"]

BUNDLE_SCHEMA = "glycoshape-entry-bundle"
BUNDLE_SCHEMA_VERSION = 1
BUNDLE_FILE = "bundle.json"
ARCHIVE_DIR = "archive"
STATIC_DIR = "static"
RESULT_FILE = "result.json"
LOG_FILE = "run.log"

# Names of lib/archive.py and lib/entry_run.py, repeated here so that this module needs neither
# (tests/test_entry_bundle.py holds them against the originals).
ARCHIVE_SCHEMA = "glycoshape-raw-trajectory"
ARCHIVE_SCHEMA_VERSION = 1
RESULT_SCHEMA = "glycoshape-entry-result"
MANIFEST_FILE = "manifest.json"
#: Files of an archive entry, in the order they have in a bundle.
ARCHIVE_FILES = ("trajectory.xtc", "topology.pdb", "structure.mol2", MANIFEST_FILE, "info.json", "name.json")
REQUIRED_ARCHIVE_FILES = ("trajectory.xtc", "topology.pdb", "structure.mol2", MANIFEST_FILE)
#: Static-ready files every bundle has, and those it has per level (as lib/entry_run.py requires them).
REQUIRED_STATIC_FILES = ("output/info.json", "output/pca.csv")
REQUIRED_LEVEL_FILES = ("dist.svg", "GLYCAM/alpha.pdb", "GLYCAM/beta.pdb", "PDB/alpha.pdb", "PDB/beta.pdb")

#: Limits of :func:`open_bundle`.  The largest trajectory of the database is 0.3 GB.
DEFAULT_MAX_BYTES = 16 << 30
DEFAULT_MAX_MEMBER_BYTES = (8 << 30) - 1       # what a ustar header can say
MAX_BUNDLE_JSON_BYTES = 8 << 20
MAX_JSON_MEMBER_BYTES = 64 << 20
MAX_MEMBERS = 4096
MAX_PATH_CHARS = 200
MAX_PATH_DEPTH = 8
MAX_TEXT_CHARS = 4000
#: Zero bytes accepted after the end of the archive (tar pads to a record of 10 KiB).
MAX_TRAILING_BYTES = 1 << 20

_BLOCK = 512
_COPY_BLOCK = 1 << 20
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PATH_PART = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.+-]*\Z")
_TAR_TYPES = {
    b"1": "hard link", b"2": "symbolic link", b"3": "character device", b"4": "block device", b"5": "directory",
    b"6": "FIFO", b"7": "contiguous file", b"x": "extended header", b"g": "global extended header",
    b"L": "GNU long name", b"K": "GNU long link name", b"S": "GNU sparse file", b"D": "GNU directory dump",
    b"M": "GNU multi-volume member", b"V": "GNU volume label",
}
_NUMBER_FIELDS = (
    "simulation_length_ns", "time_per_frame_ps", "temperature_k", "pressure_bar", "salt_concentration_mm",
)
_TEXT_FIELDS = ("name", "md_package", "force_field", "comments", "glytoucan_id", "run_id")


class BundleError(Exception):
    """A bundle cannot be written, or is not one that may be unpacked."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> Tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(_COPY_BLOCK)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def _no_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"key {key!r} occurs twice")
        out[key] = value
    return out


def _refuse_constant(text: str) -> Any:
    raise ValueError(f"{text} is not a JSON number")


def _load_json(payload: bytes, what: str) -> Dict[str, Any]:
    """A JSON object from untrusted bytes: UTF-8, no duplicate keys, no NaN or Infinity."""
    try:
        data = json.loads(payload.decode("utf-8"), object_pairs_hook=_no_duplicate_keys, parse_constant=_refuse_constant)
    except (ValueError, RecursionError) as exc:
        raise BundleError(f"{what} is not readable JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise BundleError(f"{what} is not a JSON object")
    return data


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def check_member_path(path: Any) -> str:
    """A member path as a bundle may have it; raises :class:`BundleError` for anything else.

    Relative, ``/``-separated, every part made of letters, digits and
    ``_ . + -`` and not starting with a dot (so no ``..``, no hidden file,
    no backslash, no control character), at most 8 parts and 200 characters,
    and one of: ``archive/<file of an entry>``, ``static/...``,
    ``result.json``, ``run.log``.
    """
    if not isinstance(path, str) or not path:
        raise BundleError(f"member path {path!r} is not a non-empty string")
    if len(path) > MAX_PATH_CHARS:
        raise BundleError(f"member path {path[:60]!r}... is longer than {MAX_PATH_CHARS} characters")
    if path.startswith("/"):
        raise BundleError(f"member path {path!r} is absolute")
    parts = path.split("/")
    for part in parts:
        if part in ("", ".", ".."):
            raise BundleError(f"member path {path!r} has the part {part!r}")
        if not _PATH_PART.match(part):
            raise BundleError(
                f"member path {path!r}: {part!r} is not a plain file name (letters, digits and _ . + -, "
                "not starting with a dot)"
            )
    if len(parts) > MAX_PATH_DEPTH:
        raise BundleError(f"member path {path!r} is more than {MAX_PATH_DEPTH} folders deep")
    if path in (RESULT_FILE, LOG_FILE):
        return path
    if parts[0] == ARCHIVE_DIR and len(parts) == 2 and parts[1] in ARCHIVE_FILES:
        return path
    if parts[0] == STATIC_DIR and len(parts) >= 2:
        return path
    raise BundleError(
        f"member {path!r} is not a file of a bundle ({ARCHIVE_DIR}/<file of an entry>, {STATIC_DIR}/..., "
        f"{RESULT_FILE}, {LOG_FILE})"
    )


def _check_glycam_name(name: Any) -> str:
    """The rules of ``lib.archive`` for a GLYCAM name (it becomes part of an object key)."""
    if not isinstance(name, str) or not name:
        raise BundleError("GLYCAM name is missing or empty")
    if name != name.strip() or re.search(r"[\x00-\x1f\x7f]", name):
        raise BundleError(f"GLYCAM name has surrounding whitespace or control characters: {name!r}")
    if any(part in ("", ".", "..") for part in name.split("/")):
        raise BundleError(f"GLYCAM name is not usable as an object key: {name!r}")
    if len(name.encode("utf-8")) > 1000:
        raise BundleError("GLYCAM name is longer than 1000 bytes")
    return name


def member_order(paths: List[str]) -> List[str]:
    """The order of the members of a bundle after ``bundle.json``."""
    archive = [f"{ARCHIVE_DIR}/{name}" for name in ARCHIVE_FILES if f"{ARCHIVE_DIR}/{name}" in paths]
    static = sorted(path for path in paths if path.startswith(STATIC_DIR + "/"))
    rest = [name for name in (RESULT_FILE, LOG_FILE) if name in paths]
    return archive + static + rest


def validate_description(bundle: Dict[str, Any], max_bytes: int, max_member_bytes: int) -> Dict[str, Any]:
    """Check the content of ``bundle.json``; returns it.  Raises :class:`BundleError`."""
    if bundle.get("schema") != BUNDLE_SCHEMA:
        raise BundleError(f"{BUNDLE_FILE} is not a {BUNDLE_SCHEMA} description (schema {bundle.get('schema')!r})")
    if bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION or isinstance(bundle.get("schema_version"), bool):
        raise BundleError(
            f"{BUNDLE_FILE} has schema version {bundle.get('schema_version')!r}, this code reads version "
            f"{BUNDLE_SCHEMA_VERSION}"
        )
    _check_glycam_name(bundle.get("glycam_name"))
    if not isinstance(bundle.get("created"), str) or len(bundle["created"]) > 40:
        raise BundleError(f"{BUNDLE_FILE}: created is not a time stamp")
    if not isinstance(bundle.get("pipeline"), dict):
        raise BundleError(f"{BUNDLE_FILE}: pipeline is missing")
    levels = bundle.get("levels")
    if levels not in ([1, 2], [1, 2, 3]) or any(isinstance(level, bool) for level in levels):
        raise BundleError(f"{BUNDLE_FILE}: levels must be [1, 2] or [1, 2, 3], not {levels!r}")

    files = bundle.get("files")
    if not isinstance(files, dict) or not files:
        raise BundleError(f"{BUNDLE_FILE}: files is missing or empty")
    if len(files) > MAX_MEMBERS:
        raise BundleError(f"{BUNDLE_FILE} lists {len(files)} members, at most {MAX_MEMBERS} are accepted")
    total = 0
    for path, record in files.items():
        check_member_path(path)
        if not isinstance(record, dict) or not _is_count(record.get("size")) \
                or not isinstance(record.get("sha256"), str) or not _SHA256.match(record["sha256"]):
            raise BundleError(f"{BUNDLE_FILE}: files[{path!r}] needs a size and a SHA-256 of 64 hexadecimal digits")
        if record["size"] > max_member_bytes:
            raise BundleError(f"member {path} has {record['size']} bytes, at most {max_member_bytes} are accepted")
        total += record["size"]
    if total > max_bytes:
        raise BundleError(f"the members have {total} bytes together, at most {max_bytes} are accepted")
    folded = [path.lower() for path in files]
    if len(set(folded)) != len(folded):
        raise BundleError(f"{BUNDLE_FILE}: two member paths differ only in the case of their letters")
    for path in files:
        if any(other.startswith(path + "/") for other in files):
            raise BundleError(f"{BUNDLE_FILE}: {path} is listed as a file and as a folder")
    if list(files) != member_order(list(files)):
        raise BundleError(f"{BUNDLE_FILE}: files is not in the order of a bundle (archive, static, result, log)")
    required = [f"{ARCHIVE_DIR}/{name}" for name in REQUIRED_ARCHIVE_FILES] + [RESULT_FILE, LOG_FILE]
    required += [f"{STATIC_DIR}/{name}" for name in REQUIRED_STATIC_FILES]
    required += [f"{STATIC_DIR}/output/level_{level}/{name}" for level in levels for name in REQUIRED_LEVEL_FILES]
    missing = [path for path in required if path not in files]
    if missing:
        raise BundleError(f"{BUNDLE_FILE} does not list {', '.join(missing)}")

    source = bundle.get("source")
    if not isinstance(source, dict):
        raise BundleError(f"{BUNDLE_FILE}: source is missing")
    for key in _NUMBER_FIELDS:
        value = source.get(key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
        ):
            raise BundleError(f"{BUNDLE_FILE}: source.{key} is {value!r}, not a number that is zero or more")
    for key in _TEXT_FIELDS:
        value = source.get(key)
        if value is not None and (not isinstance(value, str) or len(value) > MAX_TEXT_CHARS):
            raise BundleError(f"{BUNDLE_FILE}: source.{key} is not a text of at most {MAX_TEXT_CHARS} characters")

    extends = bundle.get("extends", False)
    if extends is False:
        raise BundleError(f"{BUNDLE_FILE}: extends is missing (null for an entry that extends none)")
    if extends is not None:
        if not isinstance(extends, dict):
            raise BundleError(f"{BUNDLE_FILE}: extends is neither null nor an object")
        _check_glycam_name(extends.get("glycam_name"))
        if not isinstance(extends.get("manifest_sha256"), str) or not _SHA256.match(extends["manifest_sha256"]):
            raise BundleError(f"{BUNDLE_FILE}: extends.manifest_sha256 is not a SHA-256")
        if not _is_count(extends.get("frames")) or extends["frames"] < 1:
            raise BundleError(f"{BUNDLE_FILE}: extends.frames is not a number of frames")
        if extends["glycam_name"] != bundle["glycam_name"]:
            raise BundleError(
                f"{BUNDLE_FILE}: the bundle is {bundle['glycam_name']} and extends {extends['glycam_name']}"
            )
    return bundle


def check_contents(bundle: Dict[str, Any], manifest: Dict[str, Any], result: Dict[str, Any]) -> None:
    """``archive/manifest.json`` and ``result.json`` have to describe what ``bundle.json`` lists.

    The three were read from the bundle; the sizes and SHA-256 of the files
    themselves were compared with ``bundle.json`` while they were read.
    """
    name = bundle["glycam_name"]
    files = bundle["files"]

    def record(path: str) -> Dict[str, Any]:
        return {"size": files[path]["size"], "sha256": files[path]["sha256"]}

    try:
        if manifest.get("schema") != ARCHIVE_SCHEMA or manifest.get("schema_version") != ARCHIVE_SCHEMA_VERSION:
            raise BundleError(f"{ARCHIVE_DIR}/{MANIFEST_FILE} is not a {ARCHIVE_SCHEMA} manifest of version 1")
        if manifest.get("glycam_name") != name:
            raise BundleError(
                f"{ARCHIVE_DIR}/{MANIFEST_FILE} is the entry {manifest.get('glycam_name')!r}, {BUNDLE_FILE} says {name!r}"
            )
        in_bundle = sorted(path.split("/", 1)[1] for path in files
                           if path.startswith(ARCHIVE_DIR + "/") and path != f"{ARCHIVE_DIR}/{MANIFEST_FILE}")
        listed = manifest["files"]
        if not isinstance(listed, dict) or sorted(listed) != in_bundle:
            raise BundleError(
                f"{ARCHIVE_DIR}/{MANIFEST_FILE} lists {', '.join(sorted(listed)) or 'no files'}; the bundle has "
                f"{', '.join(in_bundle)}"
            )
        for entry_file, described in listed.items():
            if {"size": described["size"], "sha256": described["sha256"]} != record(f"{ARCHIVE_DIR}/{entry_file}"):
                raise BundleError(
                    f"{ARCHIVE_DIR}/{entry_file} is not the file {ARCHIVE_DIR}/{MANIFEST_FILE} describes "
                    "(size or SHA-256)"
                )
        if not _is_count(manifest["frames"]) or not _is_count(manifest["atoms"]):
            raise BundleError(f"{ARCHIVE_DIR}/{MANIFEST_FILE} has no frame and atom counts")

        if result.get("schema") != RESULT_SCHEMA:
            raise BundleError(f"{RESULT_FILE} is not a {RESULT_SCHEMA} result")
        if result.get("status") != "ok":
            raise BundleError(f"{RESULT_FILE} has status {result.get('status')!r}, not ok")
        if result.get("glycam_name") != name or result["input"].get("glycam_name") != name:
            raise BundleError(f"{RESULT_FILE} is the result of {result.get('glycam_name')!r}, {BUNDLE_FILE} says {name!r}")
        analysed = result["input"]
        if analysed.get("kind") != "archive":
            raise BundleError(f"{RESULT_FILE} was not computed from an archive entry")
        if analysed.get("manifest_sha256") != files[f"{ARCHIVE_DIR}/{MANIFEST_FILE}"]["sha256"]:
            raise BundleError(f"{RESULT_FILE} was computed from another {MANIFEST_FILE} than the one in the bundle")
        if analysed.get("files") != {key: {"size": value["size"], "sha256": value["sha256"]}
                                     for key, value in listed.items()}:
            raise BundleError(f"{RESULT_FILE} was computed from other files than the entry in the bundle")
        if (analysed.get("frames"), analysed.get("atoms")) != (manifest["frames"], manifest["atoms"]):
            raise BundleError(f"{RESULT_FILE} and {ARCHIVE_DIR}/{MANIFEST_FILE} disagree on frames and atoms")
        if result["settings"].get("cluster_levels") != bundle["levels"]:
            raise BundleError(
                f"{RESULT_FILE} was computed with levels {result['settings'].get('cluster_levels')!r}, "
                f"{BUNDLE_FILE} says {bundle['levels']!r}"
            )
        produced = {
            f"{STATIC_DIR}/{item['path']}": {"size": item["size"], "sha256": item["sha256"]}
            for item in result["produced"]["static"]["files"]
        }
        in_bundle_static = {path: record(path) for path in files if path.startswith(STATIC_DIR + "/")}
        if produced != in_bundle_static:
            different = sorted(
                path for path in set(produced) | set(in_bundle_static) if produced.get(path) != in_bundle_static.get(path)
            )
            raise BundleError(
                f"the static files of the bundle are not those {RESULT_FILE} records ({len(different)} differ, "
                f"for example {different[0]})"
            )

        extends, recorded = bundle["extends"], manifest.get("extends")
        if (extends is None) != (recorded is None):
            raise BundleError(f"{BUNDLE_FILE} and {ARCHIVE_DIR}/{MANIFEST_FILE} disagree on whether an entry is extended")
        if extends is not None:
            for key in ("glycam_name", "manifest_sha256", "frames"):
                if extends[key] != recorded.get(key):
                    raise BundleError(f"{BUNDLE_FILE} and {ARCHIVE_DIR}/{MANIFEST_FILE} disagree on extends.{key}")
            if not extends["frames"] < manifest["frames"]:
                raise BundleError(f"the entry has {manifest['frames']} frames and says it extends {extends['frames']}")
    except (KeyError, TypeError, AttributeError) as exc:
        raise BundleError(f"incomplete {MANIFEST_FILE} or {RESULT_FILE}: {type(exc).__name__}: {exc}") from exc


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
class _HashingReader:
    """File object that hashes what is read from it."""

    def __init__(self, handle: BinaryIO) -> None:
        self._handle = handle
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        block = self._handle.read(size)
        self.digest.update(block)
        self.size += len(block)
        return block


class _HashingWriter:
    """File object that hashes what is written to it."""

    def __init__(self, handle: BinaryIO) -> None:
        self._handle = handle
        self.digest = hashlib.sha256()
        self.size = 0

    def write(self, block: bytes) -> int:
        self._handle.write(block)
        self.digest.update(block)
        self.size += len(block)
        return len(block)

    def tell(self) -> int:
        return self.size

    def flush(self) -> None:
        self._handle.flush()


def _tar_info(name: str, size: int, mtime: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = mtime
    info.mode = 0o644
    info.type = tarfile.REGTYPE
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def describe_pipeline(result: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, Any]:
    """``pipeline`` of ``bundle.json`` from the result of the analysis and the manifest of the entry."""
    pipeline = result.get("pipeline") or {}
    return {
        "commit": pipeline.get("commit"),
        "uncommitted_changes": pipeline.get("uncommitted_changes"),
        "version": pipeline.get("code_sha256"),
        "version_of": pipeline.get("code_sha256_of"),
        "converter_version": (manifest.get("converter") or {}).get("version"),
        "python": pipeline.get("python"),
        "packages": pipeline.get("packages"),
    }


def write_bundle(
    tar_path: PathLike,
    entry_dir: PathLike,
    result_dir: PathLike,
    source: Dict[str, Any],
    created: Optional[str] = None,
) -> Dict[str, Any]:
    """Write the bundle of an archive entry and the result folder of its analysis.

    ``entry_dir`` is the entry (``lib/archive.py``), ``result_dir`` the
    result folder ``lib/entry_run.py`` made of it (``result.json`` with
    status ``ok``, ``run.log``, ``static/``), ``source`` the description of
    what the entry was made from.  The sizes and SHA-256 that ``bundle.json``
    lists are those the manifest and the result record; every file is hashed
    while it is written, and a file that is not what was recorded stops the
    writing.  The bundle appears under its name only when it is complete.

    Returns ``{"path", "size", "sha256", "bundle"}``: the file, and the
    content of its ``bundle.json``.
    """
    tar_path, entry_dir, result_dir = Path(tar_path), Path(entry_dir), Path(result_dir)
    try:
        manifest_bytes = (entry_dir / MANIFEST_FILE).read_bytes()
        result_bytes = (result_dir / RESULT_FILE).read_bytes()
        log_bytes = (result_dir / LOG_FILE).read_bytes()
    except OSError as exc:
        raise BundleError(f"cannot read {exc.filename}: {exc.strerror}") from exc
    manifest = _load_json(manifest_bytes, str(entry_dir / MANIFEST_FILE))
    result = _load_json(result_bytes, str(result_dir / RESULT_FILE))
    if result.get("status") != "ok":
        raise BundleError(f"{result_dir / RESULT_FILE} has status {result.get('status')!r}: there is nothing to bundle")

    # (member path, file or content, size, SHA-256 as recorded)
    members: List[Tuple[str, Union[Path, bytes], int, str]] = []
    try:
        listed = manifest["files"]
        for name in ARCHIVE_FILES:
            if name == MANIFEST_FILE:
                members.append((f"{ARCHIVE_DIR}/{name}", manifest_bytes, len(manifest_bytes),
                                hashlib.sha256(manifest_bytes).hexdigest()))
            elif name in listed:
                members.append((f"{ARCHIVE_DIR}/{name}", entry_dir / name, listed[name]["size"], listed[name]["sha256"]))
        unknown = sorted(set(listed) - set(ARCHIVE_FILES))
        if unknown:
            raise BundleError(f"the manifest lists files a bundle does not carry: {', '.join(unknown)}")
        static = {item["path"]: item for item in result["produced"]["static"]["files"]}
        for path in sorted(static):
            members.append((f"{STATIC_DIR}/{path}", result_dir / STATIC_DIR / path, static[path]["size"],
                            static[path]["sha256"]))
        on_disk = sorted(
            path.relative_to(result_dir / STATIC_DIR).as_posix()
            for path in (result_dir / STATIC_DIR).rglob("*") if path.is_file() or path.is_symlink()
        )
        if on_disk != sorted(static):
            raise BundleError(f"the files under {result_dir / STATIC_DIR} are not those {RESULT_FILE} records")
        extends = manifest.get("extends")
        description = {
            "schema": BUNDLE_SCHEMA,
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "glycam_name": manifest["glycam_name"],
            "created": created or _utc_now(),
            "pipeline": describe_pipeline(result, manifest),
            "levels": result["settings"]["cluster_levels"],
            "files": {},
            "source": source,
            "extends": None if extends is None else {
                key: extends[key] for key in ("glycam_name", "manifest_sha256", "frames")
            },
        }
    except (KeyError, TypeError) as exc:
        raise BundleError(f"incomplete manifest or result: {type(exc).__name__}: {exc}") from exc
    members.append((RESULT_FILE, result_bytes, len(result_bytes), hashlib.sha256(result_bytes).hexdigest()))
    members.append((LOG_FILE, log_bytes, len(log_bytes), hashlib.sha256(log_bytes).hexdigest()))
    description["files"] = {path: {"size": size, "sha256": sha256} for path, _, size, sha256 in members}
    # What is written has to be what open_bundle accepts.
    validate_description(description, DEFAULT_MAX_BYTES, DEFAULT_MAX_MEMBER_BYTES)
    check_contents(description, manifest, result)
    description_bytes = (json.dumps(description, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if len(description_bytes) > MAX_BUNDLE_JSON_BYTES:
        raise BundleError(f"{BUNDLE_FILE} would have {len(description_bytes)} bytes")

    try:
        mtime = int(_dt.datetime.strptime(description["created"], "%Y-%m-%dT%H:%M:%SZ")
                    .replace(tzinfo=_dt.timezone.utc).timestamp())
    except ValueError as exc:
        raise BundleError(f"created must look like 2026-10-11T20:00:00Z, not {description['created']!r}") from exc
    temporary = tar_path.with_name(f"{tar_path.name}.tmp-{os.getpid()}")
    try:
        with open(temporary, "wb") as raw:
            sink = _HashingWriter(raw)
            with tarfile.open(fileobj=sink, mode="w", format=tarfile.USTAR_FORMAT) as tar:
                tar.addfile(_tar_info(BUNDLE_FILE, len(description_bytes), mtime), io.BytesIO(description_bytes))
                for path, content, size, sha256 in members:
                    if isinstance(content, bytes):
                        tar.addfile(_tar_info(path, size, mtime), io.BytesIO(content))
                        continue
                    if content.is_symlink() or not content.is_file():
                        raise BundleError(f"{content} is not a regular file")
                    if content.stat().st_size != size:
                        raise BundleError(f"{content} has {content.stat().st_size} bytes, {size} were recorded")
                    with open(content, "rb") as handle:
                        reader = _HashingReader(handle)
                        tar.addfile(_tar_info(path, size, mtime), reader)
                        if reader.size != size or reader.digest.hexdigest() != sha256 or handle.read(1):
                            raise BundleError(f"{content} is not the file that was recorded (size or SHA-256)")
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, tar_path)
    except (OSError, ValueError, tarfile.TarError) as exc:   # ValueError: a name or size ustar cannot hold
        raise BundleError(f"cannot write {tar_path}: {exc}") from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"path": str(tar_path), "size": sink.size, "sha256": sink.digest.hexdigest(), "bundle": description}


# ---------------------------------------------------------------------------
# Reading (untrusted input)
# ---------------------------------------------------------------------------
class _Stream:
    """Sequential reader of the bundle file that never hands out more than was asked for."""

    def __init__(self, handle: BinaryIO, path: Path) -> None:
        self._handle = handle
        self._path = path
        self.position = 0

    def read_exact(self, size: int, what: str) -> bytes:
        block = self._handle.read(size)
        self.position += len(block)
        if len(block) != size:
            raise BundleError(f"{self._path.name} ends inside {what} (at byte {self.position})")
        return block

    def chunks(self, size: int, what: str) -> Iterator[bytes]:
        remaining = size
        while remaining:
            block = self.read_exact(min(_COPY_BLOCK, remaining), what)
            remaining -= len(block)
            yield block

    def read_some(self, size: int) -> bytes:
        block = self._handle.read(size)
        self.position += len(block)
        return block


def _tar_number(field: bytes) -> int:
    """A numeric header field: octal text, or the GNU binary form."""
    if field[:1] == b"\x80":
        return int.from_bytes(field[1:], "big")
    if field[:1] == b"\xff":
        raise ValueError("negative number")
    text = field.split(b"\0", 1)[0].strip(b" ")
    return int(text, 8) if text else 0


def _next_header(stream: _Stream) -> Optional[Tuple[str, int]]:
    """Name and size of the next member; ``None`` at the end of the archive.

    Accepts the header of a regular file and nothing else.
    """
    block = stream.read_exact(_BLOCK, "a tar header")
    if block == bytes(_BLOCK):
        return None
    try:
        stored = _tar_number(block[148:156])
        size = _tar_number(block[124:136])
    except ValueError as exc:
        raise BundleError(f"not a tar header at byte {stream.position - _BLOCK}: {exc}") from exc
    if stored != sum(block[:148]) + 8 * 32 + sum(block[156:]):
        raise BundleError(f"not a tar header at byte {stream.position - _BLOCK}: the checksum is wrong")
    raw_name = block[:100].split(b"\0", 1)[0]
    if block[257:263] == b"ustar\0":
        prefix = block[345:500].split(b"\0", 1)[0]
        if prefix:
            raw_name = prefix + b"/" + raw_name
    try:
        name = raw_name.decode("ascii")
    except UnicodeDecodeError:
        raise BundleError(f"member name {raw_name!r} is not plain text") from None
    kind = block[156:157]
    if kind not in (b"0", b"\0"):
        raise BundleError(
            f"member {name!r} is a {_TAR_TYPES.get(kind, 'member of type ' + repr(kind))}; "
            "a bundle holds regular files only"
        )
    return name, size


def _read_bundle(
    tar_path: Path,
    staging: Optional[Path],
    max_bytes: int,
    max_member_bytes: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Read a bundle member by member, checking it; write the members under ``staging`` when given.

    Returns ``bundle.json``, ``archive/manifest.json`` and ``result.json``.
    """
    try:
        handle = open(tar_path, "rb")
    except OSError as exc:
        raise BundleError(f"cannot open {tar_path}: {exc.strerror}") from exc
    with handle:
        stream = _Stream(handle, tar_path)
        magic = handle.read(2)
        handle.seek(0)
        if magic == b"\x1f\x8b":
            raise BundleError(f"{tar_path.name} is gzip-compressed; a bundle is a plain tar file")
        first = _next_header(stream)
        if first is None or first[0] != BUNDLE_FILE:
            raise BundleError(f"the first member is {first[0] if first else 'missing'!r}, not {BUNDLE_FILE}")
        if first[1] > MAX_BUNDLE_JSON_BYTES:
            raise BundleError(f"{BUNDLE_FILE} has {first[1]} bytes, at most {MAX_BUNDLE_JSON_BYTES} are accepted")
        description_bytes = stream.read_exact(first[1], BUNDLE_FILE)
        stream.read_exact(-first[1] % _BLOCK, BUNDLE_FILE)
        bundle = validate_description(_load_json(description_bytes, BUNDLE_FILE), max_bytes, max_member_bytes)
        if staging is not None:
            _write_member(staging, BUNDLE_FILE, [description_bytes])

        kept: Dict[str, bytes] = {}
        keep = {f"{ARCHIVE_DIR}/{MANIFEST_FILE}", RESULT_FILE}
        for path, record in bundle["files"].items():
            header = _next_header(stream)
            if header is None:
                raise BundleError(f"member {path} is missing: the archive ends before it")
            name, size = header
            if name != path:
                if name in bundle["files"]:
                    raise BundleError(f"member {name} comes where {path} is expected (missing or out of order)")
                raise BundleError(f"member {name!r} is not listed in {BUNDLE_FILE} ({path} is expected here)")
            if size != record["size"]:
                raise BundleError(f"member {path} has {size} bytes, {BUNDLE_FILE} says {record['size']}")
            if path in keep and size > MAX_JSON_MEMBER_BYTES:
                raise BundleError(f"{path} has {size} bytes, at most {MAX_JSON_MEMBER_BYTES} are accepted")
            digest = hashlib.sha256()
            parts: List[bytes] = []

            def content() -> Iterator[bytes]:
                for block in stream.chunks(size, path):
                    digest.update(block)
                    if path in keep:
                        parts.append(block)
                    yield block

            if staging is not None:
                _write_member(staging, path, content())
            else:
                for _ in content():
                    pass
            stream.read_exact(-size % _BLOCK, path)
            if digest.hexdigest() != record["sha256"]:
                raise BundleError(f"member {path} does not have the SHA-256 that {BUNDLE_FILE} lists")
            if path in keep:
                kept[path] = b"".join(parts)

        extra = _next_header(stream)
        if extra is not None:
            raise BundleError(f"member {extra[0]!r} is not listed in {BUNDLE_FILE} (the archive should end here)")
        trailing = 0
        while True:
            block = stream.read_some(_COPY_BLOCK)
            if not block:
                break
            trailing += len(block)
            if block.strip(b"\0") or trailing > MAX_TRAILING_BYTES:
                raise BundleError(f"{tar_path.name} has data after the end of the archive")

    manifest = _load_json(kept[f"{ARCHIVE_DIR}/{MANIFEST_FILE}"], f"{ARCHIVE_DIR}/{MANIFEST_FILE}")
    result = _load_json(kept[RESULT_FILE], RESULT_FILE)
    check_contents(bundle, manifest, result)
    return bundle, manifest, result


def _write_member(staging: Path, path: str, blocks) -> None:
    """Write a member under ``staging``; ``path`` was checked by :func:`check_member_path`."""
    target = staging.joinpath(*path.split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o644)
    with os.fdopen(descriptor, "wb") as out:
        for block in blocks:
            out.write(block)


def check_bundle(
    tar_path: PathLike,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
    expected_glycam_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Check a bundle as :func:`open_bundle` does, without unpacking it; returns its ``bundle.json``."""
    bundle, _, _ = _read_bundle(Path(tar_path), None, max_bytes, max_member_bytes)
    _check_expected_name(bundle, expected_glycam_name)
    return bundle


def _check_expected_name(bundle: Dict[str, Any], expected: Optional[str]) -> None:
    if expected is not None and bundle["glycam_name"] != expected:
        raise BundleError(f"the bundle is {bundle['glycam_name']!r}, {expected!r} was expected")


def open_bundle(
    tar_path: PathLike,
    extract_to: PathLike,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
    expected_glycam_name: Optional[str] = None,
) -> Path:
    """Unpack a bundle into the new folder ``extract_to`` and return it; raises :class:`BundleError`.

    The bundle is untrusted.  It is unpacked only if all of this holds:

    * it is a plain tar file whose first member is ``bundle.json`` with the
      schema of this module;
    * the members that follow are exactly the files ``bundle.json`` lists,
      in that order: regular files with plain headers (no link, device,
      directory, sparse file or extended header), each with the listed size
      and SHA-256, and nothing after them;
    * every path is one of ``archive/<file of an entry>``, ``static/...``,
      ``result.json``, ``run.log``, relative, without ``..`` or hidden
      names (:func:`check_member_path`), and the files a publishable entry
      needs are all there;
    * no member is larger than ``max_member_bytes`` and all together not
      larger than ``max_bytes``;
    * ``archive/manifest.json`` and ``result.json`` are those of the GLYCAM
      name of ``bundle.json`` (and of ``expected_glycam_name`` when given),
      the manifest lists the archive files of the bundle with their sizes
      and SHA-256, the result was computed from exactly those files, with
      the levels of ``bundle.json``, and records exactly the static files of
      the bundle; ``extends`` agrees with the manifest.

    The files are written into a folder next to ``extract_to``, which is
    renamed to ``extract_to`` at the end: that folder exists only when the
    bundle passed.  ``extract_to`` must not exist (an empty folder is
    accepted).  File modes, owners and times of the tar are not used.
    """
    tar_path, extract_to = Path(tar_path), Path(extract_to)
    if extract_to.is_symlink() or (extract_to.exists() and (not extract_to.is_dir() or any(extract_to.iterdir()))):
        raise BundleError(f"{extract_to} exists; a bundle is unpacked into a new folder")
    parent = extract_to.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{extract_to.name}.partial-{os.getpid()}-{secrets.token_hex(4)}"
    os.mkdir(staging, 0o700)
    try:
        bundle, _, _ = _read_bundle(tar_path, staging, max_bytes, max_member_bytes)
        _check_expected_name(bundle, expected_glycam_name)
        os.chmod(staging, 0o755)
        if extract_to.exists():
            extract_to.rmdir()
        os.rename(staging, extract_to)
    except OSError as exc:
        raise BundleError(f"cannot unpack {tar_path} into {extract_to}: {exc}") from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return extract_to


def load_bundle_dir(bundle_dir: PathLike, verify_files: bool = True) -> Dict[str, Any]:
    """``bundle.json`` of an unpacked bundle, checked against the folder; raises :class:`BundleError`.

    For code that is handed the folder :func:`open_bundle` returned: the
    folder has to hold exactly the listed files, with their sizes (and, with
    ``verify_files``, SHA-256), and the manifest and the result have to agree
    with ``bundle.json`` as in :func:`open_bundle`.
    """
    bundle_dir = Path(bundle_dir)
    try:
        bundle = validate_description(
            _load_json((bundle_dir / BUNDLE_FILE).read_bytes(), BUNDLE_FILE), DEFAULT_MAX_BYTES, DEFAULT_MAX_MEMBER_BYTES)
        present = sorted(
            path.relative_to(bundle_dir).as_posix() for path in bundle_dir.rglob("*")
            if path.is_symlink() or not path.is_dir()
        )
        if present != sorted([BUNDLE_FILE, *bundle["files"]]):
            raise BundleError(f"{bundle_dir} does not hold exactly the files {BUNDLE_FILE} lists")
        for path, record in bundle["files"].items():
            target = bundle_dir.joinpath(*path.split("/"))
            if target.is_symlink() or not target.is_file():
                raise BundleError(f"{path} is not a regular file")
            if target.stat().st_size != record["size"] or (verify_files and _sha256_file(target)[1] != record["sha256"]):
                raise BundleError(f"{path} is not the file {BUNDLE_FILE} lists (size or SHA-256)")
        manifest = _load_json((bundle_dir / ARCHIVE_DIR / MANIFEST_FILE).read_bytes(), f"{ARCHIVE_DIR}/{MANIFEST_FILE}")
        result = _load_json((bundle_dir / RESULT_FILE).read_bytes(), RESULT_FILE)
    except OSError as exc:
        raise BundleError(f"cannot read {bundle_dir}: {exc}") from exc
    check_contents(bundle, manifest, result)
    return bundle
