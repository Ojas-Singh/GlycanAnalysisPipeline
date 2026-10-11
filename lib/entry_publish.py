"""Publish, replace or withdraw one entry of the database in S3 storage.

The analysis of an entry arrives as a bundle (``lib.entry_complete``); an
administrator approves it; this module makes it part of what is stored:

* **raw side** (archive bucket): the archive entry goes to
  ``raw/<GLYCAM name>/`` with its ``manifest.json`` last, and its record is
  merged into ``raw/index.json``;
* **static side** (database bucket): the glycan folder ``static/GSxxxxx/``
  and the root files that follow from it (``lib.static_delta``), and
  ``static/sync_manifest.json``, which the website synchronises from, last.

Only the stored root files and the folder of the one glycan are read and
written; no other glycan folder is touched or needed.

Order of a publication, and what a reader of the published prefix sees
----------------------------------------------------------------------
The website fetches ``sync_manifest.json`` and every file it lists whose
hash it does not have, and gives up, keeping its database, when one file is
not what the manifest says.  So the published prefix is valid whenever every
file the *stored* manifest lists is stored with the listed content.

1. *Raw entry.*  Nothing a reader of the static prefix fetches.
2. *New files* of the glycan folder are uploaded to their keys.  No stored
   manifest lists them.
3. *Staging.*  Files that replace a published one (the root files, and the
   files of a replaced glycan) are uploaded to
   ``<staging>/<static>/<SHA-256>``, outside the published prefix, and so is
   the new manifest.
4. *Intent.*  ``<state>/<static>/pending.json`` is written: what is about
   to be switched, with sizes and hashes.  Up to here the published prefix
   is exactly what it was.  From here on the publication is completed by
   whoever runs next (:func:`recover`): everything it needs is in the
   storage.
5. *Switch.*  The staged files are copied onto their keys inside the
   storage.  Two objects cannot be replaced at once, so between the first
   copy and step 6 the stored manifest describes files that have been
   replaced: a reader that is behind fails its verification, keeps its
   database and tries again at its next synchronisation; a reader that is
   up to date does nothing.  The window is the time of these copies, so
   they run ``workers`` at a time, in two groups (:func:`roll_forward`):
   the changed files of the glycan folder, which only a reader without the
   published folder of this glycan fetches, and then the root files, which
   every reader that is behind fetches.
6. *Manifest.*  When every copy of step 5 has succeeded, and not before,
   the new ``sync_manifest.json`` is copied into place, as one object.
   Readers see the new database.
7. *Afterwards.*  Files of the glycan that the new manifest no longer lists
   are deleted (all of them for a withdrawal), the record of the entry and
   the list of withdrawn IDs are written, the staged objects are removed
   and, last, ``pending.json``.

Every step first looks at what is stored and skips an object that is there
with the same size and SHA-256, so the same command can be run again after
any interruption; a journal in the work folder keeps what was derived, so a
retry publishes the same bytes.

Two publishers
--------------
One process publishes at a time (a lock in the process): that is the
intended use, and it is all the serialisation a single service needs.
Against a second process (a manual run) there are three guards in the
storage: a lock object ``<state>/<static>/lock.json`` with an owner and an
expiry, taken before anything is read; ``pending.json``, claimed before
anything is switched; and a comparison of the stored manifest with the one
the change was derived from, made after that claim.

Where the storage honours conditional writes (``If-None-Match``; found out
with a probe, :func:`conditional_writes`) a free lock and ``pending.json``
are created by one publisher only.  Then no two publishers switch at once
and no change is lost: the one that comes second is refused at the lock, or
stops before its switch and, run again, derives its change from the new
database.  Where the storage does not honour them, the objects are written
and read back: this stops a run that starts while another is under way, and
cannot stop two that take the same step within the same moment.  The
takeover of an expired lock is never atomic.  ``scripts/static_s3.py
publish`` and ``scripts/archive_s3.py upload`` do not take the lock: they
must not run while an entry is published.

Withdrawing publishes a manifest without the entry first and deletes
``static/GSxxxxx/`` afterwards; the raw entry stays, and the ID is recorded
as withdrawn so that it is never given to another glycan.
"""
from __future__ import annotations

import contextlib
import datetime
import fnmatch
import hashlib
import json
import os
import shutil
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from lib import archive, archive_transfer, entry_complete, glybake, id_registry, static_delta, static_publish
from lib.storage import S3Settings, create_s3_backend, file_sha256, parallel_map

PUBLISH, REPLACE, WITHDRAW = "publish", "replace", "withdraw"
JOURNAL_SCHEMA = "glycoshape-publish-journal"
PENDING_SCHEMA = "glycoshape-publish-pending"
LOCK_SCHEMA = "glycoshape-publish-lock"
RECORD_SCHEMA = "glycoshape-publish-record"
WITHDRAWN_SCHEMA = "glycoshape-withdrawn-ids"
SCHEMA_VERSION = 1
MANIFEST = static_publish.MANIFEST_NAME
DEFAULT_LOCK_TTL = 3600

Log = Callable[[str], None]
Checkpoint = Callable[[str], None]
_PROCESS_LOCKS: Dict[Tuple[int, str], threading.Lock] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()
_COUNT_GUARD = threading.Lock()


class PublishEntryError(Exception):
    """The publication stopped; the message says what a reader sees and what to do."""


class Refused(PublishEntryError):
    """A check failed before anything was written.  ``report`` holds every check."""

    def __init__(self, message: str, report: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.report = report or {}


class LockHeld(PublishEntryError):
    """Another publisher holds the lock."""


# ---------------------------------------------------------------------------
# Where, and with which rules
# ---------------------------------------------------------------------------
@dataclass
class Policy:
    """What a bundle has to be to be published, and what a publication may do.

    ``levels``: the clustering levels of the database (default: the
    configured ones).  ``pipeline_versions`` and ``pipeline_commits``:
    patterns (``fnmatch``) of which one has to match ``pipeline.version``
    and ``pipeline.commit`` of the bundle; empty accepts any.
    ``converter_versions``: accepted ``pipeline.converter_version`` (default:
    the converter of this checkout).  ``require_clean``: refuse a bundle
    made from a checkout with uncommitted changes.  ``allow_incomplete``:
    publish an entry that lacks an identity (no WURCS and no accession) or
    its SNFG image.  ``overwrite_raw``: replace a stored raw entry by one
    that does not extend it.  ``keep_previous_raw``: what of a replaced raw
    entry is copied to the history prefix first: ``auto`` (its manifest when
    the new entry extends it, every file otherwise), ``manifest`` or
    ``all``.  The manifest is always kept: it is the record of what was
    replaced, and what an interrupted replacement is resumed by.
    """

    levels: Optional[Tuple[int, ...]] = None
    pipeline_versions: Tuple[str, ...] = ()
    pipeline_commits: Tuple[str, ...] = ()
    converter_versions: Optional[Tuple[str, ...]] = None
    require_clean: bool = False
    allow_incomplete: bool = False
    overwrite_raw: bool = False
    keep_previous_raw: str = "auto"


@dataclass
class Target:
    """The buckets and prefixes of a publication.

    ``static`` and ``raw`` are storage backends bound to the database bucket
    and to the archive bucket (the same backend twice when one bucket holds
    both).  Besides the two published prefixes there are three that no
    reader looks at: ``staging_prefix`` (files waiting to be switched) and
    ``state_prefix`` (lock, pending publication, records, withdrawn IDs) in
    the database bucket, ``history_prefix`` (what a replaced raw entry was)
    in the archive bucket.

    Each of the three is followed by the prefix it belongs to, so that two
    databases or two archives in one bucket (a trial next to the real one)
    share no key: ``<staging>/<static prefix>/`` and
    ``<state>/<static prefix>/`` for the static side,
    ``<history>/<raw prefix>/`` for the raw side.  The prefixes of two
    environments must not lie inside each other.
    """

    static: Any
    raw: Any
    static_prefix: str = "static"
    raw_prefix: str = "raw"
    staging_prefix: str = static_publish.DEFAULT_STAGING_PREFIX
    state_prefix: str = "publish-state"
    history_prefix: str = "raw-history"
    workers: int = 8
    _conditional: Optional[bool] = field(default=None, repr=False)
    _conditional_reported: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("static_prefix", "raw_prefix", "staging_prefix", "state_prefix", "history_prefix"):
            value = str(getattr(self, name)).strip("/")
            if not value:
                raise PublishEntryError(f"{name} must not be empty")
            setattr(self, name, value)
        for other in (self.staging_prefix, self.state_prefix):
            if (other + "/").startswith(self.static_prefix + "/") or (self.static_prefix + "/").startswith(other + "/"):
                raise PublishEntryError(f"{other}/ must be outside the published prefix {self.static_prefix}/")
        if (self.history_prefix + "/").startswith(self.raw_prefix + "/") or (self.raw_prefix + "/").startswith(self.history_prefix + "/"):
            raise PublishEntryError(f"{self.history_prefix}/ must be outside the raw prefix {self.raw_prefix}/")

    @classmethod
    def from_settings(cls, settings: Optional[S3Settings] = None, workers: int = 8, **prefixes: str) -> "Target":
        """The target the ``GLYCOSHAPE_S3_*`` settings name."""
        settings = settings or S3Settings.from_env()
        connections = workers * settings.transfer_concurrency + 8
        return cls(
            static=create_s3_backend(settings, bucket=settings.static_bucket, max_connections=connections),
            raw=create_s3_backend(settings, bucket=settings.raw_bucket, max_connections=connections),
            static_prefix=prefixes.get("static_prefix") or settings.static_prefix,
            raw_prefix=prefixes.get("raw_prefix") or settings.raw_prefix,
            staging_prefix=prefixes.get("staging_prefix") or settings.staging_prefix,
            workers=workers,
        )

    def key(self, path: str) -> str:
        return f"{self.static_prefix}/{path}"

    def staged(self, sha256: str) -> str:
        return f"{self.staging_prefix}/{self.static_prefix}/{sha256}"

    def history(self, glycam_name: str) -> str:
        """Key prefix of what the raw entry of a glycan was before it was replaced, by the SHA-256 of its manifest.

        ``raw-history/raw/<GLYCAM name>/`` with the defaults: under the raw
        prefix like the state of the static side under the static prefix,
        without an exception for the default one, so that a listing of
        ``raw-history/<raw prefix>/`` is the history of that archive only.
        """
        return f"{self.history_prefix}/{self.raw_prefix}/{glycam_name}/"

    @property
    def lock_key(self) -> str:
        return f"{self.state_prefix}/{self.static_prefix}/lock.json"

    @property
    def pending_key(self) -> str:
        return f"{self.state_prefix}/{self.static_prefix}/pending.json"

    @property
    def withdrawn_key(self) -> str:
        return f"{self.state_prefix}/{self.static_prefix}/withdrawn.json"

    def record_key(self, glycan_id: str) -> str:
        return f"{self.state_prefix}/{self.static_prefix}/entries/{glycan_id}.json"


class Store:
    """The requests of one run to one bucket, counted; refuses to write in a dry run."""

    def __init__(self, backend: Any, stats: Dict[str, Any], dry_run: bool = False) -> None:
        self.backend, self.stats, self.dry_run = backend, stats, dry_run

    def _count(self, operation: str, **amounts: int) -> None:
        with _COUNT_GUARD:                                      # requests are made from several threads
            self.stats["requests"][operation] = self.stats["requests"].get(operation, 0) + 1
            for name, amount in amounts.items():
                self.stats[name] = self.stats.get(name, 0) + amount

    def _writing(self, key: str) -> None:
        if self.dry_run:
            raise AssertionError(f"a dry run must not write {key}")

    def head(self, key: str) -> Optional[Dict[str, Any]]:
        self._count("head")
        return self.backend.head(key)

    def holds(self, key: str, size: int, sha256: str) -> bool:
        """Whether the object is stored with this size and this SHA-256 in its metadata."""
        head = self.head(key)
        return head is not None and head["size"] == size and head["sha256"] == sha256

    def list(self, prefix: str) -> Dict[str, int]:
        self._count("list")
        return {item["key"]: item["size"] for item in self.backend.list_objects(prefix)}

    def read(self, key: str) -> bytes:
        payload = self.backend.read_binary(key)
        self._count("get", downloaded_bytes=len(payload))
        return payload

    def read_optional(self, key: str) -> Optional[bytes]:
        try:
            return self.read(key)
        except FileNotFoundError:
            self._count("get")
            return None

    def download(self, key: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.backend.download_file(key, path)
        self._count("get", downloaded_bytes=path.stat().st_size)

    def write(self, key: str, payload: bytes) -> None:
        self._writing(key)
        self.backend.write_binary(key, payload)
        self._count("put", uploaded_bytes=len(payload))

    def create(self, key: str, payload: bytes) -> Optional[bool]:
        self._writing(key)
        created = self.backend.write_binary_if_absent(key, payload)
        self._count("put", uploaded_bytes=len(payload) if created else 0)
        return created

    def upload(self, path: Path, key: str, size: int, sha256: str) -> None:
        self._writing(key)
        if file_sha256(path) != (size, sha256):
            raise PublishEntryError(f"{path} is not the file that was planned (size or SHA-256 changed)")
        self.backend.upload_file(path, key, sha256=sha256)
        self._count("put", uploaded_bytes=size)

    def copy(self, source: str, key: str, size: int, sha256: str) -> None:
        self._writing(key)
        self.backend.copy(source, key, sha256=sha256, size=size)
        self._count("copy", copied_bytes=size)

    def delete(self, key: str) -> None:
        self._writing(key)
        self.backend.unlink(key)
        self._count("delete")


def new_stats() -> Dict[str, Any]:
    return {"requests": {}, "uploaded_bytes": 0, "downloaded_bytes": 0, "copied_bytes": 0}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _utc_after(seconds: float) -> str:
    moment = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=seconds)
    return moment.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _is_past(stamp: Any) -> bool:
    try:
        moment = datetime.datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return True
    return moment <= datetime.datetime.now(datetime.timezone.utc)


def new_owner() -> str:
    """A name for this run: host, process and a random part."""
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


def _json(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(_json(payload))
    os.replace(temporary, path)


def _load_json(payload: Optional[bytes], schema: str, what: str) -> Optional[Dict[str, Any]]:
    if payload is None:
        return None
    try:
        document = json.loads(payload)
    except ValueError as exc:
        raise PublishEntryError(f"the stored {what} is not JSON: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema") != schema or document.get("schema_version") != SCHEMA_VERSION:
        raise PublishEntryError(f"the stored {what} is not a {schema} document of version {SCHEMA_VERSION}")
    return document


def _noop(_: str) -> None:
    return None


@contextlib.contextmanager
def _in_process(target: Target) -> Iterator[None]:
    """One publication at a time in this process for one database."""
    name = (id(getattr(target.static, "_client", target.static)), f"{getattr(target.static, 'bucket', '')}/{target.static_prefix}")
    with _PROCESS_LOCKS_GUARD:
        lock = _PROCESS_LOCKS.setdefault(name, threading.Lock())
    with lock:
        yield


# ---------------------------------------------------------------------------
# Lock and pending publication
# ---------------------------------------------------------------------------
def conditional_writes(target: Target, store: Store) -> bool:
    """Whether the storage creates an object only when none is there (``If-None-Match: *``).

    Asked once per target with a probe object: a conditional write onto an
    existing key has to be refused and leave the object as it is.  A storage
    that does not know the header, or accepts it and overwrites, counts as
    not honouring it.
    """
    if target._conditional is None:
        prefix = f"{target.state_prefix}/{target.static_prefix}/probe-"
        key = prefix + uuid.uuid4().hex
        try:
            store.write(key, b"1")
            refused = store.create(key, b"2") is False
            honoured = refused and store.read(key) == b"1"
        finally:
            for leftover in sorted(set(store.list(prefix)) | {key}):   # also the probe of a run that was killed
                store.delete(leftover)
        target._conditional = honoured                          # remembered once the probe is gone
    return bool(target._conditional)


def _claim(target: Target, store: Store, key: str, document: Dict[str, Any], replace: bool) -> bool:
    """Write a state object that only one publisher may hold; False when another one holds it.

    ``replace`` overwrites what is there (an expired lock, an object of this
    run).  Otherwise the object is created conditionally where the storage
    honours that; where it does not, it is written and read back.
    """
    payload = _json(document)
    if not replace and conditional_writes(target, store):
        return bool(store.create(key, payload))
    if not replace and store.head(key) is not None:
        return False
    store.write(key, payload)
    back = store.read_optional(key)
    return back is not None and json.loads(back).get("owner") == document["owner"]


def read_lock(target: Target, store: Store) -> Optional[Dict[str, Any]]:
    return _load_json(store.read_optional(target.lock_key), LOCK_SCHEMA, "lock")


def _process_alive(pid: Any) -> bool:
    """Whether a process of this machine exists; True when that cannot be told."""
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (OSError, TypeError, ValueError):
        return True
    return True


def acquire_lock(target: Target, store: Store, owner: str, action: str, glycan_id: str, ttl: float = DEFAULT_LOCK_TTL,
                 break_lock: bool = False, log: Log = print) -> Dict[str, Any]:
    """Take the lock of the database for ``ttl`` seconds; raises :class:`LockHeld`.

    A lock of another run stands until it expires, with two exceptions: the
    run was a process of this machine that no longer exists (it was killed:
    its lock is taken over), and ``break_lock`` (an operator who knows that
    the other run is gone).
    """
    existing = read_lock(target, store)
    if existing is not None and existing.get("owner") != owner and not _is_past(existing.get("expires")):
        gone = existing.get("host") == socket.gethostname() and not _process_alive(existing.get("pid"))
        if not (break_lock or gone):
            raise LockHeld(
                f"the database is locked by {existing.get('owner')} for {existing.get('action')} "
                f"{existing.get('glycoshape_id')} until {existing.get('expires')}")
        log(f"taking over the lock of {existing.get('owner')}: "
            + ("asked to break it" if break_lock else "its process on this machine is gone"))
    document = {
        "schema": LOCK_SCHEMA, "schema_version": SCHEMA_VERSION, "owner": owner,
        "host": socket.gethostname(), "pid": os.getpid(), "action": action, "glycoshape_id": glycan_id,
        "acquired": _utc_now(), "expires": _utc_after(ttl),
    }
    if not _claim(target, store, target.lock_key, document, replace=existing is not None):
        holder = read_lock(target, store) or {}
        raise LockHeld(f"the database was locked by {holder.get('owner', 'another publisher')} at the same moment")
    if target._conditional is False and not target._conditional_reported:
        target._conditional_reported = True
        log("the storage does not honour conditional writes: the lock is written and read back, which does not "
            "separate two publishers that start at the same moment")
    return document


def hold_lock(target: Target, store: Store, lock: Dict[str, Any], ttl: float = DEFAULT_LOCK_TTL) -> None:
    """Confirm that the lock is still this run's and move its expiry; raises when it was taken over."""
    current = read_lock(target, store)
    if current is None or current.get("owner") != lock["owner"]:
        raise PublishEntryError(
            f"the lock of the database was taken over by {(current or {}).get('owner', 'nobody')}: "
            "this run stops before it changes anything else")
    lock["expires"] = _utc_after(ttl)
    store.write(target.lock_key, _json(lock))


def release_lock(target: Target, store: Store, lock: Optional[Dict[str, Any]]) -> None:
    if lock is None:
        return
    current = read_lock(target, store)
    if current is not None and current.get("owner") == lock["owner"]:
        store.delete(target.lock_key)


def read_pending(target: Target, store: Store) -> Optional[Dict[str, Any]]:
    return _load_json(store.read_optional(target.pending_key), PENDING_SCHEMA, "pending publication")


def read_withdrawn(target: Target, store: Store) -> Dict[str, Any]:
    document = _load_json(store.read_optional(target.withdrawn_key), WITHDRAWN_SCHEMA, "list of withdrawn IDs")
    return document["withdrawn"] if document else {}


# ---------------------------------------------------------------------------
# Completing a pending publication (steps 5 to 7)
# ---------------------------------------------------------------------------
def roll_forward(target: Target, store: Store, pending: Dict[str, Any], log: Log = print, checkpoint: Checkpoint = _noop) -> Dict[str, Any]:
    """Switch, write the manifest and clean up, as ``pending.json`` describes; safe to run again.

    Everything it needs is in the storage: the staged objects and the
    pending document.  An object that is already at its key with the size
    and SHA-256 it should have is left alone.

    The window in which the stored manifest describes replaced files is the
    time of the copies, so they run ``target.workers`` at a time, and what
    has to be asked of the storage is asked before the first of them.  The
    order that is kept:

    * *Before the first copy*: which files are not at their keys yet, and
      that each of them is staged (so is the manifest).  A run that could
      not finish the switch from the storage does not begin or widen it.
    * *The changed files of the glycan folder*, in any order.  A reader
      fetches them only when it does not have the published folder of this
      glycan; one that is behind for another reason is not affected yet.
    * *The root files*, in any order, when every copy of the folder has
      returned.  Every reader that is behind fetches every root file, so
      once the first of these copies is done all of them fail: this group
      comes last and is the short one, however long the folder takes.
    * *The manifest*, alone, when every copy above has succeeded and none is
      in flight: the stored manifest never lists a file that is not at its
      key.  The files it lists and no stored manifest listed were uploaded
      before ``pending.json`` was written; the files it no longer lists are
      deleted after it.

    A copy that fails stops the switch: the copies in flight are waited for,
    those not started are not made, the manifest is not written and
    ``pending.json`` stays.  Whatever subset of the files was switched then,
    it is the state of step 5, which the next call completes by looking at
    every key again.
    """
    def waits(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """The item when it still has to be copied onto its key (its staged copy has to be there), or None."""
        if store.holds(item["key"], item["size"], item["sha256"]):
            return None
        staged = target.staged(item["sha256"])
        if not store.holds(staged, item["size"], item["sha256"]):
            raise PublishEntryError(
                f"the staged copy of {item['key']} is missing or damaged ({staged}): the pending "
                f"{pending['action']} of {pending['glycoshape_id']} cannot be completed from the storage")
        return item

    def place(item: Dict[str, Any]) -> str:
        store.copy(target.staged(item["sha256"]), item["key"], item["size"], item["sha256"])
        return item["key"]

    def place_together(items: List[Dict[str, Any]]) -> int:
        """Copy the items onto their keys, ``workers`` at a time; when this returns or raises none is in flight."""
        with contextlib.closing(parallel_map(place, items, target.workers)) as placed:
            for key in placed:
                checkpoint(f"switched:{key}")
        return len(items)

    def of_the_root(item: Dict[str, Any]) -> bool:
        """A file at the root of the database (every other switched file is ``<glycan>/...``)."""
        return "/" not in item["key"][len(target.static_prefix) + 1:]

    try:
        with contextlib.closing(parallel_map(waits, pending["switch"] + [pending["manifest"]], target.workers)) as looked:
            waiting = {item["key"] for item in looked if item is not None}
        of_folder = [item for item in pending["switch"] if item["key"] in waiting and not of_the_root(item)]
        of_root = [item for item in pending["switch"] if item["key"] in waiting and of_the_root(item)]
        manifest_written = pending["manifest"]["key"] in waiting
        window_started = time.perf_counter()
        switched = place_together(of_folder)
        root_started = time.perf_counter()
        switched += place_together(of_root)
        if manifest_written:
            place(pending["manifest"])
        window = round(time.perf_counter() - window_started, 3)
        root_window = round(time.perf_counter() - root_started, 3)
        checkpoint("manifest")
    except PublishEntryError:
        raise
    except (Exception, KeyboardInterrupt) as exc:
        raise PublishEntryError(
            f"stopped while the changed files were switched ({type(exc).__name__}: {exc}); the published manifest "
            "may describe files that have been replaced, so a reader that is behind fails and retries: "
            "run the command again (or publish_entry.py recover) to finish") from exc
    log(f"switched {switched} file(s) ({len(of_folder)} of the glycan folder, then {len(of_root)} root file(s); "
        f"{target.workers} at a time) and {'wrote' if manifest_written else 'found'} "
        f"{pending['manifest']['key']} (global_hash {pending['target_global_hash'][:16]}...) in {window} s, "
        f"{root_window} s of them for the root files and the manifest")

    # 7. Afterwards.  The published prefix is the new database from here on.
    try:
        doomed = list(pending.get("delete") or [])
        if pending.get("delete_prefix"):
            doomed += sorted(store.list(pending["delete_prefix"]))
        for _ in parallel_map(store.delete, doomed, target.workers):
            pass
        checkpoint("deleted")
        if pending.get("withdrawn") is not None:
            withdrawn = read_withdrawn(target, store)
            glycan_id = pending["glycoshape_id"]
            if pending["withdrawn"]:
                withdrawn[glycan_id] = pending["withdrawn"]
            else:
                withdrawn.pop(glycan_id, None)
            store.write(target.withdrawn_key, _json({
                "schema": WITHDRAWN_SCHEMA, "schema_version": SCHEMA_VERSION, "updated": _utc_now(),
                "withdrawn": dict(sorted(withdrawn.items()))}))
        if pending.get("record") is not None:
            store.write(target.record_key(pending["glycoshape_id"]), _json(pending["record"]))
        checkpoint("recorded")
        # The staged objects go before the document that names them, so none is left without a trace.
        staged = {target.staged(item["sha256"]) for item in pending["switch"] + [pending["manifest"]]}
        for _ in parallel_map(store.delete, sorted(staged), target.workers):
            pass
        checkpoint("cleaned")
        store.delete(target.pending_key)
        checkpoint("pending-cleared")
    except PublishEntryError:
        raise
    except (Exception, KeyboardInterrupt) as exc:
        raise PublishEntryError(
            f"the new database is published, and clearing up stopped ({type(exc).__name__}: {exc}): "
            "run the command again (or publish_entry.py recover) to finish") from exc
    return {"switched": switched, "manifest_written": manifest_written, "deleted": len(doomed),
            "switch_seconds": window, "switch_root_seconds": root_window}


def recover(target: Target, log: Log = print, checkpoint: Checkpoint = _noop, owner: Optional[str] = None,
            lock_ttl: float = DEFAULT_LOCK_TTL, stats: Optional[Dict[str, Any]] = None,
            break_lock: bool = False) -> Optional[Dict[str, Any]]:
    """Complete a publication that was interrupted after its ``pending.json`` was written; None when there is none."""
    stats = stats if stats is not None else new_stats()
    store = Store(target.static, stats)
    owner = owner or new_owner()
    with _in_process(target):
        pending = read_pending(target, store)
        if pending is None:
            return None
        lock = acquire_lock(target, store, owner, "recover", pending["glycoshape_id"], lock_ttl, break_lock, log)
        try:
            log(f"completing the interrupted {pending['action']} of {pending['glycoshape_id']} (by {pending['owner']})")
            done = roll_forward(target, store, pending, log, checkpoint)
        finally:
            release_lock(target, store, lock)
    return {"action": pending["action"], "glycoshape_id": pending["glycoshape_id"], "glycam_name": pending.get("glycam_name"),
            "global_hash": pending["target_global_hash"], **done, "transfers": stats}


# ---------------------------------------------------------------------------
# What is stored
# ---------------------------------------------------------------------------
@dataclass
class Database:
    """The stored database as far as a change of one glycan needs it."""

    manifest: Dict[str, Any]
    manifest_sha256: str
    stored_dir: Path                    # local copies: the manifest and the root files a change is derived from
    inventory: Dict[str, Any]
    withdrawn: Dict[str, Any]

    @property
    def global_hash(self) -> str:
        return self.manifest["global_hash"]


def read_database(target: Target, store: Store, work: Path, cache_dir: Optional[Path] = None) -> Database:
    """Fetch the stored manifest and the root files a change is derived from; check them against the manifest."""
    payload = store.read_optional(target.key(MANIFEST))
    if payload is None:
        raise Refused(
            f"no database is published under {target.static_prefix}/ (no {MANIFEST}): publish one with "
            "scripts/static_s3.py first")
    try:
        manifest = static_publish.parse_manifest(payload, f"the stored {target.key(MANIFEST)}")
        needed = static_delta.needed_root_files(manifest)
    except (static_publish.PublishError, static_delta.DeltaError) as exc:
        raise Refused(str(exc)) from exc
    stored_dir = work / "base" / manifest["global_hash"][:16]
    stored_dir.mkdir(parents=True, exist_ok=True)
    (stored_dir / MANIFEST).write_bytes(payload)
    for name in needed:
        recorded = (manifest["root_files"][name]["size"], manifest["root_files"][name]["sha256"].lower())
        local = stored_dir / name
        if local.is_file() and file_sha256(local) == recorded:
            continue                                            # kept from an earlier attempt
        cached = None if cache_dir is None else Path(cache_dir) / recorded[1]
        if cached is not None and cached.is_file() and file_sha256(cached) == recorded:
            shutil.copyfile(cached, local)
            continue
        try:
            store.download(target.key(name), local)
        except FileNotFoundError:
            raise Refused(f"{target.key(name)} is listed by the stored manifest and is not stored") from None
        if file_sha256(local) != recorded:
            local.unlink()
            raise Refused(
                f"the stored {target.key(name)} is not the file the stored manifest describes: a publication is "
                "in progress or was interrupted (publish_entry.py status)")
    try:
        inventory = json.loads((stored_dir / static_delta.INVENTORY).read_bytes())
    except ValueError as exc:
        raise Refused(f"the stored {static_delta.INVENTORY} is not JSON: {exc}") from exc
    if not isinstance(inventory, dict) or sorted(inventory) != sorted(manifest["glycans"]):
        raise Refused(f"the stored {static_delta.INVENTORY} and the stored manifest do not list the same glycans")
    return Database(manifest, hashlib.sha256(payload).hexdigest(), stored_dir, inventory, read_withdrawn(target, store))


def _published_names(database: Database, raw_index: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """GlycoShape ID -> full GLYCAM name of every published glycan."""
    aliases = id_registry.names_from_archive_index(raw_index or {})
    names = {}
    for glycan_id, entry in database.inventory.items():
        name = id_registry.full_glycam_name(entry) or ""
        names[glycan_id] = aliases.get(name, name)
    return names


# ---------------------------------------------------------------------------
# Raw side
# ---------------------------------------------------------------------------
def index_record(manifest: Dict[str, Any], manifest_size: int, manifest_sha256: str) -> Dict[str, Any]:
    """The record of an entry in ``raw/index.json``, as ``lib.archive.build_index`` makes it."""
    builder = getattr(archive, "index_record", None)
    if callable(builder):                                       # the converter's own, where it has one
        return builder(manifest, manifest_size, manifest_sha256)
    files = {archive.MANIFEST_FILE: {"size": manifest_size, "sha256": manifest_sha256}}
    files.update(manifest["files"])
    source = manifest.get("source") or {}
    record = {
        "glycam_name": manifest["glycam_name"],
        "folder": manifest["folder"],
        "object_key_prefix": manifest["object_key_prefix"],
        "source_folder": source.get("folder"),
        "atoms": manifest["atoms"],
        "frames": manifest["frames"],
        "source_size": source.get("size"),
        "source_sha256": source.get("sha256"),
        "archive_size": sum(item["size"] for item in files.values()),
        "files": files,
    }
    repaired = archive.repair_summary(manifest)
    if repaired is not None:
        record["repaired"] = repaired
    return record


def merged_index(stored: Optional[Dict[str, Any]], record: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """``raw/index.json`` with the record of one entry inserted or replaced; and whether that changes it."""
    if stored is None:
        stored = {"schema": archive.INDEX_SCHEMA, "schema_version": archive.SCHEMA_VERSION,
                  "converter_version": archive.CONVERTER_VERSION, "records": []}
    previous = next((item for item in stored["records"] if item["glycam_name"] == record["glycam_name"]), None)
    if previous == record:
        return stored, False
    records = [item for item in stored["records"] if item["glycam_name"] != record["glycam_name"]] + [record]
    records.sort(key=lambda item: item["glycam_name"])
    index = dict(stored)
    index.update(
        generated=_utc_now(),
        entries=len(records),
        totals={
            "frames": sum(item.get("frames") or 0 for item in records),
            "source_bytes": sum(item.get("source_size") or 0 for item in records),
            "archive_bytes": sum(item.get("archive_size") or 0 for item in records),
        },
        repaired=[item["glycam_name"] for item in records if "repaired" in item],
        records=records,
    )
    return index, True


def plan_raw(target: Target, store: Store, bundle: entry_complete.Bundle, policy: Policy) -> Tuple[Dict[str, Any], List[str]]:
    """What the raw side has to do for the archive entry of a bundle, and why it may not; only reads."""
    name = bundle.glycam_name
    refused: List[str] = []
    try:
        key_prefix = archive_transfer.entry_key_prefix(target.raw_prefix, name)
    except archive.ArchiveError as exc:
        return {}, [str(exc)]
    files = [(item, bundle.archive_dir / item, bundle.files[f"archive/{item}"]["size"], bundle.files[f"archive/{item}"]["sha256"])
             for item in sorted(bundle.manifest["files"])]
    manifest_file = (archive.MANIFEST_FILE, bundle.archive_dir / archive.MANIFEST_FILE,
                     bundle.files["archive/manifest.json"]["size"], bundle.files["archive/manifest.json"]["sha256"])
    listing = store.list(key_prefix)
    stored_ok = {item: key_prefix + item in listing and listing[key_prefix + item] == size
                 and store.holds(key_prefix + item, size, sha256)
                 for item, _, size, sha256 in files + [manifest_file]}
    stored_manifest = store.read_optional(key_prefix + archive.MANIFEST_FILE) if key_prefix + archive.MANIFEST_FILE in listing else None
    stored_sha256 = None if stored_manifest is None else hashlib.sha256(stored_manifest).hexdigest()
    extends = bundle.extends
    history_prefix = target.history(name)
    if not listing:
        relation = "new"
    elif all(stored_ok.values()):
        relation = "same"
    elif stored_manifest is None:
        relation = "incomplete"                                 # an interrupted upload or replacement: continued
    elif extends is not None and extends["manifest_sha256"] == stored_sha256:
        relation = "extension"
    elif all(stored_ok[item] for item, *_ in files):
        relation = "same-data"                                  # the same files under another manifest
    else:
        relation = "overwrite"
    if extends is not None and relation not in ("same", "extension"):
        resumed = stored_manifest is None and store.head(f"{history_prefix}{extends['manifest_sha256']}/{archive.MANIFEST_FILE}") is not None
        if not resumed:
            refused.append(
                f"the bundle extends the raw entry whose manifest has SHA-256 {extends['manifest_sha256'][:16]}..., "
                f"and the storage holds {'none' if stored_sha256 is None else stored_sha256[:16] + '...'} for this glycan")
    if relation == "overwrite" and not policy.overwrite_raw:
        refused.append(
            "a raw entry with other files is stored for this glycan and the bundle does not extend it: "
            "replacing it needs --overwrite-raw")
    if policy.keep_previous_raw not in ("auto", "manifest", "all"):
        refused.append(f"keep_previous_raw is {policy.keep_previous_raw!r}: auto, manifest or all")
    keep = policy.keep_previous_raw
    if keep == "auto":
        keep = "all" if relation == "overwrite" else "manifest"
    history: List[Tuple[str, str]] = []
    if stored_manifest is not None and relation in ("extension", "same-data", "overwrite"):
        kept = sorted(listing) if keep == "all" else [key_prefix + archive.MANIFEST_FILE]
        history = [(key, f"{history_prefix}{stored_sha256}/{key[len(key_prefix):]}") for key in kept]

    index_key = f"{target.raw_prefix}/{archive.INDEX_FILE}"
    stored_index_bytes = store.read_optional(index_key)
    stored_index: Optional[Dict[str, Any]] = None
    if stored_index_bytes is not None:
        try:
            stored_index = archive_transfer.parse_index(stored_index_bytes, f"the stored {index_key}")
        except archive_transfer.TransferError as exc:
            refused.append(str(exc))
    else:
        others = [prefix for prefix in target.raw.list_common_prefixes(target.raw_prefix)
                  if prefix + "/" != key_prefix]
        if others:
            refused.append(
                f"{index_key} does not exist while {len(others)} other entries are stored: finish "
                "scripts/archive_s3.py upload first, so that the index describes the archive")
    record = index_record(bundle.manifest, manifest_file[2], manifest_file[3])
    index, index_changes = merged_index(stored_index, record)
    twice = [item["glycam_name"] for item in index["records"]
             if item["glycam_name"] != name and item.get("source_folder") == record.get("source_folder")]
    if twice:
        refused.append(f"the entry {twice[0][:60]} of the stored index comes from the same source folder "
                       f"{record.get('source_folder')!r}")
    wanted = {key_prefix + item for item, *_ in files + [manifest_file]}
    plan = {
        "glycam_name": name,
        "key_prefix": key_prefix,
        "relation": relation,
        "files": files + [manifest_file],
        "stored": stored_ok,
        "stored_manifest_sha256": stored_sha256,
        "stale": sorted(set(listing) - wanted),
        "history": history,
        "index_key": index_key,
        "index": index,
        "index_changes": index_changes,
        "upload_bytes": sum(size for item, _, size, _ in files + [manifest_file] if not stored_ok[item]),
    }
    return plan, refused


def describe_raw(plan: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "key_prefix": plan["key_prefix"],
        "relation": plan["relation"],
        "upload": [item for item, *_ in plan["files"] if not plan["stored"][item]],
        "unchanged": [item for item, *_ in plan["files"] if plan["stored"][item]],
        "upload_bytes": plan["upload_bytes"],
        "delete": plan["stale"],
        "history": [destination for _, destination in plan["history"]],
        "previous_manifest_sha256": plan["stored_manifest_sha256"],
        "index": "unchanged" if not plan["index_changes"] else f"{plan['index']['entries']} entries",
    }


def publish_raw(target: Target, store: Store, plan: Dict[str, Any], log: Log = print, checkpoint: Checkpoint = _noop) -> None:
    """Store the archive entry: history first, its files, its manifest last, then the index."""
    key_prefix, manifest_key = plan["key_prefix"], plan["key_prefix"] + archive.MANIFEST_FILE
    changes = [item for item, *_ in plan["files"] if not plan["stored"][item]]
    try:
        if changes or plan["stale"]:
            for source, destination in plan["history"]:
                if store.head(destination) is None:
                    head = store.head(source)
                    if head is not None:
                        store.copy(source, destination, head["size"], head["sha256"])
            checkpoint("raw:history")
            # A stored manifest never describes files that are being replaced: it goes first and comes back last.
            if plan["stored_manifest_sha256"] is not None:
                store.delete(manifest_key)
                checkpoint("raw:manifest-removed")
            for item, path, size, sha256 in plan["files"][:-1]:
                if not plan["stored"][item]:
                    store.upload(path, key_prefix + item, size, sha256)
                    checkpoint(f"raw:uploaded:{item}")
            for key in plan["stale"]:
                store.delete(key)
            item, path, size, sha256 = plan["files"][-1]
            store.upload(path, manifest_key, size, sha256)
            checkpoint("raw:manifest")
            log(f"raw entry stored under {key_prefix} ({plan['relation']}; {len(changes)} file(s), "
                f"{plan['upload_bytes'] / 1e6:.1f} MB)")
        else:
            log(f"raw entry under {key_prefix} is already stored")
        if plan["index_changes"]:
            store.write(plan["index_key"], _json(plan["index"]))
            checkpoint("raw:index")
            log(f"{plan['index_key']} lists {plan['index']['entries']} entries")
    except PublishEntryError:
        raise
    except (Exception, KeyboardInterrupt) as exc:
        raise PublishEntryError(
            f"the raw entry was not stored completely ({type(exc).__name__}: {exc}); the published database has "
            "not changed; run the command again to continue") from exc


# ---------------------------------------------------------------------------
# Static side
# ---------------------------------------------------------------------------
def plan_static(target: Target, store: Store, database: Database, action: str, glycan_id: str,
                folder: Optional[Path], derived_dir: Path, derived: Dict[str, Any]) -> Dict[str, Any]:
    """Which files are uploaded to their keys, which are staged and switched, which are deleted afterwards."""
    prefix = target.key(f"{glycan_id}/")
    listing = store.list(prefix)
    published = (database.manifest["glycans"].get(glycan_id) or {}).get("files") or {}
    new: List[Dict[str, Any]] = []
    switch: List[Dict[str, Any]] = []
    same = 0
    entry = derived["manifest"]["glycans"].get(glycan_id)
    files = entry["files"] if entry is not None else {}

    def classify(relative: str) -> Tuple[str, Dict[str, Any]]:
        info = files[relative]
        item = {"key": target.key(info["path"]), "size": info["size"], "sha256": info["sha256"],
                "path": str(Path(folder) / relative)}
        if item["key"] not in listing:
            return "new", item
        if listing[item["key"]] == item["size"] and store.holds(item["key"], item["size"], item["sha256"]):
            return "same", item
        # an object that the stored manifest lists is replaced behind a reader's back: staged and switched
        return ("switch" if relative in published else "new"), item

    for kind, item in parallel_map(classify, sorted(files), target.workers):
        if kind == "new":
            new.append(item)
        elif kind == "switch":
            switch.append(item)
        else:
            same += 1
    new.sort(key=lambda item: item["key"])
    switch.sort(key=lambda item: item["key"])
    for name in static_delta.DERIVED_ROOT_FILES:
        info = derived["root_files"][name]
        if info["changed"]:
            switch.append({"key": target.key(name), "size": info["size"], "sha256": info["sha256"],
                           "path": str(derived_dir / name)})
    manifest = {"key": target.key(MANIFEST), "size": derived["manifest_size"], "sha256": derived["manifest_sha256"],
                "path": str(derived_dir / MANIFEST)}
    wanted = {target.key(info["path"]) for info in files.values()}
    return {
        "action": action,
        "glycoshape_id": glycan_id,
        "new": new,
        "switch": switch,
        "manifest": manifest,
        "unchanged": same,
        "stale": sorted(set(listing) - wanted),
        "base_global_hash": derived["base_global_hash"],
        "target_global_hash": derived["global_hash"],
    }


def describe_static(plan: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "upload_new": len(plan["new"]),
        "upload_new_bytes": sum(item["size"] for item in plan["new"]),
        "stage_and_switch": [item["key"] for item in plan["switch"]],
        "stage_bytes": sum(item["size"] for item in plan["switch"]) + plan["manifest"]["size"],
        "unchanged": plan["unchanged"],
        "delete_afterwards": len(plan["stale"]),
        "base_global_hash": plan["base_global_hash"],
        "target_global_hash": plan["target_global_hash"],
    }


def publish_static(target: Target, store: Store, plan: Dict[str, Any], pending_extra: Dict[str, Any], owner: str,
                   lock: Dict[str, Any], lock_ttl: float, log: Log = print, checkpoint: Checkpoint = _noop) -> Dict[str, Any]:
    """Steps 2 to 7 of the module documentation for a planned change."""
    def upload(item: Dict[str, Any]) -> None:
        store.upload(Path(item["path"]), item["key"], item["size"], item["sha256"])

    def stage(item: Dict[str, Any]) -> None:
        if not store.holds(target.staged(item["sha256"]), item["size"], item["sha256"]):   # kept from an interrupted run
            store.upload(Path(item["path"]), target.staged(item["sha256"]), item["size"], item["sha256"])

    # 2. New files, 3. staging.  Nothing that a reader of the stored manifest fetches is touched.
    try:
        for _ in parallel_map(upload, plan["new"], target.workers):
            pass
        checkpoint("uploaded")
        to_stage = list({item["sha256"]: item for item in plan["switch"] + [plan["manifest"]]}.values())
        # No publication is pending (this run holds the lock and has completed one that was): what is
        # staged and not of this change is what an abandoned attempt left.
        wanted = {target.staged(item["sha256"]) for item in to_stage}
        for key in sorted(set(store.list(f"{target.staging_prefix}/{target.static_prefix}/")) - wanted):
            store.delete(key)
        for _ in parallel_map(stage, to_stage, target.workers):
            pass
        checkpoint("staged")
    except (Exception, KeyboardInterrupt) as exc:
        raise PublishEntryError(
            f"the upload stopped ({type(exc).__name__}: {exc}); the published database has not changed for "
            "readers; run the command again to continue") from exc
    log(f"uploaded {len(plan['new'])} new file(s), staged {len(to_stage)} file(s) that replace published ones")

    # 4. Intent.  The stored manifest still has to be the one the new root files were derived from.
    def public(item: Dict[str, Any]) -> Dict[str, Any]:
        return {"key": item["key"], "size": item["size"], "sha256": item["sha256"]}

    pending = {
        "schema": PENDING_SCHEMA, "schema_version": SCHEMA_VERSION, "owner": owner, "created": _utc_now(),
        "action": plan["action"], "glycoshape_id": plan["glycoshape_id"],
        "base_global_hash": plan["base_global_hash"], "target_global_hash": plan["target_global_hash"],
        "switch": [public(item) for item in plan["switch"]],
        "manifest": public(plan["manifest"]),
        "delete": plan["stale"],
        **pending_extra,
    }
    try:
        hold_lock(target, store, lock, lock_ttl)
        if not _claim(target, store, target.pending_key, pending, replace=False):
            other = read_pending(target, store) or {}
            raise PublishEntryError(
                f"another publication is pending ({other.get('action')} of {other.get('glycoshape_id')} by "
                f"{other.get('owner')}); nothing was switched; run the command again")
        current = store.read_optional(target.key(MANIFEST))
        current_hash = None if current is None else json.loads(current).get("global_hash")
        if current_hash != plan["base_global_hash"]:
            store.delete(target.pending_key)
            raise PublishEntryError(
                "the stored manifest changed while this change was prepared (another publication finished): "
                "nothing was switched; run the command again to derive the change from the new database")
    except PublishEntryError:
        raise
    except (Exception, KeyboardInterrupt) as exc:
        raise PublishEntryError(
            f"stopped before anything was switched ({type(exc).__name__}: {exc}); the published database has not "
            "changed for readers; run the command again") from exc
    checkpoint("pending")

    # 5. Switch, 6. manifest, 7. afterwards.
    return roll_forward(target, store, pending, log, checkpoint)


# ---------------------------------------------------------------------------
# The journal of a run (in the work folder)
# ---------------------------------------------------------------------------
class Journal:
    """What a run derived and how far it came, kept next to the bundle so that a retry publishes the same bytes."""

    def __init__(self, work: Path, action: str, glycan_id: str, glycam_name: str, owner: str) -> None:
        self.path = work / "journal.json"
        previous: Dict[str, Any] = {}
        if self.path.is_file():
            try:
                previous = json.loads(self.path.read_bytes())
            except ValueError:
                previous = {}
        same = (previous.get("schema"), previous.get("action"), previous.get("glycoshape_id"), previous.get("glycam_name")) == (
            JOURNAL_SCHEMA, action, glycan_id, glycam_name)
        self.data: Dict[str, Any] = previous if same else {
            "schema": JOURNAL_SCHEMA, "schema_version": SCHEMA_VERSION, "action": action,
            "glycoshape_id": glycan_id, "glycam_name": glycam_name, "started": _utc_now(), "runs": 0, "steps": {},
        }
        self.data["runs"] = int(self.data.get("runs", 0)) + 1
        self.data["owner"] = owner
        self.data["status"] = "running"

    def step(self, name: str, **details: Any) -> None:
        self.data["steps"][name] = {"at": _utc_now(), **details}
        self.save()

    def save(self) -> None:
        self.data["updated"] = _utc_now()
        _write_json(self.path, self.data)


# ---------------------------------------------------------------------------
# Publish and replace
# ---------------------------------------------------------------------------
def check_bundle_policy(bundle: entry_complete.Bundle, policy: Policy) -> Tuple[List[str], List[str]]:
    """Reasons to refuse the bundle under the policy, and warnings."""
    import lib.config as config

    refused, warnings = [], []
    levels = tuple(policy.levels) if policy.levels is not None else tuple(config.cluster_levels)
    if tuple(bundle.levels) != levels:
        refused.append(f"the bundle was computed for levels {list(bundle.levels)} and the database has {list(levels)}")
    pipeline = bundle.pipeline
    converters = policy.converter_versions if policy.converter_versions is not None else (archive.CONVERTER_VERSION,)
    if pipeline.get("converter_version") not in converters:
        refused.append(f"the archive entry was written by converter {pipeline.get('converter_version')!r}; "
                       f"accepted: {', '.join(converters)}")
    for what, patterns in (("version", policy.pipeline_versions), ("commit", policy.pipeline_commits)):
        value = pipeline.get(what)
        if patterns and not (isinstance(value, str) and any(fnmatch.fnmatchcase(value, pattern) for pattern in patterns)):
            refused.append(f"pipeline {what} {value!r} of the bundle is not one of the accepted: {', '.join(patterns)}")
    if not pipeline.get("version") and not pipeline.get("commit"):
        refused.append("the bundle does not say which pipeline made it (no version and no commit)")
    if pipeline.get("uncommitted_changes"):
        (refused if policy.require_clean else warnings).append(
            "the bundle was made from a checkout with uncommitted changes")
    return refused, warnings


def _complete(bundle: entry_complete.Bundle, submission: entry_complete.Submission, work: Path, journal: Journal,
              offline: Optional[bool], lookup_timeout: Optional[float], lookup_budget: Optional[float],
              refresh: bool, log: Log) -> Dict[str, Any]:
    """The completed folder of the entry in the work folder: the one of an earlier run, or a new one.

    The folder of an earlier run is kept when it was made from the same
    bundle, submission and levels, is still what it was, and had no
    ``problems``.  Problems are what an entry is refused for (a file that
    is missing, no SNFG image, neither WURCS nor accession), and they say
    as much about the machine the folder was completed on as about the
    entry: a renderer that was not installed, a tool that could not be run.
    Such a folder is completed again at every run, so a machine that was
    repaired is noticed by the next run and not only by a new work folder.
    Without ``allow_incomplete`` nothing was published from it, so nothing
    is lost; with it the folder comes out the same on the same machine with
    the same answers, and ``expect_hash`` stops anything else than what was
    approved.  Lookups that were not answered are not problems: a folder
    with such gaps is kept, as what was approved, until ``refresh``.
    """
    fingerprint = hashlib.sha256(
        f"{bundle.sha256}\0{submission.sha256()}\0{list(bundle.levels)}".encode()).hexdigest()
    folder = work / "static" / submission.glycoshape_id
    report_path = work / "entry.json"
    kept = journal.data["steps"].get("complete") or {}
    if not refresh and kept.get("fingerprint") == fingerprint and folder.is_dir() and report_path.is_file():
        try:
            report = json.loads(report_path.read_bytes())
        except ValueError:
            report = None
        if report is not None and static_delta.folder_entry(folder)["hash"] == kept.get("hash"):
            if not report.get("problems"):
                log(f"using the folder completed at {kept['at']} ({folder})")
                return report
            log(f"the folder completed at {kept['at']} was not complete ({'; '.join(report['problems'])}): "
                "completing it again")
    report = entry_complete.complete_entry(
        bundle.path, submission, work / "static", offline=offline, lookup_timeout=lookup_timeout,
        lookup_budget=lookup_budget, bundle=bundle)
    report["hash"] = static_delta.folder_entry(folder)["hash"]
    _write_json(report_path, report)
    journal.step("complete", fingerprint=fingerprint, hash=report["hash"], files=report["file_count"],
                 bytes=report["total_size"], offline=report["lookups"]["offline"],
                 unanswered=report["lookups"]["unanswered"], problems=report["problems"])
    log(f"completed {folder.name}: {report['file_count']} files, {report['total_size'] / 1e6:.1f} MB; "
        f"{len(report['gaps'])} empty identity field(s), {len(report['lookups']['unanswered'])} lookup(s) unanswered")
    return report


def _derive(action: str, glycan_id: str, database: Database, folder: Optional[Path], work: Path, journal: Journal,
            generated_at: str, fetch: Optional[Callable[[str], Optional[bytes]]], offline: Optional[bool],
            log: Log) -> Tuple[Path, Dict[str, Any]]:
    """The root files of the changed database: those of an earlier run from the same base, or derived now."""
    from lib import lookups

    entry_hash = None if folder is None else static_delta.folder_entry(folder)["hash"]
    derived_dir = work / "derived" / database.global_hash[:16]
    kept = journal.data["steps"].get("derive") or {}
    report_path = derived_dir / "derived.json"
    if kept.get("base_global_hash") == database.global_hash and kept.get("entry_hash") == entry_hash and report_path.is_file():
        try:
            report = json.loads(report_path.read_bytes())
            intact = all(
                file_sha256(derived_dir / name) == (info["size"], info["sha256"])
                for name, info in list(report["root_files"].items()) + [
                    (MANIFEST, {"size": report["manifest_size"], "sha256": report["manifest_sha256"]})])
        except (ValueError, OSError, KeyError):
            intact = False
        if intact:
            log(f"using the root files derived at {kept['at']} (global_hash {report['global_hash'][:16]}...)")
            return derived_dir, report
    if derived_dir.exists():
        shutil.rmtree(derived_dir)
    with lookups.recording(offline=offline, budget=120):
        report = static_delta.apply_change(
            {PUBLISH: static_delta.ADD, REPLACE: static_delta.REPLACE, WITHDRAW: static_delta.WITHDRAW}[action],
            glycan_id, database.stored_dir, derived_dir, folder, fetch_archive=fetch,
            calculate=glybake.WurcsCalculator(), resolve=glybake.resolve_missing_accessions, generated_at=generated_at)
    _write_json(report_path, report)
    journal.step("derive", base_global_hash=database.global_hash, global_hash=report["global_hash"],
                 entry_hash=entry_hash, seconds=report["seconds"])
    log(f"derived the root files: {report['glycans']} glycans, global_hash {report['base_global_hash'][:16]}... "
        f"-> {report['global_hash'][:16]}...")
    return derived_dir, report


def _refusal(report: Dict[str, Any], refused: List[str]) -> Refused:
    report["ok"] = False
    report["refused"] = list(refused)
    shown = "; ".join(refused[:6]) + (f"; and {len(refused) - 6} more" if len(refused) > 6 else "")
    return Refused(f"not published: {shown}", report)


def publish_entry(
    action: str,
    bundle_dir,
    submission: Any,
    target: Target,
    work_dir=None,
    policy: Optional[Policy] = None,
    dry_run: bool = False,
    offline: Optional[bool] = None,
    lookup_timeout: Optional[float] = None,
    lookup_budget: Optional[float] = 300.0,
    refresh: bool = False,
    expect_hash: Optional[str] = None,
    lock_ttl: float = DEFAULT_LOCK_TTL,
    break_lock: bool = False,
    cache_dir=None,
    owner: Optional[str] = None,
    log: Log = print,
    checkpoint: Checkpoint = _noop,
) -> Dict[str, Any]:
    """Publish the entry of a bundle as a new glycan (``publish``) or in place of a published one (``replace``).

    ``bundle_dir`` is an extracted bundle, ``submission`` its submission
    record (``lib.entry_complete``).  The work folder (default
    ``<bundle_dir>.publish``) receives the completed glycan folder, the
    stored root files, the derived ones and the journal; running again with
    the same work folder continues an interrupted run and publishes the same
    bytes.  ``expect_hash`` is the folder hash an administrator approved (of
    the report of a dry run): a folder that comes out differently is not
    published.  ``refresh`` completes the folder again (new lookups); a
    folder that was completed with problems is completed again at every run
    (:func:`_complete`).

    Checked before anything is written: the bundle (its hashes, levels,
    pipeline; ``policy``), the entry (complete), the ID (new for
    ``publish``, published for this glycan for ``replace``, never one of
    another glycan or of a withdrawn one), the name (not published under
    another ID), the raw entry (extended by the bundle, or the same; another
    one only with ``policy.overwrite_raw``) and the stored database (its
    root files are those of its manifest).  A failed check raises
    :class:`Refused` with every reason; a dry run returns the report
    instead, with ``ok`` false and ``refused``.

    A dry run reads, completes the folder and derives the root files in the
    work folder, and writes nothing to the storage.  Returns a report.
    Raises :class:`PublishEntryError` when the publication stopped: the
    message says whether readers see a change, and the same call continues.
    """
    if action not in (PUBLISH, REPLACE):
        raise PublishEntryError(f"unknown action {action!r}")
    policy = policy or Policy()
    owner = owner or new_owner()
    started = time.perf_counter()
    stats = new_stats()
    static_store, raw_store = Store(target.static, stats, dry_run), Store(target.raw, stats, dry_run)
    submission = entry_complete.Submission.parse(submission)
    bundle = entry_complete.load_bundle(bundle_dir)
    if bundle.glycam_name != submission.glycam_name:
        raise entry_complete.SubmissionError(
            f"the submission is for {submission.glycam_name[:80]!r} and the bundle holds {bundle.glycam_name[:80]!r}")
    glycan_id, name = submission.glycoshape_id, submission.glycam_name
    work = Path(work_dir) if work_dir is not None else Path(str(Path(bundle_dir)) + ".publish")
    work.mkdir(parents=True, exist_ok=True)
    journal = Journal(work, action, glycan_id, name, owner)
    journal.save()
    report: Dict[str, Any] = {
        "action": action, "glycoshape_id": glycan_id, "glycam_name": name, "dry_run": dry_run, "ok": True,
        "owner": owner, "work_dir": str(work), "refused": [], "warnings": [], "transfers": stats,
    }
    refused: List[str] = []
    policy_refused, policy_warnings = check_bundle_policy(bundle, policy)
    refused += policy_refused
    report["warnings"] += policy_warnings

    entry = _complete(bundle, submission, work, journal, offline, lookup_timeout, lookup_budget, refresh, log)
    folder = work / "static" / glycan_id
    report["entry"] = {key: entry[key] for key in (
        "hash", "file_count", "total_size", "identity", "simulation", "search_meta", "gaps", "problems", "lookups",
        "bundle")}
    report["warnings"] += entry["warnings"]
    if entry["problems"] and not policy.allow_incomplete:
        refused += [f"the entry is not complete: {problem}" for problem in entry["problems"]]
    if expect_hash is not None and expect_hash != entry["hash"]:
        refused.append(f"the folder has the hash {entry['hash'][:16]}... and {expect_hash[:16]}... was approved")

    with contextlib.ExitStack() as stack:
        lock: Optional[Dict[str, Any]] = None
        if not dry_run:
            stack.enter_context(_in_process(target))
            lock = acquire_lock(target, static_store, owner, action, glycan_id, lock_ttl, break_lock, log)
            stack.callback(lambda: _release_quietly(target, static_store, lock, log))
            report["conditional_writes"] = target._conditional
            pending = read_pending(target, static_store)
            if pending is not None:                              # an interrupted publication goes first
                log(f"completing the interrupted {pending['action']} of {pending['glycoshape_id']} (by {pending['owner']})")
                roll_forward(target, static_store, pending, log, checkpoint)
        else:
            other, pending = read_lock(target, static_store), read_pending(target, static_store)
            report["lock"] = other
            report["pending"] = None if pending is None else {
                key: pending[key] for key in ("action", "glycoshape_id", "owner", "created")}
            if other is not None and not _is_past(other.get("expires")):
                report["warnings"].append(f"the database is locked by {other.get('owner')} until {other.get('expires')}")
            if pending is not None:
                refused.append(
                    f"an interrupted {pending['action']} of {pending['glycoshape_id']} is pending; it is completed "
                    "by the next publication or by publish_entry.py recover, and this dry run cannot see past it")
                _refusal(report, refused)
                report["seconds"] = round(time.perf_counter() - started, 3)
                return report

        try:
            database = read_database(target, static_store, work, cache_dir)
        except Refused as exc:
            failure = _refusal(report, refused + [str(exc)])
            report["seconds"] = round(time.perf_counter() - started, 3)
            if dry_run:
                return report
            raise failure from exc
        raw_plan, raw_refused = plan_raw(target, raw_store, bundle, policy)
        refused += raw_refused
        raw_index = raw_plan.get("index") if raw_plan else None
        names = _published_names(database, raw_index)
        in_database = glycan_id in database.inventory or glycan_id in database.manifest["glycans"]
        other_ids = sorted(other for other, published in names.items() if published == name and other != glycan_id)
        if other_ids:
            refused.append(f"this glycan is published as {', '.join(other_ids)}")
        if in_database and names.get(glycan_id) != name:
            refused.append(f"{glycan_id} is the ID of another glycan: {names.get(glycan_id, '')[:80]}")
        tombstone = database.withdrawn.get(glycan_id)
        if tombstone is not None and tombstone.get("glycam_name") != name:
            refused.append(f"{glycan_id} was the ID of a withdrawn glycan ({str(tombstone.get('glycam_name'))[:60]}): "
                           "an ID is never given to another glycan")
        for other, item in sorted(database.withdrawn.items()):
            if other != glycan_id and item.get("glycam_name") == name:
                refused.append(f"this glycan was published as {other} and withdrawn: publish it under that ID")
        already = False
        if in_database and names.get(glycan_id) == name:
            stored_entry = database.manifest["glycans"].get(glycan_id) or {}
            already = stored_entry.get("hash") == entry["hash"]
            if action == PUBLISH and not already:
                refused.append(f"{glycan_id} is already published for this glycan with other content: use replace")
        elif action == REPLACE and not in_database:
            refused.append(f"{glycan_id} is not published: use publish")
        report["database"] = {"global_hash": database.global_hash, "glycans": len(database.manifest["glycans"]),
                              "withdrawn": len(database.withdrawn)}
        if raw_plan:
            report["raw"] = describe_raw(raw_plan)
        if refused:
            journal.data["status"] = "refused"
            journal.save()
            failure = _refusal(report, refused)
            report["seconds"] = round(time.perf_counter() - started, 3)
            if dry_run:
                return report
            raise failure

        raw_unchanged = raw_plan["relation"] == "same" and not raw_plan["index_changes"] and not raw_plan["stale"]
        if already:
            # The folder is published with this content: nothing of the static side is left to do.
            report["status"] = "already-published"
            report["static"] = {"base_global_hash": database.global_hash, "target_global_hash": database.global_hash}
            if not dry_run and not raw_unchanged:
                publish_raw(target, raw_store, raw_plan, log, checkpoint)
        else:
            generated_at = journal.data["started"]

            def fetch(other_id: str) -> Optional[bytes]:
                return static_store.read_optional(target.key(f"{other_id}/{other_id}.zip"))

            try:
                derived_dir, derived = _derive(action, glycan_id, database, folder, work, journal, generated_at, fetch, offline, log)
            except static_delta.DeltaError as exc:              # what is stored cannot be changed this way
                failure = _refusal(report, [str(exc)])
                report["seconds"] = round(time.perf_counter() - started, 3)
                if dry_run:
                    return report
                raise failure from exc
            static_plan = plan_static(target, static_store, database, action, glycan_id, folder, derived_dir, derived)
            report["static"] = describe_static(static_plan)
            report["master_archive"] = derived["master_archive"]
            report["status"] = "planned"
            if not dry_run:
                publish_raw(target, raw_store, raw_plan, log, checkpoint)
                journal.step("raw", relation=raw_plan["relation"], bytes=raw_plan["upload_bytes"])
                record = _entry_record(action, glycan_id, name, owner, database.global_hash, derived["global_hash"], entry, report)
                done = publish_static(
                    target, static_store, static_plan,
                    {"glycam_name": name, "record": record, "withdrawn": {} if tombstone is not None else None},
                    owner, lock, lock_ttl, log, checkpoint)
                journal.step("static", base_global_hash=database.global_hash, global_hash=derived["global_hash"], **done)
                report["switch"] = done
                _remember(cache_dir, derived_dir, derived)
                report["status"] = "published"
    report["seconds"] = round(time.perf_counter() - started, 3)
    journal.data["status"] = "dry-run" if dry_run else report["status"]
    journal.data["transfers"] = stats
    journal.save()
    return report


def _release_quietly(target: Target, store: Store, lock: Optional[Dict[str, Any]], log: Log) -> None:
    try:
        release_lock(target, store, lock)
    except Exception as exc:                                    # the lock expires by itself
        log(f"the lock could not be released ({type(exc).__name__}); it expires at {(lock or {}).get('expires')}")


def _remember(cache_dir, derived_dir: Path, derived: Dict[str, Any]) -> None:
    """Keep the new root files by their SHA-256, as the base of the next change; drop what is not of this database."""
    if cache_dir is None:
        return
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    wanted = set()
    for name in static_delta.NEEDED_ROOT_FILES + static_delta.OPTIONAL_ROOT_FILES:
        info = derived["root_files"].get(name)
        if info is not None:
            wanted.add(info["sha256"])
            if not (cache / info["sha256"]).is_file():
                shutil.copyfile(derived_dir / name, cache / info["sha256"])
    for path in cache.iterdir():
        if path.is_file() and len(path.name) == 64 and path.name not in wanted:
            path.unlink()


def _entry_record(action: str, glycan_id: str, name: str, owner: str, base_hash: str, target_hash: str,
                  entry: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
    """What is kept in the storage about the last publication of an entry."""
    return {
        "schema": RECORD_SCHEMA, "schema_version": SCHEMA_VERSION,
        "glycoshape_id": glycan_id, "glycam_name": name, "action": action, "status": "published",
        "published_at": _utc_now(), "owner": owner,
        "global_hash_before": base_hash, "global_hash": target_hash,
        "entry": {key: entry[key] for key in ("hash", "file_count", "total_size")},
        "bundle": entry["bundle"],
        "submission": entry["submission"],
        "gaps": entry["gaps"],
        "lookups": entry["lookups"],
        "warnings": report["warnings"],
    }


# ---------------------------------------------------------------------------
# Withdraw
# ---------------------------------------------------------------------------
def withdraw_entry(
    glycan_id: str,
    target: Target,
    work_dir,
    glycam_name: Optional[str] = None,
    reason: str = "",
    dry_run: bool = False,
    offline: Optional[bool] = None,
    lock_ttl: float = DEFAULT_LOCK_TTL,
    break_lock: bool = False,
    cache_dir=None,
    owner: Optional[str] = None,
    log: Log = print,
    checkpoint: Checkpoint = _noop,
) -> Dict[str, Any]:
    """Take a published glycan out of the database.

    A manifest without the entry is published first (with the root files
    that follow); then ``static/GSxxxxx/`` is deleted.  The raw entry stays
    where it is.  The ID is recorded as withdrawn with the name of its
    glycan: it can come back for that glycan (``publish``), and is refused
    for any other.  ``glycam_name``, when given, has to be the name the ID
    is published for.  Returns a report; raises as :func:`publish_entry`.
    """
    if not id_registry.is_id(glycan_id):
        raise PublishEntryError(f"not a GlycoShape ID: {glycan_id!r}")
    owner = owner or new_owner()
    started = time.perf_counter()
    stats = new_stats()
    store = Store(target.static, stats, dry_run)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {"action": WITHDRAW, "glycoshape_id": glycan_id, "dry_run": dry_run, "ok": True,
                              "owner": owner, "work_dir": str(work), "refused": [], "warnings": [], "transfers": stats}
    with contextlib.ExitStack() as stack:
        lock: Optional[Dict[str, Any]] = None
        if not dry_run:
            stack.enter_context(_in_process(target))
            lock = acquire_lock(target, store, owner, WITHDRAW, glycan_id, lock_ttl, break_lock, log)
            stack.callback(lambda: _release_quietly(target, store, lock, log))
            report["conditional_writes"] = target._conditional
            pending = read_pending(target, store)
            if pending is not None:
                log(f"completing the interrupted {pending['action']} of {pending['glycoshape_id']} (by {pending['owner']})")
                roll_forward(target, store, pending, log, checkpoint)
        else:
            pending = read_pending(target, store)
            if pending is not None:
                _refusal(report, [f"an interrupted {pending['action']} of {pending['glycoshape_id']} is pending"])
                return report

        try:
            database = read_database(target, store, work, cache_dir)
        except Refused as exc:
            failure = _refusal(report, [str(exc)])
            if dry_run:
                return report
            raise failure from exc
        refused: List[str] = []
        raw_index_bytes = Store(target.raw, stats, dry_run).read_optional(f"{target.raw_prefix}/{archive.INDEX_FILE}")
        try:
            raw_index = None if raw_index_bytes is None else json.loads(raw_index_bytes)
        except ValueError:
            raw_index = None
        published_name = _published_names(database, raw_index).get(glycan_id)
        tombstone = database.withdrawn.get(glycan_id)
        report["database"] = {"global_hash": database.global_hash, "glycans": len(database.manifest["glycans"])}
        if glycan_id not in database.inventory:
            if tombstone is not None and (glycam_name is None or tombstone.get("glycam_name") == glycam_name):
                report.update(status="already-withdrawn", glycam_name=tombstone.get("glycam_name"),
                              seconds=round(time.perf_counter() - started, 3))
                leftover = store.list(target.key(f"{glycan_id}/"))
                if leftover and not dry_run:                    # the deletion of an earlier run did not finish
                    for _ in parallel_map(store.delete, sorted(leftover), target.workers):
                        pass
                return report
            refused.append(f"{glycan_id} is not published")
        elif glycam_name is not None and glycam_name not in (published_name, id_registry.full_glycam_name(database.inventory[glycan_id])):
            refused.append(f"{glycan_id} is published for another glycan: {str(published_name)[:80]}")
        if refused:
            failure = _refusal(report, refused)
            if dry_run:
                return report
            raise failure
        name = published_name or ""
        report["glycam_name"] = name
        journal = Journal(work, WITHDRAW, glycan_id, name, owner)
        journal.save()
        try:
            derived_dir, derived = _derive(WITHDRAW, glycan_id, database, None, work, journal, journal.data["started"],
                                           lambda other: store.read_optional(target.key(f"{other}/{other}.zip")), offline, log)
        except static_delta.DeltaError as exc:
            failure = _refusal(report, [str(exc)])
            if dry_run:
                return report
            raise failure from exc
        plan = plan_static(target, store, database, WITHDRAW, glycan_id, None, derived_dir, derived)
        plan["stale"] = []                                      # the whole folder goes, by its prefix
        report["static"] = describe_static(plan)
        report["static"]["delete_afterwards"] = len(store.list(target.key(f"{glycan_id}/")))
        report["status"] = "planned"
        if not dry_run:
            record = {
                "schema": RECORD_SCHEMA, "schema_version": SCHEMA_VERSION, "glycoshape_id": glycan_id,
                "glycam_name": name, "action": WITHDRAW, "status": "withdrawn", "withdrawn_at": _utc_now(),
                "owner": owner, "reason": reason, "global_hash_before": database.global_hash,
                "global_hash": derived["global_hash"],
            }
            done = publish_static(
                target, store, plan,
                {"glycam_name": name, "record": record, "delete_prefix": target.key(f"{glycan_id}/"),
                 "withdrawn": {"glycam_name": name, "withdrawn_at": record["withdrawn_at"], "reason": reason,
                               "global_hash_before": database.global_hash}},
                owner, lock, lock_ttl, log, checkpoint)
            journal.step("static", base_global_hash=database.global_hash, global_hash=derived["global_hash"], **done)
            report["switch"] = done
            _remember(cache_dir, derived_dir, derived)
            report["status"] = "withdrawn"
            journal.data["status"] = "withdrawn"
            journal.data["transfers"] = stats
            journal.save()
    report["seconds"] = round(time.perf_counter() - started, 3)
    return report


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
def status(target: Target, glycan_id: Optional[str] = None, glycam_name: Optional[str] = None,
           check: bool = False) -> Dict[str, Any]:
    """What is stored: the database, the lock, a pending publication, and one entry when it is named.

    Only reads.  ``check`` compares the size and the SHA-256 metadata of the
    root files, and of the files of the named entry, with the manifest.
    """
    stats = new_stats()
    store, raw_store = Store(target.static, stats, dry_run=True), Store(target.raw, stats, dry_run=True)
    report: Dict[str, Any] = {"ok": True, "problems": []}
    lock = read_lock(target, store)
    report["lock"] = None if lock is None else {**lock, "expired": _is_past(lock.get("expires"))}
    pending = read_pending(target, store)
    report["pending"] = None if pending is None else {
        key: pending.get(key) for key in ("action", "glycoshape_id", "glycam_name", "owner", "created",
                                          "base_global_hash", "target_global_hash")}
    if pending is not None:
        report["problems"].append(
            f"an interrupted {pending['action']} of {pending['glycoshape_id']} is pending: readers that are behind "
            "may fail until it is completed (publish_entry.py recover)")
    withdrawn = read_withdrawn(target, store)
    report["withdrawn"] = sorted(withdrawn)
    payload = store.read_optional(target.key(MANIFEST))
    if payload is None:
        report["problems"].append(f"no {target.key(MANIFEST)}")
        report["ok"] = False
        return report
    try:
        manifest = static_publish.parse_manifest(payload, f"the stored {target.key(MANIFEST)}")
    except static_publish.PublishError as exc:
        report["problems"].append(str(exc))
        report["ok"] = False
        return report
    report["database"] = {"global_hash": manifest["global_hash"], "glycans": len(manifest["glycans"]),
                          "generated_at": manifest.get("generated_at"), "root_files": sorted(manifest["root_files"])}

    def differs(item: Dict[str, Any]) -> Optional[str]:
        return None if store.holds(target.key(item["path"]), item["size"], item["sha256"].lower()) else item["path"]

    if check:
        wrong = sorted(path for path in parallel_map(differs, list(manifest["root_files"].values()), target.workers) if path)
        report["root_files_stored"] = not wrong
        report["problems"] += [f"{path} is not stored as the manifest describes it" for path in wrong]

    if glycam_name is not None and glycan_id is None:
        inventory = json.loads(store.read(target.key(static_delta.INVENTORY)))
        matches = [key for key, value in inventory.items() if id_registry.full_glycam_name(value) == glycam_name]
        matches += [key for key, value in withdrawn.items() if value.get("glycam_name") == glycam_name and key not in matches]
        glycan_id = matches[0] if matches else None
        report["ids_of_the_name"] = matches
    if glycan_id is not None:
        entry: Dict[str, Any] = {"glycoshape_id": glycan_id}
        listed = manifest["glycans"].get(glycan_id)
        stored_keys = store.list(target.key(f"{glycan_id}/"))
        entry["state"] = "published" if listed is not None else "withdrawn" if glycan_id in withdrawn else "not published"
        entry["withdrawn"] = withdrawn.get(glycan_id)
        entry["stored_objects"] = len(stored_keys)
        record = store.read_optional(target.record_key(glycan_id))
        entry["record"] = None if record is None else json.loads(record)
        if listed is not None:
            entry.update(hash=listed["hash"], file_count=listed["file_count"], total_size=listed["total_size"])
            unlisted = sorted(set(stored_keys) - {target.key(item["path"]) for item in listed["files"].values()})
            entry["unlisted_objects"] = unlisted
            if check:
                wrong = sorted(path for path in parallel_map(differs, list(listed["files"].values()), target.workers) if path)
                entry["files_stored"] = not wrong
                report["problems"] += [f"{path} is not stored as the manifest describes it" for path in wrong]
        elif stored_keys:
            report["problems"].append(f"{len(stored_keys)} object(s) are stored under {target.key(glycan_id)}/ "
                                      "and no manifest lists them")
        name = glycam_name or (entry["record"] or {}).get("glycam_name") or (withdrawn.get(glycan_id) or {}).get("glycam_name")
        if name:
            try:
                key_prefix = archive_transfer.entry_key_prefix(target.raw_prefix, name)
                raw_manifest = raw_store.head(key_prefix + archive.MANIFEST_FILE)
                entry["raw"] = {"key_prefix": key_prefix, "stored": raw_manifest is not None,
                                "manifest_sha256": None if raw_manifest is None else raw_manifest["sha256"]}
            except archive.ArchiveError as exc:
                entry["raw"] = {"error": str(exc)}
        report["entry"] = entry
    report["ok"] = not report["problems"]
    return report
