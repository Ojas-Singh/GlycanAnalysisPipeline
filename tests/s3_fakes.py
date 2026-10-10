"""Test doubles for the S3 storage: an in-memory client, the reader of the website, sample data.

``FakeS3Client`` has the methods of a boto3 S3 client that ``lib.storage``
calls, keeps objects in memory and records every call, so the unit tests need
neither boto3 nor a server.  ``reader_sync`` is a transcription of
``sync_gap_database`` in ``glycoshape-api/src/gap_sync.rs``: the tests use it
to see a published prefix the way the website does.
"""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


#: 267 characters, the longest GLYCAM name in the data (brackets), and a name with commas.
LONG_NAME = (
    "Neup5Aca2-3[DGalpNAcb1-4]DGalpb1-4DGlcpNAcb1-2[Neup5Aca2-3[DGalpNAcb1-4]DGalpb1-4DGlcpNAcb1-4]"
    "DManpa1-3[DGlcpNAcb1-4][Neup5Aca2-3[DGalpNAcb1-4]DGalpb1-4DGlcpNAcb1-2[Neup5Aca2-3[DGalpNAcb1-4]"
    "DGalpb1-4DGlcpNAcb1-6]DManpa1-6]DManpb1-4DGlcpNAcb1-4[LFucpa1-6]DGlcpNAcb1-OH"
)
COMMA_NAME = "LIdopA[2S]a1-4DGlcpNS[3S,6S]a1-4DGlcpA[2S,3S]b1-OH"


class FakeClientError(Exception):
    """Shaped like ``botocore.exceptions.ClientError``."""

    def __init__(self, code: str, status: int, operation: str = "") -> None:
        super().__init__(f"An error occurred ({code}) when calling the {operation} operation")
        self.response = {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}


class Interrupted(Exception):
    """Raised by a test hook to stop a transfer at a chosen request."""


class _Body:
    def __init__(self, payload: bytes, failure: Optional[BaseException]) -> None:
        self._handle = io.BytesIO(payload)
        self._failure = failure

    def read(self, amount: Optional[int] = None) -> bytes:
        if self._failure is not None:
            failure, self._failure = self._failure, None
            self._handle.read(3)   # part of the body arrived before the connection broke
            raise failure
        return self._handle.read(amount)

    def close(self) -> None:
        pass


class FakeS3Client:
    """In-memory S3 with the boto3 client methods that the storage backend uses."""

    def __init__(self, *buckets: str) -> None:
        self.objects: Dict[str, Dict[str, Dict[str, Any]]] = {bucket: {} for bucket in buckets}
        self.calls: List[Tuple[str, str, Optional[str]]] = []
        #: ``hook(operation, bucket, key)`` runs before every request and may raise.
        self.before: Optional[Callable[[str, str, Optional[str]], None]] = None
        #: Exceptions for the next body reads, one per ``get_object``.
        self.read_failures: List[BaseException] = []
        self.transfer_configs: List[Any] = []
        #: Multipart uploads that were started and neither completed nor aborted.
        self.multipart: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    # -- helpers for tests ------------------------------------------------
    def keys(self, bucket: str, prefix: str = "") -> List[str]:
        return sorted(key for key in self.objects[bucket] if key.startswith(prefix))

    def body(self, bucket: str, key: str) -> bytes:
        return self.objects[bucket][key]["body"]

    def writes(self) -> List[Tuple[str, str, Optional[str]]]:
        """The requests that change an object (a multipart copy counts when it is completed)."""
        changing = ("put_object", "upload_file", "copy_object", "complete_multipart_upload", "delete_object")
        return [call for call in self.calls if call[0] in changing]

    def count(self, operation: str) -> int:
        return sum(1 for call in self.calls if call[0] == operation)

    def _begin(self, operation: str, bucket: str, key: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            self.calls.append((operation, bucket, key))
        if self.before is not None:
            self.before(operation, bucket, key)
        if bucket not in self.objects:
            # the answer to a HEAD has no body: the client then reports the status as the code
            headed = operation in ("head_object", "download_file")
            raise FakeClientError("404" if headed else "NoSuchBucket", 404, operation)
        return self.objects[bucket]

    def _store(self, bucket: str, key: str, body: bytes, metadata: Any, content_type: Any) -> None:
        with self._lock:
            self.objects[bucket][key] = {
                "body": bytes(body), "metadata": dict(metadata or {}), "content_type": content_type,
            }

    # -- client methods ---------------------------------------------------
    def list_objects_v2(self, Bucket, Prefix="", MaxKeys=1000, ContinuationToken=None, Delimiter=None):
        objects = self._begin("list_objects_v2", Bucket, Prefix)
        with self._lock:
            entries: List[Tuple[str, bool]] = []   # (key or common prefix, is a common prefix)
            for key in sorted(k for k in objects if k.startswith(Prefix)):
                rest = key[len(Prefix):]
                if Delimiter and Delimiter in rest:
                    marker = Prefix + rest.split(Delimiter, 1)[0] + Delimiter
                    if not entries or entries[-1] != (marker, True):
                        entries.append((marker, True))
                else:
                    entries.append((key, False))
            if ContinuationToken:
                entries = [entry for entry in entries if entry[0] > ContinuationToken]
            page, rest_of_listing = entries[:MaxKeys], entries[MaxKeys:]
            response: Dict[str, Any] = {
                "Contents": [
                    {"Key": key, "Size": len(objects[key]["body"]),
                     "ETag": '"%s"' % hashlib.md5(objects[key]["body"]).hexdigest()}
                    for key, is_prefix in page if not is_prefix
                ],
                "CommonPrefixes": [{"Prefix": key} for key, is_prefix in page if is_prefix],
                "IsTruncated": bool(rest_of_listing),
                "KeyCount": len(page),
            }
            if rest_of_listing:
                response["NextContinuationToken"] = page[-1][0]
            return response

    def head_object(self, Bucket, Key):
        objects = self._begin("head_object", Bucket, Key)
        with self._lock:
            if Key not in objects:
                raise FakeClientError("404", 404, "HeadObject")
            item = objects[Key]
            return {
                "ContentLength": len(item["body"]),
                "ETag": '"%s"' % hashlib.md5(item["body"]).hexdigest(),
                "Metadata": dict(item["metadata"]),
                "ContentType": item["content_type"],
            }

    def get_object(self, Bucket, Key):
        objects = self._begin("get_object", Bucket, Key)
        with self._lock:
            if Key not in objects:
                raise FakeClientError("NoSuchKey", 404, "GetObject")
            failure = self.read_failures.pop(0) if self.read_failures else None
            return {"Body": _Body(objects[Key]["body"], failure)}

    def put_object(self, Bucket, Key, Body, Metadata=None, ContentType=None):
        self._begin("put_object", Bucket, Key)
        self._store(Bucket, Key, Body, Metadata, ContentType)
        return {}

    def delete_object(self, Bucket, Key):
        objects = self._begin("delete_object", Bucket, Key)
        with self._lock:
            objects.pop(Key, None)
        return {}

    def upload_file(self, Filename, Bucket, Key, ExtraArgs=None, Config=None):
        self._begin("upload_file", Bucket, Key)
        extra = ExtraArgs or {}
        self.transfer_configs.append(Config)
        self._store(Bucket, Key, Path(Filename).read_bytes(), extra.get("Metadata"), extra.get("ContentType"))

    def download_file(self, Bucket, Key, Filename, Config=None):
        objects = self._begin("download_file", Bucket, Key)
        with self._lock:
            if Key not in objects:
                raise FakeClientError("404", 404, "HeadObject")
            payload = objects[Key]["body"]
        Path(Filename).write_bytes(payload)

    def _source(self, copy_source, operation):
        source = self.objects.get(copy_source["Bucket"], {}).get(copy_source["Key"])
        if source is None:
            raise FakeClientError("NoSuchKey", 404, operation)
        return source

    def copy_object(self, Bucket, Key, CopySource, MetadataDirective="COPY", Metadata=None, ContentType=None):
        self._begin("copy_object", Bucket, Key)
        with self._lock:
            source = self._source(CopySource, "CopyObject")
            replace = MetadataDirective == "REPLACE"
            self._store(
                Bucket, Key, source["body"],
                Metadata if replace else source["metadata"],
                ContentType if replace else source["content_type"],
            )
        return {}

    def create_multipart_upload(self, Bucket, Key, Metadata=None, ContentType=None):
        self._begin("create_multipart_upload", Bucket, Key)
        with self._lock:
            upload_id = f"upload-{len(self.multipart) + 1}"
            self.multipart[upload_id] = {"key": Key, "metadata": Metadata, "content_type": ContentType, "parts": {}}
        return {"UploadId": upload_id}

    def upload_part_copy(self, Bucket, Key, UploadId, PartNumber, CopySource, CopySourceRange):
        self._begin("upload_part_copy", Bucket, Key)
        with self._lock:
            first, last = (int(value) for value in CopySourceRange[len("bytes="):].split("-"))
            part = self._source(CopySource, "UploadPartCopy")["body"][first:last + 1]
            self.multipart[UploadId]["parts"][PartNumber] = part
        return {"CopyPartResult": {"ETag": '"%s"' % hashlib.md5(part).hexdigest()}}

    def complete_multipart_upload(self, Bucket, Key, UploadId, MultipartUpload):
        self._begin("complete_multipart_upload", Bucket, Key)
        with self._lock:
            upload = self.multipart.pop(UploadId)
            numbers = [part["PartNumber"] for part in MultipartUpload["Parts"]]
            assert numbers == sorted(upload["parts"]) == list(range(1, len(numbers) + 1))
            body = b"".join(upload["parts"][number] for number in numbers)
            self._store(Bucket, Key, body, upload["metadata"], upload["content_type"])
        return {}

    def abort_multipart_upload(self, Bucket, Key, UploadId):
        self._begin("abort_multipart_upload", Bucket, Key)
        with self._lock:
            self.multipart.pop(UploadId, None)
        return {}


# ---------------------------------------------------------------------------
# The reader of the website (gap_sync.rs)
# ---------------------------------------------------------------------------
class ReaderError(Exception):
    """``GapSyncError`` of the reader."""


def _reader_path(value: str) -> List[str]:
    """``safe_relative_path``: the components of a manifest path, or an error."""
    value = value.strip().lstrip("/").replace("\\", "/")
    parts = [part for part in value.split("/") if part not in ("", ".")] if value else []
    if not parts or ".." in parts or value.startswith("./") or any(part.startswith(".") for part in parts):
        raise ReaderError(f"GAP manifest is invalid: unsafe object path {value!r}")
    return parts


def _reader_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdefABCDEF" for c in value)


def _reader_file(entry: Dict[str, Any]) -> List[str]:
    """``validate_file``; a missing or mistyped field fails as deserialisation does."""
    if not isinstance(entry["path"], str) or not isinstance(entry["sha256"], str) \
            or type(entry["size"]) is not int or entry["size"] < 0:
        raise ReaderError("GAP request failed: manifest does not deserialise")
    parts = _reader_path(entry["path"])
    if not _reader_sha256(entry["sha256"]):
        raise ReaderError(f"GAP manifest is invalid: invalid SHA-256 for {entry['path']}")
    return parts


def reader_validate(manifest: Dict[str, Any]) -> None:
    """``validate_manifest``."""
    try:
        if manifest["schema_version"] != 1 or type(manifest["schema_version"]) is not int \
                or manifest["hash_algorithm"] != "sha256":
            raise ReaderError("GAP manifest is invalid: expected schema_version 1 with sha256 hashes")
        if not _reader_sha256(manifest["global_hash"]) or not manifest["glycans"]:
            raise ReaderError("GAP manifest is invalid: missing global hash or glycan entries")
        for entry in manifest["root_files"].values():
            if len(_reader_file(entry)) != 1:
                raise ReaderError(f"GAP manifest is invalid: root file is not at the database root: {entry['path']}")
        for glycan_id, glycan in manifest["glycans"].items():
            if (
                not glycan_id.startswith("GS")
                or "/" in glycan_id
                or not _reader_sha256(glycan["hash"])
                or type(glycan["file_count"]) is not int
                or glycan["file_count"] != len(glycan["files"])
                or not glycan["files"]
            ):
                raise ReaderError(f"GAP manifest is invalid: invalid glycan manifest entry {glycan_id!r}")
            for entry in glycan["files"].values():
                if _reader_file(entry)[0] != glycan_id:
                    raise ReaderError(f"GAP manifest is invalid: {entry['path']} is outside {glycan_id}")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ReaderError(f"GAP request failed: manifest does not deserialise ({exc!r})") from exc


def reader_sync(backend: Any, prefix: str, database: Path, prune: bool = True) -> Dict[str, int]:
    """``sync_gap_database``: bring ``database`` to the published state, or raise and leave it alone."""
    database = Path(database)
    database.mkdir(parents=True, exist_ok=True)
    try:
        manifest_bytes = backend.read_binary(f"{prefix}/sync_manifest.json")
        manifest = json.loads(manifest_bytes)
    except (FileNotFoundError, ValueError) as exc:
        raise ReaderError(f"GAP request failed: {exc}") from exc
    reader_validate(manifest)

    manifest_path = database / "sync_manifest.json"
    try:
        previous = json.loads(manifest_path.read_bytes())
    except (OSError, ValueError):
        previous = None
    file_total = len(manifest["root_files"]) + sum(g["file_count"] for g in manifest["glycans"].values())
    if previous is not None and previous.get("global_hash") == manifest["global_hash"]:
        return {"downloaded": 0, "skipped": file_total, "pruned": 0}

    staging = database / ".gap-sync-test"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    summary = {"downloaded": 0, "skipped": 0, "pruned": 0}

    def download(entry: Dict[str, Any]) -> None:
        try:
            payload = backend.read_binary(f"{prefix}/{entry['path'].strip('/')}")
        except FileNotFoundError as exc:
            raise ReaderError(f"GAP request failed: {exc}") from exc
        if len(payload) != entry["size"] or hashlib.sha256(payload).hexdigest() != entry["sha256"].lower():
            raise ReaderError(f"GAP manifest is invalid: download verification failed for {entry['path']}")
        target = staging.joinpath(*_reader_file(entry))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        summary["downloaded"] += 1

    def unchanged(glycan_id: str, glycan: Dict[str, Any]) -> bool:
        old = (previous or {}).get("glycans", {}).get(glycan_id)
        return (
            old is not None and old["hash"] == glycan["hash"] and (database / glycan_id).is_dir()
            and all(database.joinpath(*_reader_file(entry)).is_file() for entry in glycan["files"].values())
        )

    for entry in manifest["root_files"].values():
        download(entry)
    changed = [glycan_id for glycan_id, glycan in manifest["glycans"].items() if not unchanged(glycan_id, glycan)]
    for glycan_id, glycan in manifest["glycans"].items():
        if glycan_id in changed:
            for entry in glycan["files"].values():
                download(entry)
        else:
            summary["skipped"] += glycan["file_count"]

    for entry in manifest["root_files"].values():
        parts = _reader_file(entry)
        staging.joinpath(*parts).replace(database.joinpath(*parts))
    for glycan_id in changed:
        if (database / glycan_id).exists():
            shutil.rmtree(database / glycan_id)
        (staging / glycan_id).rename(database / glycan_id)
    if prune:
        for path in list(database.iterdir()):
            if path.name.startswith("GS") and path.is_dir() and path.name not in manifest["glycans"]:
                shutil.rmtree(path)
                summary["pruned"] += 1
    manifest_path.write_bytes(manifest_bytes)
    shutil.rmtree(staging)
    if not (database / "GLYCOSHAPE.json").is_file():
        raise ReaderError("GAP manifest is invalid: sync completed without GLYCOSHAPE.json")
    return summary


# ---------------------------------------------------------------------------
# A static directory as main.py --mode static-only leaves it
# ---------------------------------------------------------------------------
def tree(root: Path) -> Dict[str, bytes]:
    """Relative path -> content of every file below ``root``."""
    root = Path(root)
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def write_static(static_dir: Path, glycans: Dict[str, Dict[str, bytes]], root_files: Optional[Dict[str, bytes]] = None):
    """Write glycan folders and root files, then ``sync_manifest.json`` with the generator of the pipeline."""
    from lib.glybake import GlycoShapeBaker

    static_dir = Path(static_dir)
    static_dir.mkdir(parents=True, exist_ok=True)
    for name, content in ({"GLYCOSHAPE.json": b'{"version": 1}'} if root_files is None else root_files).items():
        (static_dir / name).write_bytes(content)
    for glycan_id, files in glycans.items():
        for relative, content in files.items():
            target = static_dir / glycan_id / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    GlycoShapeBaker(static_dir).create_sync_manifest()
    return json.loads((static_dir / "sync_manifest.json").read_bytes())


def sample_glycans(count: int = 3, seed: str = "") -> Dict[str, Dict[str, bytes]]:
    return {
        f"GS{index:05d}": {
            "data.json": json.dumps({"archetype": {"ID": f"GS{index:05d}", "seed": seed}}).encode(),
            "snfg.svg": b"<svg/>",
            "output/pca.csv": f"x,y\n{index},{index + 1}\n".encode(),
            "output/level_1/PDB/alpha.pdb": f"ATOM {index} {seed}\n".encode() * 5,
        }
        for index in range(1, count + 1)
    }
