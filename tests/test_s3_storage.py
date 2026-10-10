"""Unit tests of the S3 storage backend (lib/storage.py) on an in-memory client.

The two tests at the end drive a real boto3 client against a local HTTP
recorder (no S3 server) and are skipped when boto3 is not installed.
"""
from __future__ import annotations

import hashlib
import http.server
import logging
import threading
import time
from pathlib import Path

import pytest

from lib import config, storage
from lib.storage import S3Settings, S3StorageBackend, StorageManager
from s3_fakes import COMMA_NAME, LONG_NAME, FakeClientError, FakeS3Client

ENVIRONMENT = {
    "GLYCOSHAPE_S3_ENDPOINT": "https://account.r2.cloudflarestorage.com",
    "GLYCOSHAPE_S3_ACCESS_KEY_ID": "key-id",
    "GLYCOSHAPE_S3_SECRET_ACCESS_KEY": "do-not-print-this-secret",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Backend operations
# ---------------------------------------------------------------------------
def test_objects_are_written_read_listed_and_deleted(fake_s3):
    backend, client = fake_s3
    backend.write_text("static/GS00001/data.json", '{"a": 1}')
    backend.write_binary("static/GS00001/output/pca.csv", b"x,y\n")
    backend.write_binary("static/GLYCOSHAPE.json", b"{}")
    backend.write_binary("staticky/other.txt", b"not below static/")

    assert backend.read_text("static/GS00001/data.json") == '{"a": 1}'
    assert backend.read_binary("/static/GS00001/output/pca.csv") == b"x,y\n"   # a leading slash is not part of a key
    assert backend.exists("static/GS00001/data.json") and backend.is_file("static/GS00001/data.json")
    assert backend.exists("static/GS00001") and backend.exists("static/GS00001/") and backend.is_dir("static/GS00001")
    assert not backend.is_file("static/GS00001") and not backend.is_dir("static/GS00001/data.json")
    assert not backend.exists("static/GS00002") and not backend.is_dir("static/GS0000")
    assert backend.is_dir("") and not backend.exists("missing")

    head = backend.head("static/GS00001/data.json")
    assert head["size"] == 8 and head["sha256"] == sha256(b'{"a": 1}') and head["etag"] and backend.head("nope") is None
    assert client.objects["bucket"]["static/GS00001/data.json"]["content_type"] == "application/json"

    listed = backend.list_files("static")
    assert listed == [Path("static/GLYCOSHAPE.json"), Path("static/GS00001/data.json"),
                      Path("static/GS00001/output/pca.csv")]
    assert backend.list_files("static", "GS*") == listed[1:] and backend.list_files_paginated("static/") == listed
    assert backend.list_files("static", "*.json") == listed[:1] and backend.list_files("missing") == []
    assert backend.list_common_prefixes("static") == ["static/GS00001"]
    assert backend.list_common_prefixes("") == ["static", "staticky"]
    assert [item["key"] for item in backend.list_objects("static/GS")] == [
        "static/GS00001/data.json", "static/GS00001/output/pca.csv"]

    with backend.open_text("static/GS00001/data.json") as handle:
        assert handle.read() == '{"a": 1}'
    with backend.open_binary("static/new.bin", "wb") as handle:
        handle.write(b"written on close")
    assert backend.open_binary("static/new.bin").read() == b"written on close"
    backend.mkdir("static/GS00009")
    assert not backend.exists("static/GS00009")

    backend.unlink("static/new.bin")
    backend.unlink("static/new.bin")   # deleting a missing object is not an error
    assert not backend.exists("static/new.bin")
    for read in (backend.read_binary, backend.read_text, backend.stream_sha256):
        with pytest.raises(FileNotFoundError):
            read("static/new.bin")


def test_listing_follows_every_page():
    client = FakeS3Client("bucket")
    backend = S3StorageBackend(client, bucket="bucket", page_size=7)
    keys = sorted(f"raw/G{entry:02d}/{name}" for entry in range(10) for name in ("a", "b", "c", "d"))
    for key in keys:
        backend.write_binary(key, key.encode())
    client.calls.clear()

    assert [item["key"] for item in backend.list_objects("raw/")] == keys
    assert client.count("list_objects_v2") == 6   # 40 keys in pages of 7
    assert [item["size"] for item in backend.list_objects("raw/G03/")] == [len("raw/G03/a")] * 4
    assert backend.list_files("raw") == [Path(key) for key in keys]
    assert backend.list_common_prefixes("raw") == [f"raw/G{entry:02d}" for entry in range(10)]
    assert backend.is_dir("raw/G09") and not backend.is_dir("raw/G10")

    client.list_objects_v2 = lambda **request: {"Contents": [], "IsTruncated": True}
    with pytest.raises(RuntimeError, match="without a continuation token"):
        list(backend.list_objects("raw/"))
    with pytest.raises(RuntimeError, match="without a continuation token"):
        backend.list_common_prefixes("raw")


def test_a_bucket_that_does_not_exist_is_an_error_where_it_can_be_told():
    backend = S3StorageBackend(FakeS3Client("bucket"), bucket="other-bucket")
    assert backend.head("raw/x") is None and not backend.is_file("raw/x")   # a HEAD cannot tell
    for action in (lambda: list(backend.list_objects("raw/")), lambda: backend.exists("raw/x"),
                   lambda: backend.is_dir("raw"), lambda: backend.read_binary("raw/x"),
                   lambda: backend.write_binary("raw/x", b"")):
        with pytest.raises(FakeClientError, match="NoSuchBucket"):
            action()


@pytest.mark.parametrize("name", [LONG_NAME, COMMA_NAME, "DGlcpa1-OH", "name with space+plus%25 ünïcode"])
def test_glycam_names_are_keys_as_they_are(fake_s3, name):
    backend, client = fake_s3
    assert len(LONG_NAME) == 267 and "[" in LONG_NAME and "," in COMMA_NAME
    key = f"raw/{name}/trajectory.xtc"
    backend.write_binary(key, b"frames")
    assert client.keys("bucket") == [key]                       # stored under exactly this key
    assert backend.read_binary(key) == b"frames" and backend.exists(key) and backend.is_dir(f"raw/{name}")
    assert [item["key"] for item in backend.list_objects(f"raw/{name}/")] == [key]
    assert backend.list_common_prefixes("raw") == [f"raw/{name}"]
    assert backend.head(key)["size"] == 6


def test_keys_longer_than_the_limit_are_refused(fake_s3):
    backend, client = fake_s3
    backend.write_binary("raw/" + "D" * 1020, b"")              # 1024 bytes: the limit
    for key in ("raw/" + "D" * 1021, "raw/" + "é" * 511):       # 1025 bytes; 1026 bytes of UTF-8
        with pytest.raises(ValueError, match="longer than 1024 bytes"):
            backend.write_binary(key, b"")
    with pytest.raises(ValueError):
        backend.write_binary("", b"")
    assert len(client.keys("bucket")) == 1


def test_files_are_uploaded_downloaded_and_copied_as_streams(tmp_path):
    client = FakeS3Client("bucket")
    backend = S3StorageBackend(client, bucket="bucket", transfer_config="transfer-config")
    payload = bytes(range(256)) * 1000
    local = tmp_path / "trajectory.xtc"
    local.write_bytes(payload)

    assert backend.upload_file(local, f"raw/{LONG_NAME}/trajectory.xtc")["sha256"] == sha256(payload)
    stored = client.objects["bucket"][f"raw/{LONG_NAME}/trajectory.xtc"]
    assert stored["body"] == payload and stored["metadata"] == {"sha256": sha256(payload)}
    assert client.transfer_configs == ["transfer-config"]       # multipart threshold and part size reach the client
    assert client.count("put_object") == 0                      # the file is never read into memory here
    backend.upload_file(local, "raw/x/known", sha256="0" * 64)   # a known hash is not computed again
    assert backend.head("raw/x/known")["sha256"] == "0" * 64

    backend.download_file(f"raw/{LONG_NAME}/trajectory.xtc", tmp_path / "back.xtc")
    assert (tmp_path / "back.xtc").read_bytes() == payload
    assert backend.stream_sha256(f"raw/{LONG_NAME}/trajectory.xtc") == (len(payload), sha256(payload))
    with pytest.raises(FileNotFoundError):
        backend.download_file("raw/missing", tmp_path / "missing")
    assert not (tmp_path / "missing").exists()

    # a copy stays inside the storage: one request, metadata of the source unless given
    backend.copy(f"raw/{LONG_NAME}/trajectory.xtc", "static/copy.bin")
    assert client.body("bucket", "static/copy.bin") == payload
    assert backend.head("static/copy.bin")["sha256"] == sha256(payload)
    backend.copy("raw/x/known", "static/copy.bin", sha256="1" * 64, size=len(payload))
    assert backend.head("static/copy.bin")["sha256"] == "1" * 64
    assert client.count("copy_object") == 2 and client.count("get_object") == 1   # only stream_sha256 downloaded
    with pytest.raises(FileNotFoundError):
        backend.copy("raw/missing", "static/copy.bin")


def test_a_large_object_is_copied_in_parts(tmp_path):
    client = FakeS3Client("bucket")
    backend = S3StorageBackend(client, bucket="bucket", copy_part_size=5 << 20)
    payload = bytes(range(256)) * (12 * 4096 + 1)                 # 12 MiB and a little: three parts
    client.put_object(Bucket="bucket", Key="publish-staging/static/abc", Body=payload)

    backend.copy("publish-staging/static/abc", "static/GlycoShape.zip", sha256=sha256(payload), size=len(payload))
    stored = client.objects["bucket"]["static/GlycoShape.zip"]
    assert stored["body"] == payload and stored["metadata"] == {"sha256": sha256(payload)}
    assert stored["content_type"] == "application/zip"
    assert (client.count("upload_part_copy"), client.count("copy_object"), client.multipart) == (3, 0, {})
    assert client.count("get_object") == client.count("head_object") == 0   # nothing was downloaded

    def breaks(operation, bucket, key):
        if operation == "upload_part_copy" and client.count("upload_part_copy") == 5:
            raise ConnectionResetError("reset")

    client.before = breaks
    with pytest.raises(ConnectionResetError):
        backend.copy("publish-staging/static/abc", "static/other.zip", size=len(payload), sha256="0" * 64)
    assert client.multipart == {} and "static/other.zip" not in client.objects["bucket"]   # aborted, nothing left


def test_prefixes_are_routed_to_their_buckets():
    client = FakeS3Client("glycoshape-archive", "glycoshape-database")
    settings = S3Settings.from_env(ENVIRONMENT)
    backend = S3StorageBackend(client, routes=settings.routes())
    backend.write_binary("raw/DGlcpa1-OH/manifest.json", b"{}")
    backend.write_binary("process/DGlcpa1-OH/output/info.json", b"{}")
    backend.write_binary("static/GS00001/data.json", b"{}")
    assert client.keys("glycoshape-archive") == ["process/DGlcpa1-OH/output/info.json", "raw/DGlcpa1-OH/manifest.json"]
    assert client.keys("glycoshape-database") == ["static/GS00001/data.json"]
    assert backend.list_files("static") == [Path("static/GS00001/data.json")]
    assert backend.list_common_prefixes("raw") == ["raw/DGlcpa1-OH"]
    assert backend.is_routed("static/x") and backend.is_routed("raw") and not backend.is_routed("rawdata/x")

    # a key outside the three prefixes belongs to no bucket
    for path in ("GlycoShape_Inventory.csv", "/scratch/project/static/GS00001/data.json", "rawdata/x"):
        assert not backend.exists(path) and not backend.is_file(path) and not backend.is_dir(path)
        assert backend.head(path) is None and backend.list_files(path) == []
        with pytest.raises(FileNotFoundError, match="no S3 bucket is configured"):
            backend.read_binary(path)
        with pytest.raises(FileNotFoundError, match="no S3 bucket is configured"):
            backend.write_binary(path, b"")
    assert all(call[1] in client.objects for call in client.calls)


# ---------------------------------------------------------------------------
# Retries while a body is read
# ---------------------------------------------------------------------------
class IncompleteReadError(Exception):
    """Named like the botocore error for a body that ended early."""


def test_a_body_that_breaks_is_read_again_with_backoff():
    client = FakeS3Client("bucket")
    pauses = []
    backend = S3StorageBackend(client, bucket="bucket", stream_attempts=4, sleep=pauses.append)
    payload = b"0123456789" * 1000
    backend.write_binary("raw/x/trajectory.xtc", payload)

    client.read_failures = [ConnectionResetError("reset"), IncompleteReadError("short"), TimeoutError("slow")]
    assert backend.read_binary("raw/x/trajectory.xtc") == payload
    assert client.count("get_object") == 4 and len(pauses) == 3
    assert pauses[0] < pauses[1] < pauses[2] and 0.25 <= pauses[0] <= 0.5 and 1.0 <= pauses[2] <= 2.0

    client.read_failures = [ConnectionResetError("reset")]
    assert backend.stream_sha256("raw/x/trajectory.xtc") == (len(payload), sha256(payload))   # hashed from the start again

    client.read_failures = [ConnectionResetError("reset")] * 4       # gives up after the configured attempts
    with pytest.raises(ConnectionResetError):
        backend.read_binary("raw/x/trajectory.xtc")
    client.read_failures.clear()

    client.calls.clear()
    client.read_failures = [ValueError("not a connection problem")]
    with pytest.raises(ValueError):
        backend.read_binary("raw/x/trajectory.xtc")
    with pytest.raises(FileNotFoundError):
        backend.read_binary("raw/x/missing")
    assert client.count("get_object") == 2                           # neither was tried twice

    def denied(**request):
        raise FakeClientError("AccessDenied", 403, "HeadObject")

    client.head_object = denied
    with pytest.raises(FakeClientError):                             # only "not found" means absent
        backend.exists("raw/x/trajectory.xtc")


def test_too_many_requests_asks_for_another_attempt_with_growing_pauses():
    class Response:
        def __init__(self, status_code):
            self.status_code = status_code

    retry = storage._retry_too_many_requests(max_attempts=4)
    pauses = [retry(response=(Response(429), {"Error": {"Code": "TooManyRequests"}}), attempts=n) for n in (1, 2, 3)]
    assert 0.5 <= pauses[0] <= 1.0 <= pauses[1] <= 2.0 <= pauses[2] <= 4.0
    assert retry(response=(Response(429), {}), attempts=4) is None                 # the last attempt was made
    assert retry(response=(Response(200), {"Error": {"Code": "TooManyRequests"}}), attempts=1) is not None
    for status in (200, 403, 404, 503):                                            # 5xx is the client's own rule
        assert retry(response=(Response(status), {"Error": {"Code": str(status)}}), attempts=1) is None
    assert retry(response=None, attempts=1, caught_exception=ConnectionError()) is None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def test_settings_come_from_the_environment_and_hide_the_secret():
    settings = S3Settings.from_env(ENVIRONMENT)
    assert settings.endpoint_url == "https://account.r2.cloudflarestorage.com"
    assert (settings.region, settings.addressing_style, settings.checksums) == ("auto", "path", "when_required")
    assert (settings.raw_bucket, settings.raw_prefix) == ("glycoshape-archive", "raw")
    assert (settings.process_bucket, settings.process_prefix) == ("glycoshape-archive", "process")
    assert (settings.static_bucket, settings.static_prefix) == ("glycoshape-database", "static")
    assert settings.staging_prefix == "publish-staging" and settings.max_attempts == 8
    assert (settings.multipart_threshold, settings.multipart_chunksize) == (64 << 20, 32 << 20)
    assert settings.routes() == {
        "raw": "glycoshape-archive", "process": "glycoshape-archive", "static": "glycoshape-database"}
    assert "do-not-print-this-secret" not in repr(settings) and "key-id" not in repr(settings)
    assert "do-not-print-this-secret" not in str(settings)

    custom = S3Settings.from_env({
        **ENVIRONMENT,
        "GLYCOSHAPE_S3_REGION": "eu-west-2", "GLYCOSHAPE_S3_RAW_BUCKET": "b1", "GLYCOSHAPE_S3_RAW_PREFIX": "/in/raw/",
        "GLYCOSHAPE_S3_STATIC_BUCKET": "b2", "GLYCOSHAPE_S3_STATIC_PREFIX": "db", "GLYCOSHAPE_S3_MAX_ATTEMPTS": "3",
        "GLYCOSHAPE_S3_MULTIPART_THRESHOLD_MB": "8", "GLYCOSHAPE_S3_MULTIPART_CHUNK_MB": "5",
        "GLYCOSHAPE_S3_ADDRESSING_STYLE": "virtual", "GLYCOSHAPE_S3_CHECKSUMS": "when_supported",
    })
    assert (custom.region, custom.raw_bucket, custom.raw_prefix, custom.static_prefix) == ("eu-west-2", "b1", "in/raw", "db")
    assert (custom.max_attempts, custom.multipart_threshold, custom.multipart_chunksize) == (3, 8 << 20, 5 << 20)
    assert (custom.addressing_style, custom.checksums) == ("virtual", "when_supported")

    with pytest.raises(ValueError) as missing:
        S3Settings.from_env({})
    assert all(name in str(missing.value) for name in ENVIRONMENT)
    for name, value in (("MAX_ATTEMPTS", "many"), ("MAX_ATTEMPTS", "0"), ("MULTIPART_CHUNK_MB", "4"),
                        ("ADDRESSING_STYLE", "dns"), ("CHECKSUMS", "never"), ("STATIC_PREFIX", "/")):
        with pytest.raises(ValueError, match=f"GLYCOSHAPE_S3_{name}"):
            S3Settings.from_env({**ENVIRONMENT, f"GLYCOSHAPE_S3_{name}": value})


def test_the_backend_is_selected_by_configuration(monkeypatch):
    for name in list(ENVIRONMENT) + ["GLYCOSHAPE_STORAGE", "GLYCOSHAPE_ORACLE_PAR_URL"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(storage.OracleStorageBackend, "_verify_par_url", lambda self: None)   # no network
    created = []
    monkeypatch.setattr(storage, "create_s3_backend", lambda: created.append("s3") or S3StorageBackend(FakeS3Client()))

    def selected():
        return config.selected_storage(), type(StorageManager().backend).__name__

    assert selected() == ("local", "LocalStorageBackend")
    monkeypatch.setenv("GLYCOSHAPE_ORACLE_PAR_URL", "https://objectstorage.example/p/token/n/ns/b/bucket/o/")
    assert selected() == ("oracle", "OracleStorageBackend")              # as before: the PAR URL selects Oracle
    for value in ("auto", "oracle", " Oracle "):
        monkeypatch.setenv("GLYCOSHAPE_STORAGE", value)
        assert selected() == ("oracle", "OracleStorageBackend")
    monkeypatch.setenv("GLYCOSHAPE_STORAGE", "local")
    assert selected() == ("local", "LocalStorageBackend")                # explicit local wins over a PAR URL
    monkeypatch.setenv("GLYCOSHAPE_STORAGE", "s3")
    assert selected() == ("s3", "S3StorageBackend") and created == ["s3"]   # and so does S3
    manager = StorageManager()
    assert manager._is_oracle                                            # S3 is remote storage for the manager

    monkeypatch.delenv("GLYCOSHAPE_ORACLE_PAR_URL")
    monkeypatch.setenv("GLYCOSHAPE_STORAGE", "oracle")
    with pytest.raises(ValueError, match="needs GLYCOSHAPE_ORACLE_PAR_URL"):
        config.selected_storage()
    monkeypatch.setenv("GLYCOSHAPE_STORAGE", "gcs")
    with pytest.raises(ValueError, match="Invalid GLYCOSHAPE_STORAGE"):
        StorageManager()


def test_s3_without_settings_fails_clearly(monkeypatch):
    for name in ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GLYCOSHAPE_STORAGE", "s3")
    with pytest.raises(ValueError, match="S3 storage is not configured: set GLYCOSHAPE_S3_ENDPOINT"):
        StorageManager()


def test_the_storage_manager_treats_s3_as_remote_storage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "upload_workers", 4)
    client = FakeS3Client("glycoshape-archive", "glycoshape-database")
    backend = S3StorageBackend(client, routes=S3Settings.from_env(ENVIRONMENT).routes())
    manager = StorageManager(backend)
    backend.write_binary("static/GS00001/data.json", b'{"remote": true}')
    backend.write_binary("raw/DGlcpa1-OH/structure.mol2", b"mol2")
    backend.write_binary("process/DGlcpa1-OH/output/info.json", b"{}")

    assert manager.exists("static/GS00001/data.json") and manager.is_dir("static/GS00001")
    assert manager.is_file("process/DGlcpa1-OH/output/info.json") and not manager.exists("static/GS00002/data.json")
    assert manager.list_files("raw") == [Path("raw/DGlcpa1-OH/structure.mol2")]   # a routed prefix lists the storage
    assert manager.list_files_paginated("static") == [Path("static/GS00001/data.json")]
    assert manager.list_dirs("static") == [Path("static/GS00001")]
    assert manager.read_text("static/GS00001/data.json") == '{"remote": true}'
    with manager.open("static/GS00001/data.json") as handle:
        assert handle.read() == '{"remote": true}'

    # writes are staged locally, and a staged file is read before the stored one
    writes = len(client.writes())
    manager.write_text("static/GS00001/data.json", '{"staged": true}')
    assert len(client.writes()) == writes and (tmp_path / "static/GS00001/data.json").is_file()
    assert manager.read_text("static/GS00001/data.json") == '{"staged": true}'
    assert backend.read_text("static/GS00001/data.json") == '{"remote": true}'

    # upload_dir sends the staged folder as streams, with the SHA-256 of each file
    (tmp_path / "static/GS00001/output").mkdir()
    (tmp_path / "static/GS00001/output/pca.csv").write_bytes(b"x,y\n")
    manager.upload_dir(tmp_path / "static/GS00001", "static/GS00001", skip_existing=False)
    assert backend.read_text("static/GS00001/data.json") == '{"staged": true}'
    assert backend.head("static/GS00001/output/pca.csv")["sha256"] == sha256(b"x,y\n")
    assert client.count("upload_file") == 2 and client.count("put_object") == 3
    manager.upload_dir(tmp_path / "static/GS00001", "static/GS00001", skip_existing=True)
    assert client.count("upload_file") == 2


def test_logging_shows_neither_bodies_nor_keys(fake_s3, caplog, tmp_path):
    backend, client = fake_s3
    caplog.set_level(logging.DEBUG)
    (tmp_path / "file.bin").write_bytes(b"BODY-OF-A-FILE")
    backend.write_binary("static/a.json", b"BODY-OF-AN-OBJECT")
    backend.upload_file(tmp_path / "file.bin", "static/b.bin")
    backend.read_binary("static/a.json")
    backend.unlink("static/a.json")
    client.read_failures = [ConnectionResetError("BODY-IN-AN-ERROR")]
    backend.read_binary("static/b.bin")
    S3StorageBackend(client, routes=S3Settings.from_env(ENVIRONMENT).routes())
    assert caplog.records and "BODY" not in caplog.text and "do-not-print-this-secret" not in caplog.text


# ---------------------------------------------------------------------------
# The real client against a local HTTP recorder
# ---------------------------------------------------------------------------
class _Recorder(http.server.BaseHTTPRequestHandler):
    def _respond(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        self.server.seen.append((self.command, self.path, {k.lower(): v for k, v in self.headers.items()}, body))
        status = self.server.statuses.pop(0) if self.server.statuses else 200
        self.send_response(status)
        if status == 200:
            self.send_header("ETag", '"0123456789abcdef0123456789abcdef"')
            self.send_header("x-amz-meta-sha256", "f" * 64)
        self.send_header("Content-Length", "0")
        self.end_headers()

    do_GET = do_HEAD = do_PUT = do_DELETE = do_POST = _respond

    def log_message(self, *arguments):
        pass


@pytest.fixture
def recorder():
    pytest.importorskip("boto3", reason="needs the real client")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Recorder)
    server.seen, server.statuses = [], []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def _real_backend(server, **overrides):
    settings = S3Settings.from_env({
        **ENVIRONMENT, "GLYCOSHAPE_S3_ENDPOINT": f"http://127.0.0.1:{server.server_address[1]}", **overrides})
    return storage.create_s3_backend(settings, bucket="glycoshape-archive")


def test_requests_are_retried_with_backoff_on_transient_errors(recorder):
    backend = _real_backend(recorder, GLYCOSHAPE_S3_MAX_ATTEMPTS="3")
    recorder.statuses[:] = [503, 500]
    assert backend.head("raw/DGlcpa1-OH/trajectory.xtc")["sha256"] == "f" * 64      # the third attempt succeeds
    assert [request[0] for request in recorder.seen] == ["HEAD"] * 3

    # 429 is what R2 answers when it limits the rate; the client alone would not retry it
    recorder.seen.clear()
    recorder.statuses[:] = [429]
    started = time.monotonic()
    backend.write_binary("raw/DGlcpa1-OH/name.json", b"{}")
    assert [request[0] for request in recorder.seen] == ["PUT"] * 2 and time.monotonic() - started >= 0.5
    recorder.seen.clear()
    recorder.statuses[:] = [429] * 10
    with pytest.raises(Exception, match="429"):
        backend.head("raw/DGlcpa1-OH/trajectory.xtc")
    assert len(recorder.seen) == 3
    recorder.statuses.clear()

    recorder.seen.clear()
    recorder.statuses[:] = [503] * 10
    with pytest.raises(Exception) as failure:                                      # attempts are bounded
        backend.head("raw/DGlcpa1-OH/trajectory.xtc")
    assert len(recorder.seen) == 3 and "do-not-print-this-secret" not in str(failure.value)
    recorder.statuses.clear()

    recorder.seen.clear()
    recorder.statuses[:] = [403]                                                    # not transient: one request
    with pytest.raises(Exception):
        backend.head("raw/DGlcpa1-OH/trajectory.xtc")
    assert len(recorder.seen) == 1


def test_requests_are_path_style_signed_and_logged_without_secrets(recorder, caplog):
    logging.getLogger("botocore").setLevel(logging.NOTSET)
    caplog.set_level(logging.DEBUG)                 # a caller that turns on debug logging for everything
    backend = _real_backend(recorder)
    key = f"raw/{LONG_NAME}/manifest.json"
    backend.write_binary(key, b"BODY-OF-AN-OBJECT")
    backend.write_binary(f"raw/{COMMA_NAME}/name.json", b"{}")
    method, path, headers, body = recorder.seen[0]

    assert (method, body) == ("PUT", b"BODY-OF-AN-OBJECT")
    # path-style: the bucket is the first path segment; brackets and commas arrive percent-encoded
    assert path == "/glycoshape-archive/raw/" + LONG_NAME.replace("[", "%5B").replace("]", "%5D") + "/manifest.json"
    assert recorder.seen[1][1] == "/glycoshape-archive/raw/" + (
        COMMA_NAME.replace("[", "%5B").replace("]", "%5D").replace(",", "%2C")) + "/name.json"
    assert headers["authorization"].startswith("AWS4-HMAC-SHA256 Credential=key-id/")
    assert "/auto/s3/aws4_request" in headers["authorization"]                     # region "auto", as R2 expects
    assert headers["x-amz-meta-sha256"] == sha256(b"BODY-OF-AN-OBJECT")
    assert not any(name.startswith("x-amz-checksum") or name == "x-amz-sdk-checksum-algorithm" for name in headers)
    assert headers.get("content-encoding") != "aws-chunked" and headers["content-length"] == "17"

    assert logging.getLogger("botocore").getEffectiveLevel() >= logging.INFO
    assert "do-not-print-this-secret" not in caplog.text and "BODY-OF-AN-OBJECT" not in caplog.text
