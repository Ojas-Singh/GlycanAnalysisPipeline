"""Fixtures for the S3 storage tests.

Unit tests use ``fake_s3`` (an in-memory client, no boto3 needed).

Integration tests use ``s3_server``, a real S3-compatible server on this
machine, and are skipped when there is none.  Every integration test runs
once for each server that can be started here:

* ``minio`` - a MinIO container, started for the test session and removed at
  its end, when Docker runs and a MinIO image is already present (nothing is
  pulled; ``GLYCOSHAPE_TEST_S3_IMAGE`` names another image).  MinIO keeps
  objects in a file system and refuses a key with a segment of more than 255
  bytes, so the tests of the 267-character GLYCAM name are skipped on it.
* ``moto`` - the moto server, in this process, when moto and flask are
  installed (``uv pip install "moto[s3]" flask flask-cors``).  It has no such
  limit, so the long name is tested here.

With ``GLYCOSHAPE_TEST_S3_ENDPOINT``, ``GLYCOSHAPE_TEST_S3_ACCESS_KEY_ID`` and
``GLYCOSHAPE_TEST_S3_SECRET_ACCESS_KEY`` that server is used instead of both.
It must be a throwaway server: the tests create and delete buckets named
``gap-test-...`` on it.  All keys of servers started here are random.
"""
from __future__ import annotations

import os
import secrets
import shutil
import socket
import subprocess
import time
import urllib.request

import pytest

from lib.storage import S3Settings, S3StorageBackend, create_s3_client
from s3_fakes import COMMA_NAME, LONG_NAME, FakeS3Client

MINIO_IMAGES = ("minio/minio:latest", "quay.io/minio/minio:latest", "ghcr.io/coollabsio/minio:latest")
SERVER_KINDS = ["endpoint"] if os.environ.get("GLYCOSHAPE_TEST_S3_ENDPOINT", "").strip() else ["minio", "moto"]


@pytest.fixture
def fake_s3():
    """``(backend, client)``: the storage backend on an in-memory client with one bucket."""
    client = FakeS3Client("bucket")
    return S3StorageBackend(client, bucket="bucket", sleep=lambda seconds: None), client


@pytest.fixture(scope="session")
def built_archive(tmp_path_factory):
    """A converted archive of three entries: a short name, one with commas, the 267-character one."""
    from lib import archive
    from test_archive import NAME_G60371DN, make_source, random_coords   # after collection, so it is rewritten

    root = tmp_path_factory.mktemp("archive-source")
    make_source(root / "data", "DGlcpa1-OH", random_coords(12, 12, seed=1), info_json="DGlcpa1-OH")
    make_source(root / "data", COMMA_NAME, random_coords(9, 14, seed=2))
    make_source(root / "data", "G60371DN", random_coords(7, 16, seed=3), info_json=NAME_G60371DN)
    for folder in ("DGlcpa1-OH", COMMA_NAME, "G60371DN"):
        archive.convert_entry(root / "data" / folder, root / "archive")
    index, problems = archive.write_index(root / "archive")
    assert not problems and NAME_G60371DN == LONG_NAME
    assert [record["glycam_name"] for record in index["records"]] == sorted(["DGlcpa1-OH", COMMA_NAME, LONG_NAME])
    return root / "archive"


@pytest.fixture
def archive_copy(built_archive, tmp_path):
    """A private copy of ``built_archive`` that a test may change."""
    shutil.copytree(built_archive, tmp_path / "archive")
    return tmp_path / "archive"


# ---------------------------------------------------------------------------
# Local S3-compatible servers
# ---------------------------------------------------------------------------
def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _docker(*arguments: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *arguments], capture_output=True, text=True, timeout=timeout, check=False)


def _start_minio() -> dict:
    if shutil.which("docker") is None:
        pytest.skip("no MinIO: Docker is not installed")
    wanted = os.environ.get("GLYCOSHAPE_TEST_S3_IMAGE")
    try:
        image = next(
            (name for name in ([wanted] if wanted else MINIO_IMAGES) if _docker("image", "inspect", name).returncode == 0),
            None,
        )
    except (OSError, subprocess.TimeoutExpired):
        image = None
    if image is None:
        pytest.skip(f"no MinIO: no image is present (docker pull one of {', '.join(MINIO_IMAGES)})")
    port = _free_port()
    server = {
        "endpoint": f"http://127.0.0.1:{port}",
        "access_key": "gaptest" + secrets.token_hex(4),
        "secret_key": secrets.token_hex(20),
        "container": f"gap-s3-test-{os.getpid()}-{secrets.token_hex(3)}",
    }
    started = _docker(
        "run", "-d", "--rm", "--name", server["container"], "-p", f"127.0.0.1:{port}:9000",
        "-e", f"MINIO_ROOT_USER={server['access_key']}", "-e", f"MINIO_ROOT_PASSWORD={server['secret_key']}",
        image, "server", "/data",
    )
    if started.returncode != 0:
        pytest.skip(f"no MinIO: could not start {image}: {started.stderr.strip()[:200]}")
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(server["endpoint"] + "/minio/health/live", timeout=2) as response:
                if response.status == 200:
                    return server
        except OSError:
            time.sleep(0.3)
    _docker("rm", "-f", server["container"])
    pytest.skip(f"no MinIO: {image} did not become ready")


def _start_moto() -> dict:
    try:
        from moto.server import ThreadedMotoServer
    except ImportError:
        pytest.skip('no moto server: moto and flask are not installed (uv pip install "moto[s3]" flask flask-cors)')
    port = _free_port()
    process = ThreadedMotoServer(ip_address="127.0.0.1", port=port, verbose=False)
    process.start()
    return {
        "endpoint": f"http://127.0.0.1:{port}",
        "access_key": "gaptest" + secrets.token_hex(4),
        "secret_key": secrets.token_hex(20),
        "region": "us-east-1",   # moto creates a bucket without a location constraint only there
        "process": process,
    }


def _accepts_long_segments(server: dict) -> bool:
    """Whether the server stores a key with a segment of more than 255 bytes (S3 and R2 do, MinIO does not)."""
    settings = S3Settings(
        server["endpoint"], server["access_key"], server["secret_key"], region=server["region"], max_attempts=2)
    client, _ = create_s3_client(settings)
    bucket = "gap-test-" + secrets.token_hex(6)
    client.create_bucket(Bucket=bucket)
    try:
        client.put_object(Bucket=bucket, Key=f"raw/{'D' * 300}/probe", Body=b"")
        client.delete_object(Bucket=bucket, Key=f"raw/{'D' * 300}/probe")
        return True
    except Exception:
        return False
    finally:
        client.delete_bucket(Bucket=bucket)


@pytest.fixture(scope="session", params=SERVER_KINDS)
def s3_server(request):
    """Endpoint, keys and ``long_segments`` of a local S3-compatible server; skips when it cannot be had."""
    pytest.importorskip("boto3", reason="the S3 integration tests need boto3")
    if request.param == "endpoint":
        server = {
            "endpoint": os.environ["GLYCOSHAPE_TEST_S3_ENDPOINT"].strip(),
            "access_key": os.environ.get("GLYCOSHAPE_TEST_S3_ACCESS_KEY_ID", ""),
            "secret_key": os.environ.get("GLYCOSHAPE_TEST_S3_SECRET_ACCESS_KEY", ""),
        }
    else:
        server = _start_minio() if request.param == "minio" else _start_moto()
    try:
        server.setdefault("region", os.environ.get("GLYCOSHAPE_TEST_S3_REGION", "auto"))
        server.update(kind=request.param, long_segments=_accepts_long_segments(server))
        yield server
    finally:
        if "container" in server:
            _docker("rm", "-f", server["container"])
        if "process" in server:
            server["process"].stop()


@pytest.fixture
def s3_settings(s3_server):
    """Settings for the test server; multipart starts at 5 MiB so that small files exercise it."""
    return S3Settings(
        endpoint_url=s3_server["endpoint"],
        access_key_id=s3_server["access_key"],
        secret_access_key=s3_server["secret_key"],
        region=s3_server["region"],
        max_attempts=3,
        multipart_threshold=5 << 20,
        multipart_chunksize=5 << 20,
    )


@pytest.fixture
def s3_bucket(s3_settings):
    """``(backend, client, bucket)`` on the test server with a new bucket, emptied and removed afterwards."""
    client, transfer_config = create_s3_client(s3_settings)
    bucket = "gap-test-" + secrets.token_hex(6)
    client.create_bucket(Bucket=bucket)
    try:
        yield S3StorageBackend(
            client, bucket=bucket, transfer_config=transfer_config, copy_part_size=5 << 20), client, bucket
    finally:
        backend = S3StorageBackend(client, bucket=bucket)
        for item in list(backend.list_objects("")):
            client.delete_object(Bucket=bucket, Key=item["key"])
        for upload in client.list_multipart_uploads(Bucket=bucket).get("Uploads") or []:
            client.abort_multipart_upload(Bucket=bucket, Key=upload["Key"], UploadId=upload["UploadId"])
        client.delete_bucket(Bucket=bucket)


# ---------------------------------------------------------------------------
# Publishing one entry
# ---------------------------------------------------------------------------
_LOOPBACK = ("127.0.0.1", "::1", "localhost")


@pytest.fixture
def no_internet(monkeypatch):
    """Any attempt to reach another machine fails the test; the S3 servers of this machine stay reachable."""
    connect, resolve = socket.socket.connect, socket.getaddrinfo

    def guarded_connect(self, address, *args, **kwargs):
        if self.family in (socket.AF_INET, socket.AF_INET6) and address[0] not in _LOOPBACK:
            raise AssertionError(f"network access attempted: {address}")
        return connect(self, address, *args, **kwargs)

    def guarded_resolve(host, *args, **kwargs):
        if host not in _LOOPBACK:
            raise AssertionError(f"name lookup attempted: {host}")
        return resolve(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_resolve)


@pytest.fixture(scope="session")
def synthetic_entry(tmp_path_factory):
    """Source, archive entry and result folder of the synthetic glycan: one run of the process stage per session."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    from bundle_fixtures import synthetic_run

    return synthetic_run(tmp_path_factory.mktemp("synthetic-entry"), info={"simulationLength": 480.0})


@pytest.fixture
def bundle_dir(synthetic_entry, tmp_path):
    """A bundle directory of the synthetic glycan that a test may change."""
    from bundle_fixtures import make_bundle

    return make_bundle(tmp_path / "bundle", synthetic_entry["result"], synthetic_entry["archive"])


@pytest.fixture
def long_name(s3_server):
    """The 267-character GLYCAM name, for a server that can store it as a key segment."""
    if not s3_server["long_segments"]:
        pytest.skip(f"the {s3_server['kind']} server refuses a key segment of more than 255 bytes")
    return LONG_NAME
