"""Integration tests of the S3 storage against a real S3-compatible server on this machine.

Skipped when there is no server; run once per server otherwise (see
tests/conftest.py).  They cover what an in-memory client cannot: multipart
upload, ranged download and multipart copy by the real client, listings of
more than one page, odd keys on the wire, and the command-line tools end to
end.  The 267-character GLYCAM name is part of every test on a server that
can store it (``long_segments``), and left out on MinIO, which cannot.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from lib import archive, archive_transfer, static_publish
from lib.storage import S3StorageBackend, StorageManager
from s3_fakes import COMMA_NAME, LONG_NAME, reader_sync, sample_glycans, tree, write_static

REPO_ROOT = Path(__file__).resolve().parents[1]
MIB = 1 << 20
ODD_NAME = "name with space+plus%25&amp;'quote' ünïcode"


def quiet(message):
    pass


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def pseudo_random(size: int, seed: int) -> bytes:
    return np.random.default_rng(seed).integers(0, 256, size=size, dtype=np.uint8).tobytes()


@pytest.fixture
def names(s3_server):
    """GLYCAM names of the entries the server can store: all three unless it limits key segments."""
    return ["DGlcpa1-OH", COMMA_NAME] + ([LONG_NAME] if s3_server["long_segments"] else [])


@pytest.fixture
def local_archive(archive_copy, names):
    """The archive of conftest.py, without the entry of the long name where the server cannot store it."""
    if LONG_NAME not in names:
        shutil.rmtree(archive_copy / archive.entry_folder_name(LONG_NAME))
        assert archive.write_index(archive_copy)[1] == []
    return archive_copy


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------
def test_large_files_are_transferred_in_parts(s3_bucket, names, tmp_path):
    backend, client, bucket = s3_bucket
    payload = pseudo_random(12 * MIB + 123, seed=1)                # three parts of 5 MiB (the test threshold)
    local = tmp_path / "trajectory.xtc"
    local.write_bytes(payload)
    key = f"raw/{names[-1]}/trajectory.xtc"

    backend.upload_file(local, key)
    head = backend.head(key)
    assert head["size"] == len(payload) and head["sha256"] == sha256(payload)
    assert head["etag"].endswith("-3")                             # a multipart upload of three parts
    assert client.list_multipart_uploads(Bucket=bucket).get("Uploads") in (None, [])   # completed, nothing left open

    backend.download_file(key, tmp_path / "back.xtc")
    assert (tmp_path / "back.xtc").read_bytes() == payload
    assert [path.name for path in tmp_path.iterdir() if path.name.startswith("back")] == ["back.xtc"]
    assert backend.stream_sha256(key) == (len(payload), sha256(payload))
    assert backend.read_binary(key) == payload

    backend.copy(key, "static/copied.bin")                         # a multipart copy inside the storage
    head = backend.head("static/copied.bin")
    assert (head["size"], head["sha256"]) == (len(payload), sha256(payload)) and head["etag"].endswith("-3")
    assert backend.stream_sha256("static/copied.bin")[1] == sha256(payload)
    assert client.list_multipart_uploads(Bucket=bucket).get("Uploads") in (None, [])

    small = tmp_path / "small.json"
    small.write_bytes(b"{}")
    backend.upload_file(small, "static/small.json")                # below the threshold: one request
    assert backend.head("static/small.json")["etag"] == hashlib.md5(b"{}").hexdigest()
    backend.copy("static/small.json", "static/small-copy.json", sha256="0" * 64)
    assert backend.head("static/small-copy.json")["sha256"] == "0" * 64   # metadata is replaced on a copy
    assert backend.read_binary("static/small-copy.json") == b"{}"
    (tmp_path / "empty").write_bytes(b"")
    backend.upload_file(tmp_path / "empty", "static/empty")
    assert backend.head("static/empty")["size"] == 0 and backend.read_binary("static/empty") == b""
    with pytest.raises(FileNotFoundError):
        backend.download_file("static/missing", tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        backend.read_binary("static/missing")
    with pytest.raises(FileNotFoundError):
        backend.copy("static/missing", "static/other")
    assert not (tmp_path / "missing").exists() and backend.head("static/missing") is None


def test_a_listing_longer_than_one_page_is_complete(s3_bucket):
    backend, client, bucket = s3_bucket
    keys = sorted(f"raw/G{entry:04d}/{name}" for entry in range(335) for name in ("a.json", "b.xtc", "c.pdb"))
    assert len(keys) == 1005                                        # more than the 1000 keys of one page
    with ThreadPoolExecutor(max_workers=32) as pool:
        list(pool.map(lambda key: backend.write_binary(key, key.encode()), keys))

    listed = list(backend.list_objects("raw/"))
    assert [item["key"] for item in listed] == keys
    assert all(item["size"] == len(item["key"]) for item in listed)
    small_pages = S3StorageBackend(client, bucket=bucket, page_size=100)
    assert [item["key"] for item in small_pages.list_objects("raw/")] == keys
    assert small_pages.list_common_prefixes("raw") == [f"raw/G{entry:04d}" for entry in range(335)]
    assert backend.list_files("raw", "G033*") == [Path(key) for key in keys if key.startswith("raw/G033")]
    assert backend.is_dir("raw/G0334") and not backend.is_dir("raw/G0335") and backend.exists("raw")


def _key_round_trip(backend, name):
    key = f"raw/{name}/manifest.json"
    backend.write_binary(key, b'{"glycam_name": 1}')
    assert backend.exists(key) and backend.is_file(key) and backend.is_dir(f"raw/{name}")
    assert backend.read_binary(key) == b'{"glycam_name": 1}'
    assert [item["key"] for item in backend.list_objects("raw/")] == [key]     # the listing returns the key itself
    assert backend.list_common_prefixes("raw") == [f"raw/{name}"]
    assert backend.head(key)["sha256"] == sha256(b'{"glycam_name": 1}')
    backend.copy(key, f"process/{name}/manifest.json")
    assert backend.read_binary(f"process/{name}/manifest.json") == b'{"glycam_name": 1}'
    backend.unlink(key)
    assert not backend.exists(key)


@pytest.mark.parametrize("name", [COMMA_NAME, ODD_NAME])
def test_odd_keys_survive_the_wire(s3_bucket, name):
    _key_round_trip(s3_bucket[0], name)


def test_the_267_character_name_is_a_key(s3_bucket, long_name):
    assert len(long_name) == 267
    _key_round_trip(s3_bucket[0], long_name)


def test_the_storage_manager_is_selected_from_the_environment(s3_bucket, s3_server, monkeypatch, tmp_path):
    backend, client, bucket = s3_bucket
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GLYCOSHAPE_STORAGE", "s3")
    monkeypatch.setenv("GLYCOSHAPE_S3_ENDPOINT", s3_server["endpoint"])
    monkeypatch.setenv("GLYCOSHAPE_S3_ACCESS_KEY_ID", s3_server["access_key"])
    monkeypatch.setenv("GLYCOSHAPE_S3_SECRET_ACCESS_KEY", s3_server["secret_key"])
    monkeypatch.setenv("GLYCOSHAPE_S3_REGION", s3_server["region"])
    for kind in ("RAW", "PROCESS", "STATIC"):
        monkeypatch.setenv(f"GLYCOSHAPE_S3_{kind}_BUCKET", bucket)
    manager = StorageManager()
    assert isinstance(manager.backend, S3StorageBackend)

    (tmp_path / "out" / "output").mkdir(parents=True)
    (tmp_path / "out" / "data.json").write_bytes(b'{"archetype": {}}')
    (tmp_path / "out" / "output" / "pca.csv").write_bytes(b"x,y\n")
    manager.upload_dir(tmp_path / "out", "static/GS00001", skip_existing=False)
    assert manager.exists("static/GS00001/data.json") and manager.is_dir("static/GS00001")
    assert manager.list_dirs("static") == [Path("static/GS00001")]
    assert manager.list_files_paginated("static") == [
        Path("static/GS00001/data.json"), Path("static/GS00001/output/pca.csv")]
    assert manager.read_text("static/GS00001/output/pca.csv") == "x,y\n"
    assert backend.head("static/GS00001/output/pca.csv")["sha256"] == sha256(b"x,y\n")


# ---------------------------------------------------------------------------
# Archive transfer
# ---------------------------------------------------------------------------
def test_the_archive_goes_up_and_comes_back_verified(s3_bucket, local_archive, names, tmp_path):
    backend, client, bucket = s3_bucket
    count = len(names)
    report = archive_transfer.upload_archive(local_archive, backend, workers=3, log=quiet)
    assert report["ok"] and report["counts"] == {"uploaded": count} and report["index_uploaded"]
    keys = [item["key"] for item in backend.list_objects("raw/")]
    assert all(f"raw/{name}/trajectory.xtc" in keys and f"raw/{name}/manifest.json" in keys for name in names)
    assert "raw/index.json" in keys and len(keys) == report["comparison"]["objects"] + 1
    for record in json.loads((local_archive / "index.json").read_text())["records"]:
        for name, recorded in record["files"].items():
            head = backend.head(f"raw/{record['glycam_name']}/{name}")
            assert (head["size"], head["sha256"]) == (recorded["size"], recorded["sha256"])

    again = archive_transfer.upload_archive(local_archive, backend, log=quiet)      # resumable: nothing to send
    assert again["ok"] and again["counts"] == {"skipped": count} and not again["index_uploaded"]

    # the whole archive comes back identical; one entry by its GLYCAM name gives a folder the pipeline reads
    mirror = tmp_path / "mirror"
    report = archive_transfer.download_archive(backend, mirror, workers=3, log=quiet)
    assert report["ok"] and report["counts"] == {"downloaded": count} and report["index_written"]
    assert tree(mirror) == tree(local_archive)
    data = tmp_path / "data"
    report = archive_transfer.download_archive(backend, data, glycans=[names[-1]], log=quiet)
    folder = data / archive.entry_folder_name(names[-1])
    assert report["ok"] and archive.verify_entry(folder)["ok"]
    assert archive.entry_inputs(folder)["glycam_name"] == names[-1]

    # content that is not the content of the manifest is refused, whatever its metadata says
    key = f"raw/{COMMA_NAME}/trajectory.xtc"
    original = backend.read_binary(key)
    client.put_object(Bucket=bucket, Key=key, Body=original[:-1] + bytes([original[-1] ^ 1]),
                      Metadata={"sha256": sha256(original)})
    report = archive_transfer.download_archive(backend, tmp_path / "other", glycans=[COMMA_NAME], log=quiet)
    assert not report["ok"] and "does not match the size and SHA-256 of the manifest" in report["failed"][0]["error"]
    assert not (tmp_path / "other" / COMMA_NAME).exists()


def test_an_upload_that_breaks_is_continued(s3_bucket, local_archive, names, monkeypatch):
    backend, client, bucket = s3_bucket
    send = backend.upload_file

    def breaks(local_path, key, sha256=None):
        if key == f"raw/{names[-1]}/topology.pdb":
            raise ConnectionError("connection lost")
        return send(local_path, key, sha256=sha256)

    monkeypatch.setattr(backend, "upload_file", breaks)
    report = archive_transfer.upload_archive(local_archive, backend, workers=2, log=quiet)
    assert not report["ok"] and report["counts"] == {"failed": 1, "uploaded": len(names) - 1}
    assert report["comparison"]["incomplete"] == [names[-1]] and not report["index_uploaded"]
    assert not backend.exists("raw/index.json") and not backend.exists(f"raw/{names[-1]}/manifest.json")

    monkeypatch.setattr(backend, "upload_file", send)
    report = archive_transfer.upload_archive(local_archive, backend, workers=2, log=quiet)
    assert report["ok"] and report["counts"] == {"skipped": len(names) - 1, "uploaded": 1} and report["index_uploaded"]
    assert archive_transfer.compare_with_index(backend, "raw", archive_transfer.load_index(local_archive))["ok"]


# ---------------------------------------------------------------------------
# Publish
# ---------------------------------------------------------------------------
ROOT_FILES = {"GLYCOSHAPE.json": b'{"version": 1}', "GlycoShape.zip": pseudo_random(6 * MIB, seed=2)}


def test_publish_check_and_the_reader_on_a_real_server(s3_bucket, tmp_path):
    backend, client, bucket = s3_bucket
    static = tmp_path / "static"
    write_static(static, sample_glycans(3), ROOT_FILES)
    report = static_publish.publish(static, backend, workers=8, log=quiet)
    assert (report["new"], report["changed"], report["manifest_written"]) == (14, 0, True)
    assert static_publish.check_published(backend, log=quiet)["ok"]
    assert static_publish.check_published(backend, quick=True, log=quiet)["ok"]
    site = tmp_path / "site"
    assert reader_sync(backend, "static", site)["downloaded"] == 14 and tree(site) == tree(static)

    # second version: a glycan changed, one added, both root files changed (the archive of 6 MiB in parts)
    (static / "GS00002" / "data.json").write_bytes(b'{"archetype": {"ID": "GS00002", "revised": true}}')
    write_static(static, {"GS00004": sample_glycans(4)["GS00004"]}, {
        "GLYCOSHAPE.json": b'{"version": 2}', "GlycoShape.zip": pseudo_random(6 * MIB, seed=3)})
    plan = static_publish.publish(static, backend, dry_run=True, log=quiet)
    assert (plan["new"], plan["changed"], plan["unchanged"]) == (4, 3, 11)
    assert static_publish.check_published(backend, quick=True, log=quiet)["ok"]     # a dry run changed nothing

    report = static_publish.publish(static, backend, workers=8, log=quiet)
    assert (report["new"], report["changed"], report["manifest_written"]) == (4, 3, True)
    assert list(backend.list_objects("publish-staging/")) == []
    head = backend.head("static/GlycoShape.zip")                                    # copied inside the storage, in parts
    assert head["sha256"] == sha256((static / "GlycoShape.zip").read_bytes()) and head["etag"].endswith("-2")
    checked = static_publish.check_published(backend, log=quiet)
    assert checked["ok"] and checked["files"] == 18 and checked["unlisted"] == []
    assert reader_sync(backend, "static", site) == {"downloaded": 2 + 4 + 4, "skipped": 8, "pruned": 0}
    assert tree(site) == tree(static)
    assert static_publish.publish(static, backend, log=quiet)["manifest_written"] is False   # nothing left to do

    # an object that is not what the manifest says is found by the check
    client.put_object(Bucket=bucket, Key="static/GS00001/snfg.svg", Body=b"<svh/>")
    report = static_publish.check_published(backend, log=quiet)
    assert not report["ok"] and report["errors"] == ["download verification failed for GS00001/snfg.svg"]


def test_a_publish_that_breaks_leaves_the_first_version_published(s3_bucket, tmp_path, monkeypatch):
    backend, client, bucket = s3_bucket
    static = tmp_path / "static"
    write_static(static, sample_glycans(3), ROOT_FILES)
    static_publish.publish(static, backend, log=quiet)
    first = tree(static)
    (static / "GS00002" / "data.json").write_bytes(b'{"archetype": {"ID": "GS00002", "revised": true}}')
    write_static(static, {"GS00004": sample_glycans(4)["GS00004"]}, {**ROOT_FILES, "GLYCOSHAPE.json": b'{"version": 2}'})

    send, sent = backend.upload_file, []

    def breaks(local_path, key, sha256=None):
        sent.append(key)
        if len(sent) == 5:
            raise ConnectionError("connection lost")
        return send(local_path, key, sha256=sha256)

    monkeypatch.setattr(backend, "upload_file", breaks)
    with pytest.raises(static_publish.PublishError, match="the published prefix has not changed for readers"):
        static_publish.publish(static, backend, workers=1, log=quiet)
    monkeypatch.setattr(backend, "upload_file", send)

    assert static_publish.check_published(backend, log=quiet)["ok"]
    site = tmp_path / "site"
    reader_sync(backend, "static", site)
    assert tree(site) == first                                                      # a new reader gets the first version
    assert static_publish.publish(static, backend, log=quiet)["manifest_written"]
    reader_sync(backend, "static", site)
    assert tree(site) == tree(static) and static_publish.check_published(backend, log=quiet)["ok"]


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def run_tool(script, *arguments, env):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), *map(str, arguments)],
        capture_output=True, text=True, check=False, env=env, cwd=REPO_ROOT)


@pytest.fixture
def tool_env(s3_bucket, s3_server):
    env = {name: value for name, value in os.environ.items() if not name.startswith("GLYCOSHAPE_")}
    env.update(
        GLYCOSHAPE_S3_ENDPOINT=s3_server["endpoint"],
        GLYCOSHAPE_S3_ACCESS_KEY_ID=s3_server["access_key"],
        GLYCOSHAPE_S3_SECRET_ACCESS_KEY=s3_server["secret_key"],
        GLYCOSHAPE_S3_REGION=s3_server["region"],
        GLYCOSHAPE_S3_RAW_BUCKET=s3_bucket[2],
        GLYCOSHAPE_S3_STATIC_BUCKET=s3_bucket[2],
        GLYCOSHAPE_S3_MULTIPART_THRESHOLD_MB="5",
        GLYCOSHAPE_S3_MULTIPART_CHUNK_MB="5",
        GLYCOSHAPE_S3_MAX_ATTEMPTS="2",
    )
    return env


def test_the_archive_tool(s3_bucket, local_archive, names, tool_env, tmp_path, s3_server):
    backend, client, bucket = s3_bucket
    count = len(names)
    run = run_tool("archive_s3.py", "upload", "--archive-dir", local_archive, "--dry-run", env=tool_env)
    assert run.returncode == 0 and f"would upload={count} skipped=0 failed=0" in run.stdout, run.stdout + run.stderr
    assert list(backend.list_objects("")) == []

    run = run_tool("archive_s3.py", "upload", "--archive-dir", local_archive, "--workers", 3,
                   "--report", tmp_path / "upload.json", env=tool_env)
    assert run.returncode == 0, run.stdout + run.stderr
    assert f"uploaded={count} skipped=0 failed=0" in run.stdout
    assert "missing=0 other-size=0 other-manifest=0 not-in-index=0" in run.stdout
    assert json.loads((tmp_path / "upload.json").read_text())["index_uploaded"] is True
    assert backend.exists(f"raw/{names[-1]}/manifest.json") and backend.exists("raw/index.json")
    run = run_tool("archive_s3.py", "upload", "--archive-dir", local_archive, env=tool_env)
    assert run.returncode == 0 and f"uploaded=0 skipped={count} failed=0" in run.stdout

    data = tmp_path / "data"
    run = run_tool("archive_s3.py", "download", "--data-dir", data, "--glycan", names[-1], env=tool_env)
    folder = data / archive.entry_folder_name(names[-1])
    assert run.returncode == 0 and f"entry folder: {folder}" in run.stdout, run.stdout + run.stderr
    assert archive.verify_entry(folder)["ok"]
    run = run_tool("archive_s3.py", "download", "--archive-dir", data, "--workers", 2, env=tool_env)
    assert run.returncode == 0 and f"downloaded={count - 1} would-download=0 skipped=1 failed=0" in run.stdout
    assert tree(data) == tree(local_archive)

    backend.unlink("raw/DGlcpa1-OH/structure.mol2")
    backend.write_binary("raw/DGalpa1-OH/manifest.json", b"{}")
    run = run_tool("archive_s3.py", "compare", "--archive-dir", local_archive, env=tool_env)
    assert run.returncode == 1 and "missing=1 other-size=0 other-manifest=0 not-in-index=1" in run.stdout
    assert "MISSING raw/DGlcpa1-OH/structure.mol2" in run.stderr and "NOT IN INDEX raw/DGalpa1-OH/manifest.json" in run.stderr
    run = run_tool("archive_s3.py", "download", "--archive-dir", tmp_path / "x", "--glycan", "DGalpb1-OH", env=tool_env)
    assert run.returncode == 1 and "FAILED DGalpb1-OH" in run.stderr

    # configuration errors name the variables; the secret is never printed
    run = run_tool("archive_s3.py", "upload", "--archive-dir", local_archive,
                   env={k: v for k, v in tool_env.items() if k != "GLYCOSHAPE_S3_SECRET_ACCESS_KEY"})
    assert run.returncode != 0 and "set GLYCOSHAPE_S3_SECRET_ACCESS_KEY" in run.stderr and "Traceback" not in run.stderr
    assert s3_server["secret_key"] not in run.stdout + run.stderr


def test_a_wrong_secret_fails_without_being_printed(s3_bucket, local_archive, tool_env, s3_server):
    if s3_server["kind"] == "moto":
        pytest.skip("the moto server does not check signatures")
    run = run_tool("archive_s3.py", "upload", "--archive-dir", local_archive,
                   env={**tool_env, "GLYCOSHAPE_S3_SECRET_ACCESS_KEY": "wrong-secret-0123456789"})
    assert run.returncode != 0 and "Traceback" not in run.stderr and "SignatureDoesNotMatch" in run.stderr
    assert "wrong-secret" not in run.stdout + run.stderr and s3_server["secret_key"] not in run.stdout + run.stderr
    assert list(s3_bucket[0].list_objects("")) == []


def test_the_static_tool(s3_bucket, tool_env, tmp_path, s3_server):
    backend, client, bucket = s3_bucket
    static = tmp_path / "static"
    write_static(static, sample_glycans(3), ROOT_FILES)
    run = run_tool("static_s3.py", "check", env=tool_env)
    assert run.returncode == 1 and "static/sync_manifest.json does not exist" in run.stderr and "NOT VALID" in run.stdout

    run = run_tool("static_s3.py", "publish", "--static-dir", static, "--dry-run", env=tool_env)
    assert run.returncode == 0 and "14 new, 0 changed, 0 unchanged" in run.stdout and "nothing was sent" in run.stdout
    assert list(backend.list_objects("")) == []
    run = run_tool("static_s3.py", "publish", "--static-dir", static, "--workers", 8, env=tool_env)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "S3 static publish complete: new=14 changed=0 unchanged=0 manifest_written=True remote=static/" in run.stdout

    run = run_tool("static_s3.py", "check", "--report", tmp_path / "check.json", env=tool_env)
    assert run.returncode == 0 and "static/: valid; 14 files" in run.stdout and "errors=0" in run.stdout
    assert json.loads((tmp_path / "check.json").read_text())["ok"] is True
    assert run_tool("static_s3.py", "check", "--quick", env=tool_env).returncode == 0
    reader_sync(backend, "static", tmp_path / "site")
    assert tree(tmp_path / "site") == tree(static)

    # a second version through the tool: staged, copied inside the storage, manifest last
    write_static(static, {}, {**ROOT_FILES, "GLYCOSHAPE.json": b'{"version": 2}'})
    run = run_tool("static_s3.py", "publish", "--static-dir", static, env=tool_env)
    assert run.returncode == 0 and "new=0 changed=1 unchanged=13 manifest_written=True" in run.stdout, run.stderr
    assert backend.read_binary("static/GLYCOSHAPE.json") == b'{"version": 2}'
    assert list(backend.list_objects("publish-staging/")) == []

    (static / "GS00003" / "snfg.svg").write_bytes(b"<svg>edited after the manifest</svg>")
    run = run_tool("static_s3.py", "publish", "--static-dir", static, env=tool_env)
    assert run.returncode != 0 and "PublishError" in run.stderr and "rebuild the static output" in run.stderr
    assert "Traceback" not in run.stderr and run_tool("static_s3.py", "check", "--quick", env=tool_env).returncode == 0

    client.delete_object(Bucket=bucket, Key="static/GS00001/data.json")
    run = run_tool("static_s3.py", "check", env=tool_env)
    assert run.returncode == 1 and "ERROR GS00001/data.json is missing" in run.stderr
    assert s3_server["secret_key"] not in run.stdout + run.stderr
