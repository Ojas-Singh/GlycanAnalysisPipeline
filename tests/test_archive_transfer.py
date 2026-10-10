"""Tests of the archive transfer (lib/archive_transfer.py) on an in-memory S3 client."""
from __future__ import annotations

import hashlib
import json
import shutil

import numpy as np
import pytest

from lib import archive, archive_transfer
from lib.archive_transfer import TransferError
from s3_fakes import COMMA_NAME, Interrupted
from test_archive import NAME_G60371DN, make_source, random_coords, snapshot

NAMES = ("DGlcpa1-OH", COMMA_NAME, NAME_G60371DN)


@pytest.fixture
def local(archive_copy):
    """A private copy of the three-entry archive of conftest.py that a test may change."""
    return archive_copy


def quiet(message):
    pass


def upload(local_dir, backend, **options):
    return archive_transfer.upload_archive(local_dir, backend, log=quiet, **options)


def download(backend, dest, **options):
    return archive_transfer.download_archive(backend, dest, log=quiet, **options)


def entry_keys(name, manifest=True):
    files = ["structure.mol2", "topology.pdb", "trajectory.xtc"] + (["manifest.json"] if manifest else [])
    if name != COMMA_NAME:
        files.append("info.json")
    return sorted(f"raw/{name}/{file}" for file in files)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
def test_upload_keys_are_the_glycam_names_of_the_manifests(local, fake_s3):
    backend, client = fake_s3
    long_folder = archive.entry_folder_name(NAME_G60371DN)
    assert long_folder != NAME_G60371DN and (local / long_folder).is_dir()     # the local folder is shortened
    before = snapshot(local)

    report = upload(local, backend, workers=3)
    assert report["ok"] and report["counts"] == {"uploaded": 3} and report["index_uploaded"]
    assert client.keys("bucket") == sorted(
        [key for name in NAMES for key in entry_keys(name)] + ["raw/index.json"])
    assert snapshot(local) == before                                            # the archive is only read

    # every object is the file, and carries the SHA-256 of the manifest as metadata
    for record in json.loads((local / "index.json").read_text())["records"]:
        for file, recorded in record["files"].items():
            key = f"raw/{record['glycam_name']}/{file}"
            assert client.body("bucket", key) == (local / record["folder"] / file).read_bytes()
            assert backend.head(key)["sha256"] == recorded["sha256"]
    assert client.body("bucket", "raw/index.json") == (local / "index.json").read_bytes()

    # within an entry the manifest goes last, and the index after everything else
    sent = [key for _, _, key in client.writes()]
    assert client.writes()[-1] == ("put_object", "bucket", "raw/index.json") and len(sent) == len(set(sent)) == 15
    assert client.count("upload_file") == 14
    for name in NAMES:
        positions = [sent.index(key) for key in entry_keys(name, manifest=False)]
        assert sent.index(f"raw/{name}/manifest.json") > max(positions)

    comparison = report["comparison"]
    assert comparison["ok"] and (comparison["entries"], comparison["objects"]) == (3, 14)
    assert comparison["missing"] == comparison["size_mismatch"] == comparison["stale"] == comparison["extra"] == []


def test_upload_skips_what_is_stored_with_size_and_sha256(local, fake_s3):
    backend, client = fake_s3
    upload(local, backend)
    client.calls.clear()

    report = upload(local, backend)
    assert report["ok"] and report["counts"] == {"skipped": 3} and not report["index_uploaded"]
    # every file compared (14), every manifest against the index (3), the index itself (1); nothing sent
    assert client.writes() == [] and client.count("head_object") == 14 + 3 + 1

    # one object gone, one with another size, one whose metadata names other content
    del client.objects["bucket"]["raw/DGlcpa1-OH/structure.mol2"]
    client.objects["bucket"][f"raw/{COMMA_NAME}/topology.pdb"]["body"] += b"x"
    client.objects["bucket"][f"raw/{NAME_G60371DN}/trajectory.xtc"]["metadata"]["sha256"] = "0" * 64
    client.calls.clear()
    report = upload(local, backend, workers=1)
    assert report["ok"] and report["counts"] == {"uploaded": 3}
    # the stored manifest of each of these entries goes before its file is replaced, and comes back last
    by_entry = {name: [(operation, key.rsplit("/", 1)[1]) for operation, _, key in client.writes()
                       if key.startswith(f"raw/{name}/")] for name in NAMES}
    assert by_entry == {
        "DGlcpa1-OH": [("delete_object", "manifest.json"), ("upload_file", "structure.mol2"),
                       ("upload_file", "manifest.json")],
        COMMA_NAME: [("delete_object", "manifest.json"), ("upload_file", "topology.pdb"),
                     ("upload_file", "manifest.json")],
        NAME_G60371DN: [("delete_object", "manifest.json"), ("upload_file", "trajectory.xtc"),
                        ("upload_file", "manifest.json")],
    }
    assert [row["transferred"] for row in report["entries"]] == [
        ["structure.mol2", "manifest.json"], ["topology.pdb", "manifest.json"], ["trajectory.xtc", "manifest.json"]]
    # no request for the two files that the listing already rules out
    assert client.count("head_object") == 12 + 3 + 1
    assert upload(local, backend)["counts"] == {"skipped": 3}


def test_the_index_waits_for_every_entry_it_pins(local, fake_s3, tmp_path):
    backend, client = fake_s3
    upload(local, backend)
    stored_index = client.body("bucket", "raw/index.json")

    # an entry is converted again: its manifest changes (here only that), and the index with it
    manifest_path = local / "DGlcpa1-OH" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["converter"]["date"] = "2027-01-01T00:00:00Z"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    assert archive.write_index(local)[1] == []

    # uploading another entry must not publish an index that pins a manifest which is not stored
    report = upload(local, backend, glycans=[COMMA_NAME])
    assert report["ok"] and report["counts"] == {"skipped": 1} and not report["index_uploaded"]
    assert report["comparison"]["stale"] == ["DGlcpa1-OH"] == report["comparison"]["incomplete"]
    assert client.body("bucket", "raw/index.json") == stored_index
    assert download(backend, tmp_path / "mirror")["ok"]                         # the stored archive is still consistent

    report = upload(local, backend)
    assert report["ok"] and report["counts"] == {"skipped": 2, "uploaded": 1} and report["index_uploaded"]
    assert client.body("bucket", "raw/index.json") == (local / "index.json").read_bytes()
    assert download(backend, tmp_path / "mirror")["counts"] == {"downloaded": 1, "skipped": 2}


def test_an_entry_that_is_being_replaced_has_no_stored_manifest(local, fake_s3, tmp_path):
    backend, client = fake_s3
    upload(local, backend)
    # the entry is converted again from a longer trajectory
    shutil.rmtree(local / COMMA_NAME)
    source = make_source(tmp_path / "data", COMMA_NAME, random_coords(30, 14, seed=9))
    archive.convert_entry(source, local)
    archive.write_index(local)

    def stop_at_the_topology(operation, bucket, key):
        if operation == "upload_file" and key == f"raw/{COMMA_NAME}/topology.pdb":
            raise Interrupted("connection lost")

    client.before = stop_at_the_topology
    report = upload(local, backend, workers=1)
    client.before = None
    assert not report["ok"] and report["comparison"]["incomplete"] == [COMMA_NAME]
    # some files are new, some old: there is no manifest that would call this an entry
    assert f"raw/{COMMA_NAME}/manifest.json" not in client.keys("bucket")
    failed = download(backend, tmp_path / "mirror", glycans=[COMMA_NAME])["failed"]
    assert "no entry in the storage" in failed[0]["error"]

    assert upload(local, backend)["index_uploaded"]
    assert download(backend, tmp_path / "mirror", glycans=[COMMA_NAME])["ok"]
    assert archive.ArchiveEntry(tmp_path / "mirror" / COMMA_NAME).n_frames == 30


def test_an_interrupted_upload_is_resumed_and_the_index_comes_last(local, fake_s3):
    backend, client = fake_s3

    def stop_at_the_long_trajectory(operation, bucket, key):
        if operation == "upload_file" and key == f"raw/{NAME_G60371DN}/trajectory.xtc":
            raise Interrupted("connection lost")

    client.before = stop_at_the_long_trajectory
    report = upload(local, backend, workers=1)
    assert not report["ok"] and report["counts"] == {"failed": 1, "uploaded": 2}
    assert "Interrupted: connection lost" in report["failed"][0]["error"]
    assert "raw/index.json" not in client.keys("bucket") and not report["index_uploaded"]
    assert f"raw/{NAME_G60371DN}/manifest.json" not in client.keys("bucket")     # no manifest: not a complete entry
    assert report["comparison"]["incomplete"] == [NAME_G60371DN] and not report["comparison"]["ok"]

    client.before = None
    client.calls.clear()
    report = upload(local, backend)
    assert report["ok"] and report["counts"] == {"skipped": 2, "uploaded": 1} and report["index_uploaded"]
    assert [key for _, _, key in client.writes()] == [
        f"raw/{NAME_G60371DN}/trajectory.xtc", f"raw/{NAME_G60371DN}/manifest.json", "raw/index.json"]


def test_a_subset_is_uploaded_by_any_of_its_names(local, fake_s3):
    backend, client = fake_s3
    report = upload(local, backend, glycans=["G60371DN"])                        # the source folder
    assert report["ok"] and [row["glycam_name"] for row in report["entries"]] == [NAME_G60371DN]
    assert client.keys("bucket") == entry_keys(NAME_G60371DN)                    # the index waits for the rest
    assert sorted(report["comparison"]["incomplete"]) == sorted([COMMA_NAME, "DGlcpa1-OH"])

    report = upload(local, backend, glycans=[COMMA_NAME, archive.entry_folder_name("DGlcpa1-OH")])
    assert report["ok"] and report["counts"] == {"uploaded": 2} and report["index_uploaded"]
    with pytest.raises(TransferError, match="not in index.json: DGalpa1-OH"):
        upload(local, backend, glycans=["DGalpa1-OH"])

    # another prefix and objects that the index does not know
    backend.write_binary("raw/DGalpa1-OH/manifest.json", b"{}")
    comparison = archive_transfer.compare_with_index(backend, "raw", archive_transfer.load_index(local))
    assert comparison["ok"] and comparison["extra"] == ["raw/DGalpa1-OH/manifest.json"]
    assert upload(local, backend, prefix="copy/raw/")["counts"] == {"uploaded": 3}
    assert f"copy/raw/{COMMA_NAME}/manifest.json" in client.keys("bucket", "copy/")


def test_upload_sends_only_what_matches_its_manifest(local, fake_s3):
    backend, client = fake_s3
    with open(local / "DGlcpa1-OH" / "trajectory.xtc", "r+b") as handle:         # same size, other content
        handle.seek(40)
        handle.write(b"\xff\xff")
    (local / COMMA_NAME / "structure.mol2").write_bytes(b"shorter")              # another size
    report = upload(local, backend)
    errors = {row["glycam_name"]: row["error"] for row in report["failed"]}
    assert not report["ok"] and report["counts"] == {"failed": 2, "uploaded": 1}
    assert "trajectory.xtc does not match the size and SHA-256 of its manifest" in errors["DGlcpa1-OH"]
    assert "structure.mol2 is missing or does not have the size of the manifest" in errors[COMMA_NAME]
    assert "raw/DGlcpa1-OH/trajectory.xtc" not in client.keys("bucket")
    assert client.keys("bucket", f"raw/{COMMA_NAME}/") == [] and "raw/index.json" not in client.keys("bucket")


def test_upload_needs_an_index_that_describes_the_archive(local, fake_s3):
    backend, client = fake_s3
    manifest_path = local / "DGlcpa1-OH" / "manifest.json"
    manifest_path.write_text(manifest_path.read_text() + "\n")                   # the entry changed after the index
    report = upload(local, backend)
    assert not report["ok"] and "index.json does not describe this entry" in report["failed"][0]["error"]
    assert client.keys("bucket", "raw/DGlcpa1-OH/") == []

    index = json.loads((local / "index.json").read_text())
    index["records"].append(dict(index["records"][0], glycam_name="DGlcpb1-OH"))  # two entries of one source folder
    (local / "index.json").write_text(json.dumps(index))
    with pytest.raises(TransferError, match="more than one entry with source_folder"):
        upload(local, backend)
    (local / "index.json").write_text("{}")
    with pytest.raises(TransferError, match="is not a glycoshape-raw-trajectory-index file"):
        upload(local, backend)
    (local / "index.json").unlink()
    with pytest.raises(TransferError, match="cannot read"):
        upload(local, backend)


def test_a_dry_run_of_the_upload_sends_nothing(local, fake_s3):
    backend, client = fake_s3
    upload(local, backend, glycans=["DGlcpa1-OH"])
    client.calls.clear()
    report = upload(local, backend, dry_run=True)
    assert report["ok"] and report["dry_run"] and report["counts"] == {"skipped": 1, "uploaded": 2}
    assert report["comparison"] is None and client.writes() == []
    assert report["bytes"] == sum(
        (local / archive.entry_folder_name(name) / file).stat().st_size
        for name in (COMMA_NAME, NAME_G60371DN)
        for file in [key.rsplit("/", 1)[1] for key in entry_keys(name)])


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def test_one_entry_is_downloaded_by_glycam_name_into_a_folder_the_pipeline_reads(local, fake_s3, tmp_path):
    backend, client = fake_s3
    upload(local, backend)
    data = tmp_path / "data"

    report = download(backend, data, glycans=[NAME_G60371DN])
    folder = data / archive.entry_folder_name(NAME_G60371DN)
    assert report["ok"] and report["counts"] == {"downloaded": 1} and not report["index_written"]
    assert [path.name for path in data.iterdir()] == [folder.name]               # no staging folder is left
    assert snapshot(folder).keys() == {"info.json", "manifest.json", "structure.mol2", "topology.pdb", "trajectory.xtc"}
    assert {name: digest for name, (digest, _) in snapshot(folder).items()} == {
        name: digest for name, (digest, _) in snapshot(local / folder.name).items()}

    # it is an archive entry: the verifier accepts it and the store step finds its inputs
    assert archive.verify_entry(folder)["ok"]
    assert archive.entry_inputs(folder)["glycam_name"] == NAME_G60371DN
    assert np.array_equal(
        archive.ArchiveEntry(folder).read_frames(unit="milliangstrom"), random_coords(7, 16, seed=3))

    client.calls.clear()
    assert download(backend, data, glycans=[NAME_G60371DN])["counts"] == {"skipped": 1}
    assert client.count("download_file") == 0
    report = download(backend, data, glycans=["DGalpa1-OH", "DGlcpa1-OH"])
    assert not report["ok"] and report["counts"] == {"downloaded": 1, "failed": 1}
    assert "no entry in the storage under raw/DGalpa1-OH/" in report["failed"][0]["error"]


def test_download_compares_every_file_with_the_manifest_and_resumes(local, fake_s3, tmp_path):
    backend, client = fake_s3
    upload(local, backend)
    data = tmp_path / "data"
    key = f"raw/{COMMA_NAME}/trajectory.xtc"
    good = client.body("bucket", key)
    client.objects["bucket"][key]["body"] = good[:50] + bytes([good[50] ^ 1]) + good[51:]   # same size

    report = download(backend, data, glycans=[COMMA_NAME])
    assert not report["ok"]
    assert "downloaded trajectory.xtc does not match the size and SHA-256 of the manifest" in report["failed"][0]["error"]
    assert not (data / COMMA_NAME).exists()                                       # no entry folder without all of it
    staging = next((data / ".staging").iterdir())
    assert sorted(path.name for path in staging.iterdir()) == ["structure.mol2", "topology.pdb"]

    client.objects["bucket"][key]["body"] = good
    client.calls.clear()
    report = download(backend, data, glycans=[COMMA_NAME])
    assert report["ok"] and report["entries"][0]["transferred"] == ["trajectory.xtc"]
    assert [call[2] for call in client.calls if call[0] == "download_file"] == [key]   # the verified files were kept
    assert archive.verify_entry(data / COMMA_NAME)["ok"] and not (data / ".staging").exists()

    # a truncated or changed local file is fetched again; --trust-local looks at sizes only
    target = data / COMMA_NAME / "topology.pdb"
    target.write_bytes(target.read_bytes()[:-1] + b"?")
    assert download(backend, data, glycans=[COMMA_NAME], trust_local=True)["counts"] == {"skipped": 1}
    report = download(backend, data, glycans=[COMMA_NAME])
    assert report["counts"] == {"downloaded": 1} and report["entries"][0]["transferred"] == ["topology.pdb"]
    assert archive.verify_entry(data / COMMA_NAME)["ok"]


def test_the_whole_archive_is_downloaded_from_the_stored_index(local, fake_s3, tmp_path):
    backend, client = fake_s3
    upload(local, backend)
    mirror = tmp_path / "mirror"

    dry = download(backend, mirror, dry_run=True)
    assert dry["ok"] and dry["counts"] == {"download": 3} and not mirror.exists()
    assert dry["bytes"] == sum(
        item["size"] for record in json.loads((local / "index.json").read_text())["records"]
        for name, item in record["files"].items() if name != "manifest.json")

    report = download(backend, mirror, workers=3)
    assert report["ok"] and report["counts"] == {"downloaded": 3} and report["index_written"]
    assert {name: digest for name, (digest, _) in snapshot(mirror).items()} == {
        name: digest for name, (digest, _) in snapshot(local).items()}
    assert archive.build_index(mirror)[1] == []
    client.calls.clear()
    assert download(backend, mirror)["counts"] == {"skipped": 3} and client.count("download_file") == 0

    # An entry was uploaded again and the index not yet.  The whole archive is pinned by the stored
    # index and refuses that entry; by name it is its own manifest that counts.
    key = "raw/DGlcpa1-OH/manifest.json"
    manifest = json.loads(client.body("bucket", key))
    manifest["warnings"] = ["converted again"]
    payload = json.dumps(manifest).encode()
    client.objects["bucket"][key]["body"] = payload
    report = download(backend, mirror)
    assert not report["ok"] and report["counts"] == {"failed": 1, "skipped": 2}
    assert "does not match the stored index.json" in report["failed"][0]["error"]
    assert json.loads((mirror / "DGlcpa1-OH" / "manifest.json").read_text())["warnings"] == []
    client.calls.clear()
    report = download(backend, mirror, glycans=["DGlcpa1-OH"])
    assert report["ok"] and report["counts"] == {"downloaded": 1}
    assert client.count("download_file") == 0                                     # its files had not changed
    assert json.loads((mirror / "DGlcpa1-OH" / "manifest.json").read_text())["warnings"] == ["converted again"]
    assert archive.verify_entry(mirror / "DGlcpa1-OH", deep=False)["errors"] == []
    assert sorted(path.name for path in mirror.iterdir()) == sorted(      # replaced in place, nothing left behind
        ["index.json"] + [archive.entry_folder_name(name) for name in NAMES])
    # once the index lists it, the whole archive is consistent again
    index = json.loads(client.body("bucket", "raw/index.json"))
    record = next(record for record in index["records"] if record["glycam_name"] == "DGlcpa1-OH")
    record["files"]["manifest.json"] = {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    client.objects["bucket"]["raw/index.json"]["body"] = json.dumps(index).encode()
    assert download(backend, mirror)["counts"] == {"skipped": 3}

    del client.objects["bucket"]["raw/index.json"]
    with pytest.raises(TransferError, match="no raw/index.json in the storage"):
        download(backend, tmp_path / "other")
    assert download(backend, tmp_path / "other", glycans=[COMMA_NAME])["ok"]     # by name it needs no index


def test_download_refuses_what_is_not_an_entry(local, fake_s3, tmp_path):
    backend, client = fake_s3
    upload(local, backend)
    data = tmp_path / "data"
    (data / "DGlcpa1-OH").mkdir(parents=True)
    (data / "DGlcpa1-OH" / "simulation.pdb").write_text("a source folder, not an entry")
    report = download(backend, data, glycans=["DGlcpa1-OH"])
    assert "exists and is not an archive entry; it is left untouched" in report["failed"][0]["error"]
    assert [path.name for path in (data / "DGlcpa1-OH").iterdir()] == ["simulation.pdb"]

    key = f"raw/{COMMA_NAME}/manifest.json"
    original = client.body("bucket", key)
    del client.objects["bucket"]["raw/index.json"]
    for change, message in (
        (lambda m: m["files"].update({"../../escape.txt": {"size": 1, "sha256": "0" * 64}}), "unexpected file"),
        (lambda m: m["files"].pop("structure.mol2"), "does not list the files of an entry"),
        (lambda m: m.update(glycam_name="DGlcpa1-OH"), "is of another entry"),
        (lambda m: m.update(folder="elsewhere"), "not the folder of its GLYCAM name"),
        (lambda m: m.update(schema="something-else"), "is not a glycoshape-raw-trajectory manifest"),
        (lambda m: m["files"]["topology.pdb"].pop("sha256"), "no size and SHA-256 for topology.pdb"),
    ):
        manifest = json.loads(original)
        change(manifest)
        client.objects["bucket"][key]["body"] = json.dumps(manifest).encode()
        report = download(backend, data, glycans=[COMMA_NAME])
        assert message in report["failed"][0]["error"], report["failed"][0]["error"]
    client.objects["bucket"][key]["body"] = b"not json"
    assert "is not JSON" in download(backend, data, glycans=[COMMA_NAME])["failed"][0]["error"]
    assert not (data / COMMA_NAME).exists() and not (tmp_path / "escape.txt").exists()


def test_names_that_cannot_be_keys_are_refused():
    assert archive_transfer.entry_key_prefix("raw", NAME_G60371DN) == "raw/" + NAME_G60371DN + "/"
    assert archive_transfer.entry_key_prefix("/in/raw/", "DGlcpa1-OH") == "in/raw/DGlcpa1-OH/"
    for name in ("", "a//b", "../x"):
        with pytest.raises(archive.NamingError):
            archive_transfer.entry_key_prefix("raw", name)
    with pytest.raises(archive.NamingError, match="exceeds 1024 bytes"):
        archive_transfer.entry_key_prefix("a-long-prefix-" * 20, "D" * 900)
