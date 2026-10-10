"""Tests of publishing the static database (lib/static_publish.py) on an in-memory S3 client.

``reader_sync`` (tests/s3_fakes.py) is the reader of the website; the tests
use it to say what a reader sees at every point of a publish.
"""
from __future__ import annotations

import copy
import json

import pytest

from lib import static_publish
from lib.static_publish import PublishError
from s3_fakes import Interrupted, ReaderError, reader_sync, reader_validate, sample_glycans, tree, write_static

ROOT_FILES = {"GLYCOSHAPE.json": b'{"version": 1}', "GLYCOSHAPE_RDF.ttl": b"@prefix : <#> .\n", "GlycoShape.zip": b"PK-1"}


def quiet(message):
    pass


def publish(static_dir, backend, **options):
    return static_publish.publish(static_dir, backend, log=quiet, **options)


def check(backend, **options):
    return static_publish.check_published(backend, log=quiet, **options)


@pytest.fixture
def static(tmp_path):
    """A static directory of three glycans with its sync_manifest.json (version 1)."""
    write_static(tmp_path / "static", sample_glycans(3), ROOT_FILES)
    return tmp_path / "static"


def second_version(static_dir):
    """Change the static directory: one glycan changed, one added, one unchanged, root files changed."""
    (static_dir / "GS00002" / "data.json").write_bytes(b'{"archetype": {"ID": "GS00002", "revised": true}}')
    (static_dir / "GS00002" / "output" / "torsions.csv").write_bytes(b"phi,psi\n")
    glycans = sample_glycans(4)
    return write_static(static_dir, {"GS00004": glycans["GS00004"]}, {
        **ROOT_FILES, "GLYCOSHAPE.json": b'{"version": 2}', "GlycoShape.zip": b"PK-2"})


CHANGED = ["static/GLYCOSHAPE.json", "static/GS00002/data.json", "static/GlycoShape.zip"]
NEW = ["static/GS00002/output/torsions.csv"] + [
    f"static/GS00004/{name}" for name in ("data.json", "output/level_1/PDB/alpha.pdb", "output/pca.csv", "snfg.svg")]


def written(client):
    """The requests that change the storage, in order: (operation, key)."""
    return [(operation, key) for operation, _, key in client.writes()]


# ---------------------------------------------------------------------------
# The manifest contract
# ---------------------------------------------------------------------------
def test_the_generated_manifest_satisfies_the_reader(static):
    manifest = json.loads((static / "sync_manifest.json").read_bytes())
    assert static_publish.validate_manifest(manifest) == []
    reader_validate(manifest)
    files = static_publish.manifest_files(manifest)
    assert len(files) == 3 + 3 * 4 and set(files) == set(tree(static)) - {"sync_manifest.json"}
    assert files["GS00001/output/pca.csv"] == {
        "size": 8, "sha256": manifest["glycans"]["GS00001"]["files"]["output/pca.csv"]["sha256"]}


def _set(path, value):
    def change(manifest):
        target = manifest
        for part in path[:-1]:
            target = target[part]
        if value is _set:
            del target[path[-1]]
        else:
            target[path[-1]] = value
    return change


GS1_FILE = ("glycans", "GS00001", "files", "data.json")


@pytest.mark.parametrize("change, message", [
    (_set(("schema_version",), 2), "expected schema_version 1"),
    (_set(("schema_version",), "1"), "expected schema_version 1"),
    (_set(("hash_algorithm",), "md5"), "expected schema_version 1 with sha256 hashes"),
    (_set(("global_hash",), "abc"), "missing global hash or glycan entries"),
    (_set(("global_hash",), _set), "missing global hash or glycan entries"),
    (_set(("glycans",), {}), "missing global hash or glycan entries"),
    (_set(("root_files", "GLYCOSHAPE_RDF.ttl", "path"), "rdf/GLYCOSHAPE_RDF.ttl"), "root file is not at the database root"),
    (_set(("root_files", "GLYCOSHAPE_RDF.ttl", "path"), "../GLYCOSHAPE_RDF.ttl"), "unsafe object path"),
    (_set(("root_files", "GLYCOSHAPE_RDF.ttl", "path"), ".hidden"), "unsafe object path"),
    (_set(("root_files", "GLYCOSHAPE_RDF.ttl", "sha256"), "z" * 64), "invalid SHA-256"),
    (_set(("root_files", "GLYCOSHAPE_RDF.ttl", "size"), -1), "invalid size"),
    (_set(("root_files", "GLYCOSHAPE_RDF.ttl", "size"), _set), "invalid size"),
    (_set(GS1_FILE + ("path",), "GS00001/.cache/data.json"), "unsafe object path"),
    (_set(GS1_FILE + ("path",), "GS00002/data.json"), "is outside GS00001"),
    (_set(GS1_FILE + ("path",), "data.json"), "is outside GS00001"),
    (_set(GS1_FILE + ("path",), ""), "unsafe object path"),
    (_set(GS1_FILE + ("sha256",), "0" * 63), "invalid SHA-256"),
    (_set(GS1_FILE + ("size",), "12"), "invalid size"),
    (_set(("glycans", "GS00001", "file_count"), 3), "invalid glycan manifest entry 'GS00001'"),
    (_set(("glycans", "GS00001", "files"), {}), "invalid glycan manifest entry 'GS00001'"),
    (_set(("glycans", "GS00001", "hash"), "not-a-hash"), "invalid glycan manifest entry 'GS00001'"),
    (_set(("glycans", "GS00001", "hash"), _set), "invalid glycan manifest entry 'GS00001'"),
    (lambda m: m["glycans"].update({"G00001": m["glycans"].pop("GS00001")}), "invalid glycan manifest entry 'G00001'"),
    (lambda m: m["glycans"].update({"GS/1": m["glycans"].pop("GS00001")}), "invalid glycan manifest entry 'GS/1'"),
])
def test_what_the_reader_refuses_is_refused(static, change, message):
    manifest = json.loads((static / "sync_manifest.json").read_bytes())
    change(manifest)
    errors = static_publish.validate_manifest(manifest)
    assert any(message in error for error in errors), errors
    with pytest.raises(ReaderError):                 # and the reader does refuse it
        reader_validate(manifest)


def test_rules_that_follow_from_what_the_reader_does(static):
    original = json.loads((static / "sync_manifest.json").read_bytes())

    manifest = copy.deepcopy(original)               # a fresh reader fails without GLYCOSHAPE.json
    del manifest["root_files"]["GLYCOSHAPE.json"]
    assert any("GLYCOSHAPE.json is not a root file" in error for error in static_publish.validate_manifest(manifest))

    manifest = copy.deepcopy(original)               # one path, two contents
    manifest["root_files"]["copy"] = dict(manifest["root_files"]["GlycoShape.zip"], path="GLYCOSHAPE.json")
    assert any("GLYCOSHAPE.json is listed twice" in error for error in static_publish.validate_manifest(manifest))

    for path in ("/GLYCOSHAPE.json", " GLYCOSHAPE.json", "GS00001\\data.json", "GS00001//data.json"):
        manifest = copy.deepcopy(original)           # the reader would rewrite these before using them
        manifest["root_files"]["GLYCOSHAPE.json"]["path"] = path
        assert any("unsafe object path" in error for error in static_publish.validate_manifest(manifest))

    assert static_publish.validate_manifest([]) == ["the manifest is not a JSON object"]
    assert static_publish.validate_manifest({"schema_version": 1, "hash_algorithm": "sha256"})[-1] == (
        "root_files and glycans must be objects")
    with pytest.raises(PublishError, match="is not JSON"):
        static_publish.parse_manifest(b"{", "manifest")
    with pytest.raises(PublishError, match="would be refused by the reader"):
        static_publish.parse_manifest(b"{}", "manifest")


def test_a_file_cannot_also_be_a_folder(static, fake_s3, tmp_path):
    """Object keys allow it, the directory tree of the reader does not."""
    backend, client = fake_s3
    publish(static, backend)
    original = json.loads(client.body("bucket", "static/sync_manifest.json"))
    for path, label in (("GS00001", "root file"), ("GS00002/output", "glycan file")):
        manifest = copy.deepcopy(original)
        entry = dict(manifest["root_files"]["GlycoShape.zip"], path=path)
        if label == "root file":
            manifest["root_files"][path] = entry
        else:
            manifest["glycans"]["GS00002"]["files"]["output"] = entry
            manifest["glycans"]["GS00002"]["file_count"] += 1
        reader_validate(manifest)                                  # passes the reader's validation...
        errors = static_publish.validate_manifest(manifest)
        assert len(errors) >= 1 and all(f"{path} is listed as a file and is a folder of" in error for error in errors)

        client.put_object(Bucket="bucket", Key=f"static/{path}", Body=client.body("bucket", "static/GlycoShape.zip"))
        client.put_object(Bucket="bucket", Key="static/sync_manifest.json", Body=json.dumps(manifest).encode())
        with pytest.raises(OSError):                               # ...and fails when it writes the files
            reader_sync(backend, "static", tmp_path / label)
        report = check(backend)
        assert not report["ok"] and "is listed as a file and is a folder of" in report["errors"][0]
        client.delete_object(Bucket="bucket", Key=f"static/{path}")


@pytest.mark.parametrize("damage, reason", [
    (lambda text: b"\xef\xbb\xbf" + text.encode(), "byte order mark"),
    (lambda text: text.encode("utf-16"), "is not JSON"),
    (lambda text: text.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1').encode(), "key twice"),
    (lambda text: text.replace('"glycan_count": 3', '"glycan_count": NaN').encode(), "NaN is not JSON"),
    (lambda text: text.replace('"static_prefix": "static"', '"static_prefix": "\\ud800"').encode(), "is not JSON"),
    (lambda text: text.encode() + b"{}", "is not JSON"),
])
def test_the_manifest_is_parsed_as_strictly_as_the_reader_parses_it(static, fake_s3, damage, reason):
    backend, client = fake_s3
    publish(static, backend)
    text = (static / "sync_manifest.json").read_text()
    payload = damage(text)
    assert payload != text.encode()
    with pytest.raises(PublishError, match=reason):
        static_publish.parse_manifest(payload, "manifest")

    # such a manifest is neither published nor accepted when it is found published
    (static / "sync_manifest.json").write_bytes(payload)
    client.calls.clear()
    with pytest.raises(PublishError, match="is not JSON"):
        publish(static, backend)
    assert client.writes() == []
    client.put_object(Bucket="bucket", Key="static/sync_manifest.json", Body=payload)
    report = check(backend)
    assert not report["ok"] and "is not JSON" in report["errors"][0]


# ---------------------------------------------------------------------------
# Publish
# ---------------------------------------------------------------------------
def test_the_first_publish_sends_everything_and_the_manifest_last(static, fake_s3, tmp_path):
    backend, client = fake_s3
    with pytest.raises(ReaderError):                 # nothing published yet
        reader_sync(backend, "static", tmp_path / "site")

    report = publish(static, backend, workers=4)
    assert (report["new"], report["changed"], report["unchanged"], report["manifest_written"]) == (15, 0, 0, True)
    sent = written(client)
    assert sent[-1] == ("put_object", "static/sync_manifest.json")
    assert sorted(sent[:-1]) == sorted(("upload_file", f"static/{path}") for path in tree(static)
                                       if path != "sync_manifest.json")
    assert client.keys("bucket") == sorted(f"static/{path}" for path in tree(static))
    manifest = json.loads((static / "sync_manifest.json").read_bytes())
    assert backend.head("static/GS00003/snfg.svg")["sha256"] == manifest["glycans"]["GS00003"]["files"]["snfg.svg"]["sha256"]

    # the website gets exactly the static directory
    summary = reader_sync(backend, "static", tmp_path / "site")
    assert summary == {"downloaded": 15, "skipped": 0, "pruned": 0}
    assert tree(tmp_path / "site") == tree(static)
    assert check(backend)["ok"]

    # nothing changed: nothing is sent, not even the manifest
    client.calls.clear()
    report = publish(static, backend)
    assert (report["new"], report["changed"], report["unchanged"], report["manifest_written"]) == (0, 0, 15, False)
    assert client.writes() == []
    assert reader_sync(backend, "static", tmp_path / "site")["downloaded"] == 0


def test_only_what_changed_is_sent_and_changed_files_are_switched_at_the_end(static, fake_s3, tmp_path):
    backend, client = fake_s3
    publish(static, backend)
    site = tmp_path / "site"
    reader_sync(backend, "static", site)
    manifest = second_version(static)
    client.calls.clear()

    report = publish(static, backend, workers=4)
    assert (report["new"], report["changed"], report["unchanged"], report["manifest_written"]) == (5, 3, 12, True)
    sent = written(client)
    uploads = [key for operation, key in sent if operation == "upload_file"]
    copies = [key for operation, key in sent if operation == "copy_object"]
    staged = sorted(key for key in uploads if key.startswith("publish-staging/static/"))

    # new files go to their keys, changed files to the staging prefix (named by content), nothing else is sent
    assert sorted(key for key in uploads if key.startswith("static/")) == NEW
    assert staged == sorted(
        "publish-staging/static/" + static_publish.manifest_files(manifest)[key[len("static/"):]]["sha256"]
        for key in CHANGED)
    # then the changed files are copied inside the storage, then the manifest, then the staging objects go
    assert sorted(copies) == CHANGED
    order = [operation for operation, _ in sent]
    assert order == ["upload_file"] * 8 + ["copy_object"] * 3 + ["put_object"] + ["delete_object"] * 3
    assert sent[11] == ("put_object", "static/sync_manifest.json")
    assert client.keys("bucket", "publish-staging/") == []
    assert client.keys("bucket") == sorted(f"static/{path}" for path in tree(static))

    # a reader that had the first version fetches the root files and the two glycans that changed
    summary = reader_sync(backend, "static", site)
    assert summary == {"downloaded": 3 + 5 + 4, "skipped": 8, "pruned": 0}
    assert tree(site) == tree(static) and check(backend)["ok"]


@pytest.mark.parametrize("stop_at", [1, 4, 8])
def test_an_interrupted_upload_changes_nothing_a_reader_sees(static, fake_s3, tmp_path, stop_at):
    backend, client = fake_s3
    publish(static, backend)
    first = tree(static)
    current, behind = tmp_path / "current", tmp_path / "behind"
    reader_sync(backend, "static", current)
    second_version(static)

    uploads = []

    def stop(operation, bucket, key):
        if operation == "upload_file":
            uploads.append(key)
            if len(uploads) == stop_at:
                raise Interrupted("connection lost")

    client.before = stop
    with pytest.raises(PublishError, match="the published prefix has not changed for readers"):
        publish(static, backend, workers=1)
    client.before = None

    # the published prefix is the first version, complete and valid, for every reader
    assert check(backend)["ok"] and check(backend, quick=True)["ok"]
    assert reader_sync(backend, "static", current) == {"downloaded": 0, "skipped": 15, "pruned": 0}
    assert reader_sync(backend, "static", behind)["downloaded"] == 15 and tree(behind) == first
    assert all(operation == "upload_file" for operation, _ in written(client)[15 + 1:])

    # the same command again continues: what arrived is not sent a second time
    # (the upload that was already running when the first one failed may have arrived too)
    arrived = len(client.keys("bucket")) - 16
    assert arrived in (stop_at - 1, stop_at)
    client.calls.clear()
    report = publish(static, backend, workers=1)
    assert report["manifest_written"] and client.count("upload_file") == 8 - arrived
    assert reader_sync(backend, "static", current)["downloaded"] == 12 and tree(current) == tree(static)
    assert reader_sync(backend, "static", behind)["downloaded"] == 12 and tree(behind) == tree(static)
    assert check(backend)["ok"] and client.keys("bucket", "publish-staging/") == []


@pytest.mark.parametrize("stop_at", ["second copy", "manifest"])
def test_an_interruption_of_the_switch_is_finished_by_running_again(static, fake_s3, tmp_path, stop_at):
    backend, client = fake_s3
    publish(static, backend)
    first = tree(static)
    current, behind = tmp_path / "current", tmp_path / "behind"
    reader_sync(backend, "static", current)
    second_version(static)

    copies = []

    def stop(operation, bucket, key):
        copies.append(key) if operation == "copy_object" else None
        if (stop_at == "second copy" and len(copies) == 2) or (
                stop_at == "manifest" and (operation, key) == ("put_object", "static/sync_manifest.json")):
            raise Interrupted("connection lost")

    client.before = stop
    with pytest.raises(PublishError, match="run the command again to finish"):
        publish(static, backend, workers=1)
    client.before = None

    # The old manifest now describes files that were replaced.  A reader that has the published
    # global_hash does nothing; one that is behind fails its verification and keeps what it has.
    assert reader_sync(backend, "static", current) == {"downloaded": 0, "skipped": 15, "pruned": 0}
    assert tree(current) == first
    with pytest.raises(ReaderError, match="download verification failed for GLYCOSHAPE.json"):
        reader_sync(backend, "static", behind)
    assert not (behind / "sync_manifest.json").exists() and not list(behind.glob("GS*"))
    report = check(backend)
    assert not report["ok"] and "download verification failed for GLYCOSHAPE.json" in report["errors"]

    client.calls.clear()
    assert publish(static, backend, workers=1)["manifest_written"]
    assert client.count("upload_file") == 0                                   # everything had arrived
    copied = {key for operation, key in written(client) if operation == "copy_object"}
    if stop_at == "second copy":   # the copy that failed, and the third unless it was already running
        assert {"static/GS00002/data.json"} <= copied <= set(CHANGED[1:])
    else:
        assert copied == set()
    for site in (current, behind):
        reader_sync(backend, "static", site)
        assert tree(site) == tree(static)
    assert check(backend)["ok"] and client.keys("bucket", "publish-staging/") == []


def test_what_cannot_be_published_is_refused_before_anything_is_sent(static, fake_s3):
    backend, client = fake_s3

    (static / "GS00001" / ".DS_Store").write_bytes(b"finder")                  # the reader refuses hidden paths
    write_static(static, {}, ROOT_FILES)
    with pytest.raises(PublishError, match="would be refused by the reader: unsafe object path 'GS00001/.DS_Store'"):
        publish(static, backend)
    (static / "GS00001" / ".DS_Store").unlink()
    write_static(static, {}, ROOT_FILES)

    (static / "GS00003" / "snfg.svg").write_bytes(b"<svg>changed after the manifest was written</svg>")
    with pytest.raises(PublishError, match=r"1 file\(s\) of sync_manifest.json are missing .* \(GS00003/snfg.svg\)"):
        publish(static, backend)
    (static / "GS00003" / "snfg.svg").unlink()
    with pytest.raises(PublishError, match="rebuild the static output"):
        publish(static, backend)
    (static / "sync_manifest.json").unlink()
    with pytest.raises(PublishError, match="cannot read"):
        publish(static, backend)
    assert client.writes() == []

    for prefix, staging in (("static", "static/staging"), ("static/v1", "static"), ("", "x"), ("static", "")):
        with pytest.raises(PublishError, match="staging prefix"):
            publish(static, backend, prefix=prefix, staging_prefix=staging)


def test_a_file_that_changed_after_the_manifest_is_not_published(static, fake_s3, tmp_path):
    backend, client = fake_s3
    publish(static, backend)
    second_version(static)
    (static / "GS00004" / "snfg.svg").write_bytes(b"<svh/>")                   # same size as the manifest says
    with pytest.raises(PublishError, match="snfg.svg does not match the size and SHA-256 of sync_manifest.json"):
        publish(static, backend, workers=1)
    assert "static/GS00004/snfg.svg" not in client.keys("bucket")
    assert reader_sync(backend, "static", tmp_path / "site")["downloaded"] == 15   # still the first version
    assert check(backend)["ok"]


def test_objects_the_manifest_no_longer_lists_are_deleted_on_request(static, fake_s3, tmp_path):
    backend, client = fake_s3
    publish(static, backend)
    site = tmp_path / "site"
    reader_sync(backend, "static", site)
    import shutil
    shutil.rmtree(static / "GS00003")
    (static / "GS00001" / "snfg.svg").unlink()
    write_static(static, {}, ROOT_FILES)
    gone = [f"static/GS00001/snfg.svg"] + [key for key in client.keys("bucket") if key.startswith("static/GS00003/")]

    report = publish(static, backend)
    assert (report["stale"], report["stale_deleted"], report["manifest_written"]) == (5, 0, True)
    assert all(key in client.keys("bucket") for key in gone)                   # kept unless asked
    assert check(backend)["unlisted"] == sorted(gone) and check(backend)["ok"]

    client.calls.clear()
    report = publish(static, backend, delete_stale=True)                       # the manifest is already published
    assert (report["stale"], report["stale_deleted"], report["manifest_written"]) == (5, 5, False)
    assert sorted(written(client)) == sorted(("delete_object", key) for key in gone)
    assert client.keys("bucket") == sorted(f"static/{path}" for path in tree(static))
    assert reader_sync(backend, "static", site)["pruned"] == 1 and tree(site) == tree(static)


def test_stale_objects_are_deleted_after_the_manifest(static, fake_s3):
    backend, client = fake_s3
    publish(static, backend)
    (static / "GS00001" / "snfg.svg").unlink()
    second_version(static)
    client.calls.clear()
    publish(static, backend, delete_stale=True, workers=1)
    sent = written(client)
    assert sent.index(("delete_object", "static/GS00001/snfg.svg")) > sent.index(("put_object", "static/sync_manifest.json"))


def test_a_dry_run_only_reads(static, fake_s3):
    backend, client = fake_s3
    report = publish(static, backend, dry_run=True)
    assert (report["new"], report["changed"], report["bytes_to_upload"] > 0, report["dry_run"]) == (15, 0, True, True)
    assert client.writes() == [] and client.keys("bucket") == []
    publish(static, backend)
    second_version(static)
    client.calls.clear()
    report = publish(static, backend, dry_run=True)
    assert (report["new"], report["changed"], report["unchanged"]) == (5, 3, 12) and client.writes() == []


def test_without_a_published_manifest_files_are_replaced_in_place(static, fake_s3, tmp_path):
    backend, client = fake_s3
    # objects copied by other means: no manifest, no SHA-256 metadata, one of them out of date
    for path, content in tree(static).items():
        if path != "sync_manifest.json":
            client.put_object(Bucket="bucket", Key=f"static/{path}", Body=content)
    client.put_object(Bucket="bucket", Key="static/GS00001/data.json", Body=b"{}")
    client.calls.clear()

    report = publish(static, backend)
    assert (report["new"], report["changed"], report["manifest_written"]) == (15, 0, True)
    assert client.count("copy_object") == 0 and client.keys("bucket", "publish-staging/") == []
    assert check(backend, quick=True)["ok"] and tree(tmp_path / "x") == {}
    reader_sync(backend, "static", tmp_path / "site")
    assert tree(tmp_path / "site") == tree(static)


def test_another_prefix_and_a_failed_clean_up(static, fake_s3):
    backend, client = fake_s3
    publish(static, backend, prefix="/releases/2026/", staging_prefix="tmp/staging")
    assert check(backend, prefix="releases/2026")["ok"] and client.keys("bucket", "static/") == []
    second_version(static)

    def no_deletes(operation, bucket, key):
        if operation == "delete_object":
            raise Interrupted("delete refused")

    client.before = no_deletes
    report = publish(static, backend, prefix="releases/2026", staging_prefix="tmp/staging")
    assert report["manifest_written"] and "Interrupted" in report["cleanup_error"]
    client.before = None
    assert check(backend, prefix="releases/2026")["ok"]
    assert len(client.keys("bucket", "tmp/staging/releases/2026/")) == 3
    publish(static, backend, prefix="releases/2026", staging_prefix="tmp/staging")   # tidies up
    assert client.keys("bucket", "tmp/") == []


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------
def test_check_reads_the_published_prefix_back_as_the_reader_does(static, fake_s3):
    backend, client = fake_s3
    assert check(backend)["errors"] == ["static/sync_manifest.json does not exist"]
    publish(static, backend)

    client.calls.clear()
    report = check(backend, workers=3)
    assert report["ok"] and (report["files"], report["glycans"], report["unlisted"]) == (15, 3, [])
    assert report["bytes"] == sum(len(content) for path, content in tree(static).items() if path != "sync_manifest.json")
    assert client.count("get_object") == 16 and client.count("head_object") == 0      # manifest and every file
    client.calls.clear()
    assert check(backend, quick=True)["ok"] and client.count("get_object") == 1 and client.count("head_object") == 15

    # other content of the same size: only the download notices; a missing object: both do
    stored = client.objects["bucket"]["static/GS00002/output/pca.csv"]
    stored["body"] = stored["body"][:-2] + b"9\n"
    del client.objects["bucket"]["static/GS00003/data.json"]
    client.put_object(Bucket="bucket", Key="static/GS00009/left.txt", Body=b"left behind")
    report = check(backend)
    assert not report["ok"] and report["errors"] == [
        "GS00003/data.json is missing", "download verification failed for GS00002/output/pca.csv"]
    assert report["unlisted"] == ["static/GS00009/left.txt"]
    assert check(backend, quick=True)["errors"] == ["GS00003/data.json is missing"]

    client.objects["bucket"]["static/GlycoShape.zip"]["metadata"] = {}
    assert any("GlycoShape.zip has no SHA-256 metadata" in error for error in check(backend, quick=True)["errors"])
    client.objects["bucket"]["static/GlycoShape.zip"]["body"] += b"!"
    assert "download verification failed for GlycoShape.zip" in check(backend)["errors"]

    manifest = json.loads(client.body("bucket", "static/sync_manifest.json"))
    manifest["glycans"]["GS00001"]["file_count"] = 99
    client.put_object(Bucket="bucket", Key="static/sync_manifest.json", Body=json.dumps(manifest).encode())
    report = check(backend)
    assert not report["ok"] and "invalid glycan manifest entry 'GS00001'" in report["errors"][0]
    client.put_object(Bucket="bucket", Key="static/sync_manifest.json", Body=b"<html>")
    assert "is not JSON" in check(backend)["errors"][0]
