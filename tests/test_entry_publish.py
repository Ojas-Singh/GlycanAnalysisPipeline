"""Publishing, replacing and withdrawing one entry (lib/entry_publish.py), on an in-memory S3 client.

The bucket holds a published database of three glycans and a raw archive (``publish_world``); the
entry is the bundle of the synthetic glycan.  ``Reader`` is the website.  No test reaches the network.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import threading
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest

from lib import archive, archive_transfer, entry_complete, entry_publish, static_publish
from lib.entry_publish import LockHeld, Policy, PublishEntryError, Refused, Store, Target, new_stats
from lib.storage import S3StorageBackend

import fake_services
import publish_world
from bundle_fixtures import make_bundle, renamed_bundle, submission_record, synthetic_run, write_bundle_json
from publish_world import PUBLISHED, Reader, quiet, stored_manifest, stored_tree
from s3_fakes import FakeClientError, FakeS3Client, Interrupted
from static_tree import full_bake, manifest_without_times, root_files
from synthetic_glycan import GLYCAM_NAME

pytestmark = pytest.mark.usefixtures("no_internet")
GS = "GS01067"
SECOND_NAME = "DManpa1-2DManpb1-OH"
STATE = "publish-state/static"
WRITES = ("put_object", "upload_file", "copy_object", "complete_multipart_upload", "delete_object")


def record(glycan_id=GS, name=GLYCAM_NAME, **fields):
    return submission_record(glycan_id, name, **fields)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """The objects of a published database and archive, built once; every test gets its own copy."""
    root = tmp_path_factory.mktemp("published")
    client = FakeS3Client("bucket")
    backend = S3StorageBackend(client, bucket="bucket", sleep=lambda seconds: None)
    with pytest.MonkeyPatch.context() as patching:
        fake_services.install(patching)
        built = publish_world.build(root, backend, raw_source=publish_world.other_archive(root / "stored-archive"))
    return {"objects": client.objects, "static": built["static"]}


class World:
    """A copy of the prepared bucket, the target of a publication in it, and a website that follows it."""

    def __init__(self, prepared, tmp_path, client_class=FakeS3Client):
        self.client = client_class("bucket")
        self.client.objects = copy.deepcopy(prepared["objects"])
        self.backend = S3StorageBackend(self.client, bucket="bucket", sleep=lambda seconds: None)
        self.target = Target(static=self.backend, raw=self.backend, workers=4)
        self.tmp = Path(tmp_path)
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.static = prepared["static"]                        # the local tree the database was published from
        self.reader = Reader(self.backend, self.tmp / "website")
        self.reader.sync()

    def writes(self):
        return [call for call in self.client.calls if call[0] in WRITES]

    def published_writes(self):
        """Requests that changed something a reader or the pipeline can see: below static/ or raw/."""
        return [call for call in self.writes() if call[2].startswith(("static/", "raw/"))]

    def keys(self, prefix=""):
        return self.client.keys("bucket", prefix)

    def store(self):
        return Store(self.backend, new_stats())

    def publish(self, bundle, submission=None, action="publish", **options):
        options.setdefault("work_dir", self.tmp / "work")
        options.setdefault("log", quiet)
        return entry_publish.publish_entry(action, bundle, submission or record(), self.target, **options)

    def withdraw(self, glycan_id, **options):
        options.setdefault("log", quiet)
        return entry_publish.withdraw_entry(glycan_id, self.target, self.tmp / f"withdraw-{glycan_id}", **options)


@pytest.fixture
def world(prepared, tmp_path, monkeypatch):
    fake_services.install(monkeypatch)
    return World(prepared, tmp_path)


@pytest.fixture
def second_bundle(bundle_dir, tmp_path):
    """The bundle of another glycan (the same analysis under another name)."""
    return renamed_bundle(tmp_path / "second-bundle", bundle_dir, SECOND_NAME)


@pytest.fixture(scope="module")
def extended_bundle(tmp_path_factory, synthetic_entry):
    """A bundle whose entry extends the one of ``bundle_dir``: the same first 480 frames and 120 more."""
    root = tmp_path_factory.mktemp("extended-entry")
    run = synthetic_run(root, frames=600, info={"simulationLength": 600.0})
    first = hashlib.sha256((synthetic_entry["archive"] / "manifest.json").read_bytes()).hexdigest()
    return make_bundle(root / "bundle", run["result"], run["archive"],
                       extends={"glycam_name": GLYCAM_NAME, "manifest_sha256": first, "frames": 480})


# ---------------------------------------------------------------------------
# Publish
# ---------------------------------------------------------------------------
def test_a_dry_run_reports_and_writes_nothing(world, bundle_dir):
    before = stored_tree(world.backend, "")
    report = world.publish(bundle_dir, dry_run=True)
    assert world.writes() == [] and stored_tree(world.backend, "") == before        # not even a lock
    assert report["ok"] and report["dry_run"] and report["status"] == "planned" and report["refused"] == []
    assert report["entry"]["file_count"] == 25 and len(report["entry"]["hash"]) == 64
    assert report["entry"]["identity"]["archetype"]["iupac"] and report["entry"]["gaps"] == []
    assert report["entry"]["simulation"]["length"] == 2.5
    assert report["database"] == {"global_hash": world.reader.global_hash, "glycans": 3, "withdrawn": 0}
    assert report["raw"]["relation"] == "new" and report["raw"]["index"] == "2 entries"
    assert report["raw"]["upload"] == ["info.json", "structure.mol2", "topology.pdb", "trajectory.xtc", "manifest.json"]
    static = report["static"]
    assert static["upload_new"] == 25 and static["unchanged"] == 0 and static["delete_afterwards"] == 0
    # the root files that change (the missing reports do not: every variant of the entry has an accession)
    assert static["stage_and_switch"] == ["static/GLYCOSHAPE.json", "static/GlycoShape.zip", "static/GLYCOSHAPE_RDF.ttl"]
    assert static["base_global_hash"] == world.reader.global_hash != static["target_global_hash"]
    assert report["transfers"]["uploaded_bytes"] == 0 and report["transfers"]["downloaded_bytes"] > 0
    assert (world.tmp / "work" / "static" / GS / "data.json").is_file()                # the folder that would be published
    assert json.loads((world.tmp / "work" / "journal.json").read_text())["status"] == "dry-run"


def test_publishing_adds_one_glycan_and_touches_no_other(world, bundle_dir, synthetic_entry, tmp_path):
    approved = world.publish(bundle_dir, dry_run=True)["entry"]["hash"]
    old = stored_manifest(world.backend)
    report = world.publish(bundle_dir, expect_hash=approved)
    assert report["status"] == "published" and report["ok"] and report["entry"]["hash"] == approved

    # the website gets the new database: the three glycans it had, untouched, and the new one
    assert world.reader.try_sync() is None
    assert world.reader.glycans() == sorted(PUBLISHED + (GS,))
    assert world.reader.global_hash == report["static"]["target_global_hash"]
    folder = world.tmp / "work" / "static" / GS
    for path in folder.rglob("*"):
        if path.is_file():
            assert (world.reader.folder / GS / path.relative_to(folder)).read_bytes() == path.read_bytes()
    assert world.reader.fresh(tmp_path / "new-website").try_sync() is None             # every listed file verifies
    assert static_publish.check_published(world.backend, "static", log=quiet)["ok"]
    new = stored_manifest(world.backend)
    assert {glycan: new["glycans"][glycan] for glycan in PUBLISHED} == old["glycans"]
    touched = {call[2] for call in world.published_writes()}
    assert not [key for key in touched if key.startswith(tuple(f"static/{glycan}/" for glycan in PUBLISHED))]
    assert world.client.calls.index(("copy_object", "bucket", "static/sync_manifest.json")) == max(
        index for index, call in enumerate(world.client.calls) if call[0] in WRITES and call[2].startswith("static/"))

    # what is stored is the full bake of the same tree
    reference = tmp_path / "reference"
    shutil.copytree(world.static, reference)
    shutil.copytree(folder, reference / GS)
    full_bake(reference)
    for name, content in root_files(reference).items():
        stored = world.backend.read_binary(f"static/{name}")
        if name == "sync_manifest.json":
            assert manifest_without_times(stored) == manifest_without_times(content)
        else:
            assert stored == content, name

    # the raw entry and its record in the index
    name = GLYCAM_NAME
    assert world.keys(f"raw/{name}/") == [f"raw/{name}/{item}" for item in (
        "info.json", "manifest.json", "structure.mol2", "topology.pdb", "trajectory.xtc")]
    assert world.backend.read_binary(f"raw/{name}/manifest.json") == (bundle_dir / "archive" / "manifest.json").read_bytes()
    index = json.loads(world.backend.read_binary("raw/index.json"))
    assert [item["glycam_name"] for item in index["records"]] == sorted([name, publish_world.OTHER_RAW_NAME])
    assert index["entries"] == 2 and index["totals"]["frames"] == sum(item["frames"] for item in index["records"])
    mine = next(item for item in index["records"] if item["glycam_name"] == name)
    assert [mine] == archive.build_index(synthetic_entry["archive"].parent)[0]["records"]   # the record of the converter
    assert archive_transfer.compare_with_index(world.backend, "raw", index)["ok"]
    assert world.keys("raw-history/") == []

    # nothing is left of the run but the record of the entry
    assert world.keys("publish-staging/") == [] and world.keys(STATE) == [f"{STATE}/entries/{GS}.json"]
    kept = json.loads(world.backend.read_binary(f"{STATE}/entries/{GS}.json"))
    assert kept["status"] == "published" and kept["glycam_name"] == name and kept["entry"]["hash"] == approved
    assert kept["global_hash"] == new["global_hash"] and kept["global_hash_before"] == old["global_hash"]
    assert kept["lookups"]["counts"]["ok"] > 0 and kept["submission"]["glycoshape_id"] == GS
    journal = json.loads((world.tmp / "work" / "journal.json").read_text())
    assert journal["status"] == "published" and sorted(journal["steps"]) == ["complete", "derive", "raw", "static"]
    assert report["transfers"]["uploaded_bytes"] > report["entry"]["total_size"]


def test_publishing_again_changes_nothing(world, bundle_dir):
    first = world.publish(bundle_dir)
    stored = stored_tree(world.backend, "")
    world.client.calls.clear()
    again = world.publish(bundle_dir)                                       # the same command, the same work folder
    assert again["status"] == "already-published" and again["ok"]
    assert again["static"]["target_global_hash"] == first["static"]["target_global_hash"]
    assert world.published_writes() == [] and stored_tree(world.backend, "") == stored
    assert again["transfers"]["uploaded_bytes"] < 2000                      # the lock, nothing else
    # from another work folder (the first one is gone): completed again, the same bytes, nothing to do
    world.client.calls.clear()
    fresh = world.publish(bundle_dir, work_dir=world.tmp / "another-work")
    assert fresh["status"] == "already-published" and fresh["entry"]["hash"] == first["entry"]["hash"]
    assert world.published_writes() == [] and stored_tree(world.backend, "") == stored
    # as a replacement it is the same: nothing changed
    assert world.publish(bundle_dir, action="replace")["status"] == "already-published"


def test_the_root_files_of_the_last_publication_are_kept_for_the_next(world, bundle_dir, tmp_path, second_bundle):
    cache = tmp_path / "cache"
    world.publish(bundle_dir, cache_dir=cache)
    fetched = {call[2] for call in world.client.calls if call[0] == "download_file"}
    assert fetched == {"static/GLYCOSHAPE.json", "static/GlycoShape.zip", "static/missing_glytoucan.txt", "static/missing_wurcs.txt"}
    assert len(list(cache.iterdir())) == 4                                  # inventory, master archive, two reports
    world.client.calls.clear()
    second = world.publish(second_bundle, record("GS01068", SECOND_NAME), work_dir=tmp_path / "work-2", cache_dir=cache)
    assert second["status"] == "published"
    # the root files the stored manifest lists are the ones the first publication derived: not fetched again
    assert [call for call in world.client.calls if call[0] == "download_file"] == []
    assert world.reader.try_sync() is None and world.reader.glycans() == sorted(PUBLISHED + (GS, "GS01068"))
    assert len(list(cache.iterdir())) == 4                                  # those of the database as it is now
    index = json.loads(world.backend.read_binary("raw/index.json"))
    assert [item["glycam_name"] for item in index["records"]] == sorted([GLYCAM_NAME, SECOND_NAME, publish_world.OTHER_RAW_NAME])


# ---------------------------------------------------------------------------
# Interruptions
# ---------------------------------------------------------------------------
class Killed:
    """The process dies at its n-th writing request: that request and every later request fail."""

    def __init__(self, client, at):
        self.at, self.seen, self.dead = at, 0, False
        client.before = self

    def __call__(self, operation, bucket, key):
        if operation in WRITES:
            self.seen += 1
            if self.seen > self.at:
                self.dead = True
        if self.dead:
            raise Interrupted(f"killed at writing request {self.at + 1}: {operation} {key}")


def check_interruptions(objects, static_dir, tmp_path, monkeypatch, run, work_template, expected_glycans, old_glycans):
    """Kill the run at every writing request; what readers see then; the same command finishes it."""
    monkeypatch.setattr(entry_publish, "_process_alive", lambda pid: False)    # the killed run is gone
    prepared = {"objects": objects, "static": static_dir}
    reference = World(prepared, tmp_path / "reference")
    shutil.copytree(work_template, reference.tmp / "work")
    old_hash = reference.reader.global_hash
    reference.client.calls.clear()
    run(reference)
    points = len(reference.writes())
    final = {"static": set(reference.keys("static/")), "raw": set(reference.keys("raw/")),
             "manifest": manifest_without_times(reference.backend.read_binary("static/sync_manifest.json")),
             "state": set(reference.keys("publish-state/"))}
    new_hash = final["manifest"]["global_hash"]
    assert points > 20 and new_hash != old_hash
    seen = {"old": 0, "window": 0, "new": 0, "window_failed": 0}
    for at in range(points):
        world = World(prepared, tmp_path / f"at-{at}")
        shutil.copytree(work_template, world.tmp / "work")
        assert world.reader.global_hash == old_hash and world.reader.glycans() == old_glycans
        Killed(world.client, at)
        try:
            run(world)
        except (PublishEntryError, Interrupted):
            pass
        else:
            assert at == points - 1, at                                         # only the release of the lock was lost
            assert f"{STATE}/lock.json" in world.keys(STATE)
        world.client.before = None

        # what the website sees at this point
        stored_hash = stored_manifest(world.backend)["global_hash"]
        pending = f"{STATE}/pending.json" in world.keys(STATE)
        assert stored_hash in (old_hash, new_hash), at
        behind = world.reader.fresh(world.tmp / "website-behind")              # has nothing: fetches every listed file
        if stored_hash == new_hash:
            seen["new"] += 1
            assert world.reader.try_sync() is None and behind.try_sync() is None, at
            assert world.reader.global_hash == new_hash and world.reader.glycans() == expected_glycans
        elif not pending:
            seen["old"] += 1                                                    # before the intent: exactly the old database
            assert world.reader.try_sync() is None and behind.try_sync() is None, at
            assert world.reader.global_hash == behind.global_hash == old_hash and behind.glycans() == old_glycans
        else:
            seen["window"] += 1                                                 # switching: old manifest, some files replaced
            assert world.reader.try_sync() is None and world.reader.global_hash == old_hash    # up to date: nothing to do
            failure = behind.try_sync()                                         # behind: gives up and keeps what it has
            seen["window_failed"] += failure is not None
            assert failure is None or "download verification failed" in failure, failure

        # the same command again finishes the publication
        report = run(world)
        assert report["ok"] and report["status"] in ("published", "already-published", "withdrawn", "already-withdrawn"), at
        assert set(world.keys("static/")) == final["static"], at
        assert set(world.keys("raw/")) == final["raw"], at
        assert manifest_without_times(world.backend.read_binary("static/sync_manifest.json")) == final["manifest"], at
        assert set(world.keys("publish-state/")) == final["state"] and world.keys("publish-staging/") == [], at
        assert world.reader.try_sync() is None and behind.try_sync() is None
        assert world.reader.global_hash == behind.global_hash == new_hash and behind.glycans() == expected_glycans
        assert static_publish.check_published(world.backend, "static", quick=True, log=quiet)["ok"]
        shutil.rmtree(world.tmp)
    assert seen["old"] >= 5 and seen["window"] >= 3 and seen["new"] >= 3 and seen["window_failed"] >= 1, seen
    print(f"killed at each of {points} writing requests: {seen}")
    return seen


@pytest.fixture(scope="module")
def completed_work(prepared, tmp_path_factory, synthetic_entry):
    """A work folder in which the entry is completed (by a dry run), so the interruption tests do not repeat that."""
    root = tmp_path_factory.mktemp("completed-work")
    bundle = make_bundle(root / "bundle", synthetic_entry["result"], synthetic_entry["archive"])
    with pytest.MonkeyPatch.context() as patching:
        fake_services.install(patching)
        World(prepared, root / "world").publish(bundle, dry_run=True, work_dir=root / "work")
    shutil.rmtree(root / "work" / "base")
    shutil.rmtree(root / "work" / "derived")
    return {"bundle": bundle, "work": root / "work"}


@pytest.fixture(scope="module")
def after_publication(prepared, completed_work, tmp_path_factory):
    """The bucket after the entry was published."""
    root = tmp_path_factory.mktemp("after-publication")
    with pytest.MonkeyPatch.context() as patching:
        fake_services.install(patching)
        world = World(prepared, root)
        shutil.copytree(completed_work["work"], root / "work")
        assert world.publish(completed_work["bundle"])["status"] == "published"
    return {"objects": world.client.objects, "static": prepared["static"]}


def test_a_publication_killed_at_any_request_leaves_the_old_or_the_new_database(prepared, completed_work, tmp_path, monkeypatch):
    fake_services.install(monkeypatch)
    seen = check_interruptions(
        prepared["objects"], prepared["static"], tmp_path, monkeypatch,
        lambda world: world.publish(completed_work["bundle"]), completed_work["work"],
        expected_glycans=sorted(PUBLISHED + (GS,)), old_glycans=sorted(PUBLISHED))
    assert seen["old"] > 30                                                    # the raw entry and the 25 new files


def test_a_withdrawal_killed_at_any_request_leaves_the_old_or_the_new_database(after_publication, tmp_path, monkeypatch):
    fake_services.install(monkeypatch)
    empty = tmp_path / "empty-work"
    empty.mkdir()
    seen = check_interruptions(
        after_publication["objects"], after_publication["static"], tmp_path, monkeypatch,
        lambda world: entry_publish.withdraw_entry(GS, world.target, world.tmp / "work", reason="test", log=quiet),
        empty, expected_glycans=sorted(PUBLISHED), old_glycans=sorted(PUBLISHED + (GS,)))
    assert seen["new"] > 25                                                    # the folder is deleted after the manifest


def test_a_replacement_killed_at_any_request_leaves_the_old_or_the_new_database(
        after_publication, extended_bundle, tmp_path, monkeypatch):
    """The replacing entry extends the raw entry, changes every file of the folder and has no SNFG image."""
    fake_services.install(monkeypatch, snfg=False)
    template = tmp_path / "template"
    World(after_publication, tmp_path / "prepare").publish(
        extended_bundle, record(), action="replace", dry_run=True, work_dir=template, policy=Policy(allow_incomplete=True))
    shutil.rmtree(template / "base")
    shutil.rmtree(template / "derived")
    seen = check_interruptions(
        after_publication["objects"], after_publication["static"], tmp_path, monkeypatch,
        lambda world: world.publish(extended_bundle, record(), action="replace", policy=Policy(allow_incomplete=True)),
        template, expected_glycans=sorted(PUBLISHED + (GS,)), old_glycans=sorted(PUBLISHED + (GS,)))
    assert seen["window"] > 15                                                 # the published files of the folder are switched


def test_an_interrupted_switch_is_reported_and_completed_by_recover(world, bundle_dir, tmp_path):
    def stop_in_the_window(name):
        if name == "switched:static/GLYCOSHAPE.json":
            raise Interrupted("stopped after the first file was switched")

    old_hash = world.reader.global_hash
    with pytest.raises(PublishEntryError, match="stopped while the changed files were switched .* run the command again"):
        world.publish(bundle_dir, checkpoint=stop_in_the_window)
    assert f"{STATE}/lock.json" not in world.keys(STATE)                    # the run ended: its lock is released
    report = entry_publish.status(world.target, GS, check=True)
    assert not report["ok"] and report["pending"]["action"] == "publish" and report["pending"]["glycoshape_id"] == GS
    assert report["database"]["global_hash"] == old_hash and report["entry"]["state"] == "not published"
    assert any("interrupted publish of GS01067 is pending" in problem for problem in report["problems"])
    assert "GLYCOSHAPE.json is not stored as the manifest describes it" in report["problems"]
    assert world.reader.fresh(tmp_path / "behind").try_sync() == (
        "GAP manifest is invalid: download verification failed for GLYCOSHAPE.json")
    # a dry run of anything says so and plans nothing on top of it
    dry = world.publish(bundle_dir, dry_run=True, work_dir=tmp_path / "dry")
    assert not dry["ok"] and "is pending" in dry["refused"][0]

    done = entry_publish.recover(world.target, log=quiet)
    assert done["action"] == "publish" and done["glycoshape_id"] == GS and done["manifest_written"]
    assert entry_publish.recover(world.target, log=quiet) is None           # nothing left to recover
    report = entry_publish.status(world.target, GS, check=True)
    assert report["ok"] and report["pending"] is None and report["lock"] is None
    assert report["entry"]["state"] == "published" and report["entry"]["files_stored"] and report["root_files_stored"]
    assert report["entry"]["raw"]["stored"] and report["entry"]["record"]["action"] == "publish"
    assert report["entry"]["unlisted_objects"] == [] and world.keys("publish-staging/") == []
    assert world.reader.try_sync() is None and world.reader.glycans() == sorted(PUBLISHED + (GS,))
    assert world.publish(bundle_dir)["status"] == "already-published"


def test_a_publication_that_was_interrupted_is_completed_before_the_next_one(world, bundle_dir, second_bundle, tmp_path):
    def stop_in_the_window(name):
        if name == "switched:static/GlycoShape.zip":
            raise Interrupted("stopped")

    with pytest.raises(PublishEntryError):
        world.publish(bundle_dir, checkpoint=stop_in_the_window)
    messages = []
    report = world.publish(second_bundle, record("GS01068", SECOND_NAME), work_dir=tmp_path / "work-2", log=messages.append)
    assert report["status"] == "published"
    assert any("completing the interrupted publish of GS01067" in message for message in messages)
    assert world.reader.try_sync() is None and world.reader.glycans() == sorted(PUBLISHED + (GS, "GS01068"))
    assert sorted(json.loads(world.backend.read_binary("static/GLYCOSHAPE.json"))) == sorted(PUBLISHED + (GS, "GS01068"))


# ---------------------------------------------------------------------------
# Replace
# ---------------------------------------------------------------------------
def test_a_replacement_with_other_metadata_switches_only_what_changed(world, bundle_dir, tmp_path):
    world.publish(bundle_dir)
    world.reader.sync()
    old = stored_manifest(world.backend)
    world.client.calls.clear()
    curated = {"common_names": ["Mannobiose"], "aliases": [], "description": "Corrected by the curator.", "keywords": []}
    report = world.publish(bundle_dir, record(curated=curated), action="replace", work_dir=tmp_path / "work-replace")
    assert report["status"] == "published" and report["raw"]["relation"] == "same" and report["raw"]["upload"] == []
    assert report["static"]["upload_new"] == 0 and report["static"]["unchanged"] == 23
    assert report["static"]["stage_and_switch"] == [
        f"static/{GS}/{GS}.zip", f"static/{GS}/data.json", "static/GLYCOSHAPE.json", "static/GlycoShape.zip",
        "static/GLYCOSHAPE_RDF.ttl"]
    assert not [call for call in world.published_writes() if call[2].startswith("raw/")]
    new = stored_manifest(world.backend)
    changed = {name for name, entry in new["glycans"][GS]["files"].items() if entry != old["glycans"][GS]["files"][name]}
    assert changed == {"data.json", f"{GS}.zip"} and new["glycans"][GS]["hash"] != old["glycans"][GS]["hash"]
    assert world.reader.try_sync() is None
    data = json.loads((world.reader.folder / GS / "data.json").read_text())
    assert data["search_meta"]["description"] == "Corrected by the curator." and data["search_meta"]["common_names"] == ["Mannobiose"]
    assert json.loads(world.backend.read_binary("static/GLYCOSHAPE.json"))[GS] == data
    assert json.loads(world.backend.read_binary(f"{STATE}/entries/{GS}.json"))["action"] == "replace"
    assert static_publish.check_published(world.backend, "static", log=quiet)["ok"]


def test_a_replacement_by_an_extended_entry_overwrites_the_raw_entry_and_keeps_its_manifest(
        world, bundle_dir, extended_bundle, tmp_path):
    world.publish(bundle_dir)
    first_manifest = (bundle_dir / "archive" / "manifest.json").read_bytes()
    first_sha256 = hashlib.sha256(first_manifest).hexdigest()
    report = world.publish(extended_bundle, record(simulation={"simulation_length_ns": 3100.0}), action="replace",
                           work_dir=tmp_path / "work-extended")
    assert report["status"] == "published" and report["raw"]["relation"] == "extension"
    assert report["raw"]["previous_manifest_sha256"] == first_sha256
    assert report["raw"]["history"] == [f"raw-history/{GLYCAM_NAME}/{first_sha256}/manifest.json"]
    assert set(report["raw"]["upload"]) == {"info.json", "trajectory.xtc", "manifest.json"}      # topology and MOL2 are the same
    # the previous trajectory is gone (the new one holds its frames); the manifest that described it is kept
    assert world.keys("raw-history/") == report["raw"]["history"]
    assert world.backend.read_binary(report["raw"]["history"][0]) == first_manifest
    stored = json.loads(world.backend.read_binary(f"raw/{GLYCAM_NAME}/manifest.json"))
    assert stored["frames"] == 600
    assert world.backend.read_binary(f"raw/{GLYCAM_NAME}/trajectory.xtc") == (extended_bundle / "archive" / "trajectory.xtc").read_bytes()
    index = json.loads(world.backend.read_binary("raw/index.json"))
    assert next(item for item in index["records"] if item["glycam_name"] == GLYCAM_NAME)["frames"] == 600
    assert archive_transfer.compare_with_index(world.backend, "raw", index)["ok"]
    assert world.reader.try_sync() is None
    assert json.loads((world.reader.folder / GS / "data.json").read_text())["archetype"]["length"] == 3.1
    assert (world.reader.folder / GS / "output" / "pca.csv").read_bytes() == (extended_bundle / "static" / "output" / "pca.csv").read_bytes()
    assert static_publish.check_published(world.backend, "static", log=quiet)["ok"]
    # the first bundle does not extend what is stored now: it cannot come back by accident
    with pytest.raises(Refused, match="replacing it needs --overwrite-raw"):
        world.publish(bundle_dir, action="replace", work_dir=tmp_path / "work-back")


def test_a_raw_entry_is_only_overwritten_when_asked_and_then_kept_in_the_history(world, bundle_dir, extended_bundle, tmp_path):
    world.publish(bundle_dir)
    unrelated = tmp_path / "unrelated"                                      # another trajectory that extends nothing
    shutil.copytree(extended_bundle, unrelated)
    write_bundle_json(unrelated, extends=None)
    stored_before = stored_tree(world.backend, "raw/")
    world.client.calls.clear()
    with pytest.raises(Refused, match="a raw entry with other files is stored for this glycan and the bundle does not extend it"):
        world.publish(unrelated, action="replace", work_dir=tmp_path / "work-unrelated")
    assert world.published_writes() == [] and stored_tree(world.backend, "raw/") == stored_before
    # a bundle that says it extends another manifest than the stored one
    write_bundle_json(unrelated, extends={"glycam_name": GLYCAM_NAME, "manifest_sha256": "0" * 64, "frames": 100})
    with pytest.raises(Refused, match="the bundle extends the raw entry whose manifest has SHA-256 0000000000000000"):
        world.publish(unrelated, action="replace", work_dir=tmp_path / "work-unrelated-2")
    write_bundle_json(unrelated, extends=None)
    report = world.publish(unrelated, action="replace", work_dir=tmp_path / "work-unrelated-3", policy=Policy(overwrite_raw=True))
    assert report["status"] == "published" and report["raw"]["relation"] == "overwrite"
    previous = hashlib.sha256((bundle_dir / "archive" / "manifest.json").read_bytes()).hexdigest()
    kept = {key[len(f"raw-history/{GLYCAM_NAME}/{previous}/"):]: content for key, content in stored_tree(world.backend, "raw-history/").items()}
    assert kept == {key[len(f"raw/{GLYCAM_NAME}/"):]: content for key, content in stored_before.items()
                    if key.startswith(f"raw/{GLYCAM_NAME}/")}                    # every file of the replaced entry


# ---------------------------------------------------------------------------
# Withdraw
# ---------------------------------------------------------------------------
def test_withdrawing_publishes_a_manifest_without_the_entry_and_then_deletes_its_folder(world, bundle_dir, second_bundle, tmp_path):
    world.publish(bundle_dir)
    world.reader.sync()
    raw_before = stored_tree(world.backend, "raw/")
    dry = world.withdraw(GS, dry_run=True)
    assert dry["ok"] and dry["status"] == "planned" and dry["static"]["delete_afterwards"] == 25
    assert world.reader.try_sync() is None and GS in world.reader.glycans()
    world.client.calls.clear()
    report = world.withdraw(GS, glycam_name=GLYCAM_NAME, reason="superseded")
    assert report["status"] == "withdrawn" and report["glycam_name"] == GLYCAM_NAME
    calls = world.client.calls
    manifest_at = calls.index(("copy_object", "bucket", "static/sync_manifest.json"))
    deletions = [index for index, call in enumerate(calls) if call[0] == "delete_object" and call[2].startswith(f"static/{GS}/")]
    assert len(deletions) == 25 and min(deletions) > manifest_at            # the manifest first, then the folder
    assert world.keys(f"static/{GS}/") == [] and stored_tree(world.backend, "raw/") == raw_before   # the raw data stays
    assert world.reader.try_sync() is None and world.reader.glycans() == sorted(PUBLISHED)
    assert sorted(json.loads(world.backend.read_binary("static/GLYCOSHAPE.json"))) == sorted(PUBLISHED)
    assert static_publish.check_published(world.backend, "static", log=quiet)["ok"]
    withdrawn = json.loads(world.backend.read_binary(f"{STATE}/withdrawn.json"))["withdrawn"]
    assert withdrawn[GS]["glycam_name"] == GLYCAM_NAME and withdrawn[GS]["reason"] == "superseded"
    kept = json.loads(world.backend.read_binary(f"{STATE}/entries/{GS}.json"))
    assert kept["status"] == "withdrawn" and kept["action"] == "withdraw"
    state = entry_publish.status(world.target, GS)
    assert state["entry"]["state"] == "withdrawn" and state["withdrawn"] == [GS] and state["entry"]["raw"]["stored"]
    assert entry_publish.status(world.target, glycam_name=GLYCAM_NAME)["ids_of_the_name"] == [GS]

    assert world.withdraw(GS)["status"] == "already-withdrawn"              # again: nothing to do
    # the ID is never given to another glycan, and the glycan never gets another ID
    with pytest.raises(Refused, match="GS01067 was the ID of a withdrawn glycan .*an ID is never given to another glycan"):
        world.publish(second_bundle, record(GS, SECOND_NAME), work_dir=tmp_path / "work-other")
    with pytest.raises(Refused, match="this glycan was published as GS01067 and withdrawn: publish it under that ID"):
        world.publish(bundle_dir, record("GS01070"), work_dir=tmp_path / "work-other-id")
    # the same glycan can come back under its ID
    back = world.publish(bundle_dir, work_dir=tmp_path / "work-back")
    assert back["status"] == "published" and back["raw"]["relation"] == "same"
    assert json.loads(world.backend.read_binary(f"{STATE}/withdrawn.json"))["withdrawn"] == {}
    assert world.reader.try_sync() is None and world.reader.glycans() == sorted(PUBLISHED + (GS,))


def test_a_withdrawal_that_cannot_be_done_is_refused(world, bundle_dir):
    world.client.calls.clear()
    with pytest.raises(Refused, match="GS01067 is not published"):
        world.withdraw(GS)
    with pytest.raises(Refused, match="GS00002 is published for another glycan"):
        world.withdraw("GS00002", glycam_name=GLYCAM_NAME)
    with pytest.raises(PublishEntryError, match="not a GlycoShape ID"):
        world.withdraw("GS2")
    assert world.published_writes() == []
    assert world.withdraw(GS, dry_run=True)["refused"] == ["GS01067 is not published"]
    for glycan_id in PUBLISHED[:2]:
        assert world.withdraw(glycan_id)["status"] == "withdrawn"
    with pytest.raises(PublishEntryError, match="the last glycan of a database cannot be withdrawn"):
        world.withdraw(PUBLISHED[2])
    assert world.reader.try_sync() is None and world.reader.glycans() == [PUBLISHED[2]]


# ---------------------------------------------------------------------------
# Checks before anything is written
# ---------------------------------------------------------------------------
def refused(world, bundle, reason, submission=None, action="publish", **options):
    """The publication is refused with this reason, as an error and in the report of a dry run; nothing is written."""
    before = stored_tree(world.backend, "")
    world.client.calls.clear()
    options.setdefault("work_dir", world.tmp / f"refused-{len(list(world.tmp.iterdir()))}")
    with pytest.raises(Refused, match=reason) as error:
        world.publish(bundle, submission, action, **options)
    assert world.published_writes() == [] and stored_tree(world.backend, "") == before
    assert error.value.report["ok"] is False and error.value.report["refused"]
    dry = world.publish(bundle, submission, action, dry_run=True, **options)
    assert dry["ok"] is False and dry["refused"] == error.value.report["refused"]
    return error.value.report


def test_the_id_and_the_name_have_to_fit_the_published_database(world, bundle_dir, second_bundle):
    refused(world, bundle_dir, "GS00002 is the ID of another glycan: DGalpb1-3DGlcpb1-OH", record("GS00002"))
    refused(world, bundle_dir, "GS01067 is not published: use publish", action="replace")
    world.publish(bundle_dir)
    refused(world, bundle_dir, "this glycan is published as GS01067", record("GS01099"))
    refused(world, bundle_dir, "this glycan is published as GS01067", record("GS01099"), action="replace")
    changed = record(curated={"description": "Another text."})
    refused(world, bundle_dir, "GS01067 is already published for this glycan with other content: use replace", changed)
    refused(world, second_bundle, "GS01067 is the ID of another glycan: DManpa1-2DManpa1-OH", record(GS, SECOND_NAME), action="replace")


def test_the_bundle_has_to_be_acceptable(world, bundle_dir, tmp_path, monkeypatch):
    refused(world, bundle_dir, r"the bundle was computed for levels \[1, 2\] and the database has \[1, 2, 3\]",
            policy=Policy(levels=(1, 2, 3)))
    refused(world, bundle_dir, "the archive entry was written by converter '1.0.0'; accepted: 0.9",
            policy=Policy(converter_versions=("0.9",)))
    version = json.loads((bundle_dir / "bundle.json").read_text())["pipeline"]["version"]
    refused(world, bundle_dir, "pipeline version .* of the bundle is not one of the accepted: abc\\*, 0123",
            policy=Policy(pipeline_versions=("abc*", "0123")))
    assert world.publish(bundle_dir, dry_run=True, policy=Policy(pipeline_versions=("abc*", version[:8] + "*")))["ok"]
    refused(world, bundle_dir, "pipeline commit .* is not one of the accepted", policy=Policy(pipeline_commits=("deadbeef*",)))
    refused(world, bundle_dir, "the folder has the hash .* and 0000000000000000... was approved", expect_hash="0" * 64)

    dirty = tmp_path / "dirty"
    shutil.copytree(bundle_dir, dirty)
    pipeline = json.loads((dirty / "bundle.json").read_text())["pipeline"]
    write_bundle_json(dirty, pipeline={**pipeline, "uncommitted_changes": True})
    assert "the bundle was made from a checkout with uncommitted changes" in world.publish(dirty, dry_run=True)["warnings"]
    refused(world, dirty, "uncommitted changes", policy=Policy(require_clean=True))
    write_bundle_json(dirty, pipeline={"converter_version": pipeline["converter_version"]})
    refused(world, dirty, "does not say which pipeline made it")

    # a bundle or a submission that cannot be read never reaches the storage
    world.client.calls.clear()
    with pytest.raises(entry_complete.SubmissionError, match="simulation.temperature is not accepted"):
        world.publish(bundle_dir, record(simulation={"temperature": 300}))
    (dirty / "static" / "output" / "pca.csv").write_text("changed after the bundle was made")
    with pytest.raises(entry_complete.BundleError, match="does not have the (size|SHA-256) bundle.json records"):
        world.publish(dirty)
    assert world.client.calls == []


def test_an_entry_without_identity_or_image_is_published_only_when_allowed(world, bundle_dir, monkeypatch):
    fake_services.install(monkeypatch, online=False, crabwurcs=False, snfg=False)
    report = refused(world, bundle_dir, "the entry is not complete: snfg.svg is missing", offline=True)
    assert len(report["refused"]) == 2 and "neither a WURCS nor a GlyTouCan accession" in report["refused"][1]
    assert report["entry"]["lookups"]["offline"] and "archetype.wurcs" in report["entry"]["gaps"]
    done = world.publish(bundle_dir, offline=True, policy=Policy(allow_incomplete=True), work_dir=world.tmp / "allowed")
    assert done["status"] == "published" and done["entry"]["file_count"] == 24
    assert world.reader.try_sync() is None and not (world.reader.folder / GS / "snfg.svg").exists()
    # the three variants without WURCS and accession are in the report that needs neither
    assert world.backend.read_binary("static/missing_wurcs.txt").decode().count(GS) == 3
    kept = json.loads(world.backend.read_binary(f"{STATE}/entries/{GS}.json"))
    assert kept["lookups"]["offline"] and "wurcs2glytoucan" not in kept["lookups"]["unanswered"]   # nothing to look up
    assert "archetype.glytoucan" in kept["gaps"]                            # what a later refresh has to fill


def test_what_is_stored_has_to_be_what_its_manifest_describes(world, bundle_dir):
    inventory = world.backend.read_binary("static/GLYCOSHAPE.json")
    world.backend.write_binary("static/GLYCOSHAPE.json", inventory + b"\n")        # replaced by something else
    refused(world, bundle_dir, "the stored static/GLYCOSHAPE.json is not the file the stored manifest describes")
    world.backend.write_binary("static/GLYCOSHAPE.json", inventory)
    index = world.backend.read_binary("raw/index.json")
    world.backend.unlink("raw/index.json")
    refused(world, bundle_dir, "raw/index.json does not exist while 1 other entries are stored")
    world.backend.write_binary("raw/index.json", b'{"schema": "something else"}')
    refused(world, bundle_dir, "raw/index.json is not a glycoshape-raw-trajectory-index file")
    world.backend.write_binary("raw/index.json", index)
    world.backend.unlink("static/sync_manifest.json")
    refused(world, bundle_dir, "no database is published under static/")


def test_an_interrupted_raw_upload_is_continued(world, bundle_dir, tmp_path):
    def stop_after_the_trajectory(name):
        if name == "raw:uploaded:trajectory.xtc":
            raise Interrupted("stopped")

    with pytest.raises(PublishEntryError, match="the raw entry was not stored completely .* the published database has not changed"):
        world.publish(bundle_dir, checkpoint=stop_after_the_trajectory)
    assert f"raw/{GLYCAM_NAME}/manifest.json" not in world.keys("raw/")      # an entry without manifest is not complete
    assert world.reader.fresh(tmp_path / "behind").try_sync() is None
    again = world.publish(bundle_dir)
    assert again["status"] == "published" and again["raw"]["relation"] == "incomplete"
    assert again["raw"]["upload"] == ["manifest.json"] and "trajectory.xtc" in again["raw"]["unchanged"]


# ---------------------------------------------------------------------------
# Two publishers
# ---------------------------------------------------------------------------
def lock_of(world, **changes):
    document = json.loads(world.backend.read_binary(f"{STATE}/lock.json"))
    document.update(changes)
    world.backend.write_binary(f"{STATE}/lock.json", json.dumps(document).encode())
    return document


def test_a_second_publisher_waits_for_the_lock_of_the_first(world, bundle_dir, tmp_path):
    entry_publish.acquire_lock(world.target, world.store(), "elsewhere:77:abcd1234", "replace", "GS00002", ttl=600, log=quiet)
    lock_of(world, host="elsewhere")
    stored = stored_tree(world.backend, "")
    world.client.calls.clear()
    with pytest.raises(LockHeld, match="the database is locked by elsewhere:77:abcd1234 for replace GS00002 until"):
        world.publish(bundle_dir)
    with pytest.raises(LockHeld):
        world.withdraw("GS00002")
    assert world.writes() == [] and stored_tree(world.backend, "") == stored
    dry = world.publish(bundle_dir, dry_run=True)                            # a dry run needs no lock, and says who holds it
    assert dry["ok"] and dry["lock"]["owner"] == "elsewhere:77:abcd1234"
    assert any("locked by elsewhere:77:abcd1234" in warning for warning in dry["warnings"])
    assert entry_publish.status(world.target)["lock"]["expired"] is False

    # a process of this machine that still runs holds its lock; one that is gone does not
    lock_of(world, host=__import__("socket").gethostname(), pid=os.getpid())
    with pytest.raises(LockHeld):
        world.publish(bundle_dir)
    messages = []
    lock_of(world, pid=2 ** 22 + 12345)
    assert world.publish(bundle_dir, log=messages.append)["status"] == "published"
    assert any("taking over the lock of elsewhere:77:abcd1234: its process on this machine is gone" in m for m in messages)
    assert f"{STATE}/lock.json" not in world.keys(STATE)


def test_an_expired_lock_is_taken_over_and_a_live_one_can_be_broken(world, bundle_dir, second_bundle, tmp_path):
    entry_publish.acquire_lock(world.target, world.store(), "elsewhere:1:aaaa", "publish", "GS09999", ttl=600, log=quiet)
    lock_of(world, host="elsewhere", expires="2020-01-01T00:00:00Z")
    assert entry_publish.status(world.target)["lock"]["expired"] is True
    assert world.publish(bundle_dir)["status"] == "published"
    entry_publish.acquire_lock(world.target, world.store(), "elsewhere:2:bbbb", "publish", "GS09999", ttl=600, log=quiet)
    lock_of(world, host="elsewhere")
    with pytest.raises(LockHeld):
        world.publish(second_bundle, record("GS01068", SECOND_NAME), work_dir=tmp_path / "work-2")
    done = world.publish(second_bundle, record("GS01068", SECOND_NAME), work_dir=tmp_path / "work-2", break_lock=True)
    assert done["status"] == "published"


def test_two_publications_in_one_process_follow_each_other(world, bundle_dir, second_bundle, tmp_path):
    reports, errors = {}, []

    def run(name, bundle, submission, work):
        try:
            reports[name] = world.publish(bundle, submission, work_dir=work)
        except Exception as exc:                                             # pragma: no cover - the assertion below shows it
            errors.append(exc)

    threads = [threading.Thread(target=run, args=("first", bundle_dir, record(), tmp_path / "work-a")),
               threading.Thread(target=run, args=("second", second_bundle, record("GS01068", SECOND_NAME), tmp_path / "work-b"))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == [] and {report["status"] for report in reports.values()} == {"published"}
    # the one that came second was derived from the database of the first
    hashes = {name: (report["static"]["base_global_hash"], report["static"]["target_global_hash"]) for name, report in reports.items()}
    assert hashes["first"][1] == hashes["second"][0] or hashes["second"][1] == hashes["first"][0]
    assert world.reader.try_sync() is None and world.reader.glycans() == sorted(PUBLISHED + (GS, "GS01068"))
    assert sorted(stored_manifest(world.backend)["glycans"]) == sorted(PUBLISHED + (GS, "GS01068"))
    assert static_publish.check_published(world.backend, "static", log=quiet)["ok"]


class AnotherProcess:
    """The same bucket through a client of its own: what a second process is to the lock inside a process."""

    def __init__(self, client):
        self._client = client

    def __getattr__(self, name):
        return getattr(self._client, name)


def target_of_another_process(world):
    backend = S3StorageBackend(AnotherProcess(world.client), bucket="bucket", sleep=lambda seconds: None)
    return Target(static=backend, raw=backend, workers=4)


def test_a_publisher_that_lost_its_lock_stops_before_it_switches_anything(world, bundle_dir, second_bundle, tmp_path):
    """Two publishers that both got past the lock: the second one to arrive at the switch notices."""
    def another_publisher_runs(name):
        if name == "staged":                                                 # everything is uploaded, nothing is switched
            world.backend.unlink(f"{STATE}/lock.json")                       # the lock did not hold the other one back
            other = entry_publish.publish_entry(
                "publish", second_bundle, record("GS01068", SECOND_NAME), target_of_another_process(world),
                work_dir=tmp_path / "work-other", log=quiet)
            assert other["status"] == "published"

    with pytest.raises(PublishEntryError, match="the lock of the database was taken over by nobody"):
        world.publish(bundle_dir, checkpoint=another_publisher_runs)
    assert sorted(stored_manifest(world.backend)["glycans"]) == sorted(PUBLISHED + ("GS01068",))
    assert f"{STATE}/pending.json" not in world.keys(STATE)
    assert world.reader.fresh(tmp_path / "behind").try_sync() is None        # the database of the other publisher, intact
    base = stored_manifest(world.backend)["global_hash"]
    again = world.publish(bundle_dir)                                        # derived again, from the new database
    assert again["status"] == "published" and again["static"]["base_global_hash"] == base
    assert sorted(stored_manifest(world.backend)["glycans"]) == sorted(PUBLISHED + (GS, "GS01068"))
    assert world.reader.try_sync() is None and static_publish.check_published(world.backend, "static", log=quiet)["ok"]


def test_a_change_derived_from_a_database_that_was_replaced_meanwhile_is_not_switched(world, bundle_dir, second_bundle, tmp_path):
    """The last guard: the stored manifest is compared with the base of the change after the intent is claimed."""
    def another_publisher_ran_unnoticed(name):
        if name == "staged":
            mine = world.backend.read_binary(f"{STATE}/lock.json")
            world.backend.unlink(f"{STATE}/lock.json")
            entry_publish.publish_entry(
                "publish", second_bundle, record("GS01068", SECOND_NAME), target_of_another_process(world),
                work_dir=tmp_path / "work-other", log=quiet)
            world.backend.write_binary(f"{STATE}/lock.json", mine)           # as if both had held it

    with pytest.raises(PublishEntryError, match="the stored manifest changed while this change was prepared"):
        world.publish(bundle_dir, checkpoint=another_publisher_ran_unnoticed)
    assert f"{STATE}/pending.json" not in world.keys(STATE)                  # the claim is given back
    assert sorted(stored_manifest(world.backend)["glycans"]) == sorted(PUBLISHED + ("GS01068",))
    assert world.reader.fresh(tmp_path / "behind").try_sync() is None
    assert world.publish(bundle_dir)["status"] == "published"
    assert sorted(stored_manifest(world.backend)["glycans"]) == sorted(PUBLISHED + (GS, "GS01068"))
    assert world.keys("publish-staging/") == []                              # what the first attempt had staged is gone


class ConditionalClient(FakeS3Client):
    """A storage that honours If-None-Match on a put, as S3 and MinIO do."""

    honour = True

    def put_object(self, Bucket, Key, Body, Metadata=None, ContentType=None, IfNoneMatch=None):
        if IfNoneMatch == "*" and self.honour and Key in self.objects.get(Bucket, {}):
            self.calls.append(("put_object_refused", Bucket, Key))
            raise FakeClientError("PreconditionFailed", 412, "PutObject")
        return super().put_object(Bucket, Key, Body, Metadata, ContentType)


class IgnoringClient(ConditionalClient):
    """A storage that accepts the header and overwrites all the same."""

    honour = False


@pytest.mark.parametrize("client_class, honoured", [(FakeS3Client, False), (ConditionalClient, True), (IgnoringClient, False)])
def test_conditional_writes_are_used_only_where_the_storage_honours_them(prepared, tmp_path, monkeypatch, bundle_dir, client_class, honoured):
    fake_services.install(monkeypatch)
    world = World(prepared, tmp_path, client_class)
    store = world.store()
    assert entry_publish.conditional_writes(world.target, store) is honoured
    assert world.keys(STATE) == []                                          # the probe is gone
    first = {"schema": "glycoshape-publish-pending", "schema_version": 1, "owner": "a"}
    assert entry_publish._claim(world.target, store, f"{STATE}/pending.json", first, replace=False) is True
    assert entry_publish._claim(world.target, store, f"{STATE}/pending.json", {**first, "owner": "b"}, replace=False) is False
    assert json.loads(world.backend.read_binary(f"{STATE}/pending.json"))["owner"] == "a"
    world.backend.unlink(f"{STATE}/pending.json")
    world.client.calls.clear()
    assert world.publish(bundle_dir)["status"] == "published"
    refusals = [call for call in world.client.calls if call[0] == "put_object_refused"]
    assert refusals == []                                                    # nobody else was there
    assert world.reader.try_sync() is None and world.reader.glycans() == sorted(PUBLISHED + (GS,))


def test_the_prefixes_of_a_target_do_not_overlap(fake_s3):
    backend, _ = fake_s3
    assert Target(backend, backend).lock_key == "publish-state/static/lock.json"
    assert Target(backend, backend, static_prefix="/db/static/").key("GS00001/data.json") == "db/static/GS00001/data.json"
    for wrong in ({"staging_prefix": "static/staging"}, {"state_prefix": "static"}, {"history_prefix": "raw/history"},
                  {"static_prefix": ""}, {"staging_prefix": "/"}):
        with pytest.raises(PublishEntryError):
            Target(backend, backend, **wrong)
