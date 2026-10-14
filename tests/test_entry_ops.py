"""Operations on published entries (lib/entry_ops.py): update-metadata, reanalyse, renames, on an in-memory S3 client.

The bucket holds the published database of ``publish_world`` with the synthetic glycan published as
GS01067 from the bundle that ``make_entry`` made of its three replicas (segments).  ``Reader`` is the
website.  No test reaches the network; the analysis of ``reanalyse`` runs ``scripts/run_entry.py`` in a
process of its own, as in the worker.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest

from lib import archive, entry_bundle, entry_complete, entry_ops, entry_publish
from lib.entry_publish import Refused, Store, Target, new_stats
from lib.storage import S3StorageBackend

import fake_services
import publish_world
import synthetic_run
from bundle_fixtures import renamed_bundle, submission_record, write_bundle_json
from publish_world import Reader, quiet, stored_tree
from s3_fakes import FakeS3Client
from synthetic_glycan import GLYCAM_NAME

pytestmark = pytest.mark.usefixtures("no_internet")
REPO_ROOT = Path(__file__).resolve().parents[1]
GS, OTHER_GS = "GS01067", "GS01068"
STATE = "publish-state/static"
NEW_NAME = "DManpa1-3DManpa1-OH"                       # a rename that keeps the anomer
BETA_NAME = "DManpa1-2DManpb1-OH"                      # the same glycan with the other reducing-end anomer
WRITES = ("put_object", "upload_file", "copy_object", "complete_multipart_upload", "delete_object")


def changes_record(**fields):
    record = {"schema": "glycoshape-entry-changes", "schema_version": 1, "glycoshape_id": GS}
    record.update(fields)
    return record


@pytest.fixture(scope="module")
def made_bundle(tmp_path_factory):
    """The bundle of the three replicas of the synthetic run, unpacked."""
    run = synthetic_run.get(tmp_path_factory)
    return entry_bundle.open_bundle(run["bundle"], tmp_path_factory.mktemp("made") / "bundle")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory, made_bundle):
    """A database with the synthetic glycan published as GS01067; every test gets its own copy of the bucket."""
    root = tmp_path_factory.mktemp("published-ops")
    client = FakeS3Client("bucket")
    backend = S3StorageBackend(client, bucket="bucket", sleep=lambda seconds: None)
    with pytest.MonkeyPatch.context() as patching:
        fake_services.install(patching)
        built = publish_world.build(root, backend, raw_source=publish_world.other_archive(root / "stored-archive"))
        record = submission_record(GS, GLYCAM_NAME, curated={
            "common_names": ["Mannobiose"], "aliases": [], "description": "", "keywords": ["Curated:yes"]})
        report = entry_publish.publish_entry("publish", made_bundle, record, built["target"], work_dir=root / "work",
                                             log=quiet)
        assert report["status"] == "published"
    return {"objects": client.objects}


class World:
    def __init__(self, prepared, tmp_path):
        self.client = FakeS3Client("bucket")
        self.client.objects = copy.deepcopy(prepared["objects"])
        self.backend = S3StorageBackend(self.client, bucket="bucket", sleep=lambda seconds: None)
        self.target = Target(static=self.backend, raw=self.backend, workers=4)
        self.tmp = Path(tmp_path)
        self.reader = Reader(self.backend, self.tmp / "website")
        self.reader.sync()

    def writes(self):
        return [call for call in self.client.calls if call[0] in WRITES]

    def keys(self, prefix=""):
        return self.client.keys("bucket", prefix)

    def data(self, glycan_id=GS):
        return json.loads((self.reader.folder / glycan_id / "data.json").read_text())

    def update(self, changes, work="work", **options):
        options.setdefault("log", quiet)
        return entry_ops.update_metadata(changes, self.target, self.tmp / work, **options)

    def reanalyse(self, work="work", **options):
        options.setdefault("log", quiet)
        options.setdefault("workers", 2)
        return entry_ops.reanalyse(GS, self.target, self.tmp / work, **options)

    def merge(self, kept=GS, absorbed=OTHER_GS, work="work", **options):
        options.setdefault("log", quiet)
        options.setdefault("workers", 2)
        return entry_ops.merge(kept, absorbed, self.target, self.tmp / work, **options)


@pytest.fixture
def world(prepared, tmp_path, monkeypatch):
    fake_services.install(monkeypatch)
    return World(prepared, tmp_path)


def versions(world, glycan_id=GS):
    """The versions of an entry in the order they were published: each names the folder it replaced."""
    documents = [json.loads(world.backend.read_binary(key)) | {"key": key}
                 for key in world.keys(f"{STATE}/history/{glycan_id}/")]
    ordered, before = [], None
    while documents:
        following = next(item for item in documents if item.get("folder_hash_before") == before)
        documents.remove(following)
        ordered.append(following)
        before = (following.get("entry") or {}).get("hash")
    return ordered


def models(pdb_text):
    """Coordinates of every ATOM record of a PDB text, as text."""
    return [line[30:54] for line in pdb_text.splitlines() if line.startswith(("ATOM", "HETATM"))]


def residue(pdb_text, number):
    return next(line[17:20] for line in pdb_text.splitlines()
                if line.startswith(("ATOM", "HETATM")) and int(line[22:26]) == number)


# ---------------------------------------------------------------------------
# The changes record
# ---------------------------------------------------------------------------
def test_a_changes_record_is_checked_field_by_field():
    changes = entry_ops.Changes.parse(changes_record(
        names={"glycam_name": NEW_NAME, "iupac": "Man(a1-3)Man"}, glytoucan={"alpha": "G00001AA"},
        keywords={"add": ["Mannose"]}, simulation={"salt_concentration_mm": 150, "water_model": "TIP3P"},
        curator_note="fixed the linkage", public_note="Linkage corrected in 2026."))
    assert changes.glycam_name == NEW_NAME and changes.names == {"iupac": "Man(a1-3)Man"}
    assert changes.record()["names"]["keep_old_name_as_alias"] is True
    with pytest.raises(entry_complete.SubmissionError) as refused:
        entry_ops.Changes.parse(changes_record(
            simulation={"salt": 150, "simulation_length": 2.0}, glytoucan={"alpha": "not-one"},
            keywords={"replace": ["a"], "add": ["b"]}, colour="blue", names={"glycam_name": " x"}))
    message = str(refused.value)
    for expected in ("simulation.salt is not accepted", "simulation.simulation_length is not accepted",
                     "glytoucan.alpha is not a GlyTouCan accession", "keywords is either", "unknown field 'colour'",
                     "names.glycam_name must be a full GLYCAM name"):
        assert expected in message, message
    with pytest.raises(entry_complete.SubmissionError):
        entry_ops.Changes.parse({**changes_record(), "schema_version": 2})


def test_the_salt_of_every_run_is_read_back_from_a_published_entry():
    data = {"archetype": {"ID": GS, "salt": "150\u2013200", "salt_by_run": {"run_1": 200.0, "GS00453/run_1": 150},
                          "temperature": 300.0}, "search_meta": {}}
    submission, curation, notes = entry_ops.published_submission(data, GLYCAM_NAME)
    assert "salt_concentration_mm" not in submission["simulation"] and submission["simulation"]["temperature_k"] == 300.0
    assert curation["salt_by_run"] == {"run_1": 200.0, "GS00453/run_1": 150.0}
    assert any("the salt of every run is kept" in note for note in notes)
    # one salt for the entry replaces the salt of every run
    _, after = entry_ops.apply_changes(submission, curation, entry_ops.Changes(
        glycoshape_id=GS, simulation={"salt_concentration_mm": 150.0}), GLYCAM_NAME, GLYCAM_NAME)
    assert "salt_by_run" not in after
    assert entry_complete.salt_range([200.0, 150.0]) == "150\u2013200"


def test_names_of_the_reducing_end():
    assert entry_ops.reducing_end_anomer("DGalpb1-4DGlcpNAcb1-OH") == "beta"
    assert entry_ops.with_anomer("DGalpb1-4DGlcpNAcb1-OH", "alpha") == "DGalpb1-4DGlcpNAca1-OH"
    assert entry_ops.with_anomer("DNeup5Aca2-6DGalpb1-OH", "beta") == "DNeup5Aca2-6DGalpb1-OH"
    assert entry_ops.reducing_end_anomer("G77147GI") is None
    with pytest.raises(ValueError):
        entry_ops.with_anomer("G77147GI", "alpha")


# ---------------------------------------------------------------------------
# update-metadata
# ---------------------------------------------------------------------------
def test_a_dry_run_of_update_metadata_shows_the_diff_and_writes_nothing(world):
    before = stored_tree(world.backend, "")
    changes = changes_record(
        simulation={"salt_concentration_mm": 100.0, "water_model": "TIP3P"}, keywords={"add": ["Mannose oligomer"]},
        curated={"common_names": ["Mannobiose", "Man2"]}, glytoucan={"beta": "G12345AB"},
        public_note="Salt corrected from the run logs.", curator_note="ticket 17", author="admin@example")
    report = world.update(changes, dry_run=True)
    assert world.writes() == [] and stored_tree(world.backend, "") == before
    assert report["ok"] and report["status"] == "planned" and report["schema"] == "glycoshape-entry-operation"
    assert report["operation"] == "update-metadata" and report["rename"] is None
    diff = report["diff"]
    assert diff["simulation"]["salt"] == {"before": 150.0, "after": 100.0}
    assert diff["simulation"]["water"] == {"before": None, "after": "TIP3P"}
    assert diff["names"]["beta.glytoucan"]["after"] == "G12345AB"
    assert diff["search_meta"]["keywords"]["added"] == ["Mannose oligomer"]
    assert diff["search_meta"]["common_names"]["added"] == ["Man2"]
    assert diff["search_meta"]["curator_note"]["after"] == "Salt corrected from the run logs."
    assert diff["analysis"]["levels"]["level_1"]["total_variation"] == 0.0           # no new analysis
    changed = {item["path"] for item in diff["files"]["changed"]}
    assert "data.json" in changed and f"{GS}.zip" in changed and not diff["files"]["added"] + diff["files"]["removed"]
    assert not any(path.startswith(("output/pca", "output/torsion", "landscape/")) for path in changed)
    assert report["stored_folder"]["kept_with_identifier_block"] == []               # every REMARK block came off
    # only the beta PDB files name another accession
    assert {path for path in changed if path.endswith(".pdb")} == {
        f"output/level_{level}/{fmt}/beta.pdb" for level in (1, 2) for fmt in ("GLYCAM", "PDB")}
    assert "Curated:yes" in report["entry"]["search_meta"]["keywords"]                # the curated keyword stays


def test_update_metadata_publishes_a_new_version_with_its_history(world):
    changes = changes_record(simulation={"salt_concentration_mm": 100.0}, keywords={"remove": ["Curated:yes"]},
                             curator_note="ticket 17", author="curator")
    dry = world.update(changes, dry_run=True)
    report = world.update(changes, expect_hash=dry["entry"]["hash"])
    assert report["status"] == "published" and report["entry"]["hash"] == dry["entry"]["hash"]
    assert world.reader.try_sync() is None and world.reader.fresh(world.tmp / "fresh").try_sync() is None
    data = world.data()
    assert data["archetype"]["salt"] == 100.0 and "Curated:yes" not in data["search_meta"]["keywords"]
    assert data["archetype"]["name"] == GLYCAM_NAME
    history = {item["action"]: item for item in versions(world)}
    assert sorted(history) == ["publish", "update-metadata"]
    version = history["update-metadata"]
    assert version["key"].endswith(f"{report['static']['target_global_hash'][:16]}-update-metadata.json")
    # the versions form a chain: each one names the folder it replaced
    assert version["folder_hash_before"] == history["publish"]["entry"]["hash"]
    assert version["curator_note"] == "ticket 17" and version["author"] == "curator"
    assert version["folder_hash_before"] != version["entry"]["hash"] == report["entry"]["hash"]
    assert version["changes"]["keywords"] == {"remove": ["Curated:yes"]}
    record = json.loads(world.backend.read_binary(f"{STATE}/entries/{GS}.json"))
    assert record["operation"] == "update-metadata" and record["curation"]["keywords"] == {"add": [], "remove": ["Curated:yes"]}
    assert world.keys("raw-history/") == []                                          # the raw entry did not change
    # the same command again: the version is published
    assert world.update(changes)["status"] == "already-published"
    # a later operation takes the curated metadata from the record of this one
    again = world.update(changes_record(curated={"description": "Two mannoses."}), work="work-2", dry_run=True)
    assert again["curated_from"] == ["curated metadata from the record of the last publication"]
    assert "Curated:yes" not in again["entry"]["search_meta"]["keywords"]
    assert again["entry"]["simulation"]["salt"] == 100.0


def test_an_approved_dry_run_is_published_only_as_it_was_approved(world):
    changes = changes_record(simulation={"temperature_k": 310.0})
    with pytest.raises(Refused) as refused:
        world.update(changes, expect_hash="0" * 64)
    assert any("was approved" in reason for reason in refused.value.report["refused"])
    assert [item["action"] for item in versions(world)] == ["publish"]
    stale = changes_record(simulation={"temperature_k": 310.0}, expect={"folder_hash": "1" * 64})
    report = world.update(stale, work="work-2", dry_run=True)
    assert not report["ok"] and any("changed since the changes were made" in reason for reason in report["refused"])


def test_a_rename_moves_the_raw_entry_and_keeps_the_old_name_as_an_alias(world):
    old_manifest = world.backend.read_binary(f"raw/{GLYCAM_NAME}/manifest.json")
    old_sha256 = hashlib.sha256(old_manifest).hexdigest()
    old_files = json.loads(old_manifest)["files"]
    changes = changes_record(names={"glycam_name": NEW_NAME}, curator_note="the linkage is 1-3",
                             expect={"glycam_name": GLYCAM_NAME})
    dry = world.update(changes, dry_run=True)
    assert dry["ok"] and dry["rename"]["from"] == GLYCAM_NAME and dry["rename"]["to"] == NEW_NAME
    raw = dry["rename"]["raw"]
    assert raw["relation"] == "move" and raw["upload"] == ["name.json", "manifest.json"]
    assert sorted(raw["copy"]) == sorted(name for name in old_files if name != "name.json")
    assert world.writes() == []
    report = world.update(changes, expect_hash=dry["entry"]["hash"])
    assert report["status"] == "published"

    # the raw entry is under the new name only, with the same data, and the index says so
    assert world.keys(f"raw/{GLYCAM_NAME}/") == []
    new_manifest = json.loads(world.backend.read_binary(f"raw/{NEW_NAME}/manifest.json"))
    assert new_manifest["glycam_name"] == NEW_NAME and new_manifest["renamed_from"]["glycam_name"] == GLYCAM_NAME
    assert new_manifest["renamed_from"]["manifest_sha256"] == old_sha256
    for name, info in old_files.items():
        if name != "name.json":
            assert new_manifest["files"][name] == info
            head = world.backend.head(f"raw/{NEW_NAME}/{name}")
            assert (head["size"], head["sha256"]) == (info["size"], info["sha256"])
    assert json.loads(world.backend.read_binary(f"raw/{NEW_NAME}/name.json")) == {"indexOrderedSequence": NEW_NAME}
    index = json.loads(world.backend.read_binary("raw/index.json"))
    names = [item["glycam_name"] for item in index["records"]]
    assert NEW_NAME in names and GLYCAM_NAME not in names and index["entries"] == len(names)
    history = f"raw-history/raw/{GLYCAM_NAME}/{old_sha256}/"
    assert world.backend.read_binary(history + "manifest.json") == old_manifest
    renamed = json.loads(world.backend.read_binary(history + "renamed.json"))
    assert renamed["from"] == GLYCAM_NAME and renamed["to"] == NEW_NAME and renamed["glycoshape_id"] == GS

    # the published entry: new name, old one an alias; the website follows
    assert world.reader.try_sync() is None
    data = world.data()
    assert data["archetype"]["name"] == NEW_NAME and GLYCAM_NAME in data["search_meta"]["aliases"]
    assert data["alpha"]["glycam"] == "DManpa1-3DManpa1-OH" and data["beta"]["glycam"] == "DManpa1-3DManpb1-OH"
    status = entry_publish.status(world.target, GS)
    assert status["entry"]["state"] == "published" and status["entry"]["raw"]["stored"]
    # the raw entry under its new name is a valid archive entry that the next publication finds
    from lib import archive_transfer

    row = archive_transfer.download_entry(world.backend, "raw", NEW_NAME, world.tmp / "check")
    assert archive.verify_entry(world.tmp / "check" / NEW_NAME, deep=True)["ok"]
    assert row["status"] == "downloaded"
    assert world.update(changes)["status"] == "already-published"


def test_a_rename_onto_another_entry_is_refused(world):
    taken = "DGalpb1-2DGlcpb1-OH"                                     # published as GS00001
    before = stored_tree(world.backend, "")
    report = world.update(changes_record(names={"glycam_name": taken}), dry_run=True)
    assert not report["ok"] and any("published as GS00001" in reason for reason in report["refused"])
    with pytest.raises(Refused):
        world.update(changes_record(names={"glycam_name": taken}), work="work-2")
    assert stored_tree(world.backend, "") == {**before, **{key: value for key, value in stored_tree(world.backend, "").items()
                                                           if key.startswith(f"{STATE}/probe")}}
    # a raw entry of a glycan that is not published also keeps its name
    unpublished = "DGlcpa1-4DGlcpa1-OH"
    world.backend.write_binary(f"raw/{unpublished}/manifest.json", b'{"glycam_name": "DGlcpa1-4DGlcpa1-OH"}')
    report = world.update(changes_record(names={"glycam_name": unpublished}), work="work-3", dry_run=True)
    assert not report["ok"] and any("raw entry of another glycan" in reason for reason in report["refused"])
    # and a withdrawn entry keeps its name too
    entry_publish.withdraw_entry("GS00002", world.target, world.tmp / "withdraw", log=quiet)
    report = world.update(changes_record(names={"glycam_name": "DGalpb1-3DGlcpb1-OH"}), work="work-4", dry_run=True)
    assert not report["ok"] and any("withdrawn" in reason for reason in report["refused"])


def test_update_metadata_refuses_to_change_the_anomer_of_the_name(world):
    report = world.update(changes_record(names={"glycam_name": BETA_NAME}), dry_run=True)
    assert not report["ok"] and any("reanalyse" in reason for reason in report["refused"])


def test_an_entry_that_is_not_published_cannot_be_updated(world):
    report = world.update(changes_record(glycoshape_id="GS09999"), dry_run=True)
    assert not report["ok"] and report["refused"] == ["GS09999 is not published"]


def test_the_identifier_block_comes_off_the_pdb_files_exactly(world, tmp_path):
    store = Store(world.backend, new_stats())
    database = entry_publish.read_database(world.target, store, tmp_path / "db")
    folder = entry_ops.download_folder(world.target, store, database, GS, tmp_path / "stored")
    ready = entry_ops.static_ready_from_folder(folder, tmp_path / "ready", (1, 2))
    assert ready["kept_with_block"] == [] and "data.json" in ready["left_out"]
    text = (tmp_path / "ready" / "output/level_1/GLYCAM/alpha.pdb").read_text()
    assert text.startswith("REMARK   GLYCAN MULTI-MODEL PDB")
    stored = (folder / "output/level_1/GLYCAM/alpha.pdb").read_text()
    assert stored.endswith(text) and stored.startswith("REMARK     GENERATED BY")


# ---------------------------------------------------------------------------
# reanalyse
# ---------------------------------------------------------------------------
def test_reanalyse_without_a_replica(world, made_bundle):
    segments = json.loads((made_bundle / "archive" / "manifest.json").read_text())["segments"]
    dropped = segments[1]["name"]
    report = world.reanalyse(exclude=[dropped], note="replica 2 is a different stereoisomer", dry_run=True)
    assert report["ok"], report["refused"]
    assert world.writes() == []
    analysis = report["analysis"]
    assert [row["name"] for row in analysis["excluded_replicas"]] == [dropped]
    assert analysis["frame_quality"]["excluded_frames"] == segments[1]["frames"]
    assert analysis["frame_quality"]["retained_frames"] <= 480 - segments[1]["frames"]
    assert report["raw"]["relation"] == "same" and report["rename"] is None
    assert report["diff"]["analysis"]["levels"]["level_1"]["total_variation"] is not None
    assert {item["path"] for item in report["diff"]["files"]["changed"]} >= {"output/pca.csv", "output/info.json"}
    assert report["memory"]["estimate_gb"] > 0 and report["bundle"]["sha256"]
    published = world.reanalyse(exclude=[dropped], note="replica 2 is a different stereoisomer",
                                 expect_hash=report["entry"]["hash"])
    assert published["status"] == "published" and published["analysis"]["reused"]
    assert world.reader.try_sync() is None
    pca = (world.reader.folder / GS / "output" / "pca.csv").read_text().splitlines()
    source = {int(line.split(",")[4]) for line in pca[1:]}
    first, count = segments[1]["first_frame"], segments[1]["frames"]
    assert not source & set(range(first, first + count))                    # none of its frames is in the analysis
    version = versions(world)[-1]
    assert version["reanalysis"]["exclude"] == [dropped] and version["curator_note"] == "replica 2 is a different stereoisomer"
    replicas = json.loads(world.backend.read_binary(f"{STATE}/replicas/{GS}.json"))
    assert [row["retained"] for row in replicas["replicas"]][1] == 0
    assert world.keys("raw-history/") == [] and world.keys(f"raw/{GLYCAM_NAME}/")    # the raw entry keeps the replica


def test_reanalyse_with_the_simulated_anomer_renames_and_files_the_structures_by_it(world):
    before = (world.reader.folder / GS / "output/level_1/GLYCAM/alpha.pdb").read_text()
    report = world.reanalyse(simulated="beta", note="simulated as beta", dry_run=True)
    assert report["ok"], report["refused"]
    assert report["anomer"]["named"] == "alpha" and report["anomer"]["declared"] == "beta"
    assert report["anomer"]["residue_code"]["residue"] == "2MA"
    assert any("residue code 2MA" in warning for warning in report["warnings"])
    assert report["glycam_name"] == BETA_NAME and report["rename"]["raw"]["relation"] == "move"
    assert report["diff"]["names"]["archetype.name"] == {"before": GLYCAM_NAME, "after": BETA_NAME}
    published = world.reanalyse(simulated="beta", note="simulated as beta", expect_hash=report["entry"]["hash"])
    assert published["status"] == "published"
    assert world.reader.try_sync() is None
    data = world.data()
    # the old name stays findable: here it is the GLYCAM name of the alpha variant, so no alias is needed
    assert data["archetype"]["name"] == BETA_NAME and data["alpha"]["glycam"] == GLYCAM_NAME
    assert "aliases" not in data["search_meta"]
    beta =(world.reader.folder / GS / "output/level_1/GLYCAM/beta.pdb").read_text()
    alpha = (world.reader.folder / GS / "output/level_1/GLYCAM/alpha.pdb").read_text()
    # the simulated coordinates are now the beta file, under a beta residue code; the alpha one is the flipped copy
    assert models(beta) == models(before) and residue(beta, 2) == "2MB" and residue(alpha, 2) == "2MA"
    assert models(alpha) != models(beta)
    assert world.keys(f"raw/{GLYCAM_NAME}/") == [] and world.keys(f"raw/{BETA_NAME}/")
    manifest = json.loads(world.backend.read_binary(f"raw/{BETA_NAME}/manifest.json"))
    assert manifest["renamed_from"]["glycam_name"] == GLYCAM_NAME and "extends" not in manifest
    # the same declaration again changes nothing: the name says it now
    again = world.reanalyse(simulated="beta", work="work-2", dry_run=True)
    assert again["ok"] and again["anomer"]["named"] == "beta" and again["rename"] is None


def test_reanalyse_refuses_a_rename_onto_a_published_name(world, made_bundle, tmp_path):
    second = renamed_bundle(tmp_path / "second", made_bundle, BETA_NAME)
    record = submission_record(OTHER_GS, BETA_NAME)
    assert entry_publish.publish_entry("publish", second, record, world.target, work_dir=tmp_path / "w2",
                                       log=quiet)["status"] == "published"
    world.client.calls.clear()
    report = world.reanalyse(simulated="beta", dry_run=True)
    assert not report["ok"] and any(f"published as {OTHER_GS}" in reason for reason in report["refused"])


def test_reanalyse_refuses_what_it_cannot_do(world):
    report = world.reanalyse(exclude=["no-such-replica"], dry_run=True)
    assert not report["ok"] and "no replica or segment named 'no-such-replica'" in report["refused"][0]
    report = world.reanalyse(memory_limit_gb=0.5, work="work-2", dry_run=True)
    assert not report["ok"] and "the limit is 0.5 GB" in report["refused"][0]
    report = world.reanalyse(simulated="sideways", work="work-3", dry_run=True)
    assert not report["ok"]
    assert world.writes() == []


def test_a_memory_estimate_that_follows_the_measurements():
    assert 19 < entry_ops.memory_estimate_gb(160016, 172, 16) < 25
    assert 2 < entry_ops.memory_estimate_gb(50005, 62, 16) < 5


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------
#: The synthetic glycan written another way.  Here it is the name of another disaccharide, and the IUPAC name the
#: static stage derives from it is made that of GLYCAM_NAME (:func:`one_glycan`), as GS00453 and GS00303 have one.
ALT_NAME = NEW_NAME


def one_glycan(patching):
    """The two names are one glycan: the same IUPAC name, so the same archetype WURCS and accession."""
    from lib import glystatic

    canonical = glystatic.canonicalize_iupac
    patching.setattr(glystatic, "canonicalize_iupac", lambda name: canonical(GLYCAM_NAME if name == ALT_NAME else name))


def publish_second(target, made_bundle, root, glycan_id=OTHER_GS, name=ALT_NAME, salt=200.0):
    """The same simulation published once more under another ID and name (as GS00453 next to GS00303)."""
    record = submission_record(glycan_id, name)
    record["simulation"]["salt_concentration_mm"] = salt
    bundle = renamed_bundle(Path(root) / f"bundle-{glycan_id}", made_bundle, name)
    report = entry_publish.publish_entry("publish", bundle, record, target, work_dir=Path(root) / f"w-{glycan_id}",
                                         log=quiet)
    assert report["status"] == "published"


@pytest.fixture(scope="module")
def prepared_pair(tmp_path_factory, prepared, made_bundle):
    """The database of ``prepared`` with the synthetic glycan published a second time, as GS01068 (salt 200 mM)."""
    root = tmp_path_factory.mktemp("published-pair")
    client = FakeS3Client("bucket")
    client.objects = copy.deepcopy(prepared["objects"])
    backend = S3StorageBackend(client, bucket="bucket", sleep=lambda seconds: None)
    with pytest.MonkeyPatch.context() as patching:
        fake_services.install(patching)
        one_glycan(patching)
        publish_second(Target(static=backend, raw=backend, workers=4), made_bundle, root)
    return {"objects": client.objects}


@pytest.fixture
def pair(prepared_pair, tmp_path, monkeypatch):
    fake_services.install(monkeypatch)
    return World(prepared_pair, tmp_path)


def test_the_two_entries_of_a_pair_are_one_glycan(pair):
    kept, absorbed = pair.data(GS), pair.data(OTHER_GS)
    assert kept["archetype"]["wurcs"] == absorbed["archetype"]["wurcs"] and kept["archetype"]["name"] != absorbed["archetype"]["name"]
    assert (kept["archetype"]["salt"], absorbed["archetype"]["salt"]) == (150.0, 200.0)


def segment_names(made_bundle):
    return [segment["name"] for segment in json.loads((made_bundle / "archive" / "manifest.json").read_text())["segments"]]


def test_a_dry_run_of_a_merge_reports_everything_and_writes_nothing(pair, made_bundle):
    before = stored_tree(pair.backend, "")
    report = pair.merge(dry_run=True)
    assert pair.writes() == [] and stored_tree(pair.backend, "") == before
    assert not report["ok"] and report["operation"] == "merge" and report["schema"] == "glycoshape-entry-operation"
    assert report["refused"] == [f"the salt concentration differs ({GS} 150.0 mM, {OTHER_GS} 200.0 mM): "
                                 f"--accept-difference salt merges them with the value of {GS}"]
    assert report["merge"]["differences"] == [{"field": "salt", "what": "salt concentration", "unit": "mM",
                                               "kept": 150.0, "absorbed": 200.0, "accepted": False}]
    assert "entry" not in report                                    # refused before anything was analysed

    report = pair.merge(dry_run=True, accept_differences=["salt"], note="the same glycan", author="curator")
    assert report["ok"], report["refused"]
    assert pair.writes() == [] and stored_tree(pair.backend, "") == before
    assert report["status"] == "planned" and report["glycoshape_id"] == GS and report["glycam_name"] == GLYCAM_NAME
    assert f"the salt concentration differs ({GS} 150.0 mM, {OTHER_GS} 200.0 mM); accepted: every run keeps its own " \
           "(archetype.salt is their range, archetype.salt_by_run the salt of each run)" in report["warnings"]
    merge = report["merge"]
    assert (merge["from"], merge["from_glycam_name"], merge["glycam_name"]) == (OTHER_GS, ALT_NAME, GLYCAM_NAME)
    assert merge["frames"] == {"kept": 480, "absorbed": 480, "total": 960}
    names = segment_names(made_bundle)
    assert [(s["name"], s["first_frame"], s["from"]) for s in merge["segments"]] == [
        (names[0], 0, GS), (names[1], 200, GS), (names[2], 330, GS),
        (f"{OTHER_GS}/{names[0]}", 480, OTHER_GS), (f"{OTHER_GS}/{names[1]}", 680, OTHER_GS),
        (f"{OTHER_GS}/{names[2]}", 810, OTHER_GS)]
    assert merge["accepted_differences"] == ["salt"] and merge["differences"][0]["accepted"]
    assert merge["drop_runs"] is False and merge["anomers"] == {"kept": "alpha", "absorbed": "alpha"}
    assert merge["salt_by_run"] == {**{name: 150.0 for name in names}, **{f"{OTHER_GS}/{name}": 200.0 for name in names}}
    assert report["memory"]["frames"] == 960 and report["analysis"]["frame_quality"]
    # the kept ID: a new version with the sum of the lengths and its own salt; the raw entry is an extension
    assert len(report["entry"]["hash"]) == 64 and report["entry"]["simulation"]["length"] == 5.0
    assert report["diff"]["simulation"] == {"length": {"before": 2.5, "after": 5.0},
                                            "salt": {"before": 150.0, "after": "150\u2013200"}}
    assert report["raw"]["relation"] == "extension" and "trajectory.xtc" in report["raw"]["upload"]
    assert report["raw"]["history"] == [f"raw-history/raw/{GLYCAM_NAME}/"
                                        f"{report['merge']['kept_manifest_sha256']}/manifest.json"]
    # the absorbed ID: withdrawn from the database the kept version makes, its raw entry to the history
    withdrawal = report["withdrawal"]
    assert withdrawal["glycoshape_id"] == OTHER_GS and withdrawal["status"] == "planned"
    assert withdrawal["merged_into"] == GS and withdrawal["static"]["delete_afterwards"] > 0
    assert withdrawal["static"]["base_global_hash"] == report["static"]["target_global_hash"]
    raw = withdrawal["raw"]
    assert raw["relation"] == "move" and raw["from_key_prefix"] == f"raw/{ALT_NAME}/"
    assert raw["history_prefix"] == f"raw-history/raw/{ALT_NAME}/{merge['absorbed_manifest_sha256']}/"
    assert "trajectory.xtc" in raw["copy"] and "manifest.json" in raw["copy"] and raw["index"] == "2 entries"
    assert sorted(raw["delete"]) == sorted(pair.keys(f"raw/{ALT_NAME}/"))


def test_a_merge_gives_the_kept_id_the_frames_of_both_and_withdraws_the_other(pair, made_bundle):
    absorbed_manifest = pair.backend.read_binary(f"raw/{ALT_NAME}/manifest.json")
    absorbed_sha256 = hashlib.sha256(absorbed_manifest).hexdigest()
    absorbed_files = {key: pair.backend.read_binary(key) for key in pair.keys(f"raw/{ALT_NAME}/")}
    kept_manifest = pair.backend.read_binary(f"raw/{GLYCAM_NAME}/manifest.json")
    kept_sha256 = hashlib.sha256(kept_manifest).hexdigest()
    options = dict(accept_differences=["salt"], note="GS01068 is GS01067 written another way", author="curator")
    dry = pair.merge(dry_run=True, **options)
    report = pair.merge(expect_hash=dry["entry"]["hash"], **options)
    assert report["status"] == "merged" and report["kept"]["status"] == "published", report["refused"]
    assert report["entry"]["hash"] == dry["entry"]["hash"]
    assert report["withdrawal"]["status"] == "withdrawn"
    assert report["database_after"]["global_hash"] == report["withdrawal"]["static"]["target_global_hash"]
    # the dry run derived the database the merge made: the kept version first, then the withdrawal on it
    assert report["static"]["target_global_hash"] == dry["static"]["target_global_hash"]
    assert report["database_after"]["global_hash"] == dry["withdrawal"]["static"]["target_global_hash"]

    # the website: one ID for the glycan, with the frames and the length of both simulations
    assert pair.reader.try_sync() is None and pair.reader.fresh(pair.tmp / "fresh").try_sync() is None
    assert OTHER_GS not in pair.reader.glycans() and GS in pair.reader.glycans()
    assert pair.reader.global_hash == report["database_after"]["global_hash"]
    data = pair.data()
    names = segment_names(made_bundle)
    by_run = {**{name: 150.0 for name in names}, **{f"{OTHER_GS}/{name}": 200.0 for name in names}}
    assert data["archetype"]["length"] == 5.0 and data["archetype"]["name"] == GLYCAM_NAME
    assert data["archetype"]["salt"] == "150\u2013200" and data["archetype"]["salt_by_run"] == by_run
    assert list(data["archetype"])[list(data["archetype"]).index("salt") + 1] == "salt_by_run"

    # the raw entry of the kept ID holds the frames of both, and extends the one it was
    manifest = json.loads(pair.backend.read_binary(f"raw/{GLYCAM_NAME}/manifest.json"))
    assert manifest["frames"] == 960 and manifest["extends"]["manifest_sha256"] == kept_sha256
    assert manifest["merged_from"]["glycoshape_id"] == OTHER_GS and manifest["merged_from"]["manifest_sha256"] == absorbed_sha256
    names = segment_names(made_bundle)
    assert [s["name"] for s in manifest["segments"]] == names + [f"{OTHER_GS}/{name}" for name in names]
    assert [s["carried_from"]["glycam_name"] for s in manifest["segments"]] == [GLYCAM_NAME] * 3 + [ALT_NAME] * 3
    assert [s.get("merged_from", {}).get("glycoshape_id") for s in manifest["segments"]] == [None] * 3 + [OTHER_GS] * 3
    assert [s["run"]["salt_concentration_mm"] for s in manifest["segments"]] == [150.0] * 3 + [200.0] * 3
    assert pair.backend.read_binary(f"raw-history/raw/{GLYCAM_NAME}/{kept_sha256}/manifest.json") == kept_manifest
    from lib import archive_transfer

    archive_transfer.download_entry(pair.backend, "raw", GLYCAM_NAME, pair.tmp / "check")
    assert archive.verify_entry(pair.tmp / "check" / GLYCAM_NAME, deep=True)["ok"]

    # the raw entry of the absorbed ID: in the history with merged.json, gone from raw/ and from the index
    assert pair.keys(f"raw/{ALT_NAME}/") == []
    history = f"raw-history/raw/{ALT_NAME}/{absorbed_sha256}/"
    for key, content in absorbed_files.items():
        assert pair.backend.read_binary(history + key[len(f"raw/{ALT_NAME}/"):]) == content
    merged = json.loads(pair.backend.read_binary(history + "merged.json"))
    assert (merged["schema"], merged["from"], merged["into"]) == ("glycoshape-raw-merge", ALT_NAME, GLYCAM_NAME)
    assert (merged["from_glycoshape_id"], merged["into_glycoshape_id"]) == (OTHER_GS, GS)
    assert merged["manifest_sha256_from"] == absorbed_sha256
    assert merged["manifest_sha256_into"] == hashlib.sha256(pair.backend.read_binary(
        f"raw/{GLYCAM_NAME}/manifest.json")).hexdigest()
    assert (merged["author"], merged["accepted_differences"]) == ("curator", ["salt"])
    index = json.loads(pair.backend.read_binary("raw/index.json"))
    names = {item["glycam_name"]: item for item in index["records"]}
    assert ALT_NAME not in names and names[GLYCAM_NAME]["frames"] == 960 and index["entries"] == len(names)

    # the records: withdrawn.json says where the ID went; both histories say what was done
    withdrawn = json.loads(pair.backend.read_binary(f"{STATE}/withdrawn.json"))["withdrawn"]
    assert withdrawn[OTHER_GS]["merged_into"] == GS and withdrawn[OTHER_GS]["glycam_name"] == ALT_NAME
    version = versions(pair)[-1]
    assert version["operation"] == "merge" and version["merge"]["from"] == OTHER_GS
    assert version["merge"]["accepted_differences"] == ["salt"] and version["author"] == "curator"
    assert version["curator_note"] == options["note"] and version["entry"]["hash"] == report["entry"]["hash"]
    other = [json.loads(pair.backend.read_binary(key)) for key in pair.keys(f"{STATE}/history/{OTHER_GS}/")]
    assert [item["action"] for item in other if item.get("merged_into") == GS] == ["withdraw"]
    status = entry_publish.status(pair.target, OTHER_GS)
    assert status["entry"]["state"] == "withdrawn" and status["entry"]["withdrawn"]["merged_into"] == GS

    # the same command again: nothing is left to do, and nothing is written but the lock
    pair.client.calls.clear()
    again = pair.merge(**options)
    assert again["ok"] and again["status"] == "already-merged"
    assert again["withdrawal"]["status"] == "already-withdrawn" and again["withdrawal"]["raw"]["relation"] == "moved"
    assert {call[2] for call in pair.writes()} <= {f"{STATE}/lock.json"} | {
        call[2] for call in pair.writes() if call[2].startswith(f"{STATE}/probe-")}
    assert pair.merge(work="work-2", dry_run=True)["status"] == "already-merged"

    # later operations keep the salt of every run; one salt for the entry replaces it
    pair.update(changes_record(curated={"description": "Two mannoses."}), work="work-3")
    assert pair.reader.try_sync() is None and pair.data()["archetype"]["salt_by_run"] == by_run
    assert pair.data()["archetype"]["salt"] == "150\u2013200"
    pair.update(changes_record(simulation={"salt_concentration_mm": 150.0}), work="work-4")
    assert pair.reader.try_sync() is None and pair.data()["archetype"]["salt"] == 150.0
    assert "salt_by_run" not in pair.data()["archetype"]


def test_after_a_merge_the_absorbed_id_and_its_old_name_are_free_for_new_glycans(pair, made_bundle):
    pair.merge(accept_differences=["salt"])
    status = entry_publish.status(pair.target, OTHER_GS)["entry"]
    assert status["state"] == "withdrawn" and status["withdrawn"]["merged_into"] == GS and status["id_free"]
    raw_history = pair.keys(f"raw-history/raw/{ALT_NAME}/")
    old_history = pair.keys(f"{STATE}/history/{OTHER_GS}/")

    # the old name of the absorbed entry, a new glycan under a new ID: not bound to the ID it had
    publish_second(pair.target, made_bundle, pair.tmp, glycan_id="GS01070", name=ALT_NAME, salt=150.0)
    # another glycan under the absorbed ID: the ID was free; its entry of withdrawn.json goes
    new_name = "DGlcpa1-2DManpa1-OH"
    record = submission_record(OTHER_GS, new_name)
    bundle = renamed_bundle(pair.tmp / "b-new", made_bundle, new_name)
    (bundle / "static" / "output" / "replicas.json").unlink()            # a bundle without replica diagnostics
    result = json.loads((bundle / "result.json").read_text())
    result["produced"]["static"]["files"] = [item for item in result["produced"]["static"]["files"]
                                             if item["path"] != "output/replicas.json"]
    (bundle / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    write_bundle_json(bundle)
    assert pair.keys(f"{STATE}/replicas/{OTHER_GS}.json")                # those of the glycan that had the ID
    report = entry_publish.publish_entry("publish", bundle, record, pair.target, work_dir=pair.tmp / "w-new", log=quiet)
    assert report["status"] == "published"
    assert pair.keys(f"{STATE}/replicas/{OTHER_GS}.json") == []          # are not shown for the new one
    assert any(f"{OTHER_GS} was free again" in warning for warning in report["warnings"])
    assert OTHER_GS not in json.loads(pair.backend.read_binary(f"{STATE}/withdrawn.json"))["withdrawn"]
    assert pair.reader.try_sync() is None and pair.data(OTHER_GS)["archetype"]["name"] == new_name
    assert pair.data("GS01070")["archetype"]["name"] == ALT_NAME
    # the records of what the ID was stay
    assert set(old_history) < set(pair.keys(f"{STATE}/history/{OTHER_GS}/"))
    assert pair.keys(f"raw-history/raw/{ALT_NAME}/") == raw_history
    assert entry_publish.status(pair.target, OTHER_GS)["entry"]["state"] == "published"

    # the merge command again leaves the new glycan of the ID, and the raw entry of the new glycan of the name, alone
    new_raw = {key: pair.backend.read_binary(key) for key in pair.keys(f"raw/{ALT_NAME}/")}
    again = pair.merge(work="work-again", accept_differences=["salt"])
    assert again["status"] == "already-merged" and again["id_taken_by"] == new_name
    assert again["withdrawal"]["raw"]["relation"] == "taken" and again["withdrawal"]["raw"]["held_by"] == ["GS01070"]
    assert new_raw and {key: pair.backend.read_binary(key) for key in pair.keys(f"raw/{ALT_NAME}/")} == new_raw
    assert pair.reader.try_sync() is None and pair.data(OTHER_GS)["archetype"]["name"] == new_name


def test_a_plain_withdrawal_still_binds_its_id_and_its_name(world, made_bundle):
    name = world.data("GS00002")["archetype"]["name"]
    entry_publish.withdraw_entry("GS00002", world.target, world.tmp / "withdraw", log=quiet)
    assert not entry_publish.status(world.target, "GS00002")["entry"]["id_free"]
    other = renamed_bundle(world.tmp / "b-other", made_bundle, "DGlcpa1-2DManpa1-OH")
    with pytest.raises(Refused) as refused:
        entry_publish.publish_entry("publish", other, submission_record("GS00002", "DGlcpa1-2DManpa1-OH"), world.target,
                                    work_dir=world.tmp / "w-other", log=quiet)
    assert any("GS00002 was the ID of a withdrawn glycan" in reason for reason in refused.value.report["refused"])
    same = renamed_bundle(world.tmp / "b-same", made_bundle, name)
    with pytest.raises(Refused) as refused:
        entry_publish.publish_entry("publish", same, submission_record("GS01070", name), world.target,
                                    work_dir=world.tmp / "w-same", log=quiet)
    assert any("published as GS00002 and withdrawn: publish it under that ID" in reason
               for reason in refused.value.report["refused"])


def test_an_interrupted_merge_is_continued_by_the_same_command(pair):
    options = dict(accept_differences=["salt"])
    stops = {"pending-cleared": 1, "merge-move:index": 1}

    def checkpoint(name):
        if stops.get(name):
            stops[name] -= 1
            raise RuntimeError(f"stopped at {name}")

    # stopped when the kept version is published, before the absorbed ID is withdrawn
    with pytest.raises(entry_publish.PublishEntryError):
        pair.merge(checkpoint=checkpoint, **options)
    assert pair.reader.try_sync() is None
    assert OTHER_GS in pair.reader.glycans() and pair.data()["archetype"]["length"] == 5.0
    # stopped while the raw entry of the absorbed ID was moved: copied to the history, out of the index, not deleted
    with pytest.raises(entry_ops.OperationError):
        pair.merge(checkpoint=checkpoint, **options)
    assert pair.reader.try_sync() is None and OTHER_GS not in pair.reader.glycans()
    assert pair.keys(f"raw/{ALT_NAME}/")                       # nothing is deleted before the index is written
    report = pair.merge(checkpoint=checkpoint, **options)
    assert report["status"] == "merged" and report["kept"]["status"] == "already-published"
    assert report["withdrawal"]["status"] == "already-withdrawn" and pair.keys(f"raw/{ALT_NAME}/") == []
    assert [item["action"] for item in versions(pair)] == ["publish", "merge"]
    assert pair.merge(**options)["status"] == "already-merged"


def test_a_merge_is_refused_for_entries_with_other_atoms(pair):
    # two atoms of the absorbed entry's topology exchanged (its stored manifest describes the file)
    key = f"raw/{ALT_NAME}/topology.pdb"
    lines = pair.backend.read_binary(key).decode().splitlines(keepends=True)
    atoms = [index for index, line in enumerate(lines) if line.startswith(("ATOM", "HETATM"))]
    lines[atoms[5]], lines[atoms[6]] = lines[atoms[6]], lines[atoms[5]]
    swapped = "".join(lines).encode()
    pair.backend.write_binary(key, swapped)
    manifest = json.loads(pair.backend.read_binary(f"raw/{ALT_NAME}/manifest.json"))
    manifest["files"]["topology.pdb"] = {"size": len(swapped), "sha256": hashlib.sha256(swapped).hexdigest()}
    pair.backend.write_binary(f"raw/{ALT_NAME}/manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    pair.client.calls.clear()
    report = pair.merge(dry_run=True, accept_differences=["salt"])
    assert not report["ok"] and pair.writes() == []
    assert report["refused"][0].startswith(
        f"the raw entries of {GS} and {OTHER_GS} cannot be merged: topology.pdb of {OTHER_GS} does not have the atoms "
        f"of topology.pdb of {GS}: 2 of 45 atoms differ in name, residue or order; the first is atom 6: ")
    with pytest.raises(Refused):
        pair.merge(work="work-2", accept_differences=["salt"])
    assert OTHER_GS in [key.split("/")[1] for key in pair.keys("static/GS")]


def test_a_merge_is_refused_for_two_glycans_the_same_id_and_an_id_that_is_not_published(pair, made_bundle):
    report = pair.merge(kept=GS, absorbed=GS, dry_run=True)
    assert not report["ok"] and report["refused"] == [f"{GS} cannot be merged into itself: name two IDs"]
    report = pair.merge(absorbed="GS09999", work="w2", dry_run=True)
    assert report["refused"] == ["GS09999 is not published"]
    report = pair.merge(kept="GS09999", work="w3", dry_run=True)
    assert report["refused"] == ["GS09999 is not published"]
    report = pair.merge(work="w4", dry_run=True, accept_differences=["colour"])
    assert report["refused"][0].startswith("--accept-difference colour: the fields that are compared are package")
    # another glycan
    publish_second(pair.target, made_bundle, pair.tmp, glycan_id="GS01069", name="DGlcpa1-2DManpa1-OH", salt=150.0)
    report = pair.merge(absorbed="GS01069", work="w5", dry_run=True)
    assert report["refused"][0] == f"{GS} and GS01069 are not the same glycan: their archetype WURCS differ"
    assert report["refused"][1].startswith(f"{GS} and GS01069 have other GlyTouCan accessions")
    # an ID that was withdrawn (and not into this one)
    entry_publish.withdraw_entry("GS01069", pair.target, pair.tmp / "withdraw", log=quiet)
    report = pair.merge(absorbed="GS01069", work="w6", dry_run=True)
    assert report["refused"] == ["GS01069 is not published (it was withdrawn)"]
    assert [item["action"] for item in versions(pair)] == ["publish"]


def test_the_command_line_of_a_merge(pair, monkeypatch, tmp_path, capsys):
    report = tmp_path / "report.json"
    common = ["merge", "--id", GS, "--from", OTHER_GS, "--work-dir", str(tmp_path / "w"), "--analysis-workers", "2",
              "--report", str(report)]
    assert _cli(pair, monkeypatch, common + ["--dry-run"]) == 2
    assert "--accept-difference salt" in json.loads(report.read_text())["refused"][0]
    capsys.readouterr()
    assert _cli(pair, monkeypatch, common + ["--dry-run", "--accept-salt-difference", "--note", "one glycan"]) == 0
    written = json.loads(report.read_text())
    assert written["merge"]["accepted_differences"] == ["salt"]
    out = capsys.readouterr().out
    assert f"merge: {OTHER_GS} ({ALT_NAME}) into {GS}; frames 480 + 480 = 960; 6 segment(s)" in out
    assert f"  salt: {GS} 150.0, {OTHER_GS} 200.0 (accepted: every run keeps its own)" in out
    assert "  salt by run: " in out and f"{OTHER_GS}/" in out
    assert "WARNING the salt concentration differs" in out
    assert f"withdraw {OTHER_GS} (merged into {GS}): planned" in out
    assert f"raw raw/{ALT_NAME}/ -> raw-history/raw/{ALT_NAME}/" in out
    assert "dry run: nothing was written to the storage; planned" in out
    assert _cli(pair, monkeypatch, common + ["--accept-salt-difference", "--note", "one glycan",
                                             "--expect-hash", written["entry"]["hash"]]) == 0
    assert f"merge: merged {OTHER_GS} into {GS}" in capsys.readouterr().out
    assert json.loads(report.read_text())["status"] == "merged"
    assert pair.reader.try_sync() is None and OTHER_GS not in pair.reader.glycans()


# ---------------------------------------------------------------------------
# merge --drop-runs: the same glycan simulated as each reducing-end anomer
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def prepared_anomers(tmp_path_factory, prepared, made_bundle):
    """The database of ``prepared`` with the beta anomer of the synthetic glycan published as GS01068 (salt 200 mM)."""
    root = tmp_path_factory.mktemp("published-anomers")
    client = FakeS3Client("bucket")
    client.objects = copy.deepcopy(prepared["objects"])
    backend = S3StorageBackend(client, bucket="bucket", sleep=lambda seconds: None)
    with pytest.MonkeyPatch.context() as patching:
        fake_services.install(patching)
        publish_second(Target(static=backend, raw=backend, workers=4), made_bundle, root, name=BETA_NAME)
    return {"objects": client.objects}


@pytest.fixture
def anomers(prepared_anomers, tmp_path, monkeypatch):
    fake_services.install(monkeypatch)
    return World(prepared_anomers, tmp_path)


def drop(world, **options):
    """Keep the beta anomer (the linkage that occurs in nature), drop the alpha one."""
    return world.merge(kept=OTHER_GS, absorbed=GS, drop_runs=True, **options)


def test_a_dry_run_of_a_drop_reports_the_withdrawal_and_writes_nothing(anomers):
    alpha, beta = anomers.data(GS), anomers.data(OTHER_GS)
    assert alpha["archetype"]["wurcs"] == beta["archetype"]["wurcs"]
    assert alpha["archetype"]["glytoucan"] == beta["archetype"]["glytoucan"]
    # pooled, the two anomers are refused: their runs are of two molecules
    refused = anomers.merge(kept=OTHER_GS, absorbed=GS, work="pooled", dry_run=True, accept_differences=["salt"])
    assert refused["refused"] == [f"{OTHER_GS} and {GS} do not have the same reducing end: beta and alpha anomer"]
    before = stored_tree(anomers.backend, "")
    report = drop(anomers, dry_run=True, note="the beta anomer is kept")
    assert anomers.writes() == [] and stored_tree(anomers.backend, "") == before
    assert report["ok"], report["refused"]
    assert report["operation"] == "merge" and report["status"] == "planned" and report["glycoshape_id"] == OTHER_GS
    merge = report["merge"]
    assert merge["drop_runs"] is True and merge["anomers"] == {"kept": "beta", "absorbed": "alpha"}
    assert (merge["from"], merge["from_glycam_name"], merge["glycam_name"]) == (GS, GLYCAM_NAME, BETA_NAME)
    assert merge["frames"] == {"kept": 480, "absorbed": 480, "total": 480} and merge["segments"] is None
    # the conditions are reported, not refused
    assert merge["differences"] == [{"field": "salt", "what": "salt concentration", "unit": "mM", "kept": 200.0,
                                     "absorbed": 150.0}] and merge["accepted_differences"] == []
    assert f"the salt concentration differs ({OTHER_GS} 200.0 mM, {GS} 150.0 mM): reported only, the runs of {GS} " \
           "are not pooled" in report["warnings"]
    # the kept entry: as it is published
    published = stored_manifest_entry(anomers, OTHER_GS)
    assert report["kept"] == {"status": "unchanged", "hash": published["hash"], "history": None}
    assert report["entry"] == {"hash": published["hash"], "file_count": published["file_count"],
                               "total_size": published["total_size"], "unchanged": True}
    assert "analysis" not in report and "raw" not in report and "diff" not in report
    assert report["static"]["base_global_hash"] == report["static"]["target_global_hash"]
    withdrawal = report["withdrawal"]
    assert (withdrawal["glycoshape_id"], withdrawal["status"], withdrawal["merged_into"], withdrawal["runs_pooled"]) == (
        GS, "planned", OTHER_GS, False)
    assert withdrawal["static"]["base_global_hash"] == report["database"]["global_hash"]
    assert withdrawal["raw"]["relation"] == "move" and withdrawal["raw"]["from_key_prefix"] == f"raw/{GLYCAM_NAME}/"


def stored_manifest_entry(world, glycan_id):
    return json.loads(world.backend.read_binary("static/sync_manifest.json"))["glycans"][glycan_id]


def test_a_drop_withdraws_the_other_anomer_and_leaves_the_kept_folder_as_it_is(anomers, made_bundle):
    kept_folder = {key: anomers.backend.read_binary(key) for key in anomers.keys(f"static/{OTHER_GS}/")}
    kept_raw = {key: anomers.backend.read_binary(key) for key in anomers.keys(f"raw/{BETA_NAME}/")}
    dropped_raw = {key: anomers.backend.read_binary(key) for key in anomers.keys(f"raw/{GLYCAM_NAME}/")}
    dropped_sha256 = hashlib.sha256(dropped_raw[f"raw/{GLYCAM_NAME}/manifest.json"]).hexdigest()
    dry = drop(anomers, dry_run=True, note="the beta anomer is kept", author="curator")
    report = drop(anomers, expect_hash=dry["entry"]["hash"], note="the beta anomer is kept", author="curator")
    assert report["status"] == "merged" and report["kept"]["status"] == "unchanged", report["refused"]
    assert report["withdrawal"]["status"] == "withdrawn"
    assert report["database_after"]["global_hash"] == dry["withdrawal"]["static"]["target_global_hash"]

    # the kept ID: its folder, its raw entry and its history as they were
    assert {key: anomers.backend.read_binary(key) for key in anomers.keys(f"static/{OTHER_GS}/")} == kept_folder
    assert stored_manifest_entry(anomers, OTHER_GS)["hash"] == dry["entry"]["hash"]
    assert {key: anomers.backend.read_binary(key) for key in anomers.keys(f"raw/{BETA_NAME}/")} == kept_raw
    assert [item["action"] for item in versions(anomers, OTHER_GS)] == ["publish"]
    assert anomers.reader.try_sync() is None and GS not in anomers.reader.glycans()
    assert anomers.data(OTHER_GS)["archetype"]["salt"] == 200.0

    # the dropped ID: withdrawn into the kept one, its runs not pooled
    withdrawn = json.loads(anomers.backend.read_binary(f"{STATE}/withdrawn.json"))["withdrawn"][GS]
    assert (withdrawn["merged_into"], withdrawn["runs_pooled"], withdrawn["raw_manifest_sha256"]) == (
        OTHER_GS, False, dropped_sha256)
    assert withdrawn["reason"] == f"dropped into {OTHER_GS} without its runs: the beta anomer is kept"
    history = [item for item in versions_of(anomers, GS) if item["action"] == "withdraw"]
    assert len(history) == 1 and history[0]["merged_into"] == OTHER_GS and history[0]["runs_pooled"] is False

    # its raw entry: in the history with its record, out of raw/ and of the index
    assert anomers.keys(f"raw/{GLYCAM_NAME}/") == []
    root = f"raw-history/raw/{GLYCAM_NAME}/{dropped_sha256}/"
    for key, content in dropped_raw.items():
        assert anomers.backend.read_binary(root + key[len(f"raw/{GLYCAM_NAME}/"):]) == content
    record = json.loads(anomers.backend.read_binary(root + "merged.json"))
    assert (record["kind"], record["runs_pooled"], record["from"], record["into"]) == ("drop", False, GLYCAM_NAME, BETA_NAME)
    assert (record["from_glycoshape_id"], record["into_glycoshape_id"], record["author"]) == (GS, OTHER_GS, "curator")
    index = json.loads(anomers.backend.read_binary("raw/index.json"))
    assert GLYCAM_NAME not in {item["glycam_name"] for item in index["records"]}

    # again: nothing left to do
    anomers.client.calls.clear()
    again = drop(anomers, expect_hash=dry["entry"]["hash"])
    assert again["status"] == "already-merged" and again["withdrawal"]["raw"]["relation"] == "moved"
    assert {call[2] for call in anomers.writes()} <= {f"{STATE}/lock.json"} | {
        call[2] for call in anomers.writes() if call[2].startswith(f"{STATE}/probe-")}
    # a pooled merge of the dropped ID is refused, and a drop of a pooled one
    refused = anomers.merge(kept=OTHER_GS, absorbed=GS, work="pooled", dry_run=True)
    assert refused["refused"] == [f"{GS} was merged into {OTHER_GS} without its runs (--drop-runs)"]

    # the dropped ID is free again: a new glycan takes it, and the drop command again leaves that one alone
    assert entry_publish.status(anomers.target, GS)["entry"]["id_free"]
    new_name = "DGlcpa1-2DManpa1-OH"
    assert entry_publish.publish_entry("publish", renamed_bundle(anomers.tmp / "b-new", made_bundle, new_name),
                                       submission_record(GS, new_name), anomers.target, work_dir=anomers.tmp / "w-new",
                                       log=quiet)["status"] == "published"
    assert GS not in json.loads(anomers.backend.read_binary(f"{STATE}/withdrawn.json"))["withdrawn"]
    again = drop(anomers, work="work-again")
    assert again["status"] == "already-merged" and again["id_taken_by"] == new_name
    assert anomers.reader.try_sync() is None and anomers.data(GS)["archetype"]["name"] == new_name


def versions_of(world, glycan_id):
    return [json.loads(world.backend.read_binary(key)) for key in world.keys(f"{STATE}/history/{glycan_id}/")]


def test_a_drop_is_refused_for_another_glycan_and_for_a_kept_entry_that_changed(anomers, made_bundle):
    report = anomers.merge(kept=OTHER_GS, absorbed=OTHER_GS, drop_runs=True, dry_run=True)
    assert report["refused"] == [f"{OTHER_GS} cannot be merged into itself: name two IDs"]
    report = anomers.merge(kept=OTHER_GS, absorbed="GS09999", drop_runs=True, work="w1", dry_run=True)
    assert report["refused"] == ["GS09999 is not published"]
    publish_second(anomers.target, made_bundle, anomers.tmp, glycan_id="GS01069", name="DGlcpa1-2DManpb1-OH")
    report = anomers.merge(kept=OTHER_GS, absorbed="GS01069", drop_runs=True, work="w2", dry_run=True)
    assert report["refused"][0] == f"{OTHER_GS} and GS01069 are not the same glycan: their archetype WURCS differ"
    assert report["refused"][1].startswith(f"{OTHER_GS} and GS01069 have other GlyTouCan accessions")

    # approved, then the kept entry changed: refused, nothing written
    dry = drop(anomers, work="w3", dry_run=True)
    anomers.update(changes_record(glycoshape_id=OTHER_GS, simulation={"temperature_k": 310.0}), work="w4")
    before = stored_tree(anomers.backend, "static/")
    with pytest.raises(Refused) as refused:
        drop(anomers, work="w3", expect_hash=dry["entry"]["hash"])
    assert refused.value.report["refused"] == [
        f"the folder of {OTHER_GS} has the hash {stored_manifest_entry(anomers, OTHER_GS)['hash'][:16]}... and "
        f"{dry['entry']['hash'][:16]}... was approved"]
    assert stored_tree(anomers.backend, "static/") == before and anomers.keys(f"static/{GS}/")   # still published


def test_the_command_line_of_a_drop(anomers, monkeypatch, tmp_path, capsys):
    report = tmp_path / "report.json"
    common = ["merge", "--id", OTHER_GS, "--from", GS, "--drop-runs", "--note", "beta kept", "--work-dir",
              str(tmp_path / "w"), "--report", str(report)]
    assert _cli(anomers, monkeypatch, common + ["--dry-run"]) == 0
    written = json.loads(report.read_text())
    assert written["merge"]["drop_runs"] is True and written["kept"]["status"] == "unchanged"
    out = capsys.readouterr().out
    assert f"merge --drop-runs: {GS} ({GLYCAM_NAME}, alpha) into {OTHER_GS} (beta); its 480 frames are not pooled; " \
           f"{OTHER_GS} keeps its 480 frames and its folder" in out
    assert f"entry {OTHER_GS} {BETA_NAME}: unchanged" in out
    assert f"  salt: {OTHER_GS} 200.0, {GS} 150.0 (reported only: the runs are not pooled)" in out
    assert f"withdraw {GS} (merged into {OTHER_GS}, runs not pooled): planned" in out
    assert _cli(anomers, monkeypatch, common + ["--expect-hash", written["entry"]["hash"]]) == 0
    assert f"merge: merged {GS} into {OTHER_GS}" in capsys.readouterr().out
    assert anomers.reader.try_sync() is None and GS not in anomers.reader.glycans()


# ---------------------------------------------------------------------------
# prune-runs
# ---------------------------------------------------------------------------
def prune(world, runs, **options):
    options.setdefault("log", quiet)
    options.setdefault("workers", 2)
    options.setdefault("note", "run 2 has an inverted stereocentre")
    return entry_ops.prune_runs(GS, runs, world.target, world.tmp / options.pop("work", "work"), **options)


def test_a_dry_run_of_a_prune_reports_the_cut_and_writes_nothing(world, made_bundle):
    names = segment_names(made_bundle)
    before = stored_tree(world.backend, "")
    report = prune(world, [names[1]], dry_run=True, author="curator")
    assert report["ok"], report["refused"]
    assert world.writes() == [] and stored_tree(world.backend, "") == before
    assert report["operation"] == "prune-runs" and report["status"] == "planned"
    prune_block = report["prune"]
    assert prune_block["removed"] == [{"name": names[1], "first_frame": 200, "frames": 130, "basis": "recorded"}]
    assert [(row["name"], row["first_frame"], row["first_frame_after"]) for row in prune_block["kept"]] == [
        (names[0], 0, 0), (names[2], 330, 200)]
    assert prune_block["frames"] == {"before": 480, "removed": 130, "after": 350}
    assert prune_block["names_after"] == [names[0], names[2]] and prune_block["names_unchanged"]
    assert prune_block["length_us"]["after"] == round(2.5 * 350 / 480, 9)
    assert report["memory"]["frames"] == 350 and len(report["entry"]["hash"]) == 64
    assert report["diff"]["simulation"]["length"]["after"] == round(2.5 * 350 / 480, 9)
    # the raw entry is replaced, every file of the entry before kept in the history
    old_sha256 = hashlib.sha256(world.backend.read_binary(f"raw/{GLYCAM_NAME}/manifest.json")).hexdigest()
    assert report["raw"]["relation"] == "overwrite" and "trajectory.xtc" in report["raw"]["upload"]
    assert sorted(report["raw"]["history"]) == sorted(f"raw-history/raw/{GLYCAM_NAME}/{old_sha256}/{key.rsplit('/', 1)[1]}"
                                                      for key in world.keys(f"raw/{GLYCAM_NAME}/"))
    assert prune_block["history_prefix"] == f"raw-history/raw/{GLYCAM_NAME}/{old_sha256}/"


def test_a_prune_cuts_the_runs_out_of_the_raw_entry_and_publishes_the_rest(world, made_bundle):
    names = segment_names(made_bundle)
    old_manifest = world.backend.read_binary(f"raw/{GLYCAM_NAME}/manifest.json")
    old_sha256 = hashlib.sha256(old_manifest).hexdigest()
    old_raw = {key: world.backend.read_binary(key) for key in world.keys(f"raw/{GLYCAM_NAME}/")}
    dry = prune(world, [names[1]], dry_run=True, author="curator")
    report = prune(world, [names[1]], expect_hash=dry["entry"]["hash"], author="curator")
    assert report["status"] == "published" and report["entry"]["hash"] == dry["entry"]["hash"]

    # the raw entry without the run, its runs named as before; the entry before in the history
    manifest = json.loads(world.backend.read_binary(f"raw/{GLYCAM_NAME}/manifest.json"))
    assert manifest["frames"] == 350 and [s["name"] for s in manifest["segments"]] == [names[0], names[2]]
    assert manifest["pruned"]["manifest_sha256"] == old_sha256 and manifest["pruned"]["author"] == "curator"
    assert manifest["pruned"]["note"] == "run 2 has an inverted stereocentre"
    for key, content in old_raw.items():
        assert world.backend.read_binary(f"raw-history/raw/{GLYCAM_NAME}/{old_sha256}/{key.rsplit('/', 1)[1]}") == content
    record = json.loads(world.backend.read_binary(f"raw-history/raw/{GLYCAM_NAME}/{old_sha256}/pruned.json"))
    assert (record["schema"], record["runs"], record["glycoshape_id"]) == ("glycoshape-raw-prune", [names[1]], GS)
    assert record["removed"][0]["first_frame"] == 200 and record["manifest_sha256_after"] == hashlib.sha256(
        world.backend.read_binary(f"raw/{GLYCAM_NAME}/manifest.json")).hexdigest()
    index = {item["glycam_name"]: item for item in json.loads(world.backend.read_binary("raw/index.json"))["records"]}
    assert index[GLYCAM_NAME]["frames"] == 350
    from lib import archive_transfer

    archive_transfer.download_entry(world.backend, "raw", GLYCAM_NAME, world.tmp / "check")
    entry = world.tmp / "check" / GLYCAM_NAME
    assert archive.verify_entry(entry)["ok"]
    # the published version is the analysis of the frames that are left
    assert world.reader.try_sync() is None
    assert world.data()["archetype"]["length"] == round(2.5 * 350 / 480, 9)
    pca = (world.reader.folder / GS / "output" / "pca.csv").read_text().splitlines()
    assert max(int(line.split(",")[4]) for line in pca[1:]) < 350
    version = versions(world)[-1]
    assert version["operation"] == "prune-runs" and version["prune"]["runs"] == [names[1]]
    assert version["raw_manifest_sha256"] == record["manifest_sha256_after"]

    # again: nothing left
    world.client.calls.clear()
    again = prune(world, [names[1]], expect_hash=dry["entry"]["hash"])
    assert again["status"] == "already-pruned"
    assert {call[2] for call in world.writes()} <= {f"{STATE}/lock.json"} | {
        call[2] for call in world.writes() if call[2].startswith(f"{STATE}/probe-")}


def test_an_interrupted_prune_is_continued_and_bad_prunes_are_refused(world, made_bundle):
    names = segment_names(made_bundle)
    report = prune(world, ["run_9"], dry_run=True)
    assert report["refused"] == [f"the runs of {GS} cannot be pruned: no run named 'run_9'; the runs of the entry "
                                 f"are {', '.join(names)}"]
    report = prune(world, names, work="w2", dry_run=True)
    assert "every run of the entry would be removed" in report["refused"][0]
    assert entry_ops.prune_runs("GS09999", ["run_1"], world.target, world.tmp / "w3", dry_run=True,
                                log=quiet)["refused"] == ["GS09999 is not published"]
    assert world.writes() == []

    stops = {"raw:manifest": 1}

    def checkpoint(name):
        if stops.get(name):
            stops[name] -= 1
            raise RuntimeError(f"stopped at {name}")

    with pytest.raises(entry_publish.PublishEntryError):        # the raw entry is replaced, the folder is not yet
        prune(world, [names[2]], work="w4", checkpoint=checkpoint)
    assert json.loads(world.backend.read_binary(f"raw/{GLYCAM_NAME}/manifest.json"))["frames"] == 330
    assert [item["action"] for item in versions(world)] == ["publish"]
    report = prune(world, [names[2]], work="w5")                # another work folder: the stored entry is taken
    assert report["status"] == "published" and report["raw"]["relation"] == "same"
    assert [item["action"] for item in versions(world)] == ["publish", "prune-runs"]
    assert world.reader.try_sync() is None and world.data()["archetype"]["length"] == round(2.5 * 330 / 480, 9)


# ---------------------------------------------------------------------------
# scrub-raw
# ---------------------------------------------------------------------------
EMAIL, USERNAME = "someone.submitter@example.org", "jdoe-submitter"


def with_personal_data(world):
    """The raw entry of GS01067 with email and username in info.json, a copy in its history and in a document."""
    key = f"raw/{GLYCAM_NAME}/info.json"
    info = json.loads(world.backend.read_binary(key))
    info.update(email=EMAIL, username=USERNAME, comments=f"submitted by {USERNAME}", contact=EMAIL)
    payload = (json.dumps(info, indent=4) + "\n").encode()
    world.backend.write_binary(key, payload)
    manifest = json.loads(world.backend.read_binary(f"raw/{GLYCAM_NAME}/manifest.json"))
    manifest["files"]["info.json"] = {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    world.backend.write_binary(f"raw/{GLYCAM_NAME}/manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    world.backend.write_binary(f"raw-history/raw/{GLYCAM_NAME}/{'0' * 64}/info.json", payload)
    history = world.keys(f"{STATE}/history/{GS}/")[0]
    document = json.loads(world.backend.read_binary(history))
    document["submitter"] = {"email": EMAIL, "username": USERNAME}
    world.backend.write_binary(history, (json.dumps(document, indent=2) + "\n").encode())
    return history


def scrub(world, **options):
    options.setdefault("log", quiet)
    options.setdefault("note", "personal data in the raw entry")
    return entry_ops.scrub_raw(GS, ["email", "username"], world.target, world.tmp / options.pop("work", "work"), **options)


def objects_holding(world, value):
    return sorted(key for key in world.keys() if value.encode() in world.backend.read_binary(key))


def test_a_scrub_removes_the_fields_from_the_raw_entry_its_history_and_the_documents(world):
    history = with_personal_data(world)
    world.client.calls.clear()
    dry = scrub(world, dry_run=True, author="curator")
    assert dry["ok"], dry["refused"]
    assert world.writes() == [] and dry["status"] == "planned"
    scrub_block = dry["scrub"]
    assert [(row["key"], row["removed"], row["replaced"]) for row in scrub_block["raw"]] == [
        (f"raw/{GLYCAM_NAME}/info.json", ["email", "username"], ["contact"])]
    assert [row["key"] for row in scrub_block["history"]] == [f"raw-history/raw/{GLYCAM_NAME}/{'0' * 64}/info.json"]
    assert [(row["key"], row["removed"]) for row in scrub_block["state"]] == [
        (history, ["submitter.email", "submitter.username"])]
    assert scrub_block["values_found"] == {"email": 1, "username": 1} and scrub_block["static"]["certain"] == []
    assert scrub_block["static"]["files_scanned"] > 20
    # what remains: what the code cannot reach, and the username inside a text, shown masked
    assert scrub_block["remains"][:3] == list(entry_ops.SCRUB_REMAINS)
    assert any("username occurs 1 time(s) as text in raw/" in line and "(\"comments\": \"submitted by <username>\"" in line
               for line in scrub_block["remains"])
    # the values are nowhere in the report or in the work folder
    assert EMAIL not in json.dumps(dry) and USERNAME not in json.dumps(dry)
    assert not [path for path in (world.tmp / "work").rglob("*") if path.is_file()
                and (EMAIL.encode() in path.read_bytes() or USERNAME.encode() in path.read_bytes())]

    report = scrub(world, expect_hash=scrub_block["plan_hash"], author="curator")
    assert report["status"] == "scrubbed"
    assert objects_holding(world, EMAIL) == []
    assert objects_holding(world, USERNAME) == [                                     # inside the comment, reported
        f"raw-history/raw/{GLYCAM_NAME}/{'0' * 64}/info.json", f"raw/{GLYCAM_NAME}/info.json"]
    assert all("<username>" in line for line in report["scrub"]["remains"] if "username occurs" in line)
    info = json.loads(world.backend.read_binary(f"raw/{GLYCAM_NAME}/info.json"))
    assert "email" not in info and "username" not in info and info["contact"] == "[removed]"
    manifest = json.loads(world.backend.read_binary(f"raw/{GLYCAM_NAME}/manifest.json"))
    record = manifest["scrubbed"][-1]
    assert record["fields"] == ["email", "username"] and record["author"] == "curator"
    from lib import archive_transfer

    archive_transfer.download_entry(world.backend, "raw", GLYCAM_NAME, world.tmp / "check")
    assert archive.verify_entry(world.tmp / "check" / GLYCAM_NAME)["ok"]
    old = f"raw-history/raw/{GLYCAM_NAME}/{record['manifest_sha256_before']}/"
    assert json.loads(world.backend.read_binary(old + "scrubbed.json"))["fields"] == ["email", "username"]
    assert world.backend.head(old + "manifest.json") is not None
    index = {item["glycam_name"]: item for item in json.loads(world.backend.read_binary("raw/index.json"))["records"]}
    assert index[GLYCAM_NAME]["files"]["info.json"]["sha256"] == manifest["files"]["info.json"]["sha256"]

    world.client.calls.clear()
    again = scrub(world)
    assert again["status"] == "already-scrubbed"
    assert {call[2] for call in world.writes()} <= {f"{STATE}/lock.json"} | {
        call[2] for call in world.writes() if call[2].startswith(f"{STATE}/probe-")}


def test_a_scrub_is_refused_when_the_published_folder_holds_a_value(world):
    with_personal_data(world)
    world.update(changes_record(curated={"common_names": [EMAIL]}))
    report = scrub(world, dry_run=True)
    assert not report["ok"] and sorted(reason.split(" (")[0] for reason in report["refused"]) == [
        f"the published folder holds email, username in static/{GS}/{GS}.zip:data.json",
        f"the published folder holds email, username in static/{GS}/data.json"]
    assert EMAIL not in json.dumps(report)
    with pytest.raises(Refused):
        scrub(world, work="w2")
    assert world.keys(f"raw-history/raw/{GLYCAM_NAME}/") == [f"raw-history/raw/{GLYCAM_NAME}/{'0' * 64}/info.json"]


def test_the_command_lines_of_prune_and_scrub(world, made_bundle, monkeypatch, tmp_path, capsys):
    names = segment_names(made_bundle)
    report = tmp_path / "r.json"
    common = ["prune-runs", "--id", GS, "--run", names[1], "--note", "stereo", "--analysis-workers", "2",
              "--work-dir", str(tmp_path / "w"), "--report", str(report)]
    assert _cli(world, monkeypatch, common + ["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert f"prune: remove {names[1]} (frames 200-329); 480 -> 350 frames; runs after: {names[0]}, {names[2]}" in out
    assert _cli(world, monkeypatch, common + ["--expect-hash", json.loads(report.read_text())["entry"]["hash"]]) == 0
    assert f"prune-runs: published {GS}" in capsys.readouterr().out
    with_personal_data(world)
    common = ["scrub-raw", "--id", GS, "--field", "email", "--field", "username", "--note", "personal data",
              "--work-dir", str(tmp_path / "s"), "--report", str(report)]
    assert _cli(world, monkeypatch, common + ["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert f"scrub email, username of {GS}: values found" in out and "  remains: " in out and EMAIL not in out
    plan = json.loads(report.read_text())["scrub"]["plan_hash"]
    assert _cli(world, monkeypatch, common + ["--expect-hash", "0" * 64]) == 2
    assert _cli(world, monkeypatch, common + ["--expect-hash", plan]) == 0
    assert f"scrub-raw: scrubbed {GS}" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------
def _cli(world, monkeypatch, argv):
    import importlib.util

    spec = importlib.util.spec_from_file_location("publish_entry_cli", REPO_ROOT / "scripts" / "publish_entry.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_target", lambda args: world.target)
    return module.main(argv)


def test_the_command_line_of_the_operations(world, monkeypatch, tmp_path, capsys):
    changes = tmp_path / "changes.json"
    changes.write_text(json.dumps(changes_record(simulation={"temperature_k": 310.0})))
    report = tmp_path / "report.json"
    status = _cli(world, monkeypatch, ["update-metadata", "--changes", str(changes), "--work-dir", str(tmp_path / "w"),
                                       "--dry-run", "--report", str(report)])
    assert status == 0
    written = json.loads(report.read_text())
    assert written["schema"] == "glycoshape-entry-operation" and written["diff"]["simulation"]["temperature"]["after"] == 310.0
    assert "simulation temperature: 300.0 -> 310.0" in capsys.readouterr().out
    status = _cli(world, monkeypatch, ["update-metadata", "--changes", str(changes), "--work-dir", str(tmp_path / "w"),
                                       "--expect-hash", written["entry"]["hash"]])
    assert status == 0 and world.reader.try_sync() is None and world.data()["archetype"]["temperature"] == 310.0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(changes_record(names={"glycam_name": "DGalpb1-2DGlcpb1-OH"})))
    assert _cli(world, monkeypatch, ["update-metadata", "--changes", str(bad), "--work-dir", str(tmp_path / "w2")]) == 2
    assert _cli(world, monkeypatch, ["reanalyse", "--id", GS, "--work-dir", str(tmp_path / "w3"), "--dry-run",
                                     "--exclude-replica", "nothing", "--report", str(report)]) == 2
    assert json.loads(report.read_text())["refused"]


def test_the_command_line_prints_and_reports_an_operation_that_moves_the_raw_entry(world, monkeypatch, tmp_path,
                                                                                   capsys):
    # Declaring the simulated anomer renames the entry, so its raw entry moves (report["raw"] describes the move).
    report = tmp_path / "report.json"
    status = _cli(world, monkeypatch, ["reanalyse", "--id", GS, "--simulated-anomer", "beta", "--note", "simulated as beta",
                                       "--work-dir", str(tmp_path / "w"), "--dry-run", "--report", str(report)])
    assert status == 0
    written = json.loads(report.read_text())
    assert written["raw"]["relation"] == "move" and written["glycam_name"] == BETA_NAME
    out = capsys.readouterr().out
    assert f"raw {written['raw']['from_key_prefix']} -> {written['raw']['to_key_prefix']}: move" in out


# ---------------------------------------------------------------------------
# On a real S3-compatible server (MinIO, moto; skipped without one)
# ---------------------------------------------------------------------------
def test_a_rename_on_a_real_server(s3_bucket, made_bundle, tmp_path, monkeypatch):
    backend, client, bucket = s3_bucket
    fake_services.install(monkeypatch)
    publish_world.build(tmp_path / "published", backend, raw_source=publish_world.other_archive(tmp_path / "stored"))
    target = Target(static=backend, raw=backend, workers=4)
    assert entry_publish.publish_entry("publish", made_bundle, submission_record(GS, GLYCAM_NAME), target,
                                       work_dir=tmp_path / "w1", log=quiet)["status"] == "published"
    reader = Reader(backend, tmp_path / "website")
    assert reader.try_sync() is None
    trajectory = backend.head(f"raw/{GLYCAM_NAME}/trajectory.xtc")
    changes = changes_record(names={"glycam_name": NEW_NAME})
    dry = entry_ops.update_metadata(changes, target, tmp_path / "w2", dry_run=True, log=quiet)
    report = entry_ops.update_metadata(changes, target, tmp_path / "w2", expect_hash=dry["entry"]["hash"], log=quiet)
    assert report["status"] == "published" and report["rename"]["raw"]["relation"] == "move"
    assert not list(backend.list_objects(f"raw/{GLYCAM_NAME}/"))
    moved = backend.head(f"raw/{NEW_NAME}/trajectory.xtc")
    assert (moved["size"], moved["sha256"]) == (trajectory["size"], trajectory["sha256"])
    assert reader.try_sync() is None and reader.fresh(tmp_path / "fresh").try_sync() is None
    assert json.loads((reader.folder / GS / "data.json").read_text())["archetype"]["name"] == NEW_NAME
    assert [item["key"] for item in backend.list_objects("publish-state/static/lock")] == []


def test_a_merge_on_a_real_server(s3_bucket, made_bundle, tmp_path, monkeypatch):
    backend, client, bucket = s3_bucket
    fake_services.install(monkeypatch)
    one_glycan(monkeypatch)
    publish_world.build(tmp_path / "published", backend, raw_source=publish_world.other_archive(tmp_path / "stored"))
    target = Target(static=backend, raw=backend, workers=4)
    assert entry_publish.publish_entry("publish", made_bundle, submission_record(GS, GLYCAM_NAME), target,
                                       work_dir=tmp_path / "w1", log=quiet)["status"] == "published"
    publish_second(target, made_bundle, tmp_path)
    absorbed = {item["key"]: backend.head(item["key"]) for item in backend.list_objects(f"raw/{ALT_NAME}/")}
    options = dict(accept_differences=["salt"], workers=2, log=quiet)
    dry = entry_ops.merge(GS, OTHER_GS, target, tmp_path / "w2", dry_run=True, **options)
    report = entry_ops.merge(GS, OTHER_GS, target, tmp_path / "w2", expect_hash=dry["entry"]["hash"], **options)
    assert report["status"] == "merged" and report["database_after"] == {
        "global_hash": dry["withdrawal"]["static"]["target_global_hash"]}
    # the absorbed raw entry was copied inside the storage to the history, then deleted
    assert not list(backend.list_objects(f"raw/{ALT_NAME}/"))
    history = f"raw-history/raw/{ALT_NAME}/{report['merge']['absorbed_manifest_sha256']}/"
    for key, head in absorbed.items():
        moved = backend.head(history + key.rsplit("/", 1)[1])
        assert (moved["size"], moved["sha256"]) == (head["size"], head["sha256"])
    assert backend.head(history + "merged.json") is not None
    reader = Reader(backend, tmp_path / "website")
    assert reader.try_sync() is None and OTHER_GS not in reader.glycans()
    assert json.loads((reader.folder / GS / "data.json").read_text())["archetype"]["length"] == 5.0
    assert entry_ops.merge(GS, OTHER_GS, target, tmp_path / "w3", workers=2, log=quiet)["status"] == "already-merged"
    assert [item["key"] for item in backend.list_objects("publish-state/static/lock")] == []


def test_prune_and_scrub_on_a_real_server(s3_bucket, made_bundle, tmp_path, monkeypatch):
    backend, client, bucket = s3_bucket
    fake_services.install(monkeypatch)
    publish_world.build(tmp_path / "published", backend, raw_source=publish_world.other_archive(tmp_path / "stored"))
    target = Target(static=backend, raw=backend, workers=4)
    assert entry_publish.publish_entry("publish", made_bundle, submission_record(GS, GLYCAM_NAME), target,
                                       work_dir=tmp_path / "w1", log=quiet)["status"] == "published"
    names = segment_names(made_bundle)
    old = backend.head(f"raw/{GLYCAM_NAME}/trajectory.xtc")
    old_sha256 = hashlib.sha256(backend.read_binary(f"raw/{GLYCAM_NAME}/manifest.json")).hexdigest()
    options = dict(note="stereo", workers=2, log=quiet)
    dry = entry_ops.prune_runs(GS, [names[2]], target, tmp_path / "w2", dry_run=True, **options)
    report = entry_ops.prune_runs(GS, [names[2]], target, tmp_path / "w2", expect_hash=dry["entry"]["hash"], **options)
    assert report["status"] == "published"
    kept = backend.head(f"raw-history/raw/{GLYCAM_NAME}/{old_sha256}/trajectory.xtc")
    assert (kept["size"], kept["sha256"]) == (old["size"], old["sha256"])          # copied inside the storage
    assert json.loads(backend.read_binary(f"raw/{GLYCAM_NAME}/manifest.json"))["frames"] == 330
    reader = Reader(backend, tmp_path / "website")
    assert reader.try_sync() is None

    world = type("World", (), {})()                             # with_personal_data needs backend and keys
    world.backend = backend
    world.keys = lambda prefix="": sorted(item["key"] for item in backend.list_objects(prefix))
    with_personal_data(world)
    dry = entry_ops.scrub_raw(GS, ["email", "username"], target, tmp_path / "w3", note="data", dry_run=True, log=quiet)
    report = entry_ops.scrub_raw(GS, ["email", "username"], target, tmp_path / "w3", note="data",
                                 expect_hash=dry["scrub"]["plan_hash"], log=quiet)
    assert report["status"] == "scrubbed"
    assert not [key for key in world.keys() if EMAIL.encode() in backend.read_binary(key)]
    from lib import archive_transfer

    archive_transfer.download_entry(backend, "raw", GLYCAM_NAME, tmp_path / "check")
    assert archive.verify_entry(tmp_path / "check" / GLYCAM_NAME)["ok"]
    assert [item["key"] for item in backend.list_objects("publish-state/static/lock")] == []
