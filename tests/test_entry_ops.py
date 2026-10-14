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
from bundle_fixtures import renamed_bundle, submission_record
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
