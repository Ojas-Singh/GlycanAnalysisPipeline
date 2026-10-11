"""A published glycan folder from a bundle and a submission record (lib/entry_complete.py).

The bundle is the one of the synthetic glycan (``conftest.bundle_dir``); the web services are
replaced (``fake_services``) and no test reaches the network.
"""
from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest

import lib.config as config
from lib import entry_complete, glystatic, name_utils, storage
from lib.entry_complete import BundleError, Submission, SubmissionError, complete_entry, load_bundle
from lib.glybake import GlycoShapeBaker
from lib.glymeta import GlycanMetadataProcessor

import fake_services
from bundle_fixtures import submission_record, write_bundle_json
from s3_fakes import tree
from synthetic_glycan import GLYCAM_NAME

pytestmark = pytest.mark.usefixtures("no_internet")
GS = "GS01067"
SIMULATION = {"simulation_length_ns": 2500.0, "md_package": "GROMACS", "force_field": "GLYCAM_06j",
              "temperature_k": 300.0, "pressure_bar": 1.0, "salt_concentration_mm": 150.0, "source_type": "manual",
              "comments": "Run by the test."}
CURATED = {"common_names": ["Mannobiose"], "aliases": [], "description": "", "keywords": ["Curated"]}


def record(**fields):
    complete = submission_record(GS, GLYCAM_NAME, simulation=dict(SIMULATION), curated=dict(CURATED))
    complete.update(fields)
    return complete


def old_static_stage(process_folder: Path, static_root: Path) -> Path:
    """The folder as main.py --mode static-only makes it: static stage, metadata, archive; the registry replaced."""
    metadata = {"ID": GS, "length": 2.5, "simulation_length_ns": None, "package": "GROMACS", "forcefield": "GLYCAM_06j",
                "temperature": 300.0, "pressure": 1.0, "salt": 150.0, "comments": SIMULATION["comments"],
                "glytoucan_id": ""}

    class Registry:
        def is_available(self):
            return True

        def get_record_by_glycoshape_id(self, gs_id, collection=None):
            return {"md_package": "GROMACS", "force_field": "GLYCAM_06j", "source_type": "manual",
                    "comments": SIMULATION["comments"]}

        def get_glycan_record_by_glycoshape_id(self, gs_id):
            return {"keywords": CURATED["keywords"], "common_names": CURATED["common_names"], "description": ""}

        def upsert_glycan_metadata(self, payload):
            self.upserted = payload

    with storage.local_storage(), patch.object(glystatic, "get_glycan_metadata", return_value=metadata), \
            patch.object(glystatic, "_source_simulation_length_ns", return_value=None):
        done = glystatic.process_glycan(str(process_folder), GLYCAM_NAME, str(static_root))
        folder = Path(done["final_output_dir"])
        assert GlycanMetadataProcessor(static_root, pb_client=Registry()).update_glycan_metadata(folder / "data.json")
        assert GlycoShapeBaker(static_root).create_individual_glycan_archive(folder)
    return folder


# ---------------------------------------------------------------------------
# The submission record
# ---------------------------------------------------------------------------
def test_the_units_of_a_submission_are_converted_once_to_those_of_data_json():
    submission = Submission.parse(record())
    assert submission.simulation() == {"length": 2.5, "package": "GROMACS", "forcefield": "GLYCAM_06j",
                                       "temperature": 300.0, "pressure": 1.0, "salt": 150.0}
    assert submission.warnings == ()
    in_microseconds = Submission.parse(record(simulation={**SIMULATION, "simulation_length_ns": None,
                                                          "simulation_length_us": 2.5}))
    assert in_microseconds.simulation() == submission.simulation() and in_microseconds == submission
    assert in_microseconds.sha256() == submission.sha256()                 # one normal form for both spellings
    assert submission.record()["simulation"]["simulation_length_ns"] == 2500.0
    assert Submission.parse(submission.record()) == submission
    assert Submission.parse(submission) is submission
    both = Submission.parse(record(simulation={**SIMULATION, "simulation_length_us": 2.5}))
    assert both.length_us == 2.5
    assert submission.enrichment() == {
        "extra_keywords": ["Curated", "MD:GROMACS", "FF:GLYCAM_06j", "Source:manual"],
        "user_common_names": ["Mannobiose"], "user_description": "", "user_comments": "Run by the test."}


@pytest.mark.parametrize("key, instead", [
    ("simulation_length", "simulation_length_ns or simulation_length_us"),
    ("length", "simulation_length_ns or simulation_length_us"), ("simulationLength", "simulation_length_ns"),
    ("temperature", "temperature_k"), ("pressure", "pressure_bar"),
    ("salt_concentration", "salt_concentration_mm"), ("salt", "salt_concentration_mm"),
])
def test_a_quantity_without_its_unit_is_refused(key, instead):
    with pytest.raises(SubmissionError, match=f"simulation.{key} is not accepted: use {instead}"):
        Submission.parse(record(simulation={**SIMULATION, key: 2}))


def test_a_submission_that_cannot_be_used_names_every_reason():
    bad = record(glycoshape_id="GS1067", glytoucan_id="G1", extra=1,
                 simulation={"simulation_length_ns": 2500, "simulation_length_us": 3.0, "temperature_k": "300",
                             "salt_concentration_mm": -1, "rate": 2},
                 curated={"keywords": "N-Glycan", "notes": []})
    with pytest.raises(SubmissionError) as error:
        Submission.parse(bad)
    message = str(error.value)
    for reason in ("glycoshape_id must be GS and five digits", "unknown field 'extra'", "do not agree",
                   "simulation.temperature_k must be a number", "simulation.salt_concentration_mm must be zero or more",
                   "unknown field simulation.rate", "curated.keywords must be a list of texts",
                   "unknown field curated.notes", "not a GlyTouCan accession"):
        assert reason in message, reason
    for wrong in ([], {"glycoshape_id": GS}, {**record(), "schema": "other"}, {**record(), "schema_version": 2},
                  {**record(), "glycam_name": " padded "}):
        with pytest.raises(SubmissionError):
            Submission.parse(wrong)


def test_values_that_look_like_a_unit_mistake_are_accepted_with_a_warning():
    nanoseconds_as_microseconds = Submission.parse(record(simulation={**SIMULATION, "simulation_length_ns": 2.5}))
    assert nanoseconds_as_microseconds.length_us == 0.0025
    assert nanoseconds_as_microseconds.warnings == ("the aggregated length is 0.0025 microseconds: check the unit",)
    celsius = Submission.parse(record(simulation={**SIMULATION, "temperature_k": 27.0}))
    assert celsius.warnings == ("the temperature is 27 K: check the unit",)
    bare = Submission.parse({"glycoshape_id": GS, "glycam_name": GLYCAM_NAME})
    assert bare.simulation() == {"length": None, "package": "", "forcefield": "", "temperature": None,
                                 "pressure": None, "salt": None}
    assert len(bare.warnings) == 4 and bare.warnings[0] == "simulation.simulation_length_ns is not given"


# ---------------------------------------------------------------------------
# The bundle
# ---------------------------------------------------------------------------
def test_a_bundle_is_read_with_its_documents(bundle_dir):
    bundle = load_bundle(bundle_dir)
    assert bundle.glycam_name == GLYCAM_NAME and bundle.levels == (1, 2) and bundle.hashes_verified
    assert bundle.manifest["frames"] == 480 and bundle.result["status"] == "ok" and bundle.extends is None
    assert sorted(name for name in bundle.files if "/" not in name) == ["result.json", "run.log"]
    assert bundle.summary()["frames"] == 480 and len(bundle.sha256) == 64
    before = tree(bundle_dir)
    write_bundle_json(bundle_dir, levels=["level_1", "level_2"])           # the other spelling of the levels
    assert load_bundle(bundle_dir).levels == (1, 2)
    assert set(tree(bundle_dir)) == set(before)


def _damage(bundle_dir: Path, what: str) -> None:
    if what == "content":
        path = bundle_dir / "static" / "output" / "pca.csv"
        content = bytearray(path.read_bytes())
        content[10] ^= 1
        path.write_bytes(bytes(content))
    elif what == "missing":
        (bundle_dir / "static" / "landscape" / "preprocessing.npz").unlink()
    elif what == "unlisted":
        (bundle_dir / "static" / "output" / "extra.txt").write_text("not in bundle.json")
    elif what == "size":
        with open(bundle_dir / "archive" / "trajectory.xtc", "ab") as handle:
            handle.write(b"\0")
    elif what == "another run":                                             # a static file that the run did not produce
        (bundle_dir / "static" / "output" / "pca.csv").write_text("x,y\n")
        write_bundle_json(bundle_dir)
    elif what == "failed run":
        result = json.loads((bundle_dir / "result.json").read_text())
        result["status"] = "failed"
        (bundle_dir / "result.json").write_text(json.dumps(result))
        write_bundle_json(bundle_dir)
    elif what == "another glycan":
        write_bundle_json(bundle_dir, glycam_name="DGalpb1-OH")
    elif what == "levels":
        write_bundle_json(bundle_dir, levels=[1, 2, 3])
    elif what == "another trajectory":                                      # the analysis is of other raw data
        result = json.loads((bundle_dir / "result.json").read_text())
        result["input"]["files"]["trajectory.xtc"]["sha256"] = "0" * 64
        (bundle_dir / "result.json").write_text(json.dumps(result))
        write_bundle_json(bundle_dir)
    elif what == "schema":
        write_bundle_json(bundle_dir, schema_version=2)
    elif what == "extends":
        write_bundle_json(bundle_dir, extends={"glycam_name": GLYCAM_NAME})


@pytest.mark.parametrize("what, reason", [
    ("content", "static/output/pca.csv does not have the SHA-256 bundle.json records"),
    ("missing", "static/landscape/preprocessing.npz is listed in bundle.json and missing"),
    ("unlisted", "static/output/extra.txt is in the bundle directory and not in bundle.json"),
    ("size", "archive/trajectory.xtc does not have the size bundle.json records"),
    ("another run", "the static files are not the ones result.json produced"),
    ("failed run", "the run did not succeed"),
    ("another glycan", "is of another glycan than bundle.json"),
    ("levels", "bundle.json does not list static/output/torsion_level_3_reps.csv"),
    ("another trajectory", "result.json was not computed from the files of this archive entry"),
    ("schema", "schema version 2"),
    ("extends", "extends is neither null nor"),
])
def test_a_bundle_that_is_not_what_it_says_is_refused(bundle_dir, tmp_path, what, reason):
    _damage(bundle_dir, what)
    with pytest.raises(BundleError, match=reason):
        load_bundle(bundle_dir)
    with pytest.raises(BundleError):
        complete_entry(bundle_dir, record(), tmp_path / "static")
    assert not (tmp_path / "static").exists() or not any((tmp_path / "static").iterdir())


def test_sizes_are_checked_even_without_hashing(bundle_dir):
    _damage(bundle_dir, "content")
    assert load_bundle(bundle_dir, verify_hashes=False).hashes_verified is False
    _damage(bundle_dir, "size")
    with pytest.raises(BundleError, match="size"):
        load_bundle(bundle_dir, verify_hashes=False)


def test_a_submission_takes_the_run_metadata_of_the_bundle(bundle_dir):
    source = {"name": "run-17", "simulation_length_ns": 480.0, "time_per_frame_ps": 1000.0, "md_package": "AMBER",
              "force_field": "GLYCAM_06j", "temperature_k": 298.15, "pressure_bar": 1.0, "salt_concentration_mm": None,
              "comments": "", "glytoucan_id": "G12345AB", "run_id": "7"}
    write_bundle_json(bundle_dir, source=source)
    bundle = load_bundle(bundle_dir)
    made = entry_complete.submission_from_bundle(bundle, GS, curated={"keywords": ["Curated"]}, source_type="puuid")
    assert made["simulation"] == {"simulation_length_ns": 480.0, "md_package": "AMBER", "force_field": "GLYCAM_06j",
                                  "temperature_k": 298.15, "pressure_bar": 1.0, "source_type": "puuid"}
    assert (made["glycoshape_id"], made["glycam_name"], made["glytoucan_id"]) == (GS, GLYCAM_NAME, "G12345AB")
    submission = Submission.parse(made)
    assert submission.length_us == 0.48 and submission.salt_mm is None and submission.keywords == ("Curated",)
    # what the registry knows better replaces what the bundle says
    in_microseconds = entry_complete.submission_from_bundle(bundle, GS, simulation_length_us=1.5, salt_concentration_mm=150)
    assert Submission.parse(in_microseconds).simulation()["length"] == 1.5
    assert in_microseconds["simulation"]["salt_concentration_mm"] == 150

    # an extending bundle states the length of the new run only
    extending = load_bundle(bundle_dir)
    extending.document["extends"] = {"glycam_name": GLYCAM_NAME, "manifest_sha256": "0" * 64, "frames": 100}
    with pytest.raises(SubmissionError, match="states the length of the new run only"):
        entry_complete.submission_from_bundle(extending, GS)
    added = entry_complete.submission_from_bundle(extending, GS, published_length_ns=2000.0)
    assert added["simulation"]["simulation_length_ns"] == 2480.0
    given = entry_complete.submission_from_bundle(extending, GS, simulation_length_ns=3000.0)
    assert given["simulation"]["simulation_length_ns"] == 3000.0


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("online", [True, False], ids=["services mocked", "offline"])
def test_the_folder_is_the_one_the_static_stage_writes(bundle_dir, synthetic_entry, tmp_path, monkeypatch, online):
    """The same glycan through the static stage (ID lookup replaced) and through complete_entry."""
    calls = fake_services.install(monkeypatch, online=online)
    old = old_static_stage(synthetic_entry["result"] / "process" / GLYCAM_NAME, tmp_path / "old")
    asked_by_the_static_stage = list(calls)
    calls.clear()
    name_utils._GLYTOUCAN_IUPAC_CACHE.clear()                               # the motif names the first run looked up
    report = complete_entry(bundle_dir, record(), tmp_path / "new", offline=not online)
    new = Path(report["folder"])

    assert new == tmp_path / "new" / GS and old.name == GS
    assert tree(new) == tree(old)                                           # every file, byte for byte
    assert sorted(tree(new)) == sorted(report["files"]) and report["file_count"] == 25
    if online:
        assert calls == asked_by_the_static_stage and len(calls) >= 8      # the same questions, in the same order
        assert report["gaps"] == [] and report["problems"] == [] and report["warnings"] == [
            "the entry says 480 ns of simulation and the submission 2500 ns"]   # info.json of the archive entry
        assert report["lookups"]["counts"]["ok"] == len(calls) and report["lookups"]["unanswered"] == []
    else:
        assert calls == []                                                  # offline: not one request
        assert asked_by_the_static_stage                                    # where the static stage tried and failed
        assert report["lookups"]["offline"] is True and report["lookups"]["counts"]["ok"] == 0
        assert "wurcs2glytoucan" in report["lookups"]["unanswered"]
        assert "archetype.glytoucan" in report["gaps"] and "archetype.wurcs" not in report["gaps"]
        assert report["problems"] == []                                     # a WURCS is enough to publish


def test_what_the_folder_holds(bundle_dir, tmp_path, monkeypatch):
    fake_services.install(monkeypatch)
    before = tree(bundle_dir)
    (tmp_path / "static").mkdir()
    (tmp_path / "static" / "GS00001").mkdir()
    (tmp_path / "static" / "GS00001" / "data.json").write_text("{}")
    (tmp_path / "static" / GS).mkdir()
    (tmp_path / "static" / GS / "stale.txt").write_text("from an earlier attempt")
    levels, manager = config.cluster_levels, storage._storage_manager
    report = complete_entry(bundle_dir, record(), tmp_path / "static")
    folder = tmp_path / "static" / GS

    assert tree(bundle_dir) == before                                       # the bundle is only read
    assert config.cluster_levels == levels and storage._storage_manager is manager
    assert sorted(path.name for path in (tmp_path / "static").iterdir()) == ["GS00001", GS]
    assert (tmp_path / "static" / "GS00001" / "data.json").read_text() == "{}" and not (folder / "stale.txt").exists()

    data = json.loads((folder / "data.json").read_text())
    assert data["archetype"]["ID"] == GS and data["archetype"]["name"] == GLYCAM_NAME
    assert data["archetype"]["glycam"] == "DManpa1-2DManp" and data["beta"]["glycam"] == "DManpa1-2DManpb1-OH"
    assert {name: data["archetype"][name] for name in glystatic.SIMULATION_FIELDS} == report["simulation"] == {
        "length": 2.5, "package": "GROMACS", "forcefield": "GLYCAM_06j", "temperature": 300.0, "pressure": 1.0,
        "salt": 150.0}
    assert sorted(data["archetype"]["cluster_levels"]) == ["level_1", "level_2"]
    assert data["search_meta"]["common_names"] == ["Mannobiose"]
    assert data["search_meta"]["keywords"] == ["Curated", "FF:GLYCAM_06j", "MD:GROMACS", "Source:manual"]
    assert data["search_meta"]["description"].endswith("Run by the test.") and "aliases" not in data["search_meta"]
    assert report["search_meta"] == data["search_meta"] and report["identity"]["alpha"]["glytoucan"]

    # a PDB file is the static-ready one under the REMARK block of its anomer
    for anomer in ("alpha", "beta"):
        ready = (bundle_dir / "static" / "output" / "level_1" / "PDB" / f"{anomer}.pdb").read_bytes()
        published = (folder / "output" / "level_1" / "PDB" / f"{anomer}.pdb").read_bytes()
        banner = published[: len(published) - len(ready)].decode()
        assert published.endswith(ready) and banner.startswith("REMARK     GENERATED BY GlycanAnalysisPipeline")
        assert all(line.startswith("REMARK") for line in banner.splitlines())
        assert f"GlyTouCan ID: {data[anomer]['glytoucan']}" in banner and f"IUPAC Name: {data[anomer]['iupac']}" in banner
    for name in ("output/pca.csv", "output/info.json", "output/torparts.npz", "landscape/pca_model.joblib",
                 "output/level_2/dist.svg"):
        assert (folder / name).read_bytes() == (bundle_dir / "static" / name).read_bytes()

    with zipfile.ZipFile(folder / f"{GS}.zip") as archive:
        assert archive.namelist() == sorted(archive.namelist()) == [
            "GLYCAM/alpha.pdb", "GLYCAM/beta.pdb", "PDB/alpha.pdb", "PDB/beta.pdb", "data.json", "dist.svg",
            "glycosmos.json", "glygen.json", "snfg.svg"]
        assert {info.date_time for info in archive.infolist()} == {(1980, 1, 1, 0, 0, 0)}
        assert archive.read("data.json") == (folder / "data.json").read_bytes()
        assert archive.read("PDB/alpha.pdb") == (folder / "output" / "level_1" / "PDB" / "alpha.pdb").read_bytes()


def test_completing_again_gives_the_same_bytes(bundle_dir, tmp_path, monkeypatch):
    fake_services.install(monkeypatch)
    first = tree(Path(complete_entry(bundle_dir, record(), tmp_path / "first")["folder"]))
    shutil.copytree(bundle_dir, tmp_path / "copy")                          # other paths and file times
    for path in (tmp_path / "copy").rglob("*"):
        os.utime(path, (1_600_000_000, 1_600_000_000))
    again = tree(Path(complete_entry(tmp_path / "copy", record(), tmp_path / "again")["folder"]))
    assert again == first and f"{GS}.zip" in first
    in_place = tree(Path(complete_entry(bundle_dir, record(), tmp_path / "first")["folder"]))   # over the first one
    assert in_place == first


def test_curated_aliases_that_the_names_do_not_give_are_kept(bundle_dir, tmp_path, monkeypatch):
    fake_services.install(monkeypatch)
    curated = {**CURATED, "aliases": ["DManpa1-2DManp", "mannobiose", "Man2", "2-O-α-D-mannosyl-D-mannose"]}
    report = complete_entry(bundle_dir, record(curated=curated), tmp_path / "static")
    data = json.loads((Path(report["folder"]) / "data.json").read_text(encoding="utf-8"))
    # the GLYCAM name and the common name are found without help; the two others are not
    assert data["search_meta"]["aliases"] == ["Man2", "2-O-α-D-mannosyl-D-mannose"]
    assert list(data["search_meta"]) == ["common_names", "description", "keywords", "aliases"]
    with zipfile.ZipFile(Path(report["folder"]) / f"{GS}.zip") as archive:
        assert json.loads(archive.read("data.json")) == data


def test_what_stands_in_the_way_of_publishing_is_reported(bundle_dir, tmp_path, monkeypatch):
    fake_services.install(monkeypatch, online=False, crabwurcs=False, snfg=False)
    report = complete_entry(bundle_dir, record(), tmp_path / "static", offline=True)
    assert report["problems"] == [
        "snfg.svg is missing (the SNFG renderer did not run)",
        "the archetype has neither a WURCS nor a GlyTouCan accession "
        "(the static stage of main.py calls such a folder incomplete)"]
    assert {"archetype.wurcs", "archetype.glytoucan", "alpha.wurcs", "beta.glytoucan"} <= set(report["gaps"])
    assert (Path(report["folder"]) / "data.json").is_file() and report["file_count"] == 24   # written all the same
    # an accession from the registry fills the archetype, and only the archetype
    with_accession = complete_entry(bundle_dir, record(glytoucan_id="G12345AB"), tmp_path / "static", offline=True)
    assert with_accession["identity"]["archetype"]["glytoucan"] == "G12345AB"
    assert with_accession["identity"]["alpha"]["glytoucan"] is None
    assert with_accession["problems"] == ["snfg.svg is missing (the SNFG renderer did not run)"]


def test_the_submission_and_the_bundle_have_to_be_of_one_glycan(bundle_dir, tmp_path):
    with pytest.raises(SubmissionError, match="the submission is for 'DGalpb1-OH' and the bundle holds"):
        complete_entry(bundle_dir, submission_record(GS, "DGalpb1-OH"), tmp_path / "static")
    with pytest.raises(SubmissionError, match="simulation.simulation_length is not accepted"):
        complete_entry(bundle_dir, record(simulation={"simulation_length": 2.5}), tmp_path / "static")
    assert not (tmp_path / "static").exists()


def test_search_enrichment_is_one_rule_for_the_registry_and_for_a_submission():
    from lib.glymeta import search_enrichment
    from lib.pocketbase import extract_search_enrichment

    glycan = {"keywords": ["N-Glycan", "", None, "Curated"], "common_names": [" GM3 ", "gm3", "", None, "Other"],
              "description": "  A ganglioside.  "}
    submission = {"md_package": " AMBER ", "force_field": "GLYCAM06", "source_type": "puuid", "comments": " note "}
    expected = {"extra_keywords": ["N-Glycan", "Curated", "MD:AMBER", "FF:GLYCAM06", "Source:puuid"],
                "user_common_names": ["GM3", "Other"], "user_description": "A ganglioside.", "user_comments": "note"}
    assert extract_search_enrichment(glycan_record=glycan, submission_record=submission) == expected
    assert search_enrichment(keywords=glycan["keywords"], common_names=glycan["common_names"],
                             description=glycan["description"], package=" AMBER ", forcefield="GLYCAM06",
                             source_type="puuid", comments=" note ") == expected
    assert extract_search_enrichment() == {"extra_keywords": [], "user_common_names": [], "user_description": "",
                                           "user_comments": ""}
    assert entry_complete.identity_gaps({"archetype": {"iupac": "Man", "wurcs": "W", "glytoucan": "G"}})[:1] == [
        "archetype.iupac_extended"]
