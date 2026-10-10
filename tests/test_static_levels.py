"""Reducing a published static database to levels 1,2 (lib/static_levels.py, scripts/static_drop_levels.py).

Small synthetic databases, no network.  ``write_database`` writes a static
database the way the pipeline does for a set of levels: ``data.json`` with the
writer of the static stage, the tables with pandas, ``info.json`` in the
format of the process stage, the root files with the baker.  The database
written with levels 1,2,3 is the source of every test and the one written with
levels 1,2 is what the tool has to produce from it.  The same comparison on a
real run of the process stage and the static stage is in
``test_levels_setting.py``.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import pytest

import lib.config as config
from lib import static_levels, static_publish, static_ready
from lib.glybake import GlycoShapeBaker
from lib.storage import S3StorageBackend
from s3_fakes import FakeS3Client, reader_sync, reader_validate

REPO_ROOT = Path(__file__).resolve().parents[1]
IDS = ["GS00001", "GS00002", "GS00003"]
RDF = b"@prefix gs: <https://glycoshape.org/ontology/> .\n"
MISSING = {
    "missing_glytoucan.txt": b"WURCS=2.0/1,1,0/[a2122h-1x_1-5]/1/\n",
    "missing_wurcs.txt": b"glycoshape_id\tvariant\tiupac\tglycam\n",
    "missing_glytoucan_no_wurcs_gags.tsv": b"glycoshape_id\tvariant\tiupac\n",
}
LEVEL_FILES = ("dist.svg", "GLYCAM/alpha.pdb", "GLYCAM/beta.pdb", "PDB/alpha.pdb", "PDB/beta.pdb")
CLUSTERS = {1: 5, 2: 9, 3: 40}


def quiet(message):
    pass


# ---------------------------------------------------------------------------
# A synthetic static database
# ---------------------------------------------------------------------------
def glycan_name(glycan_id):
    return f"DManpa1-{int(glycan_id[2:]) + 1}DManpa1-OH"


def write_glycan(folder: Path, levels=(1, 2, 3), rows: int = 6) -> None:
    """The static folder of one glycan as the static stage writes it for these levels."""
    glycan_id, number = folder.name, int(folder.name[2:])
    name = glycan_name(glycan_id)

    def percentages(level):
        count = CLUSTERS[level]
        return {f"Cluster {index}": round(100.0 / count + (index == 0) * 1e-05, 4) for index in range(count)}

    data = {
        "archetype": {
            "ID": glycan_id, "name": name, "glycam": name[:-5], "iupac": f"Man(a1-{number + 1})Man",
            "glytoucan": f"G{number:05d}AA", "wurcs": f"WURCS=2.0/{number}", "mass": 342.12, "entropy": 1.5 + number,
            "clusters": percentages(1), "coverage_clusters": percentages(2),
            "cluster_levels": {
                f"level_{level}": {"n_clusters": CLUSTERS[level], "clusters": percentages(level)} for level in levels},
            "torsion_summary": {"count": 2, "glycosidic": 2}, "length": 2.0, "package": "AMBER", "temperature": 300,
        },
        "alpha": {"ID": glycan_id, "name": name, "iupac": f"Man(a1-{number + 1})Man(a1-", "glytoucan": f"G{number:05d}BB"},
        "beta": {"ID": glycan_id, "name": name, "iupac": f"Man(a1-{number + 1})Man(b1-", "glytoucan": f"G{number:05d}CC"},
        "search_meta": {"common_names": [f"mannobiose {number} µ"], "description": "", "keywords": ["Oligomannose"]},
    }
    static_ready.save_glycan_data(data, str(folder), "data")
    static_ready.save_glycan_data({}, str(folder), "glygen")
    static_ready.save_glycan_data({}, str(folder), "glycosmos")
    (folder / "snfg.svg").write_bytes(b"<svg/>")

    info = {
        "entropy_nats": 1.5 + number,
        "torsions": {"count": 2, "glycosidic": ["phi_1", "psi_1"]},
        "levels": [
            {"level": level, "n_clusters": CLUSTERS[level],
             "clusters": [{"cluster_id": index, "cluster_size_pct": 100.0 / CLUSTERS[level], "representative_idx": index,
                           "members_count": 3, "torsion_stats": {"phi_1": {"mean": -60.5 + index}}}
                          for index in range(CLUSTERS[level])]}
            for level in levels],
        "pca": {
            "pca_analysis": {"n_components": 2, "explained_variance_ratio": [0.6, 0.25]},
            "main_clustering": {"H_nats": 1.5 + number, "silhouette_scores": {"5": 0.4}},
            "coverage_clustering": {"selected_n_clusters": CLUSTERS[2]},
            **({"level_3_clustering": {"selected_n_clusters": CLUSTERS[3]}} if 3 in levels else {}),
        },
        "schema_version": 3,
    }
    output = folder / "output"
    output.mkdir()
    (output / "info.json").write_bytes(json.dumps(info, indent=2).encode("utf-8"))
    (output / "torparts.npz").write_bytes(b"PK\x03\x04 torparts " + glycan_id.encode())

    frames = pd.DataFrame({"frame": range(rows), "phi_1": [-65.31234 + index for index in range(rows)],
                           "psi_1": [1e-05 * (index + number) for index in range(rows)]})
    for level in (1, 2, 3):          # the process stage wrote all three; the static stage leaves out the others
        frames[f"level_{level}_cluster"] = [index % CLUSTERS[level] for index in range(rows)]
        frames[f"level_{level}_center"] = [index if index < 2 else None for index in range(rows)]
    frames["label, with a comma"] = [f'"{index}"' if index == 1 else "x" for index in range(rows)]
    other = [f"level_{level}_{kind}" for level in (1, 2, 3) if level not in levels for kind in ("cluster", "center")]
    (output / "torsion_all.csv").write_text(frames.drop(columns=other).to_csv(index=False), encoding="utf-8")
    (output / "torsion_glycosidic.csv").write_text(frames[["frame", "phi_1", "psi_1"]].to_csv(index=False), encoding="utf-8")
    (output / "pca.csv").write_text(f"frame,PC1,PC2\n0,0.{number},-1.5\n", encoding="utf-8")
    for level in levels:
        (output / f"torsion_level_{level}_reps.csv").write_text(f"cluster,phi_1\n0,{-60.5 - level}\n", encoding="utf-8")
        for relative in LEVEL_FILES:
            target = output / f"level_{level}" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(f"REMARK {glycan_id} level {level} {relative}\n".encode() * (level * 3))
    (folder / "landscape").mkdir()
    for index, landscape_file in enumerate(static_ready.LANDSCAPE_FILES):
        (folder / "landscape" / landscape_file).write_bytes(f"{glycan_id} landscape {index}".encode())


def write_database(root: Path, ids=IDS, levels=(1, 2, 3)) -> Path:
    """A static database of these glycans as the static stage and the finalization leave it."""
    for glycan_id in ids:
        (root / glycan_id).mkdir(parents=True)
        write_glycan(root / glycan_id, levels)
    baker = GlycoShapeBaker(root)
    baker.create_consolidated_json()
    baker.create_master_archive()
    for name, content in MISSING.items():
        (root / name).write_bytes(content)
    (root / "GLYCOSHAPE_RDF.ttl").write_bytes(RDF)
    baker.create_sync_manifest()
    return root


def files_of(root: Path) -> dict:
    root = Path(root)
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def tree_state(root: Path) -> dict:
    """Content, time and mode of every file and the names of all folders: what must not change in a source."""
    state = {}
    for path in sorted(Path(root).rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_file():
            status = path.stat()
            state[relative] = (hashlib.sha256(path.read_bytes()).hexdigest(), status.st_size, status.st_mtime_ns, status.st_mode)
        else:
            state[relative] = "folder"
    return state


def zip_members(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def is_zip(name: str) -> bool:
    return name.endswith(".zip")


@pytest.fixture
def source(tmp_path):
    """A static database of three glycans with Levels 1, 2 and 3."""
    return write_database(tmp_path / "source")


@pytest.fixture
def expected(tmp_path):
    """The same database written by the pipeline's writers with levels 1,2."""
    return write_database(tmp_path / "expected", levels=(1, 2))


def run(source, dest, **options):
    options.setdefault("log", quiet)
    return static_levels.drop_levels(source, dest, **options)


def assert_is_the_levels_1_2_database(dest: Path, expected: Path) -> None:
    """``dest`` is ``expected``: same files, same bytes, except the time stamps that archives and manifest carry."""
    got, want = files_of(dest), files_of(expected)
    assert sorted(got) == sorted(want)
    for name in want:
        if not is_zip(name) and name != "sync_manifest.json":
            assert got[name] == want[name], name
    for glycan_id in IDS:        # the archives: same members with the same content
        assert zip_members(dest / glycan_id / f"{glycan_id}.zip") == zip_members(expected / glycan_id / f"{glycan_id}.zip")
    master, wanted = zip_members(dest / "GlycoShape.zip"), zip_members(expected / "GlycoShape.zip")
    assert sorted(master) == sorted(wanted) == [f"{glycan_id}.zip" for glycan_id in IDS]
    for glycan_id in IDS:
        assert master[f"{glycan_id}.zip"] == (dest / glycan_id / f"{glycan_id}.zip").read_bytes()
    manifest, wanted = (json.loads((folder / "sync_manifest.json").read_bytes()) for folder in (dest, expected))
    for glycan_id in IDS:
        for name, entry in wanted["glycans"][glycan_id]["files"].items():
            if not is_zip(name):
                assert manifest["glycans"][glycan_id]["files"][name]["sha256"] == entry["sha256"], name
    assert not [name for name in got if "level_3" in name]
    assert not [name for name in got if any(part.startswith(".") for part in name.split("/"))]


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------
def test_the_synthetic_source_has_level_3_everywhere(source):
    files = files_of(source)
    assert len([name for name in files if name.startswith("GS00001/")]) == 31       # as a published folder
    assert "GS00001/output/level_3/PDB/beta.pdb" in files and "GS00001/output/torsion_level_3_reps.csv" in files
    assert b"level_3_center" in files["GS00001/output/torsion_all.csv"].split(b"\n", 1)[0]
    assert b"level_3_clustering" in files["GS00001/output/info.json"]
    assert "level_3" in json.loads(files["GS00001/data.json"])["archetype"]["cluster_levels"]
    assert "level_3" in json.loads(files["GLYCOSHAPE.json"])["GS00002"]["archetype"]["cluster_levels"]
    assert "µ" in files["GS00001/data.json"].decode("utf-8")                         # not escaped, as the stage writes it


@pytest.mark.parametrize("copy", [False, True])
def test_the_reduced_database_is_the_one_written_with_levels_1_2(source, expected, tmp_path, copy):
    before = tree_state(source)
    report = run(source, tmp_path / "dest", copy=copy)
    assert report["ok"] and report["finalized"] and report["error"] is None and report["warnings"] == []
    assert_is_the_levels_1_2_database(tmp_path / "dest", expected)
    assert tree_state(source) == before                                              # the source is only read

    totals = report["totals"]
    assert (totals["glycans"], totals["done"], totals["skipped"], totals["failed"]) == (3, 3, 0, 0)
    assert (totals["files_kept"], totals["files_rewritten"], totals["files_dropped"]) == (3 * 21, 3 * 3, 3 * 6)
    assert (totals["files_linked"], totals["files_copied"]) == ((0, 63) if copy else (63, 0))
    record = report["glycans"]["GS00002"]
    assert record["status"] == "done" and record["rewritten"] == ["data.json", "output/info.json", "output/torsion_all.csv"]
    assert record["dropped"] == sorted(
        [f"output/level_3/{name}" for name in LEVEL_FILES] + ["output/torsion_level_3_reps.csv"])
    assert len(record["kept"]) == 21 and "output/level_2/GLYCAM/beta.pdb" in record["kept"] and record["extra"] == []
    assert record["bytes_before"] == sum(len(c) for n, c in files_of(source / "GS00002").items())
    assert record["bytes_after"] == sum(len(c) for n, c in files_of(tmp_path / "dest" / "GS00002").items())
    assert record["bytes_after"] < record["bytes_before"] and record["archive_bytes_before"] and record["archive_bytes_after"]
    root = report["root"]
    assert root["regenerated"] == ["GLYCOSHAPE.json", "GlycoShape.zip", "sync_manifest.json"]
    assert root["kept"] == sorted(["GLYCOSHAPE_RDF.ttl", *MISSING]) and root["unknown_kept"] == []
    assert root["checks"]["manifest"]["accepted_by_the_publisher"] and root["checks"]["manifest"]["files"] == 3 * 25 + 6
    assert root["checks"]["inventory_against_source"] == {
        "entries_in_source": 3, "entries": 3, "only_in_source": [], "only_in_destination": [], "different": [], "same": 3}
    # every file that was taken from the source, kept or read to be rewritten, is what its manifest records
    compared = root["checks"]["source_files_against_source_manifest"]
    assert compared["same"] == 3 * 24 + 4 and compared["different"] == 0 and compared["not_in_source_manifest"] == 0
    assert record["rewritten_from_sha256"] == {
        name: hashlib.sha256((source / "GS00002" / name).read_bytes()).hexdigest() for name in record["rewritten"]}
    assert json.loads((tmp_path / "dest.work" / "report.json").read_text(encoding="utf-8")) == report


def test_unchanged_files_are_hard_links_unless_copied(source, tmp_path):
    run(source, tmp_path / "linked")
    run(source, tmp_path / "copied", copy=True)
    for name in ("GS00001/output/pca.csv", "GS00003/landscape/pca_model.joblib", "GS00002/output/level_2/PDB/beta.pdb",
                 "GS00002/snfg.svg"):
        assert os.path.samefile(source / name, tmp_path / "linked" / name), name
        assert not os.path.samefile(source / name, tmp_path / "copied" / name), name
        assert (tmp_path / "copied" / name).stat().st_mtime_ns == (source / name).stat().st_mtime_ns
    for name in ("GS00001/data.json", "GS00001/output/info.json", "GS00001/output/torsion_all.csv",
                 "GS00001/GS00001.zip", "GLYCOSHAPE.json", "GlycoShape.zip", "sync_manifest.json",
                 "GLYCOSHAPE_RDF.ttl", "missing_wurcs.txt"):
        assert (tmp_path / "linked" / name).stat().st_nlink == 1, name               # written or copied, never linked
    assert (tmp_path / "linked" / "GLYCOSHAPE_RDF.ttl").stat().st_mtime_ns == (source / "GLYCOSHAPE_RDF.ttl").stat().st_mtime_ns


def test_files_that_the_pipeline_rewrites_in_place_are_not_links(expected, tmp_path):
    """glymeta writes data.json, the finalization the root files, with open(path, "w")."""
    from lib.json2rdf import convert_glycoshape_to_rdf

    dest = tmp_path / "dest"
    report = run(expected, dest)                       # nothing to reduce: every data.json is kept
    assert report["ok"] and report["totals"]["files_rewritten"] == 0
    assert (report["totals"]["files_linked"], report["totals"]["files_copied"]) == (3 * 23, 3 * 1)
    before = tree_state(expected)
    for glycan_id in IDS:
        with open(dest / glycan_id / "data.json", "w", encoding="utf-8") as handle:
            handle.write("{}")
    baker = GlycoShapeBaker(dest)
    baker.create_consolidated_json()
    baker.create_master_archive()
    for name in MISSING:
        with open(dest / name, "w", encoding="utf-8") as handle:
            handle.write("rewritten\n")
    (dest / "GLYCOSHAPE.json").write_text(json.dumps({"GS00001": {"archetype": {"ID": "GS00001"}}}), encoding="utf-8")
    assert convert_glycoshape_to_rdf(str(dest / "GLYCOSHAPE.json"), str(dest / "GLYCOSHAPE_RDF.ttl")) is not None
    baker.create_sync_manifest()
    assert (dest / "GLYCOSHAPE_RDF.ttl").read_bytes() != RDF
    assert tree_state(expected) == before


def test_a_source_without_the_level_3_folders_gives_the_same_database(source, expected, tmp_path):
    for glycan_id in IDS:                      # the copy that was transferred without them
        shutil.rmtree(source / glycan_id / "output" / "level_3")
    (source / "GS00003" / "output" / "torsion_level_3_reps.csv").unlink()
    before = tree_state(source)
    report = run(source, tmp_path / "dest")
    assert report["ok"]
    assert_is_the_levels_1_2_database(tmp_path / "dest", expected)
    assert tree_state(source) == before
    assert report["glycans"]["GS00001"]["dropped"] == ["output/torsion_level_3_reps.csv"]
    assert report["glycans"]["GS00003"]["dropped"] == []
    assert report["glycans"]["GS00001"]["rewritten"] == ["data.json", "output/info.json", "output/torsion_all.csv"]
    # the manifest of the source still lists the removed files; the others have the content it records
    compared = report["root"]["checks"]["source_files_against_source_manifest"]
    assert compared["same"] == 3 * 24 + 4 and compared["different"] == 0 and report["warnings"] == []


def test_a_database_that_has_only_levels_1_2_is_kept_as_it_is(expected, tmp_path):
    report = run(expected, tmp_path / "dest")
    assert report["ok"]
    assert_is_the_levels_1_2_database(tmp_path / "dest", expected)
    assert report["totals"]["files_rewritten"] == 0 and report["totals"]["files_dropped"] == 0
    assert report["glycans"]["GS00001"]["kept"][0] == "data.json" and len(report["glycans"]["GS00001"]["kept"]) == 24
    assert os.path.samefile(expected / "GS00001" / "output" / "info.json", tmp_path / "dest" / "GS00001" / "output" / "info.json")


def test_the_result_does_not_depend_on_the_configured_levels(source, expected, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "cluster_levels", (1, 2, 3))
    assert run(source, tmp_path / "dest", levels="1,2")["ok"]
    assert_is_the_levels_1_2_database(tmp_path / "dest", expected)


def test_keeping_levels_1_2_3_changes_no_glycan_file(source, tmp_path):
    report = run(source, tmp_path / "dest", levels=(1, 2, 3))
    assert report["ok"] and report["totals"]["files_rewritten"] == 0 and report["totals"]["files_dropped"] == 0
    got, want = files_of(tmp_path / "dest"), files_of(source)
    assert sorted(got) == sorted(want)
    assert all(got[name] == want[name] for name in want if not is_zip(name) and name != "sync_manifest.json")


def test_worker_processes_give_the_same_database(source, expected, tmp_path):
    before = tree_state(source)
    report = run(source, tmp_path / "dest", workers=3)
    assert report["ok"] and report["workers"] == 3 and report["totals"]["done"] == 3
    assert_is_the_levels_1_2_database(tmp_path / "dest", expected)
    assert tree_state(source) == before


# ---------------------------------------------------------------------------
# Files the static stage does not write
# ---------------------------------------------------------------------------
def test_an_unexpected_file_is_kept_and_reported(source, expected, tmp_path):
    (source / "GS00002" / "notes.txt").write_bytes(b"checked by hand\n")
    (source / "GS00002" / "output" / "level_1" / "extra.pdb").write_bytes(b"REMARK extra\n")
    (source / "GS00002" / "output" / "level_3" / "extra.pdb").write_bytes(b"REMARK of a dropped level\n")
    (source / "GS00002" / "PDB_format_ATOM").mkdir()
    (source / "GS00002" / "PDB_format_ATOM" / "cluster0.pdb").write_bytes(b"REMARK old layout\n")
    before = tree_state(source)
    report = run(source, tmp_path / "dest")
    assert report["ok"] and tree_state(source) == before
    extra = ["PDB_format_ATOM/cluster0.pdb", "notes.txt", "output/level_1/extra.pdb"]
    record = report["glycans"]["GS00002"]
    assert record["extra"] == extra and set(extra) <= set(record["kept"]) and len(record["kept"]) == 21 + 3
    assert "output/level_3/extra.pdb" in record["dropped"]
    assert report["glycans"]["GS00001"]["extra"] == [] and report["totals"]["files_extra"] == 3
    got = files_of(tmp_path / "dest" / "GS00002")
    for name in extra:
        assert got[name] == (source / "GS00002" / name).read_bytes()
    assert sorted(set(got) - set(files_of(expected / "GS00002"))) == extra
    manifest = json.loads((tmp_path / "dest" / "sync_manifest.json").read_bytes())
    assert set(extra) <= set(manifest["glycans"]["GS00002"]["files"])
    # the baker puts the old format folders and everything of level_1 into the archive
    members = zip_members(tmp_path / "dest" / "GS00002" / "GS00002.zip")
    assert members["extra.pdb"] == b"REMARK extra\n" and members["PDB_format_ATOM/cluster0.pdb"] == b"REMARK old layout\n"


def test_hidden_files_do_not_reach_the_destination(source, expected, tmp_path):
    (source / ".DS_Store").write_bytes(b"finder")
    (source / ".sync").mkdir()
    (source / ".sync" / "state").write_bytes(b"x")
    (source / "GS00001" / ".data.json.swp").write_bytes(b"vim")
    (source / "GS00001" / "output" / ".cache").mkdir()
    (source / "GS00001" / "output" / ".cache" / "table.bin").write_bytes(b"cache")
    report = run(source, tmp_path / "dest")
    assert report["ok"]
    assert_is_the_levels_1_2_database(tmp_path / "dest", expected)        # exactly the files of the database
    assert report["glycans"]["GS00001"]["hidden_skipped"] == [".data.json.swp", "output/.cache/table.bin"]
    assert report["root"]["hidden_skipped"] == [".DS_Store", ".sync"]
    assert [path for path in (tmp_path / "dest").rglob(".*")] == []
    # the tool's own files are next to the destination, not in it
    assert sorted(path.name for path in (tmp_path / "dest.work").iterdir()) == ["glycans", "lock", "report.json", "state.json"]


def test_root_files_the_tool_does_not_know_are_copied_and_listed(source, tmp_path):
    (source / "README.txt").write_bytes(b"GlycoShape static database\n")
    (source / "GS_notes.txt").write_bytes(b"a file, not a glycan folder\n")
    (source / "sync_manifest.json.tmp").write_bytes(b"left by an interrupted finalization")
    (source / "logs").mkdir()
    (source / "logs" / "run.log").write_bytes(b"log")
    before = tree_state(source)
    report = run(source, tmp_path / "dest")
    assert report["ok"] and tree_state(source) == before
    assert report["root"]["unknown_kept"] == ["GS_notes.txt", "README.txt"]
    assert report["root"]["ignored_directories"] == ["logs"]
    # the finalization writes the manifest by way of this name: it is never taken from the source
    assert report["root"]["left_out"] == ["sync_manifest.json.tmp"]
    assert (tmp_path / "dest" / "README.txt").read_bytes() == b"GlycoShape static database\n"
    assert not os.path.samefile(source / "README.txt", tmp_path / "dest" / "README.txt")      # root files are copies
    assert not (tmp_path / "dest" / "logs").exists()                        # the manifest lists no such folder
    manifest = json.loads((tmp_path / "dest" / "sync_manifest.json").read_bytes())
    assert sorted(manifest["root_files"]) == sorted(
        ["GLYCOSHAPE.json", "GLYCOSHAPE_RDF.ttl", "GlycoShape.zip", "GS_notes.txt", "README.txt", *MISSING])
    assert sorted(files_of(tmp_path / "dest")) == sorted(
        [*manifest["root_files"], "sync_manifest.json"]
        + [entry["path"] for glycan in manifest["glycans"].values() for entry in glycan["files"].values()])


def test_a_data_json_in_another_format_is_reduced_and_reported(source, tmp_path):
    path = source / "GS00003" / "data.json"
    data = json.loads(path.read_bytes())
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")            # edited by another program
    report = run(source, tmp_path / "dest")
    assert report["ok"]
    written = (tmp_path / "dest" / "GS00003" / "data.json").read_bytes()
    del data["archetype"]["cluster_levels"]["level_3"]
    assert json.loads(written) == data and written == json.dumps(data, indent=4, ensure_ascii=False).encode("utf-8")
    assert report["warnings"] == [
        "1 file(s) of the source do not have the content that its sync_manifest.json records, "
        "first ['GS00003/data.json']",
        "GS00003: data.json was not in the format of the static stage; it is now (same content)"]
    assert report["glycans"]["GS00001"]["notes"] == []


# ---------------------------------------------------------------------------
# Failures and resuming
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("workers", [1, 2])
def test_a_broken_info_json_fails_that_glycan_only_and_the_next_run_finishes(source, expected, tmp_path, workers):
    good = (source / "GS00002" / "output" / "info.json").read_bytes()
    (source / "GS00002" / "output" / "info.json").write_bytes(good[: len(good) // 2])
    before = tree_state(source)
    dest, work = tmp_path / "dest", tmp_path / "dest.work"
    report = run(source, dest, workers=workers)
    assert not report["ok"] and not report["finalized"] and report["root"] is None
    assert report["error"] == "1 glycan(s) failed: GS00002"
    assert report["glycans"]["GS00002"]["status"] == "failed"
    assert report["glycans"]["GS00002"]["error"].startswith("ValueError: output/info.json cannot be used")
    assert [report["glycans"][name]["status"] for name in ("GS00001", "GS00003")] == ["done", "done"]
    assert report["totals"]["failed"] == 1 and report["totals"]["done"] == 2
    # nothing of the failed glycan and no root file in the destination; nothing left in the staging folder
    assert sorted(path.name for path in dest.iterdir()) == ["GS00001", "GS00003"]
    for glycan_id in ("GS00001", "GS00003"):
        want = {name: content for name, content in files_of(expected / glycan_id).items() if not is_zip(name)}
        assert files_of(dest / glycan_id) == want
    assert not (work / "staging").exists() and sorted(path.name for path in (work / "glycans").iterdir()) == [
        "GS00001.json", "GS00003.json"]
    assert tree_state(source) == before

    # the entry is repaired in the source and the command is run again
    (source / "GS00002" / "output" / "info.json").write_bytes(good)
    finished = {name: (dest / name / "output" / "info.json").stat().st_mtime_ns for name in ("GS00001", "GS00003")}
    report = run(source, dest, workers=workers)
    assert report["ok"] and report["finalized"] and report["resumed"]
    assert [report["glycans"][name]["status"] for name in IDS] == ["skipped", "done", "skipped"]
    assert finished == {name: (dest / name / "output" / "info.json").stat().st_mtime_ns for name in finished}
    assert report["glycans"]["GS00001"]["rewritten"] == ["data.json", "output/info.json", "output/torsion_all.csv"]
    assert report["totals"]["files_dropped"] == 3 * 6                         # the records of the first run are used
    assert_is_the_levels_1_2_database(dest, expected)


@pytest.mark.parametrize("damage, message", [
    (lambda folder: (folder / "data.json").write_bytes(b"{not json"), "data.json cannot be used"),
    (lambda folder: (folder / "data.json").write_bytes(b'["a list"]'), "data.json has no archetype entry"),
    (lambda folder: (folder / "output" / "info.json").write_bytes(b'{"pca": {}}'), "has no list of levels"),
    (lambda folder: (folder / "output" / "info.json").write_bytes(b'{"levels": [{"level": 1}]}'),
     "does not describe level 2"),
    (lambda folder: (folder / "output" / "torsion_all.csv").write_bytes(b"frame,level_3_cluster\n\xff\xfe,1\n"),
     "UnicodeDecodeError"),
    (lambda folder: (folder / "output" / "level_2" / "PDB" / "alpha.pdb").unlink(),
     "incomplete for levels 1,2: no output/level_2/PDB/alpha.pdb"),
    (lambda folder: (folder / "landscape" / "pca_model.joblib").unlink(), "no landscape/pca_model.joblib"),
    (lambda folder: (folder / "data.json").unlink(), "incomplete for levels 1,2: no data.json"),
])
def test_a_folder_that_is_not_a_complete_static_folder_fails(source, tmp_path, damage, message):
    damage(source / "GS00001")
    report = run(source, tmp_path / "dest")
    assert not report["ok"] and report["glycans"]["GS00001"]["status"] == "failed"
    assert message in report["glycans"]["GS00001"]["error"], report["glycans"]["GS00001"]["error"]
    assert sorted(path.name for path in (tmp_path / "dest").iterdir()) == ["GS00002", "GS00003"]


def test_a_second_run_changes_nothing(source, tmp_path):
    dest = tmp_path / "dest"
    assert run(source, dest)["ok"]
    before = tree_state(dest)
    messages = []
    report = run(source, dest, log=messages.append)
    assert report["ok"] and report["finalized"] and report["resumed"] and report["root"] is None
    assert report["totals"]["skipped"] == 3 and report["totals"]["done"] == 0
    assert report["totals"]["files_kept"] == 63 and report["glycans"]["GS00001"]["archive_bytes_after"]
    assert tree_state(dest) == before
    assert any("is complete" in message for message in messages)
    # the root files are written again when the manifest is taken away
    (dest / "sync_manifest.json").unlink()
    report = run(source, dest)
    assert report["ok"] and report["root"]["regenerated"][-1] == "sync_manifest.json"
    assert {name: state for name, state in tree_state(dest).items() if "/" in name and not is_zip(name)} == {
        name: state for name, state in before.items() if "/" in name and not is_zip(name)}


def test_an_interrupted_run_leaves_finished_folders_and_is_continued(source, expected, tmp_path, monkeypatch):
    run_glycan = static_levels._run_glycan

    def stop_at_the_second(task):
        if task["glycan_id"] == "GS00002":
            (Path(task["staging"]) / "GS00002" / "output").mkdir(parents=True)        # in the middle of a folder
            raise KeyboardInterrupt
        return run_glycan(task)

    monkeypatch.setattr(static_levels, "_run_glycan", stop_at_the_second)
    dest, work = tmp_path / "dest", tmp_path / "dest.work"
    with pytest.raises(KeyboardInterrupt):
        run(source, dest)
    report = json.loads((work / "report.json").read_text(encoding="utf-8"))
    assert not report["ok"] and report["error"] == "interrupted" and not report["finalized"]
    assert [report["glycans"][name]["status"] for name in IDS] == ["done", "pending", "pending"]
    assert sorted(path.name for path in dest.iterdir()) == ["GS00001"] and not (work / "staging").exists()
    monkeypatch.setattr(static_levels, "_run_glycan", run_glycan)
    report = run(source, dest)
    assert report["ok"] and [report["glycans"][name]["status"] for name in IDS] == ["skipped", "done", "done"]
    assert_is_the_levels_1_2_database(dest, expected)


def test_glycan_folders_only_then_the_database(source, expected, tmp_path):
    dest = tmp_path / "dest"
    report = run(source, dest, finalize=False)
    assert report["ok"] and not report["finalized"] and report["root"] is None
    assert sorted(path.name for path in dest.iterdir()) == IDS                # what the static stage writes, no archive
    for glycan_id in IDS:
        want = {name: content for name, content in files_of(expected / glycan_id).items() if not is_zip(name)}
        assert files_of(dest / glycan_id) == want
    assert report["glycans"]["GS00001"]["archive_bytes_after"] is None
    report = run(source, dest)
    assert report["ok"] and report["finalized"] and report["totals"]["skipped"] == 3
    assert_is_the_levels_1_2_database(dest, expected)


def test_folders_that_cannot_be_used_are_refused(source, tmp_path):
    def refused(*arguments, **options):
        with pytest.raises(static_levels.StaticLevelsError) as caught:
            run(*arguments, **options)
        return str(caught.value)

    assert "not a folder" in refused(tmp_path / "missing", tmp_path / "dest")
    (tmp_path / "empty").mkdir()
    assert "no glycan folder" in refused(tmp_path / "empty", tmp_path / "dest")
    assert "outside the source" in refused(source, source)
    assert "outside the source" in refused(source, source / "reduced")
    assert "outside the source" in refused(source / "GS00001", source)
    assert "work folder must be outside" in refused(source, tmp_path / "dest", work_dir=source / "work")
    assert "work folder must be outside" in refused(source, tmp_path / "dest", work_dir=tmp_path / "dest" / ".work")
    assert "report must be outside" in refused(source, tmp_path / "dest", report_path=tmp_path / "dest" / "report.json")
    assert "cluster levels must be 1,2 or 1,2,3" in refused(source, tmp_path / "dest", levels="1")
    assert tree_state(source)["GS00001/data.json"] and not (tmp_path / "dest").exists()

    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "file.txt").write_bytes(b"something else")
    assert "is not empty" in refused(source, tmp_path / "other")
    assert files_of(tmp_path / "other") == {"file.txt": b"something else"}

    assert run(source, tmp_path / "dest", finalize=False)["ok"]
    assert "was started from" in refused(source, tmp_path / "dest", levels=(1, 2, 3))
    second = write_database(tmp_path / "second", ids=["GS00009"])
    assert "was started from" in refused(second, tmp_path / "dest")
    shutil.rmtree(tmp_path / "dest.work")                                   # its records are gone: not resumable
    assert "is not empty" in refused(source, tmp_path / "dest")


def test_two_runs_do_not_share_a_work_folder(source, tmp_path):
    import fcntl

    (tmp_path / "dest.work").mkdir()
    with open(tmp_path / "dest.work" / "lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(static_levels.StaticLevelsError, match="another run"):
            run(source, tmp_path / "dest")
    assert run(source, tmp_path / "dest")["ok"]


def test_a_stale_inventory_or_manifest_of_the_source_is_reported(source, tmp_path):
    inventory = json.loads((source / "GLYCOSHAPE.json").read_bytes())
    inventory["GS00002"]["archetype"]["mass"] = 1.0                       # not what data.json says
    inventory["GS00777"] = inventory["GS00001"]
    (source / "GLYCOSHAPE.json").write_text(json.dumps(inventory), encoding="utf-8")
    (source / "GS00003" / "snfg.svg").write_bytes(b"<svg>changed after the manifest</svg>")
    table = (source / "GS00001" / "output" / "torsion_all.csv").read_bytes()
    (source / "GS00001" / "output" / "torsion_all.csv").write_bytes(table.replace(b"-65.31234", b"-65.31235"))
    report = run(source, tmp_path / "dest")
    assert report["ok"] and len(report["warnings"]) == 2
    compared = report["root"]["checks"]["inventory_against_source"]
    assert compared["different"] == ["GS00002"] and compared["only_in_source"] == ["GS00777"] and compared["same"] == 2
    compared = report["root"]["checks"]["source_files_against_source_manifest"]      # a kept file and a rewritten one
    assert compared["different_first"] == ["GS00001/output/torsion_all.csv", "GS00003/snfg.svg"]
    assert compared["same"] == 3 * 24 + 4 - 2 and "2 file(s) of the source do not have the content" in report["warnings"][1]
    # the new inventory is the data.json files, as the finalization writes it
    assert json.loads((tmp_path / "dest" / "GLYCOSHAPE.json").read_bytes())["GS00002"]["archetype"]["mass"] == 342.12


# ---------------------------------------------------------------------------
# What the pipeline and the publisher require of the result
# ---------------------------------------------------------------------------
def test_the_required_files_are_those_of_the_static_only_check():
    from main import GlycanPipelineRunner

    runner = GlycanPipelineRunner.__new__(GlycanPipelineRunner)
    for levels in ((1, 2), (1, 2, 3)):
        config_levels = config.cluster_levels
        config.cluster_levels = levels
        try:
            assert static_levels.required_files(levels) == ["data.json"] + runner._get_static_only_required_files()
            assert set(static_levels.LEVEL_FILES) == set(runner.REQUIRED_GLYSTATIC_LEVEL_FILES)
        finally:
            config.cluster_levels = config_levels


def test_the_reduced_folders_are_complete_for_the_pipeline(source, tmp_path, monkeypatch):
    from lib.storage import get_storage_manager
    from main import GlycanPipelineRunner

    assert run(source, tmp_path / "dest")["ok"]
    runner = GlycanPipelineRunner.__new__(GlycanPipelineRunner)
    runner.storage = get_storage_manager()
    runner._wurcs_glytoucan_cache, runner._iupac_wurcs_cache = {}, {}      # every entry has its GlyTouCan ID: no lookup
    monkeypatch.setattr(config, "cluster_levels", (1, 2))
    for glycan_id in IDS:
        folder = tmp_path / "dest" / glycan_id
        assert runner._static_only_output_missing_requirements(folder, glycan_name(glycan_id)) == []
        assert runner._validate_glystatic_output_dir(str(folder), glycan_name=glycan_name(glycan_id)) == []
    assert runner._wurcs_glytoucan_cache == {} and runner._iupac_wurcs_cache == {}
    monkeypatch.setattr(config, "cluster_levels", (1, 2, 3))               # and it is no longer a Level 3 folder
    assert "output/torsion_level_3_reps.csv" in runner._static_only_output_missing_requirements(
        tmp_path / "dest" / "GS00001", glycan_name("GS00001"))


def test_the_publisher_and_the_website_accept_the_reduced_database(source, expected, tmp_path, fake_s3):
    backend, client = fake_s3
    dest = tmp_path / "dest"
    assert run(source, dest)["ok"]
    manifest = json.loads((dest / "sync_manifest.json").read_bytes())
    assert static_publish.validate_manifest(manifest) == []
    reader_validate(manifest)
    assert set(static_publish.manifest_files(manifest)) == set(files_of(dest)) - {"sync_manifest.json"}
    assert manifest["glycan_count"] == 3 and manifest["glycans"]["GS00001"]["file_count"] == 25

    plan = static_publish.publish(dest, backend, dry_run=True, log=quiet)
    assert (plan["glycans"], plan["files"], plan["new"], plan["changed"]) == (3, 3 * 25 + 6, 3 * 25 + 6, 0)
    assert client.writes() == []
    done = static_publish.publish(dest, backend, log=quiet)
    assert done["manifest_written"] and done["new"] == 81
    checked = static_publish.check_published(backend, log=quiet)
    assert checked["ok"] and checked["errors"] == [] and checked["unlisted"] == [] and checked["files"] == 81
    reader_sync(backend, "static", tmp_path / "website")                   # the website downloads and verifies it
    assert files_of(tmp_path / "website") == files_of(dest)

    # Over the published database with Level 3: only what was rewritten is sent, the Level 3 files become stale.
    published = S3StorageBackend(FakeS3Client("bucket"), bucket="bucket", sleep=lambda seconds: None)
    static_publish.publish(source, published, log=quiet)
    reader_sync(published, "static", tmp_path / "website_with_3")
    assert files_of(tmp_path / "website_with_3") == files_of(source)
    plan = static_publish.publish(dest, published, dry_run=True, log=quiet)
    assert plan["changed"] == 3 * 4 + 2 and plan["new"] == 0               # three files and the archive per glycan
    assert plan["unchanged"] == 3 * 21 + 4                                 # kept files, the RDF, the missing lists
    assert plan["stale"] == 3 * 6
    done = static_publish.publish(dest, published, delete_stale=True, log=quiet)
    assert done["stale_deleted"] == 18 and static_publish.check_published(published, log=quiet)["ok"]
    reader_sync(published, "static", tmp_path / "website_with_3")          # the website moves to the new database
    assert files_of(tmp_path / "website_with_3") == files_of(dest)


def test_the_rdf_does_not_depend_on_the_levels(source, tmp_path):
    rdflib = pytest.importorskip("rdflib")
    from rdflib.compare import isomorphic

    from lib.json2rdf import convert_glycoshape_to_rdf

    assert run(source, tmp_path / "dest")["ok"]
    assert (tmp_path / "dest" / "GLYCOSHAPE_RDF.ttl").read_bytes() == (source / "GLYCOSHAPE_RDF.ttl").read_bytes()
    with_3 = convert_glycoshape_to_rdf(str(source / "GLYCOSHAPE.json"), str(tmp_path / "with_3.ttl"))
    without = convert_glycoshape_to_rdf(str(tmp_path / "dest" / "GLYCOSHAPE.json"), str(tmp_path / "without.ttl"))
    assert len(with_3) == len(without) > 30 and isomorphic(with_3, without)
    assert isinstance(with_3, rdflib.Graph) and "level_3" not in (tmp_path / "with_3.ttl").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The pieces
# ---------------------------------------------------------------------------
def test_classify():
    def kind(name, levels=(1, 2)):
        return static_levels.classify(name, "GS00042", levels)

    assert [kind(name) for name in ("glygen.json", "glycosmos.json", "snfg.svg", "output/torparts.npz",
                                    "landscape/preprocessing.npz", "output/level_2/GLYCAM/beta.pdb",
                                    "output/level_1/dist.svg")] == ["keep"] * 7
    assert [kind(name) for name in ("output/pca.csv", "output/torsion_all.csv", "output/torsion_glycosidic.csv",
                                    "output/torsion_level_2_reps.csv")] == ["table"] * 4
    assert (kind("data.json"), kind("output/info.json"), kind("GS00042.zip")) == ("data", "info", "archive")
    assert [kind(name) for name in ("output/level_3/dist.svg", "output/level_3/PDB/alpha.pdb", "output/level_3/x/y.txt",
                                    "output/torsion_level_3_reps.csv", "output/level_12/dist.svg")] == ["drop"] * 5
    assert kind("output/level_3/dist.svg", (1, 2, 3)) == "keep" and kind("output/torsion_level_3_reps.csv", (1, 2, 3)) == "table"
    assert [kind(name) for name in ("notes.txt", "GS00001.zip", "output/level_1/extra.pdb", "output/other.csv",
                                    "output/data.json", "landscape/more.npz", "output/level_x/dist.svg",
                                    "output/torsion_level_3_reps.csv.bak")] == ["extra"] * 8
    assert [kind(name) for name in (".DS_Store", "output/.cache/x", "output/level_3/.hidden", ".git/config")] == ["hidden"] * 4


def test_data_without_other_levels():
    data = {"archetype": {"ID": "GS1", "clusters": {"Cluster 0": 100.0},
                          "cluster_levels": {"level_1": {"n": 5}, "level_2": {"n": 9}, "level_3": {"n": 512}, "note": 1}},
            "alpha": {"cluster_levels": {"level_3": "not written by the pipeline"}}}
    reduced, removed = static_levels.data_without_other_levels(data, (1, 2))
    assert removed == ["level_3"] and list(reduced["archetype"]["cluster_levels"]) == ["level_1", "level_2", "note"]
    assert list(reduced) == list(data) and list(reduced["archetype"]) == list(data["archetype"])
    assert "level_3" in data["archetype"]["cluster_levels"] and reduced["alpha"] is data["alpha"]       # input untouched
    assert static_levels.data_without_other_levels(data, (1, 2, 3)) == (data, [])
    assert static_levels.data_without_other_levels(reduced, (1, 2)) == (reduced, [])
    for other in ({}, {"archetype": None}, {"archetype": {"cluster_levels": []}}, [1, 2], None):
        assert static_levels.data_without_other_levels(other, (1, 2)) == (other, [])


def test_reduce_glycan_leaves_no_part_of_itself_behind(source, expected, tmp_path):
    record = static_levels.reduce_glycan(source / "GS00001", tmp_path / "staged" / "GS00001", (1, 2))
    want = {name: content for name, content in files_of(expected / "GS00001").items() if not is_zip(name)}
    assert files_of(tmp_path / "staged" / "GS00001") == want
    assert sorted(path.name for path in (tmp_path / "staged").iterdir()) == ["GS00001"]
    assert record["bytes_staged"] == sum(len(content) for content in want.values())
    assert record["archive_bytes_before"] == (source / "GS00001" / "GS00001.zip").stat().st_size
    with pytest.raises(FileExistsError):                                    # never into a folder that is there
        static_levels.reduce_glycan(source / "GS00001", tmp_path / "staged" / "GS00001", (1, 2))


def test_the_storage_of_the_process_is_not_used_and_is_put_back(source, tmp_path, monkeypatch):
    from lib import storage

    class Refusing(storage.LocalStorageBackend):
        def __getattribute__(self, name):
            if name in ("open_text", "write_text", "write_binary", "read_binary", "exists", "mkdir"):
                raise AssertionError(f"the storage of the process was used: {name}")
            return super().__getattribute__(name)

    manager = storage.StorageManager(Refusing())
    monkeypatch.setattr(storage, "_storage_manager", manager)
    assert run(source, tmp_path / "dest")["ok"]
    assert storage._storage_manager is manager


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------
def load_script():
    spec = importlib.util.spec_from_file_location("static_drop_levels", REPO_ROOT / "scripts" / "static_drop_levels.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_command(source, expected, tmp_path, capsys):
    script = load_script()
    good = (source / "GS00003" / "output" / "info.json").read_bytes()
    (source / "GS00003" / "output" / "info.json").write_bytes(b"")
    dest, report_file = tmp_path / "dest", tmp_path / "reports" / "report.json"
    report_file.parent.mkdir()
    arguments = ["--source", str(source), "--dest", str(dest), "--workers", "1", "--report", str(report_file)]
    assert script.main(arguments) == 1
    output = capsys.readouterr()
    assert "ERROR GS00003: ValueError: output/info.json cannot be used" in output.err
    assert "1 glycan(s) failed: GS00003; run the command again to continue" in output.err
    assert "2 glycans done, 0 already there, 1 failed" in output.out and "root files not written" in output.out
    assert json.loads(report_file.read_text(encoding="utf-8"))["glycans"]["GS00003"]["status"] == "failed"
    assert not (dest / "sync_manifest.json").exists() and not (dest / "GS00003").exists()

    (source / "GS00003" / "output" / "info.json").write_bytes(good)
    assert script.main(arguments) == 0
    output = capsys.readouterr()
    assert "1 glycans done, 2 already there, 0 failed" in output.out and "root files written" in output.out
    assert "wrote sync_manifest.json" in output.out and "ERROR" not in output.err and "WARNING" not in output.err
    assert_is_the_levels_1_2_database(dest, expected)
    assert not (tmp_path / "dest.work" / "report.json").exists()            # --report says where it goes

    assert script.main(["--source", str(source), "--dest", str(source / "inside")]) == 2
    assert "must be a new folder outside the source" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        script.main(["--source", str(source), "--dest", str(dest), "--workers", "0"])
