"""The set of clustering levels is a setting (default 1,2): process stage and static stage.

Synthetic trajectory, no network. The process stage is run once with levels 1,2,3 and once with the
default 1,2 (module fixture); the tests compare the two folders and what the static stage makes of them.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")

import polars as pl
import pyarrow.parquet as pq
import pytest

import lib.config as config
from lib import static_ready, v2
from lib.v2 import GlycanAnalysisPipeline

from synthetic_glycan import GLYCAM_NAME, write_source


@contextlib.contextmanager
def cluster_levels(levels):
    previous = config.cluster_levels
    config.cluster_levels = tuple(levels)
    try:
        yield
    finally:
        config.cluster_levels = previous


def files_of(root):
    root = Path(root)
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def table(path):
    return pl.from_arrow(pq.read_table(path))


def run_process(data_root, process_root, levels):
    with cluster_levels(levels):
        results = GlycanAnalysisPipeline(str(data_root), str(process_root)).run_analysis(GLYCAM_NAME)
    assert results["success"], results["error_message"]
    return Path(process_root) / GLYCAM_NAME


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("levels")
    write_source(root / "data", broken=(11, 200))
    return {
        "root": root,
        "data": root / "data",
        "with_3": run_process(root / "data", root / "process_123", (1, 2, 3)),
        "default": run_process(root / "data", root / "process_12", (1, 2)),
    }


# ---------------------------------------------------------------------------
# The setting
# ---------------------------------------------------------------------------
def test_default_is_levels_1_and_2():
    assert config.cluster_levels == (1, 2)
    assert config._cluster_levels_env("GLYCOSHAPE_TEST_NO_SUCH_VARIABLE", (1, 2)) == (1, 2)


@pytest.mark.parametrize("raw,expected", [("1,2", (1, 2)), ("1,2,3", (1, 2, 3)), (" 3, 1 ,2 ", (1, 2, 3)), ([2, 1], (1, 2))])
def test_parse_cluster_levels(raw, expected):
    assert config.parse_cluster_levels(raw) == expected


@pytest.mark.parametrize("raw", ["1", "1,3", "2,3", "1,2,4", "", "a,b", "0,1,2"])
def test_parse_cluster_levels_rejects_other_sets(raw):
    with pytest.raises(ValueError):
        config.parse_cluster_levels(raw)


def test_levels_from_the_environment(monkeypatch):
    monkeypatch.setenv("GLYCOSHAPE_TEST_LEVELS", "1,2,3")
    assert config._cluster_levels_env("GLYCOSHAPE_TEST_LEVELS", (1, 2)) == (1, 2, 3)
    monkeypatch.setenv("GLYCOSHAPE_TEST_LEVELS", "1,3")          # invalid: the default is kept
    assert config._cluster_levels_env("GLYCOSHAPE_TEST_LEVELS", (1, 2)) == (1, 2)


# ---------------------------------------------------------------------------
# Process stage
# ---------------------------------------------------------------------------
def test_default_run_computes_and_writes_nothing_of_level_3(runs):
    folder = runs["default"]
    names = sorted(files_of(folder))
    assert not [name for name in names if "level_3" in name]
    assert {"output/level_1/alpha.pdb", "output/level_1/beta.pdb", "output/level_1/dist.svg",
            "output/level_2/alpha.pdb", "output/level_2/beta.pdb", "output/level_2/dist.svg",
            "embedding/exports/torsion_level_1_reps.parquet",
            "embedding/exports/torsion_level_2_reps.parquet"} <= set(names)

    assert sorted(table(folder / "embedding/cluster_summary.parquet")["level"].unique().to_list()) == [1, 2]
    assert sorted(table(folder / "embedding/cluster_members.parquet")["level"].unique().to_list()) == [1, 2]
    assert json.loads((folder / "embedding/metadata.json").read_text())["levels"] == [1, 2]
    assert sorted(json.loads((folder / "embedding/clustering_results.json").read_text())) == ["1", "2"]
    columns = table(folder / "embedding/exports/torsion_all.parquet").columns
    assert {"level_1_cluster", "level_1_center", "level_2_cluster", "level_2_center"} <= set(columns)
    assert not [c for c in columns if c.startswith("level_3")]
    for name in ("embedding/info.json", "output/info.json", "embedding/torsion_stats.json"):
        document = json.loads((folder / name).read_text())
        assert [level["level"] for level in document["levels"]] == [1, 2], name
    assert "level_3_clustering" not in json.loads((folder / "embedding/pca.json").read_text())
    assert "level_3_clustering" not in json.loads((folder / "output/info.json").read_text())["pca"]
    text = "".join(files_of(folder)[name].decode("utf-8", "replace") for name in names if name.endswith(".json"))
    assert "level_3" not in text


def test_levels_1_2_3_behave_as_before(runs):
    folder = runs["with_3"]
    names = set(files_of(folder))
    assert {"output/level_3/alpha.pdb", "output/level_3/beta.pdb", "output/level_3/dist.svg",
            "embedding/exports/torsion_level_3_reps.parquet"} <= names
    assert sorted(table(folder / "embedding/cluster_summary.parquet")["level"].unique().to_list()) == [1, 2, 3]
    columns = table(folder / "embedding/exports/torsion_all.parquet").columns
    assert {"level_3_cluster", "level_3_center"} <= set(columns)
    info = json.loads((folder / "output/info.json").read_text())
    assert [level["level"] for level in info["levels"]] == [1, 2, 3]
    assert info["pca"]["level_3_clustering"]["selected_n_clusters"] == info["levels"][2]["n_clusters"]
    with cluster_levels((1, 2, 3)):
        status = v2.get_glycan_status(GLYCAM_NAME, str(runs["data"]), str(folder.parent))
    assert all(status["steps_status"].values()), status["steps_status"]


def test_levels_1_and_2_are_the_same_with_and_without_level_3(runs):
    with_3, default = runs["with_3"], runs["default"]
    a, b = files_of(with_3), files_of(default)

    # cluster assignments and representative frames
    for name in ("embedding/cluster_members.parquet", "embedding/cluster_summary.parquet"):
        assert table(with_3 / name).filter(pl.col("level") <= 2).equals(table(default / name)), name
    # representative structures, plots, exported tables: byte for byte
    for name in ("output/level_1/alpha.pdb", "output/level_1/beta.pdb", "output/level_2/alpha.pdb",
                 "output/level_2/beta.pdb", "embedding/exports/pca.parquet",
                 "embedding/exports/torsion_glycosidic.parquet",
                 "embedding/exports/torsion_level_1_reps.parquet",
                 "embedding/exports/torsion_level_2_reps.parquet",
                 "embedding/pca_conformation_landscape.parquet", "embedding/torsions.parquet",
                 "embedding/landscape/pca_model.joblib", "embedding/landscape/preprocessing.npz"):
        assert a[name] == b[name], name
    for level in (1, 2):
        name = f"output/level_{level}/dist.svg"                # ids are salted per figure, the drawing is the same
        assert comparable_svg(a[name]) == comparable_svg(b[name]), name
    all_3 = table(with_3 / "embedding/exports/torsion_all.parquet")
    assert all_3.drop(["level_3_cluster", "level_3_center"]).equals(table(default / "embedding/exports/torsion_all.parquet"))
    # info.json: the run without Level 3 writes what remains of the other one without its Level 3 entries
    reduced, changed = static_ready.filter_info_levels(json.loads(a["output/info.json"]), (1, 2))
    assert changed and json.dumps(reduced, indent=2).encode() == b["output/info.json"]


def test_folder_with_level_3_is_complete_and_left_alone(runs):
    folder = runs["with_3"]
    before = {name: (folder / name).stat().st_mtime_ns for name in files_of(folder)}
    with cluster_levels((1, 2)):
        status = v2.get_glycan_status(GLYCAM_NAME, str(runs["data"]), str(folder.parent))
        assert all(status["steps_status"].values()), status["steps_status"]
        with patch.object(v2, "kcenter_coverage_clustering", side_effect=AssertionError("clustering was recomputed")):
            results = GlycanAnalysisPipeline(str(runs["data"]), str(folder.parent)).run_analysis(GLYCAM_NAME)
    assert results["success"], results["error_message"]
    after = {name: (folder / name).stat().st_mtime_ns for name in files_of(folder)}
    changed = sorted(name for name in before if before[name] != after[name])
    # the skipped clustering step refreshes the entropy in pca.json; nothing else is touched
    assert changed == ["embedding/pca.json"]
    assert "level_3_clustering" in json.loads((folder / "embedding/pca.json").read_text())


def test_folder_without_level_3_is_incomplete_when_level_3_is_required(runs, tmp_path):
    process = tmp_path / "process"
    shutil.copytree(runs["default"], process / GLYCAM_NAME)
    with cluster_levels((1, 2, 3)):
        status = v2.get_glycan_status(GLYCAM_NAME, str(runs["data"]), str(process))["steps_status"]
        assert not status["clustering"] and not status["structures"] and not status["export"]
        results = GlycanAnalysisPipeline(str(runs["data"]), str(process)).run_analysis(GLYCAM_NAME)
        assert results["success"], results["error_message"]
        status = v2.get_glycan_status(GLYCAM_NAME, str(runs["data"]), str(process))["steps_status"]
    assert all(status.values()), status
    upgraded, reference = files_of(process / GLYCAM_NAME), files_of(runs["with_3"])
    assert sorted(upgraded) == sorted(reference)
    # Level 3 was added and everything indexed by clusters was rewritten: the folder is the one a run
    # with levels 1,2,3 writes (the compressed coordinate chunk is not byte-reproducible).
    for name in reference:
        if "coords.zarr/xyz/0." in name:
            continue
        if name.endswith("dist.svg"):
            assert comparable_svg(upgraded[name]) == comparable_svg(reference[name]), name
        else:
            assert upgraded[name] == reference[name], name


def comparable_svg(content: bytes) -> str:
    """Matplotlib salts the ids of clip paths and markers per figure and stamps the date."""
    text = re.sub(r"<dc:date>[^<]*</dc:date>", "", content.decode())
    tokens = {}
    return re.sub(r"(?<=[\"#])[A-Za-z]{1,2}[0-9a-f]{10}(?=[\")])",
                  lambda match: tokens.setdefault(match.group(0), f"id{len(tokens)}"), text)


def test_level_plots_drawn_in_worker_processes_are_the_serial_plots(runs, tmp_path, monkeypatch):
    folder = runs["with_3"]
    shutil.copytree(folder / "output", tmp_path / "output")
    for level in (1, 2, 3):
        (tmp_path / "output" / f"level_{level}" / "dist.svg").unlink()
    monkeypatch.setattr(config, "max_workers", 2)
    with cluster_levels((1, 2, 3)):
        v2.step_plot_torsion_distributions(str(folder / "embedding"), str(tmp_path / "output"), force=True)
    for level in (1, 2, 3):
        serial = comparable_svg((folder / "output" / f"level_{level}" / "dist.svg").read_bytes())
        parallel = comparable_svg((tmp_path / "output" / f"level_{level}" / "dist.svg").read_bytes())
        assert "<svg" in serial and serial == parallel, level
    assert comparable_svg((folder / "output/level_1/dist.svg").read_bytes()) != comparable_svg(
        (folder / "output/level_3/dist.svg").read_bytes())


# ---------------------------------------------------------------------------
# Static stage
# ---------------------------------------------------------------------------
def test_filter_info_levels():
    info = {"entropy_nats": 1.5, "levels": [{"level": 1, "n": 5}, {"level": 2, "n": 9}, {"level": 3, "n": 512}],
            "pca": {"main_clustering": {"H_nats": 1.5}, "coverage_clustering": {"selected_n_clusters": 9},
                    "level_3_clustering": {"selected_n_clusters": 512}}, "schema_version": 3}
    reduced, changed = static_ready.filter_info_levels(info, (1, 2))
    assert changed and [level["level"] for level in reduced["levels"]] == [1, 2]
    assert list(reduced["pca"]) == ["main_clustering", "coverage_clustering"]
    assert list(reduced) == list(info) and len(info["levels"]) == 3          # key order kept, input untouched
    assert static_ready.filter_info_levels(info, (1, 2, 3)) == (info, False)
    assert static_ready.filter_info_levels(reduced, (1, 2)) == (reduced, False)


def test_static_ready_from_a_folder_with_level_3_equals_the_one_without(runs, tmp_path):
    with cluster_levels((1, 2)):
        written = static_ready.write_static_ready(runs["with_3"], tmp_path / "from_old", GLYCAM_NAME)
        static_ready.write_static_ready(runs["default"], tmp_path / "from_new", GLYCAM_NAME)
    assert written == {"levels": ["level_1", "level_2"]}
    old, new = files_of(tmp_path / "from_old"), files_of(tmp_path / "from_new")
    assert sorted(old) == sorted(new) == sorted([
        "landscape/landscape_metadata.json", "landscape/pca_model.joblib", "landscape/preprocessing.npz",
        "output/info.json", "output/torparts.npz", "output/pca.csv", "output/torsion_all.csv",
        "output/torsion_glycosidic.csv", "output/torsion_level_1_reps.csv", "output/torsion_level_2_reps.csv",
        *[f"output/level_{level}/{name}" for level in (1, 2)
          for name in ("dist.svg", "GLYCAM/alpha.pdb", "GLYCAM/beta.pdb", "PDB/alpha.pdb", "PDB/beta.pdb")],
    ])
    for name in old:
        if not name.endswith("dist.svg"):
            assert old[name] == new[name], name
    header = old["output/torsion_all.csv"].split(b"\n", 1)[0].decode()
    assert "level_2_center" in header and "level_3" not in header
    assert [level["level"] for level in json.loads(old["output/info.json"])["levels"]] == [1, 2]
    assert b"level_3" not in old["output/info.json"]
    assert old["output/level_1/GLYCAM/alpha.pdb"].startswith(b"REMARK   GLYCAN MULTI-MODEL PDB  LEVEL LEVEL_1  ANOMER ALPHA\nMODEL")


def test_static_ready_keeps_level_3_when_configured(runs, tmp_path):
    with cluster_levels((1, 2, 3)):
        written = static_ready.write_static_ready(runs["with_3"], tmp_path / "static", GLYCAM_NAME)
    assert written == {"levels": ["level_1", "level_2", "level_3"]}
    names = set(files_of(tmp_path / "static"))
    assert {"output/level_3/PDB/beta.pdb", "output/level_3/dist.svg", "output/torsion_level_3_reps.csv"} <= names
    assert "level_3_cluster" in (tmp_path / "static/output/torsion_all.csv").read_text().split("\n", 1)[0]
    assert (tmp_path / "static/output/info.json").read_bytes() == (runs["with_3"] / "output/info.json").read_bytes()


def test_legacy_csv_table_is_copied_without_other_levels(tmp_path):
    from lib.storage import get_storage_manager

    folder = tmp_path / "process" / "glycan"
    (folder / "output").mkdir(parents=True)
    (folder / "output" / "torsion_all.csv").write_text(
        'frame,phi,level_1_cluster,level_1_center,level_3_cluster,level_3_center,"odd,name"\n'
        "0,-65.31234,2,,417,,x\n1,1e-05,0,0,3,3,\"a,b\"\n", encoding="utf-8")
    (folder / "output" / "pca.csv").write_text("frame,PC1\n0,0.1\n", encoding="utf-8")
    (tmp_path / "static").mkdir()
    storage = get_storage_manager()
    with cluster_levels((1, 2)):
        for name in ("torsion_all.csv", "pca.csv"):
            assert static_ready.copy_or_convert_analysis_table(storage, str(folder), str(tmp_path / "static"), name, "none.parquet")
    assert (tmp_path / "static" / "torsion_all.csv").read_text() == (
        'frame,phi,level_1_cluster,level_1_center,"odd,name"\n0,-65.31234,2,,x\n1,1e-05,0,0,"a,b"\n')
    assert (tmp_path / "static" / "pca.csv").read_bytes() == (folder / "output" / "pca.csv").read_bytes()
    with cluster_levels((1, 2, 3)):
        static_ready.copy_or_convert_analysis_table(storage, str(folder), str(tmp_path / "static"), "torsion_all.csv", "none.parquet")
    assert (tmp_path / "static" / "torsion_all.csv").read_bytes() == (folder / "output" / "torsion_all.csv").read_bytes()


class OfflineNames:
    """lib.name_utils without its web services: what the static stage gets when nothing resolves."""

    def __init__(self):
        from lib import name_utils

        self.glycam_info = name_utils.glycam_info

    def crabwurcs_convert(self, value, source, target):
        return f"WURCS={value}" if (source, target) == ("glycam", "wurcs") else None

    def iupac2wurcs_glytoucan(self, iupac):
        return "G00000XX", "WURCS=archetype"

    def wurcs2glytoucan(self, wurcs):            # the alpha and the beta anomer have their own accession
        return {"a1-OH": "G11111AA", "b1-OH": "G22222BB"}.get(str(wurcs)[-5:])

    def __getattr__(self, name):                 # every other lookup: nothing found
        return lambda *args, **kwargs: None


def static_stage(process_folder, static_root):
    """lib.glystatic.process_glycan with the submission metadata and all services replaced."""
    from lib import glystatic

    metadata = {"ID": "GS00042", "length": 2.0, "package": "AMBER", "forcefield": "GLYCAM06j", "temperature": 300,
                "pressure": 1, "salt": 0.2, "comments": "", "glytoucan_id": "G00000XX"}

    def fake_snfg(glycam_name, iupac, destination):
        Path(destination).write_text("<svg/>", encoding="utf-8")
        return "test"

    with patch.object(glystatic, "get_glycan_metadata", return_value=metadata), \
            patch.object(glystatic, "name_utils", OfflineNames()), \
            patch.object(glystatic, "_render_snfg_svg", side_effect=fake_snfg), \
            patch.object(glystatic, "_source_simulation_length_ns", return_value=None):
        result = glystatic.process_glycan(str(process_folder), GLYCAM_NAME, str(static_root))
    return Path(result["final_output_dir"])


def test_static_stage_rebuilds_without_level_3_from_a_folder_that_has_it(runs, tmp_path):
    with cluster_levels((1, 2)):
        static = static_stage(runs["with_3"], tmp_path / "static")
        static_ready.write_static_ready(runs["with_3"], tmp_path / "ready", GLYCAM_NAME)
    assert static.name == "GS00042"
    names = set(files_of(static))
    assert not [name for name in names if "level_3" in name]
    data = json.loads((static / "data.json").read_text())
    assert sorted(data["archetype"]["cluster_levels"]) == ["level_1", "level_2"]
    assert data["archetype"]["clusters"] and data["archetype"]["coverage_clusters"]
    assert b"level_3" not in (static / "output" / "info.json").read_bytes()
    assert "level_3" not in (static / "output" / "torsion_all.csv").read_text().split("\n", 1)[0]

    # The static stage writes the static-ready files plus the identifier files: the shared files are
    # equal, and a PDB file is the static-ready one under the GlycoShape REMARK block.
    ready = files_of(tmp_path / "ready")
    stage = files_of(static)
    assert set(stage) - set(ready) == {"data.json", "glygen.json", "glycosmos.json", "snfg.svg"}
    for name, content in ready.items():
        if name.endswith(".pdb"):
            assert stage[name].endswith(content), name
            banner = stage[name][: len(stage[name]) - len(content)].decode()
            assert banner.startswith("REMARK     GENERATED BY GlycanAnalysisPipeline")
            assert all(line.startswith("REMARK") for line in banner.splitlines())
            assert "GlyTouCan ID: " + ("G11111AA" if "alpha" in name else "G22222BB") in banner
        else:
            assert stage[name] == content, name


def test_static_stage_keeps_level_3_when_configured(runs, tmp_path):
    with cluster_levels((1, 2, 3)):
        static = static_stage(runs["with_3"], tmp_path / "static")
    names = set(files_of(static))
    assert {"output/level_3/GLYCAM/alpha.pdb", "output/level_3/PDB/beta.pdb", "output/torsion_level_3_reps.csv"} <= names
    assert sorted(json.loads((static / "data.json").read_text())["archetype"]["cluster_levels"]) == [
        "level_1", "level_2", "level_3"]


def test_completeness_checks_of_the_runner_follow_the_setting(runs, tmp_path):
    from lib.storage import get_storage_manager
    from main import GlycanPipelineRunner

    with cluster_levels((1, 2)):
        old_static = static_stage(runs["with_3"], tmp_path / "static_12")
    with cluster_levels((1, 2, 3)):
        full_static = static_stage(runs["with_3"], tmp_path / "static_123")

    def runner(process_dir, output_dir):
        instance = GlycanPipelineRunner.__new__(GlycanPipelineRunner)
        instance.storage = get_storage_manager()
        instance.process_dir = process_dir
        instance.output_dir = output_dir
        instance.data_dir = runs["data"]
        instance._wurcs_glytoucan_cache = {"WURCS=archetype": None}
        instance._iupac_wurcs_cache = {}
        return instance

    with_3 = runner(runs["with_3"].parent, tmp_path / "static_123")
    default = runner(runs["default"].parent, tmp_path / "static_12")
    with cluster_levels((1, 2)):
        assert with_3.REQUIRED_GLYSTATIC_LEVELS == ("level_1", "level_2")
        required = with_3._get_static_only_required_files()
        assert "output/torsion_level_2_reps.csv" in required and not [f for f in required if "level_3" in f]
        assert not [f for f in with_3._get_log_tail_required_files() if "level_3" in f]
        # process and static folders that still contain Level 3 are complete, and so are the new ones
        assert with_3._static_only_missing_requirements(GLYCAM_NAME) == []
        assert default._static_only_missing_requirements(GLYCAM_NAME) == []
        assert with_3.check_v2_completion(GLYCAM_NAME) and default.check_v2_completion(GLYCAM_NAME)
        for static in (full_static, old_static):
            assert with_3._validate_glystatic_output_dir(str(static), glycan_name=GLYCAM_NAME) == []
            assert with_3._static_only_output_missing_requirements(static, GLYCAM_NAME) == []
    with cluster_levels((1, 2, 3)):
        assert with_3.REQUIRED_GLYSTATIC_LEVELS == ("level_1", "level_2", "level_3")
        assert "output/torsion_level_3_reps.csv" in with_3._get_static_only_required_files()
        assert "output/level_3/PDB/alpha.pdb" in with_3._get_log_tail_required_files()
        assert with_3._static_only_missing_requirements(GLYCAM_NAME) == []
        assert with_3._validate_glystatic_output_dir(str(full_static), glycan_name=GLYCAM_NAME) == []
        missing = default._static_only_missing_requirements(GLYCAM_NAME)
        assert "output/level_3/alpha.pdb" in missing and "embedding/exports/torsion_level_3_reps.parquet" in missing
        assert not default.check_v2_completion(GLYCAM_NAME)
        errors = with_3._validate_glystatic_output_dir(str(old_static), glycan_name=GLYCAM_NAME)
        assert errors == ["missing level_3 structures"]
        assert "output/torsion_level_3_reps.csv" in with_3._static_only_output_missing_requirements(old_static, GLYCAM_NAME)
