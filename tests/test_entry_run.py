"""The one-entry mode (lib/entry_run.py, scripts/run_entry.py): end to end on a synthetic trajectory.

No network: every test in this file runs with outgoing connections disabled.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest

import lib.config as config
from lib import archive, entry_run

from synthetic_glycan import GLYCAM_NAME, write_source

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_entry.py"
STATIC_FILES = sorted([
    "landscape/landscape_metadata.json", "landscape/pca_model.joblib", "landscape/preprocessing.npz",
    "output/info.json", "output/torparts.npz", "output/pca.csv", "output/torsion_all.csv",
    "output/torsion_glycosidic.csv", "output/torsion_level_1_reps.csv", "output/torsion_level_2_reps.csv",
    *[f"output/level_{level}/{name}" for level in (1, 2)
      for name in ("dist.svg", "GLYCAM/alpha.pdb", "GLYCAM/beta.pdb", "PDB/alpha.pdb", "PDB/beta.pdb")],
])


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any attempt to open an internet connection fails the test."""
    original = socket.socket.connect

    def connect(self, address, *args, **kwargs):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise AssertionError(f"network access attempted: {address}")
        return original(self, address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: (_ for _ in ()).throw(AssertionError("name lookup attempted")))


@pytest.fixture(autouse=True)
def restore_settings(monkeypatch):
    """run_entry sets the worker count, the levels and local storage in lib.config; undo it per test."""
    for name in ("max_workers", "cluster_levels", "use_oracle_storage", "oracle_par_url", "use_pocketbase"):
        monkeypatch.setattr(config, name, getattr(config, name))
    yield
    from lib.storage import reset_storage_manager

    reset_storage_manager()


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("entry_inputs")
    source = write_source(root / "data", broken=(11, 200))
    (source / "info.json").write_text(json.dumps({"simulationLength": 480.0}), encoding="utf-8")
    entry = Path(archive.convert_entry(source, root / "archive")["entry"])
    return {"root": root, "pdb": source, "archive": entry}


@pytest.fixture(scope="module")
def archive_result(inputs, tmp_path_factory):
    """One run from the archive entry, shared by the tests that only read it."""
    output = tmp_path_factory.mktemp("entry_result") / "result"
    settings = {name: getattr(config, name) for name in
                ("max_workers", "cluster_levels", "use_oracle_storage", "oracle_par_url", "use_pocketbase")}
    try:
        result = entry_run.run_entry(str(inputs["archive"]), str(output), workers=1, workers_from="test")
    finally:
        for name, value in settings.items():
            setattr(config, name, value)
    return output, result


def files_of(root):
    root = Path(root)
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------
def test_workers_come_from_the_option_then_slurm_then_the_setting():
    environ = {"SLURM_CPUS_PER_TASK": "64", "GLYCOSHAPE_MAX_WORKERS": "4"}
    assert entry_run.resolve_workers(12, environ) == (12, "--workers")
    assert entry_run.resolve_workers(None, environ) == (64, "SLURM_CPUS_PER_TASK")
    assert entry_run.resolve_workers(None, {"GLYCOSHAPE_MAX_WORKERS": "4"}) == (4, "GLYCOSHAPE_MAX_WORKERS")
    count, origin = entry_run.resolve_workers(None, {})
    assert count >= 1 and origin.startswith("cores")
    for bad in ({"SLURM_CPUS_PER_TASK": "many"}, {"SLURM_CPUS_PER_TASK": "0"}):
        with pytest.raises(entry_run.EntryError):
            entry_run.resolve_workers(None, bad)
    with pytest.raises(entry_run.EntryError):
        entry_run.resolve_workers(0, {})


def test_environment_is_local_only_and_sized_to_the_workers(monkeypatch):
    for name in ("GLYCOSHAPE_MAX_WORKERS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "GLYCOSHAPE_ORACLE_PAR_URL",
                 "POCKETBASE_URL"):
        monkeypatch.setenv(name, "https://example.invalid/x" if "URL" in name else "1")
    entry_run.prepare_environment(6)
    assert os.environ["GLYCOSHAPE_MAX_WORKERS"] == os.environ["OMP_NUM_THREADS"] == os.environ["OPENBLAS_NUM_THREADS"] == "6"
    assert os.environ["GLYCOSHAPE_ORACLE_PAR_URL"] == "" and os.environ["POCKETBASE_URL"] == ""


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------
def test_result_folder_of_an_archive_entry(inputs, archive_result):
    output, result = archive_result
    assert result["status"] == "ok" and result["error"] is None and result["warnings"] == []
    assert result == json.loads((output / "result.json").read_text())
    assert sorted(p.name for p in output.iterdir()) == ["input", "process", "result.json", "run.log", "static"]
    assert sorted(p.name for p in (output / "input").iterdir()) == ["info.json", "manifest.json"]
    assert (output / "run.log").stat().st_size > 0

    # what it was computed from: the hashes of the archive manifest
    manifest = archive.load_manifest(inputs["archive"])
    described = result["input"]
    assert described["kind"] == "archive" and described["files_verified_against_manifest"] is True
    assert described["files"] == manifest["files"]
    assert described["coordinates_sha256"] == manifest["trajectory"]["coordinates_sha256"]
    assert described["source"]["sha256"] == manifest["source"]["sha256"]
    assert (described["atoms"], described["frames"]) == (45, 480)
    assert result["glycam_name"] == GLYCAM_NAME and result["folder"] == inputs["archive"].name

    # with which code and settings
    assert len(result["pipeline"]["code_sha256"]) == 64 and result["pipeline"]["packages"]["numpy"]
    assert result["settings"]["cluster_levels"] == [1, 2] and result["settings"]["workers"] == 1
    assert result["settings"]["max_covalent_bond_angstrom"] == 3.0

    # timings per step and the frame-quality summary
    steps = ("input", "store", "pca", "torsions", "clustering", "torsion_stats", "create_info", "structures",
             "plots", "export", "static_ready", "result", "total")
    assert list(result["timings_sec"]) == list(steps)
    assert all(isinstance(result["timings_sec"][step], float) for step in steps)
    assert result["timings_sec"]["total"] >= sum(result["timings_sec"][step] for step in steps[:-1]) * 0.99
    quality = result["frame_quality"]
    assert (quality["source_frames"], quality["retained_frames"], quality["rejected_frames"]) == (480, 478, 2)
    assert quality == json.loads((output / "process" / GLYCAM_NAME / "data" / "frame_filter.json").read_text())

    # what was produced
    analysis = result["analysis"]
    assert sorted(analysis["levels"]) == ["1", "2"] and analysis["levels"]["1"]["n_clusters"] == 5
    frames = analysis["levels"]["1"]["representative_frames"]
    sources = analysis["levels"]["1"]["representative_source_frames"]
    assert len(frames) == 5 and all(s >= f for f, s in zip(frames, sources)) and not {11, 200} & set(sources)
    assert analysis["torsions"]["glycosidic"] == 2 and len(analysis["pca_explained_variance_ratio"]) == 10
    produced = result["produced"]
    assert produced["levels"] == ["level_1", "level_2"]
    static = files_of(output / "static")
    process = files_of(output / "process" / GLYCAM_NAME)
    assert sorted(static) == STATIC_FILES
    assert [record["path"] for record in produced["static"]["files"]] == sorted(static)
    assert [record["path"] for record in produced["process"]["files"]] == sorted(process)
    import hashlib

    for record in produced["static"]["files"]:
        content = static[record["path"]]
        assert (record["size"], record["sha256"]) == (len(content), hashlib.sha256(content).hexdigest())
    assert not [name for name in list(static) + list(process) if "level_3" in name]
    assert "identifier" not in json.dumps(result["analysis"]) and result["left_to_publishing"]


def test_static_folder_is_what_the_static_stage_makes_of_the_process_folder(archive_result, tmp_path):
    from lib import static_ready

    output, _ = archive_result
    static_ready.write_static_ready(output / "process" / GLYCAM_NAME, tmp_path / "again", GLYCAM_NAME)
    first, again = files_of(output / "static"), files_of(tmp_path / "again")
    assert first == again
    # structures of the reducing-end anomers in both namings, without identifier remarks
    alpha = first["output/level_1/GLYCAM/alpha.pdb"].decode()
    assert alpha.startswith("REMARK   GLYCAN MULTI-MODEL PDB  LEVEL LEVEL_1  ANOMER ALPHA\nMODEL        1\n")
    assert "GlyTouCan" not in alpha and alpha.count("ENDMDL") == 5 and " 2MA " in alpha
    assert " MAN " in first["output/level_1/PDB/alpha.pdb"].decode()
    assert first["output/torsion_all.csv"].split(b"\n", 1)[0].endswith(b"level_2_cluster,level_2_center")


def test_pdb_folder_gives_the_same_analysis_as_its_archive_entry(inputs, archive_result, tmp_path):
    output, from_archive = archive_result
    from_pdb = entry_run.run_entry(str(inputs["pdb"]), str(tmp_path / "result"), workers=1)
    assert from_pdb["status"] == "ok"
    described = from_pdb["input"]
    assert described["kind"] == "pdb" and (described["atoms"], described["frames"]) == (45, 480)
    assert sorted(described["files"]) == ["info.json", "simulation.pdb", "structure.mol2"]
    assert described["files"]["simulation.pdb"]["sha256"] == from_archive["input"]["source"]["sha256"]
    assert sorted(p.name for p in (tmp_path / "result" / "input").iterdir()) == ["info.json"]
    assert from_pdb["frame_quality"] == from_archive["frame_quality"]
    assert from_pdb["analysis"] == from_archive["analysis"]
    a, b = files_of(output / "static"), files_of(tmp_path / "result" / "static")
    assert sorted(a) == sorted(b)
    for name in a:
        if not name.endswith("dist.svg"):
            assert a[name] == b[name], name


def test_current_result_is_reused_and_force_recomputes(inputs, archive_result, tmp_path):
    output, first = archive_result
    copy = tmp_path / "result"
    shutil.copytree(output, copy)
    log_before = (copy / "run.log").read_bytes()
    reused = entry_run.run_entry(str(inputs["archive"]), str(copy), workers=1)
    assert reused["reused"] is True and reused["finished_at"] == first["finished_at"]
    assert (copy / "run.log").read_bytes() == log_before
    assert "reused" not in json.loads((copy / "result.json").read_text())

    # other settings: the result is not current any more
    (copy / "static" / "stale.txt").write_text("from an earlier run")
    again = entry_run.run_entry(str(inputs["archive"]), str(copy), workers=1, levels="1,2,3")
    assert again["status"] == "ok" and "reused" not in again and again["settings"]["cluster_levels"] == [1, 2, 3]
    assert again["produced"]["levels"] == ["level_1", "level_2", "level_3"]
    assert not (copy / "static" / "stale.txt").exists()
    assert (copy / "static" / "output" / "level_3" / "PDB" / "alpha.pdb").is_file()

    forced = entry_run.run_entry(str(inputs["archive"]), str(copy), workers=1, levels="1,2,3", force=True)
    assert forced["status"] == "ok" and forced["started_at"] >= again["finished_at"]
    assert forced["analysis"] == again["analysis"]


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------
def test_output_folder_must_be_new_empty_or_an_earlier_result(inputs, tmp_path, capsys):
    occupied = tmp_path / "occupied"
    (occupied / "static").mkdir(parents=True)
    (occupied / "static" / "index.html").write_text("not ours")
    assert entry_run.main([str(inputs["archive"]), str(occupied), "--workers", "1"]) == 2
    assert "is not empty and holds no result.json" in capsys.readouterr().err
    assert (occupied / "static" / "index.html").read_text() == "not ours"

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "result.json").write_text('{"schema": "something else"}')
    assert entry_run.main([str(inputs["archive"]), str(foreign), "--workers", "1"]) == 2
    assert "is not a result of this tool" in capsys.readouterr().err

    assert entry_run.main([str(tmp_path / "missing"), str(tmp_path / "out"), "--workers", "1"]) == 2
    assert "input folder does not exist" in capsys.readouterr().err
    empty = tmp_path / "empty_input"
    empty.mkdir()
    assert entry_run.main([str(empty), str(tmp_path / "out"), "--workers", "1"]) == 2
    assert "neither an archive entry" in capsys.readouterr().err
    assert entry_run.main([str(inputs["archive"]), str(tmp_path / "out"), "--workers", "1", "--levels", "1,3"]) == 2
    assert "cluster levels must be 1,2 or 1,2,3" in capsys.readouterr().err


def test_archive_entry_that_does_not_match_its_manifest_is_refused(inputs, tmp_path, capsys):
    damaged = tmp_path / inputs["archive"].name
    shutil.copytree(inputs["archive"], damaged)
    content = bytearray((damaged / "trajectory.xtc").read_bytes())
    content[len(content) // 2] ^= 0x01
    (damaged / "trajectory.xtc").write_bytes(bytes(content))
    assert entry_run.main([str(damaged), str(tmp_path / "out"), "--workers", "1"]) == 2
    assert "trajectory.xtc does not have the SHA-256 of the manifest" in capsys.readouterr().err
    assert not (tmp_path / "out" / "process").exists()

    (damaged / "trajectory.xtc").write_bytes(bytes(content[:-100]))
    assert entry_run.main([str(damaged), str(tmp_path / "out"), "--workers", "1", "--no-verify-input"]) == 2
    assert "the manifest says" in capsys.readouterr().err


def test_failed_analysis_leaves_a_failed_result_and_exit_status_1(inputs, tmp_path, capsys):
    broken = tmp_path / "data" / GLYCAM_NAME
    shutil.copytree(inputs["pdb"], broken)
    mol2 = (broken / "structure.mol2").read_text()
    head, bonds = mol2.split("@<TRIPOS>BOND\n")
    bond_lines = bonds.splitlines()
    fields = bond_lines[0].split()
    fields[2] = "999"                                  # a bond to an atom the trajectory does not have
    (broken / "structure.mol2").write_text(head + "@<TRIPOS>BOND\n" + "\n".join([" ".join(fields)] + bond_lines[1:]) + "\n")

    output = tmp_path / "out"
    assert entry_run.main([str(broken), str(output), "--workers", "1"]) == 1
    message = capsys.readouterr().err
    assert "FAILED in step store" in message and "999" in message and "run.log" in message
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "failed" and result["error"]["step"] == "store"
    assert "Connectivity provenance failure" in result["error"]["message"]
    assert result["finished_at"] and result["produced"] is None and "store" not in result["timings_sec"]
    assert "Connectivity provenance failure" in (output / "run.log").read_text()

    # a failed (or killed: status "running") result is replaced by the next run
    shutil.copy2(inputs["pdb"] / "structure.mol2", broken / "structure.mol2")
    assert entry_run.main([str(broken), str(output), "--workers", "1"]) == 0
    assert json.loads((output / "result.json").read_text())["status"] == "ok"
    assert "run_entry: ok" in capsys.readouterr().out


def test_command_line_takes_the_workers_from_slurm(inputs, tmp_path):
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("GLYCOSHAPE_", "POCKETBASE", "SLURM_")) and not key.endswith("_NUM_THREADS")}
    environment.update({
        "SLURM_CPUS_PER_TASK": "2", "SLURM_JOB_ID": "4242",
        # a job environment that points at services and pins BLAS to one thread must not matter
        "GLYCOSHAPE_ORACLE_PAR_URL": "https://objectstorage.invalid/p/x/", "POCKETBASE_URL": "http://pocketbase.invalid",
        "POCKETBASE_TOKEN": "token", "OMP_NUM_THREADS": "1", "http_proxy": "http://127.0.0.1:9", "https_proxy": "http://127.0.0.1:9",
    })
    output = tmp_path / "out"
    completed = subprocess.run([sys.executable, str(SCRIPT), str(inputs["archive"]), str(output)],
                               env=environment, cwd=str(tmp_path), capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert f"run_entry: ok: {GLYCAM_NAME}" in completed.stdout and "frames kept 478/480" in completed.stdout
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "ok" and result["host"]["slurm_job_id"] == "4242"
    assert (result["settings"]["workers"], result["settings"]["workers_from"]) == (2, "SLURM_CPUS_PER_TASK")
    assert set(result["settings"]["threads"].values()) == {"2"}
    assert sorted(files_of(output / "static")) == STATIC_FILES
    assert not (tmp_path / "last_run.log").exists()            # nothing is written next to the caller

    missing = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path / "nothing"), str(tmp_path / "out2")],
                             env=environment, cwd=str(tmp_path), capture_output=True, text=True, timeout=600)
    assert missing.returncode == 2 and "input folder does not exist" in missing.stderr
