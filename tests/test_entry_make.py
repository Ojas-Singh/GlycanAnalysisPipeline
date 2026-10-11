"""The node command (lib/entry_make.py, scripts/make_entry.py): replicas of a run -> one bundle, end to end.

A synthetic trajectory, no network: every test in this file runs with outgoing connections disabled.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest

import lib.config as config
from lib import archive, archive_segments, entry_bundle, entry_make, entry_run

import synthetic_run
from synthetic_glycan import GLYCAM_NAME

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "make_entry.py"
REPLICA_NAMES = [f"{GLYCAM_NAME}_replica_{index}" for index in (1, 2, 3)]


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
    """The analysis sets the worker count, the levels and local storage in lib.config; undo it per test."""
    for name in synthetic_run.SETTINGS:
        monkeypatch.setattr(config, name, getattr(config, name))
    yield
    from lib.storage import reset_storage_manager

    reset_storage_manager()


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return synthetic_run.get(tmp_path_factory)


def members_of(tar_path):
    with tarfile.open(tar_path) as tar:
        return {member.name: tar.extractfile(member).read() for member in tar}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# What a run leaves on disk
# ---------------------------------------------------------------------------
def test_replicas_are_found_as_folders_files_and_lines_of_a_replica_map(tmp_path):
    def folder(name, *files):
        (tmp_path / name).mkdir()
        for file in files:
            (tmp_path / name / file).write_text("x")
        return tmp_path / name

    run_folder = folder("x_replica_1", "glycan.dry.pdb", "glycan.dry.mol2", "glycan.dry.prm7", "production.mdp", "info.json")
    found = entry_make.resolve_replica(str(run_folder))
    assert found == {"name": "x_replica_1", "pdb": run_folder / "glycan.dry.pdb", "mol2": run_folder / "glycan.dry.mol2",
                     "folder": run_folder, "mdp": run_folder / "production.mdp", "info": run_folder / "info.json"}
    upload = folder("upload_package", "simulation.pdb", "structure.mol2")
    assert entry_make.resolve_replica(str(upload))["pdb"] == upload / "simulation.pdb"
    assert entry_make.resolve_replica(str(upload))["mdp"] is None
    named = folder("DGlcpa1-OH", "DGlcpa1-OH.pdb", "DGlcpa1-OH.mol2")
    assert entry_make.resolve_replica(str(named))["mol2"] == named / "DGlcpa1-OH.mol2"
    # a file: its MOL2 next to it; the folder names it when the file has the name every replica has
    assert entry_make.resolve_replica(str(run_folder / "glycan.dry.pdb"))["name"] == "x_replica_1"
    assert entry_make.resolve_replica(str(run_folder / "glycan.dry.pdb"))["mol2"] == run_folder / "glycan.dry.mol2"
    assert entry_make.resolve_replica(str(named / "DGlcpa1-OH.pdb"))["name"] == "DGlcpa1-OH"
    lonely = folder("lonely", "traj.pdb")
    assert entry_make.resolve_replica(str(lonely / "traj.pdb"))["mol2"] is None
    assert entry_make.resolve_replica(f"first={run_folder}")["name"] == "first"
    equals = folder("a=b", "glycan.dry.pdb")
    assert entry_make.resolve_replica(str(equals))["name"] == "a=b"            # an existing path is a path

    (tmp_path / "replica_map.txt").write_text(f"{run_folder}|run_1\n\nupload_package|\n# comment\nDGlcpa1-OH|third\n")
    replicas = entry_make.resolve_replicas([f"fourth={lonely / 'traj.pdb'}"], tmp_path / "replica_map.txt")
    assert [replica["name"] for replica in replicas] == ["run_1", "upload_package", "third", "fourth"]
    assert replicas[1]["folder"] == upload
    prefixed = entry_make.resolve_replicas([str(run_folder)], tmp_path / "replica_map.txt", prefix="run7-")
    assert [replica["name"] for replica in prefixed] == ["run7-run_1", "run7-upload_package", "run7-third", "run7-x_replica_1"]

    for arguments, message in (
        (([], None), "no replicas were given"),
        (([str(run_folder), str(run_folder / "glycan.dry.pdb")], None), "x_replica_1 occurs more than once"),
        (([str(tmp_path / "nothing")], None), "no such replica folder or file"),
        (([str(folder("empty"))], None), "has none of glycan.dry.pdb, simulation.pdb, empty.pdb"),
        (([], tmp_path / "no_map.txt"), "cannot read the replica map"),
    ):
        with pytest.raises(entry_make.MakeEntryError, match=message):
            entry_make.resolve_replicas(*arguments)


def test_settings_of_a_production_mdp_are_read_with_their_units(tmp_path):
    mdp = tmp_path / "production.mdp"
    mdp.write_text(
        "; Run parameters\n"
        "integrator              = md        ; leap-frog integrator\n"
        "nsteps                  = 250000000 ; 2 * 250000000 = 500000 ps (500 ns)\n"
        "dt                      = 0.002     ; 2 fs\n"
        "nstxout                 = 0         ; suppress bulky .trr file by specifying \n"
        "nstxout-compressed      = 5000      ; save compressed coordinates every 10.0 ps\n"
        "ref_t                   = 300     300           ; reference temperature\n"
    )
    assert entry_make.read_mdp(mdp) == {
        "file": "production.mdp", "sha256": sha256(mdp.read_bytes()), "dt_ps": 0.002, "nsteps": 250000000,
        "length_ns": 500.0, "nstxout_compressed": 5000, "saved_every_ps": 10.0}
    mdp.write_text("dt=0.004\nnsteps = -1\nnstxtcout = 2500\n")
    values = entry_make.read_mdp(mdp)
    assert (values["dt_ps"], values["nsteps"], values["length_ns"], values["saved_every_ps"]) == (0.004, None, None, 10.0)
    mdp.write_text("; nothing\n")
    assert entry_make.read_mdp(mdp)["dt_ps"] is None
    simulation = {"saved_every_ps": 10.0}
    assert entry_make._time_per_frame(simulation, None, 5) == 50.0
    assert entry_make._time_per_frame(simulation, 20.0, None) == 20.0
    assert entry_make._time_per_frame(None, None, 5) is None and entry_make._time_per_frame(simulation, None, None) is None


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------
def test_bundle_of_the_replicas_is_the_entry_and_the_analysis_of_the_whole_trajectory(run, tmp_path):
    """Three replicas give what the converter and the one-entry mode give for the file they would be joined to."""
    reference = Path(archive.convert_entry(run["whole"], tmp_path / "archive")["entry"])
    result = entry_run.run_entry(str(reference), str(tmp_path / "result"), workers=1)
    assert result["status"] == "ok"
    members = members_of(run["bundle"])

    assert members["archive/trajectory.xtc"] == (reference / "trajectory.xtc").read_bytes()
    assert members["archive/topology.pdb"] == (reference / "topology.pdb").read_bytes()
    assert members["archive/structure.mol2"] == (reference / "structure.mol2").read_bytes()
    manifest = json.loads(members["archive/manifest.json"])
    whole = archive.load_manifest(reference)
    assert manifest["trajectory"] == whole["trajectory"] and (manifest["atoms"], manifest["frames"]) == (45, 480)
    assert [(s["name"], s["first_frame"], s["frames"]) for s in manifest["segments"]] == [
        (REPLICA_NAMES[0], 0, 200), (REPLICA_NAMES[1], 200, 130), (REPLICA_NAMES[2], 330, 150)]
    assert json.loads(members["archive/name.json"]) == {"indexOrderedSequence": GLYCAM_NAME}
    assert members["archive/info.json"] == (run["replicas"][0] / "info.json").read_bytes()

    static = {name[len("static/"):]: data for name, data in members.items() if name.startswith("static/")}
    expected = {path.relative_to(tmp_path / "result" / "static").as_posix(): path.read_bytes()
                for path in (tmp_path / "result" / "static").rglob("*") if path.is_file()}
    assert sorted(static) == sorted(expected)
    different = [name for name in static if static[name] != expected[name]]
    assert all(name.endswith("dist.svg") for name in different), different     # the figures carry ids of their run
    bundled = json.loads(members["result.json"])
    assert bundled["analysis"] == result["analysis"] and bundled["frame_quality"] == result["frame_quality"]
    assert bundled["frame_quality"]["retained_frames"] == 478 and bundled["input"]["kind"] == "archive"
    assert bundled["input"]["manifest_sha256"] == sha256(members["archive/manifest.json"])
    assert (bundled["settings"]["workers"], bundled["settings"]["workers_from"]) == (1, "--workers")

    made = run["made"]
    assert (made["status"], made["frames"], made["frames_kept"], made["replicas"]) == ("ok", 480, 478, 3)
    assert made["clusters"]["1"] == 5 and made["extends"] is None and made["levels"] == [1, 2]
    assert made["entry"]["status"] == "converted" and made["result"]["reused"] is False
    assert list(made["timings_sec"]) == ["convert", "analysis", "bundle", "total"]
    line = entry_make.summary_line(made)
    assert "\n" not in line and line.startswith(
        f"make_entry: ok: {GLYCAM_NAME}: 480 frames of 45 atoms (3 replica(s)), 478 kept; clusters level 1: 5, level 2: ")
    assert f"bundle {run['bundle']} " in line and f"sha256 {made['bundle']['sha256']}; " in line


def test_source_of_the_bundle_says_what_it_was_made_from_with_units(run):
    source = json.loads(members_of(run["bundle"])["bundle.json"])["source"]
    assert (source["kind"], source["name"]) == ("simulation-run", GLYCAM_NAME)
    assert source["trajectory"] == {"atoms": 45, "frames": 480, "frames_from_this_run": 480,
                                    "frames_from_extended_entry": 0, "segments": 3}
    replicas = source["replicas"]
    assert [(r["name"], r["first_frame"], r["frames"], r["time_per_frame_ps"], r["sampled_ns"]) for r in replicas] == [
        (REPLICA_NAMES[0], 0, 200, 50.0, 9.95), (REPLICA_NAMES[1], 200, 130, 50.0, 6.45), (REPLICA_NAMES[2], 330, 150, 50.0, 7.45)]
    for replica, folder in zip(replicas, run["replicas"]):
        pdb = (folder / "glycan.dry.pdb").read_bytes()
        assert replica["pdb"] == {"file": "glycan.dry.pdb", "size": len(pdb), "sha256": sha256(pdb)}
        assert replica["mol2"] == {"file": "glycan.dry.mol2", "sha256": sha256((folder / "glycan.dry.mol2").read_bytes())}
        assert replica["simulation"]["dt_ps"] == 0.002 and replica["simulation"]["saved_every_ps"] == 10.0
        assert replica["simulation"]["length_ns"] == replica["sampled_ns"]
    # nanoseconds, said in the name of the key (the upload form of today sends microseconds as simulationLength)
    assert source["simulation_length_ns"] == 23.85
    assert source["simulation_length_from"] == "the frames of every replica: (frames - 1) x time_per_frame_ps"
    assert (source["time_per_frame_ps"], source["frame_stride"]) == (50.0, 5)
    assert {key: source[key] for key in synthetic_run.METADATA if key != "simulation_length_ns"} == {
        "md_package": "GROMACS", "force_field": "GLYCAM_06j", "temperature_k": 300.0, "pressure_bar": 1.0,
        "salt_concentration_mm": 200.0, "comments": "Auto upload from GOTW", "glytoucan_id": None, "run_id": "run-0001"}
    assert sorted(source["units"]) == ["pressure_bar", "salt_concentration_mm", "simulation_length_ns", "temperature_k",
                                       "time_per_frame_ps"]
    assert source["info_json"] == {"file": "info.json", "sha256": sha256((run["replicas"][0] / "info.json").read_bytes())}
    assert source["host"]["workers"] == 1 and source["host"]["cores_available"] >= 1 and source["warnings"] == []

    # the manifest of the entry outlives the bundle: every segment says which run it is from
    manifest = json.loads(members_of(run["bundle"])["archive/manifest.json"])
    for segment, replica in zip(manifest["segments"], replicas):
        assert segment["run"] == {
            "run_id": "run-0001", "md_package": "GROMACS", "force_field": "GLYCAM_06j", "temperature_k": 300.0,
            "pressure_bar": 1.0, "salt_concentration_mm": 200.0, "production_mdp": replica["simulation"]}
        assert segment["time_per_frame_ps"] == 50.0 and segment["source"]["sha256"] == replica["pdb"]["sha256"]


def test_length_of_the_run_from_the_frames_the_mdp_or_the_option(run):
    manifest = archive.load_manifest(run["work"] / "archive" / GLYCAM_NAME)
    replicas = entry_make.resolve_replicas([], run["root"] / "run" / "replica_map.txt")
    simulations = [entry_make.read_mdp(replica["mdp"]) for replica in replicas]

    def describe(manifest=manifest, simulations=simulations, **metadata):
        return entry_make.describe_source(replicas, manifest, metadata, simulations, None, 1, None)

    assert describe(simulation_length_ns=30.0)["simulation_length_ns"] == 30.0
    assert describe(simulation_length_ns=30.0)["simulation_length_from"] == "--simulation-length-ns"
    assert describe(simulations=[None] * 3)["simulation_length_ns"] == 23.85              # from the frames
    # without a time per frame: what production.mdp was set to; without that: unknown, never a guess
    untimed = json.loads(json.dumps(manifest))
    for segment in untimed["segments"]:
        segment["time_per_frame_ps"] = None
    from_mdp = describe(manifest=untimed)
    assert from_mdp["simulation_length_ns"] == 23.85 and from_mdp["time_per_frame_ps"] is None
    assert from_mdp["simulation_length_from"] == "production.mdp of every replica: dt x nsteps"
    unknown = describe(manifest=untimed, simulations=[simulations[0], None, simulations[2]])
    assert unknown["simulation_length_ns"] is None and unknown["simulation_length_from"] is None
    # a replica that was cut short is said
    longer = [dict(simulation, length_ns=500.0) for simulation in simulations]
    warned = describe(simulations=longer)
    assert warned["simulation_length_ns"] == 23.85 and len(warned["warnings"]) == 3
    assert warned["warnings"][0] == (
        f"{REPLICA_NAMES[0]}: 200 frames 50.0 ps apart span 9.95 ns, production.mdp sets 500.0 ns")
    assert "--workers 4096 is more than" in entry_make.describe_source(
        replicas, manifest, {}, simulations, None, 4096, 5)["warnings"][0]


def test_command_line_in_a_job_environment(run, tmp_path):
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("GLYCOSHAPE_", "POCKETBASE", "SLURM_")) and not key.endswith("_NUM_THREADS")}
    environment.update({
        # the job of a simulation: one CPU per task, services and proxies that must not matter
        "SLURM_CPUS_PER_TASK": "1", "SLURM_JOB_ID": "4061692", "OMP_NUM_THREADS": "1",
        "GLYCOSHAPE_ORACLE_PAR_URL": "https://objectstorage.invalid/p/x/", "POCKETBASE_URL": "http://pocketbase.invalid",
        "POCKETBASE_TOKEN": "token", "http_proxy": "http://127.0.0.1:9", "https_proxy": "http://127.0.0.1:9",
    })
    output = tmp_path / "out" / "bundle.tar"
    output.parent.mkdir()
    command = [sys.executable, str(SCRIPT), "--glycam-name", GLYCAM_NAME, "--workers", "2", "--output", str(output),
               "--frame-stride", "5", "--force-field", "GLYCAM_06j", "--summary", str(tmp_path / "summary.json"),
               *map(str, run["replicas"])]
    completed = subprocess.run(command, env=environment, cwd=str(tmp_path), capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stderr[-3000:]
    lines = completed.stdout.splitlines()
    assert len(lines) == 1 and lines[0].startswith(
        f"make_entry: ok: {GLYCAM_NAME}: 480 frames of 45 atoms (3 replica(s)), 478 kept; clusters level 1: 5, ")
    assert lines[0].endswith(" with 2 workers")
    assert sorted(path.name for path in output.parent.iterdir()) == ["bundle.tar"]      # the work folder is gone
    assert not (tmp_path / "last_run.log").exists()

    opened = entry_bundle.open_bundle(output, tmp_path / "opened", expected_glycam_name=GLYCAM_NAME)
    bundle = json.loads((opened / "bundle.json").read_text())
    result = json.loads((opened / "result.json").read_text())
    assert (result["settings"]["workers"], result["settings"]["workers_from"]) == (2, "--workers")
    assert set(result["settings"]["threads"].values()) == {"2"}
    assert bundle["source"]["host"] == dict(bundle["source"]["host"], slurm_job_id="4061692", workers=2)
    assert [replica["name"] for replica in bundle["source"]["replicas"]] == ["replica_1", "replica_2", "replica_3"]
    assert bundle["source"]["force_field"] == "GLYCAM_06j" and bundle["source"]["md_package"] is None
    assert archive.verify_entry(opened / "archive", check_folder=False)["ok"]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["bundle"] == {"path": str(output), "size": output.stat().st_size, "sha256": sha256(output.read_bytes())}
    assert f"sha256 {summary['bundle']['sha256']}" in lines[0] and summary["work_dir"] is None
    # the same frames as the bundle of the session, which was made in this process with one worker
    assert members_of(output)["archive/trajectory.xtc"] == members_of(run["bundle"])["archive/trajectory.xtc"]

    for arguments, message in (
        (["--glycam-name", GLYCAM_NAME, "--output", str(output), str(run["replicas"][0])], "--workers"),
        (["--workers", "2", "--output", str(output), str(run["replicas"][0])], "--glycam-name"),
        (["--glycam-name", GLYCAM_NAME, "--workers", "2", str(run["replicas"][0])], "--output"),
    ):
        refused = subprocess.run([sys.executable, str(SCRIPT), *arguments], env=environment, cwd=str(tmp_path),
                                 capture_output=True, text=True, timeout=600)
        assert refused.returncode == 2 and message in refused.stderr and "required" in refused.stderr


def test_same_command_again_reuses_what_is_current_and_continues_after_an_interruption(run, tmp_path):
    replicas = entry_make.resolve_replicas([str(folder) for folder in synthetic_run.copy_of_run(tmp_path, run)])
    output, work = tmp_path / "bundle.tar", tmp_path / "work"

    def make(**options):
        return entry_make.make_entry(replicas, output, GLYCAM_NAME, workers=1, work_dir=work, keep_work=True, **options)

    first = make()
    assert (first["entry"]["status"], first["result"]["reused"], first["work_dir"]) == ("converted", False, str(work))
    assert sorted(path.name for path in work.iterdir()) == ["archive", "make_entry.json", "result"]
    members = members_of(output)

    again = make()
    assert (again["entry"]["status"], again["result"]["reused"]) == ("skipped", True)
    rewritten = members_of(output)
    assert {name: data for name, data in rewritten.items() if name != "bundle.json"} == {
        name: data for name, data in members.items() if name != "bundle.json"}

    # a job that was killed in the analysis: the entry is there, the result says "running"
    output.unlink()
    result = json.loads((work / "result" / "result.json").read_text())
    (work / "result" / "result.json").write_text(json.dumps(dict(result, status="running", finished_at=None)))
    shutil.rmtree(work / "result" / "static" / "landscape")
    resumed = make()
    assert (resumed["status"], resumed["entry"]["status"], resumed["result"]["reused"]) == ("ok", "skipped", False)
    assert entry_bundle.check_bundle(output)["glycam_name"] == GLYCAM_NAME
    # ... and one that was killed while the entry was written: the staging folder is not an entry
    shutil.rmtree(work / "archive" / GLYCAM_NAME)
    (work / "archive" / archive.STAGING_DIR / "leftover").mkdir(parents=True)
    rewritten = make()
    # the result was computed from the manifest of the entry that is gone: it is not reused
    assert (rewritten["entry"]["status"], rewritten["result"]["reused"], rewritten["status"]) == ("converted", False, "ok")

    # other levels: the entry stays, the analysis is done again; --force does both again
    levels = make(levels="1,2,3")
    assert (levels["entry"]["status"], levels["result"]["reused"], levels["levels"]) == ("skipped", False, [1, 2, 3])
    assert "static/output/level_3/PDB/alpha.pdb" in members_of(output)
    forced = make(levels="1,2,3", force=True)
    assert (forced["entry"]["status"], forced["result"]["reused"]) == ("converted", False)

    # without --keep-work the work folder is removed once the bundle is there
    final = entry_make.make_entry(replicas, output, GLYCAM_NAME, workers=1, work_dir=work, levels="1,2,3")
    assert final["status"] == "ok" and final["work_dir"] is None and not work.exists() and output.is_file()

    # a work folder is never shared with another entry, and nothing else is written into
    entry_make.claim_work_dir(work, "DGlcpa1-OH")
    (work / "archive").mkdir()
    with pytest.raises(entry_make.MakeEntryError, match="work folder .* belongs to DGlcpa1-OH"):
        make()
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "notes.txt").write_text("mine")
    with pytest.raises(entry_make.MakeEntryError, match="is not empty and is not a work folder of this tool"):
        entry_make.make_entry(replicas, output, GLYCAM_NAME, workers=1, work_dir=foreign)
    assert sorted(path.name for path in foreign.iterdir()) == ["notes.txt"]


def test_input_that_cannot_be_used_exits_with_2_and_writes_no_bundle(run, tmp_path, capsys):
    folders = synthetic_run.copy_of_run(tmp_path, run)
    output = tmp_path / "bundle.tar"
    base = ["--glycam-name", GLYCAM_NAME, "--workers", "1", "--output", str(output)]

    def refused(arguments, *messages, replicas=folders):
        assert entry_make.main([*base, *arguments, *map(str, replicas)]) == 2
        captured = capsys.readouterr()
        assert captured.out == "" and captured.err.startswith("make_entry: error: ")
        assert all(message in captured.err for message in messages), captured.err
        assert not output.exists()

    # a replica of another glycan: other atoms
    other = tmp_path / "run" / "replica_2" / "glycan.dry.pdb"
    original = other.read_text()
    assert " O5  2MA X   2 " in original
    other.write_text(original.replace(" O5  2MA X   2 ", " S5  2MA X   2 "))
    refused([], "replica_2 does not have the atoms of replica_1: 1 of 45 atoms differ in name, residue or order; "
                "the first is atom 18: name 'S5' residue '2MA' 2 in replica_2 against name 'O5' residue '2MA' 2 in replica_1")
    assert not (tmp_path / "bundle.tar.work" / "archive").exists()
    # a replica that was cut off
    other.write_text(original[: len(original) // 2])
    refused([], "replica_2 (", "glycan.dry.pdb): frame ", "is not terminated by ENDMDL; the file is truncated")
    assert not (tmp_path / "bundle.tar.work" / "archive" / GLYCAM_NAME).exists()
    other.write_text(original)
    refused(["--levels", "1,3"], "cluster levels must be 1,2 or 1,2,3")
    refused(["--frame-stride", "0"], "--frame-stride must be at least 1")
    refused(["--temperature-k", "-5"], "--temperature-k must not be negative")
    refused(["--info", str(tmp_path / "no-info.json")], "no such info.json")
    refused(["--extend", str(tmp_path / "run" / "replica_1")], "is not an archive entry")
    refused([], "no such replica folder or file", replicas=[tmp_path / "run" / "replica_9"])
    refused([], "no replicas were given", replicas=[])
    assert entry_make.main(["--glycam-name", "a/../b", "--workers", "1", "--output", str(output), str(folders[0])]) == 2
    assert "not usable as an object key" in capsys.readouterr().err
    assert entry_make.main(["--glycam-name", GLYCAM_NAME, "--workers", "0", "--output", str(output), str(folders[0])]) == 2
    assert "--workers must be at least 1" in capsys.readouterr().err
    assert entry_make.main(["--glycam-name", GLYCAM_NAME, "--workers", "1", "--output", str(tmp_path), str(folders[0])]) == 2
    assert "is a folder; name the bundle file" in capsys.readouterr().err
    assert entry_make.main(["--glycam-name", GLYCAM_NAME, "--workers", "1", "--output", str(tmp_path / "no" / "b.tar"),
                            str(folders[0])]) == 2
    assert "the folder of the output does not exist" in capsys.readouterr().err
    assert not output.exists()


def test_failed_analysis_exits_with_1_and_leaves_the_result_of_the_failure(run, tmp_path, capsys, monkeypatch):
    folders = synthetic_run.copy_of_run(tmp_path, run)
    output = tmp_path / "bundle.tar"
    from lib import v2

    def broken(self, glycan):
        raise RuntimeError("the cluster ran out of memory in the PCA step")

    run_analysis = v2.GlycanAnalysisPipeline.run_analysis
    monkeypatch.setattr(v2.GlycanAnalysisPipeline, "run_analysis", broken)
    arguments = ["--glycam-name", GLYCAM_NAME, "--workers", "1", "--output", str(output), "--summary",
                 str(tmp_path / "summary.json"), *map(str, folders)]
    assert entry_make.main(arguments) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "make_entry: FAILED in step process: the cluster ran out of memory" in captured.err
    assert "no bundle was written" in captured.err and not output.exists()
    work = tmp_path / "bundle.tar.work"
    result = json.loads((work / "result" / "result.json").read_text())
    assert result["status"] == "failed" and result["error"]["step"] == "process"
    assert "ran out of memory" in (work / "result" / "run.log").read_text()
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "failed"

    # the same command afterwards: the entry is reused, the analysis runs, the bundle is written
    monkeypatch.setattr(v2.GlycanAnalysisPipeline, "run_analysis", run_analysis)
    assert entry_make.main(arguments) == 0
    line = capsys.readouterr().out
    assert line.startswith(f"make_entry: ok: {GLYCAM_NAME}: 480 frames") and line.count("\n") == 1
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["entry"]["status"] == "skipped" and summary["status"] == "ok" and not work.exists()

    def no_room(*args, **kwargs):
        raise entry_bundle.BundleError("cannot write: no space left on device")

    # a run that fails leaves no bundle, not even the one of the run before it
    monkeypatch.setattr(entry_bundle, "write_bundle", no_room)
    assert output.is_file()
    assert entry_make.main(arguments) == 1
    assert "make_entry: FAILED writing the bundle: cannot write: no space left on device" in capsys.readouterr().err
    assert not output.exists()

    # ... and a bundle that the receiving side would refuse is not left under its name
    def not_a_bundle(tar_path, *args, **kwargs):
        Path(tar_path).write_bytes(b"not a tar file")
        return {"path": str(tar_path), "size": 14, "sha256": "0" * 64, "bundle": {"levels": [1, 2]}}

    monkeypatch.setattr(entry_bundle, "write_bundle", not_a_bundle)
    assert entry_make.main(arguments) == 1
    assert "make_entry: FAILED writing the bundle: " in capsys.readouterr().err and not output.exists()


# ---------------------------------------------------------------------------
# Merge and extend
# ---------------------------------------------------------------------------
def test_extending_a_stored_entry_gives_the_bundle_of_the_union(run, tmp_path, capsys):
    """Two replicas are in the database; the third is added to them: the result is that of all three."""
    folders = run["replicas"]
    stored = archive_segments.convert_segments(
        [archive_segments.SegmentSource(folder.name, folder / "glycan.dry.pdb", folder / "glycan.dry.mol2")
         for folder in folders[:2]],
        tmp_path / "raw", GLYCAM_NAME, info=folders[0] / "info.json")
    stored_dir = Path(stored["entry"])
    stored_files = {path.name: path.read_bytes() for path in stored_dir.iterdir()}
    stored_sha256 = sha256(stored_files["manifest.json"])

    output = tmp_path / "bundle.tar"
    assert entry_make.main([
        "--glycam-name", GLYCAM_NAME, "--workers", "1", "--output", str(output), "--extend", str(stored_dir),
        "--frame-stride", "5", f"third={folders[2]}"]) == 0
    line = capsys.readouterr().out
    assert "480 frames of 45 atoms (1 replica(s), 330 of them from the entry that is extended), 478 kept" in line
    assert {path.name: path.read_bytes() for path in stored_dir.iterdir()} == stored_files     # only read

    opened = entry_bundle.open_bundle(output, tmp_path / "opened")
    bundle = json.loads((opened / "bundle.json").read_text())
    assert bundle["extends"] == {"glycam_name": GLYCAM_NAME, "manifest_sha256": stored_sha256, "frames": 330}
    manifest = archive.load_manifest(opened / "archive")
    assert [(s["name"], s["first_frame"], s["frames"], s.get("carried_from", {}).get("segment")) for s in manifest["segments"]] == [
        ("replica_1", 0, 200, 0), ("replica_2", 200, 130, 1), ("third", 330, 150, None)]
    assert manifest["extends"]["manifest_sha256"] == stored_sha256 and manifest["frames"] == 480
    assert manifest["segments"][2]["time_per_frame_ps"] == 50.0 and manifest["segments"][0]["time_per_frame_ps"] is None
    # the stored frames are the first bytes of the new trajectory, and the stored entry is what was extended
    trajectory = (opened / "archive" / "trajectory.xtc").read_bytes()
    assert trajectory[:len(stored_files["trajectory.xtc"])] == stored_files["trajectory.xtc"]
    assert archive_segments.check_extension(opened / "archive", stored_dir) == {
        "ok": True, "errors": [], "trajectory_bytes_equal": True, "frames_compared": 330}
    assert archive.verify_entry(opened / "archive", check_folder=False)["ok"]
    source = bundle["source"]
    assert source["trajectory"] == {"atoms": 45, "frames": 480, "frames_from_this_run": 150,
                                    "frames_from_extended_entry": 330, "segments": 3}
    assert [replica["name"] for replica in source["replicas"]] == ["third"] and source["simulation_length_ns"] == 7.45

    # the union of the stored and the new frames is the trajectory of the three replicas: the same analysis
    members, union = members_of(output), members_of(run["bundle"])
    assert members["archive/trajectory.xtc"] == union["archive/trajectory.xtc"]
    assert members["archive/topology.pdb"] == union["archive/topology.pdb"]
    different = [name for name in union if name.startswith("static/") and members[name] != union[name]]
    assert all(name.endswith("dist.svg") for name in different), different
    assert json.loads(members["result.json"])["analysis"] == json.loads(union["result.json"])["analysis"]

    # another glycan's replica is refused, and says what differs
    other = tmp_path / "other"
    shutil.copytree(folders[2], other)
    pdb = other / "glycan.dry.pdb"
    pdb.write_text(pdb.read_text().replace(" O5  2MA X   2 ", " S5  2MA X   2 "))
    output.unlink()
    assert entry_make.main(["--glycam-name", GLYCAM_NAME, "--workers", "1", "--output", str(output), "--extend",
                            str(stored_dir), str(other)]) == 2
    message = capsys.readouterr().err
    assert "other does not have the atoms of topology.pdb of the entry to extend: 1 of 45 atoms differ in name, " \
           "residue or order; the first is atom " in message
    assert "atom 18: name 'S5' residue '2MA' 2 in other against name 'O5' residue '2MA' 2 in topology.pdb" in message
    assert entry_make.main(["--glycam-name", "DGlcpa1-OH", "--workers", "1", "--output", str(tmp_path / "second.tar"),
                            "--extend", str(stored_dir), str(folders[2])]) == 2
    assert "the entry to extend is DManpa1-2DManpa1-OH, the new trajectories are named DGlcpa1-OH" in capsys.readouterr().err
    # a replica with the name of a stored segment (runs name their replicas alike) needs another name
    assert entry_make.main(["--glycam-name", GLYCAM_NAME, "--workers", "1", "--output", str(tmp_path / "third.tar"),
                            "--extend", str(stored_dir), str(folders[1])]) == 2
    assert "replica_2 is already the name of a segment of the entry to extend; give the new trajectories other names " \
           "(make_entry.py: --replica-prefix, or NAME=PATH)" in capsys.readouterr().err
    assert not output.exists() and not (tmp_path / "second.tar").exists() and not (tmp_path / "third.tar").exists()
