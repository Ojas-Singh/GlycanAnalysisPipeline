"""A synthetic simulation run for the tests of the node command and of the bundle.

The 480-frame trajectory of ``synthetic_glycan`` as three replica folders, laid out as a run on the cluster
leaves them (``glycan.dry.pdb``, ``glycan.dry.mol2``, ``production.mdp``, ``info.json`` and a replica map), the
same frames as one ``simulation.pdb`` folder, and the bundle ``lib/entry_make.py`` makes of the replicas. The
bundle is made once per test session (:func:`get`); with at most 500 frames the PCA fit is exact, so two
analyses of the same frames give the same files.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import lib.config as config

from synthetic_glycan import EXAMPLE, GLYCAM_NAME, pdb_text, trajectory

CUTS = (0, 200, 330, 480)
#: 10 ps between saved frames, every 5th kept: 50 ps per frame; the 200 frames of the first replica span 9.95 ns.
MDP = (
    "; Run parameters\n"
    "integrator              = md        ; leap-frog integrator\n"
    "nsteps                  = {nsteps} ; 2 fs steps\n"
    "dt                      = 0.002     ; 2 fs\n"
    "nstxout-compressed      = 5000      ; save compressed coordinates every 10.0 ps\n"
)
METADATA = {
    "md_package": "GROMACS", "force_field": "GLYCAM_06j", "temperature_k": 300.0, "pressure_bar": 1.0,
    "salt_concentration_mm": 200.0, "comments": "Auto upload from GOTW", "glytoucan_id": None, "run_id": "run-0001",
    "simulation_length_ns": None,
}
SETTINGS = ("max_workers", "cluster_levels", "use_oracle_storage", "oracle_par_url", "use_pocketbase")
_RUN: dict = {}


def write_replicas(root, cuts=CUTS, frames=480, seed=7, prefix="replica", broken=()):
    """Replica folders under ``root`` and their replica map; returns the folders."""
    lines, coordinates = trajectory(frames, seed, broken)
    root = Path(root)
    folders = []
    for index, (start, stop) in enumerate(zip(cuts, cuts[1:]), start=1):
        folder = root / f"{prefix}_{index}"
        folder.mkdir(parents=True)
        (folder / "glycan.dry.pdb").write_text(pdb_text(lines, coordinates[start:stop]), encoding="ascii")
        (folder / "glycan.dry.mol2").write_bytes((EXAMPLE / f"{GLYCAM_NAME}.mol2").read_bytes())
        (folder / "production.mdp").write_text(MDP.format(nsteps=(stop - start - 1) * 25000), encoding="ascii")
        (folder / "info.json").write_text(
            json.dumps({"indexOrderedSequence": GLYCAM_NAME, "conformerID": "1ogg"}, indent=2) + "\n", encoding="utf-8")
        folders.append(folder)
    (root / "replica_map.txt").write_text(
        "".join(f"{folder}|{GLYCAM_NAME}_{folder.name}\n" for folder in folders), encoding="utf-8")
    return folders


def write_whole(root, frames=480, seed=7, broken=()):
    """The same frames as one source folder (``simulation.pdb``, ``structure.mol2``, ``info.json``)."""
    lines, coordinates = trajectory(frames, seed, broken)
    folder = Path(root) / GLYCAM_NAME
    folder.mkdir(parents=True)
    (folder / "simulation.pdb").write_text(pdb_text(lines, coordinates), encoding="ascii")
    (folder / "structure.mol2").write_bytes((EXAMPLE / f"{GLYCAM_NAME}.mol2").read_bytes())
    return folder


class keep_settings:
    """lib.entry_run sets the worker count, the levels and local storage in lib.config; undo it afterwards."""

    def __enter__(self):
        self._saved = {name: getattr(config, name) for name in SETTINGS}
        return self

    def __exit__(self, *exc):
        for name, value in self._saved.items():
            setattr(config, name, value)
        from lib.storage import reset_storage_manager

        reset_storage_manager()
        return False


def get(tmp_path_factory) -> dict:
    """The run of this test session: ``root``, ``replicas``, ``whole``, ``bundle``, ``work``, ``made``."""
    if not _RUN:
        from lib import entry_make

        root = tmp_path_factory.mktemp("synthetic_run")
        folders = write_replicas(root / "run", broken=(11, 200))
        whole = write_whole(root / "data", broken=(11, 200))
        replicas = entry_make.resolve_replicas([], root / "run" / "replica_map.txt")
        with keep_settings():
            made = entry_make.make_entry(
                replicas, root / "bundle.tar", GLYCAM_NAME, workers=1, frame_stride=5, metadata=dict(METADATA),
                keep_work=True,
            )
        assert made["status"] == "ok", made
        _RUN.update(root=root, replicas=folders, whole=whole, bundle=root / "bundle.tar",
                    work=root / "bundle.tar.work", made=made)
    return _RUN


def copy_of_run(tmp_path, run) -> list:
    """A private copy of the replica folders that a test may change."""
    shutil.copytree(run["root"] / "run", tmp_path / "run")
    return sorted(path for path in (tmp_path / "run").iterdir() if path.is_dir())
