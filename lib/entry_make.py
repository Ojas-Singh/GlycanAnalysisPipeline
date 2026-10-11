"""The dry trajectories of a simulation run -> one entry bundle, on the node that ran the simulation.

    python scripts/make_entry.py --glycam-name NAME --workers N --output BUNDLE.tar REPLICA [REPLICA ...]

Each ``REPLICA`` is what one simulation of the run left: a folder with
``glycan.dry.pdb`` and ``glycan.dry.mol2`` (or ``simulation.pdb`` and
``structure.mol2``), or the PDB file itself (its MOL2 next to it).  The
command

1. checks that all replicas have the same atoms in the same order,
2. converts them into one archive entry, each replica read and verified
   against its own text and recorded as a segment (``lib/archive_segments.py``);
   with ``--extend ENTRY`` the frames of that existing entry come first,
3. runs the analysis on that entry (``lib/entry_run.py``),
4. writes the bundle (``lib/entry_bundle.py``) and reads it back the way the
   receiving side does.

No network and no service is used.  Everything is done in a work folder next
to the output (``--work-dir``); a bundle that an earlier run left under the
name of the output is removed first, the bundle appears under its name only when
it is complete, and the work folder is removed then (``--keep-work`` keeps
it).  After an interruption the same command continues: an entry that was
converted from the same inputs and a result that is current are reused.

Exit status: ``0`` the bundle was written; ``1`` the analysis or the bundle
failed; ``2`` the input cannot be used (nothing was computed).  One line on
standard output says what was made; errors go to standard error.

Only the standard library is imported here.  ``main`` puts the worker count
into the environment before NumPy and ``lib.config`` are imported, as
``lib/entry_run.py`` does.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import shutil
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from lib import entry_run

WORK_SCHEMA = "glycoshape-make-entry-work"
WORK_FILE = "make_entry.json"
#: File names of a replica folder, most specific first.
REPLICA_FILES = (("glycan.dry.pdb", "glycan.dry.mol2"), ("simulation.pdb", "structure.mol2"))
MDP_FILE = "production.mdp"
#: Metadata of a run that is recorded with every segment of the manifest (``run``).
RUN_KEYS = ("run_id", "md_package", "force_field", "temperature_k", "pressure_bar", "salt_concentration_mm")
UNITS = {
    "simulation_length_ns": "nanoseconds, of the trajectories of this run (not of an entry that is extended)",
    "time_per_frame_ps": "picoseconds between two frames of a replica",
    "temperature_k": "kelvin",
    "pressure_bar": "bar",
    "salt_concentration_mm": "millimolar NaCl",
}


class MakeEntryError(Exception):
    """A problem with the input or the output; reported, exit status 2."""


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="make_entry.py",
        description="Archive entry, analysis and bundle of the dry trajectories of one simulation run, "
                    "without network access.",
    )
    parser.add_argument(
        "replicas", nargs="*", metavar="REPLICA",
        help="replica folder (glycan.dry.pdb + glycan.dry.mol2, or simulation.pdb + structure.mol2) or PDB file; "
             "NAME=PATH names the replica (default: the folder name)",
    )
    parser.add_argument("--replica-map", type=Path, help="file of 'folder|name' lines (replica_map.txt of a run)")
    parser.add_argument("--replica-prefix", default="",
                        help="put this before the name of every replica, for example the identifier of the run: the "
                             "segments of an entry are told apart by their names")
    parser.add_argument("--glycam-name", required=True, help="full GLYCAM name of the glycan; written as name.json")
    parser.add_argument("--workers", type=int, required=True,
                        help="cores to use (required: the job environment of a simulation says nothing useful)")
    parser.add_argument("--output", "-o", type=Path, required=True, help="the bundle to write (a .tar file)")
    parser.add_argument("--info", type=Path, help="info.json to archive (default: the one of the first replica folder that has one)")
    parser.add_argument("--mol2", type=Path, help="structure.mol2 of the entry (default: the MOL2 of the first replica)")
    parser.add_argument("--levels", default=None, help="clustering levels: 1,2 (default) or 1,2,3")
    parser.add_argument("--extend", type=Path, metavar="ENTRY",
                        help="existing archive entry of this glycan: its frames come first, the replicas are added")
    parser.add_argument("--extend-time-per-frame-ps", type=float, default=None,
                        help="time between the frames of ENTRY, when its manifest does not say")
    timing = parser.add_mutually_exclusive_group()
    timing.add_argument("--time-per-frame-ps", type=float, default=None, help="time between two frames of a replica")
    timing.add_argument("--frame-stride", type=int, default=None,
                        help="the dry trajectory holds every N-th saved frame; with production.mdp in the replica "
                             "folder this gives the time per frame (dt x nstxout-compressed x N)")
    parser.add_argument("--simulation-length-ns", type=float, default=None,
                        help="aggregated length of this run in nanoseconds (default: from the frames, else production.mdp)")
    parser.add_argument("--md-package", default=None)
    parser.add_argument("--force-field", default=None)
    parser.add_argument("--temperature-k", type=float, default=None)
    parser.add_argument("--pressure-bar", type=float, default=None)
    parser.add_argument("--salt-concentration-mm", type=float, default=None)
    parser.add_argument("--comments", default=None)
    parser.add_argument("--glytoucan-id", default=None)
    parser.add_argument("--run-id", default=None, help="identifier of the run on the submitting side")
    parser.add_argument("--source-name", default=None, help="name of the source in the manifest (default: the entry folder)")
    parser.add_argument("--work-dir", type=Path, default=None, help="work folder (default: OUTPUT.work)")
    parser.add_argument("--keep-work", action="store_true", help="keep the work folder (entry, processed data, result)")
    parser.add_argument("--summary", type=Path, default=None, help="also write what was made as JSON to this file")
    parser.add_argument("--force", action="store_true", help="convert and analyse again even when the work folder is current")
    parser.add_argument("--repair", action="store_true",
                        help="accept a replica whose PDB is damaged in one of the ways the archive converter repairs")
    parser.add_argument("--verbose", action="store_true", help="print the log of the analysis")
    return parser


# ---------------------------------------------------------------------------
# Replicas
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> Tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(4 << 20)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def read_mdp(path: Path) -> Dict[str, Any]:
    """Time step, steps and output interval of a GROMACS .mdp file, with explicit units.

    ``length_ns`` is ``dt x nsteps`` and ``saved_every_ps`` is ``dt x
    nstxout-compressed``; a value the file does not give is ``None``.
    """
    values: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = raw.split(";", 1)[0].replace("=", " ", 1).split()
        if len(fields) >= 2:
            values[fields[0].lower().replace("-", "_")] = fields[1]

    def number(key: str) -> Optional[float]:
        try:
            value = float(values[key])
        except (KeyError, ValueError):
            return None
        return value if value > 0 else None

    dt, steps = number("dt"), number("nsteps")
    interval = number("nstxout_compressed") or number("nstxtcout")
    return {
        "file": path.name,
        "sha256": _sha256(path)[1],
        "dt_ps": dt,
        "nsteps": int(steps) if steps is not None else None,
        "length_ns": round(dt * steps / 1000.0, 9) if dt and steps else None,
        "nstxout_compressed": int(interval) if interval is not None else None,
        "saved_every_ps": round(dt * interval, 9) if dt and interval else None,
    }


def resolve_replica(text: str, name: Optional[str] = None) -> Dict[str, Any]:
    """Files of one replica: ``{"name", "pdb", "mol2", "folder", "mdp", "info"}``."""
    if name is None and "=" in text and not Path(text).exists():
        name, text = text.split("=", 1)
    path = Path(text).expanduser()
    if path.is_dir():
        folder = path.resolve()
        candidates = REPLICA_FILES + ((f"{folder.name}.pdb", f"{folder.name}.mol2"),)
        found = next((pair for pair in candidates if (folder / pair[0]).is_file()), None)
        if found is None:
            raise MakeEntryError(
                f"replica folder {path} has none of " + ", ".join(pair[0] for pair in candidates)
            )
        pdb = folder / found[0]
        mol2 = next((folder / pair[1] for pair in candidates if (folder / pair[1]).is_file()), None)
        default = folder.name
    elif path.is_file():
        pdb = path.resolve()
        folder = pdb.parent
        candidates = [pdb.with_suffix(".mol2")] + [folder / pair[1] for pair in REPLICA_FILES]
        mol2 = next((candidate for candidate in candidates if candidate.is_file()), None)
        default = folder.name if pdb.name in [pair[0] for pair in REPLICA_FILES] else pdb.stem
    else:
        raise MakeEntryError(f"no such replica folder or file: {text}")
    mdp = folder / MDP_FILE
    info = folder / "info.json"
    return {
        "name": (name or default).strip(),
        "pdb": pdb,
        "mol2": mol2,
        "folder": folder,
        "mdp": mdp if mdp.is_file() else None,
        "info": info if info.is_file() else None,
    }


def read_replica_map(path: Path) -> List[Tuple[str, Optional[str]]]:
    """``folder|name`` lines of a replica map; relative folders are relative to the map."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise MakeEntryError(f"cannot read the replica map {path}: {exc.strerror}") from exc
    replicas = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        folder, _, name = line.partition("|")
        folder_path = Path(folder.strip())
        if not folder_path.is_absolute():
            folder_path = path.resolve().parent / folder_path
        replicas.append((str(folder_path), name.strip() or None))
    return replicas


def resolve_replicas(
    arguments: Sequence[str], replica_map: Optional[Path] = None, prefix: str = ""
) -> List[Dict[str, Any]]:
    """The replicas of a map and of the command line, in that order; ``prefix`` goes before every name."""
    replicas = [resolve_replica(folder, name) for folder, name in (read_replica_map(replica_map) if replica_map else [])]
    replicas += [resolve_replica(text) for text in arguments]
    if not replicas:
        raise MakeEntryError("no replicas were given (REPLICA arguments or --replica-map)")
    for replica in replicas:
        replica["name"] = f"{prefix}{replica['name']}"
    names = [replica["name"] for replica in replicas]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated or not all(names):
        raise MakeEntryError(
            f"the replicas need distinct names; {', '.join(repeated) or 'an empty name'} occurs more than once "
            "(name them with NAME=PATH)"
        )
    return replicas


def _time_per_frame(simulation: Optional[Dict[str, Any]], explicit: Optional[float], stride: Optional[int]) -> Optional[float]:
    if explicit is not None:
        return float(explicit)
    if stride is not None and simulation and simulation.get("saved_every_ps"):
        return round(simulation["saved_every_ps"] * stride, 9)
    return None


# ---------------------------------------------------------------------------
# The work folder
# ---------------------------------------------------------------------------
def claim_work_dir(work_dir: Path, glycam_name: str) -> None:
    """The work folder must be new, empty, or the work folder of this entry."""
    marker = work_dir / WORK_FILE
    if work_dir.exists() and not work_dir.is_dir():
        raise MakeEntryError(f"work path is not a folder: {work_dir}")
    if work_dir.is_dir() and any(work_dir.iterdir()):
        try:
            recorded = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            recorded = None
        if not isinstance(recorded, dict) or recorded.get("schema") != WORK_SCHEMA:
            raise MakeEntryError(
                f"work folder {work_dir} is not empty and is not a work folder of this tool; choose another "
                "(--work-dir)"
            )
        if recorded.get("glycam_name") != glycam_name:
            raise MakeEntryError(
                f"work folder {work_dir} belongs to {recorded.get('glycam_name')}; choose another (--work-dir) "
                "or delete it"
            )
    work_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({"schema": WORK_SCHEMA, "glycam_name": glycam_name}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
def describe_source(
    replicas: Sequence[Dict[str, Any]],
    manifest: Dict[str, Any],
    metadata: Dict[str, Any],
    simulations: Sequence[Optional[Dict[str, Any]]],
    info_path: Optional[Path],
    workers: int,
    stride: Optional[int],
) -> Dict[str, Any]:
    """``source`` of ``bundle.json``: the replicas, the run and its metadata, with units in the key names."""
    segments = {segment["name"]: segment for segment in manifest["segments"] if "carried_from" not in segment}
    warnings: List[str] = []
    described = []
    for replica, simulation in zip(replicas, simulations):
        segment = segments[replica["name"]]
        step = segment.get("time_per_frame_ps")
        sampled = round((segment["frames"] - 1) * step / 1000.0, 9) if step else None
        if simulation and simulation.get("length_ns") and sampled is not None \
                and abs(sampled - simulation["length_ns"]) > 1e-6 * max(1.0, simulation["length_ns"]):
            warnings.append(
                f"{replica['name']}: {segment['frames']} frames {step} ps apart span {sampled} ns, "
                f"{MDP_FILE} sets {simulation['length_ns']} ns"
            )
        described.append({
            "name": replica["name"],
            "first_frame": segment["first_frame"],
            "frames": segment["frames"],
            "time_per_frame_ps": step,
            "sampled_ns": sampled,
            "pdb": {"file": replica["pdb"].name, "size": segment["source"]["size"], "sha256": segment["source"]["sha256"]},
            "mol2": None if replica["mol2"] is None else {
                "file": replica["mol2"].name, "sha256": _sha256(replica["mol2"])[1]},
            "simulation": simulation,
        })

    length = metadata.get("simulation_length_ns")
    origin: Optional[str] = "--simulation-length-ns" if length is not None else None
    if length is None and all(item["sampled_ns"] is not None for item in described):
        length = round(sum(item["sampled_ns"] for item in described), 9)
        origin = "the frames of every replica: (frames - 1) x time_per_frame_ps"
    if length is None and all(item["simulation"] and item["simulation"].get("length_ns") for item in described):
        length = round(sum(item["simulation"]["length_ns"] for item in described), 9)
        origin = f"{MDP_FILE} of every replica: dt x nsteps"
    steps = {item["time_per_frame_ps"] for item in described}
    extends = manifest.get("extends")
    new_frames = sum(item["frames"] for item in described)
    try:
        cores = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cores = os.cpu_count()
    if cores and workers > cores:
        warnings.append(f"--workers {workers} is more than the {cores} cores this process may run on")
    return {
        "kind": "simulation-run",
        "name": manifest["source"]["folder"],
        "replicas": described,
        "trajectory": {
            "atoms": manifest["atoms"],
            "frames": manifest["frames"],
            "frames_from_this_run": new_frames,
            "frames_from_extended_entry": extends["frames"] if extends else 0,
            "segments": len(manifest["segments"]),
        },
        "simulation_length_ns": length,
        "simulation_length_from": origin,
        "time_per_frame_ps": steps.pop() if len(steps) == 1 else None,
        "frame_stride": stride,
        "md_package": metadata.get("md_package"),
        "force_field": metadata.get("force_field"),
        "temperature_k": metadata.get("temperature_k"),
        "pressure_bar": metadata.get("pressure_bar"),
        "salt_concentration_mm": metadata.get("salt_concentration_mm"),
        "comments": metadata.get("comments"),
        "glytoucan_id": metadata.get("glytoucan_id"),
        "run_id": metadata.get("run_id"),
        "units": UNITS,
        "info_json": None if info_path is None else {"file": info_path.name, "sha256": _sha256(info_path)[1]},
        "host": {
            "hostname": socket.gethostname(),
            "machine": platform.machine(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cores_available": cores,
            "workers": workers,
        },
        "warnings": warnings,
    }


def make_entry(
    replicas: Sequence[Dict[str, Any]],
    output: Path,
    glycam_name: str,
    *,
    workers: int,
    levels: Optional[str] = None,
    info: Optional[Path] = None,
    mol2: Optional[Path] = None,
    extend: Optional[Path] = None,
    extend_time_per_frame_ps: Optional[float] = None,
    time_per_frame_ps: Optional[float] = None,
    frame_stride: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
    source_name: Optional[str] = None,
    work_dir: Optional[Path] = None,
    keep_work: bool = False,
    force: bool = False,
    repair: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Convert, analyse and bundle; returns what was made.

    Raises :class:`MakeEntryError` (and ``lib.archive.ArchiveError``,
    ``lib.entry_run.EntryError``) when the input cannot be used.  The
    returned ``status`` is ``"ok"`` or ``"failed"``; a failed analysis leaves
    the work folder with ``result/result.json`` and ``result/run.log``.
    """
    from lib import archive, archive_segments, entry_bundle

    started = time.perf_counter()
    metadata = dict(metadata or {})
    output = Path(output).absolute()
    if output.is_dir():
        raise MakeEntryError(f"the output {output} is a folder; name the bundle file")
    if not output.parent.is_dir():
        raise MakeEntryError(f"the folder of the output does not exist: {output.parent}")
    work_dir = Path(work_dir).absolute() if work_dir else output.with_name(output.name + ".work")
    if frame_stride is not None and frame_stride < 1:
        raise MakeEntryError("--frame-stride must be at least 1")
    for key in ("simulation_length_ns", "temperature_k", "pressure_bar", "salt_concentration_mm"):
        if metadata.get(key) is not None and not metadata[key] >= 0:
            raise MakeEntryError(f"--{key.replace('_', '-')} must not be negative")
    if info is None:
        info = next((replica["info"] for replica in replicas if replica["info"] is not None), None)
    elif not Path(info).is_file():
        raise MakeEntryError(f"no such info.json: {info}")
    try:
        archive._require_glycam_name(glycam_name)
        if levels:
            import lib.config as config

            config.parse_cluster_levels(levels)     # refused before anything is converted
    except (archive.NamingError, ValueError) as exc:
        raise MakeEntryError(str(exc)) from exc

    simulations = [read_mdp(replica["mdp"]) if replica["mdp"] is not None else None for replica in replicas]
    # What is known about the run goes into every segment: the manifest outlives the bundle, and an
    # entry that is extended holds the trajectories of several runs.
    run = {key: metadata[key] for key in RUN_KEYS if metadata.get(key) is not None}
    sources = [
        archive_segments.SegmentSource(
            name=replica["name"],
            pdb=replica["pdb"],
            mol2=replica["mol2"],
            time_per_frame_ps=_time_per_frame(simulation, time_per_frame_ps, frame_stride),
            run=dict(run, **({"production_mdp": simulation} if simulation else {})) or None,
        )
        for replica, simulation in zip(replicas, simulations)
    ]
    claim_work_dir(work_dir, glycam_name)
    if output.is_file() or output.is_symlink():
        output.unlink()      # the bundle of an earlier run: after this run there is its bundle, or none

    timings: Dict[str, float] = {}
    step_started = time.perf_counter()
    converted = archive_segments.convert_segments(
        sources,
        work_dir / "archive",
        glycam_name,
        mol2=mol2,
        info=info,
        source_name=source_name,
        extend=extend,
        extend_time_per_frame_ps=extend_time_per_frame_ps,
        workers=workers,
        repair=repair,
        force=force,
    )
    timings["convert"] = round(time.perf_counter() - step_started, 3)
    manifest = converted["manifest"]
    entry_dir = Path(converted["entry"])

    step_started = time.perf_counter()
    result = entry_run.run_entry(
        str(entry_dir), str(work_dir / "result"),
        workers=workers, workers_from="--workers", levels=levels, force=force, verbose=verbose,
    )
    if result.get("reused") and result["input"].get("manifest_sha256") != _sha256(entry_dir / archive.MANIFEST_FILE)[1]:
        # Current for the files of the entry, but computed from an earlier manifest of them (the entry
        # was written again, for example with another time per frame): the bundle needs the two to agree.
        result = entry_run.run_entry(
            str(entry_dir), str(work_dir / "result"),
            workers=workers, workers_from="--workers", levels=levels, force=True, verbose=verbose,
        )
    timings["analysis"] = round(time.perf_counter() - step_started, 3)
    made: Dict[str, Any] = {
        "status": result["status"],
        "glycam_name": manifest["glycam_name"],
        "atoms": manifest["atoms"],
        "frames": manifest["frames"],
        "replicas": len(replicas),
        "extends": manifest.get("extends") and {
            key: manifest["extends"][key] for key in ("glycam_name", "manifest_sha256", "frames")},
        "entry": {"status": converted["status"], "path": str(entry_dir)},
        "result": {"reused": bool(result.get("reused")), "path": str(work_dir / "result")},
        "error": result.get("error"),
        "work_dir": str(work_dir),
        "workers": workers,
        "timings_sec": timings,
    }
    if result["status"] != "ok":
        timings["total"] = round(time.perf_counter() - started, 3)
        return made

    step_started = time.perf_counter()
    source = describe_source(replicas, manifest, metadata, simulations, info, workers, frame_stride)
    written = entry_bundle.write_bundle(output, entry_dir, work_dir / "result", source)
    try:
        entry_bundle.check_bundle(output, expected_glycam_name=glycam_name)   # as the receiving side reads it
    except entry_bundle.BundleError:
        output.unlink()
        raise
    timings["bundle"] = round(time.perf_counter() - step_started, 3)

    quality = result.get("frame_quality") or {}
    made.update({
        "bundle": {"path": written["path"], "size": written["size"], "sha256": written["sha256"]},
        "frames_kept": quality.get("retained_frames"),
        "clusters": {level: entry["n_clusters"] for level, entry in (result.get("analysis") or {}).get("levels", {}).items()},
        "levels": written["bundle"]["levels"],
        "simulation_length_ns": source["simulation_length_ns"],
        "warnings": list(manifest.get("warnings", [])) + source["warnings"] + list(result.get("warnings", [])),
        "peak_rss_gb": result.get("peak_rss_gb"),
        "analysis_timings_sec": result.get("timings_sec"),
    })
    if not keep_work:
        shutil.rmtree(work_dir)
        made["work_dir"] = None
        made["entry"]["path"] = made["result"]["path"] = None
    timings["total"] = round(time.perf_counter() - started, 3)
    return made


def summary_line(made: Dict[str, Any]) -> str:
    """What was made, in one line."""
    clusters = ", ".join(f"level {level}: {count}" for level, count in made.get("clusters", {}).items())
    extended = f", {made['extends']['frames']} of them from the entry that is extended" if made.get("extends") else ""
    bundle = made["bundle"]
    return (
        f"make_entry: ok: {made['glycam_name']}: {made['frames']} frames of {made['atoms']} atoms "
        f"({made['replicas']} replica(s){extended}), {made['frames_kept']} kept; clusters {clusters}; "
        f"bundle {bundle['path']} {bundle['size'] / 1e6:.1f} MB sha256 {bundle['sha256']}; "
        f"{made['timings_sec']['total']:.1f} s with {made['workers']} workers"
        + (f"; {len(made['warnings'])} warning(s), see bundle.json and archive/manifest.json" if made.get("warnings") else "")
    )


@contextlib.contextmanager
def _stdout_to_stderr():
    """Send what is written to standard output, by this process and the ones it starts, to standard error."""
    with contextlib.redirect_stdout(sys.stderr):     # print() of this process
        try:
            sys.__stdout__.flush()
            saved = os.dup(1)
        except (OSError, ValueError, AttributeError):
            yield      # no file descriptor behind standard output: nothing more to redirect
            return
        try:
            os.dup2(2, 1)                            # file descriptor 1: worker processes, C libraries
            yield
        finally:
            os.dup2(saved, 1)
            os.close(saved)


def main(argv: Optional[List[str]] = None) -> int:
    """Command line: exit status 0 bundle written, 1 failed, 2 unusable input."""
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        print("make_entry: error: --workers must be at least 1", file=sys.stderr)
        return 2
    entry_run.prepare_environment(args.workers)
    from lib import archive, entry_bundle

    metadata = {
        key: getattr(args, key) for key in (
            "simulation_length_ns", "md_package", "force_field", "temperature_k", "pressure_bar",
            "salt_concentration_mm", "comments", "glytoucan_id", "run_id",
        )
    }
    try:
        replicas = resolve_replicas(args.replicas, args.replica_map, args.replica_prefix)
        with _stdout_to_stderr():   # the analysis and its worker processes print; standard output is for the summary
            made = make_entry(
                replicas, args.output, args.glycam_name,
                workers=args.workers, levels=args.levels, info=args.info, mol2=args.mol2, extend=args.extend,
                extend_time_per_frame_ps=args.extend_time_per_frame_ps, time_per_frame_ps=args.time_per_frame_ps,
                frame_stride=args.frame_stride, metadata=metadata, source_name=args.source_name,
                work_dir=args.work_dir, keep_work=args.keep_work, force=args.force, repair=args.repair,
                verbose=args.verbose,
            )
    except (MakeEntryError, entry_run.EntryError, archive.ArchiveError) as exc:
        print(f"make_entry: error: {exc}", file=sys.stderr)
        return 2
    except entry_bundle.BundleError as exc:
        print(f"make_entry: FAILED writing the bundle: {exc}", file=sys.stderr)
        return 1
    if args.summary is not None:
        args.summary.write_text(json.dumps(made, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if made["status"] != "ok":
        error = made.get("error") or {}
        print(
            f"make_entry: FAILED in step {error.get('step')}: {error.get('message')}\n"
            f"make_entry: see {Path(made['result']['path']) / entry_run.LOG_FILE} and "
            f"{Path(made['result']['path']) / entry_run.RESULT_FILE}; no bundle was written",
            file=sys.stderr,
        )
        return 1
    print(summary_line(made))
    return 0
