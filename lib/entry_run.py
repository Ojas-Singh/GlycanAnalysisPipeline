"""Analyse one trajectory entry where it was simulated: no network, no services.

    python scripts/run_entry.py ENTRY OUTPUT [--workers N] [--levels 1,2]

``ENTRY`` is an archive entry (``lib/archive.py``: ``manifest.json``,
``trajectory.xtc``, ``topology.pdb``, ``structure.mol2``) or a folder with
``simulation.pdb`` and ``structure.mol2``. ``OUTPUT`` becomes a self-contained
result folder::

    OUTPUT/
      result.json          what was produced, from which input, with which code and settings,
                           timings per step, frame-quality summary, status
      run.log              the log of the run
      input/               manifest.json, info.json, name.json of the entry (small files only)
      process/<folder>/    the processed data, laid out as process/<glycan>/ of the pipeline
      static/              the files of static/GSxxxxx/ that need no identifier and no network
                           (lib/static_ready.py), and output/replicas.json: how well the
                           replicas of the entry agree (lib/replicas.py), which publishing
                           keeps for the administrators or publishes with the entry

Nothing here talks to PocketBase, GlyTouCan, the format services or object
storage, and nothing needs the GlycoShape ID: identifiers, metadata and upload
belong to publishing, which can either run the static stage on
``OUTPUT/process`` or complete ``OUTPUT/static``.

Only the standard library is imported here. ``main`` puts the worker count
into the environment before ``lib.config`` and the numerical libraries are
imported, because they read it at import time.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RESULT_SCHEMA = "glycoshape-entry-result"
RESULT_SCHEMA_VERSION = 1
RESULT_FILE = "result.json"
LOG_FILE = "run.log"
REPO_ROOT = Path(__file__).resolve().parent.parent
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
LEFT_TO_PUBLISHING = [
    "GlycoShape ID and the folder name static/GSxxxxx",
    "data.json (names, WURCS, GlyTouCan, composition, motifs, submission metadata)",
    "glygen.json and glycosmos.json",
    "snfg.svg",
    "the GlycoShape REMARK block (GlyTouCan ID, IUPAC name) at the top of the PDB files in static/output/level_*/",
    "GLYCOSHAPE.json, GlycoShape.zip, GLYCOSHAPE_RDF.ttl, sync_manifest.json",
    "upload",
]

logger = logging.getLogger("entry_run")


class EntryError(Exception):
    """A problem with the input or the output folder; reported, exit status 2."""


# ---------------------------------------------------------------------------
# Before lib.config is imported
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_entry.py",
        description="Process stage and static-ready files of one trajectory entry, without network access.",
    )
    parser.add_argument("entry", help="archive entry folder, or a folder with simulation.pdb and structure.mol2")
    parser.add_argument("output", help="result folder (created; an earlier result of this tool in it is replaced)")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="cores to use; default SLURM_CPUS_PER_TASK, then GLYCOSHAPE_MAX_WORKERS, "
             "then the cores this process may run on",
    )
    parser.add_argument("--levels", default=None, help="clustering levels: 1,2 (default) or 1,2,3")
    parser.add_argument("--force", action="store_true", help="recompute even when OUTPUT holds a current result")
    parser.add_argument(
        "--no-verify-input", action="store_true",
        help="do not re-hash the files of an archive entry against its manifest",
    )
    parser.add_argument("--verbose", action="store_true", help="print the full log, not only warnings and the summary")
    return parser


def resolve_workers(option: Optional[int], environ=None) -> Tuple[int, str]:
    """Worker count and where it came from."""
    environ = os.environ if environ is None else environ
    if option is not None:
        if option < 1:
            raise EntryError("--workers must be at least 1")
        return int(option), "--workers"
    for variable in ("SLURM_CPUS_PER_TASK", "GLYCOSHAPE_MAX_WORKERS"):
        raw = str(environ.get(variable, "")).strip()
        if raw:
            try:
                value = int(raw)
            except ValueError:
                raise EntryError(f"{variable}={raw!r} is not a number") from None
            if value < 1:
                raise EntryError(f"{variable}={raw!r} must be at least 1")
            return value, variable
    try:
        return max(1, len(os.sched_getaffinity(0))), "cores available to the process"
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1), "cores of the machine"


def prepare_environment(workers: int) -> None:
    """Settings that lib.config and the numerical libraries read when imported.

    Local storage only and no PocketBase, whatever the environment or a .env
    file of the checkout says; one thread-pool size for the whole run.
    """
    os.environ["GLYCOSHAPE_MAX_WORKERS"] = str(workers)
    for variable in THREAD_VARIABLES:
        os.environ[variable] = str(workers)
    os.environ["GLYCOSHAPE_ORACLE_PAR_URL"] = ""
    os.environ["POCKETBASE_URL"] = ""
    os.environ.setdefault("MPLBACKEND", "Agg")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _file_records(root: Path) -> List[Dict[str, Any]]:
    records = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        size, digest = _sha256(path)
        records.append({"path": path.relative_to(root).as_posix(), "size": size, "sha256": digest})
    return records


def pipeline_version() -> Dict[str, Any]:
    """Identity of the code that ran: the git commit when available, and a hash of the sources."""
    digest = hashlib.sha256()
    for path in sorted([REPO_ROOT / "main.py", *(REPO_ROOT / "lib").glob("*.py")]):
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    commit: Optional[str] = None
    dirty: Optional[bool] = None
    try:
        head = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=20, check=False,
        )
        if head.returncode == 0 and head.stdout.strip():
            commit = head.stdout.strip()
            status = subprocess.run(
                ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
                capture_output=True, text=True, timeout=60, check=False,
            )
            if status.returncode == 0:
                dirty = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    packages: Dict[str, Optional[str]] = {}
    from importlib import metadata

    for name in ("numpy", "scipy", "scikit-learn", "polars", "pandas", "pyarrow", "zarr", "numcodecs",
                 "joblib", "matplotlib", "seaborn", "mdtraj", "rdkit", "networkx"):
        try:
            packages[name] = metadata.version(name)
        except Exception:
            packages[name] = None
    return {
        "commit": commit,
        "uncommitted_changes": dirty,
        "code_sha256": digest.hexdigest(),
        "code_sha256_of": "main.py and lib/*.py in path order: path, zero byte, content, zero byte",
        "python": platform.python_version(),
        "packages": packages,
    }


def describe_input(entry_dir: Path, verify: bool = True) -> Dict[str, Any]:
    """Kind, files and hashes of the input; an archive entry is checked against its manifest."""
    from lib import archive

    if not entry_dir.is_dir():
        raise EntryError(f"input folder does not exist: {entry_dir}")
    if archive.is_archive_entry(entry_dir):
        try:
            manifest = archive.load_manifest(entry_dir)
        except archive.ArchiveError as exc:
            raise EntryError(f"archive entry {entry_dir}: {exc}") from exc
        files = {}
        for name, expected in manifest.get("files", {}).items():
            path = entry_dir / name
            if not path.is_file():
                raise EntryError(f"archive entry {entry_dir}: {name} is listed in the manifest and missing")
            record = {"size": int(expected["size"]), "sha256": str(expected["sha256"])}
            if path.stat().st_size != record["size"]:
                raise EntryError(
                    f"archive entry {entry_dir}: {name} has {path.stat().st_size} bytes, "
                    f"the manifest says {record['size']}"
                )
            if verify and _sha256(path)[1] != record["sha256"]:
                raise EntryError(f"archive entry {entry_dir}: {name} does not have the SHA-256 of the manifest")
            files[name] = record
        for required in (archive.TRAJECTORY_FILE, archive.TOPOLOGY_FILE, archive.MOL2_FILE):
            if required not in files:
                raise EntryError(f"archive entry {entry_dir}: the manifest does not list {required}")
        source = manifest.get("source", {}) if isinstance(manifest.get("source"), dict) else {}
        return {
            "kind": "archive",
            "path": str(entry_dir),
            "glycam_name": manifest["glycam_name"],
            "atoms": int(manifest["atoms"]),
            "frames": int(manifest["frames"]),
            "files": files,
            "files_verified_against_manifest": bool(verify),
            "manifest_sha256": _sha256(entry_dir / archive.MANIFEST_FILE)[1],
            "coordinates_sha256": manifest.get("trajectory", {}).get("coordinates_sha256"),
            "source": {key: source.get(key) for key in ("folder", "file", "size", "sha256")},
            "converter": manifest.get("converter"),
        }

    try:
        inputs = archive.source_inputs(entry_dir)
    except archive.ArchiveError as exc:
        raise EntryError(
            f"{entry_dir} is neither an archive entry (manifest.json and trajectory.xtc) "
            f"nor a simulation.pdb folder: {exc}"
        ) from exc
    try:
        glycam_name, origin, _ = archive.resolve_glycam_name(entry_dir)
    except archive.ArchiveError as exc:
        raise EntryError(f"{entry_dir}: {exc}") from exc
    files = {}
    for key in ("pdb", "mol2", "info", "name"):
        path = inputs.get(key)
        if path is not None:
            size, digest = _sha256(path)
            files[path.name] = {"size": size, "sha256": digest}
    return {
        "kind": "pdb",
        "path": str(entry_dir),
        "glycam_name": glycam_name,
        "glycam_name_from": origin,
        "atoms": None,
        "frames": None,
        "files": files,
    }


def _previous_result(output_dir: Path) -> Optional[Dict[str, Any]]:
    """None without result.json, {} for a file that is not ours, else the earlier result."""
    path = output_dir / RESULT_FILE
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) and data.get("schema") == RESULT_SCHEMA else {}


def claim_output(output_dir: Path) -> Optional[Dict[str, Any]]:
    """The output folder must be new, empty, or hold an earlier result of this tool.

    Returns the earlier result, if any. Nothing is created here: the folder
    is made once the input has been accepted.
    """
    if output_dir.exists() and not output_dir.is_dir():
        raise EntryError(f"output path is not a folder: {output_dir}")
    previous = _previous_result(output_dir)
    if previous == {}:
        raise EntryError(f"{output_dir / RESULT_FILE} exists and is not a result of this tool")
    if output_dir.is_dir() and previous is None and any(output_dir.iterdir()):
        raise EntryError(
            f"output folder {output_dir} is not empty and holds no {RESULT_FILE} of this tool; "
            "choose a new folder (process/, static/ and input/ in it would be replaced)"
        )
    return previous


def _fingerprint(input_info: Dict[str, Any], settings: Dict[str, Any], version: Dict[str, Any]) -> Dict[str, Any]:
    """What must be equal for an existing result to count as current."""
    return {
        "input_files": input_info["files"],
        "cluster_levels": settings["cluster_levels"],
        "max_covalent_bond_angstrom": settings["max_covalent_bond_angstrom"],
        "code_sha256": version["code_sha256"],
    }


def _analysis_summary(process_dir: Path) -> Dict[str, Any]:
    info = json.loads((process_dir / "output" / "info.json").read_text(encoding="utf-8"))
    import pyarrow.parquet as pq

    table = pq.read_table(process_dir / "data" / "frame_map.parquet").to_pydict()
    source_frame = dict(zip(table["frame"], table["source_frame"]))
    levels = {}
    for level in info.get("levels", []):
        clusters = level.get("clusters", [])
        frames = [cluster.get("representative_idx") for cluster in clusters]
        levels[str(level["level"])] = {
            "n_clusters": level.get("n_clusters"),
            "cluster_size_pct": [cluster.get("cluster_size_pct") for cluster in clusters],
            "representative_frames": frames,
            "representative_source_frames": [
                None if frame is None else source_frame.get(int(frame)) for frame in frames
            ],
        }
    torsions = info.get("torsions", {}) if isinstance(info.get("torsions"), dict) else {}
    pca = info.get("pca", {}) if isinstance(info.get("pca"), dict) else {}
    return {
        "entropy_nats": info.get("entropy_nats"),
        "levels": levels,
        "torsions": {
            "count": torsions.get("count"),
            "glycosidic": len(torsions.get("glycosidic", []) or []),
        },
        "pca_explained_variance_ratio": (pca.get("pca_analysis") or {}).get("explained_variance_ratio"),
        "silhouette_scores": (pca.get("main_clustering") or {}).get("silhouette_scores"),
    }


def write_replicas(entry_dir: Path, input_info: Dict[str, Any], process_dir: Path, static_dir: Path,
                   glycam_name: str) -> Optional[str]:
    """``static/output/replicas.json``: how well the replicas of the entry agree (``lib/replicas.py``).

    The replicas are those of the manifest of an archive entry (its segments, or the runs its source
    recorded); a ``simulation.pdb`` folder has no record of them.  Diagnostics never stop a run: a
    problem is returned as a warning and the file is not written.
    """
    from lib import archive, replicas

    try:
        manifest = archive.load_manifest(entry_dir) if input_info.get("kind") == "archive" else None
        quality = json.loads((process_dir / "data" / "frame_filter.json").read_text(encoding="utf-8"))
        frames = int(manifest["frames"]) if manifest is not None else int(quality["source_frames"])
        document = replicas.diagnose_result(process_dir, manifest, frames, glycam_name=glycam_name)
        replicas.write(document, static_dir / replicas.STATIC_PATH)
        verdict = document["verdict"]
        logger.info("replicas: %d (%s), verdict %s%s", document["boundaries"]["replicas"],
                    document["boundaries"]["basis"], verdict["value"],
                    "".join(f"; {reason['text']}" for reason in verdict["reasons"]))
        return None
    except Exception as exc:
        logger.warning("replica diagnostics were not written: %s: %s", type(exc).__name__, exc, exc_info=True)
        partial = static_dir / replicas.STATIC_PATH
        if partial.exists():
            partial.unlink()
        return f"replica diagnostics were not written ({type(exc).__name__}: {exc})"


def _configure_logging(log_path: Path, verbose: bool):
    """Log to OUTPUT/run.log and to stderr; returns a function that undoes it."""
    root = logging.getLogger()
    previous_handlers = list(root.handlers)
    previous_level = root.level
    for handler in previous_handlers:
        root.removeHandler(handler)
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    console.setLevel(logging.INFO if verbose else logging.WARNING)
    root.addHandler(file_handler)
    root.addHandler(console)

    def restore() -> None:
        for handler in (file_handler, console):
            root.removeHandler(handler)
        file_handler.close()
        for handler in previous_handlers:
            root.addHandler(handler)
        root.setLevel(previous_level)

    return restore


class _Terminated(BaseException):
    pass


def _raise_terminated(signum, _frame):
    raise _Terminated(f"terminated by signal {signum}")


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
def run_entry(
    entry: str,
    output: str,
    *,
    workers: int,
    workers_from: str = "caller",
    levels: Optional[str] = None,
    force: bool = False,
    verify_input: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run the process stage and the static-ready export for one entry; returns the result.

    The result is also ``OUTPUT/result.json``: status ``"ok"`` or ``"failed"``
    at the end, ``"running"`` while the run is in progress, so that a job
    that is killed leaves a folder that says so. A current result (same input
    files, levels, bond threshold and code) is returned as it is unless
    ``force`` is set; anything else in ``OUTPUT`` from an earlier run is
    replaced, not completed.
    """
    started = time.perf_counter()
    started_at = _utc_now()
    entry_dir = Path(entry).resolve()
    output_dir = Path(output).resolve()
    previous = claim_output(output_dir)

    import lib.config as config

    if levels:
        try:
            config.cluster_levels = config.parse_cluster_levels(levels)
        except ValueError as exc:
            raise EntryError(str(exc)) from exc
    # lib.config read these at import; a caller that imported it earlier gets them here.
    config.max_workers = int(workers)
    config.use_oracle_storage = False
    config.oracle_par_url = None
    config.use_pocketbase = False
    from lib.storage import reset_storage_manager

    reset_storage_manager()

    timings: Dict[str, float] = {}
    step_started = time.perf_counter()
    input_info = describe_input(entry_dir, verify=verify_input)
    timings["input"] = round(time.perf_counter() - step_started, 3)
    version = pipeline_version()
    settings = {
        "cluster_levels": [int(level) for level in config.cluster_levels],
        "workers": int(workers),
        "workers_from": workers_from,
        "threads": {variable: os.environ.get(variable) for variable in THREAD_VARIABLES},
        "max_covalent_bond_angstrom": float(config.max_covalent_bond_angstrom),
        "frame_quality_block_frames": int(config.frame_quality_block_frames),
        "store_frames_buffer": int(config.store_frames_buffer),
        "store_frame_chunk": int(config.store_frame_chunk),
    }
    folder = entry_dir.name
    process_root = output_dir / "process"
    process_dir = process_root / folder
    static_dir = output_dir / "static"
    result: Dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "running",
        "error": None,
        "warnings": [],
        "glycam_name": input_info["glycam_name"],
        "folder": folder,
        "started_at": started_at,
        "finished_at": None,
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
        "input": input_info,
        "pipeline": version,
        "settings": settings,
        "fingerprint": _fingerprint(input_info, settings, version),
        "timings_sec": timings,
        "frame_quality": None,
        "analysis": None,
        "produced": None,
        "left_to_publishing": LEFT_TO_PUBLISHING,
    }

    if (
        not force
        and previous
        and previous.get("status") == "ok"
        and previous.get("fingerprint") == result["fingerprint"]
        and process_dir.is_dir()
        and static_dir.is_dir()
    ):
        previous["reused"] = True
        return previous

    # Everything below is recomputed from the input: an unfinished or
    # outdated result is removed rather than mixed with the new one.
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in (process_root, static_dir, output_dir / "input"):
        if stale.exists():
            shutil.rmtree(stale)
    _write_json(output_dir / RESULT_FILE, result)
    restore_logging = _configure_logging(output_dir / LOG_FILE, verbose)

    previous_handler = None
    try:
        previous_handler = signal.signal(signal.SIGTERM, _raise_terminated)
    except ValueError:  # not in the main thread
        pass
    failed_step = "setup"
    try:
        logger.info(
            "entry %s -> %s (%d workers, levels %s)", entry_dir, output_dir, workers, settings["cluster_levels"]
        )
        (output_dir / "input").mkdir()
        for name in ("manifest.json", "info.json", "name.json"):
            if (entry_dir / name).is_file():
                shutil.copy2(entry_dir / name, output_dir / "input" / name)

        from lib.v2 import GlycanAnalysisPipeline
        from lib import static_ready

        failed_step = "process"
        pipeline = GlycanAnalysisPipeline(str(entry_dir.parent), str(process_root))
        analysis = pipeline.run_analysis(folder)
        timings.update({step: float(seconds) for step, seconds in analysis.get("step_durations_sec", {}).items()})
        if not analysis.get("success"):
            failed = analysis.get("steps_failed") or ["process"]
            failed_step = str(failed[0])
            raise RuntimeError(analysis.get("error_message") or "the process stage failed")
        glycam_name = analysis["paths"]["glycam_name"]
        if glycam_name != input_info["glycam_name"]:
            result["warnings"].append(
                f"the analysis used the GLYCAM name {glycam_name!r}; the input describes itself as "
                f"{input_info['glycam_name']!r}"
            )
        result["glycam_name"] = glycam_name

        failed_step = "static_ready"
        step_started = time.perf_counter()
        written = static_ready.write_static_ready(process_dir, static_dir, glycam_name)
        timings["static_ready"] = round(time.perf_counter() - step_started, 3)
        missing = [
            f"output/{level}/{name}"
            for level in static_ready.level_names()
            for name in ("dist.svg", "GLYCAM/alpha.pdb", "GLYCAM/beta.pdb", "PDB/alpha.pdb", "PDB/beta.pdb")
            if not (static_dir / "output" / level / name).is_file()
        ]
        missing += [name for name in ("output/info.json", "output/pca.csv") if not (static_dir / name).is_file()]
        if missing:
            raise RuntimeError("static-ready files were not written: " + ", ".join(missing))

        failed_step = "replicas"
        step_started = time.perf_counter()
        problem = write_replicas(entry_dir, input_info, process_dir, static_dir, glycam_name)
        if problem:
            result["warnings"].append(problem)
        timings["replicas"] = round(time.perf_counter() - step_started, 3)

        failed_step = "result"
        step_started = time.perf_counter()
        frame_quality = json.loads((process_dir / "data" / "frame_filter.json").read_text(encoding="utf-8"))
        if input_info["kind"] == "pdb":
            import pyarrow.parquet as pq

            input_info["frames"] = int(frame_quality["source_frames"])
            input_info["atoms"] = int(pq.read_metadata(process_dir / "data" / "atoms.parquet").num_rows)
        result["frame_quality"] = frame_quality
        result["analysis"] = _analysis_summary(process_dir)
        result["produced"] = {
            "levels": written["levels"],
            "process": {"path": f"process/{folder}", "files": _file_records(process_dir)},
            "static": {"path": "static", "files": _file_records(static_dir)},
        }
        timings["result"] = round(time.perf_counter() - step_started, 3)
        result["status"] = "ok"
    except BaseException as exc:
        logger.error("%s: %s", failed_step, exc, exc_info=isinstance(exc, Exception))
        result["status"] = "failed"
        result["error"] = {"step": failed_step, "type": type(exc).__name__, "message": str(exc)}
        if not isinstance(exc, (Exception, _Terminated)):
            raise
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGTERM, previous_handler)
        try:
            import resource

            result["peak_rss_gb"] = {
                "main_process": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 3),
                "largest_finished_child": round(
                    resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1048576, 3
                ),
            }
        except Exception:
            pass
        timings["total"] = round(time.perf_counter() - started, 3)
        result["finished_at"] = _utc_now()
        _write_json(output_dir / RESULT_FILE, result)
        restore_logging()
    return result


def main(argv: Optional[List[str]] = None) -> int:
    """Command line of the one-entry mode: exit status 0 ok, 1 failed, 2 unusable input or output."""
    args = build_parser().parse_args(argv)
    try:
        workers, workers_from = resolve_workers(args.workers)
        prepare_environment(workers)
        result = run_entry(
            args.entry,
            args.output,
            workers=workers,
            workers_from=workers_from,
            levels=args.levels,
            force=args.force,
            verify_input=not args.no_verify_input,
            verbose=args.verbose,
        )
    except EntryError as exc:
        print(f"run_entry: error: {exc}", file=sys.stderr)
        return 2
    output = Path(args.output).resolve()
    if result["status"] != "ok":
        error = result.get("error") or {}
        print(
            f"run_entry: FAILED in step {error.get('step')}: {error.get('message')}\n"
            f"run_entry: see {output / LOG_FILE} and {output / RESULT_FILE}",
            file=sys.stderr,
        )
        return 1
    quality = result.get("frame_quality") or {}
    levels = (result.get("analysis") or {}).get("levels", {})
    clusters = ", ".join(f"level {level}: {entry['n_clusters']}" for level, entry in levels.items())
    steps = ", ".join(f"{step} {seconds:.1f}" for step, seconds in result["timings_sec"].items() if step != "total")
    print(
        f"run_entry: ok{' (current result reused)' if result.get('reused') else ''}: {result['glycam_name']}\n"
        f"run_entry: frames kept {quality.get('retained_frames')}/{quality.get('source_frames')}; clusters {clusters}\n"
        f"run_entry: {result['timings_sec'].get('total', 0):.1f} s with {result['settings']['workers']} workers ({steps})\n"
        f"run_entry: result in {output}"
    )
    return 0
