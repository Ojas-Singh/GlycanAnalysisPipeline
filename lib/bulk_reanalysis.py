"""Re-analyse every entry of the raw archive into a new static database, check it, publish it.

The published database is derived from the raw archive, the code and the
curated metadata of each GlycoShape ID.  This is the run that derives all of
it again with the current code (``docs/bulk-reanalysis.md`` is the runbook).
Stages, each resumable and each with a summary, in a *run folder*::

    plan       which archive entry is which published ID, the cost of each, the shards
    analyse    scripts/run_entry.py for every entry of a shard (compute nodes, no network)
    complete   the published folder of each entry, with the curated metadata of the published one
    finalize   the root files: GLYCOSHAPE.json, GlycoShape.zip, GLYCOSHAPE_RDF.ttl, the missing_* reports,
               sync_manifest.json last
    compare    the new tree against the published one
    check-base whether the stored database is still the one the run was derived from (reads the storage)
    publish    the tree in one publication, under the lock of the incremental publisher

The run folder::

    plan.json
    entries/<folder>/result/           the result folder of run_entry.py (process/ removed after success)
    entries/<folder>/analysis.log      what run_entry.py printed
    entries/<folder>/status.json       analysis: ok | failed | skipped, seconds, error
    entries/<folder>/complete.json     completion: ok | failed, the folder hash, lookups, problems
    tree/                              the new static database (Levels 1+2)
    replicas/GSxxxxx.json              the replica diagnostics, for the administrators' copies
    logs/<stage>[-<shard>].json        the summary of each stage
    compare.json

Nothing here writes to the published tree, the raw archive or the storage,
except ``publish``.
"""
from __future__ import annotations

import concurrent.futures
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

PLAN_SCHEMA = "glycoshape-bulk-reanalysis-plan"
SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[1]
#: Seconds of one entry with 16 workers on a 64-core AMD node of Iridis X: 30 s plus 0.053 s per million
#: (frames x landscape atom pairs).  Fitted to job 1797685: 35 s (50,005 frames, 1,891 pairs), 40 s
#: (40,004, about 5,000) and 154 s (159,940, 14,706).
SECONDS_BASE, SECONDS_PER_MEGA = 30.0, 0.053
REGENERATED_ROOT_FILES = ("GLYCOSHAPE.json", "GlycoShape.zip", "GLYCOSHAPE_RDF.ttl", "missing_glytoucan.txt",
                          "missing_wurcs.txt", "sync_manifest.json")

Log = Callable[[str], None]


class BulkError(Exception):
    """A stage cannot run (exit status 2)."""


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def estimated_seconds(frames: int, landscape_atoms: int) -> float:
    pairs = landscape_atoms * (landscape_atoms - 1) // 2
    return round(SECONDS_BASE + SECONDS_PER_MEGA * frames * pairs / 1e6, 1)


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
def make_plan(archive_dir: Path, published_dir: Path, run_dir: Path, shards: int = 4,
              fixes: Optional[Dict[str, Any]] = None, rename_to_raw_names: bool = False,
              workers: int = 16, log: Log = print) -> Dict[str, Any]:
    """``plan.json``: every archive entry with its published ID, cost, memory and shard.

    ``fixes`` maps a GlycoShape ID (or a GLYCAM name) to ``{"exclude":
    [replica names]}``: replicas to leave out of the analysis.  Entries whose
    fix needs a new name (``simulated_anomer``, ``glycam_name``) are not done
    here: they are ``reanalyse`` / ``update-metadata`` of one entry
    afterwards, and the plan lists them.  An entry is completed under the
    name it is published under, unless ``rename_to_raw_names`` (the entries
    published under a GlyTouCan accession then get their GLYCAM name).
    """
    from lib import archive, entry_ops, id_registry

    archive_dir, published_dir, run_dir = Path(archive_dir), Path(published_dir), Path(run_dir)
    index = _read_json(archive_dir / archive.INDEX_FILE)
    if not isinstance(index, dict) or not isinstance(index.get("records"), list):
        raise BulkError(f"{archive_dir / archive.INDEX_FILE} is not the index of a raw archive")
    manifest = _read_json(published_dir / "sync_manifest.json")
    inventory = _read_json(published_dir / "GLYCOSHAPE.json")
    if not isinstance(manifest, dict) or not isinstance(inventory, dict):
        raise BulkError(f"{published_dir} is not a published static database (sync_manifest.json, GLYCOSHAPE.json)")
    aliases = id_registry.names_from_archive_index(index)
    by_name: Dict[str, str] = {}
    published_names: Dict[str, str] = {}
    for glycan_id, data in inventory.items():
        name = id_registry.full_glycam_name(data) or ""
        published_names[glycan_id] = name
        by_name[aliases.get(name, name)] = glycan_id
    fixes = dict(fixes or {})
    entries: List[Dict[str, Any]] = []
    for record in sorted(index["records"], key=lambda item: item["glycam_name"]):
        name = record["glycam_name"]
        folder = archive_dir / record["folder"]
        glycan_id = by_name.get(name)
        fix = fixes.pop(glycan_id, None) if glycan_id in fixes else fixes.pop(name, None)
        try:
            atoms = entry_ops.heavy_atoms(folder / archive.TOPOLOGY_FILE)
        except OSError:
            atoms = min(int(record.get("atoms") or 0), 300)
        frames = int(record.get("frames") or 0)
        row = {
            "glycam_name": name, "folder": record["folder"], "glycoshape_id": glycan_id,
            "published_name": published_names.get(glycan_id) if glycan_id else None,
            "complete_as": None, "frames": frames, "atoms": record.get("atoms"), "landscape_atoms": atoms,
            "seconds": estimated_seconds(frames, atoms),
            "memory_gb": entry_ops.memory_estimate_gb(frames, atoms, workers),
            "exclude": list((fix or {}).get("exclude") or []), "skip": None, "shard": None,
        }
        if glycan_id is None:
            row["skip"] = "not published: publish it with publish_entry.py and an ID from the registry"
        elif fix and (fix.get("simulated_anomer") or fix.get("glycam_name")):
            row["skip"] = "the fix renames the entry: publish_entry.py reanalyse / update-metadata after the bulk run"
            row["fix"] = fix
        else:
            row["complete_as"] = name if rename_to_raw_names else row["published_name"]
        entries.append(row)
    planned_ids = {row["glycoshape_id"] for row in entries if row["glycoshape_id"]}
    without_raw = sorted(set(inventory) - planned_ids)
    # Shards of equal estimated time: the longest entries first, each to the shard with the least so far.
    loads = [0.0] * max(1, int(shards))
    for row in sorted((row for row in entries if not row["skip"]), key=lambda item: -item["seconds"]):
        shard = min(range(len(loads)), key=lambda number: loads[number])
        row["shard"] = shard + 1
        loads[shard] += row["seconds"]
    plan = {
        "schema": PLAN_SCHEMA, "schema_version": SCHEMA_VERSION, "created": _utc_now(),
        "archive_dir": str(archive_dir.resolve()), "published_dir": str(published_dir.resolve()),
        "base_global_hash": manifest.get("global_hash"), "published_glycans": len(inventory),
        "archive_index_sha256": _sha256_file(archive_dir / archive.INDEX_FILE),
        "shards": len(loads), "workers": workers, "rename_to_raw_names": rename_to_raw_names,
        "estimate": {
            "entries": sum(1 for row in entries if not row["skip"]),
            "entry_seconds": round(sum(row["seconds"] for row in entries if not row["skip"]), 1),
            "shard_seconds": [round(load, 1) for load in loads],
            "largest_memory_gb": max((row["memory_gb"] for row in entries if not row["skip"]), default=0),
            "model": f"{SECONDS_BASE} s + {SECONDS_PER_MEGA} s per million frames x landscape atom pairs, "
                     "16 workers, one entry at a time on a 64-core node",
        },
        "published_without_raw_entry": without_raw,
        "unused_fixes": sorted(fixes),
        "entries": entries,
    }
    _write_json(run_dir / "plan.json", plan)
    log(f"plan: {plan['estimate']['entries']} entries to analyse, {len(entries) - plan['estimate']['entries']} skipped, "
        f"{len(without_raw)} published IDs without a raw entry; about {plan['estimate']['entry_seconds'] / 3600:.1f} "
        f"entry-hours; shards {', '.join(f'{seconds / 3600:.1f} h' for seconds in loads)} (one entry at a time)")
    return plan


def load_plan(run_dir: Path) -> Dict[str, Any]:
    plan = _read_json(Path(run_dir) / "plan.json")
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise BulkError(f"{Path(run_dir) / 'plan.json'} is not a plan of this tool: run plan first")
    return plan


# ---------------------------------------------------------------------------
# analyse
# ---------------------------------------------------------------------------
def _code_sha256() -> str:
    from lib.entry_run import pipeline_version

    return pipeline_version()["code_sha256"]


def analyse_entry(row: Dict[str, Any], plan: Dict[str, Any], run_dir: Path, workers: int, keep_process: bool,
                  code_sha256: str, force: bool = False) -> Dict[str, Any]:
    """``run_entry.py`` for one entry, in a process of its own; returns ``status.json``."""
    entry_dir = Path(plan["archive_dir"]) / row["folder"]
    folder = Path(run_dir) / "entries" / row["folder"]
    status_path = folder / "status.json"
    manifest_sha256 = _sha256_file(entry_dir / "manifest.json")
    fingerprint = {"manifest_sha256": manifest_sha256, "code_sha256": code_sha256, "exclude": row["exclude"]}
    previous = _read_json(status_path, {})
    if not force and previous.get("status") == "ok" and previous.get("fingerprint") == fingerprint \
            and (folder / "result" / "result.json").is_file():
        return {**previous, "resumed": True}
    folder.mkdir(parents=True, exist_ok=True)
    status = {"glycam_name": row["glycam_name"], "glycoshape_id": row["glycoshape_id"], "status": "running",
              "started": _utc_now(), "host": os.uname().nodename, "workers": workers, "fingerprint": fingerprint,
              "slurm_job": os.environ.get("SLURM_JOB_ID"), "slurm_task": os.environ.get("SLURM_ARRAY_TASK_ID")}
    _write_json(status_path, status)
    command = [sys.executable, str(REPO_ROOT / "scripts" / "run_entry.py"), str(entry_dir), str(folder / "result"),
               "--workers", str(workers)]
    if force:
        command.append("--force")
    for name in row["exclude"]:
        command += ["--exclude-replica", name]
    started = time.perf_counter()
    with open(folder / "analysis.log", "wb") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False,
                                   env=dict(os.environ, MPLBACKEND="Agg", GLYCOSHAPE_STORAGE="local"))
    status["seconds"] = round(time.perf_counter() - started, 1)
    result = _read_json(folder / "result" / "result.json", {})
    status["exit"] = completed.returncode
    status["peak_rss_gb"] = result.get("peak_rss_gb")
    if completed.returncode == 0 and result.get("status") == "ok":
        status["status"] = "ok"
        status["frames"] = (result.get("frame_quality") or {}).get("source_frames")
        status["retained"] = (result.get("frame_quality") or {}).get("retained_frames")
        if not keep_process and (folder / "result" / "process").is_dir():
            shutil.rmtree(folder / "result" / "process")       # 10x the rest; the static-ready half is what is kept
    else:
        status["status"] = "failed"
        error = result.get("error") or {}
        tail = (folder / "analysis.log").read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:]
        status["error"] = error.get("message") or (f"killed by signal {-completed.returncode} (out of memory?)"
                                                    if completed.returncode < 0 else " | ".join(tail))
        status["step"] = error.get("step")
    status["finished"] = _utc_now()
    _write_json(status_path, status)
    return status


def analyse(run_dir: Path, shard: int = 1, shards: Optional[int] = None, jobs: int = 4, workers: int = 16,
            memory_gb: float = 230.0, keep_process: bool = False, only: Iterable[str] = (), force: bool = False,
            archive_dir: Optional[Path] = None, log: Log = print) -> Dict[str, Any]:
    """Analyse the entries of one shard, ``jobs`` at a time with ``workers`` cores each, within ``memory_gb``.

    The longest entries go first.  An entry whose estimated memory does not
    fit next to the ones running waits; one that needs more than
    ``memory_gb`` alone runs alone.  Entries analysed before with the same
    raw manifest, code and exclusions are skipped (resume).  A failure is
    recorded in the ``status.json`` of the entry with its reason, and the
    other entries go on.  ``archive_dir`` is where the archive of the plan
    is on this machine (the plan may have been made on another one; its
    ``index.json`` has to be the one the plan was made from).
    """
    run_dir = Path(run_dir)
    plan = load_plan(run_dir)
    if archive_dir is not None:
        if _sha256_file(Path(archive_dir) / "index.json") != plan["archive_index_sha256"]:
            raise BulkError(f"{archive_dir} is not the archive the plan was made from (its index.json differs)")
        plan = {**plan, "archive_dir": str(Path(archive_dir).resolve())}
    if shards is not None and shards != plan["shards"]:
        raise BulkError(f"the plan has {plan['shards']} shards, not {shards}: plan again with --shards {shards}")
    only = set(only)
    rows = [row for row in plan["entries"] if not row["skip"]
            and (row["shard"] == shard if not only else (row["glycam_name"] in only or row["glycoshape_id"] in only))]
    rows.sort(key=lambda item: -item["seconds"])
    code_sha256 = _code_sha256()
    started = time.perf_counter()
    lock = threading.Condition()
    running: Dict[str, float] = {}
    results: List[Dict[str, Any]] = []

    def run(row: Dict[str, Any]) -> Dict[str, Any]:
        need = min(float(row["memory_gb"]), memory_gb)
        with lock:
            while running and sum(running.values()) + need > memory_gb:
                lock.wait()
            running[row["folder"]] = need
        try:
            status = analyse_entry(row, plan, run_dir, workers, keep_process, code_sha256, force=force)
        except Exception as exc:                                    # the driver goes on with the other entries
            status = {"glycam_name": row["glycam_name"], "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        finally:
            with lock:
                running.pop(row["folder"], None)
                lock.notify_all()
        word = "resumed" if status.get("resumed") else status["status"]
        log(f"{word:8s} {row['glycoshape_id'] or '-':8s} {status.get('seconds', 0):>7} s  {row['glycam_name'][:90]}"
            + (f"  ERROR {status.get('error')}" if status["status"] != "ok" else ""))
        return status

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        for status in pool.map(run, rows):
            results.append(status)
    summary = {
        "stage": "analyse", "shard": shard, "shards": plan["shards"], "finished": _utc_now(),
        "seconds": round(time.perf_counter() - started, 1), "jobs": jobs, "workers": workers,
        "entries": len(rows), "ok": sum(1 for item in results if item["status"] == "ok"),
        "resumed": sum(1 for item in results if item.get("resumed")),
        "failed": [{"glycam_name": item["glycam_name"], "error": item.get("error"), "step": item.get("step")}
                   for item in results if item["status"] != "ok"],
    }
    _write_json(run_dir / "logs" / (f"analyse-{shard}.json" if not only else "analyse-selected.json"), summary)
    log(f"analyse shard {shard}/{plan['shards']}: {summary['ok']}/{summary['entries']} ok "
        f"({summary['resumed']} resumed), {len(summary['failed'])} failed, {summary['seconds']} s")
    return summary


# ---------------------------------------------------------------------------
# complete
# ---------------------------------------------------------------------------
def _complete_one(task: Dict[str, Any]) -> Dict[str, Any]:
    """The published folder of one analysed entry (in a worker process)."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    from lib import entry_complete, entry_ops, replicas, static_delta

    row, run_dir, published_dir = task["row"], Path(task["run_dir"]), Path(task["published_dir"])
    folder = run_dir / "entries" / row["folder"]
    record_path = folder / "complete.json"
    glycan_id = row["glycoshape_id"]
    out = {"glycoshape_id": glycan_id, "glycam_name": row["glycam_name"], "status": "failed"}
    try:
        result_sha256 = _sha256_file(folder / "result" / "result.json")
        data = json.loads((published_dir / glycan_id / "data.json").read_text(encoding="utf-8"))
        curation_record = _read_json(Path(task["curation_dir"]) / f"{glycan_id}.json") if task.get("curation_dir") else None
        submission_record, curation, notes = entry_ops.published_submission(data, row["complete_as"], curation_record)
        submission_record["glycoshape_id"] = glycan_id
        submission = entry_complete.Submission.parse(submission_record)
        fingerprint = hashlib.sha256(json.dumps(
            [result_sha256, submission.sha256(), curation, row["complete_as"], task["levels"], task["offline"]],
            sort_keys=True).encode()).hexdigest()
        tree = run_dir / "tree"
        previous = _read_json(record_path, {})
        if previous.get("status") == "ok" and previous.get("fingerprint") == fingerprint and (tree / glycan_id).is_dir() \
                and static_delta.folder_entry(tree / glycan_id)["hash"] == previous.get("hash") and not task["refresh"]:
            return {**previous, "resumed": True}
        staging = run_dir / "tree.staging"
        if (staging / glycan_id).exists():
            shutil.rmtree(staging / glycan_id)
        target_folder, completed_data, summary = entry_complete.complete_folder(
            folder / "result" / "static", row["complete_as"], tuple(task["levels"]), submission, staging,
            offline=task["offline"], lookup_timeout=task.get("lookup_timeout"), lookup_budget=task.get("lookup_budget"),
            names=curation["names"], aliases=curation.get("aliases") or [], keywords=curation.get("keywords") or {},
            public_note=curation.get("public_note") or "", salt_by_run=curation.get("salt_by_run"))
        report = entry_complete.folder_report(target_folder, completed_data, task["levels"], summary, submission,
                                              list(submission.warnings))
        diagnostics = folder / "result" / "static" / replicas.STATIC_PATH
        if diagnostics.is_file() and not replicas.public_placement():
            (run_dir / "replicas").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(diagnostics, run_dir / "replicas" / f"{glycan_id}.json")
        if (tree / glycan_id).exists():
            shutil.rmtree(tree / glycan_id)
        tree.mkdir(parents=True, exist_ok=True)
        os.rename(target_folder, tree / glycan_id)
        out.update(status="ok" if not report["problems"] else "incomplete", fingerprint=fingerprint,
                   hash=static_delta.folder_entry(tree / glycan_id)["hash"], file_count=report["file_count"],
                   total_size=report["total_size"], problems=report["problems"], gaps=report["gaps"],
                   unanswered=report["lookups"]["unanswered"], warnings=report["warnings"], curated_from=notes,
                   complete_as=row["complete_as"], submission=submission.record(), curation=curation,
                   finished=_utc_now())
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    _write_json(record_path, out)
    return out


def complete(run_dir: Path, published_dir: Optional[Path] = None, jobs: int = 4, offline: bool = False,
             curation_dir: Optional[Path] = None, lookup_timeout: Optional[float] = None,
             lookup_budget: Optional[float] = 300.0, refresh: bool = False, levels: Tuple[int, ...] = (1, 2),
             log: Log = print) -> Dict[str, Any]:
    """Complete every analysed entry into ``tree/GSxxxxx`` with the curated metadata of its published folder.

    ``curation_dir`` holds the records of the incremental publisher
    (``publish-state/static/entries/GSxxxxx.json``) of the entries that
    have one; their submission and curation take precedence over what is
    read back from ``data.json``.  The web lookups run here (``offline``
    makes none: the accessions are then missing).  ``jobs`` processes.
    """
    run_dir = Path(run_dir)
    plan = load_plan(run_dir)
    published_dir = Path(published_dir or plan["published_dir"])
    tasks = []
    waiting, analysis_failed = [], []
    for row in plan["entries"]:
        if row["skip"]:
            continue
        status = _read_json(run_dir / "entries" / row["folder"] / "status.json", {})
        if status.get("status") == "failed":            # reported by analyse; finalize keeps the published version
            analysis_failed.append(row["glycoshape_id"])
            continue
        if status.get("status") != "ok":
            waiting.append(row["glycam_name"])
            continue
        tasks.append({"row": row, "run_dir": str(run_dir), "published_dir": str(published_dir), "offline": offline,
                      "curation_dir": str(curation_dir) if curation_dir else None, "lookup_timeout": lookup_timeout,
                      "lookup_budget": lookup_budget, "refresh": refresh, "levels": list(levels)})
    started = time.perf_counter()
    results: List[Dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, jobs)) as pool:
        for out in pool.map(_complete_one, tasks):
            results.append(out)
            log(f"{('resumed' if out.get('resumed') else out['status']):10s} {out['glycoshape_id']}  "
                + (out.get("error") or "; ".join(out.get("problems") or []) or ""))
    summary = {
        "stage": "complete", "finished": _utc_now(), "seconds": round(time.perf_counter() - started, 1),
        "offline": offline, "entries": len(tasks), "not_analysed": waiting, "analysis_failed": analysis_failed,
        "ok": sum(1 for item in results if item["status"] == "ok"),
        "resumed": sum(1 for item in results if item.get("resumed")),
        "incomplete": [{"glycoshape_id": item["glycoshape_id"], "problems": item.get("problems")}
                       for item in results if item["status"] == "incomplete"],
        "failed": [{"glycoshape_id": item["glycoshape_id"], "error": item.get("error")}
                   for item in results if item["status"] == "failed"],
        "unanswered_lookups": sum(1 for item in results if item.get("unanswered")),
    }
    _write_json(run_dir / "logs" / "complete.json", summary)
    log(f"complete: {summary['ok']}/{summary['entries']} ok ({summary['resumed']} resumed), "
        f"{len(summary['incomplete'])} incomplete, {len(summary['failed'])} failed, {len(waiting)} not analysed")
    return summary


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------
def entry_states(plan: Dict[str, Any], run_dir: Path, inventory: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Per published ID: ``reanalysed``, ``kept`` (with the reason) or ``not analysed``.

    ``kept`` are the entries whose published folder stays as it is: the
    analysis failed (``status.json``), the completion failed or left the
    folder incomplete (``complete.json``), the entry has no raw entry, or
    its fix renames it.
    """
    states: Dict[str, Dict[str, Any]] = {
        glycan_id: {"state": "kept", "reason": "no raw entry in the archive that was analysed"} for glycan_id in inventory}
    for row in plan["entries"]:
        glycan_id = row.get("glycoshape_id")
        if glycan_id not in states:
            continue
        if row["skip"]:
            states[glycan_id] = {"state": "kept", "reason": row["skip"]}
            continue
        folder = Path(run_dir) / "entries" / row["folder"]
        status = _read_json(folder / "status.json", {})
        done = _read_json(folder / "complete.json", {})
        if status.get("status") == "failed":
            states[glycan_id] = {"state": "kept", "reason": f"analysis failed: {status.get('error')}",
                                 "stage": "analyse", "step": status.get("step")}
        elif status.get("status") != "ok":
            states[glycan_id] = {"state": "not analysed", "reason": "not analysed"}
        elif done.get("status") == "ok":
            states[glycan_id] = {"state": "reanalysed", "reason": None}
        elif done.get("status") in ("failed", "incomplete"):
            reason = done.get("error") or "; ".join(done.get("problems") or [])
            states[glycan_id] = {"state": "kept", "reason": f"completion {done['status']}: {reason}", "stage": "complete"}
        else:
            states[glycan_id] = {"state": "not analysed", "reason": "analysed and not completed"}
    return states


def finalize(run_dir: Path, published_dir: Optional[Path] = None, keep_published: Iterable[str] = (),
             online: bool = True, log: Log = print) -> Dict[str, Any]:
    """The root files of ``tree/`` from its glycan folders; ``sync_manifest.json`` last.

    Every published ID gets a folder in the tree: the re-analysed one, or
    the **published folder as it is** (hard links) when the entry cannot be
    re-analysed here, with the reason (:func:`entry_states`): its analysis
    or its completion failed (a damaged trajectory, for example), it has no
    raw entry, or its fix renames it (done per entry afterwards).  Nothing
    is dropped from the database for a failure; ``kept_published`` of the
    summary lists these entries for the administrators.  An entry that was
    never analysed (a shard that did not run) stops the stage, unless its ID
    is in ``keep_published`` (``"all"``).  Root files of the published tree
    that are not regenerated here are copied.  With ``online`` the missing
    reports ask the services as the finalization does.
    """
    from lib import glybake, static_delta, static_publish
    from lib.glybake import GlycoShapeBaker
    from lib.storage import local_storage

    run_dir = Path(run_dir)
    plan = load_plan(run_dir)
    published_dir = Path(published_dir or plan["published_dir"])
    tree = run_dir / "tree"
    tree.mkdir(parents=True, exist_ok=True)
    inventory = _read_json(published_dir / "GLYCOSHAPE.json", {})
    keep = set(keep_published)
    started = time.perf_counter()
    states = entry_states(plan, run_dir, inventory)
    kept: Dict[str, str] = {}
    not_run = []
    for glycan_id in sorted(inventory):
        state = states[glycan_id]
        if state["state"] == "reanalysed" and (tree / glycan_id / "data.json").is_file():
            continue
        if state["state"] == "not analysed" and not ("all" in keep or glycan_id in keep):
            not_run.append(glycan_id)
            continue
        reason = state["reason"] if state["state"] != "not analysed" else "kept on request (--keep-published)"
        if (tree / glycan_id).exists():
            shutil.rmtree(tree / glycan_id)            # an earlier completion of an entry that fails now
        source = published_dir / glycan_id
        if not (source / "data.json").is_file():
            raise BulkError(f"{source} is not a published folder: the published tree is incomplete")
        for path in sorted(p for p in source.rglob("*") if p.is_file()):
            target = tree / glycan_id / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(path, target)
            except OSError:
                shutil.copyfile(path, target)
        kept[glycan_id] = reason
    if not_run:
        raise BulkError(f"{len(not_run)} published ID(s) were not analysed yet, first {not_run[:5]}: run analyse and "
                        "complete for them, or keep their published folders (--keep-published ID, or all)")
    for glycan_id, reason in sorted(kept.items()):
        log(f"kept as published: {glycan_id}: {reason}")
    extra = sorted(path.name for path in tree.iterdir() if path.is_dir() and path.name not in inventory)
    if extra:
        raise BulkError(f"the tree has folders of IDs that are not published: {extra[:5]}")
    for name in REGENERATED_ROOT_FILES:
        (tree / name).unlink(missing_ok=True)
    copied = []
    for path in sorted(published_dir.iterdir()):
        if path.is_file() and path.name not in REGENERATED_ROOT_FILES and not path.name.startswith("."):
            shutil.copyfile(path, tree / path.name)
            copied.append(path.name)
    baker = GlycoShapeBaker(tree)
    with local_storage():
        baker.create_consolidated_json()
        new_inventory = json.loads((tree / "GLYCOSHAPE.json").read_bytes())
        if sorted(new_inventory) != sorted(inventory):
            raise BulkError("GLYCOSHAPE.json of the tree does not hold every published ID (an unreadable data.json?)")
        baker.create_master_archive()
    static_delta.write_rdf(tree / "GLYCOSHAPE.json", tree / "GLYCOSHAPE_RDF.ttl")
    calculate = glybake.WurcsCalculator() if online else None
    resolve = glybake.resolve_missing_accessions if online else None
    missing_wurcs, rows = glybake.missing_reports(new_inventory, calculate=calculate, resolve=resolve)
    (tree / "missing_glytoucan.txt").write_text(glybake.missing_glytoucan_text(missing_wurcs), encoding="utf-8")
    (tree / "missing_wurcs.txt").write_text(glybake.missing_wurcs_text(rows), encoding="utf-8")
    manifest_path = baker.create_sync_manifest()
    manifest = static_publish.parse_manifest(manifest_path.read_bytes(), str(manifest_path))
    listed = static_publish.manifest_files(manifest)
    present = {path.relative_to(tree).as_posix() for path in tree.rglob("*") if path.is_file()} - {"sync_manifest.json"}
    if present != set(listed):
        manifest_path.unlink()
        raise BulkError(f"sync_manifest.json does not list exactly the files of the tree, e.g. {sorted(present ^ set(listed))[:5]}")
    summary = {
        "stage": "finalize", "finished": _utc_now(), "seconds": round(time.perf_counter() - started, 1),
        "glycans": len(manifest["glycans"]), "files": len(listed), "global_hash": manifest["global_hash"],
        "base_global_hash": plan["base_global_hash"], "reanalysed": len(inventory) - len(kept),
        "kept_published": [{"glycoshape_id": glycan_id, "reason": reason} for glycan_id, reason in sorted(kept.items())],
        "root_files_copied": copied,
        "missing_glytoucan": len(missing_wurcs), "missing_wurcs_rows": len(rows), "online": online,
        "bytes": sum(item["size"] for item in listed.values()),
    }
    _write_json(run_dir / "logs" / "finalize.json", summary)
    log(f"finalize: {summary['glycans']} glycans, {summary['files']} files, {summary['bytes'] / 1e9:.2f} GB; "
        f"global_hash {summary['global_hash'][:16]}... ({len(kept)} published folder(s) kept as they are)")
    return summary


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
def compare(run_dir: Path, published_dir: Optional[Path] = None, intended_renames: Iterable[str] = (),
            size_ratio: Tuple[float, float] = (0.5, 2.0), log: Log = print) -> Dict[str, Any]:
    """The new tree against the published one; ``blocking`` problems make the exit status 1.

    Blocking: another set of IDs, a name that changed and is not an intended
    rename (``intended_renames``: IDs), an entry without identity (no WURCS
    and no accession) or without a file the website needs, a manifest that
    is not valid.  Reported for a person to look at: lost or changed
    accessions and WURCS, keywords added and removed, populations (total
    variation of Level 1), representative frames, folder sizes outside
    ``size_ratio`` of the published ones, files added and removed.
    """
    from lib import entry_ops, id_registry, static_levels, static_publish

    run_dir = Path(run_dir)
    plan = load_plan(run_dir)
    published_dir = Path(published_dir or plan["published_dir"])
    tree = run_dir / "tree"
    old_manifest = _read_json(published_dir / "sync_manifest.json")
    new_manifest_path = tree / "sync_manifest.json"
    if not new_manifest_path.is_file():
        raise BulkError(f"{new_manifest_path} is missing: run finalize first")
    new_manifest = static_publish.parse_manifest(new_manifest_path.read_bytes(), str(new_manifest_path))
    old_inventory = _read_json(published_dir / "GLYCOSHAPE.json", {})
    new_inventory = _read_json(tree / "GLYCOSHAPE.json", {})
    intended = set(intended_renames) | {row["glycoshape_id"] for row in plan["entries"]
                                        if row.get("complete_as") and row["complete_as"] != row.get("published_name")}
    blocking: List[str] = []
    old_ids, new_ids = set(old_manifest["glycans"]), set(new_manifest["glycans"])
    if old_ids != new_ids:
        blocking.append(f"the IDs differ: {len(old_ids - new_ids)} only published, {len(new_ids - old_ids)} only new, "
                        f"e.g. {sorted(old_ids ^ new_ids)[:5]}")
    entries: Dict[str, Any] = {}
    counters = {"renamed": 0, "accession_lost": 0, "accession_changed": 0, "wurcs_changed": 0, "keywords_changed": 0,
                "representatives_changed": 0, "size_outside": 0, "unchanged_folders": 0}
    tv_values: List[float] = []
    required = static_levels.required_files((1, 2))
    for glycan_id in sorted(old_ids & new_ids):
        old_data, new_data = old_inventory.get(glycan_id) or {}, new_inventory.get(glycan_id) or {}
        old_entry, new_entry = old_manifest["glycans"][glycan_id], new_manifest["glycans"][glycan_id]
        if old_entry["hash"] == new_entry["hash"]:
            counters["unchanged_folders"] += 1
            continue
        diff = entry_ops.diff_entry(
            old_data, new_data, old_entry["files"], new_entry["files"],
            {level: published_dir / glycan_id / f"output/torsion_{level}_reps.csv" for level in ("level_1", "level_2")},
            {level: tree / glycan_id / f"output/torsion_{level}_reps.csv" for level in ("level_1", "level_2")})
        problems, notes = [], []
        old_name, new_name = id_registry.full_glycam_name(old_data), id_registry.full_glycam_name(new_data)
        if old_name != new_name:
            counters["renamed"] += 1
            (notes if glycan_id in intended else problems).append(f"name {old_name!r} -> {new_name!r}")
        for variant in ("archetype", "alpha", "beta"):
            before = (old_data.get(variant) or {}).get("glytoucan")
            after = (new_data.get(variant) or {}).get("glytoucan")
            if before and not after:
                counters["accession_lost"] += 1
                notes.append(f"{variant} accession {before} lost")
            elif before and after and before != after:
                counters["accession_changed"] += 1
                notes.append(f"{variant} accession {before} -> {after}")
            if (old_data.get(variant) or {}).get("wurcs") != (new_data.get(variant) or {}).get("wurcs"):
                counters["wurcs_changed"] += 1
        archetype = new_data.get("archetype") or {}
        if not archetype.get("wurcs") and not archetype.get("glytoucan"):
            problems.append("neither WURCS nor accession")
        absent = [name for name in required if name not in new_entry["files"]]
        if absent:
            problems.append(f"missing {', '.join(absent[:3])}")
        if "keywords" in diff["search_meta"]:
            counters["keywords_changed"] += 1
        level_1 = diff["analysis"]["levels"].get("level_1") or {}
        if level_1.get("total_variation") is not None:
            tv_values.append(level_1["total_variation"])
        reps = level_1.get("representative_source_frames") or {}
        if reps and not reps.get("same"):
            counters["representatives_changed"] += 1
        ratio = new_entry["total_size"] / old_entry["total_size"] if old_entry["total_size"] else None
        if ratio is not None and not size_ratio[0] <= ratio <= size_ratio[1]:
            counters["size_outside"] += 1
            notes.append(f"folder size x{ratio:.2f}")
        entries[glycan_id] = {"problems": problems, "notes": notes, "diff": diff,
                              "size_ratio": None if ratio is None else round(ratio, 3)}
        blocking += [f"{glycan_id}: {problem}" for problem in problems]
    tv_values.sort()
    summary = {
        "stage": "compare", "finished": _utc_now(),
        "published": {"glycans": len(old_ids), "files": sum(e["file_count"] for e in old_manifest["glycans"].values()),
                      "bytes": sum(e["total_size"] for e in old_manifest["glycans"].values()),
                      "global_hash": old_manifest["global_hash"]},
        "new": {"glycans": len(new_ids), "files": sum(e["file_count"] for e in new_manifest["glycans"].values()),
                "bytes": sum(e["total_size"] for e in new_manifest["glycans"].values()),
                "global_hash": new_manifest["global_hash"]},
        "base_is_plan_base": old_manifest["global_hash"] == plan["base_global_hash"],
        "kept_published": (_read_json(run_dir / "logs" / "finalize.json", {}) or {}).get("kept_published") or [],
        "counts": counters,
        "level_1_population_tv": {
            "entries": len(tv_values), "median": tv_values[len(tv_values) // 2] if tv_values else None,
            "p90": tv_values[int(len(tv_values) * 0.9)] if tv_values else None,
            "max": tv_values[-1] if tv_values else None,
            "over_0.1": sum(1 for value in tv_values if value > 0.1)},
        "blocking": blocking,
        "entries": entries,
    }
    if old_manifest["global_hash"] != plan["base_global_hash"]:
        blocking.append("the published tree given here is not the one the plan was made from")
    summary["ok"] = not blocking
    _write_json(run_dir / "compare.json", summary)
    log(f"compare: {summary['new']['glycans']} glycans ({summary['published']['glycans']} published), "
        f"{summary['new']['bytes'] / 1e9:.2f} GB ({summary['published']['bytes'] / 1e9:.2f} GB); "
        f"{counters['unchanged_folders']} folders unchanged; renamed {counters['renamed']}, accessions lost "
        f"{counters['accession_lost']}, changed {counters['accession_changed']}; keywords changed "
        f"{counters['keywords_changed']}; Level 1 TV median {summary['level_1_population_tv']['median']}, "
        f"{summary['level_1_population_tv']['over_0.1']} over 0.1; {len(blocking)} blocking problem(s)")
    for problem in blocking[:20]:
        log(f"BLOCKING {problem}")
    return summary


# ---------------------------------------------------------------------------
# check-base and publish (read / write the storage)
# ---------------------------------------------------------------------------
def check_base(target: Any, base_global_hash: str, archive_index: Optional[Path] = None,
               own_lock: Optional[str] = None) -> Dict[str, Any]:
    """Whether the stored database is still the one the run was derived from; only reads.

    A bulk publication replaces the whole published prefix with the tree, so
    it is safe only while nothing was published incrementally since the
    published tree was taken: the stored ``sync_manifest.json`` has the
    base ``global_hash``, no publication is pending or holds the lock, and
    the lock is free (or held by ``own_lock``, the owner of the caller's
    lock).  Every publication changes the global hash (one that finds its
    entry published already writes nothing), so the hash decides; the
    records of incremental publications are counted for the report.  With
    ``archive_index`` the stored ``raw/index.json`` is compared with it too.
    """
    from lib import entry_publish
    from lib.entry_publish import Store, new_stats

    store = Store(target.static, new_stats(), dry_run=True)
    raw_store = Store(target.raw, new_stats(), dry_run=True)
    report: Dict[str, Any] = {"base_global_hash": base_global_hash, "problems": []}
    payload = store.read_optional(target.key("sync_manifest.json"))
    stored = None if payload is None else json.loads(payload).get("global_hash")
    report["stored_global_hash"] = stored
    if stored != base_global_hash:
        report["problems"].append(
            f"the stored database ({str(stored)[:16]}...) is not the base of the run ({base_global_hash[:16]}...): "
            "something was published since; finalize again from the stored database (docs/bulk-reanalysis.md)")
    lock, pending = entry_publish.read_lock(target, store), entry_publish.read_pending(target, store)
    report["lock"], report["pending"] = lock, pending and {k: pending.get(k) for k in ("action", "glycoshape_id", "owner")}
    if pending is not None:
        report["problems"].append(f"a {pending['action']} of {pending['glycoshape_id']} is pending (publish_entry.py recover)")
    if lock is not None and lock.get("owner") != own_lock and not entry_publish._is_past(lock.get("expires")):
        report["problems"].append(f"the database is locked by {lock.get('owner')} until {lock.get('expires')}")
    report["entry_records"] = len(store.list(f"{target.state_prefix}/{target.static_prefix}/entries/"))
    if archive_index is not None:
        local = Path(archive_index).read_bytes()
        remote = raw_store.read_optional(f"{target.raw_prefix}/index.json")
        report["raw_index_same"] = remote == local
        if remote != local:
            report["problems"].append("the stored raw/index.json is not the index of the archive that was analysed")
    report["safe"] = not report["problems"]
    return report


def sync_published(target: Any, published_dir: Path, workers: int = 8, records_dir: Optional[Path] = None,
                   log: Log = print) -> Dict[str, Any]:
    """Make a local published tree the stored database (``static/``), as the website synchronises; reads only.

    For a run whose base moved on (something was published incrementally
    since the published tree was copied): afterwards ``plan`` and the later
    stages start from the stored database.  Files whose size and SHA-256
    are those of the stored manifest are kept, the others are downloaded
    into a temporary file, verified and renamed into place (a rename, so a
    tree whose files are hard links of another tree never changes that
    one), glycan folders that the manifest does not list are removed, and
    ``sync_manifest.json`` is written last.
    """
    from lib import static_publish
    from lib.entry_publish import Store, new_stats
    from lib.storage import file_sha256, parallel_map

    published_dir = Path(published_dir)
    store = Store(target.static, new_stats(), dry_run=True)
    payload = store.read(target.key("sync_manifest.json"))
    manifest = static_publish.parse_manifest(payload, target.key("sync_manifest.json"))
    files = static_publish.manifest_files(manifest)
    published_dir.mkdir(parents=True, exist_ok=True)
    (published_dir / "sync_manifest.json").unlink(missing_ok=True)

    def fetch(path: str) -> int:
        local, recorded = published_dir / path, files[path]
        wanted = (recorded["size"], recorded["sha256"].lower())
        if local.is_file() and local.stat().st_size == wanted[0] and file_sha256(local) == wanted:
            return 0
        local.parent.mkdir(parents=True, exist_ok=True)
        part = local.with_name(f".{local.name}.part")
        store.download(target.key(path), part)
        if file_sha256(part) != wanted:
            part.unlink()
            raise BulkError(f"{target.key(path)} is not the file the stored manifest describes")
        os.replace(part, local)
        return 1

    fetched = sum(parallel_map(fetch, sorted(files), workers))
    listed = set(manifest["glycans"])
    removed = []
    for folder in sorted(published_dir.iterdir()):
        if folder.is_dir() and folder.name.startswith("GS") and folder.name not in listed:
            shutil.rmtree(folder)
            removed.append(folder.name)
    stale = sorted(path.relative_to(published_dir).as_posix() for path in published_dir.rglob("*")
                   if path.is_file() and path.relative_to(published_dir).as_posix() not in files)
    for relative in stale:
        (published_dir / relative).unlink()
    (published_dir / "sync_manifest.json").write_bytes(payload)
    records = 0
    if records_dir is not None:
        # the curated metadata of the entries published or changed one at a time (complete --curation-dir)
        records_dir = Path(records_dir)
        records_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{target.state_prefix}/{target.static_prefix}/entries/"
        for key in sorted(store.list(prefix)):
            (records_dir / key[len(prefix):]).write_bytes(store.read(key))
            records += 1
    report = {"global_hash": manifest["global_hash"], "glycans": len(listed), "downloaded": fetched,
              "removed_folders": removed, "removed_files": len(stale), "records": records}
    log(f"{published_dir} is the stored database {manifest['global_hash'][:16]}...: {fetched} file(s) downloaded, "
        f"{len(removed)} folder(s) and {len(stale)} other file(s) removed")
    return report


def publish(run_dir: Path, target: Any, expect_base: str, workers: int = 16, dry_run: bool = False,
            delete_stale: bool = False, lock_ttl: float = 4 * 3600, log: Log = print) -> Dict[str, Any]:
    """Publish ``tree/`` in one publication (``lib.static_publish``), under the lock of the entry publisher.

    Refused unless the stored manifest has ``expect_base`` (the global hash
    of the published tree the run was compared with) once the lock is held:
    an incremental publication cannot slip in between.  Afterwards the
    administrators' copies of the replica diagnostics, a record and a
    version in the history of every entry whose folder changed, and a record
    of the whole publication are written.
    """
    from lib import entry_publish, replicas, static_publish
    from lib.entry_publish import Store, new_stats

    run_dir = Path(run_dir)
    plan = load_plan(run_dir)
    tree = run_dir / "tree"
    stats = new_stats()
    store = Store(target.static, stats, dry_run)
    owner = entry_publish.new_owner()
    report: Dict[str, Any] = {"stage": "publish", "dry_run": dry_run, "expect_base": expect_base}
    lock = None
    if not dry_run:
        lock = entry_publish.acquire_lock(target, store, owner, "bulk-reanalyse", "all", lock_ttl)
    try:
        base = check_base(target, expect_base, own_lock=owner)
        report["base"] = base
        if base["problems"]:
            raise BulkError("not published: " + "; ".join(base["problems"]))
        old_manifest = json.loads(store.read(target.key("sync_manifest.json")))
        new_manifest = json.loads((tree / "sync_manifest.json").read_bytes())
        published = static_publish.publish(tree, target.static, target.static_prefix, target.staging_prefix,
                                           workers=workers, delete_stale=delete_stale, dry_run=dry_run, log=log)
        report["static"] = published
        changed = sorted(glycan_id for glycan_id, entry in new_manifest["glycans"].items()
                         if (old_manifest["glycans"].get(glycan_id) or {}).get("hash") != entry["hash"])
        report["changed_glycans"] = len(changed)
        if dry_run:
            return report
        created = _utc_now()
        code = _read_json(run_dir / "logs" / "finalize.json", {})

        def keep(glycan_id: str) -> None:
            entry = new_manifest["glycans"][glycan_id]
            folder = _folder_of(plan, glycan_id)
            completion = _read_json(run_dir / "entries" / folder / "complete.json", {}) if folder else {}
            record = {
                "schema": entry_publish.RECORD_SCHEMA, "schema_version": entry_publish.SCHEMA_VERSION,
                "glycoshape_id": glycan_id, "glycam_name": completion.get("complete_as"), "action": "bulk-reanalyse",
                "status": "published", "published_at": created, "owner": owner,
                "global_hash_before": old_manifest["global_hash"], "global_hash": new_manifest["global_hash"],
                "entry": {key: entry[key] for key in ("hash", "file_count", "total_size")},
                # what the next operation on the entry takes its curated metadata from (lib.entry_ops)
                "submission": completion.get("submission"), "curation": completion.get("curation"),
                "completion": {key: value for key, value in completion.items() if key not in ("submission", "curation")},
            }
            if record["submission"] is None:
                del record["submission"]
            store.write(target.record_key(glycan_id), json.dumps(record, indent=2).encode() + b"\n")
            version = {**record, "kind": "version",
                       "folder_hash_before": (old_manifest["glycans"].get(glycan_id) or {}).get("hash")}
            store.write(target.history_key(glycan_id, new_manifest["global_hash"], "bulk-reanalyse"),
                        json.dumps(version, indent=2).encode() + b"\n")
            diagnostics = run_dir / "replicas" / f"{glycan_id}.json"
            if diagnostics.is_file() and not replicas.public_placement():
                payload = diagnostics.read_bytes()
                key = target.replicas_key(glycan_id)
                if not store.holds(key, len(payload), hashlib.sha256(payload).hexdigest()):
                    store.write(key, payload)

        from lib.storage import parallel_map

        for _ in parallel_map(keep, changed, workers):
            pass
        store.write(f"{target.state_prefix}/{target.static_prefix}/history/bulk/{created.replace(':', '')}.json",
                    json.dumps({"published_at": created, "owner": owner, "global_hash_before": old_manifest["global_hash"],
                                "global_hash": new_manifest["global_hash"], "changed_glycans": changed,
                                "kept_published": code.get("kept_published") or [],
                                "plan": {key: plan[key] for key in ("archive_index_sha256", "base_global_hash")},
                                "finalize": code}, indent=2).encode() + b"\n")
        report["transfers"] = stats
        _write_json(run_dir / "logs" / "publish.json", report)
        log(f"published {new_manifest['global_hash'][:16]}... ({len(changed)} glycan folders changed)")
        return report
    finally:
        if lock is not None:
            entry_publish._release_quietly(target, store, lock, log)


def _folder_of(plan: Dict[str, Any], glycan_id: str) -> Optional[str]:
    return next((row["folder"] for row in plan["entries"] if row["glycoshape_id"] == glycan_id), None)


def summary(run_dir: Path) -> Dict[str, Any]:
    """Where the run is: per stage what is done, failed and left."""
    run_dir = Path(run_dir)
    plan = load_plan(run_dir)
    counts = {"planned": 0, "skipped": 0, "analysed": 0, "analysis_failed": 0, "completed": 0, "completion_failed": 0}
    failed: List[Dict[str, Any]] = []
    for row in plan["entries"]:
        if row["skip"]:
            counts["skipped"] += 1
            continue
        counts["planned"] += 1
        status = _read_json(run_dir / "entries" / row["folder"] / "status.json", {})
        done = _read_json(run_dir / "entries" / row["folder"] / "complete.json", {})
        if status.get("status") == "ok":
            counts["analysed"] += 1
        elif status.get("status") == "failed":
            counts["analysis_failed"] += 1
            failed.append({"glycam_name": row["glycam_name"], "stage": "analyse", "error": status.get("error")})
        if done.get("status") == "ok":
            counts["completed"] += 1
        elif done.get("status") in ("failed", "incomplete"):
            counts["completion_failed"] += 1
            failed.append({"glycoshape_id": row["glycoshape_id"], "stage": "complete",
                           "error": done.get("error") or done.get("problems")})
    out = {"counts": counts, "failed": failed, "finalized": (run_dir / "tree" / "sync_manifest.json").is_file(),
           "compared": _read_json(run_dir / "compare.json", {}).get("ok")}
    _write_json(run_dir / "logs" / "summary.json", out)
    return out
