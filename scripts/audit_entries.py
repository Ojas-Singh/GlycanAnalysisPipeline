#!/usr/bin/env python3
"""issues.json for every published entry (or the ones named): the entry audit for the administrators' website.

    python scripts/audit_entries.py --static-dir STATIC --archive-dir ARCHIVE --out OUT \\
        [--replicas-dir REPLICAS] [--registry-seed registry-seed.json] \\
        [--registry-export submission_ids.json --registry-export glycans.json] \\
        [--crabwurcs BIN] [--glycan GS00020 ...] [--workers N] [--no-trajectory]
    python scripts/audit_entries.py --list-codes

For every ``GSxxxxx/`` folder of STATIC (or the ones named with --glycan) the
facts of the entry are measured from its static folder (``data.json``,
``output/info.json``, ``output/pca.csv``, the Level 1 structures), its raw
archive entry (found by the GLYCAM name of ``data.json`` as ``index.json``
lists it: ``manifest.json``, ``topology.pdb``, ``info.json`` and frames of
``trajectory.xtc``), its ``replicas.json`` (REPLICAS/GSxxxxx.json, as
``scripts/replica_diagnostics.py`` writes them) and its record in the registry;
then the collection is put together (duplicates, stereocentres across
entries, GLYCAM names with several IDs) and the rules of ``lib/audit.py`` turn
the facts into issues.  Every input is only read; nothing goes to the network.

Written into OUT:

    GSxxxxx/issues.json   the issues of the entry (schema glycoshape-entry-issues, see lib/audit.py)
    GSxxxxx/facts.json    what the rules were applied to (for checking an issue, and for auditing single
                          entries later against the rest of the collection)
    summary.json          counts by issue type and severity, what the bulk re-analysis resolves and what
                          needs the owner, the collection-level lists, the failures and the time taken

With --glycan only the named entries are measured; the collection-level
checks use the facts.json of the other entries already in OUT when there are
any (from an earlier run over the collection), else the inventory alone.

Exit status: 0 every entry was audited (some checks of an entry may have
failed: listed under ``checks_failed``), 1 an entry could not be audited, 2
the folders cannot be used.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib import audit  # noqa: E402


def archive_folders(archive_dir: Path) -> Dict[str, str]:
    """Archive folder by GLYCAM name and by source folder (the name some old entries were published under)."""
    index = json.loads((archive_dir / "index.json").read_text(encoding="utf-8"))
    by: Dict[str, str] = {}
    for record in index["records"]:
        by[record["glycam_name"]] = record["folder"]
    for record in index["records"]:
        by.setdefault(record.get("source_folder") or record["glycam_name"], record["folder"])
    return by


def _facts_job(task) -> Dict[str, Any]:
    glycan_id, static_dir, archive_dir, folder, replicas_dir, registry_record, crabwurcs, trajectory, out_dir = task
    started = time.perf_counter()
    try:
        static = Path(static_dir) / glycan_id
        data = json.loads((static / "data.json").read_text(encoding="utf-8"))
        replicas_document = audit.read_json(Path(replicas_dir) / f"{glycan_id}.json") if replicas_dir else None
        facts = audit.entry_facts(glycan_id, data, static_dir=static,
                                  archive_dir=Path(archive_dir) / folder if folder else None,
                                  replicas_document=replicas_document, registry_record=registry_record,
                                  crab=audit.Crab(crabwurcs) if crabwurcs is not None else audit.Crab(),
                                  trajectory=trajectory)
        facts["archive_folder"] = folder
        facts["seconds"] = round(time.perf_counter() - started, 3)
        audit.write(facts, Path(out_dir) / glycan_id / audit.FACTS_FILE_NAME)
        return {"glycoshape_id": glycan_id, "facts": facts}
    except Exception as exc:  # noqa: BLE001 - reported per entry
        return {"glycoshape_id": glycan_id, "error": f"{type(exc).__name__}: {exc}"[:400],
                "seconds": round(time.perf_counter() - started, 3)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--static-dir", help="published static database (GLYCOSHAPE.json, GSxxxxx/ folders)")
    parser.add_argument("--archive-dir", help="raw archive (index.json, one folder per entry)")
    parser.add_argument("--out", help="folder for GSxxxxx/issues.json, GSxxxxx/facts.json and summary.json")
    parser.add_argument("--replicas-dir", help="folder of replicas.json files (GSxxxxx.json)")
    parser.add_argument("--registry-seed", help="registry seed (registry-seed.json)")
    parser.add_argument("--registry-export", action="append", default=[],
                        help="export of the old registry (submission_ids.json, glycans.json); repeatable")
    parser.add_argument("--crabwurcs", help="crabWURCS executable (default: GLYCOSHAPE_CRABWURCS_BIN as GAP uses it)")
    parser.add_argument("--glycan", action="append", default=[], help="GlycoShape ID; repeatable (default: all)")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--no-trajectory", action="store_true",
                        help="do not read frames of the raw trajectories (no stereocentres per run)")
    parser.add_argument("--list-codes", action="store_true", help="print the issue types as JSON and stop")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_codes:
        print(json.dumps({"schema": audit.SCHEMA, "schema_version": audit.SCHEMA_VERSION,
                          "severities": list(audit.SEVERITIES), "actions": audit.ACTIONS,
                          "issue_types": audit.issue_type_table()}, indent=1, ensure_ascii=False))
        return 0
    if not (args.static_dir and args.archive_dir and args.out):
        print("audit_entries: error: --static-dir, --archive-dir and --out are required", file=sys.stderr)
        return 2
    static_dir, archive_dir, out_dir = Path(args.static_dir), Path(args.archive_dir), Path(args.out)
    if not static_dir.is_dir() or not (archive_dir / "index.json").is_file():
        print("audit_entries: error: --static-dir must be a folder and --archive-dir an archive with index.json",
              file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    crab = audit.Crab(args.crabwurcs) if args.crabwurcs else audit.Crab()
    folders = archive_folders(archive_dir)
    inventory_path = static_dir / "GLYCOSHAPE.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8")) if inventory_path.is_file() else {}
    exports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.registry_export]
    seed = json.loads(Path(args.registry_seed).read_text(encoding="utf-8")) if args.registry_seed else None
    registry_records = {}
    for export in exports:
        for record in export.get("records", []):
            if record.get("glycoshape_id") and "simulation_length" in record:
                registry_records.setdefault(record["glycoshape_id"], record)
    registry = audit.registry_names(seed, exports) if (seed or exports) else None

    ids = args.glycan or sorted(p.name for p in static_dir.iterdir() if p.is_dir() and p.name.startswith("GS"))
    tasks = []
    for glycan_id in ids:
        try:
            name = json.loads((static_dir / glycan_id / "data.json").read_text(encoding="utf-8"))["archetype"]["name"]
        except (OSError, ValueError, KeyError):
            name = None
        tasks.append((glycan_id, str(static_dir), str(archive_dir), folders.get(name) if name else None,
                      args.replicas_dir, registry_records.get(glycan_id), crab.executable, not args.no_trajectory,
                      str(out_dir)))
    with ProcessPoolExecutor(max(1, args.workers)) as pool:
        results = list(pool.map(_facts_job, tasks, chunksize=2))
    facts_seconds = time.perf_counter() - started
    facts = {r["glycoshape_id"]: r["facts"] for r in results if "facts" in r}
    failures = [r for r in results if "error" in r]

    # the collection: these entries and, when only some were named, the facts already in OUT
    collection_facts = dict(facts)
    if args.glycan:
        for path in sorted(out_dir.glob(f"GS*/{audit.FACTS_FILE_NAME}")):
            gid = path.parent.name
            if gid not in collection_facts:
                collection_facts[gid] = json.loads(path.read_text(encoding="utf-8"))
    whole = len(collection_facts) >= len([k for k in inventory if k.startswith("GS")]) > 0
    context = audit.collection_context(collection_facts, registry=registry, inventory=inventory)

    by_code: Dict[str, Counter] = defaultdict(Counter)
    entries_by_code: Dict[str, List[str]] = defaultdict(list)
    severities = Counter()
    entries_with = Counter()
    resolution = Counter()
    for gid in ids:
        if gid not in facts:
            continue
        issues = audit.entry_issues(facts[gid], context)
        document = audit.document(facts[gid], issues, collection=whole)
        audit.write(document, out_dir / gid / audit.FILE_NAME)
        worst = None
        for issue in issues:
            by_code[issue["code"]][issue["severity"]] += 1
            by_code[issue["code"]]["resolved_by_reanalysis"] += bool(issue["resolved_by_reanalysis"])
            by_code[issue["code"]]["needs_owner"] += bool(issue["needs_owner"])
            entries_by_code[issue["code"]].append(gid)
            severities[issue["severity"]] += 1
            resolution["resolved by the re-analysis" if issue["resolved_by_reanalysis"] else
                       ("needs the owner" if issue["needs_owner"] else "information")] += 1
            worst = issue["severity"] if worst is None or audit.SEVERITIES.index(issue["severity"]) < \
                audit.SEVERITIES.index(worst) else worst
        entries_with[worst or "none"] += 1

    seconds = [r["facts"]["seconds"] for r in results if "facts" in r]
    several = context["names_with_several_ids"]
    published = set(inventory)
    report: Dict[str, Any] = {
        "what": "entry audit: issues.json of published entries",
        "schema": audit.SCHEMA, "schema_version": audit.SCHEMA_VERSION,
        "inputs": {"static_dir": str(static_dir), "archive_dir": str(archive_dir), "replicas_dir": args.replicas_dir,
                   "registry_seed": args.registry_seed, "registry_exports": args.registry_export,
                   "crabwurcs": crab.executable, "crabwurcs_version": crab.version(),
                   "trajectory_frames_read": not args.no_trajectory, "network": "none"},
        "entries": len(ids), "audited": len(facts), "failed": len(failures), "collection": whole,
        "issues": sum(severities.values()),
        "by_severity": {s: severities.get(s, 0) for s in audit.SEVERITIES},
        "entries_by_worst_severity": {s: entries_with.get(s, 0) for s in (*audit.SEVERITIES, "none")},
        "by_resolution": dict(resolution),
        "by_code": {code: {"entries": len(set(entries_by_code[code])), "issues": len(entries_by_code[code]),
                           **{s: by_code[code].get(s, 0) for s in audit.SEVERITIES},
                           "resolved_by_reanalysis": by_code[code].get("resolved_by_reanalysis", 0),
                           "needs_owner": by_code[code].get("needs_owner", 0),
                           "ids": sorted(set(entries_by_code[code]))}
                    for code in audit.ISSUE_TYPES if code in by_code},
        "codes_without_issues": [code for code in audit.ISSUE_TYPES if code not in by_code],
        "collection_lists": {
            "duplicates": sorted({tuple(sorted([gid, *others])) for gid, others in context["duplicates"].items()}),
            "stereo_minority": context["stereo_minority"],
            "names_with_several_ids": [{"glycam_name": name, "ids": ids_, "published": sorted(i["glycoshape_id"] for i in ids_
                                                                                              if i["glycoshape_id"] in published)}
                                       for name, ids_ in sorted(several.items())],
        },
        "checks_failed": {gid: f["errors"] for gid, f in sorted(facts.items()) if f.get("errors")},
        "failures": failures,
        "seconds": {"total": round(time.perf_counter() - started, 1), "facts": round(facts_seconds, 1),
                    "per_entry_median": round(sorted(seconds)[len(seconds) // 2], 3) if seconds else None,
                    "per_entry_max": round(max(seconds), 3) if seconds else None, "workers": args.workers},
    }
    audit.write(report, out_dir / "summary.json")
    for failure in failures:
        print(f"audit_entries: {failure['glycoshape_id']}: {failure['error']}", file=sys.stderr)
    print(json.dumps({k: report[k] for k in ("entries", "audited", "failed", "issues", "by_severity",
                                             "entries_by_worst_severity", "by_resolution", "seconds")}, indent=1))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
