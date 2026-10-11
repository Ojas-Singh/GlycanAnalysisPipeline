"""Bundle directories for the tests and trials of publishing one entry.

A bundle is what the node side hands over (the contract is in
``lib/entry_complete.py``).  ``make_bundle`` builds the extracted directory
from the result folder of ``scripts/run_entry.py`` and the archive entry it
was computed from, with a ``bundle.json`` of the documented form;
``synthetic_run`` makes both for the small synthetic trajectory of
``synthetic_glycan.py``.  Everything is copied, so the inputs are only read.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

BUNDLE_SCHEMA = "glycoshape-entry-bundle"
ARCHIVE_ORDER = ("trajectory.xtc", "topology.pdb", "structure.mol2", "manifest.json", "info.json", "name.json")


def sha256_of(path: Path) -> Dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
            size += len(block)
    return {"size": size, "sha256": digest.hexdigest()}


def _member_order(bundle_dir: Path) -> list:
    """The files of a bundle in the order they have in ``bundle.json``: archive, static, result, log."""
    archive = [f"archive/{name}" for name in ARCHIVE_ORDER if (bundle_dir / "archive" / name).is_file()]
    archive += sorted(
        f"archive/{path.name}" for path in (bundle_dir / "archive").iterdir()
        if path.is_file() and path.name not in ARCHIVE_ORDER)
    static = sorted(path.relative_to(bundle_dir).as_posix() for path in (bundle_dir / "static").rglob("*") if path.is_file())
    top = [name for name in ("result.json", "run.log") if (bundle_dir / name).is_file()]
    other = sorted(
        path.relative_to(bundle_dir).as_posix() for path in bundle_dir.rglob("*")
        if path.is_file() and path.relative_to(bundle_dir).as_posix() not in set(archive + static + top + ["bundle.json"]))
    return archive + static + top + other


def write_bundle_json(bundle_dir: Path, **changes: Any) -> Dict[str, Any]:
    """(Re)write ``bundle.json`` for the files that are in ``bundle_dir`` now; ``changes`` replace its fields."""
    bundle_dir = Path(bundle_dir)
    result = json.loads((bundle_dir / "result.json").read_text())
    manifest = json.loads((bundle_dir / "archive" / "manifest.json").read_text())
    path = bundle_dir / "bundle.json"
    pipeline = result.get("pipeline") or {}
    document = json.loads(path.read_text()) if path.is_file() else {
        "schema": BUNDLE_SCHEMA,
        "schema_version": 1,
        "glycam_name": result["glycam_name"],
        "created": result["finished_at"],
        "pipeline": {
            "commit": pipeline.get("commit"),
            "uncommitted_changes": pipeline.get("uncommitted_changes"),
            "version": pipeline.get("code_sha256"),
            "version_of": pipeline.get("code_sha256_of"),
            "converter_version": manifest["converter"]["version"],
            "python": pipeline.get("python"),
            "packages": pipeline.get("packages"),
        },
        "levels": result["settings"]["cluster_levels"],
        "files": {},
        "source": {
            "name": manifest["source"]["folder"], "simulation_length_ns": None, "time_per_frame_ps": None,
            "md_package": None, "force_field": None, "temperature_k": None, "pressure_bar": None,
            "salt_concentration_mm": None, "comments": None, "glytoucan_id": None, "run_id": None,
        },
        "extends": None,
    }
    document.update(changes)
    document["files"] = {relative: sha256_of(bundle_dir / relative) for relative in _member_order(bundle_dir)}
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return document


def make_bundle(dest: Path, run_dir: Path, archive_entry: Path, **fields: Any) -> Path:
    """A bundle directory at ``dest`` from a result folder and the archive entry it was computed from."""
    dest, run_dir, archive_entry = Path(dest), Path(run_dir), Path(archive_entry)
    dest.mkdir(parents=True)
    shutil.copytree(archive_entry, dest / "archive")
    shutil.copytree(run_dir / "static", dest / "static")
    shutil.copyfile(run_dir / "result.json", dest / "result.json")
    shutil.copyfile(run_dir / "run.log", dest / "run.log")
    write_bundle_json(dest, **fields)
    return dest


def renamed_bundle(dest: Path, bundle_dir: Path, glycam_name: str) -> Path:
    """A copy of a bundle whose three documents name another glycan (same files otherwise).

    For tests that need a second entry and not a second analysis.
    """
    dest = Path(dest)
    shutil.copytree(bundle_dir, dest)
    (dest / "bundle.json").unlink()
    manifest = json.loads((dest / "archive" / "manifest.json").read_text())
    manifest.update(glycam_name=glycam_name, folder=glycam_name, object_key_prefix=f"raw/{glycam_name}/")
    manifest["source"]["folder"] = glycam_name
    (dest / "archive" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    result = json.loads((dest / "result.json").read_text())
    result.update(glycam_name=glycam_name, folder=glycam_name)
    result["input"]["glycam_name"] = glycam_name
    (dest / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_bundle_json(dest)
    return dest


def submission_record(glycoshape_id: str, glycam_name: str, **fields: Any) -> Dict[str, Any]:
    """A complete submission record; ``fields`` replace the top-level ones."""
    record = {
        "schema": "glycoshape-submission",
        "schema_version": 1,
        "glycoshape_id": glycoshape_id,
        "glycam_name": glycam_name,
        "simulation": {
            "simulation_length_ns": 2500.0, "md_package": "GROMACS", "force_field": "GLYCAM_06j",
            "temperature_k": 300.0, "pressure_bar": 1.0, "salt_concentration_mm": 150.0,
            "source_type": "manual", "comments": "",
        },
        "glytoucan_id": "",
        "curated": {"common_names": [], "aliases": [], "description": "", "keywords": []},
    }
    record.update(fields)
    return record


def synthetic_run(root: Path, frames: int = 480, seed: int = 7, info: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
    """Source, archive entry and result folder of the synthetic glycan below ``root``.

    Runs the process stage (a few seconds) and restores what
    ``entry_run.run_entry`` sets in ``lib.config``.
    """
    import lib.config as config
    from lib import archive, entry_run
    from lib.storage import reset_storage_manager
    from synthetic_glycan import write_source

    root = Path(root)
    source = write_source(root / "data", frames=frames, seed=seed)
    if info is not None:
        (source / "info.json").write_text(json.dumps(info), encoding="utf-8")
    entry = Path(archive.convert_entry(source, root / "archive")["entry"])
    names = ("max_workers", "cluster_levels", "use_oracle_storage", "oracle_par_url", "use_pocketbase")
    settings = {name: getattr(config, name) for name in names}
    try:
        result = entry_run.run_entry(str(entry), str(root / "result"), workers=1, workers_from="test")
    finally:
        for name, value in settings.items():
            setattr(config, name, value)
        reset_storage_manager()
    assert result["status"] == "ok", result["error"]
    return {"source": source, "archive": entry, "result": root / "result"}
