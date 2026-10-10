"""The files of a static ``GSxxxxx`` folder that need neither identifiers nor the network.

A static folder is made of two kinds of files. This module writes the first
kind, everything that follows from ``process/<glycan>/`` alone:

    output/info.json, output/torparts.npz
    output/pca.csv, torsion_glycosidic.csv, torsion_all.csv, torsion_level_<n>_reps.csv
    output/level_<n>/dist.svg
    output/level_<n>/{GLYCAM,PDB}/{alpha,beta}.pdb
    landscape/pca_model.joblib, landscape_metadata.json, preprocessing.npz

for the levels of ``config.cluster_levels``. A process folder that holds more
levels (Level 3 of older runs) is reduced to the configured ones on the way.

The second kind depends on the GlycoShape ID, the submission metadata and web
services (data.json, glygen.json, glycosmos.json, snfg.svg, the identifier
REMARKs at the top of the PDB files). ``lib.glystatic.process_glycan`` writes
those around a call to :func:`write_static_ready`; the one-entry mode
(``lib.entry_run``) stops after it.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import re
from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from lib import config, glypdbio
from lib.storage import get_storage_manager

logger = logging.getLogger(__name__)

LANDSCAPE_FILES = ("pca_model.joblib", "landscape_metadata.json", "preprocessing.npz")
ANALYSIS_TABLES = {
    "pca.csv": "pca.parquet",
    "torsion_glycosidic.csv": "torsion_glycosidic.parquet",
    "torsion_all.csv": "torsion_all.parquet",
}
_LEVEL_COLUMN = re.compile(r"level_(\d+)_(cluster|center)")
_LEVEL_PCA_KEY = re.compile(r"level_(\d+)_clustering")


def configured_levels() -> tuple:
    """Levels of the static folder (``config.cluster_levels``)."""
    return tuple(int(level) for level in config.cluster_levels)


def level_names(levels: Optional[Iterable[int]] = None) -> list:
    """``["level_1", "level_2"]`` for the configured (or given) levels."""
    return [f"level_{int(level)}" for level in (configured_levels() if levels is None else levels)]


def _other_level_columns(columns: Iterable[str], levels: Iterable[int]) -> list:
    """``level_<n>_cluster`` / ``level_<n>_center`` columns of levels that are not kept."""
    keep = {int(level) for level in levels}
    dropped = []
    for name in columns:
        match = _LEVEL_COLUMN.fullmatch(str(name))
        if match and int(match.group(1)) not in keep:
            dropped.append(name)
    return dropped


def filter_info_levels(info: dict, levels: Optional[Iterable[int]] = None) -> tuple:
    """``info.json`` content reduced to the given levels; returns ``(info, changed)``.

    Removes the entries of other levels from ``levels`` and their
    ``level_<n>_clustering`` block from ``pca``. Nothing else is touched, so an
    ``info.json`` written with Level 3 becomes the one a run without it writes.
    """
    keep = {int(level) for level in (configured_levels() if levels is None else levels)}
    if not isinstance(info, dict):
        return info, False
    changed = False
    out = dict(info)
    if isinstance(info.get("levels"), list):
        kept = [
            entry for entry in info["levels"]
            if not (isinstance(entry, dict) and entry.get("level") is not None and int(entry["level"]) not in keep)
        ]
        if len(kept) != len(info["levels"]):
            out["levels"] = kept
            changed = True
    if isinstance(info.get("pca"), dict):
        pca = {
            key: value for key, value in info["pca"].items()
            if not ((match := _LEVEL_PCA_KEY.fullmatch(str(key))) and int(match.group(1)) not in keep)
        }
        if len(pca) != len(info["pca"]):
            out["pca"] = pca
            changed = True
    return out, changed


def load_info_json(folder_path) -> dict:
    """Load and parse the info.json file from the output directory."""
    storage = get_storage_manager()
    info_path = f"{folder_path}/output/info.json"
    if not storage.exists(info_path):
        raise FileNotFoundError(f"info.json not found at {info_path}")

    with storage.open(info_path, 'r') as f:
        return json.load(f)


def extract_cluster_data(info_data: dict, folder_path: str) -> dict:
    """Extract cluster data directly from info.json without probing remote per-cluster files.

    Only the levels of ``config.cluster_levels`` are returned; they decide
    which level folders and tables the static folder gets.
    """
    info_data, _ = filter_info_levels(info_data)
    cluster_data = {
        "entropy_nats": info_data.get("entropy_nats"),
        "torsions": info_data.get("torsions", {}),
        "levels": {},
        "pca": info_data.get("pca", {})
    }

    # Process each level
    for level_info in info_data.get("levels", []):
        level_num = level_info["level"]

        # Extract cluster information for this level
        clusters = []
        for cluster in level_info.get("clusters", []):
            cluster_info = {
                "cluster_id": cluster["cluster_id"],
                "cluster_size_pct": cluster["cluster_size_pct"],
                "representative_idx": cluster["representative_idx"],
                "members_count": cluster["members_count"],
                "torsion_stats": cluster.get("torsion_stats", {})
            }
            clusters.append(cluster_info)

        cluster_data["levels"][f"level_{level_num}"] = {
            "n_clusters": level_info["n_clusters"],
            "clusters": clusters
        }

    return cluster_data


def _csv_without_columns(content: bytes, dropped: set) -> bytes:
    """A CSV file without some columns; the remaining fields are copied as text."""
    reader = csv.reader(io.StringIO(content.decode("utf-8"), newline=""))
    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator="\n")
    keep = None
    for row in reader:
        if keep is None:
            keep = [index for index, name in enumerate(row) if name not in dropped]
        writer.writerow([row[index] for index in keep if index < len(row)])
    return out.getvalue().encode("utf-8")


def copy_or_convert_analysis_table(storage, folder_path: str, output_subdir: str, csv_name: str, parquet_name: str) -> bool:
    """Write a static CSV from a Parquet process export, falling back to legacy CSV copy.

    Cluster and center columns of levels that are not configured are left out.
    """
    parquet_src = f"{folder_path}/embedding/exports/{parquet_name}"
    csv_src = f"{folder_path}/output/{csv_name}"
    csv_dst = f"{output_subdir}/{csv_name}"
    levels = configured_levels()

    if storage.exists(parquet_src):
        try:
            df = pd.read_parquet(BytesIO(storage.read_binary(parquet_src)))
            dropped = _other_level_columns(df.columns, levels)
            if dropped:
                df = df.drop(columns=dropped)
            storage.write_text(csv_dst, df.to_csv(index=False))
            logger.info(f"Converted Parquet export to static CSV: {parquet_name} -> {csv_name}")
            return True
        except Exception as exc:
            logger.warning(f"Failed to convert {parquet_src} to {csv_name}: {exc}")

    if storage.exists(csv_src):
        content = storage.read_binary(csv_src)
        try:
            header = next(csv.reader(io.StringIO(content[:1 << 20].decode("utf-8", errors="replace"))), [])
            dropped = _other_level_columns(header, levels)
            if dropped:
                content = _csv_without_columns(content, set(dropped))
        except Exception as exc:
            logger.warning(f"Could not inspect the columns of {csv_src}: {exc}")
        storage.write_binary(csv_dst, content)
        return True

    return False


def _emit_multimodel_pdbs(dest_level_dir: str, source_level_dir: str, cluster_data: dict, level_name: str, anomer_meta: dict | None = None) -> None:
    """Delegate multi-model emission to glypdbio with remarks support."""
    try:
        glypdbio.emit_multimodel_pdbs(
            source_level_dir=source_level_dir,
            dest_level_dir=dest_level_dir,
            cluster_data=cluster_data,
            level_name=level_name,
            anomer_meta=anomer_meta,
            formats=["GLYCAM", "PDB"],
        )
    except Exception as e:
        logger.warning(f"Failed to emit multi-model PDBs for {level_name}: {e}")


def copy_and_convert_pdbs(
    folder_path: str,
    output_dir: str,
    glycam_name: str,
    archetype_glytoucan: str = None,
    archetype_iupac: str = None,
    alpha_glytoucan: str = None,
    alpha_iupac: str = None,
    beta_glytoucan: str = None,
    beta_iupac: str = None,
    cluster_data: dict = None,
    identifier_remarks: bool = True,
):
    """Copy PDB files and convert them to different formats using glypdbio.

    Parameters:
        folder_path: Path to the source folder containing PDB files
        output_dir: Path to the output directory
        glycam_name: GLYCAM name of the glycan
        archetype_glytoucan: GlyTouCan ID for the archetype form
        archetype_iupac: IUPAC name for the archetype form
        alpha_glytoucan: GlyTouCan ID for the alpha anomer
        alpha_iupac: IUPAC name for the alpha anomer
        beta_glytoucan: GlyTouCan ID for the beta anomer
        beta_iupac: IUPAC name for the beta anomer
        cluster_data: Dictionary containing cluster information from info.json
        identifier_remarks: write the GlycoShape REMARK block (with the
            GlyTouCan ID and IUPAC name) at the top of every file. False leaves
            the files as their models only: what the block is later put on.
    """
    storage = get_storage_manager()
    source_output = Path(folder_path) / "output"

    # Determine levels from info.json cluster_data to be backend-agnostic
    level_names: list[str] = []
    if isinstance(cluster_data, dict) and "levels" in cluster_data and isinstance(cluster_data["levels"], dict):
        level_names = sorted([k for k in cluster_data["levels"].keys() if k.startswith("level_")])
    # Fallback to filesystem probing only if necessary
    if not level_names:
        try:
            # Best-effort: collect unique names under output/ that look like level_* prefixes
            all_objs = storage.list_files(source_output, "*")
            seen = set()
            for obj in all_objs:
                parts = Path(obj).as_posix().split('/')
                try:
                    idx = parts.index("output")
                    name = parts[idx+1]
                    if name.startswith("level_"):
                        seen.add(name)
                except Exception:
                    continue
            level_names = sorted(seen)
        except Exception:
            level_names = []

    for level_name in level_names:
        dest_level_dir = Path(output_dir) / "output" / level_name
        storage.mkdir(dest_level_dir)

        # Copy dist.svg if it exists
        dist_file = source_output / level_name / "dist.svg"
        if storage.exists(dist_file):
            # Use storage abstraction to read and write
            content = storage.read_binary(dist_file)
            storage.write_binary(dest_level_dir / "dist.svg", content)

        # Build multi-model PDBs for every discovered clustering level; do not emit per-cluster converted PDBs.
        try:
            anomer_meta = {
                "alpha": {"glytoucan": alpha_glytoucan or archetype_glytoucan, "iupac": alpha_iupac or archetype_iupac},
                "beta": {"glytoucan": beta_glytoucan or archetype_glytoucan, "iupac": beta_iupac or archetype_iupac},
            } if identifier_remarks else None
            _emit_multimodel_pdbs(
                dest_level_dir=dest_level_dir,
                source_level_dir=source_output / level_name,
                cluster_data=cluster_data,
                level_name=level_name,
                anomer_meta=anomer_meta,
            )
        except Exception as e:
            logger.warning(f"Failed to build multi-model PDBs for {level_name}: {e}")


def copy_analysis_files(folder_path, final_output_dir: str, level_keys: Iterable[str]) -> None:
    """Tables, info.json, torparts.npz and the frozen PCA landscape of a static folder."""
    storage = get_storage_manager()

    # Copy additional analysis files to output subdirectory
    output_subdir = f"{final_output_dir}/output"
    storage.mkdir(output_subdir)

    table_exports = dict(ANALYSIS_TABLES)
    for level_key in level_keys:
        table_exports[f"torsion_{level_key}_reps.csv"] = f"torsion_{level_key}_reps.parquet"

    for csv_name, parquet_name in table_exports.items():
        copy_or_convert_analysis_table(storage, folder_path, output_subdir, csv_name, parquet_name)

    for file_name in ("torparts.npz", "info.json"):
        src_file = f"{folder_path}/output/{file_name}"
        if storage.exists(src_file):
            # Use storage abstraction to copy files
            content = storage.read_binary(src_file)
            if file_name == "info.json":
                # An info.json that still describes other levels is reduced
                # to the configured ones; otherwise it is copied as it is.
                try:
                    info, changed = filter_info_levels(json.loads(content))
                    if changed:
                        content = json.dumps(info, indent=2).encode("utf-8")
                except Exception as exc:
                    logger.warning(f"Could not reduce the levels of {src_file}: {exc}")
            storage.write_binary(f"{output_subdir}/{file_name}", content)

    # Copy frozen PCA landscape data (ROH-excluded, anomer-independent)
    landscape_src_dir = f"{folder_path}/embedding/landscape"
    if storage.exists(landscape_src_dir):
        landscape_dst_dir = f"{final_output_dir}/landscape"
        storage.mkdir(landscape_dst_dir)
        for landscape_file in LANDSCAPE_FILES:
            src_file = f"{landscape_src_dir}/{landscape_file}"
            if storage.exists(src_file):
                content = storage.read_binary(src_file)
                storage.write_binary(f"{landscape_dst_dir}/{landscape_file}", content)
                logger.info(f"Copied landscape file: {landscape_file}")


def write_static_ready(folder_path, destination, glycam_name: str) -> dict:
    """Everything of a static folder that follows from ``process/<glycan>/`` alone.

    ``folder_path`` is ``process/<glycan>/``; ``destination`` gets ``output/``
    and ``landscape/`` laid out as in ``static/GSxxxxx/``. The PDB files carry
    no identifier REMARK block. Returns the levels written.
    """
    storage = get_storage_manager()
    cluster_data = extract_cluster_data(load_info_json(folder_path), str(folder_path))
    storage.mkdir(destination)
    copy_and_convert_pdbs(
        folder_path,
        str(destination),
        glycam_name,
        cluster_data=cluster_data,
        identifier_remarks=False,
    )
    copy_analysis_files(folder_path, str(destination), list(cluster_data["levels"].keys()))
    return {"levels": sorted(cluster_data["levels"].keys())}
