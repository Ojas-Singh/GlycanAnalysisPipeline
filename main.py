#!/usr/bin/env python3
"""
Main entry point for the Glycan Analysis Pipeline.

This script runs a comprehensive pipeline for each glycan:
1. v2: Individual glycan analysis (PCA, clustering, torsion analysis)
2. glystatic: Generate static database entries
3. glymeta: Update metadata and search terms
4. glybake: Create final database archives

The pipeline processes all glycans in data_dir sequentially, then finalizes
with database-wide operations.
"""

import os
import sys
import logging
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import json

import lib.config as config


def _resolve_gap_log_path() -> Path:
    log_path = os.environ.get("GLYCOSHAPE_GAP_LOG", "last_run.log").strip()
    return Path(log_path) if log_path else Path("last_run.log")


def _configure_logging() -> Path:
    log_path = _resolve_gap_log_path()
    handlers = [logging.StreamHandler()]

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="w", encoding="utf-8"))
    except Exception as exc:
        print(f"Warning: failed to open log file '{log_path}': {exc}", file=sys.stderr)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True,
    )
    return log_path


LOG_FILE_PATH = _configure_logging()
logger = logging.getLogger(__name__)

from lib.storage import get_storage_manager, reset_storage_manager
from lib.glystatic import get_glycan_metadata, process_glycan as glystatic_process_glycan
from lib.glymeta import GlycanMetadataProcessor
from lib.glybake import GlycoShapeBaker
from lib.v2 import run_glycan_analysis, get_glycan_status, Chem as V2_RDKIT_CHEM


class GlycanPipelineRunner:
    """Main pipeline runner for processing glycans."""
    REQUIRED_GLYSTATIC_ANALYSIS_FILES = (
        "pca.csv",
        "torparts.npz",
        "info.json",
    )
    REQUIRED_GLYSTATIC_LEVELS = ("level_1", "level_2", "level_3")
    REQUIRED_GLYSTATIC_LEVEL_FILES = (
        "dist.svg",
        "PDB/alpha.pdb",
        "PDB/beta.pdb",
        "GLYCAM/alpha.pdb",
        "GLYCAM/beta.pdb",
    )
    LOG_TAIL_TOP_LEVEL_FILES = (
        "data.json",
        "glygen.json",
        "glycosmos.json",
        "snfg.svg",
    )
    LOG_TAIL_OUTPUT_FILES = (
        "info.json",
        "pca.csv",
        "torparts.npz",
        "torsion_all.csv",
        "torsion_glycosidic.csv",
        "torsion_level_1_reps.csv",
        "torsion_level_2_reps.csv",
        "torsion_level_3_reps.csv",
    )
    LOG_TAIL_STEP_ORDER = (
        ("input_files", "input_files"),
        ("store", "v2.store"),
        ("pca", "v2.pca"),
        ("torsions", "v2.torsions"),
        ("clustering", "v2.clustering"),
        ("cluster_stats", "v2.cluster_stats"),
        ("create_info", "v2.create_info"),
        ("structures", "v2.structures"),
        ("plots", "v2.plots"),
        ("export", "v2.export"),
    )
    
    def __init__(self, data_dir, process_dir, output_dir, force_update: bool = False, mode: str = "incremental"):
        """Initialize the pipeline runner.
        
        Args:
            data_dir: Directory containing glycan input data (read-only)
            process_dir: Directory for storing processing outputs (data, embedding, output)
            output_dir: Directory for final database output
            force_update: Whether to force recomputation of existing results
            mode: Run mode: incremental, refresh-static, or fresh
        """
        self.data_dir = data_dir
        self.process_dir = process_dir 
        self.output_dir = output_dir
        self.mode = mode
        self.force_update = force_update or mode == "fresh"
        
        # Initialize storage manager
        self.storage = get_storage_manager()
        
        # Ensure directories exist
        self.storage.mkdir(self.data_dir)
        self.storage.mkdir(self.process_dir)
        self.storage.mkdir(self.output_dir)
        
        # Initialize PocketBase client (optional)
        self.pb_client = None
        if config.use_pocketbase:
            try:
                from lib.pocketbase import GLYCAN_COLLECTION, get_pocketbase_client
                client = get_pocketbase_client()
                if client.is_available():
                    self.pb_client = client
                    client.prefetch_all()
                    try:
                        client.prefetch_all(collection=GLYCAN_COLLECTION)
                    except Exception as e:
                        logger.warning(f"Failed to prefetch glycans metadata collection: {e}")
                    logger.info("PocketBase integration enabled")
                else:
                    logger.info("PocketBase not available, using CSV fallback")
            except Exception as e:
                logger.warning(f"PocketBase initialization failed: {e}")

        # Initialize processors
        self.metadata_processor = GlycanMetadataProcessor(self.output_dir, pb_client=self.pb_client)
        self.baker = GlycoShapeBaker(self.output_dir)
        self._known_static_output_dirs: Dict[str, str] = {}
        self._last_glystatic_result: Dict[str, Dict[str, Any]] = {}
        self._remote_static_index: Optional[Dict[str, str]] = None
        self._fast_indexes_built = False
        self._remote_process_complete: Set[str] = set()
        self._remote_static_data_dirs: Set[str] = set()
        self._pocketbase_by_glycan: Dict[str, Dict[str, Any]] = {}
    
    def _remote_output_base(self) -> str:
        """Return the remote base prefix for outputs.

        In Oracle mode, we upload/list under the configured static bucket prefix,
        regardless of the local output_dir path. Locally, use output_dir.
        """
        return config.oracle_output_prefix if config.use_oracle_storage else str(self.output_dir)

    def _remote_process_base(self) -> str:
        """Return the remote base prefix for process artifacts."""
        return config.oracle_process_prefix if config.use_oracle_storage else str(self.process_dir)

    @staticmethod
    def _normalize_storage_key(value: Any) -> str:
        """Normalize local/remote storage paths to a forward-slash string."""
        raw = value.as_posix() if hasattr(value, "as_posix") else str(value)
        return raw.replace("\\", "/").rstrip("/")

    def _cache_static_output_dir(self, glycan_name: str, output_dir: Optional[str]) -> None:
        if not output_dir:
            return
        self._known_static_output_dirs[glycan_name] = output_dir

    def _get_cached_static_output_dir(self, glycan_name: str) -> Optional[str]:
        candidate = self._known_static_output_dirs.get(glycan_name)
        if not candidate:
            last_result = self._last_glystatic_result.get(glycan_name) or {}
            candidate = str(last_result.get("final_output_dir") or "").strip() or None
        if not candidate:
            return None
        data_json = f"{candidate}/data.json"
        if self.storage.exists(data_json):
            return candidate
        return None

    def _get_pocketbase_glycoshape_id(self, glycan_name: str) -> Optional[str]:
        """Return the expected GS ID for a glycan from the prefetched submission metadata."""
        record = self._get_prefetched_pocketbase_record(glycan_name)
        gs_id = str((record or {}).get("glycoshape_id") or "").strip()
        return gs_id or None

    def _get_prefetched_pocketbase_record(self, glycan_name: str) -> Optional[Dict[str, Any]]:
        """Return a prefetched PocketBase record without doing per-glycan network I/O."""
        if glycan_name in self._pocketbase_by_glycan:
            return self._pocketbase_by_glycan[glycan_name]
        if self.pb_client is None:
            return None
        cache = getattr(self.pb_client, "_cache_by_name", {})
        record = cache.get(glycan_name) if isinstance(cache, dict) else None
        if isinstance(record, dict):
            self._pocketbase_by_glycan[glycan_name] = record
            return record
        return None

    def _list_storage_files_complete(self, prefix: str) -> List[Path]:
        """List all files under a prefix, using paginated Oracle listing when available."""
        try:
            if hasattr(self.storage, "list_files_paginated"):
                return list(self.storage.list_files_paginated(prefix, "*"))
            return list(self.storage.list_files(prefix, "*"))
        except Exception as exc:
            logger.warning(f"Failed to list storage prefix {prefix}: {exc}")
            return []

    def _static_data_is_complete(self, data: Optional[Dict[str, Any]], glycan_name: Optional[str] = None) -> bool:
        """Return True when static data is usable for incremental completion checks."""
        archetype = data.get("archetype", {}) if isinstance(data, dict) else {}
        if not isinstance(archetype, dict):
            return False

        name = str(archetype.get("name") or archetype.get("glycam") or "").strip()
        gs_id = str(archetype.get("ID") or "").strip()
        glytoucan = str(archetype.get("glytoucan") or "").strip()
        wurcs = str(archetype.get("wurcs") or "").strip()
        if not name or not gs_id:
            return False
        if glycan_name and not self._matches_glycan_name(archetype, glycan_name):
            return False
        return bool(glytoucan and glytoucan.lower() != "null") or bool(wurcs and wurcs.lower() != "null")

    def _build_fast_completion_indexes(self) -> None:
        """Build in-memory indexes used by the incremental fast-skip path."""
        if self._fast_indexes_built:
            return

        started_at = time.perf_counter()
        self._remote_process_complete = set()
        self._remote_static_data_dirs = set()
        self._remote_static_index = {}
        self._pocketbase_by_glycan = {}

        if self.pb_client is not None:
            cache = getattr(self.pb_client, "_cache_by_name", {})
            if isinstance(cache, dict):
                self._pocketbase_by_glycan = {
                    str(name): record
                    for name, record in cache.items()
                    if name and isinstance(record, dict)
                }

        process_base = self._remote_process_base()
        for obj in self._list_storage_files_complete(process_base):
            key = self._normalize_storage_key(obj)
            if not key.endswith("/output/info.json"):
                continue
            rel = key[len(process_base.rstrip("/") + "/"):] if key.startswith(process_base.rstrip("/") + "/") else key
            glycan_name = rel.split("/", 1)[0]
            if glycan_name:
                self._remote_process_complete.add(glycan_name)

        output_base = self._remote_output_base()
        for obj in self._list_storage_files_complete(output_base):
            key = self._normalize_storage_key(obj)
            if not key.endswith("/data.json"):
                continue
            static_dir = key[:-10]
            if Path(static_dir).name.startswith("GS"):
                self._remote_static_data_dirs.add(static_dir)

        local_static_loaded = 0
        local_base = Path(str(self.output_dir))
        if local_base.exists():
            for data_json in local_base.glob("GS*/data.json"):
                try:
                    with open(data_json, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue
                archetype = data.get("archetype", {}) if isinstance(data, dict) else {}
                if not isinstance(archetype, dict):
                    continue
                name = str(archetype.get("name") or archetype.get("glycam") or "").strip()
                if name and self._static_data_is_complete(data, glycan_name=name) and name not in self._remote_static_index:
                    self._remote_static_index[name] = str(data_json.parent)
                    self._cache_static_output_dir(name, str(data_json.parent))
                    local_static_loaded += 1

        missing_glytoucan_static_dirs: Set[str] = set()
        for glycan_name, record in self._pocketbase_by_glycan.items():
            gs_id = str(record.get("glycoshape_id") or "").strip()
            glytoucan = str(record.get("glytoucan_id") or "").strip()
            if not gs_id:
                continue
            static_dir = f"{output_base}/{gs_id}"
            if static_dir in self._remote_static_data_dirs:
                if glytoucan:
                    self._remote_static_index[glycan_name] = static_dir
                    self._cache_static_output_dir(glycan_name, static_dir)
                else:
                    missing_glytoucan_static_dirs.add(static_dir)

        unmatched_static_dirs = self._remote_static_data_dirs - set(self._remote_static_index.values())
        priority_static_dirs = missing_glytoucan_static_dirs & unmatched_static_dirs
        if not self._pocketbase_by_glycan:
            logger.warning(
                "PocketBase index is unavailable; reading %d remote static data.json files for fallback index",
                len(unmatched_static_dirs),
            )
        elif len(unmatched_static_dirs) > 100:
            logger.warning(
                "Skipping broad static data.json fallback scan for %d unmatched static folders; verifying %d PocketBase records without GlyTouCan",
                len(unmatched_static_dirs),
                len(priority_static_dirs),
            )
            unmatched_static_dirs = priority_static_dirs
        elif unmatched_static_dirs:
            logger.info(f"Reading {len(unmatched_static_dirs)} unmatched static data.json files for fallback index")

        def load_static_index_entry(static_dir: str) -> tuple[str, Optional[str]]:
            data = self._load_json_from_storage(f"{static_dir}/data.json")
            archetype = data.get("archetype", {}) if isinstance(data, dict) else {}
            if not isinstance(archetype, dict):
                return static_dir, None
            name = str(archetype.get("name") or archetype.get("glycam") or "").strip()
            if name and self._static_data_is_complete(data, glycan_name=name):
                return static_dir, name
            return static_dir, None

        fallback_dirs = sorted(unmatched_static_dirs)
        if fallback_dirs:
            max_workers = min(16, max(1, int(getattr(config, "download_workers", 4))))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {executor.submit(load_static_index_entry, static_dir): static_dir for static_dir in fallback_dirs}
                for future in as_completed(future_map):
                    try:
                        static_dir, name = future.result()
                    except Exception:
                        continue
                    if name and name not in self._remote_static_index:
                        self._remote_static_index[name] = static_dir
                        self._cache_static_output_dir(name, static_dir)

        self._fast_indexes_built = True
        logger.info(
            "Built fast completion indexes in %.3fs: %d process-complete glycans, %d static data.json files, %d static glycan mappings, %d PocketBase records",
            time.perf_counter() - started_at,
            len(self._remote_process_complete),
            len(self._remote_static_data_dirs),
            len(self._remote_static_index or {}),
            len(self._pocketbase_by_glycan),
        )
        if local_static_loaded:
            logger.info(f"Loaded {local_static_loaded} glycan mappings from local static mirror")

    def _ensure_fast_completion_indexes(self) -> None:
        if not self._fast_indexes_built:
            self._build_fast_completion_indexes()

    def _has_required_submission_metadata(self, glycan_name: str) -> bool:
        """Return True when glystatic will have the required stable GlycoShape ID."""
        record = self._get_prefetched_pocketbase_record(glycan_name)
        if record:
            gs_id = str(record.get("glycoshape_id") or "").strip()
            glytoucan = str(record.get("glytoucan_id") or "").strip()
            if gs_id:
                if not glytoucan:
                    logger.info(f"{glycan_name}: GlyTouCan ID missing; continuing so WURCS can be generated for registration")
                return True

        try:
            metadata = get_glycan_metadata(glycan_name)
        except Exception as exc:
            logger.warning(f"Skipping {glycan_name}: required submission metadata was not found ({exc})")
            return False

        gs_id = str((metadata or {}).get("ID") or "").strip()
        glytoucan = str((metadata or {}).get("glytoucan_id") or "").strip()
        if gs_id and gs_id != "0":
            if not glytoucan or glytoucan == "0":
                logger.info(f"{glycan_name}: GlyTouCan ID missing; continuing so WURCS can be generated for registration")
            return True

        logger.warning(
            f"Skipping {glycan_name}: required submission metadata is incomplete "
            f"(ID={gs_id or 'missing'}, GlyTouCan={glytoucan or 'missing'})"
        )
        return False

    def _load_json_from_storage(self, path: str) -> Optional[Dict[str, Any]]:
        """Read a JSON object from the active storage backend."""
        try:
            with self.storage.open(path, 'r') as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    @staticmethod
    def _matches_glycan_name(archetype: Dict[str, Any], glycan_name: str, substring_match: bool = False) -> bool:
        """Check whether an archetype payload maps to the requested glycan."""
        name_field = str(archetype.get("name") or archetype.get("glycam") or "").strip()
        if not name_field:
            return False
        if substring_match:
            return glycan_name.lower() in name_field.lower()
        return name_field == glycan_name

    def _find_static_output_dir(
        self,
        glycan_name: str,
        *,
        substring_match: bool = False,
        include_local_oracle_fallback: bool = False,
        upload_local_if_oracle: bool = False,
    ) -> Optional[str]:
        """Locate the per-glycan GS output directory for the given glycan name."""
        cached = self._get_cached_static_output_dir(glycan_name)
        if cached is not None:
            return cached

        base = self._remote_output_base()
        try:
            candidates = self.storage.list_files(base, "GS*")
        except Exception:
            candidates = []

        seen: Set[str] = set()
        for cand in candidates:
            key_raw = self._normalize_storage_key(cand)
            key = key_raw[:-10] if key_raw.endswith("/data.json") else key_raw
            if key in seen:
                continue
            seen.add(key)

            data_json = f"{key}/data.json"
            if not self.storage.exists(data_json):
                continue

            data = self._load_json_from_storage(data_json)
            archetype = data.get("archetype", {}) if isinstance(data, dict) else {}
            if isinstance(archetype, dict) and self._matches_glycan_name(archetype, glycan_name, substring_match=substring_match):
                self._cache_static_output_dir(glycan_name, key)
                return key

        if config.use_oracle_storage and include_local_oracle_fallback:
            local_base = Path(str(self.output_dir))
            if local_base.exists():
                for data_json in local_base.glob("GS*/data.json"):
                    try:
                        with open(data_json, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                    except Exception:
                        continue

                    archetype = data.get("archetype", {}) if isinstance(data, dict) else {}
                    if not isinstance(archetype, dict):
                        continue
                    if not self._matches_glycan_name(archetype, glycan_name, substring_match=substring_match):
                        continue

                    if upload_local_if_oracle:
                        remote_prefix = f"{self._remote_output_base()}/{data_json.parent.name}"
                        self.storage.upload_dir(data_json.parent, remote_prefix, skip_existing=False)
                        logger.info(f"Uploaded local GS folder to Oracle: {remote_prefix}")
                        self._cache_static_output_dir(glycan_name, str(data_json.parent))
                        return remote_prefix

                    self._cache_static_output_dir(glycan_name, str(data_json.parent))
                    return str(data_json.parent)

        return None

    def _find_remote_static_output_dir(self, glycan_name: str) -> Optional[str]:
        """Locate a remote/static GS directory using cheap exact-name checks."""
        cached = self._known_static_output_dirs.get(glycan_name)
        if cached and self._normalize_storage_key(cached).startswith(f"{self._remote_output_base()}/"):
            return cached

        if self._remote_static_index is not None:
            remote_dir = self._remote_static_index.get(glycan_name)
            if remote_dir:
                self._cache_static_output_dir(glycan_name, remote_dir)
                return remote_dir

        candidates: List[str] = []
        gs_id = self._get_pocketbase_glycoshape_id(glycan_name)
        if gs_id:
            candidates.append(f"{self._remote_output_base()}/{gs_id}")

        for candidate in candidates:
            if self._remote_static_data_dirs and candidate not in self._remote_static_data_dirs:
                continue
            if not self._remote_static_data_dirs and not self.storage.exists(f"{candidate}/data.json"):
                continue
            data = self._load_json_from_storage(f"{candidate}/data.json")
            archetype = data.get("archetype", {}) if isinstance(data, dict) else {}
            if isinstance(archetype, dict) and self._static_data_is_complete(data, glycan_name=glycan_name):
                self._cache_static_output_dir(glycan_name, candidate)
                return candidate

        if self._remote_static_index is None:
            self._remote_static_index = {}
            try:
                dirs = self.storage.list_dirs(self._remote_output_base())
            except Exception:
                dirs = []

            for entry in dirs:
                key = self._normalize_storage_key(entry)
                if not Path(key).name.startswith("GS"):
                    continue
                data = self._load_json_from_storage(f"{key}/data.json")
                archetype = data.get("archetype", {}) if isinstance(data, dict) else {}
                if not isinstance(archetype, dict):
                    continue
                name = str(archetype.get("name") or archetype.get("glycam") or "").strip()
                if name and self._static_data_is_complete(data, glycan_name=name):
                    self._remote_static_index[name] = key

        remote_dir = self._remote_static_index.get(glycan_name)
        if remote_dir:
            self._cache_static_output_dir(glycan_name, remote_dir)
        return remote_dir

    def _sync_remote_static_output_to_local(self, remote_dir: str, glycan_name: str, force: bool = False) -> Optional[str]:
        """Mirror a validated Oracle static GS directory into the local output directory."""
        if not config.use_oracle_storage:
            return remote_dir

        remote_key = self._normalize_storage_key(remote_dir)
        remote_base = f"{self._remote_output_base()}/"
        if not remote_key.startswith(remote_base):
            return remote_dir

        gs_name = Path(remote_key).name
        local_dir = Path(str(self.output_dir)) / gs_name
        if not force and (local_dir / "data.json").exists():
            self._cache_static_output_dir(glycan_name, str(local_dir))
            return str(local_dir)

        try:
            objects = self.storage.backend.list_files(remote_key, "*")
        except Exception as exc:
            logger.warning(f"Failed to list remote static output for {glycan_name} in {remote_key}: {exc}")
            return remote_dir

        if not objects:
            return remote_dir

        local_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{remote_key}/"

        for obj in objects:
            obj_key = self._normalize_storage_key(obj)
            if not obj_key.startswith(prefix):
                continue
            rel_path = obj_key[len(prefix):]
            if not rel_path:
                continue
            target_path = local_dir / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            content = self.storage.backend.read_binary(obj_key)
            target_path.write_bytes(content)

        logger.info(f"Synchronized Oracle static output for {glycan_name}: {remote_key} -> {local_dir}")
        return str(local_dir)

    def _local_static_output_dir_for_refresh(self, output_dir: str, glycan_name: str) -> str:
        """Return a local static directory suitable for refresh-static metadata writes."""
        if not config.use_oracle_storage:
            return output_dir

        normalized = self._normalize_storage_key(output_dir)
        if normalized.startswith(f"{self._remote_output_base()}/"):
            synced = self._sync_remote_static_output_to_local(normalized, glycan_name, force=False)
            if synced:
                self._cache_static_output_dir(glycan_name, synced)
                return synced

        self._cache_static_output_dir(glycan_name, output_dir)
        return output_dir

    def _upload_local_static_output_if_oracle(self, output_dir: str, glycan_name: str) -> None:
        """Upload a repaired local GS static directory back to Oracle storage."""
        if not config.use_oracle_storage:
            return

        local_dir = Path(str(output_dir))
        if not local_dir.exists() or not (local_dir / "data.json").exists():
            logger.debug(f"Skipping Oracle static upload for {glycan_name}; local data.json not found in {local_dir}")
            return

        remote_prefix = f"{self._remote_output_base()}/{local_dir.name}"
        try:
            self.storage.upload_dir(local_dir, remote_prefix, skip_existing=False)
            logger.info(f"Uploaded refreshed static output to Oracle for {glycan_name}: {remote_prefix}")
        except Exception as exc:
            logger.warning(f"Failed to upload refreshed static output for {glycan_name}: {exc}")

    def _sync_all_remote_static_to_local(self) -> None:
        """Mirror all remote static GS folders locally for database-wide finalization."""
        if not config.use_oracle_storage:
            return

        try:
            dirs = self.storage.list_dirs(self._remote_output_base())
        except Exception as exc:
            logger.warning(f"Failed to list remote static directories for finalization sync: {exc}")
            return

        synced = 0
        for entry in dirs:
            key = self._normalize_storage_key(entry)
            if not Path(key).name.startswith("GS"):
                continue
            glycan_name = ""
            data = self._load_json_from_storage(f"{key}/data.json")
            archetype = data.get("archetype", {}) if isinstance(data, dict) else {}
            if isinstance(archetype, dict):
                glycan_name = str(archetype.get("name") or archetype.get("glycam") or "").strip()
            local_dir = self._sync_remote_static_output_to_local(key, glycan_name or Path(key).name, force=self.mode != "refresh-static")
            if local_dir:
                synced += 1
                if glycan_name:
                    self._cache_static_output_dir(glycan_name, local_dir)

        logger.info(f"Synchronized {synced} remote static folders locally for finalization")

    def _fast_completed_output_dir(self, glycan_name: str, sync_static: bool = False) -> Optional[str]:
        """Return static output dir when both remote process and static artifacts are present."""
        if config.use_oracle_storage:
            self._ensure_fast_completion_indexes()
            if glycan_name not in self._remote_process_complete:
                return None
        elif not self.check_v2_completion(glycan_name):
            return None

        static_dir = self._find_remote_static_output_dir(glycan_name)
        if static_dir is None:
            return None

        if sync_static:
            synced = self._sync_remote_static_output_to_local(static_dir, glycan_name)
            if synced:
                return synced
        return static_dir

    def _sync_remote_process_dir_to_local(self, glycan_name: str) -> Optional[str]:
        """Mirror an Oracle process directory into the local process tree.

        This is used when v2 has already been uploaded to Oracle and the
        local process directory was cleaned up, but glystatic still needs to
        read the intermediate analysis files from disk.
        """
        if not config.use_oracle_storage:
            return None

        remote_key = f"{self._remote_process_base()}/{glycan_name}"
        local_dir = Path(str(self.process_dir)) / glycan_name

        try:
            objects = self.storage.backend.list_files(remote_key, "*")
        except Exception as exc:
            logger.warning(f"Failed to list remote process output for {glycan_name} in {remote_key}: {exc}")
            return None

        if not objects:
            return None

        local_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{remote_key}/"

        for obj in objects:
            obj_key = self._normalize_storage_key(obj)
            if not obj_key.startswith(prefix):
                continue
            rel_path = obj_key[len(prefix):]
            if not rel_path:
                continue
            target_path = local_dir / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            content = self.storage.backend.read_binary(obj_key)
            target_path.write_bytes(content)

        logger.info(f"Synchronized Oracle process output for {glycan_name}: {remote_key} -> {local_dir}")
        return str(local_dir)

    def _get_required_v2_steps(self) -> List[str]:
        """Return the set of v2 steps required for a complete run."""
        steps = ["store", "pca", "clustering", "create_info", "structures", "export"]
        if V2_RDKIT_CHEM is not None:
            steps.extend(["torsions", "cluster_stats", "plots"])
        return steps

    def _get_log_tail_required_files(self) -> List[str]:
        """Return the fixed per-glycan static file manifest for log_tail summaries."""
        required_files = list(self.LOG_TAIL_TOP_LEVEL_FILES)
        required_files.extend(f"output/{name}" for name in self.LOG_TAIL_OUTPUT_FILES)
        for level_name in self.REQUIRED_GLYSTATIC_LEVELS:
            required_files.extend(f"output/{level_name}/{rel_path}" for rel_path in self.REQUIRED_GLYSTATIC_LEVEL_FILES)
        return required_files

    def _collect_static_output_files(self, output_dir: Optional[str]) -> List[str]:
        """Return all files currently present under a GS output directory."""
        if not output_dir:
            return []

        normalized_output_dir = self._normalize_storage_key(output_dir)
        local_root = Path(output_dir)
        if local_root.exists():
            return sorted(p.relative_to(local_root).as_posix() for p in local_root.rglob("*") if p.is_file())

        if config.use_oracle_storage:
            try:
                objects = self.storage.list_files(normalized_output_dir, "*")
            except Exception:
                return []

            prefix = f"{normalized_output_dir}/"
            files = set()
            for obj in objects:
                raw = self._normalize_storage_key(obj)
                if not raw.startswith(prefix) or raw.endswith("/"):
                    continue
                rel_path = raw[len(prefix):]
                if rel_path:
                    files.add(rel_path)
            return sorted(files)

        root = Path(output_dir)
        if not root.exists():
            return []
        return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())

    def _build_log_tail_tree(
        self,
        root_label: str,
        actual_files: List[str],
        required_files: List[str],
    ) -> List[str]:
        """Render a merged required-vs-actual ASCII tree for the static folder."""
        actual_set = set(actual_files)
        required_set = set(required_files)
        nodes: Dict[str, Dict[str, Any]] = {
            "": {"kind": "dir", "children": set(), "present": bool(actual_set)}
        }

        def ensure_dir(dir_path: str, present: bool = False) -> None:
            if dir_path not in nodes:
                nodes[dir_path] = {"kind": "dir", "children": set(), "present": False}
            if present:
                nodes[dir_path]["present"] = True
            if dir_path:
                parent = dir_path.rsplit("/", 1)[0] if "/" in dir_path else ""
                ensure_dir(parent, present=present)
                nodes[parent]["children"].add(dir_path)

        def add_file(rel_path: str, *, required: bool, present: bool) -> None:
            parent = rel_path.rsplit("/", 1)[0] if "/" in rel_path else ""
            ensure_dir(parent, present=present)
            node = nodes.setdefault(
                rel_path,
                {"kind": "file", "children": set(), "required": False, "present": False},
            )
            node["required"] = node.get("required", False) or required
            node["present"] = node.get("present", False) or present
            nodes[parent]["children"].add(rel_path)

        for rel_path in sorted(required_set):
            add_file(rel_path, required=True, present=rel_path in actual_set)
        for rel_path in sorted(actual_set - required_set):
            add_file(rel_path, required=False, present=True)

        marker_cache: Dict[str, str] = {}

        def node_marker(path: str) -> str:
            if path in marker_cache:
                return marker_cache[path]

            node = nodes[path]
            if node["kind"] == "file":
                marker = "[MISSING]" if node.get("required") and not node.get("present") else "[OK]"
            else:
                child_markers = [node_marker(child) for child in node["children"]]
                if any(child_marker == "[MISSING]" for child_marker in child_markers):
                    marker = "[MISSING]"
                elif node.get("present") or child_markers:
                    marker = "[OK]"
                else:
                    marker = "[MISSING]"

            marker_cache[path] = marker
            return marker

        def sort_key(path: str) -> tuple:
            node = nodes[path]
            return (0 if node["kind"] == "dir" else 1, path.split("/")[-1])

        lines = [f"{root_label}/ {node_marker('')}"]

        def render(path: str, prefix: str) -> None:
            children = sorted(nodes[path]["children"], key=sort_key)
            for idx, child in enumerate(children):
                is_last = idx == len(children) - 1
                connector = "`-- " if is_last else "|-- "
                child_node = nodes[child]
                name = child.split("/")[-1] + ("/" if child_node["kind"] == "dir" else "")
                lines.append(f"{prefix}{connector}{name} {node_marker(child)}")
                if child_node["kind"] == "dir":
                    render(child, prefix + ("    " if is_last else "|   "))

        render("", "")
        return lines

    def _build_log_tail_summary(
        self,
        glycan_name: str,
        results: Dict[str, bool],
        glymeta_success: Optional[bool],
    ) -> tuple[str, str]:
        """Build the final PocketBase log_tail payload for a glycan."""
        v2_status = get_glycan_status(glycan_name, str(self.data_dir), str(self.process_dir))

        submission_record = None
        if self.pb_client is not None:
            try:
                submission_record = self.pb_client.get_record_by_glycam_name(glycan_name)
            except Exception as exc:
                logger.warning(f"Failed to resolve PocketBase submission record for {glycan_name}: {exc}")

        output_dir = self._find_static_output_dir(
            glycan_name,
            include_local_oracle_fallback=True,
        )
        actual_files = self._collect_static_output_files(output_dir)
        required_files = self._get_log_tail_required_files()
        actual_set = set(actual_files)
        required_set = set(required_files)
        missing_files = sorted(required_set - actual_set)
        extra_files = sorted(actual_set - required_set)

        output_data = self._load_json_from_storage(f"{output_dir}/data.json") if output_dir else None
        archetype = output_data.get("archetype", {}) if isinstance(output_data, dict) else {}
        if not isinstance(archetype, dict):
            archetype = {}

        glycoshape_id = str(
            archetype.get("ID")
            or (submission_record or {}).get("glycoshape_id")
            or (Path(output_dir).name if output_dir else "")
            or ""
        ).strip()
        glytoucan_id = str(
            archetype.get("glytoucan")
            or (submission_record or {}).get("glytoucan_id")
            or ""
        ).strip()

        validation_errors = (
            self._validate_glystatic_output_dir(output_dir, glycan_name=glycan_name)
            if output_dir
            else [f"missing glystatic directory: {glycoshape_id or glycan_name}"]
        )

        input_files_ok = bool(v2_status.get("input_files_exist"))
        step_status_map = dict(v2_status.get("steps_status", {}))
        required_v2_steps = self._get_required_v2_steps()
        all_required_v2_ok = input_files_ok and all(step_status_map.get(step, False) for step in required_v2_steps)
        glystatic_ok = bool(results.get("glystatic")) and not validation_errors

        if all_required_v2_ok and glystatic_ok and not missing_files:
            verdict = "COMPLETE" if glymeta_success is True else "WARNING"
        else:
            verdict = "FAILED"

        def add_issue(issues: List[str], message: str) -> None:
            if message not in issues:
                issues.append(message)

        issues: List[str] = []
        if not input_files_ok:
            add_issue(issues, "[MISSING] input files not found in the data directory")
        for step in required_v2_steps:
            if not step_status_map.get(step, False):
                add_issue(issues, f"[MISSING] v2 step incomplete: {step}")
        if not results.get("glystatic"):
            add_issue(issues, "[MISSING] glystatic step incomplete")
        if not output_dir:
            add_issue(issues, "[MISSING] per-glycan static output directory not found")
        for rel_path in missing_files:
            add_issue(issues, f"[MISSING] {rel_path}")
        for validation_error in validation_errors:
            if validation_error.startswith("missing "):
                continue
            add_issue(issues, f"[WARN] {validation_error}")
        if glymeta_success is False:
            add_issue(issues, "[WARN] glymeta step failed")
        elif glymeta_success is None and results.get("v2") and results.get("glystatic"):
            add_issue(issues, "[WARN] glymeta step did not report a final status")

        root_label = glycoshape_id or (Path(output_dir).name if output_dir else "static-output")
        tree_lines = self._build_log_tail_tree(root_label, actual_files, required_files)

        lines = [
            "GAP RUN SUMMARY",
            f"glycan: {glycan_name}",
            f"glycoshape_id: {glycoshape_id or 'unknown'}",
            f"glytoucan_id: {glytoucan_id or 'unknown'}",
            f"finished_at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"verdict: {verdict}",
            f"100_percent_complete: {'YES' if verdict == 'COMPLETE' else 'NO'}",
            "",
            "STEPS",
        ]

        lines.append(f"- input_files ................. {'[OK]' if input_files_ok else '[MISSING]'}")
        for step_key, label in self.LOG_TAIL_STEP_ORDER[1:]:
            lines.append(f"- {label:<27} {'[OK]' if step_status_map.get(step_key, False) else '[MISSING]'}")
        lines.append(f"- {'glystatic':<27} {'[OK]' if glystatic_ok else '[MISSING]'}")
        lines.append(f"- {'glymeta':<27} {'[OK]' if glymeta_success is True else '[WARN]'}")
        lines.extend([
            "",
            "COUNTS",
            f"- actual_files_total ......... {len(actual_files)}",
            f"- required_files ............. {len(required_files)}",
            f"- present_required_files ..... {len(required_set & actual_set)}",
            f"- missing_required_files ..... {len(missing_files)}",
            f"- extra_files ................ {len(extra_files)}",
            "",
            "STATIC TREE",
            *tree_lines,
            "",
            "ISSUES",
        ])

        if issues:
            lines.extend(f"- {issue}" for issue in issues)
        else:
            lines.append("- none")

        return "\n".join(lines), glycoshape_id

    def _write_log_tail_summary(
        self,
        glycan_name: str,
        results: Dict[str, bool],
        glymeta_success: Optional[bool],
    ) -> None:
        """Write the final per-glycan log_tail summary back to PocketBase."""
        if self.pb_client is None or not self.pb_client.is_available():
            return

        log_tail, glycoshape_id = self._build_log_tail_summary(glycan_name, results, glymeta_success)
        updated = self.pb_client.update_submission_log_tail(
            log_tail,
            glycoshape_id=glycoshape_id,
            glycam_name=glycan_name,
        )
        if updated is None:
            logger.warning(f"Could not update log_tail for {glycan_name}: submission record not found")
        else:
            logger.info(f"Updated PocketBase log_tail for {glycan_name}")
        
    def discover_glycans(self) -> List[str]:
        """Discover glycan directories in data_dir.
        
        Returns:
            List of glycan directory names
        """
        glycan_dirs = []
        
        # For Oracle storage, directory existence checking is different
        # We'll try to proceed and handle missing gracefully
        logger.info(f"Discovering glycans in data directory: {self.data_dir}")
        if not self.storage.exists(self.data_dir) and not config.use_oracle_storage:
            logger.error(f"Data directory does not exist: {self.data_dir}")
            return glycan_dirs
            
        # List items in data directory - different approach for Oracle vs Local
        if config.use_oracle_storage:
            logger.info("Using Oracle storage - indexing glycan input files under data prefix")
            try:
                objects = self._list_storage_files_complete(str(self.data_dir))
            except Exception as e:
                logger.error(f"Failed to list files under {self.data_dir}: {e}")
                objects = []

            data_base = self._normalize_storage_key(self.data_dir).strip("/")
            files_by_glycan: Dict[str, Set[str]] = {}
            for obj in objects:
                key = self._normalize_storage_key(obj)
                rel = key[len(data_base) + 1:] if data_base and key.startswith(f"{data_base}/") else key
                parts = rel.split("/", 1)
                if len(parts) != 2:
                    continue
                glycan_name, file_name = parts
                if not glycan_name or "/" in file_name:
                    continue
                files_by_glycan.setdefault(glycan_name, set()).add(file_name)

            for cand, names in sorted(files_by_glycan.items()):
                ok_named = f"{cand}.pdb" in names and f"{cand}.mol2" in names
                ok_alt = "simulation.pdb" in names and "structure.mol2" in names
                if ok_named or ok_alt:
                    glycan_dirs.append(cand)
                    logger.debug(f"Found glycan: {cand}")
                else:
                    logger.debug(f"Skipping {cand}: missing required files (.pdb/.mol2 or simulation.pdb/structure.mol2)")
        else:
            # Local storage - use normal directory listing
            items = self.storage.list_files(self.data_dir, "*")
            
            for item in items:
                item_name = item.name if hasattr(item, 'name') else str(item).split('/')[-1]
                
                # Check if it's a directory
                is_directory = False
                if hasattr(item, 'is_dir'):
                    is_directory = item.is_dir()
                else:
                    is_directory = self.storage.is_file(item) is False and '/' not in str(item).split('/')[-1]
                
                if is_directory:
                    # Check for required files
                    pdb_file = f"{item_name}/{item_name}.pdb"
                    mol2_file = f"{item_name}/{item_name}.mol2"
                    
                    if self.storage.exists(pdb_file, self.data_dir) and self.storage.exists(mol2_file, self.data_dir):
                        glycan_dirs.append(item_name)
                        logger.info(f"Found glycan: {item_name}")
                    else:
                        logger.warning(f"Skipping {item_name}: missing required files (.pdb and .mol2)")
                    
        logger.info(f"Discovered {len(glycan_dirs)} glycan directories")
        return sorted(glycan_dirs)
    
    def run_v2_analysis(self, glycan_name: str) -> bool:
        """Run v2 analysis for a single glycan.
        
        Args:
            glycan_name: Name of the glycan directory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Running v2 analysis for {glycan_name}")
            
            # Use the new v2 API directly with separate data and process directories
            results = run_glycan_analysis(
                glycan_name=glycan_name,
                data_dir=str(self.data_dir),
                process_dir=str(self.process_dir),
                force_update=self.force_update
            )
            
            if results['success']:
                logger.info(f"Successfully completed v2 analysis for {glycan_name}")
                logger.info(f"Completed steps: {', '.join(results['steps_completed'])}")
                return True
            else:
                logger.error(f"v2 analysis failed for {glycan_name}: {results['error_message']}")
                if results['steps_completed']:
                    logger.info(f"Partial completion - steps done: {', '.join(results['steps_completed'])}")
                return False
                
        except Exception as e:
            logger.error(f"Error running v2 analysis for {glycan_name}: {str(e)}")
            return False

    def _validate_glystatic_output_dir(self, glystatic_dir: str, glycan_name: Optional[str] = None) -> List[str]:
        """Validate the expected final glystatic output structure."""
        validation_errors: List[str] = []
        data_json = f"{glystatic_dir}/data.json"

        if not self.storage.exists(glystatic_dir):
            return [f"missing glystatic directory: {glystatic_dir}"]

        if not self.storage.exists(data_json):
            return [f"missing data.json in {glystatic_dir}"]

        try:
            with self.storage.open(data_json, 'r') as f:
                data = json.load(f)
        except Exception as e:
            return [f"failed to read data.json: {e}"]

        if not isinstance(data, dict):
            validation_errors.append("data.json is not a dictionary")

        archetype = data.get('archetype', {}) if isinstance(data, dict) else {}
        if not isinstance(archetype, dict):
            validation_errors.append("missing or invalid archetype section")
            archetype = {}

        required_fields = ["ID", "name", "iupac"]
        missing_fields = [field for field in required_fields if not archetype.get(field)]
        if missing_fields:
            validation_errors.append(f"missing required archetype fields: {missing_fields}")

        if glycan_name and archetype.get("name") != glycan_name and archetype.get("glycam") != glycan_name:
            validation_errors.append(
                f"archetype name mismatch: expected '{glycan_name}', found '{archetype.get('name') or archetype.get('glycam', '')}'"
            )

        glytoucan_id = (archetype.get("glytoucan", "") or "").strip()
        wurcs = (archetype.get("wurcs", "") or "").strip()
        if (not glytoucan_id or glytoucan_id.lower() == "null") and (not wurcs or wurcs.lower() == "null"):
            validation_errors.append("missing both GlyTouCan ID and WURCS")

        output_dir = f"{glystatic_dir}/output"
        if not self.storage.exists(output_dir):
            validation_errors.append("missing output/ folder")
            return validation_errors

        for rel_path in self.REQUIRED_GLYSTATIC_ANALYSIS_FILES:
            if not self.storage.exists(f"{output_dir}/{rel_path}"):
                validation_errors.append(f"missing output file: {rel_path}")

        for level_name in self.REQUIRED_GLYSTATIC_LEVELS:
            level_dir = f"{output_dir}/{level_name}"
            if not self.storage.exists(level_dir):
                validation_errors.append(f"missing {level_name} structures")
                continue

            for rel_path in self.REQUIRED_GLYSTATIC_LEVEL_FILES:
                if not self.storage.exists(f"{level_dir}/{rel_path}"):
                    validation_errors.append(f"missing {level_name}/{rel_path}")

        return validation_errors
    
    def validate_output_structure(self, glycan_name: str) -> bool:
        """Validate that a glycan has the expected output structure.
        
        Args:
            glycan_name: Name of the glycan
            
        Returns:
            True if structure is valid, False otherwise
        """
        output_glycan_dir = f"{self.output_dir}/{glycan_name}"
        data_json = f"{output_glycan_dir}/data.json"
        
        if not self.storage.exists(output_glycan_dir):
            logger.debug(f"Output directory missing for {glycan_name}")
            return False
            
        if not self.storage.exists(data_json):
            logger.debug(f"data.json missing for {glycan_name}")
            return False
            
        try:
            validation_errors = self._validate_glystatic_output_dir(output_glycan_dir, glycan_name=glycan_name)
            if validation_errors:
                for err in validation_errors:
                    logger.debug(f"{glycan_name}: {err}")
                return False

            logger.debug(f"Output structure validated for {glycan_name}")
            return True
        except Exception as e:
            logger.debug(f"Error validating output for {glycan_name}: {str(e)}")
            return False
    
    def check_glystatic_simple(self, glycan_name: str) -> bool:
        """Simple check if glystatic processing should be skipped.
        
        This is used when GLYCOSHAPE_DB_UPDATE is False - we only need to check
        if there's a valid data.json with a GlyTouCan ID or WURCS, not the full file structure.
        
        Args:
            glycan_name: Name of the glycan
            
        Returns:
            True if a valid data.json exists with GlyTouCan ID or WURCS, False otherwise
        """
        try:
            if config.use_oracle_storage and self._fast_indexes_built:
                remote_dir = (self._remote_static_index or {}).get(glycan_name)
                if remote_dir:
                    self._cache_static_output_dir(glycan_name, remote_dir)
                    return True
                return False

            remote_dir = self._find_remote_static_output_dir(glycan_name)
            if remote_dir is not None:
                return True

            cached_dir = self._get_cached_static_output_dir(glycan_name)
            if cached_dir is not None:
                data = self._load_json_from_storage(f"{cached_dir}/data.json")
                archetype = data.get("archetype", {}) if isinstance(data, dict) else {}
                if isinstance(archetype, dict) and self._matches_glycan_name(archetype, glycan_name):
                    return True

            # Look for any GS folder that matches this glycan name
            base = self._remote_output_base()
            candidates = self.storage.list_files(base, "GS*")
            
            for candidate in candidates:
                candidate_name = candidate.name if hasattr(candidate, 'name') else str(candidate).split('/')[-1]
                
                # Check if it's a directory
                if not self.storage.is_dir(candidate):
                    continue

                data_json = f"{candidate}/data.json"
                if not self.storage.exists(data_json):
                    continue

                # Try to read the data.json
                try:
                    with self.storage.open(data_json, 'r') as f:
                        data = json.load(f)
                except Exception:
                    continue

                # Check if archetype name matches exactly
                archetype = data.get('archetype', {})
                if not isinstance(archetype, dict):
                    continue
                    
                name_field = archetype.get('name') or archetype.get('glycam', '')
                if name_field != glycan_name:
                    continue
                
                validation_errors = self._validate_glystatic_output_dir(str(candidate), glycan_name=glycan_name)
                if not validation_errors:
                    glytoucan_id = archetype.get("glytoucan", "").strip()
                    self._cache_static_output_dir(glycan_name, str(candidate))
                    logger.debug(f"Found complete glystatic output for {glycan_name} in {candidate_name} (GlyTouCan: {glytoucan_id or 'missing'}, WURCS accepted)")
                    return True
            
            logger.debug(f"No valid data.json with GlyTouCan ID or WURCS found for {glycan_name}")
            return False

        except Exception as e:
            logger.debug(f"Error in simple glystatic check for {glycan_name}: {e}")
            return False
    
    def check_glystatic_completion(self, glycan_name: str) -> bool:
        """Check if glystatic processing is complete for a glycan.
        
        This method first checks for intermediate outputs in process_dir, then
        for final outputs in output_dir. Returns True if either complete
        intermediate processing or final database entries exist with valid structure.
        
        Args:
            glycan_name: Name of the glycan
            
        Returns:
            True if glystatic processing is complete, False otherwise
        """
        try:
            if config.use_oracle_storage and self._fast_indexes_built:
                remote_dir = (self._remote_static_index or {}).get(glycan_name)
                if remote_dir:
                    self._cache_static_output_dir(glycan_name, remote_dir)
                    return True
                return False

            remote_dir = self._find_remote_static_output_dir(glycan_name)
            if remote_dir is not None:
                return True

            cached_dir = self._get_cached_static_output_dir(glycan_name)
            if cached_dir is not None:
                validation_errors = self._validate_glystatic_output_dir(cached_dir, glycan_name=glycan_name)
                if not validation_errors:
                    return True

            # Use storage abstraction to support Oracle and local modes
            storage = self.storage
            # In Oracle, check completion against remote 'static' prefix
            base = str(self._remote_output_base())
            # List GS* candidates under output dir
            try:
                candidates = storage.list_files(base, "GS*")
            except Exception:
                candidates = []

            matching_key = None
            for cand in candidates:
                cand_key = cand.as_posix() if hasattr(cand, 'as_posix') else str(cand)
                # Normalize: Oracle list_files may return file keys; ensure we use GS folder
                base_key = cand_key
                # Normalize candidates that may be file keys (e.g. '.../GSxxxx/data.json')
                if base_key.endswith('/data.json') or base_key.endswith('data.json'):
                    base_key = str(Path(base_key).parent)
                # Build path to data.json under this candidate
                data_json = f"{base_key}/data.json"
                if not storage.exists(data_json):
                    continue
                try:
                    with storage.open(data_json, 'r') as f:
                        data = json.load(f)
                    archetype = data.get('archetype', {}) if isinstance(data, dict) else {}
                    name_field = archetype.get('name') or archetype.get('glycam', '')
                    if name_field == glycan_name:
                        matching_key = base_key
                        self._cache_static_output_dir(glycan_name, base_key)
                        break
                except Exception:
                    continue

            if not matching_key:
                try:
                    matching_key = self._find_static_output_dir(
                        glycan_name,
                        include_local_oracle_fallback=config.use_oracle_storage,
                        upload_local_if_oracle=False,
                    )
                except Exception as e:
                    logger.debug(f"Local GS scan/upload attempt failed: {e}")

            if not matching_key:
                logger.debug(f"No GS folder found with exact name match for {glycan_name}")
                return False

            validation_errors = self._validate_glystatic_output_dir(matching_key, glycan_name=glycan_name)

            if validation_errors:
                logger.warning(f"Glystatic validation failed for {glycan_name} in {matching_key}. Issues:")
                for err in validation_errors:
                    logger.warning(f"  - {err}")
                return False

            logger.debug(f"Glystatic completion validated in {matching_key} for {glycan_name}")
            return True

        except Exception as e:
            logger.error(f"Error while checking glystatic completion for {glycan_name}: {e}")
            return False
    
    def check_v2_completion(self, glycan_name: str) -> bool:
        """Check if v2 analysis is complete for a glycan.
        
        Args:
            glycan_name: Name of the glycan
            
        Returns:
            True if v2 analysis is complete, False otherwise
        """
        try:
            process_base = config.oracle_process_prefix if config.use_oracle_storage else str(self.process_dir)
            status = get_glycan_status(glycan_name, str(self.data_dir), process_base)
            
            # Check if all critical steps are completed
            critical_steps = self._get_required_v2_steps()
            all_critical_complete = all(
                status['steps_status'].get(step, False) 
                for step in critical_steps
            )
            
            return all_critical_complete
            
        except Exception as e:
            logger.debug(f"Error checking v2 status for {glycan_name}: {str(e)}")
            return False
    
    def run_glystatic(self, glycan_name: str, check_existing: bool = True) -> Dict[str, Any]:
        """Run glystatic processing for a single glycan.
        
        Args:
            glycan_name: Name of the glycan directory
            
        Returns:
            Dict containing success flag and known output paths/IDs
        """
        result: Dict[str, Any] = {
            "success": False,
            "glycoshape_id": None,
            "final_output_dir": None,
            "local_output_dir": None,
        }
        try:
            output_glycan_dir = f"{self.output_dir}/{glycan_name}"
            
            # Check if glystatic processing should be skipped
            if check_existing and not self.force_update:
                if not config.update:
                    # When GLYCOSHAPE_DB_UPDATE is False, use simple check
                    if self.check_glystatic_simple(glycan_name):
                        logger.info(f"Glystatic processing already complete for {glycan_name} (found valid data.json with GlyTouCan ID or WURCS), skipping")
                        cached = self._find_static_output_dir(glycan_name, include_local_oracle_fallback=True)
                        if cached:
                            self._cache_static_output_dir(glycan_name, cached)
                            result["final_output_dir"] = cached
                            result["local_output_dir"] = cached
                        result["success"] = True
                        return result
                else:
                    # When GLYCOSHAPE_DB_UPDATE is True, use full validation
                    if self.check_glystatic_completion(glycan_name):
                        logger.info(f"Glystatic processing already complete for {glycan_name}, skipping")
                        cached = self._find_static_output_dir(glycan_name, include_local_oracle_fallback=True)
                        if cached:
                            self._cache_static_output_dir(glycan_name, cached)
                            result["final_output_dir"] = cached
                            result["local_output_dir"] = cached
                        result["success"] = True
                        return result
                
            logger.info(f"Running glystatic for {glycan_name}")
            
            # Define paths for glystatic processing
            glycan_input_dir = f"{self.data_dir}/{glycan_name}"
            process_glycan_dir = f"{self.process_dir}/{glycan_name}"
            local_process_dir = Path(process_glycan_dir)

            if config.use_oracle_storage and not self.storage.exists(process_glycan_dir):
                synced_dir = self._sync_remote_process_dir_to_local(glycan_name)
                if synced_dir:
                    process_glycan_dir = synced_dir
                    local_process_dir = Path(synced_dir)
                else:
                    logger.error(f"Process directory not found: {process_glycan_dir}")
                    return result
            
            if not self.storage.exists(process_glycan_dir):
                logger.error(f"Process directory not found: {process_glycan_dir}")
                return result
            
            # Verify that the output subdirectory exists in process_dir
            process_output_dir = f"{process_glycan_dir}/output"
            if not self.storage.exists(process_output_dir):
                logger.error(f"Output subdirectory not found in process directory: {process_output_dir}")
                return result
            
            # Call glystatic process_glycan function with the process directory as input
            # This tells glystatic to read the intermediate outputs from process_dir/glycan_name/output/
            glystatic_result = glystatic_process_glycan(
                folder_path=str(process_glycan_dir),  # Pass process_dir/glycan_name/ (contains output/)
                glycam_name=glycan_name,
                output_static_dir=str(self.output_dir)  # Final database output location
            )
            if isinstance(glystatic_result, dict):
                result.update(glystatic_result)
            local_output_dir = str(result.get("local_output_dir") or result.get("final_output_dir") or "")
            final_output_dir = str(result.get("final_output_dir") or local_output_dir or "")
            if final_output_dir:
                self._cache_static_output_dir(glycan_name, final_output_dir)
            self._last_glystatic_result[glycan_name] = dict(result)

            # Validate the output. The glystatic processor writes final output
            # under an ID directory (for example: output_dir/GS00449). Use the
            # existing `check_glystatic_completion` which searches subfolders
            # for a matching archetype entry. Fall back to the older
            # `validate_output_structure` if needed.
            if final_output_dir:
                validation_errors = self._validate_glystatic_output_dir(final_output_dir, glycan_name=glycan_name)
                if not validation_errors:
                    if config.use_oracle_storage:
                        try:
                            gs_local_dir = Path(local_output_dir) if local_output_dir else Path(final_output_dir)
                            if gs_local_dir.exists():
                                remote_prefix = f"{self._remote_output_base()}/{gs_local_dir.name}"
                                self.storage.upload_dir(gs_local_dir, remote_prefix, skip_existing=False)
                                logger.info(f"Uploaded/updated static outputs to Oracle: {remote_prefix}")
                            else:
                                logger.warning(f"Could not locate local GS folder for {glycan_name} to upload")
                        except Exception as e:
                            logger.warning(f"Failed to upload static outputs for {glycan_name}: {e}")
                    result["success"] = True
                    logger.info(f"Successfully completed glystatic for {glycan_name}")
                    return result
                logger.error(f"Glystatic output validation failed for {glycan_name} in {final_output_dir}. Issues:")
                for err in validation_errors:
                    logger.error(f"  - {err}")
            if self.check_glystatic_completion(glycan_name):
                logger.info(f"Successfully completed glystatic for {glycan_name}")
                cached = self._find_static_output_dir(glycan_name, include_local_oracle_fallback=True)
                if cached:
                    result["final_output_dir"] = cached
                    result["local_output_dir"] = cached
                    self._cache_static_output_dir(glycan_name, cached)
                result["success"] = True
                return result
            else:
                logger.error(f"Glystatic output validation failed for {glycan_name}")
                # Fallback: check same-named output folder (legacy behavior)
                if self.validate_output_structure(glycan_name):
                    logger.info(f"Found direct output directory for {glycan_name}; marking as success")
                    result["success"] = True
                    return result
                return result
                        
        except Exception as e:
            logger.exception(f"Error running glystatic for {glycan_name}: {str(e)}")
            return result
    
    def run_glymeta(self, glycan_name: str, output_dir: Optional[str] = None) -> bool:
        """Run metadata processing for a single glycan.
        
        Args:
            glycan_name: Name of the glycan directory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Running glymeta for {glycan_name}")
            
            # Locate the glystatic output directory containing data.json for this glycan
            candidate_dir = output_dir or self._get_cached_static_output_dir(glycan_name)
            if candidate_dir is None:
                candidate_dir = self._find_static_output_dir(
                    glycan_name,
                    substring_match=True,
                    include_local_oracle_fallback=True,
                )
            if candidate_dir is None:
                logger.error(f"data.json not found for {glycan_name}")
                return False
            candidate_dir = self._local_static_output_dir_for_refresh(candidate_dir, glycan_name)
            self._cache_static_output_dir(glycan_name, candidate_dir)
            json_path = f"{candidate_dir}/data.json"
                
            # Update metadata for this specific glycan
            success = self.metadata_processor.update_glycan_metadata(json_path)
            
            if success:
                self._upload_local_static_output_if_oracle(candidate_dir, glycan_name)
                logger.info(f"Successfully completed glymeta for {glycan_name}")
            else:
                logger.error(f"Glymeta processing failed for {glycan_name}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error running glymeta for {glycan_name}: {str(e)}")
            return False
    
    def run_single_glycan_pipeline(self, glycan_name: str) -> Dict[str, bool]:
        """Run the complete pipeline for a single glycan.
        
        Args:
            glycan_name: Name of the glycan directory
            
        Returns:
            Dictionary with step results
        """
        results = {
            "v2": False,
            "glystatic": False,
            # glymeta is best-effort and not part of pipeline success criteria
        }
        glymeta_success: Optional[bool] = None
        glystatic_result: Dict[str, Any] = {}
        local_process_dir = os.path.join(str(self.process_dir), glycan_name)
        should_cleanup_process_dir = False
        stage_durations: Dict[str, float] = {}
        fast_skipped = False
        early_skipped = False
        
        logger.info(f"Starting pipeline for glycan: {glycan_name}")

        try:
            if not self.force_update:
                fast_output_dir = self._fast_completed_output_dir(
                    glycan_name,
                    sync_static=self.mode == "refresh-static",
                )
                if fast_output_dir is not None:
                    fast_skipped = True
                    results["v2"] = True
                    results["glystatic"] = True
                    glystatic_result = {
                        "success": True,
                        "final_output_dir": fast_output_dir,
                        "local_output_dir": fast_output_dir,
                    }
                    logger.info(f"fast skip: v2 complete, static complete for {glycan_name}")
                    if self.mode == "refresh-static":
                        stage_started_at = time.perf_counter()
                        glymeta_success = self.run_glymeta(glycan_name, output_dir=fast_output_dir)
                        stage_durations["glymeta"] = round(time.perf_counter() - stage_started_at, 3)
                        logger.info(f"Pipeline stage glymeta completed in {stage_durations['glymeta']:.3f}s")
                    logger.info(f"Successfully completed pipeline for glycan: {glycan_name}")
                    if stage_durations:
                        logger.info(f"Pipeline timing summary (s): {stage_durations}")
                    return results

            if self.mode == "refresh-static":
                logger.warning(f"refresh-static skipped incomplete glycan {glycan_name}")
                return results

            if not self._has_required_submission_metadata(glycan_name):
                early_skipped = True
                logger.error(f"Pipeline skipped before v2 for {glycan_name}: required submission metadata is missing")
                return results

            # Step 1: v2 analysis
            # Check if already completed (unless force update)
            if (
                not self.force_update
                and config.use_oracle_storage
                and self._fast_indexes_built
                and glycan_name in self._remote_process_complete
            ):
                logger.info(f"v2 analysis already complete for {glycan_name} (fast index), skipping")
                results["v2"] = True
            elif not self.force_update and self.check_v2_completion(glycan_name):
                logger.info(f"v2 analysis already complete for {glycan_name}, skipping")
                results["v2"] = True
            else:
                # Run v2 and upload process dir to Oracle if enabled
                stage_started_at = time.perf_counter()
                v2_ok = self.run_v2_analysis(glycan_name)
                stage_durations["v2"] = round(time.perf_counter() - stage_started_at, 3)
                logger.info(f"Pipeline stage v2 completed in {stage_durations['v2']:.3f}s")
                results["v2"] = v2_ok
                if v2_ok and config.use_oracle_storage:
                    try:
                        remote_prefix = f"{self._remote_process_base()}/{glycan_name}"
                        # Overwrite remote process outputs so Oracle always reflects the latest run.
                        upload_started_at = time.perf_counter()
                        self.storage.upload_dir(local_process_dir, remote_prefix, skip_existing=False)
                        stage_durations["process_upload"] = round(time.perf_counter() - upload_started_at, 3)
                        logger.info(f"Pipeline stage process_upload completed in {stage_durations['process_upload']:.3f}s")
                        logger.info(f"Uploaded process outputs to Oracle: {remote_prefix}")
                        should_cleanup_process_dir = config.should_cleanup_process_files()
                    except Exception as e:
                        logger.warning(f"Failed to upload/cleanup process outputs for {glycan_name}: {e}")
                
            if not results["v2"]:
                logger.error(f"Pipeline failed at v2 step for {glycan_name}")
                return results
                
            # Step 2: glystatic
            if not self.force_update and self.check_glystatic_completion(glycan_name):
                logger.info(f"glystatic already complete for {glycan_name}, skipping")
                results["glystatic"] = True
            else:
                stage_started_at = time.perf_counter()
                glystatic_result = self.run_glystatic(glycan_name, check_existing=False)
                stage_durations["glystatic"] = round(time.perf_counter() - stage_started_at, 3)
                logger.info(f"Pipeline stage glystatic completed in {stage_durations['glystatic']:.3f}s")
                results["glystatic"] = bool(glystatic_result.get("success"))

            # Post-glystatic upload is handled inside run_glystatic for Oracle mode
            if not results["glystatic"]:
                logger.error(f"Pipeline failed at glystatic step for {glycan_name}")
                return results
                
            # Step 3: glymeta (best-effort, do not fail the pipeline if it fails)
            try:
                stage_started_at = time.perf_counter()
                glymeta_success = self.run_glymeta(
                    glycan_name,
                    output_dir=str(glystatic_result.get("final_output_dir") or "") or None,
                )
                stage_durations["glymeta"] = round(time.perf_counter() - stage_started_at, 3)
                logger.info(f"Pipeline stage glymeta completed in {stage_durations['glymeta']:.3f}s")
            except Exception:
                # run_glymeta logs its own errors; continue regardless
                pass
                
            logger.info(f"Successfully completed pipeline for glycan: {glycan_name}")
            if stage_durations:
                logger.info(f"Pipeline timing summary (s): {stage_durations}")
            return results
        finally:
            if not fast_skipped and not early_skipped and results.get("v2") and results.get("glystatic"):
                try:
                    self._write_log_tail_summary(glycan_name, results, glymeta_success)
                except Exception as exc:
                    logger.warning(f"Failed to write PocketBase log_tail summary for {glycan_name}: {exc}")
            if (
                config.use_oracle_storage
                and should_cleanup_process_dir
                and results.get("v2")
                and results.get("glystatic")
            ):
                try:
                    cleanup_started_at = time.perf_counter()
                    if self.storage.delete_local_directory(local_process_dir):
                        stage_durations["process_cleanup"] = round(time.perf_counter() - cleanup_started_at, 3)
                        logger.info(f"Pipeline stage process_cleanup completed in {stage_durations['process_cleanup']:.3f}s")
                        logger.info(f"Cleaned up local process directory: {local_process_dir}")
                    else:
                        logger.warning(f"Failed to clean up local process directory: {local_process_dir}")
                except Exception as exc:
                    logger.warning(f"Failed to clean up local process directory {local_process_dir}: {exc}")
    
    def run_database_finalization(self) -> bool:
        """Run final database operations.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting database finalization")
            self._sync_all_remote_static_to_local()
            
            # Validate that we have processed glycans
            processed_glycans = []
            try:
                base = self._remote_output_base()
                candidates = self._list_storage_files_complete(base)
            except Exception:
                candidates = []
            for cand in candidates:
                key_raw = cand.as_posix() if hasattr(cand, 'as_posix') else str(cand)
                if not key_raw.endswith('/data.json') and not key_raw.endswith('data.json'):
                    continue
                key = str(Path(key_raw).parent)
                processed_glycans.append(key.split('/')[-1])
            
            if not processed_glycans:
                logger.warning("No processed glycans found in output directory")
                return False
            
            logger.info(f"Found {len(processed_glycans)} processed glycans for finalization")
            
            # Skip metadata validation here; `glymeta` is run per-glycan during
            # processing and any metadata fixes should be handled at that stage.
            logger.info("Skipping final metadata validation (glymeta is applied per-glycan)")
            
            # Run glybake to create final archives
            logger.info("Running glybake to create database archives")
            self.baker.bake_all(include_wurcs_submission=False)
            
            # Run json2rdf conversion (only if consolidated JSON exists and is valid)
            logger.info("Preparing to run json2rdf conversion")
            input_path = f"{self.output_dir}/GLYCOSHAPE.json"
            output_path = f"{self.output_dir}/GLYCOSHAPE_RDF.ttl"

            if not self.storage.exists(input_path):
                logger.warning(f"Consolidated JSON not found at expected location: {input_path}")
                logger.warning("Skipping json2rdf conversion")
            else:
                # Validate JSON is readable and non-empty
                try:
                    with self.storage.open(input_path, 'r') as jf:
                        json_data = json.load(jf)
                except Exception as e:
                    logger.error(f"Failed to read/parse consolidated JSON '{input_path}': {e}")
                    return False

                if not isinstance(json_data, dict) or len(json_data) == 0:
                    logger.warning(f"Consolidated JSON at {input_path} is empty or not an object; skipping json2rdf conversion")
                else:
                    logger.info("Running json2rdf conversion")
                    try:
                        from lib.json2rdf import convert_glycoshape_to_rdf
                        rdf_graph = convert_glycoshape_to_rdf(str(input_path), str(output_path))
                        if rdf_graph is None:
                            logger.error("json2rdf conversion returned no graph")
                            return False
                    except Exception as e:
                        logger.error(f"json2rdf conversion failed with exception: {e}")
                        return False

            # In Oracle mode, upload/update the key finalization artifacts
            if config.use_oracle_storage:
                try:
                    artifacts = [
                        "GlycoShape.zip",  # from baker
                        "GLYCOSHAPE_RDF.ttl",  # from json2rdf
                        "GLYCOSHAPE.json",     # from baker
                        "missing_glytoucan.txt",
                    ]
                    for fname in artifacts:
                        local_path = Path(str(self.output_dir)) / fname
                        if local_path.exists():
                            # Always upload under 'static/<artifact>' in Oracle
                            remote_path = f"{self._remote_output_base()}/{fname}"
                            try:
                                with open(local_path, 'rb') as f:
                                    data = f.read()
                                # Overwrite to ensure updates land
                                self.storage.backend.write_binary(remote_path, data)
                                logger.info(f"Uploaded/updated artifact to Oracle: {remote_path}")
                            except Exception as e:
                                logger.warning(f"Failed to upload artifact {local_path} -> {remote_path}: {e}")
                        else:
                            logger.warning(f"Finalization artifact not found locally: {local_path}")
                except Exception as e:
                    logger.warning(f"Failed uploading final artifacts to Oracle: {e}")
            
            logger.info("Successfully completed database finalization")
            return True
            
        except Exception as e:
            logger.error(f"Error during database finalization: {str(e)}")
            return False
    
    def run_full_pipeline(self) -> Dict[str, any]:
        """Run the complete pipeline for all glycans.
        
        Returns:
            Dictionary with comprehensive results
        """
        results = {
            "total_glycans": 0,
            "successful_glycans": 0,
            "failed_glycans": 0,
            "glycan_results": {},
            "finalization_success": False,
            "step_statistics": {
                "v2_success": 0,
                "glystatic_success": 0
            }
        }
        
        # Discover glycans
        glycan_names = self.discover_glycans()
        results["total_glycans"] = len(glycan_names)
        
        if not glycan_names:
            logger.warning("No glycans found to process")
            return results

        if not self.force_update:
            self._build_fast_completion_indexes()
            
        # Process each glycan
        logger.info(f"Starting processing of {len(glycan_names)} glycans")
        
        for i, glycan_name in enumerate(glycan_names, 1):
            logger.info(f"Processing glycan {i}/{len(glycan_names)}: {glycan_name}")
            
            glycan_results = self.run_single_glycan_pipeline(glycan_name)
            results["glycan_results"][glycan_name] = glycan_results
            
            # Update step statistics
            for step, success in glycan_results.items():
                if success:
                    results["step_statistics"][f"{step}_success"] += 1
            
            # Check if all steps succeeded
            if all(glycan_results.values()):
                results["successful_glycans"] += 1
                logger.info(f"✓ Successfully completed all steps for {glycan_name}")
            else:
                results["failed_glycans"] += 1
                failed_steps = [step for step, success in glycan_results.items() if not success]
                logger.warning(f"✗ Failed steps for {glycan_name}: {', '.join(failed_steps)}")
                
        # Run database finalization if any glycans were successfully processed
        if results["successful_glycans"] > 0:
            logger.info("Starting database finalization...")
            results["finalization_success"] = self.run_database_finalization()
        else:
            logger.warning("No glycans were successfully processed, skipping finalization")
            
        # Log comprehensive summary
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total glycans processed: {results['total_glycans']}")
        logger.info(f"Successful glycans: {results['successful_glycans']}")
        logger.info(f"Failed glycans: {results['failed_glycans']}")
        logger.info("")
        logger.info("Step-by-step success rates:")
        for step, count in results["step_statistics"].items():
            percentage = (count / results["total_glycans"] * 100) if results["total_glycans"] > 0 else 0
            logger.info(f"  {step.replace('_success', '')}: {count}/{results['total_glycans']} ({percentage:.1f}%)")
        logger.info("")
        logger.info(f"Database finalization: {'Success' if results['finalization_success'] else 'Failed'}")
        
        # List failed glycans if any
        if results["failed_glycans"] > 0:
            failed_glycans = [name for name, res in results["glycan_results"].items() 
                             if not all(res.values())]
            logger.info(f"Failed glycans: {', '.join(failed_glycans)}")
        
        logger.info("=" * 60)
        
        return results


def main():
    """Main function to run the entire pipeline."""
    parser = argparse.ArgumentParser(description="Glycan Analysis Pipeline")
    parser.add_argument(
        '--data-dir', 
        type=str, 
        default=None,
        help='Directory containing glycan input data (overrides config)'
    )
    parser.add_argument(
        '--process-dir', 
        type=str, 
        default=None,
        help='Directory for processing outputs (overrides config)'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default=None,
        help='Directory for database output (overrides config)'
    )
    parser.add_argument(
        '--update', 
        action='store_true',
        help='Force recomputation of existing results (alias for --mode fresh)'
    )
    parser.add_argument(
        '--mode',
        choices=("incremental", "refresh-static", "fresh"),
        default="incremental",
        help='Run mode: incremental skips completed glycans, refresh-static updates completed static metadata only, fresh recomputes everything'
    )
    parser.add_argument(
        '--glycan', 
        type=str, 
        default=None,
        help='Process only a specific glycan (for testing)'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Check status of glycans instead of running pipeline'
    )
    
    args = parser.parse_args()
    if args.update:
        args.mode = "fresh"
    
    # Initialize storage based on configuration
    if not config.initialize_storage():
        logger.error("Failed to initialize storage. Exiting.")
        sys.exit(1)
    
    # Get storage info
    storage_info = config.get_storage_info()
    logger.info(f"Storage mode: {storage_info['storage_mode']}")
    if storage_info['storage_mode'] == 'oracle':
        logger.info(f"Using Oracle Cloud Object Storage")
    
    # Use config defaults or command line overrides
    data_dir = args.data_dir if args.data_dir else config.data_dir
    process_dir = args.process_dir if args.process_dir else config.process_dir
    output_dir = args.output_dir if args.output_dir else config.output_dir
    
    logger.info(f"Starting Glycan Analysis Pipeline")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Process directory: {process_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Force update: {args.update}")
    logger.info(f"Run mode: {args.mode}")
    
    # Initialize pipeline runner
    runner = GlycanPipelineRunner(data_dir, process_dir, output_dir, args.update, mode=args.mode)
    
    try:
        if args.status:
            # Check status instead of running
            if args.glycan:
                # Status for single glycan
                logger.info(f"Checking status for glycan: {args.glycan}")
                
                # Check v2 status
                v2_complete = runner.check_v2_completion(args.glycan)
                glystatic_status = runner.check_glystatic_completion(args.glycan)
                
                print(f"\nStatus for {args.glycan}:")
                print(f"v2: {'✓' if v2_complete else '✗'}")
                print(f"\nglystatic: {'✓' if glystatic_status else '✗'}")
                        
            else:
                # Status for all glycans
                glycan_names = runner.discover_glycans()
                if not glycan_names:
                    print("No glycans found")
                    sys.exit(0)
                
                print(f"\nStatus Summary for {len(glycan_names)} glycans:")
                print("=" * 60)
                
                v2_complete = 0
                glystatic_complete = 0
                
                for glycan_name in glycan_names:
                    v2_status = runner.check_v2_completion(glycan_name)
                    glystatic_status = runner.check_glystatic_completion(glycan_name)
                    
                    if v2_status: v2_complete += 1
                    if glystatic_status: glystatic_complete += 1

                    v2_sym = "✓" if v2_status else "✗"
                    static_sym = "✓" if glystatic_status else "✗"

                    print(f"{glycan_name:30} v2:{v2_sym} static:{static_sym}")
                
                print("=" * 60)
                print(f"Completion rates:")
                print(f"  v2: {v2_complete}/{len(glycan_names)} ({v2_complete/len(glycan_names)*100:.1f}%)")
                print(f"  glystatic: {glystatic_complete}/{len(glycan_names)} ({glystatic_complete/len(glycan_names)*100:.1f}%)")
                # glymeta is excluded from the summary since it's applied per-glycan
            
            sys.exit(0)
            
        elif args.glycan:
            # Process single glycan (for testing)
            logger.info(f"Processing single glycan: {args.glycan}")
            results = runner.run_single_glycan_pipeline(args.glycan)
            
            if all(results.values()):
                logger.info(f"Single glycan pipeline completed successfully")
                sys.exit(0)
            else:
                logger.error(f"Single glycan pipeline failed")
                sys.exit(1)
        else:
            # Process all glycans
            results = runner.run_full_pipeline()
            
            # Exit with appropriate code
            if results["successful_glycans"] > 0 and results["finalization_success"]:
                logger.info("Glycan Analysis Pipeline completed successfully")
                sys.exit(0)
            else:
                logger.error("Glycan Analysis Pipeline failed")
                sys.exit(1)
                
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error in pipeline: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
