"""
Shared PocketBase API client for the GlycoShape pipeline.

Provides:
- PocketBaseClient: reusable HTTP client with auth, caching, and health checks
- extract_inventory_metadata(): maps PB records to the inventory dict shape
- extract_search_enrichment(): extracts keywords and description data for search_meta
"""

import os
import logging
from typing import Optional, Dict, Any, List

import requests

logger = logging.getLogger(__name__)

SUBMISSION_COLLECTION = "glycan_submission"
GLYCAN_COLLECTION = "glycans"
COLLECTION = SUBMISSION_COLLECTION


def _get_env_token() -> str:
    """Resolve the PocketBase token from supported environment variables."""
    return (os.environ.get("POCKETBASE_TOKEN", "").strip()
            or os.environ.get("POCKETBASE_ADMIN_TOKEN", "").strip())


def _safe_float(value: Any) -> Optional[float]:
    """Convert a value to float, returning None on failure."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


class PocketBaseClient:
    """Reusable PocketBase API client with in-memory caching."""

    def __init__(self, base_url: str = None, token: str = None):
        self.base_url = (base_url or os.environ.get("POCKETBASE_URL", "")).strip().rstrip("/")
        self.token = (token or _get_env_token()).strip()
        self._available: Optional[bool] = None
        self._cache_by_id: Dict[str, Any] = {}
        self._cache_by_name: Dict[str, Any] = {}
        self._glycan_cache_by_id: Dict[str, Any] = {}
        self._prefetched_collections = set()

    def is_configured(self) -> bool:
        """Check if PocketBase URL and token are set."""
        return bool(self.base_url) and bool(self.token)

    def is_available(self) -> bool:
        """Check if PocketBase is reachable. Result is cached for the session."""
        if self._available is not None:
            return self._available
        if not self.is_configured():
            self._available = False
            return False
        try:
            resp = requests.get(f"{self.base_url}/api/health", timeout=5)
            self._available = resp.status_code == 200
        except Exception:
            self._available = False
        if not self._available:
            logger.warning("PocketBase is not reachable at %s", self.base_url)
        return self._available

    def request(self, method: str, path: str, **kwargs) -> dict:
        """Generic PocketBase API request with Bearer auth."""
        url = f"{self.base_url}{path}"
        headers = kwargs.pop("headers", {})
        headers.update({
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        })
        resp = requests.request(method, url, headers=headers, timeout=15, **kwargs)
        if resp.status_code >= 400:
            raise RuntimeError(f"PocketBase {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def _escape_filter_value(self, value: str) -> str:
        return value.replace('"', '\\"')

    def get_record_by_glycoshape_id(
        self, gs_id: str, collection: str = COLLECTION
    ) -> Optional[dict]:
        """Fetch a record by glycoshape_id, with in-memory cache."""
        if collection == GLYCAN_COLLECTION:
            if gs_id in self._glycan_cache_by_id:
                return self._glycan_cache_by_id[gs_id]
            if collection in self._prefetched_collections:
                return None
        elif gs_id in self._cache_by_id:
            return self._cache_by_id[gs_id]
        elif collection in self._prefetched_collections:
            return None
        data = self.request(
            "get",
            f"/api/collections/{collection}/records",
            params={"filter": f'glycoshape_id = "{self._escape_filter_value(gs_id)}"', "perPage": 1},
        )
        items = data.get("items", [])
        record = items[0] if items else None
        if collection == GLYCAN_COLLECTION:
            self._glycan_cache_by_id[gs_id] = record
        else:
            self._cache_by_id[gs_id] = record
        return record

    def get_glycan_record_by_glycoshape_id(self, gs_id: str) -> Optional[dict]:
        """Fetch a glycans collection record by glycoshape_id."""
        return self.get_record_by_glycoshape_id(gs_id, collection=GLYCAN_COLLECTION)

    def get_record_by_glycam_name(
        self, glycam_name: str, collection: str = COLLECTION
    ) -> Optional[dict]:
        """Fetch a record by glycam_name, with in-memory cache."""
        if glycam_name in self._cache_by_name:
            return self._cache_by_name[glycam_name]
        if collection in self._prefetched_collections:
            return None
        data = self.request(
            "get",
            f"/api/collections/{collection}/records",
            params={"filter": f'glycam_name = "{self._escape_filter_value(glycam_name)}"', "perPage": 1},
        )
        items = data.get("items", [])
        record = items[0] if items else None
        self._cache_by_name[glycam_name] = record
        return record

    def prefetch_all(
        self, collection: str = COLLECTION, page_size: int = 200
    ) -> None:
        """Bulk-fetch all records into both caches for batch pipeline runs."""
        page = 1
        total_loaded = 0
        while True:
            data = self.request(
                "get",
                f"/api/collections/{collection}/records",
                params={"perPage": page_size, "page": page},
            )
            for item in data.get("items", []):
                gs_id = item.get("glycoshape_id")
                glycam_name = item.get("glycam_name")
                if gs_id and collection == GLYCAN_COLLECTION:
                    self._glycan_cache_by_id[gs_id] = item
                elif gs_id:
                    self._cache_by_id[gs_id] = item
                if glycam_name:
                    self._cache_by_name[glycam_name] = item
                total_loaded += 1
            if page >= data.get("totalPages", 1):
                break
            page += 1
        self._prefetched_collections.add(collection)
        logger.info("Prefetched %d PocketBase records", total_loaded)

    def clear_cache(self):
        """Clear all caches and reset availability check."""
        self._cache_by_id.clear()
        self._cache_by_name.clear()
        self._glycan_cache_by_id.clear()
        self._prefetched_collections.clear()
        self._available = None

    def upsert_glycan_metadata(self, payload: Dict[str, Any]) -> Optional[dict]:
        """Create or update a glycans collection record keyed by glycoshape_id."""
        glycoshape_id = str(payload.get("glycoshape_id") or "").strip()
        if not glycoshape_id:
            raise ValueError("glycoshape_id is required to upsert glycan metadata")

        normalized_payload = dict(payload)
        for key in ("aliases", "common_names", "keywords"):
            if key in normalized_payload:
                normalized_payload[key] = _normalize_string_list(normalized_payload.get(key))

        if not isinstance(normalized_payload.get("name_variants"), dict):
            normalized_payload["name_variants"] = {}

        existing = self.get_glycan_record_by_glycoshape_id(glycoshape_id)
        merged = dict(existing or {})
        merged.update(normalized_payload)
        normalized_payload["search_text"] = _build_search_text(merged)

        if existing:
            result = self.request(
                "patch",
                f"/api/collections/{GLYCAN_COLLECTION}/records/{existing['id']}",
                json=normalized_payload,
            )
        else:
            result = self.request(
                "post",
                f"/api/collections/{GLYCAN_COLLECTION}/records",
                json=normalized_payload,
            )

        if result.get("glycoshape_id"):
            self._glycan_cache_by_id[result["glycoshape_id"]] = result
        return result

    def update_submission_record(
        self,
        record_id: str,
        payload: Dict[str, Any],
    ) -> Optional[dict]:
        """Patch a glycan_submission record and refresh in-memory caches."""
        if not record_id:
            raise ValueError("record_id is required to update a submission record")

        result = self.request(
            "patch",
            f"/api/collections/{SUBMISSION_COLLECTION}/records/{record_id}",
            json=payload,
        )

        glycoshape_id = str(result.get("glycoshape_id") or "").strip()
        glycam_name = str(result.get("glycam_name") or "").strip()
        if glycoshape_id:
            self._cache_by_id[glycoshape_id] = result
        if glycam_name:
            self._cache_by_name[glycam_name] = result
        return result

    def update_submission_log_tail(
        self,
        log_tail: str,
        glycoshape_id: str = "",
        glycam_name: str = "",
    ) -> Optional[dict]:
        """Patch ``log_tail`` on the matching glycan_submission record."""
        record = None

        glycoshape_id = str(glycoshape_id or "").strip()
        glycam_name = str(glycam_name or "").strip()

        if glycoshape_id:
            record = self.get_record_by_glycoshape_id(glycoshape_id, collection=SUBMISSION_COLLECTION)
        if record is None and glycam_name:
            record = self.get_record_by_glycam_name(glycam_name, collection=SUBMISSION_COLLECTION)
        if record is None:
            return None

        return self.update_submission_record(record["id"], {"log_tail": log_tail})


def _normalize_string_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []

    normalized: List[str] = []
    seen = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)
    return normalized


def _flatten_variant_values(name_variants: Any) -> List[str]:
    if not isinstance(name_variants, dict):
        return []

    values: List[str] = []
    for variant_payload in name_variants.values():
        if not isinstance(variant_payload, dict):
            continue
        for variant_value in variant_payload.values():
            if variant_value is None:
                continue
            text = str(variant_value).strip()
            if text:
                values.append(text)
    return values


def _build_search_text(payload: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in (
        "glycoshape_id", "canonical_name", "glycam_name", "iupac_name",
        "glytoucan_id", "description",
    ):
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            parts.append(text)

    parts.extend(_normalize_string_list(payload.get("aliases")))
    parts.extend(_normalize_string_list(payload.get("common_names")))
    parts.extend(_normalize_string_list(payload.get("keywords")))
    parts.extend(_flatten_variant_values(payload.get("name_variants")))

    deduped: List[str] = []
    seen = set()
    for part in parts:
        key = part.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(part)
    return "\n".join(deduped)


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def extract_inventory_metadata(pb_record: dict) -> dict:
    """Transform a PocketBase record into the inventory metadata dict shape.

    Returns the same structure as ``get_glycan_metadata_from_inventory()``
    in ``lib/glystatic.py``, so it can be used as a drop-in replacement.
    """
    return {
        "ID": pb_record.get("glycoshape_id", ""),
        "timestamp": pb_record.get("created", ""),
        "email": pb_record.get("email", ""),
        "glycam_name": pb_record.get("glycam_name", ""),
        "transfer_method": pb_record.get("source_value", ""),
        "length": _safe_float(pb_record.get("simulation_length")),
        "package": pb_record.get("md_package", ""),
        "forcefield": pb_record.get("force_field", ""),
        "temperature": _safe_float(pb_record.get("temperature")),
        "pressure": _safe_float(pb_record.get("pressure")),
        "salt": _safe_float(pb_record.get("salt_concentration")),
        "comments": pb_record.get("comments", ""),
        "glytoucan_id": pb_record.get("glytoucan_id", ""),
    }


def extract_md_info(pb_record: dict) -> tuple:
    """Transform a PocketBase submission record into ``name_utils.get_md_info`` shape."""
    metadata = extract_inventory_metadata(pb_record)

    def _as_optional_str(value: Any) -> Optional[str]:
        if value is None or value == "":
            return None
        return str(value)

    return (
        _as_optional_str(metadata.get("ID")),
        _as_optional_str(metadata.get("length")),
        _as_optional_str(metadata.get("package")),
        _as_optional_str(metadata.get("forcefield")),
        _as_optional_str(metadata.get("temperature")),
        _as_optional_str(metadata.get("pressure")),
        _as_optional_str(metadata.get("salt")),
        _as_optional_str(metadata.get("email")),
    )


def extract_search_enrichment(
    glycan_record: Optional[dict] = None,
    submission_record: Optional[dict] = None,
) -> dict:
    """Extract data relevant for ``search_meta`` enrichment.

    Reads the user-curated fields from the ``glycans`` collection:
    - ``keywords`` (JSON array) — user-supplied and pipeline-derived keywords
    - ``common_names`` (JSON array) — common names
    - ``description`` (text) — curated or generated description

    Also derives extra keywords from submission simulation fields
    (md_package, force_field, source_type).

    Returns a dict with:
    - ``extra_keywords``: list of keywords (user-curated + derived)
    - ``user_common_names``: list of common names from PocketBase
    - ``user_description``: user-provided description from PocketBase
    - ``user_comments``: user-provided comments for description fallback
    """
    extra_keywords: List[str] = []
    glycan_record = glycan_record or {}
    submission_record = submission_record or {}

    user_keywords = glycan_record.get("keywords")
    if isinstance(user_keywords, list):
        extra_keywords.extend(str(k) for k in user_keywords if k)

    # Derived keywords from simulation fields
    md_package = (submission_record.get("md_package") or "").strip()
    if md_package:
        extra_keywords.append(f"MD:{md_package}")

    force_field = (submission_record.get("force_field") or "").strip()
    if force_field:
        extra_keywords.append(f"FF:{force_field}")

    source_type = (submission_record.get("source_type") or "").strip()
    if source_type:
        extra_keywords.append(f"Source:{source_type}")

    user_common_names = _normalize_string_list(glycan_record.get("common_names"))

    user_description = str(glycan_record.get("description") or "").strip()

    return {
        "extra_keywords": extra_keywords,
        "user_common_names": user_common_names,
        "user_description": user_description,
        "user_comments": str(submission_record.get("comments") or "").strip(),
    }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_client: Optional[PocketBaseClient] = None


def get_pocketbase_client() -> PocketBaseClient:
    """Get or create the module-level PocketBase client singleton."""
    global _client
    if _client is None:
        _client = PocketBaseClient()
    return _client


def reset_pocketbase_client():
    """Reset the singleton (useful for testing)."""
    global _client
    _client = None
