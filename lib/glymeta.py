import lib.config as config
from pathlib import Path
import json
import logging
from typing import Dict, List, Set, Optional, Union
from lib import glyclass, name_utils
from lib.storage import get_storage_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _string_list(values) -> List[str]:
    """Non-empty strings of a list, stripped, the first spelling of each (case-insensitive)."""
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


def search_enrichment(
    keywords=None,
    common_names=None,
    description=None,
    package=None,
    forcefield=None,
    source_type=None,
    comments=None,
) -> Dict:
    """Curated fields of ``search_meta`` in the form :meth:`GlycanMetadataProcessor.update_glycan_metadata` merges.

    ``keywords``, ``common_names`` and ``description`` are the curated ones;
    ``package``, ``forcefield`` and ``source_type`` add the keywords
    ``MD:<package>``, ``FF:<force field>`` and ``Source:<type>``; ``comments``
    end a generated description.  The fields come from the PocketBase
    records (``lib.pocketbase.extract_search_enrichment``) or from a
    submission record (``lib.entry_complete``).
    """
    extra_keywords: List[str] = []
    if isinstance(keywords, list):
        extra_keywords.extend(str(k) for k in keywords if k)

    # Derived keywords from simulation fields
    md_package = (package or "").strip()
    if md_package:
        extra_keywords.append(f"MD:{md_package}")

    force_field = (forcefield or "").strip()
    if force_field:
        extra_keywords.append(f"FF:{force_field}")

    source_type = (source_type or "").strip()
    if source_type:
        extra_keywords.append(f"Source:{source_type}")

    return {
        "extra_keywords": extra_keywords,
        "user_common_names": _string_list(common_names),
        "user_description": str(description or "").strip(),
        "user_comments": str(comments or "").strip(),
    }


class GlycanMetadataProcessor:
    """Class for processing and updating glycan search metadata."""
    
    # Ganglioside mapping from GlyTouCan IDs to common names
    GANGLIOSIDE_MAP = {
        "G40990LB": "GA2",
        "G00071MO": "GA1", 
        "G01269WV": "GM1b",
        "G86874EI": "GD1c",
        "G40764VP": "GM3",
        "G65156XZ": "GM2",
        "G41876BL": "GM1b",
        "G21518JT": "GD1a",
        "G97898ZO": "GT1a",
        "G10158SF": "GD3",
        "G98397EB": "GD2",
        "G08648UJ": "GT1b",
        "G71158TZ": "GQ1b",
        "G27476OM": "GT3",
        "G66281XG": "GT2",
        "G53140PB": "GT1c",
        "G54434YJ": "GQ1c",
        "G44362TS": "GP1c",
    }
    
    def __init__(self, database_dir: Union[str, Path], pb_client=None):
        """Initialize the metadata processor.

        Args:
            database_dir: Path to the database directory containing glycan folders
            pb_client: Optional PocketBaseClient for search metadata enrichment
        """
        self.database_dir = Path(database_dir)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.pb_client = pb_client
        
    def get_glycan_keywords(self, iupac_string: Optional[str], wurcs: Optional[str] = None) -> List[str]:
        """Class keywords of a glycan (N-, O-glycan, GAG and the N-glycan types), sorted.

        The keywords are decided on the residue tree of the glycan
        (:mod:`lib.glyclass`), read from its WURCS when it is given and can be
        read, else from the IUPAC-condensed name.  A glycan neither of whose
        names can be read gets no class keyword (and a warning).

        Args:
            iupac_string: IUPAC-condensed name of the glycan
            wurcs: WURCS of the glycan

        Returns:
            List of keywords describing the glycan type
        """
        keywords, source = glyclass.class_keywords(wurcs=wurcs, iupac=iupac_string)
        if source not in ("wurcs", "iupac") and (wurcs or iupac_string):
            self.logger.warning(f"No residue tree for the class keywords ({source})")
        return keywords

    def get_common_names(self, glycan_data: Dict) -> Set[str]:
        """Extract common names from glycan data.
        
        Args:
            glycan_data: Dictionary containing glycan information
            
        Returns:
            Set of common names
        """
        common_names = set()
        
        # Add Oxford name if present
        archetype = glycan_data.get("archetype", {})
        oxford_name = archetype.get("oxford")
        if oxford_name:
            common_names.add(oxford_name)
            
        # Add ganglioside names based on GlyTouCan IDs
        glytoucan_ids = []
        
        # Collect all GlyTouCan IDs
        for entry_type in ["archetype", "alpha", "beta"]:
            entry = glycan_data.get(entry_type, {})
            glytoucan_id = entry.get("glytoucan")
            if glytoucan_id:
                glytoucan_ids.append(glytoucan_id)
                
        # Map to ganglioside names
        for glytoucan_id in glytoucan_ids:
            if glytoucan_id in self.GANGLIOSIDE_MAP:
                common_names.add(self.GANGLIOSIDE_MAP[glytoucan_id])
                
        return common_names

    def _normalize_string_list(self, values: List[str]) -> List[str]:
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

    def _extract_variant_payload(self, entry: Dict) -> Dict[str, str]:
        if not isinstance(entry, dict):
            return {}

        payload: Dict[str, str] = {}
        for key in ("name", "glycam", "iupac", "iupac_extended", "glytoucan", "wurcs", "oxford"):
            value = entry.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                payload[key] = text
        return payload

    def _build_name_variants(self, glycan_data: Dict) -> Dict[str, Dict[str, str]]:
        name_variants: Dict[str, Dict[str, str]] = {}
        for variant_name in ("archetype", "alpha", "beta"):
            payload = self._extract_variant_payload(glycan_data.get(variant_name, {}))
            if payload:
                name_variants[variant_name] = payload
        return name_variants

    def _build_aliases(self, glycan_data: Dict, common_names: List[str]) -> List[str]:
        aliases: List[str] = []
        for variant_name in ("archetype", "alpha", "beta"):
            entry = glycan_data.get(variant_name, {})
            if not isinstance(entry, dict):
                continue
            for key in ("name", "glycam", "iupac", "iupac_extended", "oxford"):
                value = entry.get(key)
                if value:
                    aliases.append(str(value))

        aliases.extend(common_names)
        return self._normalize_string_list(aliases)

    def _choose_canonical_name(self, glycan_data: Dict, common_names: List[str]) -> str:
        archetype = glycan_data.get("archetype", {})
        if not isinstance(archetype, dict):
            return ""

        oxford = str(archetype.get("oxford") or "").strip()
        if oxford:
            return oxford

        if common_names:
            return common_names[0]

        for key in ("iupac_extended", "iupac", "name", "glycam", "ID"):
            value = archetype.get(key)
            if value:
                return str(value).strip()

        return ""
    
    def _enrich_from_pocketbase(self, data: Dict, search_meta: Dict) -> None:
        """Merge PocketBase-sourced keywords and description into search_meta.

        Gracefully no-ops if PocketBase is unavailable or the record is missing.
        """
        if self.pb_client is None or not self.pb_client.is_available():
            return

        gs_id = data.get("archetype", {}).get("ID")
        if not gs_id:
            return

        try:
            submission_record = self.pb_client.get_record_by_glycoshape_id(gs_id)
            glycan_record = self.pb_client.get_glycan_record_by_glycoshape_id(gs_id)
            if submission_record is None and glycan_record is None:
                self.logger.debug(f"No PocketBase record found for {gs_id}")
                return

            from lib.pocketbase import extract_search_enrichment
            enrichment = extract_search_enrichment(
                glycan_record=glycan_record,
                submission_record=submission_record,
            )
            self._apply_enrichment(data, search_meta, enrichment)
            self.logger.info(f"Enriched search_meta from PocketBase for {gs_id}")

        except Exception as e:
            self.logger.warning(f"PocketBase enrichment failed for {gs_id}: {e}")

    def _apply_enrichment(self, data: Dict, search_meta: Dict, enrichment: Dict) -> None:
        """Merge curated keywords, common names and description into ``search_meta``.

        ``enrichment`` is what :func:`search_enrichment` returns, whether its
        fields came from the PocketBase records or from a submission record.
        """
        # Add extra keywords (additive, never replace)
        extra_kw = enrichment.get("extra_keywords", [])
        if extra_kw:
            current = set(search_meta.get("keywords", []))
            current.update(extra_kw)
            search_meta["keywords"] = sorted(list(current))

        # Add user-curated common names (additive)
        user_common_names = enrichment.get("user_common_names", [])
        if user_common_names:
            current_names = set(search_meta.get("common_names", []))
            current_names.update(user_common_names)
            search_meta["common_names"] = sorted(list(current_names))

        # Use the user-curated description, or generate one
        user_desc = enrichment.get("user_description", "")
        if user_desc:
            search_meta["description"] = user_desc
        elif not search_meta.get("description"):
            search_meta["description"] = self._generate_description(
                data, enrichment
            )

    def _generate_description(self, data: Dict, pb_enrichment: Dict) -> str:
        """Generate a natural language description for search and vector relevance.

        Builds a concise sentence from glycan identity, classification,
        simulation context, common names, and composition.
        """
        parts = []
        archetype = data.get("archetype", {})
        search_meta = data.get("search_meta", {})

        # Glycan identity
        gs_id = archetype.get("ID", "")
        iupac = archetype.get("iupac", "")
        if iupac:
            parts.append(f"{gs_id} is a glycan with IUPAC notation {iupac}.")

        # Type classification
        type_keywords = [
            k for k in search_meta.get("keywords", [])
            if k in glyclass.CLASS_KEYWORDS
        ]
        if type_keywords:
            parts.append(f"Classified as: {', '.join(type_keywords)}.")

        # Simulation context from archetype fields
        sim_parts = []
        package = archetype.get("package")
        if package:
            sim_parts.append(f"MD package {package}")
        forcefield = archetype.get("forcefield")
        if forcefield:
            sim_parts.append(f"force field {forcefield}")
        temperature = archetype.get("temperature")
        if temperature is not None:
            sim_parts.append(f"{temperature}K")
        if sim_parts:
            parts.append(f"Simulated with {', '.join(sim_parts)}.")

        # Common names
        common_names = search_meta.get("common_names", [])
        if common_names:
            parts.append(f"Also known as: {', '.join(common_names)}.")

        # Composition
        components = archetype.get("components", {})
        if components:
            comp_str = ", ".join(f"{v}x {k}" for k, v in components.items())
            parts.append(f"Composed of {comp_str}.")

        # User comments from PocketBase
        user_comments = pb_enrichment.get("user_comments", "")
        if user_comments:
            parts.append(user_comments)

        return " ".join(parts)

    def _sync_pocketbase_metadata(self, data: Dict, search_meta: Dict) -> None:
        """Upsert the derived glycan metadata into the PocketBase glycans collection."""
        if self.pb_client is None or not self.pb_client.is_available():
            return

        archetype = data.get("archetype", {})
        if not isinstance(archetype, dict):
            return

        glycoshape_id = str(archetype.get("ID") or "").strip()
        if not glycoshape_id:
            return

        common_names = self._normalize_string_list(search_meta.get("common_names", []))
        keywords = self._normalize_string_list(search_meta.get("keywords", []))
        name_variants = self._build_name_variants(data)
        aliases = self._build_aliases(data, common_names)
        canonical_name = self._choose_canonical_name(data, common_names)

        payload = {
            "glycoshape_id": glycoshape_id,
            "canonical_name": canonical_name,
            "glycam_name": str(archetype.get("name") or archetype.get("glycam") or "").strip(),
            "iupac_name": str(archetype.get("iupac_extended") or archetype.get("iupac") or "").strip(),
            "glytoucan_id": str(archetype.get("glytoucan") or "").strip(),
            "name_variants": name_variants,
            "aliases": aliases,
            "common_names": common_names,
            "keywords": keywords,
            "description": str(search_meta.get("description") or "").strip(),
        }

        try:
            self.pb_client.upsert_glycan_metadata(payload)
            self.logger.info(f"Upserted glycan metadata record for {glycoshape_id}")
        except Exception as e:
            self.logger.warning(f"Failed to upsert glycan metadata for {glycoshape_id}: {e}")

    @staticmethod
    def _has_value(value) -> bool:
        text = str(value or "").strip()
        return bool(text and text.lower() != "null")

    def _repair_missing_glytoucan_ids(self, data: Dict, source_name: str) -> Dict[str, int]:
        """Fill missing GlyTouCan IDs from existing WURCS values when available."""
        stats = {
            "checked": 0,
            "resolved": 0,
            "still_missing": 0,
            "lookup_failed": 0,
        }

        for entry_type in ("archetype", "alpha", "beta"):
            entry = data.get(entry_type, {})
            if not isinstance(entry, dict):
                continue

            wurcs = str(entry.get("wurcs") or "").strip()
            if not wurcs or wurcs.lower() == "null":
                continue
            if self._has_value(entry.get("glytoucan")):
                continue

            stats["checked"] += 1
            try:
                glytoucan_id = name_utils.wurcs2glytoucan(wurcs)
            except Exception as exc:
                stats["lookup_failed"] += 1
                stats["still_missing"] += 1
                self.logger.warning(
                    f"Failed to resolve GlyTouCan ID for {source_name} ({entry_type}): {exc}"
                )
                continue

            if self._has_value(glytoucan_id):
                entry["glytoucan"] = str(glytoucan_id).strip()
                stats["resolved"] += 1
                self.logger.info(
                    f"Resolved GlyTouCan ID for {source_name} ({entry_type}): {entry['glytoucan']}"
                )
            else:
                stats["still_missing"] += 1

        if stats["checked"]:
            self.logger.info(
                "GlyTouCan repair for %s: checked=%d, resolved=%d, still_missing=%d, lookup_failed=%d",
                source_name,
                stats["checked"],
                stats["resolved"],
                stats["still_missing"],
                stats["lookup_failed"],
            )

        return stats

    def update_glycan_metadata(self, json_file_path: Union[str, Path], enrichment: Optional[Dict] = None) -> bool:
        """Update search metadata for a single glycan JSON file.

        Args:
            json_file_path: Path to the glycan data.json file
            enrichment: curated fields to merge (:func:`search_enrichment`).
                When given, PocketBase is not asked: the caller is the
                source of the curated metadata.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Normalize to Path for consistent .name access
            json_file_path = Path(json_file_path)
            # Read current data
            storage = get_storage_manager()
            with storage.open(json_file_path, 'r') as f:
                data = json.load(f)
                
            # Initialize search_meta if not present
            if "search_meta" not in data:
                data["search_meta"] = {}
                
            search_meta = data["search_meta"]
            
            # Get archetype WURCS and IUPAC for keyword generation
            archetype = data.get("archetype", {})
            iupac_string = archetype.get("iupac")
            wurcs = archetype.get("wurcs")

            # Repair GlyTouCan IDs that may have appeared after static JSON generation.
            self._repair_missing_glytoucan_ids(data, json_file_path.name)

            # Update keywords: the class keywords are decided again and replace
            # the ones a previous run wrote; every other keyword is kept (the
            # curated ones are merged again below).
            if iupac_string or wurcs:
                current_keywords = {
                    k for k in search_meta.get("keywords", []) if k not in glyclass.CLASS_KEYWORDS
                }
                new_keywords = self.get_glycan_keywords(iupac_string, wurcs=wurcs)
                current_keywords.update(new_keywords)
                search_meta["keywords"] = sorted(list(current_keywords))

                if new_keywords:
                    self.logger.info(f"Updated keywords for {json_file_path.name}: {new_keywords}")
            else:
                self.logger.warning(f"No archetype WURCS or IUPAC found in {json_file_path.name}")
                
            # Update common names
            current_common_names = set(search_meta.get("common_names", []))
            new_common_names = self.get_common_names(data)
            current_common_names.update(new_common_names)
            search_meta["common_names"] = sorted(list(current_common_names))
            
            if new_common_names:
                self.logger.info(f"Updated common names for {json_file_path.name}: {new_common_names}")

            # Curated fields: from the caller, or from PocketBase (optional, gracefully degraded)
            if enrichment is not None:
                self._apply_enrichment(data, search_meta, enrichment)
            else:
                self._enrich_from_pocketbase(data, search_meta)

            # Write updated data back to file
            with storage.open(json_file_path, 'w') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

            # Keep PocketBase glycans metadata in sync with the generated static data.
            self._sync_pocketbase_metadata(data, search_meta)
                
            return True
            
        except FileNotFoundError:
            self.logger.error(f"File not found: {json_file_path}")
            return False
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in {json_file_path}: {str(e)}")
            return False
        except Exception as e:
            try:
                name = Path(json_file_path).name  # type: ignore[arg-type]
            except Exception:
                name = str(json_file_path)
            self.logger.error(f"Error updating {name}: {str(e)}")
            return False
    
    def update_all_metadata(self) -> Dict[str, int]:
        """Update metadata for all glycan directories in the database.
        
        Returns:
            Dictionary with statistics about the update process
        """
        stats = {
            "total_processed": 0,
            "successful_updates": 0,
            "failed_updates": 0,
            "missing_files": 0
        }
        
        if not self.database_dir.exists():
            self.logger.error(f"Database directory {self.database_dir} does not exist")
            return stats
            
        self.logger.info(f"Starting metadata update in directory: {self.database_dir}")
        
        # Process each glycan directory
        for glycan_dir in self.database_dir.iterdir():
            if not (glycan_dir.is_dir() and glycan_dir.name.startswith("GS")):
                continue
                
            json_path = glycan_dir / "data.json"
            stats["total_processed"] += 1
            
            if not json_path.exists():
                self.logger.warning(f"data.json not found in {glycan_dir.name}")
                stats["missing_files"] += 1
                continue
                
            self.logger.info(f"Processing {glycan_dir.name}...")
            
            if self.update_glycan_metadata(json_path):
                stats["successful_updates"] += 1
            else:
                stats["failed_updates"] += 1
                
        # Log summary
        self.logger.info("Metadata update process completed")
        self.logger.info(f"Statistics: {stats}")
        
        return stats
    
    def validate_metadata(self) -> Dict[str, List[str]]:
        """Validate metadata across all glycan files.
        
        Returns:
            Dictionary containing validation issues
        """
        issues = {
            "missing_keywords": [],
            "missing_common_names": [],
            "invalid_iupac": [],
            "missing_search_meta": []
        }
        
        for glycan_dir in self.database_dir.glob("GS*"):
            if not glycan_dir.is_dir():
                continue
                
            json_path = glycan_dir / "data.json"
            if not json_path.exists():
                continue
                
            try:
                storage = get_storage_manager()
                json_path_str = str(json_path)
                with storage.open(json_path_str, 'r') as f:
                    data = json.load(f)
                    
                glycan_id = glycan_dir.name
                
                # Check for search_meta presence
                if "search_meta" not in data:
                    issues["missing_search_meta"].append(glycan_id)
                    continue
                    
                search_meta = data["search_meta"]
                
                # Check for keywords
                if not search_meta.get("keywords"):
                    issues["missing_keywords"].append(glycan_id)
                    
                # Check for IUPAC validity
                archetype = data.get("archetype", {})
                iupac = archetype.get("iupac")
                if not iupac:
                    issues["invalid_iupac"].append(glycan_id)
                    
            except Exception as e:
                self.logger.error(f"Error validating {glycan_id}: {str(e)}")
                
        # Log validation results
        total_issues = sum(len(issue_list) for issue_list in issues.values())
        if total_issues == 0:
            self.logger.info("Metadata validation passed - no issues found")
        else:
            self.logger.warning(f"Metadata validation found {total_issues} issues")
            for issue_type, issue_list in issues.items():
                if issue_list:
                    self.logger.warning(f"{issue_type}: {len(issue_list)} entries")
                    
        return issues


def main():
    """Main function to update glycan metadata."""
    processor = GlycanMetadataProcessor(config.output_path)
    
    # Update all metadata
    stats = processor.update_all_metadata()
    
    # Validate metadata
    validation_results = processor.validate_metadata()
    
    # Print summary
    print(f"\nMetadata Update Summary:")
    print(f"Total processed: {stats['total_processed']}")
    print(f"Successful updates: {stats['successful_updates']}")
    print(f"Failed updates: {stats['failed_updates']}")
    print(f"Missing files: {stats['missing_files']}")
    
    total_validation_issues = sum(len(issues) for issues in validation_results.values())
    print(f"\nValidation issues found: {total_validation_issues}")


if __name__ == "__main__":
    main()
