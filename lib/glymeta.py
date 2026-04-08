import lib.config as config
from pathlib import Path
import json
import logging
from typing import Dict, List, Set, Optional, Union
import re
from lib.storage import get_storage_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        
    def get_glycan_keywords(self, iupac_string: str) -> List[str]:
        """Determine keywords based on the IUPAC string.
        
        Args:
            iupac_string: IUPAC notation string for the glycan
            
        Returns:
            List of keywords describing the glycan type
        """
        if not iupac_string:
            return []
            
        keywords = set()
        
        # N-Glycan classification
        keywords.update(self._classify_n_glycan(iupac_string))
        
        # O-Glycan classification
        if self._is_o_glycan(iupac_string):
            keywords.add("O-Glycan")
            
        # GAG (Glycosaminoglycan) classification
        if self._is_gag(iupac_string):
            keywords.add("GAG")
            
        return sorted(list(keywords))
    
    def _classify_n_glycan(self, iupac_string: str) -> Set[str]:
        """Classify N-Glycan types and subtypes.
        
        Args:
            iupac_string: IUPAC notation string
            
        Returns:
            Set of N-Glycan related keywords
        """
        keywords = set()
        
        # Check if it's an N-Glycan
        n_glycan_patterns = [
            r'Man\(b1-4\)GlcNAc\(b1-4\)GlcNAc$',
            r'Man\(b1-4\)GlcNAc\(b1-4\)\[Fuc\(a1-6\)\]GlcNAc$'
        ]
        
        is_n_glycan = any(re.search(pattern, iupac_string) for pattern in n_glycan_patterns)
        
        if not is_n_glycan:
            return keywords
            
        keywords.add("N-Glycan")
        
        # Check for core fucosylation
        if 'Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc' in iupac_string:
            keywords.add("Fucosylated N-Glycan")
            
        # Classify N-Glycan subtypes
        if self._is_oligomannose(iupac_string):
            keywords.add("Oligomannose")
        elif self._is_complex_n_glycan(iupac_string):
            keywords.add("Complex N-Glycan")
            # Add antennary classification
            keywords.update(self._classify_antennary(iupac_string))
        elif self._is_hybrid_n_glycan(iupac_string):
            keywords.add("Hybrid N-Glycan")
            
        return keywords
    
    def _is_oligomannose(self, iupac_string: str) -> bool:
        """Check if the glycan is an oligomannose N-Glycan.
        
        Args:
            iupac_string: IUPAC notation string
            
        Returns:
            True if oligomannose, False otherwise
        """
        # Must have at least 3 mannose residues
        if iupac_string.count('Man') < 3:
            return False
            
        # Exclude patterns that indicate complex/hybrid glycans
        exclusion_patterns = [
            'GlcNAc(b1-2)Man',
            'Gal(',
            'Neu5Ac(',
            'Xyl(',
            'GlcNAc(b1-4)]Man',
            'GlcNAc(b1-6)[GlcNAc(b1-2)]Man',
            'GlcNAc(b1-4)[GlcNAc(b1-2)]Man'
        ]
        
        for pattern in exclusion_patterns:
            if pattern in iupac_string:
                return False
                
        # Special case: allow core fucosylation
        temp_iupac = iupac_string
        if temp_iupac.endswith('Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc'):
            temp_iupac = temp_iupac.replace('[Fuc(a1-6)]GlcNAc', 'GlcNAc')
            
        # No other fucosylation allowed
        if 'Fuc(' in temp_iupac:
            return False
            
        # Check for GlcNAc(b1-4)Man pattern (excluding core)
        if ('GlcNAc(b1-4)Man' in iupac_string and 
            'Man(b1-4)GlcNAc(b1-4)GlcNAc' not in iupac_string):
            return False
            
        return True
    
    def _is_complex_n_glycan(self, iupac_string: str) -> bool:
        """Check if the glycan is a complex N-Glycan.
        
        Args:
            iupac_string: IUPAC notation string
            
        Returns:
            True if complex N-Glycan, False otherwise
        """
        # Must have N-Glycan core
        core_patterns = [
            'Man(b1-4)GlcNAc(b1-4)GlcNAc',
            'Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc',
            'Man(b1-4)GlcNAc(b1-4)[Fuc(a1-3)]GlcNAc'
        ]
        
        has_core = any(pattern in iupac_string for pattern in core_patterns)
        if not has_core:
            return False
            
        # Must have GlcNAc extensions on both arms
        has_glcnac_extension = 'GlcNAc(b1-2)Man' in iupac_string
        
        arm_3_patterns = ['GlcNAc(b1-2)Man(a1-3)', 'GlcNAc(b1-2)]Man(a1-3)']
        arm_6_patterns = ['GlcNAc(b1-2)Man(a1-6)', 'GlcNAc(b1-2)]Man(a1-6)']
        
        has_3_arm = any(pattern in iupac_string for pattern in arm_3_patterns)
        has_6_arm = any(pattern in iupac_string for pattern in arm_6_patterns)
        
        return has_glcnac_extension and has_3_arm and has_6_arm
    
    def _is_hybrid_n_glycan(self, iupac_string: str) -> bool:
        """Check if the glycan is a hybrid N-Glycan.
        
        Args:
            iupac_string: IUPAC notation string
            
        Returns:
            True if hybrid N-Glycan, False otherwise
        """
        # Must have N-Glycan core
        core_patterns = [
            'Man(b1-4)GlcNAc(b1-4)GlcNAc',
            'Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc',
            'Man(b1-4)GlcNAc(b1-4)[Fuc(a1-3)]GlcNAc'
        ]
        
        has_core = any(pattern in iupac_string for pattern in core_patterns)
        if not has_core:
            return False
            
        # Must have more than 3 mannose residues
        if iupac_string.count('Man') <= 3:
            return False
            
        # Must have GlcNAc extension on only one arm (not both)
        arm_3_patterns = ['GlcNAc(b1-2)Man(a1-3)', 'GlcNAc(b1-2)]Man(a1-3)']
        arm_6_patterns = ['GlcNAc(b1-2)Man(a1-6)', 'GlcNAc(b1-2)]Man(a1-6)']
        
        has_3_arm = any(pattern in iupac_string for pattern in arm_3_patterns)
        has_6_arm = any(pattern in iupac_string for pattern in arm_6_patterns)
        
        # Hybrid has extension on one arm but not both
        return (has_3_arm and not has_6_arm) or (has_6_arm and not has_3_arm)
    
    def _classify_antennary(self, iupac_string: str) -> Set[str]:
        """Classify antennary structure of complex N-Glycans.
        
        Args:
            iupac_string: IUPAC notation string
            
        Returns:
            Set of antennary keywords
        """
        keywords = set()
        
        # Count GlcNAc attachments on each arm
        arm_3_attachments = 0
        arm_6_attachments = 0
        
        # Check for different attachment patterns on 3-arm
        if any(pattern in iupac_string for pattern in 
               ['GlcNAc(b1-2)Man(a1-3)', 'GlcNAc(b1-2)]Man(a1-3)']):
            arm_3_attachments += 1
            
        if any(pattern in iupac_string for pattern in
               ['GlcNAc(b1-4)Man(a1-3)', 'GlcNAc(b1-4)[Man(a1-3)', 
                'GlcNAc(b1-4)[GlcNAc(b1-2)]Man(a1-3)']):
            arm_3_attachments += 1
            
        # Check for different attachment patterns on 6-arm  
        if any(pattern in iupac_string for pattern in
               ['GlcNAc(b1-2)Man(a1-6)', 'GlcNAc(b1-2)]Man(a1-6)']):
            arm_6_attachments += 1
            
        if any(pattern in iupac_string for pattern in
               ['GlcNAc(b1-6)Man(a1-6)', 'GlcNAc(b1-6)[Man(a1-6)',
                'GlcNAc(b1-6)[GlcNAc(b1-2)]Man(a1-6)']):
            arm_6_attachments += 1
            
        # Classify based on total antennae
        total_antennae = arm_3_attachments + arm_6_attachments
        has_both_arms = arm_3_attachments > 0 and arm_6_attachments > 0
        
        if has_both_arms:
            if total_antennae == 2:
                keywords.add("Biantennary N-Glycan")
            elif total_antennae == 3:
                keywords.add("Triantennary N-Glycan")
            elif total_antennae >= 4:
                keywords.add("Tetraantennary N-Glycan")
                
        return keywords
    
    def _is_o_glycan(self, iupac_string: str) -> bool:
        """Check if the glycan is an O-Glycan.
        
        Args:
            iupac_string: IUPAC notation string
            
        Returns:
            True if O-Glycan, False otherwise
        """
        o_glycan_patterns = [
            r'GalNAc$',
            'Gal(b1-3)GalNAc',
            'GlcNAc(b1-6)[Gal(b1-3)]GalNAc',
            'GlcNAc(b1-3)GalNAc',
            'GlcNAc(b1-6)[GlcNAc(b1-3)]GalNAc'
        ]
        
        return any(pattern in iupac_string for pattern in o_glycan_patterns)
    
    def _is_gag(self, iupac_string: str) -> bool:
        """Check if the glycan is a GAG (Glycosaminoglycan).
        
        Args:
            iupac_string: IUPAC notation string
            
        Returns:
            True if GAG, False otherwise
        """
        gag_patterns = [
            'GlcA(b1-3)Gal(b1-3)Gal(b1-4)Xyl',  # Linkage region
            'IdoA'  # Iduronic acid
        ]
        
        # Check for specific GAG patterns
        for pattern in gag_patterns:
            if pattern in iupac_string:
                return True
                
        # Check for combinations indicating GAGs
        gag_combinations = [
            ('GlcA', 'GlcNAc'),   # Heparin/Heparan sulfate
            ('GlcA', 'GalNAc'),   # Chondroitin/Dermatan sulfate
            ('GlcN', 'GlcA'),     # Heparin variants
            ('GlcN', 'IdoA')      # Heparin variants
        ]
        
        for combo in gag_combinations:
            if all(component in iupac_string for component in combo):
                return True
                
        return False
    
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

            # Add extra keywords (additive, never replace)
            extra_kw = enrichment.get("extra_keywords", [])
            if extra_kw:
                current = set(search_meta.get("keywords", []))
                current.update(extra_kw)
                search_meta["keywords"] = sorted(list(current))

            # Add user-curated common names from PocketBase (additive)
            user_common_names = enrichment.get("user_common_names", [])
            if user_common_names:
                current_names = set(search_meta.get("common_names", []))
                current_names.update(user_common_names)
                search_meta["common_names"] = sorted(list(current_names))

            # Use user-curated description from PocketBase, or generate one
            user_desc = enrichment.get("user_description", "")
            if user_desc:
                search_meta["description"] = user_desc
            elif not search_meta.get("description"):
                search_meta["description"] = self._generate_description(
                    data, enrichment
                )

            self.logger.info(f"Enriched search_meta from PocketBase for {gs_id}")

        except Exception as e:
            self.logger.warning(f"PocketBase enrichment failed for {gs_id}: {e}")

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
            if k in (
                "N-Glycan", "O-Glycan", "GAG", "Oligomannose",
                "Complex N-Glycan", "Hybrid N-Glycan",
                "Biantennary N-Glycan", "Triantennary N-Glycan",
                "Tetraantennary N-Glycan", "Fucosylated N-Glycan",
            )
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

    def update_glycan_metadata(self, json_file_path: Union[str, Path]) -> bool:
        """Update search metadata for a single glycan JSON file.
        
        Args:
            json_file_path: Path to the glycan data.json file
            
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
            
            # Get archetype IUPAC for keyword generation
            archetype = data.get("archetype", {})
            iupac_string = archetype.get("iupac")
            
            # Update keywords
            if iupac_string:
                current_keywords = set(search_meta.get("keywords", []))
                new_keywords = self.get_glycan_keywords(iupac_string)
                current_keywords.update(new_keywords)
                search_meta["keywords"] = sorted(list(current_keywords))
                
                if new_keywords:
                    self.logger.info(f"Updated keywords for {json_file_path.name}: {new_keywords}")
            else:
                self.logger.warning(f"No archetype IUPAC found in {json_file_path.name}")
                
            # Update common names
            current_common_names = set(search_meta.get("common_names", []))
            new_common_names = self.get_common_names(data)
            current_common_names.update(new_common_names)
            search_meta["common_names"] = sorted(list(current_common_names))
            
            if new_common_names:
                self.logger.info(f"Updated common names for {json_file_path.name}: {new_common_names}")

            # Enrich from PocketBase (optional, gracefully degraded)
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
