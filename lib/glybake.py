import lib.config as config
from pathlib import Path
import shutil
import logging
import hashlib
from datetime import datetime, timezone
import zipfile
import glob
import json
import requests
from typing import Any, Dict, List, Union, Optional
from lib import name_utils
from lib.storage import get_storage_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GlycoShapeBaker:
    """Main class for creating GlycoShape database archives and files."""
    ARCHIVE_OUTPUT_LEVELS = ("level_1",)
    
    def __init__(self, database_dir: Union[str, Path]):
        """Initialize the baker with database directory.
        
        Args:
            database_dir: Path to the database directory containing glycan folders
        """
        self.database_dir = Path(database_dir)
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _file_hash_info(path: Path) -> Dict[str, Any]:
        """Return stable hash metadata for a file."""
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)

        stat = path.stat()
        return {
            "sha256": digest.hexdigest(),
            "size": size,
            "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        }

    @staticmethod
    def _aggregate_hash(entries: List[str]) -> str:
        """Hash sorted manifest entry lines into one deterministic digest."""
        digest = hashlib.sha256()
        for entry in sorted(entries):
            digest.update(entry.encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    def create_sync_manifest(self, output_name: str = "sync_manifest.json") -> Path:
        """Create a static sync manifest for fast Oracle-to-host updates.

        The manifest is intentionally self-contained:
        - root_files covers top-level static artifacts such as GLYCOSHAPE.json
        - glycans[GSxxxxx].hash is the quick folder-level update key
        - glycans[GSxxxxx].files is the exact file list for download/verification
        """
        root = self.database_dir
        manifest_path = root / output_name

        root_files: Dict[str, Dict[str, Any]] = {}
        for file_path in sorted(root.iterdir()):
            if not file_path.is_file() or file_path.name == output_name:
                continue
            info = self._file_hash_info(file_path)
            root_files[file_path.name] = {
                "path": file_path.name,
                "sha256": info["sha256"],
                "size": info["size"],
                "mtime": info["mtime"],
            }

        glycans: Dict[str, Dict[str, Any]] = {}
        for glycan_dir in sorted(root.glob("GS*")):
            if not glycan_dir.is_dir():
                continue

            glycan_id = glycan_dir.name
            files: Dict[str, Dict[str, Any]] = {}
            aggregate_entries: List[str] = []
            total_size = 0

            for file_path in sorted(p for p in glycan_dir.rglob("*") if p.is_file()):
                rel_to_glycan = file_path.relative_to(glycan_dir).as_posix()
                rel_to_static = file_path.relative_to(root).as_posix()
                info = self._file_hash_info(file_path)
                total_size += info["size"]
                files[rel_to_glycan] = {
                    "path": rel_to_static,
                    "sha256": info["sha256"],
                    "size": info["size"],
                    "mtime": info["mtime"],
                }
                aggregate_entries.append(f"{rel_to_glycan}\0{info['sha256']}\0{info['size']}")

            glycans[glycan_id] = {
                "hash": self._aggregate_hash(aggregate_entries),
                "file_count": len(files),
                "total_size": total_size,
                "data_json_sha256": files.get("data.json", {}).get("sha256"),
                "zip_sha256": files.get(f"{glycan_id}.zip", {}).get("sha256"),
                "files": files,
            }

        global_entries = [
            f"root/{name}\0{info['sha256']}\0{info['size']}"
            for name, info in root_files.items()
        ] + [
            f"glycan/{glycan_id}\0{info['hash']}\0{info['file_count']}\0{info['total_size']}"
            for glycan_id, info in glycans.items()
        ]

        manifest: Dict[str, Any] = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "hash_algorithm": "sha256",
            "static_prefix": "static",
            "manifest_path": output_name,
            "manifest_excludes": [output_name],
            "global_hash": self._aggregate_hash(global_entries),
            "glycan_count": len(glycans),
            "root_files": root_files,
            "glycans": glycans,
        }

        tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        tmp_path.replace(manifest_path)

        self.logger.info(
            "Created sync manifest %s with %d glycans and %d root files",
            manifest_path,
            len(glycans),
            len(root_files),
        )
        return manifest_path
        
    def create_individual_glycan_archive(self, glycan_dir: Path) -> bool:
        """Create archive for individual glycan with new structure.
        
        Args:
            glycan_dir: Path to glycan directory (e.g., GS00445/)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not glycan_dir.is_dir():
            self.logger.warning(f"Directory {glycan_dir} does not exist")
            return False
            
        glycan_id = glycan_dir.name
        archive_path = glycan_dir / f"{glycan_id}.zip"
        
        # Remove existing archive
        if archive_path.exists():
            archive_path.unlink()
            self.logger.info(f"Removed existing {glycan_id}.zip")
        
        try:
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Add main files
                main_files = [
                    "data.json",
                    "snfg.svg",
                    "glycosmos.json", 
                    "glygen.json"
                ]
                
                for file_name in main_files:
                    file_path = glycan_dir / file_name
                    if file_path.exists():
                        zf.write(file_path, file_name)
                
                # Add format directories (HETATM and ATOM versions)
                format_dirs = [
                    "PDB_format_HETATM",
                    "CHARMM_format_HETATM", 
                    "GLYCAM_format_HETATM",
                    "PDB_format_ATOM",
                    "CHARMM_format_ATOM",
                    "GLYCAM_format_ATOM"
                ]
                
                for format_dir in format_dirs:
                    format_path = glycan_dir / format_dir
                    if format_path.exists() and format_path.is_dir():
                        for file in format_path.rglob('*'):
                            if file.is_file():
                                arcname = format_dir + '/' + str(file.relative_to(format_path))
                                zf.write(file, arcname)
                
                # Only include the representative archive-ready outputs.
                output_dir = glycan_dir / "output"
                level_dirs = (
                    [output_dir / level_name for level_name in self.ARCHIVE_OUTPUT_LEVELS if (output_dir / level_name).is_dir()]
                    if output_dir.exists()
                    else []
                )
                if level_dirs:
                    for level_dir in level_dirs:
                        for file in sorted(level_dir.rglob('*')):
                            if file.is_file():
                                arcname = str(file.relative_to(level_dir))
                                zf.write(file, arcname)
                                self.logger.debug(f"Added {arcname} to archive")
                else:
                    expected_levels = ", ".join(self.ARCHIVE_OUTPUT_LEVELS)
                    self.logger.warning(f"No archive output levels ({expected_levels}) found in {glycan_dir}")
                            
            self.logger.info(f"Successfully created archive for {glycan_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create archive for {glycan_id}: {str(e)}")
            return False

    def create_master_archive(self, output_file: str = "GlycoShape.zip") -> None:
        """Create master GlycoShape archive from all individual glycan archives.
        
        Args:
            output_file: Name of the master archive file
        """
        try:
            output_path = self.database_dir / output_file
            
            # Remove existing master archive
            if output_path.exists():
                output_path.unlink()
                self.logger.info(f"Removed existing {output_file}")

            # Create individual archives first
            self.logger.info("Creating individual glycan archives...")
            success_count = 0
            total_count = 0
            
            for glycan_dir in self.database_dir.glob("GS*"):
                if glycan_dir.is_dir():
                    total_count += 1
                    if self.create_individual_glycan_archive(glycan_dir):
                        success_count += 1

            self.logger.info(f"Created {success_count}/{total_count} individual archives")

            # Create master archive
            self.logger.info("Creating master archive...")
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as master_zip:
                for glycan_dir in self.database_dir.glob("GS*"):
                    if glycan_dir.is_dir():
                        archive_path = glycan_dir / f"{glycan_dir.name}.zip"
                        if archive_path.exists():
                            master_zip.write(archive_path, archive_path.name)

            self.logger.info(f"Successfully created master archive: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to create master archive: {str(e)}")
            raise

    def create_consolidated_json(self, output_file: str = "GLYCOSHAPE.json") -> None:
        """Create consolidated JSON file from all data.json files.
        
        Args:
            output_file: Name of output JSON file
        """
        try:
            output_path = self.database_dir / output_file
            consolidated_data = {}
            
            # Process each glycan directory
            for glycan_dir in self.database_dir.glob("GS*"):
                if not glycan_dir.is_dir():
                    continue
                    
                glycan_id = glycan_dir.name
                json_path = glycan_dir / "data.json"
                
                if json_path.exists():
                    self.logger.info(f"Processing JSON for {glycan_id}")
                    try:
                        storage = get_storage_manager()
                        json_path_str = str(json_path)
                        with storage.open(json_path_str, 'r') as f:
                            consolidated_data[glycan_id] = json.load(f)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Invalid JSON in {json_path}: {str(e)}")
                    except Exception as e:
                        self.logger.error(f"Error reading {json_path}: {str(e)}")
                        
            # Write consolidated JSON
            self.logger.info(f"Writing consolidated JSON to {output_file}")
            with storage.open(output_path, 'w') as f:
                json.dump(
                    dict(sorted(consolidated_data.items())), 
                    f, 
                    indent=2, 
                    ensure_ascii=False
                )
                
            self.logger.info(f"Successfully created consolidated JSON with {len(consolidated_data)} entries")
            
        except Exception as e:
            self.logger.error(f"Failed to create consolidated JSON: {str(e)}")
            raise

    def extract_missing_glytoucan_ids(
        self, 
        json_file: str = "GLYCOSHAPE.json", 
        output_file: str = "missing_glytoucan.txt"
    ) -> None:
        """Extract WURCS entries that need GlyTouCan registration.
        
        Args:
            json_file: Input JSON file to analyze
            output_file: Output file for missing GlyTouCan entries
        """
        try:
            json_path = self.database_dir / json_file
            
            storage = get_storage_manager()
            json_path_str = str(json_path)
            with storage.open(json_path_str, 'r') as f:
                data = json.load(f)

            missing_entries = []
            unresolved_wurcs_entries = []
            calculated_wurcs_by_glycan = {}

            def calculate_wurcs_from_iupac(glycan_id: str, entry_type: str, iupac: str):
                calculated_glytoucan, calculated_wurcs = name_utils.iupac2wurcs_glytoucan(iupac)
                if calculated_wurcs:
                    return calculated_glytoucan, calculated_wurcs

                try:
                    smiles = name_utils.iupac2smiles(iupac)
                except Exception as exc:
                    self.logger.warning(
                        f"Failed to calculate SMILES for {glycan_id} ({entry_type}) from IUPAC {iupac}: {exc}"
                    )
                    return calculated_glytoucan, None

                if not smiles:
                    return calculated_glytoucan, None

                calculated_wurcs = name_utils.smiles2wurcs(smiles)
                if calculated_wurcs:
                    variants = name_utils.get_wurcs_variants(calculated_wurcs)
                    if variants:
                        calculated_wurcs_by_glycan[glycan_id] = {
                            "archetype": variants[0],
                            "alpha": variants[1],
                            "beta": variants[2],
                        }
                        calculated_wurcs = calculated_wurcs_by_glycan[glycan_id].get(entry_type) or calculated_wurcs
                    calculated_glytoucan = calculated_glytoucan or name_utils.wurcs2glytoucan(calculated_wurcs)

                return calculated_glytoucan, calculated_wurcs
            
            # Check both archetype and alpha/beta entries
            for glycan_id, glycan_data in data.items():
                entries_to_check = ['archetype']
                
                # Add alpha/beta if they exist
                if 'alpha' in glycan_data:
                    entries_to_check.append('alpha')
                if 'beta' in glycan_data:
                    entries_to_check.append('beta')
                
                for entry_type in entries_to_check:
                    entry = glycan_data.get(entry_type, {})
                    wurcs = entry.get("wurcs")
                    glytoucan = entry.get("glytoucan")
                    iupac = entry.get("iupac")
                    
                    if not wurcs and not glytoucan and iupac:
                        try:
                            calculated_wurcs = calculated_wurcs_by_glycan.get(glycan_id, {}).get(entry_type)
                            calculated_glytoucan = None
                            if not calculated_wurcs:
                                calculated_glytoucan, calculated_wurcs = calculate_wurcs_from_iupac(glycan_id, entry_type, iupac)
                        except Exception as exc:
                            calculated_glytoucan, calculated_wurcs = None, None
                            self.logger.warning(
                                f"Failed to calculate WURCS for {glycan_id} ({entry_type}) from IUPAC {iupac}: {exc}"
                            )
                        if calculated_glytoucan:
                            glytoucan = calculated_glytoucan
                        if calculated_wurcs:
                            wurcs = calculated_wurcs

                    if wurcs and not glytoucan:
                        if wurcs not in missing_entries:
                            missing_entries.append(wurcs)
                    elif not wurcs and not glytoucan:
                        unresolved_wurcs_entries.append({
                            "glycoshape_id": glycan_id,
                            "variant": entry_type,
                            "iupac": entry.get("iupac") or "",
                            "glycam": entry.get("glycam") or entry.get("name") or "",
                        })

            # Stored static JSON may predate an accession, and generated WURCS
            # can be structurally valid but non-canonical. Recheck canonical
            # WURCS in batches before calling anything a registration candidate.
            lookup_candidates = [
                wurcs for wurcs in missing_entries
                if str(wurcs).startswith("WURCS=2.0/")
            ]
            resolved_by_wurcs = name_utils.wurcs2glytoucan_many(lookup_candidates)
            missing_entries = [
                wurcs for wurcs in missing_entries
                if not resolved_by_wurcs.get(wurcs)
            ]

            # Write missing entries to file
            output_path = f"{self.database_dir}/{output_file}"
            storage = get_storage_manager()
            with storage.open(output_path, 'w') as f:
                f.write("\n".join(missing_entries))
                if missing_entries:
                    f.write("\n")

            unresolved_output_path = f"{self.database_dir}/missing_wurcs.txt"
            with storage.open(unresolved_output_path, 'w') as f:
                f.write("glycoshape_id\tvariant\tiupac\tglycam\n")
                for unresolved in unresolved_wurcs_entries:
                    f.write(
                        "\t".join([
                            unresolved["glycoshape_id"],
                            unresolved["variant"],
                            unresolved["iupac"],
                            unresolved["glycam"],
                        ]) + "\n"
                    )
                
            self.logger.info(f"Found {len(missing_entries)} WURCS entries missing GlyTouCan IDs")
            if unresolved_wurcs_entries:
                self.logger.warning(
                    f"Found {len(unresolved_wurcs_entries)} entries missing both WURCS and GlyTouCan; "
                    f"wrote {unresolved_output_path}"
                )
            
        except Exception as e:
            self.logger.error(f"Failed to extract missing GlyTouCan IDs: {str(e)}")
            raise

    def submit_wurcs_to_glytoucan(
        self, 
        contributor_id: str, 
        api_key: str, 
        file_path: Union[str, Path]
    ) -> None:
        """Submit WURCS data to GlyTouCan API for registration.
        
        Args:
            contributor_id: GlyTouCan contributor ID
            api_key: GlyTouCan API key
            file_path: Path to file containing WURCS data
        """
        url = "https://glytoucan.org/api/bulkload/wurcs"
        file_path = Path(file_path)
        
        try:
            storage = get_storage_manager()
            with storage.open(file_path, 'rb') as file:
                response = requests.post(
                    url,
                    files={"file": file},
                    auth=(contributor_id, api_key),
                    timeout=300  # 5 minute timeout
                )
            
            if response.status_code == 200:
                self.logger.info("Successfully submitted WURCS data to GlyTouCan")
                self.logger.info(f"Response: {response.text}")
            else:
                self.logger.error(f"Failed to submit WURCS data: {response.status_code}")
                self.logger.error(f"Response: {response.text}")
                response.raise_for_status()
        
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error during WURCS submission: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error during WURCS submission: {str(e)}")
            raise

    def calculate_total_simulation_length(self, json_file: str = "GLYCOSHAPE.json") -> float:
        """Calculate total simulation length from all entries.
        
        Args:
            json_file: JSON file to analyze
            
        Returns:
            Total simulation length in microseconds
        """
        try:
            json_path = self.database_dir / json_file
            
            storage = get_storage_manager()
            json_path_str = str(json_path)
            with storage.open(json_path_str, 'r') as f:
                data = json.load(f)
            
            total_length = 0.0
            processed_entries = 0
            
            for glycan_id, glycan_data in data.items():
                archetype = glycan_data.get("archetype", {})
                length = archetype.get("length")
                
                if length is not None:
                    try:
                        total_length += float(length)
                        processed_entries += 1
                    except (ValueError, TypeError):
                        self.logger.warning(f"Invalid length value for {glycan_id}: {length}")
            
            self.logger.info(f"Total simulation length: {total_length} μs from {processed_entries} entries")
            return total_length
        
        except Exception as e:
            self.logger.error(f"Failed to calculate total simulation length: {str(e)}")
            raise

    def validate_database_structure(self) -> Dict[str, List[str]]:
        """Validate the database structure and report any issues.
        
        Returns:
            Dictionary with validation results
        """
        issues = {
            "missing_data_json": [],
            "missing_snfg_svg": [],
            "missing_output_dir": [],
            "missing_format_dirs": [],
            "invalid_json": []
        }
        
        required_files = ["data.json", "snfg.svg"]
        required_format_dirs = [
            "PDB_format_HETATM", "CHARMM_format_HETATM", "GLYCAM_format_HETATM",
            "PDB_format_ATOM", "CHARMM_format_ATOM", "GLYCAM_format_ATOM"
        ]
        
        for glycan_dir in self.database_dir.glob("GS*"):
            if not glycan_dir.is_dir():
                continue
                
            glycan_id = glycan_dir.name
            
            # Check required files
            for file_name in required_files:
                if not (glycan_dir / file_name).exists():
                    issues[f"missing_{file_name.replace('.', '_')}"].append(glycan_id)
            
            # Check data.json validity
            data_json = glycan_dir / "data.json"
            if data_json.exists():
                try:
                    storage = get_storage_manager()
                    data_json_str = str(data_json)
                    with storage.open(data_json_str, 'r') as f:
                        json.load(f)
                except json.JSONDecodeError:
                    issues["invalid_json"].append(glycan_id)
            
            # Check output directory
            if not (glycan_dir / "output").exists():
                issues["missing_output_dir"].append(glycan_id)
            
            # Check format directories
            missing_formats = []
            for format_dir in required_format_dirs:
                if not (glycan_dir / format_dir).exists():
                    missing_formats.append(format_dir)
            
            if missing_formats:
                issues["missing_format_dirs"].append(f"{glycan_id}: {', '.join(missing_formats)}")
        
        # Log summary
        total_issues = sum(len(issue_list) for issue_list in issues.values())
        if total_issues == 0:
            self.logger.info("Database structure validation passed - no issues found")
        else:
            self.logger.warning(f"Database validation found {total_issues} issues")
            for issue_type, issue_list in issues.items():
                if issue_list:
                    self.logger.warning(f"{issue_type}: {len(issue_list)} entries")
        
        return issues

    def bake_all(self, include_wurcs_submission: bool = False) -> None:
        """Execute all baking operations in the correct order.
        
        Args:
            include_wurcs_submission: Whether to submit WURCS data to GlyTouCan
        """
        self.logger.info("Starting GlycoShape database baking process...")
        
        try:
            # Validate database structure first
            self.logger.info("Validating database structure...")
            validation_results = self.validate_database_structure()
            
            # (FAQ creation removed)
            
            # Create consolidated JSON
            self.logger.info("Creating consolidated JSON...")
            self.create_consolidated_json()
            
            # Calculate simulation statistics
            self.logger.info("Calculating simulation statistics...")
            total_length = self.calculate_total_simulation_length()
            
            # Create archives
            self.logger.info("Creating archives...")
            self.create_master_archive()
            
            # Extract missing GlyTouCan IDs
            self.logger.info("Extracting missing GlyTouCan IDs...")
            self.extract_missing_glytoucan_ids()
            
            # Optionally submit WURCS data
            if include_wurcs_submission:
                if hasattr(config, 'contributor_id') and hasattr(config, 'api_key'):
                    self.logger.info("Submitting WURCS data to GlyTouCan...")
                    self.submit_wurcs_to_glytoucan(
                        config.contributor_id,
                        config.api_key,
                        self.database_dir / "missing_glytoucan.txt"
                    )
                else:
                    self.logger.warning("GlyTouCan credentials not configured - skipping WURCS submission")
            
            self.logger.info("GlycoShape database baking completed successfully!")
            self.logger.info(f"Total simulation time: {total_length} μs")
            
        except Exception as e:
            self.logger.error(f"Database baking failed: {str(e)}")
            raise


def main():
    """Main function to run the baking process."""
    baker = GlycoShapeBaker(config.output_path)
    baker.bake_all(include_wurcs_submission=False)  # Set to True to submit WURCS data


if __name__ == "__main__":
    main()
