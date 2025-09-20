import lib.config as config
from pathlib import Path
import shutil
import logging
from datetime import datetime
import zipfile
import glob
import json
import requests
from typing import Dict, List, Union, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GlycoShapeBaker:
    """Main class for creating GlycoShape database archives and files."""
    
    def __init__(self, database_dir: Union[str, Path]):
        """Initialize the baker with database directory.
        
        Args:
            database_dir: Path to the database directory containing glycan folders
        """
        self.database_dir = Path(database_dir)
        self.logger = logging.getLogger(self.__class__.__name__)
        
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
                
                # Add files from output/level_2/ directly to root (flatten contents)
                level_2_dir = glycan_dir / "output" / "level_2"
                if level_2_dir.exists() and level_2_dir.is_dir():
                    for file in level_2_dir.rglob('*'):
                        if file.is_file():
                            # Add directly to root of zip (flatten - no level_2/ prefix)
                            arcname = str(file.relative_to(level_2_dir))
                            zf.write(file, arcname)
                            self.logger.debug(f"Added {arcname} from level_2 to archive root")
                else:
                    self.logger.warning(f"level_2 directory not found in {glycan_dir}/output")
                            
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
                        with open(json_path, 'r', encoding='utf-8') as f:
                            consolidated_data[glycan_id] = json.load(f)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Invalid JSON in {json_path}: {str(e)}")
                    except Exception as e:
                        self.logger.error(f"Error reading {json_path}: {str(e)}")
                        
            # Write consolidated JSON
            self.logger.info(f"Writing consolidated JSON to {output_file}")
            with open(output_path, 'w', encoding='utf-8') as f:
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
            
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            missing_entries = []
            
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
                    
                    if wurcs and not glytoucan:
                        missing_entries.append(wurcs)
                        self.logger.info(f"Missing GlyTouCan ID for {glycan_id} ({entry_type}): {wurcs}")

            # Write missing entries to file
            output_path = self.database_dir / output_file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(missing_entries))
                
            self.logger.info(f"Found {len(missing_entries)} WURCS entries missing GlyTouCan IDs")
            
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
            with open(file_path, 'rb') as file:
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
            
            with open(json_path, 'r', encoding='utf-8') as f:
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
                    with open(data_json, 'r', encoding='utf-8') as f:
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
