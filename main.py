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
from pathlib import Path
from typing import List, Dict, Optional
import json

import lib.config as config
from lib.storage import get_storage_manager, reset_storage_manager
from lib.glystatic import process_glycan as glystatic_process_glycan
from lib.glymeta import GlycanMetadataProcessor
from lib.glybake import GlycoShapeBaker
from lib.v2 import run_glycan_analysis, get_glycan_status


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GlycanPipelineRunner:
    """Main pipeline runner for processing glycans."""
    
    def __init__(self, data_dir, process_dir, output_dir, force_update: bool = False):
        """Initialize the pipeline runner.
        
        Args:
            data_dir: Directory containing glycan input data (read-only)
            process_dir: Directory for storing processing outputs (data, embedding, output)
            output_dir: Directory for final database output
            force_update: Whether to force recomputation of existing results
        """
        self.data_dir = data_dir
        self.process_dir = process_dir 
        self.output_dir = output_dir
        self.force_update = force_update
        
        # Initialize storage manager
        self.storage = get_storage_manager()
        
        # Ensure directories exist
        self.storage.mkdir(self.data_dir)
        self.storage.mkdir(self.process_dir)
        self.storage.mkdir(self.output_dir)
        
        # Initialize processors
        self.metadata_processor = GlycanMetadataProcessor(self.output_dir)
        self.baker = GlycoShapeBaker(self.output_dir)
    
    def _remote_output_base(self) -> str:
        """Return the remote base prefix for outputs.

        In Oracle mode, we always upload/list under 'static' in the bucket,
        regardless of the local output_dir path. Locally, use output_dir.
        """
        return "static" if config.use_oracle_storage else str(self.output_dir)
        
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
            # For Oracle, list immediate directories under data/ using delimiter (fast)
            logger.info("Using Oracle storage - fast listing of glycan directories under data prefix")
            try:
                dirs = self.storage.list_dirs(self.data_dir)
            except Exception as e:
                logger.error(f"Failed to list directories under {self.data_dir}: {e}")
                dirs = []

            candidates = sorted([d.name if hasattr(d, 'name') else str(d).split('/')[-1] for d in dirs])

            for cand in sorted(candidates):
                # Accept either <cand>.pdb/.mol2 or simulation.pdb/structure.mol2
                pdb_file_1 = f"{cand}/{cand}.pdb"
                mol2_file_1 = f"{cand}/{cand}.mol2"
                pdb_file_2 = f"{cand}/simulation.pdb"
                mol2_file_2 = f"{cand}/structure.mol2"
                ok_named = self.storage.exists(pdb_file_1, self.data_dir) and self.storage.exists(mol2_file_1, self.data_dir)
                ok_alt = self.storage.exists(pdb_file_2, self.data_dir) and self.storage.exists(mol2_file_2, self.data_dir)
                if ok_named or ok_alt:
                    glycan_dirs.append(cand)
                    logger.info(f"Found glycan: {cand}")
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
            # Validate JSON structure
            with self.storage.open(data_json, 'r') as f:
                data = json.load(f)
                
            # Check for required sections
            required_sections = ["archetype"]
            for section in required_sections:
                if section not in data:
                    logger.debug(f"Missing required section '{section}' in {glycan_name}/data.json")
                    return False
            
            # Check for basic archetype data
            archetype = data.get("archetype", {})
            required_archetype_fields = ["ID", "name", "iupac", "glytoucan"]
            for field in required_archetype_fields:
                if field not in archetype or not archetype[field]:
                    logger.debug(f"Missing or empty required archetype field '{field}' in {glycan_name}/data.json")
                    return False
            
            # Specifically validate that glytoucan ID exists and is not null
            glytoucan_id = archetype.get("glytoucan")
            if not glytoucan_id or glytoucan_id == "null" or glytoucan_id.strip() == "":
                logger.debug(f"Missing or invalid glytoucan ID for {glycan_name}")
                return False
                    
            logger.debug(f"Output structure validated for {glycan_name} with glytoucan ID: {glytoucan_id}")
            return True
            
        except json.JSONDecodeError as e:
            logger.debug(f"Invalid JSON in {glycan_name}/data.json: {str(e)}")
            return False
        except Exception as e:
            logger.debug(f"Error validating output for {glycan_name}: {str(e)}")
            return False
    
    def check_glystatic_simple(self, glycan_name: str) -> bool:
        """Simple check if glystatic processing should be skipped.
        
        This is used when GLYCOSHAPE_DB_UPDATE is False - we only need to check
        if there's a valid data.json with a glytoucan ID, not the full file structure.
        
        Args:
            glycan_name: Name of the glycan
            
        Returns:
            True if a valid data.json exists with glytoucan ID, False otherwise
        """
        try:
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
                
                # Simple validation: just check if glytoucan ID exists and is valid
                glytoucan_id = archetype.get("glytoucan", "").strip()
                if glytoucan_id and glytoucan_id.lower() != "null" and glytoucan_id != "":
                    logger.debug(f"Found valid glytoucan ID '{glytoucan_id}' for {glycan_name} in {candidate_name}")
                    return True
            
            logger.debug(f"No valid data.json with glytoucan ID found for {glycan_name}")
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
                        break
                except Exception:
                    continue

            if not matching_key:
                logger.debug(f"No GS folder found with exact name match for {glycan_name}")
                return False

            # Validate minimal output structure
            validation_errors = []
            data_json = f"{matching_key}/data.json"
            try:
                with storage.open(data_json, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read data.json for {matching_key}: {e}")
                return False

            if not isinstance(data, dict):
                validation_errors.append("data.json is not a dictionary")
            archetype = data.get('archetype', {}) if isinstance(data, dict) else {}
            if not isinstance(archetype, dict):
                validation_errors.append("missing or invalid archetype section")
            required_fields = ["ID", "name", "iupac", "glytoucan"]
            missing_fields = [f for f in required_fields if f not in archetype or not archetype[f]]
            if missing_fields:
                validation_errors.append(f"missing required archetype fields: {missing_fields}")
            glytoucan_id = (archetype.get("glytoucan", "") or "").strip()
            if not glytoucan_id or glytoucan_id.lower() == "null":
                validation_errors.append(f"invalid glytoucan ID '{glytoucan_id}'")

            out_folder = f"{matching_key}/output"
            if not storage.exists(out_folder):
                validation_errors.append("missing output/ folder")
            for rel in ["pca.csv", "torparts.npz"]:
                if not storage.exists(f"{out_folder}/{rel}"):
                    validation_errors.append(f"missing output file: {rel}")
            if not storage.exists(f"{out_folder}/level_1"):
                validation_errors.append("missing level_1 structures")

            if validation_errors:
                logger.warning(f"Glystatic validation failed for {glycan_name} in {matching_key}. Issues:")
                for err in validation_errors:
                    logger.warning(f"  - {err}")
                return False

            logger.debug(f"Glystatic completion validated in {matching_key} for {glycan_name} (GlyTouCan: {glytoucan_id})")
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
            status = get_glycan_status(glycan_name, str(self.data_dir), str(self.process_dir))
            
            # Check if all critical steps are completed
            critical_steps = ['store', 'pca', 'clustering', 'export']
            all_critical_complete = all(
                status['steps_status'].get(step, False) 
                for step in critical_steps
            )
            
            return all_critical_complete
            
        except Exception as e:
            logger.debug(f"Error checking v2 status for {glycan_name}: {str(e)}")
            return False
    
    def run_glystatic(self, glycan_name: str) -> bool:
        """Run glystatic processing for a single glycan.
        
        Args:
            glycan_name: Name of the glycan directory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_glycan_dir = f"{self.output_dir}/{glycan_name}"
            
            # Check if glystatic processing should be skipped
            if not self.force_update:
                if not config.update:
                    # When GLYCOSHAPE_DB_UPDATE is False, use simple check
                    if self.check_glystatic_simple(glycan_name):
                        logger.info(f"Glystatic processing already complete for {glycan_name} (found valid data.json with glytoucan ID), skipping")
                        return True
                else:
                    # When GLYCOSHAPE_DB_UPDATE is True, use full validation
                    if self.check_glystatic_completion(glycan_name):
                        logger.info(f"Glystatic processing already complete for {glycan_name}, skipping")
                        return True
                
            logger.info(f"Running glystatic for {glycan_name}")
            
            # Define paths for glystatic processing
            glycan_input_dir = f"{self.data_dir}/{glycan_name}"
            process_glycan_dir = f"{self.process_dir}/{glycan_name}"
            
            if not self.storage.exists(process_glycan_dir):
                logger.error(f"Process directory not found: {process_glycan_dir}")
                return False
            
            # Verify that the output subdirectory exists in process_dir
            process_output_dir = f"{process_glycan_dir}/output"
            if not self.storage.exists(process_output_dir):
                logger.error(f"Output subdirectory not found in process directory: {process_output_dir}")
                return False
            
            # Call glystatic process_glycan function with the process directory as input
            # This tells glystatic to read the intermediate outputs from process_dir/glycan_name/output/
            glystatic_process_glycan(
                folder_path=str(process_glycan_dir),  # Pass process_dir/glycan_name/ (contains output/)
                glycam_name=glycan_name,
                output_static_dir=str(self.output_dir)  # Final database output location
            )

            # In Oracle mode, immediately upload/update the just-written GS* folder to the bucket
            # before validating (validation reads via remote in Oracle mode).
            if config.use_oracle_storage:
                try:
                    # Find the local GS folder for this glycan by matching archetype.name
                    local_base = Path(str(self.output_dir))
                    if local_base.exists():
                        gs_local_dir: Optional[Path] = None
                        for data_json in local_base.glob("GS*/data.json"):
                            try:
                                with open(data_json, 'r', encoding='utf-8') as f:
                                    d = json.load(f)
                                arche = d.get('archetype', {}) if isinstance(d, dict) else {}
                                name_field = arche.get('name') or arche.get('glycam')
                                if name_field and glycan_name.lower() in str(name_field).lower():
                                    gs_local_dir = data_json.parent
                                    break
                            except Exception:
                                continue
                        if gs_local_dir is not None:
                            # Always upload under 'static/<GSID>' in Oracle
                            remote_prefix = f"{self._remote_output_base()}/{gs_local_dir.name}"
                            # Overwrite remote to ensure updates land
                            self.storage.upload_dir(gs_local_dir, remote_prefix, skip_existing=False)
                            logger.info(f"Uploaded/updated static outputs to Oracle: {remote_prefix}")
                        else:
                            logger.warning(f"Could not locate local GS folder for {glycan_name} to upload")
                    else:
                        logger.warning(f"Local output base not found for upload: {local_base}")
                except Exception as e:
                    logger.warning(f"Failed to upload static outputs for {glycan_name}: {e}")
            
            # Validate the output. The glystatic processor writes final output
            # under an ID directory (for example: output_dir/GS00449). Use the
            # existing `check_glystatic_completion` which searches subfolders
            # for a matching archetype entry. Fall back to the older
            # `validate_output_structure` if needed.
            if self.check_glystatic_completion(glycan_name):
                logger.info(f"Successfully completed glystatic for {glycan_name}")
                return True
            else:
                logger.error(f"Glystatic output validation failed for {glycan_name}")
                # Fallback: check same-named output folder (legacy behavior)
                if self.validate_output_structure(glycan_name):
                    logger.info(f"Found direct output directory for {glycan_name}; marking as success")
                    return True
                return False
                        
        except Exception as e:
            logger.error(f"Error running glystatic for {glycan_name}: {str(e)}")
            return False
    
    def run_glymeta(self, glycan_name: str) -> bool:
        """Run metadata processing for a single glycan.
        
        Args:
            glycan_name: Name of the glycan directory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Running glymeta for {glycan_name}")
            
            # Locate the glystatic output directory containing data.json for this glycan
            candidate_dir = None
            try:
                # In Oracle, list under 'static'; locally, under output_dir
                base = self._remote_output_base()
                candidates = self.storage.list_files(base, "GS*")
            except Exception:
                candidates = []
            for cand in candidates:
                key_raw = cand.as_posix() if hasattr(cand, 'as_posix') else str(cand)
                # Normalize returned items which may be file keys
                key = str(Path(key_raw).parent) if key_raw.endswith('/data.json') or key_raw.endswith('data.json') else key_raw
                data_json = f"{key}/data.json"
                if not self.storage.exists(data_json):
                    continue
                try:
                    with self.storage.open(data_json, 'r') as f:
                        data = json.load(f)
                except Exception:
                    continue
                arche = data.get('archetype', {}) if isinstance(data, dict) else {}
                name_field = arche.get('name') or arche.get('glycam')
                if name_field and glycan_name.lower() in str(name_field).lower():
                    candidate_dir = key
                    break
            if candidate_dir is None:
                logger.error(f"data.json not found for {glycan_name}")
                return False
            json_path = f"{candidate_dir}/data.json"
                
            # Update metadata for this specific glycan
            success = self.metadata_processor.update_glycan_metadata(json_path)
            
            if success:
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
        
        logger.info(f"Starting pipeline for glycan: {glycan_name}")
        
        # Step 1: v2 analysis
        # Check if already completed (unless force update)
        if not self.force_update and self.check_v2_completion(glycan_name):
            logger.info(f"v2 analysis already complete for {glycan_name}, skipping")
            results["v2"] = True
        else:
            # Run v2 and upload process dir to Oracle if enabled
            v2_ok = self.run_v2_analysis(glycan_name)
            results["v2"] = v2_ok
            if v2_ok and config.use_oracle_storage:
                try:
                    local_process_dir = os.path.join(str(self.process_dir), glycan_name)
                    remote_prefix = f"{self.process_dir}/{glycan_name}"
                    # Upload only new/modified files, skip existing
                    self.storage.upload_dir(local_process_dir, remote_prefix, skip_existing=True)
                    logger.info(f"Uploaded process outputs to Oracle: {remote_prefix}")
                    
                    # Clean up local folder after successful upload
                    if config.should_cleanup_process_files():
                        if self.storage.delete_local_directory(local_process_dir):
                            logger.info(f"Cleaned up local process directory: {local_process_dir}")
                        else:
                            logger.warning(f"Failed to clean up local process directory: {local_process_dir}")
                except Exception as e:
                    logger.warning(f"Failed to upload/cleanup process outputs for {glycan_name}: {e}")
            
        if not results["v2"]:
            logger.error(f"Pipeline failed at v2 step for {glycan_name}")
            return results
            
        # Step 2: glystatic
        did_run_glystatic = False
        # If outputs already exist and are valid (locally or in Oracle), skip recomputation and re-upload
        if not self.force_update and self.check_glystatic_completion(glycan_name):
            logger.info(f"glystatic already complete for {glycan_name}, skipping")
            results["glystatic"] = True
        else:
            results["glystatic"] = self.run_glystatic(glycan_name)
            did_run_glystatic = results["glystatic"]

        # Post-glystatic upload is handled inside run_glystatic for Oracle mode
        if not results["glystatic"]:
            logger.error(f"Pipeline failed at glystatic step for {glycan_name}")
            return results
            
        # Step 3: glymeta (best-effort, do not fail the pipeline if it fails)
        try:
            self.run_glymeta(glycan_name)
        except Exception:
            # run_glymeta logs its own errors; continue regardless
            pass
            
        logger.info(f"Successfully completed pipeline for glycan: {glycan_name}")
        return results
    
    def run_database_finalization(self) -> bool:
        """Run final database operations.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting database finalization")
            
            # Validate that we have processed glycans
            processed_glycans = []
            try:
                base = self._remote_output_base()
                candidates = self.storage.list_files(base, "GS*")
            except Exception:
                candidates = []
            for cand in candidates:
                key_raw = cand.as_posix() if hasattr(cand, 'as_posix') else str(cand)
                key = str(Path(key_raw).parent) if key_raw.endswith('/data.json') or key_raw.endswith('data.json') else key_raw
                if self.storage.exists(f"{key}/data.json"):
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
        help='Force recomputation of existing results'
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
    
    # Initialize pipeline runner
    runner = GlycanPipelineRunner(data_dir, process_dir, output_dir, args.update)
    
    try:
        if args.status:
            # Check status instead of running
            if args.glycan:
                # Status for single glycan
                logger.info(f"Checking status for glycan: {args.glycan}")
                
                # Check v2 status
                v2_status = get_glycan_status(args.glycan, str(data_dir), str(process_dir))
                glystatic_status = runner.check_glystatic_completion(args.glycan)
                
                print(f"\nStatus for {args.glycan}:")
                print(f"Input files exist: {v2_status['input_files_exist']}")
                print("\nv2 Analysis Steps:")
                for step, completed in v2_status['steps_status'].items():
                    symbol = "✓" if completed else "✗"
                    print(f"  {step}: {symbol}")
                
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