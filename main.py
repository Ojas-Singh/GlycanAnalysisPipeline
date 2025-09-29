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
    
    def __init__(self, data_dir: Path, process_dir: Path, output_dir: Path, force_update: bool = False):
        """Initialize the pipeline runner.
        
        Args:
            data_dir: Directory containing glycan input data (read-only)
            process_dir: Directory for storing processing outputs (data, embedding, output)
            output_dir: Directory for final database output
            force_update: Whether to force recomputation of existing results
        """
        self.data_dir = Path(data_dir)
        self.process_dir = Path(process_dir) 
        self.output_dir = Path(output_dir)
        self.force_update = force_update
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.process_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize processors
        self.metadata_processor = GlycanMetadataProcessor(self.output_dir)
        self.baker = GlycoShapeBaker(self.output_dir)
        
    def discover_glycans(self) -> List[str]:
        """Discover glycan directories in data_dir.
        
        Returns:
            List of glycan directory names
        """
        glycan_dirs = []
        
        if not self.data_dir.exists():
            logger.error(f"Data directory does not exist: {self.data_dir}")
            return glycan_dirs
            
        for item in self.data_dir.iterdir():
            if item.is_dir():
                # Check for required files
                pdb_file = item / f"{item.name}.pdb"
                mol2_file = item / f"{item.name}.mol2"
                
                if pdb_file.exists() and mol2_file.exists():
                    glycan_dirs.append(item.name)
                    logger.info(f"Found glycan: {item.name}")
                else:
                    logger.warning(f"Skipping {item.name}: missing required files (.pdb and .mol2)")
                    
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
        output_glycan_dir = self.output_dir / glycan_name
        data_json = output_glycan_dir / "data.json"
        
        if not output_glycan_dir.exists():
            logger.debug(f"Output directory missing for {glycan_name}")
            return False
            
        if not data_json.exists():
            logger.debug(f"data.json missing for {glycan_name}")
            return False
            
        try:
            # Validate JSON structure
            with open(data_json, 'r', encoding='utf-8') as f:
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
            # First, find the correct GS folder by matching the archetype name exactly
            matching_candidate = None
            
            # Iterate over possible processed ID directories to find the matching one
            for candidate in self.output_dir.iterdir():
                if not candidate.is_dir():
                    continue

                data_json = candidate / "data.json"
                if not data_json.exists():
                    continue

                # Try to read the data.json
                try:
                    with open(data_json, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception:
                    continue

                # Check if archetype name matches exactly
                archetype = data.get('archetype', {})
                name_field = archetype.get('name') or archetype.get('glycam', '')
                if name_field == glycan_name:
                    matching_candidate = candidate
                    break
            
            if not matching_candidate:
                logger.debug(f"No GS folder found with exact name match for {glycan_name}")
                return False
            
            # Now validate only the matching candidate
            validation_errors = []
            
            data_json = matching_candidate / "data.json"
            try:
                with open(data_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read data.json for {matching_candidate.name}: {e}")
                return False

            # Validate JSON structure - must be a dict with 'archetype' section
            if not isinstance(data, dict):
                validation_errors.append(f"data.json is not a dictionary")
                
            archetype = data.get('archetype', {})
            if not isinstance(archetype, dict):
                validation_errors.append(f"missing or invalid archetype section")

            # Check for required archetype fields
            required_fields = ["ID", "name", "iupac", "glytoucan"]
            missing_fields = [field for field in required_fields if field not in archetype or not archetype[field]]
            if missing_fields:
                validation_errors.append(f"missing required archetype fields: {missing_fields}")
                
            # Specifically validate glytoucan ID - must not be null, empty, or "null"
            glytoucan_id = archetype.get("glytoucan", "").strip()
            if not glytoucan_id or glytoucan_id.lower() == "null" or glytoucan_id == "":
                validation_errors.append(f"invalid glytoucan ID '{glytoucan_id}'")

            # Validate output structure - must have output/ folder with required files
            out_folder = matching_candidate / 'output'
            if not out_folder.exists():
                validation_errors.append(f"missing output/ folder")

            expected_files = [out_folder / 'pca.csv', out_folder / 'torparts.npz']
            missing_files = [str(f.relative_to(matching_candidate)) for f in expected_files if not f.exists()]
            if missing_files:
                validation_errors.append(f"missing output files: {missing_files}")

            # Ensure at least level_1 exists for representative structures
            if not (out_folder / 'level_1').exists():
                validation_errors.append(f"missing level_1 structures")

            if validation_errors:
                logger.warning(f"Glystatic validation failed for {glycan_name} in {matching_candidate.name}. Issues found:")
                for error in validation_errors:
                    logger.warning(f"  - {error}")
                return False

            # All validation passed!
            logger.debug(f"Glystatic completion fully validated in {matching_candidate} for {glycan_name} (GlyTouCan: {glytoucan_id})")
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
            output_glycan_dir = self.output_dir / glycan_name
            
            # Check if glystatic processing is already complete (unless force update)
            if not self.force_update and self.check_glystatic_completion(glycan_name):
                logger.info(f"Glystatic processing already complete for {glycan_name}, skipping")
                return True
                
            logger.info(f"Running glystatic for {glycan_name}")
            
            # Define paths for glystatic processing
            glycan_input_dir = self.data_dir / glycan_name
            process_glycan_dir = self.process_dir / glycan_name
            
            if not process_glycan_dir.exists():
                logger.error(f"Process directory not found: {process_glycan_dir}")
                return False
            
            # Verify that the output subdirectory exists in process_dir
            process_output_dir = process_glycan_dir / "output"
            if not process_output_dir.exists():
                logger.error(f"Output subdirectory not found in process directory: {process_output_dir}")
                return False
            
            # Call glystatic process_glycan function with the process directory as input
            # This tells glystatic to read the intermediate outputs from process_dir/glycan_name/output/
            glystatic_process_glycan(
                folder_path=str(process_glycan_dir),  # Pass process_dir/glycan_name/ (contains output/)
                glycam_name=glycan_name,
                output_static_dir=str(self.output_dir)  # Final database output location
            )
            
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
            for subdir in self.output_dir.iterdir():
                if not subdir.is_dir():
                    continue
                data_json = subdir / "data.json"
                if not data_json.exists():
                    continue
                # Try to match glycan_name in the archetype section
                try:
                    with open(data_json, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception:
                    continue
                arche = data.get('archetype', {}) if isinstance(data, dict) else {}
                name_field = arche.get('name') or arche.get('glycam')
                if name_field and glycan_name.lower() in str(name_field).lower():
                    candidate_dir = subdir
                    break
            if candidate_dir is None:
                logger.error(f"data.json not found for {glycan_name}")
                return False
            json_path = candidate_dir / "data.json"
                
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
            results["v2"] = self.run_v2_analysis(glycan_name)
            
        if not results["v2"]:
            logger.error(f"Pipeline failed at v2 step for {glycan_name}")
            return results
            
        # Step 2: glystatic
        results["glystatic"] = self.run_glystatic(glycan_name)
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
            for glycan_dir in self.output_dir.glob("GS*"):
                if glycan_dir.is_dir() and (glycan_dir / "data.json").exists():
                    processed_glycans.append(glycan_dir.name)
            
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
            input_path = self.output_dir / "GLYCOSHAPE.json"
            output_path = self.output_dir / "GLYCOSHAPE_RDF.ttl"

            if not input_path.exists():
                logger.warning(f"Consolidated JSON not found at expected location: {input_path}")
                logger.warning("Skipping json2rdf conversion")
            else:
                # Validate JSON is readable and non-empty
                try:
                    with open(input_path, 'r', encoding='utf-8') as jf:
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
    
    # Use config defaults or command line overrides
    data_dir = Path(args.data_dir) if args.data_dir else config.data_dir
    process_dir = Path(args.process_dir) if args.process_dir else config.process_dir
    output_dir = Path(args.output_dir) if args.output_dir else config.output_dir
    
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