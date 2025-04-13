import os
import sys
import importlib
import logging

#!/usr/bin/env python3
"""
Main entry point for the Glycan Analysis Pipeline.
This script sequentially runs the following modules:
1. cluster_first
2. recluster
3. plot_dist
4. GlycoShaep_DB_static
5. GlycoShape_DB_bake
6. json2rdf
"""


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_module(module_name):
    """Run a module by importing and executing its main function."""
    try:
        logger.info(f"Running {module_name}...")
        module = importlib.import_module(f"lib.{module_name}")
        
        # Check if the module has a main function, otherwise run the module directly
        if hasattr(module, "main"):
            module.main()
        else:
            logger.warning(f"No main function found in {module_name}, attempting to run module directly")
            # This assumes the module executes its functionality when imported
        
        logger.info(f"Completed {module_name}")
        return True
    except Exception as e:
        logger.error(f"Error running {module_name}: {str(e)}")
        return False

def main():
    """Main function to run the entire pipeline."""
    # List of modules to run in order
    modules = [
        "cluster_first",
        "recluster",
        "plot_dist",
        "GlycoShape_DB_static",
        "GlycoShape_DB_bake",
        "json2rdf",
    ]
    
    # Run each module in sequence
    for module in modules:
        success = run_module(module)
        if not success:
            logger.error(f"Pipeline failed at module: {module}")
            sys.exit(1)
    
    logger.info("Glycan Analysis Pipeline completed successfully")

if __name__ == "__main__":
    main()