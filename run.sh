#!/bin/bash

# User configurable paths
BASE_DIR="/mnt/database"                                 # Base directory
GLYCAN_DATA_DIR="${BASE_DIR}/glycoshape_data/"           # Glycan data directory
DATABASE_PATH="${BASE_DIR}/DB_static/"                   # Database path
UPLOAD_DATA_DIR="${BASE_DIR}/test_data"                    # Upload data directory
GLYCAN_INVENTORY_PATH="${UPLOAD_DATA_DIR}/GlycoShape_Inventory.csv"  # Inventory file path
PIPELINE_DIR="${BASE_DIR}/GlycanAnalysisPipeline/"       # Pipeline directory
CONDA_PATH="/home/ubuntu/miniconda3"                     # Conda installation path

# Database update flag
export GLYCAN_DB_UPDATE=False

# Export environment variables
export GLYCAN_DATA_DIR
export DATABASE_PATH 
export GLYCAN_INVENTORY_PATH

# Create timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="${UPLOAD_DATA_DIR}/pipeline_output_${TIMESTAMP}.log"

# Log header
echo "=== Pipeline execution started at $(date) ===" > "$LOGFILE"


# Step 1: Run the move script and log output
echo "Moving necessary directories to target location..." >> "$LOGFILE"
# Directory movement logic integrated directly
(
    cd "$UPLOAD_DATA_DIR" || { echo "Failed to navigate to test data directory" >> "$LOGFILE"; exit 1; }
    
    # Iterate over all directories in the test data directory
    for dir in */; do
        # Get the name of the directory (remove trailing slash)
        dir_name="${dir%/}"

        # Check if the directory already exists in the target location
        if [ -d "$GLYCAN_DATA_DIR$dir_name" ]; then
            echo "Directory '$dir_name' already exists in the target location. Skipping." >> "$LOGFILE"
        else
            # Move the directory to the target location
            mv "$dir" "$GLYCAN_DATA_DIR"
            echo "Moved directory '$dir_name' to '$GLYCAN_DATA_DIR'." >> "$LOGFILE"
        fi
    done
) >> "$LOGFILE" 2>&1

# Setup conda environment
echo "Setting up conda environment..." >> "$LOGFILE"
export PATH="${CONDA_PATH}/bin:$PATH"
source ${CONDA_PATH}/etc/profile.d/conda.sh >> "$LOGFILE" 2>&1

# Step 2: Navigate to the GlycanAnalysisPipeline directory
echo "Navigating to pipeline directory..." >> "$LOGFILE"
cd "$PIPELINE_DIR" || { echo "Failed to navigate to GlycanAnalysisPipeline directory" >> "$LOGFILE"; exit 1; }

# Activate the GAP conda environment and log output
echo "Activating conda environment..." >> "$LOGFILE"
conda activate GAP >> "$LOGFILE" 2>&1

# Run the Python scripts in sequence and log output
echo "Running main.py..." >> "$LOGFILE"
python main.py >> "$LOGFILE" 2>&1

echo "Oxigraph on GlycoShape..." >> "$LOGFILE"
oxigraph load --location "${DATABASE_PATH}/GLYCOSHAPE_RDF" --file "${DATABASE_PATH}/GLYCOSHAPE_RDF.ttl"

echo "=== Pipeline execution completed at $(date) ===" >> "$LOGFILE"
