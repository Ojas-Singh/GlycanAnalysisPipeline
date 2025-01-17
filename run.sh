#!/bin/bash

# Script configuration
PIPELINE_DIR="/mnt/database/GlycanAnalysisPipeline"
CONDA_PATH="/home/ubuntu/miniconda3"
ENV_NAME="GAP"
TARGET_DIR="/mnt/database/glycoshape_data/"

# Function to log messages with timestamps
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOGFILE"
}

# Function to check if directory exists
check_dir() {
    if [ ! -d "$1" ]; then
        echo "Error: Directory $1 does not exist"
        exit 1
    fi
}

# Function to move processed directories
move_directories() {
    local input_dir="$1"
    log "Moving processed directories to $TARGET_DIR"
    
    cd "$input_dir" || {
        log "Failed to change to input directory"
        return 1
    }
    
    for dir in */; do
        dir_name="${dir%/}"
        if [ -d "$TARGET_DIR$dir_name" ]; then
            log "Directory '$dir_name' already exists in target location. Skipping."
        else
            mv "$dir" "$TARGET_DIR"
            log "Moved directory '$dir_name' to '$TARGET_DIR'"
        fi
    done
}

# Show usage if no arguments provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_directory>"
    echo "Example: $0 /path/to/input/data"
    exit 1
fi

# Setup variables
INPUT_DIR="$1"
LOGFILE="${INPUT_DIR}/pipeline_output.log"

# Validate directories
check_dir "$INPUT_DIR"
check_dir "$PIPELINE_DIR"
check_dir "$TARGET_DIR"

# Initialize log file
> "$LOGFILE"
log "Starting GlycanAnalysis Pipeline"
log "Input directory: $INPUT_DIR"

# Setup conda environment
log "Setting up conda environment"
export PATH="${CONDA_PATH}/bin:$PATH"
source "${CONDA_PATH}/etc/profile.d/conda.sh" >> "$LOGFILE" 2>&1
conda activate $ENV_NAME >> "$LOGFILE" 2>&1

# Navigate to pipeline directory
log "Changing to pipeline directory"
cd "$PIPELINE_DIR" || {
    log "Failed to navigate to pipeline directory"
    exit 1
}

# Run pipeline scripts
log "Running analysis pipeline"

# Analysis phase
log "Phase 1: Initial Analysis"
python main.py "$INPUT_DIR" >> "$LOGFILE" 2>&1
python recluster.py "$INPUT_DIR" >> "$LOGFILE" 2>&1
python plot_dist.py "$INPUT_DIR" >> "$LOGFILE" 2>&1
python save_frames.py "$INPUT_DIR" >> "$LOGFILE" 2>&1

# Database phase
log "Phase 2: Database Processing"
python GlycoShape_DB_static.py "$INPUT_DIR" >> "$LOGFILE" 2>&1
python GlycoShape_DB_bake.py "$INPUT_DIR" >> "$LOGFILE" 2>&1

# Move processed directories
log "Phase 3: Moving Results"
move_directories "$INPUT_DIR"

# Check for any errors in log
if grep -q "Error\|Exception\|failed" "$LOGFILE"; then
    log "WARNING: Errors detected in pipeline execution"
    exit 1
else
    log "Pipeline executed successfully"
fi