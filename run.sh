#!/bin/bash

# Define the logfile
LOGFILE="/mnt/database/test_data/pipeline_output.log"

# Step 1: Run the move script and log output
./move.sh >> "$LOGFILE" 2>&1
export PATH="/home/ubuntu/miniconda3/bin:$PATH"
source ~/miniconda3/etc/profile.d/conda.sh >> "$LOGFILE" 2>&1

# Step 2: Navigate to the GlycanAnalysisPipeline directory
cd ~/GlycanAnalysisPipeline/ || { echo "Failed to navigate to GlycanAnalysisPipeline directory" >> "$LOGFILE"; exit 1; }

# Activate the GAP conda environment and log output
conda activate GAP >> "$LOGFILE" 2>&1

# Run the Python scripts in sequence and log output
python main.py >> "$LOGFILE" 2>&1
python recluster.py >> "$LOGFILE" 2>&1
python plot_dist.py >> "$LOGFILE" 2>&1
python save_frames.py >> "$LOGFILE" 2>&1

# Step 3: Navigate to the DB_scripts directory
cd /mnt/database/DB_scripts/ || { echo "Failed to navigate to DB_scripts directory" >> "$LOGFILE"; exit 1; }

# Deactivate GAP environment
conda deactivate >> "$LOGFILE" 2>&1

# Activate the GS3D conda environment and log output
conda activate GS3D >> "$LOGFILE" 2>&1

# Run the GlycoShape_DB_beta.py script and log output
python GlycoShape_DB_beta.py >> "$LOGFILE" 2>&1
python iupac_glytoucan.py >> "$LOGFILE" 2>&1


