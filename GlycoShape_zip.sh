#!/bin/bash

# Directory paths
DIR1="/mnt/database/DB_beta"
DIR2="/mnt/database/DB_temp"
FINAL_ZIP="GlycoShape.zip"

# Error handling
set -e

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Remove existing zip if present
if [ -f "$DIR2/$FINAL_ZIP" ]; then
    rm "$DIR2/$FINAL_ZIP"
    log "Removed existing $FINAL_ZIP"
fi

# Process each glycan directory
for directory in "$DIR2"/*/ "$DIR1"/*/ ; do
    if [ -d "$directory" ]; then
        cd "$directory"
        glycan=$(basename "$directory")
        log "Processing $glycan"

        # Remove existing zip if present
        [ -f "$glycan.zip" ] && rm "$glycan.zip"

        # Create zip with all required files
        zip -r "$directory/$glycan.zip" \
            "$glycan.json" \
            "$glycan.svg" \
            PDB_format_HETATM \
            CHARMM_format_HETATM \
            GLYCAM_format_HETATM \
            PDB_format_ATOM \
            CHARMM_format_ATOM \
            GLYCAM_format_ATOM
    fi
done

# Create final zip file
cd "$DIR2"
log "Creating final zip file"

# Copy all zip files to temp location
cp "$DIR1"/*/*.zip "$DIR2"
cp "$DIR2"/*/*.zip "$DIR2"

# Create final archive
zip "$FINAL_ZIP" *.zip

# Cleanup
log "Cleaning up temporary files"
find . -name "*.zip" ! -name "$FINAL_ZIP" -delete

log "Successfully created $FINAL_ZIP"