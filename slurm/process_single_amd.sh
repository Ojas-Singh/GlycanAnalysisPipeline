#!/bin/bash
#SBATCH --job-name=gap_process_single
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# Run one named local data/ glycan on an AMD node. Submit with:
#   sbatch --export=ALL,GLYCAN='<data-directory-name>' slurm/process_single_amd.sh
# Or submit a file-backed array (one directory name per line):
#   sbatch --array=1-N --export=ALL,GLYCAN_FILE=/path/to/names.txt slurm/process_single_amd.sh

set -euo pipefail

if [[ -z "${GLYCAN:-}" && -n "${GLYCAN_FILE:-}" ]]; then
    : "${SLURM_ARRAY_TASK_ID:?GLYCAN_FILE requires a Slurm array task ID.}"
    GLYCAN="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${GLYCAN_FILE}")"
fi
: "${GLYCAN:?Set GLYCAN, or GLYCAN_FILE with a Slurm array task ID.}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(pwd)"

export GLYCOSHAPE_ORACLE_PAR_URL=""
export GLYCOSHAPE_MAX_WORKERS="${GLYCOSHAPE_MAX_WORKERS:-4}"
export GLYCOSHAPE_UPLOAD_WORKERS=1
export GLYCOSHAPE_STORE_FRAMES_BUFFER=128
export GLYCOSHAPE_STORE_FRAME_CHUNK=0
export GLYCOSHAPE_GAP_LOG="${REPO_ROOT}/logs/slurm/glycans/${GLYCAN}.log"

"${REPO_ROOT}/.venv/bin/python" main.py \
    --mode process-only \
    --glycan "${GLYCAN}" \
    --data-dir "${REPO_ROOT}/data" \
    --process-dir "${REPO_ROOT}/process"
