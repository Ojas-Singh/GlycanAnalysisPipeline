#!/bin/bash
#SBATCH --job-name=gap_pbc_check
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# Scan processed coordinates for likely periodic-boundary artifacts.
#
# Submit from the repository root:
#   sbatch slurm/check_pbc_jumps_amd.sh
#
# Optional threshold/parallelism overrides:
#   WORKERS=16 JUMP_DISTANCE=10 MAX_BOND_DISTANCE=3 \
#     sbatch --export=ALL slurm/check_pbc_jumps_amd.sh

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(pwd)"
CPUS="${SLURM_CPUS_PER_TASK:-64}"
WORKERS="${WORKERS:-16}"
BLOCK_FRAMES="${BLOCK_FRAMES:-2048}"
MAX_BOND_DISTANCE="${MAX_BOND_DISTANCE:-3.0}"
JUMP_DISTANCE="${JUMP_DISTANCE:-10.0}"
TRANSLATION_DISTANCE="${TRANSLATION_DISTANCE:-10.0}"
MIN_JUMPING_ATOMS="${MIN_JUMPING_ATOMS:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/pbc_jump_report}"

if ! [[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]] || (( WORKERS > CPUS )); then
    echo "WORKERS must be a positive integer no larger than allocated CPUs (${CPUS})." >&2
    exit 2
fi

mkdir -p "${REPO_ROOT}/logs/slurm" "${OUTPUT_DIR}"

echo "PBC jump scan on ${SLURMD_NODENAME:-local}"
echo "Process directory: ${REPO_ROOT}/process"
echo "Output directory: ${OUTPUT_DIR}"
echo "Workers: ${WORKERS}/${CPUS}"
echo "Thresholds (A): bond=${MAX_BOND_DISTANCE}, internal=${JUMP_DISTANCE}, translation=${TRANSLATION_DISTANCE}"
echo "Minimum coordinated jumping heavy atoms: ${MIN_JUMPING_ATOMS}"

srun --nodes=1 --ntasks=1 --cpus-per-task="${CPUS}" --cpu-bind=cores \
    "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/check_pbc_jumps.py" \
    --process-dir "${REPO_ROOT}/process" \
    --output-dir "${OUTPUT_DIR}" \
    --workers "${WORKERS}" \
    --block-frames "${BLOCK_FRAMES}" \
    --max-bond-distance "${MAX_BOND_DISTANCE}" \
    --jump-distance "${JUMP_DISTANCE}" \
    --min-jumping-atoms "${MIN_JUMPING_ATOMS}" \
    --translation-distance "${TRANSLATION_DISTANCE}"
