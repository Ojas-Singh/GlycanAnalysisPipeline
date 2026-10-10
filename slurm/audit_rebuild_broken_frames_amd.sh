#!/bin/bash
#SBATCH --job-name=gap-frame-quality
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/slurm

DATA_DIR="${DATA_DIR:-data}"
PROCESS_DIR="${PROCESS_DIR:-process}"
REPORT="${REPORT:-frame_quality_audit.json}"
REBUILD="${REBUILD:-0}"
WORKERS="${WORKERS:-8}"

if ! [[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]] || (( WORKERS > SLURM_CPUS_PER_TASK )); then
    echo "WORKERS must be between 1 and ${SLURM_CPUS_PER_TASK}." >&2
    exit 2
fi

args=(
    --data-dir "${DATA_DIR}"
    --process-dir "${PROCESS_DIR}"
    --report "${REPORT}"
    --workers "${WORKERS}"
)
if [[ "${REBUILD}" == "1" ]]; then
    args+=(--rebuild)
fi
if [[ -n "${GLYCAN_FILE:-}" ]]; then
    args+=(--glycan-file "${GLYCAN_FILE}")
fi
if [[ -n "${BOND_REPORT:-}" ]]; then
    args+=(--bond-report "${BOND_REPORT}")
fi
if [[ -n "${RETRY_ERRORS:-}" ]]; then
    args+=(--retry-errors "${RETRY_ERRORS}")
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    --cpu-bind=cores \
    .venv/bin/python scripts/audit_and_rebuild_broken_frames.py "${args[@]}"
