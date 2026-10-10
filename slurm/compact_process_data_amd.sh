#!/bin/bash
#SBATCH --job-name=gap_compact_data
#SBATCH --partition=amd
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(pwd)"

DATA_DIR="${REPO_ROOT}/data"
PROCESS_DIR="${REPO_ROOT}/process"
LOG_DIR="${REPO_ROOT}/logs/compact_process_data"
VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"

mkdir -p "${LOG_DIR}" "${REPO_ROOT}/logs/slurm"

FRAME_CHUNK="${FRAME_CHUNK:-4096}"
FRAMES_BUFFER="${FRAMES_BUFFER:-4096}"
JOBS_PER_NODE="${JOBS_PER_NODE:-16}"
SHARD_COUNT="${SLURM_NNODES:-2}"

echo "----------------------------------------------------------------"
echo "GlycanAnalysisPipeline -- compact process/<glycan>/data"
echo "Nodes: ${SHARD_COUNT}"
echo "Cores per node: ${SLURM_CPUS_PER_TASK:-64}"
echo "Jobs per node: ${JOBS_PER_NODE}"
echo "Frame chunk: ${FRAME_CHUNK}"
echo "Frames buffer: ${FRAMES_BUFFER}"
echo "----------------------------------------------------------------"

export REPO_ROOT DATA_DIR PROCESS_DIR LOG_DIR VENV_PYTHON
export FRAME_CHUNK FRAMES_BUFFER JOBS_PER_NODE SHARD_COUNT
export GLYCOSHAPE_ORACLE_PAR_URL=""
export GLYCOSHAPE_STORAGE=local
export GLYCOSHAPE_MAX_WORKERS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

srun --nodes="${SHARD_COUNT}" --ntasks="${SHARD_COUNT}" --ntasks-per-node=1 bash -c '
    set -euo pipefail
    SHARD_INDEX="${SLURM_PROCID}"
    cd "${REPO_ROOT}"
    echo "Node ${SLURMD_NODENAME:-unknown}: shard ${SHARD_INDEX}/${SHARD_COUNT}"
    "${VENV_PYTHON}" scripts/rewrite_process_data_compact.py \
        --data-dir "${DATA_DIR}" \
        --process-dir "${PROCESS_DIR}" \
        --frame-chunk "${FRAME_CHUNK}" \
        --frames-buffer "${FRAMES_BUFFER}" \
        --jobs "${JOBS_PER_NODE}" \
        --shard-index "${SHARD_INDEX}" \
        --shard-count "${SHARD_COUNT}" \
        --log-dir "${LOG_DIR}"
'

echo "Compact rewrite completed."
