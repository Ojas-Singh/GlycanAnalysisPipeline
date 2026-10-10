#!/bin/bash
#SBATCH --job-name=gap_process_only
#SBATCH --partition=amd
#SBATCH --array=1-4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

# Process glycans under data/ incrementally in process-only mode. The
# default array has four tasks; each task requests one 64-CPU AMD node, runs
# 16 glycans concurrently, and gives each glycan four CPUs (16 * 4 == 64).
#
# The #SBATCH --array=1-4 directive makes this a four-node submission:
#   sbatch slurm/process_only_amd.sh
# An explicit --array command-line option can override that default.
#
# To intentionally recompute every assigned glycan:
#   FORCE_UPDATE=1 sbatch slurm/process_only_amd.sh
#
# Array task IDs are shard numbers, not glycan numbers. Every task reads the
# same sorted glycan list and receives every fourth entry based on its ID.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(pwd)"

DATA_DIR="${REPO_ROOT}/data"
PROCESS_DIR="${REPO_ROOT}/process"
LOG_DIR="${REPO_ROOT}/logs/slurm/glycans"
VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"
FORCE_UPDATE="${FORCE_UPDATE:-0}"
NEW_ONLY="${NEW_ONLY:-0}"

# Array task indexing: default to a single task if run outside an array. Keep
# each task's list files job-specific so concurrent submissions cannot race.
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
N_TASKS=${SLURM_ARRAY_TASK_COUNT:-1}
RUN_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}
GLYCAN_LIST="${REPO_ROOT}/logs/slurm/all_glycans_${RUN_ID}_${TASK_ID}.txt"
CHUNK_FILE="${REPO_ROOT}/logs/slurm/chunk_${RUN_ID}_${TASK_ID}.txt"

if [[ "${FORCE_UPDATE}" != "0" && "${FORCE_UPDATE}" != "1" ]]; then
    echo "FORCE_UPDATE must be 0 (incremental) or 1 (recompute)." >&2
    exit 2
fi
if [[ "${NEW_ONLY}" != "0" && "${NEW_ONLY}" != "1" ]]; then
    echo "NEW_ONLY must be 0 (all incremental candidates) or 1 (data dirs without process dirs)." >&2
    exit 2
fi

mkdir -p "${LOG_DIR}"

# Rebuild the list on every task: the login-node download stage may have added
# or removed source glycans since a previous array job.  NEW_ONLY is used for
# a strictly new-data run; it deliberately excludes legacy folders so their
# old process outputs are not implicitly sent through frame-quality repair.
if [[ "${NEW_ONLY}" == "1" ]]; then
    find "${DATA_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | \
        while IFS= read -r glycan; do
            if [[ ! -d "${PROCESS_DIR}/${glycan}" ]]; then
                printf '%s\n' "${glycan}"
            fi
        done > "${GLYCAN_LIST}"
else
    find "${DATA_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort > "${GLYCAN_LIST}"
fi
TOTAL_GLYCANS=$(wc -l < "${GLYCAN_LIST}")

awk -v n="${N_TASKS}" -v i="${TASK_ID}" '(NR - 1) % n == i - 1 {print}' "${GLYCAN_LIST}" > "${CHUNK_FILE}"
CHUNK_SIZE=$(wc -l < "${CHUNK_FILE}")

echo "----------------------------------------------------------------"
echo "GlycanAnalysisPipeline -- process-only (FORCE_UPDATE=${FORCE_UPDATE}, NEW_ONLY=${NEW_ONLY})"
echo "Task ${TASK_ID}/${N_TASKS} | Glycans in chunk: ${CHUNK_SIZE}/${TOTAL_GLYCANS}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
CPUS_PER_NODE="${SLURM_CPUS_PER_TASK:-64}"
WORKERS="${WORKERS:-4}"

if ! [[ "${CPUS_PER_NODE}" =~ ^[1-9][0-9]*$ && "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "SLURM_CPUS_PER_TASK and WORKERS must be positive integers." >&2
    exit 2
fi

# Default to one concurrent glycan per WORKERS-sized CPU group. With the Slurm
# defaults this is 64 / 4 = 16 simultaneous glycans on each node.
NJOBS="${NJOBS:-$((CPUS_PER_NODE / WORKERS))}"
if ! [[ "${NJOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NJOBS must be a positive integer." >&2
    exit 2
fi

echo "Cores allocated: ${CPUS_PER_NODE}"
echo "Concurrent glycans on this node: ${NJOBS}"
echo "CPUs per glycan: ${WORKERS}"
echo "----------------------------------------------------------------"

if [[ "${CHUNK_SIZE}" -eq 0 ]]; then
    echo "No glycans assigned to this task; exiting."
    exit 0
fi

if [[ $((NJOBS * WORKERS)) -gt "${CPUS_PER_NODE}" ]]; then
    echo "ERROR: NJOBS*WORKERS ($((NJOBS * WORKERS))) exceeds allocated cores (${CPUS_PER_NODE})." >&2
    exit 2
fi

export REPO_ROOT DATA_DIR PROCESS_DIR LOG_DIR VENV_PYTHON WORKERS FORCE_UPDATE
export GLYCOSHAPE_ORACLE_PAR_URL=""
export GLYCOSHAPE_STORAGE=local

# Run the chunk in parallel. Each line of CHUNK_FILE is one glycan name.
xargs -a "${CHUNK_FILE}" -P "${NJOBS}" -I{} \
    bash -c '
        set -euo pipefail
        GLYCAN="$1"
        cd "${REPO_ROOT}"
        update_args=()
        if [[ "${FORCE_UPDATE}" == "1" ]]; then
            update_args+=(--update)
        fi
        GLYCOSHAPE_MAX_WORKERS="${WORKERS}" \
        GLYCOSHAPE_UPLOAD_WORKERS=1 \
        GLYCOSHAPE_STORE_FRAMES_BUFFER=128 \
        GLYCOSHAPE_STORE_FRAME_CHUNK=0 \
        GLYCOSHAPE_GAP_LOG="${LOG_DIR}/${GLYCAN}.log" \
            srun \
                --exclusive \
                --exact \
                --nodes=1 \
                --ntasks=1 \
                --cpus-per-task="${WORKERS}" \
                --cpu-bind=cores \
                "${VENV_PYTHON}" main.py \
                --mode process-only \
                "${update_args[@]}" \
                --glycan "${GLYCAN}" \
                --data-dir "${DATA_DIR}" \
                --process-dir "${PROCESS_DIR}"
    ' _ {}

echo "Task ${TASK_ID} completed."
