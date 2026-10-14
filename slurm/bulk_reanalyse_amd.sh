#!/bin/bash
#SBATCH --job-name=gap_bulk_reanalyse
#SBATCH --partition=amd
#SBATCH --array=1-4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

# The analysis stage of the bulk re-analysis (docs/bulk-reanalysis.md): one array task per shard of
# the plan, one 64-core AMD node each, JOBS entries at a time with WORKERS cores each (4 x 16 = 64).
# No network is used and nothing outside RUN_DIR is written; the raw archive is only read.
#
#   RUN_DIR=/scratch/os1e24/gap-bulk/run sbatch --array=1-4 slurm/bulk_reanalyse_amd.sh
#
# The array must have as many tasks as the plan has shards (bulk_reanalyse.py plan --shards N).  Run the
# same command again after a failure, a timeout or a pre-emption: entries that are done are skipped.
# ENTRY="GS00276 GS00741" analyses only those entries (one task, --array=1).  The per-entry logs are
# RUN_DIR/entries/<folder>/analysis.log and status.json; the summary of the task is
# RUN_DIR/logs/analyse-<task>.json.  Exit status 1 when an entry failed (the others are done).

set -euo pipefail

: "${RUN_DIR:?Set RUN_DIR to the run folder made by bulk_reanalyse.py plan.}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
JOBS="${JOBS:-4}"
WORKERS="${WORKERS:-16}"
CPUS="${SLURM_CPUS_PER_TASK:-64}"
TASK="${SLURM_ARRAY_TASK_ID:-1}"
TASKS="${SLURM_ARRAY_TASK_COUNT:-1}"

if ! [[ "${JOBS}" =~ ^[1-9][0-9]*$ && "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "JOBS and WORKERS must be positive integers." >&2
    exit 2
fi
if [[ $((JOBS * WORKERS)) -gt "${CPUS}" ]]; then
    echo "JOBS*WORKERS ($((JOBS * WORKERS))) exceeds the ${CPUS} allocated cores." >&2
    exit 2
fi
# Memory for the analyses: all of the node (--mem=0) but 16 GB; MEMORY_GB overrides.
if [[ -z "${MEMORY_GB:-}" ]]; then
    total_kb=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
    MEMORY_GB=$(( total_kb / 1048576 - 16 ))
fi

export GLYCOSHAPE_STORAGE=local
export GLYCOSHAPE_ORACLE_PAR_URL=""
export GLYCOSHAPE_OFFLINE=1
export POCKETBASE_URL=""
export MPLBACKEND=Agg

arguments=(analyse --run-dir "${RUN_DIR}" --jobs "${JOBS}" --workers "${WORKERS}" --memory-gb "${MEMORY_GB}")
if [[ -n "${ARCHIVE_DIR:-}" ]]; then
    arguments+=(--archive-dir "${ARCHIVE_DIR}")      # the archive of the plan, where it is on this cluster
fi
if [[ -n "${ENTRY:-}" ]]; then
    for entry in ${ENTRY}; do
        arguments+=(--entry "${entry}")
    done
else
    arguments+=(--shard "${TASK}" --shards "${TASKS}")
fi

echo "----------------------------------------------------------------"
echo "GAP bulk re-analysis: task ${TASK}/${TASKS} on ${SLURMD_NODENAME:-$(hostname)}"
echo "run folder ${RUN_DIR}; ${JOBS} entries at a time x ${WORKERS} workers; ${MEMORY_GB} GB for the analyses"
echo "code $(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "----------------------------------------------------------------"
"${PYTHON}" "${REPO_ROOT}/scripts/bulk_reanalyse.py" "${arguments[@]}"
