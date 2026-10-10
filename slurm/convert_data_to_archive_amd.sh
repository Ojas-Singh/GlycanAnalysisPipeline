#!/bin/bash
#SBATCH --job-name=gap_archive
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# Convert the raw data directory (one folder per glycan with simulation.pdb)
# into the trajectory archive (trajectory.xtc + topology.pdb + manifest.json
# per glycan, see "Raw trajectory archive" in README.md) on one AMD node.
#
# Submit from the repository root (logs/slurm must exist):
#   mkdir -p logs/slurm && sbatch slurm/convert_data_to_archive_amd.sh
#
# The source directory is only read. Submitting again resumes: entries whose
# manifest verifies are skipped, everything else is converted again.
# The job ends with a non-zero status when an entry failed; the reasons are in
# ${ARCHIVE_DIR}/conversion_report.json under "failed" and in the error log.
#
# Afterwards, to compare every coordinate of every entry with the source text
# a second time (reads the whole source again; writes verification_report.json):
#   VERIFY=1 sbatch --export=ALL slurm/convert_data_to_archive_amd.sh

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(pwd)"

# ---- Paths: edit here, or override with sbatch --export=ALL,DATA_DIR=... ----
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"            # source, one folder per glycan (read only)
ARCHIVE_DIR="${ARCHIVE_DIR:-${REPO_ROOT}/archive}"   # output: entries, index.json, conversion_report.json
VENV_PYTHON="${VENV_PYTHON:-${REPO_ROOT}/.venv/bin/python}"

# ---- Run parameters ----
CPUS="${SLURM_CPUS_PER_TASK:-64}"
WORKERS="${WORKERS:-${CPUS}}"            # glycans converted at the same time, one core each
VERIFY="${VERIFY:-0}"                    # 1: verify the existing archive against the source, convert nothing
if [[ "${VERIFY}" == "1" ]]; then
    CHECK_FRAMES="${CHECK_FRAMES:--1}"   # frames per glycan re-read with float(); -1 is every frame
else
    CHECK_FRAMES="${CHECK_FRAMES:-100}"
fi
FORCE="${FORCE:-0}"                      # 1: convert again even when an entry verifies
GLYCAN_FILE="${GLYCAN_FILE:-}"           # optional file with source folder names, one per line

if ! [[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]] || (( WORKERS > CPUS )); then
    echo "WORKERS must be a positive integer no larger than allocated CPUs (${CPUS})." >&2
    exit 2
fi
if [[ ! -d "${DATA_DIR}" ]]; then
    echo "DATA_DIR does not exist: ${DATA_DIR}" >&2
    exit 2
fi
if ! "${VENV_PYTHON}" -c "import mdtraj" 2>/dev/null; then
    echo "MDTraj is not installed in ${VENV_PYTHON}; run:" >&2
    echo "  uv pip install --python ${VENV_PYTHON} -r ${REPO_ROOT}/requirements.txt" >&2
    exit 2
fi

mkdir -p "${REPO_ROOT}/logs/slurm" "${ARCHIVE_DIR}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

args=(
    --data-dir "${DATA_DIR}"
    --archive-dir "${ARCHIVE_DIR}"
    --workers "${WORKERS}"
    --check-frames "${CHECK_FRAMES}"
)
if [[ "${VERIFY}" == "1" ]]; then
    args+=(--verify-only)
elif [[ "${FORCE}" == "1" ]]; then
    args+=(--force)
fi
if [[ -n "${GLYCAN_FILE}" ]]; then
    args+=(--glycan-file "${GLYCAN_FILE}")
fi

echo "----------------------------------------------------------------"
if [[ "${VERIFY}" == "1" ]]; then
    echo "GlycanAnalysisPipeline -- verify trajectory archive against raw data"
else
    echo "GlycanAnalysisPipeline -- raw data -> trajectory archive"
fi
echo "Node: ${SLURMD_NODENAME:-local}"
echo "Source (read only): ${DATA_DIR}"
echo "Archive: ${ARCHIVE_DIR}"
echo "Workers: ${WORKERS}/${CPUS}"
echo "----------------------------------------------------------------"

srun --nodes=1 --ntasks=1 --cpus-per-task="${CPUS}" --cpu-bind=cores \
    "${VENV_PYTHON}" "${REPO_ROOT}/scripts/convert_data_to_archive.py" "${args[@]}"
