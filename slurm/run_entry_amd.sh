#!/bin/bash
#SBATCH --job-name=gap_run_entry
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# Analyse one trajectory entry on one node, without network access: process
# stage plus the static-ready files, into a self-contained result folder
# (see "One entry on a compute node" in README.md).
#
#   sbatch --export=ALL,ENTRY=/path/to/archive/<folder>,RESULT=/path/to/results/<folder> slurm/run_entry_amd.sh
#
# ENTRY is an archive entry or a simulation.pdb folder. A simulation job can
# run the same command as its last step instead of submitting this script.
# The worker count is SLURM_CPUS_PER_TASK; LEVELS=1,2,3 adds Level 3.

set -euo pipefail

: "${ENTRY:?Set ENTRY to the entry folder.}"
: "${RESULT:?Set RESULT to the result folder.}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(pwd)"

arguments=()
if [[ -n "${LEVELS:-}" ]]; then
    arguments+=(--levels "${LEVELS}")
fi

"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/run_entry.py" "${ENTRY}" "${RESULT}" ${arguments[@]+"${arguments[@]}"}
