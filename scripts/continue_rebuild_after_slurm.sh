#!/usr/bin/env bash
# Wait for a process-only Slurm array, then rebuild and publish static output.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 ARRAY_JOB_ID [EXPECTED_TASKS]" >&2
    exit 2
fi

ARRAY_JOB_ID="$1"
EXPECTED_TASKS="${2:-4}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if ! [[ "${ARRAY_JOB_ID}" =~ ^[0-9]+$ && "${EXPECTED_TASKS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ARRAY_JOB_ID and EXPECTED_TASKS must be positive integers." >&2
    exit 2
fi

echo "Waiting for Slurm array ${ARRAY_JOB_ID} (${EXPECTED_TASKS} tasks)..."
while squeue -h -j "${ARRAY_JOB_ID}" | grep -q .; do
    sleep 30
done

# Accounting can lag briefly after the queue entry disappears.
accounting=""
for _ in $(seq 1 12); do
    accounting="$(sacct -j "${ARRAY_JOB_ID}" -X -n -P -o State,ExitCode | sed '/^[[:space:]]*$/d')"
    record_count="$(printf '%s\n' "${accounting}" | sed '/^[[:space:]]*$/d' | wc -l)"
    if [[ "${record_count}" -ge "${EXPECTED_TASKS}" ]]; then
        break
    fi
    sleep 5
done

echo "Final Slurm accounting:"
printf '%s\n' "${accounting}"
record_count="$(printf '%s\n' "${accounting}" | sed '/^[[:space:]]*$/d' | wc -l)"
if [[ "${record_count}" -ne "${EXPECTED_TASKS}" ]]; then
    echo "Expected ${EXPECTED_TASKS} array task records, found ${record_count}; refusing static rebuild." >&2
    exit 1
fi
if printf '%s\n' "${accounting}" | grep -Evq '^COMPLETED[+]?[%|]0:0$'; then
    echo "At least one process-only task did not complete successfully; refusing static rebuild." >&2
    exit 1
fi

echo "All process-only tasks succeeded; starting forced static rebuild."
export GLYCOSHAPE_STATIC_WORKERS="${GLYCOSHAPE_STATIC_WORKERS:-4}"
export GLYCOSHAPE_MAX_WORKERS="${GLYCOSHAPE_MAX_WORKERS:-1}"
export GLYCOSHAPE_UPLOAD_WORKERS="${GLYCOSHAPE_UPLOAD_WORKERS:-1}"
export GLYCOSHAPE_GAP_LOG="${REPO_ROOT}/logs/static_after_${ARRAY_JOB_ID}.log"

"${REPO_ROOT}/.venv/bin/python" -u main.py \
    --mode static-only \
    --update \
    --process-dir "${REPO_ROOT}/process" \
    --output-dir "${REPO_ROOT}/static"

echo "Static rebuild succeeded; publishing static output to Oracle."
"${REPO_ROOT}/.venv/bin/python" -u scripts/upload_static_to_oracle.py \
    --static-dir "${REPO_ROOT}/static" \
    --workers 16 \
    --progress-every 100

echo "Process, static rebuild, and Oracle publication completed successfully."
