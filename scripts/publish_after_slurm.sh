#!/usr/bin/env bash
# Wait for a Slurm job/array, then force-rebuild and publish local static output.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <slurm-job-id> <expected-completed-tasks>" >&2
    exit 2
fi

job_id="$1"
expected_tasks="$2"
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"
echo "Waiting for Slurm job ${job_id} (${expected_tasks} task(s)) at $(date -Is)"

while :; do
    # Avoid a pipefail/SIGPIPE race from `squeue | grep -q`: grep can close
    # early while squeue is still writing, which wrongly terminates the watcher.
    active_jobs="$(squeue -h -j "$job_id" 2>/dev/null || true)"
    [[ -z "$active_jobs" ]] && break
    sleep 60
done

states="$(sacct -j "$job_id" -X -n -P --format=State 2>/dev/null | sed '/^[[:space:]]*$/d')"
completed_count="$(printf '%s\n' "$states" | grep -cx 'COMPLETED' || true)"
if [[ "$completed_count" -ne "$expected_tasks" ]] || printf '%s\n' "$states" | grep -Eq '^(FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL)'; then
    echo "Slurm job ${job_id} did not complete successfully; refusing static publish." >&2
    printf '%s\n' "$states" >&2
    exit 1
fi

set -a
. ./.env
set +a
FORCE_UPDATE=1 ./scripts/run_staged_pipeline.sh publish
