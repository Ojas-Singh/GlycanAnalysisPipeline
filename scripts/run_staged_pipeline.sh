#!/usr/bin/env bash
# Login-node helper for the split Oracle -> SLURM -> Oracle workflow.
set -euo pipefail

cd "$(dirname "$0")/.."

case "${1:-}" in
  download)
    shift
    .venv/bin/python scripts/download_data_from_oracle.py --data-dir data "$@"
    ;;
  submit)
    shift
    sbatch "$@" slurm/process_only_amd.sh
    ;;
  publish)
    shift
    static_args=()
    if [[ "${FORCE_UPDATE:-0}" == "1" ]]; then
      static_args+=(--update)
    fi
    .venv/bin/python main.py --mode static-only --process-dir process --output-dir static "${static_args[@]}"
    .venv/bin/python scripts/upload_static_to_oracle.py --static-dir static "$@"
    ;;
  *)
    echo "Usage: $0 {download|submit|publish} [arguments]" >&2
    exit 2
    ;;
esac
