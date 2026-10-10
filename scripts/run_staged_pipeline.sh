#!/usr/bin/env bash
# Login-node helper for the split storage -> SLURM -> storage workflow.
#
# The storage is the one lib/config.py selects from the environment and .env:
# GLYCOSHAPE_STORAGE=s3 uses S3 storage (Cloudflare R2), anything else the
# Oracle PAR bucket.  Only the download and publish commands differ; the
# processing in between is local and the same for both.
set -euo pipefail

cd "$(dirname "$0")/.."

storage_backend() {
  .venv/bin/python -c 'from lib import config; print(config.storage_backend)'
}

case "${1:-}" in
  download)
    shift
    storage="$(storage_backend)"   # on its own line: a failure here stops the script
    if [[ "${storage}" == "s3" ]]; then
      # Raw archive entries (raw/<GLYCAM name>/) -> data/<entry folder>/, verified.
      .venv/bin/python scripts/archive_s3.py download --archive-dir data "$@"
    else
      .venv/bin/python scripts/download_data_from_oracle.py --data-dir data "$@"
    fi
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
    storage="$(storage_backend)"
    .venv/bin/python main.py --mode static-only --process-dir process --output-dir static "${static_args[@]}"
    if [[ "${storage}" == "s3" ]]; then
      # What changed, sync_manifest.json last.
      .venv/bin/python scripts/static_s3.py publish --static-dir static "$@"
    else
      .venv/bin/python scripts/upload_static_to_oracle.py --static-dir static "$@"
    fi
    ;;
  *)
    echo "Usage: $0 {download|submit|publish} [arguments]" >&2
    exit 2
    ;;
esac
