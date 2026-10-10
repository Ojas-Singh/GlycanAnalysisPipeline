#!/usr/bin/env python3
"""Upload explicitly named local ``data/`` directories to Oracle.

This login-node tool intentionally requires one or more ``--glycan`` values,
so it cannot accidentally republish the complete data tree. It never reads or
writes ``process/`` or ``static/``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.storage import OracleStorageBackend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--remote-prefix", default="data")
    parser.add_argument("--glycan", action="append", required=True, help="Local glycan directory to upload; repeatable.")
    parser.add_argument(
        "--file",
        action="append",
        dest="files",
        help="Only upload this relative file from each selected directory; repeatable.",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Do not overwrite existing remote objects.")
    args = parser.parse_args()

    par_url = os.environ.get("GLYCOSHAPE_ORACLE_PAR_URL", "").strip()
    if not par_url:
        raise SystemExit("GLYCOSHAPE_ORACLE_PAR_URL must be set on the login node.")
    data_dir = args.data_dir.resolve()
    remote_prefix = args.remote_prefix.strip("/")
    if not remote_prefix:
        raise SystemExit("--remote-prefix must not be empty.")

    names = [name.strip() for name in args.glycan if name.strip()]
    if len(names) != len(set(names)):
        raise SystemExit("Each --glycan value must be unique.")
    source_dirs = []
    for name in names:
        if Path(name).name != name:
            raise SystemExit(f"Invalid glycan directory name: {name}")
        source = (data_dir / name).resolve()
        if not source.is_dir() or source.parent != data_dir:
            raise SystemExit(f"Local glycan directory does not exist: {source}")
        source_dirs.append(source)

    backend = OracleStorageBackend(par_url)
    uploaded = skipped = 0
    for source in source_dirs:
        if args.files:
            local_paths = []
            for relative in args.files:
                candidate = (source / relative).resolve()
                if candidate.parent != source or not candidate.is_file():
                    raise SystemExit(f"Selected file does not exist directly under {source}: {relative}")
                local_paths.append(candidate)
        else:
            local_paths = sorted(path for path in source.rglob("*") if path.is_file())
        for local_path in local_paths:
            remote_key = f"{remote_prefix}/{local_path.relative_to(data_dir).as_posix()}"
            if args.skip_existing and backend.exists(remote_key):
                skipped += 1
                continue
            backend.write_binary(remote_key, local_path.read_bytes())
            uploaded += 1
    print(f"Oracle data upload complete: uploaded={uploaded}, skipped={skipped}, remote={remote_prefix}/")


if __name__ == "__main__":
    main()
