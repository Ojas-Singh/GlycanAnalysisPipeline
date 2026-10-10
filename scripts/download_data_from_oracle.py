#!/usr/bin/env python3
"""Mirror the Oracle ``data/`` prefix into a local data directory.

This is intentionally a login-node-only operation.  It never writes to
Oracle and it does not create ``process/`` or ``static/`` artifacts.
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--remote-prefix", default="data")
    parser.add_argument("--glycan", action="append", default=[], help="Restrict to a glycan directory; repeatable.")
    parser.add_argument("--glycan-file", type=Path, help="One glycan directory name per line.")
    parser.add_argument("--overwrite", action="store_true", help="Replace already-downloaded files.")
    return parser.parse_args()


def _requested_glycans(args: argparse.Namespace) -> set[str]:
    names = {str(name).strip() for name in args.glycan if str(name).strip()}
    if args.glycan_file:
        names.update(
            line.strip()
            for line in args.glycan_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return names


def main() -> None:
    args = _parse_args()
    par_url = os.environ.get("GLYCOSHAPE_ORACLE_PAR_URL", "").strip()
    if not par_url:
        raise SystemExit("GLYCOSHAPE_ORACLE_PAR_URL must be set on the login node.")

    remote_prefix = args.remote_prefix.strip("/")
    if not remote_prefix:
        raise SystemExit("--remote-prefix must not be empty.")
    requested = _requested_glycans(args)
    backend = OracleStorageBackend(par_url)
    objects = backend.list_files(remote_prefix, "*")
    prefix = f"{remote_prefix}/"
    selected = []
    for object_path in objects:
        key = str(object_path).replace("\\", "/").lstrip("/")
        if not key.startswith(prefix):
            continue
        relative = key[len(prefix) :]
        glycan, separator, _ = relative.partition("/")
        if not separator or (requested and glycan not in requested):
            continue
        selected.append((key, relative))

    if requested:
        found = {relative.partition("/")[0] for _, relative in selected}
        missing = sorted(requested - found)
        if missing:
            raise SystemExit(f"No Oracle data objects found for: {', '.join(missing)}")
    if not selected:
        raise SystemExit(f"No data files found under Oracle prefix {remote_prefix!r}.")

    destination = args.data_dir.resolve()
    downloaded = skipped = 0
    for key, relative in selected:
        target = (destination / relative).resolve()
        try:
            target.relative_to(destination)
        except ValueError as exc:
            raise RuntimeError(f"Refusing unsafe Oracle object path: {key}") from exc
        if target.exists() and not args.overwrite:
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(backend.read_binary(key))
        downloaded += 1

    print(f"Oracle data mirror complete: downloaded={downloaded}, skipped={skipped}, local={destination}")


if __name__ == "__main__":
    main()
