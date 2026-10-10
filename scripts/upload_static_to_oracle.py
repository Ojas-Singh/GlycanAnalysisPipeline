#!/usr/bin/env python3
"""Publish local ``static/`` files to the Oracle static prefix.

Run this on a login node only, after ``main.py --mode static-only`` has
finished.  The script has no code path that reads or writes ``process/``.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.storage import OracleStorageBackend


_thread_local = threading.local()


def _get_backend(par_url: str) -> OracleStorageBackend:
    backend = getattr(_thread_local, "backend", None)
    if backend is None:
        backend = OracleStorageBackend(par_url)
        _thread_local.backend = backend
    return backend


def _upload_one(
    *,
    par_url: str,
    source: Path,
    remote_prefix: str,
    local_path: Path,
    skip_existing: bool,
) -> str:
    backend = _get_backend(par_url)
    remote_key = f"{remote_prefix}/{local_path.relative_to(source).as_posix()}"
    if skip_existing and backend.exists(remote_key):
        return "skipped"
    backend.write_binary(remote_key, local_path.read_bytes())
    return "uploaded"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-dir", type=Path, default=Path("static"))
    parser.add_argument("--remote-prefix", default=os.environ.get("GLYCOSHAPE_ORACLE_OUTPUT_PREFIX", "static"))
    parser.add_argument("--skip-existing", action="store_true", help="Do not overwrite existing remote objects.")
    parser.add_argument("--progress-every", type=int, default=250, help="Print progress after this many files.")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel upload workers.")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Relative file or directory under static-dir to publish. Repeat to upload a scoped subset.",
    )
    args = parser.parse_args()

    par_url = os.environ.get("GLYCOSHAPE_ORACLE_PAR_URL", "").strip()
    if not par_url:
        raise SystemExit("GLYCOSHAPE_ORACLE_PAR_URL must be set on the login node.")
    source = args.static_dir.resolve()
    if not source.is_dir():
        raise SystemExit(f"Static directory does not exist: {source}")

    remote_prefix = args.remote_prefix.strip("/")
    if args.include:
        selected = []
        for include in args.include:
            include_path = (source / include).resolve()
            if not include_path.exists():
                raise SystemExit(f"Included path does not exist: {include}")
            try:
                include_path.relative_to(source)
            except ValueError:
                raise SystemExit(f"Included path is outside static-dir: {include}")
            if include_path.is_file():
                selected.append(include_path)
            else:
                selected.extend(path for path in include_path.rglob("*") if path.is_file())
        files = sorted(set(selected))
    else:
        files = sorted(path for path in source.rglob("*") if path.is_file())
    uploaded = skipped = 0
    total = len(files)
    failed = 0
    workers = max(1, args.workers)
    print(f"Oracle static publish starting: files={total}, workers={workers}, remote={remote_prefix}/", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _upload_one,
                par_url=par_url,
                source=source,
                remote_prefix=remote_prefix,
                local_path=local_path,
                skip_existing=args.skip_existing,
            )
            for local_path in files
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            try:
                result = future.result()
            except Exception as exc:
                failed += 1
                print(f"Oracle static publish error: {exc}", flush=True)
            else:
                if result == "skipped":
                    skipped += 1
                else:
                    uploaded += 1
            if args.progress_every > 0 and index % args.progress_every == 0:
                print(
                    f"Oracle static publish progress: {index}/{total} files, uploaded={uploaded}, skipped={skipped}, failed={failed}",
                    flush=True,
                )
    if failed:
        raise SystemExit(f"Oracle static publish failed: uploaded={uploaded}, skipped={skipped}, failed={failed}, remote={remote_prefix}/")
    print(f"Oracle static publish complete: uploaded={uploaded}, skipped={skipped}, remote={remote_prefix}/")


if __name__ == "__main__":
    main()
