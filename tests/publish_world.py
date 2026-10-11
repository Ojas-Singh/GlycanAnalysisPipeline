"""A published database and archive in a test bucket, for the tests of publishing one entry.

``build`` bakes a small static tree (``static_tree``) and publishes it with the publisher of the whole
database (``lib.static_publish``), and uploads a raw archive of another glycan, so that the bucket is
what ``scripts/publish_entry.py`` finds in production: a manifest, root files, glycan folders, raw
entries and ``raw/index.json``.  ``Reader`` is the website: a local database that synchronises from the
bucket with the transcription of its reader (``s3_fakes.reader_sync``).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from lib import archive, archive_transfer, entry_publish, static_publish

from s3_fakes import ReaderError, reader_sync, tree
from static_tree import full_bake, write_tree

PUBLISHED = ("GS00001", "GS00002", "GS00005")
OTHER_RAW_NAME = "DGalpb1-2DGlcpb1-OH"


def quiet(message: str) -> None:
    pass


def build(root: Path, backend: Any, raw_backend: Optional[Any] = None, raw_source: Optional[Path] = None) -> Dict[str, Any]:
    """Publish a baked tree of three glycans under ``static/`` and one raw entry under ``raw/``."""
    static = write_tree(Path(root) / "static", PUBLISHED)
    full_bake(static)
    static_publish.publish(static, backend, "static", workers=4, log=quiet)
    raw_backend = raw_backend or backend
    if raw_source is not None:                                  # an archive directory with its index.json
        report = archive_transfer.upload_archive(raw_source, raw_backend, "raw", workers=2, log=quiet)
        assert report["ok"], report
    return {"static": static, "target": entry_publish.Target(static=backend, raw=raw_backend, workers=4)}


def other_archive(root: Path) -> Path:
    """An archive directory with one entry of another glycan and its index, to be uploaded as the stored archive."""
    import numpy as np

    from test_archive import make_source, random_coords

    make_source(Path(root) / "data", OTHER_RAW_NAME, random_coords(6, 10, seed=4))
    archive.convert_entry(Path(root) / "data" / OTHER_RAW_NAME, Path(root) / "archive")
    index, problems = archive.write_index(Path(root) / "archive")
    assert not problems and np.isfinite(index["totals"]["frames"])
    return Path(root) / "archive"


class Reader:
    """The website: a database folder that follows the published prefix."""

    def __init__(self, backend: Any, folder: Path, prefix: str = "static") -> None:
        self.backend, self.folder, self.prefix = backend, Path(folder), prefix

    def sync(self) -> Dict[str, int]:
        return reader_sync(self.backend, self.prefix, self.folder, prune=True)

    def try_sync(self) -> Optional[str]:
        """Synchronise; None when it succeeded, the error of the reader when it gave up (its folder is as before)."""
        before = tree(self.folder) if self.folder.exists() else {}
        try:
            self.sync()
        except ReaderError as exc:
            leftover = self.folder / ".gap-sync-test"            # the real reader removes its staging folder
            if leftover.exists():
                shutil.rmtree(leftover)
            assert tree(self.folder) == before, "a failed synchronisation changed the database of the reader"
            return str(exc)
        return None

    @property
    def global_hash(self) -> Optional[str]:
        path = self.folder / "sync_manifest.json"
        return json.loads(path.read_bytes())["global_hash"] if path.is_file() else None

    def glycans(self) -> list:
        return sorted(path.name for path in self.folder.iterdir() if path.is_dir() and path.name.startswith("GS"))

    def fresh(self, folder: Path) -> "Reader":
        """Another website that has nothing yet: it fetches every file the manifest lists."""
        return Reader(self.backend, folder, self.prefix)


def stored_manifest(backend: Any, prefix: str = "static") -> Dict[str, Any]:
    return json.loads(backend.read_binary(f"{prefix}/sync_manifest.json"))


def stored_tree(backend: Any, prefix: str) -> Dict[str, bytes]:
    """Key -> content of every object below a prefix."""
    return {item["key"]: backend.read_binary(item["key"]) for item in backend.list_objects(prefix)}
