"""
Storage abstraction layer for local and Oracle Cloud Object Storage.

Provides unified interface for file operations with support for:
- Local file system (existing functionality)  
- Oracle Cloud Object Storage via Pre-Authenticated Request (PAR) URLs
- S3-compatible object storage such as Cloudflare R2 (GLYCOSHAPE_STORAGE=s3)
- Fallback and compatibility between storage backends
"""

import contextlib
import os
import io
import json
import tempfile
import logging
import fnmatch
import hashlib
import mimetypes
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any, Callable, Dict, Iterator, List, Mapping, Tuple, Union, BinaryIO, TextIO, Optional
from urllib.parse import urlparse
import requests
from lib import config

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def exists(self, path: Union[str, Path]) -> bool:
        """Check if a file/directory exists."""
        pass
    
    @abstractmethod
    def is_file(self, path: Union[str, Path]) -> bool:
        """Check if path is a file."""
        pass
    
    @abstractmethod
    def is_dir(self, path: Union[str, Path]) -> bool:
        """Check if path is a directory."""
        pass
    
    @abstractmethod
    def list_files(self, path: Union[str, Path], pattern: str = "*") -> list:
        """List files in a directory."""
        pass
    
    @abstractmethod
    def open_text(self, path: Union[str, Path], mode: str = "r") -> TextIO:
        """Open a text file."""
        pass
    
    @abstractmethod
    def open_binary(self, path: Union[str, Path], mode: str = "rb") -> BinaryIO:
        """Open a binary file."""
        pass
    
    @abstractmethod
    def write_text(self, path: Union[str, Path], content: str, encoding: str = "utf-8") -> None:
        """Write text content to a file."""
        pass
    
    @abstractmethod
    def write_binary(self, path: Union[str, Path], content: bytes) -> None:
        """Write binary content to a file."""
        pass
    
    @abstractmethod
    def read_text(self, path: Union[str, Path], encoding: str = "utf-8") -> str:
        """Read text content from a file."""
        pass
    
    @abstractmethod
    def read_binary(self, path: Union[str, Path]) -> bytes:
        """Read binary content from a file."""
        pass
    
    @abstractmethod
    def mkdir(self, path: Union[str, Path], parents: bool = True, exist_ok: bool = True) -> None:
        """Create a directory."""
        pass
    
    @abstractmethod
    def unlink(self, path: Union[str, Path]) -> None:
        """Delete a file."""
        pass


class LocalStorageBackend(StorageBackend):
    """Local file system storage backend."""
    
    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """
        Initialize local storage backend.
        
        Args:
            base_path: Base path for all operations (optional)
        """
        self.base_path = Path(base_path) if base_path else None
        logger.info(f"Initialized LocalStorageBackend with base_path: {base_path}")
    
    def _resolve_path(self, path: Union[str, Path]) -> Path:
        """Resolve path relative to base_path if provided."""
        path = Path(path)
        if self.base_path and not path.is_absolute():
            return self.base_path / path
        return path
    
    def exists(self, path: Union[str, Path]) -> bool:
        """Check if a file/directory exists."""
        return self._resolve_path(path).exists()
    
    def is_file(self, path: Union[str, Path]) -> bool:
        """Check if path is a file."""
        return self._resolve_path(path).is_file()
    
    def is_dir(self, path: Union[str, Path]) -> bool:
        """Check if path is a directory.""" 
        return self._resolve_path(path).is_dir()
    
    def list_files(self, path: Union[str, Path], pattern: str = "*") -> list:
        """List files in a directory."""
        resolved_path = self._resolve_path(path)
        if not resolved_path.exists():
            return []
        
        if resolved_path.is_file():
            return [resolved_path]
        
        return list(resolved_path.glob(pattern))
    
    def open_text(self, path: Union[str, Path], mode: str = "r") -> TextIO:
        """Open a text file."""
        resolved_path = self._resolve_path(path)
        return open(resolved_path, mode, encoding="utf-8")
    
    def open_binary(self, path: Union[str, Path], mode: str = "rb") -> BinaryIO:
        """Open a binary file."""
        resolved_path = self._resolve_path(path)
        return open(resolved_path, mode)
    
    def write_text(self, path: Union[str, Path], content: str, encoding: str = "utf-8") -> None:
        """Write text content to a file."""
        resolved_path = self._resolve_path(path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_path.write_text(content, encoding=encoding)
        logger.debug(f"Wrote text to {resolved_path}")
    
    def write_binary(self, path: Union[str, Path], content: bytes) -> None:
        """Write binary content to a file."""
        resolved_path = self._resolve_path(path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_path.write_bytes(content)
        logger.debug(f"Wrote binary to {resolved_path}")
    
    def read_text(self, path: Union[str, Path], encoding: str = "utf-8") -> str:
        """Read text content from a file."""
        resolved_path = self._resolve_path(path)
        return resolved_path.read_text(encoding=encoding)
    
    def read_binary(self, path: Union[str, Path]) -> bytes:
        """Read binary content from a file."""
        resolved_path = self._resolve_path(path)
        return resolved_path.read_bytes()
    
    def mkdir(self, path: Union[str, Path], parents: bool = True, exist_ok: bool = True) -> None:
        """Create a directory."""
        resolved_path = self._resolve_path(path)
        resolved_path.mkdir(parents=parents, exist_ok=exist_ok)
        logger.debug(f"Created directory {resolved_path}")
    
    def unlink(self, path: Union[str, Path]) -> None:
        """Delete a file."""
        resolved_path = self._resolve_path(path)
        resolved_path.unlink()
    
    def delete_local_directory(self, path: Union[str, Path]) -> bool:
        """Safely delete a local directory and all its contents.
        
        Only works on local filesystem paths, not Oracle storage.
        
        Args:
            path: Directory path to delete
            
        Returns:
            True if successful, False otherwise
        """
        import shutil
        try:
            resolved_path = self._resolve_path(path)
            if not resolved_path.exists():
                logger.debug(f"Directory does not exist, nothing to delete: {resolved_path}")
                return True
            
            if not resolved_path.is_dir():
                logger.warning(f"Path is not a directory: {resolved_path}")
                return False
            
            # Remove directory and all contents
            shutil.rmtree(resolved_path)
            logger.debug(f"Successfully deleted directory: {resolved_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete directory {path}: {e}")
            return False


class OracleStorageBackend(StorageBackend):
    """Oracle Cloud Object Storage backend using PAR URLs."""
    
    def __init__(self, par_url: str):
        """
        Initialize Oracle storage backend.
        
        Args:
            par_url: Pre-Authenticated Request URL for Oracle bucket
        """
        # Sanitize PAR URL (remove any whitespace/newlines accidentally copied)
        self.par_url = ''.join(par_url.split()).rstrip('/')
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        self._verify_par_url()
        # Cache for directory structure since Oracle doesn't provide true directories
        self._dir_cache = {}
        logger.info(f"Initialized OracleStorageBackend with PAR URL: {self.par_url}")

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Issue an HTTP request through a persistent session."""
        return self._session.request(method, url, **kwargs)
    
    def _verify_par_url(self) -> None:
        """Verify the PAR URL is accessible."""
        try:
            response = self._request("head", self.par_url, timeout=10)
            # Some PARs may not allow HEAD on the root; accept common statuses
            if response.status_code not in [200, 204, 403, 404]:
                # Try a lightweight GET listing as a fallback check
                try:
                    r2 = self._request("get", self.par_url + "/", params={"limit": 1}, timeout=10)
                    if r2.status_code not in [200, 403, 404]:
                        raise Exception(f"PAR URL returned status {response.status_code}")
                except Exception:
                    raise Exception(f"PAR URL returned status {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to access PAR URL: {e}")
            raise
    
    def _get_object_url(self, path: Union[str, Path]) -> str:
        """Get full, safely-encoded URL for an object in the bucket."""
        from urllib.parse import quote
        # Normalize Windows backslashes to forward slashes for object keys
        path_str = str(path).replace('\\', '/').lstrip('/')
        # Encode special characters but keep slashes as path separators
        encoded = quote(path_str, safe="/")
        return f"{self.par_url}/{encoded}"
    
    def exists(self, path: Union[str, Path]) -> bool:
        """Check if a file/directory exists."""
        path_str = str(path).replace('\\', '/')
        
        # If path ends with '/', it's likely a directory
        if path_str.endswith('/'):
            return self.is_dir(path)
        
        # First try HEAD request for files
        url = self._get_object_url(path)
        try:
            response = self._request("head", url, timeout=10)
            if response.status_code == 200:
                return True
            # Some PARs disallow HEAD; fall back to GET
            if response.status_code in (400, 401, 403, 405):
                try:
                    r2 = self._request("get", url, timeout=10)
                    return r2.status_code == 200
                except Exception:
                    pass
        except Exception:
            pass
        
        # If HEAD failed and path doesn't end with '/', it might be a directory
        # Check if it's a directory by looking for objects with this prefix
        return self.is_dir(path)
    
    def is_file(self, path: Union[str, Path]) -> bool:
        """Check if path is a file."""
        return self.exists(path) and not str(path).endswith('/')
    
    def is_dir(self, path: Union[str, Path]) -> bool:
        """Check if path is a directory by probing for any objects with the prefix."""
        path_str = str(path).replace('\\', '/').rstrip('/')
        prefix = f"{path_str}/" if path_str else ""
        url = f"{self.par_url}/"
        try:
            response = self._request("get", url, params={"prefix": prefix, "limit": 1}, timeout=10)
            if response.status_code != 200:
                return False
            # Prefer JSON parsing (Oracle native API)
            try:
                data = response.json()
                objs = data.get("objects") or data.get("Objects") or []
                return len(objs) > 0
            except ValueError:
                # Fallback to XML (S3-compatible)
                try:
                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(response.text)
                    # Any <Contents><Key> indicates at least one object under the prefix
                    for el in root.findall('.//{*}Contents/{*}Key'):
                        if el.text:
                            return True
                    return False
                except Exception:
                    return False
        except Exception:
            return False
    
    def list_files(self, path: Union[str, Path], pattern: str = "*") -> list:
        """List objects under a given prefix. Returns a list of object paths (as Path)."""
        return self.list_files_paginated(path, pattern)

    def list_files_paginated(self, path: Union[str, Path], pattern: str = "*") -> list:
        """List all objects under a prefix, following Oracle pagination markers."""
        path_str = str(path).replace('\\', '/').rstrip('/')
        prefix = f"{path_str}/" if path_str else ""
        url = f"{self.par_url}/"
        names = []
        next_marker = None
        try:
            while True:
                params = {"prefix": prefix, "limit": 1000}
                if next_marker:
                    params.update({
                        "start": next_marker,
                        "startAfter": next_marker,
                        "continuation-token": next_marker,
                        "marker": next_marker,
                    })
                response = self._request("get", url, params=params, timeout=15)
                if response.status_code != 200:
                    return []

                next_marker = None
                try:
                    data = response.json()
                    objs = data.get("objects") or data.get("Objects") or []
                    for obj in objs:
                        if isinstance(obj, dict):
                            name = obj.get("name") or obj.get("key") or obj.get("Key")
                            if name:
                                names.append(name)
                    next_marker = (
                        data.get("nextStartWith")
                        or data.get("NextStartWith")
                        or data.get("nextMarker")
                        or data.get("NextMarker")
                        or data.get("nextContinuationToken")
                        or data.get("NextContinuationToken")
                    )
                except ValueError:
                    # Fallback to XML (S3-compatible API)
                    next_marker = None
                    page_names = []
                    try:
                        import xml.etree.ElementTree as ET
                        root = ET.fromstring(response.text)
                        page_names = [el.text for el in root.findall('.//{*}Contents/{*}Key') if el.text]
                        names.extend(page_names)
                        is_truncated = root.find('.//{*}IsTruncated')
                        if is_truncated is not None and is_truncated.text and is_truncated.text.lower() == 'true':
                            token_el = root.find('.//{*}NextContinuationToken')
                            marker_el = root.find('.//{*}NextMarker')
                            token = token_el.text if token_el is not None else None
                            marker = marker_el.text if marker_el is not None else None
                            next_marker = token or marker or (page_names[-1] if page_names else None)
                    except Exception:
                        return []

                if not next_marker:
                    break

            return [Path(n) for n in names]
        except Exception as e:
            logger.error(f"Failed to list files in {path}: {e}")
            return []

    def list_common_prefixes(self, prefix: Union[str, Path]) -> list:
        """List immediate child 'directories' under the given prefix using delimiter.

        Returns a list of prefix strings (e.g., 'data/folder').
        """
        base = str(prefix).replace('\\', '/').strip('/')
        url = f"{self.par_url}/"
        try:
            prefixes = []
            next_marker = None
            while True:
                params = {"prefix": (base + '/' if base else ''), "limit": 1000, "delimiter": "/"}
                if next_marker:
                    params.update({
                        "start": next_marker,
                        "startAfter": next_marker,
                        "continuation-token": next_marker,
                        "marker": next_marker,
                    })
                resp = self._request("get", url, params=params, timeout=15)
                if resp.status_code != 200:
                    break
                next_marker = None
                # Try JSON first
                try:
                    data = resp.json()
                    prefs = data.get('prefixes') or data.get('Prefixes') or []
                    for p in prefs:
                        p = str(p).lstrip('/')
                        prefixes.append(p)
                    next_marker = data.get('nextStartWith') or data.get('NextStartWith')
                except ValueError:
                    # XML fallback
                    try:
                        import xml.etree.ElementTree as ET
                        root = ET.fromstring(resp.text)
                        for cp in root.findall('.//{*}CommonPrefixes'):
                            pe = cp.find('{*}Prefix')
                            if pe is not None and pe.text:
                                prefixes.append(pe.text.lstrip('/'))
                        is_truncated = root.find('.//{*}IsTruncated')
                        if is_truncated is not None and is_truncated.text and is_truncated.text.lower() == 'true':
                            token_el = root.find('.//{*}NextContinuationToken')
                            if token_el is not None and token_el.text:
                                next_marker = token_el.text
                    except Exception:
                        pass
                if not next_marker:
                    break
            # Reduce to immediate child folders: data/foo/, not deeper
            out = []
            seen = set()
            base_with_slash = (base + '/') if base else ''
            for p in prefixes:
                rel = p[len(base_with_slash):] if p.startswith(base_with_slash) else p
                seg = rel.split('/', 1)[0]
                full = (base_with_slash + seg).rstrip('/')
                if seg and full not in seen:
                    seen.add(full)
                    out.append(full)
            return out
        except Exception as e:
            logger.error(f"Failed to list prefixes for {prefix}: {e}")
            return []
    
    def open_text(self, path: Union[str, Path], mode: str = "r") -> TextIO:
        """Open a text file."""
        if 'w' in mode:
            # For writing, return a temporary buffer that will be uploaded on close
            return OracleTextBuffer(self, path, mode)
        else:
            # For reading, download content and return BytesIO
            content = self.read_text(path)
            return io.StringIO(content)
    
    def open_binary(self, path: Union[str, Path], mode: str = "rb") -> BinaryIO:
        """Open a binary file."""
        if 'w' in mode:
            # For writing, return a temporary buffer that will be uploaded on close
            return OracleBinaryBuffer(self, path, mode)
        else:
            # For reading, download content and return BytesIO
            content = self.read_binary(path)
            return io.BytesIO(content)
    
    def write_text(self, path: Union[str, Path], content: str, encoding: str = "utf-8") -> None:
        """Write text content to a file."""
        url = self._get_object_url(path)
        try:
            response = self._request("put", url, data=content.encode(encoding), timeout=30)
            response.raise_for_status()
            logger.debug(f"Wrote text to {url}")
        except Exception as e:
            logger.error(f"Failed to write text to {url}: {e}")
            raise
    
    def write_binary(self, path: Union[str, Path], content: bytes) -> None:
        """Write binary content to a file."""
        url = self._get_object_url(path)
        try:
            response = self._request("put", url, data=content, timeout=30)
            response.raise_for_status()
            logger.debug(f"Wrote binary to {url}")
        except Exception as e:
            logger.error(f"Failed to write binary to {url}: {e}")
            raise
    
    def read_text(self, path: Union[str, Path], encoding: str = "utf-8") -> str:
        """Read text content from a file."""
        url = self._get_object_url(path)
        try:
            response = self._request("get", url, timeout=30)
            response.raise_for_status()
            return response.content.decode(encoding)
        except Exception as e:
            logger.error(f"Failed to read text from {url}: {e}")
            raise
    
    def read_binary(self, path: Union[str, Path]) -> bytes:
        """Read binary content from a file."""
        url = self._get_object_url(path)
        try:
            response = self._request("get", url, timeout=30)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Failed to read binary from {url}: {e}")
            raise
    
    def mkdir(self, path: Union[str, Path], parents: bool = True, exist_ok: bool = True) -> None:
        """No-op for object storage; prefixes do not require explicit creation."""
        logger.debug(f"Skipping mkdir for object storage prefix: {path}")
    
    def unlink(self, path: Union[str, Path]) -> None:
        """Delete a file."""
        url = self._get_object_url(path)
        try:
            response = self._request("delete", url, timeout=30)
            response.raise_for_status()
            logger.debug(f"Deleted {url}")
        except Exception as e:
            logger.error(f"Failed to delete {url}: {e}")
            raise


class OracleTextBuffer(io.StringIO):
    """Buffer for writing text files that uploads on close."""
    
    def __init__(self, backend: OracleStorageBackend, path: Union[str, Path], mode: str):
        super().__init__()
        self.backend = backend
        self.path = path
        self.mode = mode
    
    def close(self) -> None:
        """Upload the buffer content to Oracle storage."""
        if 'w' in self.mode:
            content = self.getvalue()
            self.backend.write_text(self.path, content)
        super().close()


class OracleBinaryBuffer(io.BytesIO):
    """Buffer for writing binary files that uploads on close."""
    
    def __init__(self, backend: OracleStorageBackend, path: Union[str, Path], mode: str):
        super().__init__()
        self.backend = backend
        self.path = path
        self.mode = mode
    
    def close(self) -> None:
        """Upload the buffer content to Oracle storage."""
        if 'w' in self.mode:
            content = self.getvalue()
            self.backend.write_binary(self.path, content)
        super().close()


# ---------------------------------------------------------------------------
# S3-compatible object storage (Cloudflare R2, MinIO, AWS S3)
# ---------------------------------------------------------------------------
#: User metadata key (``x-amz-meta-sha256``) that carries the SHA-256 of an
#: object.  Transfers skip an object whose size and this value match.
SHA256_METADATA_KEY = "sha256"
#: Longest object key, in bytes of UTF-8, that S3 and R2 accept.
S3_MAX_KEY_BYTES = 1024

_TRANSIENT_ERROR_NAMES = frozenset({
    "ConnectionError", "TimeoutError", "ReadTimeoutError", "ConnectTimeoutError",
    "IncompleteRead", "IncompleteReadError", "ResponseStreamingError", "ProtocolError",
})


def file_sha256(path: Union[str, Path], chunk: int = 4 << 20) -> Tuple[int, str]:
    """Size and SHA-256 (hexadecimal) of a local file, read in blocks."""
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def parallel_map(function: Callable[[Any], Any], items: Any, workers: int) -> Iterator[Any]:
    """Results of ``function(item)`` on ``workers`` threads, in the order they complete.

    An exception of a call is raised to the caller.  Calls that have not
    started are cancelled then, and when the caller stops early (Ctrl-C);
    the calls that are running are waited for, so nothing is still being
    transferred when this returns or raises.
    """
    pool = ThreadPoolExecutor(max_workers=max(1, int(workers)))
    try:
        futures = [pool.submit(function, item) for item in items]
        for future in as_completed(futures):
            yield future.result()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def _content_type(key: str) -> str:
    guessed, encoding = mimetypes.guess_type(key)
    return guessed if guessed and not encoding else "application/octet-stream"


def _s3_error(exc: BaseException) -> Tuple[str, int]:
    """Error code and HTTP status of a client error, without importing the client library."""
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return "", 0
    code = str((response.get("Error") or {}).get("Code") or "")
    status = (response.get("ResponseMetadata") or {}).get("HTTPStatusCode") or 0
    return code, int(status)


def _is_missing_object(exc: BaseException) -> bool:
    code, status = _s3_error(exc)
    return code in ("404", "NoSuchKey", "NotFound") or (status == 404 and code != "NoSuchBucket")


def _is_transient_error(exc: BaseException) -> bool:
    """A connection that broke while a body was read (the client retries requests itself)."""
    return bool(_TRANSIENT_ERROR_NAMES & {cls.__name__ for cls in type(exc).__mro__})


def _keep_client_loggers_quiet() -> None:
    """Keep the HTTP client libraries above DEBUG: their debug output shows request bodies and headers."""
    for name in ("boto3", "botocore", "s3transfer", "urllib3"):
        client_logger = logging.getLogger(name)
        if client_logger.getEffectiveLevel() < logging.INFO:
            client_logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class S3Settings:
    """Connection and bucket layout of the S3 storage (``GLYCOSHAPE_S3_*``).

    The secret is kept out of ``repr()`` and is only handed to the client.
    """

    endpoint_url: str
    access_key_id: str = field(repr=False)
    secret_access_key: str = field(repr=False)
    region: str = "auto"
    addressing_style: str = "path"
    raw_bucket: str = "glycoshape-archive"
    raw_prefix: str = "raw"
    process_bucket: str = "glycoshape-archive"
    process_prefix: str = "process"
    static_bucket: str = "glycoshape-database"
    static_prefix: str = "static"
    staging_prefix: str = "publish-staging"
    max_attempts: int = 8
    multipart_threshold: int = 64 << 20
    multipart_chunksize: int = 32 << 20
    transfer_concurrency: int = 4
    checksums: str = "when_required"

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "S3Settings":
        """Settings from the environment; raises ``ValueError`` naming what is missing or invalid."""
        env = os.environ if environ is None else environ

        def text(name: str, default: str = "") -> str:
            return str(env.get(f"GLYCOSHAPE_S3_{name}", "") or "").strip() or default

        def number(name: str, default: int, minimum: int) -> int:
            raw = text(name)
            try:
                value = int(raw) if raw else default
            except ValueError:
                raise ValueError(f"GLYCOSHAPE_S3_{name} must be an integer") from None
            if value < minimum:
                raise ValueError(f"GLYCOSHAPE_S3_{name} must be at least {minimum}")
            return value

        def prefix(name: str, default: str) -> str:
            value = text(name, default).strip("/")
            if not value:
                raise ValueError(f"GLYCOSHAPE_S3_{name} must not be empty")
            return value

        missing = [
            f"GLYCOSHAPE_S3_{name}"
            for name in ("ENDPOINT", "ACCESS_KEY_ID", "SECRET_ACCESS_KEY")
            if not text(name)
        ]
        if missing:
            raise ValueError("S3 storage is not configured: set " + ", ".join(missing))
        addressing_style = text("ADDRESSING_STYLE", "path").lower()
        if addressing_style not in ("path", "virtual", "auto"):
            raise ValueError("GLYCOSHAPE_S3_ADDRESSING_STYLE must be path, virtual or auto")
        checksums = text("CHECKSUMS", "when_required").lower()
        if checksums not in ("when_required", "when_supported"):
            raise ValueError("GLYCOSHAPE_S3_CHECKSUMS must be when_required or when_supported")
        return cls(
            endpoint_url=text("ENDPOINT"),
            access_key_id=text("ACCESS_KEY_ID"),
            secret_access_key=text("SECRET_ACCESS_KEY"),
            region=text("REGION", "auto"),
            addressing_style=addressing_style,
            raw_bucket=text("RAW_BUCKET", "glycoshape-archive"),
            raw_prefix=prefix("RAW_PREFIX", "raw"),
            process_bucket=text("PROCESS_BUCKET", "glycoshape-archive"),
            process_prefix=prefix("PROCESS_PREFIX", "process"),
            static_bucket=text("STATIC_BUCKET", "glycoshape-database"),
            static_prefix=prefix("STATIC_PREFIX", "static"),
            staging_prefix=prefix("STAGING_PREFIX", "publish-staging"),
            max_attempts=number("MAX_ATTEMPTS", 8, 1),
            # 5 MiB is the smallest part S3 and R2 accept.
            multipart_threshold=number("MULTIPART_THRESHOLD_MB", 64, 5) << 20,
            multipart_chunksize=number("MULTIPART_CHUNK_MB", 32, 5) << 20,
            transfer_concurrency=number("TRANSFER_CONCURRENCY", 4, 1),
            checksums=checksums,
        )

    def routes(self) -> Dict[str, str]:
        """Key prefix -> bucket for the three kinds of data."""
        return {
            self.raw_prefix: self.raw_bucket,
            self.process_prefix: self.process_bucket,
            self.static_prefix: self.static_bucket,
        }


def _retry_too_many_requests(max_attempts: int) -> Callable[..., Optional[float]]:
    """Retry rule for HTTP 429, which the client does not retry by itself.

    R2 answers ``TooManyRequests`` (429) when it limits the request rate, for
    example for two writes to one key within a second.  The returned handler
    of the client's ``needs-retry`` event asks for another attempt after an
    exponential backoff with jitter, up to ``max_attempts`` attempts.
    """
    def handler(response: Any = None, attempts: int = 0, **_: Any) -> Optional[float]:
        if response is None or attempts >= max_attempts:
            return None
        http_response, parsed = response
        code = ((parsed or {}).get("Error") or {}).get("Code")
        if getattr(http_response, "status_code", None) != 429 and code != "TooManyRequests":
            return None
        return min(20.0, 2.0 ** (attempts - 1)) * (0.5 + random.random() / 2)

    return handler


def create_s3_client(settings: S3Settings, max_connections: int = 10) -> Tuple[Any, Any]:
    """A boto3 S3 client and its transfer configuration for ``settings``.

    Requests are retried by the client with exponential backoff and jitter
    (``standard`` mode, ``settings.max_attempts`` attempts) on connection
    errors, throttling and 5xx responses, and here on HTTP 429.  Creating the
    client does not touch the network.
    """
    try:
        import boto3
        from boto3.s3.transfer import TransferConfig
        from botocore.config import Config
    except ImportError as exc:
        raise RuntimeError(
            "S3 storage needs boto3; install the requirements (uv pip install -r requirements.txt)"
        ) from exc

    _keep_client_loggers_quiet()
    client = boto3.client(
        "s3",
        endpoint_url=settings.endpoint_url,
        region_name=settings.region,
        aws_access_key_id=settings.access_key_id,
        aws_secret_access_key=settings.secret_access_key,
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": settings.addressing_style},
            # total_max_attempts counts the first request; max_attempts would count retries only
            retries={"total_max_attempts": settings.max_attempts, "mode": "standard"},
            max_pool_connections=max(10, int(max_connections)),
            # R2 and other S3-compatible services do not all accept the CRC
            # checksums that newer clients add to every request by default.
            request_checksum_calculation=settings.checksums,
            response_checksum_validation=settings.checksums,
        ),
    )
    client.meta.events.register("needs-retry.s3", _retry_too_many_requests(settings.max_attempts))
    transfer_config = TransferConfig(
        multipart_threshold=settings.multipart_threshold,
        multipart_chunksize=settings.multipart_chunksize,
        max_concurrency=settings.transfer_concurrency,
        use_threads=settings.transfer_concurrency > 1,
        preferred_transfer_client="classic",
    )
    return client, transfer_config


class S3StorageBackend(StorageBackend):
    """S3-compatible object storage (Cloudflare R2, MinIO, AWS S3).

    A path is an object key.  ``routes`` maps a leading key prefix to the
    bucket that holds it (``raw`` and ``process`` live in the archive bucket,
    ``static`` in the database bucket); every other key belongs to ``bucket``
    when one is given.  Keys are used exactly as given apart from a leading
    ``/``, so a GLYCAM name with ``[``, ``]`` and ``,`` is a valid key.

    ``client`` is a boto3 S3 client, or an object with the same methods.
    """

    def __init__(
        self,
        client: Any,
        bucket: Optional[str] = None,
        routes: Optional[Mapping[str, str]] = None,
        transfer_config: Any = None,
        page_size: int = 1000,
        stream_attempts: int = 4,
        sleep: Callable[[float], None] = time.sleep,
        copy_part_size: int = 1 << 30,
    ):
        self._client = client
        self.bucket = bucket
        self._routes = sorted(
            ((str(prefix).strip("/"), target) for prefix, target in (routes or {}).items()),
            key=lambda item: -len(item[0]),
        )
        self._transfer_config = transfer_config
        self._copy_part_size = max(5 << 20, int(copy_part_size))   # 5 MiB is the smallest part of a multipart copy
        self._page_size = max(1, int(page_size))
        self._stream_attempts = max(1, int(stream_attempts))
        self._sleep = sleep
        logger.info(
            "Initialized S3StorageBackend (bucket=%s, routes=%s)",
            bucket,
            {prefix: target for prefix, target in self._routes} or None,
        )

    # -- keys -----------------------------------------------------------
    @staticmethod
    def _key(path: Union[str, Path]) -> str:
        key = path.as_posix() if isinstance(path, PurePath) else str(path)
        return key.lstrip("/")

    def _route(self, key: str) -> Optional[str]:
        for prefix, target in self._routes:
            if key == prefix or key.startswith(prefix + "/"):
                return target
        return self.bucket

    def is_routed(self, path: Union[str, Path]) -> bool:
        """True when ``path`` is an object key of one of the configured buckets."""
        return self._route(self._key(path)) is not None

    def _locate(self, path: Union[str, Path]) -> Tuple[str, str]:
        key = self._key(path)
        bucket = self._route(key)
        if bucket is None:
            raise FileNotFoundError(f"no S3 bucket is configured for object key {key!r}")
        if not key or len(key.encode("utf-8")) > S3_MAX_KEY_BYTES:
            raise ValueError(f"object key is empty or longer than {S3_MAX_KEY_BYTES} bytes: {key[:80]!r}")
        return bucket, key

    def _retry_stream(self, action: Callable[[], Any], what: str) -> Any:
        """Run ``action`` again, with backoff, when the connection broke while a body was read."""
        delay = 0.5
        for attempt in range(1, self._stream_attempts + 1):
            try:
                return action()
            except Exception as exc:
                if attempt == self._stream_attempts or not _is_transient_error(exc):
                    raise
                pause = delay * (0.5 + random.random() / 2)
                logger.warning(
                    "Reading %s failed with %s (attempt %d of %d); retrying in %.1f s",
                    what, type(exc).__name__, attempt, self._stream_attempts, pause,
                )
                self._sleep(pause)
                delay = min(delay * 2, 20.0)

    def _open_body(self, bucket: str, key: str) -> Any:
        try:
            return self._client.get_object(Bucket=bucket, Key=key)["Body"]
        except Exception as exc:
            if _is_missing_object(exc):
                raise FileNotFoundError(f"no such object: {key}") from None
            raise

    # -- listing --------------------------------------------------------
    def list_objects(self, prefix: Union[str, Path] = "") -> Iterator[Dict[str, Any]]:
        """Every object whose key starts with ``prefix`` as ``{"key", "size", "etag"}``, in key order.

        The listing is followed over all its pages (continuation tokens).
        """
        key_prefix = self._key(prefix)
        bucket = self._route(key_prefix)
        if bucket is None:
            return
        token: Optional[str] = None
        while True:
            request: Dict[str, Any] = {"Bucket": bucket, "Prefix": key_prefix, "MaxKeys": self._page_size}
            if token:
                request["ContinuationToken"] = token
            page = self._client.list_objects_v2(**request)
            for item in page.get("Contents") or []:
                yield {
                    "key": item["Key"],
                    "size": int(item.get("Size", 0)),
                    "etag": str(item.get("ETag", "")).strip('"'),
                }
            if not page.get("IsTruncated"):
                return
            token = page.get("NextContinuationToken")
            if not token:
                raise RuntimeError(f"truncated listing of {key_prefix!r} without a continuation token")

    def list_files(self, path: Union[str, Path], pattern: str = "*") -> list:
        """Keys of all objects below ``path`` (recursive), as ``Path``.

        ``pattern`` filters on the first path segment below ``path``.
        """
        return self.list_files_paginated(path, pattern)

    def list_files_paginated(self, path: Union[str, Path], pattern: str = "*") -> list:
        base = self._key(path).rstrip("/")
        prefix = f"{base}/" if base else ""
        found = []
        for item in self.list_objects(prefix):
            key = item["key"]
            if key.endswith("/"):
                continue
            if pattern != "*" and not fnmatch.fnmatchcase(key[len(prefix):].split("/", 1)[0], pattern):
                continue
            found.append(Path(key))
        return found

    def list_common_prefixes(self, prefix: Union[str, Path]) -> list:
        """Immediate child 'directories' of ``prefix``, for example ``static/GS00001``."""
        base = self._key(prefix).strip("/")
        key_prefix = f"{base}/" if base else ""
        bucket = self._route(key_prefix)
        if bucket is None:
            return []
        found: List[str] = []
        token: Optional[str] = None
        while True:
            request: Dict[str, Any] = {
                "Bucket": bucket, "Prefix": key_prefix, "Delimiter": "/", "MaxKeys": self._page_size,
            }
            if token:
                request["ContinuationToken"] = token
            page = self._client.list_objects_v2(**request)
            found.extend(str(item["Prefix"]).rstrip("/") for item in page.get("CommonPrefixes") or [])
            if not page.get("IsTruncated"):
                return found
            token = page.get("NextContinuationToken")
            if not token:
                raise RuntimeError(f"truncated listing of {key_prefix!r} without a continuation token")

    # -- existence ------------------------------------------------------
    def head(self, path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Size, SHA-256 metadata and ETag of an object, or ``None`` when it does not exist.

        The answer to a HEAD request has no body, so a bucket that does not
        exist looks like a missing object here; listings and reads raise.
        """
        if not self._key(path) or not self.is_routed(path):
            return None
        bucket, key = self._locate(path)
        try:
            response = self._client.head_object(Bucket=bucket, Key=key)
        except Exception as exc:
            if _is_missing_object(exc):
                return None
            raise
        metadata = {str(name).lower(): value for name, value in (response.get("Metadata") or {}).items()}
        return {
            "size": int(response.get("ContentLength", 0)),
            "sha256": metadata.get(SHA256_METADATA_KEY),
            "etag": str(response.get("ETag", "")).strip('"'),
            "metadata": metadata,
        }

    def exists(self, path: Union[str, Path]) -> bool:
        """Check if an object, or any object below the prefix, exists."""
        key = self._key(path)
        if not key.endswith("/") and self.head(key) is not None:
            return True
        return self.is_dir(key)

    def is_file(self, path: Union[str, Path]) -> bool:
        """Check if path is an object."""
        key = self._key(path)
        return bool(key) and not key.endswith("/") and self.head(key) is not None

    def is_dir(self, path: Union[str, Path]) -> bool:
        """Check if any object exists below the prefix."""
        base = self._key(path).rstrip("/")
        key_prefix = f"{base}/" if base else ""
        bucket = self._route(key_prefix)
        if bucket is None:
            return False
        page = self._client.list_objects_v2(Bucket=bucket, Prefix=key_prefix, MaxKeys=1)
        return bool(page.get("Contents"))

    # -- reading --------------------------------------------------------
    def read_binary(self, path: Union[str, Path]) -> bytes:
        """Read an object into memory; use :meth:`download_file` for large objects."""
        bucket, key = self._locate(path)

        def read() -> bytes:
            body = self._open_body(bucket, key)
            try:
                return body.read()
            finally:
                body.close()

        return self._retry_stream(read, key)

    def read_text(self, path: Union[str, Path], encoding: str = "utf-8") -> str:
        """Read text content from an object."""
        return self.read_binary(path).decode(encoding)

    def stream_sha256(self, path: Union[str, Path], chunk: int = 1 << 20) -> Tuple[int, str]:
        """Size and SHA-256 of an object's content, downloaded as a stream and not stored."""
        bucket, key = self._locate(path)

        def read() -> Tuple[int, str]:
            body = self._open_body(bucket, key)
            digest = hashlib.sha256()
            size = 0
            try:
                for block in iter(lambda: body.read(chunk), b""):
                    digest.update(block)
                    size += len(block)
            finally:
                body.close()
            return size, digest.hexdigest()

        return self._retry_stream(read, key)

    def download_file(self, path: Union[str, Path], local_path: Union[str, Path]) -> None:
        """Download an object to a local file as a stream (ranged parts for large objects).

        The transfer writes a temporary file next to ``local_path`` and
        renames it when complete, and retries parts whose connection broke.
        """
        bucket, key = self._locate(path)
        options = {"Config": self._transfer_config} if self._transfer_config is not None else {}
        try:
            self._client.download_file(Bucket=bucket, Key=key, Filename=str(local_path), **options)
        except Exception as exc:
            if _is_missing_object(exc):
                raise FileNotFoundError(f"no such object: {key}") from None
            raise

    def open_text(self, path: Union[str, Path], mode: str = "r") -> TextIO:
        """Open a text object; a writer uploads when it is closed."""
        if 'w' in mode:
            return OracleTextBuffer(self, path, mode)
        return io.StringIO(self.read_text(path))

    def open_binary(self, path: Union[str, Path], mode: str = "rb") -> BinaryIO:
        """Open a binary object; a writer uploads when it is closed."""
        if 'w' in mode:
            return OracleBinaryBuffer(self, path, mode)
        return io.BytesIO(self.read_binary(path))

    # -- writing --------------------------------------------------------
    def write_binary(self, path: Union[str, Path], content: bytes) -> None:
        """Write a small object from memory in one request, with its SHA-256 as metadata."""
        bucket, key = self._locate(path)
        self._client.put_object(
            Bucket=bucket,
            Key=key,
            Body=content,
            ContentType=_content_type(key),
            Metadata={SHA256_METADATA_KEY: hashlib.sha256(content).hexdigest()},
        )
        logger.debug("Wrote %d bytes to s3://%s/%s", len(content), bucket, key)

    def write_text(self, path: Union[str, Path], content: str, encoding: str = "utf-8") -> None:
        """Write text content to an object."""
        self.write_binary(path, content.encode(encoding))

    def write_binary_if_absent(self, path: Union[str, Path], content: bytes) -> Optional[bool]:
        """Create a small object only when no object has this key (``If-None-Match: *``).

        ``True``: the object was created.  ``False``: an object is there and
        was left alone (the service answered 412, or 409 for a write that
        raced this one).  ``None``: the client or the service does not know
        conditional writes; nothing is known about the key then, and the
        caller has to fall back to a plain write.  A service that accepts
        the header and ignores it cannot be told apart here: see
        ``lib.entry_publish.conditional_writes``, which probes for that.
        """
        bucket, key = self._locate(path)
        try:
            self._client.put_object(
                Bucket=bucket,
                Key=key,
                Body=content,
                ContentType=_content_type(key),
                Metadata={SHA256_METADATA_KEY: hashlib.sha256(content).hexdigest()},
                IfNoneMatch="*",
            )
        except TypeError:                 # a client without the parameter
            return None
        except Exception as exc:
            code, status = _s3_error(exc)
            if status in (409, 412) or code in ("PreconditionFailed", "ConditionalRequestConflict"):
                return False
            if status == 501 or code in ("NotImplemented", "InvalidArgument", "InvalidRequest") \
                    or type(exc).__name__ == "ParamValidationError":
                return None
            raise
        return True

    def upload_file(
        self,
        local_path: Union[str, Path],
        path: Union[str, Path],
        sha256: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload a local file as a stream; multipart above the configured threshold.

        The object carries the SHA-256 of the file as user metadata.  Pass
        ``sha256`` when it is already known, otherwise the file is hashed here.
        """
        bucket, key = self._locate(path)
        if sha256 is None:
            sha256 = file_sha256(local_path)[1]
        options = {"Config": self._transfer_config} if self._transfer_config is not None else {}
        self._client.upload_file(
            Filename=str(local_path),
            Bucket=bucket,
            Key=key,
            ExtraArgs={"Metadata": {SHA256_METADATA_KEY: sha256}, "ContentType": _content_type(key)},
            **options,
        )
        logger.debug("Uploaded %s to s3://%s/%s", local_path, bucket, key)
        return {"key": key, "sha256": sha256}

    def copy(
        self,
        source: Union[str, Path],
        path: Union[str, Path],
        sha256: Optional[str] = None,
        size: Optional[int] = None,
    ) -> None:
        """Copy an object inside the storage; the content does not pass through this machine.

        One ``CopyObject`` request up to ``copy_part_size`` bytes, a multipart
        copy in parts of that size above it.  Only plain requests are used
        (no conditional copy headers, which R2 does not implement).  The copy
        carries ``sha256`` as metadata; size and SHA-256 are read from the
        source when they are not given.
        """
        source_bucket, source_key = self._locate(source)
        bucket, key = self._locate(path)
        if sha256 is None or size is None:
            head = self.head(source_key)
            if head is None:
                raise FileNotFoundError(f"no such object: {source_key}")
            sha256 = head["sha256"] if sha256 is None else sha256
            size = head["size"] if size is None else size
        copy_source = {"Bucket": source_bucket, "Key": source_key}
        described = {
            "Metadata": {SHA256_METADATA_KEY: sha256} if sha256 else {},
            "ContentType": _content_type(key),
        }
        if size <= self._copy_part_size:
            self._client.copy_object(
                Bucket=bucket, Key=key, CopySource=copy_source, MetadataDirective="REPLACE", **described)
            return
        upload_id = self._client.create_multipart_upload(Bucket=bucket, Key=key, **described)["UploadId"]
        try:
            parts = []
            for number, start in enumerate(range(0, size, self._copy_part_size), start=1):
                copied = self._client.upload_part_copy(
                    Bucket=bucket,
                    Key=key,
                    UploadId=upload_id,
                    PartNumber=number,
                    CopySource=copy_source,
                    CopySourceRange=f"bytes={start}-{min(start + self._copy_part_size, size) - 1}",
                )
                parts.append({"ETag": copied["CopyPartResult"]["ETag"], "PartNumber": number})
            self._client.complete_multipart_upload(
                Bucket=bucket, Key=key, UploadId=upload_id, MultipartUpload={"Parts": parts})
        except BaseException:
            try:
                self._client.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
            except Exception:
                logger.warning("Could not abort the multipart copy to s3://%s/%s", bucket, key)
            raise

    def mkdir(self, path: Union[str, Path], parents: bool = True, exist_ok: bool = True) -> None:
        """No-op for object storage; prefixes do not require explicit creation."""
        logger.debug(f"Skipping mkdir for object storage prefix: {path}")

    def unlink(self, path: Union[str, Path]) -> None:
        """Delete an object (deleting a missing object is not an error in S3)."""
        bucket, key = self._locate(path)
        self._client.delete_object(Bucket=bucket, Key=key)
        logger.debug("Deleted s3://%s/%s", bucket, key)


def create_s3_backend(
    settings: Optional[S3Settings] = None,
    bucket: Optional[str] = None,
    max_connections: int = 10,
) -> S3StorageBackend:
    """S3 backend from ``GLYCOSHAPE_S3_*`` (or ``settings``).

    Without ``bucket`` the backend routes keys by their prefix to the raw,
    process and static buckets; with ``bucket`` every key belongs to it.
    """
    settings = settings or S3Settings.from_env()
    client, transfer_config = create_s3_client(settings, max_connections=max_connections)
    if bucket is not None:
        return S3StorageBackend(client, bucket=bucket, transfer_config=transfer_config)
    return S3StorageBackend(client, routes=settings.routes(), transfer_config=transfer_config)


class StorageManager:
    """
    Unified storage manager that abstracts file operations across backends.
    
    Provides the same interface regardless of whether using local filesystem
    or Oracle Cloud Object Storage.
    """
    
    def __init__(self, backend: Optional[StorageBackend] = None):
        """
        Initialize storage manager.
        
        Args:
            backend: Storage backend to use. If None, will be auto-detected.
        """
        self.backend = backend
        if self.backend is None:
            self.backend = self._detect_backend()
        # Local backend used for staging writes even in Oracle mode
        self._local_backend = LocalStorageBackend()
        # "Oracle mode" below means remote object storage: the PAR backend or S3.
        self._is_oracle = isinstance(self.backend, (OracleStorageBackend, S3StorageBackend))

    def _is_data_prefix(self, path: Union[str, Path]) -> bool:
        p = str(path).lstrip('/')
        return p == 'data' or p.startswith('data/')
    
    def _is_process_prefix(self, path: Union[str, Path]) -> bool:
        """Check if path is a process directory path."""
        p = str(path).lstrip('/')
        return 'process' in p

    def _is_output_prefix(self, path: Union[str, Path]) -> bool:
        """Check if path is an output/static directory path."""
        p = str(path).lstrip('/')
        # Default output dir is 'static'; also handle any path containing '/output' subdir
        return p == 'static' or p.startswith('static/') or 'output/' in p or p.endswith('/output') or p == 'output'

    def _should_write_locally(self, path: Union[str, Path]) -> bool:
        # In Oracle mode, stage all writes locally; sync handled elsewhere
        return self._is_oracle
    
    def _detect_backend(self) -> StorageBackend:
        """Auto-detect appropriate storage backend from environment."""
        selected = config.selected_storage()
        if selected == "s3":
            logger.info("Using S3 object storage (GLYCOSHAPE_STORAGE=s3)")
            return create_s3_backend()

        # Check if Oracle PAR URL is provided
        par_url = os.environ.get("GLYCOSHAPE_ORACLE_PAR_URL")
        if par_url and selected != "local":
            logger.info(f"Using Oracle Cloud Storage (PAR URL detected)")
            return OracleStorageBackend(par_url)
        
        # Default to local storage
        logger.info("Using local file system storage")
        return LocalStorageBackend()
    
    def _wrap_path(self, path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> Union[str, Path]:
        """Wrap path with base directory if provided."""
        if base_dir is not None:
            return Path(base_dir) / path
        return path
    
    def open(self, path: Union[str, Path], mode: str = "r", base_dir: Optional[Union[str, Path]] = None) -> Union[TextIO, BinaryIO]:
        """
        Open a file using the appropriate backend.
        
        Args:
            path: File path
            mode: File mode ('r', 'w', 'rb', 'wb', etc.)
            base_dir: Base directory (used when path is relative)
            
        Returns:
            File-like object
        """
        path = self._wrap_path(path, base_dir)
        
        # Absolute local reads should always use local filesystem
        try:
            pstr = str(path)
            if base_dir is not None and not os.path.isabs(pstr):
                pstr = str(self._wrap_path(path, base_dir))
            if 'r' in mode and not any(f in mode for f in ('w','a','x','+')):
                if os.path.isabs(pstr) and os.path.exists(pstr):
                    if 'b' in mode:
                        return self._local_backend.open_binary(pstr, mode)
                    return self._local_backend.open_text(pstr, mode)
        except Exception:
            pass

        # Prefer local for reads when file exists locally (staged outputs)
        is_write = any(flag in mode for flag in ('w', 'a', 'x', '+'))
        if not is_write and self._is_oracle:
            try:
                if self._local_backend.exists(path):
                    if 'b' in mode:
                        return self._local_backend.open_binary(path, mode)
                    return self._local_backend.open_text(path, mode)
            except Exception:
                pass

        # Route writes to local when in Oracle mode to avoid PAR write limits
        if is_write and self._should_write_locally(path):
            if 'b' in mode:
                return self._local_backend.open_binary(path, mode)
            return self._local_backend.open_text(path, mode)

        if 'b' in mode:
            return self.backend.open_binary(path, mode)
        else:
            return self.backend.open_text(path, mode)
    
    def exists(self, path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> bool:
        """Check if a file/directory exists."""
        path = self._wrap_path(path, base_dir)
        # Absolute local path short-circuit
        try:
            pstr = str(path)
            if os.path.isabs(pstr) and os.path.exists(pstr):
                return True
        except Exception:
            pass
        # In Oracle mode, always prefer fresh local staged files first, then fall back to remote.
        if self._is_oracle:
            try:
                if self._local_backend.exists(path):
                    return True
            except Exception:
                pass
            path_str = str(path).lower()
            if 'inventory' in path_str and path_str.endswith('.csv'):
                return self.backend.exists(path)
            return self.backend.exists(path)
        return self.backend.exists(path)
    
    def is_file(self, path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> bool:
        """Check if path is a file."""
        path = self._wrap_path(path, base_dir)
        try:
            pstr = str(path)
            if os.path.isabs(pstr) and os.path.isfile(pstr):
                return True
        except Exception:
            pass
        # In Oracle mode, prefer local staged files first, then fall back to remote
        if self._is_oracle:
            try:
                if self._local_backend.is_file(path):
                    return True
            except Exception:
                pass
            return self.backend.is_file(path)
        return self.backend.is_file(path)
    
    def is_dir(self, path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> bool:
        """Check if path is a directory."""
        path = self._wrap_path(path, base_dir)
        try:
            pstr = str(path)
            if os.path.isabs(pstr) and os.path.isdir(pstr):
                return True
        except Exception:
            pass
        # In Oracle mode, prefer local staged dirs first, then fall back to remote
        if self._is_oracle:
            try:
                if self._local_backend.is_dir(path):
                    return True
            except Exception:
                pass
            return self.backend.is_dir(path)
        return self.backend.is_dir(path)
    
    def list_files(self, path: Union[str, Path], pattern: str = "*", base_dir: Optional[Union[str, Path]] = None) -> list:
        """List files in a directory."""
        path = self._wrap_path(path, base_dir)
        try:
            pstr = str(path)
            if os.path.isabs(pstr):
                return self._local_backend.list_files(path, pattern)
        except Exception:
            pass
        # In Oracle mode, prioritize remote for process and data prefixes
        if self._is_oracle:
            path_str = str(path)
            if self._is_process_prefix(path_str) or self._is_data_prefix(path_str) or self._is_output_prefix(path_str):
                return self.backend.list_files(path, pattern)
            # S3: the configured raw/process/static prefixes need not have the names above
            if isinstance(self.backend, S3StorageBackend) and self.backend.is_routed(path_str):
                return self.backend.list_files(path, pattern)
            # For other directories, prefer local listings (best effort)
            try:
                return self._local_backend.list_files(path, pattern)
            except Exception:
                return []
        return self.backend.list_files(path, pattern)

    def list_files_paginated(self, path: Union[str, Path], pattern: str = "*", base_dir: Optional[Union[str, Path]] = None) -> list:
        """List all files under a directory/prefix when the backend supports pagination."""
        path = self._wrap_path(path, base_dir)
        if self._is_oracle and hasattr(self.backend, "list_files_paginated"):
            return self.backend.list_files_paginated(path, pattern)
        return self.list_files(path, pattern)

    def list_dirs(self, path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> list:
        """List immediate subdirectories for the given path.

        - For Oracle backend, uses delimiter to fetch common prefixes (fast).
        - For local backend, returns directories under the path.
        Returns a list of Path-like entries.
        """
        path = self._wrap_path(path, base_dir)
        if isinstance(self.backend, (OracleStorageBackend, S3StorageBackend)):
            base = str(path)
            prefixes = self.backend.list_common_prefixes(base)
            return [Path(p) for p in prefixes]
        # Local: filter directories under path
        items = self._local_backend.list_files(path, '*')
        dirs = []
        for it in items:
            try:
                if self._local_backend.is_dir(it):
                    dirs.append(it)
            except Exception:
                pass
        return dirs
    
    def mkdir(self, path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None, 
              parents: bool = True, exist_ok: bool = True) -> None:
        """Create a directory."""
        path = self._wrap_path(path, base_dir)
        # For Oracle, create local dirs; remote prefixes don't need mkdir
        if self._is_oracle:
            self._local_backend.mkdir(path, parents, exist_ok)
        else:
            self.backend.mkdir(path, parents, exist_ok)
    
    def read_text(self, path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None, encoding: str = "utf-8") -> str:
        """Read text content from a file."""
        path = self._wrap_path(path, base_dir)
        try:
            pstr = str(path)
            if os.path.isabs(pstr) and os.path.exists(pstr):
                return self._local_backend.read_text(pstr, encoding)
        except Exception:
            pass
        # In Oracle mode, prefer freshly staged local files, then fall back to remote.
        if self._is_oracle:
            if self._local_backend.exists(path):
                return self._local_backend.read_text(path, encoding)
            path_str = str(path)
            if self._is_process_prefix(path_str) or self._is_output_prefix(path_str) or ('inventory' in path_str.lower() and path_str.lower().endswith('.csv')):
                return self.backend.read_text(path, encoding)
        return self.backend.read_text(path, encoding)
    
    def read_binary(self, path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> bytes:
        """Read binary content from a file."""
        path = self._wrap_path(path, base_dir)
        try:
            pstr = str(path)
            if os.path.isabs(pstr) and os.path.exists(pstr):
                return self._local_backend.read_binary(pstr)
        except Exception:
            pass
        # In Oracle mode, prefer freshly staged local files, then fall back to remote.
        if self._is_oracle:
            if self._local_backend.exists(path):
                return self._local_backend.read_binary(path)
            path_str = str(path)
            if self._is_process_prefix(path_str) or self._is_output_prefix(path_str) or ('inventory' in path_str.lower() and path_str.lower().endswith('.csv')):
                return self.backend.read_binary(path)
        return self.backend.read_binary(path)
    
    def write_text(self, path: Union[str, Path], content: str, base_dir: Optional[Union[str, Path]] = None, encoding: str = "utf-8") -> None:
        """Write text content to a file."""
        path = self._wrap_path(path, base_dir)
        if self._should_write_locally(path):
            self._local_backend.write_text(path, content, encoding)
        else:
            self.backend.write_text(path, content, encoding)
    
    def write_binary(self, path: Union[str, Path], content: bytes, base_dir: Optional[Union[str, Path]] = None) -> None:
        """Write binary content to a file."""
        path = self._wrap_path(path, base_dir)
        if self._should_write_locally(path):
            self._local_backend.write_binary(path, content)
        else:
            self.backend.write_binary(path, content)
    
    def unlink(self, path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> None:
        """Delete a file."""
        path = self._wrap_path(path, base_dir)
        # Try local first then remote
        try:
            if self._is_oracle and self._local_backend.exists(path):
                return self._local_backend.unlink(path)
        except Exception:
            pass
        self.backend.unlink(path)

    def upload_dir(self, local_dir: Union[str, Path], remote_prefix: Union[str, Path], skip_existing: bool = True) -> None:
        """Recursively upload a local directory to the current backend under remote_prefix.

        Args:
            local_dir: Local directory to traverse and upload
            remote_prefix: Destination prefix/path in the active backend
            skip_existing: When True, do not overwrite files that already exist remotely

        Note: Intended for syncing local outputs to remote object storage in Oracle mode.
        """
        local_dir = Path(local_dir)
        remote_prefix = str(remote_prefix).rstrip('/')
        if not local_dir.exists():
            logger.debug(f"upload_dir: local path does not exist: {local_dir}")
            return
        upload_jobs = []
        for root, _, files in os.walk(local_dir):
            for fname in files:
                lpath = Path(root) / fname
                rel = lpath.relative_to(local_dir).as_posix()
                rpath = f"{remote_prefix}/{rel}" if remote_prefix else rel
                upload_jobs.append((lpath, rpath))

        success_count = 0
        fail_count = 0
        skip_count = 0

        def _upload_one(job: tuple[Path, str]) -> tuple[str, Path, str, Optional[str]]:
            lpath, rpath = job
            try:
                if skip_existing:
                    try:
                        if self.backend.exists(rpath):
                            return ("skipped", lpath, rpath, None)
                    except Exception:
                        pass
                if isinstance(self.backend, S3StorageBackend):
                    # streamed, multipart for large files; nothing is read into memory
                    self.backend.upload_file(lpath, rpath)
                    return ("uploaded", lpath, rpath, None)
                with open(lpath, 'rb') as f:
                    data = f.read()
                self.backend.write_binary(rpath, data)
                return ("uploaded", lpath, rpath, None)
            except Exception as exc:
                return ("failed", lpath, rpath, str(exc))

        max_workers = max(1, int(getattr(config, "upload_workers", 1)))
        if isinstance(self.backend, (OracleStorageBackend, S3StorageBackend)) and len(upload_jobs) > 1 and max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {executor.submit(_upload_one, job): job for job in upload_jobs}
                for future in as_completed(future_map):
                    status, lpath, rpath, error = future.result()
                    if status == "uploaded":
                        success_count += 1
                    elif status == "skipped":
                        skip_count += 1
                    else:
                        logger.warning(f"upload_dir: failed to upload {lpath} -> {rpath}: {error}")
                        fail_count += 1
        else:
            for job in upload_jobs:
                status, lpath, rpath, error = _upload_one(job)
                if status == "uploaded":
                    success_count += 1
                elif status == "skipped":
                    skip_count += 1
                else:
                    logger.warning(f"upload_dir: failed to upload {lpath} -> {rpath}: {error}")
                    fail_count += 1
        logger.info(f"upload_dir: uploaded {success_count} files to '{remote_prefix}' (skipped {skip_count} existing, {fail_count} failures)")

    def delete_local_directory(self, path: Union[str, Path]) -> bool:
        """Safely delete a local directory and all its contents.
        
        Delegates to the local backend since this operation only works
        on local filesystem paths, not Oracle storage.
        
        Args:
            path: Directory path to delete
            
        Returns:
            True if successful, False otherwise
        """
        return self._local_backend.delete_local_directory(path)


# Global storage manager instance
_storage_manager: Optional[StorageManager] = None


def get_storage_manager() -> StorageManager:
    """Get the global storage manager instance."""
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = StorageManager()
    return _storage_manager


@contextlib.contextmanager
def local_storage() -> Iterator[None]:
    """Local files for the pipeline functions that go through the storage manager.

    Inside the block the storage manager of the process is the local file
    system, whatever ``GLYCOSHAPE_STORAGE``, ``GLYCOSHAPE_ORACLE_PAR_URL`` or
    a ``.env`` file selects; the manager that was there is put back
    afterwards.  For code that works on local folders only (completing and
    publishing one entry) in a process whose environment names a remote
    storage.  The manager is one per process: not for concurrent use.
    """
    global _storage_manager
    previous = _storage_manager
    _storage_manager = StorageManager(LocalStorageBackend())
    try:
        yield
    finally:
        _storage_manager = previous


def reset_storage_manager() -> None:
    """Reset the global storage manager instance (useful for testing)."""
    global _storage_manager
    _storage_manager = None


# Convenience functions that use the global storage manager
def open_file(path: Union[str, Path], mode: str = "r", base_dir: Optional[Union[str, Path]] = None) -> Union[TextIO, BinaryIO]:
    """Open a file using the appropriate storage backend."""
    return get_storage_manager().open(path, mode, base_dir)


def file_exists(path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> bool:
    """Check if a file exists."""
    return get_storage_manager().exists(path, base_dir)


def file_is_file(path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> bool:
    """Check if path is a file."""
    return get_storage_manager().is_file(path, base_dir)


def file_is_dir(path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> bool:
    """Check if path is a directory."""
    return get_storage_manager().is_dir(path, base_dir)


def list_directory(path: Union[str, Path], pattern: str = "*", base_dir: Optional[Union[str, Path]] = None) -> list:
    """List files in a directory."""
    return get_storage_manager().list_files(path, pattern, base_dir)


def create_directory(path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None, 
                    parents: bool = True, exist_ok: bool = True) -> None:
    """Create a directory."""
    return get_storage_manager().mkdir(path, base_dir, parents, exist_ok)


def read_file_text(path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None, encoding: str = "utf-8") -> str:
    """Read text content from a file."""
    return get_storage_manager().read_text(path, base_dir, encoding)


def read_file_binary(path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> bytes:
    """Read binary content from a file."""
    return get_storage_manager().read_binary(path, base_dir)


def write_file_text(path: Union[str, Path], content: str, base_dir: Optional[Union[str, Path]] = None, encoding: str = "utf-8") -> None:
    """Write text content to a file."""
    return get_storage_manager().write_text(path, content, base_dir, encoding)


def write_file_binary(path: Union[str, Path], content: bytes, base_dir: Optional[Union[str, Path]] = None) -> None:
    """Write binary content to a file."""
    return get_storage_manager().write_binary(path, content, base_dir)


def delete_file(path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> None:
    """Delete a file."""
    return get_storage_manager().unlink(path, base_dir)


def upload_directory(local_dir: Union[str, Path], remote_prefix: Union[str, Path]) -> None:
    """Convenience wrapper to upload a local directory tree to current backend."""
    return get_storage_manager().upload_dir(local_dir, remote_prefix)
