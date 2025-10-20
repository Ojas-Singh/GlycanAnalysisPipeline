"""
Storage abstraction layer for local and Oracle Cloud Object Storage.

Provides unified interface for file operations with support for:
- Local file system (existing functionality)  
- Oracle Cloud Object Storage via Pre-Authenticated Request (PAR) URLs
- Fallback and compatibility between storage backends
"""

import os
import io
import json
import tempfile
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, BinaryIO, TextIO, Optional
from urllib.parse import urlparse
import requests

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
        self._verify_par_url()
        # Cache for directory structure since Oracle doesn't provide true directories
        self._dir_cache = {}
        logger.info(f"Initialized OracleStorageBackend with PAR URL: {self.par_url}")
    
    def _verify_par_url(self) -> None:
        """Verify the PAR URL is accessible."""
        try:
            response = requests.head(self.par_url, timeout=10)
            # Some PARs may not allow HEAD on the root; accept common statuses
            if response.status_code not in [200, 204, 403, 404]:
                # Try a lightweight GET listing as a fallback check
                try:
                    r2 = requests.get(self.par_url + "/", params={"limit": 1}, timeout=10)
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
            response = requests.head(url, timeout=10)
            if response.status_code == 200:
                return True
            # Some PARs disallow HEAD; fall back to GET
            if response.status_code in (400, 401, 403, 405):
                try:
                    r2 = requests.get(url, timeout=10)
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
            response = requests.get(url, params={"prefix": prefix, "limit": 1}, timeout=10)
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
        path_str = str(path).replace('\\', '/').rstrip('/')
        prefix = f"{path_str}/" if path_str else ""
        url = f"{self.par_url}/"
        try:
            response = requests.get(url, params={"prefix": prefix, "limit": 1000}, timeout=15)
            if response.status_code != 200:
                return []

            # Prefer JSON (Oracle native API)
            try:
                data = response.json()
                objs = data.get("objects") or data.get("Objects") or []
                names = []
                for obj in objs:
                    if isinstance(obj, dict):
                        name = obj.get("name") or obj.get("key") or obj.get("Key")
                        if name:
                            names.append(name)
                return [Path(n) for n in names]
            except ValueError:
                # Fallback to XML (S3-compatible API)
                try:
                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(response.text)
                    names = [el.text for el in root.findall('.//{*}Contents/{*}Key') if el.text]
                    return [Path(n) for n in names]
                except Exception:
                    return []
        except Exception as e:
            logger.error(f"Failed to list files in {path}: {e}")
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
            response = requests.put(url, data=content.encode(encoding), timeout=30)
            response.raise_for_status()
            logger.debug(f"Wrote text to {url}")
        except Exception as e:
            logger.error(f"Failed to write text to {url}: {e}")
            raise
    
    def write_binary(self, path: Union[str, Path], content: bytes) -> None:
        """Write binary content to a file."""
        url = self._get_object_url(path)
        try:
            response = requests.put(url, data=content, timeout=30)
            response.raise_for_status()
            logger.debug(f"Wrote binary to {url}")
        except Exception as e:
            logger.error(f"Failed to write binary to {url}: {e}")
            raise
    
    def read_text(self, path: Union[str, Path], encoding: str = "utf-8") -> str:
        """Read text content from a file."""
        url = self._get_object_url(path)
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.content.decode(encoding)
        except Exception as e:
            logger.error(f"Failed to read text from {url}: {e}")
            raise
    
    def read_binary(self, path: Union[str, Path]) -> bytes:
        """Read binary content from a file."""
        url = self._get_object_url(path)
        try:
            response = requests.get(url, timeout=30)
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
            response = requests.delete(url, timeout=30)
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
        self._is_oracle = isinstance(self.backend, OracleStorageBackend)

    def _is_data_prefix(self, path: Union[str, Path]) -> bool:
        p = str(path).lstrip('/')
        return p == 'data' or p.startswith('data/')
    
    def _is_process_prefix(self, path: Union[str, Path]) -> bool:
        """Check if path is a process directory path."""
        p = str(path).lstrip('/')
        return 'process' in p

    def _should_write_locally(self, path: Union[str, Path]) -> bool:
        # In Oracle mode, stage all writes locally; sync handled elsewhere
        return self._is_oracle
    
    def _detect_backend(self) -> StorageBackend:
        """Auto-detect appropriate storage backend from environment."""
        # Check if Oracle PAR URL is provided
        par_url = os.environ.get("GLYCOSHAPE_ORACLE_PAR_URL")
        if par_url:
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
        # In Oracle mode, prioritize Oracle for inventory CSV
        if self._is_oracle:
            path_str = str(path).lower()
            if 'inventory' in path_str and path_str.endswith('.csv'):
                # For inventory CSV, check Oracle first
                return self.backend.exists(path)
            # For other files, prefer local staged files first, then fall back to remote
            try:
                if self._local_backend.exists(path):
                    return True
            except Exception:
                pass
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
            if self._is_process_prefix(path_str) or self._is_data_prefix(path_str):
                return self.backend.list_files(path, pattern)
            # For other directories, prefer local listings (best effort)
            try:
                return self._local_backend.list_files(path, pattern)
            except Exception:
                return []
        return self.backend.list_files(path, pattern)
    
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
        # For Oracle mode, prioritize remote reads for process directories and inventory CSV
        if self._is_oracle:
            path_str = str(path)
            if self._is_process_prefix(path_str) or ('inventory' in path_str.lower() and path_str.lower().endswith('.csv')):
                # For process directories and inventory CSV, read from Oracle bucket first
                return self.backend.read_text(path, encoding)
            # For other directories, prefer local staged file
            elif self._local_backend.exists(path):
                return self._local_backend.read_text(path, encoding)
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
        # For Oracle mode, prioritize remote reads for process directories and inventory CSV
        if self._is_oracle:
            path_str = str(path)
            if self._is_process_prefix(path_str) or ('inventory' in path_str.lower() and path_str.lower().endswith('.csv')):
                # For process directories and inventory CSV, read from Oracle bucket first
                return self.backend.read_binary(path)
            # For other directories, prefer local staged file
            elif self._local_backend.exists(path):
                return self._local_backend.read_binary(path)
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

    def upload_dir(self, local_dir: Union[str, Path], remote_prefix: Union[str, Path]) -> None:
        """Recursively upload a local directory to the current backend under remote_prefix.

        Note: Intended for syncing local outputs to remote object storage in Oracle mode.
        """
        local_dir = Path(local_dir)
        remote_prefix = str(remote_prefix).rstrip('/')
        if not local_dir.exists():
            logger.debug(f"upload_dir: local path does not exist: {local_dir}")
            return
        success_count = 0
        fail_count = 0
        for root, _, files in os.walk(local_dir):
            for fname in files:
                lpath = Path(root) / fname
                rel = lpath.relative_to(local_dir).as_posix()
                rpath = f"{remote_prefix}/{rel}" if remote_prefix else rel
                try:
                    with open(lpath, 'rb') as f:
                        data = f.read()
                    self.backend.write_binary(rpath, data)
                    success_count += 1
                except Exception as e:
                    logger.warning(f"upload_dir: failed to upload {lpath} -> {rpath}: {e}")
                    fail_count += 1
        logger.info(f"upload_dir: uploaded {success_count} files to '{remote_prefix}' ({fail_count} failures)")

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
