import os
from pathlib import Path
from typing import Union, Optional

# Base path where the script is located
base_dir = Path(__file__).resolve().parent.parent

# Attempt to load a .env file from the project root so environment variables
# defined there are available to the rest of this module. Prefer using
# python-dotenv if it's installed; otherwise fall back to a tiny parser.
def _load_dotenv_file(dotenv_path: Path) -> bool:
    """
    Load environment variables from a .env file.

    Returns True if a file was found and processed, False otherwise.
    """
    try:
        # Try using python-dotenv if available
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(dotenv_path=str(dotenv_path))
        return True
    except Exception:
        # Fallback: parse simple KEY=VALUE lines
        if not dotenv_path.exists():
            return False

        try:
            for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                # handle optional leading export
                if line.lower().startswith("export "):
                    line = line[len("export "):].lstrip()
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                # remove surrounding quotes if present
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                # Only set if not already set in the environment
                os.environ.setdefault(key, val)
            return True
        except Exception:
            return False


# Look for a .env file at the repository root (base_dir)
try:
    dotenv_file = base_dir / ".env"
    _load_dotenv_file(dotenv_file)
except Exception:
    # Non-fatal: if loading .env fails, continue with existing environment
    pass

# Storage configuration
# Oracle Cloud Object Storage PAR URL (optional - if provided, uses cloud storage)
oracle_par_url = os.environ.get("GLYCOSHAPE_ORACLE_PAR_URL")

# Determine storage mode
use_oracle_storage = oracle_par_url is not None and oracle_par_url.strip() != ""

if use_oracle_storage:
    # Oracle Cloud Object Storage mode
    # Paths in Oracle mode are subfolders within the bucket
    data_dir = os.environ.get("GLYCOSHAPE_DATA_DIR", "data")
    process_dir = os.environ.get("GLYCOSHAPE_PROCESS_DIR", "process") 
    output_dir = os.environ.get("GLYCOSHAPE_OUTPUT_DIR", "static")
    rdf_path = os.environ.get("GLYCOSHAPE_RDF_DIR", "rdf")
    inventory_path = os.environ.get("GLYCOSHAPE_INVENTORY_PATH", "GlycoShape_Inventory.csv")
else:
    # Local file system mode (existing functionality)
    data_dir = Path(os.environ.get("GLYCOSHAPE_DATA_DIR", base_dir / "dummy_database_data"))
    process_dir = Path(os.environ.get("GLYCOSHAPE_PROCESS_DIR", base_dir / "dummy_database_process"))
    output_dir = Path(os.environ.get("GLYCOSHAPE_OUTPUT_DIR", base_dir / "dummy_database_static"))
    rdf_path = Path(os.environ.get("GLYCOSHAPE_RDF_DIR", base_dir / "GLYCOSHAPE_RDF"))
    inventory_path = Path(os.environ.get("GLYCOSHAPE_INVENTORY_PATH", base_dir / "GlycoShape_Inventory.csv"))

# Other configuration
update = os.environ.get("GLYCOSHAPE_DB_UPDATE", "True").lower() in ("true", "1", "yes")

# Process cleanup configuration
# When using Oracle storage, automatically clean up local process files after successful upload
cleanup_process_files = os.environ.get("GLYCOSHAPE_CLEANUP_PROCESS_FILES", "auto").lower()

def should_cleanup_process_files() -> bool:
    """
    Determine whether to clean up local process files after upload.
    
    Returns:
        True if cleanup should be performed, False otherwise
    """
    global cleanup_process_files, use_oracle_storage
    
    if cleanup_process_files == "auto":
        # Auto mode: clean up when using Oracle storage
        return use_oracle_storage
    else:
        # Explicit setting
        return cleanup_process_files in ("true", "1", "yes")

contributor_id = os.environ.get("GLYTOUCAN_CONTRIBUTOR_ID", "dummy_contributor_id")
api_key = os.environ.get("GLYTOUCAN_API_KEY", "dummy_glytoucan_api")


def get_storage_info() -> dict:
    """
    Get information about current storage configuration.
    
    Returns:
        Dictionary with storage mode and configuration info
    """
    return {
        "storage_mode": "oracle" if use_oracle_storage else "local",
        "oracle_par_url": oracle_par_url if use_oracle_storage else None,
        "data_dir": str(data_dir),
        "process_dir": str(process_dir), 
        "output_dir": str(output_dir),
        "rdf_path": str(rdf_path),
        "inventory_path": str(inventory_path)
    }


def initialize_storage() -> bool:
    """
    Initialize storage directories/folders.
    
    For Oracle storage, creates the required folder structure.
    For local storage, ensures directories exist.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if use_oracle_storage:
            # Oracle storage: do not attempt to create folders or files via PAR (may be read-only).
            # Prefixes are virtual; assume they exist or will be created by writes elsewhere.
            # Simply return True so status/listing can proceed.
            return True
        else:
            # Local storage - ensure directories exist
            data_dir.mkdir(parents=True, exist_ok=True)
            process_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            rdf_path.mkdir(parents=True, exist_ok=True)
            
            # Ensure inventory CSV exists (create empty if needed)
            if not inventory_path.exists():
                inventory_path.write_text("# GlycoShape Inventory\n")
        
        return True
    except Exception as e:
        print(f"Failed to initialize storage: {e}")
        return False


def get_path_type(path: Union[str, Path]) -> str:
    """
    Get the type of a path (relative string vs Path object).
    
    Args:
        path: The path to check
        
    Returns:
        'string' if path is a string, 'path' if Path object
    """
    return "string" if isinstance(path, str) else "path"


def ensure_path_format(path: Union[str, Path]) -> Union[str, Path]:
    """
    Ensure path is in the correct format for the current storage mode.
    
    For Oracle mode: returns string paths
    For local mode: returns Path objects
    
    Args:
        path: The path to normalize
        
    Returns:
        Path in appropriate format
    """
    if use_oracle_storage:
        return str(path)
    else:
        return Path(path)