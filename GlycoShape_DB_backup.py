from pathlib import Path
import json
import logging
import hashlib
from datetime import datetime
import zipfile
from concurrent.futures import ThreadPoolExecutor
import xxhash  # Faster hashing algorithm
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_file_hash(file_path: Path) -> str:
    """Calculate fast hash of file using xxHash."""
    hash_obj = xxhash.xxh3_64()  # Use XXH3_64bits
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            hash_obj.update(byte_block)
    return hash_obj.hexdigest()

def get_last_backup_info(backup_dir: Path) -> dict:
    """Get information about last backup."""
    info_file = backup_dir / "backup_info.json"
    if info_file.exists():
        return json.loads(info_file.read_text())
    return {
        "directories": {
            "input": {"last_backup": None, "file_hashes": {}},
            "output": {"last_backup": None, "file_hashes": {}}
        }
    }

def save_backup_info(backup_dir: Path, info: dict) -> None:
    """Save backup information."""
    info_file = backup_dir / "backup_info.json"
    info_file.write_text(json.dumps(info, indent=2))

def compute_file_hash(file_path: Path):
    """Compute hash and modification time for a single file."""
    current_hash = calculate_file_hash(file_path)
    mtime = file_path.stat().st_mtime
    return str(file_path), {"hash": current_hash, "mtime": mtime}

def get_modified_files(directory: Path, dir_info: dict) -> list:
    """Get list of modified files based on hash or modification time."""
    modified_files = []
    current_hashes = {}
    file_paths = [file_path for file_path in directory.rglob("*") if file_path.is_file()]

    with ThreadPoolExecutor() as executor:
        results = executor.map(compute_file_hash, file_paths)
        for abs_path, file_info in results:
            rel_path = str(Path(abs_path).relative_to(directory))
            current_hashes[rel_path] = file_info
            
            # Add to modified files if hash or mtime differ
            if rel_path not in dir_info["file_hashes"] or \
               dir_info["file_hashes"][rel_path] != file_info:
                modified_files.append((Path(abs_path), file_info["hash"]))

    return modified_files, current_hashes

def create_incremental_backup(input_dir: Path, backup_dir: Path, dir_type: str) -> Path:
    """Create incremental backup of directory.
    
    Args:
        input_dir: Directory to backup
        backup_dir: Directory to store backups
        dir_type: Type of directory ('input' or 'output')
        
    Returns:
        Path to created backup file
    """
    input_dir = Path(input_dir)
    backup_dir = Path(backup_dir)
    
    # Create backup directory
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Get last backup info
    backup_info = get_last_backup_info(backup_dir)
    dir_info = backup_info["directories"][dir_type]
    
    # Get modified files
    modified_files, current_hashes = get_modified_files(input_dir, dir_info)
    
    if not modified_files:
        logger.info(f"No files modified in {dir_type} directory since last backup")
        return None
        
    # Create backup filename
    timestamp = datetime.now()
    backup_name = f'{dir_type}_backup_{timestamp.strftime("%Y%m%d_%H%M%S")}.zip'
    backup_path = backup_dir / backup_name
    
    # Create zip with modified files
    with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path, _ in modified_files:
            arcname = file_path.relative_to(input_dir)
            zf.write(file_path, arcname)
            logger.info(f"Backed up: {arcname}")
    
    # Update backup info
    dir_info["last_backup"] = timestamp.timestamp()
    dir_info["file_hashes"] = current_hashes
    save_backup_info(backup_dir, backup_info)
    
    logger.info(f"Created incremental backup for {dir_type}: {backup_path}")
    return backup_path

def backup_all_directories():
    """Backup both input and output directories."""
    for dir_type, dir_path in [
        ("input", config.input_path),
        ("output", config.output_path)
    ]:
        try:
            backup_path = create_incremental_backup(
                Path(dir_path),
                Path(config.backup_dir),
                dir_type
            )
            if backup_path:
                logger.info(f"{dir_type.title()} backup created at: {backup_path}")
        except Exception as e:
            logger.error(f"Failed to backup {dir_type} directory: {str(e)}")

if __name__ == "__main__":
    backup_all_directories()
