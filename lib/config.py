import os
from pathlib import Path

# Base path where the script is located
base_dir = Path(__file__).resolve().parent.parent

# Get paths from environment variables or use defaults

# data_dir: Read-only directory containing input glycan data (PDB, MOL2 files)
data_dir = Path(os.environ.get("GLYCOSHAPE_DATA_DIR", base_dir / "dummy_database_data"))

# process_dir: Working directory for storing all processing outputs (data, embedding, output folders)
process_dir = Path(os.environ.get("GLYCOSHAPE_PROCESS_DIR", base_dir / "dummy_database_process"))

# output_dir: Final database output directory (separate from processing)
output_dir = Path(os.environ.get("GLYCOSHAPE_OUTPUT_DIR", base_dir / "dummy_database_static"))

# rdf_path: Directory containing RDF files
rdf_path = Path(os.environ.get("GLYCOSHAPE_RDF_DIR", base_dir / "GLYCOSHAPE_RDF"))

inventory_path = Path(os.environ.get("GLYCOSHAPE_INVENTORY_PATH", base_dir / "GlycoShape_Inventory.csv"))

update = os.environ.get("GLYCOSHAPE_DB_UPDATE", "True").lower() in ("true", "1", "yes")

contributor_id = os.environ.get("GLYTOUCAN_CONTRIBUTOR_ID", "dummy_contributor_id")
api_key = os.environ.get("GLYTOUCAN_API_KEY", "dummy_glytoucan_api")