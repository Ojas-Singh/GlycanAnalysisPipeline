number_of_dimensions = 20  # Number of dimensions for PCA
explained_variance = 0.8  # Explained variance cut-off
hard_dim = 3  # Hard cut-off

import os
from pathlib import Path

# Base path where the script is located
base_dir = Path(__file__).resolve().parent.parent

# Get paths from environment variables or use defaults

data_dir = Path(os.environ.get("GLYCAN_DATA_DIR", base_dir / "dummy_data_dir"))
output_path = Path(os.environ.get("DATABASE_PATH", base_dir / "dummy_database"))
inventory_path = Path(os.environ.get("GLYCAN_INVENTORY_PATH", base_dir / "GlycoShape_Inventory.csv"))

update = os.environ.get("GLYCAN_DB_UPDATE", "True").lower() in ("true", "1", "yes")


backup_dir = Path(os.environ.get("GLYCAN_BACKUP_DIR", base_dir / "backup"))

contributor_id = os.environ.get("GLYTOUCAN_CONTRIBUTOR_ID", "dummy_contributor")
api_key = os.environ.get("GLYTOUCAN_API_KEY", "dummy_api")