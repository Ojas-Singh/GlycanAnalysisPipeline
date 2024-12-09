from pathlib import Path

# Base path where the script is located
base_dir = Path(__file__).resolve().parent

data_dir = base_dir / "dummy_data_dir"
number_of_dimensions = 20  # Number of dimensions for PCA
explained_variance = 0.8  # Explained variance cut-off
hard_dim = 3  # Hard cut-off

input_path = data_dir
output_path = base_dir / "dummy_database"
inventory_path = base_dir / "GlycoShape_Inventory.csv"
sugarbase_path = base_dir / "v9_sugarbase.csv"
update = True

backup_dir = base_dir / "backup"

contributor_id = "dummy_contributor"
api_key = "dummy_api"