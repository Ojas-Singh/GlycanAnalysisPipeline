import os, shutil
from lib import flip, pdb
import config
import numpy as np

def process_folders(root_dir):
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            pdb_file = os.path.join(folder_path, folder_name + ".pdb")
            subdirectory_path = os.path.join(folder_path, "clusters/pack")
            alpha_file_path = os.path.join(subdirectory_path, "alpha.npy")
            beta_file_path = os.path.join(subdirectory_path, "beta.npy")

            if os.path.isfile(pdb_file):
                print(f"Processing {pdb_file}")
                try:
                    if os.path.isfile(alpha_file_path) and os.path.isfile(beta_file_path):
                        print(f"Skipping {folder_name} as alpha.npy and beta.npy already exist.")
                        continue

                    output_pdb_file = None
                    if flip.is_alpha(folder_name):
                        output_pdb_file = os.path.join(folder_path, "output/beta.pdb")
                    else:
                        output_pdb_file = os.path.join(folder_path, "output/alpha.pdb")

                    print(f"Will try to save flipped PDB to {output_pdb_file}")
                    flip.flip_alpha_beta_multi(pdb_file, output_pdb_file)
                    print(f"Saved flipped PDB to {output_pdb_file}")

                    # Ensure the subdirectory exists
                    os.makedirs(subdirectory_path, exist_ok=True)

                    if flip.is_alpha(folder_name):
                        output_file_path = alpha_file_path
                        a, frame = pdb.multi(pdb_file)
                        np.save(output_file_path, frame)

                        output_file_path = beta_file_path
                        b, frame = pdb.multi(output_pdb_file)
                        np.save(output_file_path, frame)
                    else:
                        output_file_path = beta_file_path
                        a, frame = pdb.multi(pdb_file)
                        np.save(output_file_path, frame)

                        output_file_path = alpha_file_path
                        b, frame = pdb.multi(output_pdb_file)
                        np.save(output_file_path, frame)

                    print(f"Saved flipped .npy to /clusters/pack/alpha.npy and /clusters/pack/beta.npy")
                    os.remove(output_pdb_file)
                    print(f"Removed {output_pdb_file}")
                except Exception as e:
                    print(f"Error processing {pdb_file}: {e}")
            else:
                print(f"No PDB file found in {folder_path}")



process_folders(config.data_dir)
