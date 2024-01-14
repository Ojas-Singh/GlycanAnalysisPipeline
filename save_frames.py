import os,shutil
from lib import flip,pdb
import config
import numpy as np

def process_folders(root_dir):
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        output_pdb_file = None
        if os.path.isdir(folder_path):
            pdb_file = os.path.join(folder_path, folder_name + ".pdb")
            if flip.is_alpha(folder_name):
                output_pdb_file = os.path.join(folder_path,  "/output/beta.pdb")
            else:
                output_pdb_file = os.path.join(folder_path,  "/output/alpha.pdb")

            if os.path.isfile(pdb_file):
                print(f"Processing {pdb_file}")
                try:
                    print(f"will try to Save flipped PDB to {output_pdb_file}")
                    flip.flip_alpha_beta_multi(pdb_file, output_pdb_file)
                    print(f"Saved flipped PDB to {output_pdb_file}")
                    if flip.is_alpha(folder_name):
                        subdirectory_path = os.path.join(folder_path, "clusters/pack")
                        output_file_path = os.path.join(subdirectory_path, "alpha.npy")
                        a,frame = pdb.multi(pdb_file)
                        np.save(output_file_path, frame)
                        output_file_path = os.path.join(subdirectory_path, "beta.npy")
                        b,frame = pdb.multi(output_pdb_file)
                        np.save(output_file_path, frame)
                    else:
                        subdirectory_path = os.path.join(folder_path, "clusters/pack")
                        output_file_path = os.path.join(subdirectory_path, "beta.npy")
                        a,frame = pdb.multi(pdb_file)
                        np.save(output_file_path, frame)
                        output_file_path = os.path.join(subdirectory_path, "alpha.npy")
                        b,frame = pdb.multi(output_pdb_file)
                        np.save(output_file_path, frame)
                    os.remove(output_pdb_file)
                except Exception as e:
                    print(f"Error processing {pdb_file}: {e}")
            else:
                print(f"No PDB file found in {folder_path}")


process_folders(config.data_dir)
