import os
import shutil
import lib.config as config

def delete_output_folders(directory):
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            if dir_name == 'output':
                output_path = os.path.join(root, dir_name)
                shutil.rmtree(output_path)
                print(f"Deleted: {output_path}")
            if dir_name == 'clusters':
                output_path = os.path.join(root, dir_name)
                shutil.rmtree(output_path)
                print(f"Deleted: {output_path}")


delete_output_folders(config.data_dir)
