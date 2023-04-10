from lib import flip
import config
import os

def list_directories(folder_path):
    directories = [
        d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))
    ]
    return directories

if __name__ == "__main__":
    folder_path = config.data_dir
    directory_list = list_directories(folder_path)
    print("Directories in folder: ",config.data_dir)
    for directory in directory_list:
        if flip.is_alpha(str(directory)):
            isExist = os.path.exists(config.data_dir+directory+'/clusters/')
            if isExist:
                    print("Processing : ",directory)
                    k=config.data_dir+directory+'/clusters/'
                    if not os.path.exists(config.data_dir+directory+'/clusters/beta/'):
                        os.makedirs(config.data_dir+directory+'/clusters/beta/')
                    try:
                        pdb_files = [f for f in os.listdir(k) if f.endswith(".pdb")]
                        print(pdb_files)
                        for i in pdb_files:
                            flip.flip_alpha_beta(k+i,k+"beta/"+i)
                        print("success!")
                    except:
                        print("failed!")
                        pass
        else:
            print("Skipping : ",directory)