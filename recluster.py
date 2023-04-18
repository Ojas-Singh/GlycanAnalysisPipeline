from lib import flip,pdb,clustering
import config
import os,sys,traceback
import pandas as pd

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
        isExist = os.path.exists(config.data_dir+directory+'/clusters/')
        try:    
            if not isExist:
                        print("Processing : ",directory)
                        os.makedirs(config.data_dir+directory+'/clusters/')
                        os.makedirs(config.data_dir+directory+'/clusters/alpha')
                        os.makedirs(config.data_dir+directory+'/clusters/beta')
                        with open(config.data_dir+directory+'/output/info.txt', 'r') as file:
                            lines = file.readlines()
                            exec(lines[0])
                            exec(lines[1])
                            if flip.is_alpha(str(directory)):
                                output_dir = config.data_dir+directory+'/clusters/alpha/'
                                output_dir_flip = config.data_dir+directory+'/clusters/beta/'
                                pca = config.data_dir+directory + "/output/pca.csv"
                                input_file = config.data_dir + directory + "/" + directory + ".pdb"
                                pca_df = pd.read_csv(pca)
                                selected_columns = [str(i) for i in range(1, n_dim+1)]
                                clustering_labels,pp = clustering.best_clustering(n_clus,pca_df[selected_columns])
                                pca_df.insert(1,"cluster",clustering_labels,False)
                                pca_df["cluster"] = pca_df["cluster"].astype(str)
                                popp = clustering.kde_c(n_clus,pca_df,selected_columns) 
                                pdb.exportframeidPDB(input_file,popp,output_dir)
                                try:
                                    pdb_files = [f for f in os.listdir(output_dir) if f.endswith(".pdb")]
                                    print(pdb_files)
                                    for i in pdb_files:
                                        flip.flip_alpha_beta(output_dir+i,output_dir_flip+i)
                                    print("success!")
                                except:
                                    print("failed!")
                                    pass
                            else: 
                                output_dir = config.data_dir+directory+'/clusters/beta/'
                                output_dir_flip = config.data_dir+directory+'/clusters/alpha/'
                                pca = config.data_dir+directory + "/output/pca.csv"
                                input_file = config.data_dir + directory + "/" + directory + ".pdb"
                                pca_df = pd.read_csv(pca)
                                selected_columns = [str(i) for i in range(1, n_dim+1)]
                                clustering_labels,pp = clustering.best_clustering(n_clus,pca_df[selected_columns])
                                pca_df.insert(1,"cluster",clustering_labels,False)
                                pca_df["cluster"] = pca_df["cluster"].astype(str)
                                popp = clustering.kde_c(n_clus,pca_df,selected_columns) 
                                pdb.exportframeidPDB(input_file,popp,output_dir)
                                try:
                                    pdb_files = [f for f in os.listdir(output_dir) if f.endswith(".pdb")]
                                    print(pdb_files)
                                    for i in pdb_files:
                                        flip.flip_alpha_beta(output_dir+i,output_dir_flip+i)
                                    print("success!")
                                except:
                                    print("failed!")
                                    pass
                            
            else:
                print("Skipping : ",directory)
        except Exception:
            traceback.print_exc()
            pass