from lib import flip,pdb,clustering,align
import config
import os,sys,traceback,shutil,json
import pandas as pd



def get_number_after_underscore(filename):
                return float(filename.split("_")[1].split(".")[0])
            

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
                        os.makedirs(config.data_dir+directory+'/clusters/pack')
                        os.makedirs(config.data_dir+directory+'/clusters/alpha')
                        os.makedirs(config.data_dir+directory+'/clusters/beta')
                        with open(config.data_dir+directory+'/output/info.txt', 'r') as file:
                            lines = file.readlines()
                            exec(lines[0])
                            exec(lines[1])
                            exec(lines[2])
                            exec(lines[3])
                            exec(lines[4])
                            if flip.is_alpha(str(directory)):
                                shutil.copy(config.data_dir+directory + "/output/torparts.npz", config.data_dir+directory+'/clusters/pack/torparts.npz')
                                shutil.copy(config.data_dir+directory + "/output/PCA_variance.png", config.data_dir+directory+'/clusters/pack/PCA_variance.png')
                                shutil.copy(config.data_dir+directory + "/output/Silhouette_Score.png", config.data_dir+directory+'/clusters/pack/Silhouette_Score.png')

                                output_info = config.data_dir+directory+'/clusters/info.txt'
                                input_torsion = config.data_dir+directory+'/output/torsions.csv'
                                output_torsion = config.data_dir+directory+'/clusters/pack/torsions.csv'
                                output_pca = config.data_dir+directory+'/clusters/pack/pca.csv'
                                output_dir = config.data_dir+directory+'/clusters/alpha_temp/'
                                output_dir_final = config.data_dir+directory+'/clusters/alpha/'
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
                                sorted_popp = sorted(popp, key=lambda x: int(x[1]))
                                sorted_popp_values = [int(item[0]) for item in sorted_popp]
                                s_scores = [float(i) for i in s_scores]
                                data = {
                                    "n_clus": n_clus,
                                    "n_dim": n_dim,
                                    "popp": sorted_popp_values,
                                    "s_scores": s_scores,
                                    "flexibility": int(flexibility)
                                }
                                


                                with open(config.data_dir+directory+'/clusters/pack/info.json', "w") as json_file:
                                    json.dump(data, json_file, indent=4)

                                df = pd.read_csv(input_torsion)
                                df.insert(1,"cluster",clustering_labels,False)
                                # Remove columns containing the word "internal" from df
                                cols_to_drop = [col for col in df.columns if "internal" in col]
                                df = df.drop(columns=cols_to_drop)
                                df.to_csv(output_torsion,index=False)
                                pca_df = pca_df[["0", "1", "2", "i", "cluster"]]
                                pca_df.to_csv(output_pca,index=False)
                                # Saving the n_dim and n_clus to output/info.txt

                                filenames = os.listdir(output_dir)
                                pdb_files = [filename for filename in filenames if filename.endswith(".pdb")]
                                files_to_align =[output_dir+x for x in pdb_files]
                                sorted_pdb_files = sorted(pdb_files, key=get_number_after_underscore,reverse=True)
                                reference_file = output_dir+sorted_pdb_files[0]
                                output_files = [output_dir_final+ x for x in sorted_pdb_files]  
                                align.align_pdb_files(reference_file, files_to_align, output_files)
                                shutil.rmtree(output_dir)

                                with open(output_info, 'w') as file:
                                    file.write(f"n_clus = {n_clus}\n")
                                    file.write(f"n_dim = {n_dim}\n")
                                    file.write(f"popp = {list(popp)}\n")
                                try:
                                    pdb_files = [f for f in os.listdir(output_dir_final) if f.endswith(".pdb")]
                                    print(pdb_files)
                                    for i in pdb_files:
                                        flip.flip_alpha_beta(output_dir_final+i,output_dir_flip+i)
                                    print("success!")
                                except:
                                    print("failed!")
                                    pass
                            else: 
                                shutil.copy(config.data_dir+directory + "/output/torparts.npz", config.data_dir+directory+'/clusters/pack/torparts.npz')
                                shutil.copy(config.data_dir+directory + "/output/PCA_variance.png", config.data_dir+directory+'/clusters/pack/PCA_variance.png')
                                shutil.copy(config.data_dir+directory + "/output/Silhouette_Score.png", config.data_dir+directory+'/clusters/pack/Silhouette_Score.png')

                                output_info = config.data_dir+directory+'/clusters/info.txt'
                                input_torsion = config.data_dir+directory+'/output/torsions.csv'
                                output_torsion = config.data_dir+directory+'/clusters/pack/torsions.csv'
                                output_pca = config.data_dir+directory+'/clusters/pack/pca.csv'
                                output_dir = config.data_dir+directory+'/clusters/beta_temp/'
                                output_dir_final = config.data_dir+directory+'/clusters/beta/'
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

                                sorted_popp = sorted(popp, key=lambda x: int(x[1]))
                                sorted_popp_values = [item[0] for item in sorted_popp]
                                s_scores = [str(i) for i in s_scores]
                                data = {
                                    "n_clus": n_clus,
                                    "n_dim": n_dim,
                                    "popp": sorted_popp_values,
                                    "s_scores": s_scores,
                                    "flexibility": flexibility
                                }
                                

                                with open(config.data_dir+directory+'/clusters/pack/info.json', "w") as json_file:
                                    json.dump(data, json_file, indent=4)


                                df = pd.read_csv(input_torsion)
                                df.insert(1,"cluster",clustering_labels,False)
                                # Remove columns containing the word "internal" from df
                                cols_to_drop = [col for col in df.columns if "internal" in col]
                                df = df.drop(columns=cols_to_drop)
                                df.to_csv(output_torsion,index=False)
                                pca_df = pca_df[["0", "1", "2", "i", "cluster"]]
                                pca_df.to_csv(output_pca,index=False)
                                # Saving the n_dim and n_clus to output/info.txt

                                
                                filenames = os.listdir(output_dir)
                                pdb_files = [filename for filename in filenames if filename.endswith(".pdb")]
                                files_to_align =[output_dir+x for x in pdb_files]
                                sorted_pdb_files = sorted(pdb_files, key=get_number_after_underscore,reverse=True)
                                reference_file = output_dir+sorted_pdb_files[0]
                                output_files = [output_dir_final+ x for x in sorted_pdb_files]  
                                align.align_pdb_files(reference_file, files_to_align, output_files)
                                shutil.rmtree(output_dir)

                                with open(output_info, 'w') as file:
                                    file.write(f"n_clus = {n_clus}\n")
                                    file.write(f"n_dim = {n_dim}\n")
                                    file.write(f"popp = {list(popp)}\n")
                                try:
                                    pdb_files = [f for f in os.listdir(output_dir_final) if f.endswith(".pdb")]
                                    print(pdb_files)
                                    for i in pdb_files:
                                        flip.flip_alpha_beta(output_dir_final+i,output_dir_flip+i)
                                    print("success!")
                                except:
                                    print("failed!")
                                    pass
                            
            else:
                print("Skipping : ",directory)
        except Exception:
            traceback.print_exc()
            pass