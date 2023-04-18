import numpy as np
import pandas as pd
from lib import pdb,graph,dihedral,clustering,tfindr
import config
import os,sys,traceback

def extract_first_frame(input_pdb, output_pdb):
    with open(input_pdb, 'r') as infile, open(output_pdb, 'w') as outfile:
        model_started = False
        for line in infile:
            if line.startswith('MODEL'):
                if not model_started:
                    model_started = True
                else:
                    break
            if model_started:
                outfile.write(line)
                
        if not model_started:
            print(f"No model found in {input_pdb}.")
            return
    print(f"First frame extracted and saved as {output_pdb}.")

def list_directories(folder_path):
    directories = [
        d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))
    ]
    return directories

def big_calculations(name):
    # Set the input and output file paths
    input_file = config.data_dir + name + "/" + name + ".pdb"
    output_structure = config.data_dir + name + "/output/structure.pdb"
    output_pca = config.data_dir + name + "/output/pca.csv"
    output_torsions = config.data_dir + name + "/output/torsions.csv"
    output_info = config.data_dir + name + "/output/info.txt"
    output_cluster_folder = config.data_dir + name + "/output/cluster_default/"

    # Extract the first frame from the input file and save it to the output file
    extract_first_frame(input_file, output_structure)

    # Load the data and convert it to DataFrame format
    pdb_data, frames = pdb.multi(input_file)
    df = pdb.to_DF(pdb_data)

    # Filter out hydrogen atoms and get their indices
    idx_noH = df.loc[df['Element'] != "H", 'Number'] - 1

    # Perform PCA with Gaussian clustering and plot the Silhouette score, plot_Silhouette function also gives n_clus (best number of clusters)
    pcaG, n_dim = clustering.pcawithG(frames, idx_noH, config.number_of_dimensions, name)
    n_clus = clustering.plot_Silhouette(pcaG, name, n_dim)

    # Saving the n_dim and n_clus to output/info.txt
    with open(output_info, 'w') as file:
        file.write(f"n_clus = {n_clus}\n")
        file.write(f"n_dim = {n_dim}\n")

    # Save the PCA data to a CSV file
    pcaG.to_csv(output_pca, index_label="i")

    # clustering 
    pca_df = pd.read_csv(output_pca)
    selected_columns = [str(i) for i in range(1, n_dim+1)]
    clustering_labels,pp = clustering.best_clustering(n_clus,pca_df[selected_columns])
    pca_df.insert(1,"cluster",clustering_labels,False)
    pca_df["cluster"] = pca_df["cluster"].astype(str)
    popp = clustering.kde_c(n_clus,pca_df,selected_columns) 
    pdb.exportframeidPDB(input_file,popp,output_cluster_folder)

    # Compute torsion pairs for the protein structure
    pairs, external, internal = tfindr.torsionspairs(pdb_data, name)
    pairs = np.asarray(pairs)

    # Convert pairs to torsion names
    torsion_names = dihedral.pairtoname(external, df)
    ext_DF = dihedral.pairstotorsion(external, frames, torsion_names)

    # Add "internal" to torsion_names for internal torsions
    for _ in range(len(internal)):
        torsion_names.append("internal")

    # Calculate torsion data and save it to a DataFrame
    torsion_data_DF = dihedral.pairstotorsion(pairs, frames, torsion_names)

    # Save the torsion data to a CSV file
    torsion_data_DF.to_csv(output_torsions, index_label="i")


if __name__ == "__main__":
    folder_path = config.data_dir
    directory_list = list_directories(folder_path)
    print("Directories in folder: ",config.data_dir)
    for directory in directory_list:
        isExist = os.path.exists(config.data_dir+directory+'/output')
        if not isExist:
                print("Processing : ",directory)
                os.makedirs(config.data_dir+directory+'/output')
                try:
                    big_calculations(directory)
                    print("success!")
                except Exception:
                    traceback.print_exc()
                    print("failed!")
                    pass
        else:
            print("Skipping : ",directory)