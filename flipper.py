import numpy as np
import pandas as pd
from lib import pdb,graph,dihedral,clustering,tfindr
import config
import os

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
    f=config.data_dir+name+"/"+name+".pdb"
    extract_first_frame(f,config.data_dir+name+"/output/structure.pdb")
    pdbdata, frames = pdb.multi(f)
    df = pdb.to_DF(pdbdata)
    idx_noH=df.loc[(df['Element']!="H"),['Number']].iloc[:]['Number']-1
    pcaG= clustering.pcawithG(frames,idx_noH,config.number_of_dimensions,name)
    clustering.plot_Silhouette(pcaG,name)
    pcaG.to_csv(config.data_dir+name+"/output/pca.csv",index_label="i")

    pairs,external,internal = tfindr.torsionspairs(pdbdata,name)
    pairs = np.asarray(pairs)

    torsion_names = dihedral.pairtoname(external,df)
    ext_DF = dihedral.pairstotorsion(external,frames,torsion_names)
    for i in range(len(internal)):
        torsion_names.append("internal")

    torsiondataDF= dihedral.pairstotorsion(pairs,frames,torsion_names)
    torsiondataDF.to_csv(config.data_dir+name+"/output/torsions.csv",index_label="i")


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
                except:
                    print("failed!")
                    pass
        else:
            print("Skipping : ",directory)