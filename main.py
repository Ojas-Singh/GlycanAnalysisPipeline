import numpy as np
import pandas as pd
from lib import pdb,graph,dihedral,clustering,tfindr
import config
from scipy import stats
import os

def list_directories(folder_path):
    directories = [
        d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))
    ]
    return directories

def big_calculations(name):
    f=config.data_dir+name+"/"+name+".pdb"
    pdbdata, frames = pdb.multi(f)
    df = pdb.to_DF(pdbdata)
    idx_noH=df.loc[(df['Element']!="H"),['Number']].iloc[:]['Number']-1
    pcaG= clustering.pcawithG(frames,idx_noH,config.number_of_dimensions,name)
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
    print("Directories in folder:")
    for directory in directory_list:
        isExist = os.path.exists(config.data_dir+directory+'/output')
        if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(config.data_dir+directory+'/output')
        big_calculations(directory)