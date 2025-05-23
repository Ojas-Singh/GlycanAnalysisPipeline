#PDB Format from https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html
import numpy as np
import pandas as pd
import os
import shutil
import re
import glob
from collections import defaultdict
import lib.config as config


def to_DF(pdbddata):
    df = pd.DataFrame(data=pdbddata)
    df = df.transpose()
    df.columns = ['Number','Name','ResName','Chain','ResId','X','Y','Z','Element']
    return df

def to_normal(df):
    Number = df['Number'].tolist()
    Name = df['Name'].tolist()
    ResName = df['ResName'].tolist()
    Chain = df['Chain'].tolist()
    ResId = df['ResId'].tolist()
    X = df['X'].tolist()
    Y = df['Y'].tolist()
    Z = df['Z'].tolist()
    Element = df['Element'].tolist()
    pdbdata=[Number,Name,ResName,Chain,ResId,X,Y,Z,Element]
    return pdbdata

def parse(f):
    Number = []
    Name = []
    ResName = []
    Chain = []
    ResId = []
    X = []
    Y = []
    Z = []
    Element = []
    pdbdata=[Number,Name,ResName,Chain,ResId,X,Y,Z,Element]
    with open(f, 'r') as f:
            lines = f.readlines()
            i=1
            for line in lines:
                if line.startswith("ATOM"):
                    pdbdata[0].append(int((line[7:11]).strip(" ")))
                    pdbdata[1].append((line[12:16]).strip(" "))
                    pdbdata[2].append((line[17:20]).strip(" "))
                    pdbdata[3].append((line[20:22]).strip(" "))
                    pdbdata[4].append(int((line[22:26]).strip(" ")))
                    pdbdata[5].append(float(line[31:38]))
                    pdbdata[6].append(float(line[39:46]))
                    pdbdata[7].append(float(line[47:54]))
                    pdbdata[8].append((line[76:78]).strip(" "))
                    i+=1
                if  line.startswith("END"):
                    break
            o = len(pdbdata[0])
    return pdbdata


def exportPDB(fout,pdbdata):
    fn= open(fout,"w+")
    k=""
    for i in range(len(pdbdata[0])):
        line=list("ATOM".ljust(80))
        line[6:10] = str(pdbdata[0][i]).rjust(5) 
        line[12:15] = str(pdbdata[1][i]).ljust(4) 
        line[17:19] = str(pdbdata[2][i]).rjust(3) 
        line[20:21] = str(pdbdata[3][i]).rjust(2) 
        line[22:25] = str(pdbdata[4][i]).rjust(4) 
        line[30:37] = str('{:0.3f}'.format(pdbdata[5][i])).rjust(8) 
        line[38:45] = str('{:0.3f}'.format(pdbdata[6][i])).rjust(8) 
        line[46:53] = str('{:0.3f}'.format(pdbdata[7][i])).rjust(8) 
        line[75:77] = str(pdbdata[8][i]).rjust(3) 
        line= ''.join(line)
        fn.write(line+"\n")
        k=k+line+"\n"
    return k
                
def multi(f):
    frames=[]
    pdbdata = parse(f)
    with open(f, 'r') as f:
            lines = f.readlines()
            mat = np.zeros((len(pdbdata[0]),3))
            j=1
            i=0
            for line in lines:
                if line.startswith("ATOM"):
                    mat[i,0]=float(line[30:38])
                    mat[i,1]=float(line[38:46])
                    mat[i,2]=float(line[46:54])
                    i+=1
                if line.startswith("ENDMDL"):
                    j+=1
                    i=0
                    frames.append(mat)
                    mat = np.zeros((len(pdbdata[0]),3))

    return pdbdata,frames



def exportPDBmulti(fout,pdbdata,id):
    fn= open(fout,"a")
    k=""
    fn.write("MODEL "+str(id)+"\n")
    for i in range(len(pdbdata[0])):
        line=list("ATOM".ljust(80))
        line[6:10] = str(pdbdata[0][i]).rjust(5) 
        line[12:15] = str(pdbdata[1][i]).ljust(4) 
        line[17:19] = str(pdbdata[2][i]).rjust(3) 
        line[20:21] = str(pdbdata[3][i]).rjust(2) 
        line[22:25] = str(pdbdata[4][i]).rjust(4) 
        line[30:37] = str('{:0.3f}'.format(pdbdata[5][i])).rjust(8) 
        line[38:45] = str('{:0.3f}'.format(pdbdata[6][i])).rjust(8) 
        line[46:53] = str('{:0.3f}'.format(pdbdata[7][i])).rjust(8) 
        line[75:77] = str(pdbdata[8][i]).rjust(3) 
        line= ''.join(line)
        fn.write(line+"\n")
        k=k+line+"\n"
    fn.write("ENDMDL\n")
    return k

def exportframeidPDB(f,framesid,output_folder):
    isExist = os.path.exists(output_folder)
    if not isExist:
            os.makedirs(output_folder)
    frames=[]
    framesid.sort()
    for i in framesid:
        frames.append([])
    with open(f, 'r') as f:
            lines = f.readlines()
            k=0
            pp=False
            i=1
            for line in lines:
                if line.startswith("MODEL") :
                    if framesid[k][0]==i and k<len(framesid):
                        pp=True
                    i+=1
                if pp:
                    frames[k].append(line)
                if pp== True and line.startswith("ENDMDL"):
                    pp=False
                    k+=1
                if k == len(framesid):
                    break
            for i in range(len(framesid)):
                fn= open(output_folder+str(framesid[i][1])+"_"+str("{:.2f}".format(framesid[i][2]))+".pdb","w+")
                fn.write("# Cluster : "+str(i)+" Size : "+str("{:.2f}".format(framesid[i][2]))+"\n")
                for line in frames[i]:
                    fn.write(line)
                fn.close()
                

# Function for adding remark line...
def pdb_remark_adder(filename):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write("REMARK    GENERATED BY GlycanAnalysisPipeline from GlycoShape     \n")
        f.write("REMARK        ____ _                ____  _                       \n")
        f.write("REMARK       / ___| |_   _  ___ ___/ ___|| |__   __ _ _ __   ___  \n")
        f.write("REMARK      | |  _| | | | |/ __/ _ \___ \| '_ \ / _` | '_ \ / _ \ \n")
        f.write("REMARK      | |_| | | |_| | (_| (_) |__) | | | | (_| | |_) |  __/ \n")
        f.write("REMARK       \____|_|\__, |\___\___/____/|_| |_|\__,_| .__/ \___| \n")
        f.write("REMARK               |___/                           |_|          \n")
        f.write("REMARK                   https://GlycoShape.org                   \n")
        f.write("REMARK   Cite:   Restoring protein glycosylation with GlycoShape.\n")
        f.write("REMARK   Nat Methods 21, 2117–2127 (2024).  https://doi.org/10.1038/s41592-024-02464-7 \n")
        f.write("REMARK   Callum M. Ives* and Ojas Singh*, Silvia D’Andrea, Carl A. Fogarty, \n")
        f.write("REMARK   Aoife M. Harbison, Akash Satheesan, Beatrice Tropea, Elisa Fadda\n")
        f.write("REMARK   Data available under CC BY-NC-ND 4.0 for academic use only.\n")
        f.write("REMARK   Contact elisa.fadda@soton.ac.uk for commercial licence.\n")
        f.write(content)



def convert_pdbs(ID, bonded_atoms):
    """
    Converts the PDB files in the output directory to the following formats:
    - GLYCAM format (ATOM and HETATM)
    - PDB format (ATOM and HETATM)
    - CHARMM format (ATOM and HETATM)
    """
    directory = f"{config.output_path}/{ID}/"
    
    os.chdir(directory)
    if os.path.exists("GLYCAM_format_ATOM"):
        shutil.rmtree("GLYCAM_format_ATOM")
        shutil.rmtree("PDB_format_ATOM")
        shutil.rmtree("CHARMM_format_ATOM")
        shutil.rmtree("GLYCAM_format_HETATM")
        shutil.rmtree("PDB_format_HETATM")
        shutil.rmtree("CHARMM_format_HETATM")
    os.mkdir("GLYCAM_format_ATOM")
    os.mkdir("PDB_format_ATOM")
    os.mkdir("CHARMM_format_ATOM")
    os.mkdir("GLYCAM_format_HETATM")
    os.mkdir("PDB_format_HETATM")
    os.mkdir("CHARMM_format_HETATM")

    pdb_files = glob.glob("*pdb")
    for pdb in pdb_files:

        add_conect_cards(pdb, bonded_atoms)  

        # Tidied GLYCAM name..
        try:
            with open(pdb, 'r') as file:
                filedata = file.read()
            filedata = filedata.replace("ATOM  ", "HETATM")
            with open(f"GLYCAM_format_HETATM/{pdb.split('.')[0]}.GLYCAM.pdb", 'w') as file:
                file.write(filedata)
        except OSError:
            print(OSError)
            pass

        # Tidied PDB name..
        with open(pdb, 'r') as file:
            filedata = file.read()
        filedata = filedata.replace("ATOM  ", "HETATM")
        filedata = re.sub("\s\wYA", " NDG", filedata) # GlcNAc alpha
        filedata = re.sub("\s\wYB", " NAG", filedata) # GlcNAc beta
        filedata = re.sub("\s\wVA", " A2G", filedata) # GalNAc alpha
        filedata = re.sub("\s\wVB", " NGA", filedata) # GalNAc beta
        filedata = re.sub("\s\wGA", " GLC", filedata) # Glc alpha
        filedata = re.sub("\s\wGB", " BGC", filedata) # Glc beta
        filedata = re.sub("\s\wGL", " NGC", filedata) # Neu5Gc alpha
        filedata = re.sub("\s\wLA", " GLA", filedata) # Gal alpha
        filedata = re.sub("\s\wLB", " GAL", filedata) # Gal beta
        filedata = re.sub("\s\wfA", " FUC", filedata) # L-Fuc alpha
        filedata = re.sub("\s\wfB", " FUL", filedata) # L-Fuc beta
        filedata = re.sub("\s\wMB", " BMA", filedata) # Man beta
        filedata = re.sub("\s\wMA", " MAN", filedata) # Man alpha
        filedata = re.sub("\s\wSA", " SIA", filedata) # Neu5Ac alpha
        filedata = re.sub("\s\wSA", " SLB", filedata) # Neu5Ac beta
        filedata = re.sub("\s\wZA", " GCU", filedata) # GlcA alpha
        filedata = re.sub("\s\wZB", " BDP", filedata) # GlcA beta
        filedata = re.sub("\s\wXA", " XYS", filedata) # Xyl alpha
        filedata = re.sub("\s\wXB", " XYP", filedata) # Xyl beta
        filedata = re.sub("\s\wuA", " IDR", filedata) # IdoA alpha
        filedata = re.sub("\s\whA", " RAM", filedata) # Rha alpha
        filedata = re.sub("\s\whB", " RHM", filedata) # Rha beta
        filedata = re.sub("\s\wRA", " RIB", filedata) # Rib alpha
        filedata = re.sub("\s\wRB", " BDR", filedata) # Rib beta
        filedata = re.sub("\s\wAA", " ARA", filedata) # Ara alpha
        filedata = re.sub("\s\wAB", " ARB", filedata) # Ara beta
        try:
            with open(f"PDB_format_HETATM/{pdb.split('.')[0]}.PDB.pdb", 'w') as file:
                file.write(filedata)
        except OSError:
            print(OSError)
            pass

        # Tidied CHARMM name..
        with open(pdb, 'r') as file:
            filedata = file.read()
        filedata = filedata.replace("ATOM  ", "HETATM")
        filedata = re.sub("\s\wYA ", " AGLC", filedata) # GlcNAc alpha
        filedata = re.sub("\s\wYB ", " BGLC", filedata) # GlcNAc beta
        filedata = re.sub("\s\wVA ", " AGAL", filedata) # GalNAc alpha
        filedata = re.sub("\s\wVB ", " BGAL", filedata) # GalNAc beta
        filedata = re.sub("\s\wGA ", " AGLC", filedata) # Glc alpha
        filedata = re.sub("\s\wGB ", " BGLC", filedata) # Glc beta
        filedata = re.sub("\s\wLA ", " AGAL", filedata) # Gal alpha
        filedata = re.sub("\s\wLB ", " BGAL", filedata) # Gal beta
        filedata = re.sub("\s\wf[A|B]", " FUC", filedata) # Fuc alpha and beta
        filedata = re.sub("\s\wMA ", " AMAN", filedata) # Man alpha
        filedata = re.sub("\s\wMB ", " BMAN", filedata) # Man beta
        filedata = re.sub("\s\wSA ", " ANE5", filedata) # Neu5Ac alpha
        filedata = re.sub("\s\wGL ", " ANE5", filedata) # Neu5Gc 
        filedata = re.sub("\s\wXA ", " AXYL", filedata) # Xyl alpha
        filedata = re.sub("\s\wXB ", " BXYL", filedata) # Xyl beta
        filedata = re.sub("\s\wuA ", " AIDO", filedata) # IdoA alpha
        filedata = re.sub("\s\wZA ", " AGLC", filedata) # GlcA alpha
        filedata = re.sub("\s\wZB ", " BGLC", filedata) # GlcA beta
        filedata = re.sub("\s\whA ", " ARHM", filedata) # Rha alpha
        filedata = re.sub("\s\whB ", " BRHM", filedata) # Rha beta
        filedata = re.sub("\s\wAA ", " AARB", filedata) # Ara alpha
        filedata = re.sub("\s\wAB ", " BARB", filedata) # Ara beta
        filedata = re.sub("\s\wRA ", " ARIB", filedata) # Rib alpha
        filedata = re.sub("\s\wRB ", " BRIB", filedata) # Rib beta
        try:
            with open(f"CHARMM_format_HETATM/{pdb.split('.')[0]}.CHARMM.pdb", 'w') as file:
                file.write(filedata)
        except OSError:
            print(OSError)
            pass

        with open("CHARMM_format_HETATM/README.txt", "w") as file:
            file.write("Warning:\n\nSome Glycan residues in the CHARMM naming format have residue names longer than the maximum four characters that are permitted in the PDB format. Therefore, it can be difficult to differentiate between similar residues (i.e. Glc and GlcNAc) on their residue name alone.")

        # Tidied GLYCAM name..
        with open(pdb, 'r') as file:
            filedata = file.read()
        with open(f"GLYCAM_format_ATOM/{pdb.split('.')[0]}.GLYCAM.pdb", 'w') as file:
            file.write(filedata)

        # Tidied PDB name..
        with open(pdb, 'r') as file:
            filedata = file.read()
        filedata = re.sub("\s\wYA", " NDG", filedata) # GlcNAc alpha
        filedata = re.sub("\s\wYB", " NAG", filedata) # GlcNAc beta
        filedata = re.sub("\s\wGA", " GLC", filedata) # Glc alpha
        filedata = re.sub("\s\wGB", " BGC", filedata) # Glc beta
        filedata = re.sub("\s\wVA", " A2G", filedata) # GalNAc alpha
        filedata = re.sub("\s\wVB", " NGA", filedata) # GalNAc beta
        filedata = re.sub("\s\wGL", " NGC", filedata) # Neu5Gc alpha
        filedata = re.sub("\s\wLA", " GLA", filedata) # Gal alpha
        filedata = re.sub("\s\wLB", " GAL", filedata) # Gal beta
        filedata = re.sub("\s\wfA", " FUC", filedata) # L-Fuc alpha
        filedata = re.sub("\s\wfB", " FUL", filedata) # L-Fuc beta
        filedata = re.sub("\s\wMB", " BMA", filedata) # Man beta
        filedata = re.sub("\s\wMA", " MAN", filedata) # Man alpha
        filedata = re.sub("\s\wSA", " SIA", filedata) # Neu5Ac alpha
        filedata = re.sub("\s\wSA", " SLB", filedata) # Neu5Ac beta
        filedata = re.sub("\s\wZA", " GCU", filedata) # GlcA alpha
        filedata = re.sub("\s\wZB", " BDP", filedata) # GlcA beta
        filedata = re.sub("\s\wXA", " XYS", filedata) # Xyl alpha
        filedata = re.sub("\s\wXB", " XYP", filedata) # Xyl beta
        filedata = re.sub("\s\wuA", " IDR", filedata) # IdoA alpha
        filedata = re.sub("\s\whA", " RAM", filedata) # Rha alpha
        filedata = re.sub("\s\whB", " RHM", filedata) # Rha beta
        filedata = re.sub("\s\wRA", " RIB", filedata) # Rib alpha
        filedata = re.sub("\s\wRB", " BDR", filedata) # Rib beta
        filedata = re.sub("\s\wAA", " ARA", filedata) # Ara alpha
        filedata = re.sub("\s\wAB", " ARB", filedata) # Ara beta
        try:
            with open(f"PDB_format_ATOM/{pdb.split('.')[0]}.PDB.pdb", 'w') as file:
                file.write(filedata)
        except OSError:
            print(OSError)
            pass

        # Tidied CHARMM name..
        with open(pdb, 'r') as file:
            filedata = file.read()
        filedata = re.sub("\s\wYA ", " AGLC", filedata) # GlcNAc alpha
        filedata = re.sub("\s\wYB ", " BGLC", filedata) # GlcNAc beta
        filedata = re.sub("\s\wVA ", " AGAL", filedata) # GalNAc alpha
        filedata = re.sub("\s\wVB ", " BGAL", filedata) # GalNAc beta
        filedata = re.sub("\s\wGA ", " AGLC", filedata) # Glc alpha
        filedata = re.sub("\s\wGB ", " BGLC", filedata) # Glc beta
        filedata = re.sub("\s\wLA ", " AGAL", filedata) # Gal alpha
        filedata = re.sub("\s\wLB ", " BGAL", filedata) # Gal beta
        filedata = re.sub("\s\wf[A|B]", " FUC", filedata) # Fuc alpha and beta
        filedata = re.sub("\s\wMA ", " AMAN", filedata) # Man alpha
        filedata = re.sub("\s\wMB ", " BMAN", filedata) # Man beta
        filedata = re.sub("\s\wSA ", " ANE5", filedata) # Neu5Ac alpha
        filedata = re.sub("\s\wGL ", " ANE5", filedata) # Neu5Gc 
        filedata = re.sub("\s\wXA ", " AXYL", filedata) # Xyl alpha
        filedata = re.sub("\s\wXB ", " BXYL", filedata) # Xyl beta
        filedata = re.sub("\s\wuA ", " AIDO", filedata) # IdoA alpha
        filedata = re.sub("\s\wZA ", " AGLC", filedata) # GlcA alpha
        filedata = re.sub("\s\wZB ", " BGLC", filedata) # GlcA beta
        filedata = re.sub("\s\whA ", " ARHM", filedata) # Rha alpha
        filedata = re.sub("\s\whB ", " BRHM", filedata) # Rha beta
        filedata = re.sub("\s\wAA ", " AARB", filedata) # Ara alpha
        filedata = re.sub("\s\wAB ", " BARB", filedata) # Ara beta
        filedata = re.sub("\s\wRA ", " ARIB", filedata) # Rib alpha
        filedata = re.sub("\s\wRB ", " BRIB", filedata) # Rib beta
        try:
            with open(f"CHARMM_format_ATOM/{pdb.split('.')[0]}.CHARMM.pdb", 'w') as file:
                file.write(filedata)
        except:
            pass

        with open("CHARMM_format_ATOM/README.txt", "w") as file:
            file.write("Warning:\n\nSome Glycan residues in the CHARMM naming format have residue names longer than the maximum four characters that are permitted in the PDB format. Therefore, it can be difficult to differentiate between similar residues (i.e. Glc and GlcNAc) on their residue name alone.")


        os.remove(pdb)
        
def add_conect_cards(pdb_file, bonded_atoms):
    """
    Adds CONECT records to a PDB file according to bonded_atoms.
    - pdb_file: path to the pdb file to update (in-place)
    - bonded_atoms: list of tuples/lists, each with atom indices (1-based) that are bonded, e.g. [(1,2), (2,3,4)]
    """
    with open(pdb_file, 'r') as f:
        lines = f.readlines()

    # Find where to insert CONECT cards (before END/ENDMDL if present)
    insert_idx = len(lines)
    for i, line in enumerate(lines):
        if line.startswith('END') or line.startswith('ENDMDL'):
            insert_idx = i
            break

    # Build CONECT lines, avoid duplicates
    conect_set = set()
    for bond in bonded_atoms:
        for i, atom1 in enumerate(bond):
            for atom2 in bond[i+1:]:
                key = tuple(sorted((atom1, atom2)))
                if key not in conect_set:
                    conect_set.add(key)

    conect_lines = []
    atom_bonds = defaultdict(list)
    for a1, a2 in conect_set:
        atom_bonds[a1].append(a2)
        atom_bonds[a2].append(a1)
    for atom, bonded in sorted(atom_bonds.items()):
        line = f"CONECT{atom:5d}" + ''.join(f"{b:5d}" for b in bonded)
        conect_lines.append(line + '\n')

    # Insert CONECT lines
    new_lines = lines[:insert_idx] + conect_lines + lines[insert_idx:]

    with open(pdb_file, 'w') as f:
        f.writelines(new_lines)