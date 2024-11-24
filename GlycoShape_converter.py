#!/usr/bin/env python3

import os
import re
import glob
import shutil
from config import output_path

def create_directories(base_path):
    formats = ['GLYCAM', 'PDB', 'CHARMM']
    types = ['ATOM', 'HETATM']
    for format in formats:
        for type in types:
            dir_path = os.path.join(base_path, f"{format}_format_{type}")
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            os.mkdir(dir_path)

def process_file(input_file, output_dir, conversion_type):
    with open(input_file, 'r') as file:
        filedata = file.read()
    
    if conversion_type == 'GLYCAM_HETATM':
        filedata = filedata.replace("ATOM  ", "HETATM")
    elif conversion_type in ['PDB', 'CHARMM']:
        filedata = apply_conversions(filedata, conversion_type)
    
    output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_file))[0]}.{conversion_type}.pdb")
    with open(output_file, 'w') as file:
        file.write(filedata)

def apply_conversions(filedata, conversion_type):
    conversions = {
        'PDB': {
            "\s\wYA": " NDG", "\s\wYB": " NAG", "\s\wVA": " A2G", "\s\wVB": " NGA",
            "\s\wGA": " GLC", "\s\wGB": " BGC", "\s\wGL": " NGC", "\s\wLA": " GLA",
            "\s\wLB": " GAL", "\s\wfA": " FUC", "\s\wfB": " FUL", "\s\wMB": " BMA",
            "\s\wMA": " MAN", "\s\wSA": " SIA", "\s\wZA": " GCU", "\s\wZB": " BDP",
            "\s\wXA": " XYS", "\s\wXB": " XYP", "\s\wuA": " IDR", "\s\whA": " RAM",
            "\s\whB": " RHM", "\s\wRA": " RIB", "\s\wRB": " BDR", "\s\wAA": " ARA",
            "\s\wAB": " ARB"
        },
        'CHARMM': {
            "\s\wYA ": " AGLC", "\s\wYB ": " BGLC", "\s\wVA ": " AGAL", "\s\wVB ": " BGAL",
            "\s\wGA ": " AGLC", "\s\wGB ": " BGLC", "\s\wLA ": " AGAL", "\s\wLB ": " BGAL",
            "\s\wf[A|B]": " FUC", "\s\wMA ": " AMAN", "\s\wMB ": " BMAN", "\s\wSA ": " ANE5",
            "\s\wGL ": " ANE5", "\s\wXA ": " AXYL", "\s\wXB ": " BXYL", "\s\wuA ": " AIDO",
            "\s\wZA ": " AGLC", "\s\wZB ": " BGLC", "\s\whA ": " ARHM", "\s\whB ": " BRHM",
            "\s\wAA ": " AARB", "\s\wAB ": " BARB", "\s\wRA ": " ARIB", "\s\wRB ": " BRIB"
        }
    }
    
    for pattern, replacement in conversions[conversion_type].items():
        filedata = re.sub(pattern, replacement, filedata)
    
    if conversion_type == 'PDB':
        filedata = filedata.replace("ATOM  ", "HETATM")
    
    return filedata

def write_charmm_readme(directory):
    readme_path = os.path.join(directory, "README.txt")
    with open(readme_path, "w") as file:
        file.write("Warning:\n\nSome Glycan residues in the CHARMM naming format have residue names longer than the maximum four characters that are permitted in the PDB format. Therefore, it can be difficult to differentiate between similar residues (i.e. Glc and GlcNAc) on their residue name alone.")

def main():
    directories = glob.glob(f"{output_path}/*/")
    for directory in directories:
        print(f"Converting for {os.path.basename(os.path.dirname(directory))}")
        
        create_directories(directory)
        
        pdb_files = glob.glob(os.path.join(directory, "*pdb"))
        for pdb in pdb_files:
            for format in ['GLYCAM', 'PDB', 'CHARMM']:
                for type in ['ATOM', 'HETATM']:
                    output_dir = os.path.join(directory, f"{format}_format_{type}")
                    conversion_type = f"{format}_{type}" if format == 'GLYCAM' else format
                    process_file(pdb, output_dir, conversion_type)
            
            os.remove(pdb)
        
        write_charmm_readme(os.path.join(directory, "CHARMM_format_ATOM"))
        write_charmm_readme(os.path.join(directory, "CHARMM_format_HETATM"))

if __name__ == "__main__":
    main()