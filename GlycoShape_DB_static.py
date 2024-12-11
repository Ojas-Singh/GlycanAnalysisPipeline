#!/usr/bin/env python3

import ast
import json
import shutil
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors

import logging
import config
from lib import name, pdb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Function for replacing "_" with " " in evaluated strings
def remove_underscores_eval(input):
    if input is None:
        return None
    else:
        mod = []
        input_mod = ast.literal_eval(input)
        for entry in input_mod:
            mod.append(entry.replace("_", " "))
        mod2 = list(set(mod))
        if "undetermined" in mod2:
            mod2.remove("undetermined")
        return mod2

# Function for replacing "_" with " " in lists
def remove_underscores_list(input):
    if input is None:
        return None
    else:
        mod = []
        for entry in input:
            mod.append(entry.replace("_", " "))
        mod2 = list(set(mod))
        if "undetermined" in mod2:
            mod2.remove("undetermined")
        return mod2

def save_glycan_data(glycan_data: dict, ID, filename) -> None:
    """Save glycan data as JSON file.
    
    Args:
        glycan_data: Dictionary containing glycan data
        ID: Glycan ID
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(config.output_path) / str(ID)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON file
        output_file = output_dir / f"{filename}.json"
        with open(output_file, 'w') as f:
            json.dump(glycan_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved glycan data to {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to save glycan data: {str(e)}")
        raise

def main():
    """Main execution function."""
    data_dir = Path(config.data_dir)
    glycoshape_list = [f.name for f in data_dir.iterdir() if f.is_dir()]
    
    for glycan in glycoshape_list:
        try:
            # Get glycan data
            ID, length, package, FF, temp, pressure, salt, contributor = name.get_md_info(glycan)
            logger.info(f"Processing {glycan} with ID {ID}")
            
            # Create output directory
            output_dir = Path(config.output_path) / str(ID)
            if output_dir.exists() and not config.update:
                logger.info(f"Skipping {glycan} - output exists")
                continue
                
            if output_dir.exists() and config.update:
                shutil.rmtree(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process glycan data
            glycam_tidy, end_pos, end_link = name.glycam_info(glycan)

            iupac = name.glycam2iupac(glycam_tidy)
            iupac_alpha = f"{iupac}(a{end_pos}-"
            iupac_beta = f"{iupac}(b{end_pos}-"
            glycam_alpha = f"{glycam_tidy}a{end_pos}-OH"
            glycam_beta = f"{glycam_tidy}b{end_pos}-OH"
            
            try:
                mass, tpsa, rot_bonds, hbond_donor, hbond_acceptor = name.iupac2properties(iupac)
            except Exception as e:
                logger.warning(f"Could not retrieve properties for {iupac}: {e}")
                mass, tpsa, rot_bonds, hbond_donor, hbond_acceptor = None, None, None, None, None

            

           
            try:
                Composition = ast.literal_eval(Composition)
            except Exception:
                Composition = None

            snfg = name.iupac2snfg(iupac, ID)
            smiles = name.iupac2smiles(iupac)
            glycoct = name.iupac2glycoct(iupac)
            composition = name.iupac2composition(iupac)
            try:
                termini = name.iupac2termini(iupac)
            except Exception as e:
                logger.warning(f"Could not retrieve termini for {iupac}: {e}")
                termini = None

            cluster_dict_alpha = name.get_clusters_alpha(glycan, ID)
            cluster_dict_beta = name.get_clusters_beta(glycan, ID)

            cluster_dict = cluster_dict_alpha if cluster_dict_alpha.get("None") != "None" else cluster_dict_beta

            # Define oxford nomenclature mapping
            oxford_mapping = {
                "DManpa1-3[DManpa1-6]DManpb1-4DGlcpNAcb1-4DGlcpNAc": "M3",
                "DManpa1-3[DManpa1-6]DManpa1-6[DManpa1-3]DManpb1-4DGlcpNAcb1-4DGlcpNAc": "M5",
                "DGlcpNAcb1-2DManpa1-3[DGlcpNAcb1-2DManpa1-6]DManpb1-4DGlcpNAcb1-4DGlcpNAc": "A2",
                "DGalpb1-4DGlcpNAcb1-2DManpa1-3[DGalpb1-4DGlcpNAcb1-2DManpa1-6]DManpb1-4DGlcpNAcb1-4DGlcpNAc": "A2G2",
                "DGlcpNAcb1-2DManpa1-3[DGlcpNAcb1-2DManpa1-6]DManpb1-4DGlcpNAcb1-4[LFucpa1-6]DGlcpNAc": "FA2",
                "DGalpb1-4DGlcpNAcb1-2DManpa1-3[DGalpb1-4DGlcpNAcb1-2DManpa1-6]DManpb1-4DGlcpNAcb1-4[LFucpa1-6]DGlcpNAc": "FA2G2",
                "DGlcpNAcb1-6[DGlcpNAcb1-2]DManpa1-6[DGlcpNAcb1-2DManpa1-3]DManpb1-4DGlcpNAcb1-4DGlcpNAc": "A3",
                "DGalpb1-4DGlcpNAcb1-6[DGalpb1-4DGlcpNAcb1-2]DManpa1-6[DGlcpNAcb1-4[DGlcpNAcb1-2]DManpa1-3]DManpb1-4DGlcpNAcb1-4DGlcpNAc": "A4G4",
                "DGlcpNAcb1-6[DGlcpNAcb1-2]DManpa1-6[DGlcpNAcb1-4[DGlcpNAcb1-2]DManpa1-3]DManpb1-4DGlcpNAcb1-4DGlcpNAc": "A4"
            }

            # Pattern-based matching for more complex cases
            m6_patterns = [
                "DManpa1-3[DManpa1-3[DManpa1-2DManpa1-6]DManpa1-6]DManpb1-4DGlcpNAcb1-4DGlcpNAc",
                "DManpa1-2DManpa1-3[DManpa1-3[DManpa1-6]DManpa1-6]DManpb1-4DGlcpNAcb1-4DGlcpNAc",
                "DManpa1-3[DManpa1-2DManpa1-3[DManpa1-6]DManpa1-6]DManpb1-4DGlcpNAcb1-4DGlcpNAc"
            ]

            m7_patterns = [
                "DManpa1-3[DManpa1-2DManpa1-3[DManpa1-2DManpa1-6]DManpa1-6]DManpb1-4DGlcpNAcb1-4DGlcpNAc",
                "DManpa1-2DManpa1-2DManpa1-3[DManpa1-3[DManpa1-6]DManpa1-6]DManpb1-4DGlcpNAcb1-4DGlcpNAc",
                "DManpa1-2DManpa1-3[DManpa1-2DManpa1-3[DManpa1-6]DManpa1-6]DManpb1-4DGlcpNAcb1-4DGlcpNAc",
                "DManpa1-2DManpa1-3[DManpa1-3[DManpa1-2DManpa1-6]DManpa1-6]DManpb1-4DGlcpNAcb1-4DGlcpNAc"
            ]

            # Get oxford nomenclature
            oxford = oxford_mapping.get(glycam_tidy)
            if not oxford:
                if glycam_tidy in m6_patterns:
                    oxford = "M6"
                elif glycam_tidy in m7_patterns:
                    oxford = "M7"
                else:
                    oxford = None
            
            glytoucan, wurcs = name.iupac2wurcs_glytoucan(iupac)
            glytoucan_alpha, wurcs_alpha = name.iupac2wurcs_glytoucan(iupac_alpha)
            glytoucan_beta, wurcs_beta = name.iupac2wurcs_glytoucan(iupac_beta)

            # if wurcs is None get it from Mol2WURCS using smiles
            if wurcs == None:
                print("WURCS is None, will try to get it from Mol2WURCS")
                wurcs = name.smiles2wurcs(smiles)
                wurcs, wurcs_alpha, wurcs_beta = name.get_wurcs_variants(wurcs)

            try: 
                if glytoucan != None:
                    motifs = name.glytoucan2motif(glytoucan)
                else:
                    motifs = None
            except Exception as e:
                logger.warning(f"Could not retrieve motifs for {iupac}: {e}")
                motifs = None

            glycan_data = {
                "archetype" :{

                # General info
                "ID": ID,
                "name": glycan,
                "glycam": glycam_tidy,
                "iupac": iupac,
                "iupac_extended": name.wurcs2extendediupac(wurcs),
                "glytoucan": glytoucan,
                "wurcs": wurcs,
                "glycoct": glycoct,
                "smiles": smiles,
                "oxford": oxford,

                # Chemical properties
                "mass": round(Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles)), 2),
                "motifs": motifs,
                "termini": termini,
                "components":composition,
                "composition":Composition,

                "components_search":composition,
                "composition_search":Composition,
                "rot_bonds":rot_bonds, 
                "hbond_donor":hbond_donor, 
                "hbond_acceptor":hbond_acceptor, 
                

                # Molecular Dynamics info
                "clusters": cluster_dict,
                "length": length,
                "package": package,
                "forcefield": FF,
                "temperature": temp,
                "pressure": pressure,
                "salt": salt,
            },

            "alpha":{
                

                # General info
                "ID": ID,
                "name": glycan,
                "glycam": glycam_alpha,
                "iupac": iupac_alpha,
                "iupac_extended": name.wurcs2extendediupac(wurcs_alpha),
                "glytoucan": glytoucan_alpha,
                "wurcs": wurcs_alpha,
                "glycoct": glycoct,
                "smiles": smiles,
                "oxford": oxford,

                # Chemical properties
                "mass": round(Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles)), 2),
                "motifs": motifs,
                "termini": termini,
                "components":composition,
                "composition":Composition,

                "components_search":composition,
                "composition_search":Composition,
                "rot_bonds":rot_bonds, 
                "hbond_donor":hbond_donor, 
                "hbond_acceptor":hbond_acceptor, 
                
                


                # Molecular Dynamics info
                "clusters": cluster_dict,
                "length": length,
                "package": package,
                "forcefield": FF,
                "temperature": temp,
                "pressure": pressure,
                "salt": salt,
            },
            
            "beta":{
                

                # General info
                "ID": ID,
                "name": glycan,
                "glycam": glycam_beta,
                "iupac": iupac_beta,
                "iupac_extended": name.wurcs2extendediupac(wurcs_beta),
                "glytoucan": glytoucan_beta,
                "wurcs": wurcs_beta,
                "glycoct": glycoct,
                "smiles": smiles,
                "oxford": oxford,

                # Chemical properties
                "mass": round(Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles)), 2),
                "motifs": motifs,
                "termini": termini,
                "components":composition,
                "composition":Composition,

                "components_search":composition,
                "composition_search":Composition,
                "rot_bonds":rot_bonds, 
                "hbond_donor":hbond_donor, 
                "hbond_acceptor":hbond_acceptor, 
                

                # Molecular Dynamics info
                "clusters": cluster_dict,
                "length": length,
                "package": package,
                "forcefield": FF,
                "temperature": temp,
                "pressure": pressure,
                "salt": salt,
            }
            
            }

            # Save data
            save_glycan_data(glycan_data, ID, "data")
            save_glycan_data(name.glytoucan2glygen(glytoucan), ID, "glygen")
            save_glycan_data(name.glytoucan2glycosmos(glytoucan), ID, "glycosmos")

            
            
            pdb.convert_pdbs(ID)
            shutil.copytree(Path(config.input_path) / f"{glycan}/clusters/pack", Path(config.output_path) / f"{ID}/output", dirs_exist_ok=True)

            
        except Exception as e:
            logger.error(f"Failed to process {glycan}: {str(e)}")
            continue

if __name__ == "__main__":
    main()

