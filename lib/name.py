
import re
import sys
import shutil
import requests
import collections
import pandas as pd
from pathlib import Path
from glypy.io import iupac as glypy_iupac

# Import glycowork functions (thankyou daniel for this amazing package!)
from glycowork.motif.draw import GlycoDraw
from glycowork.motif.processing import canonicalize_iupac
from glycowork.motif.annotate import annotate_glycan
from glycowork.motif.annotate import get_molecular_properties
from glycowork.motif.processing import IUPAC_to_SMILES
from glycowork.motif.annotate import get_terminal_structures

import lib.config as config
import logging
import subprocess

from lib import pdb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import json
from glycowork.motif.graph import compare_glycans



def get_oxford(iupac, oxford_dict='oxford.json'):
    with open(oxford_dict, 'r') as f:
        oxford_data = json.load(f)
    
    input_length = len(iupac)
    
    for oxford_name, iupac_list in oxford_data.items():
        for iupac_from_json in iupac_list:
            if len(iupac_from_json) == input_length:
                if compare_glycans(iupac_from_json, iupac):
                    return oxford_name
    
    return None


# Function to sort a dictionary by value...
def sort_dict(d, reverse = True):
  return dict(sorted(d.items(), key = lambda x: x[1], reverse = reverse))



    

def clean_iupac(iupac):
    iupac = iupac.replace("D-", "")
    iupac = iupac.replace("L-", "") 
    return iupac

# Function to calculate the monosaccharide composition from the condensed IUPAC nomenclature...
def iupac2composition(iupac):
    iupac_mod = re.sub("\[|\]", "", iupac)
    iupac_mod = re.sub("\(.*?\)", ".", iupac_mod)
    components = [component for component in iupac_mod.split(".")]
    composition =dict(collections.Counter(components))

    return sort_dict(composition)


def glycam2iupac(glycam):
    # Define a dictionary of default stereochemistry for common monosaccharides
    default_stereochemistry = {
        "4eLeg": "D", "6dAlt": "L", "6dAltNAc": "L", "6dGul": "D",
        "6dTal": "D", "6dTalNAc": "D", "8eAci": "D", "8eLeg": "L",
        "Abe": "D", "Aci": "L", "All": "D", "AllA": "D", "AllN": "D",
        "AllNAc": "D", "Alt": "L", "AltA": "L", "AltN": "L", "AltNAc": "L",
        "Api": "L", "Ara": "L", "Bac": "D", "Col": "L", "DDmanHep": "D",
        "Dha": "D", "Dig": "D", "Fru": "D", "Fuc": "L", "FucNAc": "L",
        "Gal": "D", "GalA": "D", "GalN": "D", "GalNAc": "D", "Glc": "D",
        "GlcA": "D", "GlcN": "D", "GlcNAc": "D", "Gul": "D", "GulA": "D",
        "GulN": "D", "GulNAc": "D", "Ido": "L", "IdoA": "L", "IdoN": "L",
        "IdoNAc": "L", "Kdn": "D", "Kdo": "D", "Leg": "D", "LDmanHep": "L",
        "Lyx": "D", "Man": "D", "ManA": "D", "ManN": "D", "ManNAc": "D",
        "Mur": "D", "MurNAc": "D", "MurNGc": "D", "Neu": "D", "Neu5Ac": "D",
        "Neu5Gc": "D", "Oli": "D", "Par": "D", "Pse": "L", "Psi": "D",
        "Qui": "D", "QuiNAc": "D", "Rha": "L", "RhaNAc": "L", "Rib": "D",
        "Sia": "D", "Sor": "L", "Tag": "D", "Tal": "D", "TalA": "D",
        "TalN": "D", "TalNAc": "D", "Tyv": "D", "Xyl": "D"
    }
    
    if glycam == "DGalpb1-4DGalpa1-3[2,4-diacetimido-2,4,6-trideoxyhexose]":
        iupac = "Gal(b1-4)Gal(a1-3)2,4-diacetimido-2,4,6-trideoxyhexose"
    else:
        glycam_components = glycam.split("-")
        mod_component_list = []
        for component in glycam_components:
            mod_component = component
            
            # Check for stereochemistry and modify appropriately
            for sugar, default in default_stereochemistry.items():
                if sugar in mod_component:
                    if default == "D":
                        mod_component = mod_component.replace("D", "")
                        mod_component = mod_component.replace("L", "L-")
                    elif default == "L":
                        mod_component = mod_component.replace("L", "")
                        mod_component = mod_component.replace("D", "D-")
            
            mod_component = mod_component.replace("p", "")
            # mod_component = mod_component.replace("f", "")
            
            # Handle glycosidic linkages
            if component != glycam_components[-1]:
                mod_component = mod_component.replace(mod_component[-2:], f"({mod_component[-2:]}-", 1)
            if component != glycam_components[0]:
                mod_component = mod_component.replace(mod_component[0], f"{mod_component[0]})", 1)
            
            # Replace common modifications
            mod_component = mod_component.replace("[2S]", "2S")
            mod_component = mod_component.replace("[3S]", "3S")
            mod_component = mod_component.replace("[4S]", "4S")
            mod_component = mod_component.replace("[6S]", "6S")
            mod_component = mod_component.replace("[3S-6S]", "3S6S")
            mod_component = mod_component.replace("[3S,6S]", "3S6S")
            mod_component = mod_component.replace("[4S,6S]", "4S6S")
            mod_component = mod_component.replace("[2Me]", "2Me")
            mod_component = mod_component.replace("[2Me-3Me]", "2Me3Me")
            mod_component = mod_component.replace("[2Me,3Me]", "2Me3Me")
            mod_component = mod_component.replace("[2Me-4Me]", "2Me4Me")
            mod_component = mod_component.replace("[2Me,4Me]", "2Me4Me")
            mod_component = mod_component.replace("[2Me-6Me]", "2Me6Me")
            mod_component = mod_component.replace("[2Me,6Me]", "2Me6Me")
            mod_component = mod_component.replace("[2Me-3Me-4Me]", "2Me3Me4Me")
            mod_component = mod_component.replace("[2Me,3Me,4Me]", "2Me3Me4Me")
            mod_component = mod_component.replace("[3Me]", "3Me")
            mod_component = mod_component.replace("[4Me]", "4Me")
            mod_component = mod_component.replace("[6Me]", "6Me")
            mod_component = mod_component.replace("[9Me]", "9Me")
            mod_component = mod_component.replace("[2A]", "2Ac")
            mod_component = mod_component.replace("[4A]", "4Ac")
            mod_component = mod_component.replace("[9A]", "9Ac")
            mod_component = mod_component.replace("[6PC]", "6Pc")
            mod_component = mod_component.replace("DKDN", "Kdn")
            mod_component = mod_component.replace("DKDO", "Kdo")
            mod_component = mod_component.replace("LKDN", "LKdn")
            mod_component = mod_component.replace("LKDO", "LKdo")
            
            mod_component_list.append(mod_component)
        
        iupac = "".join(mod_component_list)
    return iupac



def iupac2wurcs_glytoucan(iupac_condensed):
    """
    Converts IUPAC condensed format to WURCS format and retrieves the GlyTouCan accession number.

    Parameters:
        iupac_condensed (str): The IUPAC Condensed format string.

    Returns:
        dict: A dictionary containing the GlyTouCan accession number and WURCS format, 
              or an error message if the request fails.
    """
    # Base URL for the API endpoint
    url = f"https://api.glycosmos.org/glycanformatconverter/2.10.0/iupaccondensed2wurcs/{iupac_condensed}"
    
    try:
        # Make the API request
        response = requests.get(url)
        # Raise an exception if the request failed
        response.raise_for_status()
        
        # Parse the JSON response
        data = response.json()
        return data.get("id"),data.get("WURCS")
    except requests.exceptions.RequestException as e:
        # Handle any request exceptions
        return {"error": str(e)}
    except KeyError:
        # Handle unexpected response structure
        return {"error": "Unexpected response structure"}

def wurcs2extendediupac(wurcs):
    """
    Converts WURCS format to extended IUPAC using GlyCosmos API.
    
    Args:
        wurcs (str): WURCS format glycan string
        
    Returns:
        str: Extended IUPAC format or None if request fails
    """
    url = "https://api.glycosmos.org/glycanformatconverter/2.10.0/wurcs2iupacextended"
    try:
        response = requests.post(url, json={"input": wurcs})
        response.raise_for_status()
        response.encoding = 'utf-8'
        raw_iupac = response.json()["IUPACextended"]
        return raw_iupac
    except Exception as e:
        logger.error(f"Failed to convert WURCS to extended IUPAC: {str(e)}")
        return None

# def wurcs2mass(wurcs):
#     """
#     Gets mass from WURCS string using GlyCosmos API.
    
#     Args:
#         wurcs (str): WURCS format glycan string
        
#     Returns:
#         float: Mass of glycan or None if request fails
#     """
#     url = "https://api.glycosmos.org/wurcsframework/1.2.14/wurcs2mass"
#     try:
#         response = requests.post(url, json=[wurcs])
#         response.raise_for_status()
#         data = response.json()
#         return data[0].get("mass")
#     except Exception as e:
#         logger.error(f"Failed to get mass from WURCS: {str(e)}")
#         return None
    

def wurcsRDFmatch(wurcs):
    """
    Gets matches for a WURCS string using GlyCosmos partial match API.
    
    Args:
        wurcs (str): WURCS format glycan string
        
    Returns:
        list: List of dicts containing id and wurcs for matches
    """
    encoded_wurcs = requests.utils.quote(wurcs)
    url = f"https://api.glycosmos.org/partialmatch/wurcsrdf?wurcs={encoded_wurcs}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to get WURCS matches: {str(e)}")
        return []

# Function to look up a GlycoCT name from IUPAC using GlyConnect...
def iupac2glycoct(iupac):
    try:
        return str(glypy_iupac.loads(iupac, dialect="simple"))
    except:
        return None


def iupac2smiles(iupac):
    smiles = IUPAC_to_SMILES([iupac])[0]
    return smiles


def iupac2snfg(iupac, ID):
    output_path = Path(config.output_path) / str(ID) / 'snfg.svg'
    snfg = GlycoDraw(iupac, filepath=str(output_path), show_linkage=True)
    return str(output_path)


def canonicalize_iupac(iupac):
    # return canonicalize_iupac(iupac)
    return canonicalize_iupac([iupac])

    
# Function to get the glycan name, end position and end link from a Glycam name...
def glycam_info(glycam):
    glycan_tidy = glycam
    end_pos = None
    end_link = None
    if glycam[-3:] == "-OH":
        glycan_tidy = glycam[:-5]
        end_pos = glycam[-4] 
        end_link = glycam[-5]
    return glycan_tidy, end_pos, end_link


# Function to get motifs from an IUPAC name...
def iupac2motif(iupac):
    df = annotate_glycan(iupac)
    df = df.loc[:, (df != 0).any(axis=0)]
    motifs = df.columns.tolist()[1:]
    return motifs

def iupac2properties(iupac):
    
    try:
        df = get_molecular_properties([iupac], placeholder=True)
        mass = df["exact_mass"].values.tolist()[0]
        tpsa = df["tpsa"].values.tolist()[0]
        rot_bonds = df["rotatable_bond_count"].values.tolist()[0]
        hbond_donor = df["h_bond_donor_count"].values.tolist()[0]
        hbond_acceptor = df["h_bond_acceptor_count"].values.tolist()[0]
        return mass, tpsa, rot_bonds, hbond_donor, hbond_acceptor
    except:
        sys.stderr.write(f'{iupac} properties were not found\n')
        mass, tpsa, rot_bonds, hbond_donor, hbond_acceptor = None, None, None, None, None

def iupac2termini(iupac):
    termini = get_terminal_structures(iupac)
    if termini != []:
        termini_mod = list(set(termini))
        return termini_mod
    else:
        return None



# Function to fetch data from SugarBase for the glycan of interest...
def iupac2sugarbase(iupac: str) -> tuple:
    """Fetch data from SugarBase for glycan.
    
    Args:
        iupac: IUPAC name of glycan
        
    Returns:
        Tuple of glycan properties or tuple of None values if not found
    """
    try:
        if not hasattr(iupac2sugarbase, '_sugarbase_df'):
            df = pd.read_csv(config.sugarbase_path)
            df = df.where(pd.notnull(df), None)
            iupac2sugarbase._sugarbase_df = df
            logger.info(f"Loaded sugarbase from {config.sugarbase_path}")
            
        sugarbase = iupac2sugarbase._sugarbase_df
        df = sugarbase.loc[sugarbase["glycan"] == iupac]
        if df.shape[0] != 0:
            value_list = []
            for parameter in ['Species', 'Genus', 'Family', 'Order', 'Class', 'Phylum',
                            'Kingdom', 'Domain','glycan_type',
                            'disease_association','tissue_sample','Composition']:
                values = df[parameter].values.tolist()[0]
                if values == "[]":
                    value_list.append(None)
                else:
                    value_list.append(values)
        else:
            print(f"{iupac} not found in SugarBase")
            value_list = [None for n in range(12)]
        return value_list
            
    except Exception as e:
        logger.error(f"Failed to process sugarbase data: {str(e)}")
        return tuple([None] * 13)
    



# Function to get MD simulation info for a glycan submission...
def get_md_info(glycan: str) -> tuple:
    """Get MD simulation info for a glycan submission.
    
    Args:
        glycan: Glycan name
        
    Returns:
        Tuple of (ID, length, package, FF, temp, pressure, salt, contributor)
    """
    # Cache the inventory DataFrame
    if not hasattr(get_md_info, '_inventory_df'):
        try:
            latest_inventory = Path(config.inventory_path)
            if not latest_inventory.exists():
                logger.error(f"Inventory file not found: {latest_inventory}")
                return 0, 0, 0, 0, 0, 0, 0, "0"
                
            df = pd.read_csv(latest_inventory)
            df = df[df['Timestamp'].notna()]
            get_md_info._inventory_df = df
            logger.info(f"Loaded inventory from {latest_inventory}")
        except Exception as e:
            logger.error(f"Failed to load inventory: {str(e)}")
            return 0, 0, 0, 0, 0, 0, 0, "0"
    
    try:
        df = get_md_info._inventory_df
        glycan_data = df.loc[df[df.columns[3]] == glycan]
        
        if glycan_data.empty:
            logger.warning(f"No inventory data found for {glycan}")
            return 0, 0, 0, 0, 0, 0, 0, "0"
            
        length = str(glycan_data[df.columns[5]].iloc[0])
        package = glycan_data[df.columns[6]].iloc[0]
        FF = glycan_data[df.columns[7]].iloc[0]
        temp = str(glycan_data[df.columns[8]].iloc[0])
        pressure = str(glycan_data[df.columns[9]].iloc[0])
        salt = glycan_data[df.columns[10]].iloc[0]
        contributor = glycan_data[df.columns[2]].iloc[0]
        ID = glycan_data[df.columns[0]].iloc[0]
        
        # Replace NaN values with None
        ID = None if pd.isna(ID) else str(ID)
        length = None if pd.isna(length) else str(length)
        package = None if pd.isna(package) else package
        FF = None if pd.isna(FF) else FF
        temp = None if pd.isna(temp) else str(temp)
        pressure = None if pd.isna(pressure) else str(pressure)
        salt = None if pd.isna(salt) else salt
        contributor = None if pd.isna(contributor) else contributor

        return ID, length, package, FF, temp, pressure, salt, contributor
        
    except Exception as e:
        logger.error(f"Error processing inventory data for {glycan}: {str(e)}")
        return 0, 0, 0, 0, 0, 0, 0, "0"

# Function to get the cluster information for the glycan of interest...
def get_clusters_alpha(glycan: str, ID: str) -> dict:
    """Get cluster information for alpha conformer.
    
    Args:
        glycan: Glycan name
        ID: Glycan ID
        
    Returns:
        Dictionary of cluster information
    """
    cluster_dict = {}
    conf = "alpha"
    
    try:
        # Setup paths
        cluster_path = Path(config.data_dir) / glycan / "clusters" / conf
        output_path = Path(config.output_path) / str(ID)
        
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize cluster lists
        cluster_ids = []
        cluster_occupancies = []
        
        # Check for PDB files
        files = list(cluster_path.glob("*.pdb"))
        if not files:
            logger.warning(f"No cluster files found for {glycan} {conf}")
            cluster_ids.append("None")
            cluster_occupancies.append("None")
        else:
            # Sort files by occupancy
            files_mod = sorted([float(f.stem.split("_")[1]) for f in files], reverse=True)
            
            # Process each file
            for file_num, occupancy in enumerate(files_mod):
                try:
                    # Find matching PDB file
                    pdb_file = next(cluster_path.glob(f'*_{occupancy:.2f}.pdb'))
                except StopIteration:
                    pdb_file = next(cluster_path.glob(f'*_{occupancy:.1f}.pdb'))
                
                # Copy and process file
                output_file = output_path / f"cluster{file_num}_{conf}.pdb"
                shutil.copyfile(pdb_file, output_file)
                pdb.pdb_remark_adder(output_file)
                
                # Store cluster info
                cluster_ids.append(f"Cluster {file_num}")
                cluster_occupancies.append(occupancy)
        
        # Create cluster dictionary
        for n in range(len(cluster_ids)):
            cluster_dict[cluster_ids[n]] = cluster_occupancies[n]
            
        return cluster_dict
        
    except Exception as e:
        logger.error(f"Failed to process {conf} clusters for {glycan}: {str(e)}")
        return {"None": "None"}
    
    # Function to get the cluster information for the glycan of interest...
def get_clusters_beta(glycan: str, ID: str) -> dict:
    """Get cluster information for beta conformer.
    
    Args:
        glycan: Glycan name
        ID: Glycan ID
        
    Returns:
        Dictionary of cluster information
    """
    cluster_dict = {}
    conf = "beta"
    
    try:
        # Setup paths
        cluster_path = Path(config.data_dir) / glycan / "clusters" / conf
        output_path = Path(config.output_path) / str(ID)
        
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize cluster lists
        cluster_ids = []
        cluster_occupancies = []
        
        # Check for PDB files
        files = list(cluster_path.glob("*.pdb"))
        if not files:
            logger.warning(f"No cluster files found for {glycan} {conf}")
            cluster_ids.append("None")
            cluster_occupancies.append("None")
        else:
            # Sort files by occupancy
            files_mod = sorted([float(f.stem.split("_")[1]) for f in files], reverse=True)
            
            # Process each file
            for file_num, occupancy in enumerate(files_mod):
                try:
                    # Find matching PDB file
                    pdb_file = next(cluster_path.glob(f'*_{occupancy:.2f}.pdb'))
                except StopIteration:
                    pdb_file = next(cluster_path.glob(f'*_{occupancy:.1f}.pdb'))
                
                # Copy and process file
                output_file = output_path / f"cluster{file_num}_{conf}.pdb"
                shutil.copyfile(pdb_file, output_file)
                pdb.pdb_remark_adder(output_file)
                
                # Store cluster info
                cluster_ids.append(f"Cluster {file_num}")
                cluster_occupancies.append(occupancy)
        
        # Create cluster dictionary
        for n in range(len(cluster_ids)):
            cluster_dict[cluster_ids[n]] = cluster_occupancies[n]
            
        return cluster_dict
        
    except Exception as e:
        # logger.error(f"Failed to process {conf} clusters for {glycan}: {str(e)}")
        return {"None": "None"}
    

def glytoucan2glygen(glytoucan_ac):
    """Get Glygen data for a GlyTouCan accession number.

    Args:
        glytoucan_ac (str): GlyTouCan accession number
        
    Returns:
        dict: Glygen data or None if request fails
    """
    url = f"https://api.glygen.org/glycan/detail/{glytoucan_ac}/"

    try:
        response = requests.post(
            url,
            headers={'accept': 'application/json', 'Content-Type': 'application/json'},
            json={'glytoucan_ac': glytoucan_ac},
            verify=False
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to get Glygen data: {str(e)}")
        return None
    
def glytoucan2glycosmos(glytoucan_id):
    """Get taxonomy data from GlyCosmos API for a GlyTouCan ID.
    
    Args:
        glytoucan_id (str): GlyTouCan accession number
        
    Returns:
        dict: Taxonomy data or None if request fails
    """
    url = f"https://api.glycosmos.org/sparqlist/Glycan-taxonomy?id={glytoucan_id}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to get GlyCosmos taxonomy data: {str(e)}")
        return None
    

def smiles2wurcs(smiles):
    jar_path = Path(__file__).parent / "MolWURCS.jar"
    print("Using MolWURCS at :",jar_path)
    try:
        result = subprocess.run(
            ["java", "-jar", str(jar_path), "--in", "smi", "--out", "wurcs", smiles],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to convert SMILES to WURCS: {e.stderr}")
        return None


def glytoucan2motif(glytoucan_id):
    """Get glycan motif data from GlyCosmos API for a GlyTouCan ID.
    
    Args:
        glytoucan_id (str): GlyTouCan accession number
        
    Returns:
        dict: Motif data or None if request fails
    """
    url = f"https://api.alpha.glycosmos.org/sparqlist/get_glycan_motif?id={glytoucan_id}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to get glycan motif data: {str(e)}")
        return None

# not perfect     
def get_wurcs_variants(input_wurcs: str):
    """
    Given an input WURCS string (archetype, alpha, beta, or neutral),
    return a dictionary containing all three forms: archetype, alpha, beta.
    Returns None if the input WURCS does not match expected patterns.
    """
    # Define the WURCS pattern
    pattern = re.compile(
        r'^WURCS=2\.0/(\d+),(\d+),(\d+)/(\[.*?\]+)/([^/]+)/([^/]+)$'
    )
    match = pattern.match(input_wurcs)
    if not match:
        print(f"Invalid WURCS format: {input_wurcs}")
        return None

    S, D, B, residues_str, linkage, annotation = match.groups()

    # Extract residues
    residues = re.findall(r'\[([^\]]+)\]', residues_str)
    if not residues:
        print(f"No residues found in the WURCS string: {input_wurcs}")
        return None

    first_residue = residues[0]

    # Identify the type based on the first residue
    if first_residue.startswith('u') or first_residue.startswith('U'):
        wtype = 'archetype'
    elif '-1a_' in first_residue:
        wtype = 'alpha'
    elif '-1b_' in first_residue:
        wtype = 'beta'
    elif '-1x_' in first_residue:
        wtype = 'neutral'
    else:
        print(f"Could not determine the type of the WURCS string: {input_wurcs}")
        return None

    # Create copies for each form
    arch_residues = residues.copy()
    alpha_residues = residues.copy()
    beta_residues = residues.copy()

    if wtype == 'archetype':
        # Archetype is input; generate alpha and beta
        match_alpha = re.match(r'u(\w+)(.*)', arch_residues[0], re.IGNORECASE)
        if match_alpha:
            # Replace [uXXXX...] with [aXXXX-1a_1-5...]
            alpha_residues[0] = f'a{match_alpha.group(1)}-1a_1-5{match_alpha.group(2)}'
            # Replace [uXXXX...] with [aXXXX-1b_1-5...]
            beta_residues[0] = f'a{match_alpha.group(1)}-1b_1-5{match_alpha.group(2)}'
        else:
            print(f"Archetype first residue does not match expected pattern: {first_residue}")
            return None

    elif wtype == 'alpha':
        # Alpha is input; generate archetype and beta
        match_arch = re.match(r'a(\w+)-1a_(\d+-\d+)(.*)', alpha_residues[0])
        if match_arch:
            # Replace [aXXXX-1a_Y-Z...] with [uXXXX...]
            arch_residues[0] = f'u{match_arch.group(1)}{match_arch.group(3)}'
            # Replace [aXXXX-1a_Y-Z...] with [aXXXX-1b_Y-Z...]
            beta_residues[0] = f'a{match_arch.group(1)}-1b_{match_arch.group(2)}{match_arch.group(3)}'
        else:
            print(f"Alpha first residue does not match expected pattern: {first_residue}")
            return None

    elif wtype == 'beta':
        # Beta is input; generate archetype and alpha
        match_arch = re.match(r'a(\w+)-1b_(\d+-\d+)(.*)', beta_residues[0])
        if match_arch:
            # Replace [aXXXX-1b_Y-Z...] with [uXXXX...]
            arch_residues[0] = f'u{match_arch.group(1)}{match_arch.group(3)}'
            # Replace [aXXXX-1b_Y-Z...] with [aXXXX-1a_Y-Z...]
            alpha_residues[0] = f'a{match_arch.group(1)}-1a_{match_arch.group(2)}{match_arch.group(3)}'
        else:
            print(f"Beta first residue does not match expected pattern: {first_residue}")
            return None

    elif wtype == 'neutral':
        # Neutral is input; generate archetype, alpha, beta
        match_neutral = re.match(r'a(\w+)-1x_(\d+-\d+)(.*)', first_residue)
        if match_neutral:
            # Archetype: replace 'a' with 'u' and remove suffix
            arch_residues[0] = f'u{match_neutral.group(1)}{match_neutral.group(3)}'
            # Alpha: replace '-1x_' with '-1a_'
            alpha_residues[0] = f'a{match_neutral.group(1)}-1a_{match_neutral.group(2)}{match_neutral.group(3)}'
            # Beta: replace '-1x_' with '-1b_'
            beta_residues[0] = f'a{match_neutral.group(1)}-1b_{match_neutral.group(2)}{match_neutral.group(3)}'
        else:
            print(f"Neutral first residue does not match expected pattern: {first_residue}")
            return None

    # Reconstruct residues strings
    arch_residues_str = ''.join([f'[{r}]' for r in arch_residues])
    alpha_residues_str = ''.join([f'[{r}]' for r in alpha_residues])
    beta_residues_str = ''.join([f'[{r}]' for r in beta_residues])

    # Reconstruct WURCS strings
    archetype_wurcs = f'WURCS=2.0/{S},{D},{B}/{arch_residues_str}/{linkage}/{annotation}'
    alpha_wurcs = f'WURCS=2.0/{S},{D},{B}/{alpha_residues_str}/{linkage}/{annotation}'
    beta_wurcs = f'WURCS=2.0/{S},{D},{B}/{beta_residues_str}/{linkage}/{annotation}'


    return normalize_wurcs(archetype_wurcs), normalize_wurcs(alpha_wurcs), normalize_wurcs(beta_wurcs)

def validate_wurcs(wurcs):
    """
    Validates a WURCS string using the GlyCosmos API.
    Returns the standardized WURCS string if valid, the original WURCS string if a server error (HTTP 500) occurs,
    or None if validation errors are found.

    Args:
        wurcs (str): WURCS format string to validate

    Returns:
        str or None: Validated WURCS string, original string on server error, or None if invalid.
    """
    url = "https://api.glycosmos.org/wurcsframework/1.3.1/wurcsvalidator"
    
    try:
        response = requests.post(url, json=[wurcs])
        if response.status_code == 500:
            logger.warning("Server error during WURCS validation (status 500), returning original WURCS.")
            return wurcs
        response.raise_for_status()
        
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            validation_result = data[0]
            error_reports = validation_result.get("m_mapTypeToReports", {}).get("ERROR", [])
            
            # If no errors, return the standardized WURCS string if available; otherwise, return the original.
            if not error_reports:
                return validation_result.get("m_sStandardString", wurcs)
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to validate WURCS: {str(e)}")
        return None

def normalize_wurcs(wurcs):
    url = "https://api.glycosmos.org/glycanformatconverter/2.10.4/wurcs2wurcs"
    try:
        response = requests.post(url, json=[wurcs])
        response.raise_for_status()
        data = response.json()
        if data and isinstance(data, list) and len(data) > 0:
            # Return the normalized WURCS string from the response.
            return data[0].get("WURCS")
        return None
    except Exception as e:
        logger.error(f"Failed to normalize WURCS: {str(e)}")
        return None  
    
def wurcs2glytoucan(wurcs):
    """Convert WURCS to GlyTouCan ID using GlyCosmos API.
    
    Args:
        wurcs (str): WURCS format string
        
    Returns:
        str: GlyTouCan ID or None if request fails
    """
    try:
        # URL encode the WURCS string
        encoded_wurcs = requests.utils.quote(wurcs)
        url = f"https://api.glycosmos.org/sparqlist/wurcs2gtcids?wurcs={encoded_wurcs}"
        
        # Make the request
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse response
        data = response.json()
        if data and isinstance(data, list) and len(data) > 0:
            return data[0].get("id")
            
        return None
        
    except Exception as e:
        logger.error(f"Failed to convert WURCS to GlyTouCan ID: {str(e)}")
        return None