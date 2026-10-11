
import re
import sys
import shutil
import os
import requests
import collections
import polars as pl
import math
from pathlib import Path
from io import BytesIO
from lib.storage import get_storage_manager
from glypy.io import iupac as glypy_iupac

# Import glycowork functions (thankyou daniel for this amazing package!)
from glycowork.motif.draw import GlycoDraw
from glycowork.motif.processing import canonicalize_iupac
from glycowork.motif.annotate import annotate_glycan
from glycowork.motif.annotate import get_molecular_properties
from glycowork.motif.processing import IUPAC_to_SMILES
from glycowork.motif.annotate import get_terminal_structures

import lib.config as config
from lib import lookups
import logging
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import json
from glycowork.motif.graph import compare_glycans



def get_oxford(iupac, oxford_dict='oxford.json'):
    # Convert to absolute path relative to this file's location
    if not Path(oxford_dict).is_absolute():
        oxford_dict = Path(__file__).parent / oxford_dict
    
    storage = get_storage_manager()
    with storage.open(oxford_dict, 'r') as f:
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
    # Prefer the pinned local converter so WURCS generation is reproducible
    # and benefits from crabWURCS fixes even when GlyCosmos also accepts the
    # input.  GlyCosmos remains the fallback and is still used to resolve the
    # corresponding GlyTouCan accession.
    local_wurcs = crabwurcs_convert(iupac_condensed, "iupac-condensed", "wurcs")
    if local_wurcs:
        return wurcs2glytoucan(local_wurcs), local_wurcs

    encoded_iupac = requests.utils.quote(str(iupac_condensed or "").strip(), safe="")
    url = f"https://api.glycosmos.org/glycanformatconverter/2.10.0/iupaccondensed2wurcs/{encoded_iupac}"
    
    try:
        # Make the API request
        response = lookups.get(url, lookup="iupac2wurcs", subject=iupac_condensed, timeout=30)
        # Raise an exception if the request failed
        response.raise_for_status()
        
        # Parse the JSON response
        data = response.json()
        return data.get("id"), data.get("WURCS")
    except requests.exceptions.RequestException as e:
        # Handle any request exceptions
        logger.error(f"Failed to convert IUPAC to WURCS/GlyTouCan for {iupac_condensed}: {e}")
        return None, crabwurcs_convert(iupac_condensed, "iupac-condensed", "wurcs")
    except (KeyError, ValueError) as e:
        # Handle unexpected response structure
        logger.error(f"Unexpected IUPAC to WURCS response for {iupac_condensed}: {e}")
        return None, crabwurcs_convert(iupac_condensed, "iupac-condensed", "wurcs")


def crabwurcs_convert(value, source_format, target_format, timeout=30):
    """Convert a glycan notation locally with crabWURCS.

    Returns ``None`` when crabWURCS is unavailable or cannot represent the
    input, allowing callers to retain their existing remote/Java fallbacks.
    """
    text_value = str(value or "").strip()
    if not text_value:
        return None

    command = [
        config.crabwurcs_executable,
        "convert",
        "--from",
        source_format,
        "--to",
        target_format,
    ]
    try:
        result = subprocess.run(
            command,
            input=text_value,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.debug("crabWURCS conversion unavailable: %s", exc)
        if isinstance(exc, FileNotFoundError):
            lookups.tool("crabwurcs", False, f"executable not found: {config.crabwurcs_executable}")
        return None

    lookups.tool("crabwurcs", True, str(config.crabwurcs_executable))
    converted = result.stdout.strip()
    if result.returncode == 0 and converted:
        return converted

    detail = result.stderr.strip() or converted or f"exit code {result.returncode}"
    logger.debug(
        "crabWURCS %s -> %s conversion failed: %s",
        source_format,
        target_format,
        detail,
    )
    return None

def wurcs2extendediupac(wurcs):
    """
    Converts WURCS format to extended IUPAC using GlyCosmos API.
    
    Args:
        wurcs (str): WURCS format glycan string
        
    Returns:
        str: Extended IUPAC format or None if request fails
    """
    local_iupac = crabwurcs_convert(wurcs, "wurcs", "iupac-extended")
    if local_iupac:
        return local_iupac

    url = "https://api.glycosmos.org/glycanformatconverter/2.10.0/wurcs2iupacextended"
    try:
        response = lookups.post(url, lookup="wurcs2extendediupac", subject=wurcs, json={"input": wurcs}, timeout=30)
        response.raise_for_status()
        response.encoding = 'utf-8'
        payload = response.json()
        raw_iupac = payload.get("IUPACextended")
        if not raw_iupac:
            raise ValueError(f"response did not contain IUPACextended: {payload}")
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
        response = lookups.get(url, lookup="wurcsRDFmatch", subject=wurcs)
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
        lookups.bound_pubchem()
        with lookups.labelled("pubchem_properties", iupac):
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
        # Each terminus once, in the order glycowork returns them.  (A set here gave an order that
        # changed from one process to the next, and with it the bytes of data.json.)
        termini_mod = list(dict.fromkeys(termini))
        return termini_mod
    else:
        return None

# Function to get MD simulation info for a glycan submission...
def get_md_info(glycan: str) -> tuple:
    """Get MD simulation info for a glycan submission.
    
    Args:
        glycan: Glycan name
        
    Returns:
        Tuple of (ID, length, package, FF, temp, pressure, salt, contributor)
    """
    if config.use_pocketbase:
        try:
            from lib.pocketbase import extract_md_info, get_pocketbase_client

            client = get_pocketbase_client()
            if client.is_available():
                record = client.get_record_by_glycam_name(glycan)
                if record is not None:
                    logger.info(f"Loaded MD info for {glycan} from PocketBase")
                    return extract_md_info(record)
                logger.warning(f"No PocketBase submission metadata found for {glycan}; falling back to CSV")
            else:
                logger.warning("PocketBase is configured but unavailable; falling back to CSV")
        except Exception as e:
            logger.warning(f"Failed to load MD info for {glycan} from PocketBase: {e}")

    # Cache the inventory DataFrame
    if not hasattr(get_md_info, '_inventory_df'):
        try:
            storage_manager = get_storage_manager()
            inv_path = str(config.inventory_path)
            if not storage_manager.exists(inv_path):
                candidates = []
                try:
                    for obj in storage_manager.list_files("", "*"):
                        name = obj.as_posix() if hasattr(obj, 'as_posix') else str(obj)
                        if name.lower().endswith("glycoshape_inventory.csv"):
                            candidates.append(name)
                except Exception:
                    pass

                if candidates:
                    candidates.sort(key=lambda s: len(s))
                    inv_path = candidates[0]
                else:
                    logger.error(f"Inventory file not found: {inv_path}")
                    return 0, 0, 0, 0, 0, 0, 0, "0"

            csv_bytes = storage_manager.read_binary(inv_path)
            df = pl.read_csv(BytesIO(csv_bytes))

            # Keep only rows with a non-null Timestamp
            if 'Timestamp' in df.columns:
                df = df.filter(pl.col('Timestamp').is_not_null())

            get_md_info._inventory_df = df
            logger.info(f"Loaded inventory from {inv_path}")
        except Exception as e:
            logger.error(f"Failed to load inventory: {str(e)}")
            return 0, 0, 0, 0, 0, 0, 0, "0"
    
    try:
        df = get_md_info._inventory_df

        # Use positional column access similar to the original implementation
        cols = list(df.columns)
        if len(cols) < 11:
            logger.error("Inventory CSV does not have the expected number of columns")
            return 0, 0, 0, 0, 0, 0, 0, "0"

        glycan_col = cols[3]
        glycan_data = df.filter(pl.col(glycan_col) == glycan)

        if glycan_data.height == 0:
            logger.warning(f"No inventory data found for {glycan}")
            return 0, 0, 0, 0, 0, 0, 0, "0"

        # Get first matching row values
        def _get_val(df_row, col_name):
            try:
                val = df_row[0][col_name]
            except Exception:
                val = None
            # Convert Polars nulls and NaNs to None
            if val is None:
                return None
            try:
                if isinstance(val, float) and math.isnan(val):
                    return None
            except Exception:
                pass
            return val

        row0 = glycan_data.head(1)

        ID = _get_val(row0, cols[0])
        length = _get_val(row0, cols[5])
        package = _get_val(row0, cols[6])
        FF = _get_val(row0, cols[7])
        temp = _get_val(row0, cols[8])
        pressure = _get_val(row0, cols[9])
        salt = _get_val(row0, cols[10])
        contributor = _get_val(row0, cols[2])

        # Normalize to expected return types (strings or None)
        ID = None if ID is None else str(ID)
        length = None if length is None else str(length)
        package = None if package is None else package
        FF = None if FF is None else FF
        temp = None if temp is None else str(temp)
        pressure = None if pressure is None else str(pressure)
        salt = None if salt is None else salt
        contributor = None if contributor is None else contributor

        return ID, length, package, FF, temp, pressure, salt, contributor

    except Exception as e:
        logger.error(f"Error processing inventory data for {glycan}: {str(e)}")
        return 0, 0, 0, 0, 0, 0, 0, "0"



def glytoucan2glygen(glytoucan_ac):
    """Get Glygen data for a GlyTouCan accession number.

    Args:
        glytoucan_ac (str): GlyTouCan accession number
        
    Returns:
        dict: Glygen data or None if request fails
    """
    url = f"https://api.glygen.org/glycan/detail/{glytoucan_ac}/"

    try:
        response = lookups.post(
            url,
            lookup="glytoucan2glygen",
            subject=glytoucan_ac,
            headers={'accept': 'application/json', 'Content-Type': 'application/json'},
            json={'glytoucan_ac': glytoucan_ac},
            verify=False,
            timeout=30,
        )
        if response.status_code == 500:
            try:
                payload = response.json()
            except ValueError:
                payload = None
            reason = payload.get("reason", {}) if isinstance(payload, dict) else {}
            error_list = payload.get("error_list", []) if isinstance(payload, dict) else []
            known_missing = (
                reason.get("type") in {"discontinued_in_glygen", "non-existent-record"}
                or any(
                    isinstance(item, dict) and item.get("error_code") == "non-existent-record"
                    for item in error_list
                )
            )
            if known_missing:
                logger.info(
                    "GlyGen has no active record for %s: %s",
                    glytoucan_ac,
                    reason.get("description") or reason.get("type") or "non-existent record",
                )
                return payload
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
        response = lookups.get(url, lookup="glytoucan2glycosmos", subject=glytoucan_id)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to get GlyCosmos taxonomy data: {str(e)}")
        return None
    

def _detect_local_java_home():
    """Return a usable local Java home bundled with the pipeline, if present."""
    java_root = Path(__file__).resolve().parents[1] / ".java"
    direct_java = java_root / "bin" / "java"
    if direct_java.exists():
        return java_root

    for java_bin in sorted(java_root.glob("*/bin/java")):
        if java_bin.exists():
            return java_bin.parent.parent

    return None


#: The MolWURCS fallback is a local Java program; a run that hangs must not hold the static stage.
MOLWURCS_TIMEOUT_SECONDS = 120


def smiles2wurcs(smiles):
    crabwurcs_result = crabwurcs_convert(smiles, "smiles", "wurcs")
    if crabwurcs_result:
        return crabwurcs_result

    jar_path = Path(__file__).parent / "MolWURCS.jar"
    env = os.environ.copy()
    java_home = env.get("JAVA_HOME", "").strip()
    if not java_home:
        detected_java_home = _detect_local_java_home()
        if detected_java_home:
            java_home = str(detected_java_home)
    if java_home:
        lib_paths = [str(Path(java_home) / "lib"), str(Path(java_home) / "lib" / "server")]
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join([*lib_paths, existing]) if existing else ":".join(lib_paths)
        env["PATH"] = f"{Path(java_home) / 'bin'}:{env.get('PATH', '')}"
    try:
        result = subprocess.run(
            ["java", "-jar", str(jar_path), "--in", "smi", "--out", "wurcs", smiles],
            capture_output=True,
            text=True,
            check=True,
            env=env,
            timeout=MOLWURCS_TIMEOUT_SECONDS,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to convert SMILES to WURCS: {e.stderr}")
        return None
    except subprocess.TimeoutExpired:
        logger.error(f"MolWURCS did not finish in {MOLWURCS_TIMEOUT_SECONDS} seconds")
        return None


GLYCOSMOS_SPARQL_ENDPOINT = "https://ts.glycosmos.org/sparql"


#: Answers of :func:`glytoucan2iupac`.  Only sequences that were found are kept: a service that was
#: down (or lookups switched off) for one entry must not answer ``None`` for the rest of the process.
_GLYTOUCAN_IUPAC_CACHE = {}


def glytoucan2iupac(glytoucan_id):
    """Resolve a GlyTouCan accession to condensed IUPAC via GlyCosmos RDF.

    Motif accessions use the same GlyTouCan identifier space as ordinary
    glycans (for example, ``G00026MO``).  The motif endpoint returns the
    identifier and label but not the sequence, so the RDF endpoint is queried
    for the condensed IUPAC sequence.

    ``None`` is returned for an empty accession, an empty result, malformed
    JSON, or a network/API failure.  The caller can therefore retain the motif
    record and serialize an explicit ``iupac: null`` value.
    """
    glytoucan_id = str(glytoucan_id or "").strip()
    if not glytoucan_id:
        return None
    if glytoucan_id in _GLYTOUCAN_IUPAC_CACHE:
        return _GLYTOUCAN_IUPAC_CACHE[glytoucan_id]

    # JSON quoting keeps the accession safe when embedded in the SPARQL
    # string while retaining compatibility with the public endpoint.
    accession_literal = json.dumps(glytoucan_id)
    query = f"""
PREFIX glycan: <http://purl.jp/bio/12/glyco/glycan#>
PREFIX glytoucan: <http://www.glytoucan.org/glyco/owl/glytoucan#>
SELECT DISTINCT ?sequence WHERE {{
  ?saccharide glytoucan:has_primary_id ?accession .
  FILTER (STR(?accession) = {accession_literal})
  ?saccharide glycan:has_glycosequence ?glycosequence .
  ?glycosequence glycan:has_sequence ?sequence .
  ?glycosequence glycan:in_carbohydrate_format
      glycan:carbohydrate_format_iupac_condensed .
}}
LIMIT 1
"""

    try:
        response = lookups.get(
            GLYCOSMOS_SPARQL_ENDPOINT,
            lookup="glytoucan2iupac",
            subject=glytoucan_id,
            params={"query": query},
            headers={"Accept": "application/sparql-results+json"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        bindings = payload.get("results", {}).get("bindings", [])
        if not isinstance(bindings, list) or not bindings:
            logger.warning("No condensed IUPAC found for GlyTouCan accession %s", glytoucan_id)
            return None

        sequence = bindings[0].get("sequence", {})
        value = sequence.get("value") if isinstance(sequence, dict) else None
        if not value:
            return None
        if len(_GLYTOUCAN_IUPAC_CACHE) >= 4096:
            _GLYTOUCAN_IUPAC_CACHE.clear()
        _GLYTOUCAN_IUPAC_CACHE[glytoucan_id] = str(value).strip()
        return _GLYTOUCAN_IUPAC_CACHE[glytoucan_id]
    except Exception as e:
        logger.warning("Failed to get condensed IUPAC for %s: %s", glytoucan_id, e)
        return None


def glytoucan2motif(glytoucan_id):
    """Get glycan motif data from GlyCosmos and enrich it with condensed IUPAC.
    
    Args:
        glytoucan_id (str): GlyTouCan accession number
        
    Returns:
        list: Motif data, with an ``iupac`` key on each motif object, or None
        if the motif request fails
    """
    url = f"https://api.glycosmos.org/sparqlist/get_glycan_motif?id={glytoucan_id}"
    
    try:
        response = lookups.get(url, lookup="glytoucan2motif", subject=glytoucan_id)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            return payload

        enriched = []
        for motif in payload:
            if not isinstance(motif, dict):
                enriched.append(motif)
                continue

            motif_data = dict(motif)
            motif_id = str(motif_data.get("motif") or "").strip()
            if not str(motif_data.get("iupac") or "").strip():
                motif_data["iupac"] = glytoucan2iupac(motif_id) if motif_id else None
            else:
                # Make the output schema explicit even when an upstream
                # response already provides the sequence.
                motif_data.setdefault("iupac", None)
            enriched.append(motif_data)
        return enriched
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
    # Define the WURCS pattern - allow empty annotation section
    pattern = re.compile(
        r'^WURCS=2\.0/(\d+),(\d+),(\d+)/(\[.*?\]+)/([^/]*)/([^/]*)$'
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


    return (
        normalize_wurcs(archetype_wurcs) or archetype_wurcs,
        normalize_wurcs(alpha_wurcs) or alpha_wurcs,
        normalize_wurcs(beta_wurcs) or beta_wurcs,
    )

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
        response = lookups.post(url, lookup="validate_wurcs", subject=wurcs, json=[wurcs])
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
        response = lookups.post(url, lookup="normalize_wurcs", subject=wurcs, json=[wurcs])
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
    """Resolve a WURCS string to a GlyTouCan ID using GlyCosmos.

    GlyTouCan stores canonical WURCS strings, while valid WURCS produced by
    converters may differ only in residue/linkage ordering.  The historical
    ``wurcs2gtcids`` lookup is an exact-string query and reports those valid
    aliases as missing.  Ask the format converter first because it
    canonicalizes the structure and includes the matching accession.
    
    Args:
        wurcs (str): WURCS format string
        
    Returns:
        str: GlyTouCan ID or None if request fails
    """
    wurcs = str(wurcs or "").strip()
    if not wurcs:
        return None

    errors = []

    try:
        response = lookups.post(
            "https://api.glycosmos.org/glycanformatconverter/2.10.4/wurcs2wurcs",
            lookup="wurcs2glytoucan",
            subject=wurcs,
            json=[wurcs],
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            glytoucan_id = str(data[0].get("id") or "").strip()
            if glytoucan_id:
                return glytoucan_id
    except Exception as exc:
        errors.append(f"canonical lookup: {exc}")

    # Retain the exact-match endpoint as a fallback in case the format
    # converter accepts the WURCS but does not include accession metadata.
    try:
        response = lookups.get(
            "https://api.glycosmos.org/sparqlist/wurcs2gtcids",
            lookup="wurcs2glytoucan_exact",
            subject=wurcs,
            params={"wurcs": wurcs},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list):
            for match in data:
                if not isinstance(match, dict):
                    continue
                glytoucan_id = str(match.get("id") or "").strip()
                if glytoucan_id:
                    return glytoucan_id
    except Exception as exc:
        errors.append(f"exact lookup: {exc}")

    if errors:
        logger.error(
            "Failed to convert WURCS to GlyTouCan ID (%s)",
            "; ".join(errors),
        )
    return None


def wurcs2glytoucan_many(wurcs_values, batch_size=100):
    """Resolve multiple WURCS strings with GlyCosmos' batch converter.

    Returns a mapping for every non-empty input.  Values are ``None`` only
    when no accession could be found.  Malformed or partial batch responses
    fall back to the scalar resolver rather than silently misaligning IDs.
    """
    unique_wurcs = list(dict.fromkeys(
        str(wurcs or "").strip()
        for wurcs in wurcs_values
        if str(wurcs or "").strip()
    ))
    results = {}

    for start in range(0, len(unique_wurcs), batch_size):
        batch = unique_wurcs[start:start + batch_size]
        try:
            response = lookups.post(
                "https://api.glycosmos.org/glycanformatconverter/2.10.4/wurcs2wurcs",
                lookup="wurcs2glytoucan_many",
                subject=f"{len(batch)} WURCS",
                json=batch,
                timeout=60,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list) or len(payload) != len(batch):
                raise ValueError(
                    f"expected {len(batch)} converter results, got "
                    f"{len(payload) if isinstance(payload, list) else type(payload).__name__}"
                )

            for wurcs, item in zip(batch, payload):
                glytoucan_id = (
                    str(item.get("id") or "").strip()
                    if isinstance(item, dict)
                    else ""
                )
                results[wurcs] = glytoucan_id or None
        except Exception as exc:
            logger.warning(
                "Batch WURCS-to-GlyTouCan lookup failed for %d entries; "
                "falling back to individual lookup: %s",
                len(batch),
                exc,
            )
            for wurcs in batch:
                results[wurcs] = wurcs2glytoucan(wurcs)

    return results
