import config
from pathlib import Path
import shutil
import logging
from datetime import datetime
import zipfile
import glob
import json
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

faq_dict = {
    "What is GlycoShape?" : "GlycoShape is an OA database of glycans 3D structural data and information that can be downloaded or used with Re-Glyco to rebuild glycoproteins from the RCSB PDB or EMBL-EBI AlphaFold repositories.",
    "How to search?":"You can search by GlyTouCan ID, IUPAC, GLYCAM, WURCS, SMILES or you can draw your own glycan using draw' button in search bar and search the closest match from our database.",
    "What are clusters?":"Clusters and their centroids are crucial for understanding the dynamic behavior of glycan structures in our database. A cluster groups similar conformations of glycans based on molecular dynamics simulations, simplifying the complex data from these simulations. The cluster centroid represents the most typical conformation within its cluster, offering a clear view of probable glycan shapes and orientations. This approach helps us explore and quantify the glycan's conformational space, enhancing our understanding of its biological interactions and functions.",
    "How are clusters calculated?" : "The conformational ensembles from multi-microsecond MD simulations are clustered into representative conformations. To do this, a principal component analysis (PCA) is conducted on the trajectory for dimensionality reduction. Clusters from the PCA are then identified using a Gaussian mixture model (GMM), with representative structures for each conformational cluster identified from the kernel density (more details at The conformational ensembles from multi-microsecond MD simulations are clustered into representative conformations. To do this, a principal component analysis (PCA) is conducted on the trajectory for dimensionality reduction. Clusters from the PCA are then identified using a Gaussian mixture model (GMM), with representative structures for each conformational cluster identified from the kernel density (more details at https://doi.org/10.1101/2023.12.11.571101 ). ",
    "What is Re-Glyco?" : "Re-Glyco is a tool we designed to restore the missing glycosylation on glycoproteins deposited in the RCSB PDB, the EBI-EMBL AlphaFold protein structure database or on your own structure file in PDB format.",
    "What is GlcNAc Scan?" : "The ability of Re-Glyco to resolve steric clashes can be used within GlycoShape also to assess the potential occupancy of N-glycosylation sites through an implementation we called ‘GlcNAc Scanning’. Where Re-Glyco will try to fit a single GlcNAc monosaccharide into all the NXS/T sequons in the protein. The process outputs a list of sequons that passed the test, marked with a simple ‘yes’ or ‘no’ label.",
    "Having Performance Issue?" : "The website is optimized for the Chrome browser. If you are using a different browser and experiencing slowdowns while using Re-Glyco, we recommend trying Chrome.",
    "Advanced (Site-by-Site) Glycosylation" : "This option allows users to select different glycans for different residues. It is intended for use with up to 5-10 residues at a time, as using it with more may cause your browser to slow down. If you wish to glycosylate more than 10 residues, we recommend using our API through a Python script (more information at https://glycoshape.org/api-docs).",
    "What is the Re-Glyco Ensemble?" : "The Ensemble option becomes available only after a successful Re-Glyco run, appearing next to the download PDB button. It outputs a multiframe PDB of the glycoprotein, along with the SASA (Solvent Accessible Surface Area) of the protein, taking into account the influence of the glycans.",
    "Experiencing timeout with Re-Glyco job?":"We have set a 4000-second hard timeout for any job with Re-Glyco. If you wish to process an extremely large system, we recommend contacting us directly via email at elisa.fadda@soton.ac.uk. We would love to help and build your system.",
    "I would like a new feature!":"Please contact us at elisa.fadda@soton.ac.uk, we would be glad to hear any feedback or feature requests."
}


def create_glycoshape_archive(dir: Path, output_file: str = "GlycoShape.zip") -> None:
    """Create GlycoShape archive from directory contents.
    
    Args:
        dir1: Path to first database directory
        dir2: Path to second database directory
        output_file: Name of final zip file
    """
    try:
        # Convert to Path objects
        dir = Path(dir)
        output_path = dir / output_file
        
        # Remove existing archive if present
        if output_path.exists():
            output_path.unlink()
            logger.info(f"Removed existing {output_file}")

        # Process each glycan directory
        directories = list(dir.glob("*/"))
        
        for directory in directories:
            if not directory.is_dir():
                continue
                
            glycan = directory.name
            logger.info(f"Processing Zip for {glycan}")
            
            # Remove existing zip if present
            glycan_zip = directory / f"{glycan}.zip"
            if glycan_zip.exists():
                glycan_zip.unlink()
            
            # Files to include
            files_to_zip = [
                directory / "data.json",
                directory / "snfg.svg",
                directory / "PDB_format_HETATM",
                directory / "CHARMM_format_HETATM", 
                directory / "GLYCAM_format_HETATM",
                directory / "PDB_format_ATOM",
                directory / "CHARMM_format_ATOM",
                directory / "GLYCAM_format_ATOM"
            ]
            
            # Create zip for glycan
            with zipfile.ZipFile(glycan_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file_path in files_to_zip:
                    if file_path.exists():
                        if file_path.is_dir():
                            # Handle directories
                            for file in file_path.rglob('*'):
                                if file.is_file():
                                    arcname = file.relative_to(directory)
                                    zf.write(file, arcname)
                        else:
                            # Handle files
                            zf.write(file_path, file_path.name)

        # Create final archive
        logger.info("Creating final zip file")
        
        # Copy all zip files to temp location
        temp_zips = []
        for src_dir in [dir]:
            for zip_file in src_dir.rglob("*.zip"):
                if zip_file.name != output_file:
                    shutil.copy2(zip_file, dir)
                    temp_zips.append(dir / zip_file.name)

        # Create final archive
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as final_zip:
            for zip_file in temp_zips:
                final_zip.write(zip_file, zip_file.name)

        # Cleanup
        logger.info("Cleaning up temporary files")
        for temp_zip in temp_zips:
            temp_zip.unlink()

        logger.info(f"Successfully created {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to create archive: {str(e)}")
        raise


def create_glycoshape_json(dir: Path, output_file: str = "GLYCOSHAPE.json") -> None:
    """Create consolidated JSON file from all data.json files in subdirectories.
    
    Args:
        dir: Path to database directory
        output_file: Name of output JSON file
    """
    
    try:
        dir = Path(dir)
        output_path = dir / output_file
        consolidated_data = {}
        
        # Process each glycan directory
        for directory in dir.glob("*/"):
            if not directory.is_dir():
                continue
                
            glycan = directory.name
            json_path = directory / "data.json"
            
            if json_path.exists():
                logger.info(f"Processing JSON for {glycan}")
                with open(json_path) as f:
                    consolidated_data[glycan] = json.load(f)
                    
        # Write consolidated JSON
        logger.info(f"Writing consolidated JSON to {output_file}")
        with open(output_path, 'w') as f:
            json.dump(consolidated_data, f, indent=2, ensure_ascii=False, sort_keys=True)
            
        logger.info("Successfully created consolidated JSON file")
        
    except Exception as e:
        logger.error(f"Failed to create consolidated JSON: {str(e)}")
        raise

def save_faq_json(dir: Path, faq_data: dict, output_file: str = "faq.json") -> None:
    """Save FAQ data to JSON file.
    
    Args:
        dir: Path to output directory
        faq_data: Dictionary containing FAQ data
        output_file: Name of output JSON file
    """
    try:
        dir = Path(dir)
        output_path = dir / output_file
        
        logger.info(f"Writing FAQ JSON to {output_file}")
        with open(output_path, 'w') as f:
            json.dump(faq_data, f, indent=2)
            
        logger.info("Successfully created FAQ JSON file")
        
    except Exception as e:
        logger.error(f"Failed to create FAQ JSON: {str(e)}")
        raise
def wurcs_registration(dir: Path, file: str = "GLYCOSHAPE.json", output_file: str = "missing_glytoucan.txt") -> None:

    # Load the JSON data
    file_path = dir / file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Initialize a list to store the output
    output_lines = []

    # Iterate through the data to extract relevant information
    for key, glycan in data.items():
        wurcs_values = {
            "wurcs": glycan.get("wurcs"),
            "wurcs_alpha": glycan.get("wurcs_alpha"),
            "wurcs_beta": glycan.get("wurcs_beta")
        }
        glytoucan_values = {
            "glytoucan": glycan.get("glytoucan"),
            "glytoucan_alpha": glycan.get("glytoucan_alpha"),
            "glytoucan_beta": glycan.get("glytoucan_beta")
        }

        # Write non-null wurcs if corresponding glytoucan is null
        for wurcs_key, wurcs_value in wurcs_values.items():
            glytoucan_key = wurcs_key.replace("wurcs", "glytoucan")
            
            if wurcs_value and not glytoucan_values.get(glytoucan_key):
                output_lines.append(f"{wurcs_value}")

    # Write the output to a text file
    output_file_path = dir / output_file
    with open(output_file_path, 'w') as output_file:
        output_file.write("\n".join(output_lines))

    output_file_path

def submit_wurcs(contributor_id: str, api_key: str, file_path: Path) -> None:
    """Submit WURCS data to GlyTouCan API.
    
    Args:
        contributor_id: Contributor ID for authentication
        api_key: API key for authentication
        file_path: Path to the text file containing WURCS data
    """
    url = "https://glytoucan.org/api/bulkload/wurcs"
    try:
        with open(file_path, 'rb') as file:
            response = requests.post(
                url,
                files={"file": file},
                auth=(contributor_id, api_key)
            )
        
        if response.status_code == 200:
            logger.info("Successfully submitted WURCS data")
        else:
            logger.error(f"Failed to submit WURCS data: {response.status_code} {response.text}")
            response.raise_for_status()
    
    except Exception as e:
        logger.error(f"Error during WURCS submission: {str(e)}")
        raise

def total_simulation_length(file_path: Path) -> float:
    """Sum the 'length' values in a JSON file.
    
    Args:
        file_path: Path to the JSON file
    
    Returns:
        The sum of all 'length' values as a float
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        total_length = 0.0
        for entry in data.values():
            length = entry.get("length")
            if length is not None:
                total_length += float(length)
        
        return total_length
    
    except Exception as e:
        logger.error(f"Failed to sum lengths in JSON: {str(e)}")
        raise

if __name__ == "__main__":
    save_faq_json(
        dir=config.output_path,
        faq_data=faq_dict,
    )
    create_glycoshape_json(
        dir=config.output_path,)
    total_simulation_length(file_path=config.output_path / "GLYCOSHAPE.json")
    create_glycoshape_archive(
        dir=config.output_path,
    )
    wurcs_registration(
        dir=config.output_path,
        file="GLYCOSHAPE.json",
        output_file="missing_glytoucan.txt"
    )
    # submit_wurcs(
    #     contributor_id=config.contributor_id,
    #     api_key=config.api_key, file_path=config.output_path / "missing_glytoucan.txt")