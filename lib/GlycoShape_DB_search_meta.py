import lib.config as config
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_keywords(iupac_string: str) -> list:
    """
    Determines keywords based on the IUPAC string.
    """
    keywords = []
    if not iupac_string:
        return keywords

    # N-Glycans
    is_n_glycan = False
    if iupac_string.endswith('Man(b1-4)GlcNAc(b1-4)GlcNAc') or \
       iupac_string.endswith('Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc'):
        is_n_glycan = True
        keywords.append("N-Glycan")

        # Check for core fucosylation specifically for N-Glycans
        if iupac_string.endswith('Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc'):
            keywords.append("Fucosylated N-Glycan")

        # Oligomannose
        is_oligomannose = True
        if not (iupac_string.count('Man') >= 3):
            is_oligomannose = False
        if 'GlcNAc(b1-2)Man' in iupac_string:
            is_oligomannose = False
        if 'Gal(' in iupac_string:
            is_oligomannose = False
        if 'Neu5Ac(' in iupac_string:
            is_oligomannose = False
        if 'Xyl(' in iupac_string: 
            is_oligomannose = False
        if 'GlcNAc(b1-4)]Man' in iupac_string: 
            is_oligomannose = False
        if 'GlcNAc(b1-4)Man' in iupac_string and not 'Man(b1-4)GlcNAc(b1-4)GlcNAc' in iupac_string :
            is_oligomannose = False
        if 'GlcNAc(b1-6)[GlcNAc(b1-2)]Man' in iupac_string: 
            is_oligomannose = False
        if 'GlcNAc(b1-4)[GlcNAc(b1-2)]Man' in iupac_string: 
            is_oligomannose = False
        
        temp_iupac_for_oligomannose = iupac_string
        if temp_iupac_for_oligomannose.endswith('Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc'):
            temp_iupac_for_oligomannose = temp_iupac_for_oligomannose.replace('[Fuc(a1-6)]GlcNAc', 'GlcNAc')
        
        if 'Fuc(' in temp_iupac_for_oligomannose:
            is_oligomannose = False
        
        if is_oligomannose:
            keywords.append("Oligomannose")

        # Complex
        is_complex = False
        if ('Man(b1-4)GlcNAc(b1-4)GlcNAc' in iupac_string or
            'Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc' in iupac_string or
            'Man(b1-4)GlcNAc(b1-4)[Fuc(a1-3)]GlcNAc' in iupac_string) and \
           'GlcNAc(b1-2)Man' in iupac_string and \
           ('GlcNAc(b1-2)Man(a1-6)' in iupac_string or "GlcNAc(b1-2)]Man(a1-6)" in iupac_string) and \
           ('GlcNAc(b1-2)Man(a1-3)' in iupac_string or "GlcNAc(b1-2)]Man(a1-3)" in iupac_string):
            is_complex = True
            keywords.append("Complex N-Glycan")

        # Hybrid
        if not is_complex and \
           ('Man(b1-4)GlcNAc(b1-4)GlcNAc' in iupac_string or
            'Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc' in iupac_string or
            'Man(b1-4)GlcNAc(b1-4)[Fuc(a1-3)]GlcNAc' in iupac_string) and \
           ((('GlcNAc(b1-2)Man(a1-3)' in iupac_string or "GlcNAc(b1-2)]Man(a1-3)" in iupac_string) and
             not ('GlcNAc(b1-2)Man(a1-6)' in iupac_string or "GlcNAc(b1-2)]Man(a1-6)" in iupac_string)) or
            (('GlcNAc(b1-2)Man(a1-6)' in iupac_string or "GlcNAc(b1-2)]Man(a1-6)" in iupac_string) and
             not ('GlcNAc(b1-2)Man(a1-3)' in iupac_string or "GlcNAc(b1-2)]Man(a1-3)" in iupac_string))) and \
           iupac_string.count('Man') > 3: 
            if not is_oligomannose: 
                 keywords.append("Hybrid N-Glycan")

        # Antennary classification
        if not is_oligomannose: 
            b2_on_a3 = "GlcNAc(b1-2)Man(a1-3)" in iupac_string or "GlcNAc(b1-2)]Man(a1-3)" in iupac_string 
            b4_on_a3 = "GlcNAc(b1-4)Man(a1-3)" in iupac_string or "GlcNAc(b1-4)[Man(a1-3)" in iupac_string or "GlcNAc(b1-4)[GlcNAc(b1-2)]Man(a1-3)" in iupac_string
            b2_on_a6 = "GlcNAc(b1-2)Man(a1-6)" in iupac_string or "GlcNAc(b1-2)]Man(a1-6)" in iupac_string
            b6_on_a6 = "GlcNAc(b1-6)Man(a1-6)" in iupac_string or "GlcNAc(b1-6)[Man(a1-6)" in iupac_string or "GlcNAc(b1-6)[GlcNAc(b1-2)]Man(a1-6)" in iupac_string

            num_a3_glcnac_attachments = 0
            if b2_on_a3: num_a3_glcnac_attachments +=1
            if b4_on_a3: num_a3_glcnac_attachments +=1

            num_a6_glcnac_attachments = 0
            if b2_on_a6: num_a6_glcnac_attachments +=1
            if b6_on_a6: num_a6_glcnac_attachments +=1

            total_antennae_points = num_a3_glcnac_attachments + num_a6_glcnac_attachments
            has_branches_on_both_arms = num_a3_glcnac_attachments > 0 and num_a6_glcnac_attachments > 0

            if has_branches_on_both_arms:
                if total_antennae_points == 2: 
                    keywords.append("Biatennary N-Glycan")
                elif total_antennae_points == 3: 
                    keywords.append("Triantennary N-Glycan")
                elif total_antennae_points >= 4: 
                    keywords.append("Tetraantennary N-Glycan")

    # O-Glycans
    if iupac_string.endswith('GalNAc') or \
       'Gal(b1-3)GalNAc' in iupac_string or \
       'GlcNAc(b1-6)[Gal(b1-3)]GalNAc' in iupac_string or \
       'GlcNAc(b1-3)GalNAc' in iupac_string or \
       'GlcNAc(b1-6)[GlcNAc(b1-3)]GalNAc' in iupac_string:
        keywords.append("O-Glycan")

    # GAGs (Glycosaminoglycans)
    if 'GlcA(b1-3)Gal(b1-3)Gal(b1-4)Xyl' in iupac_string or \
       'IdoA' in iupac_string or \
       ('GlcA' in iupac_string and 'GlcNAc' in iupac_string) or \
       ('GlcA' in iupac_string and 'GalNAc' in iupac_string) or \
       ('GlcN' in iupac_string and 'GlcA' in iupac_string) or \
       ('GlcN' in iupac_string and 'IdoA' in iupac_string):
        keywords.append("GAG")
             
    return sorted(list(set(keywords))) # Return unique, sorted keywords

ganglioside_map = {
    "G40990LB": "GA2",
    "G00071MO": "GA1",
    "G01269WV": "GM1b",
    "G86874EI": "GD1c",
    "G40764VP": "GM3",
    "G65156XZ": "GM2",
    "G41876BL": "GM1b", 
    "G21518JT": "GD1a",
    "G97898ZO": "GT1a",
    "G10158SF": "GD3",
    "G98397EB": "GD2",
    "G08648UJ": "GT1b",
    "G71158TZ": "GQ1b",
    "G27476OM": "GT3",
    "G66281XG": "GT2",
    "G53140PB": "GT1c",
    "G54434YJ": "GQ1c",
    "G44362TS": "GP1c",
}

def update_meta(json_file_path: Path):
    """
    Update the search metadata in the given JSON file.
    """
    try:
        with open(json_file_path, 'r+') as f:
            data = json.load(f)
            
            archetype_data = data.get("archetype", {})
            archetype_iupac = archetype_data.get("iupac")
            archetype_oxford = archetype_data.get("oxford")
            archetype_glytoucan = archetype_data.get("glytoucan")
            alpha_data = data.get("alpha", {})
            alpha_glytoucan = alpha_data.get("glytoucan")
            beta_data = data.get("beta", {})
            beta_glytoucan = beta_data.get("glytoucan")

            
            if "search_meta" not in data:
                data["search_meta"] = {}
            
            # Initialize or get existing common names and keywords
            common_names = set(data["search_meta"].get("common_names", []))
            current_keywords = set(data["search_meta"].get("keywords", []))

            # Add Oxford name to common_names if it exists
            if archetype_oxford:
                common_names.add(archetype_oxford)

            # Add Ganglioside name from map if GlyTouCan ID matches
            # Add Ganglioside name from map if GlyTouCan ID matches (archetype, alpha, or beta)
            glytoucan_ids = [archetype_glytoucan, alpha_glytoucan, beta_glytoucan]
            for glytoucan_id in glytoucan_ids:
                if glytoucan_id and glytoucan_id in ganglioside_map:
                    common_names.add(ganglioside_map[glytoucan_id])
            
            data["search_meta"]["common_names"] = sorted(list(common_names))
            
            # Update keywords
            if archetype_iupac:
                new_keywords = get_keywords(archetype_iupac)
                for kw in new_keywords:
                    current_keywords.add(kw)
                data["search_meta"]["keywords"] = sorted(list(current_keywords))
                logger.info(f"Updated keywords for {json_file_path.name}: {data['search_meta']['keywords']}")
            else:
                logger.warning(f"No archetype IUPAC string found in {json_file_path.name}. Skipping keyword update.")

            # Move file pointer to the beginning to overwrite
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()
            if common_names: # Log if common names were added/updated
                 logger.info(f"Updated common names for {json_file_path.name}: {data['search_meta']['common_names']}")

    except FileNotFoundError:
        logger.error(f"File not found: {json_file_path}")
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from file: {json_file_path}")
    except Exception as e:
        logger.error(f"An error occurred while updating {json_file_path.name}: {e}")

def main():
    """
    Main function to update the search metadata.
    """
    data_dir = Path(config.output_path)
    
    if not data_dir.exists():
        logger.error(f"Output directory {data_dir} does not exist.")
        return

    logger.info(f"Starting metadata update in directory: {data_dir}")
    for folder in data_dir.iterdir():
        if folder.is_dir() and folder.name.startswith("GS"):
            json_path = folder / "data.json"
            if json_path.exists():
                logger.info(f"Processing {folder.name}...")
                update_meta(json_path)
            else:
                logger.warning(f"data.json not found in {folder.name}")
    logger.info("Metadata update process finished.")

if __name__ == "__main__":
    main()