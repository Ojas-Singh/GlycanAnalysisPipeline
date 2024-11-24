import json

# Load the JSON data from the provided file
file_path = '/mnt/database/DB_beta/GLYCOSHAPE.json'

# Read the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

# Modify the extraction to include 'null' if 'glytoucan_id' doesn't exist
iupac_glytoucan_dict_with_null = {}
for key, value in data.items():
    iupac_name = value.get('iupac', None)
    glytoucan_id = value.get('glytoucan_id', None)
    
    # Only add to dictionary if iupac name exists
    if iupac_name:
        iupac_glytoucan_dict_with_null[iupac_name] = glytoucan_id if glytoucan_id else None

# Save the updated extracted data to a new JSON file
output_path_with_null = '/mnt/database/DB_beta/iupac_glytoucan_mapping.json'
with open(output_path_with_null, 'w') as output_file:
    json.dump(iupac_glytoucan_dict_with_null, output_file)

