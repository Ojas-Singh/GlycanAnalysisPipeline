import lib.config as config
import json
from rdflib import Graph, Literal, URIRef, Namespace, BNode
from rdflib.namespace import RDF, RDFS, XSD, DCTERMS, OWL
import os # Import os for file path handling

# --- Define Namespaces ---
# Using a more persistent URL for the custom ontology is recommended if possible
GS = Namespace("http://glycoshape.io/ontology/")
GSO = Namespace("http://glycoshape.io/resource/") # Namespace for specific instances/resources
GLYCORDF = Namespace("http://purl.jp/bio/12/glyco/glycan#")
GLYTOUCAN = Namespace("http://rdf.glytoucan.org/glycan/") # GlyTouCan RDF namespace
# Consider adding CHEBI or PUBCHEM if linking monosaccharides externally

# --- Helper Function to Add Literal if Value Exists ---
def add_literal(graph, subject, predicate, obj, datatype=None):
    """Adds a literal triple if the object value is not None or empty."""
    if obj is not None and obj != "":
        # Ensure boolean values are correctly typed
        if isinstance(obj, bool):
            graph.add((subject, predicate, Literal(obj, datatype=XSD.boolean)))
        # Use specific datatypes if provided
        elif datatype:
            try:
                # Attempt conversion for numeric types to handle strings like "2.0"
                if datatype in [XSD.integer, XSD.float, XSD.double, XSD.decimal]:
                    if datatype == XSD.integer:
                        literal_value = Literal(int(float(obj)), datatype=datatype) # Handle potential floats like "300.0" for integer
                    else:
                         literal_value = Literal(float(obj), datatype=datatype)
                else:
                    literal_value = Literal(str(obj), datatype=datatype) # Keep others as string
                graph.add((subject, predicate, literal_value))
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not convert value '{obj}' to {datatype} for {subject} {predicate}. Adding as string. Error: {e}")
                graph.add((subject, predicate, Literal(str(obj), datatype=XSD.string)))

        # Infer datatype if not provided
        else:
            if isinstance(obj, bool):
                 graph.add((subject, predicate, Literal(obj, datatype=XSD.boolean)))
            elif isinstance(obj, int):
                 graph.add((subject, predicate, Literal(obj, datatype=XSD.integer)))
            elif isinstance(obj, float):
                 graph.add((subject, predicate, Literal(obj, datatype=XSD.double))) # Use double for floats
            else:
                 # Default to string for anything else
                 graph.add((subject, predicate, Literal(str(obj), datatype=XSD.string)))


# --- Helper Function to Add Sequence ---
def add_sequence(graph, glycan_variant_uri, sequence_value, sequence_type_uri, sequence_format_label):
    """Adds a Glycosequence node for a given sequence string and format."""
    if sequence_value:
        # Use a URI for the sequence node based on the variant and format
        # Replace potentially problematic characters in label for URI
        safe_label = sequence_format_label.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
        seq_uri = URIRef(f"{str(glycan_variant_uri)}/sequence/{safe_label}")

        graph.add((glycan_variant_uri, GLYCORDF.has_glycosequence, seq_uri))
        graph.add((seq_uri, RDF.type, GLYCORDF.Glycosequence))
        # Use has_sequence for the actual string - ensure it's treated as a string
        add_literal(graph, seq_uri, GLYCORDF.has_sequence, sequence_value, XSD.string)
        # Link to the format type
        graph.add((seq_uri, GLYCORDF.in_carbohydrate_format, sequence_type_uri))
        # Optionally add a label to the format URI itself (useful for custom formats)
        if sequence_type_uri.startswith(str(GS)):
             add_literal(graph, sequence_type_uri, RDFS.label, sequence_format_label)


# --- Main Processing Function for a Single Glycan Variant ---
def process_glycan_variant(glycan_variant_uri, data, g):
    """Processes a single glycan variant (archetype, alpha, beta) and adds to the graph 'g'."""

    # Basic Type Information
    g.add((glycan_variant_uri, RDF.type, GS.GlycanVariant)) # Custom type for the specific variant
    g.add((glycan_variant_uri, RDF.type, GLYCORDF.Saccharide)) # It is a saccharide

    # --- Identifiers ---
    gs_id = data.get("ID") # This should be the ID of the main entry
    if gs_id:
        # Don't add gs:glycoShapeID here, it belongs to the main entry URI
        pass # The main entry URI already has this ID

    glytoucan_id = data.get("glytoucan")
    if glytoucan_id:
        add_literal(g, glycan_variant_uri, GS.glytoucanID, glytoucan_id)
        # Link to the canonical GlyTouCan RDF resource
        glytoucan_uri = GLYTOUCAN[glytoucan_id]
        g.add((glycan_variant_uri, OWL.sameAs, glytoucan_uri))
        # Also state it's an identifier using dcterms
        g.add((glycan_variant_uri, DCTERMS.identifier, Literal(glytoucan_id)))


    # --- Names and Labels ---
    add_literal(g, glycan_variant_uri, RDFS.label, data.get("name")) # Use rdfs:label for the primary name
    add_literal(g, glycan_variant_uri, GS.iupacName, data.get("iupac"))
    add_literal(g, glycan_variant_uri, GS.iupacExtendedName, data.get("iupac_extended"))
    add_literal(g, glycan_variant_uri, GS.glycamName, data.get("glycam"))
    add_literal(g, glycan_variant_uri, GS.oxfordName, data.get("oxford"))


    # --- Sequence Representations ---
    add_sequence(g, glycan_variant_uri, data.get("wurcs"), GLYCORDF.carbohydrate_format_wurcs, "WURCS")
    add_sequence(g, glycan_variant_uri, data.get("glycoct"), GLYCORDF.carbohydrate_format_glycoct, "GlycoCT")
    add_sequence(g, glycan_variant_uri, data.get("iupac"), GLYCORDF.carbohydrate_format_iupac_condensed, "IUPAC Condensed")
    add_sequence(g, glycan_variant_uri, data.get("iupac_extended"), GS.carbohydrate_format_iupac_extended, "IUPAC Extended")
    add_sequence(g, glycan_variant_uri, data.get("glycam"), GS.carbohydrate_format_glycam, "GLYCAM")
    add_sequence(g, glycan_variant_uri, data.get("smiles"), GS.carbohydrate_format_smiles, "SMILES")


    # --- Physical / Chemical Properties ---
    add_literal(g, glycan_variant_uri, GS.mass, data.get("mass"), XSD.double)
    add_literal(g, glycan_variant_uri, GS.hydrogenBondAcceptors, data.get("hbond_acceptor"), XSD.integer)
    add_literal(g, glycan_variant_uri, GS.hydrogenBondDonors, data.get("hbond_donor"), XSD.integer)
    add_literal(g, glycan_variant_uri, GS.rotatableBonds, data.get("rot_bonds"), XSD.integer)


    # --- Structural Features: Motifs ---
    motifs = data.get("motifs")
    if motifs and isinstance(motifs, list): # Ensure it's a list
        for motif in motifs:
             # Check if motif is a dictionary with expected keys
            if isinstance(motif, dict):
                motif_id = motif.get("motif")
                motif_label = motif.get("motif_label")
                if motif_id:
                    motif_uri = GSO["motif/" + motif_id]
                    g.add((glycan_variant_uri, GLYCORDF.has_motif, motif_uri))
                    g.add((motif_uri, RDF.type, GLYCORDF.Motif))
                    add_literal(g, motif_uri, DCTERMS.identifier, motif_id)
                    if motif_label:
                        add_literal(g, motif_uri, RDFS.label, motif_label)
            else:
                 print(f"Warning: Unexpected motif format found in {glycan_variant_uri}: {motif}")


    # --- Structural Features: Termini ---
    termini = data.get("termini")
    if termini and isinstance(termini, list):
        for terminus in termini:
            add_literal(g, glycan_variant_uri, GLYCORDF.has_terminal_residue, terminus)


    # --- Composition: Components ---
    components = data.get("components")
    if components and isinstance(components, dict):
        for mono_name, count in components.items():
            # Use Blank Node for component instance to avoid complex URI generation for now
            comp_node = BNode()
            g.add((glycan_variant_uri, GLYCORDF.has_component, comp_node))
            g.add((comp_node, RDF.type, GLYCORDF.Component))

            # Link component instance to the monosaccharide *type*
            mono_type_uri = GSO["monosaccharide/" + mono_name] # Example: http://glycoshape.io/resource/monosaccharide/Man
            g.add((comp_node, GLYCORDF.has_monosaccharide, mono_type_uri))
            add_literal(g, mono_type_uri, RDFS.label, mono_name) # Label the monosaccharide type URI

            # Add the count (cardinality)
            add_literal(g, comp_node, GLYCORDF.has_cardinality, count, XSD.integer)

    # Handle 'composition' field if it exists and is not null
    composition = data.get("composition")
    if composition:
         add_literal(g, glycan_variant_uri, GS.compositionString, composition)


    # --- Simulation Parameters ---
    add_literal(g, glycan_variant_uri, GS.simulationPackage, data.get("package"))
    add_literal(g, glycan_variant_uri, GS.simulationForcefield, data.get("forcefield"))
    add_literal(g, glycan_variant_uri, GS.simulationLength, data.get("length")) # Keep as string or convert? String is safer unless units are clear
    add_literal(g, glycan_variant_uri, GS.simulationTemperature, data.get("temperature"), XSD.double)
    add_literal(g, glycan_variant_uri, GS.simulationPressure, data.get("pressure"), XSD.double)
    add_literal(g, glycan_variant_uri, GS.simulationSaltConcentration, data.get("salt")) # Keep as string or numeric? Numeric if units (mM) are consistent


    # --- Simulation Results: Clusters ---
    clusters = data.get("clusters")
    if clusters and isinstance(clusters, dict):
        for cluster_label, percentage in clusters.items():
            # Use Blank Node for cluster result instance
            cluster_result_node = BNode()
            g.add((glycan_variant_uri, GS.hasClusterResult, cluster_result_node))
            g.add((cluster_result_node, RDF.type, GS.ClusterResult))

            # Add properties to the cluster result node
            safe_label = cluster_label.replace(" ", "_") # Make label safe for potential use in URI/queries
            add_literal(g, cluster_result_node, RDFS.label, cluster_label)
            add_literal(g, cluster_result_node, GS.clusterLabel, safe_label) # Store a safe version of the label
            add_literal(g, cluster_result_node, RDF.value, percentage, XSD.double) # Use rdf:value for the numeric percentage
            add_literal(g, cluster_result_node, GS.clusterPercentage, percentage, XSD.double) # Custom predicate


# --- Main Conversion Function ---
def convert_glycoshape_to_rdf(input_path, output_path):
    """
    Convert a GlycoShape JSON database (dict mapping IDs to entries) to RDF Turtle.
    """

    # Create master graph
    g_all = Graph()

    # Bind namespaces for readable output
    g_all.bind("gs", GS)
    g_all.bind("gso", GSO)
    g_all.bind("glycordf", GLYCORDF)
    g_all.bind("glytoucan", GLYTOUCAN)
    g_all.bind("dcterms", DCTERMS)
    g_all.bind("owl", OWL)
    g_all.bind("rdf", RDF)
    g_all.bind("rdfs", RDFS)
    g_all.bind("xsd", XSD)

    # Read JSON data
    print(f"Reading JSON data from: {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {input_path}: {e}")
        return None

    if not isinstance(data, dict):
        print(f"Error: Expected top-level JSON structure to be a dictionary (mapping IDs to entries), but found {type(data)}.")
        return None

    print(f"Processing {len(data)} entries...")
    entry_count = 0
    # Iterate through each main glycan entry in the database
    for main_glycan_id, entry_data in data.items():
        entry_count += 1
        print(f"Processing entry {entry_count}/{len(data)}: {main_glycan_id}")

        if not isinstance(entry_data, dict):
            print(f"Warning: Skipping entry '{main_glycan_id}' because its value is not a dictionary.")
            continue

        # Create the main resource URI for this GlycoShape entry
        main_entry_uri = GSO[main_glycan_id]
        g_all.add((main_entry_uri, RDF.type, GS.GlycoShapeEntry))
        add_literal(g_all, main_entry_uri, RDFS.label, f"GlycoShape Entry {main_glycan_id}")
        # Add the main ID as both dcterms:identifier and a specific gs:glycoShapeID
        add_literal(g_all, main_entry_uri, DCTERMS.identifier, main_glycan_id)
        add_literal(g_all, main_entry_uri, GS.glycoShapeID, main_glycan_id)

        archetype_uri = None # Keep track of archetype URI for linking anomers

        # Process each variant (archetype, alpha, beta) if it exists within the entry
        for variant_type in ["archetype", "alpha", "beta"]:
            if variant_type in entry_data and entry_data[variant_type] and isinstance(entry_data[variant_type], dict):
                variant_data = entry_data[variant_type]

                # Check if the variant data has its own ID and if it matches the main ID
                variant_id_check = variant_data.get("ID")
                if variant_id_check and variant_id_check != main_glycan_id:
                     print(f"Warning: ID mismatch for {main_glycan_id}/{variant_type}. Variant ID is '{variant_id_check}'. Using main ID '{main_glycan_id}' for URI.")
                     # Ensure the variant data dictionary retains the correct main ID for potential internal use by process_glycan_variant
                     # variant_data["ID"] = main_glycan_id # No, don't modify input data, just use main_glycan_id for URI

                # Create a URI for this specific variant
                variant_uri = GSO[f"{main_glycan_id}/{variant_type}"]

                # Link the main entry to its variant
                g_all.add((main_entry_uri, GS.hasVariant, variant_uri))

                # Add specific type and link predicate based on variant type
                if variant_type == "archetype":
                    g_all.add((variant_uri, RDF.type, GS.ArchetypeGlycan))
                    g_all.add((main_entry_uri, GS.hasArchetype, variant_uri))
                    archetype_uri = variant_uri # Store for later linking
                elif variant_type == "alpha":
                    g_all.add((variant_uri, RDF.type, GS.AlphaAnomerGlycan))
                    g_all.add((main_entry_uri, GS.hasAlphaAnomer, variant_uri))
                elif variant_type == "beta":
                    g_all.add((variant_uri, RDF.type, GS.BetaAnomerGlycan))
                    g_all.add((main_entry_uri, GS.hasBetaAnomer, variant_uri))

                # Process the detailed data for this variant, adding triples to g_all
                process_glycan_variant(variant_uri, variant_data, g_all)

                # Add relationships between variants if archetype exists
                if variant_type in ["alpha", "beta"] and archetype_uri:
                    g_all.add((variant_uri, GS.isAnomerOf, archetype_uri))
            # else:
            #     print(f"Info: Variant '{variant_type}' not found or is not valid data for entry '{main_glycan_id}'.")


    # Write the combined graph to file
    print(f"\nSerializing RDF graph to: {output_path}")
    try:
        g_all.serialize(destination=output_path, format='turtle')
        print(f"Successfully wrote RDF data to {output_path}")
    except Exception as e:
        print(f"Error serializing RDF graph: {e}")
        return None

    return g_all

def main():
    
    input_file = config.output_path / "GLYCOSHAPE.json"
    output_file = config.output_path / "GLYCOSHAPE_RDF.ttl"

    # Run the conversion
    print("\n--- Starting RDF Conversion ---")
    convert_glycoshape_to_rdf(input_file, output_file)
    print("--- RDF Conversion Finished ---")

if __name__ == "__main__":

    main()
    