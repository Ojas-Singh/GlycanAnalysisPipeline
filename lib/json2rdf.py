import lib.config as config
import json
from rdflib import Graph, Literal, URIRef, Namespace, BNode
from rdflib.namespace import RDF, RDFS, XSD, DCTERMS, OWL, SKOS
import os # Import os for file path handling
from lib.storage import get_storage_manager

# --- Define Namespaces (Best Practice with rdf.glycoshape.org) ---
GS = Namespace("http://rdf.glycoshape.org/ontology/")          # Ontology/vocabulary terms
GSR = Namespace("http://rdf.glycoshape.org/resource/")         # Resource instances
GSM = Namespace("http://rdf.glycoshape.org/metadata/")         # Metadata concepts
GLYCORDF = Namespace("http://purl.jp/bio/12/glyco/glycan#")    # GlycoRDF ontology
GLYTOUCAN = Namespace("http://rdf.glytoucan.org/glycan/")      # GlyTouCan resources
QUDT = Namespace("http://qudt.org/schema/qudt/")               # Units and quantities
UNIT = Namespace("http://qudt.org/vocab/unit/")                # Standard units
CHEBI = Namespace("http://purl.obolibrary.org/obo/CHEBI_")     # Chemical entities

# --- Helper Function to Add Literal if Value Exists ---
def add_literal(graph, subject, predicate, obj, datatype=None):
    """Adds a literal triple if the object value is not None or empty."""
    if obj is not None and obj != "":
        # Handle boolean values
        if isinstance(obj, bool):
            graph.add((subject, predicate, Literal(obj, datatype=XSD.boolean)))
        # Use specific datatypes if provided
        elif datatype:
            try:
                if datatype in [XSD.integer, XSD.float, XSD.double, XSD.decimal]:
                    if datatype == XSD.integer:
                        literal_value = Literal(int(float(obj)), datatype=datatype)
                    else:
                        literal_value = Literal(float(obj), datatype=datatype)
                else:
                    literal_value = Literal(str(obj), datatype=datatype)
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
                graph.add((subject, predicate, Literal(obj, datatype=XSD.double)))
            else:
                graph.add((subject, predicate, Literal(str(obj), datatype=XSD.string)))


# --- Helper Function to Add Quantity with Unit ---
def add_quantity_with_unit(graph, subject, predicate, value, unit_uri, quantity_type=None):
    """Adds a quantity value with associated unit using QUDT pattern."""
    if value is not None and value != "":
        # Create a blank node for the quantity
        quantity_node = BNode()
        graph.add((subject, predicate, quantity_node))
        graph.add((quantity_node, RDF.type, QUDT.Quantity))
        if quantity_type:
            graph.add((quantity_node, RDF.type, quantity_type))
        
        # Add the numeric value
        add_literal(graph, quantity_node, QUDT.numericValue, value, XSD.double)
        # Add the unit
        graph.add((quantity_node, QUDT.hasUnit, unit_uri))


# --- Helper Function to Add Sequence ---
def add_sequence(graph, glycan_variant_uri, sequence_value, sequence_type_uri, sequence_format_label):
    """Adds a Glycosequence node for a given sequence string and format."""
    if sequence_value:
        # Use a URI for the sequence node based on the variant and format
        safe_label = sequence_format_label.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
        seq_uri = URIRef(f"{str(glycan_variant_uri)}/sequence/{safe_label}")

        graph.add((glycan_variant_uri, GLYCORDF.has_glycosequence, seq_uri))
        graph.add((seq_uri, RDF.type, GLYCORDF.Glycosequence))
        add_literal(graph, seq_uri, GLYCORDF.has_sequence, sequence_value, XSD.string)
        graph.add((seq_uri, GLYCORDF.in_carbohydrate_format, sequence_type_uri))
        
        # Add format label if it's a custom format
        if sequence_type_uri.startswith(str(GS)):
            add_literal(graph, sequence_type_uri, RDFS.label, sequence_format_label)
            add_literal(graph, sequence_type_uri, DCTERMS.title, sequence_format_label)


# --- Helper Function to Process Search Metadata ---
def process_search_metadata(graph, main_entry_uri, search_meta):
    """Process the search_meta field and add appropriate triples."""
    if not search_meta or not isinstance(search_meta, dict):
        return
    
    # Common names as alternative labels
    common_names = search_meta.get("common_names", [])
    if isinstance(common_names, list):
        for name in common_names:
            if name:
                add_literal(graph, main_entry_uri, SKOS.altLabel, name)
                add_literal(graph, main_entry_uri, GS.commonName, name)
    
    # Description
    description = search_meta.get("description")
    if description:
        add_literal(graph, main_entry_uri, DCTERMS.description, description)
        add_literal(graph, main_entry_uri, RDFS.comment, description)
    
    # Keywords as subject tags
    keywords = search_meta.get("keywords", [])
    if isinstance(keywords, list):
        for keyword in keywords:
            if keyword:
                # Create concept URIs for keywords
                keyword_safe = keyword.lower().replace(' ', '_').replace('-', '_')
                keyword_uri = GSM[f"keyword/{keyword_safe}"]
                graph.add((main_entry_uri, DCTERMS.subject, keyword_uri))
                graph.add((keyword_uri, RDF.type, SKOS.Concept))
                add_literal(graph, keyword_uri, RDFS.label, keyword)
                add_literal(graph, keyword_uri, SKOS.prefLabel, keyword)


# --- Main Processing Function for a Single Glycan Variant ---
def process_glycan_variant(glycan_variant_uri, data, g):
    """Processes a single glycan variant (archetype, alpha, beta) and adds to the graph 'g'."""

    # Basic Type Information
    g.add((glycan_variant_uri, RDF.type, GS.GlycanVariant))
    g.add((glycan_variant_uri, RDF.type, GLYCORDF.Saccharide))

    # --- Identifiers ---
    glytoucan_id = data.get("glytoucan")
    if glytoucan_id:
        add_literal(g, glycan_variant_uri, GS.glytoucanID, glytoucan_id)
        glytoucan_uri = GLYTOUCAN[glytoucan_id]
        g.add((glycan_variant_uri, OWL.sameAs, glytoucan_uri))
        g.add((glycan_variant_uri, DCTERMS.identifier, Literal(glytoucan_id)))

    # --- Names and Labels (Using Dublin Core and SKOS best practices) ---
    name = data.get("name")
    if name:
        add_literal(g, glycan_variant_uri, RDFS.label, name)
        add_literal(g, glycan_variant_uri, DCTERMS.title, name)
        add_literal(g, glycan_variant_uri, SKOS.prefLabel, name)
    
    # Alternative names
    for name_type, predicate in [
        ("iupac", GS.iupacName),
        ("iupac_extended", GS.iupacExtendedName),
        ("glycam", GS.glycamName),
        ("oxford", GS.oxfordName)
    ]:
        name_value = data.get(name_type)
        if name_value:
            add_literal(g, glycan_variant_uri, predicate, name_value)
            add_literal(g, glycan_variant_uri, SKOS.altLabel, name_value)

    # --- Sequence Representations ---
    sequence_mappings = [
        ("wurcs", GLYCORDF.carbohydrate_format_wurcs, "WURCS"),
        ("glycoct", GLYCORDF.carbohydrate_format_glycoct, "GlycoCT"),
        ("iupac", GLYCORDF.carbohydrate_format_iupac_condensed, "IUPAC Condensed"),
        ("iupac_extended", GS.carbohydrate_format_iupac_extended, "IUPAC Extended"),
        ("glycam", GS.carbohydrate_format_glycam, "GLYCAM"),
        ("smiles", GS.carbohydrate_format_smiles, "SMILES")
    ]
    
    for seq_key, format_uri, format_label in sequence_mappings:
        seq_value = data.get(seq_key)
        if seq_value:
            add_sequence(g, glycan_variant_uri, seq_value, format_uri, format_label)

    # --- Physical / Chemical Properties ---
    add_literal(g, glycan_variant_uri, GS.mass, data.get("mass"), XSD.double)
    add_literal(g, glycan_variant_uri, GS.hydrogenBondAcceptors, data.get("hbond_acceptor"), XSD.integer)
    add_literal(g, glycan_variant_uri, GS.hydrogenBondDonors, data.get("hbond_donor"), XSD.integer)
    add_literal(g, glycan_variant_uri, GS.rotatableBonds, data.get("rot_bonds"), XSD.integer)
    
    # Entropy with proper scientific notation
    entropy = data.get("entropy")
    if entropy is not None:
        add_literal(g, glycan_variant_uri, GS.entropy, entropy, XSD.double)

    # --- Structural Features: Motifs ---
    motifs = data.get("motifs")
    if motifs and isinstance(motifs, list):
        for motif in motifs:
            if isinstance(motif, dict):
                motif_id = motif.get("motif")
                motif_label = motif.get("motif_label")
                if motif_id:
                    motif_uri = GSR[f"motif/{motif_id}"]
                    g.add((glycan_variant_uri, GLYCORDF.has_motif, motif_uri))
                    g.add((motif_uri, RDF.type, GLYCORDF.Motif))
                    g.add((motif_uri, RDF.type, SKOS.Concept))
                    add_literal(g, motif_uri, DCTERMS.identifier, motif_id)
                    if motif_label:
                        add_literal(g, motif_uri, RDFS.label, motif_label)
                        add_literal(g, motif_uri, SKOS.prefLabel, motif_label)
            else:
                print(f"Warning: Unexpected motif format in {glycan_variant_uri}: {motif}")

    # --- Structural Features: Termini ---
    termini = data.get("termini")
    if termini and isinstance(termini, list):
        for terminus in termini:
            if terminus:
                # Create terminus concept
                terminus_safe = terminus.replace('(', '_').replace(')', '_').replace('-', '_')
                terminus_uri = GSR[f"terminus/{terminus_safe}"]
                g.add((glycan_variant_uri, GLYCORDF.has_terminal_residue, terminus_uri))
                g.add((terminus_uri, RDF.type, GS.TerminalResidue))
                g.add((terminus_uri, RDF.type, SKOS.Concept))
                add_literal(g, terminus_uri, RDFS.label, terminus)
                add_literal(g, terminus_uri, SKOS.prefLabel, terminus)

    # --- Composition: Components (monosaccharide components) ---
    components = data.get("components")
    if components and isinstance(components, dict):
        for mono_name, count in components.items():
            comp_node = BNode()
            g.add((glycan_variant_uri, GLYCORDF.has_component, comp_node))
            g.add((comp_node, RDF.type, GLYCORDF.Component))

            mono_type_uri = GSR[f"monosaccharide/{mono_name}"]
            g.add((comp_node, GLYCORDF.has_monosaccharide, mono_type_uri))
            g.add((mono_type_uri, RDF.type, GLYCORDF.Monosaccharide))
            g.add((mono_type_uri, RDF.type, SKOS.Concept))
            add_literal(g, mono_type_uri, RDFS.label, mono_name)
            add_literal(g, mono_type_uri, SKOS.prefLabel, mono_name)
            add_literal(g, comp_node, GLYCORDF.has_cardinality, count, XSD.integer)

    # --- Composition: Residue Type Composition (new structure) ---
    composition = data.get("composition")
    if composition and isinstance(composition, dict):
        for residue_type, count in composition.items():
            comp_node = BNode()
            g.add((glycan_variant_uri, GS.hasResidueTypeComposition, comp_node))
            g.add((comp_node, RDF.type, GS.ResidueTypeComposition))
            
            residue_type_uri = GSR[f"residue_type/{residue_type}"]
            g.add((comp_node, GS.hasResidueType, residue_type_uri))
            g.add((residue_type_uri, RDF.type, GS.ResidueType))
            g.add((residue_type_uri, RDF.type, SKOS.Concept))
            add_literal(g, residue_type_uri, RDFS.label, residue_type)
            add_literal(g, residue_type_uri, SKOS.prefLabel, residue_type)
            add_literal(g, comp_node, GS.hasCount, count, XSD.integer)

    # --- Simulation Parameters with Units ---
    add_literal(g, glycan_variant_uri, GS.simulationPackage, data.get("package"))
    add_literal(g, glycan_variant_uri, GS.simulationForcefield, data.get("forcefield"))
    
    # Simulation length (assuming microseconds based on values like 1.5, 4.0)
    length = data.get("length")
    if length is not None:
        add_quantity_with_unit(g, glycan_variant_uri, GS.simulationLength, 
                             length, UNIT.MicroSec, GS.SimulationTime)
    
    # Temperature (Kelvin)
    temp = data.get("temperature")
    if temp is not None:
        add_quantity_with_unit(g, glycan_variant_uri, GS.simulationTemperature, 
                             temp, UNIT.K, GS.Temperature)
    
    # Pressure (assuming bar/atm)
    pressure = data.get("pressure")
    if pressure is not None:
        add_quantity_with_unit(g, glycan_variant_uri, GS.simulationPressure, 
                             pressure, UNIT.BAR, GS.Pressure)
    
    # Salt concentration (assuming mM based on values like "200", "150")
    salt = data.get("salt")
    if salt is not None:
        try:
            salt_val = float(salt)
            add_quantity_with_unit(g, glycan_variant_uri, GS.simulationSaltConcentration,
                                 salt_val, UNIT.MilliM, GS.Concentration)
        except (ValueError, TypeError):
            add_literal(g, glycan_variant_uri, GS.simulationSaltConcentration, salt)

    # --- Simulation Results: Clusters ---
    clusters = data.get("clusters")
    if clusters and isinstance(clusters, dict):
        for cluster_label, percentage in clusters.items():
            cluster_result_node = BNode()
            g.add((glycan_variant_uri, GS.hasClusterResult, cluster_result_node))
            g.add((cluster_result_node, RDF.type, GS.ClusterResult))

            safe_label = cluster_label.replace(" ", "_")
            add_literal(g, cluster_result_node, RDFS.label, cluster_label)
            add_literal(g, cluster_result_node, GS.clusterLabel, safe_label)
            add_literal(g, cluster_result_node, RDF.value, percentage, XSD.double)
            add_literal(g, cluster_result_node, GS.clusterPercentage, percentage, XSD.double)

    # --- Coverage Clusters (if different from clusters) ---
    coverage_clusters = data.get("coverage_clusters")
    if coverage_clusters and isinstance(coverage_clusters, dict):
        for cluster_label, percentage in coverage_clusters.items():
            coverage_result_node = BNode()
            g.add((glycan_variant_uri, GS.hasCoverageClusterResult, coverage_result_node))
            g.add((coverage_result_node, RDF.type, GS.CoverageClusterResult))

            safe_label = cluster_label.replace(" ", "_")
            add_literal(g, coverage_result_node, RDFS.label, cluster_label)
            add_literal(g, coverage_result_node, GS.clusterLabel, safe_label)
            add_literal(g, coverage_result_node, RDF.value, percentage, XSD.double)
            add_literal(g, coverage_result_node, GS.coveragePercentage, percentage, XSD.double)


# --- Main Conversion Function ---
def convert_glycoshape_to_rdf(input_path, output_path):
    """
    Convert a GlycoShape JSON database to RDF Turtle using best practices.
    """

    # Create master graph
    g_all = Graph()

    # Bind namespaces for readable output
    g_all.bind("gs", GS)
    g_all.bind("gsr", GSR)
    g_all.bind("gsm", GSM)
    g_all.bind("glycordf", GLYCORDF)
    g_all.bind("glytoucan", GLYTOUCAN)
    g_all.bind("dcterms", DCTERMS)
    g_all.bind("skos", SKOS)
    g_all.bind("owl", OWL)
    g_all.bind("qudt", QUDT)
    g_all.bind("unit", UNIT)
    g_all.bind("chebi", CHEBI)

    # Read JSON data
    print(f"Reading JSON data from: {input_path}")
    try:
        storage = get_storage_manager()
        with storage.open(input_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {input_path}: {e}")
        return None

    if not isinstance(data, dict):
        print(f"Error: Expected top-level JSON structure to be a dictionary, but found {type(data)}.")
        return None

    print(f"Processing {len(data)} entries...")
    entry_count = 0
    
    for main_glycan_id, entry_data in data.items():
        entry_count += 1
        print(f"Processing entry {entry_count}/{len(data)}: {main_glycan_id}")

        if not isinstance(entry_data, dict):
            print(f"Warning: Skipping entry '{main_glycan_id}' because its value is not a dictionary.")
            continue

        # Create the main resource URI
        main_entry_uri = GSR[main_glycan_id]
        g_all.add((main_entry_uri, RDF.type, GS.GlycoShapeEntry))
        g_all.add((main_entry_uri, RDF.type, SKOS.Concept))
        add_literal(g_all, main_entry_uri, RDFS.label, f"GlycoShape Entry {main_glycan_id}")
        add_literal(g_all, main_entry_uri, DCTERMS.title, f"GlycoShape Entry {main_glycan_id}")
        add_literal(g_all, main_entry_uri, DCTERMS.identifier, main_glycan_id)
        add_literal(g_all, main_entry_uri, GS.glycoShapeID, main_glycan_id)

        # Process search metadata
        search_meta = entry_data.get("search_meta")
        if search_meta:
            process_search_metadata(g_all, main_entry_uri, search_meta)

        archetype_uri = None

        # Process each variant
        for variant_type in ["archetype", "alpha", "beta"]:
            if variant_type in entry_data and entry_data[variant_type] and isinstance(entry_data[variant_type], dict):
                variant_data = entry_data[variant_type]

                # Create variant URI
                variant_uri = GSR[f"{main_glycan_id}/{variant_type}"]

                # Link main entry to variant
                g_all.add((main_entry_uri, GS.hasVariant, variant_uri))

                # Add specific types and relationships
                if variant_type == "archetype":
                    g_all.add((variant_uri, RDF.type, GS.ArchetypeGlycan))
                    g_all.add((main_entry_uri, GS.hasArchetype, variant_uri))
                    archetype_uri = variant_uri
                elif variant_type == "alpha":
                    g_all.add((variant_uri, RDF.type, GS.AlphaAnomerGlycan))
                    g_all.add((main_entry_uri, GS.hasAlphaAnomer, variant_uri))
                elif variant_type == "beta":
                    g_all.add((variant_uri, RDF.type, GS.BetaAnomerGlycan))
                    g_all.add((main_entry_uri, GS.hasBetaAnomer, variant_uri))

                # Process variant data
                process_glycan_variant(variant_uri, variant_data, g_all)

                # Add anomer relationships
                if variant_type in ["alpha", "beta"] and archetype_uri:
                    g_all.add((variant_uri, GS.isAnomerOf, archetype_uri))

    # Write the graph
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

    print("\n--- Starting RDF Conversion with Best Practices ---")
    convert_glycoshape_to_rdf(input_file, output_file)
    print("--- RDF Conversion Finished ---")


if __name__ == "__main__":
    main()