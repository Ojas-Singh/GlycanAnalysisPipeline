# GlycoShape RDF Database Structure Documentation

## Namespace Overview

| Prefix | URI | Purpose |
|--------|-----|---------|
| `gs:` | `http://rdf.glycoshape.org/ontology/` | Ontology terms and properties |
| `gsr:` | `http://rdf.glycoshape.org/resource/` | Resource instances (glycans, motifs, etc.) |
| `gsm:` | `http://rdf.glycoshape.org/metadata/` | Metadata concepts (keywords, etc.) |
| `glycordf:` | `http://purl.jp/bio/12/glyco/glycan#` | GlycoRDF ontology terms |
| `glytoucan:` | `http://rdf.glytoucan.org/glycan/` | GlyTouCan resource URIs |
| `qudt:` | `http://qudt.org/schema/qudt/` | Quantities, units, dimensions |
| `unit:` | `http://qudt.org/vocab/unit/` | Standard unit definitions |

## Core Entity Structure

### 1. Main Entry (`gs:GlycoShapeEntry`)

**URI Pattern:** `gsr:GS00XXX`

**Example:** `gsr:GS00445`

**Core Properties:**
- `rdf:type`: `gs:GlycoShapeEntry`, `skos:Concept`
- `dcterms:identifier`: GlycoShape ID (string)
- `gs:glycoShapeID`: GlycoShape ID (string)
- `rdfs:label`: "GlycoShape Entry {ID}" (string)
- `dcterms:title`: "GlycoShape Entry {ID}" (string)

**Search Metadata Properties:**
- `skos:altLabel`: Common names (string, multiple values)
- `gs:commonName`: Common names (string, multiple values)
- `dcterms:description`: Description text (string)
- `rdfs:comment`: Description text (string)
- `dcterms:subject`: Links to keyword concepts (URI)

**Variant Relationships:**
- `gs:hasVariant`: Links to variant URIs (URI, multiple)
- `gs:hasArchetype`: Links to archetype variant (URI)
- `gs:hasAlphaAnomer`: Links to alpha anomer (URI)
- `gs:hasBetaAnomer`: Links to beta anomer (URI)

### 2. Glycan Variants (`gs:GlycanVariant`)

**URI Pattern:** `gsr:GS00XXX/{variant_type}`

**Example:** `gsr:GS00445/archetype`

**Core Types:**
- `gs:ArchetypeGlycan` (archetype variants)
- `gs:AlphaAnomerGlycan` (alpha anomer variants)  
- `gs:BetaAnomerGlycan` (beta anomer variants)
- `gs:GlycanVariant` (all variants)
- `glycordf:Saccharide` (all variants)

#### 2.1 Identifiers and Names

| Property | Value Type | Example | Description |
|----------|------------|---------|-------------|
| `gs:glytoucanID` | xsd:string | "G23706QF" | GlyTouCan identifier |
| `owl:sameAs` | URI | `glytoucan:G23706QF` | Link to GlyTouCan |
| `dcterms:identifier` | xsd:string | "G23706QF" | Generic identifier |
| `rdfs:label` | xsd:string | "DManpa1-2DManpa1-OH" | Primary name |
| `dcterms:title` | xsd:string | "DManpa1-2DManpa1-OH" | Display title |
| `skos:prefLabel` | xsd:string | "DManpa1-2DManpa1-OH" | Preferred label |
| `gs:iupacName` | xsd:string | "Man(a1-2)Man" | IUPAC name |
| `gs:iupacExtendedName` | xsd:string | "α-D-Manp-(1→2)-D-Man" | Extended IUPAC |
| `gs:glycamName` | xsd:string | "DManpa1-2DManp" | GLYCAM name |
| `gs:oxfordName` | xsd:string | "FA2BG2S2" | Oxford notation |
| `skos:altLabel` | xsd:string | Multiple alternative names | Alternative names |

#### 2.2 Sequence Representations

**Pattern:** Variant → `glycordf:has_glycosequence` → Sequence Node

**Sequence Node Structure:**
- `rdf:type`: `glycordf:Glycosequence`
- `glycordf:has_sequence`: Sequence string (xsd:string)
- `glycordf:in_carbohydrate_format`: Format URI

**Supported Formats:**

| Format | Property | Format URI | Example Value |
|--------|----------|------------|---------------|
| WURCS | `glycordf:has_glycosequence` | `glycordf:carbohydrate_format_wurcs` | "WURCS=2.0/2,2,1/..." |
| GlycoCT | `glycordf:has_glycosequence` | `glycordf:carbohydrate_format_glycoct` | "RES\n1b:b-dman-HEX..." |
| IUPAC | `glycordf:has_glycosequence` | `glycordf:carbohydrate_format_iupac_condensed` | "Man(a1-2)Man" |
| IUPAC Extended | `glycordf:has_glycosequence` | `gs:carbohydrate_format_iupac_extended` | "α-D-Manp-(1→2)-D-Man" |
| GLYCAM | `glycordf:has_glycosequence` | `gs:carbohydrate_format_glycam` | "DManpa1-2DManp" |
| SMILES | `glycordf:has_glycosequence` | `gs:carbohydrate_format_smiles` | "O1C(O)[C@@H]..." |

#### 2.3 Chemical Properties

| Property | Value Type | Example | Description |
|----------|------------|---------|-------------|
| `gs:mass` | xsd:double | 342.12 | Molecular mass in Da |
| `gs:hydrogenBondAcceptors` | xsd:integer | 11 | Number of H-bond acceptors |
| `gs:hydrogenBondDonors` | xsd:integer | 8 | Number of H-bond donors |
| `gs:rotatableBonds` | xsd:integer | 4 | Number of rotatable bonds |
| `gs:entropy` | xsd:double | 8.726120274593718 | Conformational entropy |

#### 2.4 Structural Features

**Motifs:** `glycordf:has_motif` → Motif URI

**Motif Structure:**
- **URI Pattern:** `gsr:motif/{motif_id}`
- **Example:** `gsr:motif/G00065MO`
- **Types:** `glycordf:Motif`, `skos:Concept`
- **Properties:**
  - `dcterms:identifier`: Motif ID (xsd:string)
  - `rdfs:label`: Motif label (xsd:string)
  - `skos:prefLabel`: Motif label (xsd:string)

**Terminal Residues:** `glycordf:has_terminal_residue` → Terminus URI

**Terminus Structure:**
- **URI Pattern:** `gsr:terminus/{safe_terminus_name}`
- **Example:** `gsr:terminus/Man_a1_2_`
- **Types:** `gs:TerminalResidue`, `skos:Concept`
- **Properties:**
  - `rdfs:label`: Original terminus string (xsd:string)
  - `skos:prefLabel`: Original terminus string (xsd:string)

#### 2.5 Composition

**A. Monosaccharide Components** (`glycordf:has_component`)

**Component Structure (Blank Node):**
- **Type:** `glycordf:Component`
- **Links to:** `glycordf:has_monosaccharide` → Monosaccharide URI
- **Count:** `glycordf:has_cardinality` (xsd:integer)

**Monosaccharide URI:**
- **Pattern:** `gsr:monosaccharide/{name}`
- **Example:** `gsr:monosaccharide/Man`
- **Types:** `glycordf:Monosaccharide`, `skos:Concept`
- **Properties:** `rdfs:label`, `skos:prefLabel`

**B. Residue Type Composition** (`gs:hasResidueTypeComposition`)

**Residue Type Composition Structure (Blank Node):**
- **Type:** `gs:ResidueTypeComposition`
- **Links to:** `gs:hasResidueType` → Residue Type URI
- **Count:** `gs:hasCount` (xsd:integer)

**Residue Type URI:**
- **Pattern:** `gsr:residue_type/{type}`
- **Example:** `gsr:residue_type/Hex`
- **Types:** `gs:ResidueType`, `skos:Concept`
- **Properties:** `rdfs:label`, `skos:prefLabel`

#### 2.6 Simulation Parameters

**Simple Properties:**
| Property | Value Type | Example | Description |
|----------|------------|---------|-------------|
| `gs:simulationPackage` | xsd:string | "AMBER" | Simulation software |
| `gs:simulationForcefield` | xsd:string | "GLYCAM_06j" | Force field used |

**Quantities with Units (QUDT Pattern):**

Each creates a blank node with:
- `rdf:type`: `qudt:Quantity`, specific type
- `qudt:numericValue`: Numeric value (xsd:double)
- `qudt:hasUnit`: Unit URI

| Property | Quantity Type | Unit | Example Value | Example Unit |
|----------|---------------|------|---------------|--------------|
| `gs:simulationLength` | `gs:SimulationTime` | `unit:MicroSec` | 1.5 | Microseconds |
| `gs:simulationTemperature` | `gs:Temperature` | `unit:K` | 300.0 | Kelvin |
| `gs:simulationPressure` | `gs:Pressure` | `unit:BAR` | 1.0 | Bar |
| `gs:simulationSaltConcentration` | `gs:Concentration` | `unit:MilliM` | 200.0 | Millimolar |

#### 2.7 Simulation Results

**A. Cluster Results** (`gs:hasClusterResult`)

**Cluster Result Structure (Blank Node):**
- **Type:** `gs:ClusterResult`
- **Properties:**
  - `rdfs:label`: Original cluster label (xsd:string)
  - `gs:clusterLabel`: Safe cluster label (xsd:string)
  - `rdf:value`: Percentage value (xsd:double)
  - `gs:clusterPercentage`: Percentage value (xsd:double)

**B. Coverage Cluster Results** (`gs:hasCoverageClusterResult`)

**Coverage Cluster Result Structure (Blank Node):**
- **Type:** `gs:CoverageClusterResult`
- **Properties:**
  - `rdfs:label`: Original cluster label (xsd:string)
  - `gs:clusterLabel`: Safe cluster label (xsd:string)
  - `rdf:value`: Percentage value (xsd:double)
  - `gs:coveragePercentage`: Percentage value (xsd:double)

#### 2.8 Anomer Relationships

| Property | Domain | Range | Description |
|----------|--------|-------|-------------|
| `gs:isAnomerOf` | `gs:AlphaAnomerGlycan`, `gs:BetaAnomerGlycan` | `gs:ArchetypeGlycan` | Links anomers to archetype |

### 3. Metadata Concepts

#### Keywords (`gsm:keyword/{safe_keyword}`)

**Example:** `gsm:keyword/biantennary_n_glycan`

**Properties:**
- **Types:** `skos:Concept`
- **Properties:**
  - `rdfs:label`: Original keyword (xsd:string)
  - `skos:prefLabel`: Original keyword (xsd:string)
- **Linked from entries via:** `dcterms:subject`

## Example SPARQL Queries

### Query 1: Basic Information Retrieval
```sparql
PREFIX gs: <http://rdf.glycoshape.org/ontology/>
PREFIX gsr: <http://rdf.glycoshape.org/resource/>
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?entry ?id ?name ?description WHERE {
  ?entry a gs:GlycoShapeEntry ;
         gs:glycoShapeID ?id ;
         rdfs:label ?name .
  OPTIONAL { ?entry dcterms:description ?description }
}
```

### Query 2: Find Glycans by Mass Range
```sparql
PREFIX gs: <http://rdf.glycoshape.org/ontology/>
PREFIX gsr: <http://rdf.glycoshape.org/resource/>

SELECT ?entry ?variant ?mass WHERE {
  ?entry gs:hasVariant ?variant .
  ?variant gs:mass ?mass .
  FILTER(?mass >= 300 && ?mass <= 400)
}
```

### Query 3: Find Fucosylated Glycans
```sparql
PREFIX gs: <http://rdf.glycoshape.org/ontology/>
PREFIX gsr: <http://rdf.glycoshape.org/resource/>
PREFIX glycordf: <http://purl.jp/bio/12/glyco/glycan#>

SELECT ?entry ?variant ?fuc_count WHERE {
  ?entry gs:hasVariant ?variant .
  ?variant glycordf:has_component ?comp .
  ?comp glycordf:has_monosaccharide gsr:monosaccharide/Fuc ;
        glycordf:has_cardinality ?fuc_count .
}
```

### Query 4: Simulation Parameters Query
```sparql
PREFIX gs: <http://rdf.glycoshape.org/ontology/>
PREFIX qudt: <http://qudt.org/schema/qudt/>

SELECT ?entry ?variant ?temp_value ?pressure_value WHERE {
  ?entry gs:hasVariant ?variant .
  ?variant gs:simulationTemperature ?temp_qty ;
           gs:simulationPressure ?press_qty .
  ?temp_qty qudt:numericValue ?temp_value .
  ?press_qty qudt:numericValue ?pressure_value .
}
```

### Query 5: Find Glycans with Specific Motifs
```sparql
PREFIX gs: <http://rdf.glycoshape.org/ontology/>
PREFIX gsr: <http://rdf.glycoshape.org/resource/>
PREFIX glycordf: <http://purl.jp/bio/12/glyco/glycan#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?entry ?variant ?motif_label WHERE {
  ?entry gs:hasVariant ?variant .
  ?variant glycordf:has_motif ?motif .
  ?motif rdfs:label ?motif_label .
  FILTER(CONTAINS(LCASE(?motif_label), "n-glycan"))
}
```

### Query 6: Cluster Analysis
```sparql
PREFIX gs: <http://rdf.glycoshape.org/ontology/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?entry ?variant ?cluster_label ?percentage WHERE {
  ?entry gs:hasVariant ?variant .
  ?variant gs:hasClusterResult ?cluster .
  ?cluster rdfs:label ?cluster_label ;
           rdf:value ?percentage .
  FILTER(?percentage > 50)
}
ORDER BY DESC(?percentage)
```

### Query 7: Alternative Names and Sequences
```sparql
PREFIX gs: <http://rdf.glycoshape.org/ontology/>
PREFIX gsr: <http://rdf.glycoshape.org/resource/>
PREFIX glycordf: <http://purl.jp/bio/12/glyco/glycan#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT ?entry ?iupac ?wurcs_seq WHERE {
  ?entry gs:hasVariant ?variant .
  ?variant gs:iupacName ?iupac ;
           glycordf:has_glycosequence ?seq .
  ?seq glycordf:in_carbohydrate_format glycordf:carbohydrate_format_wurcs ;
       glycordf:has_sequence ?wurcs_seq .
}
```

## Data Types Summary

| XSD Type | Used For | Example |
|----------|----------|---------|
| `xsd:string` | Names, sequences, labels | "Man(a1-2)Man" |
| `xsd:integer` | Counts, bond numbers | 4 |
| `xsd:double` | Masses, percentages, measurements | 342.12 |
| `xsd:boolean` | Boolean flags | true |

## URI Construction Patterns

| Entity Type | Pattern | Example |
|-------------|---------|---------|
| Main Entry | `gsr:{ID}` | `gsr:GS00445` |
| Variant | `gsr:{ID}/{type}` | `gsr:GS00445/archetype` |
| Sequence | `gsr:{ID}/{type}/sequence/{format}` | `gsr:GS00445/archetype/sequence/wurcs` |
| Monosaccharide | `gsr:monosaccharide/{name}` | `gsr:monosaccharide/Man` |
| Motif | `gsr:motif/{id}` | `gsr:motif/G00065MO` |
| Terminus | `gsr:terminus/{safe_name}` | `gsr:terminus/Man_a1_2_` |
| Residue Type | `gsr:residue_type/{type}` | `gsr:residue_type/Hex` |
| Keyword | `gsm:keyword/{safe_keyword}` | `gsm:keyword/n_glycan` |