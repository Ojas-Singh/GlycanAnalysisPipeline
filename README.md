# GlycanAnalysisPipeline

Glycan conformations analysis pipeline for [GlycoShape Database](https://glycoshape.org).


![Schematic overview of Glycan Analysis Pipeline (GAP) used to build the GlycoShape Glycan Database (GDB). Panel a) Multiple uncorrelated replica molecular dynamics (MD) simulations are performed for each glycan in the GDB, to comprehensively sample its structural dynamics. The resulting MD frames are then transformed into a graph matrix representation, as depicted in Panel b), simplified by flattening the lower half as shown in Panel c). This step enables a dimensionality reduction via principal component analysis (PCA), shown in Panel d). These data are clustered by Gaussian Mixture Model (GMM) and the results of which are displayed in terms of cluster distributions, see Panel e). Panel f) Representative 3D structures for each cluster are selected based on KDE maxima, along with comprehensive torsion angle profiles for the highest populated clusters, showing the wide breadth of the conformational space covered by GAP. Panel g) Structures derived from GAP are clearly presented on the GlycoShape GDB web platform, in addition to biological and chemical information.](docs/Figure.jpg)



From every glycan simulated, a directory is made titled with the name of the particular glycan in GLYCAM condensed format. Within this directory is a multiframe PDB of the concatenated replicas of MD simulation, and a single frame MOL2 file. The GAP pipeline is then ran on these directories to create further subdirectories titled "output" and "clusters" which contain the outputs of both the PCA and GMM and the representative cluster structures, respectively.



# Installation
```
conda create -n GAP python=3.10
conda activate GAP
pip install -r requirements.txt

```
modify config.py to set data_dir variable to the folder where we have all the simulations multiframe pdb and mol2 file for the molecule, the folder name should be the GLYCAM name of the glycan.

# Configure

Before running the pipeline, ensure the following environment variables are set up.

```
export GLYCAN_DATA_DIR=/path/to/glycan/data
export DATABASE_PATH=/path/to/database
export GLYCAN_INVENTORY_PATH=/path/to/glycan/inventory
export GLYCAN_DB_UPDATE=True
export GLYTOUCAN_CONTRIBUTOR_ID=your_contributor_id
export GLYTOUCAN_API_KEY=your_api_key
```


Ensure the required directories and files exist, or modify the paths as needed.

# Running
```
python main.py 
```
this will produce "clusters" and "output" folder in each glycan dir with required files for Database and [Re-Glyco](https://github.com/Ojas-Singh/Re-Glyco).

# SPARQL Endpoint

Install [oxigraph](https://github.com/oxigraph/oxigraph) to serve GLYCOSHAPE_RDF.ttl file in the database output dir.

```
./oxigraph load --location GLYCOSHAPE_RDF --file glycoshape_rdf.ttl
./oxigraph serve --location GLYCOSHAPE_RDF
```
This will host the SPARQL endpoint at http://localhost:7878


# Note 
The DB script then takes the structural information from these directories, coupled with APIs and other packages, to create the information necessary for the GDB. For the is, the DB directory contains subdirectories titled with the name of each glycan in IUPAC condensed format. Within these subdirectories are JSON files with the relecant nomeclature, chemical, and biological data of the glycan and an SVG file of the glycan 2D structure in SNFG format. Also located within this directory are further subdirectories containing the representative cluster structures in different naming formats, specifically CHARMM, GLYCAM, and PDB.

The final output database has format of dummy_database/. This directory format is used by [Re-Glyco](https://glycoshape.org/reglyco) to build glycoproteins. The code for Re-Glyco is [here](https://github.com/Ojas-Singh/Re-Glyco)

# Citation

All of the data provided is freely available for academic use under Creative Commons Attribution 4.0 (CC BY-NC-ND 4.0 Deed) licence terms. Please contact us at elisa.fadda@soton.ac.uk for Commercial licence. If you use this resource, please cite the following papers:

Callum M Ives and Ojas Singh et al. Restoring Protein Glycosylation with GlycoShape [Nat Methods (2024).](https://doi.org/10.1038/s41592-024-02464-7).


