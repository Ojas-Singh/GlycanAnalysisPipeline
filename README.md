# GlycanAnalysisPipeline

Glycan conformations analysis pipeline for [GlycoShape Database](https://glycoshape.org).


![Schematic overview of Glycan Analysis Pipeline (GAP) used to build the GlycoShape Glycan Database (GDB). Panel a) Multiple uncorrelated replica molecular dynamics (MD) simulations are performed for each glycan in the GDB, to comprehensively sample its structural dynamics. The resulting MD frames are then transformed into a graph matrix representation, as depicted in Panel b), simplified by flattening the lower half as shown in Panel c). This step enables a dimensionality reduction via principal component analysis (PCA), shown in Panel d). These data are clustered by Gaussian Mixture Model (GMM) and the results of which are displayed in terms of cluster distributions, see Panel e). Panel f) Representative 3D structures for each cluster are selected based on KDE maxima, along with comprehensive torsion angle profiles for the highest populated clusters, showing the wide breadth of the conformational space covered by GAP. Panel g) Structures derived from GAP are clearly presented on the GlycoShape GDB web platform, in addition to biological and chemical information.](Figure.jpg)



From every glycan simulated, a directory is made titled with the name of the particular glycan in GLYCAM condensed format. Within this directory is a multiframe PDB of the concatenated replicas of MD simulation, and a single frame MOL2 file. The GAP pipeline is then ran on these directories to create further subdirectories titled "output" and "clusters" which contain the outputs of both the PCA and GMM and the representative cluster structures, respectively.



# Installation
```
conda create -n GAP python=3.10
conda activate GAP
pip install -r requirements.txt

```
modify config.py to set data_dir variable to the folder where we have all the simulations multiframe pdb and mol2 file for the molecule, the folder name should be the GLYCAM name of the glycan.


# Running
```
python main.py && python recluster.py && python plot_dist.py && python save_frames.py
```
this will produce "clusters" and "output" folder in each glycan dir with required files for Database and [Re-Glyco](https://github.com/Ojas-Singh/Re-Glyco).

# Note 
The DB script then takes the structural information from these directories, coupled with APIs and other packages, to create the information necessary for the GDB. For the is, the DB directory contains subdirectories titled with the name of each glycan in IUPAC condensed format. Within these subdirectories are JSON files with the relecant nomeclature, chemical, and biological data of the glycan and an SVG file of the glycan 2D structure in SNFG format. Also located within this directory are further subdirectories containing the representative cluster structures in different naming formats, specifically CHARMM, GLYCAM, and PDB.

The final output database has format of dummy_database/. This directory format is used by [Re-Glyco](https://glycoshape.org/reglyco) to build glycoproteins. The code for Re-Glyco is [here](https://github.com/Ojas-Singh/Re-Glyco)

# Citation

All of the data provided is freely available for academic use under Creative Commons Attribution 4.0 (CC BY-NC-ND 4.0 Deed) licence terms. Please contact us at elisa.fadda@mu.ie for Commercial licence. If you use this resource, please cite the following papers:

Callum M Ives and Ojas Singh et al. Restoring Protein Glycosylation with GlycoShape [bioRxiv (2023)](https://www.biorxiv.org/content/10.1101/2023.12.11.571101v1.full).

