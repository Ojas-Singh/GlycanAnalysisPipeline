from Bio.SVDSuperimposer import SVDSuperimposer
from Bio.PDB import PDBParser, PDBIO
import numpy as np


def align_pdb_files(reference_file, files_to_align, output_files):
    parser = PDBParser()
    reference_structure = parser.get_structure("reference", reference_file)
    
    reference_residues = list(reference_structure.get_residues())
    residue_count = len(reference_residues)
    
    if residue_count < 4:
        reference_atoms = [atom.get_coord() for residue in reference_residues for atom in residue.get_atoms()]
    else:
        reference_atoms = [atom.get_coord() for residue in reference_residues[:4] for atom in residue.get_atoms()]
    
    reference_centroid = np.mean(reference_atoms, axis=0)

    for file, output_file in zip(files_to_align, output_files):
        structure_to_align = parser.get_structure("to_align", file)
        
        to_align_residues = list(structure_to_align.get_residues())
        to_align_residue_count = len(to_align_residues)

        if to_align_residue_count < 4:
            atoms_to_align = [atom.get_coord() for residue in to_align_residues for atom in residue.get_atoms()]
        else:
            atoms_to_align = [atom.get_coord() for residue in to_align_residues[:4] for atom in residue.get_atoms()]
        
        align_centroid = np.mean(atoms_to_align, axis=0)

        # Ensure that the structures have the same number of atoms in the selected residues
        if len(reference_atoms) != len(atoms_to_align):
            print(f"The number of atoms in the selected residues of {file} does not match the reference structure.")
            continue

        super_imposer = SVDSuperimposer()
        super_imposer.set(np.array(reference_atoms) - reference_centroid, np.array(atoms_to_align) - align_centroid)
        super_imposer.run()

        # Get the rotation and translation matrix for the alignment of the selected residues
        rot, tran = super_imposer.get_rotran()

        # Apply the transformation to all atoms in the structure to align
        for atom in structure_to_align.get_atoms():
            atom.set_coord(np.dot(atom.get_coord() - align_centroid, rot) + reference_centroid + tran)

        io = PDBIO()
        io.set_structure(structure_to_align)
        io.save(output_file)




def merge_pdb_files(input_files, output_file):
    parser = PDBParser()
    io = PDBIO()

    # Create a new structure object to hold all the chains
    from Bio.PDB.Structure import Structure
    merged_structure = Structure('merged')

    chain_names = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    chain_index = 0

    # Iterate through the input files, load each structure, and add to the merged structure
    for file in input_files:
        structure = parser.get_structure("to_merge", file)
        for model in structure:
            for chain in model:
                if chain_index >= len(chain_names):
                    print("Warning: Too many chains, ran out of unique chain identifiers.")
                else:
                    chain.id = chain_names[chain_index]
                    chain_index += 1

                # Append the chain to the merged structure
                merged_structure.add(chain)

    # Save the merged structure to the output file
    io.set_structure(merged_structure)
    io.save(output_file)


