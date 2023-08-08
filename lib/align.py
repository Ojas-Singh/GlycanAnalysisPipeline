from Bio.SVDSuperimposer import SVDSuperimposer
from Bio.PDB import PDBParser, PDBIO
import numpy as np

def align_pdb_files(reference_file, files_to_align, output_files):
    parser = PDBParser()
    reference_structure = parser.get_structure("reference", reference_file)
    reference_atoms = [atom.get_coord() for atom in reference_structure.get_atoms()]

    for file, output_file in zip(files_to_align, output_files):
        structure_to_align = parser.get_structure("to_align", file)
        atoms_to_align = [atom.get_coord() for atom in structure_to_align.get_atoms()]

        # Ensure that the structures have the same number of atoms
        if len(reference_atoms) != len(atoms_to_align):
            print(f"The number of atoms in {file} does not match the reference structure.")
            continue

        super_imposer = SVDSuperimposer()
        super_imposer.set(np.array(reference_atoms), np.array(atoms_to_align))
        super_imposer.run()

        rot, tran = super_imposer.get_rotran()
        for atom, new_coord in zip(structure_to_align.get_atoms(), super_imposer.get_transformed()):
            atom.set_coord(new_coord)

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


