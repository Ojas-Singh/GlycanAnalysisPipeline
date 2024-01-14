import re
from Bio.PDB import PDBParser, PDBIO
import numpy as np

def flip_alpha_beta(input_pdb, output_pdb):
    parser = PDBParser()
    structure = parser.get_structure("input", input_pdb)

    C1 = None
    H1 = None
    O1 = None
    HO1 = None

    for residue in structure.get_residues():
        if residue.get_id()[1] == 2:  # Checks for ResID 2
            resname = residue.get_resname()
            # Switch the last character from 'A' to 'B' or vice versa
            if resname[-1] == "A":
                residue.resname = resname[:-1] + 'B'
            elif resname[-1] == "B":
                residue.resname = resname[:-1] + 'A'

        for atom in residue:
            if atom.get_name() == "C1" and residue.get_id()[1] == 2:
                C1 = atom
            elif atom.get_name() == "H1" and residue.get_id()[1] == 2:
                H1 = atom
            elif atom.get_name() == "O1" and residue.get_id()[1] == 1:
                O1 = atom
            elif atom.get_name() == "HO1" and residue.get_id()[1] == 1:
                HO1 = atom

    if not (C1 and H1 and O1 and HO1):
        raise ValueError("One or more of the required atoms (C1, H1, O1, HO1) not found in the input PDB file.")

    CO1_vector = O1.coord - C1.coord
    CH1_vector = H1.coord - C1.coord
    HO1O1_vector = HO1.coord - O1.coord

    new_O1_coord = C1.coord + CH1_vector
    new_O1_coord = C1.coord + CH1_vector * 1.43 / np.linalg.norm(CH1_vector)
    O1.set_coord(new_O1_coord)

    new_H1_coord = C1.coord + CO1_vector
    new_H1_coord = C1.coord + (new_H1_coord - C1.coord) * 1.09 / np.linalg.norm(new_H1_coord - C1.coord)
    H1.set_coord(new_H1_coord)

    new_HO1_coord = new_O1_coord + HO1O1_vector
    HO1.set_coord(new_HO1_coord)

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)


def get_glycan_sequence_from_filename(filename):
    pattern = re.compile(r'[A-Za-z0-9]+-?OH?')
    sequence = pattern.findall(filename)
    return sequence

def find_alpha_or_beta_oh_from_filename(filename):
    glycan_sequence = get_glycan_sequence_from_filename(filename)
    last_residue = glycan_sequence[-1]
    
    if "a1-OH" in last_residue or "a2-OH" in last_residue:
        return "Alpha OH"
    elif "b1-OH" in last_residue:
        return "Beta OH"
    else:
        return "OH not found"

def is_alpha(name):
    if find_alpha_or_beta_oh_from_filename(name)=="Alpha OH":
        return True
    else:
        return False
    



def flip_alpha_beta_multi(input_pdb, output_pdb):
    parser = PDBParser()
    structure = parser.get_structure("input", input_pdb)

    for model in structure:
            # Initialize variables for each model
            C1 = None
            H1 = None
            O1 = None
            HO1 = None

            for chain in model:
                for residue in chain:
                    if residue.get_id()[1] == 2:  # Checks for ResID 2
                        resname = residue.get_resname()
                        # Switch the last character from 'A' to 'B' or vice versa
                        if resname[-1] == "A":
                            residue.resname = resname[:-1] + 'B'
                        elif resname[-1] == "B":
                            residue.resname = resname[:-1] + 'A'

                    for atom in residue:
                        if atom.get_name() == "C1" and residue.get_id()[1] == 2:
                            C1 = atom
                        elif atom.get_name() == "H1" and residue.get_id()[1] == 2:
                            H1 = atom
                        elif atom.get_name() == "O1" and residue.get_id()[1] == 1:
                            O1 = atom
                        elif atom.get_name() == "HO1" and residue.get_id()[1] == 1:
                            HO1 = atom

            if not (C1 and H1 and O1 and HO1):
                raise ValueError("One or more of the required atoms (C1, H1, O1, HO1) not found in the input PDB file.")

            CO1_vector = O1.coord - C1.coord
            CH1_vector = H1.coord - C1.coord
            HO1O1_vector = HO1.coord - O1.coord

            new_O1_coord = C1.coord + CH1_vector
            new_O1_coord = C1.coord + CH1_vector * 1.43 / np.linalg.norm(CH1_vector)
            O1.set_coord(new_O1_coord)

            new_H1_coord = C1.coord + CO1_vector
            new_H1_coord = C1.coord + (new_H1_coord - C1.coord) * 1.09 / np.linalg.norm(new_H1_coord - C1.coord)
            H1.set_coord(new_H1_coord)

            new_HO1_coord = new_O1_coord + HO1O1_vector
            HO1.set_coord(new_HO1_coord)

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)
