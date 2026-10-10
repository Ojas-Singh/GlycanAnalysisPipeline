#!/usr/bin/env python3
"""
Fix glycan PDB files by adding complete CONECT records.
This script reads a PDB file and generates complete connectivity information
for all glycan residues based on standard carbohydrate chemistry.
"""

import re
from collections import defaultdict


class GlycanConnectivity:
    """Defines standard connectivity patterns for glycan residues."""

    # Standard ring connectivity for pyranose sugars
    RING_CONNECTIVITY = {
        'C1': ['H1', 'O5', 'C2'],
        'C2': ['H2', 'C1', 'C3'],
        'C3': ['H3', 'C2', 'C4'],
        'C4': ['H4', 'C3', 'C5'],
        'C5': ['H5', 'C4', 'O5', 'C6'],
        'C6': ['H61', 'H62', 'O6'],
        'O5': ['C1', 'C5'],
        'O6': ['C6'],
    }

    # N-acetyl group connectivity (for NAG, UYB, 0YB, 4YB)
    N_ACETYL_CONNECTIVITY = {
        'N2': ['H2N', 'C2', 'C2N'],
        'C2N': ['N2', 'O2N', 'CME'],
        'O2N': ['C2N'],
        'CME': ['C2N', 'H1M', 'H2M', 'H3M'],
        'H2N': ['N2'],
        'H1M': ['CME'],
        'H2M': ['CME'],
        'H3M': ['CME'],
    }

    # Standard C2 modifications for different residue types
    C2_MODIFICATIONS = {
        'NAG': 'N2',      # N-acetylglucosamine
        'UYB': 'N2',      # GLYCAM N-acetylglucosamine code
        '0YB': 'N2',      # GLYCAM N-acetylglucosamine code
        '4YB': 'N2',      # GLYCAM N-acetylglucosamine code
        'FUC': 'O2',      # Fucose
        '0fA': 'O2',      # GLYCAM fucose code
        'MAN': 'O2',      # Mannose
        '4MA': 'O2',      # GLYCAM mannose code
        'BMA': 'O2',      # Glucose
        'QMB': 'O2',      # GLYCAM glucose code
        'GLC': 'O2',      # Glucose
        'GAL': 'O2',      # Galactose
        'ROH': 'O1',      # Reducing end
    }

    @classmethod
    def has_n_acetyl(cls, residue_name):
        """Check if residue has N-acetyl group."""
        return residue_name in ['NAG', 'UYB', '0YB', '4YB']

    @classmethod
    def get_c2_atom(cls, residue_name):
        """Get the C2 modification atom for a residue type."""
        return cls.C2_MODIFICATIONS.get(residue_name, 'O2')


class PDBAtom:
    """Represents a single atom in a PDB file."""

    def __init__(self, line):
        self.serial = int(line[6:11].strip())
        self.name = line[12:16].strip()
        self.alt_loc = line[16].strip()
        self.res_name = line[17:20].strip()
        self.chain_id = line[21].strip()
        self.res_seq = int(line[22:26].strip())
        self.x = float(line[30:38].strip())
        self.y = float(line[38:46].strip())
        self.z = float(line[46:54].strip())
        self.element = line[76:78].strip() if len(line) > 76 else self._infer_element()

    def _infer_element(self):
        """Infer element from atom name."""
        name = self.name
        # Handle special cases like C2N, O2N, CME
        if name.startswith('C'):
            return 'C'
        elif name.startswith('O'):
            return 'O'
        elif name.startswith('N'):
            return 'N'
        elif name.startswith('H'):
            return 'H'
        elif name.startswith('S'):
            return 'S'
        elif name.startswith('P'):
            return 'P'
        return 'C'

    def __repr__(self):
        return f"Atom({self.serial}, {self.name}, {self.res_name}{self.res_seq})"


class ConnectivityBuilder:
    """Builds complete connectivity for glycan PDB files."""

    def __init__(self):
        self.atoms = []
        self.residues = defaultdict(list)
        self.atom_by_serial = {}

    def parse_pdb(self, pdb_lines):
        """Parse PDB file and organize atoms by residue."""
        for line in pdb_lines:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom = PDBAtom(line)
                self.atoms.append(atom)
                self.residues[(atom.res_seq, atom.res_name, atom.chain_id)].append(atom)
                self.atom_by_serial[atom.serial] = atom

    def get_residue_connectivity(self, res_seq, res_name, chain_id):
        """Get complete connectivity for a residue."""
        residue_atoms = self.residues.get((res_seq, res_name, chain_id), [])
        atom_map = {atom.name: atom.serial for atom in residue_atoms}

        connectivity = defaultdict(list)

        # Add ring connectivity
        for atom_name, connections in GlycanConnectivity.RING_CONNECTIVITY.items():
            if atom_name in atom_map:
                from_serial = atom_map[atom_name]
                for conn_name in connections:
                    if conn_name in atom_map:
                        connectivity[from_serial].append(atom_map[conn_name])

        # Add C2 modification connectivity
        c2_atom = GlycanConnectivity.get_c2_atom(res_name)
        if c2_atom == 'N2' and GlycanConnectivity.has_n_acetyl(res_name):
            # Add N-acetyl group connectivity
            for atom_name, connections in GlycanConnectivity.N_ACETYL_CONNECTIVITY.items():
                if atom_name in atom_map:
                    from_serial = atom_map[atom_name]
                    for conn_name in connections:
                        if conn_name in atom_map:
                            connectivity[from_serial].append(atom_map[conn_name])
        elif c2_atom in atom_map:
            # Add O2 connectivity
            if 'C2' in atom_map and 'O2' in atom_map:
                connectivity[atom_map['C2']].append(atom_map['O2'])
            if 'O2' in atom_map and 'H2O' in atom_map:
                connectivity[atom_map['O2']].append(atom_map['H2O'])

        # Add oxygen hydrogens
        for atom in residue_atoms:
            if atom.name == 'O6' and 'H6O' in atom_map:
                connectivity[atom.serial].append(atom_map['H6O'])
            elif atom.name == 'O3' and 'H3O' in atom_map:
                connectivity[atom.serial].append(atom_map['H3O'])
            elif atom.name == 'O4' and 'H4O' in atom_map:
                connectivity[atom.serial].append(atom_map['H4O'])
            elif atom.name == 'O2' and 'H2O' in atom_map:
                connectivity[atom.serial].append(atom_map['H2O'])
            elif atom.name == 'O1' and 'HO1' in atom_map:
                connectivity[atom.serial].append(atom_map['HO1'])

        return connectivity

    def get_inter_residue_bonds(self):
        """Find inter-residue bonds (glycosidic linkages)."""
        # This is a simplified implementation - in practice you'd use
        # distance-based detection or LINK records
        bonds = []

        # Look for common glycosidic linkage patterns
        # O4 of one residue to C1 of another, etc.
        for i, atom1 in enumerate(self.atoms):
            if atom1.name in ['O4', 'O3', 'O6', 'O2']:
                # Find nearby C1 atoms from other residues
                for atom2 in self.atoms[i+1:]:
                    if atom2.name == 'C1' and atom1.res_seq != atom2.res_seq:
                        # Calculate distance
                        dist = self._distance(atom1, atom2)
                        if 1.3 < dist < 1.6:  # Typical C-O bond length
                            bonds.append((atom1.serial, atom2.serial))

        return bonds

    def _distance(self, atom1, atom2):
        """Calculate Euclidean distance between two atoms."""
        import math
        dx = atom1.x - atom2.x
        dy = atom1.y - atom2.y
        dz = atom1.z - atom2.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def build_all_connectivity(self):
        """Build complete connectivity for the entire structure."""
        all_connectivity = defaultdict(list)

        # Get intra-residue connectivity
        processed_residues = set()
        for atom in self.atoms:
            res_key = (atom.res_seq, atom.res_name, atom.chain_id)
            if res_key not in processed_residues:
                res_connectivity = self.get_residue_connectivity(*res_key)
                for from_atom, to_atoms in res_connectivity.items():
                    all_connectivity[from_atom].extend(to_atoms)
                processed_residues.add(res_key)

        # Add inter-residue bonds
        inter_bonds = self.get_inter_residue_bonds()
        for from_atom, to_atom in inter_bonds:
            all_connectivity[from_atom].append(to_atom)
            all_connectivity[to_atom].append(from_atom)

        return all_connectivity


def generate_conect_records(connectivity):
    """Generate CONECT records from connectivity dictionary."""
    conect_lines = []

    for from_atom in sorted(connectivity.keys()):
        to_atoms = sorted(set(connectivity[from_atom]))
        if to_atoms:
            # CONECT format: columns 7-11: from atom, columns 12-31: up to 4 connected atoms
            line = f"CONECT{from_atom:5d}"

            # Split into groups of 4 (PDB format allows 4 connections per line)
            for i in range(0, len(to_atoms), 4):
                if i > 0:
                    line = f"CONECT{from_atom:5d}"
                group = to_atoms[i:i+4]
                for to_atom in group:
                    line += f"{to_atom:5d}"
                conect_lines.append(line)

    return conect_lines


def fix_pdb_file(input_file, output_file):
    """Fix a PDB file by adding complete CONECT records."""
    print(f"Reading PDB file: {input_file}")

    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Filter out existing CONECT records
    atom_lines = [line for line in lines if line.startswith('ATOM') or line.startswith('HETATM')]
    other_lines = [line for line in lines if not line.startswith('CONECT') and not line.startswith('END')]

    # Build connectivity
    builder = ConnectivityBuilder()
    builder.parse_pdb(atom_lines)
    connectivity = builder.build_all_connectivity()

    # Generate CONECT records
    conect_lines = generate_conect_records(connectivity)

    # Write output
    print(f"Writing output to: {output_file}")
    print(f"Generated {len(conect_lines)} CONECT records for {len(atom_lines)} atoms")

    with open(output_file, 'w') as f:
        # Write ATOM/HETATM records
        for line in atom_lines:
            f.write(line)

        # Write other records (TER, etc.)
        for line in other_lines:
            if not line.startswith('CONECT'):
                f.write(line)

        # Write CONECT records
        for line in conect_lines:
            f.write(line + '\n')

        # Write END
        if any(line.startswith('END') for line in lines):
            f.write('END\n')

    print("Done!")
    return len(conect_lines)


def main():
    """Main entry point."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python fix_glycan_conect.py <input.pdb> [output.pdb]")
        print("\nIf output file is not specified, will create <input>_fixed.pdb")
        sys.exit(1)

    input_file = sys.argv[1]

    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        base, ext = input_file.rsplit('.', 1)
        output_file = f"{base}_fixed.{ext}"

    try:
        fix_pdb_file(input_file, output_file)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
