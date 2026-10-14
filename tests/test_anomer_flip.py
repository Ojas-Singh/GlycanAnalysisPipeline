"""The other anomer of the representative structures (lib/v2.py ``_flip_anomer_model``), on real topologies.

The fixtures are the first frames (``topology.pdb``) of four entries of the raw archive: three 2-ketoses at the
reducing end (Kdo, a Neu5Ac trisaccharide, a Fru trisaccharide whose reducing Fru carries the next residue on O1)
and an aldose (Gal).  The audit of the published database (issue ``level1-file-anomer``) found that for the
ketoses the generated "other anomer" file still held the simulated one: the flip only knew C1/H1.  A flip has to
invert the configuration at the anomeric carbon, keep every bond length, and move nothing else but the exocyclic
groups it exchanges (with what is linked to them).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lib import frame_quality
from lib.v2 import _flip_anomer_model

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "anomer"
NEU5AC = "DNeup5Aca2-9DNeup5Aca2-9DNeup5Aca2-OH"
FRU = "DFrufb2-1DFrufb2-1DFrufa2-OH"


def read(name):
    names, residues, res_names, elements, xyz = [], [], [], [], []
    for line in (FIXTURES / name / "topology.pdb").read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            names.append(line[12:16].strip())
            res_names.append(line[17:21].strip())
            residues.append(int(line[22:26]))
            elements.append(line[76:78].strip() or line[12:16].strip()[0])
            xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return names, residues, res_names, elements, np.asarray(xyz)


def bonds_by_distance(elements, xyz):
    """Pairs within the covalent distance of their elements (one frame)."""
    symbols = [frame_quality._normalise_element(element) for element in elements]
    pairs = []
    for a in range(len(xyz)):
        for b in range(a + 1, len(xyz)):
            if frame_quality._chemically_allowed(symbols[a], symbols[b]) and \
                    np.linalg.norm(xyz[a] - xyz[b]) <= frame_quality._covalent_cutoff(symbols[a], symbols[b]):
                pairs.append((a, b))
    return pairs


def index(names, residues, name, residue):
    return next(i for i, (n, r) in enumerate(zip(names, residues)) if n == name and r == residue)


def chirality(xyz, centre, a, b, c):
    """Sign of the volume of three substituents about a centre: the configuration of that centre."""
    return float(np.sign(np.dot(xyz[a] - xyz[centre], np.cross(xyz[b] - xyz[centre], xyz[c] - xyz[centre]))))


CASES = [
    # entry, anomeric carbon, ring oxygen, ring neighbour, reducing residue code -> other anomer's code
    ("DKDOpa2-OH", "C2", "O6", "C3", ("0KO", "0KO")),          # Kdo: the code does not name the anomer
    (NEU5AC, "C2", "O6", "C3", ("9SA", "9SB")),
    (FRU, "C2", "O5", "C3", ("1CD", "1CU")),                    # furanose: D alpha, U beta
    ("DGalpa1-OH", "C1", "O5", "C2", ("0LA", "0LB")),
]


@pytest.mark.parametrize("entry, anomeric, ring_oxygen, neighbour, codes", CASES, ids=[case[0][:20] for case in CASES])
def test_the_other_anomer_inverts_the_anomeric_carbon_and_nothing_else(entry, anomeric, ring_oxygen, neighbour, codes):
    names, residues, res_names, elements, xyz = read(entry)
    bonds = bonds_by_distance(elements, xyz)
    centre = index(names, residues, anomeric, 2)
    oxygen = index(names, residues, "O1", 1)                         # the aglycone (ROH) oxygen
    ring_o, ring_c = index(names, residues, ring_oxygen, 2), index(names, residues, neighbour, 2)
    assert res_names[centre] == codes[0]

    flipped, flipped_names = _flip_anomer_model(xyz, names, residues, res_names, target_anomer="beta", bonds=bonds)

    before = chirality(xyz, centre, ring_o, ring_c, oxygen)
    after = chirality(flipped, centre, ring_o, ring_c, oxygen)
    assert before != 0 and after == -before                          # the configuration at the anomeric carbon
    for a, b in bonds:                                               # every bond keeps its length
        assert np.linalg.norm(flipped[a] - flipped[b]) == pytest.approx(np.linalg.norm(xyz[a] - xyz[b]), abs=1e-6)
    ring = [i for i, (name, residue) in enumerate(zip(names, residues))
            if residue == 2 and name in (anomeric, ring_oxygen, neighbour, "C4", "C5", "C6", "O5", "O4")]
    assert np.allclose(flipped[ring], xyz[ring])                      # the ring stays where it is
    assert {flipped_names[i] for i in range(len(names)) if residues[i] == 2} == {codes[1]}


def test_a_ketose_moves_its_c1_branch_with_what_is_linked_to_it():
    names, residues, res_names, elements, xyz = read(FRU)
    bonds = bonds_by_distance(elements, xyz)
    flipped, _ = _flip_anomer_model(xyz, names, residues, res_names, target_anomer="beta", bonds=bonds)
    moved = {residues[i] for i in range(len(names)) if not np.allclose(flipped[i], xyz[i])}
    assert moved == {1, 2, 3, 4}                         # ROH, the C1 branch of residue 2, and the two Fru on its O1
    downstream = [i for i, residue in enumerate(residues) if residue in (3, 4)]
    distances = lambda coordinates: np.linalg.norm(coordinates[downstream][:, None] - coordinates[downstream][None], axis=2)
    assert np.allclose(distances(flipped), distances(xyz))            # moved as one body
    # the C1 branch takes the place of the aglycone oxygen and the other way round
    c2, c1, o1 = index(names, residues, "C2", 2), index(names, residues, "C1", 2), index(names, residues, "O1", 1)
    direction = lambda coordinates, atom: (coordinates[atom] - coordinates[c2]) / np.linalg.norm(coordinates[atom] - coordinates[c2])
    assert np.allclose(direction(flipped, c1), direction(xyz, o1)) and np.allclose(direction(flipped, o1), direction(xyz, c1))
    # and the flip of the flip is the simulated structure
    again, _ = _flip_anomer_model(flipped, names, residues, res_names, target_anomer="alpha", bonds=bonds)
    assert np.allclose(again, xyz)


def test_a_ketose_without_a_bond_table_moves_the_atoms_named_after_c1():
    names, residues, res_names, elements, xyz = read("DKDOpa2-OH")
    with_bonds, _ = _flip_anomer_model(xyz, names, residues, res_names, target_anomer="beta",
                                       bonds=bonds_by_distance(elements, xyz))
    without, _ = _flip_anomer_model(xyz, names, residues, res_names, target_anomer="beta")
    assert np.allclose(with_bonds, without)


def test_the_aldose_flip_is_the_one_it_was():
    """Aldoses keep their exchange of H1 and the ROH oxygen: the files of the database do not change."""
    names, residues, res_names, elements, xyz = read("DGalpa1-OH")
    flipped, _ = _flip_anomer_model(xyz, names, residues, res_names, target_anomer="beta",
                                    bonds=bonds_by_distance(elements, xyz))
    c1, h1 = index(names, residues, "C1", 2), index(names, residues, "H1", 2)
    o1, ho1 = index(names, residues, "O1", 1), index(names, residues, "HO1", 1)
    unit = lambda vector: vector / np.linalg.norm(vector)
    assert np.allclose(flipped[o1], xyz[c1] + unit(xyz[h1] - xyz[c1]) * np.linalg.norm(xyz[o1] - xyz[c1]))
    assert np.allclose(flipped[h1], xyz[c1] + unit(xyz[o1] - xyz[c1]) * np.linalg.norm(xyz[h1] - xyz[c1]))
    assert np.allclose(flipped[ho1], flipped[o1] + (xyz[ho1] - xyz[o1]))
    untouched = [i for i in range(len(names)) if i not in (o1, h1, ho1)]
    assert np.array_equal(flipped[untouched], xyz[untouched])
