"""The definitions of the stored glycosidic torsions, and phi/psi of furanose donors.

The stored omega of a 1-6 linkage is the IUPAC O5-C5-C6-O6, so gg, gt and tg are near -60, +60 and
180. Up to version 1 of the definitions it was C4-C5-C6-O6, that angle plus about 120 degrees.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from lib import v2
from lib.torsion import (
    TORSION_CONVENTIONS,
    build_torsion_metadata,
    classify_glycosidic_from_name,
    get_glycan_torsions_enhanced,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "dummy_database_data" / "DManpa1-2DManpa1-OH" / "DManpa1-2DManpa1-OH.pdb"
OMEGA_ATOMS = ("O5", "C5", "C6", "O6")
RING = [("C1", "C2"), ("C2", "C3"), ("C3", "C4"), ("C4", "C5"), ("C5", "O5"), ("O5", "C1"),
        ("C2", "O2"), ("C3", "O3"), ("C4", "O4"), ("C5", "C6"), ("C6", "O6")]


def dihedral(p0, p1, p2, p3) -> float:
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def wrap(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def first_model():
    residues = {}
    for line in EXAMPLE.read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if line.startswith("ATOM"):
            residues.setdefault(int(line[22:26]), {})[line[12:16].strip()] = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return residues


def rotate_about(point, origin, axis, degrees):
    axis = axis / np.linalg.norm(axis)
    angle = np.radians(degrees)
    v = point - origin
    return origin + v * np.cos(angle) + np.cross(axis, v) * np.sin(angle) + axis * np.dot(axis, v) * (1 - np.cos(angle))


@pytest.mark.parametrize("residue", [2, 3])
def test_stored_omega_is_the_iupac_omega(residue):
    atoms = first_model()[residue]
    offsets = []
    for target, rotamer in ((-60.0, "gg"), (60.0, "gt"), (180.0, "tg")):
        # turn O6 about C5-C6 until O5-C5-C6-O6 is the IUPAC value of the rotamer
        now = dihedral(atoms["O5"], atoms["C5"], atoms["C6"], atoms["O6"])
        o6 = rotate_about(atoms["O6"], atoms["C6"], atoms["C6"] - atoms["C5"], target - now)
        positions = {**atoms, "O6": o6}
        stored = dihedral(*(positions[name] for name in OMEGA_ATOMS))
        assert abs(wrap(stored - target)) < 1e-6, rotamer
        # the former definition, C4-C5-C6-O6, lay about 120 degrees from it
        offsets.append(wrap(dihedral(atoms["C4"], atoms["C5"], atoms["C6"], o6) - target))
    assert all(105.0 < offset < 135.0 for offset in offsets)


def man_a1_6_man():
    """Heavy atoms and bonds of Man(a1-6)Man: residue 2 is the acceptor, residue 3 the donor."""
    names, res = [], []
    for residue in (2, 3):
        for name in ("C1", "C2", "C3", "C4", "C5", "C6", "O2", "O3", "O4", "O5", "O6"):
            names.append(name)
            res.append(residue)
    index = {(r, n): i for i, (r, n) in enumerate(zip(res, names))}
    bonds = [(index[(r, a)], index[(r, b)]) for r in (2, 3) for a, b in RING]
    bonds.append((index[(3, "C1")], index[(2, "O6")]))
    atoms = pl.DataFrame({"name": names, "element": [n[0] for n in names], "res_id": res,
                          "res_name": ["VMA" if r == 2 else "0MA" for r in res]})
    return atoms, pl.DataFrame({"a": [a for a, _ in bonds], "b": [b for _, b in bonds]})


def test_detected_omega_of_a_1_6_linkage_is_o5_c5_c6_o6():
    atoms, bonds = man_a1_6_man()
    torsions = get_glycan_torsions_enhanced(atoms, bonds)
    metadata = build_torsion_metadata(torsions, atoms, bonds_df=bonds)
    glycosidic = {e["canonical_name"]: e for e in metadata if e["family"] == "glycosidic"}
    assert set(glycosidic) == {"3_2_phi", "3_2_psi", "3_2_omega"}
    assert glycosidic["3_2_omega"]["atom_names"] == list(OMEGA_ATOMS)
    assert glycosidic["3_2_omega"]["residue_ids"] == [2, 2, 2, 2]
    assert glycosidic["3_2_phi"]["atom_names"] == ["O5", "C1", "O6", "C6"]
    assert glycosidic["3_2_psi"]["atom_names"] == ["C1", "O6", "C6", "C5"]
    assert glycosidic["3_2_psi"]["residue_ids"] == [3, 2, 2, 2]


def test_conventions_are_written_with_the_torsions():
    assert TORSION_CONVENTIONS["omega"]["atoms"]["1-6 to a hexopyranose"] == list(OMEGA_ATOMS)
    assert "O5-C5-C6-O6" in TORSION_CONVENTIONS["omega"]["definition"]
    block = v2._level_summary_from_metadata([])
    assert block["conventions"] == TORSION_CONVENTIONS
    assert block["substituent"] == []


@pytest.mark.parametrize("name,expected", [
    ("O4_3-C1_3-O5_2-C5_2", "3_2_phi"),          # Araf(a1-5)Araf: aldofuranose donor, ring oxygen O4
    ("C5_2-O5_2-C1_3-O4_3", "3_2_phi"),          # the same torsion written the other way round
    ("O4_3-C1_3-O3_2-C3_2", "3_2_phi"),          # Araf(1-3)
    ("O5_2-C2_2-O1_1-C1_1", "2_1_phi"),          # ketofuranose donor (Fruf), anomeric C2, ring oxygen O5
    ("C1_3-O5_2-C5_2-C4_2", "3_2_psi"),          # psi of a 1-5 linkage: the glycosidic oxygen is O5
    ("O5_3-C1_3-O5_2-C5_2", "3_2_phi"),          # pyranose donor on the O5 of a furanose
    ("O5_3-C1_3-O4_2-C4_2", "3_2_phi"),          # unchanged: pyranose donor
    ("C1_3-O4_2-C4_2-C3_2", "3_2_psi"),          # unchanged
])
def test_furanose_and_1_5_linkages_are_glycosidic(name, expected):
    assert classify_glycosidic_from_name(name) == expected


@pytest.mark.parametrize("name", [
    "O4_3-C1_3-C2_3-C3_3",                       # ring torsion of a furanose
    "O4_2-C1_3-O5_3-C5_3",                       # pyranose ring torsion with a flank in the acceptor
    "C4_2-C5_2-C6_2-O6_2",                       # free C6 side chain
    "O5_2-C2_2-C3_2-C4_2",                       # ring torsion of a ketofuranose
])
def test_ring_and_side_chain_torsions_stay_non_glycosidic(name):
    assert classify_glycosidic_from_name(name).startswith("other_")
