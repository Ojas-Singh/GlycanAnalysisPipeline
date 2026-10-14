"""The IUPAC torsion definitions, checked by hand on real linkages of the collection.

The structures in ``tests/fixtures/linkages/`` are the first frame (``topology.pdb``) and the MOL2 of
raw archive entries; ``glycam_name.txt`` names each one.  For every linkage the four atoms of phi,
psi and omega are written out below as the IUPAC-IUB recommendations (1981) define them, with the
same choice of reference atom where they do not spell it out (``lib.torsion.TORSION_CONVENTIONS``):

- phi   O5'-C1'-Ox-Cx (O4'-C1' aldofuranose, O6'-C2' sialic acid, O5'-C2' ketofuranose)
- psi   C1'-Ox-Cx-C(x-1) (C2' for ketose donors; C2 where the acceptor is the C1 of a ketose)
- omega O5-C5-C6-O6 for 1-6, and O-C(x-1)-Cx-Ox for every acceptor carbon outside the ring
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from lib import v2
from lib.frame_quality import recover_topology
from lib.glypdbio import parse_mol2_bonds
from lib.glystore import _parse_pdb_atom_line
from lib.torsion import (
    TORSION_CONVENTIONS,
    build_torsion_metadata,
    get_glycan_torsions_enhanced,
    get_torsion_values_batch,
)

FIXTURES = Path(__file__).parent / "fixtures" / "linkages"


def load(name: str):
    """Atoms, bonds (MOL2 completed from distances) and coordinates (1 frame) of a fixture."""
    folder = FIXTURES / name
    rows = [
        _parse_pdb_atom_line(line)
        for line in (folder / "topology.pdb").read_text().splitlines()
        if line[:6] in ("ATOM  ", "HETATM")
    ]
    atoms = pl.DataFrame({
        "serial": [r[0] for r in rows], "name": [r[1] for r in rows], "element": [r[2] for r in rows],
        "res_name": [r[3] for r in rows], "res_id": [r[4] for r in rows],
    })
    xyz = np.asarray([[r[6], r[7], r[8]] for r in rows], dtype=np.float32)[None]
    index = {serial: i for i, serial in enumerate(atoms["serial"].to_list())}
    pairs = [(index[a], index[b]) for a, b in parse_mol2_bonds(folder / "structure.mol2")]
    recovered = recover_topology(xyz, atoms, pairs)
    bonds = pl.DataFrame({"a": [a for a, _ in recovered.bonds], "b": [b for _, b in recovered.bonds]})
    return atoms, bonds, xyz


def linkage_torsions(name: str):
    atoms, bonds, xyz = load(name)
    torsions = get_glycan_torsions_enhanced(atoms, bonds)
    metadata = build_torsion_metadata(torsions, atoms, bonds_df=bonds)
    found = {
        entry["canonical_name"]: (entry["family"], [f"{a}_{r}" for a, r in zip(entry["atom_names"], entry["residue_ids"])])
        for entry in metadata
        if entry["family"] in ("glycosidic", "substituent")
    }
    return found, metadata, atoms, xyz


# (fixture, canonical name, family, the four atoms as <atom>_<residue>); the donor is the first residue
# of the name.
HAND_CHECKED = [
    # Gal(b1-4)GlcNAc: Gal 3 on O4 of GlcNAc 2
    ("gal_b1_4_glcnac", "3_2_phi", "glycosidic", ["O5_3", "C1_3", "O4_2", "C4_2"]),
    ("gal_b1_4_glcnac", "3_2_psi", "glycosidic", ["C1_3", "O4_2", "C4_2", "C3_2"]),
    # Man(a1-6)Man in the N-glycan core: Man 6 on O6 of the beta-Man 5
    ("man_a1_6_man_core", "6_5_phi", "glycosidic", ["O5_6", "C1_6", "O6_5", "C6_5"]),
    ("man_a1_6_man_core", "6_5_psi", "glycosidic", ["C1_6", "O6_5", "C6_5", "C5_5"]),
    ("man_a1_6_man_core", "6_5_omega", "glycosidic", ["O5_5", "C5_5", "C6_5", "O6_5"]),
    # and its Man(b1-4)GlcNAc: psi on C3, not C5
    ("man_a1_6_man_core", "5_4_psi", "glycosidic", ["C1_5", "O4_4", "C4_4", "C3_4"]),
    # Neu5Ac(a2-3)Gal: the anomeric carbon is C2, the ring oxygen O6; psi on C2 of Gal, not C4
    ("neu5ac_a2_3_gal", "3_2_phi", "glycosidic", ["O6_3", "C2_3", "O3_2", "C3_2"]),
    ("neu5ac_a2_3_gal", "3_2_psi", "glycosidic", ["C2_3", "O3_2", "C3_2", "C2_2"]),
    # Neu5Ac(a2-6)Gal: a 1-6 type linkage with omega
    ("neu5ac_a2_6_gal", "4_3_phi", "glycosidic", ["O6_4", "C2_4", "O6_3", "C6_3"]),
    ("neu5ac_a2_6_gal", "4_3_psi", "glycosidic", ["C2_4", "O6_3", "C6_3", "C5_3"]),
    ("neu5ac_a2_6_gal", "4_3_omega", "glycosidic", ["O5_3", "C5_3", "C6_3", "O6_3"]),
    # Fuc(a1-6)GlcNAc (core fucose)
    ("fuc_a1_6_glcnac", "3_2_phi", "glycosidic", ["O5_3", "C1_3", "O6_2", "C6_2"]),
    ("fuc_a1_6_glcnac", "3_2_psi", "glycosidic", ["C1_3", "O6_2", "C6_2", "C5_2"]),
    ("fuc_a1_6_glcnac", "3_2_omega", "glycosidic", ["O5_2", "C5_2", "C6_2", "O6_2"]),
    # D-Araf: ring oxygen O4; 1-5 to Araf has an omega about C4-C5 (C5 is outside the ring)
    ("araf", "3_2_phi", "glycosidic", ["O4_3", "C1_3", "O5_2", "C5_2"]),
    ("araf", "3_2_psi", "glycosidic", ["C1_3", "O5_2", "C5_2", "C4_2"]),
    ("araf", "3_2_omega", "glycosidic", ["O4_2", "C4_2", "C5_2", "O5_2"]),
    ("araf", "5_4_phi", "glycosidic", ["O4_5", "C1_5", "O2_4", "C2_4"]),
    ("araf", "5_4_psi", "glycosidic", ["C1_5", "O2_4", "C2_4", "C1_4"]),
    # Fruf(b2-1)Fruf: ketofuranose donor (O5, C2); the acceptor C1 has no C0, so psi takes C2, and the
    # exocyclic C1 gives an omega about C1-C2 with the ring oxygen
    ("fruf_b2_1", "3_2_phi", "glycosidic", ["O5_3", "C2_3", "O1_2", "C1_2"]),
    ("fruf_b2_1", "3_2_psi", "glycosidic", ["C2_3", "O1_2", "C1_2", "C2_2"]),
    ("fruf_b2_1", "3_2_omega", "glycosidic", ["O5_2", "C2_2", "C1_2", "O1_2"]),
    # Neu5Ac(a2-8)Neu5Ac and (a2-9): omega with the hydroxyl oxygen of C7 / C8
    ("neu5ac_a2_8", "3_2_psi", "glycosidic", ["C2_3", "O8_2", "C8_2", "C7_2"]),
    ("neu5ac_a2_8", "3_2_omega", "glycosidic", ["O7_2", "C7_2", "C8_2", "O8_2"]),
    ("neu5ac_a2_9", "3_2_psi", "glycosidic", ["C2_3", "O9_2", "C9_2", "C8_2"]),
    ("neu5ac_a2_9", "3_2_omega", "glycosidic", ["O8_2", "C8_2", "C9_2", "O9_2"]),
    # Gal6S: the sulfate (residue 4) on O6 of Gal 3 is a substituent, not a linkage
    ("gal6s_b1_4_glc", "4_3_omega", "substituent", ["O5_3", "C5_3", "C6_3", "O6_3"]),
]


@pytest.mark.parametrize("fixture,name,family,atoms", HAND_CHECKED, ids=[f"{f}:{n}" for f, n, _, _ in HAND_CHECKED])
def test_hand_checked_linkage_torsions(fixture, name, family, atoms):
    found, *_ = linkage_torsions(fixture)
    assert found[name] == (family, atoms)


@pytest.mark.parametrize("fixture,expected", [
    ("gal_b1_4_glcnac", {"3_2_phi", "3_2_psi"}),
    ("fuc_a1_6_glcnac", {"3_2_phi", "3_2_psi", "3_2_omega"}),
    ("neu5ac_a2_3_gal", {"3_2_phi", "3_2_psi"}),
    ("gal6s_b1_4_glc", {"3_2_phi", "3_2_psi", "4_3_omega"}),
    ("fruf_b2_1", {"3_2_phi", "3_2_psi", "3_2_omega", "4_3_phi", "4_3_psi", "4_3_omega"}),
])
def test_no_other_linkage_torsions(fixture, expected):
    found, *_ = linkage_torsions(fixture)
    assert set(found) == expected


def dihedral(p0, p1, p2, p3) -> float:
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def wrap(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def position(atoms: pl.DataFrame, xyz: np.ndarray, label: str) -> np.ndarray:
    atom, residue = label.rsplit("_", 1)
    index = atoms.with_row_index().filter((pl.col("name") == atom) & (pl.col("res_id") == int(residue)))["index"][0]
    return xyz[0, int(index)].astype(np.float64)


@pytest.mark.parametrize("fixture,name", [("man_a1_6_man_core", "6_5_omega"), ("fuc_a1_6_glcnac", "3_2_omega"),
                                          ("neu5ac_a2_6_gal", "4_3_omega")])
def test_omega_is_the_iupac_angle_and_the_former_one_was_120_degrees_off(fixture, name):
    found, metadata, atoms, xyz = linkage_torsions(fixture)
    entry = next(e for e in metadata if e["canonical_name"] == name)
    stored = float(get_torsion_values_batch(xyz.astype(np.float64), [tuple(entry["atom_indices"])])[0, 0])
    o5, c5, c6, o6, c4 = (position(atoms, xyz, f"{a}_{entry['residue_ids'][0]}") for a in ("O5", "C5", "C6", "O6", "C4"))
    assert stored == pytest.approx(dihedral(o5, c5, c6, o6), abs=1e-3)
    # the starting structures are built in a staggered rotamer: gg -60, gt +60 or tg 180 on this scale
    assert min(abs(wrap(stored - centre)) for centre in (-60.0, 60.0, 180.0)) < 30.0
    assert 105.0 < wrap(dihedral(c4, c5, c6, o6) - stored) < 135.0


@pytest.mark.parametrize("fixture,name,former", [
    ("gal_b1_4_glcnac", "3_2_psi", "C5_2"),          # the bond table had C5 first
    ("neu5ac_a2_3_gal", "3_2_psi", "C4_2"),          # the sialic acid rule took C3-C4
])
def test_psi_with_c_x_plus_1_was_about_120_degrees_from_the_iupac_psi(fixture, name, former):
    found, metadata, atoms, xyz = linkage_torsions(fixture)
    labels = found[name][1]
    p = [position(atoms, xyz, label) for label in labels]
    old = dihedral(p[0], p[1], p[2], position(atoms, xyz, former))
    assert 110.0 < abs(wrap(dihedral(*p) - old)) < 130.0


def test_conventions_list_the_four_atoms_of_every_case():
    assert TORSION_CONVENTIONS["version"] == 2
    assert TORSION_CONVENTIONS["phi"]["atoms"]["aldopyranose donor"] == ["O5'", "C1'", "Ox", "Cx"]
    assert TORSION_CONVENTIONS["psi"]["atoms"]["aldose donor, x > 1"] == ["C1'", "Ox", "Cx", "C(x-1)"]
    assert TORSION_CONVENTIONS["psi"]["examples"]["2-3 (sialic acid)"] == ["C2'", "O3", "C3", "C2"]
    assert TORSION_CONVENTIONS["omega"]["atoms"]["1-6 to a hexopyranose"] == ["O5", "C5", "C6", "O6"]
    assert TORSION_CONVENTIONS["omega"]["rotamers"]["1-6 to a D-hexopyranose"] == {"gg": -60, "gt": 60, "tg": 180}
    for block in ("phi", "psi", "omega", "substituent"):
        assert all(len(atoms) == 4 for atoms in TORSION_CONVENTIONS[block]["atoms"].values())


def test_info_json_lists_substituent_torsions_apart():
    _, metadata, *_ = linkage_torsions("gal6s_b1_4_glc")
    block = v2._level_summary_from_metadata(metadata)
    assert block["conventions"] == TORSION_CONVENTIONS
    assert [e["canonical_name"] for e in block["substituent"]] == ["4_3_omega"]
    assert "4_3_omega" not in [e["canonical_name"] for e in block["glycosidic"]]
    assert "4_3_omega" not in [e["canonical_name"] for e in block["non_glycosidic"]]
    assert block["glycosidic_summary"]["omega_count"] == 0
    assert block["glycosidic_summary"]["substituent_torsions"] == 1
    assert block["family_summary"]["substituent"] == 1
    total = len(block["glycosidic"]) + len(block["substituent"]) + len(block["non_glycosidic"])
    assert total == len(metadata)
