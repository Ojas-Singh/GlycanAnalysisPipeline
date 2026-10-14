"""Bonds missing from a MOL2 are completed from distances, and residue rings are checked.

GS01019 (G04508HC) and twelve other entries were analysed with an open ring: their MOL2 lacks bonds
inside a residue (the core fucose of G04508HC lacks C1-O5 and C1-C2), and the former recovery only
added bonds that joined disconnected components, so the ring stayed open and its anomeric carbon had
no phi.  The recovery now adds every undeclared pair at covalent distance in most frames, checks each
residue with a ring template, and stops when a bond cannot be resolved.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from lib import frame_quality
from lib.frame_quality import filter_broken_frames, recover_topology
from lib.glypdbio import parse_mol2_bonds
from lib.glystore import _parse_pdb_atom_line, store_from_pdb
from lib.residue_templates import (
    ALDOFURANOSE,
    ALDOPYRANOSE,
    KETOFURANOSE,
    KETOPYRANOSE,
    ring_template,
)
from lib.torsion import build_torsion_metadata, get_glycan_torsions_enhanced

FIXTURES = Path(__file__).parent / "fixtures" / "linkages"


def fixture_atoms(name: str):
    """Atoms, MOL2 bonds (index pairs) and coordinates (1 frame) of a structure in ``fixtures/linkages``."""
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
    pairs = [tuple(sorted((index[a], index[b]))) for a, b in parse_mol2_bonds(FIXTURES / name / "structure.mol2")]
    return atoms, sorted(set(pairs)), xyz


def atom(atoms: pl.DataFrame, name: str, residue: int) -> int:
    return int(atoms.with_row_index().filter((pl.col("name") == name) & (pl.col("res_id") == residue))["index"][0])


def frames_from(xyz: np.ndarray, count: int = 20, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (xyz + rng.normal(0.0, 0.02, size=(count,) + xyz.shape[1:])).astype(np.float32)


@pytest.mark.parametrize("residue_name,template", [
    ("0fA", ALDOPYRANOSE), ("4YB", ALDOPYRANOSE), ("UYS", ALDOPYRANOSE), ("0SA", KETOPYRANOSE),
    ("8GL", KETOPYRANOSE), ("0KN", KETOPYRANOSE), ("0JA", KETOPYRANOSE), ("5AD", ALDOFURANOSE),
    ("2aU", ALDOFURANOSE), ("1CU", KETOFURANOSE), ("0PD", KETOFURANOSE),
    ("SO3", None), ("ROH", None), ("MEX", None), ("GZL", None),
])
def test_ring_templates_from_glycam_names(residue_name, template):
    assert ring_template(residue_name) == template


def test_a_ring_bond_missing_from_the_mol2_is_completed():
    atoms, pairs, xyz = fixture_atoms("fuc_a1_6_glcnac")
    c1, o5 = atom(atoms, "C1", 3), atom(atoms, "O5", 3)
    without = [pair for pair in pairs if pair != tuple(sorted((c1, o5)))]
    assert len(without) == len(pairs) - 1
    recovered = recover_topology(frames_from(xyz), atoms, without)
    assert recovered.completed_from_distances == [tuple(sorted((c1, o5)))]
    assert recovered.joined_components == []
    assert recovered.ring_templates_checked == 2
    bonds = pl.DataFrame({"a": [a for a, _ in recovered.bonds], "b": [b for _, b in recovered.bonds]})
    metadata = build_torsion_metadata(get_glycan_torsions_enhanced(atoms, bonds), atoms, bonds_df=bonds)
    names = {e["canonical_name"] for e in metadata if e["family"] == "glycosidic"}
    assert names == {"3_2_phi", "3_2_psi", "3_2_omega"}
    # no torsion about a ring bond of the fucose (they were there while the ring was open)
    assert not [e for e in metadata if e["residue_ids"][1] == e["residue_ids"][2] == 3
                and set(e["central_bond_atoms"]) <= set(ALDOPYRANOSE)]


def test_the_former_recovery_left_that_ring_open():
    # without C1-O5 the fucose is still one component, so joining components adds nothing
    atoms, pairs, xyz = fixture_atoms("fuc_a1_6_glcnac")
    c1, o5 = atom(atoms, "C1", 3), atom(atoms, "O5", 3)
    without = [pair for pair in pairs if pair != tuple(sorted((c1, o5)))]
    graph = frame_quality._UnionFind(len(atoms))
    for a, b in without:
        graph.union(a, b)
    assert graph.component_count() == 1


def test_a_missing_glycosidic_bond_is_completed():
    # the MOL2 of this entry lacks one bond, as 302 entries lack a bond of a branch
    atoms, pairs, xyz = fixture_atoms("man_a1_6_man_core")
    recovered = recover_topology(frames_from(xyz), atoms, pairs)
    assert len(recovered.completed_from_distances) == 1
    assert recovered.joined_components == []
    assert recovered.ring_templates_checked == 5


def test_a_bond_that_cannot_be_resolved_stops_the_recovery():
    # in every frame a hydrogen of the GlcNAc sits at bonding distance from C2 of the fucose,
    # whose four bonds are all declared: no bond can be added, and none can be left out
    atoms, pairs, xyz = fixture_atoms("fuc_a1_6_glcnac")
    frames = frames_from(xyz)
    h, c2 = atom(atoms, "H1", 2), atom(atoms, "C2", 3)
    frames[:, h] = frames[:, c2] + np.array([1.0, 0.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="cannot be resolved"):
        recover_topology(frames, atoms, pairs)


def ring_residue(close_ring: bool, chord: bool = False):
    """The ring atoms of a glucopyranose (residue 0GB) on a zigzag: bonds of 1.46 A, C1 and O5 6.3 A apart."""
    names = list(ALDOPYRANOSE)
    atoms = pl.DataFrame({"name": names, "element": [n[0] for n in names], "res_name": ["0GB"] * 6, "res_id": [1] * 6})
    xyz = np.asarray([[1.25 * i, 0.75 * (i % 2), 0.0] for i in range(6)], dtype=np.float32)[None]
    pairs = [(i, i + 1) for i in range(5)]
    if close_ring:
        pairs.append((0, 5))
    if chord:
        pairs.append((0, 3))
    return atoms, pairs, np.repeat(xyz, 4, axis=0)


def test_a_ring_that_distances_cannot_close_stops_the_recovery():
    atoms, pairs, frames = ring_residue(close_ring=False)
    with pytest.raises(ValueError, match="ring of residue 0GB 1 is open: no bond O5-C1"):
        recover_topology(frames, atoms, pairs)


def test_a_bond_across_a_ring_stops_the_recovery():
    atoms, pairs, frames = ring_residue(close_ring=True, chord=True)
    with pytest.raises(ValueError, match="bond across it: C1-C4"):
        recover_topology(frames, atoms, pairs)


def test_the_store_manifest_records_the_completed_bonds(tmp_path):
    atoms, pairs, xyz = fixture_atoms("fuc_a1_6_glcnac")
    serials = atoms["serial"].to_list()
    c1, o5 = atom(atoms, "C1", 3), atom(atoms, "O5", 3)
    serial_pairs = [(serials[a], serials[b]) for a, b in pairs if (a, b) != tuple(sorted((c1, o5)))]
    store = tmp_path / "data"
    store_from_pdb(str(FIXTURES / "fuc_a1_6_glcnac" / "topology.pdb"), str(store),
                   get_connectivity=serial_pairs, connectivity_kind="serial")
    result = filter_broken_frames(store)
    manifest = json.loads((store / "frame_filter.json").read_text())
    assert manifest["filter_version"] == frame_quality.FILTER_VERSION == 2
    assert manifest["bonds_completed_from_distances"] == 1
    assert manifest["bonds_joining_components"] == 0
    assert manifest["ring_templates_checked"] == 2
    assert result.inferred_bonds == 1 and result.topology_repaired
    stored = {tuple(row) for row in pl.read_parquet(store / "bonds.parquet").rows()}
    assert tuple(sorted((c1, o5))) in stored
