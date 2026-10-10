"""Regression tests for PDB residue naming and connectivity provenance."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from lib.glypdbio import (
    _convert_pdb_text_naming,
    _pdb_conect_block,
    flip_anomeric_residue_names,
    load_bonds,
    normalize_anomeric_pdb_text,
    normalize_anomeric_residue_names,
    parse_mol2_bonds,
)
from lib.torsion import classify_glycosidic_from_name
from lib.glystore import store_from_pdb
from lib.v2 import _flip_anomer_model


def _atom_line(residue: str, serial: int = 1, res_id: int = 1) -> str:
    return (
        f"HETATM{serial:5d}   C1 {residue:>3s} A{res_id:4d}"
        "       0.000   0.000   0.000  1.00  0.00           C\n"
    )


class PdbResidueNamingTest(unittest.TestCase):
    def test_reducing_residue_name_follows_target_anomer(self) -> None:
        names = ["ROH", "UYB", "UYB", "4GA"]
        residue_ids = [1, 2, 2, 3]

        self.assertEqual(
            normalize_anomeric_residue_names(names, residue_ids, "alpha"),
            ["ROH", "UYA", "UYA", "4GA"],
        )
        self.assertEqual(
            normalize_anomeric_residue_names(names, residue_ids, "beta"),
            names,
        )
        self.assertEqual(
            flip_anomeric_residue_names(names, residue_ids),
            ["ROH", "UYA", "UYA", "4GA"],
        )

    def test_pdb_text_normalization_handles_glycam_and_pdb_names(self) -> None:
        glycam = _atom_line("4YA", serial=1, res_id=2)
        pdb = _atom_line("NDG", serial=2, res_id=2)
        self.assertEqual(
            normalize_anomeric_pdb_text(glycam + pdb, "beta").splitlines()[0][17:20],
            "4YB",
        )
        self.assertEqual(
            normalize_anomeric_pdb_text(glycam + pdb, "beta").splitlines()[1][17:20],
            "NAG",
        )

    def test_flipped_model_name_is_updated_even_when_geometry_is_degenerate(self) -> None:
        xyz = np.zeros((4, 3), dtype=float)
        names = ["O1", "HO1", "C1", "H1"]
        residue_ids = [1, 1, 2, 2]
        flipped_xyz, flipped_names = _flip_anomer_model(
            xyz,
            names,
            residue_ids,
            ["ROH", "ROH", "4YA", "4YA"],
            target_anomer="beta",
        )
        np.testing.assert_array_equal(flipped_xyz, xyz)
        self.assertEqual(flipped_names, ["ROH", "ROH", "4YB", "4YB"])

    def test_glycam_residue_mapping_is_single_pass_and_fixed_width(self) -> None:
        # These two cases caused the reported SNFG corruption: the old sequential
        # regex pass transformed NGA into GLC and BMA into MAN.
        cases = {
            "0VB": "NGA",  # beta-GalNAc: yellow SNFG circle
            "0MB": "BMA",  # beta-Man: green SNFG circle
            "0VA": "A2G",
            "0YA": "NDG",
            "0YB": "NAG",
            "0SA": "SIA",
        }
        conect = "CONECT    1    2\n"
        remark = "REMARK   1 0VB must not be converted outside atom records\n"

        for source, expected in cases.items():
            with self.subTest(source=source):
                original = _atom_line(source) + conect + remark
                converted = _convert_pdb_text_naming(original, "PDB")
                atom, emitted_conect, emitted_remark = converted.splitlines()
                self.assertEqual(atom[17:20], expected)
                self.assertEqual(atom[:17], original.splitlines()[0][:17])
                self.assertEqual(atom[20:], original.splitlines()[0][20:])
                self.assertEqual(emitted_conect, conect.rstrip())
                self.assertEqual(emitted_remark, remark.rstrip())

    def test_unknown_neu5ac_beta_source_is_left_unchanged(self) -> None:
        # There is no verified distinct GLYCAM source code for beta-Neu5Ac in
        # this converter.  It must not be guessed or overwritten as SLB.
        original = _atom_line("0SB")
        self.assertEqual(_convert_pdb_text_naming(original, "PDB"), original)

    def test_acceptor_owned_linkage_oxygen_is_classified_as_psi(self) -> None:
        # The linkage oxygen and acceptor carbon use the acceptor residue ID in
        # PDB/GLYCAM atom records. This is GS00971's missing distribution-plot
        # torsion and is not a residue-local bond torsion.
        self.assertEqual(
            classify_glycosidic_from_name("C1_3-O4_2-C4_2-C5_2"),
            "3_2_psi",
        )


class ConnectivityProvenanceTest(unittest.TestCase):
    def test_missing_pyranose_ring_closure_is_repaired(self) -> None:
        # The atom graph is still connected without C1--C2, so this specifically
        # exercises the residue-aware repair rather than the component heuristic.
        mol2 = """@<TRIPOS>MOLECULE
ring
6 5 0 0 0
SMALL
NO_CHARGES
@<TRIPOS>ATOM
1 C1 0.0 0.0 0.0 C.3 6 4YB 0.0
2 C2 1.0 0.0 0.0 C.3 6 4YB 0.0
3 O5 0.0 1.0 0.0 O.3 6 4YB 0.0
4 C5 1.0 1.0 0.0 C.3 6 4YB 0.0
5 C4 2.0 1.0 0.0 C.3 6 4YB 0.0
6 C3 2.0 0.0 0.0 C.3 6 4YB 0.0
@<TRIPOS>BOND
1 1 3 1
2 3 4 1
3 4 5 1
4 5 6 1
5 6 2 1
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ring.mol2"
            path.write_text(mol2, encoding="utf-8")
            self.assertIn((1, 2), parse_mol2_bonds(path))

    def test_emitted_conect_uses_all_stored_bonds(self) -> None:
        serials = np.array([10, 20, 30], dtype=np.int32)
        bonds = pl.DataFrame({"a": [0, 1], "b": [1, 2]})
        self.assertEqual(
            _pdb_conect_block(serials, bonds),
            "CONECT   10   20\nCONECT   20   10   30\nCONECT   30   20\n",
        )

    def test_conect_repairs_missing_covalent_bond_from_geometry(self) -> None:
        atoms = pl.DataFrame({
            "serial": [10, 20, 30],
            "name": ["C1", "O1", "HO1"],
            "res_name": ["GLC", "GLC", "GLC"],
            "res_id": [1, 1, 1],
            "chain_id_u8": [0, 0, 0],
            "element": ["C", "O", "H"],
        })
        bonds = pl.DataFrame({"a": [0], "b": [1]})
        xyz = np.array([[0.0, 0.0, 0.0], [1.43, 0.0, 0.0], [2.39, 0.0, 0.0]])
        self.assertEqual(
            _pdb_conect_block(atoms["serial"].to_numpy(), bonds, atoms, xyz),
            "CONECT   10   20\nCONECT   20   10   30\nCONECT   30   20\n",
        )

    def test_invalid_stored_bond_fails_instead_of_being_dropped(self) -> None:
        with self.assertRaisesRegex(ValueError, "Cannot emit PDB CONECT records"):
            _pdb_conect_block(
                np.array([10], dtype=np.int32),
                pl.DataFrame({"a": [0], "b": [1]}),
            )

    def test_mol2_serial_mismatch_fails_with_provenance_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pdb_path = Path(tmp) / "input.pdb"
            pdb_path.write_text(_atom_line("0VB", serial=1), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "MOL2 bond atom IDs do not match"):
                store_from_pdb(
                    str(pdb_path),
                    str(Path(tmp) / "stored"),
                    get_connectivity=[(1, 999)],
                    connectivity_kind="serial",
                )

    def test_input_bond_is_stored_and_emitted_as_conect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pdb_path = Path(tmp) / "input.pdb"
            pdb_path.write_text(
                _atom_line("0VB", serial=10) + _atom_line("0MB", serial=20),
                encoding="utf-8",
            )
            stored = Path(tmp) / "stored"
            store_from_pdb(
                str(pdb_path),
                str(stored),
                get_connectivity=[(10, 20)],
                connectivity_kind="serial",
            )
            bonds = load_bonds(str(stored))
            self.assertEqual(bonds.rows(), [(0, 1)])
            self.assertEqual(
                _pdb_conect_block(np.array([10, 20], dtype=np.int32), bonds),
                "CONECT   10   20\nCONECT   20   10\n",
            )


if __name__ == "__main__":
    unittest.main()
