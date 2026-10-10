import unittest
import tempfile
from pathlib import Path

import pandas as pd
import polars as pl

from lib.torsion import (
    build_torsion_metadata,
    classify_glycosidic_from_name,
    get_glycan_torsions_enhanced,
    plot_torsion_distribution,
)


class GlycosidicOmegaClassificationTests(unittest.TestCase):
    @staticmethod
    def _atoms():
        return pl.DataFrame(
            {
                "name": ["C4", "C5", "C6", "O6", "C1"],
                "res_id": [2, 2, 2, 2, 3],
                "res_name": ["GlcNAc", "GlcNAc", "GlcNAc", "GlcNAc", "GalNAc"],
            }
        )

    def test_residue_local_c6_torsion_is_not_glycosidic_omega(self):
        bonds = pl.DataFrame({"a": [0, 1, 2], "b": [1, 2, 3]})
        metadata = build_torsion_metadata(
            [(0, 1, 2, 3)],
            self._atoms(),
            bonds_df=bonds,
        )

        self.assertEqual(metadata[0]["family"], "residue_local")
        self.assertNotEqual(metadata[0]["subtype"], "omega")
        self.assertTrue(
            classify_glycosidic_from_name("C4_2-C5_2-C6_2-O6_2").startswith("other_")
        )

    def test_inter_residue_o6_linkage_is_glycosidic_omega(self):
        bonds = pl.DataFrame({"a": [0, 1, 2, 3], "b": [1, 2, 3, 4]})
        metadata = build_torsion_metadata(
            [(0, 1, 2, 3)],
            self._atoms(),
            bonds_df=bonds,
        )

        self.assertEqual(metadata[0]["family"], "glycosidic")
        self.assertEqual(metadata[0]["subtype"], "omega")
        self.assertEqual(metadata[0]["canonical_name"], "3_2_omega")

    def test_o6_sulfate_substitution_retains_omega(self):
        atoms = self._atoms().with_columns(
            pl.Series("name", ["C4", "C5", "C6", "O6", "S1"]),
            pl.Series("res_name", ["UYA", "UYA", "UYA", "UYA", "SO3"]),
        )
        bonds = pl.DataFrame({"a": [0, 1, 2, 3], "b": [1, 2, 3, 4]})
        metadata = build_torsion_metadata(
            [(0, 1, 2, 3)],
            atoms,
            bonds_df=bonds,
        )

        self.assertEqual(metadata[0]["family"], "glycosidic")
        self.assertEqual(metadata[0]["subtype"], "omega")
        self.assertEqual(metadata[0]["canonical_name"], "3_2_omega")

    @staticmethod
    def _neu5_atoms():
        return pl.DataFrame(
            {
                "name": ["C6", "C7", "C8", "O8", "O6", "C2", "C1"],
                "element": ["C", "C", "C", "O", "O", "C", "C"],
                "res_id": [2, 2, 2, 2, 3, 3, 3],
                "res_name": ["8SA", "8SA", "8SA", "8SA", "0SA", "0SA", "0SA"],
            }
        )

    def test_residue_local_neu5_c8_torsion_is_not_glycosidic_omega(self):
        bonds = pl.DataFrame({"a": [0, 1, 2], "b": [1, 2, 3]})
        metadata = build_torsion_metadata(
            [(0, 1, 2, 3)],
            self._neu5_atoms(),
            bonds_df=bonds,
        )

        self.assertEqual(metadata[0]["family"], "residue_local")
        self.assertNotEqual(metadata[0]["subtype"], "omega")

    def test_neu5_alpha2_8_linkage_has_glycosidic_omega(self):
        bonds = pl.DataFrame(
            {
                "a": [0, 1, 2, 3, 4, 5],
                "b": [1, 2, 3, 5, 5, 6],
            }
        )
        torsions = get_glycan_torsions_enhanced(self._neu5_atoms(), bonds)
        self.assertIn((0, 1, 2, 3), torsions)

        metadata = build_torsion_metadata(
            torsions,
            self._neu5_atoms(),
            bonds_df=bonds,
        )
        omega = [entry for entry in metadata if entry["subtype"] == "omega"]
        self.assertEqual(len(omega), 1)
        self.assertEqual(omega[0]["family"], "glycosidic")
        self.assertEqual(omega[0]["canonical_name"], "3_2_omega")

    def test_neu5_alpha2_8_linkage_has_phi_and_psi(self):
        bonds = pl.DataFrame(
            {
                "a": [0, 1, 2, 3, 4, 5],
                "b": [1, 2, 3, 5, 5, 6],
            }
        )
        torsions = get_glycan_torsions_enhanced(self._neu5_atoms(), bonds)
        self.assertIn((4, 5, 3, 2), torsions)
        self.assertIn((5, 3, 2, 1), torsions)

        metadata = build_torsion_metadata(
            torsions,
            self._neu5_atoms(),
            bonds_df=bonds,
        )
        glycosidic = {
            entry["subtype"]: entry["canonical_name"]
            for entry in metadata
            if entry["family"] == "glycosidic"
        }
        self.assertEqual(glycosidic["phi"], "3_2_phi")
        self.assertEqual(glycosidic["psi"], "3_2_psi")
        self.assertEqual(glycosidic["omega"], "3_2_omega")

    def test_neu5_phi_psi_name_classification(self):
        self.assertEqual(
            classify_glycosidic_from_name("O6_3-C2_3-O3_2-C3_2"),
            "3_2_phi",
        )
        self.assertEqual(
            classify_glycosidic_from_name("C2_3-O3_2-C3_2-C4_2"),
            "3_2_psi",
        )

    def test_neu5_alpha2_6_linkage_has_phi_and_psi(self):
        atoms = pl.DataFrame(
            {
                "name": ["C5", "C6", "O6", "O6", "C2", "C1"],
                "element": ["C", "C", "O", "O", "C", "C"],
                "res_id": [2, 2, 2, 3, 3, 3],
                "res_name": ["6LB", "6LB", "6LB", "0SA", "0SA", "0SA"],
            }
        )
        bonds = pl.DataFrame(
            {
                "a": [0, 1, 2, 3, 4],
                "b": [1, 2, 4, 4, 5],
            }
        )
        torsions = get_glycan_torsions_enhanced(atoms, bonds)
        self.assertIn((3, 4, 2, 1), torsions)
        self.assertIn((4, 2, 1, 0), torsions)

        metadata = build_torsion_metadata(torsions, atoms, bonds_df=bonds)
        neu5 = {
            entry["subtype"]: entry["canonical_name"]
            for entry in metadata
            if entry["family"] == "glycosidic"
        }
        self.assertEqual(neu5["phi"], "3_2_phi")
        self.assertEqual(neu5["psi"], "3_2_psi")

    def test_distribution_plot_never_falls_back_to_local_torsions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            torsions_path = temp_path / "torsions.parquet"
            output_path = temp_path / "dist.svg"
            pd.DataFrame(
                {
                    "frame": [0, 1],
                    "C4_2-C5_2-C6_2-O6_2": [60.0, -60.0],
                }
            ).to_parquet(torsions_path)

            plot_torsion_distribution(str(torsions_path), str(output_path))

            svg = output_path.read_text(encoding="utf-8")
            self.assertIn("No glycosidic torsions", svg)
            self.assertNotIn("omega", svg.lower())


if __name__ == "__main__":
    unittest.main()
