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
from test_torsion_definitions import linkage_torsions, load


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
        # Fuc(a1-6)GlcNAc: O6 of GlcNAc bonded to the anomeric carbon of a sugar
        found, *_ = linkage_torsions("fuc_a1_6_glcnac")
        self.assertEqual(found["3_2_omega"], ("glycosidic", ["O5_2", "C5_2", "C6_2", "O6_2"]))

    def test_o6_sulfate_omega_is_a_substituent_torsion(self):
        # Gal6S(b1-4)Glc: the sulfate (residue 4) on O6 of Gal is not a sugar.  Its omega
        # is kept, under the same name, as a substituent torsion apart from the
        # glycosidic ones (it was listed among them).
        found, metadata, atoms, xyz = linkage_torsions("gal6s_b1_4_glc")
        entry = next(e for e in metadata if e["canonical_name"] == "4_3_omega")
        self.assertEqual(entry["family"], "substituent")
        self.assertEqual(entry["subtype"], "omega")
        self.assertNotIn("glycosidic_name", entry)
        self.assertEqual(entry["residue_names"], ["6LB"] * 4)
        glycosidic = [e["canonical_name"] for e in metadata if e["family"] == "glycosidic"]
        self.assertEqual(sorted(glycosidic), ["3_2_phi", "3_2_psi"])
        # the column is still computed: the torsion is in the list the values are taken from
        atoms_df, bonds_df, _ = load("gal6s_b1_4_glc")
        self.assertIn(tuple(entry["atom_indices"]), get_glycan_torsions_enhanced(atoms_df, bonds_df))
        # the free C6 of Glc stays residue-local
        local = [e for e in metadata if e["family"] == "residue_local" and e["central_bond_atoms"] == ["C5", "C6"]]
        self.assertEqual([e["residue_ids"][0] for e in local], [2])

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

    def test_neu5_alpha2_8_linkage_has_phi_psi_and_omega(self):
        found, *_ = linkage_torsions("neu5ac_a2_8")
        self.assertEqual(found["3_2_phi"], ("glycosidic", ["O6_3", "C2_3", "O8_2", "C8_2"]))
        self.assertEqual(found["3_2_psi"], ("glycosidic", ["C2_3", "O8_2", "C8_2", "C7_2"]))
        self.assertEqual(found["3_2_omega"], ("glycosidic", ["O7_2", "C7_2", "C8_2", "O8_2"]))

    def test_neu5_phi_psi_name_classification(self):
        self.assertEqual(
            classify_glycosidic_from_name("O6_3-C2_3-O3_2-C3_2"),
            "3_2_phi",
        )
        self.assertEqual(
            classify_glycosidic_from_name("C2_3-O3_2-C3_2-C2_2"),
            "3_2_psi",
        )

    def test_neu5_alpha2_6_linkage_has_phi_and_psi(self):
        found, *_ = linkage_torsions("neu5ac_a2_6_gal")
        self.assertEqual(found["4_3_phi"], ("glycosidic", ["O6_4", "C2_4", "O6_3", "C6_3"]))
        self.assertEqual(found["4_3_psi"], ("glycosidic", ["C2_4", "O6_3", "C6_3", "C5_3"]))

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
