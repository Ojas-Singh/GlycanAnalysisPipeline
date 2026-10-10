import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lib import glystatic
from lib.pocketbase import extract_inventory_metadata
from lib.storage import LocalStorageBackend, StorageManager


class GlystaticOutputSafetyTests(unittest.TestCase):
    def test_explicit_nanoseconds_are_preserved_in_inventory_metadata(self):
        metadata = extract_inventory_metadata(
            {
                "glycoshape_id": "GS00001",
                "simulation_length": 500,
                "simulation_length_ns": 1500,
            }
        )

        self.assertEqual(metadata["length"], 500.0)
        self.assertEqual(metadata["simulation_length_ns"], 1500.0)

    def test_explicit_nanoseconds_are_converted_to_microseconds(self):
        metadata = {"length": 500.0}

        with patch.object(glystatic, "_source_simulation_length_ns", return_value=1500.0):
            length_us = glystatic._simulation_length_us(metadata, "Example", storage=None)

        self.assertEqual(
            length_us,
            1.5,
        )

    def test_legacy_microsecond_metadata_is_preserved(self):
        metadata = {"length": 2.0, "simulation_length_ns": None}

        self.assertEqual(
            glystatic._simulation_length_us(metadata, "Example", storage=None),
            2.0,
        )

    def test_manual_upload_info_exposes_simulation_length_in_nanoseconds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            source_dir = data_dir / "Example"
            source_dir.mkdir(parents=True)
            (source_dir / "info.json").write_text(
                '{"simulationLength": 1500.0}',
                encoding="utf-8",
            )
            storage = StorageManager(LocalStorageBackend())

            with patch.object(glystatic.config, "data_dir", data_dir):
                length_ns = glystatic._source_simulation_length_ns("Example", storage)

            self.assertEqual(length_ns, 1500.0)

    def test_archetype_glytoucan_is_never_copied_to_variants(self):
        archetype, alpha, beta = glystatic._apply_glytoucan_fallbacks(
            glytoucan="GARCHETYPE",
            glytoucan_alpha=None,
            glytoucan_beta="GBETA",
            inventory_glytoucan="GINVENTORY",
        )

        self.assertEqual(archetype, "GARCHETYPE")
        self.assertIsNone(alpha)
        self.assertEqual(beta, "GBETA")

    def test_inventory_glytoucan_only_fills_archetype(self):
        archetype, alpha, beta = glystatic._apply_glytoucan_fallbacks(
            glytoucan=None,
            glytoucan_alpha=None,
            glytoucan_beta=None,
            inventory_glytoucan="GINVENTORY",
        )

        self.assertEqual(archetype, "GINVENTORY")
        self.assertIsNone(alpha)
        self.assertIsNone(beta)

    def test_metadata_recovers_id_from_glycans_collection(self):
        submission = {
            "glycoshape_id": "",
            "glycam_name": "Example",
            "glytoucan_id": "",
        }
        glycan_record = {
            "glycoshape_id": "GS00957",
            "glycam_name": "Example",
            "glytoucan_id": "G78018KO",
        }

        class FakeClient:
            def is_available(self):
                return True

            def get_record_by_glycam_name(self, _name, collection="glycan_submission"):
                return glycan_record if collection == "glycans" else submission

            def get_record_by_glycoshape_id(self, _gs_id, collection="glycan_submission"):
                return None

        with (
            patch.object(glystatic.config, "use_pocketbase", True),
            patch("lib.pocketbase.get_pocketbase_client", return_value=FakeClient()),
        ):
            metadata = glystatic.get_glycan_metadata("Example")

        self.assertEqual(metadata["ID"], "GS00957")
        self.assertEqual(metadata["glytoucan_id"], "G78018KO")

    def test_missing_id_cannot_delete_static_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            process_dir = root / "process" / "example"
            (process_dir / "output").mkdir(parents=True)
            static_dir = root / "static"
            static_dir.mkdir()
            sentinel = static_dir / "keep-me.txt"
            sentinel.write_text("safe", encoding="utf-8")
            storage = StorageManager(LocalStorageBackend())

            metadata = {
                "ID": "",
                "length": "",
                "package": "",
                "forcefield": "",
                "temperature": "",
                "pressure": "",
                "salt": "",
                "comments": "",
                "glytoucan_id": "",
            }
            with (
                patch("lib.storage.get_storage_manager", return_value=storage),
                patch.object(glystatic, "load_info_json", return_value={}),
                patch.object(glystatic, "extract_cluster_data", return_value={}),
                patch.object(glystatic, "get_glycan_metadata", return_value=metadata),
            ):
                with self.assertRaisesRegex(ValueError, "Invalid or missing GlycoShape ID"):
                    glystatic.process_glycan(
                        str(process_dir),
                        "Example",
                        str(static_dir),
                    )

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "safe")
            self.assertEqual(list(static_dir.iterdir()), [sentinel])


if __name__ == "__main__":
    unittest.main()
