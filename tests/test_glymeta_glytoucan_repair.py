import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import main as pipeline_main
import requests
from lib import name_utils
from lib import storage as storage_module
from lib.glybake import GlycoShapeBaker
from lib.glymeta import GlycanMetadataProcessor
from lib.pocketbase import PocketBaseClient
from lib.storage import LocalStorageBackend, StorageManager


class GlyTouCanRepairTests(unittest.TestCase):
    def setUp(self):
        self._previous_storage_manager = storage_module._storage_manager
        storage_module._storage_manager = StorageManager(LocalStorageBackend())

    def tearDown(self):
        storage_module._storage_manager = self._previous_storage_manager

    def test_wurcs_lookup_uses_normalizing_converter_accession(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = [{"id": "G77945NE"}]

        with (
            patch("lib.name_utils.requests.post", return_value=response) as post,
            patch("lib.name_utils.requests.get") as get,
        ):
            result = name_utils.wurcs2glytoucan(
                "WURCS=2.0/1,1,0/[noncanonical]/1/"
            )

        self.assertEqual(result, "G77945NE")
        self.assertEqual(
            post.call_args.kwargs["json"],
            ["WURCS=2.0/1,1,0/[noncanonical]/1/"],
        )
        self.assertEqual(post.call_args.kwargs["timeout"], 30)
        get.assert_not_called()

    def test_wurcs_lookup_falls_back_to_exact_match(self):
        canonical_response = Mock()
        canonical_response.raise_for_status.return_value = None
        canonical_response.json.return_value = [{"WURCS": "WURCS=2.0/example"}]
        exact_response = Mock()
        exact_response.raise_for_status.return_value = None
        exact_response.json.return_value = [{"id": "G40990LB"}]

        with (
            patch(
                "lib.name_utils.requests.post",
                return_value=canonical_response,
            ),
            patch(
                "lib.name_utils.requests.get",
                return_value=exact_response,
            ) as get,
        ):
            result = name_utils.wurcs2glytoucan("WURCS=2.0/example")

        self.assertEqual(result, "G40990LB")
        self.assertEqual(
            get.call_args.kwargs["params"],
            {"wurcs": "WURCS=2.0/example"},
        )
        self.assertEqual(get.call_args.kwargs["timeout"], 30)

    def test_bulk_lookup_deduplicates_and_maps_converter_results(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = [
            {"id": "G11111AA"},
            {"message": "Accession not found."},
        ]

        sparql = Mock()
        sparql.raise_for_status.return_value = None
        sparql.json.return_value = {"results": {"bindings": []}}

        def answer(url, **kwargs):
            return sparql if url == name_utils.GLYTOUCAN_SPARQL_ENDPOINT else response

        with (
            patch("lib.name_utils.requests.post", side_effect=answer) as post,
            patch.object(name_utils, "crabwurcs_normalize", side_effect=lambda wurcs, timeout=30: wurcs),
        ):
            result = name_utils.wurcs2glytoucan_many(
                ["WURCS=2.0/one", "WURCS=2.0/two", "WURCS=2.0/one"]
            )

        self.assertEqual(
            result,
            {
                "WURCS=2.0/one": "G11111AA",
                "WURCS=2.0/two": None,
            },
        )
        self.assertEqual(
            post.call_args_list[0].kwargs["json"],
            ["WURCS=2.0/one", "WURCS=2.0/two"],
        )
        # What GlyCosmos has no accession for is asked of GlyTouCan (it may not have synchronised yet).
        self.assertEqual(post.call_args_list[1].args[0], name_utils.GLYTOUCAN_SPARQL_ENDPOINT)
        self.assertIn('"WURCS=2.0/two"', post.call_args_list[1].kwargs["data"]["query"])
        self.assertNotIn('"WURCS=2.0/one"', post.call_args_list[1].kwargs["data"]["query"])

    def test_missing_report_rechecks_and_deduplicates_wurcs(self):
        wurcs = "WURCS=2.0/1,1,0/[a2122h-1b_1-5]/1/"
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "GLYCOSHAPE.json").write_text(
                json.dumps({
                    "GS00001": {
                        "archetype": {"wurcs": wurcs, "glytoucan": ""},
                        "alpha": {"wurcs": wurcs, "glytoucan": None},
                    }
                }),
                encoding="utf-8",
            )

            with patch(
                "lib.glybake.name_utils.wurcs2glytoucan_many",
                return_value={wurcs: "G12345AB"},
            ) as lookup:
                GlycoShapeBaker(root).extract_missing_glytoucan_ids()

            self.assertEqual(
                (root / "missing_glytoucan.txt").read_text(encoding="utf-8"),
                "",
            )
            lookup.assert_called_once_with([wurcs])

    def test_update_glycan_metadata_repairs_missing_variant_glytoucan_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "GS00001" / "data.json"
            data_path.parent.mkdir()
            data_path.write_text(
                json.dumps(
                    {
                        "archetype": {
                            "ID": "GS00001",
                            "name": "Example",
                            "iupac": "Gal(b1-4)Glc",
                            "glytoucan": "GEXISTING",
                            "wurcs": "WURCS_EXISTING",
                        },
                        "alpha": {
                            "ID": "GS00001",
                            "name": "Example",
                            "glytoucan": "",
                            "wurcs": "WURCS_ALPHA",
                        },
                        "beta": {
                            "ID": "GS00001",
                            "name": "Example",
                            "glytoucan": "GKEEP",
                            "wurcs": "WURCS_BETA",
                        },
                        "search_meta": {"common_names": [], "keywords": []},
                    }
                ),
                encoding="utf-8",
            )

            calls = []

            def fake_lookup(wurcs):
                calls.append(wurcs)
                return {"WURCS_ALPHA": "G40990LB"}[wurcs]

            processor = GlycanMetadataProcessor(tmpdir)
            with patch("lib.glymeta.name_utils.wurcs2glytoucan", side_effect=fake_lookup):
                self.assertTrue(processor.update_glycan_metadata(data_path))

            updated = json.loads(data_path.read_text(encoding="utf-8"))
            self.assertEqual(updated["alpha"]["glytoucan"], "G40990LB")
            self.assertEqual(updated["beta"]["glytoucan"], "GKEEP")
            self.assertEqual(calls, ["WURCS_ALPHA"])
            self.assertIn("GA2", updated["search_meta"]["common_names"])

    def test_update_glycan_metadata_leaves_unresolved_wurcs_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "GS00002" / "data.json"
            data_path.parent.mkdir()
            data_path.write_text(
                json.dumps(
                    {
                        "archetype": {
                            "ID": "GS00002",
                            "name": "Example",
                            "iupac": "Gal",
                            "glytoucan": None,
                            "wurcs": "WURCS_UNRESOLVED",
                        },
                        "search_meta": {},
                    }
                ),
                encoding="utf-8",
            )

            processor = GlycanMetadataProcessor(tmpdir)
            with patch("lib.glymeta.name_utils.wurcs2glytoucan", return_value=None):
                self.assertTrue(processor.update_glycan_metadata(data_path))

            updated = json.loads(data_path.read_text(encoding="utf-8"))
            self.assertIsNone(updated["archetype"]["glytoucan"])

    def test_baker_missing_glytoucan_output_reflects_repaired_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gs_dir = Path(tmpdir) / "GS00003"
            gs_dir.mkdir()
            (gs_dir / "data.json").write_text(
                json.dumps(
                    {
                        "archetype": {
                            "ID": "GS00003",
                            "name": "Example",
                            "glytoucan": "GRESOLVED",
                            "wurcs": "WURCS_RESOLVED",
                        },
                        "alpha": {
                            "ID": "GS00003",
                            "name": "Example",
                            "glytoucan": "",
                            "wurcs": "WURCS_STILL_MISSING",
                        },
                    }
                ),
                encoding="utf-8",
            )

            baker = GlycoShapeBaker(tmpdir)
            baker.create_consolidated_json()
            baker.extract_missing_glytoucan_ids()

            missing = (Path(tmpdir) / "missing_glytoucan.txt").read_text(encoding="utf-8")
            self.assertNotIn("WURCS_RESOLVED", missing)
            self.assertIn("WURCS_STILL_MISSING", missing)

    def test_refresh_static_finalization_sync_preserves_repaired_local_json(self):
        class FakeBackend:
            def __init__(self, remote_data):
                self.remote_data = remote_data

            def list_common_prefixes(self, base):
                return [f"{base}/GS00004"]

            def list_files(self, remote_key, pattern="*"):
                return [f"{remote_key}/data.json"]

            def read_binary(self, key):
                return self.remote_data[key]

        class FakeStorage:
            def __init__(self, backend):
                self.backend = backend

            def list_dirs(self, base):
                return [Path(f"{base}/GS00004")]

            def open(self, path, mode="r"):
                return open(path, mode, encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "static"
            local_dir = output_dir / "GS00004"
            local_dir.mkdir(parents=True)
            repaired = {
                "archetype": {
                    "ID": "GS00004",
                    "name": "Example",
                    "glytoucan": "GREPAIRED",
                    "wurcs": "WURCS_REPAIRED",
                }
            }
            stale = {
                "archetype": {
                    "ID": "GS00004",
                    "name": "Example",
                    "glytoucan": None,
                    "wurcs": "WURCS_REPAIRED",
                }
            }
            (local_dir / "data.json").write_text(json.dumps(repaired), encoding="utf-8")

            runner = pipeline_main.GlycanPipelineRunner.__new__(pipeline_main.GlycanPipelineRunner)
            runner.output_dir = output_dir
            runner.mode = "refresh-static"
            runner._known_static_output_dirs = {}
            runner.storage = FakeStorage(
                FakeBackend({"static/GS00004/data.json": json.dumps(stale).encode("utf-8")})
            )

            with patch.object(pipeline_main.config, "use_oracle_storage", True), patch.object(
                pipeline_main.config, "oracle_output_prefix", "static"
            ):
                runner._sync_all_remote_static_to_local()

            updated = json.loads((local_dir / "data.json").read_text(encoding="utf-8"))
            self.assertEqual(updated["archetype"]["glytoucan"], "GREPAIRED")

    def test_refresh_static_uploads_repaired_local_static_output_to_oracle(self):
        class FakeStorage:
            def __init__(self):
                self.uploads = []

            def upload_dir(self, local_dir, remote_prefix, skip_existing=True):
                self.uploads.append((Path(local_dir), remote_prefix, skip_existing))

        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / "static" / "GS00005"
            local_dir.mkdir(parents=True)
            (local_dir / "data.json").write_text("{}", encoding="utf-8")

            storage = FakeStorage()
            runner = pipeline_main.GlycanPipelineRunner.__new__(pipeline_main.GlycanPipelineRunner)
            runner.storage = storage

            with patch.object(pipeline_main.config, "use_oracle_storage", True), patch.object(
                pipeline_main.config, "oracle_output_prefix", "static"
            ):
                runner._upload_local_static_output_if_oracle(str(local_dir), "Example")

            self.assertEqual(storage.uploads, [(local_dir, "static/GS00005", False)])

    def test_pocketbase_get_retries_transient_timeout(self):
        class FakeResponse:
            status_code = 200

            def json(self):
                return {"ok": True}

        class FakeSession:
            def __init__(self):
                self.calls = 0

            def request(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise requests.exceptions.Timeout("slow")
                return FakeResponse()

        client = PocketBaseClient("https://example.test", "token")
        client.timeout = 0.1
        client.retries = 2
        client.retry_backoff = 0
        client.session = FakeSession()

        self.assertEqual(client.request("get", "/api/test"), {"ok": True})
        self.assertEqual(client.session.calls, 2)

    def test_pocketbase_post_timeout_is_not_retried(self):
        class FakeSession:
            def __init__(self):
                self.calls = 0

            def request(self, *args, **kwargs):
                self.calls += 1
                raise requests.exceptions.Timeout("slow")

        client = PocketBaseClient("https://example.test", "token")
        client.timeout = 0.1
        client.retries = 3
        client.retry_backoff = 0
        client.session = FakeSession()

        with self.assertRaises(requests.exceptions.Timeout):
            client.request("post", "/api/test", json={"x": 1})
        self.assertEqual(client.session.calls, 1)


if __name__ == "__main__":
    unittest.main()
