import subprocess
import unittest
from unittest.mock import Mock, patch

from lib import name_utils


class CrabwurcsConversionTests(unittest.TestCase):
    def test_crabwurcs_convert_uses_stdin(self):
        completed = subprocess.CompletedProcess(
            [],
            0,
            "α-D-Galp-(1→4)-D-Glcp\n",
            "",
        )
        with (
            patch.object(name_utils.config, "crabwurcs_executable", "/opt/bin/crabwurcs"),
            patch.object(name_utils.subprocess, "run", return_value=completed) as run,
        ):
            converted = name_utils.crabwurcs_convert(
                "WURCS=2.0/example",
                "wurcs",
                "iupac-extended",
            )

        self.assertEqual(converted, "α-D-Galp-(1→4)-D-Glcp")
        self.assertEqual(
            run.call_args.args[0],
            [
                "/opt/bin/crabwurcs",
                "convert",
                "--from",
                "wurcs",
                "--to",
                "iupac-extended",
            ],
        )
        self.assertEqual(run.call_args.kwargs["input"], "WURCS=2.0/example")

    def test_extended_iupac_prefers_local_crabwurcs(self):
        with (
            patch.object(
                name_utils,
                "crabwurcs_convert",
                return_value="α-D-Galp-(1→4)-D-Glcp",
            ) as convert,
            patch.object(name_utils.requests, "post") as post,
        ):
            result = name_utils.wurcs2extendediupac("WURCS=2.0/example")

        self.assertEqual(result, "α-D-Galp-(1→4)-D-Glcp")
        convert.assert_called_once_with(
            "WURCS=2.0/example",
            "wurcs",
            "iupac-extended",
        )
        post.assert_not_called()

    def test_iupac_to_wurcs_prefers_local_crabwurcs(self):
        with (
            patch.object(
                name_utils,
                "crabwurcs_convert",
                return_value="WURCS=2.0/local",
            ) as convert,
            patch.object(
                name_utils,
                "wurcs2glytoucan",
                return_value="G00000AA",
            ) as lookup,
            patch.object(name_utils.requests, "get") as get,
        ):
            glytoucan, wurcs = name_utils.iupac2wurcs_glytoucan("GlcNAc")

        self.assertEqual(glytoucan, "G00000AA")
        self.assertEqual(wurcs, "WURCS=2.0/local")
        convert.assert_called_once_with("GlcNAc", "iupac-condensed", "wurcs")
        lookup.assert_called_once_with("WURCS=2.0/local")
        get.assert_not_called()

    def test_discontinued_glygen_record_is_preserved_without_error(self):
        payload = {
            "error_list": [{"error_code": "non-existent-record"}],
            "reason": {
                "type": "discontinued_in_glygen",
                "description": "GlyTouCan accession discontinued in GlyGen",
            },
        }
        response = Mock(status_code=500)
        response.json.return_value = payload

        with patch.object(name_utils.requests, "post", return_value=response):
            result = name_utils.glytoucan2glygen("G99460PJ")

        self.assertEqual(result, payload)
        response.raise_for_status.assert_not_called()


if __name__ == "__main__":
    unittest.main()
