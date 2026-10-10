import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lib import glystatic


class SnfgRendererTests(unittest.TestCase):
    def test_crabwurcs_renders_glycam_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "snfg.svg"

            def fake_run(command, **kwargs):
                Path(command[command.index("--output") + 1]).write_text(
                    '<svg xmlns="http://www.w3.org/2000/svg"/>',
                    encoding="utf-8",
                )
                self.assertEqual(kwargs["input"], "DGalpb1-4DGlcpa1-OH")
                return subprocess.CompletedProcess(command, 0, "", "")

            with (
                patch.object(glystatic.config, "snfg_renderer", "crabwurcs"),
                patch.object(glystatic.config, "crabwurcs_executable", "/opt/bin/crabwurcs"),
                patch.object(glystatic.subprocess, "run", side_effect=fake_run) as run,
            ):
                renderer = glystatic._render_snfg_svg(
                    "DGalpb1-4DGlcpa1-OH",
                    "Gal(b1-4)Glc",
                    destination,
                )

            self.assertEqual(renderer, "crabWURCS")
            self.assertEqual(
                run.call_args.args[0],
                [
                    "/opt/bin/crabwurcs",
                    "render",
                    "--from",
                    "glycam",
                    "--output",
                    str(destination),
                ],
            )

    def test_glycodraw_remains_available(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "snfg.svg"

            def fake_glycodraw(iupac, filepath, show_linkage):
                self.assertEqual(iupac, "Gal(b1-4)Glc")
                self.assertTrue(show_linkage)
                Path(filepath).write_text("<svg/>", encoding="utf-8")

            with (
                patch.object(glystatic.config, "snfg_renderer", "glycodraw"),
                patch.object(glystatic, "GlycoDraw", side_effect=fake_glycodraw) as draw,
            ):
                renderer = glystatic._render_snfg_svg(
                    "DGalpb1-4DGlcpa1-OH",
                    "Gal(b1-4)Glc",
                    destination,
                )

            self.assertEqual(renderer, "GlycoDraw")
            draw.assert_called_once()

    def test_crabwurcs_error_includes_cli_detail(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(glystatic.config, "snfg_renderer", "crabwurcs"),
                patch.object(glystatic.config, "crabwurcs_executable", "crabwurcs"),
                patch.object(
                    glystatic.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess([], 2, "", "unsupported residue"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "unsupported residue"):
                    glystatic._render_snfg_svg(
                        "bad-glycan",
                        "bad-glycan",
                        Path(temp_dir) / "snfg.svg",
                    )


if __name__ == "__main__":
    unittest.main()
