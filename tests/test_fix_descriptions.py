"""scripts/fix_descriptions.py: what it takes out of a published description and which Oxford names it drops."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from lib.name_utils import oxford_fits

_SPEC = importlib.util.spec_from_file_location(
    "fix_descriptions", Path(__file__).resolve().parents[1] / "scripts" / "fix_descriptions.py")
fix = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fix)

HEAD = "GS00113 is a glycan with IUPAC notation Neu5Ac(a2-3)Gal(b1-4)GlcNAc. Classified as: O-Glycan. "


def test_the_comment_after_the_generated_description_goes():
    tail = "Composed of 1x Gal, 1x GlcNAc, 1x Neu5Ac."
    assert fix.repaired(HEAD + "Simulated with MD package GROMACS, force field GLYCAM_06j, 300.0K. " + tail + " Fermanagh",
                        "GS00113") == HEAD + "Simulated with MD package GROMACS, force field GLYCAM_06j, 300.0K. " + tail
    assert fix.repaired(HEAD + tail + " http://glycoshape.healoor.me:8080/  GOTW script used. Not for running.", "GS00113") \
        == HEAD + tail


def test_a_zero_temperature_goes():
    text = HEAD + "Simulated with MD package GROMACS, force field CHARMM36m, 0.0K. Composed of 2x Man."
    assert fix.repaired(text, "GS00113") == HEAD + "Simulated with MD package GROMACS, force field CHARMM36m. Composed of 2x Man."
    assert fix.repaired(HEAD + "Simulated with 0.0K. Composed of 2x Man.", "GS00113") == HEAD + "Composed of 2x Man."


def test_a_clean_or_curated_description_stays():
    assert fix.repaired(HEAD + "Composed of 2x Man.", "GS00113") is None
    assert fix.repaired("Sialyl Lewis X, curated. Donegal", "GS00113") is None


def test_oxford_names_that_name_a_missing_galnac_or_sulfate_are_refused():
    a1 = "GlcNAc(b1-2)Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc"
    assert not oxford_fits("A1GalNAc1", a1)
    assert not oxford_fits("A1GalNAc1[SO4-2]1", a1)
    assert oxford_fits("A1", a1)
    assert oxford_fits("A1GalNAc1[SO4-2]1", "GalNAc4S(b1-4)" + a1)
    assert not oxford_fits("A2GalNAc2[SO4-2]2", "GalNAc(b1-4)" + a1)
