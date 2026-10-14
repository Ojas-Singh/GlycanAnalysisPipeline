"""Ring templates of GLYCAM residues, used to check a recovered topology.

A GLYCAM residue name has three characters: the positions that carry
linkages (``0`` for none, ``4`` for O4, ``V`` for O4 and O6, ...), the
monosaccharide (upper case D, lower case L) and the anomer and ring size
(``A``/``B``: alpha/beta pyranose, ``D``/``U``: alpha/beta furanose).  The
ulosonic acids have a two-letter monosaccharide code instead (``SA``/``SB``
Neu5Ac, ``GL`` Neu5Gc, ``KN`` Kdn, ``KO`` Kdo).

The template of a residue is its ring, in order around the ring, by atom
name.  Only the name decides it; the atoms of the topology are then checked
against it (``lib.frame_quality``).  Residues that the rules below do not
cover (substituents such as SO3, MEX, ACX, the aglycone ROH, and residue
names outside these rules) have no template: they are not checked.
"""
from __future__ import annotations

from typing import Optional, Tuple

ALDOPYRANOSE: Tuple[str, ...] = ("C1", "C2", "C3", "C4", "C5", "O5")
ALDOFURANOSE: Tuple[str, ...] = ("C1", "C2", "C3", "C4", "O4")
#: 2-ulopyranoses: sialic acids, Kdn, Kdo, and the ketohexoses as pyranoses.
KETOPYRANOSE: Tuple[str, ...] = ("C2", "C3", "C4", "C5", "C6", "O6")
#: 2-ulofuranoses: Fruf, Psif.
KETOFURANOSE: Tuple[str, ...] = ("C2", "C3", "C4", "C5", "O5")

#: GLYCAM one-letter codes of the 2-ketoses (Fru, Psi, Sor, Tag) and of Neu5Ac.
_KETOSE_LETTERS = frozenset("CPBJS")
#: Two-letter codes of the ulosonic acids whose third character is not an anomer.
_ULOSONIC_CODES = frozenset({"SA", "SB", "GL", "KN", "KO"})

__all__ = [
    "ALDOPYRANOSE",
    "ALDOFURANOSE",
    "KETOPYRANOSE",
    "KETOFURANOSE",
    "ring_template",
]


def ring_template(residue_name: object) -> Optional[Tuple[str, ...]]:
    """The ring atoms of a GLYCAM residue in ring order, or None when the name has no template."""
    code = str(residue_name or "").strip()
    if len(code) != 3:
        return None
    if code[1:] in _ULOSONIC_CODES:
        return KETOPYRANOSE
    sugar, form = code[1], code[2]
    ketose = sugar.upper() in _KETOSE_LETTERS
    # "S" as the third character: the sulfated GlcNAc residues (UYS, 4YS, 0YS,
    # ...), pyranoses like the others.
    if form in "ABS":
        return KETOPYRANOSE if ketose else ALDOPYRANOSE
    if form in "DU":
        return KETOFURANOSE if ketose else ALDOFURANOSE
    return None
