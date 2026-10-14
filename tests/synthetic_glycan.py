"""A small synthetic trajectory with a real topology, for end-to-end tests of the process stage.

The atoms, the first frame and the bonds are those of the Man(a1-2)Man example shipped in
``dummy_database_data/`` (45 atoms, read from the first model of its PDB and from its MOL2). The
trajectory is generated here: the non-reducing residue is rotated about the two bonds of the
glycosidic linkage by angles drawn around three conformational states, and every atom gets a small
Gaussian displacement. Bond lengths stay plausible, so the frame-quality filter keeps every frame
unless a frame is deliberately broken.

At most 500 frames keeps the PCA fit on scikit-learn's exact solver (the randomized one, used for larger
inputs, is seeded as well), so two runs of the pipeline give the same numbers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
GLYCAM_NAME = "DManpa1-2DManpa1-OH"
EXAMPLE = REPO_ROOT / "dummy_database_data" / GLYCAM_NAME
HEADER = (
    "HEADER    \n"
    "TITLE     SYNTHETIC TRAJECTORY FOR TESTS\n"
    "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n"
)
STATES = ((0.0, 0.0), (55.0, -70.0), (-80.0, 95.0))          # (phi, psi) offsets in degrees
POPULATIONS = (0.5, 0.3, 0.2)


def first_model_lines() -> list:
    """ATOM lines of the first model of the example PDB."""
    lines = []
    with open(EXAMPLE / f"{GLYCAM_NAME}.pdb", "r", encoding="ascii") as handle:
        for line in handle:
            if line.startswith("ATOM"):
                lines.append(line)
            elif line.startswith("ENDMDL"):
                break
    return lines


def _rotate(points: np.ndarray, origin: np.ndarray, axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    angle = np.deg2rad(degrees)
    shifted = points - origin
    return (
        shifted * np.cos(angle)
        + np.cross(axis, shifted) * np.sin(angle)
        + axis * (shifted @ axis)[:, None] * (1 - np.cos(angle))
        + origin
    )


def trajectory(frames: int = 480, seed: int = 7, broken=()) -> tuple:
    """``(atom_lines, coordinates)``; coordinates are integers of 0.001 A, shape (frames, atoms, 3)."""
    lines = first_model_lines()
    names = [line[12:16].strip() for line in lines]
    residues = np.array([int(line[22:26]) for line in lines])
    start = np.array([[float(line[30:38]), float(line[38:46]), float(line[46:54])] for line in lines])

    def atom(name, residue):
        return next(i for i, (n, r) in enumerate(zip(names, residues)) if n == name and r == residue)

    c2, o2, c1 = atom("C2", 2), atom("O2", 2), atom("C1", 3)
    moving = np.flatnonzero(residues == 3)
    rng = np.random.default_rng(seed)
    out = np.empty((frames, len(lines), 3), dtype=np.int64)
    for frame in range(frames):
        phi, psi = STATES[rng.choice(len(STATES), p=POPULATIONS)]
        xyz = start.copy()
        xyz[moving] = _rotate(xyz[moving], xyz[o2], xyz[o2] - xyz[c2], psi + rng.normal(0, 7))
        xyz[moving] = _rotate(xyz[moving], xyz[c1], xyz[c1] - xyz[o2], phi + rng.normal(0, 7))
        xyz += rng.normal(0, 0.02, size=xyz.shape)
        if frame in broken:
            xyz[atom("H6O", 3)] += (9.0, 0.0, 0.0)             # O6-H6O torn apart: a broken frame
        out[frame] = np.rint(xyz * 1000).astype(np.int64)
    return lines, out


def pdb_text(lines: list, coordinates: np.ndarray) -> str:
    parts = [HEADER]
    for index, frame in enumerate(coordinates):
        parts.append("MODEL     %4d\n" % (index + 1))
        for line, (x, y, z) in zip(lines, frame):
            parts.append("%s%8.3f%8.3f%8.3f%s" % (line[:30], x / 1000.0, y / 1000.0, z / 1000.0, line[54:]))
        parts.append("ENDMDL\n")
    parts.append("END\n")
    return "".join(parts)


def write_source(root, folder: str = GLYCAM_NAME, frames: int = 480, seed: int = 7, broken=()) -> Path:
    """Create ``root/folder`` with simulation.pdb and structure.mol2; return its path."""
    source = Path(root) / folder
    source.mkdir(parents=True)
    lines, coordinates = trajectory(frames, seed, broken)
    (source / "simulation.pdb").write_text(pdb_text(lines, coordinates), encoding="ascii")
    (source / "structure.mol2").write_bytes((EXAMPLE / f"{GLYCAM_NAME}.mol2").read_bytes())
    return source
