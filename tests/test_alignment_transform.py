#!/usr/bin/env python3
"""Unit test for the RDKit-alignment transform extraction fix in
``lib.glypdbio.align_structures_by_residues``.

Background: prior to the fix, the RDKit branch performed alignment on a
temporary RDKit molecule but never extracted or applied the transform to
the original ``target_coords``, so the function returned untransformed
coordinates. The Kabsch fallback (and a rigid synthetic transform) were
the only way to verify alignment actually worked.
"""

from __future__ import annotations

import unittest

import numpy as np
import polars as pl

from lib.glypdbio import align_structures_by_residues


def _make_synthetic_glycan(n_atoms: int = 30, n_residues: int = 3) -> tuple[pl.DataFrame, np.ndarray]:
    """Build a small synthetic glycan: ``n_residues`` residues of 10 atoms each."""
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(n_atoms, 3)).astype(np.float32) * 5.0
    rows = []
    atoms_per_res = n_atoms // n_residues
    for i in range(n_atoms):
        rid = (i // atoms_per_res) + 1
        rows.append(
            {
                "serial": i + 1,
                "name": f"A{i % atoms_per_res + 1}",
                "res_name": "ROH" if i < atoms_per_res else "GLC",
                "res_id": int(rid),
                "element": "C" if i % 3 != 0 else "O",
            }
        )
    return pl.DataFrame(rows), xyz


class AlignmentTransformTest(unittest.TestCase):
    def test_rigid_rotation_recovered_exactly(self) -> None:
        atoms, ref = _make_synthetic_glycan()
        rng = np.random.default_rng(42)
        theta = rng.uniform(-np.pi, np.pi)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        t = rng.normal(size=3) * 10.0
        tgt = ref @ R.T + t

        aligned, matrix = align_structures_by_residues(ref, tgt, atoms, atoms, (1, 3), use_rdkit=True)
        # det(R) must be 1
        self.assertAlmostEqual(float(np.linalg.det(matrix[:3, :3])), 1.0, places=5)
        # Rigid transform must recover the original target exactly
        diff = aligned - ref
        self.assertLess(float(np.max(np.abs(diff))), 1e-4)
        # The recovered transform maps ref -> ref under the alignment cost
        # (recovered translation may differ from the applied t when the
        # alignment subset is a strict subset of the molecule; we don't
        # require exact equality here).

    def test_rigid_rotation_recovered_via_kabsch_path(self) -> None:
        atoms, ref = _make_synthetic_glycan()
        rng = np.random.default_rng(7)
        theta = 1.2
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        t = np.array([5.0, -3.0, 2.0])
        tgt = ref @ R.T + t

        aligned, matrix = align_structures_by_residues(ref, tgt, atoms, atoms, (1, 3), use_rdkit=False)
        diff = aligned - ref
        self.assertLess(float(np.max(np.abs(diff))), 1e-4)
        self.assertAlmostEqual(float(np.linalg.det(matrix[:3, :3])), 1.0, places=5)

    def test_alignment_modifies_target(self) -> None:
        """Regression test for the original bug: alignment must change the
        coordinates (not return them unchanged)."""
        atoms, ref = _make_synthetic_glycan()
        rng = np.random.default_rng(99)
        R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        t = rng.normal(size=3) * 7.0
        tgt = ref @ R.T + t

        aligned, _ = align_structures_by_residues(ref, tgt, atoms, atoms, (1, 3), use_rdkit=True)
        # The aligned target must differ from the raw target -- the bug
        # returned target.copy() so the two arrays were identical.
        self.assertGreater(float(np.max(np.abs(aligned - tgt))), 1.0)

    def test_rdkit_and_kabsch_agree(self) -> None:
        atoms, ref = _make_synthetic_glycan()
        rng = np.random.default_rng(13)
        R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        t = rng.normal(size=3) * 4.0
        tgt = ref @ R.T + t

        a_rdkit, _ = align_structures_by_residues(ref, tgt, atoms, atoms, (1, 3), use_rdkit=True)
        a_kab, _ = align_structures_by_residues(ref, tgt, atoms, atoms, (1, 3), use_rdkit=False)
        self.assertLess(float(np.max(np.abs(a_rdkit - a_kab))), 1e-3)


if __name__ == "__main__":
    unittest.main()
