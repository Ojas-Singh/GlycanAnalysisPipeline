#!/usr/bin/env python3
"""Verify alignment of multi-model PDBs in static/<GS>/output/.

For each multi-model PDB, uses MODEL 1's heavy-atom subset of the first
``--core-residues`` (default 5) residues as the alignment reference and
computes per-model RMSD on:
  * the alignment core (residues 1..N) -- should be ~0 after alignment
  * the remaining residues (branches) -- should remain non-zero (flexible)

Prints a pass/fail summary per (level, format, anomer).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _parse_pdb_atoms(text: str) -> list[dict]:
    """Return list of atom dicts: serial, name, res_name, chain_id, res_id, x, y, z, element."""
    atoms: list[dict] = []
    for line in text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        try:
            serial = int(line[6:11])
            name = line[12:16].strip()
            alt = line[16:17]
            if alt not in (" ", "", "A"):
                continue
            res_name = line[17:20].strip()
            chain_id = line[21:22].strip() or "X"
            res_id = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            element = line[76:78].strip() if len(line) >= 78 else name[0]
            atoms.append(
                {
                    "serial": serial,
                    "name": name,
                    "res_name": res_name,
                    "chain_id": chain_id,
                    "res_id": res_id,
                    "x": x,
                    "y": y,
                    "z": z,
                    "element": element,
                }
            )
        except Exception:
            continue
    return atoms


def _split_models(atoms: list[dict]) -> list[list[dict]]:
    """Split atoms into per-model lists based on serial resets."""
    if not atoms:
        return []
    models: list[list[dict]] = []
    current: list[dict] = []
    last_serial = 0
    for a in atoms:
        if a["serial"] <= last_serial and current:
            models.append(current)
            current = []
        current.append(a)
        last_serial = a["serial"]
    if current:
        models.append(current)
    return models


def _heavy_mask(atoms: list[dict]) -> np.ndarray:
    return np.array(
        [
            not str(a["element"]).strip().upper().startswith("H")
            for a in atoms
        ],
        dtype=bool,
    )


def _residue_mask(atoms: list[dict], lo: int, hi: int) -> np.ndarray:
    return np.array(
        [lo <= int(a["res_id"]) <= hi for a in atoms], dtype=bool
    )


def _model_xyz(atoms: list[dict]) -> np.ndarray:
    return np.asarray([[a["x"], a["y"], a["z"]] for a in atoms], dtype=float)


@dataclass
class FileReport:
    path: Path
    n_models: int
    n_atoms_per_model: int
    core_rmsds: list[float]
    branch_rmsds: list[float]

    @property
    def max_core_rmsd(self) -> float:
        return max(self.core_rmsds) if self.core_rmsds else float("nan")

    @property
    def median_branch_rmsd(self) -> float:
        return float(np.median(self.branch_rmsds)) if self.branch_rmsds else float("nan")


def _verify_file(path: Path, core_residues: int, core_tol: float) -> FileReport:
    text = path.read_text()
    atoms = _parse_pdb_atoms(text)
    models = _split_models(atoms)
    if len(models) < 2:
        return FileReport(path, len(models), len(models[0]) if models else 0, [], [])
    ref_atoms = models[0]
    ref_xyz = _model_xyz(ref_atoms)
    ref_heavy = _heavy_mask(ref_atoms)
    ref_core = _residue_mask(ref_atoms, 1, core_residues)
    ref_core_idx = np.where(ref_heavy & ref_core)[0]
    ref_branch_idx = np.where(ref_heavy & ~ref_core)[0]
    core_rmsds: list[float] = []
    branch_rmsds: list[float] = []
    n_atoms = len(ref_atoms)
    for m in models[1:]:
        if len(m) != n_atoms:
            core_rmsds.append(float("nan"))
            branch_rmsds.append(float("nan"))
            continue
        xyz = _model_xyz(m)
        heavy = _heavy_mask(m)
        core = _residue_mask(m, 1, core_residues)
        ci = np.where(heavy & core)[0]
        bi = np.where(heavy & ~core)[0]
        if ci.size:
            d = np.linalg.norm(xyz[ci] - ref_xyz[ci], axis=1)
            core_rmsds.append(float(np.sqrt(np.mean(d * d))))
        else:
            core_rmsds.append(float("nan"))
        if bi.size:
            d = np.linalg.norm(xyz[bi] - ref_xyz[bi], axis=1)
            branch_rmsds.append(float(np.sqrt(np.mean(d * d))))
        else:
            branch_rmsds.append(float("nan"))
    return FileReport(path, len(models), n_atoms, core_rmsds, branch_rmsds)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--gs-dir",
        required=True,
        help="Path to the static/<GS_ID>/ folder to verify (e.g. static/GS00141)",
    )
    p.add_argument("--core-residues", type=int, default=5, help="Number of N-terminal residues used for alignment (default 5)")
    p.add_argument("--core-tol", type=float, default=0.5, help="Pass threshold (Å) for max core RMSD per file (default 0.5)")
    p.add_argument("--branch-min", type=float, default=0.1, help="Pass threshold (Å) for median branch RMSD (default 0.1)")
    args = p.parse_args()

    gs_dir = Path(args.gs_dir).resolve()
    if not gs_dir.is_dir():
        print(f"ERROR: {gs_dir} is not a directory", file=sys.stderr)
        return 2
    files = sorted(gs_dir.glob("output/level_*/PDB/alpha.pdb")) + sorted(
        gs_dir.glob("output/level_*/PDB/beta.pdb")
    ) + sorted(gs_dir.glob("output/level_*/GLYCAM/alpha.pdb")) + sorted(
        gs_dir.glob("output/level_*/GLYCAM/beta.pdb")
    )
    if not files:
        print(f"ERROR: no multi-model PDBs found under {gs_dir}/output/", file=sys.stderr)
        return 2

    print(
        f"Verifying {len(files)} multi-model PDBs under {gs_dir} "
        f"(core residues 1..{args.core_residues}, core_tol={args.core_tol} Å, branch_min={args.branch_min} Å)\n"
    )
    print(
        f"{'File':75s} {'#mdl':>5s} {'natoms':>7s}  {'max core (Å)':>13s}  {'med branch (Å)':>15s}  {'status':>8s}"
    )
    print("-" * 130)
    all_pass = True
    n_pass = 0
    for f in files:
        rep = _verify_file(f, args.core_residues, args.core_tol)
        core_ok = rep.max_core_rmsd <= args.core_tol
        branch_ok = rep.median_branch_rmsd >= args.branch_min
        status = "PASS" if (core_ok and branch_ok) else "FAIL"
        if status == "FAIL":
            all_pass = False
        else:
            n_pass += 1
        rel = f.relative_to(gs_dir)
        print(
            f"{str(rel):75s} {rep.n_models:5d} {rep.n_atoms_per_model:7d}  "
            f"{rep.max_core_rmsd:13.4f}  {rep.median_branch_rmsd:15.4f}  {status:>8s}"
        )
    print("-" * 130)
    print(f"{n_pass}/{len(files)} files PASS")
    if not all_pass:
        print("FAILURES:")
        for f in files:
            rep = _verify_file(f, args.core_residues, args.core_tol)
            core_ok = rep.max_core_rmsd <= args.core_tol
            branch_ok = rep.median_branch_rmsd >= args.branch_min
            if not (core_ok and branch_ok):
                rel = f.relative_to(gs_dir)
                reasons = []
                if not core_ok:
                    reasons.append(
                        f"core RMSD {rep.max_core_rmsd:.4f} > {args.core_tol}"
                    )
                if not branch_ok:
                    reasons.append(
                        f"branch RMSD {rep.median_branch_rmsd:.4f} < {args.branch_min}"
                    )
                print(f"  - {rel}: {'; '.join(reasons)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
