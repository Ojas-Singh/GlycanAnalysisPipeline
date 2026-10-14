"""Torsion (dihedral) utilities for glycan / small-molecule frames.

This module centralizes torsion extraction and manipulation logic that was
previously embedded in an exploratory notebook. Functions are designed to
work directly with RDKit molecules produced e.g. via
``glypdbio.get_mol_from_frame``.

Public functions:
	- get_movable_torsions(mol): list[(i,j,k,l)] of dihedral atom indices.
	- get_torsion_values(mol, torsion_list, conf_id=0): list[float] in degrees.
	- set_torsion_values(mol, torsion_list, torsion_values, template_conf_id=0): add new conformer.
	- generate_wiggles(mol, torsion_list, degrees=5.0, per_torsion=2): small torsion perturbations.

Utilities intentionally keep dependencies light (only RDKit + numpy). Plotting
or network visualization code remains in the notebook / higher-level modules.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import os
import re

import numpy as np
from lib.storage import get_storage_manager

try:  # Optional dependency import block
	from rdkit import Chem  # type: ignore
	from rdkit.Chem import rdMolTransforms  # type: ignore
	import networkx as nx  # type: ignore
except Exception as _exc:  # pragma: no cover - handled lazily
	Chem = None  # type: ignore
	rdMolTransforms = None  # type: ignore
	nx = None  # type: ignore

Torsion = Tuple[int, int, int, int]

#: Version of the torsion definitions below.  1: psi with C(x-1) or C(x+1) as
#: the bond table had it, omega C4-C5-C6-O6, the omega of 6-O-sulfates and the
#: like listed as glycosidic.  2: the IUPAC definitions for every linkage.
TORSION_DEFINITIONS_VERSION = 2

#: How the stored torsions are defined.  It is written into the torsion block
#: of ``info.json`` (``torsions.conventions``); the four atoms of each stored
#: torsion are in its metadata (``atom_names``, ``residue_ids``).
#:
#: These are the definitions of the IUPAC-IUB recommendations (Joint
#: Commission on Biochemical Nomenclature, "Symbols for specifying the
#: conformation of polysaccharide chains", 1981; Eur. J. Biochem. 131, 5-7,
#: 1983) with heavy atoms only: phi O5'-C1'-Ox-Cx, psi C1'-Ox-Cx-C(x-1), and for
#: a 1-6 linkage omega O5-C5-C6-O6 (the same angle as O6-C6-C5-O5).  They are
#: extended in the same way to the linkages the recommendations do not spell
#: out: the ring oxygen and anomeric carbon of furanose and 2-ketose donors in
#: phi, a ring oxygen or C(x+1) for psi where there is no C(x-1), and an omega
#: for every acceptor carbon outside the ring (C5 of a pentofuranose, C6 of a
#: hexofuranose, C8 and C9 of a sialic acid, C1 of a 2-ketose).
TORSION_CONVENTIONS: Dict[str, Any] = {
	"version": TORSION_DEFINITIONS_VERSION,
	"reference": (
		"IUPAC-IUB JCBN, Symbols for specifying the conformation of polysaccharide chains (1981), "
		"Eur. J. Biochem. 131, 5-7 (1983); heavy atoms only"
	),
	"units": "degrees, -180 to 180",
	"naming": (
		"<donor residue>_<acceptor residue>_<phi|psi|omega>, with the residue numbers of the topology. "
		"Primed atoms belong to the donor, x is the acceptor position. The four atoms of every torsion "
		"of the entry are in its metadata (atom_names, residue_ids)"
	),
	"phi": {
		"definition": "ring oxygen' - anomeric carbon' - Ox - Cx",
		"atoms": {
			"aldopyranose donor": ["O5'", "C1'", "Ox", "Cx"],
			"aldofuranose donor (Araf, Galf, Ribf)": ["O4'", "C1'", "Ox", "Cx"],
			"2-ketopyranose donor (Neu5Ac, Neu5Gc, Kdn, Kdo, Tagp, Sorp)": ["O6'", "C2'", "Ox", "Cx"],
			"2-ketofuranose donor (Fruf, Psif)": ["O5'", "C2'", "Ox", "Cx"],
		},
	},
	"psi": {
		"definition": (
			"anomeric carbon' - Ox - Cx - C(x-1); where the acceptor has no C(x-1): its ring oxygen when Cx "
			"is the anomeric C1 of an aldose (1-1), otherwise C(x+1) (2-1 to the C1 of a ketose)"
		),
		"atoms": {
			"aldose donor, x > 1": ["C1'", "Ox", "Cx", "C(x-1)"],
			"2-ketose donor, x > 1": ["C2'", "Ox", "Cx", "C(x-1)"],
			"1-1 to an aldose": ["C1'", "O1", "C1", "O5"],
			"2-1 to a 2-ketose": ["C2'", "O1", "C1", "C2"],
		},
		"examples": {
			"1-4": ["C1'", "O4", "C4", "C3"],
			"1-6": ["C1'", "O6", "C6", "C5"],
			"2-3 (sialic acid)": ["C2'", "O3", "C3", "C2"],
			"2-6 (sialic acid)": ["C2'", "O6", "C6", "C5"],
			"2-8 (sialic acid)": ["C2'", "O8", "C8", "C7"],
		},
	},
	"omega": {
		"definition": (
			"for an acceptor carbon Cx outside the ring: O - C(x-1) - Cx - Ox, the oxygen being the ring oxygen "
			"when C(x-1) is a ring atom and the hydroxyl oxygen of C(x-1) otherwise (C(x+1) and its ring "
			"oxygen for 2-1). For a 1-6 linkage this is the IUPAC omega O5-C5-C6-O6, the same angle as "
			"O6-C6-C5-O5"
		),
		"atoms": {
			"1-6 to a hexopyranose": ["O5", "C5", "C6", "O6"],
			"1-6 to a hexofuranose (Galf)": ["O5", "C5", "C6", "O6"],
			"1-5 to a pentofuranose (Araf)": ["O4", "C4", "C5", "O5"],
			"2-8 to a sialic acid": ["O7", "C7", "C8", "O8"],
			"2-9 to a sialic acid": ["O8", "C8", "C9", "O9"],
			"2-1 to a 2-ketofuranose (Fruf)": ["O5", "C2", "C1", "O1"],
		},
		"rotamers": {
			"1-6 to a D-hexopyranose": {"gg": -60, "gt": 60, "tg": 180},
		},
	},
	"substituent": {
		"definition": (
			"the omega of an exocyclic carbon whose oxygen carries a group that is not a sugar (sulfate, "
			"phosphate, methyl, acetyl ...), with the atoms of a glycosidic omega. Listed in "
			"torsions.substituent, not among the glycosidic torsions, and named "
			"<substituent residue>_<sugar residue>_omega"
		),
		"atoms": {
			"6-O-substituted hexopyranose": ["O5", "C5", "C6", "O6"],
			"8-O-substituted sialic acid": ["O7", "C7", "C8", "O8"],
			"9-O-substituted sialic acid": ["O8", "C8", "C9", "O9"],
		},
	},
	"other": (
		"every other rotatable bond, in one residue or between residues, with an oxygen at each end where "
		"there is one, named after its four atoms"
	),
}

__all__ = [
	"Torsion",
	"TORSION_CONVENTIONS",
	"TORSION_DEFINITIONS_VERSION",
	"LINKAGE_FAMILIES",
	"Linkage",
	"find_linkages",
	"linkage_roles",
	"get_movable_torsions",
	"get_torsion_values",
	"get_torsion_values_batch",
	"set_torsion_values",
	"generate_wiggles",
	"get_glycan_torsions_enhanced",
	"annotate_torsion",
	"build_torsion_metadata",
	"summarize_torsion_metadata",
	"classify_glycosidic_from_name",
	"circular_stats",
	"save_torparts_npz",
	"plot_torsion_distribution",
]


def _require_rdkit():  # Internal helper to raise a clearer error
	if Chem is None or rdMolTransforms is None:
		raise ImportError("RDKit is required for torsion utilities. Install rdkit-pypi or RDKit distribution.")


def get_movable_torsions(mol: object) -> List[Torsion]:
	"""Return a list of rotatable torsions as 4-tuples of atom indices (i,j,k,l).

	Heuristic: identify single, non-ring bonds between two non-terminal atoms
	/ not triple or double bonds, then pick one heavy (or fallback) neighbor
	on each side to define the dihedral.
	"""
	_require_rdkit()
	patt = Chem.MolFromSmarts('[!$(*#*)&!D1]-[!$(*#*)&!D1]')
	potential_bonds = mol.GetSubstructMatches(patt)
	rot_bonds = []
	for a1, a2 in potential_bonds:
		bond = mol.GetBondBetweenAtoms(a1, a2)
		if not bond:
			continue
		if bond.GetBondType() == Chem.BondType.SINGLE and not bond.IsInRing():
			rot_bonds.append((a1, a2))
	torsion_list: List[Torsion] = []
	for j, k in rot_bonds:
		atom_j = mol.GetAtomWithIdx(j)
		atom_k = mol.GetAtomWithIdx(k)
		i_cands = [n for n in atom_j.GetNeighbors() if n.GetIdx() != k]
		if not i_cands:
			continue
		i_heavy = [n for n in i_cands if n.GetAtomicNum() != 1]
		i = i_heavy[0].GetIdx() if i_heavy else i_cands[0].GetIdx()
		l_cands = [n for n in atom_k.GetNeighbors() if n.GetIdx() != j]
		if not l_cands:
			continue
		l_heavy = [n for n in l_cands if n.GetAtomicNum() != 1]
		l = l_heavy[0].GetIdx() if l_heavy else l_cands[0].GetIdx()
		torsion_list.append((i, j, k, l))
	return torsion_list


def get_torsion_values(mol: object, torsion_list: Sequence[Torsion], conf_id: int = -1) -> List[float]:
	"""Return torsion angles in degrees for the specified conformer (default last)."""
	_require_rdkit()
	conf = mol.GetConformer(conf_id)
	return [rdMolTransforms.GetDihedralDeg(conf, *tors) for tors in torsion_list]


def get_torsion_values_batch(coords_array: np.ndarray, torsion_list: Sequence[Torsion]) -> np.ndarray:
	"""Compute torsion angles for multiple frames using vectorized operations.
	
	Parameters
	----------
	coords_array : np.ndarray
		Coordinates array with shape (n_frames, n_atoms, 3)
	torsion_list : Sequence[Torsion]
		List of torsion definitions as 4-tuples of atom indices
		
	Returns
	-------
	np.ndarray
		Torsion angles in degrees with shape (n_frames, n_torsions)
	"""
	n_frames = coords_array.shape[0]
	n_torsions = len(torsion_list)
	
	if n_torsions == 0:
		return np.zeros((n_frames, 0))
	
	# Pre-allocate result array
	torsion_angles = np.zeros((n_frames, n_torsions))
	
	# Process each torsion
	for t_idx, (i, j, k, l) in enumerate(torsion_list):
		# Extract coordinates for the 4 atoms across all frames
		# Shape: (n_frames, 4, 3)
		torsion_coords = coords_array[:, [i, j, k, l], :]
		
		# Compute torsion angles for all frames at once
		angles_deg = _compute_dihedral_batch(torsion_coords)
		torsion_angles[:, t_idx] = angles_deg
	
	return torsion_angles


def _compute_dihedral_batch(coords: np.ndarray) -> np.ndarray:
	"""Compute dihedral angles for a batch of 4-atom coordinate sets.
	
	Parameters
	----------
	coords : np.ndarray
		Coordinates with shape (n_frames, 4, 3) where the 4 atoms define the dihedral
		
	Returns
	-------
	np.ndarray
		Dihedral angles in degrees with shape (n_frames,)
	"""
	# Extract the 4 atom positions
	# coords has shape (n_frames, 4, 3)
	p0, p1, p2, p3 = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
	
	# Compute bond vectors
	b1 = p1 - p0  # vector from atom 0 to atom 1
	b2 = p2 - p1  # vector from atom 1 to atom 2  
	b3 = p3 - p2  # vector from atom 2 to atom 3
	
	# Normalize b2 (the central bond)
	b2_norm = np.linalg.norm(b2, axis=1, keepdims=True)
	b2_norm = np.where(b2_norm == 0, 1, b2_norm)  # Avoid division by zero
	b2_unit = b2 / b2_norm
	
	# Compute normal vectors to the planes
	n1 = np.cross(b1, b2)  # normal to plane defined by atoms 0,1,2
	n2 = np.cross(b2, b3)  # normal to plane defined by atoms 1,2,3
	
	# Normalize the normal vectors
	n1_norm = np.linalg.norm(n1, axis=1, keepdims=True)
	n2_norm = np.linalg.norm(n2, axis=1, keepdims=True)
	
	# Handle cases where norm is zero
	n1_norm = np.where(n1_norm == 0, 1, n1_norm)
	n2_norm = np.where(n2_norm == 0, 1, n2_norm)
	
	n1_unit = n1 / n1_norm
	n2_unit = n2 / n2_norm
	
	# Compute the dihedral angle using atan2 for proper quadrant handling
	# cos(dihedral) = n1 · n2
	cos_dihedral = np.sum(n1_unit * n2_unit, axis=1)
	
	# sin(dihedral) = (n1 × n2) · b2_unit
	cross_n1_n2 = np.cross(n1_unit, n2_unit)
	sin_dihedral = np.sum(cross_n1_n2 * b2_unit, axis=1)
	
	# Calculate dihedral angle in radians, then convert to degrees
	dihedral_rad = np.arctan2(sin_dihedral, cos_dihedral)
	dihedral_deg = np.degrees(dihedral_rad)
	
	return dihedral_deg


def set_torsion_values(
	mol: object,
	torsion_list: Sequence[Torsion],
	torsion_values: Sequence[float],
	template_conf_id: int = -1,
) -> int:
	"""Create a new conformer with updated torsion angles.

	Returns the new conformer ID.
	"""
	_require_rdkit()
	template_conf = mol.GetConformer(template_conf_id)
	from rdkit import Chem as _Chem  # local alias
	new_conf = _Chem.Conformer(template_conf)
	for tors, val in zip(torsion_list, torsion_values):
		rdMolTransforms.SetDihedralDeg(new_conf, *tors, float(val))
	new_conf_id = mol.AddConformer(new_conf, assignId=True)
	return new_conf_id


def generate_wiggles(
	mol: object,
	torsion_list: Sequence[Torsion],
	degrees: float = 5.0,
	per_torsion: int = 2,
	conf_id: int = 0,
) -> List[int]:
	"""Generate small torsion perturbation conformers.

	Parameters
	----------
	mol : RDKit Mol with at least one conformer.
	torsion_list : iterable of torsion tuples (i,j,k,l).
	degrees : float, magnitude for wiggle.
	per_torsion : 1 -> only +deg, 2 -> +/-deg.
	conf_id : base conformer to perturb.

	Returns
	-------
	list[int]
		New conformer IDs.
	"""
	_require_rdkit()
	base_values = get_torsion_values(mol, torsion_list, conf_id=conf_id)
	new_ids: List[int] = []
	for idx, base in enumerate(base_values):
		if per_torsion == 1:
			offsets = [degrees]
		else:
			offsets = [-degrees, degrees]
		for off in offsets:
			vals = list(base_values)
			vals[idx] = base + off
			cid = set_torsion_values(mol, torsion_list, vals, template_conf_id=conf_id)
			new_ids.append(cid)
	return new_ids


def describe_torsions(mol: object, torsion_list: Sequence[Torsion], conf_id: int = 0) -> List[str]:
	"""Return human-readable labels for torsions (element+idx sequences)."""
	_require_rdkit()
	labels = []
	for (i,j,k,l) in torsion_list:
		atoms = [mol.GetAtomWithIdx(a) for a in (i,j,k,l)]
		labels.append("-".join(f"{a.GetSymbol()}{a.GetIdx()}" for a in atoms))
	values = get_torsion_values(mol, torsion_list, conf_id=conf_id)
	return [f"{lab}: {val:.2f} deg" for lab, val in zip(labels, values)]


def _sanitize_torsion_token(value: Any) -> str:
	text = "NA" if value is None else str(value).strip()
	if not text:
		text = "NA"
	text = re.sub(r"[^A-Za-z0-9]+", "_", text)
	text = text.strip("_")
	return text or "NA"


def _torsion_label_parts_from_name(torsion_name: str) -> Optional[List[Tuple[str, int]]]:
	parts = torsion_name.split('-')
	if len(parts) != 4:
		return None
	parsed: List[Tuple[str, int]] = []
	for part in parts:
		if '_' not in part:
			return None
		atom_name, res_id = part.rsplit('_', 1)
		try:
			parsed.append((atom_name, int(res_id)))
		except ValueError:
			return None
	return parsed


def _classify_glycosidic_from_parts(atom_names: Sequence[str], residue_ids: Sequence[int]) -> Optional[str]:
	if len(atom_names) != 4 or len(residue_ids) != 4:
		return None
	atom1, atom2, atom3, atom4 = atom_names
	res1, res2, res3, res4 = residue_ids

	if (
		atom1 == "O5"
		and atom2 == "C1"
		and atom3.startswith("O")
		and atom4.startswith("C")
		and res1 == res2
		and res3 == res4
		and res1 != res3
	):
		return f"{res1}_{res3}_phi"

	# Sialic acids use C2 as the anomeric carbon and O6 as the ring oxygen.
	# Keep the same ring-oxygen convention used above for ordinary pyranoses.
	if (
		atom1 == "O6"
		and atom2 == "C2"
		and atom3.startswith("O")
		and atom4.startswith("C")
		and res1 == res2
		and res3 == res4
		and res1 != res3
	):
		return f"{res1}_{res3}_phi"

	# Furanoses, with the same ring-oxygen convention: O4 is the ring oxygen
	# of an aldofuranose (Araf, Galf, Ribf), O5 that of a ketofuranose (Fruf,
	# Psif), whose anomeric carbon is C2.  Neither bond exists in a pyranose.
	if (
		(atom1, atom2) in (("O4", "C1"), ("O5", "C2"))
		and atom3.startswith("O")
		and atom4.startswith("C")
		and res1 == res2
		and res3 == res4
		and res1 != res3
	):
		return f"{res1}_{res3}_phi"

	if (
		atom1 == "C1"
		and atom2.startswith("O")
		and atom3.startswith("C")
		and atom4.startswith(("C", "O"))
		# In GLYCAM/PDB connectivity the bridging oxygen belongs to the
		# acceptor residue: C1(donor)-O(acceptor)-Cx(acceptor)-X(acceptor).
		# The previous donor-owned oxygen rule mislabeled valid psi torsions
		# (for example GS00971 C1_3-O4_2-C4_2-C5_2) as residue-local.
		and res1 != res2
		and res2 == res3 == res4
	):
		return f"{res1}_{res3}_psi"

	if (
		atom1 == "C2"
		and atom2.startswith("O")
		and atom3.startswith("C")
		and atom4.startswith(("C", "O"))
		and res1 != res2
		and res2 == res3 == res4
	):
		return f"{res1}_{res3}_psi"

	if (
		atom1.startswith("O")
		and atom1 != "O5"
		and atom2 == "C6"
		and atom3 == "C5"
		and atom4 == "O5"
		and res2 == res3 == res4
		and res1 != res2
	):
		return f"{res1}_{res2}_omega"

	return None


def classify_glycosidic_from_name(torsion_name: str) -> str:
	"""Classify a torsion as glycosidic (phi/psi/omega) based on atom names and positions.
	
	Expected format: "atom1_res1-atom2_res2-atom3_res3-atom4_res4"
	Examples:
	- phi: "O5_1-C1_1-O3_2-C3_2" -> "1_2_phi"
	- psi: "C1_1-O3_2-C3_2-C2_2" -> "1_2_psi"  
	- omega: "O3_2-C6_3-C5_3-O5_3" -> "2_3_omega"
	"""
	try:
		# Parse torsion name: "atom1_res1-atom2_res2-atom3_res3-atom4_res4"
		parts = torsion_name.split('-')
		if len(parts) != 4:
			return f"other_{torsion_name}"
		
		atoms_res = []
		for part in parts:
			if '_' not in part:
				return f"other_{torsion_name}"
			atom_name, res_id = part.rsplit('_', 1)
			try:
				res_id = int(res_id)
			except ValueError:
				return f"other_{torsion_name}"
			atoms_res.append((atom_name, res_id))
		
		# Extract atom names and residue IDs
		atom1, res1 = atoms_res[0]
		atom2, res2 = atoms_res[1] 
		atom3, res3 = atoms_res[2]
		atom4, res4 = atoms_res[3]
		
		# Check for phi angle: O5-C1-O(gly)-Cx
		# Pattern: ring oxygen - anomeric carbon - glycosidic oxygen - acceptor carbon
		if (atom1 in ['O5'] and atom2 in ['C1'] and 
			atom3.startswith('O') and atom3 != 'O5' and
			atom4.startswith('C') and res1 != res3):
			return f"{res1}_{res3}_phi"
		
		# Check for psi angle: C1-O(gly)-Cx-C(x±1)
		# Pattern: anomeric carbon - glycosidic oxygen - acceptor carbon - next carbon
		if (atom1 in ['C1'] and atom2.startswith('O') and atom2 != 'O5' and
			atom3.startswith('C') and atom4.startswith('C') and 
			res1 != res2 and res2 == res3):
			return f"{res1}_{res2}_psi"
		
		# Check for omega angle: O(gly)-C6-C5-O5 (for 1→6 linkages)
		# Pattern: glycosidic oxygen - C6 - C5 - ring oxygen (oxygen at both ends!)
		if (atom1.startswith('O') and atom1 != 'O5' and
			atom2 in ['C6'] and atom3 in ['C5'] and atom4 in ['O5'] and
			res1 != res2 and res2 == res3 == res4):
			return f"{res1}_{res2}_omega"
		
		return f"other_{torsion_name}"
		
	except Exception:
		return f"other_{torsion_name}"


def classify_glycosidic_from_name(torsion_name: str) -> str:
	"""Classify a torsion as glycosidic (phi/psi/omega) based on atom names and positions."""
	try:
		parts = _torsion_label_parts_from_name(torsion_name)
		if not parts:
			return f"other_{torsion_name}"

		atom_names = [atom_name for atom_name, _ in parts]
		residue_ids = [res_id for _, res_id in parts]

		forward = _classify_glycosidic_from_parts(atom_names, residue_ids)
		if forward:
			return forward

		reverse = _classify_glycosidic_from_parts(atom_names[::-1], residue_ids[::-1])
		if reverse:
			return reverse

		return f"other_{torsion_name}"
	except Exception:
		return f"other_{torsion_name}"


def _torsion_atom_metadata(
	torsion: Torsion,
	atoms_df,
) -> tuple[List[int], List[str], List[int], List[str]]:
	atom_indices = [int(i) for i in torsion]
	atom_names_col = atoms_df["name"].to_list() if "name" in atoms_df.columns else [f"X{i}" for i in range(len(atoms_df))]
	residue_ids_col = atoms_df["res_id"].to_list() if "res_id" in atoms_df.columns else [1] * len(atoms_df)
	residue_names_col = atoms_df["res_name"].to_list() if "res_name" in atoms_df.columns else ["UNK"] * len(atoms_df)

	atom_names = [str(atom_names_col[i]) for i in atom_indices]
	residue_ids = [int(residue_ids_col[i]) for i in atom_indices]
	residue_names = [str(residue_names_col[i]) for i in atom_indices]
	return atom_indices, atom_names, residue_ids, residue_names


def _make_original_torsion_name(atom_names: Sequence[str], residue_ids: Sequence[int]) -> str:
	return "-".join(f"{atom_name}_{res_id}" for atom_name, res_id in zip(atom_names, residue_ids))


def _canonicalize_non_glycosidic_orientation(
	atom_names: Sequence[str],
	residue_ids: Sequence[int],
	residue_names: Sequence[str],
) -> tuple[List[str], List[int], List[str]]:
	forward = [
		f"c0_{residue_ids[1]}_{_sanitize_torsion_token(residue_names[1])}_{_sanitize_torsion_token(atom_names[1])}",
		f"c1_{residue_ids[2]}_{_sanitize_torsion_token(residue_names[2])}_{_sanitize_torsion_token(atom_names[2])}",
		f"f0_{residue_ids[0]}_{_sanitize_torsion_token(residue_names[0])}_{_sanitize_torsion_token(atom_names[0])}",
		f"f1_{residue_ids[3]}_{_sanitize_torsion_token(residue_names[3])}_{_sanitize_torsion_token(atom_names[3])}",
	]
	reverse = [
		f"c0_{residue_ids[2]}_{_sanitize_torsion_token(residue_names[2])}_{_sanitize_torsion_token(atom_names[2])}",
		f"c1_{residue_ids[1]}_{_sanitize_torsion_token(residue_names[1])}_{_sanitize_torsion_token(atom_names[1])}",
		f"f0_{residue_ids[3]}_{_sanitize_torsion_token(residue_names[3])}_{_sanitize_torsion_token(atom_names[3])}",
		f"f1_{residue_ids[0]}_{_sanitize_torsion_token(residue_names[0])}_{_sanitize_torsion_token(atom_names[0])}",
	]
	if tuple(reverse) < tuple(forward):
		return list(atom_names[::-1]), list(residue_ids[::-1]), list(residue_names[::-1])
	return list(atom_names), list(residue_ids), list(residue_names)


def _infer_non_glycosidic_identity(
	atom_names: Sequence[str],
	residue_ids: Sequence[int],
	residue_names: Sequence[str],
) -> tuple[str, str, Optional[str]]:
	center_pair = tuple(atom_names[1:3])
	center_pair_set = frozenset(center_pair)
	same_central_residue = residue_ids[1] == residue_ids[2]

	if same_central_residue:
		if center_pair_set == frozenset({"C5", "C6"}) and {"C4", "O6"}.issubset(set(atom_names)):
			return "residue_local", "exocyclic_c6", "exocyclic_c6"
		if center_pair_set == frozenset({"C2", "N2"}):
			return "residue_local", "n_acetyl_linker", "n_acetyl"
		if center_pair_set == frozenset({"N2", "C7"}):
			return "residue_local", "n_acetyl_amide", "n_acetyl"
		if center_pair_set == frozenset({"C7", "C8"}):
			return "residue_local", "n_acetyl_methyl", "n_acetyl"
		return "residue_local", f"bond_{_sanitize_torsion_token(atom_names[1])}_{_sanitize_torsion_token(atom_names[2])}", None

	return "inter_residue", f"bond_{_sanitize_torsion_token(atom_names[1])}_{_sanitize_torsion_token(atom_names[2])}", None


def _build_non_glycosidic_canonical_name(
	atom_names: Sequence[str],
	residue_ids: Sequence[int],
	residue_names: Sequence[str],
) -> str:
	atom_names, residue_ids, residue_names = _canonicalize_non_glycosidic_orientation(atom_names, residue_ids, residue_names)
	center_left = f"r{residue_ids[1]}_{_sanitize_torsion_token(residue_names[1])}_{_sanitize_torsion_token(atom_names[1])}"
	center_right = f"r{residue_ids[2]}_{_sanitize_torsion_token(residue_names[2])}_{_sanitize_torsion_token(atom_names[2])}"
	flank_left = f"r{residue_ids[0]}_{_sanitize_torsion_token(residue_names[0])}_{_sanitize_torsion_token(atom_names[0])}"
	flank_right = f"r{residue_ids[3]}_{_sanitize_torsion_token(residue_names[3])}_{_sanitize_torsion_token(atom_names[3])}"
	return f"{center_left}__{center_right}__{flank_left}__{flank_right}"


#: Families of torsions that belong to a linkage: a glycosidic linkage, or a
#: substituent (sulfate, methyl ...) on an exocyclic oxygen.
LINKAGE_FAMILIES = ("glycosidic", "substituent")


def annotate_torsion(
	torsion: Torsion,
	atoms_df,
	index: int,
	original_name: Optional[str] = None,
	linkage_role: Optional[Tuple[str, str, str]] = None,
	classify_by_name: bool = True,
) -> Dict[str, Any]:
	"""Build stable metadata for a torsion definition.

	``linkage_role`` is ``(family, canonical name, subtype)`` when the torsion
	belongs to a linkage found in the structure (``find_linkages``).  Without
	it, ``classify_by_name`` decides whether the atom names alone may make the
	torsion glycosidic (only when the structure is not at hand).
	"""
	atom_indices, atom_names, residue_ids, residue_names = _torsion_atom_metadata(torsion, atoms_df)
	original_name = original_name or _make_original_torsion_name(atom_names, residue_ids)

	family: Optional[str] = None
	canonical: Optional[str] = None
	if linkage_role is not None:
		family, canonical, _ = linkage_role
	elif classify_by_name:
		by_name = classify_glycosidic_from_name(original_name)
		if by_name and not by_name.startswith("other_"):
			family, canonical = "glycosidic", by_name
	if family in LINKAGE_FAMILIES and canonical:
		subtype = canonical.rsplit("_", 1)[-1]
		entry = {
			"index": int(index),
			"original_name": original_name,
			"canonical_name": canonical,
			"family": family,
			"subtype": subtype,
			"common_name": subtype,
			"atom_indices": atom_indices,
			"atom_names": atom_names,
			"residue_ids": residue_ids,
			"residue_names": residue_names,
			"central_bond_atoms": [atom_names[1], atom_names[2]],
			"central_bond_residues": [residue_ids[1], residue_ids[2]],
		}
		if family == "glycosidic":
			entry["glycosidic_name"] = canonical
		else:
			entry["substituent_name"] = canonical
		return entry

	family, subtype, common_name = _infer_non_glycosidic_identity(atom_names, residue_ids, residue_names)
	canonical_name = _build_non_glycosidic_canonical_name(atom_names, residue_ids, residue_names)
	return {
		"index": int(index),
		"original_name": original_name,
		"canonical_name": canonical_name,
		"family": family,
		"subtype": subtype,
		"common_name": common_name,
		"atom_indices": atom_indices,
		"atom_names": atom_names,
		"residue_ids": residue_ids,
		"residue_names": residue_names,
		"central_bond_atoms": [atom_names[1], atom_names[2]],
		"central_bond_residues": [residue_ids[1], residue_ids[2]],
	}


def linkage_roles(atoms_df, bonds_df) -> Dict[Torsion, Tuple[str, str, str]]:
	"""{torsion (either direction, as the smaller tuple): (family, canonical name, subtype)} of a structure."""
	roles: Dict[Torsion, Tuple[str, str, str]] = {}
	for linkage in find_linkages(atoms_df, bonds_df):
		for subtype, torsion in linkage.torsions():
			roles.setdefault(
				_normalised_torsion(torsion),
				(linkage.kind, f"{linkage.donor}_{linkage.acceptor}_{subtype}", subtype),
			)
	return roles


def build_torsion_metadata(
	torsion_list: Sequence[Torsion],
	atoms_df,
	original_names: Optional[Sequence[str]] = None,
	bonds_df=None,
) -> List[Dict[str, Any]]:
	"""Metadata of every torsion: family (glycosidic, substituent, residue_local, inter_residue), names, atoms.

	With the bond table the linkages are found in the structure
	(``find_linkages``) and a torsion is glycosidic or substituent exactly when
	it is one of their phi, psi or omega; this is what tells the omega of a
	1-6 linkage (O6 bonded to the anomeric carbon of another sugar) from the
	omega of a 6-O-sulfate and from a free C6 hydroxyl.  Without the bond table
	the atom names decide (``classify_glycosidic_from_name``).
	"""
	roles: Optional[Dict[Torsion, Tuple[str, str, str]]] = None
	if bonds_df is not None and len(torsion_list):
		roles = linkage_roles(atoms_df, bonds_df)

	metadata: List[Dict[str, Any]] = []
	for index, torsion in enumerate(torsion_list):
		original_name = original_names[index] if original_names and index < len(original_names) else None
		metadata.append(annotate_torsion(
			torsion,
			atoms_df=atoms_df,
			index=index,
			original_name=original_name,
			linkage_role=roles.get(_normalised_torsion(torsion)) if roles is not None else None,
			classify_by_name=roles is None,
		))
	return metadata


def summarize_torsion_metadata(metadata: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
	by_family: Dict[str, int] = {}
	by_subtype: Dict[str, int] = {}
	by_common_name: Dict[str, int] = {}
	for entry in metadata:
		family = str(entry.get("family") or "unknown")
		subtype = str(entry.get("subtype") or "unknown")
		common_name = entry.get("common_name")
		by_family[family] = by_family.get(family, 0) + 1
		by_subtype[subtype] = by_subtype.get(subtype, 0) + 1
		if common_name:
			by_common_name[str(common_name)] = by_common_name.get(str(common_name), 0) + 1

	return {
		"total": len(metadata),
		"glycosidic": sum(1 for entry in metadata if entry.get("family") == "glycosidic"),
		"substituent": sum(1 for entry in metadata if entry.get("family") == "substituent"),
		"non_glycosidic": sum(1 for entry in metadata if entry.get("family") not in LINKAGE_FAMILIES),
		"by_family": dict(sorted(by_family.items())),
		"by_subtype": dict(sorted(by_subtype.items())),
		"by_common_name": dict(sorted(by_common_name.items())),
	}


def circular_stats(deg_values: np.ndarray) -> dict:
	"""Return circular mean/std (degrees) along with linear mean/std."""
	if deg_values.size == 0:
		return {"mean_deg": None, "std_deg": None, "circular_mean_deg": None, "circular_std_deg": None}
	# Linear
	lin_mean = float(np.mean(deg_values))
	lin_std = float(np.std(deg_values, ddof=1)) if deg_values.size > 1 else 0.0
	# Circular
	rad = np.deg2rad(deg_values)
	s = np.sin(rad).sum()
	c = np.cos(rad).sum()
	n = deg_values.size
	R = np.sqrt(s*s + c*c)/n
	circ_mean = np.rad2deg(np.arctan2(s, c))
	# Circ std: sqrt(-2 ln R) in radians
	circ_std = float(np.rad2deg(np.sqrt(max(0.0, -2.0*np.log(max(R, 1e-12)))))) if R > 0 else 180.0
	return {
		"mean_deg": lin_mean,
		"std_deg": lin_std,
		"circular_mean_deg": float(circ_mean),
		"circular_std_deg": circ_std,
	}



_CARBON_LOCANT = re.compile(r"C(\d+)")


def _carbon_locant(name: object) -> Optional[int]:
	"""x of an atom named Cx (C1, C12), else None."""
	match = _CARBON_LOCANT.fullmatch(str(name).strip())
	return int(match.group(1)) if match else None


def _normalised_torsion(torsion: Sequence[int]) -> Torsion:
	forward = tuple(int(atom) for atom in torsion)
	return min(forward, forward[::-1])  # type: ignore[return-value]


class Linkage:
	"""The torsions of one linkage: a glycosidic linkage, or a substituent on an exocyclic oxygen.

	``kind`` is ``"glycosidic"`` (``donor`` is the sugar whose anomeric carbon
	makes the bond) or ``"substituent"`` (``donor`` is the residue of the group
	that is not a sugar, SO3, MEX, ACX ...; only ``omega`` is set).  The
	torsions are 4-tuples of atom indices in the order of
	``TORSION_CONVENTIONS``.
	"""

	__slots__ = ("kind", "donor", "acceptor", "phi", "psi", "omega", "acceptor_carbon")

	def __init__(self, kind, donor, acceptor, phi=None, psi=None, omega=None, acceptor_carbon=None):
		self.kind = kind
		self.donor = donor
		self.acceptor = acceptor
		self.phi = phi
		self.psi = psi
		self.omega = omega
		self.acceptor_carbon = acceptor_carbon

	@property
	def name(self) -> str:
		return f"{self.donor}_{self.acceptor}"

	def torsions(self) -> List[Tuple[str, Torsion]]:
		return [
			(subtype, getattr(self, subtype))
			for subtype in ("phi", "psi", "omega")
			if getattr(self, subtype) is not None
		]

	def __repr__(self) -> str:  # pragma: no cover - debugging aid
		return f"Linkage({self.kind}, {self.name}, phi={self.phi}, psi={self.psi}, omega={self.omega})"


def _heavy_atom_graph(atoms_df, bonds_df):
	"""Bond graph without hydrogens, with the atom names, residue ids and elements (upper case)."""
	if nx is None:
		raise ImportError("NetworkX is required for graph-based torsion detection.")
	atom_count = len(atoms_df)
	graph = nx.Graph()
	# Bonds might be 1-based; arrays are 0-based.
	for a, b in bonds_df.iter_rows():
		a_idx = int(a) if int(a) < atom_count else int(a) - 1
		b_idx = int(b) if int(b) < atom_count else int(b) - 1
		if 0 <= a_idx < atom_count and 0 <= b_idx < atom_count:
			graph.add_edge(a_idx, b_idx)
	if "name" in atoms_df.columns:
		names = [str(value) for value in atoms_df["name"].to_list()]
	else:
		names = [f"X{i}" for i in range(atom_count)]
	if "element" in atoms_df.columns:
		elements = [str(value).strip().upper() for value in atoms_df["element"].to_list()]
	else:  # the first letter of the atom name (C1, O5, H1 ...)
		elements = [next((ch for ch in name if ch.isalpha()), "X").upper() for name in names]
	graph.remove_nodes_from([i for i, element in enumerate(elements) if element == "H"])
	res_ids = atoms_df["res_id"].to_list() if "res_id" in atoms_df.columns else [1] * atom_count
	return graph, names, res_ids, elements


def _residue_rings(graph, res_ids) -> Dict[Any, set]:
	"""The ring atoms of every residue (rings within one residue)."""
	by_residue: Dict[Any, List[int]] = {}
	for node in graph.nodes():
		by_residue.setdefault(res_ids[node], []).append(node)
	rings: Dict[Any, set] = {}
	for residue, nodes in by_residue.items():
		ring_atoms: set = set()
		for cycle in nx.cycle_basis(graph.subgraph(nodes)):
			ring_atoms.update(cycle)
		if ring_atoms:
			rings[residue] = ring_atoms
	return rings


def _anomeric_carbons(graph, names, res_ids, elements, rings) -> Dict[int, int]:
	"""{anomeric carbon: ring oxygen} of every residue with a ring.

	The anomeric carbon is the ring carbon bonded both to the ring oxygen and
	to an oxygen outside the ring: C1 of an aldose, C2 of a 2-ketose (the other
	ring carbon next to the ring oxygen carries a carbon, C5 or C6).
	"""
	result: Dict[int, int] = {}
	for residue, ring in rings.items():
		candidates = []
		for atom in ring:
			if elements[atom] != "C":
				continue
			ring_oxygens = [n for n in graph.neighbors(atom) if n in ring and elements[n] == "O"]
			outside_oxygens = [n for n in graph.neighbors(atom) if n not in ring and elements[n] == "O"]
			if ring_oxygens and outside_oxygens:
				locant = _carbon_locant(names[atom])
				candidates.append((locant if locant is not None else 10**6, atom, min(ring_oxygens)))
		if candidates:
			_, atom, ring_oxygen = min(candidates)
			result[atom] = ring_oxygen
	return result


def _psi_reference(graph, names, res_ids, elements, rings, carbon: int, oxygen: int) -> Optional[int]:
	"""The fourth atom of psi: C(x-1); the ring oxygen of an anomeric C1; C(x+1); else a neighbour."""
	residue = res_ids[carbon]
	neighbours = [n for n in graph.neighbors(carbon) if n != oxygen and res_ids[n] == residue]
	by_name = {names[n]: n for n in neighbours}
	x = _carbon_locant(names[carbon])
	if x is not None:
		if f"C{x - 1}" in by_name:
			return by_name[f"C{x - 1}"]
		ring = rings.get(residue, set())
		if carbon in ring:
			ring_oxygens = [n for n in neighbours if n in ring and elements[n] == "O"]
			if ring_oxygens:
				return ring_oxygens[0]
		if f"C{x + 1}" in by_name:
			return by_name[f"C{x + 1}"]
	carbons = [n for n in neighbours if elements[n] == "C"]
	if carbons:
		return carbons[0]
	return neighbours[0] if neighbours else None


def _omega(graph, names, res_ids, elements, rings, carbon: int, oxygen: int, reference: Optional[int]) -> Optional[Torsion]:
	"""O-C(x-1)-Cx-Ox when Cx lies outside the ring of a sugar residue, else None.

	The first oxygen is the ring oxygen when C(x-1) is a ring atom (O5 of a
	hexopyranose, O4 of Araf, O5 of Fruf), otherwise the hydroxyl oxygen of
	C(x-1) (O7 and O8 of a sialic acid).
	"""
	residue = res_ids[carbon]
	ring = rings.get(residue)
	if not ring or carbon in ring or reference is None or elements[reference] != "C":
		return None
	oxygens = [
		n for n in graph.neighbors(reference)
		if n != carbon and res_ids[n] == residue and elements[n] == "O"
	]
	if not oxygens:
		return None
	ring_oxygens = [n for n in oxygens if n in ring]
	if ring_oxygens:
		outer = ring_oxygens[0]
	else:
		locant = _carbon_locant(names[reference])
		named = [n for n in oxygens if locant is not None and names[n] == f"O{locant}"]
		outer = named[0] if named else oxygens[0]
	return (outer, reference, carbon, oxygen)


def _find_linkages_in_graph(graph, names, res_ids, elements) -> List[Linkage]:
	rings = _residue_rings(graph, res_ids)
	anomeric = _anomeric_carbons(graph, names, res_ids, elements, rings)
	node_order = {node: position for position, node in enumerate(graph.nodes())}
	# C1 donors first, then the 2-ketoses, each in the order of the graph.
	donors = sorted(anomeric, key=lambda atom: (names[atom] != "C1", node_order[atom]))

	linkages: List[Linkage] = []
	glycosidic_bonds: set = set()
	for anomeric_carbon in donors:
		donor = res_ids[anomeric_carbon]
		ring_oxygen = anomeric[anomeric_carbon]
		for oxygen in graph.neighbors(anomeric_carbon):
			if elements[oxygen] != "O" or oxygen in rings.get(donor, set()):
				continue
			for carbon in graph.neighbors(oxygen):
				if carbon == anomeric_carbon or elements[carbon] != "C" or res_ids[carbon] == donor:
					continue
				reference = _psi_reference(graph, names, res_ids, elements, rings, carbon, oxygen)
				linkages.append(Linkage(
					"glycosidic",
					donor,
					res_ids[carbon],
					phi=(ring_oxygen, anomeric_carbon, oxygen, carbon),
					psi=(anomeric_carbon, oxygen, carbon, reference) if reference is not None else None,
					omega=_omega(graph, names, res_ids, elements, rings, carbon, oxygen, reference),
					acceptor_carbon=carbon,
				))
				glycosidic_bonds.add((oxygen, carbon))

	# A group that is not a sugar (sulfate, phosphate, methyl, acetyl ...) on
	# the oxygen of a carbon outside the ring of a sugar: the omega of that
	# carbon, listed apart from the glycosidic torsions.
	for residue, ring in rings.items():
		carbons = sorted((n for n in graph.nodes() if res_ids[n] == residue), key=node_order.get)
		for carbon in carbons:
			if elements[carbon] != "C" or carbon in ring:
				continue
			for oxygen in graph.neighbors(carbon):
				if elements[oxygen] != "O" or (oxygen, carbon) in glycosidic_bonds:
					continue
				group = {
					res_ids[n] for n in graph.neighbors(oxygen)
					if n != carbon and res_ids[n] != residue
				}
				if res_ids[oxygen] != residue:
					group.add(res_ids[oxygen])
				if not group:
					continue
				reference = _psi_reference(graph, names, res_ids, elements, rings, carbon, oxygen)
				omega = _omega(graph, names, res_ids, elements, rings, carbon, oxygen, reference)
				if omega is not None:
					linkages.append(Linkage(
						"substituent", min(group), residue, omega=omega, acceptor_carbon=carbon,
					))
	return linkages


def find_linkages(atoms_df, bonds_df) -> List[Linkage]:
	"""Glycosidic linkages and substituents of a structure, with their torsions (``TORSION_CONVENTIONS``)."""
	graph, names, res_ids, elements = _heavy_atom_graph(atoms_df, bonds_df)
	return _find_linkages_in_graph(graph, names, res_ids, elements)


def get_glycan_torsions_enhanced(atoms_df, bonds_df) -> List[Torsion]:
	"""The torsions of a glycan: those of its linkages, then every other rotatable bond.

	First phi and psi of every glycosidic linkage, then the omegas (of the
	linkages whose acceptor carbon lies outside the ring, and of substituents
	on such a carbon), all as ``TORSION_CONVENTIONS`` defines them (IUPAC):

	- phi: O5'-C1'-Ox-Cx (O4'-C1' for an aldofuranose, O6'-C2' for sialic
	  acids and other 2-ketopyranoses, O5'-C2' for 2-ketofuranoses)
	- psi: C1'-Ox-Cx-C(x-1) (C2' for 2-ketose donors; O5 for a 1-1 linkage,
	  C2 for a 2-1 linkage to the C1 of a ketose)
	- omega: O5-C5-C6-O6 for 1-6, and O-C(x-1)-Cx-Ox for any acceptor carbon
	  outside the ring (O4-C4-C5-O5 for 1-5 to Araf, O7-C7-C8-O8 for 2-8,
	  O8-C8-C9-O9 for 2-9, O5-C2-C1-O1 for 2-1 to Fruf)

	Then every other single bond outside a ring between two non-terminal heavy
	atoms, with an oxygen at each end where there is one.
	"""
	_require_rdkit()
	G, atom_names, res_ids, elements = _heavy_atom_graph(atoms_df, bonds_df)
	atom_count = len(atoms_df)
	linkages = _find_linkages_in_graph(G, atom_names, res_ids, elements)
	node_order = {node: position for position, node in enumerate(G.nodes())}

	torsions: List[Torsion] = []
	for linkage in linkages:
		if linkage.kind == "glycosidic":
			torsions.extend(t for t in (linkage.phi, linkage.psi) if t is not None)
	with_omega = sorted(
		(linkage for linkage in linkages if linkage.omega is not None),
		key=lambda linkage: node_order.get(linkage.acceptor_carbon, 0),
	)
	torsions.extend(linkage.omega for linkage in with_omega)

	# Method 3: Additional rotatable bonds (for completeness)
	# Look for other single bonds that might be rotatable
	try:
		cycles = list(nx.cycle_basis(G))
		cycle_edges = set()
		for cycle in cycles:
			for i in range(len(cycle)):
				edge = tuple(sorted([cycle[i], cycle[(i+1) % len(cycle)]]))
				cycle_edges.add(edge)
	except:
		cycle_edges = set()
	
	for edge in G.edges():
		edge_sorted = tuple(sorted(edge))
		if edge_sorted not in cycle_edges:  # Not a ring bond
			node1, node2 = edge
			
			# Skip if this bond was already covered by phi/psi/omega
			already_covered = False
			for existing_torsion in torsions:
				if (node1 in existing_torsion[1:3] and node2 in existing_torsion[1:3]):
					already_covered = True
					break
			
			if not already_covered and G.degree(node1) > 1 and G.degree(node2) > 1:
				node1_neighbors = [n for n in G.neighbors(node1) if n != node2]
				node2_neighbors = [n for n in G.neighbors(node2) if n != node1]
				
				if node1_neighbors and node2_neighbors:
					# Prefer oxygen atoms at the ends
					end1_oxygens = [n for n in node1_neighbors if elements[n] == "O"]
					end2_oxygens = [n for n in node2_neighbors if elements[n] == "O"]
					
					end1_atom = end1_oxygens[0] if end1_oxygens else node1_neighbors[0]
					end2_atom = end2_oxygens[0] if end2_oxygens else node2_neighbors[0]
					
					torsions.append((end1_atom, node1, node2, end2_atom))
	
	# Remove duplicates and filter invalid torsions
	unique_torsions = []
	seen = set()
	
	for torsion in torsions:
		# Check if all atoms are valid
		if all(0 <= atom < atom_count for atom in torsion):
			# Check if all atoms are different
			if len(set(torsion)) == 4:
				# Normalize torsion (handle both orientations)
				normalized = tuple(sorted([torsion, torsion[::-1]])[0])
				if normalized not in seen:
					seen.add(normalized)
					unique_torsions.append(torsion)
	
	return unique_torsions


def save_torparts_npz(torsion_list: List[Torsion], bonds_df, atoms_df, output_path: str) -> None:
	"""Save torsion parts data to npz file with same structure as tfindr.py.
	
	Args:
		torsion_list: List of torsion angle definitions (4-tuples of atom indices)
		bonds_df: DataFrame containing bond information with atom pairs
		atoms_df: DataFrame containing atom information 
		output_path: Path where to save the torparts.npz file
	"""
	if nx is None:
		raise ImportError("NetworkX is required for torparts generation. Install networkx.")
	
	# Convert bonds to list of tuples
	bonds_list = [(int(a), int(b)) for a, b in bonds_df.iter_rows()]
	
	# Generate torsion parts array
	anotherlist = []
	for torsion in torsion_list:
		G = nx.Graph()
		G.add_edges_from(bonds_list)
		node1, node2 = torsion[1], torsion[2]  # middle two atoms of torsion
		try:
			G.remove_edge(node1, node2)
			# Create boolean array marking which atoms are on one side of the bond
			arr = np.ones(len(atoms_df), dtype=bool)
			for node in nx.node_connected_component(G, node1):
				arr[node-1] = False  # Convert from 1-based to 0-based indexing
			anotherlist.append(arr)
		except Exception:
			# Create default array if edge removal fails
			arr = np.ones(len(atoms_df), dtype=bool)
			anotherlist.append(arr)
	
	# Save torparts.npz file with same structure as tfindr.py
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	np.savez_compressed(output_path, a=torsion_list, b=anotherlist, allow_pickle=True)


def _torsion_plot_table(data, torsion_cols: List[str], display_names: Dict[str, str]):
	"""Long table (Value, Torsion, Frame) of the ridge plot: one row per torsion and frame.

	This is the table the plot was always drawn from, built from whole columns.
	It used to be filled value by value from ``data.iterrows()``, one Python
	call per value. ``data.to_numpy()`` holds exactly the row values iterrows
	yields (all columns in their common dtype), so the values are the same.
	The torsion names are a categorical column whose categories are in order
	of first appearance, the order seaborn derives from a column of strings:
	the grid has the same rows and colours, and selecting the frames of one
	torsion compares small integers instead of every string of the table
	(seaborn does that for each pair of torsions, three times per figure).
	Mixed (object) tables and duplicate column names take the former loop.
	"""
	import pandas as pd

	matrix = data.to_numpy()
	if matrix.dtype == object or len(set(data.columns)) != len(data.columns):
		plot_data = []
		for col in torsion_cols:
			for _, row in data.iterrows():
				plot_data.append({
					'Value': row[col],
					'Torsion': display_names[col],
					'Frame': row['frame']
				})
		return pd.DataFrame(plot_data)
	position = {name: index for index, name in enumerate(data.columns)}
	names = [display_names[col] for col in torsion_cols]
	return pd.DataFrame({
		'Value': np.concatenate([matrix[:, position[col]] for col in torsion_cols]),
		'Torsion': pd.Categorical.from_codes(
			np.repeat(
				[list(dict.fromkeys(names)).index(name) for name in names], len(data)
			),
			categories=list(dict.fromkeys(names)),
		),
		'Frame': np.tile(matrix[:, position['frame']], len(torsion_cols)),
	})


def plot_torsion_distribution(
	torsion_csv_path: str,
	output_path: str,
	cluster_representatives: List[int] = None,
	level: int = None
) -> None:
	"""Plot torsion angle distributions for a clustering level.
	
	Creates a ridge plot showing the distribution of torsion angles with 
	cluster representatives marked as vertical lines.
	
	Parameters:
		torsion_csv_path: Path to torsions.parquet or legacy torsions.csv file
		output_path: Path to save the distribution plot (SVG format)
		cluster_representatives: List of frame indices for cluster representatives
		level: Clustering level number for plot title
	"""
	try:
		import pandas as pd
		import seaborn as sns
		import matplotlib.pyplot as plt
		from matplotlib.lines import Line2D
		import json
	except ImportError as e:
		raise ImportError("Plotting requires pandas, seaborn, and matplotlib") from e
	
	storage = get_storage_manager()
	if not (os.path.exists(torsion_csv_path) or storage.exists(torsion_csv_path)):
		raise FileNotFoundError(f"Torsion table file not found: {torsion_csv_path}")
	
	# Read torsion data
	if torsion_csv_path.endswith(".parquet"):
		if os.path.exists(torsion_csv_path):
			data = pd.read_parquet(torsion_csv_path)
		else:
			import io
			data = pd.read_parquet(io.BytesIO(storage.read_binary(torsion_csv_path)))
	elif os.path.exists(torsion_csv_path):
		data = pd.read_csv(torsion_csv_path)
	else:
		with storage.open(torsion_csv_path, 'r') as fh:
			data = pd.read_csv(fh)
	if 'frame' not in data.columns:
		raise ValueError("Torsions table must contain 'frame' column")
	
	# Try to read info.json to get glycosidic torsion information
	info_json_path = os.path.join(os.path.dirname(torsion_csv_path), 'info.json')
	glycosidic_original_names = set()
	name_mapping = {}
	
	if storage.exists(info_json_path):
		try:
			with storage.open(info_json_path, 'r') as f:
				info_data = json.load(f)
			
			# Get glycosidic torsion names from info.json
			if 'torsions' in info_data and 'glycosidic' in info_data['torsions']:
				for glycosidic_torsion in info_data['torsions']['glycosidic']:
					original_name = glycosidic_torsion['original_name']
					glycosidic_name = glycosidic_torsion['glycosidic_name']
					glycosidic_original_names.add(original_name)
					name_mapping[original_name] = glycosidic_name
		except (json.JSONDecodeError, KeyError) as e:
			print(f"Warning: Could not read glycosidic info from {info_json_path}: {e}")
	
	# Get torsion columns - only glycosidic linkages
	torsion_cols = []
	for col in data.columns:
		if col != 'frame':
			# If we have glycosidic names from info.json, use those original names
			if glycosidic_original_names and col in glycosidic_original_names:
				torsion_cols.append(col)
			# Fallback: check for phi, psi, omega patterns
			elif not glycosidic_original_names and any(angle in col for angle in ['phi', 'psi', 'omega']):
				torsion_cols.append(col)
			# Recognize structurally explicit phi/psi/omega labels when metadata is
			# unavailable. Residue-local C6 torsions deliberately do not match.
			elif col not in torsion_cols and not classify_glycosidic_from_name(col).startswith('other_'):
				torsion_cols.append(col)
	
	if not torsion_cols:
		# Never substitute residue-local torsions into a glycosidic distribution
		# plot. Emit a complete placeholder so static validation still succeeds.
		os.makedirs(os.path.dirname(output_path), exist_ok=True)
		with open(output_path, "w", encoding="utf-8") as handle:
			handle.write(
				'<svg xmlns="http://www.w3.org/2000/svg" width="720" height="120" '
				'viewBox="0 0 720 120" role="img" aria-label="No glycosidic torsions">'
				'<rect width="720" height="120" fill="white"/>'
				'<text x="24" y="68" font-family="sans-serif" font-size="20" fill="#444">'
				'No glycosidic torsions for this structure.'
				'</text></svg>\n'
			)
		print(f"No-glycosidic-torsion placeholder saved to {output_path}")
		return
	
	# Replace Greek letter names for better display
	display_names = {}
	for col in torsion_cols:
		# Use mapped name if available, otherwise default Greek letter mapping
		if col in name_mapping:
			display_name = name_mapping[col].replace('phi', 'ϕ').replace('psi', 'ψ').replace('omega', 'ω')
		else:
			display_name = col.replace('phi', 'ϕ').replace('psi', 'ψ').replace('omega', 'ω')
		display_names[col] = display_name
	
	# Prepare data for plotting
	df = _torsion_plot_table(data, torsion_cols, display_names)
	
	# Create the plot
	num_torsions = len(torsion_cols)
	facet_height = max(1, 3 - num_torsions * 0.25)
	facet_aspect = 10 if num_torsions <= 5 else 20
	
	# Set up colors for representative lines
	line_colors = ["#1B9C75", "#D55D02", "#746FB1", "#E12886", "#939242"]
	
	# Create ridge plot
	g = sns.FacetGrid(df, row="Torsion", hue="Torsion",
					 aspect=facet_aspect, height=facet_height,
					 palette=sns.cubehelix_palette(len(torsion_cols), rot=-.25, light=.7),
					 xlim=(-240, 200))
	
	# Add KDE plots
	g.map_dataframe(sns.kdeplot, "Value", bw_adjust=1, clip_on=False,
				   fill=True, alpha=1, linewidth=4, multiple='stack')
	
	# Add vertical lines for cluster representatives
	if cluster_representatives:
		for ax_idx, (torsion_col, ax) in enumerate(zip(torsion_cols, g.axes.flat)):
			# Get representative values for this torsion
			rep_data = data[data['frame'].isin(cluster_representatives)]
			rep_values = rep_data[torsion_col].values
			
			# Add colored lines for each representative
			for i, rep_val in enumerate(rep_values):
				if i < len(line_colors):
					ax.axvline(rep_val, color=line_colors[i], linestyle="-", 
							  linewidth=4, ymax=0.25)
	
	# Add labels and formatting
	def label_func(x, color, label):
		ax = plt.gca()
		ax.text(0, .06, label, fontweight="bold", fontsize=32, color=color,
			   ha="left", va="center", transform=ax.transAxes)
	
	g.map(label_func, "Value")
	g.refline(y=0, linewidth=4, linestyle="-", color=None, clip_on=False)
	
	# Format the plot
	g.figure.subplots_adjust(hspace=-.75)
	g.set_titles("")
	g.set(yticks=[], 
		  xticks=[-180,-150,-120,-90,-60,-30,0,30,60,90,120,150,180],
		  ylabel="")
	
	for ax in g.axes.flat:
		ax.tick_params(axis='x', labelsize=24)
	g.despine(bottom=True, left=True)
	
	# Save the plot
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	plt.savefig(output_path, transparent=True, dpi=450, bbox_inches='tight')
	plt.close(g.figure)
	
	print(f"Torsion distribution plot saved to {output_path}")
