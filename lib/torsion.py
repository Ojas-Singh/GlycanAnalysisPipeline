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

from typing import Iterable, List, Sequence, Tuple
import os

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

__all__ = [
	"Torsion",
	"get_movable_torsions",
	"get_torsion_values",
	"get_torsion_values_batch",
	"set_torsion_values",
	"generate_wiggles",
	"get_glycan_torsions_enhanced",
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



def get_glycan_torsions_enhanced(atoms_df, bonds_df) -> List[Torsion]:
	"""Enhanced glycan torsion detection using domain knowledge.
	
	This method implements the correct phi, psi, and omega definitions:
	- phi: O5-C1-O-Cx' (glycosidic linkage angle)
	- psi: C1-O-Cx'-Cx+1' (continuation angle)  
	- omega: C4-C5-C6-O6 (side chain angle)
	"""
	_require_rdkit()
	
	try:
		import networkx as nx
	except ImportError:
		raise ImportError("NetworkX is required for graph-based torsion detection.")
	
	# Build graph - handle both 0-based and 1-based indexing
	G = nx.Graph()
	atom_count = len(atoms_df)
	
	# Create index mapping (bonds might use 1-based, arrays use 0-based)
	for a, b in bonds_df.iter_rows():
		a_idx = int(a) if int(a) < atom_count else int(a) - 1
		b_idx = int(b) if int(b) < atom_count else int(b) - 1
		if 0 <= a_idx < atom_count and 0 <= b_idx < atom_count:
			G.add_edge(a_idx, b_idx)
	
	# Remove hydrogens
	elements = atoms_df["element"].to_list()
	hydrogen_nodes = [i for i, element in enumerate(elements) if element == "H"]
	G.remove_nodes_from(hydrogen_nodes)
	
	# Get atom information
	atom_names = atoms_df["name"].to_list() if "name" in atoms_df.columns else [f"X{i}" for i in range(len(atoms_df))]
	res_ids = atoms_df["res_id"].to_list() if "res_id" in atoms_df.columns else [1] * len(atoms_df)
	
	torsions = []
	
	# Method 1: Find glycosidic linkage torsions (phi and psi)
	for node in G.nodes():
		if node >= len(atom_names):
			continue
			
		atom_name = atom_names[node]
		
		# Look for C1 atoms (anomeric carbons) - these are key for glycosidic linkages
		if atom_name == "C1":
			neighbors = list(G.neighbors(node))
			
			# Find O5 (ring oxygen) and glycosidic oxygen
			o5_node = None
			glycosidic_oxygens = []
			
			for n in neighbors:
				if n < len(atom_names):
					if atom_names[n] == "O5":
						o5_node = n
					elif elements[n] == "O" and atom_names[n] != "O5":
						# Check if this connects to different residue
						if n < len(res_ids) and res_ids[n] != res_ids[node]:
							glycosidic_oxygens.append(n)
						else:
							# Check neighbors of oxygen for different residue
							for on in G.neighbors(n):
								if on < len(res_ids) and res_ids[on] != res_ids[node]:
									glycosidic_oxygens.append(n)
									break
			
			# Build phi and psi torsions for each glycosidic oxygen
			for glyc_o in glycosidic_oxygens:
				# Find the carbon on the other side of glycosidic oxygen
				other_carbons = [n for n in G.neighbors(glyc_o) 
				               if n != node and elements[n] == "C"]
				
				if other_carbons and o5_node is not None:
					other_c = other_carbons[0]  # Cx'
					
					# PHI torsion: O5-C1-O-Cx'
					phi_torsion = (o5_node, node, glyc_o, other_c)
					torsions.append(phi_torsion)
					
					# Find next carbon for PSI torsion (Cx+1')
					next_carbons = [n for n in G.neighbors(other_c) 
					              if n != glyc_o and elements[n] == "C"]
					
					# Prefer oxygen over carbon for fourth atom if available
					next_atoms = [n for n in G.neighbors(other_c) 
					            if n != glyc_o and elements[n] == "O"]
					if not next_atoms:
						next_atoms = next_carbons
					
					if next_atoms:
						next_atom = next_atoms[0]  # Cx+1' or Ox'
						# PSI torsion: C1-O-Cx'-Cx+1' (or C1-O-Cx'-Ox')
						psi_torsion = (node, glyc_o, other_c, next_atom)
						torsions.append(psi_torsion)
	
	# Method 2: Find omega torsions (C4-C5-C6-O6)
	for node in G.nodes():
		if node >= len(atom_names):
			continue
			
		atom_name = atom_names[node]
		
		if atom_name == "C6":
			neighbors = list(G.neighbors(node))
			
			# Find C5 and O6 neighbors
			c5_node = None
			o6_node = None
			
			for n in neighbors:
				if n < len(atom_names):
					if atom_names[n] == "C5":
						c5_node = n
					elif atom_names[n] == "O6":
						o6_node = n
			
			if c5_node is not None and o6_node is not None:
				# Find C4 (neighbor of C5)
				c4_candidates = [n for n in G.neighbors(c5_node) 
				               if n < len(atom_names) and atom_names[n] == "C4"]
				
				if c4_candidates:
					c4_node = c4_candidates[0]
					# OMEGA torsion: C4-C5-C6-O6
					torsions.append((c4_node, c5_node, node, o6_node))
	
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
		torsion_csv_path: Path to torsions.csv file
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
	
	if not os.path.exists(torsion_csv_path):
		raise FileNotFoundError(f"Torsion CSV file not found: {torsion_csv_path}")
	
	# Read torsion data
	data = pd.read_csv(torsion_csv_path)
	if 'frame' not in data.columns:
		raise ValueError("Torsions CSV must contain 'frame' column")
	
	# Try to read info.json to get glycosidic torsion information
	info_json_path = os.path.join(os.path.dirname(torsion_csv_path), 'info.json')
	glycosidic_original_names = set()
	name_mapping = {}
	
	storage = get_storage_manager()
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
	
	if not torsion_cols:
		raise ValueError("No glycosidic torsion columns found in CSV")
	
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
	plot_data = []
	for col in torsion_cols:
		for _, row in data.iterrows():
			plot_data.append({
				'Value': row[col],
				'Torsion': display_names[col],
				'Frame': row['frame']
			})
	
	df = pd.DataFrame(plot_data)
	
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

