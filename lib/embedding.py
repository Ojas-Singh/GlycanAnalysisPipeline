"""Embedding utilities (PCA) for conformational landscapes.

The main entry point here is `pca_conformation_landscape`, which accepts the
2D numpy array produced by `lib.glypdbio.get_conformation_landscape` and
returns a Polars DataFrame containing principal component coordinates for
each frame.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl

try:
	from lib.torsion import circular_stats
except ImportError:
	circular_stats = None

try:
	from sklearn.decomposition import PCA as _SkPCA  # type: ignore
	_HAS_SKLEARN = True
except Exception:  # pragma: no cover - fallback if sklearn missing
	_HAS_SKLEARN = False


# Optional clustering / KDE dependencies. If missing, clustering functions will raise
try:
	from typing import List, Dict, Tuple, Any
	import logging
	from sklearn.neighbors import KernelDensity
	from sklearn.model_selection import GridSearchCV
	from sklearn.preprocessing import StandardScaler
	from scipy.optimize import minimize
	import matplotlib.pyplot as _plt
	_HAS_CLUSTERING = True
	logger = logging.getLogger(__name__)

except Exception:
	_HAS_CLUSTERING = False
	# lightweight placeholders
	KernelDensity = None  # type: ignore
	GridSearchCV = None  # type: ignore
	StandardScaler = None  # type: ignore
	silhouette_score = None  # type: ignore
	calinski_harabasz_score = None  # type: ignore
	GaussianMixture = None  # type: ignore
	minimize = None  # type: ignore
	linkage = None  # type: ignore
	fcluster = None  # type: ignore
	dendrogram = None  # type: ignore
	_plt = None  # type: ignore

# Optional imports for entropy estimation (lightweight fallbacks provided if missing)
try:  # pragma: no cover - optional dependency handling
	from sklearn.neighbors import NearestNeighbors  # type: ignore
	_HAS_SKLEARN_NEIGHBORS = True
except Exception:  # pragma: no cover
	NearestNeighbors = None  # type: ignore
	_HAS_SKLEARN_NEIGHBORS = False

try:  # pragma: no cover
	from scipy.special import digamma as _digamma  # type: ignore
	_HAS_DIGAMMA = True
except Exception:  # pragma: no cover
	_HAS_DIGAMMA = False
	_digamma = None  # type: ignore

import math
import os
import json


def _read_parquet_robust(path: str):
	"""Try pyarrow first (avoids glob issues), then fall back to Polars."""
	if not os.path.exists(path):
		raise FileNotFoundError(path)
	try:
		import pyarrow.parquet as pq  # type: ignore
		table = pq.read_table(path)
		return pl.from_arrow(table)
	except Exception:
		try:
			import glob
			return pl.read_parquet(glob.escape(path))
		except Exception:
			return pl.read_parquet(path)

def _read_csv_robust(path: str, **kwargs):
	"""Read CSV robustly, handling special characters in paths (like square brackets)."""
	if not os.path.exists(path):
		raise FileNotFoundError(path)
	try:
		# Try with glob.escape first to handle special characters
		import glob
		return pl.read_csv(glob.escape(path), **kwargs)
	except Exception:
		# Fall back to direct path
		return pl.read_csv(path, **kwargs)

def _digamma_approx(x: np.ndarray | float) -> np.ndarray | float:
	"""Simple digamma approximation ln(x) - 1/(2x) for x>0.

	Used only if scipy is unavailable. Accuracy is sufficient for kNN entropy constants.
	"""
	return np.log(x) - 1.0 / (2.0 * x)

def _safe_digamma(x: np.ndarray | float) -> np.ndarray | float:
	if _HAS_DIGAMMA:
		return _digamma(x)  # type: ignore
	return _digamma_approx(x)


def _pca_svd(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
	"""Lightweight PCA via SVD (fallback when scikit-learn unavailable).

	Returns (scores, explained_variance_ratio).
	"""
	# Center
	Xc = X - X.mean(axis=0, keepdims=True)
	U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
	# Eigenvalues of covariance = (S^2)/(n_samples-1)
	eigvals = (S ** 2) / (X.shape[0] - 1)
	total_var = eigvals.sum()
	scores = U[:, :n_components] * S[:n_components]
	evr = eigvals[:n_components] / total_var if total_var > 0 else np.zeros(n_components)
	return scores, evr


def pca_conformation_landscape(
	landscape: np.ndarray,
	n_components: int = 20,
	scale: bool = False,
	center: bool = True,
	per_pair_scale: bool = False,
	use_sklearn: bool = True,
	frame_col: str = "frame",
	prefix: str = "PC",
) -> pl.DataFrame:
	"""Perform PCA on a conformation landscape matrix and return a Polars DataFrame.

	Parameters
	----------
	landscape : np.ndarray
		2D array of shape (n_frames, n_features) from `get_conformation_landscape`.
	n_components : int, default 2
		Number of principal components to compute.
	scale : bool, default False
		If True, standardize features to unit variance after centering.
	center : bool, default True
		If True, subtract feature means.
	per_pair_scale : bool, default False
		If True, normalize each atom pair distance to unit variance independently.
		This is applied before the general 'scale' parameter if both are True.
	use_sklearn : bool, default True
		If False or scikit-learn not available, falls back to a lightweight SVD PCA.
	frame_col : str, default 'frame'
		Name of the frame index column in the output.
	prefix : str, default 'PC'
		Column name prefix for components (e.g., PC1, PC2, ...).

	Returns
	-------
	pl.DataFrame
		Columns: frame_col, f"{prefix}1"..f"{prefix}n", and explained variance ratios
		as a struct column 'explained_variance_ratio'.
	"""
	X = np.asarray(landscape, dtype=float)
	if X.ndim != 2:
		raise ValueError("landscape must be 2D (n_frames, n_features)")
	n_frames, n_features = X.shape
	if n_frames == 0:
		return pl.DataFrame({frame_col: [], f"{prefix}1": []})

	# Adjust n_components if larger than allowable rank
	max_components = min(n_frames, n_features)
	if n_components > max_components:
		n_components = max_components

	X_proc = X.copy()
	
	# Apply per-pair scaling first (before centering and general scaling)
	if per_pair_scale:
		# Normalize each feature (atom pair distance) to unit variance
		pair_std = X_proc.std(axis=0, ddof=1)
		# Avoid division by zero for constant features
		pair_std[pair_std == 0] = 1.0
		X_proc = X_proc / pair_std[np.newaxis, :]
		
	if center:
		X_proc -= X_proc.mean(axis=0, keepdims=True)
	if scale:
		std = X_proc.std(axis=0, ddof=1, keepdims=True)
		std[std == 0] = 1.0
		X_proc /= std

	if use_sklearn and _HAS_SKLEARN:
		pca = _SkPCA(n_components=n_components, svd_solver="auto")
		scores = pca.fit_transform(X_proc)
		evr = pca.explained_variance_ratio_
	else:
		scores, evr = _pca_svd(X_proc, n_components)

	# Build columns
	data = {frame_col: np.arange(n_frames, dtype=int)}
	for i in range(n_components):
		data[f"{prefix}{i+1}"] = scores[:, i]
	# Include explained variance ratio as a separate struct column for clarity
	data["explained_variance_ratio"] = [evr] * n_frames
	return pl.DataFrame(data)


def gmm_optimize_silhouette(data: np.ndarray, range_clusters: tuple = (3, 8)) -> tuple[np.ndarray, int]:
	"""Find optimal GMM clustering using silhouette score."""
	if not _HAS_CLUSTERING:
		raise ImportError("Clustering dependencies not available")
	from sklearn.mixture import GaussianMixture
	from sklearn.metrics import silhouette_score
	
	best_score = -1
	best_labels = None
	best_n = 0
	
	for n in range(range_clusters[0], range_clusters[1] + 1):
		gmm = GaussianMixture(n_components=n, random_state=42)
		labels = gmm.fit_predict(data)
		
		if len(np.unique(labels)) > 1:
			score = silhouette_score(data, labels)
			logger.info(f"GMM with {n} clusters: silhouette={score:.4f}")
			
			if score > best_score:
				best_score = score
				best_labels = labels
				best_n = n
	
	logger.info(f"Best GMM: {best_n} clusters with silhouette={best_score:.4f}")
	return best_labels, best_n


def gmm_optimize_glycosidic_deviation(
	data: np.ndarray,
	torsion_df: pl.DataFrame,
	glycosidic_info: list,
	max_deviation_threshold: float = 20.0,
	max_clusters: int = 100,
) -> tuple[np.ndarray, int]:
	"""Find GMM clustering that minimizes glycosidic torsion deviation within clusters.

	Parameters
	----------
	stopping_criterion : str, default 'max'
		Criterion to stop clustering:
		- 'max': Stop when the maximum glycosidic deviation across all clusters <= threshold
		- 'average': Stop when the average glycosidic deviation across all torsions <= threshold
		- 'average_max': Stop when the average of per-cluster max deviations <= threshold
	"""
	if not _HAS_CLUSTERING:
		raise ImportError("Clustering dependencies not available")
	from sklearn.mixture import GaussianMixture

	# Extract glycosidic torsion column indices
	glycosidic_indices = [info['index'] for info in glycosidic_info]
	
	# Log the glycosidic torsions being used
	glycosidic_names = [info.get('original_name', f"torsion_{info['index']}") for info in glycosidic_info]
	logger.info(f"Using {len(glycosidic_indices)} glycosidic torsions: {glycosidic_names}")

	if circular_stats is None:
		raise ImportError("circular_stats is required for glycosidic-deviation checks but was not found")

	# Helper: compute per-cluster, per-glycosidic-torsion std (degrees)
	def compute_cluster_torsion_stds(labels):
		cluster_torsion_stds = []  # List of lists: [ [std1, std2, ...], ... ]
		all_stds = []  # flat list for global average
		for cluster_id in np.unique(labels):
			mask = labels == cluster_id
			if np.sum(mask) <= 1:
				cluster_torsion_stds.append([0.0 for _ in glycosidic_indices])
				all_stds.extend([0.0 for _ in glycosidic_indices])
				continue
			mask_series = pl.Series(mask)
			cluster_rows = torsion_df.filter(mask_series)
			torsion_stds = []
			for idx in glycosidic_indices:
				col_name = torsion_df.columns[idx + 1]  # +1 because 'frame' is first
				if len(cluster_rows) == 0:
					torsion_stds.append(0.0)
					continue
				angle_values = cluster_rows[col_name].to_numpy()
				stats = circular_stats(angle_values)
				std_val = float(stats.get('circular_std_deg', 0.0))
				torsion_stds.append(std_val)
				all_stds.append(std_val)
			cluster_torsion_stds.append(torsion_stds)
		return cluster_torsion_stds, all_stds

	# Try increasing number of clusters until stopping criterion is met
	for n in range(1, max_clusters + 1):
		gmm = GaussianMixture(n_components=n, random_state=42)
		labels = gmm.fit_predict(data)
		# Compute per-cluster, per-glycosidic-torsion stds
		cluster_torsion_stds, all_stds = compute_cluster_torsion_stds(labels)
		# Log per-cluster, per-torsion stds
		# for cid, torsion_stds in enumerate(cluster_torsion_stds):
		# 	logger.info(f"GMM n={n} cluster={cid}: glycosidic torsion stds = {[f'{std:.2f}' for std in torsion_stds]}")
		# Compute average of all stds (across all clusters and torsions)
		avg_deviation = float(np.mean(all_stds)) if all_stds else 0.0
		logger.info(f"GMM n={n}: average glycosidic torsion std = {avg_deviation:.2f}° (across all clusters and torsions)")
		# Stopping criterion: average of all stds
		criterion_value = avg_deviation
		criterion_name = 'average_deviation'
		logger.info(f"GMM n={n}: {criterion_name} = {criterion_value:.2f}° (threshold: {max_deviation_threshold:.1f}°)")
		if criterion_value <= max_deviation_threshold:
			logger.info(f"Found solution with {n} clusters: {criterion_name} <= {max_deviation_threshold:.1f}°")
			return labels, n

	# If no solution meets criterion, fallback to max clusters
	final_gmm = GaussianMixture(n_components=max_clusters, random_state=42)
	labels = final_gmm.fit_predict(data)
	return labels, max_clusters


def kcenter_coverage_clustering(
	data: np.ndarray,
	max_clusters: int = 128,
	coverage_threshold: float = 0.95,
	min_coverage_improvement: float = 0.01,
	seed: int = 42
) -> tuple[np.ndarray, int]:
	"""Coverage-based farthest-point (k-center) clustering procedure.
	
	This algorithm iteratively selects cluster centers that maximize coverage of the
	data space, using a greedy farthest-point strategy. It stops when either:
	1. Maximum number of clusters is reached
	2. Coverage threshold is achieved  
	3. Coverage improvement becomes negligible
	
	Parameters
	----------
	data : np.ndarray
		Input data of shape (n_samples, n_features)
	max_clusters : int, default 128
		Maximum number of clusters to create
	coverage_threshold : float, default 0.95
		Stop when this fraction of points are within reasonable distance of centers
	min_coverage_improvement : float, default 0.01
		Stop when adding a new center improves coverage by less than this amount
	seed : int, default 42
		Random seed for reproducible center selection
		
	Returns
	-------
	labels : np.ndarray
		Cluster labels for each data point
	n_clusters : int
		Number of clusters found
	"""
	np.random.seed(seed)
	n_samples, n_features = data.shape
	
	if n_samples <= max_clusters:
		# If we have fewer points than max clusters, each point is its own cluster
		logger.info(f"k-center: {n_samples} samples <= {max_clusters} max clusters, using identity clustering")
		return np.arange(n_samples), n_samples
	
	# Standardize data for distance calculations
	if _HAS_CLUSTERING:
		scaler = StandardScaler()
		data_scaled = scaler.fit_transform(data)
	else:
		# Simple standardization if sklearn not available
		data_scaled = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
	
	# Initialize with random first center
	centers = []
	center_indices = []
	first_idx = np.random.randint(0, n_samples)
	centers.append(data_scaled[first_idx])
	center_indices.append(first_idx)
	
	logger.info(f"k-center: Starting with random center at index {first_idx}")
	
	# Greedy farthest-point selection until we have `max_clusters` centers.
	# This ensures we always return the requested number of representatives
	# (unless n_samples <= max_clusters which is handled above).
	for k in range(1, max_clusters):
		# Find distances from each point to nearest existing center
		distances_to_centers = np.full(n_samples, np.inf)
		for center in centers:
			distances = np.linalg.norm(data_scaled - center, axis=1)
			distances_to_centers = np.minimum(distances_to_centers, distances)

		# Select next center as the point farthest from all existing centers
		farthest_idx = int(np.argmax(distances_to_centers))
		# If the farthest point is already selected (duplicate), pick a random remaining index
		if farthest_idx in center_indices:
			remaining = [i for i in range(n_samples) if i not in center_indices]
			if not remaining:
				break
			farthest_idx = int(np.random.choice(remaining))

		centers.append(data_scaled[farthest_idx])
		center_indices.append(farthest_idx)
	
	n_clusters = len(centers)
	logger.info(f"k-center: Final clustering with {n_clusters} centers")
	
	# Assign each point to nearest center
	labels = np.zeros(n_samples, dtype=int)
	for i, point in enumerate(data_scaled):
		distances = [np.linalg.norm(point - center) for center in centers]
		labels[i] = np.argmin(distances)
	
	# # Log cluster sizes
	# unique_labels, counts = np.unique(labels, return_counts=True)
	# size_info = ", ".join([f"cluster {label}: {count} points" for label, count in zip(unique_labels, counts)])
	# logger.info(f"k-center: Cluster sizes: {size_info}")
	
	return labels, n_clusters


def create_cluster_info(labels: np.ndarray, pca_df: pl.DataFrame, n_clusters: int) -> list:
	"""Create cluster info compatible with existing framework using KDE representatives.
	
	Clusters are sorted by decreasing size, with cluster ID 0 being the largest cluster,
	cluster ID 1 being the second largest, and so on.
	"""
	reps = []
	
	# First, collect cluster sizes and sort by decreasing size
	cluster_sizes = []
	for cluster_id in range(n_clusters):
		indices = np.where(labels == cluster_id)[0]
		cluster_sizes.append((cluster_id, len(indices)))
	
	# Sort by size (descending) and create mapping from old ID to new ID
	cluster_sizes.sort(key=lambda x: x[1], reverse=True)
	old_to_new_id = {old_id: new_id for new_id, (old_id, _) in enumerate(cluster_sizes)}
	
	# Extract PCA component columns for KDE analysis
	pc_cols = [col for col in pca_df.columns if col.startswith('PC') and col[2:].isdigit()]
	if not pc_cols:
		# Fallback to original simple approach if no PC columns found
		for new_cluster_id, (old_cluster_id, _) in enumerate(cluster_sizes):
			indices = np.where(labels == old_cluster_id)[0]
			if len(indices) > 0:
				cluster_size_pct = 100.0 * len(indices) / len(labels)
				cluster_frames = pca_df.filter(pl.col("frame").is_in(indices.tolist()))
				rep_idx = int(cluster_frames["frame"][0]) if len(cluster_frames) > 0 else int(indices[0])
				reps.append({
					'cluster_id': int(new_cluster_id),
					'representative_idx': rep_idx,
					'cluster_size_pct': float(cluster_size_pct),
					'n_points': len(indices),
					'n_clusters': int(n_clusters),
					'members': indices.tolist()
				})
		return reps
	
	# Extract PCA data for KDE analysis
	pca_data = pca_df.select(pc_cols).to_numpy().astype(float)
	
	# Standardize the PCA data
	if _HAS_CLUSTERING:
		scaler = StandardScaler()
		pca_data_scaled = scaler.fit_transform(pca_data)
	else:
		# Simple standardization if sklearn not available
		pca_data_scaled = (pca_data - np.mean(pca_data, axis=0)) / np.std(pca_data, axis=0)
	
	for new_cluster_id, (old_cluster_id, _) in enumerate(cluster_sizes):
		# Get indices for this cluster (using original cluster ID)
		indices = np.where(labels == old_cluster_id)[0]
		
		if len(indices) > 0:
			# Calculate percentage
			cluster_size_pct = 100.0 * len(indices) / len(labels)
			
			# Get cluster data
			cluster_data_scaled = pca_data_scaled[indices]
			
			# Find KDE representative if clustering dependencies are available
			if _HAS_CLUSTERING and len(cluster_data_scaled) > 1:
				try:
					# Find optimal bandwidth
					if len(cluster_data_scaled) < 5:
						bandwidth = 0.5
					else:
						grid = GridSearchCV(
							KernelDensity(kernel='gaussian'),
							{'bandwidth': np.linspace(0.1, 1.0, 30)},
							cv=min(5, len(cluster_data_scaled)), n_jobs=-1
						)
						grid.fit(cluster_data_scaled)
						bandwidth = float(grid.best_params_['bandwidth'])
					
					# Fit KDE model
					kde_model = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(cluster_data_scaled)
					
					# Find KDE maximum
					bounds = [(float(cluster_data_scaled[:, dim].min()), float(cluster_data_scaled[:, dim].max()))
							  for dim in range(cluster_data_scaled.shape[1])]
					
					result = minimize(
						lambda x: -kde_model.score_samples([x])[0],
						cluster_data_scaled.mean(axis=0),
						bounds=bounds,
						method='L-BFGS-B'
					)
					kde_center_scaled = result.x
					
					# Find closest point to KDE maximum
					distances = np.linalg.norm(cluster_data_scaled - kde_center_scaled, axis=1)
					closest_local_idx = int(np.argmin(distances))
					rep_idx = int(indices[closest_local_idx])
					
				except Exception as e:
					# Fallback to first frame if KDE fails
					cluster_frames = pca_df.filter(pl.col("frame").is_in(indices.tolist()))
					rep_idx = int(cluster_frames["frame"][0]) if len(cluster_frames) > 0 else int(indices[0])
			else:
				# Fallback to first frame if dependencies not available or single point
				cluster_frames = pca_df.filter(pl.col("frame").is_in(indices.tolist()))
				rep_idx = int(cluster_frames["frame"][0]) if len(cluster_frames) > 0 else int(indices[0])
			
			reps.append({
				'cluster_id': int(new_cluster_id),
				'representative_idx': rep_idx,
				'cluster_size_pct': float(cluster_size_pct),
				'n_points': len(indices),
				'n_clusters': int(n_clusters),
				'members': indices.tolist(),
				'has_kde_center': _HAS_CLUSTERING and len(indices) > 1
			})
	
	return reps

def save_clustering_results(all_levels: Dict[int, List[Dict[str, Any]]], output_path: str = "clustering_results.json") -> None:
	import json
	serializable_levels: Dict[str, List[Dict[str, Any]]] = {}
	for level, representatives in all_levels.items():
		serializable_reps: List[Dict[str, Any]] = []
		for rep in representatives:
			serializable_rep = rep.copy()
			if 'kde_center_scaled' in rep and isinstance(rep['kde_center_scaled'], np.ndarray):
				serializable_rep['kde_center_scaled'] = rep['kde_center_scaled'].tolist()
			# Ensure members (indices) are JSON serializable
			if 'members' in serializable_rep and isinstance(serializable_rep['members'], np.ndarray):
				serializable_rep['members'] = serializable_rep['members'].tolist()
			serializable_reps.append(serializable_rep)
		# Store under sequential level key (stringified)
		serializable_levels[str(level)] = serializable_reps
	with open(output_path, 'w') as f:
		json.dump(serializable_levels, f, indent=2)


def save_clustering_results_parquet(
	all_levels: Dict[int, List[Dict[str, Any]]],
	output_dir: str = "clustering_results",
	*,
	include_members: bool = True,
) -> None:
	"""Persist clustering results efficiently as Parquet files using Polars.

	Creates two (or one) parquet files in ``output_dir``:

	1. cluster_summary.parquet: one row per (level, cluster_id) with
	   representative_idx, percentages, n_points, etc.
	2. cluster_members.parquet: (optional) one row per member with columns
	   [n_clusters, cluster_id, member_idx]. This normalized layout lets you
	   filter / join quickly without exploding lists in memory.

	Parameters
	----------
	all_levels : Dict[int, List[Dict[str, Any]]]
		Structure returned by ``hierarchical_clustering_pipeline`` (now augmented
		with 'members' per cluster).
	output_dir : str
		Directory to create / overwrite parquet outputs.
	include_members : bool, default True
		Whether to write the (potentially large) membership table.
	"""
	os.makedirs(output_dir, exist_ok=True)
	
	logger.info(f"Saving clustering results with {len(all_levels)} levels to {output_dir}")

	# Build summary rows (use sequential 'level' and include actual 'n_clusters' per level)
	summary_rows = []
	members_rows = []
	for level, reps in all_levels.items():
		logger.info(f"Processing level {level} with {len(reps)} clusters")
		# derive n_clusters for this level from reps if available
		n_clusters_in_level = reps[0].get('n_clusters') if reps else None
		for rep in reps:
			# Ensure all values are properly typed and handle None values
			rep_idx = rep.get('representative_idx')
			if rep_idx is not None:
				rep_idx = int(rep_idx)
			
			cluster_size_pct = rep.get('cluster_size_pct')
			if cluster_size_pct is not None:
				cluster_size_pct = float(cluster_size_pct)
			
			n_points = rep.get('n_points')
			if n_points is not None:
				n_points = int(n_points)
			
			summary_rows.append({
				'level': int(level),
				'n_clusters': int(n_clusters_in_level) if n_clusters_in_level is not None else None,
				'cluster_id': int(rep['cluster_id']),
				'representative_idx': rep_idx,
				'cluster_size_pct': cluster_size_pct,
				'n_points': n_points,
				'has_kde_center': rep.get('kde_center_scaled') is not None,
			})
			if include_members and 'members' in rep:
				for m in rep['members']:
					members_rows.append({
						'level': int(level),
						'n_clusters': int(n_clusters_in_level) if n_clusters_in_level is not None else None,
						'cluster_id': int(rep['cluster_id']),
						'member_idx': int(m),
					})

	logger.info(f"Generated {len(summary_rows)} summary rows and {len(members_rows)} member rows")

	cluster_summary_df = pl.DataFrame(summary_rows) if summary_rows else pl.DataFrame(schema={
		'level': pl.Int64, 'n_clusters': pl.Int64, 'cluster_id': pl.Int64, 'representative_idx': pl.Int64,
		'cluster_size_pct': pl.Float64, 'n_points': pl.Int64, 'has_kde_center': pl.Boolean
	})
	
	logger.info(f"Summary DataFrame before saving: shape {cluster_summary_df.shape}")
	if cluster_summary_df.height > 0:
		logger.info(f"Summary DataFrame columns: {cluster_summary_df.columns}")
	
	cluster_summary_df.write_parquet(os.path.join(output_dir, 'cluster_summary.parquet'))

	if include_members:
		cluster_members_df = pl.DataFrame(members_rows) if members_rows else pl.DataFrame(schema={
			'level': pl.Int64, 'n_clusters': pl.Int64, 'cluster_id': pl.Int64, 'member_idx': pl.Int64
		})
		cluster_members_df.write_parquet(os.path.join(output_dir, 'cluster_members.parquet'))

	# Lightweight metadata JSON (e.g., available levels)
	# metadata: list of sequential levels and mapping to actual number of clusters
	meta = {
		'levels': sorted(int(k) for k in all_levels.keys()),
		'level_to_n_clusters': {int(k): (v[0].get('n_clusters') if v else None) for k, v in all_levels.items()},
		'has_members': include_members,
		'format_version': 2,
	}
	with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
		json.dump(meta, f, indent=2)


class ClusteringResults:
	"""Convenience API wrapper for loaded clustering results.

	Usage::

		cr = load_clustering_results_parquet('clustering_results')
		cr.levels() -> list of cluster levels (n_clusters)
		cr.representatives(level) -> Polars DataFrame of representatives for that level
		cr.clusters(level) -> list of cluster IDs
		cr.members(level, cluster_id) -> Python list of member indices
		cr.membership_vector(level) -> numpy array of length max_index+1 mapping index->cluster_id
	"""

	def __init__(self, summary: pl.DataFrame, members: Optional[pl.DataFrame]):
		self.summary = summary
		# store membership table under a private name to avoid shadowing the
		# public method `members(level, cluster_id)`
		self._members = members
		# build mapping level -> n_clusters if present in summary
		if 'level' in self.summary.columns and 'n_clusters' in self.summary.columns:
			self._level_to_nclusters = {int(r['level']): int(r['n_clusters']) for r in self.summary.select(['level','n_clusters']).unique().iter_rows(named=True)}
		else:
			self._level_to_nclusters = {}

	def levels(self) -> list[int]:
		"""Return sequential levels (1,2,3...)."""
		if self.summary.height == 0:
			return []
		if 'level' in self.summary.columns:
			return sorted(self.summary['level'].unique().to_list())
		# Backwards compatibility: fall back to using n_clusters as levels
		return sorted(self.summary['n_clusters'].unique().to_list()) if 'n_clusters' in self.summary.columns else []

	def representatives(self, level: int) -> pl.DataFrame:
		"""Get representatives for a specific sequential level (1-based).

		If the stored summary uses a 'level' column this will filter by that. If
		not, it will fall back to interpreting the provided `level` as the actual
		number of clusters (backwards compatibility).
		"""
		logger.info(f"Getting representatives for level {level}")
		if 'level' in self.summary.columns:
			result = self.summary.filter(pl.col('level') == int(level))
		else:
			# legacy behavior: filter by n_clusters
			result = self.summary.filter(pl.col('n_clusters') == int(level))
		return result

	def clusters(self, level: int) -> list[int]:
		return self.representatives(level)['cluster_id'].to_list()

	def get_n_clusters(self, level: int) -> Optional[int]:
		"""Return the actual number of clusters for a sequential `level` (or None).

		Uses an internal mapping if available, falls back to summary table. For
		backwards compatibility, if no metadata is available it will interpret the
		provided `level` as the n_clusters value and return it.
		"""
		lvl = int(level)
		# prefer mapping built at construction
		if hasattr(self, '_level_to_nclusters') and self._level_to_nclusters:
			return self._level_to_nclusters.get(lvl)
		# try to read from summary table
		if 'level' in self.summary.columns and 'n_clusters' in self.summary.columns:
			df = self.summary.filter(pl.col('level') == lvl)
			if df.height == 0:
				return None
			vals = df['n_clusters'].unique().to_list()
			return int(vals[0]) if vals else None
		# fallback: assume user passed n_clusters directly
		try:
			return int(level)
		except Exception:
			return None

	def get_representative_idx(self, level: int, cluster_id: int) -> Optional[int]:
		"""Return the representative index for given (level, cluster_id) or None."""
		rep_df = self.representatives(level).filter(pl.col('cluster_id') == int(cluster_id))
		if rep_df.height == 0:
			return None
		val = rep_df['representative_idx'].to_list()[0]
		return int(val) if val is not None else None

	def get_percentage(self, level: int, cluster_id: int) -> Optional[float]:
		"""Return cluster size percentage (0-100) for given (level, cluster_id) or None."""
		rep_df = self.representatives(level).filter(pl.col('cluster_id') == int(cluster_id))
		if rep_df.height == 0:
			return None
		val = rep_df['cluster_size_pct'].to_list()[0]
		return float(val) if val is not None else None

	def members_df(self, level: Optional[int] = None) -> pl.DataFrame:
		if self._members is None:
			raise ValueError('No membership data stored.')
		if level is None:
			return self._members
		# prefer 'level' column in members table, else fall back to 'n_clusters' for compatibility
		if 'level' in self._members.columns:
			return self._members.filter(pl.col('level') == int(level))
		if 'n_clusters' in self._members.columns:
			return self._members.filter(pl.col('n_clusters') == int(level))
		# no matching column - return full table
		return self._members

	def members(self, level: int, cluster_id: int) -> list[int]:
		if self._members is None:
			raise ValueError('No membership data stored.')
		return self.members_df(level).filter(pl.col('cluster_id') == cluster_id)['member_idx'].to_list()

	def representative_indices(self, level: int) -> list[int]:
		df = self.representatives(level)
		return [r for r in df['representative_idx'].to_list() if r is not None]

	def membership_vector(self, level: int) -> np.ndarray:
		"""Return an array mapping frame index -> cluster id for a given level.

		If memberships are sparse (frames missing), output length equals 1 + max(member_idx).
		"""
		if self._members is None:
			raise ValueError('No membership data stored.')
		mdf = self.members_df(level)
		if mdf.height == 0:
			return np.array([], dtype=int)
		max_idx = int(mdf['member_idx'].max())
		vec = np.full(max_idx + 1, -1, dtype=int)
		for row in mdf.iter_rows(named=True):
			vec[int(row['member_idx'])] = int(row['cluster_id'])
		return vec


def load_clustering_results_parquet(output_dir: str) -> ClusteringResults:
	"""Load clustering results previously saved by ``save_clustering_results_parquet``.

	Parameters
	----------
	output_dir : str
		Directory containing parquet + metadata files.

	Returns
	-------
	ClusteringResults
		Wrapper object with convenience query methods.
	"""
	import json
	meta_path = os.path.join(output_dir, 'metadata.json')
	if not os.path.exists(meta_path):
		raise FileNotFoundError(f'Metadata file not found: {meta_path}')
	with open(meta_path, 'r') as f:
		meta = json.load(f)
	
	logger.info(f"Loading clustering results from {output_dir}, metadata: {meta}")
	
	summary_path = os.path.join(output_dir, 'cluster_summary.parquet')
	if not os.path.exists(summary_path):
		raise FileNotFoundError(f'cluster_summary.parquet missing in {output_dir}')
	
	summary_df = _read_parquet_robust(summary_path)
	logger.info(f"Loaded summary DataFrame with shape {summary_df.shape}")
	logger.info(f"Summary columns: {summary_df.columns}")
	if summary_df.height > 0:
		logger.info(f"First few summary rows:\n{summary_df.head()}")
	else:
		logger.warning("Summary DataFrame is empty!")
	
	members_df = None
	if meta.get('has_members'):
		members_path = os.path.join(output_dir, 'cluster_members.parquet')
		if os.path.exists(members_path):
			members_df = _read_parquet_robust(members_path)
			logger.info(f"Loaded members DataFrame with shape {members_df.shape}")
		else:
			logger.warning('Metadata indicates members present but file missing.')

	# --- Normalize old schema to new: ensure 'level' (sequential) and 'n_clusters' columns exist.
	# If the saved summary used 'n_clusters' column as the key (legacy), transform to
	# sequential level indices where levels are ordered by increasing 'n_clusters' and
	# the 'level' column stores 1..N and 'n_clusters' stores the actual cluster count.
	if 'level' not in summary_df.columns and 'n_clusters' in summary_df.columns:
		unique_nc = sorted(int(x) for x in summary_df['n_clusters'].unique().to_list())
		# map actual n_clusters -> sequential level index
		nc_to_level = {nc: i+1 for i, nc in enumerate(unique_nc)}
		# add level column
		summary_df = summary_df.with_columns(pl.col('n_clusters').apply(lambda x: int(nc_to_level[int(x)])).alias('level'))
		# reorder columns to place level first for readability
		cols = ['level'] + [c for c in summary_df.columns if c != 'level']
		summary_df = summary_df.select(cols)

	# Ensure members_df has 'level' column as well, preferring to compute from n_clusters if needed
	if members_df is not None and 'level' not in members_df.columns:
		if 'n_clusters' in members_df.columns:
			# map n_clusters -> level using summary mapping (if available)
			unique_nc = sorted(int(x) for x in summary_df['n_clusters'].unique().to_list())
			nc_to_level = {nc: i+1 for i, nc in enumerate(unique_nc)}
			members_df = members_df.with_columns(pl.col('n_clusters').apply(lambda x: int(nc_to_level.get(int(x), int(x)))).alias('level'))

	return ClusteringResults(summary_df, members_df)


def knn_conformational_entropy(
	pca_df: pl.DataFrame,
	component_cols: Optional[list[str]] = None,
	n_components: Optional[int] = None,
	k: int = 3,
	center: bool = False,
	standardize: bool = False,
	return_details: bool = False,
	boltzmann: bool = False,
	temperature: Optional[float] = None,
	k_B: float = 1.380649e-23,
) -> float | dict:
	"""Estimate differential conformational entropy from PCA coordinates using kNN.

	Implements the Kozachenko-Leonenko (kNN) entropy estimator:

	H = psi(N) - psi(k) + ln(V_d) + (d/N) * sum_i ln(r_{i,k})

	where r_{i,k} is the distance from point i to its k-th nearest neighbour (excluding itself),
	psi is the digamma function, d is dimensionality and V_d is the volume of the d-dimensional
	unit ball. Returns entropy in natural units (nats). If ``boltzmann`` is True, also multiplies
	by k_B (and optionally temperature to yield an approximate k_B*T*H free-energy-like scale).

	Parameters
	----------
	pca_df : pl.DataFrame
		DataFrame returned by ``pca_conformation_landscape`` (or equivalent). Must contain
		principal component columns.
	component_cols : list[str], optional
		Specific columns to use. If None, all columns starting with 'PC' (case-insensitive)
		and followed by digits are used, in sorted numerical order.
	k : int, default 3
		k for kNN entropy. Must be >=1 and < N.
	center : bool, default False
		If True, subtract mean of each component before computing distances (PCA output is
		already centered unless further transformations were applied; optional here).
	standardize : bool, default False
		If True, divide each component by its standard deviation (useful if comparing entropy
		across sets where scaling differs, but changes absolute value).
	return_details : bool, default False
		Return a dict with metadata instead of a bare float.
	boltzmann : bool, default False
		Include physical entropy S = k_B * H (and if temperature supplied, k_B * T * H).
	temperature : float, optional
		Temperature in Kelvin. Only used if ``boltzmann`` is True. Adds field 'kBT_H' (J).
	k_B : float, default 1.380649e-23
		Boltzmann constant to use.

	Returns
	-------
	float | dict
		Entropy H in nats (float) or a dictionary of details if ``return_details`` True.

	Notes
	-----
	* Differential entropy can be negative; comparisons between systems are typically of interest.
	* For very large N (> 1e5) this may become costly; consider subsampling or larger k.
	* If scikit-learn is unavailable, a pure NumPy O(N^2) fallback distance computation is used.
	"""
	if component_cols is None:
		# Auto-detect PC columns
		component_cols = [c for c in pca_df.columns if c.lower().startswith("pc") and c[2:].isdigit()]
		# Sort numerically by component index
		component_cols.sort(key=lambda x: int(x[2:]) if x[2:].isdigit() else 0)
		# Optionally limit to first n_components
		if n_components is not None:
			if n_components < 1:
				raise ValueError("n_components must be >= 1")
			component_cols = component_cols[:n_components]
	if not component_cols:
		raise ValueError("No PCA component columns found for entropy calculation")
	data_np = pca_df.select(component_cols).to_numpy().astype(float)
	N, d = data_np.shape
	if N <= 1:
		raise ValueError("Need at least two points for entropy estimation")
	if k < 1:
		raise ValueError("k must be >=1")
	if k >= N:
		raise ValueError("k must be < number of points")
	# Optional centering / scaling
	if center:
		data_np -= data_np.mean(axis=0, keepdims=True)
	if standardize:
		std = data_np.std(axis=0, ddof=1, keepdims=True)
		std[std == 0] = 1.0
		data_np /= std

	# Compute k-th neighbour distances
	if _HAS_SKLEARN_NEIGHBORS:
		# Use k+1 to include self then discard
		nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
		nbrs.fit(data_np)
		distances, _ = nbrs.kneighbors(data_np)
		r_k = distances[:, k]  # k-th excluding self at index 0
	else:
		# Fallback O(N^2) distances (avoid for large N)
		diff = data_np[:, None, :] - data_np[None, :, :]
		dmat = np.linalg.norm(diff, axis=2)
		# Replace diagonal with inf so argsort ignores self
		np.fill_diagonal(dmat, np.inf)
		r_k = np.partition(dmat, kth=k-1, axis=1)[:, k-1]

	if np.any(r_k <= 0):  # numerical safety
		# Add a tiny jitter to zero distances (duplicate points)
		jitter = 1e-12
		r_k = np.where(r_k <= 0, r_k + jitter, r_k)

	# Volume of d-dim unit ball
	V_d = math.pi ** (d / 2.0) / math.gamma(d / 2.0 + 1.0)
	H = float(_safe_digamma(N) - _safe_digamma(k) + math.log(V_d) + (d / N) * np.sum(np.log(r_k)))

	if not return_details and not boltzmann:
		return H

	result = {
		'H_nats': H,
		'k': k,
		'n_points': N,
	'dim': d,
		'mean_r_k': float(np.mean(r_k)),
		'component_cols': component_cols,
	'used_n_components': len(component_cols),
		'centered': center,
		'standardized': standardize,
	}
	if boltzmann:
		S = k_B * H  # J/K
		result['S_J_per_K'] = S
		if temperature is not None:
			result['kBT_H_J'] = S * temperature  # k_B * T * H (not a standard thermodynamic quantity but useful scale)
	return result if return_details or boltzmann else H




__all__ = [
	"pca_conformation_landscape",
	"gmm_optimize_silhouette",
	"gmm_optimize_glycosidic_deviation",
	"kcenter_coverage_clustering",
	"create_cluster_info",
	"save_clustering_results",
	"save_clustering_results_parquet",
	"load_clustering_results_parquet",
	"ClusteringResults",
	"knn_conformational_entropy",
	"_read_csv_robust",
]

