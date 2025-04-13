from pathlib import Path
from typing import List, Tuple, Any, Optional
import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from lib import graph
import lib.config as config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pcawithG(frames: np.ndarray, idx_noH: List[int], dim: int, name: str) -> Tuple[pd.DataFrame, int]:
    """Perform PCA on molecular frames.
    
    Args:
        frames: Molecular frame data
        idx_noH: Non-hydrogen atom indices
        dim: Number of PCA dimensions
        name: Molecule name
        
    Returns:
        PCA components DataFrame and optimal dimension
    """
    logger.info(f"Starting PCA analysis for {name}")
    try:
        # Calculate distance matrix
        G = np.zeros((len(frames), int(len(frames[0][np.asarray(idx_noH,dtype=int)])*(len(frames[0][np.asarray(idx_noH,dtype=int)])-1)/2)))
        for i in range(len(frames)):
            G[i] = graph.G_flatten(frames[i][np.asarray(idx_noH,dtype=int)])

        # Perform PCA
        pca = PCA(n_components=dim)
        transformed = pca.fit_transform(G)
        pca_df = pd.DataFrame(transformed)
        
        # Calculate explained variance
        exp_var = pca.explained_variance_ratio_
        cum_sum = np.cumsum(exp_var)
        n_dim = config.hard_dim
        
        # Plot variance
        _plot_variance(exp_var, cum_sum, name)
        _plot_eigenvalues(pca.explained_variance_, name)
        
        logger.info(f"PCA completed with {n_dim} dimensions")
        return pca_df, n_dim

    except Exception as e:
        logger.error(f"PCA failed: {str(e)}")
        raise

def _plot_variance(exp_var: np.ndarray, cum_sum: np.ndarray, name: str) -> None:
    """Plot PCA variance."""
    plt.figure()
    plt.bar(range(len(exp_var)), exp_var, alpha=0.5, align='center', 
            label='Individual explained variance')
    plt.step(range(len(cum_sum)), cum_sum, where='mid',
             label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(Path(config.data_dir) / name / 'output' / 'PCA_variance.png', dpi=450)
    plt.close()

def _plot_eigenvalues(eigenvalues: np.ndarray, name: str) -> None:
    """Plot PCA eigenvalues."""
    components = sum(eigenvalues > 1)
    logger.info(f"Components with eigenvalues > 1: {components}")
    
    plt.figure()
    plt.plot(eigenvalues, 'o-')
    plt.title('Scree Plot')
    plt.xlabel('Component number')
    plt.ylabel('Eigenvalue')
    plt.axhline(y=1, color='r', linestyle='--')
    plt.savefig(Path(config.data_dir) / name / 'output' / 'PCA_eigen.png', dpi=450)
    plt.close()

def find_optimal_bandwidth(data: np.ndarray) -> float:
    """Find optimal KDE bandwidth using cross-validation."""
    grid = GridSearchCV(
        KernelDensity(kernel='gaussian'),
        {'bandwidth': np.linspace(0.1, 1.0, 30)},
        cv=5, n_jobs=-1
    )
    grid.fit(data)
    return grid.best_params_['bandwidth']

def kde_score(point: np.ndarray, kde_model: KernelDensity) -> float:
    """Calculate KDE score for a point."""
    return -kde_model.score_samples([point])

def find_closest_point_to_max_kde(cluster_data: np.ndarray, kde_model: KernelDensity) -> np.ndarray:
    """Find point closest to KDE maximum."""
    bounds = [(min(cluster_data[:, dim]), max(cluster_data[:, dim])) 
             for dim in range(cluster_data.shape[1])]
    result = minimize(
        lambda x: -kde_model.score_samples([x])[0],
        cluster_data.mean(axis=0),
        bounds=bounds,
        method='L-BFGS-B'
    )
    return result.x

def best_clustering(n: int, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Perform Gaussian Mixture clustering."""
    gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
    gmm.fit(data)
    return gmm.predict(data), gmm.means_

def kde_c(n_clusters: int, pca_df: pd.DataFrame, selected_columns: List[str]) -> List[List[Any]]:
    """Calculate KDE centers for clusters."""
    logger.info(f"Calculating KDE centers for {n_clusters} clusters")
    pca_np = pca_df[selected_columns].to_numpy()
    kde_centers = []
    popp = []

    for i in range(n_clusters):
        data = pca_df.loc[pca_df["cluster"] == str(i), selected_columns].to_numpy()
        data = StandardScaler().fit_transform(data)
        
        kde_model = KernelDensity(
            kernel='gaussian',
            bandwidth=find_optimal_bandwidth(data)
        ).fit(data)
        
        kde_centers.append(find_closest_point_to_max_kde(data, kde_model))
        
        cluster_df = pca_df.loc[pca_df["cluster"] == str(i)]
        distances = [
            [np.linalg.norm(kde_centers[i] - pca_np[j]), 
             cluster_df["i"].iloc[j],
             cluster_df["cluster"].iloc[j]]
            for j in range(len(cluster_df))
        ]
        distances.sort()
        popp.append([distances[0][1], distances[0][2]])
        
        size = 100 * len(cluster_df) / len(pca_df)
        popp[i].append(size)
    
    return popp

def plot_Silhouette(pca_df: pd.DataFrame, name: str, n_dim: int) -> Tuple[int, List[float]]:
    """Plot silhouette scores for different cluster numbers."""
    logger.info("Calculating silhouette scores")
    selected_columns = list(range(1, n_dim))
    x_range = range(2, 12)
    scores = []
    
    for i in x_range:
        score = metrics.silhouette_score(
            pca_df[selected_columns],
            best_clustering(i, pca_df[selected_columns])[0],
            metric='euclidean'
        )
        scores.append(score)
    
    plt.figure()
    plt.plot(x_range, scores)
    plt.scatter(x_range, scores)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title(f'Silhouette Score vs Number of Clusters [n_dim: {n_dim}]')
    plt.tight_layout()
    plt.savefig(Path(config.data_dir) / name / 'output' / 'Silhouette_Score.png', dpi=450)
    plt.close()
    
    n_clus = find_peaks(scores)[0] + 2
    logger.info(f"Optimal number of clusters: {n_clus}")
    return n_clus, scores

def find_peaks(array: List[float]) -> List[int]:
    """Find peak indices in array."""
    peaks = []
    if len(array) >= 3:
        for i in range(1, len(array)-1):
            if array[i] > array[i-1] and array[i] > array[i+1]:
                peaks.append((i, array[i]))
    
    peaks = [i for i, _ in sorted(peaks, key=lambda x: x[1], reverse=True)]
    return peaks if peaks else [0]