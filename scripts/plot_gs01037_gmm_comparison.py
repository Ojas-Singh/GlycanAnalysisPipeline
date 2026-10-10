#!/usr/bin/env python3
"""Compare 3- and 5-component GMMs for the GS01037 PCA landscape.

The script fits both models in exactly the exported PC1--PC3 space and writes
plots plus a compact machine-readable diagnostics file.  It is intentionally
standalone so the comparison can be repeated without changing pipeline output.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture


PC_NAMES = ("PC1", "PC2", "PC3")
COLORS = plt.get_cmap("tab10").colors


def read_pca(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    points = np.asarray([[float(row[pc]) for pc in PC_NAMES] for row in rows])
    source_labels = np.asarray([int(row["cluster"]) for row in rows])
    return points, source_labels


def plot_3d(ax, points: np.ndarray, labels: np.ndarray, model: GaussianMixture, title: str) -> None:
    for cluster_id in range(model.n_components):
        mask = labels == cluster_id
        ax.scatter(
            points[mask, 0], points[mask, 1], points[mask, 2],
            s=2, alpha=0.25, color=COLORS[cluster_id], label=f"Cluster {cluster_id}",
            linewidths=0,
        )
    ax.scatter(
        model.means_[:, 0], model.means_[:, 1], model.means_[:, 2],
        marker="X", s=90, color="black", edgecolor="white", linewidth=0.8,
        label="GMM mean", zorder=5,
    )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.view_init(elev=22, azim=-58)


def plot_pairwise(axs, points: np.ndarray, labels: np.ndarray, model: GaussianMixture, title: str) -> None:
    pairs = ((0, 1), (0, 2), (1, 2))
    for ax, (first, second) in zip(axs, pairs):
        for cluster_id in range(model.n_components):
            mask = labels == cluster_id
            ax.scatter(
                points[mask, first], points[mask, second], s=2, alpha=0.25,
                color=COLORS[cluster_id], linewidths=0,
            )
        ax.scatter(
            model.means_[:, first], model.means_[:, second], marker="X", s=75,
            color="black", edgecolor="white", linewidth=0.8, zorder=5,
        )
        ax.set_xlabel(PC_NAMES[first])
        ax.set_ylabel(PC_NAMES[second])
        ax.grid(alpha=0.2)
    axs[0].set_title(title)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pca-csv",
        type=Path,
        default=Path("static/GS01037/output/pca.csv"),
        help="Exported PC1--PC3 and level-1-cluster assignments.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/GS01037_gmm_3_vs_5"),
        help="Directory for comparison artefacts.",
    )
    parser.add_argument("--plot-sample", type=int, default=10_000)
    parser.add_argument("--silhouette-sample", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    points, exported_labels = read_pca(args.pca_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    plot_indices = rng.choice(len(points), size=min(args.plot_sample, len(points)), replace=False)
    silhouette_indices = rng.choice(
        len(points), size=min(args.silhouette_sample, len(points)), replace=False
    )
    plot_points = points[plot_indices]

    results: dict[str, object] = {
        "source": str(args.pca_csv),
        "n_frames": int(len(points)),
        "space": list(PC_NAMES),
        "plot_sample_size": int(len(plot_indices)),
        "silhouette_sample_size": int(len(silhouette_indices)),
        "models": {},
    }
    fitted: dict[int, tuple[GaussianMixture, np.ndarray]] = {}
    for n_components in (3, 5):
        model = GaussianMixture(n_components=n_components, random_state=42)
        labels = model.fit_predict(points)
        fitted[n_components] = (model, labels)
        sample_labels = labels[silhouette_indices]
        results["models"][str(n_components)] = {
            "n_components": n_components,
            "bic": float(model.bic(points)),
            "aic": float(model.aic(points)),
            "silhouette_sample": float(silhouette_score(points[silhouette_indices], sample_labels)),
            "cluster_sizes": {str(c): int((labels == c).sum()) for c in range(n_components)},
            "means": model.means_.tolist(),
        }

    # This confirms whether the exported level-1 labels are the same GMM fit,
    # while tolerating arbitrary component-number permutations.
    results["exported_level_1_vs_refit_gmm_5_ari"] = float(
        adjusted_rand_score(exported_labels, fitted[5][1])
    )
    results["exported_level_1_cluster_sizes"] = {
        str(c): int((exported_labels == c).sum()) for c in sorted(set(exported_labels))
    }

    # Direct visualisation of the saved CSV: neither refit nor joined data are
    # used here.  The displayed X markers are empirical per-cluster centroids.
    source_centers = np.asarray([
        points[exported_labels == c].mean(axis=0) for c in sorted(set(exported_labels))
    ])
    source_model = SimpleNamespace(n_components=len(source_centers), means_=source_centers)
    fig = plt.figure(figsize=(8, 7), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    plot_3d(
        ax, plot_points, exported_labels[plot_indices], source_model,
        "GS01037: saved pca.csv level-1 labels\nPC1–PC3; 10,000 sampled frames",
    )
    ax.legend(loc="upper left", fontsize=8, markerscale=2)
    fig.savefig(args.output_dir / "pca_csv_level_1_3d.png", dpi=220)
    plt.close(fig)

    fig = plt.figure(figsize=(15, 7), constrained_layout=True)
    for plot_number, n_components in enumerate((3, 5), start=1):
        model, labels = fitted[n_components]
        ax = fig.add_subplot(1, 2, plot_number, projection="3d")
        plot_3d(
            ax, plot_points, labels[plot_indices], model,
            f"GS01037: {n_components}-component GMM\nPC1–PC3; 10,000 sampled frames",
        )
        ax.legend(loc="upper left", fontsize=8, markerscale=2)
    fig.savefig(args.output_dir / "gmm_3_vs_5_3d.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for row, n_components in enumerate((3, 5)):
        model, labels = fitted[n_components]
        plot_pairwise(
            axes[row], plot_points, labels[plot_indices], model,
            f"{n_components}-component GMM (same frame sample)",
        )
    fig.savefig(args.output_dir / "gmm_3_vs_5_pairwise.png", dpi=220)
    plt.close(fig)

    with (args.output_dir / "diagnostics.json").open("w") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")

    with (args.output_dir / "README.md").open("w") as handle:
        handle.write(
            "# GS01037 GMM comparison\n\n"
            "Both models are fit directly to the exported PC1, PC2, PC3 coordinates. "
            "Black X markers are GMM means.  The plots use the same reproducible "
            "10,000-frame sample; diagnostics use the complete 40,004-frame dataset "
            "except for the explicitly sampled silhouette calculation.  "
            "pca_csv_level_1_3d.png is plotted directly from the saved CSV labels, "
            "without refitting or joining.\n"
        )


if __name__ == "__main__":
    main()
