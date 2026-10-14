"""The clustering step: the Level 2 to Level 1 map in the published numbering, and a reproducible analysis.

- ``coverage_clusters_per_main`` names Level 1 and Level 2 clusters by the numbers they are published
  under (sorted by size), not by the labels of GMM and k-center.
- The PCA projection does not depend on the global random generator (the randomized SVD is seeded).
- The first k-center is the most central frame, not a seeded random index that is the same for every
  trajectory of the same length.
- ``lib.embedding`` works, with its representatives, when matplotlib is not installed.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import lib.config as config
from lib import embedding, v2

ROOT = Path(__file__).resolve().parents[1]


# --- coverage_clusters_per_main ---------------------------------------------------------------------------------
def test_map_uses_the_published_numbers_of_both_levels():
    # Level 1: cluster 0 (the largest) holds frames 0-5, cluster 1 frames 6-9
    level_1 = [{"cluster_id": 0, "members": [0, 1, 2, 3, 4, 5]}, {"cluster_id": 1, "members": [6, 7, 8, 9]}]
    # Level 2, published numbers: 0 = {0,1,2}, 1 = {6,7,8}, 2 = {3,4}, 3 = {5, 9}: a tie, the lower Level 1 number
    level_2 = [{"cluster_id": 2, "members": [3, 4]}, {"cluster_id": 0, "members": [0, 1, 2]},
               {"cluster_id": 3, "members": [5, 9]}, {"cluster_id": 1, "members": [6, 7, 8]}]
    assert v2.coverage_clusters_per_main(level_1, level_2, n_frames=10) == {0: [0, 2, 3], 1: [1]}
    assert v2.coverage_clusters_per_main(level_1, level_2, n_frames=10, n_main=3) == {0: [0, 2, 3], 1: [1], 2: []}


def blobs(sizes, seed=1):
    """3-D points in well separated blobs of the given sizes, the smaller ones first."""
    rng = np.random.default_rng(seed)
    centres = [(0, 0, 0), (9, 0, 0), (0, 9, 0), (0, 0, 9), (9, 9, 9)]
    return np.concatenate([rng.normal(loc=c, scale=1.0, size=(n, 3)) for c, n in zip(centres, sizes)])


def test_clustering_step_writes_the_map_in_the_published_numbering(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "max_workers", 1)
    monkeypatch.setattr(config, "cluster_levels", (1, 2))
    points = blobs((40, 90, 160, 260, 400))
    rng = np.random.default_rng(2)
    columns = {"frame": np.arange(len(points))}
    for i in range(10):
        columns[f"PC{i + 1}"] = points[:, i] if i < 3 else rng.normal(scale=0.1, size=len(points))
    pca_path = tmp_path / "pca_conformation_landscape.parquet"
    pl.DataFrame(columns).write_parquet(pca_path)

    v2.step_clustering_gmm(str(tmp_path), str(pca_path), force=True)

    stored = json.loads((tmp_path / "pca.json").read_text())["main_clustering"]["coverage_clusters_per_main"]
    results = embedding.load_clustering_results_parquet(str(tmp_path))
    main, cover = results.membership_vector(1), results.membership_vector(2)
    expected = {str(k): [] for k in range(results.get_n_clusters(1))}
    for cluster in range(results.get_n_clusters(2)):
        labels, counts = np.unique(main[cover == cluster], return_counts=True)
        expected[str(int(labels[np.argmax(counts)]))].append(cluster)
    assert stored == expected
    # published Level 1 cluster 0 is the largest blob, and the Level 2 clusters listed under it lie in it
    sizes = [int((main == k).sum()) for k in range(5)]
    assert sizes == sorted(sizes, reverse=True)
    for cluster in stored["0"]:
        assert set(np.unique(main[cover == cluster])) == {0}


def test_a_new_run_replaces_the_lists_of_pca_json(tmp_path, monkeypatch):
    from lib import storage as storage_module
    from lib.storage import LocalStorageBackend, StorageManager

    monkeypatch.setattr(storage_module, "_storage_manager", StorageManager(LocalStorageBackend()))
    (tmp_path / "pca.json").write_text(json.dumps({
        "pca_analysis": {"explained_variance_ratio": [0.5, 0.2], "params": {"seed": 0}},
        "main_clustering": {"H_nats": 9.0, "coverage_clusters_per_main": {"0": [4, 15], "1": [0]}},
    }))
    v2._update_pca_json(str(tmp_path), {
        "pca_analysis": {"explained_variance_ratio": [0.4, 0.3]},
        "main_clustering": {"coverage_clusters_per_main": {"0": [0, 1], "1": [2]}},
    })
    stored = json.loads((tmp_path / "pca.json").read_text())
    assert stored["pca_analysis"] == {"explained_variance_ratio": [0.4, 0.3], "params": {"seed": 0}}
    assert stored["main_clustering"] == {"H_nats": 9.0, "coverage_clusters_per_main": {"0": [0, 1], "1": [2]}}


# --- PCA ----------------------------------------------------------------------------------------------------------
def test_pca_projection_does_not_depend_on_the_global_generator(tmp_path):
    from test_process_equalities import stored_chain

    from lib.glypdbio import get_conformation_landscape

    data_dir = stored_chain(tmp_path, frames=900, atoms=50)          # 1,225 pairs: the randomized solver
    landscape, chosen = get_conformation_landscape(data_dir, method="noH_noROH_fast", sample_size=300, seed=0,
                                                   return_atom_indices=True)
    frames = []
    for name, global_seed in (("one", 1), ("two", 2)):
        np.random.seed(global_seed)
        frame, evr, _ = v2._fit_pca_and_write_landscape(data_dir, str(tmp_path / name), landscape, chosen)
        frames.append((frame, evr))
    assert frames[0][0].equals(frames[1][0]) and frames[0][1] == frames[1][1]
    import joblib
    model = joblib.load(tmp_path / "one" / "landscape" / "pca_model.joblib")
    assert model._fit_svd_solver == "randomized" and model.random_state == embedding.PCA_RANDOM_STATE


# --- k-center -----------------------------------------------------------------------------------------------------
def test_first_kcenter_is_the_most_central_frame_and_depends_on_the_trajectory():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(600, 3))
    b = rng.normal(size=(600, 3)) * (1.0, 2.0, 0.5) + 3.0          # another trajectory of the same length
    firsts = []
    for data in (a, b):
        scaled = (data - data.mean(axis=0)) / data.std(axis=0)
        central = int(np.argmin(np.linalg.norm(scaled, axis=1)))
        assert embedding.kcenter_first_center(scaled) == central
        np.random.seed(0)
        labels, count = embedding.kcenter_coverage_clustering(data, max_clusters=20)
        assert count == 20 and labels[central] == 0              # the cluster of the first center is cluster 0
        np.random.seed(99)
        again, _ = embedding.kcenter_coverage_clustering(data, max_clusters=20)
        assert np.array_equal(labels, again)
        firsts.append(central)
    assert firsts[0] != firsts[1]


# --- without matplotlib -------------------------------------------------------------------------------------------
SCRIPT = """
import json, sys
import numpy as np, polars as pl
if sys.argv[1] == "without":
    sys.modules["matplotlib"] = None
    sys.modules["matplotlib.pyplot"] = None
import lib.config as config
config.max_workers = 1
from lib import embedding
rng = np.random.default_rng(3)
data = np.concatenate([rng.normal(loc=c, size=(80, 3)) for c in ((0, 0, 0), (6, 0, 0), (0, 6, 0))])
labels, n = embedding.kcenter_coverage_clustering(data, max_clusters=6)
df = pl.DataFrame({"frame": np.arange(len(data)), **{f"PC{i + 1}": data[:, i] for i in range(3)}})
reps = embedding.create_cluster_info_fast(labels, df, n)
print(json.dumps({"labels": labels.tolist(), "reps": [r["representative_idx"] for r in reps],
                  "kde": [r["has_kde_center"] for r in reps], "plt": embedding._plt is None}))
"""


def test_embedding_works_without_matplotlib():
    outputs = {}
    for case in ("with", "without"):
        run = subprocess.run([sys.executable, "-c", SCRIPT, case], cwd=ROOT, capture_output=True, text=True)
        assert run.returncode == 0, run.stderr[-2000:]
        outputs[case] = json.loads(run.stdout.strip().splitlines()[-1])
    assert outputs["without"]["plt"] and not outputs["with"]["plt"]
    assert all(outputs["without"]["kde"])
    for key in ("labels", "reps", "kde"):
        assert outputs["without"][key] == outputs["with"][key]
