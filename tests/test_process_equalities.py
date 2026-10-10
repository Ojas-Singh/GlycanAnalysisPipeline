"""Regression checks for the speed-ups of the process stage: each must give what the code it replaces gave.

The reference implementations below are the former code paths, kept here in their original form.
Random data only, no network.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import polars as pl
import pytest
import zarr

import lib.config as config
from lib import embedding, glystore, torsion, v2
from lib.glypdbio import get_conformation_landscape, parse_mol2_bonds

from test_archive import make_source, random_coords
from test_archive_store import chain_source

REPO_ROOT = Path(__file__).resolve().parents[1]


def same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


# ---------------------------------------------------------------------------
# Store step: every Zarr chunk is written once
# ---------------------------------------------------------------------------
@pytest.fixture
def chunk_writes(monkeypatch):
    """Record (frames already stored, frames appended, chunk length) of every append to the store."""
    calls = []
    original = glystore._append_frames

    def recording(zarr_arr, xyz_frames):
        calls.append((int(zarr_arr.shape[0]), int(xyz_frames.shape[0]), int(zarr_arr.chunks[0])))
        return original(zarr_arr, xyz_frames)

    monkeypatch.setattr(glystore, "_append_frames", recording)
    return calls


@pytest.mark.parametrize("frames_buffer,frame_chunk", [(128, 0), (1, 0), (7, None), (4, 16), (7, 5), (1000, 50), (3, 1)])
@pytest.mark.parametrize("kind", ["pdb", "archive"])
def test_store_never_appends_into_a_chunk_it_has_written(tmp_path, chunk_writes, kind, frames_buffer, frame_chunk):
    from lib import archive

    frames = 333
    source, coords = chain_source(tmp_path / "data", "DGlcpa1-OH", frames=frames)
    bonds = parse_mol2_bonds(source / "structure.mol2")
    if kind == "archive":
        entry = Path(archive.convert_entry(source, tmp_path / "archive")["entry"])
        chunk_writes.clear()
        glystore.store_from_archive(str(entry), str(tmp_path / "store"), get_connectivity=bonds,
                                    connectivity_kind="serial", frames_buffer=frames_buffer, frame_chunk=frame_chunk)
        del chunk_writes[0]                       # the first-frame store of the topology, replaced afterwards
    else:
        glystore.store_from_pdb(str(source / "simulation.pdb"), str(tmp_path / "store"), get_connectivity=bonds,
                                connectivity_kind="serial", frames_buffer=frames_buffer, frame_chunk=frame_chunk)

    chunk = frames if not frame_chunk else frame_chunk
    stored = zarr.open(str(tmp_path / "store" / "coords.zarr"), mode="r")["xyz"]
    assert stored.shape == (frames, 6, 3) and stored.chunks == (chunk, 6, 3)
    assert same_bits(stored[:], (coords / 1000.0).astype(np.float32))
    # Appends start on a chunk boundary, and only the last one may end inside a chunk: no chunk is
    # read back, extended and compressed again. (The former code appended frames_buffer frames at a time,
    # which with one chunk for the trajectory rewrote that growing chunk on every append.)
    assert sum(count for _, count, _ in chunk_writes) == frames
    for position, (start, count, chunk_length) in enumerate(chunk_writes):
        assert chunk_length == chunk and start % chunk == 0
        if position < len(chunk_writes) - 1:
            assert count % chunk == 0
    if not frame_chunk:
        assert len(chunk_writes) == 1
    assert pl.read_parquet(tmp_path / "store" / "frames.parquet")["n_frames"].to_list() == [frames]


def test_single_chunk_has_the_length_the_model_count_gave(tmp_path):
    # One chunk for the trajectory used to take its length from a separate pass that counted MODEL
    # records; the count now comes from the parsing pass and must be the same number.
    source = make_source(tmp_path / "data", "DGlcpa1-OH", random_coords(9, 12))
    pdb = source / "simulation.pdb"
    assert glystore._count_pdb_frames(str(pdb)) == 9
    glystore.store_from_pdb(str(pdb), str(tmp_path / "plain"), frames_buffer=4, frame_chunk=0)
    assert zarr.open(str(tmp_path / "plain" / "coords.zarr"), mode="r")["xyz"].chunks[0] == 9

    # a MODEL record without atoms at the end is counted but holds no frame
    text = pdb.read_text().replace("END   \n", "MODEL     10\nENDMDL\nEND   \n")
    pdb.write_text(text)
    assert glystore._count_pdb_frames(str(pdb)) == 10
    glystore.store_from_pdb(str(pdb), str(tmp_path / "trailing"), frames_buffer=4, frame_chunk=0)
    stored = zarr.open(str(tmp_path / "trailing" / "coords.zarr"), mode="r")["xyz"]
    assert stored.shape[0] == 9 and stored.chunks[0] == 10

    # no MODEL records at all: one frame
    lines = [line for line in text.splitlines(keepends=True) if line.startswith("ATOM")][:12]
    pdb.write_text("".join(lines) + "END\n")
    assert glystore._count_pdb_frames(str(pdb)) == 1
    glystore.store_from_pdb(str(pdb), str(tmp_path / "single"), frames_buffer=4, frame_chunk=None)
    stored = zarr.open(str(tmp_path / "single" / "coords.zarr"), mode="r")["xyz"]
    assert stored.shape[0] == 1 and stored.chunks[0] == 1


def test_compact_rewrite_script_accepts_its_default_chunk(tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("rewrite_compact", REPO_ROOT / "scripts" / "rewrite_process_data_compact.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.DEFAULT_FRAME_CHUNK == 0
    source, coords = chain_source(tmp_path / "data", "DGlcpa1-OH", frames=40)
    arguments = ["rewrite", "--data-dir", str(tmp_path / "data"), "--process-dir", str(tmp_path / "process"),
                 "--log-dir", str(tmp_path / "logs")]
    monkeypatch.setattr(sys, "argv", arguments)
    assert module.main() == 0                    # the default --frame-chunk 0 used to be refused
    stored = zarr.open(str(tmp_path / "process" / "DGlcpa1-OH" / "data" / "coords.zarr"), mode="r")["xyz"]
    assert stored.shape == (40, 6, 3) and stored.chunks[0] == 40
    assert same_bits(stored[:], (coords / 1000.0).astype(np.float32))
    monkeypatch.setattr(sys, "argv", arguments + ["--frame-chunk", "-1"])
    with pytest.raises(SystemExit):
        module.main()
    capsys.readouterr()


# ---------------------------------------------------------------------------
# k-center
# ---------------------------------------------------------------------------
def reference_kcenter(data, max_clusters, seed=42):
    """lib.embedding.kcenter_coverage_clustering before the speed-up, loop for loop."""
    from sklearn.preprocessing import StandardScaler

    np.random.seed(seed)
    n_samples, _ = data.shape
    if n_samples <= max_clusters:
        return np.arange(n_samples), n_samples
    data_scaled = StandardScaler().fit_transform(data)
    centers = []
    center_indices = []
    first_idx = np.random.randint(0, n_samples)
    centers.append(data_scaled[first_idx])
    center_indices.append(first_idx)
    for k in range(1, max_clusters):
        distances_to_centers = np.full(n_samples, np.inf)
        for center in centers:
            distances = np.linalg.norm(data_scaled - center, axis=1)
            distances_to_centers = np.minimum(distances_to_centers, distances)
        farthest_idx = int(np.argmax(distances_to_centers))
        if farthest_idx in center_indices:
            remaining = [i for i in range(n_samples) if i not in center_indices]
            if not remaining:
                break
            farthest_idx = int(np.random.choice(remaining))
        centers.append(data_scaled[farthest_idx])
        center_indices.append(farthest_idx)
    labels = np.zeros(n_samples, dtype=int)
    for i, point in enumerate(data_scaled):
        distances = [np.linalg.norm(point - center) for center in centers]
        labels[i] = np.argmin(distances)
    return labels, len(centers)


def kcenter_cases():
    rng = np.random.default_rng(3)
    blobs = np.concatenate([rng.normal(loc=center, scale=0.4, size=(300, 3)) for center in ((0, 0, 0), (4, 1, -2), (-3, 5, 2))])
    lattice = np.stack(np.meshgrid(np.arange(8.0), np.arange(8.0), np.arange(5.0)), axis=-1).reshape(-1, 3)
    duplicates = np.repeat(rng.normal(size=(7, 3)), 30, axis=0)     # 7 distinct points: the random branch
    return [
        ("blobs, 16 centers", blobs, 16),
        ("blobs, 128 centers", blobs, 128),
        ("blobs, 512 centers", blobs, 512),
        ("lattice with equidistant points, 40 centers", lattice, 40),
        ("duplicates, more centers than distinct points", duplicates, 12),
        ("fewer points than centers", blobs[:20], 64),
    ]


@pytest.mark.parametrize("name,data,max_clusters", kcenter_cases(), ids=[case[0] for case in kcenter_cases()])
def test_kcenter_gives_the_labels_of_the_former_loops(name, data, max_clusters):
    expected_labels, expected_count = reference_kcenter(data, max_clusters)
    state_after_reference = np.random.get_state()[1].copy()
    labels, count = embedding.kcenter_coverage_clustering(data, max_clusters=max_clusters)
    assert count == expected_count
    assert labels.dtype == expected_labels.dtype and np.array_equal(labels, expected_labels)
    # the global generator is left where the former code left it
    assert np.array_equal(np.random.get_state()[1], state_after_reference)


def test_nearest_center_labels_decide_ties_like_the_scalar_loop():
    rng = np.random.default_rng(11)
    centers = [np.array(c, dtype=float) for c in ((0, 0, 0), (2, 0, 0), (0, 2, 0), (2, 0, 0), (1e-9, 0, 0))]
    points = np.concatenate([
        rng.normal(size=(500, 3)),
        np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 5.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),   # exact ties
        rng.normal(size=(40, 3)) * 1e-12,
    ])
    reference = np.array([int(np.argmin([np.linalg.norm(point - center) for center in centers])) for point in points])
    for block_rows in (1, 7, 4096):
        assert np.array_equal(embedding._nearest_center_labels(points, centers, block_rows=block_rows), reference)
    one = embedding._nearest_center_labels(points, centers[:1])
    assert one.dtype == reference.dtype and not one.any()


# ---------------------------------------------------------------------------
# Silhouette of several labelings
# ---------------------------------------------------------------------------
def test_shared_silhouette_is_sklearns_silhouette_bit_for_bit():
    from sklearn.metrics import silhouette_score
    from sklearn.mixture import GaussianMixture

    rng = np.random.default_rng(5)
    data = np.concatenate([rng.normal(loc=center, scale=0.9, size=(450, 3)) for center in ((0, 0, 0), (3, 1, -2), (-2, 4, 1))])
    # the clustering step hands over a Fortran-ordered array (polars), keep that layout
    data = np.asfortranarray(data)
    labelings = [GaussianMixture(n_components=n, random_state=42).fit_predict(data) for n in range(2, 9)]
    labelings.append(rng.integers(0, 4, size=len(data)) * 3 + 10)          # labels that are not 0..k-1
    expected = [float(silhouette_score(data, labels)) for labels in labelings]

    # few points: scikit-learn computes the distance matrix in one call, and so does the shared version
    for n_jobs in (1, 2):
        scores = embedding.silhouette_scores_shared(data, labelings, n_jobs=n_jobs)
        assert [score.hex() for score in scores] == [score.hex() for score in expected], n_jobs


def test_shared_silhouette_in_blocks_is_sklearns_silhouette_bit_for_bit():
    from sklearn import get_config
    from sklearn.metrics import silhouette_score

    # more points than scikit-learn takes in one call: it works in chunks of rows, the shared version
    # in smaller blocks spread over worker processes
    rng = np.random.default_rng(6)
    centres = ((0, 0, 0), (3, 1, -2), (-2, 4, 1), (5, 5, 5))
    data = np.asfortranarray(np.concatenate([rng.normal(loc=c, scale=1.1, size=(2900, 3)) for c in centres]))
    count = len(data)
    assert get_config()["working_memory"] * 2**20 // (8 * count) < count
    labelings = [np.repeat(np.arange(4), 2900), rng.integers(0, 7, size=count)]
    expected = [float(silhouette_score(data, labels)) for labels in labelings]
    for n_jobs, block_bytes in ((1, 8 * 2**20), (2, 64 * 2**20)):
        scores = embedding.silhouette_scores_shared(data, labelings, n_jobs=n_jobs, block_bytes=block_bytes)
        assert [score.hex() for score in scores] == [score.hex() for score in expected], (n_jobs, block_bytes)


def test_silhouette_blocks_are_even_and_never_a_single_row():
    assert embedding._even_ranges(0, 10, 3) == [(0, 3), (3, 6), (6, 10)]
    assert embedding._even_ranges(5, 7, 4) == [(5, 6), (6, 7)]
    for count, parts in ((160016, 3053), (50005, 299), (11, 5), (4, 2)):
        ranges = embedding._even_ranges(0, count, parts)
        lengths = [last - first for first, last in ranges]
        assert ranges[0][0] == 0 and ranges[-1][1] == count and sum(lengths) == count
        assert max(lengths) - min(lengths) <= 1 and min(lengths) >= 2


def test_shared_silhouette_refuses_what_sklearn_refuses():
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(2)
    data = rng.normal(size=(60, 3))
    one_label = np.zeros(60, dtype=int)
    all_different = np.arange(60)
    good = rng.integers(0, 3, size=60)
    for labels in (one_label, all_different):
        with pytest.raises(ValueError):
            silhouette_score(data, labels)
    scores = embedding.silhouette_scores_shared(data, [one_label, good, all_different], n_jobs=1)
    assert scores[0] is None and scores[2] is None
    assert scores[1].hex() == float(silhouette_score(data, good)).hex()
    assert embedding.silhouette_scores_shared(data, [one_label]) == [None]


# ---------------------------------------------------------------------------
# PCA step
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rows,columns,block_bytes", [(1000, 7, 8 * 1000 * 2), (1000, 8, 8 * 1000 * 3), (4097, 33, 8 * 4097 * 5),
                                                      (5000, 301, 2**18), (3, 2, 64), (50, 1, 64), (2, 5, 1)])
def test_column_statistics_in_blocks_equal_numpy_bit_for_bit(monkeypatch, rows, columns, block_bytes):
    rng = np.random.default_rng(rows + columns)
    X = np.ascontiguousarray(rng.normal(size=(rows, columns)) * rng.uniform(0.1, 50, size=columns) + rng.normal(size=columns))
    blocks = v2._column_blocks(rows, columns, block_bytes)
    assert blocks[0][0] == 0 and blocks[-1][1] == columns
    assert all(stop == next_start for (_, stop), (next_start, _) in zip(blocks, blocks[1:]))
    assert columns < 2 or all(stop - start >= 2 for start, stop in blocks)
    original = v2._column_blocks
    monkeypatch.setattr(v2, "_column_blocks", lambda r, c: original(r, c, block_bytes))
    for threads in (1, 3):
        assert same_bits(v2._column_mean(X, threads), X.mean(axis=0))
        assert same_bits(v2._column_std(X, threads), X.std(axis=0, ddof=1))
        Y = X.copy()
        v2._scale_rows(Y, np.subtract, X.mean(axis=0), threads)
        Z = X.copy()
        Z -= X.mean(axis=0)
        assert same_bits(Y, Z)
        v2._scale_rows(Y, np.divide, X.std(axis=0, ddof=1), threads)
        Z /= X.std(axis=0, ddof=1)
        assert same_bits(Y, Z)


def stored_chain(tmp_path, frames=700, atoms=14):
    """A store with `atoms` heavy atoms: data/ of a process folder."""
    rng = np.random.default_rng(8)
    coords = np.cumsum(rng.normal(scale=1.2, size=(1, atoms, 3)), axis=1) + rng.normal(scale=0.3, size=(frames, atoms, 3))
    table = {
        "serial": np.arange(1, atoms + 1), "name": [f"C{i}" for i in range(atoms)], "element": ["C"] * atoms,
        "res_name": ["0GB"] * atoms, "res_id": np.full(atoms, 2, dtype=np.int16), "chain_id_u8": np.zeros(atoms, dtype=np.uint8),
    }
    glystore.store_from_memory(str(tmp_path / "data"), table, coords.astype(np.float32),
                               connectivity_pairs=[(i, i + 1) for i in range(1, atoms)])
    return str(tmp_path / "data")


def test_landscape_in_float64_and_in_threads_is_the_float32_landscape(tmp_path):
    data_dir = stored_chain(tmp_path)
    reference, atoms = get_conformation_landscape(data_dir, method="noH_noROH_fast", sample_size=300, seed=0,
                                                  return_atom_indices=True)
    assert reference.dtype == np.float32 and reference.shape == (700, 14 * 13 // 2) and reference.flags.c_contiguous
    for threads, chunk in ((1, 1000), (3, 64), (4, 1)):
        wide, wide_atoms = get_conformation_landscape(data_dir, method="noH_noROH_fast", sample_size=300, seed=0,
                                                      return_atom_indices=True, dtype=np.float64, n_threads=threads,
                                                      chunk_size=chunk)
        assert wide.dtype == np.float64 and wide.flags.c_contiguous and np.array_equal(atoms, wide_atoms)
        assert same_bits(wide, reference.astype(np.float64))
        narrow = get_conformation_landscape(data_dir, method="noH_noROH_fast", sample_size=300, seed=0,
                                            n_threads=threads, chunk_size=chunk)
        assert same_bits(narrow, reference)


def reference_pca(landscape, per_pair_scale=False):
    """The fit of lib.v2._fit_pca_and_write_landscape before the memory changes."""
    from sklearn.decomposition import PCA

    X = np.asarray(landscape, dtype=float)
    n_frames, n_pairs = X.shape
    X_proc = X.copy()
    pair_stds = None
    if per_pair_scale:
        pair_stds = X_proc.std(axis=0, ddof=1)
        pair_stds = np.nan_to_num(pair_stds, nan=1.0, posinf=1.0, neginf=1.0)
        pair_stds[pair_stds == 0] = 1.0
        X_proc = X_proc / pair_stds[np.newaxis, :]
    feature_means = X_proc.mean(axis=0)
    X_proc -= feature_means
    feature_stds = X_proc.std(axis=0, ddof=1)
    feature_stds = np.nan_to_num(feature_stds, nan=1.0, posinf=1.0, neginf=1.0)
    feature_stds[feature_stds == 0] = 1.0
    X_proc /= feature_stds
    pca = PCA(n_components=min(10, n_frames, n_pairs), svd_solver="auto")
    scores = pca.fit_transform(X_proc)
    return pca, scores, feature_means, feature_stds, pair_stds


@pytest.mark.parametrize("frames,atoms,solver", [(1000, 14, "covariance_eigh"), (400, 20, "full"), (900, 50, "randomized")])
@pytest.mark.parametrize("per_pair_scale", [False, True])
def test_pca_fit_in_place_equals_the_former_fit(tmp_path, monkeypatch, frames, atoms, solver, per_pair_scale):
    import joblib

    data_dir = stored_chain(tmp_path, frames=frames, atoms=atoms)
    landscape, chosen = get_conformation_landscape(data_dir, method="noH_noROH_fast", sample_size=300, seed=0,
                                                   return_atom_indices=True)
    np.random.seed(123)                     # the randomized solver draws from the global generator
    pca, scores, means, stds, pair_stds = reference_pca(landscape, per_pair_scale)
    assert pca._fit_svd_solver == solver

    monkeypatch.setattr(config, "max_workers", 3)
    monkeypatch.setattr(v2, "_column_blocks", lambda r, c, _original=v2._column_blocks: _original(r, c, 8 * r * 7))
    wide = get_conformation_landscape(data_dir, method="noH_noROH_fast", sample_size=300, seed=0, dtype=np.float64)
    np.random.seed(123)
    frame, evr, cumulative = v2._fit_pca_and_write_landscape(
        data_dir, str(tmp_path / "embedding"), wide, chosen, per_pair_scale=per_pair_scale, overwrite_landscape=True)

    assert frame["frame"].to_list() == list(range(frames))
    assert same_bits(frame.select([f"PC{i}" for i in range(1, 11)]).to_numpy(), np.ascontiguousarray(scores))
    assert evr == [float(x) for x in pca.explained_variance_ratio_]
    stored = np.load(tmp_path / "embedding" / "landscape" / "preprocessing.npz")
    assert same_bits(stored["feature_means"], means) and same_bits(stored["feature_stds"], stds)
    if per_pair_scale:
        assert same_bits(stored["pair_stds"], pair_stds)
    model = joblib.load(tmp_path / "embedding" / "landscape" / "pca_model.joblib")
    assert model.get_params() == pca.get_params() and model.copy is True
    for attribute in ("components_", "mean_", "explained_variance_", "singular_values_"):
        assert same_bits(getattr(model, attribute), getattr(pca, attribute)), attribute
    assert model.noise_variance_ == pca.noise_variance_

    # a float32 landscape that the caller keeps is not modified
    kept = landscape.copy()
    np.random.seed(123)
    again, _, _ = v2._fit_pca_and_write_landscape(data_dir, str(tmp_path / "again"), landscape, chosen,
                                                  per_pair_scale=per_pair_scale)
    assert same_bits(landscape, kept) and again.equals(frame)


# ---------------------------------------------------------------------------
# Plots, representatives, cluster representatives
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtypes", [("int32", "float32"), ("int64", "float64"), ("int64", "float32")])
def test_plot_table_is_the_table_the_row_loop_built(dtypes):
    rng = np.random.default_rng(4)
    data = pd.DataFrame({
        "frame": np.arange(300, dtype=dtypes[0]),
        "a": rng.uniform(-180, 180, 300).astype(dtypes[1]),
        "skipped": rng.uniform(-180, 180, 300).astype(dtypes[1]),
        "b": rng.uniform(-180, 180, 300).astype(dtypes[1]),
    })
    columns, names = ["b", "a"], {"a": "2_3_ϕ", "b": "2_3_ψ"}
    rows = []
    for column in columns:
        for _, row in data.iterrows():
            rows.append({"Value": row[column], "Torsion": names[column], "Frame": row["frame"]})
    expected = pd.DataFrame(rows)
    produced = torsion._torsion_plot_table(data, columns, names)
    # same values in the same rows; the torsion names are a categorical column whose categories are in
    # the order seaborn takes from a column of strings (first appearance)
    assert list(produced.columns) == ["Value", "Torsion", "Frame"]
    assert list(produced["Torsion"].cat.categories) == list(expected["Torsion"].unique()) == ["2_3_ψ", "2_3_ϕ"]
    pd.testing.assert_frame_equal(produced.astype({"Torsion": object}), expected, check_exact=True)
    assert same_bits(produced["Value"].to_numpy(), expected["Value"].to_numpy())
    assert same_bits(produced["Frame"].to_numpy(), expected["Frame"].to_numpy())

    shared = torsion._torsion_plot_table(data, ["a", "b"], {"a": "same", "b": "same"})      # two columns, one name
    assert list(shared["Torsion"].cat.categories) == ["same"] and len(shared) == 600

    mixed = data.assign(note="x")                                   # an object column: the former loop is used
    pd.testing.assert_frame_equal(torsion._torsion_plot_table(mixed, columns, names).astype({"Value": float, "Frame": float}),
                                  expected, check_exact=True)


def test_representative_frames_are_read_in_one_pass(tmp_path):
    data_dir = stored_chain(tmp_path, frames=60, atoms=5)
    coords = zarr.open(os.path.join(data_dir, "coords.zarr"), mode="r")["xyz"]
    wanted = [59, 0, 17, 17, 3, 60, -1, 1000]
    loaded = v2._load_frames(coords, wanted)
    assert sorted(loaded) == [0, 3, 17, 59]
    for index, xyz in loaded.items():
        assert same_bits(xyz, coords[index])
    assert v2._load_frames(coords, []) == {}


def test_cluster_representatives_do_not_depend_on_how_the_grid_search_is_run(monkeypatch):
    rng = np.random.default_rng(9)
    sizes = [40, 9, 3, 150, 25]                                     # with the threshold below: 150 gets a parallel grid
    centres = rng.normal(scale=6, size=(len(sizes), 10))
    data = np.concatenate([centre + rng.normal(size=(size, 10)) for centre, size in zip(centres, sizes)])
    labels = np.repeat(np.arange(len(sizes)), sizes)
    order = rng.permutation(len(labels))
    data, labels = data[order], labels[order]
    frame = pl.DataFrame({"frame": np.arange(len(labels)), **{f"PC{i + 1}": data[:, i] for i in range(10)}})

    sequential = embedding.create_cluster_info_fast(labels, frame, len(sizes), n_jobs=1)
    monkeypatch.setattr(embedding, "_GRID_PARALLEL_MIN_SIZE", 100)
    parallel = embedding.create_cluster_info_fast(labels, frame, len(sizes), n_jobs=2)
    assert json.dumps(parallel, sort_keys=True) == json.dumps(sequential, sort_keys=True)
    assert [entry["n_points"] for entry in sequential] == sorted(sizes, reverse=True)
    assert [entry["has_kde_center"] for entry in sequential] == [True, True, True, True, False]
