"""The processing step reads archive entries: same store as from the PDB text."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import zarr

from lib import archive
from lib.glypdbio import parse_mol2_bonds
from lib.glystore import store_from_archive, store_from_pdb, store_trajectory

from test_archive import NAME_G60371DN, NAME_G77147GI, make_source, mol2_text, random_coords

CHAIN = [("C1", "0GB", 1, "C"), ("C2", "0GB", 1, "C"), ("O2", "0GB", 1, "O"),
         ("C3", "0GB", 1, "C"), ("O3", "0GB", 1, "O"), ("H3O", "0GB", 1, "H")]


def chain_source(root, folder, frames=50, broken=(), **options):
    """A six-atom chain with plausible bond lengths, jittered from frame to frame."""
    rng = np.random.default_rng(11)
    base = np.cumsum([[0, 0, 0], [1520, 0, 0], [0, 1430, 0], [1430, 0, 0], [0, 1430, 0], [960, 0, 0]], axis=0)
    coords = base[None] + rng.integers(-40, 41, size=(frames, 6, 3))
    for frame in broken:
        coords[frame, 5] += (9000, 0, 0)                 # O3-H3O torn apart
    source = make_source(root, folder, coords, atoms=CHAIN, ter=False, **options)
    (source / "structure.mol2").write_text(mol2_text([1, 2, 3, 4, 5, 6]), encoding="utf-8")
    return source, coords


def store_files(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(Path(root).rglob("*")) if p.is_file()}


@pytest.fixture
def single_threaded_blosc():
    # Blosc lays out its blocks in thread-completion order; one thread makes
    # the compressed chunks reproducible so that whole stores can be compared.
    from numcodecs import blosc

    previous = blosc.use_threads
    blosc.use_threads = False
    yield
    blosc.use_threads = previous


@pytest.mark.parametrize("frames_buffer,frame_chunk", [(1, None), (128, 0), (4, 16), (7, 5), (1000, 50)])
def test_store_from_archive_equals_store_from_pdb(tmp_path, single_threaded_blosc, frames_buffer, frame_chunk):
    source, coords = chain_source(tmp_path / "data", "DGlcpa1-OH")
    entry = Path(archive.convert_entry(source, tmp_path / "archive")["entry"])
    options = dict(connectivity_kind="serial", frames_buffer=frames_buffer, frame_chunk=frame_chunk)
    store_from_pdb(str(source / "simulation.pdb"), str(tmp_path / "from_pdb"),
                   get_connectivity=parse_mol2_bonds(source / "structure.mol2"), **options)
    store_from_archive(str(entry), str(tmp_path / "from_archive"),
                       get_connectivity=parse_mol2_bonds(entry / "structure.mol2"), **options)

    a = zarr.open(str(tmp_path / "from_pdb" / "coords.zarr"), mode="r")["xyz"]
    b = zarr.open(str(tmp_path / "from_archive" / "coords.zarr"), mode="r")["xyz"]
    assert a.shape == b.shape == (50, 6, 3) and a.chunks == b.chunks and a.dtype == b.dtype == np.float32
    assert np.array_equal(a[:].view(np.uint32), b[:].view(np.uint32))          # bit for bit
    assert np.array_equal(a[:], (coords / 1000.0).astype(np.float32))
    for table in ("atoms.parquet", "bonds.parquet", "frames.parquet"):
        assert pl.read_parquet(tmp_path / "from_pdb" / table).equals(pl.read_parquet(tmp_path / "from_archive" / table))
    assert pl.read_parquet(tmp_path / "from_archive" / "atoms.parquet")["name"].to_list() == [a[0] for a in CHAIN]
    assert pl.read_parquet(tmp_path / "from_archive" / "bonds.parquet").rows() == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    assert store_files(tmp_path / "from_pdb") == store_files(tmp_path / "from_archive")   # every file, every byte


def test_store_trajectory_accepts_a_pdb_an_entry_or_its_xtc(tmp_path, single_threaded_blosc):
    source, _ = chain_source(tmp_path / "data", "DGlcpa1-OH")
    entry = Path(archive.convert_entry(source, tmp_path / "archive")["entry"])
    bonds = parse_mol2_bonds(source / "structure.mol2")
    for name, trajectory in (("pdb", source / "simulation.pdb"), ("entry", entry), ("xtc", entry / "trajectory.xtc")):
        store_trajectory(str(trajectory), str(tmp_path / name), get_connectivity=bonds,
                         connectivity_kind="serial", frames_buffer=16, frame_chunk=0)
    assert store_files(tmp_path / "pdb") == store_files(tmp_path / "entry") == store_files(tmp_path / "xtc")


def test_store_step_gives_the_same_result_for_both_inputs(tmp_path, single_threaded_blosc):
    from lib.v2 import GlycanAnalysisPipeline, step_store

    source, coords = chain_source(tmp_path / "data", "G77147GI", frames=60, broken=(7, 41), name_json=NAME_G77147GI)
    entry = Path(archive.convert_entry(source, tmp_path / "archive")["entry"])
    assert entry.name == NAME_G77147GI

    from_pdb = GlycanAnalysisPipeline(str(tmp_path / "data"), str(tmp_path / "process_pdb")).get_glycan_paths("G77147GI")
    from_archive = GlycanAnalysisPipeline(str(tmp_path / "archive"), str(tmp_path / "process_xtc")).get_glycan_paths(entry.name)
    assert from_pdb["pdb_path"].endswith("simulation.pdb") and from_archive["pdb_path"].endswith("trajectory.xtc")
    assert from_pdb["glycam_name"] == from_archive["glycam_name"] == NAME_G77147GI
    assert GlycanAnalysisPipeline(str(tmp_path / "archive"), str(tmp_path / "x")).validate_inputs(from_archive)

    results = [
        step_store(paths["input_glycan_dir"], paths["frame_data_dir"], paths["pdb_path"], paths["mol2_path"], force=True)
        for paths in (from_pdb, from_archive)
    ]
    assert results[0] == results[1]
    assert (results[0].source_frames, results[0].retained_frames, results[0].rejected_frames) == (60, 58, 2)
    assert store_files(from_pdb["frame_data_dir"]) == store_files(from_archive["frame_data_dir"])
    kept = zarr.open(os.path.join(from_archive["frame_data_dir"], "coords.zarr"), mode="r")["xyz"][:]
    assert np.array_equal(kept, (np.delete(coords, [7, 41], axis=0) / 1000.0).astype(np.float32))


def test_glycan_discovery_lists_archive_entries(tmp_path):
    from lib.storage import get_storage_manager
    from main import GlycanPipelineRunner

    source, _ = chain_source(tmp_path / "data", "G60371DN", info_json=NAME_G60371DN)
    make_source(tmp_path / "data", "DGlcpa1-OH", random_coords(3, 12))
    for folder in ("G60371DN", "DGlcpa1-OH"):
        archive.convert_entry(tmp_path / "data" / folder, tmp_path / "archive")
    archive.write_index(tmp_path / "archive")

    runner = GlycanPipelineRunner.__new__(GlycanPipelineRunner)
    runner.storage = get_storage_manager()
    runner.data_dir = tmp_path / "archive"
    assert runner.discover_glycans() == sorted(["DGlcpa1-OH", archive.entry_folder_name(NAME_G60371DN)])
    runner.data_dir = tmp_path / "data"
    assert runner.discover_glycans() == ["DGlcpa1-OH", "G60371DN"]
