"""A torsion table computed with earlier torsion definitions is computed again, never relabelled.

Its columns are named after their four atoms, so a table of the former definitions (psi with
C(x+1), omega C4-C5-C6-O6) has other column names than the current ones give.  Metadata of the
current torsions over those values would describe atoms that the values were not measured on.
"""
from __future__ import annotations

import polars as pl
import pytest

from lib import v2
from synthetic_glycan import GLYCAM_NAME, write_source


@pytest.fixture
def analysed(tmp_path):
    source = write_source(tmp_path / "data", frames=40)
    pipeline = v2.GlycanAnalysisPipeline(str(tmp_path / "data"), str(tmp_path / "process"))
    paths = pipeline.get_glycan_paths(GLYCAM_NAME)
    v2.step_store(GLYCAM_NAME, paths["frame_data_dir"], paths["pdb_path"], paths["mol2_path"], force=True)
    torsion_list, labels = v2.step_torsions(paths["frame_data_dir"], paths["embedding_dir"], force=True)
    return paths, torsion_list, labels, pipeline


def psi_label(labels):
    # Man(a1-2)Man: residue 3 on O2 of residue 2; psi is C1'-O2-C2-C1
    return next(label for label in labels if label.startswith("C1_3-O2_2-C2_2-"))


def test_psi_of_the_example_is_the_iupac_one(analysed):
    _, _, labels, _ = analysed
    assert psi_label(labels) == "C1_3-O2_2-C2_2-C1_2"


def make_stale(paths, labels):
    """Rename the psi column as the former definitions could have named it (C(x+1))."""
    table_path = v2._torsions_parquet_path(paths["embedding_dir"])
    table = pl.read_parquet(table_path)
    old = psi_label(labels)
    table.rename({old: "C1_3-O2_2-C2_2-C3_2"}).write_parquet(table_path)


def test_a_stale_table_is_recognised_and_computed_again(analysed):
    paths, _, labels, pipeline = analysed
    assert v2.torsion_table_is_current(paths["frame_data_dir"], paths["embedding_dir"])
    assert pipeline.get_analysis_status(GLYCAM_NAME)["steps_status"]["torsions"]
    make_stale(paths, labels)
    assert not v2.torsion_table_is_current(paths["frame_data_dir"], paths["embedding_dir"])
    # an incremental run takes the entry as unfinished
    assert not pipeline.get_analysis_status(GLYCAM_NAME)["steps_status"]["torsions"]
    # an incremental torsion step does not skip it
    torsion_list, new_labels = v2.step_torsions(paths["frame_data_dir"], paths["embedding_dir"], force=False)
    assert torsion_list and new_labels == labels
    header = pl.read_parquet(v2._torsions_parquet_path(paths["embedding_dir"])).columns
    assert psi_label(header[1:]) == "C1_3-O2_2-C2_2-C1_2"
    assert v2.torsion_table_is_current(paths["frame_data_dir"], paths["embedding_dir"])


def test_metadata_is_not_built_over_a_stale_table(analysed):
    paths, _, labels, _ = analysed
    make_stale(paths, labels)
    stale = [label if label != psi_label(labels) else "C1_3-O2_2-C2_2-C3_2" for label in labels]
    with pytest.raises(ValueError, match="earlier definitions"):
        v2._load_or_build_torsion_metadata(paths["frame_data_dir"], paths["embedding_dir"], stale)


def test_metadata_of_a_current_table_is_structural(analysed):
    paths, _, labels, _ = analysed
    metadata = v2._load_or_build_torsion_metadata(paths["frame_data_dir"], paths["embedding_dir"], labels)
    glycosidic = {e["canonical_name"]: e["atom_names"] for e in metadata if e["family"] == "glycosidic"}
    assert glycosidic == {"3_2_phi": ["O5", "C1", "O2", "C2"], "3_2_psi": ["C1", "O2", "C2", "C1"]}
