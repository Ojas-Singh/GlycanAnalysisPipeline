"""A glycan that cannot be read or archived stops the root files instead of being left out of them."""
from __future__ import annotations

import json
import zipfile

import pytest

from lib import storage as storage_module
from lib.glybake import BakeError, GlycoShapeBaker
from lib.storage import LocalStorageBackend, StorageManager


@pytest.fixture
def database(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_module, "_storage_manager", StorageManager(LocalStorageBackend()))
    for glycan_id in ("GS00001", "GS00002", "GS00003"):
        folder = tmp_path / glycan_id
        (folder / "output" / "level_1").mkdir(parents=True)
        (folder / "output" / "level_1" / "cluster_0.pdb").write_text(f"REMARK {glycan_id}\nEND\n")
        (folder / "data.json").write_text(json.dumps({"archetype": {"ID": glycan_id}}))
    return tmp_path


def test_complete_database_is_baked(database):
    baker = GlycoShapeBaker(database)
    baker.create_consolidated_json()
    baker.create_master_archive()
    assert sorted(json.loads((database / "GLYCOSHAPE.json").read_text())) == ["GS00001", "GS00002", "GS00003"]
    with zipfile.ZipFile(database / "GlycoShape.zip") as master:
        assert master.namelist() == ["GS00001.zip", "GS00002.zip", "GS00003.zip"]


@pytest.mark.parametrize("damage,reason", [
    (lambda folder: (folder / "data.json").write_text('{"archetype": '), "invalid JSON"),
    (lambda folder: (folder / "data.json").unlink(), "no data.json"),
    (lambda folder: (folder / "data.json").write_text("[1, 2]"), "not hold a JSON object"),
])
def test_unreadable_data_json_stops_the_inventory(database, damage, reason):
    damage(database / "GS00002")
    with pytest.raises(BakeError) as raised:
        GlycoShapeBaker(database).create_consolidated_json()
    assert [glycan_id for glycan_id, _ in raised.value.failures] == ["GS00002"]
    assert reason in raised.value.failures[0][1] and "GS00002" in str(raised.value)
    assert not (database / "GLYCOSHAPE.json").exists()


def test_failed_glycan_archive_stops_the_master_archive(database, monkeypatch):
    baker = GlycoShapeBaker(database)
    baker.create_master_archive()                                        # a complete master archive first
    stale = (database / "GS00002" / "GS00002.zip").read_bytes()
    (database / "GS00002" / "output" / "level_1" / "cluster_0.pdb").write_text("REMARK changed\nEND\n")
    original = GlycoShapeBaker.create_individual_glycan_archive
    monkeypatch.setattr(GlycoShapeBaker, "create_individual_glycan_archive",
                        lambda self, folder: False if folder.name == "GS00002" else original(self, folder))
    with pytest.raises(BakeError) as raised:
        baker.create_master_archive()
    assert [glycan_id for glycan_id, _ in raised.value.failures] == ["GS00002"]
    # neither the old master archive nor one with the stale archive of GS00002 is left behind
    assert not (database / "GlycoShape.zip").exists()
    assert (database / "GS00002" / "GS00002.zip").read_bytes() == stale


def test_bake_all_fails_on_a_partial_database(database):
    (database / "GS00003" / "data.json").write_text("{")
    with pytest.raises(BakeError):
        GlycoShapeBaker(database).bake_all(include_wurcs_submission=False)
