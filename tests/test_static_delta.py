"""The root files of a static database: written by the finalization, and derived for one changed glycan.

``lib/glybake.py`` (archives, inventory, missing reports, manifest), ``lib/json2rdf.py`` and
``lib/static_delta.py``.  The reference of every comparison is the finalization of ``main.py`` on the
same tree (``static_tree.full_bake``).  No test reaches the network.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from lib import glybake, static_delta, static_publish
from lib.glybake import GlycoShapeBaker
from lib.static_delta import ADD, REPLACE, WITHDRAW, DeltaError

import fake_services
from s3_fakes import reader_validate
from static_tree import full_bake, glycan_data, manifest_without_times, root_files, write_glycan, write_tree

pytestmark = pytest.mark.usefixtures("no_internet")
REPO_ROOT = Path(__file__).resolve().parents[1]
DERIVED = static_delta.DERIVED_ROOT_FILES


@pytest.fixture
def offline(monkeypatch):
    """No service answers and there is no crabWURCS: the reports follow from the inventory alone."""
    return fake_services.install(monkeypatch, online=False, crabwurcs=False)


@pytest.fixture
def baked(tmp_path, offline):
    """A baked tree of three glycans and a directory holding what the storage would hold of it."""
    tree = write_tree(tmp_path / "static")
    write_glycan(tree, "GS00002", name="DGalpb1-3DGlcpb1-OH", archive=False,
                 alpha={"glytoucan": None}, beta={"glytoucan": None, "wurcs": None})   # one WURCS to register, one unresolved
    full_bake(tree)
    return tree


def stored_copy(tree: Path, into: Path) -> Path:
    """The manifest and the root files a change needs, as downloaded from the storage; no glycan folder."""
    into.mkdir(parents=True)
    manifest = json.loads((tree / "sync_manifest.json").read_bytes())
    for name in ["sync_manifest.json"] + static_delta.needed_root_files(manifest):
        shutil.copyfile(tree / name, into / name)
    return into


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Archives
# ---------------------------------------------------------------------------
def test_an_archive_is_a_function_of_the_names_and_contents_of_its_members(tmp_path):
    members = {"data.json": b'{"a": 1}', "PDB/alpha.pdb": b"ATOM\n" * 100, "dist.svg": b"<svg/>"}
    (tmp_path / "file").write_bytes(members["dist.svg"])
    glybake.write_archive(tmp_path / "a.zip", members.items())
    glybake.write_archive(tmp_path / "b.zip", [
        ("dist.svg", tmp_path / "file"), ("PDB/alpha.pdb", lambda: members["PDB/alpha.pdb"]), ("data.json", members["data.json"])])
    assert (tmp_path / "a.zip").read_bytes() == (tmp_path / "b.zip").read_bytes()      # any order, any source
    with zipfile.ZipFile(tmp_path / "a.zip") as archive:
        assert archive.namelist() == ["PDB/alpha.pdb", "data.json", "dist.svg"] and archive.testzip() is None
        for info in archive.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0) and info.compress_type == zipfile.ZIP_DEFLATED
            assert info.external_attr >> 16 == 0o100644 and info.create_system == 3 and not info.extra
            assert archive.read(info) == members[info.filename]
    assert glybake.archive_holds(tmp_path / "a.zip", members)
    assert not glybake.archive_holds(tmp_path / "a.zip", {**members, "data.json": b'{"a": 2}'})
    assert not glybake.archive_holds(tmp_path / "a.zip", {name: members[name] for name in ("data.json", "dist.svg")})
    assert not glybake.archive_holds(tmp_path / "missing.zip", members)
    (tmp_path / "broken.zip").write_bytes(b"not a zip")
    assert not glybake.archive_holds(tmp_path / "broken.zip", members)
    with pytest.raises(ValueError, match="named twice"):
        glybake.write_archive(tmp_path / "c.zip", [("x", b"1"), ("x", b"2")])
    assert sorted(path.name for path in tmp_path.iterdir()) == ["a.zip", "b.zip", "broken.zip", "file"]   # no temporary file


def test_the_archive_of_a_glycan_does_not_depend_on_file_times_or_modes(tmp_path):
    folder = write_glycan(tmp_path / "one", "GS00007", archive=False)
    other = tmp_path / "two" / "GS00007"
    shutil.copytree(folder, other)
    for path in other.rglob("*"):
        os.utime(path, (1_500_000_000, 1_500_000_000))
        if path.is_file():
            path.chmod(0o600)
    assert GlycoShapeBaker(tmp_path / "one").create_individual_glycan_archive(folder)
    assert GlycoShapeBaker(tmp_path / "two").create_individual_glycan_archive(other)
    first = (folder / "GS00007.zip").read_bytes()
    assert first == (other / "GS00007.zip").read_bytes()
    with zipfile.ZipFile(folder / "GS00007.zip") as archive:                # what it has always held: Level 1 only
        assert archive.namelist() == ["GLYCAM/alpha.pdb", "GLYCAM/beta.pdb", "PDB/alpha.pdb", "PDB/beta.pdb",
                                      "data.json", "dist.svg", "glycosmos.json", "glygen.json", "snfg.svg"]
        assert archive.read("dist.svg") == b"<svg>level 1</svg>"


def test_an_archive_that_already_holds_the_files_is_kept_as_it_is(tmp_path):
    """An archive written before archives were reproducible stays what it is until its content changes."""
    folder = write_glycan(tmp_path, "GS00007", archive=False)
    with zipfile.ZipFile(folder / "GS00007.zip", "w", zipfile.ZIP_DEFLATED) as archive:   # as the baker wrote it before
        for name, path in glybake.glycan_archive_members(folder):
            archive.write(path, name)
    published = (folder / "GS00007.zip").read_bytes()
    baker = GlycoShapeBaker(tmp_path)
    assert baker.create_individual_glycan_archive(folder) and (folder / "GS00007.zip").read_bytes() == published
    (folder / "data.json").write_text('{"changed": true}')
    assert baker.create_individual_glycan_archive(folder)
    rewritten = (folder / "GS00007.zip").read_bytes()
    assert rewritten != published
    with zipfile.ZipFile(folder / "GS00007.zip") as archive:
        assert archive.read("data.json") == b'{"changed": true}'
        assert {info.date_time for info in archive.infolist()} == {(1980, 1, 1, 0, 0, 0)}
    (folder / "GS00007.zip").unlink()
    assert baker.create_individual_glycan_archive(folder) and (folder / "GS00007.zip").read_bytes() == rewritten
    assert not baker.create_individual_glycan_archive(tmp_path / "GS99999")


def test_the_finalization_is_a_function_of_the_tree(tmp_path, offline):
    """Two trees with the same files, written at other times and listed in another order, give the same root files."""
    first = write_tree(tmp_path / "first", ("GS00003", "GS00001", "GS00002"))
    second = write_tree(tmp_path / "second", ("GS00002", "GS00003", "GS00001"))
    for glycan_id in ("GS00001", "GS00002", "GS00003"):                    # the same content in both
        shutil.rmtree(second / glycan_id)
        shutil.copytree(first / glycan_id, second / glycan_id)
    for path in second.rglob("*"):
        os.utime(path, (1_500_000_000, 1_500_000_000))
    full_bake(first)
    full_bake(second)
    full_bake(second)                                                       # and again in place
    one, two = root_files(first), root_files(second)
    assert sorted(one) == ["GLYCOSHAPE.json", "GLYCOSHAPE_RDF.ttl", "GlycoShape.zip", "missing_glytoucan.txt",
                           "missing_glytoucan_no_wurcs_gags.tsv", "missing_wurcs.txt", "sync_manifest.json"]
    for name in one:
        if name != "sync_manifest.json":
            assert one[name] == two[name], name
    assert manifest_without_times(one["sync_manifest.json"]) == manifest_without_times(two["sync_manifest.json"])
    assert one["sync_manifest.json"] != two["sync_manifest.json"]           # the times are not a function of the tree
    with zipfile.ZipFile(first / "GlycoShape.zip") as master:
        assert master.namelist() == ["GS00001.zip", "GS00002.zip", "GS00003.zip"]
        assert master.read("GS00002.zip") == (first / "GS00002" / "GS00002.zip").read_bytes()
        assert {info.date_time for info in master.infolist()} == {(1980, 1, 1, 0, 0, 0)}


def test_the_rdf_file_is_the_same_in_another_process(tmp_path, offline):
    tree = write_tree(tmp_path / "static")
    full_bake(tree)
    here = (tree / "GLYCOSHAPE_RDF.ttl").read_bytes()
    assert b"gsr:GS00001" in here and b"_:" not in here                     # blank nodes are written inline
    program = (
        "import sys; sys.path.insert(0, sys.argv[1])\n"
        "import logging; logging.disable(logging.CRITICAL)\n"
        "from lib.json2rdf import convert_glycoshape_to_rdf\n"
        "assert convert_glycoshape_to_rdf(sys.argv[2], sys.argv[3]) is not None\n")
    for seed in ("1", "4242"):                                              # string hashes differ between the runs
        environment = {**os.environ, "PYTHONHASHSEED": seed, "GLYCOSHAPE_STORAGE": "local"}
        done = subprocess.run([sys.executable, "-c", program, str(REPO_ROOT), str(tree / "GLYCOSHAPE.json"),
                               str(tmp_path / f"rdf-{seed}.ttl")], env=environment, capture_output=True, text=True, timeout=300)
        assert done.returncode == 0, done.stderr[-2000:]
        assert (tmp_path / f"rdf-{seed}.ttl").read_bytes() == here


def test_the_shared_functions_write_what_the_baker_writes(baked):
    inventory = json.loads((baked / "GLYCOSHAPE.json").read_bytes())
    assert glybake.inventory_bytes(dict(reversed(list(inventory.items())))) == (baked / "GLYCOSHAPE.json").read_bytes()
    manifest = json.loads((baked / "sync_manifest.json").read_bytes())
    rebuilt = glybake.build_manifest(manifest["root_files"], manifest["glycans"], generated_at=manifest["generated_at"])
    assert glybake.manifest_bytes(rebuilt) == (baked / "sync_manifest.json").read_bytes()
    assert static_delta.folder_entry(baked / "GS00002") == manifest["glycans"]["GS00002"]
    missing, rows = glybake.missing_reports(inventory)                      # from the inventory alone
    assert glybake.missing_glytoucan_text(missing) == (baked / "missing_glytoucan.txt").read_text() == "WURCS=2.0/2/[a]\n"
    assert glybake.missing_wurcs_text(rows) == (baked / "missing_wurcs.txt").read_text() == (
        "glycoshape_id\tvariant\tiupac\tglycam\nGS00002\tbeta\tMan(a1-2)Man\tDGalpb1-3DGlcpb1-OH\n")


# ---------------------------------------------------------------------------
# One changed glycan against the finalization of the changed tree
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("action, glycan_id", [(ADD, "GS00003"), (ADD, "GS00000"), (ADD, "GS00009"),
                                               (REPLACE, "GS00002"), (WITHDRAW, "GS00001"), (WITHDRAW, "GS00002")])
def test_the_derived_root_files_are_those_of_a_full_bake(baked, tmp_path, action, glycan_id):
    stored = stored_copy(baked, tmp_path / "stored")
    folder = None
    if action != WITHDRAW:                                                  # the new folder, made somewhere else
        folder = write_glycan(tmp_path / "new", glycan_id, name="DManpa1-6DManpa1-OH", seed="new",
                              alpha={"glytoucan": None}, beta={"wurcs": None, "glytoucan": ""})
    before = {path: path.read_bytes() for path in stored.iterdir()}
    report = static_delta.apply_change(action, glycan_id, stored, tmp_path / "out", folder,
                                       calculate=glybake.WurcsCalculator(), resolve=glybake.resolve_missing_accessions)

    if (baked / glycan_id).exists():                                        # the same change of the tree, then a full bake
        shutil.rmtree(baked / glycan_id)
    if folder is not None:
        shutil.copytree(folder, baked / glycan_id)
    full_bake(baked)
    derived, reference = root_files(tmp_path / "out"), root_files(baked)
    assert sorted(derived) == sorted(DERIVED) + ["sync_manifest.json"]
    for name in DERIVED:
        assert derived[name] == reference[name], name                       # byte for byte
    assert manifest_without_times(derived["sync_manifest.json"]) == manifest_without_times(reference["sync_manifest.json"])
    assert {path: path.read_bytes() for path in stored.iterdir()} == before   # what is stored is only read

    manifest = json.loads(derived["sync_manifest.json"])
    assert report["global_hash"] == manifest["global_hash"] == json.loads(reference["sync_manifest.json"])["global_hash"]
    assert report["base_global_hash"] == json.loads(before[stored / "sync_manifest.json"])["global_hash"]
    assert report["glycans"] == len(manifest["glycans"]) and (glycan_id in manifest["glycans"]) == (action != WITHDRAW)
    assert report["manifest_sha256"] == sha256(derived["sync_manifest.json"])
    reader_validate(manifest)                                               # the website accepts it
    assert static_publish.validate_manifest(manifest) == []
    assert report["root_files"]["GLYCOSHAPE.json"]["changed"] and report["root_files"]["GlycoShape.zip"]["changed"]
    assert report["master_archive"] == {"members": len(manifest["glycans"]), "repaired": [], "dropped": []}


def test_only_the_changed_entries_of_the_manifest_change(baked, tmp_path):
    stored = stored_copy(baked, tmp_path / "stored")
    old = json.loads((stored / "sync_manifest.json").read_bytes())
    folder = write_glycan(tmp_path / "new", "GS00003", seed="new")
    report = static_delta.apply_change(ADD, "GS00003", stored, tmp_path / "out", folder, generated_at="2026-10-11T00:00:00Z")
    new = report["manifest"]
    assert new["generated_at"] == "2026-10-11T00:00:00Z" and new["glycan_count"] == 4
    for glycan_id in old["glycans"]:
        assert new["glycans"][glycan_id] == old["glycans"][glycan_id]       # times included: the stored entries
    assert list(new["glycans"]) == sorted(new["glycans"])
    # root files that the change does not touch keep their entry; so does a report whose content stayed
    assert new["root_files"]["missing_glytoucan_no_wurcs_gags.tsv"] == old["root_files"]["missing_glytoucan_no_wurcs_gags.tsv"]
    assert new["root_files"]["missing_wurcs.txt"] == old["root_files"]["missing_wurcs.txt"]
    assert not report["root_files"]["missing_wurcs.txt"]["changed"] and not report["root_files"]["missing_glytoucan.txt"]["changed"]
    assert new["root_files"]["GLYCOSHAPE.json"]["sha256"] != old["root_files"]["GLYCOSHAPE.json"]["sha256"]
    assert new["glycans"]["GS00003"]["files"]["data.json"]["path"] == "GS00003/data.json"
    assert new["glycans"]["GS00003"]["zip_sha256"] == sha256((folder / "GS00003.zip").read_bytes())
    assert report["entry"] == {key: new["glycans"]["GS00003"][key] for key in ("hash", "file_count", "total_size")}
    # deriving it again gives the same files (and, with the same time, the same manifest but for the file times)
    again = static_delta.apply_change(ADD, "GS00003", stored, tmp_path / "again", folder, generated_at="2026-10-11T00:00:00Z")
    assert again["global_hash"] == report["global_hash"]
    for name in DERIVED:
        assert (tmp_path / "again" / name).read_bytes() == (tmp_path / "out" / name).read_bytes()


def test_changes_can_follow_each_other_without_a_full_bake(baked, tmp_path):
    """Each change is derived from the result of the one before; the end is the full bake of the final tree."""
    stored = stored_copy(baked, tmp_path / "stored-0")
    steps = [(ADD, "GS00004", "x"), (REPLACE, "GS00001", "y"), (WITHDRAW, "GS00005", ""), (REPLACE, "GS00004", "z"),
             (ADD, "GS00005", "back")]
    for number, (action, glycan_id, seed) in enumerate(steps, start=1):
        folder = None if action == WITHDRAW else write_glycan(tmp_path / f"new-{number}", glycan_id, seed=seed)
        static_delta.apply_change(action, glycan_id, stored, tmp_path / f"out-{number}", folder)
        stored = stored_copy(tmp_path / f"out-{number}", tmp_path / f"stored-{number}")
        if (baked / glycan_id).exists():
            shutil.rmtree(baked / glycan_id)
        if folder is not None:
            shutil.copytree(folder, baked / glycan_id)
    full_bake(baked)
    derived, reference = root_files(tmp_path / f"out-{len(steps)}"), root_files(baked)
    for name in DERIVED:
        assert derived[name] == reference[name], name
    assert manifest_without_times(derived["sync_manifest.json"]) == manifest_without_times(reference["sync_manifest.json"])


# ---------------------------------------------------------------------------
# The missing reports
# ---------------------------------------------------------------------------
def inventory_of(*entries):
    return {glycan_id: glycan_data(glycan_id, "DManpa1-2DManpa1-OH", **variants) for glycan_id, variants in entries}


def test_the_decisions_of_the_last_finalization_stand_for_the_other_glycans():
    before = inventory_of(
        ("GS00001", {"alpha": {"glytoucan": None}}),                        # listed
        ("GS00002", {"alpha": {"glytoucan": None}}),                        # GlyTouCan knew it by then: not listed
        ("GS00003", {"beta": {"glytoucan": None, "wurcs": None}}),          # a WURCS was calculated for it: no row
        ("GS00004", {"beta": {"glytoucan": None, "wurcs": None}}),          # unresolved: a row
    )
    stored_missing = "WURCS=2.0/1/[a]\nWURCS=2.0/calculated-for-GS00003\n"
    stored_rows = "glycoshape_id\tvariant\tiupac\tglycam\nGS00004\tbeta\tMan(a1-2)Man\tDManpa1-2DManpb1-OH\n"
    after = dict(before)
    after["GS00000"] = glycan_data("GS00000", "DManpa1-2DManpa1-OH", archetype={"glytoucan": None},
                                   alpha={"glytoucan": None, "wurcs": None})
    after = dict(sorted(after.items()))
    missing, rows = static_delta.change_missing_reports(stored_missing, stored_rows, before, after, "GS00000")
    assert missing == "WURCS=2.0/0/[u]\nWURCS=2.0/1/[a]\nWURCS=2.0/calculated-for-GS00003\n"   # inventory order, then the unexplained
    assert rows == ("glycoshape_id\tvariant\tiupac\tglycam\nGS00000\talpha\tMan(a1-2)Man\tDManpa1-2DManpa1-OH\n"
                    "GS00004\tbeta\tMan(a1-2)Man\tDManpa1-2DManpb1-OH\n")
    # taking the glycan out again gives the stored reports back
    assert static_delta.change_missing_reports(missing, rows, after, before, "GS00000") == (stored_missing, stored_rows)
    # a replaced glycan loses its old lines and gets the new ones; a WURCS it shares with another glycan stays
    replaced = dict(before)
    replaced["GS00004"] = glycan_data("GS00004", "DManpa1-2DManpa1-OH", alpha={"glytoucan": None, "wurcs": "WURCS=2.0/1/[a]"},
                                      beta={"glytoucan": None, "wurcs": "WURCS=2.0/4/[b]"})
    missing, rows = static_delta.change_missing_reports(stored_missing, stored_rows, before, replaced, "GS00004")
    assert missing == "WURCS=2.0/1/[a]\nWURCS=2.0/4/[b]\nWURCS=2.0/calculated-for-GS00003\n"
    assert rows == "glycoshape_id\tvariant\tiupac\tglycam\n"
    withdrawn = {key: value for key, value in before.items() if key != "GS00001"}
    assert static_delta.change_missing_reports(stored_missing, stored_rows, before, withdrawn, "GS00001")[0] == (
        "WURCS=2.0/calculated-for-GS00003\n")
    # without stored reports the inventory decides alone
    assert static_delta.change_missing_reports(None, None, before, before, "GS00009") == (
        "WURCS=2.0/1/[a]\nWURCS=2.0/2/[a]\n",
        "glycoshape_id\tvariant\tiupac\tglycam\nGS00003\tbeta\tMan(a1-2)Man\tDManpa1-2DManpb1-OH\n"
        "GS00004\tbeta\tMan(a1-2)Man\tDManpa1-2DManpb1-OH\n")


def test_the_changed_glycan_is_decided_with_the_services_when_they_are_given():
    before = inventory_of(("GS00001", {}))
    after = dict(before)
    after["GS00002"] = glycan_data("GS00002", "DManpa1-2DManpa1-OH", archetype={"glytoucan": None},
                                   alpha={"glytoucan": None}, beta={"glytoucan": None, "wurcs": None})
    asked = []

    def calculate(glycan_id, variant_name, iupac):
        asked.append((glycan_id, variant_name))
        return None, "WURCS=2.0/calculated"

    def resolve(candidates):
        asked.append(tuple(candidates))
        return {"WURCS=2.0/2/[u]": "G00002ZZ"}                              # registered by now

    missing, rows = static_delta.change_missing_reports("", "glycoshape_id\tvariant\tiupac\tglycam\n", before, after,
                                                        "GS00002", calculate=calculate, resolve=resolve)
    assert missing == "WURCS=2.0/2/[a]\nWURCS=2.0/calculated\n" and rows == "glycoshape_id\tvariant\tiupac\tglycam\n"
    assert asked == [("GS00002", "beta"), ("WURCS=2.0/2/[u]", "WURCS=2.0/2/[a]", "WURCS=2.0/calculated")]


# ---------------------------------------------------------------------------
# What is stored is checked before anything is derived from it
# ---------------------------------------------------------------------------
def test_a_stored_archive_that_is_not_the_one_of_the_manifest_is_fetched_or_refused(baked, tmp_path):
    stored = stored_copy(baked, tmp_path / "stored")
    manifest = json.loads((stored / "sync_manifest.json").read_bytes())
    expected = {glycan_id: entry["zip_sha256"] for glycan_id, entry in manifest["glycans"].items()}
    real = {glycan_id: (baked / glycan_id / f"{glycan_id}.zip").read_bytes() for glycan_id in expected}
    glybake.write_archive(tmp_path / "damaged.zip", [                       # one member stale, one missing, one unknown
        ("GS00001.zip", b"an older archive"), ("GS00005.zip", real["GS00005"]), ("GS00777.zip", b"withdrawn long ago")])
    new = (write_glycan(tmp_path / "new", "GS00003") / "GS00003.zip").read_bytes()
    with pytest.raises(DeltaError, match="does not hold the archive that the stored manifest records for GS00001"):
        static_delta.change_master_archive(tmp_path / "damaged.zip", tmp_path / "out.zip", ADD, "GS00003", new, expected)
    assert not (tmp_path / "out.zip").exists() and not (tmp_path / "out.zip.tmp").exists()
    fetched = []

    def fetch(glycan_id):
        fetched.append(glycan_id)
        return real[glycan_id]

    report = static_delta.change_master_archive(
        tmp_path / "damaged.zip", tmp_path / "out.zip", ADD, "GS00003", new, expected, fetch)
    assert report == {"members": 4, "repaired": ["GS00001", "GS00002"], "dropped": ["GS00777.zip"]} and fetched == report["repaired"]
    static_delta.change_master_archive(stored / "GlycoShape.zip", tmp_path / "reference.zip", ADD, "GS00003", new, expected)
    assert (tmp_path / "out.zip").read_bytes() == (tmp_path / "reference.zip").read_bytes()
    with pytest.raises(DeltaError, match="and neither does its folder"):
        static_delta.change_master_archive(tmp_path / "damaged.zip", tmp_path / "x.zip", ADD, "GS00003", new, expected, lambda _: b"other")
    with pytest.raises(DeltaError, match="records no archive for GS00002"):
        static_delta.change_master_archive(stored / "GlycoShape.zip", tmp_path / "x.zip", ADD, "GS00003", new, {**expected, "GS00002": None})
    with pytest.raises(DeltaError, match="the archive of GS00003 is missing"):
        static_delta.change_master_archive(stored / "GlycoShape.zip", tmp_path / "x.zip", ADD, "GS00003", None, expected)


def test_a_change_that_cannot_be_derived_is_refused_before_anything_is_written(baked, tmp_path):
    stored = stored_copy(baked, tmp_path / "stored")
    folder = write_glycan(tmp_path / "new", "GS00003")
    out = tmp_path / "out"

    def refused(reason, action=ADD, glycan_id="GS00003", glycan_dir=folder, stored_dir=stored):
        with pytest.raises(DeltaError, match=reason):
            static_delta.apply_change(action, glycan_id, stored_dir, out, glycan_dir)
        assert not out.exists()

    refused("GS00002 is already in the stored GLYCOSHAPE.json", glycan_id="GS00002",
            glycan_dir=write_glycan(tmp_path / "other", "GS00002"))
    refused("GS00003 is not in the stored GLYCOSHAPE.json", action=REPLACE)
    refused("GS00003 is not in the stored GLYCOSHAPE.json", action=WITHDRAW, glycan_dir=None)
    refused("the folder of GS00004 is needed, named GS00004", glycan_id="GS00004")
    refused("the folder of GS00003 is needed", glycan_dir=None)
    refused("unknown action 'publish'", action="publish")
    incomplete = write_glycan(tmp_path / "incomplete", "GS00003", archive=False)
    refused("the folder of GS00003 is not complete", glycan_dir=incomplete)

    damaged = tmp_path / "damaged"
    shutil.copytree(stored, damaged)
    (damaged / "GLYCOSHAPE.json").write_bytes((stored / "GLYCOSHAPE.json").read_bytes() + b" ")   # replaced in a publication
    refused("the stored GLYCOSHAPE.json is not the file the stored manifest describes", stored_dir=damaged)
    (damaged / "GLYCOSHAPE.json").unlink()
    refused("the stored GLYCOSHAPE.json cannot be read", stored_dir=damaged)
    (damaged / "sync_manifest.json").write_bytes(b'{"schema_version": 2}')
    refused("would be refused by the reader", stored_dir=damaged)

    # the folder of another glycan under this name
    wrong = tmp_path / "wrong" / "GS00003"
    shutil.copytree(tmp_path / "other" / "GS00002", wrong)
    (wrong / "GS00002.zip").rename(wrong / "GS00003.zip")
    with pytest.raises(DeltaError, match="data.json is not the one of GS00003"):
        static_delta.apply_change(ADD, "GS00003", stored, tmp_path / "out-wrong", wrong)


def test_the_last_glycan_cannot_be_withdrawn(tmp_path, offline):
    tree = write_tree(tmp_path / "static", ("GS00001",))
    full_bake(tree)
    stored = stored_copy(tree, tmp_path / "stored")
    with pytest.raises(DeltaError, match="the last glycan of a database cannot be withdrawn"):
        static_delta.apply_change(WITHDRAW, "GS00001", stored, tmp_path / "out")
