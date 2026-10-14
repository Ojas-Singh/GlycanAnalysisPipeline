"""Class keywords of ``search_meta`` (N-glycan, O-glycan, GAG and the N-glycan types) from the residue tree.

The reference is the rule-based classification of the 1,001 published entries in the thesis
(``tests/fixtures/glycan_classes_reference.json``): the keywords of every entry follow from its class
there, read from its WURCS and, where the name can be read, from its IUPAC-condensed name.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from lib import glyclass
from lib import storage as storage_module
from lib.glymeta import GlycanMetadataProcessor
from lib.storage import LocalStorageBackend, StorageManager

REFERENCE = json.loads((Path(__file__).parent / "fixtures" / "glycan_classes_reference.json").read_text())
ENTRIES = {e["id"]: e for e in REFERENCE["entries"]}
# IUPAC fields of the inventory that are not IUPAC-condensed names (problem list, section 1)
BROKEN_IUPAC = {"GS00430", "GS00728", "GS00954"}


def expected_keywords(entry: dict) -> list:
    """The class keywords that follow from the thesis class of an entry, in GAP's vocabulary."""
    if entry["class"] == "gag":
        return ["GAG"]
    if entry["class"] == "o_galnac":
        return ["O-Glycan"]
    if entry["class"] != "n_glycan":
        return []
    out = {"N-Glycan"}
    if entry["core_fucose"]:
        out.add("Fucosylated N-Glycan")
    out |= {"oligomannose": {"Oligomannose"}, "hybrid": {"Hybrid N-Glycan"}}.get(entry["subclass"], set())
    if entry["subclass"] == "complex":
        out.add("Complex N-Glycan")
        out |= {2: {"Biantennary N-Glycan"}, 3: {"Triantennary N-Glycan"}, 4: {"Tetraantennary N-Glycan"}}.get(
            entry["antennae"], set())
    return sorted(out)


@pytest.fixture
def processor(tmp_path):
    return GlycanMetadataProcessor(tmp_path)


# --- the cases of the audit -----------------------------------------------------------------------------------
@pytest.mark.parametrize("iupac", [
    "GalNAc(a1-3)[Neu5Ac(a2-6)]GalNAc",                    # core 5 with a sialic acid at O-6 (GS00003)
    "Neu5Ac(a2-6)GalNAc",                                  # sialyl-Tn
    "GlcNAc(b1-6)GalNAc",                                  # core 6
    "Gal(a1-3)GalNAc",                                     # core 8
    "Gal(b1-3)GalNAc",                                     # core 1
    "Neu5Ac(a2-3)Gal(b1-4)GlcNAc(b1-6)[Neu5Ac(a2-3)Gal(b1-3)]GalNAc",   # core 2
])
def test_galnac_at_the_reducing_end_is_an_o_glycan(processor, iupac):
    assert processor.get_glycan_keywords(iupac) == ["O-Glycan"]


@pytest.mark.parametrize("iupac", [
    "Gal(b1-3)GalNAc(b1-4)[Neu5Ac(a2-3)]Gal(b1-4)Glc",     # GM1a headgroup
    "GalNAc(b1-4)[Neu5Ac(a2-3)]Gal(b1-4)Glc",              # GM2
    "GalNAc(b1-3)Gal(a1-4)Gal(b1-4)Glc",                   # globoside Gb4
    "Fuc(a1-2)[Gal(a1-3)]Gal(b1-3)GalNAc(b1-3)Gal(a1-4)Gal(b1-4)Glc",   # GS00103, globo series
    "GalNAc",                                              # a monosaccharide has no class
])
def test_glycolipid_headgroups_are_not_o_glycans(processor, iupac):
    assert processor.get_glycan_keywords(iupac) == []


def test_core_fucose_a1_3_n_glycan(processor):
    iupac = ("Gal(b1-4)GlcNAc(b1-2)Man(a1-3)[Gal(b1-4)GlcNAc(b1-2)Man(a1-6)]"
             "Man(b1-4)GlcNAc(b1-4)[Fuc(a1-3)]GlcNAc")      # GS00402
    assert processor.get_glycan_keywords(iupac) == [
        "Biantennary N-Glycan", "Complex N-Glycan", "Fucosylated N-Glycan", "N-Glycan"]


def test_substituted_core_fucose_a1_6_n_glycan(processor):
    iupac = "GlcNAc(b1-2)Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)[Gal(b1-4)Fuc(a1-6)]GlcNAc"   # GS00860
    assert processor.get_glycan_keywords(iupac) == ["Complex N-Glycan", "Fucosylated N-Glycan", "N-Glycan"]


def test_n_glycan_whose_iupac_field_is_not_a_name_is_read_from_its_wurcs(processor):
    entry = ENTRIES["GS00728"]
    assert entry["iupac"] == "G60371D-N"
    assert processor.get_glycan_keywords(entry["iupac"]) == []
    keywords = processor.get_glycan_keywords(entry["iupac"], wurcs=entry["wurcs"])
    assert "N-Glycan" in keywords and keywords == expected_keywords(entry)


def test_antennae_are_counted_on_both_arms(processor):
    # GlcNAc(b1-6) on the a1-6 arm: triantennary, not biantennary (GS00062)
    iupac = ("Neu5Ac(a2-3)Gal(b1-4)GlcNAc(b1-6)[Gal(b1-4)GlcNAc(b1-2)]Man(a1-6)[Gal(b1-4)GlcNAc(b1-2)Man(a1-3)]"
             "Man(b1-4)GlcNAc(b1-4)GlcNAc")
    assert processor.get_glycan_keywords(iupac) == ["Complex N-Glycan", "N-Glycan", "Triantennary N-Glycan"]


def test_trimannosyl_core_is_not_oligomannose(processor):
    assert processor.get_glycan_keywords("Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc") == ["N-Glycan"]
    assert processor.get_glycan_keywords(
        "Man(a1-2)Man(a1-3)[Man(a1-3)[Man(a1-6)]Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc") == ["N-Glycan", "Oligomannose"]


@pytest.mark.parametrize("iupac,keywords", [
    ("GlcA3S(b1-3)Gal(b1-3)Gal(b1-4)Xyl", ["GAG"]),           # linkage region with a sulfated GlcA (GS00193)
    ("GlcA3S(b1-3)Gal(b1-4)GlcNAc", []),                     # HNK-1 epitope, not a GAG (GS00132)
    ("GlcNS6S(a1-4)GlcA(b1-4)GlcNS3S6S(a1-4)IdoA2S(a1-4)GlcNS6S", ["GAG"]),
])
def test_glycosaminoglycans(processor, iupac, keywords):
    assert processor.get_glycan_keywords(iupac) == keywords


def test_stale_class_keywords_are_replaced_and_others_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_module, "_storage_manager", StorageManager(LocalStorageBackend()))
    folder = tmp_path / "GS00103"
    folder.mkdir()
    data = {
        "archetype": {"ID": "GS00103", "iupac": "Fuc(a1-2)[Gal(a1-3)]Gal(b1-3)GalNAc(b1-3)Gal(a1-4)Gal(b1-4)Glc",
                      "wurcs": None, "glytoucan": "G00000XX"},
        "alpha": {}, "beta": {},
        "search_meta": {"common_names": [], "description": "",
                        "keywords": ["FF:GLYCAM_06j", "MD:AMBER", "O-Glycan", "Source:manual", "globoside"]},
    }
    (folder / "data.json").write_text(json.dumps(data))
    assert GlycanMetadataProcessor(tmp_path).update_glycan_metadata(folder / "data.json", enrichment={})
    written = json.loads((folder / "data.json").read_text())
    assert written["search_meta"]["keywords"] == ["FF:GLYCAM_06j", "MD:AMBER", "Source:manual", "globoside"]


# --- the whole collection against the thesis classification -----------------------------------------------------
def test_every_published_entry_has_the_keywords_of_its_class_from_its_wurcs():
    differ = []
    for entry in REFERENCE["entries"]:
        keywords, source = glyclass.class_keywords(wurcs=entry["wurcs"])
        if source != "wurcs" or keywords != expected_keywords(entry):
            differ.append((entry["id"], source, keywords, expected_keywords(entry)))
    assert differ == []


def test_every_published_iupac_name_gives_the_same_keywords():
    differ, unreadable = [], set()
    for entry in REFERENCE["entries"]:
        keywords, source = glyclass.class_keywords(iupac=entry["iupac"])
        if source != "iupac":
            unreadable.add(entry["id"])
        elif keywords != expected_keywords(entry):
            differ.append((entry["id"], keywords, expected_keywords(entry)))
    assert differ == []
    assert unreadable == BROKEN_IUPAC


def test_class_counts_of_the_collection():
    counts = Counter()
    for entry in REFERENCE["entries"]:
        counts.update(glyclass.class_keywords(wurcs=entry["wurcs"], iupac=entry["iupac"])[0])
    assert (counts["N-Glycan"], counts["O-Glycan"], counts["GAG"]) == (324, 99, 79)
    # the counts of the string patterns these rules replace
    published = Counter(k for e in REFERENCE["entries"] for k in e["published_keywords"])
    assert (published["N-Glycan"], published["O-Glycan"], published["GAG"]) == (290, 66, 85)


def test_names_with_open_or_odd_linkages_are_read():
    tree = glyclass.parse_iupac_condensed("Neu5Ac(a2-3/6)Gal(b1-3/4)GlcNAc")
    assert [r.parent_pos for r in tree] == ["3/6", "3/4", None]
    tree = glyclass.parse_wurcs(ENTRIES["GS00728"]["wurcs"])
    assert tree[0].base == "GlcNAc" and tree[0].parent is None
    with pytest.raises(ValueError):
        glyclass.parse_iupac_condensed("Gal(b1-4")
