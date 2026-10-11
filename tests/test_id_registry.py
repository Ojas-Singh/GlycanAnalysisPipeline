"""GlycoShape IDs for the registry of the publishing service (lib/id_registry.py)."""
from __future__ import annotations

import pytest

from lib import id_registry, name_utils
from lib.id_registry import RegistryError, full_glycam_name, name_is_consistent, next_id, seed_registry

from static_tree import glycan_data

LONG = "DNeup5Aca2-3[DGalpNAcb1-4]DGalpb1-4DGlcpNAcb1-OH"


def entry(glycan_id, name):
    """An inventory entry with the glycam fields the static stage derives from ``name``."""
    stem, position, _ = name_utils.glycam_info(name)
    data = glycan_data(glycan_id, name)
    data["archetype"]["glycam"] = stem
    data["alpha"]["glycam"] = f"{stem}a{position}-OH"
    data["beta"]["glycam"] = f"{stem}b{position}-OH"
    return data


def test_the_next_id_follows_the_highest_one_ever_issued():
    # the old registry issued up to GS01066 while GS01063 is the highest published one
    issued = ["GS00001", "GS01066", "", None, "GS00120", "not an id", "GS1064"]
    inventory = {"GS00002": {}, "GS01063": {}}
    assert next_id(issued, inventory) == "GS01067"
    assert next_id([], inventory) == "GS01064" and next_id(["GS00005"]) == "GS00006"
    assert next_id([]) == "GS00001" and next_id(["GS00007"], {"GS00009": {}}) == "GS00010"   # a gap is never filled
    with pytest.raises(RegistryError, match="five digits|5 digits"):
        next_id(["GS99999"])
    assert id_registry.id_number("GS01066") == 1066 and id_registry.format_id(7) == "GS00007"
    for wrong in ("GS1066", "GS010666", "gs01066", "GS0106a", 1066, None, "GS01066 "):
        assert not id_registry.is_id(wrong)
        with pytest.raises(RegistryError):
            id_registry.id_number(wrong)


def test_the_full_glycam_name_of_an_inventory_entry_is_its_name_not_its_glycam_field():
    alpha = entry("GS00002", "LFucpa1-2DGalpa1-3DGalpb1-3DGalpNAca1-OH")
    assert alpha["archetype"]["glycam"] == "LFucpa1-2DGalpa1-3DGalpb1-3DGalpNAc"          # without the reducing end
    assert full_glycam_name(alpha) == "LFucpa1-2DGalpa1-3DGalpb1-3DGalpNAca1-OH" == alpha["alpha"]["glycam"]
    beta = entry("GS00003", "DAllpb1-OH")
    assert (beta["archetype"]["glycam"], beta["alpha"]["glycam"], beta["beta"]["glycam"]) == ("DAllp", "DAllpa1-OH", "DAllpb1-OH")
    assert full_glycam_name(beta) == "DAllpb1-OH" and name_is_consistent(beta) and name_is_consistent(alpha)
    ketose = entry("GS00004", "DNeup5Aca2-OH")
    assert ketose["beta"]["glycam"] == "DNeup5Acb2-OH" and name_is_consistent(ketose)
    # a name without the ending: an entry built from a folder named by an accession, or a name that has none
    for name in ("G50508SG", "DGalpb1-4DGalpa1-3[2,4-diacetimido-2,4,6-trideoxyhexose]"):
        plain = entry("GS00722", name)
        assert plain["alpha"]["glycam"] == name + "aNone-OH" and full_glycam_name(plain) == name and name_is_consistent(plain)
        del plain["archetype"]["name"]
        assert full_glycam_name(plain) == name                                             # recoverable without the name
    del alpha["archetype"]["name"]
    assert full_glycam_name(alpha) is None and not name_is_consistent(alpha)                # the anomer is not recoverable
    beta["alpha"]["glycam"] = "DAllpa2-OH"
    assert not name_is_consistent(beta)
    assert full_glycam_name({}) is None and full_glycam_name({"archetype": "x"}) is None


def test_a_registry_is_seeded_from_the_inventory_and_the_issued_ids():
    inventory = {
        "GS00002": entry("GS00002", "DManpa1-OH"),
        "GS00005": entry("GS00005", "DGlcpb1-OH"),
        "GS00722": entry("GS00722", "G50508SG"),                  # published under the name of its source folder
        "GS00900": entry("GS00900", "DGalpb1-OH"),
    }
    issued = [
        {"glycoshape_id": "GS00002", "glycam_name": "DManpa1-OH", "status": "inbucket"},
        {"glycoshape_id": "GS00005", "glycam_name": "DGlcpb1-OH"},
        {"glycoshape_id": "GS00300", "glycam_name": "DManpa1-OH"},       # the same glycan was issued a second ID
        {"glycoshape_id": "GS00722", "glycam_name": "G50508SG"},
        {"glycoshape_id": "GS00901", "glycam_name": "DXylpb1-OH"},       # issued, never published
        {"glycoshape_id": "GS00901", "glycam_name": "LFucpa1-OH"},       # and issued again for another glycan
        {"glycoshape_id": "", "glycam_name": "DAllpb1-OH"},
        {"glycoshape_id": "GS902", "glycam_name": "DTalpa1-OH"},
    ]
    names = id_registry.names_from_archive_index({"records": [
        {"glycam_name": LONG, "source_folder": "G50508SG"}, {"glycam_name": "DManpa1-OH", "source_folder": "DManpa1-OH"}]})
    assert names == {"G50508SG": LONG}
    seed = seed_registry(inventory, issued, names)

    assert [(item["glycoshape_id"], item["status"]) for item in seed["entries"]] == [
        ("GS00002", "published"), ("GS00005", "published"), ("GS00300", "issued"), ("GS00722", "published"),
        ("GS00900", "published"), ("GS00901", "issued")]
    assert seed["entries"][3] == {"glycoshape_id": "GS00722", "glycam_name": LONG, "status": "published",
                                  "published_as": "G50508SG"}
    assert (seed["published"], seed["issued_only"], seed["next_id"]) == (4, 2, "GS00902")
    # the key is the full GLYCAM name; a name with two IDs is left to a person
    assert seed["by_name"] == {"DGalpb1-OH": "GS00900", "DGlcpb1-OH": "GS00005", "DXylpb1-OH": "GS00901", LONG: "GS00722"}
    assert seed["conflicts"] == {
        "name_with_several_ids": [{"glycam_name": "DManpa1-OH", "ids": [
            {"glycoshape_id": "GS00002", "status": "published"}, {"glycoshape_id": "GS00300", "status": "issued"}]}],
        "id_with_several_names": [{"glycoshape_id": "GS00901", "status": "issued",
                                   "glycam_names": ["DXylpb1-OH", "LFucpa1-OH"]}],
        "issued_without_id": [{"glycam_name": "DAllpb1-OH"}],
        "invalid_id": [{"glycoshape_id": "GS902", "glycam_name": "DTalpa1-OH", "status": "issued"}],
        "published_not_issued": ["GS00900"],
    }

    # without the names of the archive the accession stays, and is reported
    plain = seed_registry(inventory, issued)
    assert plain["by_name"]["G50508SG"] == "GS00722"
    assert plain["conflicts"]["name_not_glycam"] == [{"glycoshape_id": "GS00722", "name": "G50508SG"}]
    # an ID published for one name and issued for another
    moved = seed_registry({"GS00002": entry("GS00002", "DManpa1-OH")}, [{"glycoshape_id": "GS00002", "glycam_name": "DManpb1-OH"}])
    assert moved["conflicts"] == {"id_with_several_names": [
        {"glycoshape_id": "GS00002", "status": "published", "glycam_names": ["DManpa1-OH", "DManpb1-OH"]}]}
    broken = entry("GS00007", "DManpa1-OH")
    broken["archetype"]["glycam"] = "DManpa1-OH"
    assert seed_registry({"GS00007": broken, "GS8": {}}, [])["conflicts"] == {
        "invalid_id": [{"glycoshape_id": "GS8", "status": "published"}],
        "published_not_issued": ["GS00007"],
        "inconsistent_entry": [{"glycoshape_id": "GS00007", "reason": "the glycam fields are not those of archetype.name"}],
    }
    assert seed_registry({}, []) == {"entries": [], "by_name": {}, "next_id": "GS00001", "published": 0,
                                     "issued_only": 0, "conflicts": {}}
