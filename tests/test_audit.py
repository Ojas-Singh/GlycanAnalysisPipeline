"""The entry audit (lib/audit.py, lib/audit_structure.py, lib/audit_classes.py, scripts/audit_entries.py).

Unit tests use the first models of a few published Level 1 structures
(``tests/fixtures/audit``) and facts written by hand.  The last test audits
the known problem entries of the local collection and is skipped where the
collection is not on disk (``GLYCOSHAPE_AUDIT_STATIC``, ``..._ARCHIVE``,
``..._REPLICAS``, ``..._REGISTRY_SEED``, ``..._REGISTRY_EXPORTS`` and
``..._CRABWURCS`` name other places).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from lib import audit
from lib import audit_classes as classes
from lib import audit_structure as structure

FIXTURES = Path(__file__).parent / "fixtures" / "audit"
WURCS = {
    "GS00002": "WURCS=2.0/4,4,3/[a2112h-1a_1-5_2*NCC/3=O][a2112h-1b_1-5][a2112h-1a_1-5][a1221m-1a_1-5]/1-2-3-4/a3-b1_b3-c1_c2-d1",
    "GS00020": "WURCS=2.0/5,14,13/[a2122h-1b_1-5_2*NCC/3=O][a1122h-1b_1-5][a1122h-1a_1-5][a2112h-1b_1-5][a1221m-1a_1-5]"
               "/1-1-2-3-1-4-1-4-3-1-4-1-4-5/a4-b1_a6-n1_b4-c1_c3-d1_c6-i1_d2-e1_d4-g1_e4-f1_g4-h1_i2-j1_i6-l1_j4-k1_l4-m1",
    "GS00032": "WURCS=2.0/6,15,14/[a2122h-1b_1-5_2*NCC/3=O][a1122h-1b_1-5][a1122h-1a_1-5][a1221m-1a_1-5][a2112h-1b_1-5]"
               "[Aad21122h-2a_2-6_5*NCC/3=O]/1-1-2-3-1-4-5-6-3-1-5-6-1-5-6/a4-b1_b4-c1_c3-d1_c6-i1_d2-e1_e3-f1_e4-g1_g3-h2"
               "_i2-j1_i6-m1_j4-k1_k6-l2_m4-n1_n3-o2",
    "GS00938": "WURCS=2.0/4,5,4/[a2122h-1a_1-5_2*NSO/3=O/3=O_3*OSO/3=O/3=O_6*OSO/3=O/3=O][a2121A-1a_1-5_2*OSO/3=O/3=O]"
               "[a2122h-1a_1-5_2*NCC/3=O][a2122A-1b_1-5]/1-2-3-4-3/a4-b1_b4-c1_c4-d1_d4-e1",
    "GS00901": "WURCS=2.0/1,1,0/[Aad21122h-2a_2-6_5*NCC/3=O]/1/",
}


def read(name: str) -> structure.Reading:
    return structure.read(structure.read_first_model(FIXTURES / name))


# ---------------------------------------------------------------------------
# Reading a glycan from its coordinates
# ---------------------------------------------------------------------------
def test_a_structure_that_is_its_name_reads_back_as_it():
    reading = read("GS00002_level1_alpha_model1.pdb")
    assert [r.code for r in reading.residues] == ["a2112h-1a_1-5", "a2112h-1b_1-5", "a2112h-1a_1-5", "a1221m-1a_1-5"]
    assert structure.compare(reading, structure.declared(WURCS["GS00002"]))["outcome"] == "match"


def test_the_skeleton_codes_and_anomers_of_ketoses_are_read():
    neu5ac = read("GS00901_level1_alpha_model1.pdb")
    assert [(r.skeleton, r.anomer, r.ring_code) for r in neu5ac.residues] == [("Aad21122h", "a", "2-6")]
    fructose = read("GS00894_level1_beta_model1.pdb")
    assert [(r.skeleton, r.ring_code) for r in fructose.residues] == [("ha122h", "2-5")]


def test_the_other_anomer_at_the_reducing_end_is_told_apart_from_another_glycan():
    # the beta file of a Neu5Ac entry still holds the simulated alpha (GAP cannot flip C2 of a ketose)
    beta_file = read("GS00901_level1_beta_model1.pdb")
    named_beta = structure.declared(WURCS["GS00901"].replace("-2a_", "-2b_"))
    result = structure.compare(beta_file, named_beta)
    assert result["outcome"] == "other_anomer"
    assert result["structure_reducing_end"]["anomer"] == "a"


def test_g56655cc_alpha_file_holds_beta_because_the_simulation_was_alpha():
    # GS00020 is named ...DGlcpNAcb1-OH but was simulated as alpha: GAP wrote the flipped frame as alpha.pdb
    reading = read("GS00020_level1_alpha_model1.pdb")
    root = reading.residues[reading.reducing_end]
    assert root.residue_name == "NDG"          # the wwPDB code of alpha-GlcNAc: what the file is called
    assert root.anomer == "b"                  # what its atoms are
    assert structure.compare(reading, structure.declared(WURCS["GS00020"]))["outcome"] == "match"


def test_g46171qf_has_an_alpha_glcnac_where_the_name_says_beta_1_6():
    reading = read("GS00032_level1_beta_model1.pdb")     # the observed frame (the name declares beta)
    result = structure.compare(reading, structure.declared(WURCS["GS00032"]))
    assert result["outcome"] == "different"
    assert ["a2122h_1-5", "a", 1, 6, "a1122h_1-5"] in result["linkages_in_structure_not_in_name"]
    assert ["a2122h_1-5", "b", 1, 6, "a1122h_1-5"] in result["linkages_in_name_not_in_structure"]
    words = audit.describe_difference(result, WURCS["GS00032"])
    assert "GlcNAc(a1-6)Man where the name has GlcNAc(b1-6)Man" in words
    assert "the reducing end is alpha where the name says beta" in words


def test_an_iduronic_acid_with_the_c2_of_guluronic_acid_is_found():
    reading = read("GS00938_level1_alpha_model1.pdb")
    result = structure.compare(reading, structure.declared(WURCS["GS00938"]))
    assert result["outcome"] == "different"
    assert result["kinds"] == ["configuration"]
    assert result["residues_in_structure_not_in_name"] == ["a1121A-1a_1-5"]
    assert "L-GulA where the name has IdoA2S" in audit.describe_difference(result, WURCS["GS00938"])


def test_a_declared_residue_with_unknown_stereochemistry_matches_any():
    reading = read("GS00002_level1_alpha_model1.pdb")
    wurcs = WURCS["GS00002"].replace("[a1221m-1a_1-5]", "[axxxxm-1x_]")
    assert structure.compare(reading, structure.declared(wurcs))["outcome"] == "match"


def test_stereocentres_change_sign_in_the_mirror_image():
    atoms = structure.read_first_model(FIXTURES / "GS00002_level1_alpha_model1.pdb")
    centres = structure.stereocentres(atoms)
    assert len(centres) >= 16
    signs = structure.centre_signs(atoms.xyz[None], centres)
    mirrored = atoms.xyz * np.array([-1.0, 1.0, 1.0])
    assert (structure.centre_signs(mirrored[None], centres) == ~signs).all()
    assert structure.centre_label(atoms, centres[0]).count("(") == 1


# ---------------------------------------------------------------------------
# The class rules of the thesis
# ---------------------------------------------------------------------------
def test_the_class_rules_flag_a_core_without_the_beta_mannose():
    gs01039 = classes.parse_wurcs(
        "WURCS=2.0/6,12,11/[u2122h_2*NCC/3=O][a2122h-1b_1-5_2*NCC/3=O][a1122h-1a_1-5][a2112h-1b_1-5][a1221m-1a_1-5]"
        "/1-2-3-3-2-4-2-4-3-2-4-5/a4-b1_a6-l1_b4-c1_c3-d1_c6-j1_d2-e1_d4-g1_e4-f1_g4-h1_j2-k1_k4-i1")
    result = classes.classify(gs01039)
    assert result["class"] == "other" and result["detail"]["n_glycan_like"]


# ---------------------------------------------------------------------------
# Rules on facts
# ---------------------------------------------------------------------------
def facts(**parts):
    base = {"glycoshape_id": "GS99999", "errors": [], "inputs": {"static": True, "raw": True},
            "entry": {"name": "DGlcpb1-4DGlcpb1-OH", "glytoucan": {"archetype": "G00000AA"}, "mass": 342.12,
                      "length_us": 1.5, "package": "GROMACS", "forcefield": "GLYCAM_06j", "temperature": 300.0,
                      "pressure": 1.0, "salt": 200.0},
            "notation": {"name_is_glycam": True, "variants": {}}, "tree": {"residues": 2},
            "frames": {"archive": 30003, "kept": 30003, "removed": 0, "runs_if_10001": 3},
            "raw": {"glycam_name": "DGlcpb1-4DGlcpb1-OH"}}
    for key, value in parts.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return base


def codes(issues):
    return sorted(i["code"] for i in issues)


def test_a_clean_entry_has_no_issues():
    assert audit.entry_issues(facts()) == []


def test_simulation_metadata_rules():
    assert codes(audit.metadata_issues(facts(entry={"length_us": 0, "package": "", "temperature": 0, "pressure": 0,
                                                     "forcefield": "", "salt": 0}), {})) == \
        ["forcefield-placeholder", "sim-metadata-blank", "sim-time-missing"]
    issue = audit.metadata_issues(facts(entry={"length_us": 0.5}, frames={"archive": 160016, "runs_if_10001": 16}), {})
    assert codes(issue) == ["sim-time-frames-disagree", "sim-time-short"]
    disagree = next(i for i in issue if i["code"] == "sim-time-frames-disagree")
    assert disagree["evidence"]["time_by_frames_us"] == 8.0
    assert codes(audit.metadata_issues(facts(entry={"forcefield": "See Build Directory Files", "salt": 1.5}), {})) == \
        ["forcefield-placeholder", "salt-implausible"]
    unit = audit.metadata_issues(facts(registry={"simulation_length": 1500}), {})
    assert codes(unit) == ["sim-time-unit"] and unit[0]["actions"][0]["options"]["length_us"] == 1.5


def test_stereocentres_that_differ_between_runs_are_blocking():
    centre = {"label": "0LB C1 (C2, O5, O4)", "atom": "C1", "residue": "0LB", "residue_number": 5,
              "neighbours": ["C2", "O5", "O4"], "positive": [0.0, 1.0, 1.0]}
    f = facts(stereo={"runs": ["run_1", "run_2", "run_3"], "frames_read": {}, "centres": [centre]},
              frames={"runs": {"basis": "assumed", "names": ["run_1", "run_2", "run_3"]}})
    issues = audit.identity_issues(f, {})
    assert codes(issues) == ["stereo-differs-between-runs"]
    assert issues[0]["severity"] == "blocking"
    assert issues[0]["evidence"]["centres"][0]["sign_per_run"] == "-++"
    assert issues[0]["actions"][0]["options"]["exclude_replicas"] == ["run_1"]


def test_an_exchanged_anomer_suggests_declaring_the_simulated_one():
    f = facts(structure={"simulated": {"source": "raw topology.pdb", "comparison": {"outcome": "other_anomer"},
                                       "reducing_end": {"anomer": "a", "residue": "UYA", "anomer_from_residue_name": "a"}},
                         "level1": {}})
    issue = audit.identity_issues(f, {})[0]
    assert issue["code"] == "anomer-exchanged" and issue["severity"] == "blocking"
    assert issue["actions"][0] == {"action": "reanalyse", "why": issue["actions"][0]["why"],
                                   "options": {"declare_simulated_anomer": "a"}}
    assert issue["actions"][1]["options"]["to"] == "DGlcpb1-4DGlcpa1-OH"
    assert issue["needs_owner"] and not issue["resolved_by_reanalysis"]


def test_a_ketose_level1_file_that_was_not_flipped_is_explained():
    end = {"anomer": "a", "ring": "2-6", "skeleton": "Aad21122h"}
    f = facts(entry={"name": "DNeup5Aca2-OH"},
              structure={"simulated": {"comparison": {"outcome": "match"}},
                         "level1": {"beta": {"path": "output/level_1/PDB/beta.pdb", "reducing_end_anomer": "a",
                                             "reducing_end": end}}})
    issue = audit.identity_issues(f, {})[0]
    assert issue["key"] == "level1-file-anomer:beta"
    assert "ketose" in issue["evidence"]["cause"] and not issue["resolved_by_reanalysis"]


def test_the_name_disagrees_port():
    info = {"torsions": {"glycosidic": [
        {"canonical_name": "3_2_phi", "subtype": "phi", "residue_names": ["0YA", "0YA", "6MA", "6MA"],
         "atom_names": ["O5", "C1", "O6", "C6"]},
        {"canonical_name": "3_2_psi", "subtype": "psi", "residue_names": ["0YA", "6MA", "6MA", "6MA"],
         "atom_names": ["C1", "O6", "C6", "C5"]}]}}
    found = audit._name_disagrees(info, "GlcNAc(b1-6)Man", [2, 3])
    assert found == {"in_name_not_in_structure": {"b1-6": 1}, "in_structure_not_in_name": {"a1-6": 1}}
    assert audit._name_disagrees(info, "GlcNAc(a1-6)Man", [2, 3]) is None


def test_mass_rule_tells_a_sulfate_and_hydrogens():
    f = facts(entry={"mass": 2267.28}, tree={"residues": 10, "groups": {"sulfate": 6, "sialic": 0, "uronic": 5,
                                                                        "phosphocholine": 0}},
              structure={"elements": {"C": 64, "H": 89, "N": 5, "O": 74, "S": 7}})
    # the atoms of GS00039 (G87874MT): the heparan sulfate decasaccharide of its name with a seventh sulfate group
    issues = [i for i in audit.notation_issues(f, {}) if i["code"] == "mass-mismatch"]
    assert len(issues) == 1
    assert "1 sulfate group more" in issues[0]["title"]


def test_duplicates_and_registry_conflicts_come_from_the_collection():
    def entry(gid, name, wurcs):
        return {"glycoshape_id": gid, "entry": {"name": name},
                "notation": {"variants": {"archetype": {"open_wurcs": wurcs}}}}
    collection = {"GS00303": entry("GS00303", "Xa1-OH", "W1"), "GS00453": entry("GS00453", "Ya1-OH", "W1"),
                  "GS00013": entry("GS00013", "Za1-OH", "W2"), "GS00037": entry("GS00037", "Zb1-OH", "W2")}
    seed = {"entries": [{"glycoshape_id": "GS00534", "glycam_name": "N", "status": "published"},
                        {"glycoshape_id": "GS00481", "glycam_name": "N", "status": "issued"}]}
    exports = [{"collection": "glycans", "records": [{"glycoshape_id": "GS11E892", "glycam_name": "N"}]}]
    registry = audit.registry_names(seed, exports)
    assert [i["glycoshape_id"] for i in registry["N"]] == ["GS00481", "GS00534", "GS11E892"]
    assert [i["valid"] for i in registry["N"]] == [True, True, False]
    context = audit.collection_context(collection, registry=registry)
    assert context["duplicates"] == {"GS00303": ["GS00453"], "GS00453": ["GS00303"]}
    f = facts(glycoshape_id="GS00534", entry={"name": "N"}, raw={"glycam_name": "N"})
    issue = audit.registry_issues(f, context)[0]
    assert issue["code"] == "id-several-for-name" and issue["actions"][0]["action"] == "merge-ids"


def test_replica_verdicts_become_issues():
    reasons = [{"code": "cluster_missing", "clusters": [{"cluster": 4, "pooled": 0.2, "replicas": ["run_2"]}],
                "text": "Level 1 cluster 4 has less than 1% of the frames of run_2"}]
    f = facts(replicas={"verdict": "more_sampling_needed", "reasons": reasons, "replicas": 3})
    issue = audit.replica_issues(f, {})[0]
    assert issue["code"] == "replicas-more-sampling-needed" and issue["severity"] == "should-fix"
    assert [a["action"] for a in issue["actions"]] == ["request-replicas", "reanalyse", "none"]
    assert issue["actions"][1]["options"]["exclude_replicas"] == ["run_2"]
    assert audit.replica_issues(facts(replicas={"verdict": "ok", "reasons": []}), {}) == []


def test_runs_whose_coordinates_do_not_match_the_topology_are_blocking():
    def run(name, first, last, inverted):
        frames = [{"frame": f, "bonds_over_cutoff": 0, "bonds_over_3A": 0, "longest_bond": 1.6,
                   "centres_inverted": ["0LB C1 (C2, O5, O4)"] if inverted else []} for f in (first, last)]
        return {"run": name, "first_frame": first, "last_frame": last, "frames": frames, "matches": not inverted}

    f = facts(segments={"bonds": 145, "bonds_from": "structure.mol2", "centres": 30,
                        "runs": [run("run_1", 0, 10000, False), run("run_2", 10001, 20001, True),
                                 run("run_3", 20002, 30002, True)]})
    issue = audit.raw_issues(f, {})[0]
    assert issue["code"] == "run-topology-mismatch" and issue["severity"] == "blocking"
    assert "run_2 (frames 10,001 to 20,001), run_3 (frames 20,002 to 30,002)" in issue["title"]
    assert [a["action"] for a in issue["actions"]] == ["reanalyse", "redo"]
    assert issue["actions"][0]["options"]["exclude_replicas"] == ["run_2", "run_3"]


def test_an_email_in_the_raw_info_json_is_masked():
    f = facts(raw={"glycam_name": "N", "info_json": {"emails": [audit.mask_email("jane.doe@example.org")],
                                                     "fields": ["email"]}})
    issue = audit.raw_issues(f, {})[0]
    assert issue["code"] == "personal-data-in-raw" and issue["severity"] == "blocking"
    assert issue["evidence"]["emails_masked"] == ["j***@e***.org"]


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------
def test_every_issue_type_is_described_and_every_action_is_known():
    for code, t in audit.ISSUE_TYPES.items():
        assert t["severity"] in audit.SEVERITIES
        assert t["what"] and t["found_by"] and t["group"]
        assert isinstance(t["resolved_by_reanalysis"], bool) and isinstance(t["needs_owner"], bool)
    with pytest.raises(ValueError):
        audit._issue("sim-time-short", "t", {}, actions=[{"action": "delete"}])


def test_the_document_is_versioned_and_counts_by_severity(tmp_path):
    f = facts(entry={"length_us": 0.5})
    issues = audit.entry_issues(f)
    doc = audit.document(f, issues)
    assert doc["schema"] == "glycoshape-entry-issues" and doc["schema_version"] == 1
    assert doc["counts"] == {"blocking": 0, "should-fix": 1, "note": 1}
    assert [i["severity"] for i in doc["issues"]] == ["should-fix", "note"]
    for issue in doc["issues"]:
        assert set(issue) == {"key", "code", "severity", "title", "evidence", "files", "actions",
                              "resolved_by_reanalysis", "needs_owner", "found_by"}
    path = audit.write(doc, tmp_path / "GS99999" / "issues.json")
    assert json.loads(path.read_text()) == doc
    assert audit.encode(doc) == audit.encode(json.loads(path.read_text()))


def test_the_list_of_codes(capsys):
    from scripts import audit_entries

    assert audit_entries.main(["--list-codes"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert [t["code"] for t in listed["issue_types"]] == list(audit.ISSUE_TYPES)
    assert set(listed["actions"]) == {"rename", "edit-metadata", "reanalyse", "request-replicas", "redo", "withdraw",
                                      "merge-ids", "none"}


# ---------------------------------------------------------------------------
# The known problems of the collection (skipped where the collection is not on disk)
# ---------------------------------------------------------------------------
HOME = Path.home()
STATIC = Path(os.environ.get("GLYCOSHAPE_AUDIT_STATIC", HOME / "gap-work/static-l12"))
ARCHIVE = Path(os.environ.get("GLYCOSHAPE_AUDIT_ARCHIVE", HOME / "gap-work/archive"))
REPLICAS = Path(os.environ.get("GLYCOSHAPE_AUDIT_REPLICAS", HOME / "scratch/replica-diagnostics"))
SEED = Path(os.environ.get("GLYCOSHAPE_AUDIT_REGISTRY_SEED", HOME / "gap-work/publish-work/registry-seed.json"))
EXPORTS = [Path(p) for p in os.environ.get(
    "GLYCOSHAPE_AUDIT_REGISTRY_EXPORTS",
    f"{HOME}/gap-work/pocketbase-export/submission_ids.json:{HOME}/gap-work/pocketbase-export/glycans.json").split(":")]
CRABWURCS = Path(os.environ.get("GLYCOSHAPE_AUDIT_CRABWURCS",
                                HOME / "scratch/entry-audit/crabwurcs-target/release/crabwurcs"))

#: The problems found by the thesis and listed in problem_to_look.md, sections 1 to 3: entry -> issue codes.
KNOWN = {
    # alpha and beta files exchanged
    "GS00020": {"anomer-exchanged"}, "GS00021": {"anomer-exchanged"}, "GS00028": {"anomer-exchanged"},
    # structure contradicts the name at a 1-6 linkage (G46171QF, G57923KF, G03346FO, G89222IF)
    "GS00032": {"structure-contradicts-name"}, "GS00033": {"structure-contradicts-name"},
    "GS00036": {"structure-contradicts-name"},
    "GS00399": {"structure-contradicts-name", "name-not-glycam"},
    # runs that are different stereoisomers (G04196SD, G71956IS, G97921YP)
    # (and whose runs do not match the topology: the torsion work found GS00276)
    "GS00276": {"stereo-differs-between-runs", "run-topology-mismatch"},
    "GS00180": {"stereo-differs-between-runs", "run-topology-mismatch"},
    "GS00250": {"stereo-differs-between-runs", "run-topology-mismatch"},
    # stored WURCS: substituents lost, furanose rings, others
    "GS00237": {"stored-wurcs-mismatch"}, "GS00238": {"stored-wurcs-mismatch"}, "GS00314": {"stored-wurcs-mismatch"},
    "GS00386": {"stored-wurcs-mismatch"}, "GS00667": {"stored-wurcs-mismatch"}, "GS00893": {"stored-wurcs-mismatch"},
    "GS00907": {"stored-wurcs-mismatch"}, "GS00616": {"stored-wurcs-mismatch"},
    # names that are not GLYCAM sequences; broken inventory fields
    "GS00430": {"name-not-glycam", "inventory-fields-broken"}, "GS00722": {"name-not-glycam"},
    "GS00723": {"name-not-glycam"}, "GS00728": {"name-not-glycam", "inventory-fields-broken"},
    "GS00954": {"stored-wurcs-mismatch", "inventory-fields-broken", "sim-time-unit", "personal-data-in-raw"},
    # duplicates; probable input errors
    "GS00303": {"duplicate-entry"}, "GS00453": {"duplicate-entry"},
    "GS01039": {"probable-input-error"}, "GS01043": {"probable-input-error", "glytoucan-missing", "sim-time-missing"},
    "GS00690": {"probable-input-error"},
    # atoms or charge against the name
    "GS00039": {"mass-mismatch"}, "GS00617": {"mass-mismatch"}, "GS00621": {"mass-mismatch"},
    "GS00632": {"mass-mismatch"},
    # simulation metadata
    "GS01032": {"sim-time-missing", "sim-metadata-blank"}, "GS00709": {"sim-time-frames-disagree"},
    "GS00041": {"sim-time-frames-disagree"}, "GS00091": {"salt-implausible"}, "GS00948": {"sim-time-unit"},
    # raw irregularities, frames removed, short trajectories
    "GS00968": {"raw-source-repaired"}, "GS00055": {"raw-source-repaired"}, "GS00741": {"raw-source-repaired"},
    "GS00636": {"frames-removed"}, "GS00491": {"frames-removed"}, "GS00936": {"trajectory-short"},
    # a GLYCAM name with two IDs (GS00481 issued, GS00534 published)
    "GS00534": {"id-several-for-name"},
}


@pytest.mark.skipif(not (STATIC.is_dir() and (ARCHIVE / "index.json").is_file() and os.access(CRABWURCS, os.X_OK)),
                    reason="the local collection or the crabWURCS binary is not on this machine")
def test_the_known_problems_of_the_collection_are_found(tmp_path):
    from scripts import audit_entries

    argv = ["--static-dir", str(STATIC), "--archive-dir", str(ARCHIVE), "--out", str(tmp_path),
            "--crabwurcs", str(CRABWURCS), "--workers", "4"]
    if REPLICAS.is_dir():
        argv += ["--replicas-dir", str(REPLICAS)]
    if SEED.is_file():
        argv += ["--registry-seed", str(SEED)]
    for export in EXPORTS:
        if export.is_file():
            argv += ["--registry-export", str(export)]
    for gid in KNOWN:
        argv += ["--glycan", gid]
    assert audit_entries.main(argv) == 0
    missing = {}
    for gid, expected in KNOWN.items():
        doc = json.loads((tmp_path / gid / "issues.json").read_text())
        assert doc["checks_failed"] == [], gid
        found = {i["code"] for i in doc["issues"]}
        if not expected <= found:
            missing[gid] = sorted(expected - found)
    assert missing == {}
    gs00020 = json.loads((tmp_path / "GS00020" / "issues.json").read_text())
    exchanged = next(i for i in gs00020["issues"] if i["code"] == "anomer-exchanged")
    assert exchanged["evidence"]["declared_anomer"] == "b" and exchanged["evidence"]["simulated_anomer"] == "a"
    assert exchanged["evidence"]["level1_files_read_as"] == {"alpha": "b", "beta": "a"}
    gs00954 = json.loads((tmp_path / "GS00954" / "issues.json").read_text())
    personal = next(i for i in gs00954["issues"] if i["code"] == "personal-data-in-raw")
    assert "@" in personal["evidence"]["emails_masked"][0] and "***" in personal["evidence"]["emails_masked"][0]
    gs00616 = json.loads((tmp_path / "GS00616" / "issues.json").read_text())
    wurcs = next(i for i in gs00616["issues"] if i["code"] == "stored-wurcs-mismatch")
    assert wurcs["evidence"]["kinds"] == ["ring"] and wurcs["resolved_by_reanalysis"]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["failed"] == 0 and summary["inputs"]["network"] == "none"
