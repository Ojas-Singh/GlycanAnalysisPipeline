"""Replica diagnostics (lib/replicas.py): synthetic entries with known answers, and one real entry against the thesis."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lib import replicas

DATA = Path(__file__).resolve().parent / "fixtures"


# ---------------------------------------------------------------------------
# Synthetic entries
# ---------------------------------------------------------------------------
def _metadata(version=1):
    """Torsion metadata of one Man a1-6 Man linkage (phi, psi, omega), as info.json records it.

    Version 1 of the torsion definitions stored omega as C4-C5-C6-O6, version 2 as O5-C5-C6-O6.
    """
    omega = ["C4", "C5", "C6", "O6"] if version == 1 else ["O5", "C5", "C6", "O6"]
    return [
        {"canonical_name": "2_1_phi", "family": "glycosidic", "atom_names": ["O5", "C1", "O6", "C6"],
         "residue_ids": [2, 2, 1, 1], "residue_names": ["0MA", "0MA", "2MA", "2MA"]},
        {"canonical_name": "2_1_psi", "family": "glycosidic", "atom_names": ["C1", "O6", "C6", "C5"],
         "residue_ids": [2, 1, 1, 1], "residue_names": ["0MA", "2MA", "2MA", "2MA"]},
        {"canonical_name": "2_1_omega", "family": "glycosidic", "atom_names": omega,
         "residue_ids": [1, 1, 1, 1], "residue_names": ["2MA", "2MA", "2MA", "2MA"]},
    ]


CENTRES = np.array([[-6.0, -6.0], [6.0, -6.0], [-6.0, 6.0], [6.0, 6.0], [0.0, 0.0]])


def _run(rng, frames, populations, omega=None, switch=0.02):
    """One replica: Level 1 clusters drawn with ``populations``; omega a Markov chain over the three rotamers
    (switching with probability ``switch`` per frame), or fixed at ``omega`` degrees."""
    cluster = rng.choice(5, size=frames, p=populations)
    pcs = CENTRES[cluster] + rng.normal(0.0, 0.8, (frames, 2))
    phi = rng.normal(-70.0, 12.0, frames)
    psi = rng.normal(150.0, 15.0, frames)
    if omega is None:
        state = np.empty(frames, dtype=int)
        state[0] = 0
        flips = rng.random(frames) < switch
        steps = rng.integers(1, 3, frames)
        for i in range(1, frames):
            state[i] = (state[i - 1] + steps[i]) % 3 if flips[i] else state[i - 1]
        om = np.array([60.0, 180.0, -60.0])[state] + rng.normal(0.0, 8.0, frames)
    else:
        om = omega + rng.normal(0.0, 8.0, frames)
    torsions = (np.stack([phi, psi, om], axis=1) + 180.0) % 360.0 - 180.0
    return cluster, pcs, torsions


def synthetic(runs, keep=None, version=1):
    """FrameTables and the manifest of an old entry whose runs are recorded by END records.

    The omega of ``runs`` is given on the scale of version 1 (C4-C5-C6-O6: gg +60, gt 180, tg -60); with
    ``version=2`` the same conformations are stored on the IUPAC scale, 120 degrees lower.
    """
    rng = np.random.default_rng(7)
    parts = [_run(rng, **run) for run in runs]
    cluster = np.concatenate([p[0] for p in parts])
    pcs = np.concatenate([p[1] for p in parts])
    torsions = np.concatenate([p[2] for p in parts])
    if version == 2:
        torsions[:, 2] = (torsions[:, 2] - 120.0 + 180.0) % 360.0 - 180.0
    frames = len(cluster)
    source_frame = np.arange(frames) if keep is None else np.flatnonzero(keep)
    lengths = [run["frames"] for run in runs]
    ends = list(np.cumsum(lengths))
    manifest = {"glycam_name": "DManpa1-6DManpa1-OH", "frames": frames,
                "source": {"model_numbers": {"first": 1, "last": lengths[-1], "largest": max(lengths),
                                             "sequential": False, "breaks": len(runs) - 1},
                           "records_between_frames": {"END": {"count": len(ends), "before_frame": ends}}}}
    tables = replicas.FrameTables(source_frame=source_frame, cluster=cluster[source_frame], pcs=pcs[source_frame],
                                  torsions=torsions[source_frame], torsion_names=["2_1_phi", "2_1_psi", "2_1_omega"],
                                  torsion_metadata=_metadata(version), conventions_version=version)
    return tables, manifest


EVEN = [0.4, 0.3, 0.2, 0.06, 0.04]


# ---------------------------------------------------------------------------
# The measures
# ---------------------------------------------------------------------------
def test_distances_between_distributions_have_their_known_values():
    same, other = np.array([2.0, 2.0, 0.0]), np.array([0.0, 0.0, 5.0])
    assert replicas.tv(same, same) == 0.0 and replicas.tv(same, other) == 1.0
    assert replicas.tv(np.array([3.0, 1.0]), np.array([1.0, 3.0])) == pytest.approx(0.5)
    assert replicas.overlap(same, same) == pytest.approx(1.0) and replicas.overlap(same, other) == 0.0
    assert replicas.jsd(same, same) == pytest.approx(0.0) and replicas.jsd(same, other) == pytest.approx(1.0)
    # half the mass shared: JSD of [1, 0] and [0.5, 0.5] in bits
    expected = 0.5 * (np.log2(1 / 0.75)) + 0.5 * (0.5 * np.log2(0.5 / 0.75) + 0.5 * np.log2(0.5 / 0.25))
    assert replicas.jsd(np.array([1.0, 0.0]), np.array([1.0, 1.0])) == pytest.approx(expected)


def test_transitions_are_counted_between_the_cores_of_the_rotamers():
    # 60 -> (fluctuation across the border at 120 does not count) -> 180 -> -60 -> 60
    angles = np.array([60.0, 65.0, 119.0, 125.0, 95.0, 180.0, 170.0, -60.0, -55.0, 61.0])
    assert replicas.transitions(angles) == 3
    assert replicas.transitions(np.array([60.0, 99.0, 61.0, 98.0])) == 0      # never leaves gg
    assert replicas.transitions(np.array([179.0])) == 0


def test_linkages_are_classified_from_the_torsion_metadata():
    (link,) = replicas.linkages_of(_metadata(), ["2_1_phi", "2_1_psi", "2_1_omega"])
    assert (link["class"], link["subclass"], link["positions"], link["label"]) == ("1-6", "1-6", "1-6", "Man α1–6 Man")
    assert link["torsions"] == {"phi": 0, "psi": 1, "omega": 2}
    substituent = [{"canonical_name": "1_9_omega", "family": "glycosidic", "atom_names": ["C4", "C5", "C6", "O6"],
                    "residue_ids": [1, 1, 1, 1], "residue_names": ["6LB", "6LB", "6LB", "6LB"]}]
    (alone,) = replicas.linkages_of(substituent, ["1_9_omega"])
    assert alone["class"] == "substituent"


# ---------------------------------------------------------------------------
# Replica boundaries
# ---------------------------------------------------------------------------
def test_runs_recorded_by_follows_the_evidence_audit():
    records = {"records_between_frames": {"END": {"before_frame": [30003]}, "CRYST1": {"before_frame": [10001, 20002]}}}
    assert replicas.runs_recorded_by(records, 30003) == ("records", [10001, 10001, 10001])
    restart = {"model_numbers": {"first": 1, "largest": 10001, "last": 10001, "breaks": 3, "sequential": False}}
    assert replicas.runs_recorded_by(restart, 40004) == ("model_numbers", [10001] * 4)
    assert replicas.runs_recorded_by({"model_numbers": {"sequential": True}}, 10001) == ("one_block", [10001])
    assert replicas.runs_recorded_by({"model_numbers": {"largest": 9999}}, 30003) == ("wrapped_numbers", [30003])
    assert replicas.runs_recorded_by({}, 777) == ("unknown", [777])


def test_boundaries_of_old_entries_recorded_single_assumed_and_unknown():
    recorded = replicas.replica_boundaries({"frames": 30003, "source": {
        "records_between_frames": {"END": {"before_frame": [10001, 20002, 30003]}}}})
    assert (recorded.basis, recorded.recorded_by, recorded.starts) == ("recorded", "records", [0, 10001, 20002])
    single = replicas.replica_boundaries({"frames": 10001, "source": {"model_numbers": {"sequential": True}}})
    assert (single.basis, single.lengths) == ("single", [10001])
    wrapped = {"frames": 40004, "source": {"model_numbers": {"largest": 9999}}}
    assumed = replicas.replica_boundaries(wrapped)
    assert (assumed.basis, assumed.recorded_by, assumed.lengths) == ("assumed", None, [10001] * 4)
    assert assumed.evidence["class_test"]["entries"] == 105
    unknown = replicas.replica_boundaries({"frames": 12345, "source": {"model_numbers": {"largest": 9999}}})
    assert (unknown.basis, unknown.lengths) == ("none", [12345]) and "no equal division" in unknown.evidence["why"]
    folder = replicas.replica_boundaries(None, 480)
    assert (folder.basis, folder.lengths) == ("none", [480])


def test_boundaries_of_new_and_extended_entries_are_their_segments():
    manifest = {"frames": 25000, "segments": [
        {"name": "stored", "first_frame": 0, "frames": 20002, "time_per_frame_ps": None,
         "carried_from": {"glycam_name": "x", "manifest_sha256": "0" * 64, "segment": None},
         "source": {"records_between_frames": {"CRYST1": {"before_frame": [10001]}, "END": {"before_frame": [20002]}}}},
        {"name": "run7_replica_1", "first_frame": 20002, "frames": 4998, "time_per_frame_ps": 50.0},
    ]}
    b = replicas.replica_boundaries(manifest)
    assert [r.name for r in b.replicas] == ["stored_run_1", "stored_run_2", "run7_replica_1"]
    assert b.starts == [0, 10001, 20002] and b.lengths == [10001, 10001, 4998]
    assert (b.basis, b.recorded_by) == ("recorded", "segments")
    assert [r.recorded_by for r in b.replicas] == ["records", "records", "segments"]
    assert b.replicas[2].time_per_frame_ps == 50.0 and b.replicas[2].segment == 1
    with pytest.raises(replicas.ReplicaError):
        replicas.replica_boundaries({"frames": 10, "segments": [{"name": "a", "first_frame": 0, "frames": 4},
                                                               {"name": "b", "first_frame": 5, "frames": 5}]})


def test_class_test_finds_the_finest_supported_division():
    rng = np.random.default_rng(3)
    percentiles = {}
    names = ["a", "b", "c", "d", "e", "f"]
    for name in names:
        p = rng.random(3000)
        p[[1000, 2000]] = 0.995                                   # the change into the first frame of a run
        percentiles[name] = p
    result = replicas.class_test({3000: names, 5000: ["a"]}, percentiles)
    assert result[3000]["accepted"]["runs"] == 3 and result[3000]["accepted"]["run_frames"] == 1000
    assert result[5000]["accepted"] is None and result[5000]["tested"] == 1


# ---------------------------------------------------------------------------
# Verdicts on synthetic entries with known answers
# ---------------------------------------------------------------------------
def test_replicas_of_one_process_agree_and_are_ok():
    tables, manifest = synthetic([{"frames": 4000, "populations": EVEN}] * 3)
    doc = replicas.diagnose(tables, replicas.replica_boundaries(manifest), glycam_name=manifest["glycam_name"])
    assert doc["boundaries"]["basis"] == "recorded" and doc["boundaries"]["full_length_replicas"] == 3
    assert doc["verdict"]["value"] == "ok" and doc["verdict"]["reasons"] == []
    between, floor = doc["between"], doc["floor"]
    assert between["population_tv_mean"] < 0.05 and floor["population_tv"] < 0.05
    assert between["overlap_mean"] > 0.8 and len(between["pairs"]) == 3
    assert doc["clusters"]["pooled"] == pytest.approx(EVEN, abs=0.02)
    (link,) = doc["linkages"]
    assert link["class"] == "1-6" and link["torsions"]["omega"]["rotamer_names"] == ["gg", "gt", "tg"]
    assert all(t > 20 for t in link["torsions"]["omega"]["transitions"])
    assert link["jsd"] < 0.05 and link["rotamer_missing"] == []


def test_replicas_that_sample_different_states_need_more_sampling():
    runs = [{"frames": 3000, "populations": [1, 0, 0, 0, 0], "omega": 60.0},
            {"frames": 3000, "populations": [0, 1, 0, 0, 0], "omega": 180.0},
            {"frames": 3000, "populations": [0, 0, 1, 0, 0], "omega": -60.0}]
    tables, manifest = synthetic(runs)
    doc = replicas.diagnose(tables, replicas.replica_boundaries(manifest))
    assert doc["between"]["population_tv_mean"] == 1.0 and doc["between"]["population_tv_max"] == 1.0
    assert doc["between"]["top_cluster_same"] is False and doc["between"]["overlap_mean"] < 0.05
    assert doc["clusters"]["pooled"][:3] == pytest.approx([1 / 3] * 3, abs=1e-4)
    assert [r["clusters_missing"] for r in doc["replicas"]] == [[1, 2], [0, 2], [0, 1]]
    verdict = doc["verdict"]
    assert verdict["value"] == "more_sampling_needed"
    assert [reason["code"] for reason in verdict["reasons"]] == [
        "population_tv", "cluster_missing", "rotamer_missing", "omega_never_changes"]
    stuck = next(r for r in verdict["reasons"] if r["code"] == "omega_never_changes")["replicas"]
    assert [(s["replica"], s["rotamer"], s["transitions"]) for s in stuck] == [
        ("run_1", "gg", 0), ("run_2", "gt", 0), ("run_3", "tg", 0)]
    omega = doc["linkages"][0]["torsions"]["omega"]
    assert [max(r) for r in omega["rotamers"]] == [1.0, 1.0, 1.0] and omega["jsd"] == pytest.approx(1.0, abs=0.02)


def test_one_stuck_omega_alone_is_a_suggestion():
    runs = [{"frames": 4000, "populations": EVEN}, {"frames": 4000, "populations": EVEN},
            {"frames": 4000, "populations": EVEN, "omega": 60.0}]
    tables, manifest = synthetic(runs)
    doc = replicas.diagnose(tables, replicas.replica_boundaries(manifest))
    assert doc["verdict"]["value"] == "more_sampling_suggested"
    codes = [reason["code"] for reason in doc["verdict"]["reasons"]]
    assert "omega_never_changes" in codes and "population_tv" not in codes and "cluster_missing" not in codes


# ---------------------------------------------------------------------------
# omega rotamers under the two torsion definitions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("info,version", [
    ({}, 1),                                                                         # published before October 2026
    ({"torsions": {"metadata": []}}, 1),
    ({"torsions": {"conventions": {"omega": "C4-C5-C6-O6 of the acceptor ..."}}}, 1),  # text conventions of PR #7
    ({"torsions": {"conventions": {"version": 2}}}, 2),
])
def test_the_conventions_version_is_read_from_info_json(info, version):
    assert replicas.conventions_version(info) == version


STUCK = [{"frames": 3000, "populations": [1, 0, 0, 0, 0], "omega": 60.0},     # gg (C4-C5-C6-O6 +60)
         {"frames": 3000, "populations": [0, 1, 0, 0, 0], "omega": 180.0},    # gt
         {"frames": 3000, "populations": [0, 0, 1, 0, 0], "omega": -60.0}]    # tg


def test_version_1_omega_c4_c5_c6_o6_names_the_thirds_gg_gt_tg():
    tables, manifest = synthetic(STUCK, version=1)
    doc = replicas.diagnose(tables, replicas.replica_boundaries(manifest))
    assert doc["torsion_conventions_version"] == 1
    omega = doc["linkages"][0]["torsions"]["omega"]
    assert omega["rotamer_names"] == ["gg", "gt", "tg"] and omega["rotamer_centres"] == [-60.0, 60.0, 180.0]
    stuck = next(r for r in doc["verdict"]["reasons"] if r["code"] == "omega_never_changes")["replicas"]
    assert [s["rotamer"] for s in stuck] == ["gg", "gt", "tg"]


def test_version_2_omega_o5_c5_c6_o6_names_the_thirds_gt_tg_gg():
    tables, manifest = synthetic(STUCK, version=2)
    doc = replicas.diagnose(tables, replicas.replica_boundaries(manifest))
    assert doc["torsion_conventions_version"] == 2
    omega = doc["linkages"][0]["torsions"]["omega"]
    assert omega["rotamer_names"] == ["gt", "tg", "gg"] and omega["rotamer_centres"] == [60.0, 180.0, -60.0]
    # the same conformations get the same names under both versions
    stuck = next(r for r in doc["verdict"]["reasons"] if r["code"] == "omega_never_changes")["replicas"]
    assert [s["rotamer"] for s in stuck] == ["gg", "gt", "tg"]


def by_rotamer(omega):
    """Occupancy of every replica by rotamer name."""
    return [dict(zip(omega["rotamer_names"], occupancy)) for occupancy in omega["rotamers"]]


def test_occupancies_of_the_two_versions_agree_by_rotamer_name():
    runs = [{"frames": 4000, "populations": EVEN}, {"frames": 4000, "populations": EVEN, "switch": 0.005}]
    docs = []
    for version in (1, 2):
        tables, manifest = synthetic(runs, version=version)
        docs.append(replicas.diagnose(tables, replicas.replica_boundaries(manifest)))
    old, new = (doc["linkages"][0]["torsions"]["omega"] for doc in docs)
    assert old["rotamers"] != new["rotamers"]                       # the thirds are in another order
    assert by_rotamer(old) == by_rotamer(new)
    assert old["transitions"] == new["transitions"]
    assert docs[0]["verdict"] == docs[1]["verdict"]


def _sia_2_8_metadata(version):
    omega = ["C6", "C7", "C8", "O8"] if version == 1 else ["O7", "C7", "C8", "O8"]
    return [
        {"canonical_name": "3_2_phi", "family": "glycosidic", "atom_names": ["O6", "C2", "O8", "C8"],
         "residue_ids": [3, 3, 2, 2], "residue_names": ["0SA", "0SA", "8SA", "8SA"]},
        {"canonical_name": "3_2_psi", "family": "glycosidic", "atom_names": ["C2", "O8", "C8", "C7"],
         "residue_ids": [3, 2, 2, 2], "residue_names": ["0SA", "8SA", "8SA", "8SA"]},
        {"canonical_name": "3_2_omega", "family": "glycosidic", "atom_names": omega,
         "residue_ids": [2, 2, 2, 2], "residue_names": ["8SA", "8SA", "8SA", "8SA"]},
    ]


@pytest.mark.parametrize("version,centres", [(1, [-60.0, 60.0, 180.0]), (2, [60.0, 180.0, -60.0])])
def test_omega_of_a_2_8_linkage_has_the_iupac_centres_of_its_thirds(version, centres):
    (link,) = replicas.linkages_of(_sia_2_8_metadata(version), ["3_2_phi", "3_2_psi", "3_2_omega"], version)
    assert (link["class"], link["subclass"]) == ("sia", "sia 2-8")
    # O7-C7-C8-O8 at the centre of each third; no gg/gt/tg names for the C7-C8 bond
    assert list(link["omega_rotamer_centres"]) == centres and "omega_rotamer_names" not in link


def test_omega_atoms_that_contradict_the_version_are_refused():
    with pytest.raises(replicas.ReplicaError, match="disagree"):
        replicas.linkages_of(_metadata(1), ["2_1_phi", "2_1_psi", "2_1_omega"], 2)
    with pytest.raises(replicas.ReplicaError, match="disagree"):
        replicas.linkages_of(_sia_2_8_metadata(2), ["3_2_phi", "3_2_psi", "3_2_omega"], 1)


def test_a_static_entry_of_version_2_is_read_as_version_2(tmp_path):
    import polars as pl

    tables, manifest = synthetic([{"frames": 2000, "populations": EVEN}] * 2, version=2)
    n = len(tables.source_frame)
    output = tmp_path / "output"
    output.mkdir()
    pl.DataFrame({"frame": np.arange(n), "PC1": tables.pcs[:, 0].astype(float), "PC2": tables.pcs[:, 1].astype(float),
                  "source_frame": tables.source_frame, "cluster": tables.cluster}).write_csv(output / "pca.csv")
    pl.DataFrame({"frame": np.arange(n), "source_frame": tables.source_frame,
                  **{name: tables.torsions[:, j] for j, name in enumerate(tables.torsion_names)},
                  "cluster": tables.cluster}).write_csv(output / "torsion_glycosidic.csv")
    (output / "info.json").write_text(json.dumps(
        {"torsions": {"metadata": _metadata(2), "conventions": {"version": 2}}}))
    read = replicas.tables_from_static(tmp_path)
    assert read.conventions_version == 2
    doc = replicas.diagnose_static(tmp_path, manifest, glycam_name="x")
    assert doc["linkages"][0]["torsions"]["omega"]["rotamer_names"] == ["gt", "tg", "gg"]


def test_short_and_filtered_replicas_are_not_compared():
    runs = [{"frames": 4000, "populations": EVEN}, {"frames": 4000, "populations": EVEN},
            {"frames": 1500, "populations": [0, 0, 0, 0, 1]},              # short: not full length
            {"frames": 4000, "populations": [0, 0, 0, 0, 1]}]             # most frames removed by the filter
    keep = np.ones(13500, dtype=bool)
    keep[9500:9500 + 3200] = False
    tables, manifest = synthetic(runs, keep=keep)
    doc = replicas.diagnose(tables, replicas.replica_boundaries(manifest))
    assert [r["full_length"] for r in doc["replicas"]] == [True, True, False, False]
    assert [r["retained"] for r in doc["replicas"]] == [4000, 4000, 1500, 800]
    assert doc["between"]["replicas"] == [0, 1] and len(doc["between"]["pairs"]) == 1
    assert doc["verdict"]["value"] == "ok"
    # every replica is still described on its own frames
    assert doc["replicas"][2]["population"][4] == 1.0 and doc["replicas"][3]["clusters_missing"] == [0, 1, 2]


def test_one_replica_or_unknown_runs_are_not_assessable():
    tables, manifest = synthetic([{"frames": 3000, "populations": EVEN}, {"frames": 1000, "populations": EVEN}])
    one = replicas.diagnose(tables, replicas.replica_boundaries(manifest))
    assert one["boundaries"]["basis"] == "recorded" and [r["full_length"] for r in one["replicas"]] == [True, False]
    assert one["verdict"]["value"] == "not_assessable" and one["between"] is None and one["floor"] is None
    assert [r["code"] for r in one["verdict"]["reasons"]] == ["too_few_replicas"]
    assert one["halves"]["population_tv_within"] < 0.1
    unknown = replicas.diagnose(tables, replicas.replica_boundaries(None, 4000))
    assert [r["code"] for r in unknown["verdict"]["reasons"]] == ["replicas_unknown", "too_few_replicas"]


def test_thresholds_are_parameters():
    tables, manifest = synthetic([{"frames": 4000, "populations": EVEN}] * 3)
    boundaries = replicas.replica_boundaries(manifest)
    strict = replicas.Parameters(population_tv=0.0)
    doc = replicas.diagnose(tables, boundaries, parameters=strict)
    assert doc["verdict"]["value"] == "more_sampling_suggested"
    assert doc["verdict"]["thresholds"]["population_tv"] == 0.0 and doc["parameters"]["population_tv"] == 0.0
    three = replicas.diagnose(tables, boundaries, parameters=replicas.Parameters(min_full_replicas=4))
    assert three["verdict"]["value"] == "not_assessable"


def test_floor_is_what_frames_dealt_at_random_give():
    tables, manifest = synthetic([{"frames": 3000, "populations": [1, 0, 0, 0, 0]},
                                  {"frames": 3000, "populations": [0, 1, 0, 0, 0]}])
    boundaries = replicas.replica_boundaries(manifest)
    m = replicas.measure(tables, boundaries, seed=11)
    assert m["between"]["population_tv"][0] == 1.0 and m["floor"]["population_tv"] < 0.05
    again = replicas.measure(tables, boundaries, seed=11)["floor"]
    assert again["population_tv"] == m["floor"]["population_tv"] and np.array_equal(again["jsd"], m["floor"]["jsd"])
    assert replicas.floor_seed("GS00760") == 760 and replicas.floor_seed(None, "x") == replicas.floor_seed(None, "x")


# ---------------------------------------------------------------------------
# The file
# ---------------------------------------------------------------------------
def test_document_is_compact_deterministic_json_with_a_drawable_landscape(tmp_path):
    tables, manifest = synthetic([{"frames": 3000, "populations": EVEN}] * 3)
    boundaries = replicas.replica_boundaries(manifest)
    first = replicas.write(replicas.diagnose(tables, boundaries, glycam_name="x"), tmp_path / "a" / "replicas.json")
    second = replicas.write(replicas.diagnose(tables, boundaries, glycam_name="x"), tmp_path / "b" / "replicas.json")
    assert first.read_bytes() == second.read_bytes() and first.read_bytes().endswith(b"}\n")
    assert first.stat().st_size < 60_000
    doc = json.loads(first.read_bytes())
    assert list(doc)[:4] == ["schema", "schema_version", "glycam_name", "glycoshape_id"]
    land = doc["landscape"]
    assert [p["mass"] for p in land["pooled"]] == [0.6, 0.85, 0.97] and all(p["lines"] for p in land["pooled"])
    for item in land["replicas"]:
        assert item["lines"] and all(len(line) % 2 == 0 and len(line) >= 6 for line in item["lines"])
    grid = land["overlap_grid"]
    assert [sum(item["counts"]) for item in grid["replicas"]] == [3000, 3000, 3000]
    assert all(0 <= cell < grid["size"] ** 2 for item in grid["replicas"] for cell in item["cells"])
    histograms = doc["torsion_histograms"]
    assert histograms["bins"] == 36 and [sum(c) for c in histograms["torsions"][0]["counts"]] == [3000] * 3


def test_tables_of_a_static_entry_and_of_a_process_folder_give_the_same_document(tmp_path):
    import polars as pl

    tables, manifest = synthetic([{"frames": 2000, "populations": EVEN}] * 2)
    n = len(tables.source_frame)
    pca = pl.DataFrame({"frame": np.arange(n), "PC1": tables.pcs[:, 0].astype(float), "PC2": tables.pcs[:, 1].astype(float),
                        "PC3": np.zeros(n), "source_frame": tables.source_frame, "cluster": tables.cluster})
    tor = pl.DataFrame({"frame": np.arange(n), "source_frame": tables.source_frame,
                        **{name: tables.torsions[:, j] for j, name in enumerate(tables.torsion_names)},
                        "cluster": tables.cluster, "center": [None] * n})
    info = {"torsions": {"metadata": _metadata()}}
    static, process = tmp_path / "static", tmp_path / "process"
    (static / "output").mkdir(parents=True)
    (process / "embedding" / "exports").mkdir(parents=True)
    (process / "output").mkdir(parents=True)
    pca.write_csv(static / "output" / "pca.csv")
    tor.write_csv(static / "output" / "torsion_glycosidic.csv")
    pca.write_parquet(process / "embedding" / "exports" / "pca.parquet")
    tor.write_parquet(process / "embedding" / "exports" / "torsion_glycosidic.parquet")
    for folder in (static, process):
        (folder / "output" / "info.json").write_text(json.dumps(info))
    from_static = replicas.diagnose_static(static, manifest, glycam_name="x")
    from_process = replicas.diagnose_result(process, manifest, manifest["frames"], glycam_name="x")
    assert replicas.encode(from_static) == replicas.encode(from_process)
    assert from_static == replicas.diagnose(tables, replicas.replica_boundaries(manifest), glycam_name="x")


def test_folders_named_after_a_branched_glycan_are_read(tmp_path):
    # Process folders carry the GLYCAM name, whose brackets polars would expand as a glob.
    import polars as pl

    tables, manifest = synthetic([{"frames": 2000, "populations": EVEN}] * 2)
    n = len(tables.source_frame)
    pca = pl.DataFrame({"frame": np.arange(n), "PC1": tables.pcs[:, 0].astype(float), "PC2": tables.pcs[:, 1].astype(float),
                        "PC3": np.zeros(n), "source_frame": tables.source_frame, "cluster": tables.cluster})
    tor = pl.DataFrame({"frame": np.arange(n), "source_frame": tables.source_frame,
                        **{name: tables.torsions[:, j] for j, name in enumerate(tables.torsion_names)},
                        "cluster": tables.cluster, "center": [None] * n})
    info = {"torsions": {"metadata": _metadata()}}
    name = "DManpa1-3[DManpa1-6]DManpb1-4DGlcpNAcb1-4[LFucpa1-6]DGlcpNAcb1-OH"
    process, static = tmp_path / "process" / name, tmp_path / "static" / name
    (process / "embedding" / "exports").mkdir(parents=True)
    (process / "output").mkdir(parents=True)
    (static / "output").mkdir(parents=True)
    pca.write_parquet(process / "embedding" / "exports" / "pca.parquet")
    tor.write_parquet(process / "embedding" / "exports" / "torsion_glycosidic.parquet")
    pca.write_csv(static / "output" / "pca.csv")
    tor.write_csv(static / "output" / "torsion_glycosidic.csv")
    for folder in (process, static):
        (folder / "output" / "info.json").write_text(json.dumps(info))
    expected = replicas.diagnose(tables, replicas.replica_boundaries(manifest), glycam_name=name)
    assert replicas.diagnose_result(process, manifest, manifest["frames"], glycam_name=name) == expected
    assert replicas.diagnose_static(static, manifest, glycam_name=name) == expected


def test_tables_that_disagree_are_refused():
    with pytest.raises(replicas.ReplicaError):
        replicas.FrameTables(source_frame=[0, 2, 1], cluster=[0, 0, 0], pcs=np.zeros((3, 2)), torsions=np.zeros((3, 0)),
                             torsion_names=[])
    tables, manifest = synthetic([{"frames": 600, "populations": EVEN}] * 2)
    with pytest.raises(replicas.ReplicaError):
        replicas.measure(tables, replicas.replica_boundaries(None, 1000 - 1))


def test_placement_is_one_setting():
    assert replicas.public_placement({}) is False
    assert replicas.public_placement({"GLYCOSHAPE_REPLICAS_PUBLIC": "1"}) is True
    assert replicas.public_placement({"GLYCOSHAPE_REPLICAS_PUBLIC": "no"}) is False
    assert replicas.admin_key("publish-state", "static/", "GS00760") == "publish-state/static/replicas/GS00760.json"
    assert replicas.STATIC_PATH == "output/replicas.json"


# ---------------------------------------------------------------------------
# A real entry: the numbers of the thesis
# ---------------------------------------------------------------------------
def test_real_entry_reproduces_the_thesis():
    """GS00877 (G47329OU, Glc b1-6 Glc): two recorded runs of 10,001 frames, as published.

    The expected values are those of robustness.json and robustness-linkages.json of the thesis
    (figures/GlycoShape/data, 2026-10-12), which robustness_data.py computed from the same tables.
    """
    z = np.load(DATA / "replicas_GS00877.npz")
    manifest = json.loads(str(z["manifest"]))
    tables = replicas.FrameTables(source_frame=np.arange(len(z["cluster"])), cluster=z["cluster"], pcs=z["pcs"],
                                  torsions=z["torsions"], torsion_names=[str(n) for n in z["names"]],
                                  torsion_metadata=json.loads(str(z["metadata"])))
    boundaries = replicas.replica_boundaries(manifest)
    assert (boundaries.basis, boundaries.recorded_by, boundaries.starts) == ("recorded", "records", [0, 10001])
    m = replicas.measure(tables, boundaries, seed=replicas.floor_seed("GS00877"))
    row = replicas.entry_row(m)
    expected = {"tv": 0.1005, "tv_max": 0.1005, "tv_floor": 0.0096, "absdiff_max": 0.0971, "se_max": 0.0486,
                "overlap": 0.8701, "overlap_min": 0.8701, "overlap_floor": 0.9351, "tv_within": 0.1408,
                "tv_across": 0.1262, "overlap_within": 0.8122, "overlap_across": 0.835}
    assert {key: round(row[key], 4) for key in expected} == expected
    assert row["top_same"] is True and row["absent_clusters"] == [] and row["rotamer_missing"] == 0
    (link,) = row["linkage_rows"]
    assert link["label"] == "Glc β1–6 Glc" and link["jsd_torsion"] == "omega"
    assert {key: round(link[key], 5) for key in ("jsd", "jsd_floor", "jsd_within", "jsd_across")} == {
        "jsd": 0.00319, "jsd_floor": 0.00164, "jsd_within": 0.00866, "jsd_across": 0.00768}
    assert {kind: round(t["jsd"], 5) for kind, t in link["torsions"].items()} == {
        "phi": 0.00196, "psi": 0.00104, "omega": 0.00319}
    omega = link["torsions"]["omega"]["column"]
    assert [m["replica_transitions"][omega][i] for i in (0, 1)] == [805, 783]
    assert [[round(round(float(x), 4), 3) for x in m["rotamers"][i, omega]] for i in (0, 1)] == [
        [0.621, 0.242, 0.137], [0.661, 0.199, 0.14]]
    doc = replicas.diagnose(tables, boundaries, glycoshape_id="GS00877", glycam_name=manifest["glycam_name"])
    assert doc["verdict"]["value"] == "ok" and doc["between"]["population_tv_mean"] == 0.1005
    assert doc["floor"]["seed"] == 877 and doc["replicas"][1]["boundary_rank"] > 0.9
