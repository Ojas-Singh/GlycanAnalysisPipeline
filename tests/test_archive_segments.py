"""Archive entries made of several trajectories and entries that extend another (lib/archive_segments.py).

Every trajectory is synthetic and built in the test; nothing needs the network.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from lib import archive
from lib import archive_segments as segs

from test_archive import atom_table, make_source, mol2_text, pdb_text, random_coords
from test_archive_repair import cut_source

NAME = "DGlcpa1-OH"
ATOMS = 14
CELLS = (
    "CRYST1   29.288   31.797   32.511  90.00  90.00  90.00               1\n",
    "CRYST1   30.100   30.200   30.300  90.00  90.00  90.00               1\n",
    "CRYST1   31.000   31.500   32.000  90.00  90.00  90.00               1\n",
)
LEGACY_KEYS = ["schema", "schema_version", "glycam_name", "folder", "object_key_prefix", "atoms", "frames", "box",
               "trajectory", "files", "source", "verification", "converter", "warnings"]
# cpptraj numbers its TER records, MDAnalysis writes a chain and a segment
LAYOUT_COLUMNS = ["serial number (columns 7-11)", "chain (column 22)",
                  "columns 55-80 (occupancy, B factor, segment, element, charge)"]
LEGACY_SOURCE_KEYS = ["folder", "file", "size", "sha256", "glycam_name_from", "model_numbers",
                      "records_between_frames", "negative_zero_coordinates", "frames_parsed_line_by_line"]


def replica(root, name, coords, header=CELLS[0], mol2=True, **options):
    """A replica folder as a run leaves it: glycan.dry.pdb and glycan.dry.mol2."""
    folder = Path(root) / name
    folder.mkdir(parents=True)
    (folder / "glycan.dry.pdb").write_text(pdb_text(coords, header=header, **options), encoding="ascii")
    if mol2:
        (folder / "glycan.dry.mol2").write_text(mol2_text(list(range(1, coords.shape[1] + 1))), encoding="utf-8")
    return segs.SegmentSource(name, folder / "glycan.dry.pdb", folder / "glycan.dry.mol2" if mol2 else None)


def run_of(root, coords, cuts, cells=CELLS):
    """Replica folders for the pieces of ``coords`` between ``cuts``."""
    return [
        replica(root, f"replica_{index + 1}", coords[start:stop], header=cells[index % len(cells)])
        for index, (start, stop) in enumerate(zip(cuts, cuts[1:]))
    ]


def whole_file_entry(root, sources):
    """The entry the converter makes of the replica files written one after the other (as ``cat`` does)."""
    folder = Path(root) / "data" / NAME
    folder.mkdir(parents=True)
    (folder / "simulation.pdb").write_bytes(b"".join(source.pdb.read_bytes() for source in sources))
    shutil.copy(sources[0].mol2, folder / "structure.mol2")
    return Path(archive.convert_entry(folder, Path(root) / "whole")["entry"])


def files_of(folder):
    return {path.name: path.read_bytes() for path in sorted(Path(folder).iterdir())}


def wrong_count(path):
    """Stands in for ``segs.count_frames``: one frame too many for the second replica."""
    return segs.archive._record_offsets(Path(path), (b"ENDMDL",))[0][b"ENDMDL"].size + ("replica_2" in str(path))


# ---------------------------------------------------------------------------
# Several sources
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("workers", [1, 3])
def test_replicas_give_the_entry_of_the_file_they_would_be_concatenated_to(tmp_path, workers):
    coords = random_coords(60, ATOMS, seed=1)
    sources = run_of(tmp_path / "run", coords, [0, 25, 40, 60])
    reference = whole_file_entry(tmp_path, sources)

    result = segs.convert_segments(sources, tmp_path / "archive", NAME, workers=workers)
    entry, manifest = Path(result["entry"]), result["manifest"]
    assert result["status"] == "converted" and entry.name == NAME
    assert manifest == archive.load_manifest(entry)

    # the trajectory and the topology are those of the converted whole file, byte for byte
    whole = archive.load_manifest(reference)
    assert (entry / "trajectory.xtc").read_bytes() == (reference / "trajectory.xtc").read_bytes()
    assert (entry / "topology.pdb").read_bytes() == (reference / "topology.pdb").read_bytes()
    assert manifest["trajectory"] == whole["trajectory"] and manifest["box"] == whole["box"]
    assert (manifest["atoms"], manifest["frames"]) == (ATOMS, 60)
    assert manifest["box"]["kind"] == "varying" and [cell["first_frame"] for cell in manifest["box"]["cells"]] == [0, 25, 40]
    assert np.array_equal(archive.ArchiveEntry(entry).read_frames(unit="milliangstrom"), coords)
    assert segs.compare_trajectories(entry, reference) == {"equal": True, "frames_compared": 60, "first_difference": None}

    # every key a manifest has always had, and the segments
    assert [key for key in manifest if key in LEGACY_KEYS] == LEGACY_KEYS
    assert sorted(set(manifest) - set(LEGACY_KEYS)) == ["segments", "segments_version"]
    assert sorted(manifest["files"]) == ["name.json", "structure.mol2", "topology.pdb", "trajectory.xtc"]
    assert json.loads((entry / "name.json").read_text()) == {"indexOrderedSequence": NAME}
    source = manifest["source"]
    assert (source["folder"], source["file"], source["glycam_name_from"], source["segments"]) == (NAME, None, "option", 3)
    assert source["size"] == sum(s.pdb.stat().st_size for s in sources)
    hashes = [hashlib.sha256(s.pdb.read_bytes()).hexdigest() for s in sources]
    assert source["sha256"] == hashlib.sha256("\n".join(hashes).encode()).hexdigest()

    segments = manifest["segments"]
    assert [(s["index"], s["name"], s["first_frame"], s["frames"]) for s in segments] == [
        (0, "replica_1", 0, 25), (1, "replica_2", 25, 15), (2, "replica_3", 40, 20)]
    trajectory = (entry / "trajectory.xtc").read_bytes()
    position = 0
    for segment, part, digest in zip(segments, (coords[:25], coords[25:40], coords[40:]), hashes):
        assert segment["source"]["file"] == "glycan.dry.pdb" and segment["source"]["sha256"] == digest
        assert segment["coordinates_sha256"] == hashlib.sha256(part.astype("<i4").tobytes()).hexdigest()
        assert segment["trajectory_offset"] == position and segment["atom_records"] == "identical"
        assert segment["verification"]["frames_compared"] == segment["frames"]
        assert segment["verification"]["mismatched_coordinates"] == 0 and segment["time_per_frame_ps"] is None
        assert segment["source"]["model_numbers"]["first"] == 1 and "carried_from" not in segment
        position += segment["trajectory_bytes"]
        # the bytes of a segment are a trajectory of their own, with the steps it has in the whole
        steps = np.frombuffer(trajectory[segment["trajectory_offset"] + 8:][:4], dtype=">i4")
        assert int(steps[0]) == segment["first_frame"]
    assert position == len(trajectory)
    assert manifest["verification"]["frames_compared"] == 60 and manifest["verification"]["segments_checked"] == 3
    assert manifest["converter"]["name"] == "lib/archive_segments.py" and manifest["warnings"] == []

    # the readers of an entry read it
    report = archive.verify_entry(entry)
    assert report["ok"] and report["segments"] == report["segments_checked"] == 3 and report["frames"] == 60
    index, problems = archive.build_index(tmp_path / "archive")
    assert problems == [] and index["entries"] == 1 and index["repaired"] == []
    assert index["records"][0]["source_sha256"] == source["sha256"] and index["records"][0]["frames"] == 60
    assert archive.entry_inputs(entry)["glycam_name"] == NAME
    assert not (tmp_path / "archive" / archive.STAGING_DIR).exists() or not any(
        (tmp_path / "archive" / archive.STAGING_DIR).iterdir())


def test_entry_of_one_simulation_pdb_is_written_as_before(tmp_path, monkeypatch):
    """convert_entry does not know about segments: its manifest has the keys it always had, in their order."""
    monkeypatch.setattr(archive, "_utc_now", lambda: "2026-01-01T00:00:00Z")
    source = make_source(tmp_path / "data", NAME, random_coords(20, ATOMS, seed=2), info_json=NAME)
    (source / "structure.mol2").write_text(mol2_text(list(range(1, ATOMS + 1))))
    first = archive.convert_entry(source, tmp_path / "a")
    manifest = first["manifest"]
    assert list(manifest) == LEGACY_KEYS and list(manifest["source"]) == LEGACY_SOURCE_KEYS
    assert manifest["converter"]["name"] == "lib/archive.py" and manifest["converter"]["version"] == "1.0.0"
    assert archive.segments_of(manifest) == [] and archive.repair_summary(manifest) is None
    again = archive.convert_entry(source, tmp_path / "b")
    assert files_of(first["entry"]) == files_of(again["entry"])
    # ... and the same source as the only segment gives the same trajectory and topology
    one = segs.convert_segments([segs.SegmentSource("only", source / "simulation.pdb", source / "structure.mol2")],
                                tmp_path / "c", NAME)
    for name in ("trajectory.xtc", "topology.pdb", "structure.mol2"):
        assert one["manifest"]["files"][name] == manifest["files"][name]


def test_time_per_frame_and_what_is_known_about_the_run_are_recorded(tmp_path):
    coords = random_coords(12, ATOMS, seed=3)
    sources = run_of(tmp_path / "run", coords, [0, 5, 12])
    run = {"dt_ps": 0.002, "nsteps": 250000000, "length_ns": 500.0}
    sources = [segs.SegmentSource(s.name, s.pdb, s.mol2, 50.0, run) for s in sources]
    info = tmp_path / "info.json"
    info.write_text(json.dumps({"indexOrderedSequence": "DGlcpb1-OH", "conformerID": "1ogg"}))
    manifest = segs.convert_segments(sources, tmp_path / "archive", NAME, info=info, source_name="run-42")["manifest"]
    assert [(s["time_per_frame_ps"], s["run"]) for s in manifest["segments"]] == [(50.0, run), (50.0, run)]
    assert manifest["source"]["folder"] == "run-42"
    entry = tmp_path / "archive" / NAME
    assert (entry / "info.json").read_bytes() == info.read_bytes() and "info.json" in manifest["files"]
    assert manifest["warnings"] == [
        "info.json indexOrderedSequence differs from the GLYCAM name taken from option: DGlcpb1-OH"]
    with pytest.raises(archive.SourceError, match="time per frame must be positive"):
        segs.convert_segments([segs.SegmentSource("a", sources[0].pdb, sources[0].mol2, 0.0)], tmp_path / "x", NAME)
    with pytest.raises(archive.SourceError, match="distinct, non-empty names"):
        segs.convert_segments([sources[0], sources[0]], tmp_path / "x", NAME)
    with pytest.raises(archive.SourceError, match="no such trajectory file"):
        segs.convert_segments([segs.SegmentSource("a", tmp_path / "missing.pdb")], tmp_path / "x", NAME)
    with pytest.raises(archive.SourceError, match="outside"):
        segs.convert_segments(sources, sources[0].pdb.parent / "archive", NAME)
    assert not (tmp_path / "x").exists() or not any((tmp_path / "x").iterdir())


def test_source_without_cryst1_has_no_box(tmp_path):
    """In one concatenated file the cell of the run before would stay in force; a replica file stands alone."""
    coords = random_coords(30, ATOMS, seed=4)
    sources = [
        replica(tmp_path / "run", "a", coords[:10], header=CELLS[0]),
        replica(tmp_path / "run", "b", coords[10:20], header=""),
        replica(tmp_path / "run", "c", coords[20:], header=CELLS[1]),
    ]
    result = segs.convert_segments(sources, tmp_path / "archive", NAME)
    manifest = result["manifest"]
    assert [segment["box"]["kind"] for segment in manifest["segments"]] == ["constant", "none", "constant"]
    assert manifest["box"] == {"kind": "varying", "cells_total": 3, "cells": [
        {"first_frame": 0, "lengths_angstrom": [29.288, 31.797, 32.511], "angles_degree": [90.0, 90.0, 90.0]},
        {"first_frame": 10, "lengths_angstrom": None, "angles_degree": None},
        {"first_frame": 20, "lengths_angstrom": [30.1, 30.2, 30.3], "angles_degree": [90.0, 90.0, 90.0]},
    ]}
    boxes = archive._read_back(Path(result["entry"]), ATOMS)["boxes"]
    assert (boxes[:10] != 0).any() and not boxes[10:20].any() and (boxes[20:] != 0).any()
    assert archive.verify_entry(result["entry"])["ok"]

    for headers, kind in ((("", ""), "none"), ((CELLS[0], CELLS[0]), "constant")):
        pair = [replica(tmp_path / kind, name, coords[:5], header=header) for name, header in zip("ab", headers)]
        assert segs.convert_segments(pair, tmp_path / f"archive-{kind}", NAME)["manifest"]["box"]["kind"] == kind


# ---------------------------------------------------------------------------
# Same atoms, same order
# ---------------------------------------------------------------------------
def changed_atoms(change):
    atoms = atom_table(ATOMS)
    if change == "renamed":
        atoms[6] = ("N2",) + atoms[6][1:]
    elif change == "other residue":
        atoms[9] = (atoms[9][0], "0MA") + atoms[9][2:]
    elif change == "other order":
        atoms[3], atoms[4] = atoms[4], atoms[3]
    elif change == "residue number":
        atoms[12] = atoms[12][:2] + (7,) + atoms[12][3:]
    return atoms


@pytest.mark.parametrize("change,message", [
    ("renamed", r"1 of 14 atoms differ in name, residue or order; the first is atom 7: name 'N2' residue '0GB' 2 in "
                r"replica_2 against name 'H7' residue '0GB' 2 in replica_1"),
    ("other residue", r"the first is atom 10: name 'O1' residue '0MA' 2 in replica_2 against name 'O1' residue '0GB' 2"),
    ("other order", r"2 of 14 atoms differ in name, residue or order; the first is atom 4"),
    ("residue number", r"the first is atom 13: name 'O4' residue '4YB' 7 in replica_2 against name 'O4' residue '4YB' 3"),
    ("fewer", r"replica_2 has 13 atoms, replica_1 has 14"),
])
def test_replica_with_other_atoms_is_refused_before_anything_is_converted(tmp_path, change, message):
    coords = random_coords(20, ATOMS, seed=5)
    first = replica(tmp_path / "run", "replica_1", coords[:10])
    if change == "fewer":
        second = replica(tmp_path / "run", "replica_2", coords[10:, :13])
    else:
        second = replica(tmp_path / "run", "replica_2", coords[10:], atoms=changed_atoms(change))
    with pytest.raises(segs.SegmentMismatchError, match=message):
        segs.convert_segments([first, second], tmp_path / "archive", NAME)
    assert not (tmp_path / "archive").exists()             # refused on the first frames: nothing was written

    # a difference that only shows after the first frame is found by the conversion of that replica
    text = second.pdb.read_text().split("MODEL")
    good = pdb_text(coords[10:12]).split("MODEL")
    second.pdb.write_text("MODEL".join([good[0], good[1]] + text[2:]))
    second.mol2.write_bytes(first.mol2.read_bytes())
    with pytest.raises(archive.PdbFormatError, match=r"^replica_2 \(.*glycan.dry.pdb\): frame 1.* the first frame"):
        segs.convert_segments([first, second], tmp_path / "archive", NAME)
    assert not (tmp_path / "archive" / NAME).exists() and not any((tmp_path / "archive" / archive.STAGING_DIR).iterdir())


def test_other_columns_of_the_atom_records_are_recorded_not_refused(tmp_path):
    """Serial numbers, chain and the columns behind the coordinates are layout: MDAnalysis and cpptraj differ there."""
    coords = random_coords(16, ATOMS, seed=6)
    first = replica(tmp_path / "run", "cpptraj", coords[:8])
    second = replica(tmp_path / "run", "mdanalysis", coords[8:], header="", dialect="mdanalysis")
    manifest = segs.convert_segments([first, second], tmp_path / "archive", NAME)["manifest"]
    assert manifest["segments"][0]["atom_records"] == "identical"
    assert manifest["segments"][1]["atom_records"] == {"differ_in": LAYOUT_COLUMNS, "atoms": 14}
    assert manifest["warnings"] == [
        "the atom records of segment 1 (mdanalysis) differ from topology.pdb in serial number (columns 7-11), "
        "chain (column 22), columns 55-80 (occupancy, B factor, segment, element, charge); names, residues and "
        "order are the same"]
    assert np.array_equal(archive.ArchiveEntry(tmp_path / "archive" / NAME).read_frames(unit="milliangstrom"), coords)


def test_structures_of_the_replicas_have_to_agree(tmp_path):
    coords = random_coords(10, ATOMS, seed=7)
    first, second = run_of(tmp_path / "run", coords, [0, 5, 10])
    text = second.mol2.read_text()
    second.mol2.write_text(text.replace("13 13 14 1\n", ""))                       # one bond less
    with pytest.raises(segs.SegmentMismatchError, match="0 bonds only in the MOL2 of replica_2, 1 only in structure.mol2"):
        segs.convert_segments([first, second], tmp_path / "archive", NAME)
    second.mol2.write_text(text.replace(" X5 ", " Y5 "))
    with pytest.raises(segs.SegmentMismatchError, match="1 of 14 atom names differ; the first is atom 5: 'Y5' against 'X5'"):
        segs.convert_segments([first, second], tmp_path / "archive", NAME)
    second.mol2.write_text(mol2_text(list(range(1, 13))))
    with pytest.raises(segs.SegmentMismatchError, match="the MOL2 of replica_2 has 12 atoms"):
        segs.convert_segments([first, second], tmp_path / "archive", NAME)
    first.mol2.write_text(mol2_text(list(range(1, 13))))
    with pytest.raises(segs.SegmentMismatchError, match="has 12 atoms, the trajectory has 14"):
        segs.convert_segments([first, second], tmp_path / "archive", NAME)
    first.mol2.write_text(text.replace("5 5 6 1", "5 5 99 1"))
    with pytest.raises(archive.SourceError, match="a bond names atom '99'"):
        segs.convert_segments([first, second], tmp_path / "archive", NAME)
    # a structure given from outside may come from other tools: differences are warnings then
    first.mol2.write_text(text)
    second.mol2.write_text(text)
    outside = tmp_path / "other.mol2"
    outside.write_text(text.replace(" C.3 ", " Cg ").replace("13 13 14 1\n", ""))
    manifest = segs.convert_segments([first, second], tmp_path / "archive", NAME, mol2=outside)["manifest"]
    assert len(manifest["warnings"]) == 2 and "14 atoms differ in type or substructure" in manifest["warnings"][0]
    assert (tmp_path / "archive" / NAME / "structure.mol2").read_bytes() == outside.read_bytes()
    with pytest.raises(archive.SourceError, match="no MOL2 file"):
        segs.convert_segments([segs.SegmentSource("a", first.pdb)], tmp_path / "none", NAME)


def test_mol2_as_the_post_processing_of_a_run_writes_it_is_read(tmp_path):
    """ParmEd: a CRYSIN and a SUBSTRUCTURE section, right-aligned columns, charges."""
    path = tmp_path / "glycan.dry.mol2"
    path.write_text(
        "@<TRIPOS>MOLECULE\nAGLC\n4 3 2 0 1\nBIOPOLYMER\nUSER_CHARGES\n"
        "@<TRIPOS>CRYSIN\n   41.6050    41.6050    41.6050    90.0000    90.0000    90.0000 1 1\n"
        "@<TRIPOS>ATOM\n"
        "       1 C1          17.8000    29.7600    24.4000 CC3162        1 AGLC       0.340000\n"
        "       2 H1          17.1600    30.6600    24.2700 HCA1          1 AGLC       0.090000\n"
        "       3 O1          18.5400    29.9100    25.5900 OC311         1 AGLC      -0.650000\n"
        "       4 C1          19.7600    28.9300    23.3700 CC3162        2 BGLC       0.340000\n"
        "@<TRIPOS>BOND\n     1     1     2 1\n     2     3     1 1\n     3     3     4 1\n"
        "@<TRIPOS>SUBSTRUCTURE\n     1 AGLC            1 RESIDUE           0 ****  ROOT      0\n"
        "     2 BGLC            4 RESIDUE           0 ****  ****      0\n"
    )
    read = segs.read_mol2(path)
    assert read["atoms"] == [("C1", "CC3162", "1", "AGLC"), ("H1", "HCA1", "1", "AGLC"), ("O1", "OC311", "1", "AGLC"),
                             ("C1", "CC3162", "2", "BGLC")]
    assert read["bonds"] == {(0, 1), (0, 2), (2, 3)}
    same = dict(read, bonds=set(read["bonds"]))
    assert segs.compare_mol2(read, same, "a", "b", same_run=True) == []
    path.write_text("@<TRIPOS>MOLECULE\nempty\n")
    with pytest.raises(archive.SourceError, match="has no @<TRIPOS>ATOM records"):
        segs.read_mol2(path)
    with pytest.raises(archive.SourceError, match="cannot read"):
        segs.read_mol2(tmp_path / "missing.mol2")


# ---------------------------------------------------------------------------
# Rerun, damaged sources, parallel conversion
# ---------------------------------------------------------------------------
def test_second_run_skips_and_a_replica_that_was_simulated_again_is_converted_again(tmp_path):
    coords = random_coords(20, ATOMS, seed=8)
    sources = run_of(tmp_path / "run", coords, [0, 10, 20])
    first = segs.convert_segments(sources, tmp_path / "archive", NAME)
    again = segs.convert_segments(sources, tmp_path / "archive", NAME)
    assert again["status"] == "skipped" and again["manifest"] == first["manifest"] and again["entry"] == first["entry"]

    # same size, other coordinates: only the content says that it is another simulation
    size = sources[1].pdb.stat().st_size
    sources[1].pdb.write_text(pdb_text(random_coords(10, ATOMS, seed=9), header=CELLS[1]), encoding="ascii")
    assert sources[1].pdb.stat().st_size == size
    changed = segs.convert_segments(sources, tmp_path / "archive", NAME)
    assert changed["status"] == "converted"
    assert changed["manifest"]["segments"][0] == first["manifest"]["segments"][0]
    assert changed["manifest"]["segments"][1]["coordinates_sha256"] != first["manifest"]["segments"][1]["coordinates_sha256"]

    for change in (
        {"force": True}, {"source_name": "other"},
        {"sources": [segs.SegmentSource(s.name, s.pdb, s.mol2, 50.0) for s in sources]},
        {"sources": sources[:1]},
    ):
        arguments = {"sources": sources, **change}
        assert segs.convert_segments(arguments.pop("sources"), tmp_path / "archive", NAME, **arguments)["status"] == "converted"
    sources[0].mol2.write_text(sources[0].mol2.read_text().replace("SMALL", "BIOPOLYMER"))
    assert segs.convert_segments(sources[:1], tmp_path / "archive", NAME)["status"] == "converted"
    assert segs.convert_segments(sources[:1], tmp_path / "archive", NAME)["status"] == "skipped"

    # the converter of one simulation.pdb does not replace such an entry unasked
    plain = make_source(tmp_path / "data", NAME, coords)
    with pytest.raises(archive.ArchiveError, match="holds 1 segments"):
        archive.convert_entry(plain, tmp_path / "archive")
    assert archive.convert_entry(plain, tmp_path / "archive", force=True)["status"] == "converted"
    occupied = tmp_path / "occupied" / NAME
    occupied.mkdir(parents=True)
    (occupied / "notes.txt").write_text("not an entry")
    with pytest.raises(archive.ArchiveError, match="is not an archive entry; it is left untouched"):
        segs.convert_segments(sources, tmp_path / "occupied", NAME)
    assert (occupied / "notes.txt").read_text() == "not an entry"


def test_damaged_replica_is_refused_and_named_or_repaired_into_its_segment(tmp_path):
    source, kept = cut_source(tmp_path / "cut", folder="damaged", frames=20, at=11, seed=11)
    coords = random_coords(20, kept.shape[1], seed=10)
    good = replica(tmp_path / "run", "good", coords)
    damaged = segs.SegmentSource("damaged", source / "simulation.pdb")
    for workers in (1, 2):
        with pytest.raises(archive.PdbFormatError, match=r"^damaged \(.*simulation.pdb\): frame 11 .*--repair"):
            segs.convert_segments([good, damaged], tmp_path / "archive", NAME, workers=workers)
    assert not (tmp_path / "archive" / NAME).exists()

    result = segs.convert_segments([good, damaged], tmp_path / "archive", NAME, repair=True, workers=2)
    manifest = result["manifest"]
    assert "source_irregularities" not in manifest and "source_irregularities" not in manifest["segments"][0]
    record = manifest["segments"][1]["source_irregularities"]
    assert (record["frames_in_source"], record["frames_kept"], record["frames_dropped"]) == (20, 19, 1)
    assert manifest["segments"][1]["frames"] == 19 and manifest["frames"] == 39
    assert record["items"][0]["frames_kept_before"] == 11          # counted in its source, as in its own entry
    assert "segment 1 (damaged) comes from an irregular source" in manifest["warnings"][0]
    summary = archive.repair_summary(manifest)
    assert summary == {"frames_in_source": 20, "frames_dropped": 1, "bytes_ignored": record["bytes_ignored"],
                       "truncated": False, "counts": {"incomplete_frame": 1}, "segments": [1]}
    report = archive.verify_entry(result["entry"])
    assert report["ok"] and report["repaired"] is True
    index, problems = archive.build_index(tmp_path / "archive")
    assert problems == [] and index["repaired"] == [NAME] and index["records"][0]["repaired"] == summary
    frames = archive.ArchiveEntry(result["entry"]).read_frames(unit="milliangstrom")
    assert np.array_equal(frames[:20], coords) and np.array_equal(frames[20:], kept)


def test_parts_are_written_again_when_a_source_has_other_frames_than_were_counted(tmp_path, monkeypatch):
    coords = random_coords(40, ATOMS, seed=12)
    sources = run_of(tmp_path / "run", coords, [0, 10, 25, 40])
    assert [segs.count_frames(source.pdb) for source in sources] == [10, 15, 15]
    single = tmp_path / "single.pdb"
    single.write_text("".join(line for line in pdb_text(coords[:1]).splitlines(keepends=True)
                              if not line.startswith(("MODEL", "ENDMDL"))))
    assert segs.count_frames(single) == 1                      # a file without MODEL records is one frame
    reference = segs.convert_segments(sources, tmp_path / "serial", NAME, workers=1)
    monkeypatch.setattr(segs, "count_frames", wrong_count)
    result = segs.convert_segments(sources, tmp_path / "parallel", NAME, workers=3)
    assert files_of(result["entry"])["trajectory.xtc"] == files_of(reference["entry"])["trajectory.xtc"]
    assert [s["first_frame"] for s in result["manifest"]["segments"]] == [0, 10, 25]
    assert archive.verify_entry(result["entry"])["ok"]


def test_verifier_holds_a_segmented_entry_to_its_records(tmp_path):
    coords = random_coords(30, ATOMS, seed=13)
    entry = Path(segs.convert_segments(run_of(tmp_path / "run", coords, [0, 10, 30]), tmp_path / "archive", NAME)["entry"])
    original = (entry / "manifest.json").read_text()

    def verify_with(change):
        manifest = json.loads(original)
        change(manifest)
        (entry / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return archive.verify_entry(entry)

    def other_digest(manifest):
        manifest["segments"][1]["coordinates_sha256"] = manifest["segments"][0]["coordinates_sha256"]

    def gap(manifest):
        manifest["segments"][1]["first_frame"] = 11

    def too_short(manifest):
        manifest["segments"][1]["frames"] = 19

    def incomplete(manifest):
        del manifest["segments"][0]["coordinates_sha256"]

    assert verify_with(other_digest)["errors"] == [
        "coordinates of segment 1 (replica_2) do not match the digest of its record"]
    assert verify_with(gap)["errors"] == ["segment 1 holds frames 11 to 31, expected to start at frame 10"]
    assert verify_with(too_short)["errors"] == ["the segments hold 29 frames, the manifest says 30"]
    assert "segment 0 is incomplete" in verify_with(incomplete)["errors"][0]
    (entry / "manifest.json").write_text(original)
    assert archive.verify_entry(entry)["ok"]
    report = archive.verify_entry(entry, source_dir=tmp_path / "run" / "replica_1")
    assert report["errors"] == ["the entry was made from 2 segments; it cannot be compared with one source folder"]
    # in a folder of another name, as in an unpacked bundle
    moved = tmp_path / "bundle" / "archive"
    shutil.copytree(entry, moved)
    assert archive.verify_entry(moved)["errors"] == [f"entry folder is archive, the manifest says {NAME}"]
    assert archive.verify_entry(moved, check_folder=False)["ok"]


# ---------------------------------------------------------------------------
# Extending an entry
# ---------------------------------------------------------------------------
@pytest.fixture
def stored(tmp_path):
    """An entry as the archive holds it, made from one simulation.pdb, and its coordinates."""
    coords = random_coords(30, ATOMS, seed=20)
    source = make_source(tmp_path / "data", NAME, coords, info_json=NAME, header=CELLS[2])
    (source / "structure.mol2").write_text(mol2_text(list(range(1, ATOMS + 1))))
    return Path(archive.convert_entry(source, tmp_path / "stored")["entry"]), coords


@pytest.mark.parametrize("workers", [1, 2])
def test_extended_entry_starts_with_the_frames_of_the_entry_unchanged(tmp_path, stored, workers):
    old, old_coords = stored
    before = files_of(old)
    old_manifest = archive.load_manifest(old)
    new_coords = random_coords(25, ATOMS, seed=21)
    sources = run_of(tmp_path / "run", new_coords, [0, 10, 25])
    sources = [segs.SegmentSource(s.name, s.pdb, s.mol2, 50.0) for s in sources]

    result = segs.convert_segments(sources, tmp_path / "merged", NAME, extend=old, workers=workers,
                                   extend_time_per_frame_ps=20.0)
    entry, manifest = Path(result["entry"]), result["manifest"]
    assert files_of(old) == before                                         # the entry that is extended is only read

    # the stored frames first, as the bytes they were; then the new ones
    old_trajectory = before["trajectory.xtc"]
    trajectory = (entry / "trajectory.xtc").read_bytes()
    assert trajectory[:len(old_trajectory)] == old_trajectory and len(trajectory) > len(old_trajectory)
    frames = archive.ArchiveEntry(entry).read_frames(unit="milliangstrom")
    assert np.array_equal(frames[:30], old_coords) and np.array_equal(frames[30:], new_coords)
    assert np.array_equal(archive.ArchiveEntry(entry).read_frames(stop=30), archive.ArchiveEntry(old).read_frames())
    back = archive._read_back(entry, ATOMS)
    assert back["frames"] == 55 and back["index_ok"]                       # steps 0..54 without a gap
    assert np.array_equal(back["boxes"][:30], archive._read_back(old, ATOMS)["boxes"])
    assert (entry / "topology.pdb").read_bytes() == before["topology.pdb"]
    assert (entry / "structure.mol2").read_bytes() == before["structure.mol2"]
    assert (entry / "info.json").read_bytes() == before["info.json"]
    assert json.loads((entry / "name.json").read_text()) == {"indexOrderedSequence": NAME}

    # where every segment came from
    old_sha256 = hashlib.sha256(before["manifest.json"]).hexdigest()
    extends = manifest["extends"]
    assert (extends["glycam_name"], extends["manifest_sha256"], extends["frames"], extends["atoms"]) == (
        NAME, old_sha256, 30, ATOMS)
    assert extends["trajectory"] == {
        "size": len(old_trajectory), "sha256": hashlib.sha256(old_trajectory).hexdigest(),
        "coordinates_sha256": old_manifest["trajectory"]["coordinates_sha256"]}
    assert extends["files"] == old_manifest["files"] and extends["converter"] == old_manifest["converter"]
    assert extends["source"]["sha256"] == old_manifest["source"]["sha256"]
    assert extends["repaired"] is None and extends["previous"] is None
    first, second, third = manifest["segments"]
    assert (first["index"], first["name"], first["first_frame"], first["frames"]) == (0, NAME, 0, 30)
    assert first["carried_from"] == {"glycam_name": NAME, "manifest_sha256": old_sha256, "segment": None}
    assert first["source"] == old_manifest["source"] and first["verification"] == old_manifest["verification"]
    assert first["coordinates_sha256"] == old_manifest["trajectory"]["coordinates_sha256"]
    assert (first["trajectory_offset"], first["trajectory_bytes"], first["time_per_frame_ps"]) == (0, len(old_trajectory), 20.0)
    assert [(s["index"], s["name"], s["first_frame"], s["frames"], s["time_per_frame_ps"]) for s in (second, third)] == [
        (1, "replica_1", 30, 10, 50.0), (2, "replica_2", 40, 15, 50.0)]
    assert second["trajectory_offset"] == len(old_trajectory) and "carried_from" not in second
    assert manifest["frames"] == 55 and manifest["source"]["segments"] == 3
    assert manifest["source"]["size"] == old_manifest["source"]["size"] + sum(s.pdb.stat().st_size for s in sources)
    assert manifest["box"]["kind"] == "varying" and [c["first_frame"] for c in manifest["box"]["cells"]] == [0, 30, 40]
    verification = manifest["verification"]
    assert (verification["frames_compared"], verification["frames_carried_over"]) == (25, 30)
    assert verification["carried_over"]["trajectory_sha256"] == hashlib.sha256(old_trajectory).hexdigest()

    assert archive.verify_entry(entry)["ok"]
    assert segs.check_extension(entry, old) == {"ok": True, "errors": [], "trajectory_bytes_equal": True,
                                               "frames_compared": 30}
    assert segs.convert_segments(sources, tmp_path / "merged", NAME, extend=old)["status"] == "converted"   # no time given
    assert segs.convert_segments(sources, tmp_path / "merged", NAME, extend=old)["status"] == "skipped"

    # extended once more: the segments keep their records, the history is in extends
    more = replica(tmp_path / "later", "replica_3", random_coords(5, ATOMS, seed=22), header="")
    twice = segs.convert_segments([more], tmp_path / "merged-again", NAME, extend=entry)
    second_manifest = twice["manifest"]
    merged_sha256 = hashlib.sha256((entry / "manifest.json").read_bytes()).hexdigest()
    assert [(s["name"], s["first_frame"], s["frames"]) for s in second_manifest["segments"]] == [
        (NAME, 0, 30), ("replica_1", 30, 10), ("replica_2", 40, 15), ("replica_3", 55, 5)]
    assert [s.get("carried_from", {}).get("segment", "new") for s in second_manifest["segments"]] == [0, 1, 2, "new"]
    assert second_manifest["segments"][1]["source"] == archive.load_manifest(entry)["segments"][1]["source"]
    assert second_manifest["extends"]["manifest_sha256"] == merged_sha256
    assert second_manifest["extends"]["previous"]["manifest_sha256"] == old_sha256
    assert second_manifest["box"]["cells"][-1] == {"first_frame": 55, "lengths_angstrom": None, "angles_degree": None}
    assert archive.verify_entry(twice["entry"])["ok"] and segs.check_extension(twice["entry"], entry)["ok"]
    assert not segs.check_extension(twice["entry"], old)["ok"]             # it extends the merged entry, not the first

    # the segments of an entry are told apart by their names
    with pytest.raises(archive.SourceError, match="replica_2 is already the name of a segment of the entry to extend"):
        segs.convert_segments([sources[1]], tmp_path / "clash", NAME, extend=entry)
    same_name = segs.SegmentSource(NAME, sources[0].pdb, sources[0].mol2)
    with pytest.raises(archive.SourceError, match=f"{NAME} is already the name of a segment of the entry to extend"):
        segs.convert_segments([same_name], tmp_path / "clash", NAME, extend=old)
    assert not (tmp_path / "clash").exists()


def test_extension_is_refused_when_the_atoms_differ_and_says_what_differs(tmp_path, stored):
    old, _ = stored
    before = files_of(old)
    coords = random_coords(8, ATOMS, seed=23)
    other = replica(tmp_path / "run", "replica_1", coords, atoms=changed_atoms("renamed"))
    with pytest.raises(segs.SegmentMismatchError) as refused:
        segs.convert_segments([other], tmp_path / "merged", NAME, extend=old)
    assert str(refused.value) == (
        "replica_1 does not have the atoms of topology.pdb of the entry to extend: 1 of 14 atoms differ in name, "
        "residue or order; the first is atom 7: name 'N2' residue '0GB' 2 in replica_1 against name 'H7' residue "
        "'0GB' 2 in topology.pdb of the entry to extend")
    fewer = replica(tmp_path / "run", "replica_2", coords[:, :12])
    with pytest.raises(segs.SegmentMismatchError, match="replica_2 has 12 atoms, topology.pdb of the entry to extend has 14"):
        segs.convert_segments([fewer], tmp_path / "merged", NAME, extend=old)
    same = replica(tmp_path / "run", "replica_3", coords)
    with pytest.raises(archive.NamingError, match="the entry to extend is DGlcpa1-OH, the new trajectories are named DGlcpb1-OH"):
        segs.convert_segments([same], tmp_path / "merged", "DGlcpb1-OH", extend=old)
    with pytest.raises(archive.SourceError, match="is not an archive entry"):
        segs.convert_segments([same], tmp_path / "merged", NAME, extend=tmp_path / "run" / "replica_3")
    with pytest.raises(archive.SourceError, match="the entry to extend is the entry that would be written"):
        segs.convert_segments([same], old.parent, NAME, extend=old)
    assert files_of(old) == before and not (tmp_path / "merged").exists()

    # a structure of the run that differs from the stored one is a warning: the stored one is kept
    same.mol2.write_text(same.mol2.read_text().replace("13 13 14 1\n", ""))
    manifest = segs.convert_segments([same], tmp_path / "merged", NAME, extend=old)["manifest"]
    assert manifest["warnings"] == [
        "the MOL2 of replica_3 and structure.mol2 of the entry to extend: 0 bonds only in the MOL2 of replica_3, "
        "1 only in structure.mol2 of the entry to extend (for example atoms 13-14)"]
    assert (tmp_path / "merged" / NAME / "structure.mol2").read_bytes() == before["structure.mol2"]

    # the stored entry has to be what its manifest says
    damaged = tmp_path / "damaged" / NAME
    shutil.copytree(old, damaged)
    content = bytearray(before["trajectory.xtc"])
    content[len(content) // 2] ^= 0x01
    (damaged / "trajectory.xtc").write_bytes(bytes(content))
    with pytest.raises(archive.VerificationError, match="the entry to extend does not verify: trajectory.xtc does not match"):
        segs.convert_segments([same], tmp_path / "merged2", NAME, extend=damaged)


def test_extension_of_an_entry_in_another_layout_of_the_atom_records(tmp_path):
    """Most stored entries were written by MDAnalysis, new runs by cpptraj: the same atoms in other columns."""
    old_coords, new_coords = random_coords(12, ATOMS, seed=24), random_coords(6, ATOMS, seed=25)
    source = make_source(tmp_path / "data", NAME, old_coords, dialect="mdanalysis")
    (source / "structure.mol2").write_text(mol2_text(list(range(1, ATOMS + 1))))
    old = archive.convert_entry(source, tmp_path / "stored")["entry"]
    result = segs.convert_segments([replica(tmp_path / "run", "replica_1", new_coords)], tmp_path / "merged", NAME, extend=old)
    manifest = result["manifest"]
    assert manifest["segments"][1]["atom_records"]["differ_in"] == LAYOUT_COLUMNS
    assert manifest["box"]["kind"] == "varying"                # placeholder cell, then the cell of the new run
    assert [cell["first_frame"] for cell in manifest["box"]["cells"]] == [0, 12]
    frames = archive.ArchiveEntry(result["entry"]).read_frames(unit="milliangstrom")
    assert np.array_equal(frames, np.concatenate([old_coords, new_coords]))
    assert segs.check_extension(result["entry"], old)["ok"]


def test_repaired_entry_is_extended_with_its_record(tmp_path):
    source, kept = cut_source(tmp_path, folder=NAME, frames=20, at=7, seed=26)
    (source / "structure.mol2").write_text(mol2_text(list(range(1, kept.shape[1] + 1))))
    old = Path(archive.convert_entry(source, tmp_path / "stored", repair=True)["entry"])
    old_manifest = archive.load_manifest(old)
    new_coords = random_coords(9, kept.shape[1], seed=27)
    result = segs.convert_segments([replica(tmp_path / "run", "replica_1", new_coords)], tmp_path / "merged", NAME, extend=old)
    manifest = result["manifest"]
    assert "source_irregularities" not in manifest
    assert manifest["segments"][0]["source_irregularities"] == old_manifest["source_irregularities"]
    assert manifest["extends"]["repaired"] == archive.repair_summary(old_manifest)
    assert archive.repair_summary(manifest) == dict(archive.repair_summary(old_manifest), segments=[0])
    assert manifest["warnings"][0].startswith("segment 0 (DGlcpa1-OH) comes from an irregular source")
    report = archive.verify_entry(result["entry"])
    assert report["ok"] and report["repaired"] is True
    index, problems = archive.build_index(tmp_path / "merged")
    assert problems == [] and index["repaired"] == [NAME]
    frames = archive.ArchiveEntry(result["entry"]).read_frames(unit="milliangstrom")
    assert np.array_equal(frames, np.concatenate([kept, new_coords]))
    assert segs.check_extension(result["entry"], old)["ok"]


def test_check_of_an_extension_notices_frames_that_were_changed(tmp_path, stored):
    old, _ = stored
    new = replica(tmp_path / "run", "replica_1", random_coords(6, ATOMS, seed=28))
    entry = Path(segs.convert_segments([new], tmp_path / "merged", NAME, extend=old)["entry"])
    assert segs.check_extension(entry, old)["ok"]

    # another stored entry of that name (one frame differs): not what was extended
    other_coords = random_coords(30, ATOMS, seed=20)
    other_coords[17, 3, 1] += 1
    source = make_source(tmp_path / "other-data", NAME, other_coords, info_json=NAME, header=CELLS[2])
    (source / "structure.mol2").write_text(mol2_text(list(range(1, ATOMS + 1))))
    other = archive.convert_entry(source, tmp_path / "other")["entry"]
    report = segs.check_extension(entry, other)
    assert not report["ok"] and report["trajectory_bytes_equal"] is False
    assert report["errors"][0] == "extends.manifest_sha256 is not the SHA-256 of the manifest of the extended entry"
    compared = segs.compare_trajectories(entry, other)
    assert compared == {"equal": False, "frames_compared": 0, "first_difference": "coordinates of frame 17"}
    assert segs.check_extension(old, old)["errors"] == ["the manifest has no extends record"]


# ---------------------------------------------------------------------------
# Merging two stored entries of one glycan
# ---------------------------------------------------------------------------
KEPT = "LFucpa1-3[DGalpb1-4]DGlcpNAcb1-OH"            # the same glycan, its branches written in two orders
ABSORBED = "DGalpb1-4[LFucpa1-3]DGlcpNAcb1-OH"


def stored_entry(root, name, coords, **options):
    """An entry converted from one simulation.pdb, as most stored entries are."""
    source = make_source(Path(root) / "data", name, coords, info_json=name, **options)
    (source / "structure.mol2").write_text(mol2_text(list(range(1, coords.shape[1] + 1))))
    return Path(archive.convert_entry(source, Path(root) / "stored")["entry"])


def runs_by_model_numbers(runs):
    """MODEL numbers that restart for every run of ``runs`` frames: the runs are recorded."""
    return lambda index: "MODEL     %d" % (index % runs + 1)


@pytest.mark.parametrize("atoms", [ATOMS, 6])                   # compressed frames, and the plain ones of <= 9 atoms
def test_merged_entry_holds_the_frames_of_both_entries(tmp_path, atoms):
    kept_coords, absorbed_coords = random_coords(30, atoms, seed=40), random_coords(25, atoms, seed=41)
    kept = stored_entry(tmp_path / "kept", KEPT, kept_coords, header=CELLS[2])
    sources = run_of(tmp_path / "run", absorbed_coords, [0, 10, 25])
    sources = [segs.SegmentSource(s.name, s.pdb, s.mol2, 50.0) for s in sources]
    absorbed = Path(segs.convert_segments(sources, tmp_path / "absorbed", ABSORBED)["entry"])
    before = {"kept": files_of(kept), "absorbed": files_of(absorbed)}

    result = segs.merge_entries(kept, absorbed, tmp_path / "merged", kept_id="GS00303", absorbed_id="GS00453")
    entry, manifest = Path(result["entry"]), result["manifest"]
    assert result["status"] == "merged" and entry.name == KEPT and manifest == archive.load_manifest(entry)
    assert {"kept": files_of(kept), "absorbed": files_of(absorbed)} == before        # both are only read

    # the frames of the kept entry as the bytes they were, then those of the absorbed one
    kept_xtc, absorbed_xtc = before["kept"]["trajectory.xtc"], before["absorbed"]["trajectory.xtc"]
    trajectory = (entry / "trajectory.xtc").read_bytes()
    assert trajectory[:len(kept_xtc)] == kept_xtc and len(trajectory) == len(kept_xtc) + len(absorbed_xtc)
    frames = archive.ArchiveEntry(entry).read_frames(unit="milliangstrom")
    assert np.array_equal(frames, np.concatenate([kept_coords, absorbed_coords]))
    back = archive._read_back(entry, atoms)
    assert back["frames"] == 55 and back["index_ok"]                       # steps and times 0..54
    assert np.array_equal(back["boxes"][30:], archive._read_back(absorbed, atoms)["boxes"])
    # only step and time of the absorbed frames changed: numbered from 0 again they are its trajectory
    (tmp_path / "tail.xtc").write_bytes(trajectory[len(kept_xtc):])
    with open(tmp_path / "again.xtc", "wb") as handle:
        again = segs.renumber_xtc(tmp_path / "tail.xtc", handle, 0, atoms)
    assert (tmp_path / "again.xtc").read_bytes() == absorbed_xtc and again["frames"] == 25
    assert again["sha256_written"] == hashlib.sha256(absorbed_xtc).hexdigest()

    # the files of the kept entry
    for name in ("topology.pdb", "structure.mol2", "info.json"):
        assert (entry / name).read_bytes() == before["kept"][name]
    assert json.loads((entry / "name.json").read_text()) == {"indexOrderedSequence": KEPT}
    assert manifest["glycam_name"] == KEPT and manifest["source"]["folder"] == KEPT
    assert manifest["source"]["segments"] == 3 and manifest["warnings"] == []

    # the segments: the kept entry, then those of the absorbed one under its ID
    kept_sha256 = hashlib.sha256(before["kept"]["manifest.json"]).hexdigest()
    absorbed_sha256 = hashlib.sha256(before["absorbed"]["manifest.json"]).hexdigest()
    first, second, third = manifest["segments"]
    assert [(s["index"], s["name"], s["first_frame"], s["frames"]) for s in manifest["segments"]] == [
        (0, KEPT, 0, 30), (1, "GS00453/replica_1", 30, 10), (2, "GS00453/replica_2", 40, 15)]
    assert first["carried_from"] == {"glycam_name": KEPT, "manifest_sha256": kept_sha256, "segment": None}
    assert "merged_from" not in first and first["trajectory_offset"] == 0
    absorbed_segments = archive.load_manifest(absorbed)["segments"]
    for index, (segment, old) in enumerate(zip((second, third), absorbed_segments)):
        assert segment["carried_from"] == {"glycam_name": ABSORBED, "manifest_sha256": absorbed_sha256, "segment": index}
        assert segment["merged_from"] == {"glycoshape_id": "GS00453", "glycam_name": ABSORBED,
                                          "manifest_sha256": absorbed_sha256}
        assert segment["trajectory_offset"] == len(kept_xtc) + old["trajectory_offset"]
        assert (segment["coordinates_sha256"], segment["source"], segment["time_per_frame_ps"]) == (
            old["coordinates_sha256"], old["source"], 50.0)
    assert manifest["extends"]["manifest_sha256"] == kept_sha256 and manifest["extends"]["frames"] == 30
    merged_from = manifest["merged_from"]
    assert (merged_from["glycoshape_id"], merged_from["kept_glycoshape_id"], merged_from["glycam_name"]) == (
        "GS00453", "GS00303", ABSORBED)
    assert (merged_from["manifest_sha256"], merged_from["frames"], merged_from["first_frame"]) == (absorbed_sha256, 25, 30)
    assert merged_from["atom_records"] == "identical"
    assert manifest["box"]["kind"] == "varying" and [c["first_frame"] for c in manifest["box"]["cells"]] == [0, 30, 40]
    verification = manifest["verification"]
    assert (verification["frames_carried_over"], verification["frames_merged"], verification["segments_checked"]) == (30, 25, 3)
    assert verification["merged"]["trajectory_sha256"] == hashlib.sha256(absorbed_xtc).hexdigest()

    # every reader reads it; it extends the kept entry
    report = archive.verify_entry(entry)
    assert report["ok"] and report["segments_checked"] == 3 and report["frames"] == 55
    assert segs.check_extension(entry, kept)["ok"]
    assert segs.compare_trajectories(entry, kept) == {"equal": True, "frames_compared": 30, "first_difference": None}
    index, problems = archive.build_index(tmp_path / "merged")
    assert problems == [] and index["records"][0]["frames"] == 55
    from lib import replicas

    runs = replicas.replica_boundaries(manifest).replicas
    assert [run.name for run in runs][1:] == ["GS00453/replica_1", "GS00453/replica_2"]

    # once more: the merge of these two entries is there
    assert segs.merge_entries(kept, absorbed, tmp_path / "merged", kept_id="GS00303", absorbed_id="GS00453")["status"] == "skipped"
    assert segs.merge_entries(kept, absorbed, tmp_path / "merged", kept_id="GS00303", absorbed_id="GS00453",
                              force=True)["status"] == "merged"


def test_an_absorbed_entry_of_one_simulation_pdb_is_one_segment_named_by_its_id(tmp_path):
    kept = stored_entry(tmp_path / "kept", KEPT, random_coords(20, ATOMS, seed=42), header=CELLS[2],
                        model=runs_by_model_numbers(10))
    absorbed = stored_entry(tmp_path / "absorbed", ABSORBED, random_coords(20, ATOMS, seed=43), header=CELLS[2],
                            model=runs_by_model_numbers(10))
    manifest = segs.merge_entries(kept, absorbed, tmp_path / "merged", kept_id="GS00303", absorbed_id="GS00453",
                                  kept_time_per_frame_ps=50.0, kept_run={"salt_concentration_mm": 200.0},
                                  absorbed_run={"salt_concentration_mm": 150.0})["manifest"]
    assert [(s["name"], s["first_frame"], s["frames"], s["time_per_frame_ps"]) for s in manifest["segments"]] == [
        (KEPT, 0, 20, 50.0), ("GS00453", 20, 20, None)]
    # what each published entry said about its simulations, in the run record of its segments
    assert [s["run"] for s in manifest["segments"]] == [{"salt_concentration_mm": 200.0}, {"salt_concentration_mm": 150.0}]
    assert manifest["segments"][1]["carried_from"]["segment"] is None
    assert manifest["box"]["kind"] == "constant"
    # the runs of each part as its source recorded them; those of the absorbed entry under its ID
    from lib import replicas

    runs = replicas.replica_boundaries(manifest).replicas
    assert [(run.name, run.first_frame, run.frames) for run in runs] == [
        ("stored_run_1", 0, 10), ("stored_run_2", 10, 10), ("GS00453/run_1", 20, 10), ("GS00453/run_2", 30, 10)]
    assert archive.verify_entry(Path(tmp_path / "merged" / KEPT))["ok"]
    from lib import entry_ops

    assert entry_ops.salt_by_run(manifest) == {"stored_run_1": 200.0, "stored_run_2": 200.0,
                                               "GS00453/run_1": 150.0, "GS00453/run_2": 150.0}
    # without IDs the absorbed entry is named by its GLYCAM name
    manifest = segs.merge_entries(kept, absorbed, tmp_path / "merged-2")["manifest"]
    assert manifest["segments"][1]["name"] == ABSORBED and manifest["merged_from"]["glycoshape_id"] is None


def test_a_merge_is_refused_when_the_atoms_or_bonds_differ_and_says_what_differs(tmp_path):
    kept = stored_entry(tmp_path / "kept", KEPT, random_coords(12, ATOMS, seed=44))
    coords = random_coords(8, ATOMS, seed=45)
    swapped = stored_entry(tmp_path / "swapped", ABSORBED, coords, atoms=changed_atoms("other order"))
    before = files_of(kept)
    with pytest.raises(segs.SegmentMismatchError) as refused:
        segs.merge_entries(kept, swapped, tmp_path / "merged", kept_id="GS00303", absorbed_id="GS00453")
    assert str(refused.value) == (
        "topology.pdb of GS00453 does not have the atoms of topology.pdb of GS00303: 2 of 14 atoms differ in name, "
        "residue or order; the first is atom 4: name 'O5' residue 'ROH' 1 in topology.pdb of GS00453 against name "
        "'C4' residue 'ROH' 1 in topology.pdb of GS00303")
    fewer = stored_entry(tmp_path / "fewer", ABSORBED, coords[:, :12])
    with pytest.raises(segs.SegmentMismatchError, match="topology.pdb of GS00453 has 12 atoms, topology.pdb of GS00303 has 14"):
        segs.merge_entries(kept, fewer, tmp_path / "merged", kept_id="GS00303", absorbed_id="GS00453")

    # other bonds in structure.mol2 are refused; other atom types are a warning
    other = tmp_path / "other" / ABSORBED
    shutil.copytree(stored_entry(tmp_path / "bonds", ABSORBED, coords), other)
    mol2 = (other / "structure.mol2").read_text()

    def with_mol2(text):
        (other / "structure.mol2").write_text(text)
        manifest = archive.load_manifest(other)
        manifest["files"]["structure.mol2"] = {"size": len(text.encode()), "sha256": hashlib.sha256(text.encode()).hexdigest()}
        (other / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    with_mol2(mol2.replace("13 13 14 1\n", ""))
    with pytest.raises(segs.SegmentMismatchError, match=r"structure.mol2 of GS00453 does not have the bonds of "
                                                        r"structure.mol2 of GS00303: 0 bonds only in .*, 1 only in .* "
                                                        r"\(for example atoms 13-14\)"):
        segs.merge_entries(kept, other, tmp_path / "merged", kept_id="GS00303", absorbed_id="GS00453")
    with_mol2(mol2.replace(" C.3 1 RES", " C.ar 1 RES", 1))
    manifest = segs.merge_entries(kept, other, tmp_path / "merged", kept_id="GS00303", absorbed_id="GS00453")["manifest"]
    assert manifest["warnings"] == ["structure.mol2 of GS00453 and structure.mol2 of GS00303: 1 atoms differ in type "
                                    "or substructure"]
    shutil.rmtree(tmp_path / "merged")

    # an entry with itself, an entry of the same name, an archive inside an entry, an entry that does not verify
    with pytest.raises(archive.SourceError, match="both entries are named"):
        segs.merge_entries(kept, kept, tmp_path / "merged")
    with pytest.raises(archive.SourceError, match="the archive directory must be outside"):
        segs.merge_entries(kept, other, kept / "inside")
    with pytest.raises(archive.SourceError, match="the kept entry is the entry that would be written"):
        segs.merge_entries(kept, other, kept.parent)
    (other / "trajectory.xtc").write_bytes((other / "trajectory.xtc").read_bytes()[:-4])
    with pytest.raises(archive.VerificationError, match="the absorbed entry does not verify"):
        segs.merge_entries(kept, other, tmp_path / "merged")
    assert files_of(kept) == before and not (tmp_path / "merged").exists()


# ---------------------------------------------------------------------------
# Pruning runs of a stored entry
# ---------------------------------------------------------------------------
def runs_of_entry(entry):
    from dataclasses import asdict

    from lib import replicas

    return [asdict(run) for run in replicas.replica_boundaries(archive.load_manifest(entry)).replicas]


def test_a_pruned_entry_keeps_the_frames_of_the_other_runs_under_their_names(tmp_path):
    coords = random_coords(30, ATOMS, seed=50)
    entry = stored_entry(tmp_path / "stored", KEPT, coords, header=CELLS[2], model=runs_by_model_numbers(10))
    before = files_of(entry)
    runs = runs_of_entry(entry)
    assert [(run["name"], run["first_frame"], run["frames"], run["basis"]) for run in runs] == [
        ("run_1", 0, 10, "recorded"), ("run_2", 10, 10, "recorded"), ("run_3", 20, 10, "recorded")]
    record = {"glycoshape_id": "GS00276", "note": "run 2 has an inverted stereocentre", "author": "curator",
              "pruned_at": "2026-10-14T00:00:00Z"}
    result = segs.prune_entry(entry, tmp_path / "pruned", runs, ["run_2"], record=record)
    pruned, manifest = Path(result["entry"]), result["manifest"]
    assert result["status"] == "pruned" and pruned.name == KEPT and files_of(entry) == before   # only read

    # the frames of run_1 and run_3, step and time renumbered, nothing else changed
    assert manifest["frames"] == 20
    frames = archive.ArchiveEntry(pruned).read_frames(unit="milliangstrom")
    assert np.array_equal(frames, np.concatenate([coords[:10], coords[20:]]))
    back = archive._read_back(pruned, ATOMS)
    assert back["frames"] == 20 and back["index_ok"]
    report = archive.verify_entry(pruned)
    assert report["ok"] and report["segments_checked"] == 2
    for name in ("topology.pdb", "structure.mol2", "info.json"):
        assert (pruned / name).read_bytes() == before[name]

    # every kept run is a segment of its name: the replica diagnostics name the runs as before
    assert [(s["name"], s["first_frame"], s["frames"]) for s in manifest["segments"]] == [("run_1", 0, 10), ("run_3", 10, 10)]
    assert [run["name"] for run in runs_of_entry(pruned)] == ["run_1", "run_3"]
    sha256 = hashlib.sha256(before["manifest.json"]).hexdigest()
    third = manifest["segments"][1]
    assert third["pruned_from"]["manifest_sha256"] == sha256 and third["pruned_from"]["first_frame"] == 20
    assert third["coordinates_sha256"] == hashlib.sha256(coords[20:].astype("<i4").tobytes()).hexdigest()
    assert manifest["box"]["kind"] == "constant"
    record_in = manifest["pruned"]
    assert (record_in["manifest_sha256"], record_in["frames_before"], record_in["note"]) == (sha256, 30, record["note"])
    assert record_in["removed"] == [{"name": "run_2", "first_frame": 10, "frames": 10, "basis": "recorded"}]
    assert [(row["name"], row["first_frame"], row["first_frame_after"]) for row in record_in["kept"]] == [
        ("run_1", 0, 0), ("run_3", 20, 10)]
    assert "extends" not in manifest and manifest["source"]["sha256"] == archive.load_manifest(entry)["source"]["sha256"]

    # once more: the same prune is there; refusals
    assert segs.prune_entry(entry, tmp_path / "pruned", runs, ["run_2"], record=record)["status"] == "skipped"
    with pytest.raises(archive.SourceError, match="no run named 'run_9'; the runs of the entry are run_1, run_2, run_3"):
        segs.prune_entry(entry, tmp_path / "other", runs, ["run_9"])
    with pytest.raises(archive.SourceError, match="every run of the entry would be removed"):
        segs.prune_entry(entry, tmp_path / "other", runs, ["run_1", "run_2", "run_3"])
    with pytest.raises(archive.SourceError, match="the archive directory must be outside"):
        segs.prune_entry(entry, entry / "inside", runs, ["run_2"])
    assert not (tmp_path / "other").exists()


def test_pruning_an_entry_of_segments_keeps_the_records_of_the_whole_segments(tmp_path):
    coords = random_coords(60, ATOMS, seed=51)
    entry = Path(segs.convert_segments(run_of(tmp_path / "run", coords, [0, 25, 40, 60]), tmp_path / "archive",
                                       NAME)["entry"])
    old = archive.load_manifest(entry)["segments"]
    manifest = segs.prune_entry(entry, tmp_path / "pruned", runs_of_entry(entry), ["replica_1"])["manifest"]
    assert [(s["name"], s["first_frame"], s["frames"]) for s in manifest["segments"]] == [("replica_2", 0, 15),
                                                                                         ("replica_3", 15, 20)]
    for segment, before in zip(manifest["segments"], old[1:]):
        assert {key: value for key, value in segment.items() if key not in ("index", "first_frame", "trajectory_offset")} \
            == {key: value for key, value in before.items() if key not in ("index", "first_frame", "trajectory_offset")}
    assert [cell["first_frame"] for cell in manifest["box"]["cells"]] == [0, 15]
    entry_after = tmp_path / "pruned" / NAME
    assert archive.verify_entry(entry_after)["ok"]
    assert np.array_equal(archive.ArchiveEntry(entry_after).read_frames(unit="milliangstrom"), coords[25:])
