"""Tests of the converter on damaged sources (``repair`` of lib/archive.py, ``--repair`` of the script).

Every trajectory is synthetic and built in the test.  The damage is of the
kinds found in the raw data: a frame that the file cuts short, in the middle
(one run of several was cut and the next appended) or at the end, and frames
without a MODEL record.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from lib import archive

from test_archive import (
    atom_line, atom_table, make_source, mol2_text, pdb_text, random_coords, read_xtc, run_converter, snapshot)

ATOMS = 12
FRAMES = 30


# ---------------------------------------------------------------------------
# Building damaged text
# ---------------------------------------------------------------------------
def split_frames(text):
    """``(header, frames, tail)`` of a multi-model PDB: lists of lines, one list per frame."""
    lines = text.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.startswith("MODEL")]
    ends = [i for i, line in enumerate(lines) if line.startswith("ENDMDL")]
    assert len(starts) == len(ends)
    return lines[:starts[0]], [lines[a:b + 1] for a, b in zip(starts, ends)], lines[ends[-1] + 1:]


def join(header, frames, tail):
    return "".join(header + [line for frame in frames for line in frame] + tail)


def cut(frame, atoms, fragment=""):
    """The lines of ``frame`` up to its ``atoms``-th atom record, then ``fragment``; no ENDMDL."""
    kept, seen = [], 0
    for line in frame[1:]:
        if line.startswith("ATOM"):
            if seen == atoms:
                break
            seen += 1
        elif line.startswith("ENDMDL"):
            break
        kept.append(line)
    return [frame[0]] + kept + ([fragment] if fragment else [])


def atom_lines_of(frame):
    return [line for line in frame if line.startswith("ATOM")]


def piece(frame, atom, length):
    """The first ``length`` bytes of an atom record of ``frame``: what a cut leaves of the line."""
    return atom_lines_of(frame)[atom][:length]


def regular(frames=FRAMES, atoms=ATOMS, seed=0, **options):
    coords = random_coords(frames, atoms, seed=seed)
    return coords, pdb_text(coords, **options)


def convert(tmp_path, text, repair=True, folder="DGlcpa1-OH", **options):
    source = make_source(tmp_path / "data", folder, text=text)
    result = archive.convert_entry(source, tmp_path / "archive", repair=repair, **options)
    return source, result


def offset_of(frames, header, index):
    return len("".join(header + [line for frame in frames[:index] for line in frame]))


def check_entry(result, source, coords):
    """A repaired entry reads like any other and verifies against its source."""
    entry = archive.ArchiveEntry(result["entry"])
    manifest = entry.manifest
    assert entry.n_frames == manifest["frames"] == len(coords)
    assert np.array_equal(entry.read_frames(unit="milliangstrom"), coords)
    xyz, time, step, _ = read_xtc(entry.path)
    assert np.array_equal(step, np.arange(len(coords))) and np.array_equal(time, np.arange(len(coords), dtype=np.float32))
    assert manifest["trajectory"]["coordinates_sha256"] == hashlib.sha256(coords.astype("<i4").tobytes()).hexdigest()
    assert manifest["verification"]["passed"] and manifest["verification"]["frames_compared"] == len(coords)
    assert manifest["verification"]["source_irregularities"]["bytes_ignored"] == (
        manifest["source_irregularities"]["bytes_ignored"])
    assert manifest["source"]["sha256"] == hashlib.sha256((source / "simulation.pdb").read_bytes()).hexdigest()
    assert len(manifest["warnings"]) == 1 and "archived with repair" in manifest["warnings"][0]
    assert archive.verify_entry(entry.path)["ok"] and archive.verify_entry(entry.path, deep=False)["ok"]
    report = archive.verify_entry(entry.path, source, check_frames=-1)
    assert report["ok"], report["errors"]
    assert report["repaired"] and report["reference_frames_checked"] == len(coords)
    assert report["mismatched_coordinates"] == 0 and report["source_frames_compared"] == len(coords)
    return manifest


# ---------------------------------------------------------------------------
# A frame that the file cuts short, the next MODEL record following
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("atoms,length", [
    (7, 0),                                       # the cut fell on a line end
    (1, 0),
    (7, 36),                                      # a line cut inside its x coordinate
    (7, 60),                                      # a shorter line that still has its three coordinates
    (11, 60),                                     # the same in the last atom record: no atom is missing but this
    (11, 78),                                     # the last atom record without its last two blanks
    (0, 3),                                       # nothing but the beginning of the first atom record
])
def test_incomplete_frame_before_the_next_model_is_dropped(tmp_path, atoms, length):
    coords, text = regular()
    header, frames, tail = split_frames(text)
    # The line that the cut went through ends where the next file begins; here with a line end.
    fragment = piece(frames[11], atoms, length) + "\n" if length else ""
    cut_bytes = len(fragment)
    damaged = list(frames)
    damaged[11] = cut(frames[11], atoms, fragment)
    damaged_text = join(header, damaged, tail)

    source = make_source(tmp_path / "data", "DGlcpa1-OH", text=damaged_text)
    before = snapshot(source)
    with pytest.raises(archive.PdbFormatError, match=r"frame 11 \(byte \d+\): MODEL record at byte \d+ before the "
                                                     r"ENDMDL of this frame; with repair \(--repair"):
        archive.convert_entry(source, tmp_path / "archive")
    assert not (tmp_path / "archive" / "DGlcpa1-OH").exists()

    result = archive.convert_entry(source, tmp_path / "archive", repair=True)
    assert result["status"] == "converted" and snapshot(source) == before
    kept = np.delete(coords, 11, axis=0)
    if (atoms, length) == (11, 78):
        # Every column of the record is there and the line is whole: the frame has its twelve atom
        # records and lacks only TER and ENDMDL.  It is kept, as a frame without ENDMDL.
        manifest = check_entry(result, source, coords)
        item, = manifest["source_irregularities"]["items"]
        assert (item["kind"], item["action"], item["atoms_found"], item["cut_line_bytes"]) == (
            "frame_without_endmdl", "kept", ATOMS, 0)
        assert manifest["source_irregularities"]["frames_dropped"] == 0
        return
    manifest = check_entry(result, source, kept)
    dropped = "".join(damaged[11])
    assert manifest["source_irregularities"] == {
        "accepted_with": "repair",
        "frames_in_source": FRAMES,
        "frames_kept": FRAMES - 1,
        "frames_dropped": 1,
        "bytes_ignored": len(dropped),
        "truncated": False,
        "counts": {"incomplete_frame": 1},
        "items": [{
            "kind": "incomplete_frame",
            "action": "dropped",
            "source_frame": 11,
            "frames_kept_before": 11,
            "offset": offset_of(damaged, header, 11),
            "bytes": len(dropped),
            "bytes_ignored": len(dropped),
            "sha256": hashlib.sha256(dropped.encode()).hexdigest(),
            "model_record": "MODEL     12",
            "atoms_found": atoms,                    # the cut line is never one of them
            "atoms_expected": ATOMS,
            "cut_line_bytes": cut_bytes,
            "followed_by": "MODEL record",
        }],
    }
    # the frames on both sides of the gap were read again from the text, whatever the sample
    sampled = archive.convert_entry(source, tmp_path / "archive", repair=True, force=True, check_frames=2)
    assert sampled["manifest"]["verification"]["reference_frames_checked"] == 4
    assert sampled["manifest"]["source"]["model_numbers"] == {
        "first": 1, "last": FRAMES, "largest": FRAMES, "sequential": True, "breaks": 0}


def test_cut_line_that_runs_into_the_cryst1_of_the_next_run(tmp_path):
    # Two runs written one after the other; the first was cut in the middle of a line, so the
    # CRYST1 record of the second stands on that line.
    coords = random_coords(40, ATOMS, seed=3)
    cells = [(43.501, 57.057, 51.057), (46.597, 55.385, 51.097)]
    runs = [pdb_text(coords[20 * run:20 * run + 20],
                     header="CRYST1%9.3f%9.3f%9.3f  90.00  90.00  90.00               1\n" % cell)
            for run, cell in enumerate(cells)]
    header, frames, _ = split_frames(runs[0])
    first = join(header, frames[:12] + [cut(frames[12], 5, "ATO")], [])
    assert first.endswith("\nATO")
    text = first + runs[1]
    assert "\nATOCRYST1   46.597" in text

    source, result = convert(tmp_path, text)
    kept = np.concatenate([coords[:12], coords[20:]])
    manifest = check_entry(result, source, kept)
    item, = manifest["source_irregularities"]["items"]
    dropped = "".join(cut(frames[12], 5, "ATO"))
    assert (item["kind"], item["source_frame"], item["frames_kept_before"]) == ("incomplete_frame", 12, 12)
    assert (item["atoms_found"], item["atoms_expected"], item["cut_line_bytes"]) == (5, ATOMS, 3)
    assert item["followed_by"] == "CRYST1 record on the same line"
    assert (item["offset"], item["bytes"]) == (len(first) - len(dropped), len(dropped))
    assert item["sha256"] == hashlib.sha256(dropped.encode()).hexdigest()
    assert manifest["source_irregularities"]["frames_in_source"] == 33 and not manifest["source_irregularities"]["truncated"]

    # the record on the cut line is the cell of the frames that follow it
    assert manifest["box"] == {"kind": "varying", "cells_total": 2, "cells": [
        {"first_frame": 0, "lengths_angstrom": list(cells[0]), "angles_degree": [90.0, 90.0, 90.0]},
        {"first_frame": 12, "lengths_angstrom": list(cells[1]), "angles_degree": [90.0, 90.0, 90.0]}]}
    vectors = read_xtc(result["entry"])[3]
    assert np.allclose(vectors[:, 0, 0] * 10, np.repeat([43.501, 46.597], [12, 20]), atol=1e-5)
    assert manifest["source"]["records_between_frames"] == {
        "CRYST1": {"count": 1, "before_frame": [12]}, "END": {"count": 1, "before_frame": [32]}}
    assert manifest["source"]["model_numbers"] == {
        "first": 1, "last": 20, "largest": 20, "sequential": False, "breaks": 1}

    # text other than CRYST1 on the cut line is not read as a record, nor is a CRYST1 that cannot be read
    with pytest.raises(archive.PdbFormatError, match="begins like a record of the frame"):
        convert(tmp_path / "other", first + "HEADER    next run\n" + runs[1])
    with pytest.raises(archive.PdbFormatError, match="unreadable CRYST1 record"):
        convert(tmp_path / "unreadable", first + runs[1].replace("   46.597", "   46.5x7", 1))


@pytest.mark.parametrize("left,newline,atoms,cut_bytes,kind", [
    ("MODEL   ", "\n", None, None, "model_record_without_frame"),       # the cut went through the MODEL record
    ("MODEL     13", "\n", None, None, "model_record_without_frame"),   # ... or took its line end only
    ("MODEL     13\r", "\r\n", None, None, "model_record_without_frame"),
    (80, "\n", 5, 80, "incomplete_frame"),                               # an atom record, all but its line end
    (80, "\r\n", 5, 80, "incomplete_frame"),
    (81, "\r\n", 5, 81, "incomplete_frame"),                             # the cut fell between \r and \n
    (54, "\n", 5, 54, "incomplete_frame"),                               # just behind the coordinates
    (30, "\n", 5, 30, "incomplete_frame"),                               # just before them
    (0, "\n", 5, 0, "incomplete_frame"),                                 # on a line end: CRYST1 has its own line
])
def test_cryst1_of_the_next_run_is_read_wherever_the_cut_fell(tmp_path, left, newline, atoms, cut_bytes, kind):
    coords = random_coords(40, ATOMS, seed=3)
    cells = [(43.501, 57.057, 51.057), (46.597, 55.385, 51.097)]
    runs = [pdb_text(coords[20 * run:20 * run + 20], newline=newline,
                     header="CRYST1%9.3f%9.3f%9.3f  90.00  90.00  90.00               1\n" % cell)
            for run, cell in enumerate(cells)]
    header, frames, _ = split_frames(runs[0])
    if atoms is None:
        last = [left]
    else:
        last = cut(frames[12], atoms, atom_lines_of(frames[12])[atoms][:left])
    first = join(header, frames[:12] + [last], [])
    text = first + runs[1]
    assert first + "CRYST1   46.597" in text and text.count("CRYST1") == 2

    source = make_source(tmp_path / "data", "DGlcpa1-OH", text=text.encode())
    with pytest.raises(archive.PdbFormatError, match="before the ENDMDL of this frame"):
        archive.convert_entry(source, tmp_path / "archive")
    result = archive.convert_entry(source, tmp_path / "archive", repair=True)
    manifest = check_entry(result, source, np.concatenate([coords[:12], coords[20:]]))
    item = manifest["source_irregularities"]["items"][0]
    assert item["kind"] == kind and item["frames_kept_before"] == 12
    own_line = left in (0, 81) or str(left).endswith("\r")               # a \r alone ends a line too
    assert item["followed_by"] == ("CRYST1 record" if own_line else "CRYST1 record on the same line")
    assert (item.get("atoms_found"), item.get("cut_line_bytes")) == (atoms, cut_bytes)
    assert item["offset"] + item["bytes"] == len(first)                    # up to the C of CRYST1
    # every frame of the second run has the cell of the second run
    assert [cell["first_frame"] for cell in manifest["box"]["cells"]] == [0, 12]
    assert manifest["box"]["cells"][1]["lengths_angstrom"] == list(cells[1])
    assert np.allclose(read_xtc(result["entry"])[3][:, 0, 0] * 10, np.repeat([43.501, 46.597], [12, 20]), atol=1e-5)
    assert manifest["source"]["records_between_frames"]["CRYST1"] == {"count": 1, "before_frame": [12]}


def test_cryst1_behind_text_that_is_no_cut_record_is_refused(tmp_path):
    coords, text = regular()
    header, frames, tail = split_frames(text)
    for line in ("ATOM      9  XX CRYST1   46.597   55.385   51.097  90.00  90.00  90.00               1\n",
                 "REMARK see CRYST1\n"):
        damaged = frames[:11] + [cut(frames[11], 7, line)] + frames[12:]
        source = make_source(tmp_path / line[:4] / "data", "DGlcpa1-OH", text=join(header, damaged, tail))
        with pytest.raises(archive.PdbFormatError, match="CRYST1 stands on the line at byte .* behind text that is not"):
            archive.convert_entry(source, tmp_path / "archive", repair=True)
    damaged = frames[:11] + [["MODEL  {i+1} CRYST1   46.597   55.385   51.097  90.00  90.00  90.00\n"]] + frames[12:]
    source = make_source(tmp_path / "model" / "data", "DGlcpa1-OH", text=join(header, damaged, tail))
    with pytest.raises(archive.PdbFormatError, match="frame 11 .* line of the MODEL record behind text that is not"):
        archive.convert_entry(source, tmp_path / "archive", repair=True)


def test_cut_in_a_ter_record_with_the_cryst1_of_the_next_run_on_its_line(tmp_path):
    coords = random_coords(40, ATOMS, seed=3)
    runs = [pdb_text(coords[20 * run:20 * run + 20],
                     header="CRYST1%9.3f%9.3f%9.3f  90.00  90.00  90.00               1\n" % cell)
            for run, cell in enumerate([(43.501, 57.057, 51.057), (46.597, 55.385, 51.097)])]
    header, frames, _ = split_frames(runs[0])
    assert frames[12][8].startswith("TER       8      0GB")              # the TER after the seventh atom record
    for atoms, kept in ((7, np.concatenate([coords[:12], coords[20:]])), (ATOMS, np.concatenate([coords[:13], coords[20:]]))):
        ter = [line for line in frames[12] if line.startswith("TER")][0 if atoms == 7 else 1]
        whole = cut(frames[12], atoms)
        whole = whole[:-1] if whole[-1].startswith("TER") else whole     # the TER itself is the line that is cut
        text = join(header, frames[:12] + [whole + [ter[:9]]], []) + runs[1]
        assert "\n" + ter[:9] + "CRYST1   46.597" in text
        source, result = convert(tmp_path / str(atoms), text)
        manifest = check_entry(result, source, kept)
        item, = manifest["source_irregularities"]["items"]
        assert (item["kind"], item["atoms_found"], item["cut_line_bytes"], item["followed_by"]) == (
            "incomplete_frame" if atoms == 7 else "frame_without_endmdl", atoms, 9, "CRYST1 record on the same line")
        assert item["bytes_ignored"] == (item["bytes"] if atoms == 7 else 9)
        first_frame_of_second_run = 12 if atoms == 7 else 13
        assert [cell["first_frame"] for cell in manifest["box"]["cells"]] == [0, first_frame_of_second_run]
        vectors = read_xtc(result["entry"])[3]
        assert np.allclose(vectors[:, 0, 0] * 10, np.repeat([43.501, 46.597], [first_frame_of_second_run, 20]), atol=1e-5)


# ---------------------------------------------------------------------------
# A file that ends inside a frame
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("atoms,length", [(8, 3), (8, 0), (0, 12), (5, 36), (11, 60), (11, 78), (11, 80)])
def test_truncated_file_loses_its_last_frame_only(tmp_path, atoms, length):
    coords, text = regular()
    header, frames, _ = split_frames(text)
    # What is left of the line the file ends in; with 60 columns and more it has its three
    # coordinates, with 80 it is the whole record without its line end.
    fragment = piece(frames[-1], atoms, length)
    last = cut(frames[-1], atoms, fragment)
    damaged_text = join(header, frames[:-1] + [last], [])

    source = make_source(tmp_path / "data", "DGlcpa1-OH", text=damaged_text)
    with pytest.raises(archive.PdbFormatError, match=r"frame 29 \(byte \d+\) is not terminated by ENDMDL; the file "
                                                     r"is truncated; with repair \(--repair"):
        archive.convert_entry(source, tmp_path / "archive")

    result = archive.convert_entry(source, tmp_path / "archive", repair=True)
    manifest = check_entry(result, source, coords[:-1])
    dropped = "".join(last)
    assert manifest["source_irregularities"] == {
        "accepted_with": "repair",
        "frames_in_source": FRAMES,
        "frames_kept": FRAMES - 1,
        "frames_dropped": 1,
        "bytes_ignored": len(dropped),
        "truncated": True,
        "counts": {"incomplete_frame": 1},
        "items": [{
            "kind": "incomplete_frame",
            "action": "dropped",
            "source_frame": FRAMES - 1,
            "frames_kept_before": FRAMES - 1,
            "offset": len(damaged_text) - len(dropped),
            "bytes": len(dropped),
            "bytes_ignored": len(dropped),
            "sha256": hashlib.sha256(dropped.encode()).hexdigest(),
            "model_record": "MODEL     30",
            "atoms_found": atoms,                    # a line without its end is never an atom record
            "atoms_expected": ATOMS,
            "cut_line_bytes": len(fragment),
            "followed_by": "end of file",
            "file_truncated": True,
            "file_size": len(damaged_text),
        }],
    }
    assert manifest["source"]["size"] == len(damaged_text)
    assert manifest["source"]["model_numbers"]["last"] == FRAMES          # the MODEL records of the source
    assert (Path(result["entry"]) / "topology.pdb").read_text() == "".join(header + frames[0][1:-1]) + "END\n"


@pytest.mark.parametrize("left", ["", "ATO", "whole record and \r"])
def test_cut_frames_in_a_file_with_windows_line_endings(tmp_path, left):
    coords, text = regular(newline="\r\n")
    header, frames, _ = split_frames(text)
    assert all(line.endswith("\r\n") for line in frames[11])
    middle = cut(frames[11], 7, piece(frames[11], 7, 40) + "\r\n")
    left = piece(frames[-1], 8, 80) + "\r" if left.startswith("whole") else left   # the cut fell between \r and \n
    damaged_text = join(header, frames[:11] + [middle] + frames[12:-1] + [cut(frames[-1], 8, left)], [])

    source = make_source(tmp_path / "data", "DGlcpa1-OH", text=damaged_text.encode())
    with pytest.raises(archive.PdbFormatError, match="before the ENDMDL of this frame"):
        archive.convert_entry(source, tmp_path / "archive")
    result = archive.convert_entry(source, tmp_path / "archive", repair=True)
    manifest = check_entry(result, source, np.delete(coords, [11, FRAMES - 1], axis=0))
    assert manifest["source"]["frames_parsed_line_by_line"] == 1
    summary = manifest["source_irregularities"]
    assert [(item["kind"], item["source_frame"], item["frames_kept_before"], item["atoms_found"],
             item["cut_line_bytes"], item["followed_by"], item["model_record"]) for item in summary["items"]] == [
        ("incomplete_frame", 11, 11, 7, 42, "MODEL record", "MODEL     12"),
        ("incomplete_frame", FRAMES - 1, FRAMES - 2, 8, len(left), "end of file", "MODEL     30")]
    assert summary["truncated"] and (summary["frames_in_source"], summary["frames_kept"]) == (FRAMES, FRAMES - 2)


@pytest.mark.parametrize("ending", ["MODEL     30\n", "MODEL     3", "MODEL\n"])
def test_model_record_at_the_end_without_a_frame(tmp_path, ending):
    coords, text = regular()
    header, frames, _ = split_frames(text)
    damaged_text = join(header, frames[:-1], []) + ending
    source = make_source(tmp_path / "data", "DGlcpa1-OH", text=damaged_text)
    with pytest.raises(archive.PdbFormatError, match="is not terminated by ENDMDL; the file is truncated"):
        archive.convert_entry(source, tmp_path / "archive")

    result = archive.convert_entry(source, tmp_path / "archive", repair=True)
    manifest = check_entry(result, source, coords[:-1])
    unnumbered = ending == "MODEL\n"                # listed too, because the source needed a repair
    assert manifest["source_irregularities"] == {
        "accepted_with": "repair",
        "frames_in_source": FRAMES - 1,              # a MODEL record alone is not a frame
        "frames_kept": FRAMES - 1,
        "frames_dropped": 0,
        "bytes_ignored": len(ending),
        "truncated": False,
        "counts": {"model_record_without_frame": 1, **({"model_records_without_number": 1} if unnumbered else {})},
        "items": [{
            "kind": "model_record_without_frame",
            "action": "ignored",
            "source_frames_before": FRAMES - 1,
            "frames_kept_before": FRAMES - 1,
            "offset": len(damaged_text) - len(ending),
            "bytes": len(ending),
            "bytes_ignored": len(ending),
            "sha256": hashlib.sha256(ending.encode()).hexdigest(),
            "model_record": ending.rstrip("\n"),
            "followed_by": "end of file",
        }] + ([{"kind": "model_records_without_number", "action": "noted", "count": 1, "example": "MODEL"}]
              if unnumbered else []),
    }


# ---------------------------------------------------------------------------
# A frame with every atom record and no ENDMDL
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("place,ending", [
    ("middle", ""), ("middle", "no TER"), ("middle", "ENDM\n"),
    ("end", ""), ("end", "no TER"), ("end", "TER      14      4Y"), ("end", "ENDM"), ("end", "END\n"),
])
def test_complete_frame_without_endmdl_is_kept_and_recorded(tmp_path, place, ending):
    coords, text = regular()
    header, frames, tail = split_frames(text)
    index = 9 if place == "middle" else FRAMES - 1
    assert frames[index][-2].startswith("TER      14      4YB") and frames[index][-1] == "ENDMDL\n"
    # every atom record, and then: the TER of the frame, nothing, or what is left of a line
    whole = frames[index][:-2] if ending.startswith(("TER", "no TER")) else frames[index][:-1]
    is_record = ending == "END\n"                    # a record of its own: it follows the frame
    ending = "" if ending == "no TER" else ending
    damaged = list(frames)
    damaged[index] = whole + ([ending] if ending and not is_record else [])
    damaged_text = join(header, damaged, tail if place == "middle" else ([ending] if is_record else []))

    source = make_source(tmp_path / "data", "DGlcpa1-OH", text=damaged_text)
    with pytest.raises(archive.PdbFormatError, match="before the ENDMDL of this frame" if place == "middle"
                       else "is not terminated by ENDMDL"):
        archive.convert_entry(source, tmp_path / "archive")

    result = archive.convert_entry(source, tmp_path / "archive", repair=True)
    manifest = check_entry(result, source, coords)             # no frame is lost
    frame_text = "".join(damaged[index])
    cut_bytes = 0 if is_record else len(ending)
    truncated = place == "end" and not is_record
    item = {
        "kind": "frame_without_endmdl",
        "action": "kept",
        "source_frame": index,
        "frames_kept_before": index,
        "offset": offset_of(damaged, header, index),
        "bytes": len(frame_text),
        "bytes_ignored": cut_bytes,
        "sha256": hashlib.sha256(frame_text.encode()).hexdigest(),
        "model_record": "MODEL     %d" % (index + 1),
        "atoms_found": ATOMS,
        "atoms_expected": ATOMS,
        "cut_line_bytes": cut_bytes,
        "followed_by": "MODEL record" if place == "middle" else "END record" if is_record else "end of file",
    }
    if truncated:
        item.update(file_truncated=True, file_size=len(damaged_text))
    assert manifest["source_irregularities"] == {
        "accepted_with": "repair",
        "frames_in_source": FRAMES,
        "frames_kept": FRAMES,
        "frames_dropped": 0,
        "bytes_ignored": cut_bytes,
        "truncated": truncated,
        "counts": {"frame_without_endmdl": 1},
        "items": [item],
    }
    assert manifest["verification"]["source_irregularities"]["endmdl_records"] == FRAMES - 1


# ---------------------------------------------------------------------------
# Frames without a MODEL record
# ---------------------------------------------------------------------------
def test_unformatted_model_records_and_a_first_frame_without_one(tmp_path):
    # What a writer produced that printed its template instead of the number, after the frame
    # instead of before it: no MODEL before the first frame, "MODEL {i+1}" after every ENDMDL.
    coords = random_coords(40, 14, seed=6)
    text = pdb_text(coords, dialect="mdanalysis", model=lambda i: "MODEL {i+1}", newline="\r\n")
    header, frames, _ = split_frames(text)
    assert all(frame[0] == "MODEL {i+1}\r\n" for frame in frames)
    damaged_text = join(header, [frames[0][1:]] + frames[1:], ["MODEL {i+1}\r\n"])
    assert damaged_text.count("MODEL {i+1}") == 40 and damaged_text.startswith("HEADER")

    source = make_source(tmp_path / "data", "DGlcpa1-OH", text=damaged_text.encode())
    with pytest.raises(archive.PdbFormatError, match=r"ENDMDL record at byte \d+ after ATOM records that are outside "
                                                     r"any MODEL; with repair \(--repair"):
        archive.convert_entry(source, tmp_path / "archive")

    result = archive.convert_entry(source, tmp_path / "archive", repair=True)
    manifest = check_entry(result, source, coords)
    assert manifest["source"]["frames_parsed_line_by_line"] == 1          # the block parser read the rest
    assert manifest["source_irregularities"] == {
        "accepted_with": "repair",
        "frames_in_source": 40,
        "frames_kept": 40,
        "frames_dropped": 0,
        "bytes_ignored": 13,
        "truncated": False,
        "counts": {"frame_without_model_record": 1, "model_record_without_frame": 1, "model_records_without_number": 40},
        "items": [
            {
                "kind": "model_record_without_frame",
                "action": "ignored",
                "source_frames_before": 40,
                "frames_kept_before": 40,
                "offset": len(damaged_text) - 13,
                "bytes": 13,
                "bytes_ignored": 13,
                "sha256": hashlib.sha256(b"MODEL {i+1}\r\n").hexdigest(),
                "model_record": "MODEL {i+1}",
                "followed_by": "end of file",
            },
            {"kind": "frame_without_model_record", "action": "kept", "count": 1, "source_frames": [0],
             "offsets": [len("".join(header))]},
            {"kind": "model_records_without_number", "action": "noted", "count": 40, "example": "MODEL {i+1}"},
        ],
    }
    # the numbers of MODEL records are not kept in the archive; -1 stands for a record without one
    assert manifest["source"]["model_numbers"] == {
        "first": -1, "last": -1, "largest": -1, "sequential": False, "breaks": 79}
    assert manifest["verification"]["source_irregularities"] == {
        "items_checked": 1, "model_records": 40, "endmdl_records": 40, "bytes_ignored": 13,
        "comparison": manifest["verification"]["source_irregularities"]["comparison"]}


@pytest.mark.parametrize("missing", [[17], [0, 5, 6, 29], list(range(FRAMES))])
def test_frames_that_endmdl_ends_and_no_model_starts(tmp_path, missing):
    coords, text = regular()
    header, frames, tail = split_frames(text)
    damaged = [frame[1:] if index in missing else frame for index, frame in enumerate(frames)]
    damaged_text = join(header, damaged, tail)
    source = make_source(tmp_path / "data", "DGlcpa1-OH", text=damaged_text)
    with pytest.raises(archive.PdbFormatError, match="ATOM record outside MODEL/ENDMDL at byte" if missing[0]
                       else "ENDMDL record at byte .* outside any MODEL"):
        archive.convert_entry(source, tmp_path / "archive")

    result = archive.convert_entry(source, tmp_path / "archive", repair=True, batch_bytes=3000)
    manifest = check_entry(result, source, coords)
    assert manifest["source_irregularities"] == {
        "accepted_with": "repair",
        "frames_in_source": FRAMES,
        "frames_kept": FRAMES,
        "frames_dropped": 0,
        "bytes_ignored": 0,
        "truncated": False,
        "counts": {"frame_without_model_record": len(missing)},
        "items": [{"kind": "frame_without_model_record", "action": "kept", "count": len(missing),
                   "source_frames": missing, "offsets": [offset_of(damaged, header, index) for index in missing]}],
    }
    assert manifest["verification"]["source_irregularities"]["model_records"] == FRAMES - len(missing)
    assert (manifest["source"]["model_numbers"] is None) == (len(missing) == FRAMES)


# ---------------------------------------------------------------------------
# Several things in one file, and how the file is read in blocks
# ---------------------------------------------------------------------------
def several_irregularities():
    """A source with a frame without MODEL, two cut frames, a frame without ENDMDL, and a cut end."""
    coords, text = regular(frames=60, seed=4)
    header, frames, _ = split_frames(text)
    whole = atom_lines_of(frames[20])
    damaged = list(frames)
    damaged[3] = frames[3][1:]                                    # no MODEL record
    damaged[20] = cut(frames[20], 4, whole[4][:40] + "\n")       # cut short, the line too
    damaged[21] = frames[21][:-1]                                 # every atom record, no ENDMDL
    damaged[40] = cut(frames[40], 0)                              # MODEL record alone
    damaged[41] = cut(frames[41], 11)                             # one atom record short
    damaged[59] = cut(frames[59], 6, piece(frames[59], 6, 14))   # the file ends here
    kept = np.delete(coords, [20, 40, 41, 59], axis=0)
    return kept, join(header, damaged, [])


@pytest.mark.parametrize("block_bytes,batch_bytes", [(61, 1), (997, 300), (4096, 1 << 20), (1 << 20, 2000)])
def test_block_and_batch_sizes_do_not_change_what_repair_reads(tmp_path, block_bytes, batch_bytes):
    kept, text = several_irregularities()
    path = tmp_path / "simulation.pdb"
    path.write_text(text)
    reference = archive.PdbTrajectory(path, repair=True)
    expected = np.concatenate([block for block, _ in reference.blocks()])
    assert np.array_equal(expected, kept)

    reader = archive.PdbTrajectory(path, block_bytes=block_bytes, batch_bytes=batch_bytes, repair=True)
    blocks = [block for block, _ in reader.blocks()]
    assert all(len(block) for block in blocks) and np.array_equal(np.concatenate(blocks), kept)
    assert reader.irregularities == reference.irregularities
    assert (reader.n_frames, reader.source_frames) == (56, 59)
    assert reader.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert np.array_equal(reader.frame_offsets, reference.frame_offsets)
    payload = path.read_bytes()
    for offset, length in zip(reader.frame_offsets, reader.frame_lengths):   # the kept frames, where they are
        frame = payload[offset:offset + length]
        assert frame.count(b"\nATOM") + frame.startswith(b"ATOM") == ATOMS and frame.endswith(b"\n")

    summary = reference.irregularities
    assert summary["counts"] == {
        "frame_without_endmdl": 1, "frame_without_model_record": 1, "incomplete_frame": 3,
        "model_record_without_frame": 1}
    assert [(item["kind"], item.get("source_frame", item.get("source_frames_before")), item["frames_kept_before"])
            for item in summary["items"][:5]] == [
        ("incomplete_frame", 20, 20), ("frame_without_endmdl", 21, 20), ("model_record_without_frame", 40, 39),
        ("incomplete_frame", 40, 39), ("incomplete_frame", 58, 56)]
    assert summary["items"][5]["source_frames"] == [3]
    assert (summary["frames_in_source"], summary["frames_kept"], summary["frames_dropped"]) == (59, 56, 3)
    assert summary["truncated"] and summary["items"][4]["file_truncated"]


def test_entry_with_several_irregularities_verifies(tmp_path):
    kept, text = several_irregularities()
    source, result = convert(tmp_path, text, check_frames=3)
    manifest = check_entry(result, source, kept)
    # the first, the middle and the last frame, and both sides of everything that is not a regular frame
    assert manifest["verification"]["reference_frames_checked"] == len({0, 28, 55, 19, 20, 38, 39})
    assert manifest["verification"]["source_irregularities"]["items_checked"] == 5

    # The record is also held against where the reader says the frames of the archive are in the file.
    pdb = source / "simulation.pdb"
    reader = archive.PdbTrajectory(pdb, repair=True)
    assert sum(len(block) for block, _ in reader.blocks()) == 56
    recorded = manifest["source_irregularities"]
    assert archive._check_irregularities(pdb, recorded, 56, ATOMS, reader)["items_checked"] == 5

    class Shifted:   # a reader that takes frame 31 of the archive from where frame 32 is
        frame_offsets = reader.frame_offsets.copy()
        frame_lengths = reader.frame_lengths.copy()

    Shifted.frame_offsets[31] = Shifted.frame_offsets[32]
    with pytest.raises(archive.VerificationError, match="are not the frames that the ENDMDL records of the file end"):
        archive._check_irregularities(pdb, recorded, 56, ATOMS, Shifted)


def test_more_irregular_frames_than_the_limit_are_refused(tmp_path, monkeypatch):
    coords, text = regular()
    header, frames, tail = split_frames(text)
    damaged = [cut(frame, 5) if index in (4, 8, 12, 16) else frame for index, frame in enumerate(frames)]
    source = make_source(tmp_path / "data", "DGlcpa1-OH", text=join(header, damaged, tail))
    monkeypatch.setattr(archive, "MAX_REPAIRED_FRAMES", 3)
    with pytest.raises(archive.PdbFormatError, match=r"frame 16 .* 3 frames or MODEL records .* repaired already"):
        archive.convert_entry(source, tmp_path / "archive", repair=True)
    monkeypatch.setattr(archive, "MAX_REPAIRED_FRAMES", 4)
    assert archive.convert_entry(source, tmp_path / "archive", repair=True)["manifest"]["frames"] == FRAMES - 4


# ---------------------------------------------------------------------------
# What repair does not accept
# ---------------------------------------------------------------------------
def _replace_atom(frame, atom, change):
    lines = list(frame)
    position = [i for i, line in enumerate(lines) if line.startswith("ATOM")][atom]
    lines[position] = change(lines[position])
    return lines


def _without_atom(frame, atom):
    lines = list(frame)
    del lines[[i for i, line in enumerate(lines) if line.startswith("ATOM")][atom]]
    return lines


REFUSED = {
    # frames that ENDMDL ends are never repaired
    "atom missing, ENDMDL there": (
        lambda f: _without_atom(f[11], 4), archive.AtomCountError, "frame 11 has 11 atoms, the first frame has 12"),
    "atom twice, ENDMDL there": (
        lambda f: f[11][:2] + f[11][1:], archive.AtomCountError, "frame 11 has 13 atoms"),
    "other residue, ENDMDL there": (
        lambda f: _replace_atom(f[11], 5, lambda line: line[:17] + "XYZ" + line[20:]),
        archive.TopologyChangeError, "frame 11, atom 5: columns 1-30"),
    "unreadable coordinate, ENDMDL there": (
        lambda f: _replace_atom(f[11], 5, lambda line: line[:38] + "********" + line[46:]),
        archive.PdbFormatError, "frame 11, atom 5: coordinate field '\\*{8}'"),
    # frames without ENDMDL: only the records of the first frame, in order, as far as they go
    "more atoms than the first frame": (
        lambda f: f[11][:-1] + atom_lines_of(f[11])[:1], archive.AtomCountError, "frame 11 .* has more than 12 atoms"),
    "cut frame with another atom": (
        lambda f: cut(_replace_atom(f[11], 2, lambda line: line[:12] + " XX " + line[16:]), 7),
        archive.TopologyChangeError, "frame 11, atom 2: columns 1-30"),
    "cut frame that lacks an atom in the middle": (
        lambda f: cut(_without_atom(f[11], 2), 7), archive.TopologyChangeError, "frame 11, atom 2: columns 1-30"),
    "cut frame with an unreadable coordinate": (
        lambda f: cut(_replace_atom(f[11], 2, lambda line: line[:30] + "  1.2345" + line[38:]), 7),
        archive.PdbFormatError, "frame 11, atom 2: coordinate field"),
    "whole frame without ENDMDL with an unreadable coordinate": (
        lambda f: _replace_atom(f[11], 2, lambda line: line[:30] + "     nan" + line[38:])[:-1],
        archive.PdbFormatError, "frame 11, atom 2: coordinate field"),
    "last atom record with other text at its end": (
        lambda f: cut(f[11], 11, atom_lines_of(f[11])[11].rstrip() + "x\n"),
        archive.TopologyChangeError, "frame 11, atom 11: columns 55-80"),
    "cut line that is not the beginning of the record": (
        lambda f: cut(f[11], 7, "ATOM     10  \n"), archive.PdbFormatError,
        "the atom record at byte \\d+ is not the record of atom 7 of the first frame and not the beginning of it"),
    "short line of another atom with coordinates": (
        lambda f: cut(f[11], 7, atom_lines_of(f[11])[8][:60] + "\n"), archive.TopologyChangeError,
        "frame 11, atom 7: columns 1-30"),
    "cut line with other text after it": (
        lambda f: cut(f[11], 7, "ATOMODEL\n"), archive.PdbFormatError, "begins like a record of the frame"),
    "atom records after the cut line": (
        lambda f: cut(f[11], 7, "ATO\n") + atom_lines_of(f[11])[8:], archive.PdbFormatError,
        "ATOM record outside MODEL/ENDMDL at byte \\d+ \\(after frame 12\\): the MODEL record at byte \\d+ comes "
        "before any ENDMDL"),
    "atom records after the frame, neither MODEL nor ENDMDL": (
        lambda f: f[11] + atom_lines_of(f[11])[:3], archive.PdbFormatError,
        "ATOM record outside MODEL/ENDMDL at byte \\d+ \\(after frame 12\\)"),
    "ENDMDL twice": (
        lambda f: f[11] + ["ENDMDL\n"], archive.PdbFormatError, "ENDMDL without MODEL at byte"),
}


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_repair_refuses_what_is_not_a_cut_frame_or_a_missing_model_record(tmp_path, case):
    damage, error, message = REFUSED[case]
    coords, text = regular()
    header, frames, tail = split_frames(text)
    damaged = list(frames)
    damaged[11] = damage(frames)
    source = make_source(tmp_path / "data", "DGlcpa1-OH", text=join(header, damaged, tail))
    with pytest.raises(archive.PdbFormatError):
        archive.convert_entry(source, tmp_path / "archive")
    with pytest.raises(error, match=message):
        archive.convert_entry(source, tmp_path / "archive", repair=True)
    assert [p.name for p in (tmp_path / "archive").iterdir()] == [archive.STAGING_DIR]
    assert not any((tmp_path / "archive" / archive.STAGING_DIR).iterdir())


def test_repair_needs_a_complete_first_frame(tmp_path):
    coords, text = regular()
    header, frames, tail = split_frames(text)
    source = make_source(tmp_path / "data", "DGlcpa1-OH", text=join(header, [cut(frames[0], 7)] + frames[1:], tail))
    with pytest.raises(archive.PdbFormatError, match="frame 0 .* repaired only when its first frame is complete"):
        archive.convert_entry(source, tmp_path / "archive", repair=True)

    # atom records with neither MODEL nor ENDMDL before the first MODEL record
    source = make_source(tmp_path / "other", "DGlcpa1-OH", text=join(header, [frames[0][1:-1]] + frames[1:], tail))
    for repair in (False, True):
        with pytest.raises(archive.PdbFormatError, match="MODEL record at byte" if not repair
                           else "the MODEL record at byte .* comes before any ENDMDL"):
            archive.convert_entry(source, tmp_path / "archive", repair=repair)


# ---------------------------------------------------------------------------
# A regular source is not touched by repair
# ---------------------------------------------------------------------------
def _single_structure():
    coords = random_coords(1, ATOMS)
    lines = "".join(atom_line(i + 1, atom, xyz) for i, (atom, xyz) in enumerate(zip(atom_table(ATOMS), coords[0])))
    return "REMARK single\n" + lines + "TER\nEND\n"


REGULAR = {
    "cpptraj": lambda: regular()[1],
    "mdanalysis": lambda: regular(dialect="mdanalysis")[1],
    "windows": lambda: regular(newline="\r\n")[1],
    "MODEL without number": lambda: regular(model=lambda i: "MODEL     ****")[1],
    "two runs": lambda: regular(seed=1)[1] + regular(seed=2)[1],
    "no line end at the end": lambda: regular()[1].rstrip("\n"),
    "ENDMDL is the last record, without line end": lambda: regular(tail="")[1].rstrip("\n"),
    "records after the last frame": lambda: regular(tail="CONECT    1    2\nEND\n")[1],
    "single structure": _single_structure,
}


@pytest.mark.parametrize("case", sorted(REGULAR))
def test_regular_source_gives_the_same_entry_with_and_without_repair(tmp_path, case):
    source = make_source(tmp_path / "data", "DGlcpa1-OH", text=REGULAR[case](), info_json="DGlcpa1-OH")
    plain = archive.convert_entry(source, tmp_path / "plain", block_bytes=2000)
    repaired = archive.convert_entry(source, tmp_path / "repair", block_bytes=2000, repair=True)

    assert list(repaired["manifest"]) == list(plain["manifest"]) == [
        "schema", "schema_version", "glycam_name", "folder", "object_key_prefix", "atoms", "frames", "box",
        "trajectory", "files", "source", "verification", "converter", "warnings"]
    assert "source_irregularities" not in repaired["manifest"]["verification"]
    for name in ("trajectory.xtc", "topology.pdb", "structure.mol2", "info.json"):
        assert (Path(repaired["entry"]) / name).read_bytes() == (Path(plain["entry"]) / name).read_bytes()

    def without_date(path):
        text = (Path(path) / "manifest.json").read_text()
        date = json.loads(text)["converter"]["date"]
        assert text.count(date) == 1
        return text.replace(date, "")

    assert without_date(repaired["entry"]) == without_date(plain["entry"])            # the bytes of the manifest
    assert "repaired" not in archive.verify_entry(repaired["entry"], source)
    assert archive.build_index(tmp_path / "repair")[0]["repaired"] == []
    assert "repaired" not in archive.build_index(tmp_path / "repair")[0]["records"][0]


# ---------------------------------------------------------------------------
# Verifier, resume, index
# ---------------------------------------------------------------------------
def cut_source(tmp_path, folder="DGlcpa1-OH", frames=FRAMES, at=11, seed=0, **options):
    """A source whose frame ``at`` is cut after seven atom records, and the frames that are complete."""
    coords, text = regular(frames=frames, seed=seed)
    header, lines, tail = split_frames(text)
    damaged = list(lines)
    last = at == frames - 1
    damaged[at] = cut(lines[at], 7, piece(lines[at], 7, 14) + ("" if last else "\n"))
    if last:
        tail = []
    source = make_source(tmp_path / "data", folder, text=join(header, damaged, tail), **options)
    return source, np.delete(coords, at, axis=0)


def test_verifier_holds_a_repaired_entry_to_its_record(tmp_path):
    source, kept = cut_source(tmp_path)
    entry = Path(archive.convert_entry(source, tmp_path / "archive", repair=True)["entry"])
    assert archive.verify_entry(entry, source)["ok"]
    original = (entry / "manifest.json").read_text()
    pdb = (source / "simulation.pdb").read_bytes()

    def errors_with(change):
        manifest = json.loads(original)
        change(manifest)
        (entry / "manifest.json").write_text(json.dumps(manifest))
        errors = archive.verify_entry(entry, source)["errors"]
        (entry / "manifest.json").write_text(original)
        return errors

    def item(manifest):
        return manifest["source_irregularities"]["items"][0]

    # a manifest that hides the repair: the source is then read as a regular one, and is not
    errors = errors_with(lambda manifest: manifest.pop("source_irregularities"))
    assert len(errors) == 1 and "before the ENDMDL of this frame" in errors[0]

    # a record that says something else than the file
    for change, message in [
        (lambda m: item(m).update(atoms_found=6), "7 whole atom records and a cut line of 15 bytes"),
        (lambda m: item(m).update(source_frame=12), "11 frames of the source precede it"),
        (lambda m: item(m).update(frames_kept_before=10), "11 complete frames precede it"),
        (lambda m: item(m).update(offset=item(m)["offset"] + 1), "no MODEL record there"),
        (lambda m: item(m).update(bytes=item(m)["bytes"] - 1), "do not have the recorded SHA-256"),
        (lambda m: item(m).update(sha256="0" * 64), "do not have the recorded SHA-256"),
        (lambda m: item(m).update(cut_line_bytes=0), "whole atom records and a cut line of 0 bytes"),
        (lambda m: item(m).update(followed_by="end of file"), "followed by MODEL record, not by end of file"),
        (lambda m: item(m).update(kind="frame_without_endmdl"), "the numbers of frames in the source"),
        (lambda m: m["source_irregularities"].update(frames_in_source=FRAMES + 1), "the numbers of frames in the source"),
        (lambda m: m["source_irregularities"].update(bytes_ignored=1), "bytes ignored, not 1"),
        (lambda m: m["source_irregularities"].update(truncated=True), "truncated"),
        (lambda m: item(m).pop("offset"), "the numbers of frames in the source"),
        (lambda m: m["source_irregularities"].pop("items"), "incomplete source_irregularities: KeyError"),
    ]:
        errors = errors_with(change)
        assert any(message in error for error in errors), (message, errors)
        assert any("are not those source_irregularities records" in error for error in errors) or len(errors) == 1
    errors = archive.verify_entry(entry, deep=True)["errors"]
    assert errors == []
    manifest = json.loads(original)
    manifest["source_irregularities"]["frames_kept"] = FRAMES
    (entry / "manifest.json").write_text(json.dumps(manifest))
    assert archive.verify_entry(entry)["errors"] == [
        f"source_irregularities says {FRAMES} frames were kept, the manifest says {FRAMES - 1} frames"]
    (entry / "manifest.json").write_text(original)

    # a source that changed: the dropped frame is whole now
    coords, text = regular()
    (source / "simulation.pdb").write_text(text)
    errors = archive.verify_entry(entry, source)["errors"]
    assert len(errors) == 1 and "29 frames of 12 atoms where the source has at least 30 frames" in errors[0]
    (source / "simulation.pdb").write_bytes(pdb)
    assert archive.verify_entry(entry, source)["ok"]


def test_second_run_skips_a_repaired_entry_with_and_without_repair(tmp_path):
    source, kept = cut_source(tmp_path)
    target = tmp_path / "archive"
    with pytest.raises(archive.PdbFormatError):
        archive.convert_entry(source, target)
    first = archive.convert_entry(source, target, repair=True)
    entry = Path(first["entry"])
    before = snapshot(entry)

    for repair in (True, False):   # the entry is there and describes this source
        again = archive.convert_entry(source, target, repair=repair)
        assert again["status"] == "skipped" and again["manifest"] == first["manifest"]
        assert snapshot(entry) == before

    # converting it again without repair fails and leaves the entry alone
    with pytest.raises(archive.PdbFormatError, match="before the ENDMDL of this frame"):
        archive.convert_entry(source, target, force=True)
    assert snapshot(entry) == before and archive.verify_entry(entry, source)["ok"]
    assert archive.convert_entry(source, target, force=True, repair=True)["status"] == "converted"

    # a damaged entry is converted again, and only with repair
    with open(entry / "trajectory.xtc", "r+b") as handle:
        handle.truncate(1000)
    with pytest.raises(archive.PdbFormatError):
        archive.convert_entry(source, target)
    assert archive.convert_entry(source, target, repair=True)["status"] == "converted"
    assert np.array_equal(archive.ArchiveEntry(entry).read_frames(unit="milliangstrom"), kept)


def test_index_lists_repaired_entries(tmp_path):
    cut_middle, _ = cut_source(tmp_path, "DGlcpa1-OH")
    cut_end, _ = cut_source(tmp_path, "DGalpa1-OH", at=FRAMES - 1, seed=1)
    whole = make_source(tmp_path / "data", "DManpa1-OH", random_coords(8, ATOMS, seed=2))
    target = tmp_path / "archive"
    for source in (cut_middle, cut_end, whole):
        archive.convert_entry(source, target, repair=True)
    index, problems = archive.write_index(target)
    assert problems == [] and index["entries"] == 3
    assert index["repaired"] == ["DGalpa1-OH", "DGlcpa1-OH"] == json.loads((target / "index.json").read_text())["repaired"]
    records = {record["glycam_name"]: record for record in index["records"]}
    assert "repaired" not in records["DManpa1-OH"]
    assert list(records["DManpa1-OH"]) == [
        "glycam_name", "folder", "object_key_prefix", "source_folder", "atoms", "frames", "source_size",
        "source_sha256", "archive_size", "files"]
    assert records["DGlcpa1-OH"]["frames"] == FRAMES - 1
    assert records["DGlcpa1-OH"]["repaired"] == {
        "frames_in_source": FRAMES, "frames_dropped": 1, "bytes_ignored": records["DGlcpa1-OH"]["repaired"]["bytes_ignored"],
        "truncated": False, "counts": {"incomplete_frame": 1}}
    assert records["DGalpa1-OH"]["repaired"]["truncated"] is True
    assert index["totals"]["frames"] == 2 * (FRAMES - 1) + 8


# ---------------------------------------------------------------------------
# Converter command line
# ---------------------------------------------------------------------------
def test_converter_cli_repairs_only_when_asked(tmp_path):
    data, target = tmp_path / "data", tmp_path / "archive"
    _, kept_middle = cut_source(tmp_path, "DGlcpa1-OH")
    _, kept_end = cut_source(tmp_path, "DGalpa1-OH", at=FRAMES - 1, seed=1, info_json="DGalpa1-OH")
    whole = random_coords(20, 14, seed=2)
    make_source(data, "DManpa1-OH", whole)
    coords, text = regular(seed=5)
    header, frames, tail = split_frames(text)
    make_source(data, "DAllpb1-OH", text=join(header, frames[:6] + [frames[6][:3] + frames[6][4:]] + frames[7:], tail))
    before = snapshot(data)

    # without the option: the regular entry is converted, the others fail and say what to do
    run = run_converter("--data-dir", data, "--archive-dir", target, "--workers", 2)
    assert run.returncode == 1
    report = json.loads((target / "conversion_report.json").read_text())
    assert report["counts"] == {"converted": 1, "skipped": 0, "failed": 3}
    assert report["repair"] is False and report["repaired"] == []
    reasons = {row["source_folder"]: row["reason"] for row in report["failed"]}
    assert "with repair (--repair of the converter)" in reasons["DGlcpa1-OH"]
    assert "the file is truncated; with repair (--repair of the converter)" in reasons["DGalpa1-OH"]
    assert reasons["DAllpb1-OH"] == "frame 6 has 11 atoms, the first frame has 12"
    plain = snapshot(target / "DManpa1-OH")
    first_report = (target / "conversion_report.json").read_bytes()

    # with it: the two cut sources are archived, the one that is wrong in another way still fails
    other_report = tmp_path / "reports" / "repair.json"
    other_report.parent.mkdir()
    run = run_converter("--data-dir", data, "--archive-dir", target, "--workers", 2, "--repair", "--report", other_report)
    assert run.returncode == 1 and snapshot(data) == before
    assert (target / "conversion_report.json").read_bytes() == first_report       # --report: that one stays
    report = json.loads(other_report.read_text())
    assert report["counts"] == {"converted": 2, "skipped": 1, "failed": 1} and report["repair"] is True
    assert [row["source_folder"] for row in report["failed"]] == ["DAllpb1-OH"]
    assert snapshot(target / "DManpa1-OH") == plain
    assert [(row["source_folder"], row["status"], row["frames"], row["frames_in_source"], row["frames_dropped"],
             row["truncated"], row["counts"]) for row in report["repaired"]] == [
        ("DGalpa1-OH", "converted", FRAMES - 1, FRAMES, 1, True, {"incomplete_frame": 1}),
        ("DGlcpa1-OH", "converted", FRAMES - 1, FRAMES, 1, False, {"incomplete_frame": 1})]
    assert all("repaired" in row for row in report["converted"])
    assert run.stderr.count("REPAIRED ") == 2 and "REPAIRED DGlcpa1-OH (converted): 29 of 30 frames" in run.stderr
    assert "2 of them repaired" in run.stdout and "REPAIRED: 29 of 30 frames of the source kept" in run.stdout

    index = json.loads((target / "index.json").read_text())
    assert index["entries"] == 3 and index["repaired"] == ["DGalpa1-OH", "DGlcpa1-OH"]
    assert np.array_equal(archive.ArchiveEntry(target / "DGlcpa1-OH").read_frames(unit="milliangstrom"), kept_middle)
    assert np.array_equal(archive.ArchiveEntry(target / "DGalpa1-OH").read_frames(unit="milliangstrom"), kept_end)
    assert np.array_equal(archive.ArchiveEntry(target / "DManpa1-OH").read_frames(unit="milliangstrom"), whole)
    manifest = json.loads((target / "DGalpa1-OH" / "manifest.json").read_text())
    assert manifest["source_irregularities"]["truncated"] and "info.json" in manifest["files"]

    # a second run skips the repaired entries, with the option or without, and still lists them
    entries = snapshot(target / "DGlcpa1-OH")
    for extra in (["--repair"], []):
        run = run_converter("--data-dir", data, "--archive-dir", target, *extra)
        report = json.loads((target / "conversion_report.json").read_text())
        assert run.returncode == 1 and report["counts"] == {"converted": 0, "skipped": 3, "failed": 1}
        assert [(row["source_folder"], row["status"]) for row in report["repaired"]] == [
            ("DGalpa1-OH", "skipped"), ("DGlcpa1-OH", "skipped")]
        assert snapshot(target / "DGlcpa1-OH") == entries
        assert json.loads((target / "index.json").read_text())["repaired"] == ["DGalpa1-OH", "DGlcpa1-OH"]

    # verification of the archive against the source reads each source as its manifest says
    run = run_converter("--verify-only", "--data-dir", data, "--archive-dir", target, "--check-frames", -1)
    assert run.returncode == 0, run.stdout + run.stderr
    summary = json.loads((target / "verification_report.json").read_text())
    assert (summary["entries"], summary["ok"], summary["failed"]) == (3, 3, [])
    assert {row["folder"]: row.get("repaired", False) for row in summary["reports"]} == {
        "DGalpa1-OH": True, "DGlcpa1-OH": True, "DManpa1-OH": False}
    assert all(row["mismatched_coordinates"] == 0 for row in summary["reports"])

    # --report moves the verification report too; it does not go into the source or a missing directory
    stored = (target / "verification_report.json").read_bytes()
    run = run_converter("--verify-only", "--archive-dir", target, "--report", other_report.with_name("verify.json"))
    assert run.returncode == 0 and json.loads(other_report.with_name("verify.json").read_text())["ok"] == 3
    assert (target / "verification_report.json").read_bytes() == stored
    for place in (data / "report.json", data / "DManpa1-OH" / "report.json", tmp_path / "missing" / "report.json"):
        run = run_converter("--data-dir", data, "--archive-dir", target, "--report", place)
        assert run.returncode != 0 and "--report" in run.stderr and not place.exists()
    assert snapshot(data) == before


# ---------------------------------------------------------------------------
# Readers of a repaired entry
# ---------------------------------------------------------------------------
CHAIN = [("C1", "0GB", 1, "C"), ("C2", "0GB", 1, "C"), ("O2", "0GB", 1, "O"),
         ("C3", "0GB", 1, "C"), ("O3", "0GB", 1, "O"), ("H3O", "0GB", 1, "H")]


def damaged_chain(root, folder, shape):
    """A six-atom chain with plausible bonds; frames 7 and 41 have a torn bond; the text is damaged."""
    rng = np.random.default_rng(11)
    base = np.cumsum([[0, 0, 0], [1520, 0, 0], [0, 1430, 0], [1430, 0, 0], [0, 1430, 0], [960, 0, 0]], axis=0)
    coords = base[None] + rng.integers(-40, 41, size=(60, 6, 3))
    for frame in (7, 41):
        coords[frame, 5] += (9000, 0, 0)
    header, frames, tail = split_frames(pdb_text(coords, atoms=CHAIN, ter=False))
    if shape == "cut":                                  # frame 30 cut short in the middle, the last one at the end
        frames[30] = cut(frames[30], 4, piece(frames[30], 4, 15) + "\n")
        frames[59] = cut(frames[59], 2, "ATO")
        kept, tail = np.delete(coords, [30, 59], axis=0), []
    else:                                               # no MODEL before the first frame, a template after each
        frames = [frames[0][1:]] + [["MODEL {i+1}\n"] + frame[1:] for frame in frames[1:]]
        kept, tail = coords, ["MODEL {i+1}\n"]
    source = make_source(root, folder, text=join(header, frames, tail))
    (source / "structure.mol2").write_text(mol2_text([1, 2, 3, 4, 5, 6]), encoding="utf-8")
    return source, coords, kept


@pytest.mark.parametrize("shape", ["cut", "template"])
def test_store_step_reads_a_repaired_entry(tmp_path, shape):
    import polars as pl
    import zarr

    from lib.glypdbio import parse_mol2_bonds
    from lib.glystore import store_from_pdb, store_trajectory
    from lib.v2 import GlycanAnalysisPipeline, step_store

    source, coords, kept = damaged_chain(tmp_path / "data", "DGlcpa1-OH", shape)
    with pytest.raises(archive.PdbFormatError):
        archive.convert_entry(source, tmp_path / "archive")
    entry = Path(archive.convert_entry(source, tmp_path / "archive", repair=True)["entry"])
    assert archive.ArchiveEntry(entry).n_frames == len(kept) == (58 if shape == "cut" else 60)
    assert archive.entry_inputs(entry)["trajectory_path"] == str(entry / "trajectory.xtc")

    # the store: every kept frame, bit for bit what the text reader gets from the complete frames of the text
    options = dict(get_connectivity=parse_mol2_bonds(source / "structure.mol2"), connectivity_kind="serial",
                   frames_buffer=16, frame_chunk=20)
    store_trajectory(str(entry), str(tmp_path / "from_archive"), **options)
    store_from_pdb(str(source / "simulation.pdb"), str(tmp_path / "from_pdb"), **options)
    stored = zarr.open(str(tmp_path / "from_archive" / "coords.zarr"), mode="r")["xyz"][:]
    assert stored.shape == (len(kept), 6, 3)
    assert np.array_equal(stored, (kept / 1000.0).astype(np.float32))
    from_text = zarr.open(str(tmp_path / "from_pdb" / "coords.zarr"), mode="r")["xyz"][:]
    assert np.array_equal(stored.view(np.uint32), from_text.view(np.uint32))
    for table in ("atoms.parquet", "bonds.parquet", "frames.parquet"):
        assert pl.read_parquet(tmp_path / "from_archive" / table).equals(pl.read_parquet(tmp_path / "from_pdb" / table))
    assert pl.read_parquet(tmp_path / "from_archive" / "frames.parquet")["n_frames"].to_list() == [len(kept)]

    # the store step of the pipeline, with its frame filter
    paths = GlycanAnalysisPipeline(str(tmp_path / "archive"), str(tmp_path / "process")).get_glycan_paths(entry.name)
    assert paths["pdb_path"].endswith("trajectory.xtc") and paths["glycam_name"] == "DGlcpa1-OH"
    result = step_store(paths["input_glycan_dir"], paths["frame_data_dir"], paths["pdb_path"], paths["mol2_path"], force=True)
    assert (result.source_frames, result.retained_frames, result.rejected_frames) == (len(kept), len(kept) - 2, 2)
    retained = zarr.open(str(Path(paths["frame_data_dir"]) / "coords.zarr"), mode="r")["xyz"][:]
    torn = [7, 40] if shape == "cut" else [7, 41]          # frame 41 of the source is frame 40 of the archive
    assert np.array_equal(retained, (np.delete(kept, torn, axis=0) / 1000.0).astype(np.float32))


def test_entry_description_and_upload_of_a_repaired_entry(tmp_path, fake_s3):
    from lib import archive_transfer, entry_run

    source, coords, kept = damaged_chain(tmp_path / "data", "DGlcpa1-OH", "cut")
    make_source(tmp_path / "data", "DManpa1-OH", random_coords(8, ATOMS, seed=2))
    target = tmp_path / "archive"
    entry = Path(archive.convert_entry(source, target, repair=True)["entry"])
    archive.convert_entry(tmp_path / "data" / "DManpa1-OH", target)
    archive.write_index(target)

    described = entry_run.describe_input(entry)
    manifest = json.loads((entry / "manifest.json").read_text())
    assert (described["kind"], described["frames"], described["atoms"]) == ("archive", 58, 6)
    assert described["files_verified_against_manifest"] and described["source"]["sha256"] == manifest["source"]["sha256"]
    assert described["manifest_sha256"] == hashlib.sha256((entry / "manifest.json").read_bytes()).hexdigest()

    backend, client = fake_s3
    report = archive_transfer.upload_archive(target, backend, log=lambda message: None)
    assert report["ok"] and report["counts"] == {"uploaded": 2} and report["index_uploaded"]
    assert client.body("bucket", "raw/DGlcpa1-OH/manifest.json") == (entry / "manifest.json").read_bytes()
    assert json.loads(client.body("bucket", "raw/index.json"))["repaired"] == ["DGlcpa1-OH"]
    comparison = archive_transfer.compare_with_index(backend, "raw", archive_transfer.load_index(target))
    assert comparison["ok"] and comparison["entries"] == 2
    again = archive_transfer.upload_archive(target, backend, log=lambda message: None)
    assert again["ok"] and again["counts"] == {"skipped": 2}

    mirror = tmp_path / "mirror"
    assert archive_transfer.download_archive(backend, mirror, log=lambda message: None)["ok"]
    assert (mirror / "DGlcpa1-OH" / "manifest.json").read_bytes() == (entry / "manifest.json").read_bytes()
    assert archive.verify_entry(mirror / "DGlcpa1-OH", source)["ok"]
    assert np.array_equal(archive.ArchiveEntry(mirror / "DGlcpa1-OH").read_frames(unit="milliangstrom"), kept)
