"""Tests of the raw-trajectory archive (lib/archive.py) and its converter.

Every trajectory is synthetic and built in the test; nothing needs the network.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lib import archive

REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERTER = REPO_ROOT / "scripts" / "convert_data_to_archive.py"

# The GLYCAM names of the three source folders that are named by GlyTouCan ID.
NAME_G50508SG = (
    "DNeup5Aca2-3[DGalpNAcb1-4]DGalpb1-4DGlcpNAcb1-2[DNeup5Aca2-3[DGalpNAcb1-4]DGalpb1-4DGlcpNAcb1-4]"
    "DManpa1-3[DNeup5Aca2-3[DGalpNAcb1-4]DGalpb1-4DGlcpNAcb1-2[DNeup5Aca2-3[DGalpNAcb1-4]DGalpb1-4"
    "DGlcpNAcb1-6]DManpa1-6]DManpb1-4DGlcpNAcb1-4[LFucpa1-6]DGlcpNAca1-OH"
)
NAME_G60371DN = (
    "Neup5Aca2-3[DGalpNAcb1-4]DGalpb1-4DGlcpNAcb1-2[Neup5Aca2-3[DGalpNAcb1-4]DGalpb1-4DGlcpNAcb1-4]"
    "DManpa1-3[DGlcpNAcb1-4][Neup5Aca2-3[DGalpNAcb1-4]DGalpb1-4DGlcpNAcb1-2[Neup5Aca2-3[DGalpNAcb1-4]"
    "DGalpb1-4DGlcpNAcb1-6]DManpa1-6]DManpb1-4DGlcpNAcb1-4[LFucpa1-6]DGlcpNAcb1-OH"
)
NAME_G77147GI = (
    "DNeup5Gca2-6DGalpb1-4DGlcpNAcb1-2DManpa1-3[DManpa1-3[DManpa1-6]DManpa1-6]DManpb1-4DGlcpNAcb1-4DGlcpNAca1-OH"
)


# ---------------------------------------------------------------------------
# Synthetic source folders
# ---------------------------------------------------------------------------
def atom_table(count):
    """(name, residue name, residue number, element) for ``count`` atoms."""
    elements = "COHCOCHO"
    return [
        (f"{elements[i % 8]}{i % 9 + 1}", ("ROH", "0GB", "4YB")[min(i // 5, 2)], min(i // 5, 2) + 1, elements[i % 8])
        for i in range(count)
    ]


def atom_line(serial, atom, xyz, dialect="cpptraj", record="ATOM"):
    name, residue, number, element = atom
    name = name if len(name) == 4 else " " + name
    x, y, z = (float(value) / 1000.0 for value in xyz)
    if dialect == "cpptraj":   # 80 columns, no chain
        return "%-6s%5d %-4s %3s  %4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s  \n" % (
            record, serial, name, residue, number, x, y, z, element)
    # MDAnalysis PDBWriter: 78 columns, chain X, segment SYST
    return "%-6s%5d %-4s %3s X%4d    %8.3f%8.3f%8.3f  1.00  0.00      SYST%2s\n" % (
        record, serial, name, residue, number, x, y, z, element)


def pdb_text(coords, dialect="cpptraj", atoms=None, model=None, header=None, ter=None, frame_prefix=None,
             newline="\n", tail=None):
    """Multi-model PDB text for integer coordinates (frames, atoms, 3) in 0.001 A."""
    coords = np.asarray(coords)
    atoms = atoms or atom_table(coords.shape[1])
    if model is None:
        model = (lambda i: "MODEL     %d" % (i + 1)) if dialect == "cpptraj" else (
            lambda i: "MODEL     %4d" % ((i + 1) % 10000))
    if header is None:
        header = ("CRYST1   29.288   31.797   32.511  90.00  90.00  90.00               1\n"
                  if dialect == "cpptraj" else
                  "HEADER    \nTITLE     MDANALYSIS FRAMES FROM 0, STEP 1: Created by PDBWriter\n"
                  "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n")
    ter = (dialect == "cpptraj") if ter is None else ter
    parts = [header]
    for index, frame in enumerate(coords):
        if frame_prefix is not None:
            parts.append(frame_prefix(index))
        parts.append(model(index) + "\n")
        serial = 0
        for position, (atom, xyz) in enumerate(zip(atoms, frame)):
            serial += 1
            parts.append(atom_line(serial, atom, xyz, dialect))
            if ter and position == len(atoms) // 2:   # cpptraj puts TER between molecules too
                serial += 1
                parts.append("TER   %5d      %3s  %4d \n" % (serial, atom[1], atom[2]))
        if ter:
            parts.append("TER   %5d      %3s  %4d \n" % (serial + 1, atoms[-1][1], atoms[-1][2]))
        parts.append("ENDMDL\n")
    parts.append(("END   \n" if dialect == "cpptraj" else "END\n") if tail is None else tail)
    return "".join(parts).replace("\n", newline)


MOL2 = """@<TRIPOS>MOLECULE
chain
{count} {bonds} 0 0 0
SMALL
NO_CHARGES
@<TRIPOS>ATOM
{atom_lines}
@<TRIPOS>BOND
{bond_lines}
"""


def mol2_text(serials):
    atom_lines = "\n".join(f"{s} X{s} 0.0 0.0 0.0 C.3 1 RES 0.0" for s in serials)
    bond_lines = "\n".join(f"{i + 1} {a} {b} 1" for i, (a, b) in enumerate(zip(serials, serials[1:])))
    return MOL2.format(count=len(serials), bonds=len(serials) - 1, atom_lines=atom_lines, bond_lines=bond_lines)


def make_source(root, folder, coords=None, text=None, name_json=None, info_json=None, **options):
    """Create ``root/folder`` with simulation.pdb and structure.mol2; return its path."""
    source = Path(root) / folder
    source.mkdir(parents=True)
    if text is None:
        text = pdb_text(coords, **options)
    (source / "simulation.pdb").write_bytes(text.encode("ascii") if isinstance(text, str) else text)
    (source / "structure.mol2").write_text(mol2_text([1, 2, 3]), encoding="utf-8")
    if name_json is not None:
        (source / "name.json").write_text(json.dumps({"indexOrderedSequence": name_json}), encoding="utf-8")
    if info_json is not None:
        (source / "info.json").write_text(
            json.dumps({"status": "new", "indexOrderedSequence": info_json}, indent=2), encoding="utf-8")
    return source


def random_coords(frames, atoms, seed=0, spread=30000):
    return np.random.default_rng(seed).integers(-spread, spread + 1, size=(frames, atoms, 3), dtype=np.int64)


def snapshot(root):
    """Content, size and modification time of every file below ``root``."""
    return {
        str(path.relative_to(root)): (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in sorted(Path(root).rglob("*")) if path.is_file()
    }


def read_xtc(entry_dir):
    from mdtraj.formats import XTCTrajectoryFile

    with XTCTrajectoryFile(str(Path(entry_dir) / "trajectory.xtc")) as handle:
        return handle.read()


def run_converter(*arguments):
    return subprocess.run(
        [sys.executable, str(CONVERTER), *map(str, arguments)], capture_output=True, text=True, check=False)


# ---------------------------------------------------------------------------
# Fixed-width fields
# ---------------------------------------------------------------------------
def test_field_parser_reads_exactly_the_digits_of_the_text():
    values = np.concatenate([
        np.arange(-999_999, 10_000_000, 997),
        [-999_999, -1000, -999, -1, 0, 1, 999, 1000, 9_999_999],
    ])
    text = b"".join(b"%8.3f" % (value / 1000.0) for value in values.tolist())
    parsed, valid, negative = archive.parse_coordinate_fields(np.frombuffer(text, dtype="<u8"))
    assert valid.all()
    assert np.array_equal(parsed, values)
    assert np.array_equal(negative, values < 0)

    parsed, valid, negative = archive.parse_coordinate_fields(np.frombuffer(b"  -0.000", dtype="<u8"))
    assert valid.all() and parsed[0] == 0 and negative[0]


@pytest.mark.parametrize("field", [
    b"        ", b" 1.2345 ", b"   1.23 ", b"12345.67", b"  +1.000", b"   -.123", b"    .123", b" 012.345",
    b"  1e+003", b"     nan", b"1 2.000 ", b"--12.345", b"- 12.345", b"12.345\n ", b"\x00" * 8, b"\xff" * 8,
    b"  1,234 ", b"  12.3a5", b"  1a.345", b"********",
])
def test_field_parser_rejects_everything_that_is_not_a_placed_number(field):
    good = b"  16.372"
    _, valid, _ = archive.parse_coordinate_fields(np.frombuffer(good + field + good, dtype="<u8"))
    assert valid.tolist() == [True, False, True]


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dialect", ["cpptraj", "mdanalysis"])
@pytest.mark.parametrize("atoms", [3, 9, 10, 37])
def test_round_trip_is_exact_at_pdb_precision(tmp_path, dialect, atoms):
    coords = random_coords(60, atoms, seed=atoms)
    coords[0, 0] = (-999_999, 5_120_003, 0)       # the extremes that are still exact
    coords[1, 0] = (1, -1, 999)
    source = make_source(tmp_path / "data", "DGlcpa1-OH", coords, dialect=dialect)
    before = snapshot(source)

    result = archive.convert_entry(source, tmp_path / "archive")
    assert result["status"] == "converted"
    assert snapshot(source) == before                 # the source is only read

    entry = archive.ArchiveEntry(result["entry"])
    manifest = entry.manifest
    assert (manifest["atoms"], manifest["frames"]) == (atoms, 60)
    assert np.array_equal(entry.read_frames(unit="milliangstrom"), coords)

    # float32 angstrom values are those a PDB reader gets from the text
    text = (source / "simulation.pdb").read_text()
    from_text = np.array(
        [[float(line[30:38]), float(line[38:46]), float(line[46:54])]
         for line in text.splitlines() if line.startswith("ATOM")], dtype=np.float32).reshape(60, atoms, 3)
    angstrom = entry.read_frames()
    assert angstrom.dtype == np.float32
    assert np.array_equal(angstrom.view(np.uint32), from_text.view(np.uint32))

    # what the manifest says was measured
    verification = manifest["verification"]
    assert verification["passed"] and verification["mismatched_coordinates"] == 0
    assert verification["frames_compared"] == 60
    assert verification["coordinates_compared"] == coords.size
    assert verification["max_abs_difference_angstrom"] < 5e-4
    assert manifest["source"]["sha256"] == hashlib.sha256((source / "simulation.pdb").read_bytes()).hexdigest()
    assert manifest["source"]["size"] == (source / "simulation.pdb").stat().st_size
    assert manifest["trajectory"]["coordinates_sha256"] == hashlib.sha256(coords.astype("<i4").tobytes()).hexdigest()
    for name, recorded in manifest["files"].items():
        payload = (entry.path / name).read_bytes()
        assert (len(payload), hashlib.sha256(payload).hexdigest()) == (recorded["size"], recorded["sha256"])
    assert (entry.path / "structure.mol2").read_bytes() == (source / "structure.mol2").read_bytes()

    # the file itself: standard XTC, precision 10000, step and time are the frame index
    xyz, time, step, _ = read_xtc(entry.path)
    assert np.array_equal(step, np.arange(60)) and np.array_equal(time, np.arange(60, dtype=np.float32))
    assert np.array_equal(np.rint(xyz.astype(np.float64) * 10000).astype(np.int64), coords)
    head = (entry.path / "trajectory.xtc").read_bytes()[:60]
    assert struct.unpack(">ii", head[:8]) == (1995, atoms)
    if atoms > 9:                                    # smaller systems are stored as plain floats
        assert struct.unpack(">f", head[56:60])[0] == 10000.0

    # topology.pdb is the header and the first frame, line for line
    header, _, rest = text.partition("MODEL")
    first_frame = rest[rest.index("\n") + 1:rest.index("ENDMDL")]
    assert (entry.path / "topology.pdb").read_text() == header + first_frame + "END\n"
    assert len(entry.atom_lines()) == atoms
    assert "".join(entry.atom_lines()) == "".join(
        line for line in first_frame.splitlines(keepends=True) if line.startswith("ATOM"))

    assert archive.verify_entry(entry.path)["ok"]
    report = archive.verify_entry(entry.path, source, check_frames=-1)
    assert report["ok"], report["errors"]
    assert report["reference_frames_checked"] == 60 and report["mismatched_coordinates"] == 0


@pytest.mark.parametrize("block_bytes,batch_bytes", [(61, 1), (997, 300), (4096, 1 << 20), (1 << 20, 2000)])
def test_block_and_batch_sizes_do_not_change_what_is_read(tmp_path, block_bytes, batch_bytes):
    coords = random_coords(230, 14, seed=5)
    path = tmp_path / "simulation.pdb"
    path.write_text(pdb_text(coords))
    reader = archive.PdbTrajectory(path, block_bytes=block_bytes, batch_bytes=batch_bytes)
    blocks = [block for block, _ in reader.blocks()]
    assert np.array_equal(np.concatenate(blocks), coords)
    assert reader.n_frames == 230 and reader.n_atoms == 14
    assert reader.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert reader.line_parsed_frames == 1            # every other frame went through the block parser
    text = path.read_bytes()
    for offset, length in zip(reader.frame_offsets[[0, 9, 229]], reader.frame_lengths[[0, 9, 229]]):
        frame = text[offset:offset + length]
        assert frame.startswith(b"MODEL") and frame.endswith(b"ENDMDL\n")


def test_negative_zero_is_stored_as_zero_and_counted(tmp_path):
    coords = random_coords(5, 12, seed=1)
    coords[2, 3, 1] = 0
    text = pdb_text(coords)
    lines = text.splitlines(keepends=True)
    target = [i for i, line in enumerate(lines) if line.startswith("ATOM")][2 * 12 + 3]
    assert lines[target][38:46] == "   0.000"
    lines[target] = lines[target][:38] + "  -0.000" + lines[target][46:]
    source = make_source(tmp_path / "data", "DGlcpa1-OH", text="".join(lines))
    result = archive.convert_entry(source, tmp_path / "archive")
    assert result["manifest"]["source"]["negative_zero_coordinates"] == 1
    assert np.array_equal(archive.ArchiveEntry(result["entry"]).read_frames(unit="milliangstrom"), coords)


def test_coordinates_beyond_the_exact_range_fail_instead_of_being_rounded(tmp_path):
    coords = random_coords(3, 12, seed=2)
    coords[1, 4] = (5_120_004, 5_120_015, 9_999_999)   # come back one unit off through float32
    source = make_source(tmp_path / "data", "DGlcpa1-OH", coords)
    with pytest.raises(archive.VerificationError, match="coordinates differ from the source text.*5120 A or more"):
        archive.convert_entry(source, tmp_path / "archive")
    assert not (tmp_path / "archive" / "DGlcpa1-OH").exists()
    assert not any((tmp_path / "archive" / archive.STAGING_DIR).iterdir())


# ---------------------------------------------------------------------------
# Box
# ---------------------------------------------------------------------------
def test_box_from_cryst1_is_written_to_every_frame(tmp_path):
    coords = random_coords(20, 12)
    source = make_source(tmp_path / "data", "DGlcpa1-OH", coords)
    result = archive.convert_entry(source, tmp_path / "archive")
    box = result["manifest"]["box"]
    assert box == {"kind": "constant", "lengths_angstrom": [29.288, 31.797, 32.511], "angles_degree": [90.0, 90.0, 90.0]}
    _, _, _, vectors = read_xtc(result["entry"])
    assert vectors.shape == (20, 3, 3)
    assert np.allclose(vectors, np.diag([2.9288, 3.1797, 3.2511]), atol=1e-6)
    assert (Path(result["entry"]) / "topology.pdb").read_text().startswith("CRYST1   29.288   31.797   32.511")


def test_without_cryst1_there_is_no_box(tmp_path):
    source = make_source(tmp_path / "data", "DGlcpa1-OH", random_coords(8, 12), header="")
    result = archive.convert_entry(source, tmp_path / "archive")
    assert result["manifest"]["box"] == {"kind": "none"}
    assert read_xtc(result["entry"])[3] is None          # MDTraj reports an all-zero box as None


def test_placeholder_unit_cell_is_not_written_as_a_box(tmp_path):
    source = make_source(tmp_path / "data", "DGlcpa1-OH", random_coords(8, 12), dialect="mdanalysis")
    result = archive.convert_entry(source, tmp_path / "archive")
    assert result["manifest"]["box"]["kind"] == "placeholder"
    assert result["manifest"]["box"]["lengths_angstrom"] == [1.0, 1.0, 1.0]
    assert read_xtc(result["entry"])[3] is None
    assert "CRYST1    1.000    1.000    1.000" in (Path(result["entry"]) / "topology.pdb").read_text()


def test_triclinic_box_follows_the_gromacs_convention(tmp_path):
    from mdtraj.utils.unitcell import lengths_and_angles_to_box_vectors

    header = "CRYST1   30.125   41.250   52.375  80.00 100.50  70.25 P 1           1\n"
    source = make_source(tmp_path / "data", "DGlcpa1-OH", random_coords(6, 12), header=header)
    result = archive.convert_entry(source, tmp_path / "archive")
    assert result["manifest"]["box"]["kind"] == "constant"
    assert result["manifest"]["verification"]["box_max_abs_difference_angstrom"] < 5e-4
    assert result["manifest"]["verification"]["box_max_abs_difference_degree"] < 5e-3
    expected = np.array(lengths_and_angles_to_box_vectors(3.0125, 4.1250, 5.2375, 80.0, 100.5, 70.25))
    assert np.allclose(read_xtc(result["entry"])[3], expected[None], atol=2e-6)


def test_cryst1_before_every_frame_gives_a_box_per_frame(tmp_path):
    coords = random_coords(7, 12)

    def prefix(index):
        return "CRYST1%9.3f%9.3f%9.3f  90.00  90.00  90.00 P 1           1\n" % (30 + index, 31.5, 32.25)

    source = make_source(tmp_path / "data", "DGlcpa1-OH", coords, header="TITLE     t\n", frame_prefix=prefix)
    result = archive.convert_entry(source, tmp_path / "archive")
    box = result["manifest"]["box"]
    assert box["kind"] == "varying" and box["cells_total"] == 7
    assert [cell["first_frame"] for cell in box["cells"]] == list(range(7))
    assert box["cells"][3] == {"first_frame": 3, "lengths_angstrom": [33.0, 31.5, 32.25], "angles_degree": [90.0] * 3}
    vectors = read_xtc(result["entry"])[3]
    assert np.allclose(vectors[:, 0, 0] * 10, 30 + np.arange(7), atol=1e-5)
    assert np.allclose(vectors[:, 1, 1] * 10, 31.5, atol=1e-5)
    assert np.array_equal(archive.ArchiveEntry(result["entry"]).read_frames(unit="milliangstrom"), coords)
    assert result["manifest"]["source"]["records_between_frames"]["CRYST1"] == {
        "count": 6, "before_frame": [1, 2, 3, 4, 5, 6]}


def test_concatenated_runs_keep_their_boxes_and_boundaries(tmp_path):
    # Several runs written one after the other into one file: each has its own
    # CRYST1, restarts its MODEL numbers and ends with END.
    coords = random_coords(120, 12, seed=8)
    cells = [(43.501, 57.057, 51.057), (43.528, 55.582, 51.088), (46.597, 55.385, 51.097)]
    text = "".join(
        pdb_text(coords[40 * run:40 * run + 40],
                 header="CRYST1%9.3f%9.3f%9.3f  90.00  90.00  90.00               1\n" % cell)
        for run, cell in enumerate(cells))
    source = make_source(tmp_path / "data", "DGlcpa1-OH", text=text)
    result = archive.convert_entry(source, tmp_path / "archive")
    manifest = result["manifest"]
    assert manifest["frames"] == 120 and manifest["source"]["frames_parsed_line_by_line"] == 1
    assert manifest["box"] == {"kind": "varying", "cells_total": 3, "cells": [
        {"first_frame": 40 * run, "lengths_angstrom": list(cell), "angles_degree": [90.0, 90.0, 90.0]}
        for run, cell in enumerate(cells)]}
    assert manifest["source"]["model_numbers"] == {
        "first": 1, "last": 40, "largest": 40, "sequential": False, "breaks": 2}
    assert manifest["source"]["records_between_frames"] == {
        "CRYST1": {"count": 2, "before_frame": [40, 80]}, "END": {"count": 3, "before_frame": [40, 80, 120]}}
    vectors = read_xtc(result["entry"])[3]
    assert np.allclose(vectors[:, 0, 0] * 10, np.repeat([43.501, 43.528, 46.597], 40), atol=1e-5)
    assert np.array_equal(archive.ArchiveEntry(result["entry"]).read_frames(unit="milliangstrom"), coords)
    assert archive.verify_entry(result["entry"], source, check_frames=-1)["ok"]


# ---------------------------------------------------------------------------
# MODEL numbering and other shapes of the text
# ---------------------------------------------------------------------------
def test_model_numbers_that_grow_in_width(tmp_path):
    coords = random_coords(1005, 11, seed=3)          # MODEL 1 ... MODEL 1005: four line widths
    source = make_source(tmp_path / "data", "DGlcpa1-OH", coords)
    result = archive.convert_entry(source, tmp_path / "archive")
    manifest = result["manifest"]
    assert manifest["frames"] == 1005
    assert manifest["source"]["model_numbers"] == {
        "first": 1, "last": 1005, "largest": 1005, "sequential": True, "breaks": 0}
    assert manifest["source"]["frames_parsed_line_by_line"] == 1
    assert np.array_equal(archive.ArchiveEntry(result["entry"]).read_frames(unit="milliangstrom"), coords)


def test_model_numbers_that_wrap(tmp_path):
    # MDAnalysis writes "MODEL     %4d" of the number modulo 10000: 1 ... 9999, 0, 1, ...
    coords = random_coords(10050, 10, seed=4, spread=9000)
    source = make_source(tmp_path / "data", "DGlcpa1-OH", coords, dialect="mdanalysis")
    assert b"MODEL     9999\nATOM" in (source / "simulation.pdb").read_bytes()
    assert (source / "simulation.pdb").read_bytes().count(b"MODEL        1\n") == 2
    result = archive.convert_entry(source, tmp_path / "archive")
    manifest = result["manifest"]
    assert manifest["frames"] == 10050
    assert manifest["source"]["model_numbers"] == {
        "first": 1, "last": 50, "largest": 9999, "sequential": False, "breaks": 1}
    entry = archive.ArchiveEntry(result["entry"])
    assert np.array_equal(entry.read_frames(unit="milliangstrom"), coords)   # all frames, in source order
    assert archive.verify_entry(entry.path, source)["ok"]


@pytest.mark.parametrize("model", [lambda i: "MODEL", lambda i: "MODEL        1", lambda i: "MODEL     ****"])
def test_model_lines_without_a_usable_number(tmp_path, model):
    coords = random_coords(30, 12)
    source = make_source(tmp_path / "data", "DGlcpa1-OH", coords, model=model)
    result = archive.convert_entry(source, tmp_path / "archive")
    assert result["manifest"]["frames"] == 30
    assert result["manifest"]["source"]["model_numbers"]["sequential"] is False
    assert np.array_equal(archive.ArchiveEntry(result["entry"]).read_frames(unit="milliangstrom"), coords)


def test_records_after_the_last_frame_are_kept_in_the_topology(tmp_path):
    coords = random_coords(6, 12)
    conect = "CONECT    1    2\nCONECT    2    1    3\nMASTER        0    0    0\n"
    source = make_source(tmp_path / "data", "DGlcpa1-OH", coords, tail=conect + "END\n")
    result = archive.convert_entry(source, tmp_path / "archive")
    topology = (Path(result["entry"]) / "topology.pdb").read_text()
    assert topology.endswith("TER      14      4YB     3 \n" + conect + "END\n")
    assert result["manifest"]["source"]["records_between_frames"] == {
        "CONECT": {"count": 2, "before_frame": [6, 6]},
        "END": {"count": 1, "before_frame": [6]},
        "MASTER": {"count": 1, "before_frame": [6]},
    }
    assert result["manifest"]["warnings"] == []
    assert archive.verify_entry(result["entry"], source)["ok"]


def test_windows_line_endings(tmp_path):
    coords = random_coords(25, 12)
    source = make_source(tmp_path / "data", "DGlcpa1-OH", coords, newline="\r\n")
    result = archive.convert_entry(source, tmp_path / "archive")
    assert result["manifest"]["source"]["frames_parsed_line_by_line"] == 1
    assert np.array_equal(archive.ArchiveEntry(result["entry"]).read_frames(unit="milliangstrom"), coords)


def test_file_without_model_records_is_one_frame(tmp_path):
    coords = random_coords(1, 12)
    text = "".join(atom_line(i + 1, atom, xyz) for i, (atom, xyz) in enumerate(zip(atom_table(12), coords[0])))
    source = make_source(tmp_path / "data", "DGlcpa1-OH", text="REMARK single\n" + text + "TER\nEND\n")
    result = archive.convert_entry(source, tmp_path / "archive")
    assert (result["manifest"]["frames"], result["manifest"]["atoms"]) == (1, 12)
    assert result["manifest"]["source"]["model_numbers"] is None
    assert np.array_equal(archive.ArchiveEntry(result["entry"]).read_frames(unit="milliangstrom"), coords)
    assert (Path(result["entry"]) / "topology.pdb").read_text() == "REMARK single\n" + text + "TER\nEND\n"


def test_frame_in_another_layout_is_read_line_by_line(tmp_path):
    coords = random_coords(12, 12, seed=7)
    coords[8, 2, 0] = 12_345
    lines = pdb_text(coords, ter=False).splitlines(keepends=True)
    atoms = [i for i, line in enumerate(lines) if line.startswith("ATOM")]
    for i in atoms[4 * 12:5 * 12]:                    # frame 4: trailing blanks removed
        lines[i] = lines[i].rstrip() + "\n"
    target = atoms[8 * 12 + 2]                        # frame 8: a leading zero, still a valid number
    assert lines[target][30:38] == "  12.345"
    lines[target] = lines[target][:30] + " 012.345" + lines[target][38:]
    source = make_source(tmp_path / "data", "DGlcpa1-OH", text="".join(lines))
    result = archive.convert_entry(source, tmp_path / "archive")
    assert np.array_equal(archive.ArchiveEntry(result["entry"]).read_frames(unit="milliangstrom"), coords)
    # Line by line: frame 0, frame 4 (other layout), frame 5 (back to the first
    # layout) and frame 8 (a field the block parser does not accept).
    assert result["manifest"]["source"]["frames_parsed_line_by_line"] == 4
    assert archive.verify_entry(result["entry"], source, check_frames=-1)["ok"]


# ---------------------------------------------------------------------------
# Sources that must fail cleanly
# ---------------------------------------------------------------------------
def _drop_atom_line(text, frame, atoms, which=0):
    lines = text.splitlines(keepends=True)
    del lines[[i for i, line in enumerate(lines) if line.startswith("ATOM")][frame * atoms + which]]
    return "".join(lines)


@pytest.mark.parametrize("frame", [1, 37, 79])
def test_frame_with_another_atom_count_fails_cleanly(tmp_path, frame):
    text = _drop_atom_line(pdb_text(random_coords(80, 12), ter=False), frame, 12)
    source = make_source(tmp_path / "data", "DGlcpa1-OH", text=text)
    before = snapshot(tmp_path / "data")
    with pytest.raises(archive.AtomCountError, match=rf"frame {frame} has 11 atoms, the first frame has 12"):
        archive.convert_entry(source, tmp_path / "archive", batch_bytes=4000)
    assert snapshot(tmp_path / "data") == before
    assert [p.name for p in (tmp_path / "archive").iterdir()] == [archive.STAGING_DIR]
    assert not any((tmp_path / "archive" / archive.STAGING_DIR).iterdir())


def test_frame_with_an_extra_atom_fails_cleanly(tmp_path):
    lines = pdb_text(random_coords(9, 12), ter=False).splitlines(keepends=True)
    position = [i for i, line in enumerate(lines) if line.startswith("ATOM")][6 * 12]
    lines.insert(position, lines[position])
    source = make_source(tmp_path / "data", "DGlcpa1-OH", text="".join(lines))
    with pytest.raises(archive.AtomCountError, match="frame 6 has 13 atoms"):
        archive.convert_entry(source, tmp_path / "archive")


def test_frame_with_other_atom_records_fails_cleanly(tmp_path):
    lines = pdb_text(random_coords(9, 12)).splitlines(keepends=True)
    position = [i for i, line in enumerate(lines) if line.startswith("ATOM")][3 * 12 + 5]
    lines[position] = lines[position][:17] + "XYZ" + lines[position][20:]
    source = make_source(tmp_path / "data", "DGlcpa1-OH", text="".join(lines))
    with pytest.raises(archive.TopologyChangeError, match="frame 3, atom 5"):
        archive.convert_entry(source, tmp_path / "archive")


def test_frame_with_other_occupancy_or_element_columns_fails_cleanly(tmp_path):
    lines = pdb_text(random_coords(9, 12)).splitlines(keepends=True)
    position = [i for i, line in enumerate(lines) if line.startswith("ATOM")][7 * 12 + 1]
    assert lines[position][54:60] == "  1.00"
    lines[position] = lines[position][:54] + "  0.50" + lines[position][60:]
    source = make_source(tmp_path / "data", "DGlcpa1-OH", text="".join(lines))
    with pytest.raises(archive.TopologyChangeError, match="frame 7, atom 1: columns 55-80"):
        archive.convert_entry(source, tmp_path / "archive")


@pytest.mark.parametrize("damage,message", [
    (lambda text: text[:text.rindex("ATOM")], "not terminated by ENDMDL"),
    (lambda text: text.replace("ENDMDL\nMODEL     4\n", "MODEL     4\n", 1), "before the ENDMDL of this frame"),
    (lambda text: text.replace("ENDMDL\n", "ENDMDL\nENDMDL\n", 1), "ENDMDL without MODEL"),
    (lambda text: text.replace("ENDMDL\n", "ENDMDL\n" + text.splitlines()[2] + "\n", 1), "ATOM record outside MODEL/ENDMDL"),
    (lambda text: text.replace("  16.372", "  16.37 ", 1), "not a number with three decimals"),
    (lambda text: text.replace("  16.372", "*" * 8, 1), "not a number with three decimals"),
    (lambda text: "REMARK nothing here\nEND\n", "no ATOM/HETATM records"),
])
def test_malformed_text_fails_with_the_reason(tmp_path, damage, message):
    coords = random_coords(8, 12)
    coords[5, 5, 1] = 16_372
    source = make_source(tmp_path / "data", "DGlcpa1-OH", text=damage(pdb_text(coords)))
    with pytest.raises(archive.PdbFormatError, match=message):
        archive.convert_entry(source, tmp_path / "archive")
    assert not (tmp_path / "archive" / "DGlcpa1-OH").exists()


def test_source_without_mol2_or_trajectory_fails(tmp_path):
    source = make_source(tmp_path / "data", "DGlcpa1-OH", random_coords(2, 12))
    (source / "structure.mol2").unlink()
    with pytest.raises(archive.SourceError, match="missing structure.mol2"):
        archive.convert_entry(source, tmp_path / "archive")
    (source / "simulation.pdb").unlink()
    with pytest.raises(archive.SourceError, match="missing simulation.pdb"):
        archive.convert_entry(source, tmp_path / "archive")


def test_archive_may_not_be_written_into_the_source(tmp_path):
    source = make_source(tmp_path / "data", "DGlcpa1-OH", random_coords(2, 12))
    before = snapshot(tmp_path / "data")
    for target in (tmp_path / "data", source, source / "archive"):
        with pytest.raises(archive.SourceError, match="outside the source data directory"):
            archive.convert_entry(source, target)
    assert snapshot(tmp_path / "data") == before


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------
def test_folder_is_the_glycam_name_when_it_fits():
    assert archive.entry_folder_name("DAllpb1-OH") == "DAllpb1-OH"
    assert archive.entry_folder_name("DGlcpAb1-4DGlcpNSa1-4DGlcpA[2S]b1-4DGlcpNS[6S]a1-OH") == (
        "DGlcpAb1-4DGlcpNSa1-4DGlcpA[2S]b1-4DGlcpNS[6S]a1-OH")
    assert archive.entry_folder_name(NAME_G77147GI) == NAME_G77147GI
    longest = "D" * 252 + "-OH"
    assert len(longest) == 255 and archive.entry_folder_name(longest) == longest


def test_long_name_gets_a_deterministic_shortened_folder(tmp_path):
    assert (len(NAME_G50508SG), len(NAME_G60371DN)) == (257, 267)
    for name in (NAME_G50508SG, NAME_G60371DN, "D" * 253 + "-OH"):
        folder = archive.entry_folder_name(name)
        assert folder == name[:200] + "~" + hashlib.sha256(name.encode()).hexdigest()[:16]
        assert len(folder) == 217
        assert archive.entry_folder_name(name) == folder
        (tmp_path / folder).mkdir()                    # a real directory name
    # names that share their first 200 characters still get different folders
    assert archive.entry_folder_name(NAME_G50508SG) != archive.entry_folder_name(NAME_G50508SG[:-4] + "b1-OH")
    assert archive.entry_folder_name("é" * 150) == "é" * 100 + "~" + hashlib.sha256(("é" * 150).encode()).hexdigest()[:16]


def test_names_that_cannot_be_a_directory_are_shortened_too():
    assert archive.entry_folder_name("DGalpb1-3/4DGlcpa1-OH") == (
        "DGalpb1-3_4DGlcpa1-OH~" + hashlib.sha256(b"DGalpb1-3/4DGlcpa1-OH").hexdigest()[:16])
    assert archive.entry_folder_name(".hidden").startswith("hidden~")
    lookalike = "DGlcpa1-OH~0123456789abcdef"            # must not pass for a shortened folder
    assert archive.entry_folder_name(lookalike) == lookalike + "~" + hashlib.sha256(lookalike.encode()).hexdigest()[:16]
    for bad in ("", " DGlcpa1-OH", "a\nb", "a//b", "../x", "a/./b"):
        with pytest.raises(archive.NamingError):
            archive.entry_folder_name(bad)


def test_object_key_prefix_is_raw_glycam_name():
    assert archive.object_key_prefix("DAllpb1-OH") == "raw/DAllpb1-OH/"
    assert archive.object_key_prefix(NAME_G60371DN) == "raw/" + NAME_G60371DN + "/"
    with pytest.raises(archive.NamingError):
        archive.object_key_prefix("")
    with pytest.raises(archive.NamingError, match="exceeds 1024 bytes"):
        archive.object_key_prefix("D" * 1010)


@pytest.mark.parametrize("folder,name_json,info_json,expected,origin", [
    ("DAllpb1-OH", None, "DAllpb1-OH", "DAllpb1-OH", "folder"),
    ("DAllpb1-OH", None, None, "DAllpb1-OH", "folder"),
    ("G50508SG", NAME_G50508SG, NAME_G50508SG, NAME_G50508SG, "name.json"),
    ("G60371DN", None, NAME_G60371DN, NAME_G60371DN, "info.json"),
    ("G77147GI", NAME_G77147GI, NAME_G77147GI, NAME_G77147GI, "name.json"),
])
def test_glycam_name_of_a_source_folder(tmp_path, folder, name_json, info_json, expected, origin):
    source = make_source(tmp_path / "data", folder, random_coords(4, 12), name_json=name_json, info_json=info_json)
    assert archive.resolve_glycam_name(source) == (expected, origin, [])

    result = archive.convert_entry(source, tmp_path / "archive")
    manifest = result["manifest"]
    assert manifest["glycam_name"] == expected
    assert manifest["folder"] == archive.entry_folder_name(expected) == Path(result["entry"]).name
    assert manifest["object_key_prefix"] == "raw/" + expected + "/"
    assert manifest["source"]["folder"] == folder and manifest["source"]["glycam_name_from"] == origin
    entry = archive.ArchiveEntry(result["entry"])
    assert entry.glycam_name == expected
    assert entry.object_keys()["trajectory.xtc"] == f"raw/{expected}/trajectory.xtc"
    assert set(entry.object_keys()) == set(manifest["files"]) | {"manifest.json"}
    for name, present in (("name.json", name_json), ("info.json", info_json)):   # copied unchanged
        assert (entry.path / name).exists() == (present is not None)
        if present is not None:
            assert (entry.path / name).read_bytes() == (source / name).read_bytes()
    assert archive.verify_entry(entry.path, source)["ok"]


def test_name_json_is_authoritative_and_differences_are_reported(tmp_path):
    source = make_source(tmp_path / "data", "DGlcpa1-OH", random_coords(2, 12),
                         name_json="DGlcpb1-OH", info_json="DGlcpa1-OME")
    name, origin, warnings = archive.resolve_glycam_name(source)
    assert (name, origin) == ("DGlcpb1-OH", "name.json")
    assert len(warnings) == 1 and "DGlcpa1-OME" in warnings[0]
    assert archive.convert_entry(source, tmp_path / "archive")["manifest"]["warnings"] == warnings


def test_glytoucan_folder_without_a_name_fails(tmp_path):
    source = make_source(tmp_path / "data", "G12345AB", random_coords(2, 12))
    with pytest.raises(archive.NamingError, match="named by a GlyTouCan ID"):
        archive.convert_entry(source, tmp_path / "archive")
    (source / "name.json").write_text("{}")
    with pytest.raises(archive.NamingError, match="name.json gives no GLYCAM name"):
        archive.convert_entry(source, tmp_path / "archive")


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------
def test_reader_streams_frames_in_chunks(tmp_path):
    coords = random_coords(103, 12, seed=9)
    source = make_source(tmp_path / "data", "DGlcpa1-OH", coords)
    entry = archive.ArchiveEntry(archive.convert_entry(source, tmp_path / "archive")["entry"])
    assert (entry.n_frames, entry.n_atoms) == (103, 12)

    chunks = list(entry.iter_frames(chunk=10, unit="milliangstrom"))
    assert [len(chunk) for chunk in chunks] == [10] * 10 + [3]
    assert np.array_equal(np.concatenate(chunks), coords)
    assert all(chunk.dtype == np.int32 for chunk in chunks)

    chunks = list(entry.iter_frames(chunk=1000))
    assert [chunk.shape for chunk in chunks] == [(103, 12, 3)] and chunks[0].dtype == np.float32
    assert np.array_equal(chunks[0], (coords / 1000.0).astype(np.float32))

    part = list(entry.iter_frames(chunk=7, start=40, stop=61, unit="milliangstrom"))
    assert [len(chunk) for chunk in part] == [7, 7, 7]
    assert np.array_equal(np.concatenate(part), coords[40:61])
    assert np.array_equal(entry.read_frames(98, unit="milliangstrom"), coords[98:])
    assert entry.read_frames(50, 50).shape == (0, 12, 3)
    nanometre = entry.read_frames(0, 5, unit="nm")
    assert np.allclose(nanometre * 10, coords[:5] / 1000.0, atol=1e-5)
    with pytest.raises(ValueError):
        list(entry.iter_frames(unit="bohr"))
    with pytest.raises(ValueError):
        list(entry.iter_frames(start=200))

    # entry_inputs() is what the processing step uses to recognise an entry
    assert archive.entry_inputs(entry.path) == {
        "glycam_name": "DGlcpa1-OH",
        "trajectory_path": str(entry.path / "trajectory.xtc"),
        "mol2_path": str(entry.path / "structure.mol2"),
    }
    assert archive.entry_inputs(source) is None and archive.entry_inputs(tmp_path / "missing") is None
    assert archive.ArchiveEntry(entry.path / "trajectory.xtc").path == entry.path


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------
def test_verifier_notices_changed_files(tmp_path):
    source = make_source(tmp_path / "data", "DGlcpa1-OH", random_coords(40, 12))
    entry = Path(archive.convert_entry(source, tmp_path / "archive")["entry"])
    assert archive.verify_entry(entry)["ok"] and archive.verify_entry(entry, deep=False)["ok"]

    payload = bytearray((entry / "trajectory.xtc").read_bytes())
    payload[len(payload) // 2] ^= 0x01
    (entry / "trajectory.xtc").write_bytes(payload)
    report = archive.verify_entry(entry, deep=False)
    assert not report["ok"] and "trajectory.xtc does not match" in report["errors"][0]

    shutil.rmtree(entry)
    entry = Path(archive.convert_entry(source, tmp_path / "archive")["entry"])
    (entry / "structure.mol2").unlink()
    assert archive.verify_entry(entry)["errors"] == ["missing structure.mol2"]

    renamed = entry.with_name("other")
    entry.rename(renamed)
    assert any("entry folder is other" in error for error in archive.verify_entry(renamed)["errors"])
    assert not archive.verify_entry(tmp_path / "nothing")["ok"]


def test_verifier_decodes_the_trajectory_not_only_its_hash(tmp_path):
    coords = random_coords(40, 12)
    source = make_source(tmp_path / "data", "DGlcpa1-OH", coords)
    entry = Path(archive.convert_entry(source, tmp_path / "archive")["entry"])

    # replace the trajectory by one that is a frame short and make the manifest agree on its hash
    other = make_source(tmp_path / "other", "DGlcpa1-OH", coords[:-1])
    shutil.copyfile(Path(archive.convert_entry(other, tmp_path / "archive2")["entry"]) / "trajectory.xtc",
                    entry / "trajectory.xtc")
    manifest = json.loads((entry / "manifest.json").read_text())
    payload = (entry / "trajectory.xtc").read_bytes()
    manifest["files"]["trajectory.xtc"] = {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    (entry / "manifest.json").write_text(json.dumps(manifest))
    assert archive.verify_entry(entry, deep=False)["ok"]
    errors = archive.verify_entry(entry)["errors"]
    assert any("holds 39 frames, the manifest says 40" in error for error in errors)
    assert any("do not match the digest" in error for error in errors)


def test_verifier_compares_with_the_source_text(tmp_path, monkeypatch):
    coords = random_coords(40, 12)
    coords[17, 3, 2] = 16_372
    source = make_source(tmp_path / "data", "DGlcpa1-OH", coords)
    entry = Path(archive.convert_entry(source, tmp_path / "archive")["entry"])
    assert archive.verify_entry(entry, source)["ok"]

    text = (source / "simulation.pdb").read_text()
    assert text.count("  16.372") == 1
    (source / "simulation.pdb").write_text(text.replace("  16.372", "  16.373"))
    errors = archive.verify_entry(entry, source)["errors"]
    assert any("does not match the size and SHA-256" in error for error in errors)
    assert any("1 of 1440 coordinates differ from the source text (first at frame/atom (17, 3))" in error
               for error in errors)

    renamed = make_source(tmp_path / "data", "G22222BB", coords, name_json="DGlcpb1-OH")
    errors = archive.verify_entry(entry, renamed)["errors"]
    assert errors == ["the source folder has GLYCAM name DGlcpb1-OH, the manifest says DGlcpa1-OH"]
    monkeypatch.chdir(entry)
    assert archive.verify_entry(".")["ok"]


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------
def test_second_run_skips_entries_whose_manifest_verifies(tmp_path):
    coords = random_coords(40, 12)
    source = make_source(tmp_path / "data", "DGlcpa1-OH", coords)
    target = tmp_path / "archive"
    first = archive.convert_entry(source, target)
    entry = Path(first["entry"])
    before = snapshot(entry)

    second = archive.convert_entry(source, target)
    assert second["status"] == "skipped" and second["manifest"] == first["manifest"]
    assert snapshot(entry) == before

    # a damaged entry is converted again
    with open(entry / "trajectory.xtc", "r+b") as handle:
        handle.truncate(1000)
    third = archive.convert_entry(source, target)
    assert third["status"] == "converted" and archive.verify_entry(entry, source)["ok"]

    # a changed source is converted again
    (source / "simulation.pdb").write_text(pdb_text(np.concatenate([coords, coords[:1]])))
    fourth = archive.convert_entry(source, target)
    assert fourth["status"] == "converted" and fourth["manifest"]["frames"] == 41

    # changed metadata too
    (source / "info.json").write_text(json.dumps({"indexOrderedSequence": "DGlcpa1-OH"}))
    fifth = archive.convert_entry(source, target)
    assert fifth["status"] == "converted" and "info.json" in fifth["manifest"]["files"]
    assert archive.convert_entry(source, target)["status"] == "skipped"
    assert archive.convert_entry(source, target, force=True)["status"] == "converted"


def test_leftovers_of_an_interrupted_run_are_replaced(tmp_path):
    source = make_source(tmp_path / "data", "DGlcpa1-OH", random_coords(10, 12))
    target = tmp_path / "archive"
    stale = target / archive.STAGING_DIR / hashlib.sha256(b"DGlcpa1-OH").hexdigest()[:32]
    stale.mkdir(parents=True)
    (stale / "trajectory.xtc").write_bytes(b"half written")
    result = archive.convert_entry(source, target)
    assert result["status"] == "converted" and not stale.exists()
    assert archive.verify_entry(result["entry"], source)["ok"]


def test_foreign_folder_at_the_target_is_left_alone(tmp_path):
    source = make_source(tmp_path / "data", "DGlcpa1-OH", random_coords(10, 12))
    foreign = tmp_path / "archive" / "DGlcpa1-OH"
    foreign.mkdir(parents=True)
    (foreign / "notes.txt").write_text("not an archive entry")
    with pytest.raises(archive.ArchiveError, match="is not an archive entry"):
        archive.convert_entry(source, tmp_path / "archive")
    assert (foreign / "notes.txt").read_text() == "not an archive entry"


def test_entry_of_another_source_folder_is_not_replaced_silently(tmp_path):
    first = make_source(tmp_path / "data", "DGlcpa1-OH", random_coords(10, 12, seed=1))
    second = make_source(tmp_path / "data", "G11111AA", random_coords(12, 12, seed=2), name_json="DGlcpa1-OH")
    target = tmp_path / "archive"
    entry = Path(archive.convert_entry(first, target)["entry"])
    before = snapshot(entry)
    with pytest.raises(archive.NamingError, match="converted from source folder DGlcpa1-OH"):
        archive.convert_entry(second, target)
    assert snapshot(entry) == before
    replaced = archive.convert_entry(second, target, force=True)
    assert replaced["manifest"]["source"]["folder"] == "G11111AA" and replaced["manifest"]["frames"] == 12


# ---------------------------------------------------------------------------
# Converter command line
# ---------------------------------------------------------------------------
def test_converter_cli_converts_reports_and_resumes(tmp_path):
    data, target = tmp_path / "data", tmp_path / "archive"
    good = {
        "DGlcpa1-OH": random_coords(30, 12, seed=1),
        "DManpa1-2DManpa1-OH": random_coords(45, 23, seed=2),
        "G50508SG": random_coords(20, 14, seed=3),
    }
    make_source(data, "DGlcpa1-OH", good["DGlcpa1-OH"], info_json="DGlcpa1-OH")
    make_source(data, "DManpa1-2DManpa1-OH", good["DManpa1-2DManpa1-OH"], dialect="mdanalysis")
    make_source(data, "G50508SG", good["G50508SG"], name_json=NAME_G50508SG, info_json=NAME_G50508SG)
    make_source(data, "DGalpb1-OH", text=_drop_atom_line(pdb_text(random_coords(15, 12), ter=False), 9, 12))
    make_source(data, "G00000ZZ", random_coords(3, 12))                 # no GLYCAM name anywhere
    (data / "notes.txt").write_text("not a glycan folder")
    (data / "empty").mkdir()
    before = snapshot(data)

    run = run_converter("--data-dir", data, "--archive-dir", target, "--workers", 2)
    assert run.returncode == 1, run.stderr
    assert snapshot(data) == before
    long_folder = archive.entry_folder_name(NAME_G50508SG)
    assert sorted(p.name for p in target.iterdir()) == sorted(
        ["DGlcpa1-OH", "DManpa1-2DManpa1-OH", long_folder, "index.json", "conversion_report.json"])

    index = json.loads((target / "index.json").read_text())
    assert index["entries"] == 3 and [r["glycam_name"] for r in index["records"]] == sorted(
        ["DGlcpa1-OH", "DManpa1-2DManpa1-OH", NAME_G50508SG])
    for record in index["records"]:
        manifest = json.loads((target / record["folder"] / "manifest.json").read_text())
        coords = good[record["source_folder"]]
        assert (record["frames"], record["atoms"]) == coords.shape[:2]
        assert record["object_key_prefix"] == "raw/" + record["glycam_name"] + "/"
        assert record["source_size"] == (data / record["source_folder"] / "simulation.pdb").stat().st_size
        assert record["source_sha256"] == manifest["source"]["sha256"]
        assert set(record["files"]) == set(manifest["files"]) | {"manifest.json"}
        for name, recorded in record["files"].items():
            payload = (target / record["folder"] / name).read_bytes()
            assert (len(payload), hashlib.sha256(payload).hexdigest()) == (recorded["size"], recorded["sha256"])
        assert record["archive_size"] == sum(item["size"] for item in record["files"].values())
        assert np.array_equal(archive.ArchiveEntry(target / record["folder"]).read_frames(unit="milliangstrom"), coords)
    assert index["totals"]["frames"] == 95

    report = json.loads((target / "conversion_report.json").read_text())
    assert report["counts"] == {"converted": 3, "skipped": 0, "failed": 3}
    reasons = {row["source_folder"]: row["reason"] for row in report["failed"]}
    assert reasons["DGalpb1-OH"] == "frame 9 has 11 atoms, the first frame has 12"
    assert "named by a GlyTouCan ID" in reasons["G00000ZZ"]
    assert reasons["empty"] == "missing simulation.pdb"
    assert "FAILED DGalpb1-OH: frame 9 has 11 atoms" in run.stderr
    assert all(row["mb_per_cpu_second"] > 0 for row in report["converted"])

    # second run: the three good entries are skipped, the failures are tried again
    entries = snapshot(target / "DGlcpa1-OH")
    run = run_converter("--data-dir", data, "--archive-dir", target, "--workers", 2)
    report = json.loads((target / "conversion_report.json").read_text())
    assert run.returncode == 1 and report["counts"] == {"converted": 0, "skipped": 3, "failed": 3}
    assert snapshot(target / "DGlcpa1-OH") == entries
    assert json.loads((target / "index.json").read_text())["entries"] == 3

    # repair one source and convert only that one; the index still covers the whole archive
    (data / "DGalpb1-OH" / "simulation.pdb").write_text(pdb_text(random_coords(15, 12)))
    run = run_converter("--data-dir", data, "--archive-dir", target, "--glycan", "DGalpb1-OH")
    assert run.returncode == 0, run.stderr
    assert json.loads((target / "conversion_report.json").read_text())["counts"] == {
        "converted": 1, "skipped": 0, "failed": 0}
    assert json.loads((target / "index.json").read_text())["entries"] == 4

    # verification of the whole archive against the source
    run = run_converter("--verify-only", "--data-dir", data, "--archive-dir", target, "--check-frames", -1)
    assert run.returncode == 0, run.stdout + run.stderr
    summary = json.loads((target / "verification_report.json").read_text())
    assert (summary["entries"], summary["ok"], summary["failed"]) == (4, 4, [])
    assert all(r["against_source"] and r["mismatched_coordinates"] == 0 for r in summary["reports"])

    assert run_converter("--data-dir", data, "--archive-dir", data / "archive").returncode != 0
    assert snapshot(data) == {**before, "DGalpb1-OH/simulation.pdb": snapshot(data)["DGalpb1-OH/simulation.pdb"]}


def test_converter_cli_refuses_two_sources_with_one_glycam_name(tmp_path):
    data, target = tmp_path / "data", tmp_path / "archive"
    make_source(data, "DGlcpa1-OH", random_coords(5, 12))
    make_source(data, "G11111AA", random_coords(5, 12), name_json="DGlcpa1-OH")
    make_source(data, "DGalpa1-OH", random_coords(5, 12))
    run = run_converter("--data-dir", data, "--archive-dir", target)
    report = json.loads((target / "conversion_report.json").read_text())
    assert run.returncode == 1 and report["counts"] == {"converted": 1, "skipped": 0, "failed": 2}
    assert {row["source_folder"] for row in report["failed"]} == {"DGlcpa1-OH", "G11111AA"}
    assert all("have the same GLYCAM name" in row["reason"] for row in report["failed"])
    assert not (target / "DGlcpa1-OH").exists()
