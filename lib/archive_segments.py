"""Archive entries made of several trajectories: the replicas of a run, and an entry that is extended.

``lib/archive.py`` turns one ``simulation.pdb`` into an entry.  A simulation
run leaves one multi-model PDB per replica, and a glycan that is already in
the database can be given more trajectories.  :func:`convert_segments` writes
one entry from

* one or more **sources**, each a multi-model PDB that is read, written and
  verified against its own text exactly as ``convert_entry`` does it, and
* optionally an **existing entry** (``extend``), whose frames come first.

Nothing is concatenated as text.  Every source is converted to a part of the
trajectory with the step numbers it has in the whole; the parts are then
appended byte for byte, which is all an XTC needs (it has no header and its
frames do not refer to each other).  An extended entry starts with the bytes
of the trajectory it extends: those frames are not decoded and encoded again.

The entry is read by everything that reads an entry.  Its manifest has every
key a manifest has always had and, in addition::

    "segments_version": 1,
    "segments": [                     the trajectories, in the order of the frames
      {
        "index": 0,
        "name": "replica_1",          the name the caller gave the source
        "first_frame": 0,             its frames are first_frame .. first_frame + frames of the entry
        "frames": 10001,
        "coordinates_sha256": "...",  of these frames (int32 little-endian, 0.001 A, as for the whole)
        "trajectory_offset": 0,       its bytes in trajectory.xtc
        "trajectory_bytes": 17551234,
        "time_per_frame_ps": 50.0,    null when the caller did not know it
        "run": {...},                 what the caller recorded about the simulation, verbatim; optional
                                      (lib/entry_make.py: run_id, md_package, force_field, temperature_k,
                                      pressure_bar, salt_concentration_mm, production_mdp)
        "box": {...},                 unit cell of this source, frames counted from first_frame
        "atom_records": "identical",  or what differs from topology.pdb outside names, residues and order
        "source": {"file", "size", "sha256", "model_numbers", "records_between_frames",
                   "negative_zero_coordinates", "frames_parsed_line_by_line"},
        "source_irregularities": {},  only for a damaged source that was archived with repair
        "verification": {...},        what the comparison with its text measured
        "carried_from": {...}         only for frames taken over from the entry that was extended
      }, ...
    ],
    "extends": {                      only when an existing entry was extended
      "glycam_name", "manifest_sha256", "frames", "atoms", "trajectory": {"size", "sha256",
      "coordinates_sha256"}, "files", "source", "converter", "repaired", "previous"
    }

``source`` of such a manifest describes all sources together: ``folder`` is
the name of the source (by default the folder name of the entry), ``file`` is
null, ``size`` the bytes of all source texts and ``sha256`` the SHA-256 of the
SHA-256 values of the source files in segment order (hexadecimal, joined by
line feeds).

Rules:

* **Same atoms.**  Every source has to have the atoms of the first one (of
  ``topology.pdb`` of the extended entry): the same number, and for every atom
  the same name, residue name and residue number, in the same order.  Anything
  else is an error that names the first atom that differs.  A difference in
  the other columns of the atom records (serial number, chain, occupancy,
  element) is not one: it is recorded as ``atom_records`` of the segment and
  as a warning.
* **A source without CRYST1 has no box**; the cell of the source before it is
  not carried over (in one concatenated file it would be).
* **A repaired source** keeps its ``source_irregularities`` in its segment,
  with the frame numbers of that source.  ``archive.repair_summary`` adds the
  segments up, so ``index.json`` lists such an entry under ``repaired`` like
  any other.  An extended entry that was repaired becomes a segment with the
  record it had.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import platform
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from lib import archive
from lib.archive import ArchiveError, NamingError, PathLike, SourceError, VerificationError

#: ``source.glycam_name_from`` of an entry whose name was given by the caller.
NAME_FROM_OPTION = "option"
PARTS_DIR = "parts"
_COPY_BLOCK = 4 << 20
#: The first frame of a source is looked at before anything is converted, up to this many bytes.
_FIRST_FRAME_BYTES = 64 << 20


class SegmentMismatchError(ArchiveError):
    """A source does not have the atoms of the trajectory it is to be part of."""


@dataclass(frozen=True)
class SegmentSource:
    """One trajectory of an entry: a multi-model PDB and what is known about it."""

    name: str
    pdb: Path
    mol2: Optional[Path] = None
    time_per_frame_ps: Optional[float] = None
    #: Recorded verbatim as ``run`` of the segment (JSON values only).
    run: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Atom records and MOL2
# ---------------------------------------------------------------------------
_OTHER_COLUMNS = (
    ("record name (columns 1-6)", 0, 6),
    ("serial number (columns 7-11)", 6, 11),
    ("alternate location (column 17)", 16, 17),
    ("chain (column 22)", 21, 22),
    ("columns 28-30", 27, 30),
    ("columns 55-80 (occupancy, B factor, segment, element, charge)", 30, None),
)


def atom_identity(line: bytes) -> bytes:
    """The part of an atom record that every frame repeats: columns 1-30 and 55-80."""
    line = line.rstrip(b"\r\n")
    return line[:30] + line[54:80].rstrip()


def _atom_fields(identity: bytes) -> Tuple[str, str, str]:
    """Name, residue name and residue number (with insertion code) of an atom record."""
    text = identity.decode("latin-1")
    return text[12:16].strip(), text[17:21].strip(), text[22:27].strip()


def _describe_atom(identity: bytes) -> str:
    name, residue, number = _atom_fields(identity)
    return f"name {name!r} residue {residue!r} {number}"


def compare_atom_records(
    reference: Sequence[bytes], found: Sequence[bytes], reference_name: str, found_name: str
) -> Any:
    """Compare the atom records of a source with those of the trajectory it joins.

    Raises :class:`SegmentMismatchError` when the atoms are not the same
    (number, or name, residue name and residue number of any atom, in
    order).  Returns ``"identical"`` or, when only other columns differ,
    ``{"differ_in": [...], "atoms": n}``.
    """
    if len(found) != len(reference):
        raise SegmentMismatchError(
            f"{found_name} has {len(found)} atoms, {reference_name} has {len(reference)}"
        )
    different = [
        index for index, (a, b) in enumerate(zip(found, reference)) if a != b and _atom_fields(a) != _atom_fields(b)
    ]
    if different:
        first = different[0]
        raise SegmentMismatchError(
            f"{found_name} does not have the atoms of {reference_name}: {len(different)} of {len(reference)} atoms "
            f"differ in name, residue or order; the first is atom {first + 1}: {_describe_atom(found[first])} in "
            f"{found_name} against {_describe_atom(reference[first])} in {reference_name}"
        )
    other = [index for index, (a, b) in enumerate(zip(found, reference)) if a != b]
    if not other:
        return "identical"
    columns = [
        label for label, start, stop in _OTHER_COLUMNS
        if any(found[index][start:stop] != reference[index][start:stop] for index in other)
    ]
    return {"differ_in": columns, "atoms": len(other)}


def first_frame_identity(path: PathLike, limit: int = _FIRST_FRAME_BYTES) -> Optional[List[bytes]]:
    """Atom records of the first frame of a PDB, read from the beginning of the file only.

    ``None`` when the first frame does not end within ``limit`` bytes; the
    conversion then decides.  Used to refuse sources with other atoms before
    any of them is converted.
    """
    data = b""
    with open(path, "rb") as handle:
        while len(data) < limit:
            block = handle.read(1 << 20)
            if not block:
                break
            data += block
            end = data.find(b"\nENDMDL")
            if end >= 0:
                data = data[: end + 1]
                break
        else:
            return None
    return [atom_identity(line) for line in data.split(b"\n") if line[:6] in (b"ATOM  ", b"HETATM")]


def read_mol2(path: PathLike) -> Dict[str, Any]:
    """Atoms (name, type, substructure) and bonds (pairs of atom positions) of a MOL2 file."""
    path = Path(path)
    atoms: List[Tuple[str, str, str, str]] = []
    ids: Dict[str, int] = {}
    bonds = set()
    section = ""
    try:
        with open(path, "r", encoding="latin-1") as handle:
            for line in handle:
                if line.startswith("@<TRIPOS>"):
                    section = line.strip()[len("@<TRIPOS>"):].upper()
                    continue
                fields = line.split()
                if section == "ATOM" and len(fields) >= 6:
                    ids[fields[0]] = len(atoms)
                    atoms.append((fields[1], fields[5], fields[6] if len(fields) > 6 else "",
                                  fields[7] if len(fields) > 7 else ""))
                elif section == "BOND" and len(fields) >= 3:
                    first, second = ids[fields[1]], ids[fields[2]]
                    bonds.add((min(first, second), max(first, second)))
    except OSError as exc:
        raise SourceError(f"cannot read {path}: {exc}") from exc
    except KeyError as exc:
        raise SourceError(f"{path.name}: a bond names atom {exc}, which the file does not have") from exc
    if not atoms:
        raise SourceError(f"{path.name} has no @<TRIPOS>ATOM records")
    return {"atoms": atoms, "bonds": bonds}


def compare_mol2(
    reference: Dict[str, Any], found: Dict[str, Any], reference_name: str, found_name: str, same_run: bool
) -> List[str]:
    """Compare two MOL2 files; raises for other atoms, returns warnings.

    Atom names have to be the same, in order.  Between the replicas of one
    run (``same_run``) types, substructures and bonds have to be the same
    too; against the structure of an entry that is extended they only give a
    warning, because that file may come from other tools.
    """
    warnings: List[str] = []
    a, b = found["atoms"], reference["atoms"]
    if len(a) != len(b):
        raise SegmentMismatchError(f"{found_name} has {len(a)} atoms, {reference_name} has {len(b)}")
    names = [index for index, (x, y) in enumerate(zip(a, b)) if x[0] != y[0]]
    if names:
        first = names[0]
        raise SegmentMismatchError(
            f"{found_name} does not have the atoms of {reference_name}: {len(names)} of {len(b)} atom names "
            f"differ; the first is atom {first + 1}: {a[first][0]!r} against {b[first][0]!r}"
        )
    details = []
    other = sum(1 for x, y in zip(a, b) if x != y)
    if other:
        details.append(f"{other} atoms differ in type or substructure")
    if found["bonds"] != reference["bonds"]:
        only_found = sorted(found["bonds"] - reference["bonds"])
        only_reference = sorted(reference["bonds"] - found["bonds"])
        example = (only_found or only_reference)[0]
        details.append(
            f"{len(only_found)} bonds only in {found_name}, {len(only_reference)} only in {reference_name} "
            f"(for example atoms {example[0] + 1}-{example[1] + 1})"
        )
    if details:
        message = f"{found_name} and {reference_name}: " + "; ".join(details)
        if same_run:
            raise SegmentMismatchError(message)
        warnings.append(message)
    return warnings


# ---------------------------------------------------------------------------
# Unit cell of several sources
# ---------------------------------------------------------------------------
Change = Tuple[int, Optional[Tuple[float, ...]]]


def _changes_of_cells(cells: Sequence[Tuple[int, Sequence[float]]], first_frame: int) -> List[Change]:
    """Frames at which the cell of a source changes; ``None`` is no cell."""
    changes: List[Change] = []
    if not cells or cells[0][0] > 0:
        changes.append((first_frame, None))
    for frame, cell in cells:
        changes.append((first_frame + int(frame), tuple(float(value) for value in cell)))
    return changes


def _changes_of_summary(summary: Dict[str, Any]) -> Tuple[List[Change], int]:
    """The same from the ``box`` of a manifest, and how many changes it stands for."""
    kind = summary.get("kind")
    if kind in ("constant", "placeholder"):
        return [(0, tuple(summary["lengths_angstrom"]) + tuple(summary["angles_degree"]))], 1
    if kind == "varying":
        changes: List[Change] = []
        for item in summary["cells"]:
            if item["lengths_angstrom"] is None:
                changes.append((int(item["first_frame"]), None))
            else:
                changes.append((int(item["first_frame"]), tuple(item["lengths_angstrom"]) + tuple(item["angles_degree"])))
        return changes, int(summary.get("cells_total", len(changes)))
    return [(0, None)], 1


def box_summary(changes: Sequence[Change], not_listed: int = 0) -> Dict[str, Any]:
    """``box`` of a manifest from the cell changes of all its frames.

    The kinds are those of ``PdbTrajectory.box_summary``.  Frames without a
    cell next to frames with one make the box ``varying``; they are listed
    with null lengths and angles.  ``not_listed`` counts changes of an
    extended entry that its manifest did not list.
    """
    merged: List[Change] = []
    for frame, cell in sorted(changes, key=lambda item: item[0]):
        if not merged or merged[-1][1] != cell:
            merged.append((frame, cell))
    cells = [cell for _, cell in merged if cell is not None]
    if not cells:
        return {"kind": "none"}
    if all(archive.is_placeholder_cell(cell) for cell in cells):
        kind = "placeholder"
    elif len(merged) == 1 and not not_listed:
        kind = "constant"
    else:
        return {
            "kind": "varying",
            "cells_total": len(merged) + not_listed,
            "cells": [
                {
                    "first_frame": frame,
                    "lengths_angstrom": None if cell is None else list(cell[:3]),
                    "angles_degree": None if cell is None else list(cell[3:]),
                }
                for frame, cell in merged[: archive.MAX_LISTED_POSITIONS]
            ],
        }
    return {"kind": kind, "lengths_angstrom": list(cells[0][:3]), "angles_degree": list(cells[0][3:])}


# ---------------------------------------------------------------------------
# One source -> one part of the trajectory
# ---------------------------------------------------------------------------
def count_frames(path: PathLike) -> int:
    """Frames of a regular multi-model PDB: its ENDMDL records (a file without any is one frame)."""
    found, _ = archive._record_offsets(Path(path), (b"ENDMDL",))
    return max(1, len(found[b"ENDMDL"]))


def _convert_part(task: Dict[str, Any]) -> Dict[str, Any]:
    """Worker: convert one source into ``part_dir/trajectory.xtc`` and verify it against its text.

    The frames get the steps ``first_frame`` onwards.  Returns what the
    entry needs to know about the source, or ``{"error": (type, message)}``.
    With ``expected_frames`` a source that turns out to have another number
    of frames is not verified: ``{"frames": n}`` tells the caller to convert
    it again where it belongs.
    """
    started = time.perf_counter()
    path = Path(task["pdb"])
    part_dir = Path(task["part_dir"])
    first = int(task["first_frame"])
    try:
        if part_dir.exists():
            shutil.rmtree(part_dir)
        part_dir.mkdir(parents=True)
        source = archive.PdbTrajectory(path, repair=task["repair"])
        writer = archive._XtcWriter(part_dir / archive.TRAJECTORY_FILE, first_index=first)
        try:
            for coordinates, cell in source.blocks():
                writer.write(coordinates, cell)
        finally:
            writer.close()
        expected = task.get("expected_frames")
        if expected is not None and source.n_frames != expected:
            shutil.rmtree(part_dir, ignore_errors=True)
            return {"frames": source.n_frames}
        archive._fsync_path(part_dir / archive.TRAJECTORY_FILE)
        parsed = time.perf_counter()
        verification = archive._verify_trajectory(path, part_dir, source, task["check_frames"], first_index=first)
        finished = time.perf_counter()
    except ArchiveError as exc:
        shutil.rmtree(part_dir, ignore_errors=True)
        return {"error": (type(exc).__name__, str(exc))}
    except OSError as exc:
        shutil.rmtree(part_dir, ignore_errors=True)
        return {"error": ("SourceError", f"{type(exc).__name__}: {exc}")}
    return {
        "first_frame": first,
        "atoms": source.n_atoms,
        "frames": source.n_frames,
        "identity": list(source._identity),
        "header": source.header,
        "first_frame_text": source.first_frame,
        "trailer": source.trailer,
        "trailer_dropped": source.trailer_dropped,
        "size": source.source_size,
        "sha256": source.source_sha256,
        "coordinates_sha256": source.coordinates_sha256,
        "cells": [(int(frame), list(cell)) for frame, cell in source.cells],
        "box": source.box_summary(),
        "model_numbers": source.model_numbers,
        "records_between_frames": source.records_summary(),
        "negative_zero_coordinates": source.negative_zeros,
        "frames_parsed_line_by_line": source.line_parsed_frames,
        "irregularities": source.irregularities,
        "verification": verification,
        "seconds": {"parse_and_write": parsed - started, "verify": finished - parsed},
    }


def _pool_context():
    """Worker processes are forked, as in the converter script: no thread of this process is in
    NumPy or MDTraj when they start, and a caller needs no ``__main__`` guard."""
    try:
        return multiprocessing.get_context("fork")
    except ValueError:  # no fork on this platform
        return multiprocessing.get_context()


def _raise_part_error(source: SegmentSource, error: Tuple[str, str]) -> None:
    kind = getattr(archive, error[0], ArchiveError)
    if not (isinstance(kind, type) and issubclass(kind, ArchiveError)):
        kind = ArchiveError
    raise kind(f"{source.name} ({source.pdb}): {error[1]}")


def _convert_parts(
    sources: Sequence[SegmentSource], parts_dir: Path, first_frame: int, workers: int, repair: bool, check_frames: int
) -> List[Dict[str, Any]]:
    """Convert every source to its part; in ``workers`` processes when their frames can be counted first."""

    def task(index: int, first: int, expected: Optional[int]) -> Dict[str, Any]:
        return {
            "pdb": str(sources[index].pdb), "part_dir": str(parts_dir / f"{index:04d}"), "first_frame": first,
            "expected_frames": expected, "repair": repair, "check_frames": check_frames,
        }

    results: List[Optional[Dict[str, Any]]] = [None] * len(sources)
    processes = max(1, min(int(workers), len(sources)))
    if processes > 1 and not repair:
        # The step numbers of a part depend on the frames before it.  A regular source has one
        # frame per ENDMDL record, so the frames are counted first and all parts are written at
        # once; a part whose source has another number of frames is written again below.
        with ProcessPoolExecutor(max_workers=processes, mp_context=_pool_context()) as pool:
            counts = list(pool.map(count_frames, [str(source.pdb) for source in sources]))
            firsts = first_frame + np.concatenate(([0], np.cumsum(counts)[:-1]))
            tasks = [task(index, int(firsts[index]), int(counts[index])) for index in range(len(sources))]
            for index, result in enumerate(pool.map(_convert_part, tasks)):
                if "error" in result:
                    _raise_part_error(sources[index], result["error"])
                results[index] = result if "atoms" in result else None
    # In this process: every source when there is one process, and otherwise a source whose
    # frames were not what was counted, with every source after it (their steps are off).
    position = first_frame
    for index, source in enumerate(sources):
        result = results[index]
        if result is None or result["first_frame"] != position:
            result = _convert_part(task(index, position, None))
            if "error" in result:
                _raise_part_error(source, result["error"])
            results[index] = result
        position += result["frames"]
    return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Comparing trajectories
# ---------------------------------------------------------------------------
def _iter_xtc_frames(path: Path, frames: int, chunk: int = archive.DEFAULT_CHUNK_FRAMES) -> Iterator[Tuple[Any, ...]]:
    return archive._iter_xtc(path, chunk, 0, frames)


def compare_trajectories(first: PathLike, second: PathLike, frames: Optional[int] = None) -> Dict[str, Any]:
    """Compare two entries frame for frame: coordinates, step, time and box of the first ``frames`` frames.

    ``frames`` defaults to the frames of the shorter one.  Returns
    ``{"equal", "frames_compared", "first_difference"}``.
    """
    a, b = archive.ArchiveEntry(first), archive.ArchiveEntry(second)
    if frames is None:
        frames = min(a.n_frames, b.n_frames)
    if a.n_atoms != b.n_atoms:
        return {"equal": False, "frames_compared": 0, "first_difference": f"{a.n_atoms} against {b.n_atoms} atoms"}
    compared = 0
    difference: Optional[str] = None
    for (xyz_a, time_a, step_a, box_a), (xyz_b, time_b, step_b, box_b) in zip(
        _iter_xtc_frames(a.trajectory_path, frames), _iter_xtc_frames(b.trajectory_path, frames)
    ):
        if xyz_a.shape != xyz_b.shape:
            difference = f"frames from {compared}: {xyz_a.shape} against {xyz_b.shape}"
            break
        for label, x, y in (("coordinates", xyz_a, xyz_b), ("step", step_a, step_b), ("time", time_a, time_b),
                            ("box", box_a, box_b)):
            if (x is None) != (y is None) or (x is not None and not np.array_equal(x, y)):
                where = 0 if x is None or y is None else int(np.argwhere(
                    (np.asarray(x) != np.asarray(y)).reshape(len(xyz_a), -1).any(axis=1))[0][0])
                difference = f"{label} of frame {compared + where}"
                break
        if difference:
            break
        compared += len(xyz_a)
    if difference is None and compared != frames:
        difference = f"only {compared} of {frames} frames could be read from both"
    return {"equal": difference is None, "frames_compared": compared, "first_difference": difference}


def _prefix_sha256(path: Path, size: int) -> str:
    digest = hashlib.sha256()
    remaining = size
    with open(path, "rb") as handle:
        while remaining > 0:
            block = handle.read(min(_COPY_BLOCK, remaining))
            if not block:
                break
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest() if remaining == 0 else ""


def check_extension(entry_dir: PathLike, extended_dir: PathLike, deep: bool = True) -> Dict[str, Any]:
    """Does ``entry_dir`` hold the frames of ``extended_dir`` unchanged, before its own?

    Checked: ``extends`` of its manifest names the manifest of
    ``extended_dir`` (SHA-256), the first bytes of its trajectory are the
    trajectory of ``extended_dir``, ``topology.pdb`` and ``structure.mol2``
    are the same files, and with ``deep`` both trajectories are decoded and
    compared frame for frame.  Returns ``{"ok", "errors", ...}``.
    """
    entry_dir, extended_dir = Path(entry_dir), Path(extended_dir)
    report: Dict[str, Any] = {"ok": False, "errors": []}
    errors: List[str] = report["errors"]
    try:
        new, old = archive.load_manifest(entry_dir), archive.load_manifest(extended_dir)
        extends = new.get("extends")
        if not isinstance(extends, dict):
            errors.append("the manifest has no extends record")
            return report
        old_manifest_sha256 = archive._sha256_file(extended_dir / archive.MANIFEST_FILE)[1]
        if extends.get("manifest_sha256") != old_manifest_sha256:
            errors.append("extends.manifest_sha256 is not the SHA-256 of the manifest of the extended entry")
        if new["glycam_name"] != old["glycam_name"] or extends.get("glycam_name") != old["glycam_name"]:
            errors.append(f"GLYCAM names differ: {new['glycam_name']} extends {old['glycam_name']}")
        frames = int(old["frames"])
        if extends.get("frames") != frames or int(new["frames"]) <= frames or new["atoms"] != old["atoms"]:
            errors.append(
                f"the entry has {new['frames']} frames of {new['atoms']} atoms and says it extends "
                f"{extends.get('frames')}; the extended entry has {frames} frames of {old['atoms']} atoms"
            )
        size = int(old["files"][archive.TRAJECTORY_FILE]["size"])
        prefix = _prefix_sha256(entry_dir / archive.TRAJECTORY_FILE, size)
        report["trajectory_bytes_equal"] = (
            prefix == old["files"][archive.TRAJECTORY_FILE]["sha256"]
            and prefix == archive._sha256_file(extended_dir / archive.TRAJECTORY_FILE)[1]
        )
        if not report["trajectory_bytes_equal"]:
            errors.append(f"the first {size} bytes of the trajectory are not the trajectory of the extended entry")
        for name in (archive.TOPOLOGY_FILE, archive.MOL2_FILE):
            if new["files"][name] != old["files"][name]:
                errors.append(f"{name} is not the file of the extended entry")
        if deep and not errors:
            compared = compare_trajectories(entry_dir, extended_dir, frames)
            report["frames_compared"] = compared["frames_compared"]
            if not compared["equal"]:
                errors.append(f"the first {frames} frames differ: {compared['first_difference']}")
    except (ArchiveError, KeyError, TypeError, ValueError, OSError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    report["ok"] = not errors
    return report


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
def _hash_of_hashes(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("ascii")).hexdigest()


def _append_file(target, path: Path) -> Tuple[int, str]:
    """Append the bytes of ``path`` to the open file ``target``; their size and SHA-256."""
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(_COPY_BLOCK)
            if not block:
                break
            target.write(block)
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def _name_json(glycam_name: str) -> bytes:
    return (json.dumps({"indexOrderedSequence": glycam_name}, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _run_matches(recorded: Any, wanted: Any) -> bool:
    return json.loads(json.dumps(recorded)) == json.loads(json.dumps(wanted))


def _existing_is_current(
    final: Path,
    glycam_name: str,
    sources: Sequence[SegmentSource],
    small_files: Dict[str, Optional[bytes]],
    extends_sha256: Optional[str],
    source_name: str,
    extend_time_per_frame_ps: Optional[float],
) -> Optional[Dict[str, Any]]:
    """Manifest of ``final`` when it verifies and was made from exactly these inputs, else None.

    ``small_files`` are the files that are copied (name to content; ``None``
    for one the entry must not have).  The source files are hashed for this:
    a simulation that was run again leaves files of the same size.
    """
    if not (final / archive.MANIFEST_FILE).is_file():
        return None
    if not archive.verify_entry(final, deep=False)["ok"]:
        return None
    manifest = archive.load_manifest(final)
    try:
        if manifest["glycam_name"] != glycam_name or manifest["source"]["folder"] != source_name:
            return None
        if (manifest.get("extends") or {}).get("manifest_sha256") != extends_sha256:
            return None
        new = [segment for segment in archive.segments_of(manifest) if "carried_from" not in segment]
        if len(new) != len(sources):
            return None
        for segment in archive.segments_of(manifest):
            # the one segment that stands for an extended entry without segments has the time given here
            if segment.get("carried_from", {}).get("segment", 0) is None \
                    and segment.get("time_per_frame_ps") != extend_time_per_frame_ps:
                return None
        for segment, source in zip(new, sources):
            if (
                segment["name"] != source.name
                or segment["source"]["file"] != source.pdb.name
                or segment["source"]["size"] != source.pdb.stat().st_size
                or segment.get("time_per_frame_ps") != source.time_per_frame_ps
                or not _run_matches(segment.get("run"), source.run)
                or segment["source"]["sha256"] != archive._sha256_file(source.pdb)[1]
            ):
                return None
        for name, payload in small_files.items():
            recorded = manifest["files"].get(name)
            if payload is None:
                if recorded is not None:
                    return None
            elif recorded is None or recorded["sha256"] != hashlib.sha256(payload).hexdigest():
                return None
    except (KeyError, TypeError, OSError):
        return None
    return manifest


def convert_segments(
    sources: Sequence[SegmentSource],
    archive_dir: PathLike,
    glycam_name: str,
    *,
    mol2: Optional[PathLike] = None,
    info: Optional[PathLike] = None,
    source_name: Optional[str] = None,
    extend: Optional[PathLike] = None,
    extend_time_per_frame_ps: Optional[float] = None,
    workers: int = 1,
    repair: bool = False,
    force: bool = False,
    check_frames: int = archive.DEFAULT_CHECK_FRAMES,
    glycam_name_from: str = NAME_FROM_OPTION,
) -> Dict[str, Any]:
    """Write one archive entry from several sources, optionally after the frames of an existing entry.

    ``sources`` are converted in order; ``glycam_name`` names the entry and
    is written as ``name.json``.  ``mol2`` becomes ``structure.mol2`` (default:
    the MOL2 of the first source); the MOL2 of every source is compared with
    it.  ``info`` is archived unchanged as ``info.json``.

    ``extend`` is an existing entry folder of the same GLYCAM name.  Its
    frames come first, as the bytes they are; its ``topology.pdb``,
    ``structure.mol2`` and ``info.json`` are kept, and the sources have to
    have its atoms.  ``extend_time_per_frame_ps`` is recorded for its frames
    when its manifest does not say.

    ``workers`` processes convert the sources at the same time (not with
    ``repair``, where the frames of a source are known only afterwards).
    The sources and the extended entry are only read.  As in
    ``archive.convert_entry`` the entry is written to a staging folder and
    renamed when everything verified; an existing entry that was made from
    exactly these inputs is left alone (``status`` ``"skipped"``) unless
    ``force`` is set.

    Returns ``{"status", "manifest", "entry", "timing"}``; raises an
    :class:`ArchiveError` with the reason otherwise.
    """
    started = time.perf_counter()
    sources = [
        SegmentSource(s.name, Path(s.pdb), Path(s.mol2) if s.mol2 else None, s.time_per_frame_ps, s.run)
        for s in sources
    ]
    if not sources:
        raise SourceError("no source trajectories were given")
    names = [source.name for source in sources]
    if len(set(names)) != len(names) or not all(isinstance(name, str) and name.strip() for name in names):
        raise SourceError(f"the sources need distinct, non-empty names; got {', '.join(map(repr, names))}")
    for source in sources:
        if not source.pdb.is_file():
            raise SourceError(f"{source.name}: no such trajectory file: {source.pdb}")
        if source.mol2 is not None and not source.mol2.is_file():
            raise SourceError(f"{source.name}: no such MOL2 file: {source.mol2}")
        if source.time_per_frame_ps is not None and not source.time_per_frame_ps > 0:
            raise SourceError(f"{source.name}: time per frame must be positive, not {source.time_per_frame_ps}")
    archive._require_glycam_name(glycam_name)
    archive_dir = Path(archive_dir)
    folder = archive.entry_folder_name(glycam_name)
    final = archive_dir / folder
    source_name = source_name or folder
    extend_dir = Path(extend).resolve() if extend is not None else None

    # Nothing is written into, or next to, what is only read.
    archive_real = archive_dir.resolve()
    read_only = [source.pdb.resolve().parent for source in sources] + ([extend_dir] if extend_dir else [])
    for folder_real in read_only:
        if archive_real == folder_real or folder_real in archive_real.parents:
            raise SourceError(f"the archive directory must be outside {folder_real}")
    if extend_dir is not None and extend_dir == final.resolve():
        raise SourceError("the entry to extend is the entry that would be written; give another archive directory")
    if final.exists() and not (final / archive.MANIFEST_FILE).is_file():
        raise ArchiveError(f"{final} exists and is not an archive entry; it is left untouched")

    # The entry that is extended: verified against its manifest before anything is taken from it.
    warnings: List[str] = []
    extended: Optional[Dict[str, Any]] = None
    extended_sha256: Optional[str] = None
    reference_identity: Optional[List[bytes]] = None
    reference_name = sources[0].name
    if extend_dir is not None:
        if not archive.is_archive_entry(extend_dir):
            raise SourceError(f"{extend_dir} is not an archive entry (manifest.json and trajectory.xtc)")
        checked = archive.verify_entry(extend_dir, deep=False, check_folder=False)
        if not checked["ok"]:
            raise VerificationError(f"the entry to extend does not verify: {'; '.join(checked['errors'])}")
        extended = archive.load_manifest(extend_dir)
        extended_sha256 = archive._sha256_file(extend_dir / archive.MANIFEST_FILE)[1]
        if extended["glycam_name"] != glycam_name:
            raise NamingError(
                f"the entry to extend is {extended['glycam_name']}, the new trajectories are named {glycam_name}"
            )
        taken = {segment.get("name") for segment in archive.segments_of(extended)} or {extended["source"]["folder"]}
        clash = sorted(taken & set(names))
        if clash:
            raise SourceError(
                f"{', '.join(clash)} is already the name of a segment of the entry to extend; give the new "
                "trajectories other names (make_entry.py: --replica-prefix, or NAME=PATH)"
            )
        reference_identity = [
            atom_identity(line.encode("latin-1")) for line in archive.ArchiveEntry(extend_dir).atom_lines()
        ]
        reference_name = f"{archive.TOPOLOGY_FILE} of the entry to extend"
        if len(reference_identity) != int(extended["atoms"]):
            raise VerificationError(
                f"{reference_name} holds {len(reference_identity)} atoms, its manifest says {extended['atoms']}"
            )

    # Inputs that are copied.
    structure_path = Path(mol2) if mol2 is not None else sources[0].mol2
    if extended is None and structure_path is None:
        raise SourceError("no MOL2 file: give one, or one with the first source")
    if structure_path is not None and not structure_path.is_file():
        raise SourceError(f"no such MOL2 file: {structure_path}")
    info_payload = Path(info).read_bytes() if info is not None else None
    name_payload = _name_json(glycam_name)
    if extended is not None:
        # The entry keeps the files of the entry it extends.
        if archive.INFO_FILE in extended["files"]:
            if info_payload is not None and info_payload != (extend_dir / archive.INFO_FILE).read_bytes():
                warnings.append(
                    f"{archive.INFO_FILE} of the new trajectories is not archived: the entry keeps the "
                    f"{archive.INFO_FILE} of the entry it extends"
                )
            info_payload = (extend_dir / archive.INFO_FILE).read_bytes()
        if archive.NAME_FILE in extended["files"]:
            name_payload = (extend_dir / archive.NAME_FILE).read_bytes()
    small_files = {archive.INFO_FILE: info_payload, archive.NAME_FILE: name_payload}

    if not force:
        copied = dict(small_files)
        if extended is None:   # the structure of an extended entry is pinned by the SHA-256 of its manifest
            copied[archive.MOL2_FILE] = structure_path.read_bytes()
        manifest = _existing_is_current(
            final, glycam_name, sources, copied, extended_sha256, source_name, extend_time_per_frame_ps)
        if manifest is not None:
            return {"status": "skipped", "manifest": manifest, "entry": str(final), "timing": {}}

    # Other atoms are refused before anything is converted.
    atom_records: List[Any] = []
    for source in sources:
        identity = first_frame_identity(source.pdb)
        if identity is None:
            continue
        if not identity:
            raise archive.PdbFormatError(f"{source.name} ({source.pdb}): no ATOM/HETATM records in the first frame")
        if reference_identity is None:
            reference_identity = identity
        compare_atom_records(reference_identity, identity, reference_name, source.name)
    structures = {source.name: read_mol2(source.mol2) for source in sources if source.mol2 is not None}
    if extended is not None:
        structure = read_mol2(extend_dir / archive.MOL2_FILE)
        structure_label = f"{archive.MOL2_FILE} of the entry to extend"
    else:
        structure = read_mol2(structure_path)
        structure_label = f"{archive.MOL2_FILE} ({structure_path.name} of {sources[0].name})" \
            if mol2 is None else f"{archive.MOL2_FILE} ({structure_path})"
    if reference_identity is not None and len(structure["atoms"]) != len(reference_identity):
        raise SegmentMismatchError(
            f"{structure_label} has {len(structure['atoms'])} atoms, the trajectory has {len(reference_identity)}"
        )
    # The replicas of a run have one topology; a structure given from outside, or kept from the
    # entry that is extended, may have been written by other tools.
    same_run = extended is None and mol2 is None
    for name, found in structures.items():
        warnings.extend(compare_mol2(structure, found, structure_label, f"the MOL2 of {name}", same_run))

    staging = archive_dir / archive.STAGING_DIR / hashlib.sha256(glycam_name.encode("utf-8")).hexdigest()[:32]
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        frames_before = int(extended["frames"]) if extended is not None else 0
        parts = _convert_parts(sources, staging / PARTS_DIR, frames_before, workers, repair, check_frames)
        converted = time.perf_counter()

        atoms = int(extended["atoms"]) if extended is not None else int(parts[0]["atoms"])
        if reference_identity is None:
            reference_identity = parts[0]["identity"]
        for source, part in zip(sources, parts):
            atom_records.append(compare_atom_records(reference_identity, part["identity"], reference_name, source.name))
            if part["atoms"] != atoms:
                raise SegmentMismatchError(f"{source.name} has {part['atoms']} atoms, {reference_name} has {atoms}")
        if len(structure["atoms"]) != atoms:
            raise SegmentMismatchError(f"{structure_label} has {len(structure['atoms'])} atoms, the trajectory has {atoms}")

        # The trajectory: the bytes of the extended entry, then the parts in order.
        segments: List[Dict[str, Any]] = []
        changes: List[Change] = []
        not_listed = 0
        offset = 0
        trajectory = staging / archive.TRAJECTORY_FILE
        with open(trajectory, "wb") as target:
            if extended is not None:
                size, sha256 = _append_file(target, extend_dir / archive.TRAJECTORY_FILE)
                recorded = extended["files"][archive.TRAJECTORY_FILE]
                if size != recorded["size"] or sha256 != recorded["sha256"]:
                    raise VerificationError("the trajectory of the entry to extend changed while it was copied")
                offset = size
                carried_from = {"glycam_name": extended["glycam_name"], "manifest_sha256": extended_sha256}
                if archive.segments_of(extended):
                    for index, segment in enumerate(archive.segments_of(extended)):
                        segments.append(dict(segment, index=index, carried_from=dict(carried_from, segment=index)))
                else:
                    segment = {
                        "index": 0,
                        "name": extended["source"]["folder"],
                        "first_frame": 0,
                        "frames": frames_before,
                        "coordinates_sha256": extended["trajectory"]["coordinates_sha256"],
                        "trajectory_offset": 0,
                        "trajectory_bytes": size,
                        "time_per_frame_ps": extend_time_per_frame_ps,
                        "box": extended["box"],
                        "atom_records": "identical",
                        "source": dict(extended["source"]),
                    }
                    if extended.get("source_irregularities") is not None:
                        segment["source_irregularities"] = extended["source_irregularities"]
                    segment["verification"] = extended.get("verification")
                    segment["carried_from"] = dict(carried_from, segment=None)
                    segments.append(segment)
                box_changes, total = _changes_of_summary(extended["box"])
                changes.extend(box_changes)
                not_listed = total - len(box_changes)
            position = frames_before
            for index, (source, part) in enumerate(zip(sources, parts)):
                part_dir = staging / PARTS_DIR / f"{index:04d}"
                size, _ = _append_file(target, part_dir / archive.TRAJECTORY_FILE)
                shutil.rmtree(part_dir)
                segment = {
                    "index": len(segments),
                    "name": source.name,
                    "first_frame": position,
                    "frames": part["frames"],
                    "coordinates_sha256": part["coordinates_sha256"],
                    "trajectory_offset": offset,
                    "trajectory_bytes": size,
                    "time_per_frame_ps": source.time_per_frame_ps,
                }
                if source.run is not None:
                    segment["run"] = source.run
                segment["box"] = part["box"]
                segment["atom_records"] = atom_records[index]
                segment["source"] = {
                    "file": source.pdb.name,
                    "size": part["size"],
                    "sha256": part["sha256"],
                    "model_numbers": part["model_numbers"],
                    "records_between_frames": part["records_between_frames"],
                    "negative_zero_coordinates": part["negative_zero_coordinates"],
                    "frames_parsed_line_by_line": part["frames_parsed_line_by_line"],
                }
                if part["irregularities"] is not None:
                    segment["source_irregularities"] = part["irregularities"]
                segment["verification"] = part["verification"]
                segments.append(segment)
                changes.extend(_changes_of_cells(part["cells"], position))
                offset += size
                position += part["frames"]
            target.flush()
            os.fsync(target.fileno())
        shutil.rmtree(staging / PARTS_DIR, ignore_errors=True)
        frames = position

        # The whole trajectory once more, as it will be read: frames, steps, and the coordinates
        # of every segment against what its text gave (or, for frames taken over, its record).
        digests = archive._SegmentDigests([segment["frames"] for segment in segments])
        back = archive._read_back(staging, atoms, on_frames=digests)
        if back["frames"] != frames:
            raise VerificationError(f"{archive.TRAJECTORY_FILE} holds {back['frames']} frames, {frames} were written")
        if not back["index_ok"]:
            raise VerificationError("step and time of the trajectory are not the frame index")
        if back["precision"] not in (None, float(archive.XTC_PRECISION)):
            raise VerificationError(f"{archive.TRAJECTORY_FILE} was written at precision {back['precision']}")
        for segment, found in zip(segments, digests.hexdigests()):
            if found != segment["coordinates_sha256"]:
                raise VerificationError(
                    f"the coordinates of segment {segment['index']} ({segment['name']}) are not those of its "
                    + ("record in the entry that is extended" if "carried_from" in segment else "source text")
                )
        verified = time.perf_counter()

        # The other files.
        files: Dict[str, Dict[str, Any]] = {}
        size, sha256 = archive._sha256_file(trajectory)
        files[archive.TRAJECTORY_FILE] = {"size": size, "sha256": sha256}
        if extended is not None:
            for name in (archive.TOPOLOGY_FILE, archive.MOL2_FILE):
                files[name] = archive._copy_unchanged(extend_dir / name, staging / name)
                if files[name] != {key: extended["files"][name][key] for key in ("size", "sha256")}:
                    raise VerificationError(f"{name} of the entry to extend changed while it was copied")
        else:
            first, last = parts[0], parts[-1]
            topology = first["header"] + first["first_frame_text"] + last["trailer"] + b"END\n"
            archive._write_bytes(staging / archive.TOPOLOGY_FILE, topology)
            files[archive.TOPOLOGY_FILE] = {"size": len(topology), "sha256": hashlib.sha256(topology).hexdigest()}
            files[archive.MOL2_FILE] = archive._copy_unchanged(structure_path, staging / archive.MOL2_FILE)
            if last["trailer_dropped"]:
                warnings.append(
                    f"{last['trailer_dropped']} bytes of records after the last frame are not kept in "
                    f"{archive.TOPOLOGY_FILE}"
                )
        for name, payload in small_files.items():
            if payload is not None:
                archive._write_bytes(staging / name, payload)
                files[name] = {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}

        if info_payload is not None:
            try:
                info_name = str(json.loads(info_payload.decode("utf-8")).get("indexOrderedSequence") or "").strip()
            except (ValueError, AttributeError) as exc:
                warnings.append(f"{archive.INFO_FILE} gives no GLYCAM name ({exc})")
            else:
                if info_name and info_name != glycam_name:
                    warnings.append(
                        f"{archive.INFO_FILE} indexOrderedSequence differs from the GLYCAM name taken from "
                        f"{glycam_name_from}: {info_name}"
                    )
        for segment in segments:
            if segment.get("atom_records", "identical") != "identical" and "carried_from" not in segment:
                warnings.append(
                    f"the atom records of segment {segment['index']} ({segment['name']}) differ from "
                    f"{archive.TOPOLOGY_FILE} in {', '.join(segment['atom_records']['differ_in'])}; names, "
                    "residues and order are the same"
                )
            irregularities = segment.get("source_irregularities")
            if irregularities is not None:
                warnings.append(
                    f"segment {segment['index']} ({segment['name']}) comes from an irregular source that was "
                    f"archived with repair: {irregularities['frames_kept']} of its "
                    f"{irregularities['frames_in_source']} frames are in the archive, "
                    f"{irregularities['bytes_ignored']} of its bytes are not (source_irregularities of the segment)"
                )

        new_segments = [segment for segment in segments if "carried_from" not in segment]
        new_frames = sum(segment["frames"] for segment in new_segments)
        verification = {
            "passed": True,
            "comparison": (
                "every coordinate of trajectory.xtc, read back and rounded to 0.001 A, against the digits of "
                "the source text of its segment"
                + ("; the frames of the entry that is extended against the bytes of its trajectory and the "
                   "coordinate digests of its manifest" if extended is not None else "")
            ),
            "frames_compared": new_frames,
            "coordinates_compared": new_frames * atoms * 3,
            "mismatched_coordinates": 0,
            "max_abs_difference_angstrom": back["max_abs_difference_angstrom"],
            "max_abs_coordinate_angstrom": back["max_abs_coordinate_angstrom"],
            "reference_frames_checked": sum(part["verification"]["reference_frames_checked"] for part in parts),
            "box_max_abs_difference_angstrom": max(
                part["verification"]["box_max_abs_difference_angstrom"] for part in parts),
            "box_max_abs_difference_degree": max(
                part["verification"]["box_max_abs_difference_degree"] for part in parts),
            "segments_checked": len(segments),
        }
        if extended is not None:
            recorded = extended["files"][archive.TRAJECTORY_FILE]
            verification["frames_carried_over"] = frames_before
            verification["carried_over"] = {
                "trajectory_bytes": recorded["size"],
                "trajectory_sha256": recorded["sha256"],
                "comparison": (
                    "the first trajectory_bytes bytes of trajectory.xtc are the trajectory of the entry that is "
                    "extended (SHA-256 while they were copied); read back, the coordinates of each of its "
                    "segments have the digest its manifest records"
                ),
            }

        import mdtraj

        hashes = [segment["source"].get("sha256") for segment in segments]
        manifest: Dict[str, Any] = {
            "schema": archive.ARCHIVE_SCHEMA,
            "schema_version": archive.SCHEMA_VERSION,
            "glycam_name": glycam_name,
            "folder": folder,
            "object_key_prefix": archive.object_key_prefix(glycam_name),
            "atoms": atoms,
            "frames": frames,
            "box": box_summary(changes, not_listed),
            "trajectory": {
                "file": archive.TRAJECTORY_FILE,
                "format": "xtc",
                "precision": archive.XTC_PRECISION,
                "resolution_nm": 1.0 / archive.XTC_PRECISION,
                "resolution_angstrom": 10.0 / archive.XTC_PRECISION,
                "step_and_time": "frame index from 0; the source carries no times",
                "coordinates_sha256": back["coordinates_sha256"],
                "coordinates_sha256_of": "int32 little-endian [frames, atoms, 3] in units of 0.001 angstrom",
            },
            "files": files,
            "source": {
                "folder": source_name,
                "file": None,
                "size": sum(int(segment["source"].get("size") or 0) for segment in segments),
                "sha256": _hash_of_hashes(hashes) if all(isinstance(value, str) for value in hashes) else None,
                "sha256_of": (
                    "the SHA-256 values of the source files of the segments, in order, as hexadecimal text "
                    "joined by line feeds"
                ),
                "glycam_name_from": glycam_name_from,
                "segments": len(segments),
                "negative_zero_coordinates": sum(
                    int(segment["source"].get("negative_zero_coordinates") or 0) for segment in segments),
                "frames_parsed_line_by_line": sum(
                    int(segment["source"].get("frames_parsed_line_by_line") or 0) for segment in segments),
            },
            "segments_version": archive.SEGMENTS_VERSION,
            "segments": segments,
        }
        if extended is not None:
            manifest["extends"] = {
                "glycam_name": extended["glycam_name"],
                "manifest_sha256": extended_sha256,
                "frames": frames_before,
                "atoms": int(extended["atoms"]),
                "trajectory": {
                    "size": extended["files"][archive.TRAJECTORY_FILE]["size"],
                    "sha256": extended["files"][archive.TRAJECTORY_FILE]["sha256"],
                    "coordinates_sha256": extended["trajectory"]["coordinates_sha256"],
                },
                "files": extended["files"],
                "source": {key: extended["source"].get(key) for key in ("folder", "file", "size", "sha256")},
                "converter": extended.get("converter"),
                "repaired": archive.repair_summary(extended),
                "previous": extended.get("extends"),
            }
        manifest.update({
            "verification": verification,
            "converter": {
                "name": "lib/archive_segments.py",
                "version": archive.CONVERTER_VERSION,
                "date": archive._utc_now(),
                "mdtraj": mdtraj.__version__,
                "numpy": np.__version__,
                "python": platform.python_version(),
                "machine": platform.machine(),
            },
            "warnings": warnings,
        })
        archive._write_json_atomic(staging / archive.MANIFEST_FILE, manifest)
        archive._fsync_path(staging)

        if final.exists():
            shutil.rmtree(final)
        os.rename(staging, final)
        archive._fsync_path(archive_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    finished = time.perf_counter()
    return {
        "status": "converted",
        "manifest": manifest,
        "entry": str(final),
        "timing": {
            "seconds": finished - started,
            "convert_seconds": converted - started,
            "read_back_seconds": verified - converted,
            "parse_and_write_cpu_seconds": sum(part["seconds"]["parse_and_write"] for part in parts),
            "verify_cpu_seconds": sum(part["seconds"]["verify"] for part in parts),
        },
    }
