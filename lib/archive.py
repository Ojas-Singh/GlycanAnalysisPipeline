"""Archive format for the raw trajectory of one glycan.

The raw data of GlycoShape is one folder per glycan holding ``simulation.pdb``
(a multi-model PDB, three decimals in angstrom), ``structure.mol2`` and
sometimes ``info.json`` / ``name.json``.  This module stores the same content
in a compact entry::

    <archive>/<folder>/
        trajectory.xtc   all frames in source order, precision 0.0001 nm
        topology.pdb     header, first frame and records after the last frame
                         (CONECT, ...) of the source, verbatim lines
        structure.mol2   unchanged
        info.json        unchanged, when the source has one
        name.json        unchanged, when the source has one
        manifest.json    names, counts, hashes, provenance, verification

An entry is identified by its full GLYCAM name (``manifest.json`` ->
``glycam_name``).  ``<folder>`` is that name when it can be a directory name,
otherwise the shortened name of :func:`entry_folder_name`; the object-storage
key prefix is ``raw/<GLYCAM name>/`` (:func:`object_key_prefix`).

Coordinates are handled as integers in units of 0.001 angstrom from the PDB
text to the XTC and back, so the archive holds exactly the digits of the
source.  The public pieces are

* :class:`PdbTrajectory` - block-wise reader of the source text,
* :func:`convert_entry` - writer (writes, reads back, verifies, renames),
* :class:`ArchiveEntry` - reader that streams frames in chunks,
* :func:`verify_entry` - verifier, optionally against the source folder,
* :func:`build_index` / :func:`write_index` - ``index.json`` of an archive.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import platform
import re
import shutil
import struct
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, "os.PathLike[str]"]

ARCHIVE_SCHEMA = "glycoshape-raw-trajectory"
INDEX_SCHEMA = "glycoshape-raw-trajectory-index"
SCHEMA_VERSION = 1
CONVERTER_VERSION = "1.0.0"

TRAJECTORY_FILE = "trajectory.xtc"
TOPOLOGY_FILE = "topology.pdb"
MOL2_FILE = "structure.mol2"
INFO_FILE = "info.json"
NAME_FILE = "name.json"
MANIFEST_FILE = "manifest.json"
INDEX_FILE = "index.json"
SOURCE_TRAJECTORY_FILE = "simulation.pdb"
STAGING_DIR = ".staging"

#: XTC stores round(x_nm * precision); 10000 is 0.0001 nm = 0.001 angstrom,
#: the resolution of the %8.3f coordinate columns of a PDB file.
XTC_PRECISION = 10000
#: Below this magnitude every coordinate survives the float32 interface of
#: the XTC library exactly (measured over all values of a %8.3f column: the
#: first one that comes back one unit off is 5120.004 A).  An entry with a
#: coordinate that does not come back exactly fails verification.
XTC_EXACT_LIMIT_ANGSTROM = 5120.0

OBJECT_KEY_ROOT = "raw"
#: ``source.glycam_name_from`` of an entry whose name was given by the names file of the converter.
NAMES_FILE_ORIGIN = "names-file"
MAX_OBJECT_KEY_BYTES = 1024
MAX_FOLDER_NAME_BYTES = 255
SHORT_FOLDER_PREFIX_BYTES = 200
SHORT_FOLDER_HASH_CHARS = 16

DEFAULT_BLOCK_BYTES = 4 << 20
DEFAULT_BATCH_BYTES = 1 << 20
DEFAULT_CHECK_FRAMES = 100
DEFAULT_MAX_FRAME_BYTES = 256 << 20
DEFAULT_CHUNK_FRAMES = 1024
#: Records after the last frame (CONECT, MASTER, ...) kept in topology.pdb, at most.
MAX_TRAILER_BYTES = 4 << 20
#: Positions (of records between frames, of unit-cell changes) listed in a manifest at most.
MAX_LISTED_POSITIONS = 1000

_GLYTOUCAN_ID = re.compile(r"G\d{5}[A-Z]{2}\Z")
_SHORT_FOLDER = re.compile(r"~[0-9a-f]{%d}\Z" % SHORT_FOLDER_HASH_CHARS)
_UNSAFE_FOLDER_CHARS = re.compile(r"[/\\\x00-\x1f\x7f]")
_COORD_FIELD = re.compile(rb" *(-?)(\d+)\.(\d{3})\Z")
_ATOM_RECORDS = (b"ATOM  ", b"HETATM")
_MODEL = b"MODEL"
_MODEL_BYTES = np.frombuffer(_MODEL, dtype=np.uint8)
_COORD_COLUMNS = np.arange(30, 54)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class ArchiveError(Exception):
    """Base class of the failures this module reports for an entry."""


class NamingError(ArchiveError):
    """The GLYCAM name of a source folder cannot be established or used."""


class SourceError(ArchiveError):
    """A source folder lacks a required file."""


class PdbFormatError(ArchiveError):
    """The source PDB cannot be represented as topology + trajectory."""


class AtomCountError(PdbFormatError):
    """A frame has a different number of atoms than the first frame."""


class TopologyChangeError(PdbFormatError):
    """A frame has the atom count of the first frame but other atom records."""


class VerificationError(ArchiveError):
    """An archive entry does not reproduce its source, or its manifest."""


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------
def is_glytoucan_id(text: str) -> bool:
    """True for a GlyTouCan accession such as ``G50508SG``."""
    return bool(_GLYTOUCAN_ID.match(text))


def _require_glycam_name(glycam_name: str) -> str:
    if not isinstance(glycam_name, str) or not glycam_name:
        raise NamingError("GLYCAM name is empty")
    if glycam_name != glycam_name.strip():
        raise NamingError(f"GLYCAM name has surrounding whitespace: {glycam_name!r}")
    if re.search(r"[\x00-\x1f\x7f]", glycam_name):
        raise NamingError(f"GLYCAM name contains control characters: {glycam_name!r}")
    if any(part in ("", ".", "..") for part in glycam_name.split("/")):
        raise NamingError(f"GLYCAM name is not usable as an object key: {glycam_name!r}")
    return glycam_name


def entry_folder_name(glycam_name: str) -> str:
    """Folder name of an entry on a filesystem.

    The folder is the GLYCAM name itself when that is a usable directory
    name: at most 255 bytes of UTF-8 (the usual ``NAME_MAX``), no ``/``,
    ``\\`` or control character, not starting with ``.`` and not ending in
    ``~`` plus 16 hexadecimal digits.  Any other name is shortened to::

        <first 200 bytes of the name>~<first 16 hex digits of its SHA-256>

    with the unusable characters of the prefix replaced by ``_``.  The rule
    depends on the name only, so it gives the same folder on every machine,
    and a shortened folder is recognisable by its ``~<hash>`` ending.  The
    GLYCAM name itself is always read from ``manifest.json``.
    """
    _require_glycam_name(glycam_name)
    encoded = glycam_name.encode("utf-8")
    plain = (
        len(encoded) <= MAX_FOLDER_NAME_BYTES
        and not _UNSAFE_FOLDER_CHARS.search(glycam_name)
        and not glycam_name.startswith(".")
        and not _SHORT_FOLDER.search(glycam_name)
    )
    if plain:
        return glycam_name
    safe = _UNSAFE_FOLDER_CHARS.sub("_", glycam_name).lstrip(".") or "_"
    prefix = safe.encode("utf-8")[:SHORT_FOLDER_PREFIX_BYTES].decode("utf-8", "ignore")
    digest = hashlib.sha256(encoded).hexdigest()[:SHORT_FOLDER_HASH_CHARS]
    return f"{prefix}~{digest}"


def object_key_prefix(glycam_name: str) -> str:
    """Object-storage key prefix of an entry: ``raw/<GLYCAM name>/``."""
    _require_glycam_name(glycam_name)
    prefix = f"{OBJECT_KEY_ROOT}/{glycam_name}/"
    longest = max(len(name) for name in (TRAJECTORY_FILE, TOPOLOGY_FILE, MOL2_FILE, INFO_FILE, NAME_FILE, MANIFEST_FILE))
    if len(prefix.encode("utf-8")) + longest > MAX_OBJECT_KEY_BYTES:
        raise NamingError(
            f"object key for {glycam_name[:60]}... exceeds {MAX_OBJECT_KEY_BYTES} bytes"
        )
    return prefix


def _sequence_from_json(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("not a JSON object")
    name = str(data.get("indexOrderedSequence") or "").strip()
    if not name:
        raise ValueError("indexOrderedSequence is missing or empty")
    return name


def resolve_glycam_name(source_dir: PathLike, override: Optional[str] = None) -> Tuple[str, str, List[str]]:
    """GLYCAM name of a source folder: ``(name, origin, warnings)``.

    ``name.json`` is authoritative when present (as in the processing step).
    A folder named by a GlyTouCan accession takes the name from ``info.json``.
    Every other folder is named by its GLYCAM name.  ``origin`` is
    ``"name.json"``, ``"info.json"`` or ``"folder"``.

    ``override`` is a name given from outside (the ``--names`` file of the
    converter).  It takes precedence over all three, ``origin`` is then
    ``"names-file"``, and a different name in ``name.json`` or ``info.json``
    is reported as a warning.
    """
    source_dir = Path(source_dir)
    folder = source_dir.name
    warnings: List[str] = []
    name_path = source_dir / NAME_FILE
    info_path = source_dir / INFO_FILE

    info_name: Optional[str] = None
    info_error: Optional[str] = None
    if info_path.is_file():
        try:
            info_name = _sequence_from_json(info_path)
        except (OSError, ValueError) as exc:
            info_error = f"{INFO_FILE} gives no GLYCAM name ({exc})"

    if override is not None:
        name, origin = override, NAMES_FILE_ORIGIN
        if name_path.is_file():
            try:
                json_name = _sequence_from_json(name_path)
            except (OSError, ValueError) as exc:
                warnings.append(f"{NAME_FILE} gives no GLYCAM name ({exc})")
            else:
                if json_name != name:
                    warnings.append(
                        f"{NAME_FILE} indexOrderedSequence differs from the GLYCAM name taken from {origin}: {json_name}"
                    )
    elif name_path.is_file():
        try:
            name = _sequence_from_json(name_path)
        except (OSError, ValueError) as exc:
            raise NamingError(f"{NAME_FILE} gives no GLYCAM name ({exc})") from exc
        origin = NAME_FILE
    elif is_glytoucan_id(folder):
        if info_name is None:
            raise NamingError(
                f"folder {folder} is named by a GlyTouCan ID and has no {NAME_FILE}; "
                + (info_error or f"{INFO_FILE} is missing")
            )
        name, origin = info_name, INFO_FILE
    else:
        name, origin = folder, "folder"

    _require_glycam_name(name)
    if info_error and origin != INFO_FILE:
        warnings.append(info_error)
    if info_name is not None and info_name != name:
        warnings.append(
            f"{INFO_FILE} indexOrderedSequence differs from the GLYCAM name taken from {origin}: {info_name}"
        )
    return name, origin, warnings


def source_inputs(source_dir: PathLike) -> Dict[str, Optional[Path]]:
    """Files of a source folder (``simulation.pdb`` layout or ``<name>.pdb``)."""
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise SourceError(f"source folder does not exist: {source_dir}")
    pdb = source_dir / SOURCE_TRAJECTORY_FILE
    mol2 = source_dir / MOL2_FILE
    try:
        named_pdb = source_dir / f"{source_dir.name}.pdb"
        named_mol2 = source_dir / f"{source_dir.name}.mol2"
        if not pdb.is_file() and named_pdb.is_file():
            pdb = named_pdb
        if not mol2.is_file() and named_mol2.is_file():
            mol2 = named_mol2
    except OSError:  # folder name too long to carry a suffix
        pass
    if not pdb.is_file():
        raise SourceError(f"missing {SOURCE_TRAJECTORY_FILE}")
    if not mol2.is_file():
        raise SourceError(f"missing {MOL2_FILE}")
    info = source_dir / INFO_FILE
    name = source_dir / NAME_FILE
    return {
        "pdb": pdb,
        "mol2": mol2,
        "info": info if info.is_file() else None,
        "name": name if name.is_file() else None,
    }


# ---------------------------------------------------------------------------
# Fixed-width coordinate fields
# ---------------------------------------------------------------------------
# One %8.3f field is read as a little-endian uint64 and decoded with word
# arithmetic.  The folding below yields the integer part and the three
# decimals for any input; a field is accepted only when its eight bytes equal
# the canonical text of the decoded value, so nothing but a correctly placed
# "%8.3f" number can pass.
_FOLD_SPAN = 16666  # one more than the largest value the folding can produce
_CANONICAL: Optional[Tuple[np.ndarray, np.ndarray]] = None


def _canonical_tables() -> Tuple[np.ndarray, np.ndarray]:
    global _CANONICAL
    if _CANONICAL is None:
        whole = np.zeros(2 * _FOLD_SPAN, dtype=np.uint64)
        for value in range(10000):
            whole[value] = int.from_bytes(b"%4d" % value, "little")
        for value in range(1000):
            whole[_FOLD_SPAN + value] = int.from_bytes(b"%4s" % (b"-%d" % value), "little")
        decimals = np.zeros(_FOLD_SPAN, dtype=np.uint64)
        for value in range(1000):
            decimals[value] = int.from_bytes(b".%03d" % value, "little")
        _CANONICAL = (whole, decimals)
    return _CANONICAL


def parse_coordinate_fields(words: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode ``%8.3f`` fields given as uint64 (eight ASCII bytes, little-endian).

    Returns ``(values, valid, negative)``: int32 values in units of 0.001,
    a mask of the fields that are canonical ``%8.3f`` text, and a mask of the
    fields carrying a minus sign (``-0.000`` decodes to 0).
    """
    u = np.uint64
    words = words.astype(np.uint64, copy=False)
    whole_text, decimal_text = _canonical_tables()

    digit = words & u(0x1010101010101010)
    digit -= digit >> u(4)                     # 0x0F under every byte with bit 4 set
    digit &= words                             # digit values; ' ', '-', '.' give 0
    digit *= u(2561)                           # pairs of digits -> 0..99
    digit >>= u(8)
    digit &= u(0x00FF00FF00FF00FF)
    digit *= u(6553601)                        # pairs of pairs -> 0..9999
    digit >>= u(16)
    digit &= u(0x0000FFFF0000FFFF)
    whole = digit & u(0xFFFFFFFF)
    decimals = digit >> u(32)

    # '-' is the only accepted byte with bit 3 set and bit 4 clear.
    negative = ((words >> u(3)) & ~(words >> u(4)) & u(0x01010101)) != 0

    lookup = (whole + negative * u(_FOLD_SPAN)).view(np.int64)
    valid = (words & u(0xFFFFFFFF)) == np.take(whole_text, lookup)
    valid &= (words >> u(32)) == np.take(decimal_text, decimals.view(np.int64))

    whole *= u(1000)
    whole += decimals
    values = whole.astype(np.int32)
    np.negative(values, out=values, where=negative)
    return values, valid, negative


def _parse_fields_lenient(line: bytes, frame: int, atom: int) -> Tuple[List[int], int]:
    """Columns 31-54 of one atom line as integers of 0.001; regex based."""
    values: List[int] = []
    negative_zero = 0
    for axis in range(3):
        field = line[30 + 8 * axis: 38 + 8 * axis]
        match = _COORD_FIELD.match(field) if len(field) == 8 else None
        if match is None:
            raise PdbFormatError(
                f"frame {frame}, atom {atom}: coordinate field {field.decode('latin-1')!r} "
                f"(columns {31 + 8 * axis}-{38 + 8 * axis}) is not a number with three decimals"
            )
        value = int(match.group(2)) * 1000 + int(match.group(3))
        if match.group(1):
            if value == 0:
                negative_zero += 1
            value = -value
        values.append(value)
    return values, negative_zero


def reference_frame_angstrom(frame_text: bytes) -> np.ndarray:
    """Coordinates of one frame with ``float()`` on the PDB columns.

    This is the per-line arithmetic of ``lib.glystore`` and shares no code
    with the block parser; verification uses it on a sample of frames.
    """
    rows = []
    for line in frame_text.split(b"\n"):
        if line[:6] in _ATOM_RECORDS:
            rows.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.asarray(rows, dtype=np.float64).reshape(-1, 3)


# ---------------------------------------------------------------------------
# Unit cell
# ---------------------------------------------------------------------------
Cell = Tuple[float, float, float, float, float, float]


def _parse_cryst1(line: bytes) -> Cell:
    try:
        return (
            float(line[6:15]), float(line[15:24]), float(line[24:33]),
            float(line[33:40]), float(line[40:47]), float(line[47:54]),
        )
    except ValueError as exc:
        raise PdbFormatError(f"unreadable CRYST1 record: {line.decode('latin-1')!r}") from exc


def is_placeholder_cell(cell: Cell) -> bool:
    """The 1 x 1 x 1 A, 90 degree cell that PDB writers emit when there is no box."""
    return tuple(cell) == (1.0, 1.0, 1.0, 90.0, 90.0, 90.0)


def cell_to_vectors_nm(cell: Optional[Cell]) -> np.ndarray:
    """Box vectors in nm (GROMACS convention); zeros when there is no box."""
    box = np.zeros((3, 3), dtype=np.float64)
    if cell is None or is_placeholder_cell(cell):
        return box.astype(np.float32)
    a, b, c, alpha, beta, gamma = cell

    def cos(angle: float) -> float:
        return 0.0 if angle == 90.0 else math.cos(math.radians(angle))

    sin_gamma = 1.0 if gamma == 90.0 else math.sin(math.radians(gamma))
    box[0, 0] = a
    box[1, 0] = b * cos(gamma)
    box[1, 1] = b * sin_gamma
    box[2, 0] = c * cos(beta)
    box[2, 1] = c * (cos(alpha) - cos(beta) * cos(gamma)) / sin_gamma
    box[2, 2] = math.sqrt(max(c * c - box[2, 0] ** 2 - box[2, 1] ** 2, 0.0))
    return (box / 10.0).astype(np.float32)


def vectors_to_cell(box_nm: np.ndarray) -> np.ndarray:
    """Lengths (A) and angles (degree) of box vectors in nm, shape (..., 6)."""
    box = np.asarray(box_nm, dtype=np.float64) * 10.0
    lengths = np.linalg.norm(box, axis=-1)
    safe = np.where(lengths > 0, lengths, 1.0)

    def angle(i: int, j: int) -> np.ndarray:
        cosine = (box[..., i, :] * box[..., j, :]).sum(axis=-1) / (safe[..., i] * safe[..., j])
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    return np.concatenate(
        [lengths, np.stack([angle(1, 2), angle(0, 2), angle(0, 1)], axis=-1)], axis=-1
    )


# ---------------------------------------------------------------------------
# Source text reader
# ---------------------------------------------------------------------------
class _Buffer:
    """Unconsumed bytes of the source file, refilled block by block."""

    def __init__(self, handle, block_bytes: int, digest) -> None:
        self._handle = handle
        self._block_bytes = block_bytes
        self._digest = digest
        self.data = b""
        self.base = 0      # file offset of data[0]
        self.pos = 0       # first unconsumed byte of data
        self.size = 0      # bytes read from the file so far
        self.eof = False

    def refill(self) -> bool:
        """Drop the consumed bytes and append the next block; False at end of file."""
        if self.eof:
            return False
        chunk = self._handle.read(self._block_bytes)
        if not chunk:
            self.eof = True
            return False
        if self._digest is not None:
            self._digest.update(chunk)
        self.size += len(chunk)
        rest = self.data[self.pos:]
        self.base += self.pos
        self.pos = 0
        self.data = rest + chunk if rest else chunk
        return True


class PdbTrajectory:
    """Block-wise reader of a multi-model PDB trajectory.

    ``blocks()`` yields ``(coordinates, cell)`` where ``coordinates`` is an
    int32 array ``(frames, atoms, 3)`` in units of 0.001 A and ``cell`` the
    CRYST1 values in force for those frames (or ``None``).  Frames are found
    by their order in the file; the number on the MODEL line is not used, so
    numbers that grow in width or wrap are harmless.

    Every frame must repeat the atom records of the first frame.  Frames whose
    bytes outside the coordinate columns equal those of a known frame are
    decoded with NumPy on whole blocks; any other frame is read line by line
    and compared with the first frame field by field.  A frame with another
    atom count raises :class:`AtomCountError`, other atom records raise
    :class:`TopologyChangeError`, and a file that ends inside a frame raises
    :class:`PdbFormatError`.

    A file without MODEL records is one frame holding all its atoms.
    """

    def __init__(
        self,
        path: PathLike,
        block_bytes: int = DEFAULT_BLOCK_BYTES,
        batch_bytes: int = DEFAULT_BATCH_BYTES,
        max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
        hash_source: bool = True,
    ) -> None:
        self.path = Path(path)
        self._block_bytes = max(1, int(block_bytes))
        self._batch_bytes = max(1, int(batch_bytes))
        self._max_frame_bytes = int(max_frame_bytes)
        self._source_digest = hashlib.sha256() if hash_source else None
        self._coordinate_digest = hashlib.sha256()
        self._consumed = False

        self.n_atoms = 0
        self.n_frames = 0
        self.header = b""                 # every line before the first frame
        self.first_frame = b""            # lines of the first frame, without MODEL/ENDMDL
        self.trailer = b""                # records after the last frame, without END
        self.trailer_dropped = 0          # bytes of such records beyond MAX_TRAILER_BYTES
        self.source_size = 0
        self.source_sha256: Optional[str] = None
        self.coordinates_sha256: Optional[str] = None
        self.negative_zeros = 0
        self.line_parsed_frames = 0       # frames read line by line (first one included)
        self.has_model_records = False
        self.records_between_frames: Counter = Counter()
        self.record_positions: Dict[str, List[int]] = {}   # record -> frames each one precedes
        self.cells: List[Tuple[int, Cell]] = []   # (first frame it applies to, cell)

        self._cell: Optional[Cell] = None
        self._identity: List[bytes] = []
        self._template: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self._offsets: List[np.ndarray] = []
        self._lengths: List[np.ndarray] = []
        self._model_first: Optional[int] = None
        self._model_last: Optional[int] = None
        self._model_largest: Optional[int] = None
        self._model_breaks = 0

    # -- results -----------------------------------------------------------
    @property
    def frame_offsets(self) -> np.ndarray:
        """Byte offset of every frame in the source file."""
        return np.concatenate(self._offsets) if self._offsets else np.zeros(0, dtype=np.int64)

    @property
    def frame_lengths(self) -> np.ndarray:
        """Byte length of every frame in the source file."""
        return np.concatenate(self._lengths) if self._lengths else np.zeros(0, dtype=np.int64)

    @property
    def model_numbers(self) -> Optional[Dict[str, Any]]:
        """First, last and largest MODEL number, and whether they count up by one."""
        if self._model_first is None:
            return None
        return {
            "first": self._model_first,
            "last": self._model_last,
            "largest": self._model_largest,
            "sequential": self._model_breaks == 0,
            "breaks": self._model_breaks,
        }

    def box_summary(self) -> Dict[str, Any]:
        """Unit cell of the source: none, placeholder, constant, or varying.

        A varying cell (for example one CRYST1 per concatenated run) lists the
        frame at which every new cell starts; the XTC holds the box per frame.
        """
        if not self.cells:
            return {"kind": "none"}
        first = self.cells[0][1]
        distinct = {cell for _, cell in self.cells}
        if all(is_placeholder_cell(cell) for cell in distinct):
            kind = "placeholder"
        elif len(distinct) == 1 and self.cells[0][0] == 0:
            kind = "constant"
        else:
            changes = [item for i, item in enumerate(self.cells) if i == 0 or item[1] != self.cells[i - 1][1]]
            return {
                "kind": "varying",
                "cells_total": len(changes),
                "cells": [
                    {"first_frame": frame, "lengths_angstrom": list(cell[:3]), "angles_degree": list(cell[3:])}
                    for frame, cell in changes[:MAX_LISTED_POSITIONS]
                ],
            }
        return {
            "kind": kind,
            "lengths_angstrom": list(first[:3]),
            "angles_degree": list(first[3:]),
        }

    def records_summary(self) -> Dict[str, Any]:
        """Records between frames (END, CRYST1, ...): count and the frames they precede."""
        return {
            record: {"count": count, "before_frame": self.record_positions[record]}
            for record, count in sorted(self.records_between_frames.items())
        }

    def topology_text(self) -> bytes:
        """The source as a single-structure PDB: header, first frame, trailing records."""
        return self.header + self.first_frame + self.trailer + b"END\n"

    # -- reading -----------------------------------------------------------
    def blocks(self) -> Iterator[Tuple[np.ndarray, Optional[Cell]]]:
        if self._consumed:
            raise RuntimeError("a PdbTrajectory can be read once")
        self._consumed = True
        with open(self.path, "rb", buffering=0) as handle:
            try:
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
            except (AttributeError, OSError):
                pass
            buf = _Buffer(handle, self._block_bytes, self._source_digest)
            while True:
                if buf.pos >= len(buf.data):
                    if not buf.refill():
                        break
                    continue
                data, pos = buf.data, buf.pos
                if data.startswith(_MODEL, pos):
                    self.has_model_records = True
                    coordinates = self._frames(buf)
                    if coordinates is None:
                        buf.refill()   # at end of file _frames raises instead
                        continue
                    self.trailer = b""   # what preceded this frame was not the end of the file
                    self.trailer_dropped = 0
                    yield coordinates, self._cell
                    continue
                end = data.find(b"\n", pos)
                if end < 0:
                    if buf.refill():
                        continue
                    end = following = len(data)
                else:
                    following = end + 1
                line = data[pos:end]
                if line[:6] in _ATOM_RECORDS:
                    if self.has_model_records:
                        raise PdbFormatError(
                            f"ATOM record outside MODEL/ENDMDL at byte {buf.base + pos} "
                            f"(after frame {self.n_frames})"
                        )
                    yield self._single_frame(buf), self._cell
                    continue
                self._other_record(line, data[pos:following], buf.base + pos)
                buf.pos = following
            self.source_size = buf.size
        if self.n_frames == 0:
            raise PdbFormatError("no ATOM/HETATM records")
        if self._source_digest is not None:
            self.source_sha256 = self._source_digest.hexdigest()
        self.coordinates_sha256 = self._coordinate_digest.hexdigest()

    def _other_record(self, line: bytes, raw: bytes, offset: int) -> None:
        record = line[:6].rstrip()
        if record == b"CRYST1":
            self._cell = _parse_cryst1(line)
            self.cells.append((self.n_frames, self._cell))
        elif record == b"ENDMDL":
            raise PdbFormatError(f"ENDMDL without MODEL at byte {offset}")
        if self.n_frames == 0:
            self.header += raw
        elif line.strip():
            name = record.decode("latin-1")
            self.records_between_frames[name] += 1
            positions = self.record_positions.setdefault(name, [])
            if len(positions) < MAX_LISTED_POSITIONS:
                positions.append(self.n_frames)
            if record != b"END":
                if len(self.trailer) + len(raw) <= MAX_TRAILER_BYTES:
                    self.trailer += raw if raw.endswith(b"\n") else raw + b"\n"
                else:
                    self.trailer_dropped += len(raw)

    # -- frames with a known layout ----------------------------------------
    def _frames(self, buf: _Buffer) -> Optional[np.ndarray]:
        """Decode the frame(s) at ``buf.pos``; ``None`` when more bytes are needed."""
        template = self._template
        if template is not None:
            data, pos = buf.data, buf.pos
            eol = data.find(b"\n", pos, pos + 256)
            if eol >= 0:
                head = eol + 1 - pos
                stride = head + len(template[0])
                rows = min((len(data) - pos) // stride, max(1, self._batch_bytes // stride))
                if rows == 0 and not buf.eof:
                    return None
                if rows:
                    coordinates = self._frames_by_template(buf, head, stride, rows)
                    if coordinates is not None:
                        return coordinates
        return self._frame_by_lines(buf)

    def _frames_by_template(self, buf: _Buffer, head: int, stride: int, rows: int) -> Optional[np.ndarray]:
        body_template, coord_index, static = self._template
        data, pos = buf.data, buf.pos
        table = np.frombuffer(data, dtype=np.uint8, count=rows * stride, offset=pos).reshape(rows, stride)
        lines = table[:, :head]
        body = table[:, head:]

        differs = body != body_template
        columns = differs.any(axis=0)
        columns &= static
        head_ok = (
            (lines[:, :5] == _MODEL_BYTES).all()
            and (lines[:, head - 1] == 10).all()
            and not (lines[:, 5:head - 1] == 10).any()
        )
        if columns.any() or not head_ok:
            irregular = (
                (lines[:, :5] != _MODEL_BYTES).any(axis=1)
                | (lines[:, head - 1] != 10)
                | (lines[:, 5:head - 1] == 10).any(axis=1)
            )
            if columns.any():
                irregular |= differs[:, columns].any(axis=1)
            rows = int(np.argmax(irregular))
            if rows == 0:
                return None
            lines, body = lines[:rows], body[:rows]

        fields = np.take(body, coord_index, axis=1)
        values, valid, negative = parse_coordinate_fields(fields.view("<u8"))
        if not valid.all():
            rows = int(np.argmax(~valid.all(axis=1)))
            if rows == 0:
                return None
            lines, values, negative = lines[:rows], values[:rows], negative[:rows]

        self.negative_zeros += int(np.count_nonzero(negative & (values == 0)))
        self._note_model_numbers(_model_numbers(lines[:, 5:head - 1]))
        self._offsets.append(buf.base + pos + stride * np.arange(rows, dtype=np.int64))
        self._lengths.append(np.full(rows, stride, dtype=np.int64))
        coordinates = np.ascontiguousarray(values.reshape(rows, self.n_atoms, 3), dtype="<i4")
        self._coordinate_digest.update(coordinates)
        self.n_frames += rows
        buf.pos = pos + rows * stride
        return coordinates

    # -- frames read line by line ------------------------------------------
    def _frame_by_lines(self, buf: _Buffer) -> Optional[np.ndarray]:
        data, pos = buf.data, buf.pos
        frame_index = self.n_frames
        where = f"frame {frame_index} (byte {buf.base + pos})"
        end = data.find(b"\nENDMDL", pos)
        if end >= 0:
            stop = data.find(b"\n", end + 1)
            if stop < 0 and not buf.eof:
                return None
            stop = len(data) if stop < 0 else stop + 1
        else:
            if len(data) - pos > self._max_frame_bytes:
                raise PdbFormatError(f"{where}: no ENDMDL within {self._max_frame_bytes} bytes")
            if not buf.eof:
                return None
            raise PdbFormatError(f"{where} is not terminated by ENDMDL; the file is truncated")

        frame = data[pos:stop]
        nested = frame.find(b"\nMODEL", 1)
        if nested >= 0:
            raise PdbFormatError(
                f"{where}: MODEL record at byte {buf.base + pos + nested + 1} before the ENDMDL of this frame"
            )
        head = frame.find(b"\n") + 1
        body = frame[head:]
        coordinates = self._accept_frame(body, frame[:head], frame_index, ends_with_endmdl=True)
        self._offsets.append(np.array([buf.base + pos], dtype=np.int64))
        self._lengths.append(np.array([len(frame)], dtype=np.int64))
        buf.pos = stop
        return coordinates

    def _single_frame(self, buf: _Buffer) -> np.ndarray:
        """A file without MODEL records: all atoms up to the end are one frame."""
        while buf.refill():
            if len(buf.data) - buf.pos > self._max_frame_bytes:
                raise PdbFormatError(
                    f"no MODEL record and more than {self._max_frame_bytes} bytes of atom records"
                )
        start = buf.base + buf.pos
        body = buf.data[buf.pos:]
        for marker in (b"MODEL", b"ENDMDL"):
            if body.startswith(marker) or b"\n" + marker in body:
                raise PdbFormatError(
                    f"{marker.decode()} record after ATOM records that are outside any MODEL"
                )
        coordinates = self._accept_frame(body, b"", 0, ends_with_endmdl=False)
        self._offsets.append(np.array([start], dtype=np.int64))
        self._lengths.append(np.array([len(body)], dtype=np.int64))
        buf.pos = len(buf.data)
        return coordinates

    def _accept_frame(self, body: bytes, model_line: bytes, frame_index: int, ends_with_endmdl: bool) -> np.ndarray:
        """Check one frame against the first and return its coordinates (1, atoms, 3)."""
        atom_lines: List[bytes] = []
        atom_offsets: List[int] = []
        kept: List[bytes] = []
        offset = 0
        for raw in body.splitlines(keepends=True):
            record = raw[:6]
            if record in _ATOM_RECORDS:
                atom_lines.append(raw.rstrip(b"\r\n"))
                atom_offsets.append(offset)
            elif record == b"CRYST1":
                self._cell = _parse_cryst1(raw)
                self.cells.append((frame_index, self._cell))
            if record.rstrip() not in (b"ENDMDL", b"END"):
                kept.append(raw)
            offset += len(raw)

        if not atom_lines:
            if frame_index == 0:
                raise PdbFormatError("frame 0 has no ATOM/HETATM records")
            raise AtomCountError(f"frame {frame_index} has 0 atoms, the first frame has {self.n_atoms}")
        short = [i for i, line in enumerate(atom_lines) if len(line) < 54]
        if short:
            raise PdbFormatError(f"frame {frame_index}, atom {short[0]}: record shorter than 54 columns")
        identity = [line[:30] + line[54:80].rstrip() for line in atom_lines]

        if frame_index == 0:
            self.n_atoms = len(atom_lines)
            self._identity = identity
            self.first_frame = b"".join(kept)
            if self.first_frame and not self.first_frame.endswith(b"\n"):
                self.first_frame += b"\n"
        else:
            if len(atom_lines) != self.n_atoms:
                raise AtomCountError(
                    f"frame {frame_index} has {len(atom_lines)} atoms, the first frame has {self.n_atoms}"
                )
            if identity != self._identity:
                atom = next(i for i, (a, b) in enumerate(zip(identity, self._identity)) if a != b)
                found, first = identity[atom], self._identity[atom]
                part = "columns 1-30" if found[:30] != first[:30] else "columns 55-80"
                cut = slice(0, 30) if found[:30] != first[:30] else slice(30, None)
                raise TopologyChangeError(
                    f"frame {frame_index}, atom {atom}: {part} of the atom record differ from the first frame "
                    f"({found[cut].decode('latin-1')!r} against {first[cut].decode('latin-1')!r})"
                )

        values = np.empty((1, self.n_atoms, 3), dtype="<i4")
        for atom, line in enumerate(atom_lines):
            values[0, atom], negative_zero = _parse_fields_lenient(line, frame_index, atom)
            self.negative_zeros += negative_zero

        if ends_with_endmdl and body.endswith(b"\n"):
            # Later frames with these bytes outside the coordinate columns
            # are the same atoms in the same layout.
            coord_index = (np.asarray(atom_offsets)[:, None] + _COORD_COLUMNS).ravel()
            static = np.ones(len(body), dtype=bool)
            static[coord_index] = False
            self._template = (np.frombuffer(body, dtype=np.uint8).copy(), coord_index, static)

        if model_line:
            self._note_model_numbers(
                _model_numbers(np.frombuffer(model_line[5:].rstrip(b"\r\n"), dtype=np.uint8)[None, :])
            )
        self._coordinate_digest.update(values)
        self.line_parsed_frames += 1
        self.n_frames += 1
        return values

    def _note_model_numbers(self, numbers: np.ndarray) -> None:
        if numbers.size == 0:
            return
        previous = numbers if self._model_last is None else np.concatenate(([self._model_last], numbers))
        self._model_breaks += int(np.count_nonzero(np.diff(previous) != 1))
        self._model_breaks += int(np.count_nonzero(numbers < 0))
        if self._model_first is None:
            self._model_first = int(numbers[0])
        self._model_last = int(numbers[-1])
        largest = int(numbers.max())
        self._model_largest = largest if self._model_largest is None else max(self._model_largest, largest)


def _model_numbers(text: np.ndarray) -> np.ndarray:
    """Integer on each MODEL line (bytes after ``MODEL``); -1 where there is none."""
    rows = text.shape[0]
    if text.shape[1] == 0:
        return np.full(rows, -1, dtype=np.int64)
    is_digit = (text >= 48) & (text <= 57)
    digits_right = is_digit[:, ::-1].cumsum(axis=1)[:, ::-1] - is_digit
    values = (np.where(is_digit, text - 48, 0).astype(np.int64) * 10 ** np.minimum(digits_right, 18)).sum(axis=1)
    starts = is_digit[:, 0].astype(np.int64) + (is_digit[:, 1:] & ~is_digit[:, :-1]).sum(axis=1)
    other = (~is_digit & (text != 32) & (text != 13)).any(axis=1)
    return np.where((starts == 1) & ~other, values, -1)


# ---------------------------------------------------------------------------
# XTC through MDTraj
# ---------------------------------------------------------------------------
def _xtc_file_class():
    try:
        from mdtraj.formats import XTCTrajectoryFile
    except ImportError as exc:  # pragma: no cover - environment problem
        raise ArchiveError("the trajectory archive needs MDTraj (see requirements.txt)") from exc
    # XTCTrajectoryFile.write() fixes the precision at 1000 (0.001 nm), which
    # would drop the last digit of the PDB.  Its _write() takes the precision
    # of every frame; requirements.txt pins the version that has it.
    if not callable(getattr(XTCTrajectoryFile, "_write", None)):
        raise ArchiveError(
            "this MDTraj has no XTCTrajectoryFile._write(xyz, time, step, box, prec); "
            "install the version pinned in requirements.txt"
        )
    return XTCTrajectoryFile


class _XtcWriter:
    """Appends integer coordinates (0.001 A) to an XTC at precision 0.0001 nm."""

    def __init__(self, path: PathLike) -> None:
        self._file = _xtc_file_class()(str(path), "w")
        self.frames = 0

    def write(self, coordinates: np.ndarray, cell: Optional[Cell]) -> None:
        count = len(coordinates)
        xyz = np.empty(coordinates.shape, dtype=np.float32)
        np.divide(coordinates, float(XTC_PRECISION), out=xyz, casting="same_kind")
        index = np.arange(self.frames, self.frames + count)
        box = np.ascontiguousarray(np.broadcast_to(cell_to_vectors_nm(cell), (count, 3, 3)))
        self._file._write(
            xyz,
            index.astype(np.float32),
            index.astype(np.int32),
            box,
            np.full(count, XTC_PRECISION, dtype=np.float32),
        )
        self.frames += count

    def close(self) -> None:
        self._file.close()


def _iter_xtc(
    path: PathLike, chunk: int, start: int = 0, stop: Optional[int] = None
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """Yield ``(xyz_nm, time, step, box)`` for successive chunks of frames."""
    handle = _xtc_file_class()(str(path), "r")
    try:
        if start:
            handle.seek(int(start))
        position = int(start)
        while stop is None or position < stop:
            want = chunk if stop is None else min(chunk, stop - position)
            xyz, times, steps, box = handle.read(n_frames=int(want))
            if len(xyz) == 0:
                break
            position += len(xyz)
            yield xyz, times, steps, box
    finally:
        handle.close()


def nm_to_milliangstrom(xyz_nm: np.ndarray) -> np.ndarray:
    """Integers of 0.001 A from XTC coordinates written at precision 10000."""
    return np.rint(xyz_nm.astype(np.float64) * float(XTC_PRECISION)).astype(np.int32)


def milliangstrom_to_angstrom(values: np.ndarray) -> np.ndarray:
    """float32 angstrom, bit for bit what ``float(text)`` of the PDB column gives."""
    return (values / 1000.0).astype(np.float32)


def _xtc_header_precision(path: PathLike) -> Optional[float]:
    """Precision stored in the first frame; None below 10 atoms (stored uncompressed)."""
    with open(path, "rb") as handle:
        head = handle.read(60)
    if len(head) < 56:
        raise VerificationError(f"{TRAJECTORY_FILE} is too short to hold a frame")
    magic, atoms = struct.unpack(">ii", head[:8])
    if magic != 1995:
        raise VerificationError(f"{TRAJECTORY_FILE} does not start with the XTC magic number")
    if atoms <= 9:
        return None
    return float(struct.unpack(">f", head[56:60])[0])


# ---------------------------------------------------------------------------
# Small file helpers
# ---------------------------------------------------------------------------
def _sha256_file(path: PathLike, chunk: int = 4 << 20) -> Tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def _fsync_path(path: PathLike) -> None:
    """Flush a file or directory to disk where the filesystem supports it."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_bytes(path: Path, payload: bytes) -> None:
    with open(path, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    _write_bytes(temporary, (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    os.replace(temporary, path)


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------
def is_archive_entry(path: PathLike) -> bool:
    """True when ``path`` is an entry folder (manifest and trajectory present)."""
    try:
        path = Path(path)
        return (path / MANIFEST_FILE).is_file() and (path / TRAJECTORY_FILE).is_file()
    except OSError:
        return False


def load_manifest(entry_dir: PathLike) -> Dict[str, Any]:
    path = Path(entry_dir) / MANIFEST_FILE
    try:
        with open(path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as exc:
        raise VerificationError(f"unreadable {MANIFEST_FILE}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != ARCHIVE_SCHEMA:
        raise VerificationError(f"{MANIFEST_FILE} is not a {ARCHIVE_SCHEMA} manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise VerificationError(
            f"{MANIFEST_FILE} has schema version {manifest.get('schema_version')!r}, "
            f"this code reads version {SCHEMA_VERSION}"
        )
    return manifest


class ArchiveEntry:
    """One archive entry; streams its frames from the XTC in chunks."""

    def __init__(self, path: PathLike) -> None:
        path = Path(path)
        if path.is_file():
            path = path.parent
        self.path = path
        self.manifest = load_manifest(path)
        precision = self.manifest["trajectory"]["precision"]
        if precision != XTC_PRECISION:
            raise VerificationError(f"unsupported trajectory precision {precision!r}")

    @property
    def glycam_name(self) -> str:
        return self.manifest["glycam_name"]

    @property
    def n_atoms(self) -> int:
        return int(self.manifest["atoms"])

    @property
    def n_frames(self) -> int:
        return int(self.manifest["frames"])

    @property
    def box(self) -> Dict[str, Any]:
        return self.manifest["box"]

    @property
    def trajectory_path(self) -> Path:
        return self.path / TRAJECTORY_FILE

    @property
    def topology_path(self) -> Path:
        return self.path / TOPOLOGY_FILE

    @property
    def mol2_path(self) -> Path:
        return self.path / MOL2_FILE

    def iter_frames(
        self,
        chunk: int = DEFAULT_CHUNK_FRAMES,
        start: int = 0,
        stop: Optional[int] = None,
        unit: str = "angstrom",
    ) -> Iterator[np.ndarray]:
        """Yield arrays ``(k, atoms, 3)`` of at most ``chunk`` frames each.

        ``unit`` is ``"angstrom"`` (float32, the values a PDB reader gets from
        the source text), ``"milliangstrom"`` (int32, the stored integers) or
        ``"nm"`` (float32 as decoded from the XTC).
        """
        if unit not in ("angstrom", "milliangstrom", "nm"):
            raise ValueError("unit must be 'angstrom', 'milliangstrom' or 'nm'")
        if chunk < 1:
            raise ValueError("chunk must be >= 1")
        stop = self.n_frames if stop is None else min(int(stop), self.n_frames)
        if start < 0 or start > stop:
            raise ValueError(f"start {start} is outside 0..{stop}")
        if start == stop:
            return
        position = start
        for xyz, _, _, _ in _iter_xtc(self.trajectory_path, chunk, start, stop):
            if xyz.shape[1] != self.n_atoms:
                raise VerificationError(
                    f"{TRAJECTORY_FILE} has {xyz.shape[1]} atoms, the manifest says {self.n_atoms}"
                )
            position += len(xyz)
            if unit == "nm":
                yield xyz
            elif unit == "milliangstrom":
                yield nm_to_milliangstrom(xyz)
            else:
                yield milliangstrom_to_angstrom(nm_to_milliangstrom(xyz))
        if position != stop:
            raise VerificationError(
                f"{TRAJECTORY_FILE} ended at frame {position}, the manifest says {self.n_frames} frames"
            )

    def read_frames(self, start: int = 0, stop: Optional[int] = None, unit: str = "angstrom") -> np.ndarray:
        """Frames ``start:stop`` in memory."""
        parts = list(self.iter_frames(DEFAULT_CHUNK_FRAMES, start, stop, unit))
        if parts:
            return np.concatenate(parts, axis=0)
        return np.zeros((0, self.n_atoms, 3), dtype=np.int32 if unit == "milliangstrom" else np.float32)

    def atom_lines(self) -> List[str]:
        """ATOM/HETATM lines of ``topology.pdb`` (the first frame of the source)."""
        with open(self.topology_path, "r", encoding="latin-1") as handle:
            return [line for line in handle if line[:6] in ("ATOM  ", "HETATM")]

    def object_keys(self) -> Dict[str, str]:
        """Object-storage key of every file of the entry."""
        prefix = object_key_prefix(self.glycam_name)
        names = list(self.manifest["files"]) + [MANIFEST_FILE]
        return {name: prefix + name for name in names}


def entry_inputs(path: PathLike) -> Optional[Dict[str, str]]:
    """Inputs of the processing step when ``path`` is an archive entry, else None."""
    if not is_archive_entry(path):
        return None
    entry = ArchiveEntry(path)
    return {
        "glycam_name": entry.glycam_name,
        "trajectory_path": str(entry.trajectory_path),
        "mol2_path": str(entry.mol2_path),
    }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def _read_back(
    entry_dir: Path,
    atoms: int,
    on_frames: Optional[Callable[[int, np.ndarray], None]] = None,
    chunk: int = DEFAULT_CHUNK_FRAMES,
) -> Dict[str, Any]:
    """Decode ``trajectory.xtc`` completely and summarise what it holds.

    ``on_frames(first_frame, values)`` receives every decoded chunk as int32
    in units of 0.001 A.
    """
    digest = hashlib.sha256()
    frames = 0
    max_difference = 0.0
    max_abs = 0
    index_ok = True
    boxes: List[np.ndarray] = []
    for xyz, times, steps, box in _iter_xtc(entry_dir / TRAJECTORY_FILE, chunk):
        if xyz.shape[1] != atoms:
            raise VerificationError(f"{TRAJECTORY_FILE} has {xyz.shape[1]} atoms, expected {atoms}")
        exact = xyz.astype(np.float64)
        values = np.rint(exact * float(XTC_PRECISION)).astype("<i4")
        digest.update(np.ascontiguousarray(values))
        # exact is nm as stored; values / 1000 is the decimal number of the text
        max_difference = max(max_difference, float(np.abs(exact * 10.0 - values / 1000.0).max()))
        max_abs = max(max_abs, int(np.abs(values).max()))
        expected = np.arange(frames, frames + len(xyz))
        index_ok = index_ok and np.array_equal(steps, expected) and np.array_equal(times, expected.astype(np.float32))
        if on_frames is not None:
            on_frames(frames, values)
        boxes.append(np.zeros((len(xyz), 3, 3), dtype=np.float32) if box is None else box)
        frames += len(xyz)
    return {
        "frames": frames,
        "coordinates_sha256": digest.hexdigest(),
        "max_abs_difference_angstrom": max_difference,
        "max_abs_coordinate_angstrom": max_abs / 1000.0,
        "index_ok": bool(index_ok),
        "boxes": np.concatenate(boxes) if boxes else np.zeros((0, 3, 3), dtype=np.float32),
        "precision": _xtc_header_precision(entry_dir / TRAJECTORY_FILE) if frames else None,
    }


def _check_boxes(boxes: np.ndarray, cells: List[Tuple[int, Cell]], frames: int) -> Tuple[float, float]:
    """Largest length (A) and angle (degree) difference between XTC boxes and CRYST1."""
    expected = np.zeros((frames, 3, 3), dtype=np.float32)
    real = np.zeros(frames, dtype=bool)
    for position, (first, cell) in enumerate(cells):
        last = cells[position + 1][0] if position + 1 < len(cells) else frames
        expected[first:last] = cell_to_vectors_nm(cell)
        real[first:last] = not is_placeholder_cell(cell)
    if boxes.shape != expected.shape:
        raise VerificationError("box of the trajectory has the wrong shape")
    if (boxes[~real] != 0).any():
        raise VerificationError("frames without a unit cell carry a box in the trajectory")
    if not real.any():
        return 0.0, 0.0
    got = vectors_to_cell(boxes[real])
    want = np.zeros((frames, 6))
    for position, (first, cell) in enumerate(cells):
        last = cells[position + 1][0] if position + 1 < len(cells) else frames
        want[first:last] = cell
    delta = np.abs(got - want[real])
    return float(delta[:, :3].max()), float(delta[:, 3:].max())


def _sample_indices(frames: int, count: int) -> List[int]:
    """Evenly spaced frame indices including the first and the last."""
    if count < 0 or count >= frames:
        return list(range(frames))
    if count == 0:
        return []
    return sorted(set(np.rint(np.linspace(0, frames - 1, count)).astype(int).tolist()))


class _ReferenceCheck:
    """Compares frames of the archive with the source text read by ``float()``.

    It is called with the decoded chunks of the trajectory.  The frames with
    the given indices are read again from the source file at their byte
    offsets, so nothing but one chunk is held in memory.  ``checked`` counts
    the frames that agreed; the first disagreement is kept in ``error``.
    """

    def __init__(self, pdb_path: Path, source: PdbTrajectory, indices: Sequence[int]) -> None:
        self._handle = open(pdb_path, "rb")
        self._offsets = source.frame_offsets
        self._lengths = source.frame_lengths
        self._wanted = np.asarray(sorted(i for i in indices if i < source.n_frames), dtype=np.int64)
        self.checked = 0
        self.error: Optional[str] = None

    def __call__(self, first: int, values: np.ndarray) -> None:
        if self.error is not None:
            return
        low, high = np.searchsorted(self._wanted, [first, first + len(values)])
        for index in self._wanted[low:high].tolist():
            self._handle.seek(int(self._offsets[index]))
            reference = reference_frame_angstrom(self._handle.read(int(self._lengths[index])))
            archive = values[index - first]
            if reference.shape != archive.shape:
                self.error = (
                    f"frame {index}: {reference.shape[0]} atoms in the source text, "
                    f"{archive.shape[0]} in the archive"
                )
                return
            digits = np.rint(reference * 1000.0).astype(np.int64)
            same_floats = np.array_equal(reference.astype(np.float32), milliangstrom_to_angstrom(archive))
            if not (np.array_equal(digits, archive) and same_floats):
                atom = int(np.argmax((digits != archive).any(axis=1)))
                self.error = (
                    f"frame {index}, atom {atom}: the source text reads {reference[atom].tolist()} A, "
                    f"the archive holds {(archive[atom] / 1000.0).tolist()} A"
                )
                return
            self.checked += 1

    def close(self) -> None:
        self._handle.close()


def _compare_stream(pdb_path: Path, entry_dir: Path) -> Dict[str, Any]:
    """Compare every coordinate of the source text with the archive, frame by frame."""
    source = PdbTrajectory(pdb_path)
    mismatched = 0
    compared = 0
    largest = 0.0
    first: Optional[Tuple[int, int]] = None
    frame = 0
    handle = _xtc_file_class()(str(entry_dir / TRAJECTORY_FILE), "r")
    try:
        for coordinates, _ in source.blocks():
            xyz, _, _, _ = handle.read(n_frames=len(coordinates))
            if xyz.shape != coordinates.shape:
                raise VerificationError(
                    f"the archive holds {frame + len(xyz)} frames of {xyz.shape[1]} atoms where the source "
                    f"has at least {frame + len(coordinates)} frames of {coordinates.shape[1]} atoms"
                )
            exact = xyz.astype(np.float64)
            wrong = np.rint(exact * float(XTC_PRECISION)).astype(np.int64) != coordinates
            if wrong.any():
                if first is None:
                    where = np.argwhere(wrong)[0]
                    first = (frame + int(where[0]), int(where[1]))
                mismatched += int(wrong.sum())
            largest = max(largest, float(np.abs(exact * 10.0 - coordinates / 1000.0).max()))
            compared += coordinates.size
            frame += len(coordinates)
        extra, _, _, _ = handle.read(n_frames=1)
        if len(extra):
            raise VerificationError(f"the archive holds more than the {frame} frames of the source")
    finally:
        handle.close()
    return {
        "source": source,
        "frames": frame,
        "coordinates_compared": compared,
        "mismatched_coordinates": mismatched,
        "max_abs_difference_angstrom": largest,
        "first_mismatch": first,
    }


def _verify_trajectory(pdb_path: Path, entry_dir: Path, source: PdbTrajectory, check_frames: int) -> Dict[str, Any]:
    """Read the written XTC back and compare it with what ``source`` parsed."""
    check = _ReferenceCheck(pdb_path, source, _sample_indices(source.n_frames, check_frames))
    try:
        back = _read_back(entry_dir, source.n_atoms, on_frames=check)
    finally:
        check.close()
    precision = back["precision"]
    if precision is not None and precision != float(XTC_PRECISION):
        raise VerificationError(f"{TRAJECTORY_FILE} was written at precision {precision}, not {XTC_PRECISION}")
    if back["frames"] != source.n_frames or back["coordinates_sha256"] != source.coordinates_sha256:
        detail = _compare_stream(pdb_path, entry_dir)
        where = detail["first_mismatch"]
        beyond = back["max_abs_coordinate_angstrom"] >= XTC_EXACT_LIMIT_ANGSTROM
        raise VerificationError(
            f"{detail['mismatched_coordinates']} of {detail['coordinates_compared']} coordinates differ from the "
            f"source text after reading the trajectory back (largest difference "
            f"{detail['max_abs_difference_angstrom']:.6f} A"
            + (f", first at frame {where[0]} atom {where[1]}" if where else "")
            + ")"
            + (f"; coordinates of {XTC_EXACT_LIMIT_ANGSTROM:.0f} A or more may not be stored exactly" if beyond else "")
        )
    if check.error is not None:
        raise VerificationError(check.error)
    if not back["index_ok"]:
        raise VerificationError("step and time of the trajectory are not the frame index")
    box_length, box_angle = _check_boxes(back["boxes"], source.cells, source.n_frames)
    if box_length > 5e-4 or box_angle > 5e-3:
        raise VerificationError(
            f"box of the trajectory differs from CRYST1 by {box_length:.6f} A / {box_angle:.6f} degree"
        )
    return {
        "passed": True,
        "comparison": (
            "every coordinate of trajectory.xtc, read back and rounded to 0.001 A, "
            "against the digits of the source text"
        ),
        "frames_compared": source.n_frames,
        "coordinates_compared": source.n_frames * source.n_atoms * 3,
        "mismatched_coordinates": 0,
        "max_abs_difference_angstrom": back["max_abs_difference_angstrom"],
        "max_abs_coordinate_angstrom": back["max_abs_coordinate_angstrom"],
        "reference_frames_checked": check.checked,
        "box_max_abs_difference_angstrom": box_length,
        "box_max_abs_difference_degree": box_angle,
    }


def verify_entry(
    entry_dir: PathLike,
    source_dir: Optional[PathLike] = None,
    deep: bool = True,
    check_frames: int = DEFAULT_CHECK_FRAMES,
    name_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify an archive entry; returns ``{"ok": bool, "errors": [...], ...}``.

    Always: the manifest is readable and every file it lists has the recorded
    size and SHA-256.  ``deep``: the XTC decodes to the recorded number of
    frames and atoms, its step and time are the frame index, and its
    coordinates hash to the recorded digest.  With ``source_dir``: the source
    ``simulation.pdb`` has the recorded size and SHA-256 and every one of its
    coordinates equals the archive.

    ``name_override`` is the name the names file gives this source folder, as
    in :func:`convert_entry`.  The name of an entry that was converted with
    such a name cannot be confirmed from the source folder alone: without
    ``name_override`` it is not compared and the report says so
    (``glycam_name_checked`` is false).
    """
    entry_dir = Path(entry_dir)
    report: Dict[str, Any] = {"ok": False, "errors": [], "entry": str(entry_dir)}
    errors: List[str] = report["errors"]
    try:
        manifest = load_manifest(entry_dir)
    except VerificationError as exc:
        errors.append(str(exc))
        return report
    report["glycam_name"] = manifest.get("glycam_name")

    try:
        if manifest["folder"] != entry_folder_name(manifest["glycam_name"]):
            errors.append("manifest folder is not the folder name of its GLYCAM name")
        folder = os.path.basename(os.path.abspath(entry_dir))
        if manifest["folder"] != folder:
            errors.append(f"entry folder is {folder}, the manifest says {manifest['folder']}")
        files = manifest["files"]
        for required in (TRAJECTORY_FILE, TOPOLOGY_FILE, MOL2_FILE):
            if required not in files:
                errors.append(f"manifest does not list {required}")
        for name, recorded in files.items():
            path = entry_dir / name
            if not path.is_file():
                errors.append(f"missing {name}")
                continue
            size, sha256 = _sha256_file(path)
            if size != recorded["size"] or sha256 != recorded["sha256"]:
                errors.append(f"{name} does not match the size and SHA-256 of the manifest")
        atoms, frames = int(manifest["atoms"]), int(manifest["frames"])
    except (KeyError, TypeError, ValueError, NamingError) as exc:
        errors.append(f"incomplete manifest: {type(exc).__name__}: {exc}")
        return report
    if errors:
        return report

    # With a source: every coordinate of the text against the archive first;
    # this also gives the byte offsets of the frames for the reference check.
    check: Optional[_ReferenceCheck] = None
    if source_dir is not None:
        try:
            inputs = source_inputs(source_dir)
            named_by_file = (manifest.get("source") or {}).get("glycam_name_from") == NAMES_FILE_ORIGIN
            report["glycam_name_checked"] = name_override is not None or not named_by_file
            if report["glycam_name_checked"]:
                source_name = resolve_glycam_name(source_dir, name_override)[0]
                if source_name != manifest["glycam_name"]:
                    errors.append(f"the source folder has GLYCAM name {source_name}, the manifest says {manifest['glycam_name']}")
            detail = _compare_stream(inputs["pdb"], entry_dir)
            source: PdbTrajectory = detail["source"]
            recorded = manifest["source"]
            if source.source_size != recorded["size"] or source.source_sha256 != recorded["sha256"]:
                errors.append("the source trajectory does not match the size and SHA-256 of the manifest")
            if detail["mismatched_coordinates"]:
                errors.append(
                    f"{detail['mismatched_coordinates']} of {detail['coordinates_compared']} coordinates "
                    f"differ from the source text (first at frame/atom {detail['first_mismatch']})"
                )
            if source.topology_text() != (entry_dir / TOPOLOGY_FILE).read_bytes():
                errors.append(f"{TOPOLOGY_FILE} is not the header, first frame and trailing records of the source")
            report["source_frames_compared"] = detail["frames"]
            report["mismatched_coordinates"] = detail["mismatched_coordinates"]
            report["max_abs_difference_angstrom"] = detail["max_abs_difference_angstrom"]
            check = _ReferenceCheck(inputs["pdb"], source, _sample_indices(source.n_frames, check_frames))
        except (ArchiveError, OSError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    if deep or check is not None:
        try:
            back = _read_back(entry_dir, atoms, on_frames=check)
            if deep:
                if back["frames"] != frames:
                    errors.append(f"{TRAJECTORY_FILE} holds {back['frames']} frames, the manifest says {frames}")
                if back["coordinates_sha256"] != manifest["trajectory"]["coordinates_sha256"]:
                    errors.append("coordinates of the trajectory do not match the digest of the manifest")
                if not back["index_ok"]:
                    errors.append("step and time of the trajectory are not the frame index")
                if back["precision"] not in (None, float(XTC_PRECISION)):
                    errors.append(f"{TRAJECTORY_FILE} was written at precision {back['precision']}")
                if len(ArchiveEntry(entry_dir).atom_lines()) != atoms:
                    errors.append(f"{TOPOLOGY_FILE} does not hold {atoms} atoms")
                report["frames"] = back["frames"]
                report.setdefault("max_abs_difference_angstrom", back["max_abs_difference_angstrom"])
            if check is not None:
                if check.error is not None:
                    errors.append(check.error)
                report["reference_frames_checked"] = check.checked
        except (ArchiveError, OSError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            if check is not None:
                check.close()

    report["ok"] = not errors
    return report


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
def _copy_unchanged(source: Path, target: Path) -> Dict[str, Any]:
    payload = source.read_bytes()
    _write_bytes(target, payload)
    return {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _existing_entry_is_current(final: Path, glycam_name: str, source_dir: Path, inputs: Dict[str, Optional[Path]]) -> Optional[Dict[str, Any]]:
    """Manifest of ``final`` when it verifies and describes this source, else None."""
    if not (final / MANIFEST_FILE).is_file():
        return None
    if not verify_entry(final, deep=False)["ok"]:
        return None
    manifest = load_manifest(final)
    source = manifest.get("source", {})
    files = manifest["files"]
    try:
        current = (
            manifest["glycam_name"] == glycam_name
            and source.get("folder") == source_dir.name
            and source.get("file") == inputs["pdb"].name
            and source.get("size") == inputs["pdb"].stat().st_size
        )
        for key, name in (("mol2", MOL2_FILE), ("info", INFO_FILE), ("name", NAME_FILE)):
            path = inputs[key]
            if path is None:
                current = current and name not in files
            else:
                current = current and name in files and _sha256_file(path)[1] == files[name]["sha256"]
    except OSError:
        return None
    return manifest if current else None


def convert_entry(
    source_dir: PathLike,
    archive_dir: PathLike,
    force: bool = False,
    check_frames: int = DEFAULT_CHECK_FRAMES,
    block_bytes: int = DEFAULT_BLOCK_BYTES,
    batch_bytes: int = DEFAULT_BATCH_BYTES,
    name_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert one source folder into an archive entry under ``archive_dir``.

    The source is only read.  The entry is written to a staging folder inside
    ``archive_dir``, read back and compared with the source, and renamed to
    its final name only when that verification passes.  An existing entry
    whose manifest verifies and describes this source is left alone
    (``status`` is ``"skipped"``) unless ``force`` is set.

    ``name_override`` names the entry instead of ``name.json``, ``info.json``
    or the folder (:func:`resolve_glycam_name`); the manifest then records
    ``"names-file"`` as the source of the name.

    Returns ``{"status", "manifest", "entry", "timing"}``; raises an
    :class:`ArchiveError` with the reason when the source cannot be archived.
    """
    started = time.perf_counter()
    cpu_started = time.process_time()
    source_dir = Path(source_dir)
    archive_dir = Path(archive_dir)
    inputs = source_inputs(source_dir)
    glycam_name, origin, warnings = resolve_glycam_name(source_dir, name_override)
    folder = entry_folder_name(glycam_name)
    final = archive_dir / folder

    # The source is never written to: no entry, and no staging folder, may
    # land in the source folder or next to it.
    source_real = source_dir.resolve()
    archive_real = archive_dir.resolve()
    if archive_real in (source_real, source_real.parent) or source_real in archive_real.parents:
        raise SourceError("the archive directory must be outside the source data directory")
    if final.exists() and not (final / MANIFEST_FILE).is_file():
        raise ArchiveError(f"{final} exists and is not an archive entry; it is left untouched")

    if not force:
        manifest = _existing_entry_is_current(final, glycam_name, source_dir, inputs)
        if manifest is not None:
            return {"status": "skipped", "manifest": manifest, "entry": str(final), "timing": {}}
        try:
            other = load_manifest(final)["source"]["folder"] if final.exists() else source_dir.name
        except (VerificationError, KeyError, TypeError):
            other = source_dir.name   # unreadable entry: replaced below
        if other != source_dir.name:
            raise NamingError(
                f"the entry of this GLYCAM name was converted from source folder {other}; "
                "two source folders have the same GLYCAM name (force replaces the entry)"
            )

    staging = archive_dir / STAGING_DIR / hashlib.sha256(glycam_name.encode("utf-8")).hexdigest()[:32]
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        source = PdbTrajectory(inputs["pdb"], block_bytes=block_bytes, batch_bytes=batch_bytes)
        writer = _XtcWriter(staging / TRAJECTORY_FILE)
        try:
            for coordinates, cell in source.blocks():
                writer.write(coordinates, cell)
        finally:
            writer.close()
        _fsync_path(staging / TRAJECTORY_FILE)
        _write_bytes(staging / TOPOLOGY_FILE, source.topology_text())
        parsed = time.perf_counter()

        files: Dict[str, Dict[str, Any]] = {}
        for name in (TRAJECTORY_FILE, TOPOLOGY_FILE):
            size, sha256 = _sha256_file(staging / name)
            files[name] = {"size": size, "sha256": sha256}
        for key, name in (("mol2", MOL2_FILE), ("info", INFO_FILE), ("name", NAME_FILE)):
            if inputs[key] is not None:
                files[name] = _copy_unchanged(inputs[key], staging / name)

        verification = _verify_trajectory(inputs["pdb"], staging, source, check_frames)
        verified = time.perf_counter()

        others = sorted(
            path.name for path in source_dir.iterdir()
            if path not in (inputs["pdb"], inputs["mol2"], inputs["info"], inputs["name"])
        )
        if others:
            warnings.append("source folder has files that are not archived: " + ", ".join(others))
        if source.trailer_dropped:
            warnings.append(
                f"{source.trailer_dropped} bytes of records after the last frame are not kept in {TOPOLOGY_FILE}"
            )

        import mdtraj

        manifest = {
            "schema": ARCHIVE_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "glycam_name": glycam_name,
            "folder": folder,
            "object_key_prefix": object_key_prefix(glycam_name),
            "atoms": source.n_atoms,
            "frames": source.n_frames,
            "box": source.box_summary(),
            "trajectory": {
                "file": TRAJECTORY_FILE,
                "format": "xtc",
                "precision": XTC_PRECISION,
                "resolution_nm": 1.0 / XTC_PRECISION,
                "resolution_angstrom": 10.0 / XTC_PRECISION,
                "step_and_time": "frame index from 0; the source carries no times",
                "coordinates_sha256": source.coordinates_sha256,
                "coordinates_sha256_of": "int32 little-endian [frames, atoms, 3] in units of 0.001 angstrom",
            },
            "files": files,
            "source": {
                "folder": source_dir.name,
                "file": inputs["pdb"].name,
                "size": source.source_size,
                "sha256": source.source_sha256,
                "glycam_name_from": origin,
                "model_numbers": source.model_numbers,
                "records_between_frames": source.records_summary(),
                "negative_zero_coordinates": source.negative_zeros,
                "frames_parsed_line_by_line": source.line_parsed_frames,
            },
            "verification": verification,
            "converter": {
                "name": "lib/archive.py",
                "version": CONVERTER_VERSION,
                "date": _utc_now(),
                "mdtraj": mdtraj.__version__,
                "numpy": np.__version__,
                "python": platform.python_version(),
                "machine": platform.machine(),
            },
            "warnings": warnings,
        }
        _write_json_atomic(staging / MANIFEST_FILE, manifest)
        _fsync_path(staging)

        if final.exists():
            shutil.rmtree(final)
        os.rename(staging, final)
        _fsync_path(archive_dir)
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
            "cpu_seconds": time.process_time() - cpu_started,
            "parse_and_write_seconds": parsed - started,
            "verify_seconds": verified - parsed,
        },
    }


# ---------------------------------------------------------------------------
# Index of an archive
# ---------------------------------------------------------------------------
def iter_entry_dirs(archive_dir: PathLike) -> Iterator[Path]:
    """Entry folders of an archive (folders holding a manifest), sorted."""
    archive_dir = Path(archive_dir)
    if not archive_dir.is_dir():
        return
    for path in sorted(archive_dir.iterdir()):
        if path.is_dir() and not path.name.startswith(".") and (path / MANIFEST_FILE).is_file():
            yield path


def build_index(archive_dir: PathLike) -> Tuple[Dict[str, Any], List[str]]:
    """Index of every entry of ``archive_dir`` and the problems met while reading."""
    records: List[Dict[str, Any]] = []
    problems: List[str] = []
    for path in iter_entry_dirs(archive_dir):
        try:
            manifest = load_manifest(path)
            if manifest["folder"] != path.name:
                raise VerificationError(f"manifest names folder {manifest['folder']}")
            size, sha256 = _sha256_file(path / MANIFEST_FILE)
            files = {MANIFEST_FILE: {"size": size, "sha256": sha256}}
            files.update(manifest["files"])
            records.append({
                "glycam_name": manifest["glycam_name"],
                "folder": manifest["folder"],
                "object_key_prefix": manifest["object_key_prefix"],
                "source_folder": manifest["source"]["folder"],
                "atoms": manifest["atoms"],
                "frames": manifest["frames"],
                "source_size": manifest["source"]["size"],
                "source_sha256": manifest["source"]["sha256"],
                "archive_size": sum(item["size"] for item in files.values()),
                "files": files,
            })
        except (ArchiveError, KeyError, TypeError, OSError) as exc:
            problems.append(f"{path.name}: {type(exc).__name__}: {exc}")
    records.sort(key=lambda record: record["glycam_name"])
    by_source = Counter(record["source_folder"] for record in records)
    for folder, count in sorted(by_source.items()):
        if count > 1:
            problems.append(f"{count} entries come from source folder {folder}")
    index = {
        "schema": INDEX_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "generated": _utc_now(),
        "converter_version": CONVERTER_VERSION,
        "entries": len(records),
        "totals": {
            "frames": sum(record["frames"] for record in records),
            "source_bytes": sum(record["source_size"] for record in records),
            "archive_bytes": sum(record["archive_size"] for record in records),
        },
        "records": records,
    }
    return index, problems


def write_index(archive_dir: PathLike) -> Tuple[Dict[str, Any], List[str]]:
    """Write ``index.json`` for the whole archive; returns ``(index, problems)``."""
    index, problems = build_index(archive_dir)
    _write_json_atomic(Path(archive_dir) / INDEX_FILE, index)
    return index, problems
