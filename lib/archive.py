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

A source that is damaged in one of a few known ways (a frame that the file
cuts short, frames without a MODEL record) is refused unless ``repair`` is
asked for.  The entry then holds every complete frame, in source order, and
its manifest says under ``source_irregularities`` what was wrong with the
source and which of its bytes are not in the archive; see
:class:`PdbTrajectory`.

An entry can also hold several trajectories one after the other: the replicas
of a run, or an existing entry and the trajectories it was extended with.  Its
manifest then lists them under ``segments`` (and ``extends``), next to
everything a manifest has always had, and the readers here read it like any
other entry.  ``lib/archive_segments.py`` writes such entries; a manifest
without ``segments`` is that of one ``simulation.pdb`` and is written here,
as before.
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
#: Version of the ``segments`` / ``extends`` records of a manifest (``segments_version``).
SEGMENTS_VERSION = 1

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

# Irregularities of a source that ``repair`` accepts (``kind`` of an item of
# ``source_irregularities`` in a manifest).
#: A frame without ENDMDL that has fewer atom records than the first frame; it is dropped.
INCOMPLETE_FRAME = "incomplete_frame"
#: A frame with every atom record and no ENDMDL before the next MODEL record or the end; it is kept.
FRAME_WITHOUT_ENDMDL = "frame_without_endmdl"
#: A MODEL record that no atom record follows.
MODEL_WITHOUT_FRAME = "model_record_without_frame"
#: Frames that ENDMDL ends and no MODEL record starts; they are kept.
FRAME_WITHOUT_MODEL = "frame_without_model_record"
#: MODEL records that carry no number; listed for a repaired source only (they are read anyway).
MODEL_WITHOUT_NUMBER = "model_records_without_number"
#: Frames dropped or kept without ENDMDL, and MODEL records without a frame, accepted at most:
#: each is listed in the manifest, and a source with more needs a look, not a repair.
MAX_REPAIRED_FRAMES = 1000
END_OF_FILE = "end of file"

_GLYTOUCAN_ID = re.compile(r"G\d{5}[A-Z]{2}\Z")
_SHORT_FOLDER = re.compile(r"~[0-9a-f]{%d}\Z" % SHORT_FOLDER_HASH_CHARS)
_UNSAFE_FOLDER_CHARS = re.compile(r"[/\\\x00-\x1f\x7f]")
_COORD_FIELD = re.compile(rb" *(-?)(\d+)\.(\d{3})\Z")
_ATOM_RECORDS = (b"ATOM  ", b"HETATM")
_MODEL = b"MODEL"
_MODEL_BYTES = np.frombuffer(_MODEL, dtype=np.uint8)
_COORD_COLUMNS = np.arange(30, 54)
_COORD_BYTES = frozenset(b" -.0123456789")
#: Record names of the PDB format other than ATOM and HETATM; in a frame that is not ended by
#: ENDMDL, a line with one of them that the first frame does not have is where the frame ends.
_PDB_RECORDS = frozenset(
    b"HEADER OBSLTE TITLE SPLIT CAVEAT COMPND SOURCE KEYWDS EXPDTA NUMMDL MDLTYP AUTHOR REVDAT SPRSDE "
    b"JRNL REMARK DBREF DBREF1 DBREF2 SEQADV SEQRES MODRES HET HETNAM HETSYN FORMUL HELIX SHEET SSBOND "
    b"LINK CISPEP SITE CRYST1 ORIGX1 ORIGX2 ORIGX3 SCALE1 SCALE2 SCALE3 MTRIX1 MTRIX2 MTRIX3 MODEL "
    b"ANISOU SIGATM SIGUIJ TER ENDMDL CONECT MASTER END".split()
)
_WITH_REPAIR = "with repair (--repair of the converter)"


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

    ``repair`` accepts a source that is damaged in one of the following ways,
    and no other; without it each of them raises :class:`PdbFormatError`.
    What was accepted is in :attr:`irregularities`.

    * A frame that is not ended by ENDMDL: the next MODEL record, or the end
      of the file, comes first.  Its records have to be those of the first
      frame, in order, as far as they go, every one a whole line; after them
      the frame may hold one last line that is the beginning of the record
      expected there (the line that the cut went through).  With fewer atom
      records than the first frame the frame is *dropped*
      (``incomplete_frame``); with all of them it is *kept*
      (``frame_without_endmdl``), because nothing but the ENDMDL is missing
      and its coordinates are checked like those of every other frame.
      A cut line never counts as an atom record, whatever it holds.  When
      one file was cut and another appended, the CRYST1 record of the next
      file stands on the cut line; it is read as the record it is, and
      CRYST1 behind any other text is refused.
    * A MODEL record that no atom record follows
      (``model_record_without_frame``), whole or cut like any other line.
    * Frames that ENDMDL ends and no MODEL record starts
      (``frame_without_model_record``).

    Everything else stays an error with ``repair`` too: a frame that ENDMDL
    ends with another atom count, atom records that are not those of the
    first frame, an unreadable coordinate, a first frame that is not
    complete, atom records with neither MODEL before nor ENDMDL after them,
    a cut line with MODEL or any text but CRYST1 behind it.  Not seen, with
    or without ``repair``: text behind the word ENDMDL or END on its line.
    :attr:`n_frames`, the frame offsets, the cells and the positions of
    records count the frames that are kept, which are the frames of the
    archive; :attr:`source_frames` counts the dropped ones too, and ``frame
    N`` in an error message is that index.
    """

    def __init__(
        self,
        path: PathLike,
        block_bytes: int = DEFAULT_BLOCK_BYTES,
        batch_bytes: int = DEFAULT_BATCH_BYTES,
        max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
        hash_source: bool = True,
        repair: bool = False,
    ) -> None:
        self.path = Path(path)
        self.repair = bool(repair)
        self._block_bytes = max(1, int(block_bytes))
        self._batch_bytes = max(1, int(batch_bytes))
        self._max_frame_bytes = int(max_frame_bytes)
        self._source_digest = hashlib.sha256() if hash_source else None
        self._coordinate_digest = hashlib.sha256()
        self._consumed = False

        self.n_atoms = 0
        self.n_frames = 0                 # frames read, which are the frames of the archive
        self.source_frames = 0            # frames of the source: those and the ones repair dropped
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
        self._model_unnumbered = 0
        self._model_unnumbered_example: Optional[str] = None

        # What repair accepted.
        self._repaired: List[Dict[str, Any]] = []          # one item per frame or MODEL record, in source order
        self._without_model = 0                            # frames without a MODEL record
        self._without_model_frames: List[int] = []         # their source index and byte offset, the first ones
        self._without_model_offsets: List[int] = []
        self._layout: Optional[Tuple[List[Tuple[bool, bytes]], List[int], frozenset]] = None

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
        """First, last and largest MODEL number, and whether they count up by one.

        The numbers are those of every MODEL record of the source, in order;
        -1 stands for a record without a number.  The archive does not keep
        them: a frame is its position in the trajectory.
        """
        if self._model_first is None:
            return None
        return {
            "first": self._model_first,
            "last": self._model_last,
            "largest": self._model_largest,
            "sequential": self._model_breaks == 0,
            "breaks": self._model_breaks,
        }

    @property
    def irregularities(self) -> Optional[Dict[str, Any]]:
        """What ``repair`` accepted in this source; ``None`` for a source that needed none of it.

        This is ``source_irregularities`` of a manifest.  ``items`` lists, in
        source order, every frame that was dropped or kept without its
        ENDMDL and every MODEL record without a frame, each with the byte
        range of the source it stands for (``offset``, ``bytes``, and the
        SHA-256 of those bytes) and how many of those bytes are not in the
        archive (``bytes_ignored``).  ``source_frame`` is the index of a
        frame among the frames of the source, ``frames_kept_before`` the
        number of frames the archive holds before it, which is the index of
        the next frame of the archive.  The frames without a MODEL record
        follow as one item, and the MODEL records without a number as
        another.  Read it after :meth:`blocks` is exhausted.
        """
        if not self._repaired and not self._without_model:
            return None
        items: List[Dict[str, Any]] = []
        truncated = False
        for item in self._repaired:
            item = dict(item)
            if item["followed_by"] == END_OF_FILE and item["kind"] != MODEL_WITHOUT_FRAME:
                truncated = True          # the file ends inside this frame
                item["file_truncated"] = True
                item["file_size"] = self.source_size
            items.append(item)
        counts = Counter(item["kind"] for item in items)
        if self._without_model:
            counts[FRAME_WITHOUT_MODEL] = self._without_model
            items.append({
                "kind": FRAME_WITHOUT_MODEL,
                "action": "kept",
                "count": self._without_model,
                "source_frames": list(self._without_model_frames),
                "offsets": list(self._without_model_offsets),
            })
        if self._model_unnumbered:
            counts[MODEL_WITHOUT_NUMBER] = self._model_unnumbered
            items.append({
                "kind": MODEL_WITHOUT_NUMBER,
                "action": "noted",
                "count": self._model_unnumbered,
                "example": self._model_unnumbered_example,
            })
        return {
            "accepted_with": "repair",
            "frames_in_source": self.source_frames,
            "frames_kept": self.n_frames,
            "frames_dropped": self.source_frames - self.n_frames,
            "bytes_ignored": sum(item.get("bytes_ignored", 0) for item in items),
            "truncated": truncated,
            "counts": dict(sorted(counts.items())),
            "items": items,
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
                    if len(coordinates):   # nothing when repair dropped what the MODEL record starts
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
                    if self.repair:
                        coordinates = self._frame_without_model(buf)
                        if coordinates is None:
                            buf.refill()   # at end of file _frame_without_model decides
                            continue
                        self.trailer = b""
                        self.trailer_dropped = 0
                        yield coordinates, self._cell
                        continue
                    if self.has_model_records:
                        raise PdbFormatError(
                            f"ATOM record outside MODEL/ENDMDL at byte {buf.base + pos} "
                            f"(after frame {self.source_frames}); {_WITH_REPAIR} a frame without "
                            "its MODEL record is accepted when ENDMDL ends it, and recorded in the manifest"
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
        numbers = _model_numbers(lines[:, 5:head - 1])
        self._note_model_numbers(numbers, lambda row: lines[row, :head - 1].tobytes())
        self._offsets.append(buf.base + pos + stride * np.arange(rows, dtype=np.int64))
        self._lengths.append(np.full(rows, stride, dtype=np.int64))
        coordinates = np.ascontiguousarray(values.reshape(rows, self.n_atoms, 3), dtype="<i4")
        self._coordinate_digest.update(coordinates)
        self.n_frames += rows
        self.source_frames += rows
        buf.pos = pos + rows * stride
        return coordinates

    # -- frames read line by line ------------------------------------------
    def _frame_by_lines(self, buf: _Buffer) -> Optional[np.ndarray]:
        data, pos = buf.data, buf.pos
        where = f"frame {self.source_frames} (byte {buf.base + pos})"
        end = data.find(b"\nENDMDL", pos)
        if end >= 0:
            stop = data.find(b"\n", end + 1)
            if stop < 0 and not buf.eof:
                return None
            stop = len(data) if stop < 0 else stop + 1
            nested = data.find(b"\nMODEL", pos + 1, stop)
            if nested < 0:   # the regular case: MODEL ... ENDMDL
                frame = data[pos:stop]
                head = frame.find(b"\n") + 1
                coordinates = self._accept_frame(frame[head:], frame[:head], self.n_frames, ends_with_endmdl=True)
                self._offsets.append(np.array([buf.base + pos], dtype=np.int64))
                self._lengths.append(np.array([len(frame)], dtype=np.int64))
                buf.pos = stop
                return coordinates
            if not self.repair:
                raise PdbFormatError(
                    f"{where}: MODEL record at byte {buf.base + nested + 1} before the ENDMDL of this frame; "
                    f"{_WITH_REPAIR} a frame without ENDMDL is dropped when atom records are missing, "
                    "kept when none is, and recorded in the manifest"
                )
        else:
            # With repair the next MODEL record ends the frame; without it the ENDMDL has to come.
            nested = data.find(b"\nMODEL", pos + 1) if self.repair else -1
            if nested < 0:
                if len(data) - pos > self._max_frame_bytes:
                    raise PdbFormatError(f"{where}: no ENDMDL within {self._max_frame_bytes} bytes")
                if not buf.eof:
                    return None
                if not self.repair:
                    raise PdbFormatError(
                        f"{where} is not terminated by ENDMDL; the file is truncated; {_WITH_REPAIR} "
                        "the frames before it are archived and the cut is recorded in the manifest"
                    )
        return self._unterminated_frame(buf, len(data) if nested < 0 else nested + 1, at_end=nested < 0)

    def _single_frame(self, buf: _Buffer) -> np.ndarray:
        """A file without MODEL records: all atoms up to the end are one frame."""
        checked = 0   # bytes after buf.pos that hold neither MODEL nor ENDMDL
        while True:
            self._refuse_frame_records(buf, checked)
            checked = len(buf.data) - buf.pos
            if not buf.refill():
                break
            if len(buf.data) - buf.pos > self._max_frame_bytes:
                self._refuse_frame_records(buf, checked)
                raise PdbFormatError(
                    f"no MODEL record and more than {self._max_frame_bytes} bytes of atom records"
                )
        start = buf.base + buf.pos
        body = buf.data[buf.pos:]
        coordinates = self._accept_frame(body, b"", 0, ends_with_endmdl=False)
        self._offsets.append(np.array([start], dtype=np.int64))
        self._lengths.append(np.array([len(body)], dtype=np.int64))
        buf.pos = len(buf.data)
        return coordinates

    def _refuse_frame_records(self, buf: _Buffer, checked: int) -> None:
        """Raise when MODEL or ENDMDL follows atom records that started outside any MODEL."""
        begin = buf.pos + max(checked - 8, 0)   # a record name may lie across what was checked before
        found = [
            (at, marker) for marker in (b"ENDMDL", b"MODEL")
            for at in [buf.data.find(b"\n" + marker, begin)] if at >= 0
        ]
        if not found:
            return
        at, marker = min(found)
        hint = (
            f"; {_WITH_REPAIR} frames that ENDMDL ends and no MODEL record starts are accepted "
            "and recorded in the manifest" if marker == b"ENDMDL" else ""
        )
        raise PdbFormatError(
            f"{marker.decode()} record at byte {buf.base + at + 1} after ATOM records that are outside any MODEL"
            + hint
        )

    # -- repair ---------------------------------------------------------------
    def _frame_without_model(self, buf: _Buffer) -> Optional[np.ndarray]:
        """Repair: atom records at ``buf.pos`` that no MODEL record starts.

        They are a frame when ENDMDL ends them before any MODEL record, and
        the single frame of a file that has neither.  ``None`` when more
        bytes are needed.
        """
        data, pos = buf.data, buf.pos
        start = buf.base + pos
        end = data.find(b"\nENDMDL", pos)
        nested = data.find(b"\nMODEL", pos, len(data) if end < 0 else end)
        if nested >= 0:
            raise PdbFormatError(
                f"ATOM record outside MODEL/ENDMDL at byte {start} (after frame {self.source_frames}): "
                f"the MODEL record at byte {buf.base + nested + 1} comes before any ENDMDL, "
                "and atom records with neither are not repaired"
            )
        if end < 0:
            if len(data) - pos > self._max_frame_bytes:
                raise PdbFormatError(
                    f"no MODEL record and more than {self._max_frame_bytes} bytes of atom records"
                    if self.n_frames == 0 and not self.has_model_records else
                    f"ATOM record outside MODEL/ENDMDL at byte {start} (after frame {self.source_frames}): "
                    f"no ENDMDL within {self._max_frame_bytes} bytes"
                )
            if not buf.eof:
                return None
            if self.n_frames == 0 and not self.has_model_records:
                return self._single_frame(buf)   # a file with neither MODEL nor ENDMDL is one frame
            raise PdbFormatError(
                f"ATOM record outside MODEL/ENDMDL at byte {start} (after frame {self.source_frames}): "
                "no ENDMDL ends these atom records, and atom records with neither are not repaired"
            )
        stop = data.find(b"\n", end + 1)
        if stop < 0 and not buf.eof:
            return None
        stop = len(data) if stop < 0 else stop + 1
        frame = data[pos:stop]
        index = self.source_frames
        coordinates = self._accept_frame(frame, b"", self.n_frames, ends_with_endmdl=True)
        self._offsets.append(np.array([start], dtype=np.int64))
        self._lengths.append(np.array([len(frame)], dtype=np.int64))
        self._without_model += 1
        if len(self._without_model_frames) < MAX_LISTED_POSITIONS:
            self._without_model_frames.append(index)
            self._without_model_offsets.append(start)
        buf.pos = stop
        return coordinates

    def _first_frame_layout(self) -> Tuple[List[Tuple[bool, bytes]], List[int], frozenset]:
        """Lines of the first frame, the position of each atom record among them, the other record names."""
        if self._layout is None:
            layout = [
                (raw[:6] in _ATOM_RECORDS, raw.rstrip(b"\r\n"))
                for raw in self.first_frame.splitlines(keepends=True)
            ]
            positions = [index for index, (is_atom, _) in enumerate(layout) if is_atom]
            records = frozenset(line[:6].rstrip() for is_atom, line in layout if not is_atom)
            self._layout = (layout, positions, records)
        return self._layout

    def _unterminated_frame(self, buf: _Buffer, region_end: int, at_end: bool) -> np.ndarray:
        """Repair: the MODEL record at ``buf.pos`` is not followed by an ENDMDL.

        ``buf.data[buf.pos:region_end]`` runs to the next MODEL record, or to
        the end of the file (``at_end``).  The frame is the MODEL record, the
        records of the first frame that follow it as whole lines, and a last
        line that was cut short; what stands after that is left to be read
        as records between frames.  Returns the coordinates of the frame,
        none when it is dropped.
        """
        data, pos = buf.data, buf.pos
        start = buf.base + pos
        index = self.source_frames
        where = f"frame {index} (byte {start})"
        if self.n_frames == 0:
            raise PdbFormatError(
                f"{where} is not terminated by ENDMDL; a source is repaired only when its first frame is complete"
            )
        if len(self._repaired) >= MAX_REPAIRED_FRAMES:
            raise PdbFormatError(
                f"{where} is not terminated by ENDMDL, and {MAX_REPAIRED_FRAMES} frames or MODEL records "
                "of this source were repaired already; such a source is not repaired"
            )
        region = data[pos:region_end]
        head = region.find(b"\n") + 1 or len(region)
        cr = region.find(b"\r", 0, head)
        if 0 <= cr < head - 2:   # a \r without its \n ends the line too, as it does for the lines below
            head = cr + 1
        glued: Optional[str] = None
        glue = region.find(b"CRYST1", len(_MODEL), head)
        if glue >= 0:        # the MODEL record is the line that was cut: nothing of the frame follows it
            if region[len(_MODEL):glue].strip(b" 0123456789"):
                raise PdbFormatError(
                    f"{where}: CRYST1 stands on the line of the MODEL record behind text that is not its number"
                )
            head, glued = glue, "CRYST1"
        frame_records = self._first_frame_layout()[2]
        atoms = 0
        extent = head        # bytes of the region that are the frame
        cut = 0              # of those, the bytes of a last line that was cut short
        for raw in region[head:].splitlines(keepends=True) if glued is None else ():
            line = raw.rstrip(b"\r\n")
            # A whole line of the frame ends with a line end and has no CRYST1 behind other text: a
            # record that was cut can be all there but for its line end (the columns of an atom
            # record that are compared end at 80), with the next file following on the same line.
            whole = raw.endswith(b"\n") and line.find(b"CRYST1", 1) < 0
            record = line[:6]
            if (whole and record in _ATOM_RECORDS and atoms < self.n_atoms and len(line) >= 54
                    and line[:30] + line[54:80].rstrip() == self._identity[atoms]):
                _parse_fields_lenient(line, index, atoms)   # an unreadable coordinate is an error here too
                atoms += 1
            elif whole and record not in _ATOM_RECORDS and record.rstrip() in frame_records:
                pass                                        # TER and the like, as in the first frame
            else:
                cut, glued = self._cut_record(line, atoms, where, start + extent)
                if cut and glued is None:
                    cut = len(raw)                          # the cut line with its line end, if it has one
                extent += cut
                break
            extent += len(raw)

        rest = region[extent:]
        if glued is not None:
            followed_by = f"{glued} record on the same line"
        elif rest.strip():
            followed_by = rest.lstrip()[:6].rstrip().decode("latin-1") + " record"
        else:
            followed_by = END_OF_FILE if at_end else "MODEL record"
        model_line = region[:head]
        place = {
            "frames_kept_before": self.n_frames,
            "offset": start,
            "bytes": extent,
            "bytes_ignored": extent,
            "sha256": hashlib.sha256(region[:extent]).hexdigest(),
            "model_record": model_line.rstrip(b"\r\n").decode("latin-1"),
        }
        counted = {"atoms_found": atoms, "atoms_expected": self.n_atoms, "cut_line_bytes": cut}
        if atoms == self.n_atoms:
            # Every atom record is there, each a whole line that is the record of the first frame:
            # only the ENDMDL (and what the cut line was) is missing.  _accept_frame checks the
            # records again and reads the coordinates.
            coordinates = self._accept_frame(
                region[head:extent - cut], model_line, self.n_frames, ends_with_endmdl=False)
            self._offsets.append(np.array([start], dtype=np.int64))
            self._lengths.append(np.array([extent - cut], dtype=np.int64))
            place["bytes_ignored"] = cut
            item = {"kind": FRAME_WITHOUT_ENDMDL, "action": "kept", "source_frame": index, **place, **counted}
        else:
            coordinates = np.zeros((0, self.n_atoms, 3), dtype="<i4")
            self._note_model_numbers(
                _model_numbers(np.frombuffer(model_line[5:].rstrip(b"\r\n"), dtype=np.uint8)[None, :]),
                lambda row: model_line,
            )
            if atoms == 0 and cut == 0:   # nothing of a frame is there: not a frame of the source
                item = {"kind": MODEL_WITHOUT_FRAME, "action": "ignored", "source_frames_before": index, **place}
            else:
                item = {"kind": INCOMPLETE_FRAME, "action": "dropped", "source_frame": index, **place, **counted}
                self.source_frames += 1
        item["followed_by"] = followed_by
        self._repaired.append(item)
        buf.pos = pos + extent
        return coordinates

    def _cut_record(self, line: bytes, atoms: int, where: str, at: int) -> Tuple[int, Optional[str]]:
        """Classify a line of an unterminated frame that is not a whole record of the frame.

        ``atoms`` atom records precede it.  Returns ``(bytes, record)``:
        the bytes at the start of the line that are a record cut short, and
        the record that follows them on the same line (CRYST1 is the only
        one read there).  ``(0, None)`` says that the frame ends before this
        line.  A line that is none of this raises.
        """
        layout, positions, _ = self._first_frame_layout()
        low = positions[atoms - 1] + 1 if atoms else 0
        if atoms < self.n_atoms:
            expected = layout[low:positions[atoms] + 1]
        else:
            expected = layout[low:] + [(False, b"ENDMDL")]
        longest = max((_common_prefix(line, text, is_atom) for is_atom, text in expected), default=0)
        glue = line.find(b"CRYST1", 1)
        if glue > 0:
            if glue <= longest:
                return glue, "CRYST1"
            raise PdbFormatError(
                f"{where}: CRYST1 stands on the line at byte {at} behind text that is not the beginning "
                "of the record expected there"
            )
        if longest and longest == len(line) and line.strip() != b"END":
            return longest, None
        record = line[:6]
        if record in _ATOM_RECORDS:
            if atoms >= self.n_atoms:
                raise AtomCountError(
                    f"{where} has more than {self.n_atoms} atoms, the number of the first frame"
                )
            if len(line) >= 54:
                found, first = line[:30] + line[54:80].rstrip(), self._identity[atoms]
                if found == first:
                    return len(line), None   # the whole record without its line end
                part = "columns 1-30" if found[:30] != first[:30] else "columns 55-80"
                chosen = slice(0, 30) if found[:30] != first[:30] else slice(30, None)
                raise TopologyChangeError(
                    f"frame {self.source_frames}, atom {atoms}: {part} of the atom record differ from the first "
                    f"frame ({found[chosen].decode('latin-1')!r} against {first[chosen].decode('latin-1')!r})"
                )
            raise PdbFormatError(
                f"{where}: the atom record at byte {at} is not the record of atom {atoms} of the first frame "
                "and not the beginning of it"
            )
        if record.rstrip() in _PDB_RECORDS or not longest:
            return 0, None
        raise PdbFormatError(
            f"{where}: the line at byte {at} begins like a record of the frame and is neither that record, "
            "nor one cut short, nor one cut short with CRYST1 after it"
        )

    def _accept_frame(self, body: bytes, model_line: bytes, frame_index: int, ends_with_endmdl: bool) -> np.ndarray:
        """Check one frame against the first and return its coordinates (1, atoms, 3).

        ``frame_index`` is the index the frame gets in the archive; messages name its index in the source.
        """
        number = self.source_frames
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
            raise AtomCountError(f"frame {number} has 0 atoms, the first frame has {self.n_atoms}")
        short = [i for i, line in enumerate(atom_lines) if len(line) < 54]
        if short:
            raise PdbFormatError(f"frame {number}, atom {short[0]}: record shorter than 54 columns")
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
                    f"frame {number} has {len(atom_lines)} atoms, the first frame has {self.n_atoms}"
                )
            if identity != self._identity:
                atom = next(i for i, (a, b) in enumerate(zip(identity, self._identity)) if a != b)
                found, first = identity[atom], self._identity[atom]
                part = "columns 1-30" if found[:30] != first[:30] else "columns 55-80"
                cut = slice(0, 30) if found[:30] != first[:30] else slice(30, None)
                raise TopologyChangeError(
                    f"frame {number}, atom {atom}: {part} of the atom record differ from the first frame "
                    f"({found[cut].decode('latin-1')!r} against {first[cut].decode('latin-1')!r})"
                )

        values = np.empty((1, self.n_atoms, 3), dtype="<i4")
        for atom, line in enumerate(atom_lines):
            values[0, atom], negative_zero = _parse_fields_lenient(line, number, atom)
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
                _model_numbers(np.frombuffer(model_line[5:].rstrip(b"\r\n"), dtype=np.uint8)[None, :]),
                lambda row: model_line,
            )
        self._coordinate_digest.update(values)
        self.line_parsed_frames += 1
        self.n_frames += 1
        self.source_frames += 1
        return values

    def _note_model_numbers(self, numbers: np.ndarray, record: Callable[[int], bytes]) -> None:
        """Take the numbers of successive MODEL records; ``record(row)`` is the text of one of them."""
        if numbers.size == 0:
            return
        previous = numbers if self._model_last is None else np.concatenate(([self._model_last], numbers))
        self._model_breaks += int(np.count_nonzero(np.diff(previous) != 1))
        unnumbered = numbers < 0
        self._model_breaks += int(np.count_nonzero(unnumbered))
        if unnumbered.any():
            self._model_unnumbered += int(np.count_nonzero(unnumbered))
            if self._model_unnumbered_example is None:
                text = record(int(np.argmax(unnumbered)))
                self._model_unnumbered_example = text.rstrip(b"\r\n").decode("latin-1")
        if self._model_first is None:
            self._model_first = int(numbers[0])
        self._model_last = int(numbers[-1])
        largest = int(numbers.max())
        self._model_largest = largest if self._model_largest is None else max(self._model_largest, largest)


def _common_prefix(line: bytes, record: bytes, is_atom: bool) -> int:
    """Length of the longest beginning of ``line`` that can be the beginning of ``record``.

    ``record`` is a line of the first frame.  Outside the coordinate columns
    of an atom record the bytes have to be its bytes; inside them, where
    every frame has its own, anything a coordinate is written with.
    """
    length = 0
    for column, (found, wanted) in enumerate(zip(line, record)):
        if is_atom and 30 <= column < 54:
            if found not in _COORD_BYTES:
                break
        elif found != wanted:
            break
        length += 1
    return length


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
    """Appends integer coordinates (0.001 A) to an XTC at precision 0.0001 nm.

    ``first_index`` is the step and time of the first frame written: 0 for a
    trajectory of its own, the number of frames before it for a part that is
    appended to others (``lib/archive_segments.py``).
    """

    def __init__(self, path: PathLike, first_index: int = 0) -> None:
        self._file = _xtc_file_class()(str(path), "w")
        self.first_index = int(first_index)
        self.frames = 0

    def write(self, coordinates: np.ndarray, cell: Optional[Cell]) -> None:
        count = len(coordinates)
        xyz = np.empty(coordinates.shape, dtype=np.float32)
        np.divide(coordinates, float(XTC_PRECISION), out=xyz, casting="same_kind")
        index = np.arange(self.first_index + self.frames, self.first_index + self.frames + count)
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
    first_index: int = 0,
) -> Dict[str, Any]:
    """Decode ``trajectory.xtc`` completely and summarise what it holds.

    ``on_frames(first_frame, values)`` receives every decoded chunk as int32
    in units of 0.001 A.  ``first_index`` is the step and time the first
    frame has to carry: 0 for an entry, the frames before it for a part.
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
        expected = np.arange(first_index + frames, first_index + frames + len(xyz))
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


def _sample_indices(frames: int, count: int, irregularities: Optional[Dict[str, Any]] = None) -> List[int]:
    """Evenly spaced frame indices including the first and the last.

    For a repaired source also the frames on both sides of everything that
    was dropped or ignored, and the frames kept without ENDMDL: the places
    where the frames of the archive and of the source stop running in step.
    """
    if count < 0 or count >= frames:
        return list(range(frames))
    if count == 0:
        return []
    chosen = set(np.rint(np.linspace(0, frames - 1, count)).astype(int).tolist())
    for item in (irregularities or {}).get("items", []):
        if "frames_kept_before" in item:
            chosen.update(
                index for index in (item["frames_kept_before"] - 1, item["frames_kept_before"])
                if 0 <= index < frames
            )
    return sorted(chosen)


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


def _record_offsets(
    path: Path, records: Sequence[bytes], block_bytes: int = 8 << 20
) -> Tuple[Dict[bytes, np.ndarray], int]:
    """Byte offset of every line of ``path`` that starts with one of ``records``, and the file size.

    A plain search of the bytes of the file; it shares nothing with :class:`PdbTrajectory`.
    """
    found: Dict[bytes, List[int]] = {record: [] for record in records}
    carry = {record: b"\n" for record in records}   # the line break before the first byte of the file
    size = 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(block_bytes)
            if not block:
                break
            for record in records:
                needle = b"\n" + record
                data = carry[record] + block
                base = size - len(carry[record])
                at = data.find(needle)
                while at >= 0:
                    found[record].append(base + at + 1)
                    at = data.find(needle, at + 1)
                carry[record] = data[-len(record):]   # shorter than the needle: nothing is found twice
            size += len(block)
    return {record: np.asarray(offsets, dtype=np.int64) for record, offsets in found.items()}, size


def _check_irregularities(
    pdb_path: Path, recorded: Dict[str, Any], frames: int, atoms: int, source: Optional[PdbTrajectory] = None
) -> Dict[str, Any]:
    """Compare ``source_irregularities`` of a manifest with the source file, read once more.

    The block reader does not read here.  The MODEL and ENDMDL records of
    the file are located by a search of its bytes; their numbers have to be
    those the record implies, and every listed item has to be, at its
    offset, what the record says: the frame of the source with that index,
    after that many frames of the archive, not ended by ENDMDL, with those
    bytes (SHA-256) and that many whole atom records in them.  ``frames``
    and ``atoms`` are those of the entry.

    With ``source``, the reader that read the file with ``repair``, the byte
    range it gives every frame of the archive (the ranges the reference
    check reads frames from) is held against the same search: the n-th frame
    that has an ENDMDL has to be the bytes around the n-th ENDMDL record of
    the file.  Raises :class:`VerificationError`; returns what was counted.
    """
    def refuse(message: str) -> None:
        raise VerificationError(f"the source is not what source_irregularities of the manifest says: {message}")

    try:
        items = recorded["items"]
        unknown = sorted({item["kind"] for item in items} - {
            INCOMPLETE_FRAME, FRAME_WITHOUT_ENDMDL, MODEL_WITHOUT_FRAME, FRAME_WITHOUT_MODEL, MODEL_WITHOUT_NUMBER})
        if unknown:
            refuse(f"unknown kind {', '.join(unknown)}")
        listed = [item for item in items if "offset" in item]
        dropped = [item for item in listed if item["kind"] == INCOMPLETE_FRAME]
        without_endmdl = [item for item in listed if item["kind"] == FRAME_WITHOUT_ENDMDL]
        orphans = [item for item in listed if item["kind"] == MODEL_WITHOUT_FRAME]
        without_model = next((item for item in items if item["kind"] == FRAME_WITHOUT_MODEL), None)
        without_model_count = without_model["count"] if without_model else 0
        without_model_offsets = without_model["offsets"] if without_model else []

        if recorded["frames_kept"] != frames:
            refuse(f"{recorded['frames_kept']} frames kept, the entry holds {frames}")
        if recorded["frames_dropped"] != len(dropped) or recorded["frames_in_source"] != frames + len(dropped):
            refuse("the numbers of frames in the source, kept and dropped do not agree with the items")
        if [item["offset"] for item in listed] != sorted(item["offset"] for item in listed):
            refuse("the items are not in source order")

        found, size = _record_offsets(pdb_path, (b"MODEL", b"ENDMDL"))
        models, ends = found[b"MODEL"], found[b"ENDMDL"]
        if len(ends) != frames - len(without_endmdl):
            refuse(f"{len(ends)} ENDMDL records for {frames} frames, {len(without_endmdl)} of them without one")
        expected_models = recorded["frames_in_source"] - without_model_count + len(orphans)
        if len(models) != expected_models:
            refuse(f"{len(models)} MODEL records, {expected_models} expected")
        if source is not None:
            starts, stops = source.frame_offsets, source.frame_offsets + source.frame_lengths
            ended = np.ones(frames, dtype=bool)
            ended[[item["frames_kept_before"] for item in without_endmdl]] = False
            if len(starts) != frames or not ((starts[ended] < ends) & (ends < stops[ended])).all():
                refuse("the frames of the archive are not the frames that the ENDMDL records of the file end")

        ignored = 0
        with open(pdb_path, "rb") as handle:
            for offset in without_model_offsets:
                handle.seek(offset)
                if handle.read(6) not in _ATOM_RECORDS or offset in models:
                    refuse(f"byte {offset} is not the first atom record of a frame without MODEL record")
            for item in listed:
                kind, offset, length = item["kind"], item["offset"], item["bytes"]
                what = f"{kind} at byte {offset}"
                at = int(np.searchsorted(models, offset))
                if at == len(models) or models[at] != offset:
                    refuse(f"{what}: no MODEL record there")
                following = int(models[at + 1]) if at + 1 < len(models) else size
                if offset + length > following:
                    refuse(f"{what}: its {length} bytes run past the next MODEL record")
                if np.searchsorted(ends, offset) != np.searchsorted(ends, following):
                    refuse(f"{what}: an ENDMDL record follows before the next MODEL record")
                handle.seek(offset)
                text = handle.read(length)
                if len(text) != length or hashlib.sha256(text).hexdigest() != item["sha256"]:
                    refuse(f"{what}: its {length} bytes do not have the recorded SHA-256")
                # What stands between it and the next MODEL record, or the end of the file.
                after = handle.read(min(following - offset - length, 1 << 16))
                if not after.strip():
                    followed_by = END_OF_FILE if following == size else "MODEL record"
                else:
                    followed_by = after.lstrip()[:6].rstrip().decode("latin-1") + " record"
                    if not text.endswith((b"\n", b"\r")) and not after[:1].isspace():
                        followed_by += " on the same line"
                if item["followed_by"] != followed_by:
                    refuse(f"{what}: followed by {followed_by}, not by {item['followed_by']}")
                truncated = item["followed_by"] == END_OF_FILE and kind != MODEL_WITHOUT_FRAME
                if item.get("file_truncated", False) != truncated:
                    refuse(f"{what}: file_truncated is {item.get('file_truncated', False)}")
                if truncated and item["file_size"] != size:
                    refuse(f"{what}: the file has {size} bytes, not {item['file_size']}")

                # Its place among the frames: the MODEL records before it that start a frame and
                # the frames without one; for the archive, the ENDMDL records before it.
                if without_model_count == len(without_model_offsets):
                    in_source = (
                        at - sum(1 for other in orphans if other["offset"] < offset)
                        + sum(1 for other in without_model_offsets if other < offset)
                    )
                    if in_source != item.get("source_frame", item.get("source_frames_before")):
                        refuse(f"{what}: {in_source} frames of the source precede it")
                kept_before = int(np.searchsorted(ends, offset)) + sum(
                    1 for other in without_endmdl if other["offset"] < offset)
                if kept_before != item["frames_kept_before"]:
                    refuse(f"{what}: {kept_before} complete frames precede it")

                cut = item.get("cut_line_bytes", 0)
                lines, cut_line = text[:length - cut], text[length - cut:]
                atom_records = sum(1 for line in lines.split(b"\n")[:-1] if line[:6] in _ATOM_RECORDS)
                if kind == MODEL_WITHOUT_FRAME:
                    expected_atoms, expected_ignored = 0, length
                elif kind == INCOMPLETE_FRAME:
                    expected_atoms, expected_ignored = item["atoms_found"], length
                    if not 0 <= item["atoms_found"] < item["atoms_expected"] == atoms:
                        refuse(f"{what}: {item['atoms_found']} of {item['atoms_expected']} atom records")
                else:
                    expected_atoms, expected_ignored = atoms, cut
                whole_lines = lines.endswith(b"\n") or (kind == MODEL_WITHOUT_FRAME and b"\n" not in lines)
                if atom_records != expected_atoms or not whole_lines or b"\n" in cut_line[:-1]:
                    refuse(f"{what}: {atom_records} whole atom records and a cut line of {len(cut_line)} bytes")
                if item["bytes_ignored"] != expected_ignored:
                    refuse(f"{what}: {expected_ignored} of its bytes are not in the archive, not {item['bytes_ignored']}")
                ignored += item["bytes_ignored"]
        if ignored != recorded["bytes_ignored"]:
            refuse(f"{ignored} bytes ignored, not {recorded['bytes_ignored']}")
        if recorded["truncated"] != any(item.get("file_truncated") for item in listed):
            refuse(f"truncated is {recorded['truncated']}")
    except (KeyError, TypeError, IndexError) as exc:
        raise VerificationError(f"incomplete source_irregularities: {type(exc).__name__}: {exc}") from exc
    return {
        "items_checked": len(listed),
        "model_records": int(len(models)),
        "endmdl_records": int(len(ends)),
        "bytes_ignored": ignored,
    }


def _compare_stream(pdb_path: Path, entry_dir: Path, repair: bool = False) -> Dict[str, Any]:
    """Compare every coordinate of the source text with the archive, frame by frame.

    ``repair`` reads the source as a repaired entry was converted: its
    frames are then the frames that were kept, in source order.
    """
    source = PdbTrajectory(pdb_path, repair=repair)
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


def _verify_trajectory(
    pdb_path: Path, entry_dir: Path, source: PdbTrajectory, check_frames: int, first_index: int = 0
) -> Dict[str, Any]:
    """Read the written XTC back and compare it with what ``source`` parsed.

    ``first_index`` is the step and time of its first frame (see :func:`_read_back`).
    """
    irregularities = source.irregularities
    check = _ReferenceCheck(pdb_path, source, _sample_indices(source.n_frames, check_frames, irregularities))
    try:
        back = _read_back(entry_dir, source.n_atoms, on_frames=check, first_index=first_index)
    finally:
        check.close()
    precision = back["precision"]
    if precision is not None and precision != float(XTC_PRECISION):
        raise VerificationError(f"{TRAJECTORY_FILE} was written at precision {precision}, not {XTC_PRECISION}")
    if back["frames"] != source.n_frames or back["coordinates_sha256"] != source.coordinates_sha256:
        detail = _compare_stream(pdb_path, entry_dir, repair=source.repair)
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
    verification = {
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
    if irregularities is not None:
        # A repaired source: what was left out is compared with the file once more, without the reader.
        verification["source_irregularities"] = dict(
            _check_irregularities(pdb_path, irregularities, source.n_frames, source.n_atoms, source),
            comparison=(
                "MODEL and ENDMDL records of the source located by a search of its bytes: their numbers, an "
                "ENDMDL in the bytes of every frame of the archive but those kept without one, and every "
                "listed item read at its offset (place among the frames, no ENDMDL, SHA-256, whole atom records)"
            ),
        )
    return verification


def segments_of(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """``segments`` of a manifest; empty for an entry made from one ``simulation.pdb``.

    An entry made from several trajectories (the replicas of a run, or an
    entry that was extended) lists them in order: each segment is the frames
    ``first_frame`` to ``first_frame + frames`` of the trajectory.  See
    ``lib/archive_segments.py``, which writes such entries.
    """
    segments = manifest.get("segments")
    return segments if isinstance(segments, list) else []


class _SegmentDigests:
    """SHA-256 of the coordinates of every segment, from the decoded chunks of the trajectory.

    A callback of :func:`_read_back`; ``lengths`` are the frames of the
    segments in order.  The digest of a segment is that of its coordinates as
    int32 little-endian in units of 0.001 A, as for the whole trajectory.
    """

    def __init__(self, lengths: Sequence[int]) -> None:
        self._bounds = np.concatenate(([0], np.cumsum(np.asarray(lengths, dtype=np.int64))))
        self._digests = [hashlib.sha256() for _ in lengths]
        self.frames = 0

    def __call__(self, first: int, values: np.ndarray) -> None:
        last = first + len(values)
        self.frames = max(self.frames, last)
        segment = int(np.searchsorted(self._bounds, first, side="right")) - 1
        while first < last and segment < len(self._digests):
            stop = min(last, int(self._bounds[segment + 1]))
            if stop > first:
                self._digests[segment].update(np.ascontiguousarray(values[: stop - first]))
                values = values[stop - first:]
                first = stop
            segment += 1

    def hexdigests(self) -> List[str]:
        return [digest.hexdigest() for digest in self._digests]


def check_segment_records(manifest: Dict[str, Any]) -> List[str]:
    """Problems in the ``segments`` of a manifest: they have to cover its frames once, in order."""
    problems: List[str] = []
    position = 0
    for index, segment in enumerate(segments_of(manifest)):
        try:
            first, frames = int(segment["first_frame"]), int(segment["frames"])
            if not isinstance(segment["coordinates_sha256"], str):
                raise TypeError("coordinates_sha256 is not a string")
        except (KeyError, TypeError, ValueError) as exc:
            problems.append(f"segment {index} is incomplete: {type(exc).__name__}: {exc}")
            return problems
        if first != position or frames < 1:
            problems.append(
                f"segment {index} holds frames {first} to {first + frames}, expected to start at frame {position}"
            )
            return problems
        position += frames
    if segments_of(manifest) and position != int(manifest["frames"]):
        problems.append(f"the segments hold {position} frames, the manifest says {manifest['frames']}")
    return problems


def verify_entry(
    entry_dir: PathLike,
    source_dir: Optional[PathLike] = None,
    deep: bool = True,
    check_frames: int = DEFAULT_CHECK_FRAMES,
    name_override: Optional[str] = None,
    check_folder: bool = True,
) -> Dict[str, Any]:
    """Verify an archive entry; returns ``{"ok": bool, "errors": [...], ...}``.

    Always: the manifest is readable and every file it lists has the recorded
    size and SHA-256.  ``deep``: the XTC decodes to the recorded number of
    frames and atoms, its step and time are the frame index, and its
    coordinates hash to the recorded digest.  With ``source_dir``: the source
    ``simulation.pdb`` has the recorded size and SHA-256 and every one of its
    coordinates equals the archive.

    The manifest decides how the source is read.  An entry without
    ``source_irregularities`` is compared with a source that has to be
    regular.  A repaired entry (``repaired`` in the report) is compared with
    the source read with ``repair``: the frames that are kept have to be the
    frames of the archive, the irregularities found have to be exactly the
    recorded ones, and the recorded ones are checked against the bytes of
    the file once more (:func:`_check_irregularities`).

    ``name_override`` is the name the names file gives this source folder, as
    in :func:`convert_entry`.  The name of an entry that was converted with
    such a name cannot be confirmed from the source folder alone: without
    ``name_override`` it is not compared and the report says so
    (``glycam_name_checked`` is false).

    ``check_folder=False`` accepts an entry in a folder that is not named by
    its manifest, such as ``archive/`` of an extracted entry bundle.

    An entry with ``segments`` (several trajectories, ``lib/archive_segments.py``)
    is checked the same way; ``deep`` also compares the coordinates of every
    segment with the digest of its record (``segments_checked``).  It cannot
    be compared with one source folder: ``source_dir`` is then an error.
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
        if check_folder and manifest["folder"] != folder:
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
        irregularities = manifest.get("source_irregularities")
        if irregularities is not None:
            report["repaired"] = True
            if irregularities["frames_kept"] != frames:
                errors.append(
                    f"source_irregularities says {irregularities['frames_kept']} frames were kept, "
                    f"the manifest says {frames} frames"
                )
        segments = segments_of(manifest)
        errors.extend(check_segment_records(manifest))
        if segments:
            report["segments"] = len(segments)
            if repair_summary(manifest) is not None:
                report["repaired"] = True
            if source_dir is not None:
                errors.append(
                    f"the entry was made from {len(segments)} segments; it cannot be compared with one source folder"
                )
    except (KeyError, TypeError, ValueError, NamingError) as exc:
        errors.append(f"incomplete manifest: {type(exc).__name__}: {exc}")
        return report
    if errors:
        return report

    # With a source: every coordinate of the text against the archive first;
    # this also gives the byte offsets of the frames for the reference check.
    check: Optional[_ReferenceCheck] = None
    segment_digests = _SegmentDigests([int(segment["frames"]) for segment in segments]) if segments else None
    if source_dir is not None:
        try:
            inputs = source_inputs(source_dir)
            named_by_file = (manifest.get("source") or {}).get("glycam_name_from") == NAMES_FILE_ORIGIN
            report["glycam_name_checked"] = name_override is not None or not named_by_file
            if report["glycam_name_checked"]:
                source_name = resolve_glycam_name(source_dir, name_override)[0]
                if source_name != manifest["glycam_name"]:
                    errors.append(f"the source folder has GLYCAM name {source_name}, the manifest says {manifest['glycam_name']}")
            detail = _compare_stream(inputs["pdb"], entry_dir, repair=irregularities is not None)
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
            if irregularities is not None:
                if source.irregularities != irregularities:
                    errors.append("the irregularities of the source are not those source_irregularities records")
                report["source_irregularities_checked"] = _check_irregularities(
                    inputs["pdb"], irregularities, frames, atoms, source)["items_checked"]
            report["source_frames_compared"] = detail["frames"]
            report["mismatched_coordinates"] = detail["mismatched_coordinates"]
            report["max_abs_difference_angstrom"] = detail["max_abs_difference_angstrom"]
            check = _ReferenceCheck(
                inputs["pdb"], source, _sample_indices(source.n_frames, check_frames, irregularities))
        except (ArchiveError, OSError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    if deep or check is not None:
        try:
            # A segmented entry has no source folder to compare with, so there is no reference check then.
            back = _read_back(entry_dir, atoms, on_frames=check if check is not None else segment_digests)
            if deep:
                if back["frames"] != frames:
                    errors.append(f"{TRAJECTORY_FILE} holds {back['frames']} frames, the manifest says {frames}")
                if back["coordinates_sha256"] != manifest["trajectory"]["coordinates_sha256"]:
                    errors.append("coordinates of the trajectory do not match the digest of the manifest")
                if segment_digests is not None:
                    for index, (segment, found) in enumerate(zip(segments, segment_digests.hexdigests())):
                        if found != segment["coordinates_sha256"]:
                            errors.append(
                                f"coordinates of segment {index} ({segment.get('name')}) do not match the digest "
                                "of its record"
                            )
                    report["segments_checked"] = len(segments)
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
    repair: bool = False,
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

    ``repair`` archives a source that is damaged in one of the ways
    :class:`PdbTrajectory` lists instead of refusing it.  The entry holds the
    complete frames, its manifest has ``source_irregularities`` (what was
    wrong, the frames of the source against those kept, the bytes left out)
    and a warning, and the verification also checks that record against the
    file.  A source that needs no repair gives the same entry with and
    without it, with no such key; an existing entry is skipped as described
    above whether or not it was repaired.

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
            existing = load_manifest(final) if final.exists() else None
            other = existing["source"]["folder"] if existing is not None else source_dir.name
        except (VerificationError, KeyError, TypeError):
            existing, other = None, source_dir.name   # unreadable entry: replaced below
        if existing is not None and segments_of(existing):
            raise ArchiveError(
                f"the entry of this GLYCAM name holds {len(segments_of(existing))} segments "
                "(lib/archive_segments.py); the conversion of one source folder does not replace it "
                "(force does)"
            )
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
        source = PdbTrajectory(inputs["pdb"], block_bytes=block_bytes, batch_bytes=batch_bytes, repair=repair)
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
        irregularities = source.irregularities
        if irregularities is not None:
            warnings.append(
                "the source is irregular and was archived with repair: "
                f"{irregularities['frames_kept']} of its {irregularities['frames_in_source']} frames are in the "
                f"archive, {irregularities['bytes_ignored']} of its bytes are not (source_irregularities)"
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
        }
        if irregularities is not None:   # a regular source has no such key: its manifest is what it always was
            manifest["source_irregularities"] = irregularities
        manifest.update({
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
        })
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


def repair_summary(manifest: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The short form of ``source_irregularities`` for an index or a report; None for a regular entry.

    An entry with ``segments`` keeps the record of a damaged source in the
    segment that came from it; the short form then adds up the segments that
    have one (``frames_in_source`` counts the frames of those sources only)
    and names them under ``segments``.
    """
    irregularities = manifest.get("source_irregularities")
    if irregularities is None:
        repaired = [
            (index, segment["source_irregularities"]) for index, segment in enumerate(segments_of(manifest))
            if isinstance(segment, dict) and segment.get("source_irregularities") is not None
        ]
        if not repaired:
            return None
        counts: Counter = Counter()
        for _, record in repaired:
            counts.update(record["counts"])
        return {
            "frames_in_source": sum(record["frames_in_source"] for _, record in repaired),
            "frames_dropped": sum(record["frames_dropped"] for _, record in repaired),
            "bytes_ignored": sum(record["bytes_ignored"] for _, record in repaired),
            "truncated": any(record["truncated"] for _, record in repaired),
            "counts": dict(sorted(counts.items())),
            "segments": [index for index, _ in repaired],
        }
    return {
        "frames_in_source": irregularities["frames_in_source"],
        "frames_dropped": irregularities["frames_dropped"],
        "bytes_ignored": irregularities["bytes_ignored"],
        "truncated": irregularities["truncated"],
        "counts": irregularities["counts"],
    }


def build_index(archive_dir: PathLike) -> Tuple[Dict[str, Any], List[str]]:
    """Index of every entry of ``archive_dir`` and the problems met while reading.

    ``repaired`` names the entries that were archived with repair; the
    record of each of them, and of no other entry, has ``repaired`` with the
    short form of its ``source_irregularities``.
    """
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
            record = {
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
            }
            repaired = repair_summary(manifest)
            if repaired is not None:
                record["repaired"] = repaired
            records.append(record)
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
        "repaired": [record["glycam_name"] for record in records if "repaired" in record],
        "records": records,
    }
    return index, problems


def write_index(archive_dir: PathLike) -> Tuple[Dict[str, Any], List[str]]:
    """Write ``index.json`` for the whole archive; returns ``(index, problems)``."""
    index, problems = build_index(archive_dir)
    _write_json_atomic(Path(archive_dir) / INDEX_FILE, index)
    return index, problems
