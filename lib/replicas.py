"""Replica diagnostics: how well the independent simulations of an entry agree.

An entry is made of independent simulations (replicas, runs).  This module
compares them on what the analysis already has, without the trajectory: the
Level 1 cluster of every retained frame, its first two principal components
and its glycosidic torsions.  It answers, for every entry, whether its runs
agree and whether it needs more sampling, and writes the answer to
``replicas.json``.

The measurements are those of the sampling-robustness analysis of the thesis
(``figures/GlycoShape/scripts/robustness_data.py`` of the thesis repository,
section "How reproducible the sampling is" of the GlycoShape chapter), made a
part of the pipeline: the same definitions, grids and thresholds, so that the
numbers of an entry are the numbers of the thesis for it.

Definitions (all on the retained frames of a replica, i.e. the rows of
``pca.csv`` and ``torsion_glycosidic.csv``):

* torsion histogram   36 bins of 10 degrees over [-180, 180)
* JSD                 Jensen-Shannon divergence, base 2 (0 identical, 1 disjoint), between two histograms
* JSD of a linkage    the largest JSD among its torsions (phi, psi and, where there is one, omega)
* population          fraction of the frames of a replica in each Level 1 cluster (column ``cluster``)
* TV                  total variation distance between two population vectors, 0.5 * sum |p - q|: the
                      fraction of frames that would have to change cluster to turn one into the other
* overlap             sum over the cells of a 25 x 25 grid on (PC1, PC2) of min(P, Q) for the frame
                      fractions P, Q of two replicas (1 identical, 0 disjoint); the grid spans the 0.5th to
                      99.5th percentile of all frames of the entry along each axis, frames outside fall into
                      the edge cells
* full-length replica at least 95% of the length of the longest replica of the entry (in frames; in
                      simulated time when every replica records its time per frame), of which at least a
                      quarter of the frames passed the frame-quality filter
* between replicas    mean (and maximum) of a measure over all pairs of full-length replicas
* halves              first against second half of the same replica ("within"), and first half of one
                      replica against the second half of another ("across"): the same amount of data
* floor               the same between-replica measure after the frames of the entry were dealt at random
                      to its replicas: what uncorrelated frames of one distribution give for this many frames
* rotamer             a third of the circle of the stored angle, [0, 120), [120, 240) or [240, 360) degrees,
                      centred on +60, 180 and -60.  Which rotamer of omega a third is depends on the torsion
                      definitions of the entry (``torsions.conventions.version`` of ``info.json``,
                      :func:`conventions_version`): version 2 stores the IUPAC omega O5-C5-C6-O6 (O7-C7-C8-O8
                      for 2-8), so the thirds are gt, tg and gg; version 1 (entries analysed before October
                      2026) stored C4-C5-C6-O6 (C6-C7-C8-O8), about 120 degrees more, so they are gg, gt and
                      tg.  Every omega carries ``rotamer_centres``, the centre of each third on the IUPAC
                      scale, so that the occupancies of the two versions can be set side by side
* transitions         changes of rotamer along a replica, counted between the cores of the rotamers
                      (within 40 degrees of 60, 180 and -60 degrees), so that fluctuation across a border
                      does not count

Replica boundaries (:func:`replica_boundaries`)
-----------------------------------------------
An entry made by ``scripts/make_entry.py`` lists its replicas as ``segments``
in the archive manifest: they are *recorded*.  An entry converted from one
``simulation.pdb`` of the old database has no segments; its runs are
recovered as the evidence audit of the thesis recovered them
(``audit_entries.py``, ``runs_of``; ``robustness_data.py``, ``segment_all``):

``recorded``  the source file recorded the runs: an ``END`` or ``CRYST1`` record stands between two
              frames (``records``: GROMACS wrote one file per run and the files were joined), or the
              MODEL numbers restart and the frames are restarts x largest number (``model_numbers``);
``single``    one block of 10,001 frames with an unbroken MODEL count: one run of the recipe;
``assumed``   no record; the frames are cut into equal runs where the class test of the thesis supports
              that division for all entries of that frame count together (:data:`ASSUMED_DIVISIONS`,
              :func:`class_test`).  Supported for the class, not for the single entry;
``none``      not determinable: the entry is one replica of unknown composition.

The stored entry inside an extended entry (the segment with ``carried_from``
and no segment of its own) is divided the same way from its own ``source``.

Verdict (:func:`verdict`)
-------------------------
``ok``                         two or more full-length replicas and none of the reasons below;
``more_sampling_suggested``    at least one reason;
``more_sampling_needed``       the first three reasons at once: the rule by which the thesis lists
                               "the first candidates for more sampling" (``worst_cases``);
``not_assessable``             fewer than two full-length replicas, or replicas that cannot be told apart.

Reasons, with their default thresholds (all parameters of :class:`Parameters`):

``population_tv``          mean TV between replicas >= 0.5 (thesis criterion 1);
``cluster_missing``        a Level 1 cluster with >= 10% of the pooled frames has < 1% of the frames of
                           a replica (thesis criterion 2);
``rotamer_missing``        a torsion of a linkage has a rotamer with >= 25% of the frames of one replica
                           and <= 2.5% of another (thesis criterion 3);
``omega_never_changes``    the omega of a 1-6 (or sialic acid 2-6) linkage changes rotamer fewer than once
                           in a full-length replica: that replica reports only the rotamer it started in
                           (thesis: "in 30 of 1,708 runs it never changes").

The file
--------
``replicas.json`` (schema ``glycoshape-replica-diagnostics``, version 1) is a
function of the entry and the parameters only (no time stamp), so the same
analysis gives the same bytes.  Besides the numbers it holds what a web page
needs to draw the overlap panel of the thesis without the trajectory: for
every replica the outline that encloses 80% of its frames on (PC1, PC2), the
pooled landscape at three levels, the 25 x 25 overlap grid as counts, and the
histograms of the torsions whose replicas agree least.

It is written into the static-ready folder of a result as
``output/replicas.json`` (:data:`STATIC_PATH`) and travels in the bundle
there.  Whether it becomes part of the published ``static/GSxxxxx/`` folder
or is kept by the publisher for the administrators only is one setting:
:func:`public_placement` (``GLYCOSHAPE_REPLICAS_PUBLIC``, default off: admin
only, stored at ``<state prefix>/<static prefix>/replicas/GSxxxxx.json``).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, "os.PathLike[str]"]

SCHEMA = "glycoshape-replica-diagnostics"
SCHEMA_VERSION = 1
FILE_NAME = "replicas.json"
#: Where the file is in a static-ready folder, and in ``static/GSxxxxx/`` when it is published.
STATIC_PATH = "output/replicas.json"
#: The setting that publishes the file with the entry instead of keeping it for the administrators.
PUBLIC_ENV = "GLYCOSHAPE_REPLICAS_PUBLIC"
#: Folder of the administrators' copies under ``<state prefix>/<static prefix>/``.
ADMIN_FOLDER = "replicas"

OK = "ok"
SUGGESTED = "more_sampling_suggested"
NEEDED = "more_sampling_needed"
NOT_ASSESSABLE = "not_assessable"
VERDICTS = (OK, SUGGESTED, NEEDED, NOT_ASSESSABLE)

RECORDED, SINGLE, ASSUMED, NONE = "recorded", "single", "assumed", "none"

#: Equal runs of the old entries without a run record, by frame count: ``{frames: frames per run}``.
#: The divisions the class test of the thesis accepted (``robustness.json`` -> ``runs.assumed.classes``):
#: entries of that frame count, joint z of the mean boundary rank.
ASSUMED_DIVISIONS: Dict[int, int] = {
    8000: 1000, 9000: 1000, 30000: 10000, 30003: 10001, 40000: 10000, 40004: 10001,
    45000: 5000, 50005: 10001, 60006: 10001, 80008: 10001, 90000: 10000,
}
ASSUMED_DIVISION_EVIDENCE: Dict[int, Dict[str, float]] = {
    8000: {"entries": 5, "joint_z": 10.2}, 9000: {"entries": 14, "joint_z": 17.7},
    30000: {"entries": 16, "joint_z": 7.7}, 30003: {"entries": 157, "joint_z": 14.3},
    40000: {"entries": 11, "joint_z": 8.8}, 40004: {"entries": 105, "joint_z": 26.7},
    45000: {"entries": 4, "joint_z": 9.6}, 50005: {"entries": 17, "joint_z": 9.0},
    60006: {"entries": 27, "joint_z": 18.5}, 80008: {"entries": 48, "joint_z": 31.5},
    90000: {"entries": 9, "joint_z": 14.3},
}

# GLYCAM residue codes (second letter; lower case is the L sugar) for the labels of linkages
_SUGAR = {"A": "Ara", "D": "Lyx", "R": "Rib", "X": "Xyl", "N": "All", "E": "Alt", "L": "Gal", "G": "Glc", "K": "Gul",
          "I": "Ido", "M": "Man", "T": "Tal", "C": "Fru", "P": "Psi", "B": "Sor", "J": "Tag", "F": "Fuc", "Q": "Qui",
          "H": "Rha", "O": "GalA", "Z": "GlcA", "U": "IdoA", "V": "GalNAc", "Y": "GlcNAc", "W": "ManNAc"}
_SPECIAL = {"SA": "Neu5Ac", "GL": "Neu5Gc", "KN": "Kdn", "YS": "GlcNS"}
_SIALIC = ("SA", "GL", "KN")
#: Names of the omega rotamers of a 1-6 (or sialic acid 2-6) linkage by their IUPAC angle O5-C5-C6-O6.
IUPAC_OMEGA_ROTAMERS = {-60.0: "gg", 60.0: "gt", 180.0: "tg"}
#: Centres of the three rotamer thirds of a stored angle ([0, 120), [120, 240), [240, 360)).
STORED_CENTRES = (60.0, 180.0, -60.0)
#: The omega rotamers of the thirds in a version 1 entry (omega C4-C5-C6-O6); kept for older readers.
OMEGA_ROTAMERS = ("gg", "gt", "tg")


class ReplicaError(Exception):
    """The tables of an entry cannot be compared (they disagree with each other or with the trajectory)."""


@dataclass(frozen=True)
class Parameters:
    """Grids, rules and verdict thresholds.  The defaults are those of the thesis analysis."""

    # measures (robustness_data.py: BINS, GRID, NSLICE, MIN_RUN_FRACTION, MIN_RETAINED, RUN_FRAMES)
    histogram_bins: int = 36
    pca_grid: int = 25
    pca_percentiles: Tuple[float, float] = (0.5, 99.5)
    time_slices: int = 50
    min_run_fraction: float = 0.95
    min_retained: float = 0.25
    single_run_frames: int = 10001
    rotamer_core_degrees: float = 40.0
    # verdict (robustness_data.py worst_cases; chapter, "How reproducible the sampling is")
    min_full_replicas: int = 2
    population_tv: float = 0.5
    cluster_pooled_min: float = 0.10
    cluster_replica_max: float = 0.01
    rotamer_present_min: float = 0.25
    rotamer_absent_max: float = 0.025
    omega_min_transitions: int = 1
    omega_classes: Tuple[str, ...] = ("1-6", "sia 2-6")
    # drawing (fig_robustness.py: 64 x 64 grid one fifth wider, Gaussian of 1.3 cells, 80% of the frames)
    landscape_grid: int = 64
    landscape_margin: float = 0.2
    landscape_sigma: float = 1.3
    landscape_mass: float = 0.8
    landscape_pooled_masses: Tuple[float, ...] = (0.6, 0.85, 0.97)
    histogram_torsions: int = 4

    def describe(self) -> Dict[str, Any]:
        return {key: (list(value) if isinstance(value, tuple) else value) for key, value in asdict(self).items()}


#: Where each verdict threshold comes from.
THRESHOLD_SOURCES = {
    "population_tv": "thesis worst_cases criterion 1: mean between-run TV of 0.5 or more (31 of 474 entries)",
    "cluster_missing": "thesis worst_cases criterion 2: a Level 1 cluster with at least 10% of the pooled frames "
                       "has less than 1% of the frames of one run (98 of 474)",
    "rotamer_missing": "thesis worst_cases criterion 3: a rotamer that holds at least 25% of one run and at "
                       "most 2.5% of another (105 of 474)",
    "omega_never_changes": "thesis chapter: the omega of a 1-6 linkage never changes rotamer in 30 of 1,708 "
                           "runs, and such a run reports only the rotamer in which it started",
    "more_sampling_needed": "thesis worst_cases: all three criteria at once, 19 of 474 entries, 'the first "
                            "candidates for more sampling'",
}


def public_placement(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Whether ``replicas.json`` is published with the entry (``GLYCOSHAPE_REPLICAS_PUBLIC`` = 1, true, yes, on).

    Off by default: the file is kept for the administrators
    (:func:`admin_key`) and is not part of ``static/GSxxxxx/``.
    """
    value = (os.environ if environ is None else environ).get(PUBLIC_ENV, "")
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def admin_key(state_prefix: str, static_prefix: str, glycoshape_id: str) -> str:
    """Key of the administrators' copy: ``<state>/<static>/replicas/GSxxxxx.json`` (outside the published prefix)."""
    return f"{state_prefix.strip('/')}/{static_prefix.strip('/')}/{ADMIN_FOLDER}/{glycoshape_id}.json"


# ---------------------------------------------------------------------------
# The per-frame tables of an analysed entry
# ---------------------------------------------------------------------------
@dataclass
class FrameTables:
    """What the analysis has per retained frame (row), as the thesis read it from the static tables.

    ``source_frame`` is the frame of the trajectory a row was computed from (rows of frames that the
    frame-quality filter removed are missing); ``cluster`` the Level 1 cluster; ``pcs`` at least PC1 and
    PC2 (PC3 is used where an entry has no glycosidic torsion); ``torsions`` the glycosidic torsions in
    degrees, one column per name of ``torsion_names``; ``torsion_metadata`` the ``torsions.metadata``
    records of ``info.json`` (``canonical_name``, ``atom_names``, ``residue_ids``, ``residue_names``).
    """

    source_frame: np.ndarray
    cluster: np.ndarray
    pcs: np.ndarray
    torsions: np.ndarray
    torsion_names: List[str]
    torsion_metadata: List[Dict[str, Any]] = field(default_factory=list)
    #: ``torsions.conventions.version`` of the entry (:func:`conventions_version`): 1 before October 2026
    #: (psi with C(x-1) or C(x+1), omega C4-C5-C6-O6), 2 the IUPAC definitions.
    conventions_version: int = 1

    def __post_init__(self) -> None:
        self.source_frame = np.asarray(self.source_frame, dtype=np.int64)
        self.cluster = np.asarray(self.cluster).astype(np.int64)
        self.pcs = np.asarray(self.pcs, dtype=np.float32)
        torsions = np.asarray(self.torsions, dtype=np.float32)
        self.torsions = np.zeros((len(self.source_frame), 0), np.float32) if torsions.size == 0 else torsions
        self.torsion_names = [str(name) for name in self.torsion_names]
        n = len(self.source_frame)
        if self.cluster.shape != (n,) or self.pcs.ndim != 2 or len(self.pcs) != n or self.pcs.shape[1] < 2:
            raise ReplicaError("source_frame, cluster and the principal components do not have one row per frame")
        if self.torsions.ndim != 2 or len(self.torsions) != n or self.torsions.shape[1] != len(self.torsion_names):
            raise ReplicaError("the torsion table does not have one row per frame and one column per name")
        if n and np.any(np.diff(self.source_frame) <= 0):
            raise ReplicaError("source_frame does not increase")
        if n and self.source_frame[0] < 0:
            raise ReplicaError("source_frame is negative")


_NON_TORSION_COLUMNS = ("frame", "source_frame", "cluster", "center")


def conventions_version(info: Mapping[str, Any]) -> int:
    """The version of the torsion definitions of an entry: ``torsions.conventions.version`` of ``info.json``.

    Entries analysed before the version was written (no ``conventions``, or the text conventions of
    early October 2026) are version 1: omega C4-C5-C6-O6 and C6-C7-C8-O8.
    """
    torsions = info.get("torsions") if isinstance(info, Mapping) else None
    conventions = torsions.get("conventions") if isinstance(torsions, Mapping) else None
    version = conventions.get("version") if isinstance(conventions, Mapping) else None
    return int(version) if isinstance(version, int) and not isinstance(version, bool) and version >= 1 else 1


def _wrap_centre(angle: float) -> float:
    angle = float(angle) % 360.0
    return angle - 360.0 if angle > 180.0 else angle


def omega_rotamer_centres(atom_names: Sequence[str], version: int) -> Tuple[float, float, float]:
    """The IUPAC angle (O5-C5-C6-O6, O7-C7-C8-O8 ...) at the centre of each third of a stored omega.

    Version 2 stores the IUPAC omega: +60, 180, -60.  Version 1 stored the angle with the carbon two
    bonds in (C4-C5-C6-O6, C6-C7-C8-O8), about 120 degrees more: -60, +60, 180.  The atoms of the
    torsion, when known, must agree with the version (an oxygen first in version 2, a carbon in 1).
    """
    iupac = int(version) >= 2
    first = str(atom_names[0]) if atom_names else ""
    if first[:1] in ("O", "C") and (first[:1] == "O") != iupac:
        raise ReplicaError(
            f"an omega with the atoms {'-'.join(map(str, atom_names))} in an entry of torsion definitions "
            f"version {version}: the metadata and the conventions of info.json disagree"
        )
    shift = 0.0 if iupac else -120.0
    return tuple(_wrap_centre(centre + shift) for centre in STORED_CENTRES)  # type: ignore[return-value]


def _tables(pca, tor, info: Dict[str, Any]) -> FrameTables:
    """FrameTables from the PCA and glycosidic torsion tables (polars frames) and ``info.json``."""
    if "source_frame" not in pca.columns or "cluster" not in pca.columns:
        raise ReplicaError("the PCA table has no source_frame or no cluster column")
    if tor is not None:
        if tor.height != pca.height:
            raise ReplicaError(f"rows differ: PCA {pca.height}, torsions {tor.height}")
        for column in ("frame", "source_frame"):
            if column in tor.columns and column in pca.columns \
                    and not np.array_equal(pca[column].to_numpy(), tor[column].to_numpy()):
                raise ReplicaError(f"{column} differs between the PCA and the torsion table")
        if "cluster" in tor.columns and not np.array_equal(pca["cluster"].to_numpy(), tor["cluster"].to_numpy()):
            raise ReplicaError("the Level 1 cluster differs between the PCA and the torsion table")
        names = [c for c in tor.columns if c not in _NON_TORSION_COLUMNS]
    else:
        names = []
    values = tor.select(names).to_numpy().astype(np.float32) if names else np.zeros((pca.height, 0), np.float32)
    meta = [m for m in ((info.get("torsions") or {}).get("metadata") or []) if m.get("family") == "glycosidic"]
    components = [c for c in ("PC1", "PC2", "PC3") if c in pca.columns]
    return FrameTables(
        source_frame=pca["source_frame"].to_numpy(),
        cluster=pca["cluster"].to_numpy(),
        pcs=pca.select(components).to_numpy().astype(np.float32),
        torsions=values,
        torsion_names=names,
        torsion_metadata=meta,
        conventions_version=conventions_version(info),
    )


def tables_from_static(folder: PathLike) -> FrameTables:
    """The tables of a static entry: ``output/pca.csv``, ``output/torsion_glycosidic.csv``, ``output/info.json``.

    ``folder`` is ``static/GSxxxxx/`` or the static-ready folder of a result.
    """
    import polars as pl

    output = Path(folder) / "output"
    pca = pl.read_csv(output / "pca.csv")
    path = output / "torsion_glycosidic.csv"
    tor = pl.read_csv(path, infer_schema_length=2000) if path.is_file() else None
    info = json.loads((output / "info.json").read_text(encoding="utf-8"))
    return _tables(pca, tor, info)


def tables_from_process(process_dir: PathLike) -> FrameTables:
    """The same tables from a process folder: ``embedding/exports/*.parquet`` and ``output/info.json``."""
    import polars as pl

    process_dir = Path(process_dir)
    exports = process_dir / "embedding" / "exports"
    pca = pl.read_parquet(exports / "pca.parquet")
    path = exports / "torsion_glycosidic.parquet"
    tor = pl.read_parquet(path) if path.is_file() else None
    info = json.loads((process_dir / "output" / "info.json").read_text(encoding="utf-8"))
    return _tables(pca, tor, info)


# ---------------------------------------------------------------------------
# Which frames belong to which replica
# ---------------------------------------------------------------------------
@dataclass
class Replica:
    """One replica: frames ``first_frame`` to ``first_frame + frames`` of the trajectory."""

    name: str
    first_frame: int
    frames: int
    basis: str
    recorded_by: Optional[str]
    time_per_frame_ps: Optional[float] = None
    time_from: Optional[str] = None
    segment: Optional[int] = None


@dataclass
class Boundaries:
    """The replicas of an entry and how they were found."""

    frames: int
    replicas: List[Replica]
    basis: str
    recorded_by: Optional[str]
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def starts(self) -> List[int]:
        return [replica.first_frame for replica in self.replicas]

    @property
    def lengths(self) -> List[int]:
        return [replica.frames for replica in self.replicas]


def runs_recorded_by(source: Mapping[str, Any], frames: int) -> Tuple[str, List[int]]:
    """Run boundaries that a converted ``simulation.pdb`` recorded; ``(kind, [frames per run])``.

    The evidence audit of the thesis (``audit_entries.py``, ``runs_of``):
    ``records`` an ``END`` or ``CRYST1`` record stands between two frames; ``one_block`` sequential MODEL
    numbers; ``wrapped_numbers`` MODEL numbers that wrap at 9999; ``model_numbers`` the numbers restart at
    1 for every run and frames = (restarts + 1) x largest number; ``unknown`` none of these.  Only
    ``records`` and ``model_numbers`` divide the frames.
    """
    between = source.get("records_between_frames") or {}
    ends = sorted(set((between.get("END") or {}).get("before_frame", []))
                  | set((between.get("CRYST1") or {}).get("before_frame", [])))
    cuts = [int(e) for e in ends if 0 < int(e) < frames]
    numbers = source.get("model_numbers") or {}
    if cuts:
        edges = [0] + cuts + [frames]
        return "records", [b - a for a, b in zip(edges[:-1], edges[1:])]
    if numbers.get("sequential"):
        return "one_block", [frames]
    if numbers.get("largest") == 9999:
        return "wrapped_numbers", [frames]
    if numbers.get("first") == 1 and (numbers.get("breaks") or 0) >= 1 and (numbers.get("largest") or 0) > 0 \
            and frames == (numbers["breaks"] + 1) * numbers["largest"]:
        return "model_numbers", [numbers["largest"]] * (numbers["breaks"] + 1)
    return "unknown", [frames]


def _divide_block(source: Mapping[str, Any], frames: int, first_frame: int, prefix: str,
                  time_per_frame_ps: Optional[float], time_from: Optional[str], parameters: Parameters,
                  divisions: Mapping[int, int], segment: Optional[int] = None) -> Tuple[List[Replica], Dict[str, Any]]:
    """The runs of one converted ``simulation.pdb`` (an old entry, or the stored part of an extended one)."""
    kind, runs = runs_recorded_by(source, frames)
    evidence: Dict[str, Any] = {"frames": frames, "first_frame": first_frame, "source_record": kind}
    if kind in ("records", "model_numbers"):
        basis, sizes = RECORDED, runs
    elif kind == "one_block" and frames == parameters.single_run_frames:
        basis, sizes = SINGLE, [frames]
    elif frames in divisions and frames % divisions[frames] == 0:
        basis, sizes = ASSUMED, [divisions[frames]] * (frames // divisions[frames])
        evidence["class_test"] = {"frames": frames, "run_frames": divisions[frames],
                                  **ASSUMED_DIVISION_EVIDENCE.get(frames, {})}
    else:
        basis, sizes = NONE, [frames]
        evidence["why"] = (f"{frames} frames without a run record, and no equal division of {frames} frames is "
                           "supported by the class test")
    replicas, position = [], first_frame
    for index, size in enumerate(sizes):
        name = f"{prefix}{index + 1}" if len(sizes) > 1 or basis != NONE else f"{prefix}all"
        replicas.append(Replica(name, position, int(size), basis, kind if basis in (RECORDED, SINGLE) else None,
                                time_per_frame_ps, time_from, segment))
        position += int(size)
    evidence["basis"] = basis
    return replicas, evidence


def replica_boundaries(manifest: Optional[Mapping[str, Any]], frames: Optional[int] = None, *,
                       time_per_frame_ps: Optional[float] = None, parameters: Optional[Parameters] = None,
                       divisions: Optional[Mapping[int, int]] = None) -> Boundaries:
    """The replicas of an archive entry from its manifest (see the module docstring).

    ``time_per_frame_ps`` is used for the parts that do not record it (a converted ``simulation.pdb``);
    ``frames`` stands for the manifest's count when there is no manifest (a ``simulation.pdb`` folder),
    which makes the entry one replica of unknown composition.
    """
    parameters = parameters or Parameters()
    divisions = ASSUMED_DIVISIONS if divisions is None else divisions
    hint_from = "entry length" if time_per_frame_ps is not None else None
    if manifest is None:
        if frames is None:
            raise ReplicaError("neither a manifest nor a number of frames")
        replica = Replica("all", 0, int(frames), NONE, None, time_per_frame_ps, hint_from)
        return Boundaries(int(frames), [replica], NONE, None,
                          {"why": "the input has no archive manifest, so no record of its runs"})
    total = int(manifest["frames"])
    segments = manifest.get("segments") if isinstance(manifest.get("segments"), list) else []
    if not segments:
        replicas, evidence = _divide_block(manifest.get("source") or {}, total, 0, "run_", time_per_frame_ps,
                                           hint_from, parameters, divisions)
        basis = replicas[0].basis
        return Boundaries(total, replicas, basis, replicas[0].recorded_by if basis in (RECORDED, SINGLE) else None,
                          evidence)

    replicas: List[Replica] = []
    parts: List[Dict[str, Any]] = []
    position = 0
    for index, segment in enumerate(segments):
        first, size = int(segment["first_frame"]), int(segment["frames"])
        if first != position:
            raise ReplicaError(f"segment {index} starts at frame {first}, expected {position}")
        position += size
        tpf = segment.get("time_per_frame_ps")
        tpf = float(tpf) if isinstance(tpf, (int, float)) and not isinstance(tpf, bool) and tpf > 0 else None
        carried = segment.get("carried_from") if isinstance(segment.get("carried_from"), dict) else None
        if carried is not None and carried.get("segment") is None and isinstance(segment.get("source"), dict):
            # the stored entry of an extension, itself one simulation.pdb: its runs as its source recorded them
            block, evidence = _divide_block(segment["source"], size, first, "stored_run_",
                                            tpf if tpf is not None else time_per_frame_ps,
                                            "segment" if tpf is not None else hint_from,
                                            parameters, divisions, segment=index)
            replicas += block
            parts.append({"segment": index, **evidence})
        else:
            replicas.append(Replica(str(segment.get("name") or f"segment_{index + 1}"), first, size, RECORDED,
                                    "segments", tpf, "segment" if tpf is not None else None, index))
            parts.append({"segment": index, "frames": size, "first_frame": first, "basis": RECORDED})
    if position != total:
        raise ReplicaError(f"the segments hold {position} frames, the manifest says {total}")
    bases = {replica.basis for replica in replicas}
    basis = NONE if NONE in bases else ASSUMED if ASSUMED in bases else SINGLE if len(replicas) == 1 else RECORDED
    return Boundaries(total, replicas, basis, "segments", {"segments": parts})


def step_percentile(tables: FrameTables) -> Optional[np.ndarray]:
    """Percentile rank (0..1) of the change from frame b-1 to frame b among all such changes of the entry.

    The change is the root mean square over the glycosidic torsions of the chord 2 sin(delta/2); for an
    entry without glycosidic torsions it is the distance in (PC1, PC2, PC3).  Index b is the trajectory
    frame; NaN where frame b-1 or b was removed by the frame-quality filter.  At a run boundary the glycan
    jumps from the end of one simulation to the start of another, so the rank there is high
    (``robustness_data.py``, ``step_percentile``).
    """
    sf = tables.source_frame
    if len(sf) < 3:
        return None
    if tables.torsions.shape[1]:
        t = np.radians(tables.torsions.astype(np.float64))
        d = np.sqrt((4 * np.sin(np.diff(t, axis=0) / 2) ** 2).mean(axis=1))
    else:
        d = np.linalg.norm(np.diff(tables.pcs[:, :3].astype(np.float64), axis=0), axis=1)
    ok = np.diff(sf) == 1
    if not ok.any():
        return None
    rank = np.empty(int(ok.sum()))
    rank[np.argsort(d[ok], kind="stable")] = (np.arange(ok.sum()) + 0.5) / ok.sum()
    out = np.full(int(sf[-1]) + 1, np.nan)
    out[sf[1:][ok]] = rank
    return out


def _divisions(frames: int):
    for k in range(2, 21):
        if frames % k == 0 and frames // k >= 500:
            yield k, frames // k


def class_test(groups: Mapping[int, Sequence[str]], percentiles: Mapping[str, Optional[np.ndarray]]) -> Dict[int, Dict[str, Any]]:
    """Equal runs for all entries of one frame count together (``robustness_data.py``, ``class_test``).

    ``groups`` maps a frame count to the entries of that count whose source recorded no runs;
    ``percentiles`` the :func:`step_percentile` of each.  For k runs of L = frames / k frames (every k from
    2 to 20 with L >= 500 a whole number) the ranks at frames L, 2L, ... are averaged over the n entries of
    that count (three or more).  A k is accepted when the mean rank at every one of its k - 1 positions
    exceeds 0.5 by 2.5 standard errors, sqrt(1/12n), and the mean over all positions by 5; the largest
    accepted k is taken.  Returns ``{frames: {"accepted": {runs, run_frames, ...} or None, "entries", "tried"}}``.
    """
    out: Dict[int, Dict[str, Any]] = {}
    for frames, members in sorted(groups.items()):
        P = {name: percentiles[name] for name in members if percentiles.get(name) is not None}
        best, tried = None, []
        if len(P) >= 3:
            n = min(len(p) for p in P.values())
            M = np.array([p[:n] for p in P.values()])
            for k, L in _divisions(frames):
                positions = [i * L for i in range(1, k)]
                if positions[-1] >= n:
                    continue
                at = M[:, positions]
                count = np.isfinite(at).sum(axis=0)
                if count.min() < 3:
                    continue
                mean = np.nanmean(at, axis=0)
                joint = (np.nanmean(at) - 0.5) / np.sqrt(1 / 12 / np.isfinite(at).sum())
                ok = bool(np.all(mean > 0.5 + 2.5 * np.sqrt(1 / 12 / count)) and joint > 5)
                tried.append({"runs": k, "run_frames": L, "mean_rank": [round(float(x), 3) for x in mean],
                              "joint_z": round(float(joint), 1), "accepted": ok})
                if ok:
                    best = tried[-1]
        out[int(frames)] = {"entries": len(members), "tested": len(P), "accepted": best, "tried": tried}
    return out


# ---------------------------------------------------------------------------
# Linkages
# ---------------------------------------------------------------------------
def _residue_label(code: str) -> str:
    code = code.strip()
    if len(code) == 3:
        if code[1:] in _SPECIAL:
            return _SPECIAL[code[1:]]
        if code[1].upper() in _SUGAR and code[2] in "ABDU":
            return _SUGAR[code[1].upper()] + ("f" if code[2] in "DU" else "")
    return code


def _anomer(code: str) -> str:
    code = code.strip()
    if len(code) != 3:
        return ""
    if code[1:] in _SIALIC or code[1:] == "YS":
        return "α"
    return {"A": "α", "B": "β", "D": "α", "U": "β"}.get(code[2], "")


def linkages_of(metadata: Sequence[Mapping[str, Any]], names: Sequence[str],
                conventions_version: int = 1) -> List[Dict[str, Any]]:
    """Glycosidic linkages from the torsion metadata of ``info.json`` (``robustness_data.py``, ``linkages_of``).

    A linkage is the pair (donor, acceptor) of the canonical names ``d_a_phi``, ``d_a_psi`` and
    ``d_a_omega``.  Its class is ``sia`` (the donor is a sialic acid; subclass ``sia 2-3``, ``sia 2-6``,
    ``sia 2-8``, ``sia 2-9``), ``1-6`` (another sugar on O6: the linkage has omega), ``other``, or
    ``substituent`` (an omega alone, of a residue whose O6 carries a group that is not a sugar: not a
    linkage, and left out of the statistics of linkages; version 1 entries list it among the glycosidic
    torsions, version 2 entries apart).  A linkage with omega gets ``omega_rotamer_centres`` (the IUPAC
    angle of each rotamer third, :func:`omega_rotamer_centres`) and, for a 1-6 or sialic acid 2-6
    linkage, ``omega_rotamer_names`` (gg, gt, tg in the order of the thirds).
    """
    by = {m["canonical_name"]: m for m in metadata if "canonical_name" in m}
    links: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for column, name in enumerate(names):
        m = by.get(name)
        if m is None:
            continue
        try:
            d, a, kind = name.rsplit("_", 2)
        except ValueError:
            continue
        link = links.setdefault((d, a), {"donor": d, "acceptor": a, "torsions": {}})
        link["torsions"][kind] = column
        codes = dict(zip((str(i) for i in m.get("residue_ids", [])), m.get("residue_names", [])))
        atom_names = list(m.get("atom_names", []))
        if kind in ("phi", "psi"):
            link.setdefault("donor_code", codes.get(d, "?"))
            link.setdefault("acceptor_code", codes.get(a, "?"))
            oxygen = [n for n, i in zip(atom_names, m.get("residue_ids", [])) if n.startswith("O") and n != "O5" and str(i) == a]
            if kind == "phi" and len(atom_names) == 4:
                link["anomeric_carbon"], link["ring_oxygen"] = atom_names[1], atom_names[0]
                link["position"] = "".join(ch for ch in atom_names[2] if ch.isdigit())
            elif oxygen:
                link.setdefault("position", "".join(ch for ch in oxygen[0] if ch.isdigit()))
        elif atom_names:
            link.setdefault("position", "".join(ch for ch in atom_names[-1] if ch.isdigit()))
            link.setdefault("acceptor_code", codes.get(a, "?"))
        if kind == "omega":
            link["omega_rotamer_centres"] = omega_rotamer_centres(atom_names, conventions_version)
    out = []
    for link in links.values():
        donor, acceptor = link.get("donor_code", "?"), link.get("acceptor_code", "?")
        position = link.get("position", "?")
        sugar = "phi" in link["torsions"] or "psi" in link["torsions"]
        sialic = donor[1:] in _SIALIC or (link.get("anomeric_carbon") == "C2" and link.get("ring_oxygen") == "O6")
        carbon = "2" if sialic or donor[1:2].upper() == "C" else "1"
        if not sugar:
            cls, sub = "substituent", f"omega of a residue substituted at O{position}"
        elif sialic:
            cls, sub = "sia", f"sia 2-{position}"
        elif position == "6":
            cls, sub = "1-6", "1-6"
        else:
            cls, sub = "other", "other"
        link["class"], link["subclass"], link["positions"] = cls, sub, f"{carbon}-{position}"
        link["donor_code"], link["acceptor_code"] = donor, acceptor
        link["label"] = (f"{_residue_label(donor)} {_anomer(donor)}{carbon}–{position} {_residue_label(acceptor)}" if sugar
                         else f"{_residue_label(acceptor)} O{position} substituent")
        if "omega_rotamer_centres" in link and link["positions"].endswith("-6") and cls in ("1-6", "sia"):
            link["omega_rotamer_names"] = [IUPAC_OMEGA_ROTAMERS[c] for c in link["omega_rotamer_centres"]]
        out.append(link)
    return out


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------
def jsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence (base 2) along the last axis of two arrays of counts."""
    p = p / np.maximum(p.sum(-1, keepdims=True), 1e-300)
    q = q / np.maximum(q.sum(-1, keepdims=True), 1e-300)
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(p > 0, p * np.log2(p / m), 0.0).sum(-1)
        b = np.where(q > 0, q * np.log2(q / m), 0.0).sum(-1)
    return 0.5 * (a + b)


def tv(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Total variation distance along the last axis of two arrays of counts."""
    p = p / np.maximum(p.sum(-1, keepdims=True), 1e-300)
    q = q / np.maximum(q.sum(-1, keepdims=True), 1e-300)
    return 0.5 * np.abs(p - q).sum(-1)


def overlap(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Sum of min(P, Q) along the last axis of two arrays of counts, each normalised."""
    p = p / np.maximum(p.sum(-1, keepdims=True), 1e-300)
    q = q / np.maximum(q.sum(-1, keepdims=True), 1e-300)
    return np.minimum(p, q).sum(-1)


def transitions(angles: np.ndarray, core: float = 40.0) -> int:
    """Changes of rotamer along a series of angles, counted between the cores of the three rotamers."""
    a = angles % 360.0
    state = np.full(len(a), -1)
    for k, centre in enumerate((60.0, 180.0, 300.0)):
        state[np.abs(a - centre) <= core] = k
    visited = state[state >= 0]
    return int((np.diff(visited) != 0).sum()) if len(visited) > 1 else 0


def _rotamer_state(torsions: np.ndarray) -> np.ndarray:
    return (np.floor((torsions % 360.0) / 120.0).astype(int)) % 3


def _histograms(run, slc, n_runs, tors, cluster, cell, p: Parameters, K: int):
    """Counts per replica and time slice: torsions [R, S, T, bins], clusters [R, S, K], PCA cells [R, S, grid**2]."""
    S, B, G = p.time_slices, p.histogram_bins, p.pca_grid
    base = run * S + slc
    T = tors.shape[1]
    Ht = np.zeros((n_runs * S, T, B))
    if T:
        b = np.clip(np.floor((tors + 180.0) / (360.0 / B)).astype(int), 0, B - 1)
        for j in range(T):
            Ht[:, j, :] = np.bincount(base * B + b[:, j], minlength=n_runs * S * B).reshape(-1, B)
    assigned = cluster >= 0
    Hc = np.bincount(base[assigned] * K + cluster[assigned], minlength=n_runs * S * K).reshape(n_runs, S, K).astype(float)
    Hp = np.bincount(base * G * G + cell, minlength=n_runs * S * G * G).reshape(n_runs, S, -1).astype(float)
    return Ht.reshape(n_runs, S, T, B), Hc, Hp


def floor_seed(glycoshape_id: Optional[str] = None, glycam_name: Optional[str] = None) -> int:
    """Seed of the random deal of the floor: the GlycoShape number (as in the thesis), else from the name.

    An analysis on the node has no GlycoShape ID yet, so its floor is one other random deal than the one
    of the same entry diagnosed after publication: the two differ by the noise of the deal only.
    """
    if glycoshape_id and str(glycoshape_id)[2:].isdigit():
        return int(str(glycoshape_id)[2:])
    digest = hashlib.sha256(str(glycam_name or "").encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def measure(tables: FrameTables, boundaries: Boundaries, parameters: Optional[Parameters] = None,
            seed: int = 0) -> Dict[str, Any]:
    """Every measure of one entry at full precision (``robustness_data.py``, ``measure``, extended per replica).

    Per replica (every replica, on all its retained frames): retained frames, population, rotamer
    occupancy and transitions of every torsion, histograms.  Between and within replicas, and the floor:
    over the full-length replicas only, as in the thesis.
    """
    p = parameters or Parameters()
    sf, cluster_all = tables.source_frame, tables.cluster
    starts = np.asarray(boundaries.starts, dtype=np.int64)
    length = np.asarray(boundaries.lengths, dtype=np.int64)
    n_rep = len(starts)
    if len(sf) and sf[-1] >= boundaries.frames:
        raise ReplicaError(f"a row is of frame {int(sf[-1])}, the trajectory has {boundaries.frames} frames")
    rid = np.searchsorted(starts, sf, side="right") - 1
    retained = np.bincount(rid, minlength=n_rep)
    tpf = [replica.time_per_frame_ps for replica in boundaries.replicas]
    by_time = all(t is not None for t in tpf) and len({t for t in tpf}) > 1
    size = length * np.asarray(tpf, dtype=float) if by_time else length
    full = (size >= p.min_run_fraction * size.max()) & (retained >= p.min_retained * length)
    K = max(5, int(cluster_all.max()) + 1) if len(cluster_all) else 5
    T = tables.torsions.shape[1]
    out: Dict[str, Any] = {
        "rows": int(len(sf)), "replicas": n_rep, "run_frames": [int(x) for x in length],
        "retained": [int(x) for x in retained], "full": [bool(x) for x in full], "full_replicas": int(full.sum()),
        "full_length_by": "simulated time" if by_time else "frames", "clusters": K,
        "unassigned_rows": int((cluster_all < 0).sum()),
    }
    links = linkages_of(tables.torsion_metadata, tables.torsion_names, tables.conventions_version)
    out["linkages"] = links
    out["conventions_version"] = int(tables.conventions_version)

    # every replica on all its rows
    ok = cluster_all >= 0
    counts = np.bincount(rid[ok] * K + cluster_all[ok], minlength=n_rep * K).reshape(n_rep, K).astype(float)
    out["replica_population"] = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    out["replica_assigned"] = counts.sum(1)
    if T:
        state = _rotamer_state(tables.torsions)
        rot = np.zeros((n_rep, T, 3))
        for j in range(T):
            rot[:, j, :] = np.bincount(rid * 3 + state[:, j], minlength=n_rep * 3).reshape(n_rep, 3)
        out["replica_rotamers"] = rot / np.maximum(rot.sum(-1, keepdims=True), 1)
        out["replica_transitions"] = [[transitions(tables.torsions[rid == i, j], p.rotamer_core_degrees)
                                       for i in range(n_rep)] for j in range(T)]
        B = p.histogram_bins
        b = np.clip(np.floor((tables.torsions + 180.0) / (360.0 / B)).astype(int), 0, B - 1)
        out["replica_histograms"] = np.stack(
            [np.bincount(rid * B + b[:, j], minlength=n_rep * B).reshape(n_rep, B) for j in range(T)], axis=1)
    lo, hi = np.percentile(tables.pcs[:, :2], list(p.pca_percentiles), axis=0)
    out["pc_lo"], out["pc_hi"] = lo, hi
    G = p.pca_grid
    ij_all = np.clip(((tables.pcs[:, :2] - lo) / np.maximum(hi - lo, 1e-9) * G).astype(int), 0, G - 1)
    cell_all = ij_all[:, 0] * G + ij_all[:, 1]
    out["replica_cells"] = np.bincount(rid * G * G + cell_all, minlength=n_rep * G * G).reshape(n_rep, G * G)
    out["rid"] = rid
    if full.sum() < 1:
        return out

    renumber = np.full(n_rep, -1)
    renumber[full] = np.arange(full.sum())
    use = full[rid]
    r = renumber[rid[use]]
    R = int(full.sum())
    S = p.time_slices
    pos = (sf[use] - starts[rid[use]]) / np.maximum(length[rid[use]] - 1, 1)
    slc = np.minimum((pos * S).astype(int), S - 1)
    tors, cluster, cell = tables.torsions[use], cluster_all[use], cell_all[use]
    Ht, Hc, Hp = _histograms(r, slc, R, tors, cluster, cell, p, K)
    half = S // 2

    pops = Hc.sum(1) / np.maximum(Hc.sum((1, 2))[:, None], 1e-300)
    pooled = Hc.sum((0, 1)) / max(Hc.sum(), 1e-300)
    out["full_index"] = [int(i) for i in np.flatnonzero(full)]
    out["population"] = pops
    out["pooled"] = pooled
    out["order"] = [int(x) for x in np.argsort(-pooled)]
    out["within"] = {
        "population_tv": tv(Hc[:, :half].sum(1), Hc[:, half:].sum(1)),
        "overlap": overlap(Hp[:, :half].sum(1), Hp[:, half:].sum(1)),
    }
    if T:
        out["within"]["jsd"] = jsd(Ht[:, :half].sum(1), Ht[:, half:].sum(1))           # [R, T]
        state = _rotamer_state(tors)
        rot = np.zeros((R, T, 3))
        for j in range(T):
            rot[:, j, :] = np.bincount(r * 3 + state[:, j], minlength=R * 3).reshape(R, 3)
        out["rotamers"] = rot / rot.sum(-1, keepdims=True)
    if R >= 2:
        pairs = list(combinations(range(R), 2))
        a, b = np.array([x[0] for x in pairs]), np.array([x[1] for x in pairs])
        between: Dict[str, Any] = {"pairs": [(out["full_index"][i], out["full_index"][j]) for i, j in pairs]}
        between["population_tv"] = tv(Hc.sum(1)[a], Hc.sum(1)[b])
        between["overlap"] = overlap(Hp.sum(1)[a], Hp.sum(1)[b])
        between["population_absdiff_max"] = float(np.abs(pops[a] - pops[b]).max())
        between["population_sd"] = pops.std(axis=0, ddof=1)
        between["top_cluster_same"] = bool(len(set(int(np.argmax(x)) for x in pops)) == 1)
        if T:
            between["jsd"] = jsd(Ht.sum(1)[a], Ht.sum(1)[b])                               # [pairs, T]
            between["rotamer_absdiff_max"] = np.abs(out["rotamers"][a] - out["rotamers"][b]).max((0, 2))
        ordered = [(i, k) for i in range(R) for k in range(R) if i != k]
        oa, ob = np.array([x[0] for x in ordered]), np.array([x[1] for x in ordered])
        across = {"population_tv": float(np.mean(tv(Hc[oa, :half].sum(1), Hc[ob, half:].sum(1)))),
                  "overlap": float(np.mean(overlap(Hp[oa, :half].sum(1), Hp[ob, half:].sum(1))))}
        if T:
            across["jsd"] = jsd(Ht[oa, :half].sum(1), Ht[ob, half:].sum(1)).mean(0)
        out["across"] = across
        # floor: the frames dealt at random to the replicas (and time slices), at the full length
        deal = np.random.default_rng(seed).permutation(len(r))
        Ht0, Hc0, Hp0 = _histograms(r[deal], slc[deal], R, tors, cluster, cell, p, K)
        out["floor"] = {"seed": int(seed),
                        "population_tv": float(tv(Hc0.sum(1)[a], Hc0.sum(1)[b]).mean()),
                        "overlap": float(overlap(Hp0.sum(1)[a], Hp0.sum(1)[b]).mean())}
        if T:
            out["floor"]["jsd"] = jsd(Ht0.sum(1)[a], Ht0.sum(1)[b]).mean(0)
        out["between"] = between
    return out


def _round4(values: np.ndarray) -> np.ndarray:
    """Values rounded to four decimals as ``round`` does it (the precision of the file and of the thesis)."""
    values = np.asarray(values, dtype=float)
    return np.array([round(float(x), 4) for x in values.ravel()]).reshape(values.shape)


def linkage_rows(m: Dict[str, Any], parameters: Optional[Parameters] = None) -> List[Dict[str, Any]]:
    """Per linkage (substituent omegas included, marked by their class): the measures of its torsions.

    ``jsd`` of a linkage is the largest mean JSD between replicas among its torsions; ``rotamer_missing``
    lists the torsions with a rotamer that holds at least ``rotamer_present_min`` of one full-length
    replica and at most ``rotamer_absent_max`` of another (thesis: ``rotamer_missing_in_a_run``).
    """
    p = parameters or Parameters()
    b = m.get("between")
    rows = []
    for link in m["linkages"]:
        torsions = {}
        for kind, j in link["torsions"].items():
            t: Dict[str, Any] = {"column": j}
            if "rotamers" in m:
                shown = _round4(m["rotamers"][:, j])
                hi, lo = shown.max(0), shown.min(0)
                t["rotamer_hi"], t["rotamer_lo"] = hi, lo
                t["missing"] = bool(any(h >= p.rotamer_present_min and l_ <= p.rotamer_absent_max for h, l_ in zip(hi, lo)))
                t["jsd_within"] = float(m["within"]["jsd"][:, j].mean())
            if b is not None:
                t.update({"jsd": float(b["jsd"][:, j].mean()), "jsd_max": float(b["jsd"][:, j].max()),
                          "jsd_floor": float(m["floor"]["jsd"][j]), "jsd_across": float(m["across"]["jsd"][j]),
                          "rotamer_absdiff": float(b["rotamer_absdiff_max"][j])})
            torsions[kind] = t
        row = {k: link[k] for k in ("donor", "acceptor", "label", "class", "subclass", "positions")}
        for key in ("omega_rotamer_centres", "omega_rotamer_names"):
            if key in link:
                row[key] = list(link[key])
        row["torsions"] = torsions
        if b is not None and torsions:
            kmax = max(torsions, key=lambda k: torsions[k]["jsd"])
            row.update({"jsd": torsions[kmax]["jsd"], "jsd_torsion": kmax,
                        "jsd_floor": max(t["jsd_floor"] for t in torsions.values()),
                        "jsd_within": max(t["jsd_within"] for t in torsions.values()),
                        "jsd_across": max(t["jsd_across"] for t in torsions.values()),
                        "rotamer_absdiff": max(t["rotamer_absdiff"] for t in torsions.values())})
        row["rotamer_missing"] = [k for k, t in torsions.items() if t.get("missing")]
        rows.append(row)
    return rows


def entry_row(m: Dict[str, Any], parameters: Optional[Parameters] = None) -> Dict[str, Any]:
    """The summary of an entry the thesis tabulated (``robustness_data.py``, ``_rows``), at full precision."""
    p = parameters or Parameters()
    row: Dict[str, Any] = {"runs": m["replicas"], "full_runs": m["full_replicas"]}
    links = linkage_rows(m, p)
    real = [link for link in links if link["class"] != "substituent"]
    row["has_1_6"] = any(link["class"] == "1-6" for link in real)
    row["has_sia"] = any(link["class"] == "sia" for link in real)
    row["linkages"] = len(real)
    b = m.get("between")
    if "within" in m:
        row["tv_within"] = float(np.mean(m["within"]["population_tv"]))
        row["overlap_within"] = float(np.mean(m["within"]["overlap"]))
    if b is None:
        return row
    # thresholds apply to the populations as the file shows them, to four decimals (as in the thesis)
    pops = _round4(m["population"][:, m["order"]])
    pooled = _round4(m["pooled"][m["order"]])
    se = pops.std(axis=0, ddof=1) / np.sqrt(len(pops))
    order = m["order"]
    absent = [{"cluster": int(order[k]), "pooled": float(pooled[k]), "replicas": [float(x) for x in pops[:, k]]}
              for k in range(len(pooled)) if pooled[k] >= p.cluster_pooled_min and pops[:, k].min() < p.cluster_replica_max]
    row.update({
        "tv": float(b["population_tv"].mean()), "tv_max": float(b["population_tv"].max()),
        "tv_floor": m["floor"]["population_tv"], "absdiff_max": b["population_absdiff_max"],
        "top_same": b["top_cluster_same"], "se_max": float(se.max()), "se_top": float(se[0]),
        "overlap": float(b["overlap"].mean()), "overlap_min": float(b["overlap"].min()),
        "overlap_floor": m["floor"]["overlap"], "tv_across": m["across"]["population_tv"],
        "overlap_across": m["across"]["overlap"], "absent_clusters": absent,
    })
    if real:
        worst = max(real, key=lambda link: link["jsd"])
        row.update({"jsd_max": worst["jsd"], "jsd_max_label": worst["label"], "jsd_max_class": worst["subclass"],
                    "jsd_max_torsion": worst["jsd_torsion"],
                    "jsd_median": float(np.median([link["jsd"] for link in real])),
                    "rotamer_missing": sum(bool(link["rotamer_missing"]) for link in real)})
    row["linkage_rows"] = links
    return row


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
def verdict(m: Dict[str, Any], boundaries: Boundaries, parameters: Optional[Parameters] = None,
            row: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """``ok``, ``more_sampling_suggested``, ``more_sampling_needed`` or ``not_assessable``, with the reasons."""
    p = parameters or Parameters()
    row = entry_row(m, p) if row is None else row
    names = [replica.name for replica in boundaries.replicas]
    thresholds = {
        "population_tv": p.population_tv, "cluster_pooled_min": p.cluster_pooled_min,
        "cluster_replica_max": p.cluster_replica_max, "rotamer_present_min": p.rotamer_present_min,
        "rotamer_absent_max": p.rotamer_absent_max, "omega_min_transitions": p.omega_min_transitions,
        "omega_classes": list(p.omega_classes), "min_full_replicas": p.min_full_replicas,
    }
    result: Dict[str, Any] = {"value": None, "reasons": [], "criteria": {}, "thresholds": thresholds,
                              "thresholds_from": "the sampling-robustness analysis of the thesis "
                                                 "(robustness_data.py, worst_cases); see lib/replicas.py"}
    blocked = []
    if boundaries.basis == NONE:
        blocked.append({"code": "replicas_unknown",
                        "text": "the replicas cannot be told apart in the stored trajectory: "
                                + str(boundaries.evidence.get("why") or "part of it has no record of its runs")})
    if m["full_replicas"] < p.min_full_replicas:
        blocked.append({"code": "too_few_replicas", "value": m["full_replicas"], "threshold": p.min_full_replicas,
                        "text": f"{m['full_replicas']} full-length replica(s) of {m['replicas']}; "
                                f"at least {p.min_full_replicas} are needed to compare"})
    if blocked:
        result["value"], result["reasons"] = NOT_ASSESSABLE, blocked
        return result

    reasons = []
    tv_mean = round(row["tv"], 4)
    criteria = {"population_tv": tv_mean >= p.population_tv, "cluster_missing": bool(row["absent_clusters"]),
                "rotamer_missing": row.get("rotamer_missing", 0) > 0}
    if criteria["population_tv"]:
        reasons.append({"code": "population_tv", "value": round(tv_mean, 4), "threshold": p.population_tv,
                        "floor": round(row["tv_floor"], 4),
                        "text": f"the Level 1 populations of the replicas differ: mean TV {tv_mean:.2f} "
                                f"(threshold {p.population_tv:g}; uncorrelated frames give {row['tv_floor']:.2f})"})
    if criteria["cluster_missing"]:
        items = []
        for absent in row["absent_clusters"]:
            lows = [names[m["full_index"][i]] for i, x in enumerate(absent["replicas"]) if x < p.cluster_replica_max]
            items.append({"cluster": absent["cluster"], "pooled": round(absent["pooled"], 4), "replicas": lows})
        reasons.append({"code": "cluster_missing", "clusters": items,
                        "text": "; ".join(f"Level 1 cluster {it['cluster']} ({it['pooled']:.0%} of the frames) has "
                                          f"less than {p.cluster_replica_max:.0%} of the frames of "
                                          + ", ".join(it["replicas"]) for it in items)})
    if criteria["rotamer_missing"]:
        items = [{"linkage": link["label"], "torsions": link["rotamer_missing"]}
                 for link in row["linkage_rows"] if link["class"] != "substituent" and link["rotamer_missing"]]
        reasons.append({"code": "rotamer_missing", "linkages": items,
                        "text": f"{len(items)} linkage(s) with a rotamer present in one replica and missing in "
                                "another: " + ", ".join(f"{it['linkage']} ({', '.join(it['torsions'])})" for it in items)})
    stuck = []
    for link in row["linkage_rows"]:
        if "omega" not in link["torsions"] or link["subclass"] not in p.omega_classes:
            continue
        j = link["torsions"]["omega"]["column"]
        for i in m["full_index"]:
            count = m["replica_transitions"][j][i]
            if count < p.omega_min_transitions:
                occupancy = m["replica_rotamers"][i, j]
                third = int(np.argmax(occupancy))
                if link.get("omega_rotamer_names"):
                    rotamer = link["omega_rotamer_names"][third]
                elif link.get("omega_rotamer_centres"):
                    rotamer = f"{link['omega_rotamer_centres'][third]:+.0f}°"
                else:
                    rotamer = OMEGA_ROTAMERS[third]
                stuck.append({"linkage": link["label"], "replica": names[i], "transitions": int(count),
                              "rotamer": rotamer, "occupancy": round(float(occupancy.max()), 4)})
    criteria["omega_never_changes"] = bool(stuck)
    if stuck:
        reasons.append({"code": "omega_never_changes", "replicas": stuck,
                        "text": "; ".join(f"omega of {s['linkage']} never changes rotamer in {s['replica']} "
                                          f"({s['rotamer']} {s['occupancy']:.0%})" for s in stuck)})
    result["criteria"] = criteria
    result["reasons"] = reasons
    if criteria["population_tv"] and criteria["cluster_missing"] and criteria["rotamer_missing"]:
        result["value"] = NEEDED
    elif reasons:
        result["value"] = SUGGESTED
    else:
        result["value"] = OK
    return result


# ---------------------------------------------------------------------------
# What a web page needs to draw the landscape panel
# ---------------------------------------------------------------------------
def hdr_level(density: np.ndarray, mass: float) -> float:
    """The density level whose super-level set holds ``mass`` of the total (``fig_robustness.py``)."""
    v = np.sort(density.ravel())[::-1]
    c = np.cumsum(v)
    return float(v[min(np.searchsorted(c, mass * c[-1]), len(v) - 1)])


def _decimals(span: float) -> int:
    return int(max(0, 3 - math.floor(math.log10(span)))) if span > 0 else 3


def _outline(xs: np.ndarray, ys: np.ndarray, density: np.ndarray, level: float, decimals: int) -> List[List[float]]:
    """Polylines of one level of a density on the grid ``xs`` x ``ys``: ``[[x0, y0, x1, y1, ...], ...]``."""
    from contourpy import LineType, contour_generator

    if not np.isfinite(level) or level <= 0 or density.max() <= 0:
        return []
    generator = contour_generator(xs, ys, density.T, line_type=LineType.Separate)
    lines = []
    for line in generator.lines(level):
        if len(line) < 3:
            continue
        lines.append([round(float(v), decimals) for v in np.asarray(line).ravel()])
    return lines


def landscape(tables: FrameTables, m: Dict[str, Any], parameters: Optional[Parameters] = None) -> Dict[str, Any]:
    """Outlines on (PC1, PC2): the pooled landscape at three levels and, per replica, the outline that
    encloses ``landscape_mass`` of its frames (64 x 64 grid one fifth wider than the overlap grid,
    Gaussian smoothing of 1.3 cells, as the figures of the thesis draw it); and the counts of the
    overlap grid per replica."""
    from scipy.ndimage import gaussian_filter

    p = parameters or Parameters()
    lo, hi = np.asarray(m["pc_lo"], dtype=float), np.asarray(m["pc_hi"], dtype=float)
    span = hi - lo
    lo2, hi2 = lo - p.landscape_margin * span, hi + p.landscape_margin * span
    fine = p.landscape_grid
    rid = m["rid"]
    n_rep = m["replicas"]
    pc = tables.pcs[:, :2].astype(np.float64)
    ij = np.floor((pc - lo2) / np.maximum(hi2 - lo2, 1e-9) * fine).astype(int)
    inside = ((ij >= 0) & (ij < fine)).all(axis=1)
    dens = np.bincount(rid[inside] * fine * fine + ij[inside, 0] * fine + ij[inside, 1],
                       minlength=n_rep * fine * fine).reshape(n_rep, fine, fine).astype(float)
    xs = lo2[0] + (np.arange(fine) + 0.5) * (hi2[0] - lo2[0]) / fine
    ys = lo2[1] + (np.arange(fine) + 0.5) * (hi2[1] - lo2[1]) / fine
    decimals = _decimals(float(max(hi2 - lo2)))
    pooled = gaussian_filter(dens.sum(0), p.landscape_sigma, mode="constant")
    out = {
        "pc1_range": [round(float(lo2[0]), decimals), round(float(hi2[0]), decimals)],
        "pc2_range": [round(float(lo2[1]), decimals), round(float(hi2[1]), decimals)],
        "grid": fine, "sigma_cells": p.landscape_sigma, "mass": p.landscape_mass,
        "pooled": [{"mass": mass, "lines": _outline(xs, ys, pooled, hdr_level(pooled, mass), decimals)}
                   for mass in p.landscape_pooled_masses],
        "replicas": [],
    }
    for i in range(n_rep):
        d = gaussian_filter(dens[i], p.landscape_sigma, mode="constant")
        out["replicas"].append({
            "replica": i, "frames_outside": int(((rid == i) & ~inside).sum()),
            "lines": _outline(xs, ys, d, hdr_level(d, p.landscape_mass), decimals) if d.max() > 0 else [],
        })
    G = p.pca_grid
    cells = []
    for i in range(n_rep):
        nonzero = np.flatnonzero(m["replica_cells"][i])
        cells.append({"replica": i, "cells": [int(c) for c in nonzero],
                      "counts": [int(c) for c in m["replica_cells"][i][nonzero]]})
    out["overlap_grid"] = {
        "size": G, "pc1": [round(float(lo[0]), decimals), round(float(hi[0]), decimals)],
        "pc2": [round(float(lo[1]), decimals), round(float(hi[1]), decimals)],
        "cell": "PC1 bin * size + PC2 bin; frames outside the range are in the edge cells", "replicas": cells,
    }
    return out


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------
def _r(value: Any, digits: int = 4) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_r(v, digits) for v in value]
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


def diagnose(tables: FrameTables, boundaries: Boundaries, *, glycam_name: Optional[str] = None,
             glycoshape_id: Optional[str] = None, parameters: Optional[Parameters] = None,
             seed: Optional[int] = None) -> Dict[str, Any]:
    """The ``replicas.json`` document of an entry."""
    p = parameters or Parameters()
    seed = floor_seed(glycoshape_id, glycam_name) if seed is None else int(seed)
    m = measure(tables, boundaries, p, seed)
    row = entry_row(m, p)
    links = linkage_rows(m, p)
    ranks = step_percentile(tables)
    names = [replica.name for replica in boundaries.replicas]
    full_rank = {i: k for k, i in enumerate(m.get("full_index", []))}
    T = tables.torsions.shape[1]

    replicas = []
    for i, replica in enumerate(boundaries.replicas):
        tpf = replica.time_per_frame_ps
        item: Dict[str, Any] = {
            "index": i, "name": replica.name, "first_frame": replica.first_frame, "frames": replica.frames,
            "retained": m["retained"][i], "basis": replica.basis, "recorded_by": replica.recorded_by,
            "segment": replica.segment, "time_per_frame_ps": tpf, "time_from": replica.time_from,
            "simulated_ns": _r((replica.frames - 1) * tpf / 1000.0, 3) if tpf else None,
            "full_length": m["full"][i],
            "population": _r(m["replica_population"][i]),
            "boundary_rank": None,
        }
        if i > 0 and ranks is not None and replica.first_frame < len(ranks) and np.isfinite(ranks[replica.first_frame]):
            item["boundary_rank"] = round(float(ranks[replica.first_frame]), 4)
        if "pooled" in m and m["retained"][i]:
            pooled4, mine4 = _round4(m["pooled"]), _round4(m["replica_population"][i])
            item["clusters_missing"] = [int(k) for k in range(m["clusters"])
                                        if pooled4[k] >= p.cluster_pooled_min and mine4[k] < p.cluster_replica_max]
        if i in full_rank:
            k = full_rank[i]
            item["halves"] = {"population_tv": _r(m["within"]["population_tv"][k]),
                              "overlap": _r(m["within"]["overlap"][k])}
        replicas.append(item)

    doc: Dict[str, Any] = {
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
        "glycam_name": glycam_name, "glycoshape_id": glycoshape_id,
        "generator": "GlycanAnalysisPipeline lib/replicas.py",
        "parameters": p.describe(),
        # the torsion definitions of the entry: which rotamer each third of omega is
        "torsion_conventions_version": int(tables.conventions_version),
        "frames": {"trajectory": boundaries.frames, "retained": m["rows"], "unassigned": m["unassigned_rows"]},
        "boundaries": {"basis": boundaries.basis, "recorded_by": boundaries.recorded_by,
                       "replicas": len(boundaries.replicas), "full_length_replicas": m["full_replicas"],
                       "full_length_by": m["full_length_by"], "evidence": boundaries.evidence},
        "replicas": replicas,
        "clusters": None, "between": None, "halves": None, "floor": None,
    }
    if "pooled" in m:
        doc["clusters"] = {"ids": list(range(m["clusters"])), "pooled": _r(m["pooled"]),
                           "by_population": m["order"], "pooled_over": "full-length replicas"}
    if "between" in m:
        b = m["between"]
        doc["between"] = {
            "replicas": m["full_index"],
            "pairs": [{"replicas": list(pair), "population_tv": _r(t), "overlap": _r(o)}
                      for pair, t, o in zip(b["pairs"], b["population_tv"], b["overlap"])],
            "population_tv_mean": _r(row["tv"]), "population_tv_max": _r(row["tv_max"]),
            "population_absdiff_max": _r(row["absdiff_max"]), "population_se_max": _r(row["se_max"]),
            "population_se": _r(b["population_sd"] / np.sqrt(len(m["full_index"]))),
            "top_cluster_same": row["top_same"],
            "overlap_mean": _r(row["overlap"]), "overlap_min": _r(row["overlap_min"]),
            "linkage_jsd_max": _r(row.get("jsd_max"), 5), "linkage_jsd_max_label": row.get("jsd_max_label"),
            "linkage_jsd_median": _r(row.get("jsd_median"), 5),
        }
        doc["floor"] = {"population_tv": _r(m["floor"]["population_tv"]), "overlap": _r(m["floor"]["overlap"]),
                        "seed": m["floor"]["seed"],
                        "what": "the same between-replica measures after the frames were dealt at random to the replicas"}
    if "within" in m:
        doc["halves"] = {"population_tv_within": _r(row["tv_within"]), "overlap_within": _r(row["overlap_within"])}
        if "across" in m:
            doc["halves"].update({"population_tv_across": _r(row["tv_across"]),
                                  "overlap_across": _r(row["overlap_across"])})

    out_links = []
    for link in links:
        item = {k: link[k] for k in ("donor", "acceptor", "label", "class", "subclass", "positions")}
        for key, digits in (("jsd", 5), ("jsd_floor", 5), ("jsd_within", 5), ("jsd_across", 5), ("rotamer_absdiff", 4)):
            if key in link:
                item[key] = _r(link[key], digits)
        if "jsd_torsion" in link:
            item["jsd_torsion"] = link["jsd_torsion"]
        item["rotamer_missing"] = link["rotamer_missing"]
        torsions = {}
        for kind, t in link["torsions"].items():
            j = t["column"]
            entry: Dict[str, Any] = {"name": tables.torsion_names[j]}
            for key, digits in (("jsd", 5), ("jsd_max", 5), ("jsd_floor", 5), ("jsd_within", 5), ("jsd_across", 5),
                                ("rotamer_absdiff", 4)):
                if key in t:
                    entry[key] = _r(t[key], digits)
            entry["rotamers"] = [_r(m["replica_rotamers"][i, j]) for i in range(len(boundaries.replicas))]
            entry["transitions"] = [int(x) for x in m["replica_transitions"][j]]
            if kind == "omega" and "omega_rotamer_centres" in link:
                entry["rotamer_centres"] = [float(c) for c in link["omega_rotamer_centres"]]
                if "omega_rotamer_names" in link:
                    entry["rotamer_names"] = list(link["omega_rotamer_names"])
            torsions[kind] = entry
        item["torsions"] = torsions
        out_links.append(item)
    doc["linkages"] = out_links

    # histograms of the torsions whose replicas agree least (or the first ones), for the torsion strip
    if T:
        if "between" in m:
            shown = [int(j) for j in np.argsort(-m["between"]["jsd"].mean(0), kind="stable")[:p.histogram_torsions]]
        else:
            shown = list(range(min(T, p.histogram_torsions)))
        doc["torsion_histograms"] = {
            "bins": p.histogram_bins, "from_degrees": -180.0,
            "torsions": [{"name": tables.torsion_names[j],
                          "counts": [[int(c) for c in m["replica_histograms"][i, j]] for i in range(len(boundaries.replicas))]}
                         for j in shown],
        }
    doc["verdict"] = verdict(m, boundaries, p, row)
    doc["landscape"] = landscape(tables, m, p) if len(tables.source_frame) else None
    return doc


def encode(document: Dict[str, Any]) -> bytes:
    """The bytes of ``replicas.json``: compact JSON, UTF-8, one line and a newline."""
    return (json.dumps(document, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def write(document: Dict[str, Any], path: PathLike) -> Path:
    """Write ``replicas.json`` (atomically: it appears under its name when it is complete)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(encode(document))
    os.replace(temporary, path)
    return path


def diagnose_static(folder: Optional[PathLike], manifest: Optional[Mapping[str, Any]], *,
                    glycam_name: Optional[str] = None, glycoshape_id: Optional[str] = None,
                    length_us: Optional[float] = None, parameters: Optional[Parameters] = None,
                    tables: Optional[FrameTables] = None) -> Dict[str, Any]:
    """``replicas.json`` of a published static entry (``static/GSxxxxx/``) and its archive manifest.

    The Level 1 labels are those of the stored entry: nothing is clustered again.  ``length_us`` (the
    simulation length of ``data.json``) gives the time per frame of replicas that do not record it.
    ``tables`` stands for the tables of ``folder`` when the caller has read them already.
    """
    tables = tables_from_static(folder) if tables is None else tables
    frames = int(manifest["frames"]) if manifest is not None else int(tables.source_frame[-1]) + 1
    boundaries = replica_boundaries(manifest, frames, parameters=parameters)
    tpf = implied_time_per_frame(boundaries, length_us)
    if tpf is not None:
        for replica in boundaries.replicas:
            if replica.time_per_frame_ps is None:
                replica.time_per_frame_ps, replica.time_from = tpf, "entry length"
    return diagnose(tables, boundaries, glycam_name=glycam_name or (manifest or {}).get("glycam_name"),
                    glycoshape_id=glycoshape_id, parameters=parameters)


def implied_time_per_frame(boundaries: Boundaries, length_us: Optional[float]) -> Optional[float]:
    """Picoseconds per frame from the total length of an entry: length / (frames - replicas).

    Only for an entry whose replicas are known (recorded, single or assumed) and none of which records its
    own time; ``None`` otherwise.
    """
    if not length_us or boundaries.basis == NONE:
        return None
    if any(replica.time_per_frame_ps is not None for replica in boundaries.replicas):
        return None
    steps = boundaries.frames - len(boundaries.replicas)
    return round(float(length_us) * 1e6 / steps, 6) if steps > 0 else None


def diagnose_result(process_dir: PathLike, manifest: Optional[Mapping[str, Any]], frames: int, *,
                    glycam_name: Optional[str] = None, parameters: Optional[Parameters] = None) -> Dict[str, Any]:
    """``replicas.json`` from the process folder of a run (``lib/entry_run.py``) and the manifest of its input."""
    tables = tables_from_process(process_dir)
    boundaries = replica_boundaries(manifest, frames, parameters=parameters)
    return diagnose(tables, boundaries, glycam_name=glycam_name, parameters=parameters)


def summary(document: Mapping[str, Any]) -> Dict[str, Any]:
    """One line of a collection: verdict, reasons, basis and the main numbers."""
    between = document.get("between") or {}
    floor = document.get("floor") or {}
    verdict_ = document.get("verdict") or {}
    return {
        "glycoshape_id": document.get("glycoshape_id"), "glycam_name": document.get("glycam_name"),
        "verdict": verdict_.get("value"), "reasons": [reason["code"] for reason in verdict_.get("reasons", [])],
        "basis": (document.get("boundaries") or {}).get("basis"),
        "recorded_by": (document.get("boundaries") or {}).get("recorded_by"),
        "replicas": (document.get("boundaries") or {}).get("replicas"),
        "full_length_replicas": (document.get("boundaries") or {}).get("full_length_replicas"),
        "frames": (document.get("frames") or {}).get("trajectory"),
        "retained": (document.get("frames") or {}).get("retained"),
        "population_tv": between.get("population_tv_mean"), "population_tv_floor": floor.get("population_tv"),
        "overlap": between.get("overlap_mean"), "overlap_floor": floor.get("overlap"),
        "linkage_jsd_max": between.get("linkage_jsd_max"),
    }
