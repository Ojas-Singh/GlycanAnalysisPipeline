"""Entry audit: a list of issues for every GlycoShape entry, for the administrators' website.

The owner reviews each problem of an entry on the admin pages and acts on it.
This module finds the problems.  For one entry it reads what the collection
holds about it (the stored static entry, its raw archive entry, its
``replicas.json`` and the registry of GlycoShape IDs), measures what the
problems are made of (:func:`entry_facts`), puts the entries side by side for
the problems that only a collection shows (:func:`collection_context`) and
turns both into issues (:func:`entry_issues`), which :func:`document` writes as
the ``issues.json`` of the entry.

Every issue type is the check of the thesis audit that found it, ported (the
scripts are named in :data:`ISSUE_TYPES`, ``found_by``); nothing here changes
an entry.

The file
--------
``issues.json`` (schema ``glycoshape-entry-issues``, version 1) is a function
of the inputs only (no time stamp), so the same audit gives the same bytes::

    {
      "schema": "glycoshape-entry-issues", "schema_version": 1,
      "glycoshape_id": "GS00020", "glycam_name": "...", "glytoucan": "G56655CC",
      "generator": "GlycanAnalysisPipeline lib/audit.py",
      "inputs": {"static": true, "raw": true, "replicas": true, "registry": true, "collection": true},
      "counts": {"blocking": 1, "should-fix": 2, "note": 1},
      "issues": [
        {"key": "anomer-exchanged",            # stable across audits: code, plus ":<detail>" when an entry
                                               # can have more than one issue of the type
         "code": "anomer-exchanged",           # the issue type (ISSUE_TYPES)
         "severity": "blocking",               # blocking | should-fix | note
         "title": "One sentence in plain language.",
         "evidence": {...},                    # numbers, atoms, names: what the check measured
         "files": [{"store": "raw", "path": "<GLYCAM name>/topology.pdb", "what": "..."}],
         "actions": [{"action": "reanalyse", "options": {"declare_simulated_anomer": "a"}, "why": "..."}],
         "resolved_by_reanalysis": false,      # the bulk re-analysis with the fixed GAP clears it by itself
         "needs_owner": true,                  # a decision of the owner is needed
         "found_by": "..."}                    # the thesis check that this one ports
      ]
    }

``files`` name a file by its store and its path in that store: ``static``
(``static/`` of the database: ``GSxxxxx/...``), ``raw`` (the raw archive:
``<GLYCAM name>/...``), ``replicas`` (the administrators' replica diagnostics:
``GSxxxxx.json``) and ``registry`` (the registry seed and the export of the
old registry).  ``actions`` are taken from :data:`ACTIONS`, a fixed set.

Severity: ``blocking`` the entry shows something that is not what it says it
is (or must not be shown); ``should-fix`` a record is wrong or a result is
weaker than it looks; ``note`` worth knowing, no fault established.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from lib import audit_classes as classes
from lib import audit_structure as structure

PathLike = Union[str, "os.PathLike[str]"]

SCHEMA = "glycoshape-entry-issues"
SCHEMA_VERSION = 1
FILE_NAME = "issues.json"
FACTS_FILE_NAME = "facts.json"
GENERATOR = "GlycanAnalysisPipeline lib/audit.py"

BLOCKING, SHOULD_FIX, NOTE = "blocking", "should-fix", "note"
SEVERITIES = (BLOCKING, SHOULD_FIX, NOTE)

#: The actions an issue can suggest; the website offers these and no others.
ACTIONS = {
    "rename": "give the entry another GLYCAM name (and with it its notations and accessions)",
    "edit-metadata": "correct a recorded field (simulation time, force field, salt, raw info.json, ...)",
    "reanalyse": "run the analysis again on the stored trajectory; options: exclude_replicas (names of runs to "
                 "leave out), declare_simulated_anomer ('a' or 'b': analyse under the anomer that was simulated)",
    "request-replicas": "ask for more or new independent simulations",
    "redo": "simulate the glycan again from a new build",
    "withdraw": "take the entry out of the published database",
    "merge-ids": "keep one GlycoShape ID for a glycan that has several",
    "none": "keep the entry as it is (the issue is acknowledged)",
}

# ---------------------------------------------------------------------------
# Thresholds (all from the thesis audit unless said otherwise)
# ---------------------------------------------------------------------------
FRAMES_PER_RUN = 10001                 # 500 ns written every 50 ps (evidence-audit section 2.6)
US_PER_RUN = 0.5
SHORT_TIME_US = 1.5                    # "at least 1.5 us per glycan" of the 2024 protocol
SHORT_TRAJECTORY_FRAMES = 1000         # G43523TH has 101; the next smallest entries have 1,500
FRAMES_REMOVED_SHOULD_FIX = 0.10       # audit: "entries that lost more than 10%"
FRAMES_REMOVED_NOTE = 0.01             # audit: "entries that lost more than 1%"
SALTS_OF_THE_COLLECTION = (0.0, 150.0, 200.0)
REGISTRY_LENGTH_NS = 100.0             # a recorded length above this many "microseconds" is nanoseconds
MASS_TOLERANCE = 0.05                  # Da (coverage_data.py)
STEREO_FRAMES_PER_RUN = 12             # frames read per run for the stereocentres (the thesis read all)
STEREO_LOW, STEREO_HIGH = 0.01, 0.99   # robustness_data.py _stereo
FORCEFIELD_PLACEHOLDERS = ("See Build Directory Files",)
MONOISOTOPIC = {"C": 12.0, "H": 1.00782503207, "N": 14.0030740048, "O": 15.9949146196, "S": 31.972071,
                "P": 30.97376163}
SO3 = 3 * MONOISOTOPIC["O"] + MONOISOTOPIC["S"]
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_ACCESSION = re.compile(r"^G\d{5}[A-Z]{2}$")
_ANOMER_END = re.compile(r"([ab])(\d|None)-OH$")
#: GLYCAM residue codes whose anomer is not the third character (representatives_data.py ANOMER_EXCEPTIONS)
ANOMER_EXCEPTIONS = {"GL": "a", "YS": "a", "KN": "a"}

# ---------------------------------------------------------------------------
# Issue types
# ---------------------------------------------------------------------------
#: code -> what the website shows about the type, and which thesis check it ports.
ISSUE_TYPES: Dict[str, Dict[str, Any]] = {
    # identity: the simulated molecule against the name
    "anomer-exchanged": {
        "group": "identity", "severity": BLOCKING, "resolved_by_reanalysis": False, "needs_owner": True,
        "what": "The reducing end was simulated as the other anomer than the name declares, so the alpha and beta "
                "files are exchanged.",
        "found_by": "crabwurcs_data.py (structures: other_anomer, both read-back routes) and the chirality at C1; "
                    "here the configuration is read from the coordinates (lib/audit_structure.py)"},
    "structure-contradicts-name": {
        "group": "identity", "severity": BLOCKING, "resolved_by_reanalysis": False, "needs_owner": True,
        "what": "The simulated structure is another glycan than its name: a linkage anomer, a linkage position, the "
                "configuration of a residue or the residue itself differs.",
        "found_by": "representatives_data.py (checks.name_and_structure_disagree) and crabwurcs_data.py "
                    "(structures: different), confirmed from the coordinates"},
    "level1-file-anomer": {
        "group": "identity", "severity": SHOULD_FIX, "resolved_by_reanalysis": False, "needs_owner": False,
        "what": "A published Level 1 file holds the other anomer at the reducing end than its file name says, "
                "although the simulated structure is what its name says (the evidence gives the cause: a ketose, "
                "whose anomeric carbon GAP does not flip, or a name without an anomer).",
        "found_by": "crabwurcs_data.py (distributed structures read from the atoms)"},
    "stereo-differs-between-runs": {
        "group": "identity", "severity": BLOCKING, "resolved_by_reanalysis": False, "needs_owner": True,
        "what": "The runs of the entry are not the same stereoisomer: a stereocentre is inverted in some runs, so "
                "the populations pool two molecules.",
        "found_by": "robustness_data.py (stereocentres: centres_that_differ_between_runs)"},
    "stereo-family-minority": {
        "group": "identity", "severity": NOTE, "resolved_by_reanalysis": False, "needs_owner": True,
        "what": "A stereocentre has the other configuration than the same-named centre in most entries of the "
                "collection, and the read-back from the coordinates could not settle whether the residue is what "
                "its name says (when it can, the entry gets an identity issue or none).",
        "found_by": "robustness_data.py (stereocentres: centre_types_with_both_configurations)"},
    # notation
    "stored-wurcs-mismatch": {
        "group": "notation", "severity": SHOULD_FIX, "resolved_by_reanalysis": True, "needs_owner": False,
        "what": "A stored WURCS (archetype, alpha or beta) does not describe the glycan of the name; the accession "
                "found for it names that wrong structure.",
        "found_by": "crabwurcs_data.py (notations: wurcs_differs_from_the_database), with the fixed crabWURCS"},
    "stored-iupac-mismatch": {
        "group": "notation", "severity": SHOULD_FIX, "resolved_by_reanalysis": True, "needs_owner": False,
        "what": "A stored IUPAC name reads as another glycan than the GLYCAM name.",
        "found_by": "crabwurcs_data.py (notations: stored_iupac_reads_as_another_wurcs)"},
    "name-not-glycam": {
        "group": "notation", "severity": BLOCKING, "resolved_by_reanalysis": False, "needs_owner": True,
        "what": "The entry's name is not a GLYCAM sequence (a GlyTouCan accession or free text), so its residues "
                "cannot be read from it and the re-analysis cannot name it.",
        "found_by": "coverage_data.py (validation.glycam_names_not_parsed) and evidence-audit section 2.3"},
    "inventory-fields-broken": {
        "group": "notation", "severity": SHOULD_FIX, "resolved_by_reanalysis": True, "needs_owner": False,
        "what": "Inventory fields derived from the name (iupac, components, mass) are broken.",
        "found_by": "coverage_data.py (audit_comparison: residues vs n_monosaccharides; mass_inventory_missing)"},
    "mass-mismatch": {
        "group": "notation", "severity": SHOULD_FIX, "resolved_by_reanalysis": False, "needs_owner": True,
        "what": "The atoms of the structure file do not add up to the glycan of the name (a group or hydrogens "
                "more or fewer).",
        "found_by": "coverage_data.py (size.mass_differs_from_structure)"},
    "duplicate-entry": {
        "group": "notation", "severity": SHOULD_FIX, "resolved_by_reanalysis": False, "needs_owner": True,
        "what": "The same glycan with the same reducing-end anomer is published under another ID too.",
        "found_by": "audit_summary.py (counts.archetype_wurcs_shared_by_two_entries: same glycan and anomer twice)"},
    "probable-input-error": {
        "group": "notation", "severity": SHOULD_FIX, "resolved_by_reanalysis": False, "needs_owner": True,
        "what": "The class rules find something not canonical (an N-glycan core without Man(b1-4), a beta-Man arm, "
                "an unusual residue on the core, linkage positions left open): probably an input error.",
        "found_by": "coverage_classes.py / coverage_data.py (open_to_argument: N-glycan core flags)"},
    "glytoucan-missing": {
        "group": "notation", "severity": NOTE, "resolved_by_reanalysis": False, "needs_owner": False,
        "what": "The archetype has no GlyTouCan accession.",
        "found_by": "evidence-audit section 2.2"},
    # simulation metadata
    "sim-time-missing": {
        "group": "metadata", "severity": SHOULD_FIX, "resolved_by_reanalysis": False, "needs_owner": True,
        "what": "No simulated time is recorded.",
        "found_by": "audit_summary.py (simulated_time.entries_with_zero)"},
    "sim-time-frames-disagree": {
        "group": "metadata", "severity": SHOULD_FIX, "resolved_by_reanalysis": False, "needs_owner": True,
        "what": "The recorded simulated time does not agree with the number of frames (runs of 10,001 frames are "
                "500 ns each).",
        "found_by": "audit_summary.py (simulated_time.frames_multiple_but_length_differs)"},
    "sim-time-short": {
        "group": "metadata", "severity": NOTE, "resolved_by_reanalysis": False, "needs_owner": False,
        "what": "Less than 1.5 microseconds of simulation are recorded.",
        "found_by": "audit_summary.py (simulated_time.entries_under_1.5_us_nonzero)"},
    "sim-time-unit": {
        "group": "metadata", "severity": SHOULD_FIX, "resolved_by_reanalysis": False, "needs_owner": True,
        "what": "The registry holds nanoseconds in its microsecond field.",
        "found_by": "registry seed (simulation_length of the old registry)"},
    "sim-metadata-blank": {
        "group": "metadata", "severity": SHOULD_FIX, "resolved_by_reanalysis": False, "needs_owner": True,
        "what": "Package, temperature or pressure of the simulation are not recorded.",
        "found_by": "evidence-audit section 2.7 (blank package, 0 K, 0 bar)"},
    "forcefield-placeholder": {
        "group": "metadata", "severity": SHOULD_FIX, "resolved_by_reanalysis": False, "needs_owner": True,
        "what": "The force field is recorded as a placeholder or not at all.",
        "found_by": "evidence-audit section 2.7 ('See Build Directory Files', blank)"},
    "salt-implausible": {
        "group": "metadata", "severity": SHOULD_FIX, "resolved_by_reanalysis": False, "needs_owner": True,
        "what": "The salt concentration is not one the collection was simulated at (0, 150, 200 mM).",
        "found_by": "evidence-audit section 2.7 (salt 1.5)"},
    "salt-zero": {
        "group": "metadata", "severity": NOTE, "resolved_by_reanalysis": False, "needs_owner": True,
        "what": "Salt recorded as 0: no salt, or not recorded.",
        "found_by": "evidence-audit section 7, question 5"},
    # raw data and frames
    "raw-source-repaired": {
        "group": "raw", "severity": NOTE, "resolved_by_reanalysis": False, "needs_owner": False,
        "what": "The source trajectory was irregular and was archived with a repair (a frame dropped, a cut file, "
                "malformed records).",
        "found_by": "raw archive manifest (source_irregularities)"},
    "frames-removed": {
        "group": "raw", "severity": NOTE, "resolved_by_reanalysis": False, "needs_owner": False,
        "what": "The frame-quality filter removed frames (a bond over 3 angstrom or a non-finite coordinate).",
        "found_by": "audit_entries.py / audit_summary.py (frames.entries_removed_over_10pct, over 1%)"},
    "run-topology-mismatch": {
        "group": "raw", "severity": BLOCKING, "resolved_by_reanalysis": False, "needs_owner": True,
        "what": "The coordinates of a run do not match the entry's topology (bonds of the topology stretched, or a "
                "stereocentre inverted against it, on the first and last frame of the run), so a re-analysis fails "
                "or pools another molecule.",
        "found_by": "the torsion work on GS00276 (2026-10-14); first and last frame of every run against "
                    "structure.mol2 / topology.pdb, with the criteria of the frame-quality filter and the "
                    "stereocentre test of robustness_data.py"},
    "trajectory-short": {
        "group": "raw", "severity": SHOULD_FIX, "resolved_by_reanalysis": False, "needs_owner": True,
        "what": "The trajectory has very few frames.",
        "found_by": "evidence-audit section 5.1 (minimum 101 frames)"},
    "personal-data-in-raw": {
        "group": "raw", "severity": BLOCKING, "resolved_by_reanalysis": False, "needs_owner": True,
        "what": "The raw archive entry holds personal data (an e-mail address).",
        "found_by": "evidence-audit section 7, question 16"},
    "pca-variance-accumulated": {
        "group": "raw", "severity": NOTE, "resolved_by_reanalysis": True, "needs_owner": False,
        "what": "archetype.pca_variance holds the variances of several PCA runs; the website shows the first ten.",
        "found_by": "audit_entries.py (evr_inventory_vs_model)"},
    # registry
    "id-several-for-name": {
        "group": "registry", "severity": SHOULD_FIX, "resolved_by_reanalysis": False, "needs_owner": True,
        "what": "The GLYCAM name of the entry has two or three GlycoShape IDs in the registry.",
        "found_by": "registry seed (conflicts.name_with_several_ids) and the old glycan records"},
    # replicas
    "replicas-more-sampling-needed": {
        "group": "replicas", "severity": SHOULD_FIX, "resolved_by_reanalysis": False, "needs_owner": True,
        "what": "The independent runs disagree on all three criteria of the thesis: more sampling is needed.",
        "found_by": "lib/replicas.py verdict (robustness_data.py worst_cases)"},
    "replicas-more-sampling-suggested": {
        "group": "replicas", "severity": NOTE, "resolved_by_reanalysis": False, "needs_owner": True,
        "what": "The independent runs disagree on at least one criterion: more sampling is suggested.",
        "found_by": "lib/replicas.py verdict (robustness_data.py worst_cases)"},
    "replicas-not-assessable": {
        "group": "replicas", "severity": NOTE, "resolved_by_reanalysis": False, "needs_owner": False,
        "what": "The runs cannot be compared (fewer than two full-length runs, or no record of the runs).",
        "found_by": "lib/replicas.py verdict"},
}


def issue_type_table() -> List[Dict[str, Any]]:
    """The issue types as rows (code, group, default severity, resolution, what, found by)."""
    return [{"code": code, **{k: v for k, v in t.items()}} for code, t in ISSUE_TYPES.items()]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def read_json(path: Optional[PathLike]) -> Optional[Any]:
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def declared_anomer(name: Optional[str]) -> Optional[str]:
    match = _ANOMER_END.search(name or "")
    return match.group(1) if match else None


def other_anomer_name(name: str) -> Optional[str]:
    """The GLYCAM name with the other anomer at the reducing end (``crabwurcs_data.py``, ``exchanged``)."""
    match = _ANOMER_END.search(name or "")
    if not match:
        return None
    return name[:match.start()] + ("b" if match.group(1) == "a" else "a") + match.group(2) + "-OH"


def mask_email(text: str) -> str:
    """``jane.doe@example.org`` -> ``j***@e***.org``: enough to recognise it, not to use it."""
    local, _, domain = text.partition("@")
    host, _, tld = domain.rpartition(".")
    return f"{local[:1]}***@{host[:1]}***.{tld}"


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


class Crab:
    """The crabWURCS command line (``GLYCOSHAPE_CRABWURCS_BIN``, as ``lib/name_utils.py`` calls it)."""

    def __init__(self, executable: Optional[str] = None, timeout: int = 60):
        if executable is None:
            from lib import config  # noqa: WPS433 (only for the default path)

            executable = config.crabwurcs_executable
        self.executable = str(executable)
        self.timeout = timeout

    def _run(self, args: Sequence[str], text: str) -> Tuple[Optional[str], str]:
        try:
            run = subprocess.run([self.executable, *args], input=text, capture_output=True, text=True,
                                 timeout=self.timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return None, f"{type(exc).__name__}: {exc}"[:240]
        out = run.stdout.strip()
        if run.returncode == 0 and out:
            return out, ""
        return None, (run.stderr.strip() or f"exit {run.returncode}")[:240]

    def to_wurcs(self, text: Optional[str], fmt: str) -> Tuple[Optional[str], str]:
        """Normalised WURCS of ``text`` given in ``fmt`` (computed; the bundled table is not consulted)."""
        if not text:
            return None, "empty"
        return self._run(["convert", "--from", fmt, "--to", "wurcs", "--table", "off"], text)

    def normalise(self, wurcs: Optional[str]) -> Optional[str]:
        if not wurcs:
            return None
        out, _ = self._run(["normalize-wurcs"], wurcs)
        return out

    def version(self) -> Optional[str]:
        try:
            return subprocess.run([self.executable, "--version"], capture_output=True, text=True,
                                  timeout=self.timeout).stdout.strip() or None
        except OSError:
            return None


# ---------------------------------------------------------------------------
# Facts of one entry
# ---------------------------------------------------------------------------
VARIANTS = ("archetype", "alpha", "beta")


def notation_facts(data: Mapping[str, Any], crab: Optional[Crab]) -> Dict[str, Any]:
    """The names and notations of the inventory record against what the GLYCAM name gives.

    Port of ``crabwurcs_data.py`` (``notations_one``) with the crabWURCS that writes furanose rings and
    bracketed substituents correctly: per variant the WURCS from the GLYCAM name, the stored WURCS normalised
    and the WURCS of the stored IUPAC name.
    """
    archetype = data.get("archetype") or {}
    name = archetype.get("name")
    out: Dict[str, Any] = {"name": name, "variants": {}}
    try:
        classes.parse_glycam(name or "")
        out["name_is_glycam"], out["name_error"] = True, None
    except (ValueError, KeyError, AttributeError) as exc:
        out["name_is_glycam"], out["name_error"] = False, str(exc)[:200]
    out["name_is_accession"] = bool(_ACCESSION.match(name or ""))
    for variant in VARIANTS:
        record = data.get(variant) or {}
        # the full name of each variant: the archetype is the simulated anomer (archetype.name)
        full = name if variant == "archetype" else record.get("glycam")
        row: Dict[str, Any] = {"glycam": full, "glytoucan": record.get("glytoucan"),
                               "stored_wurcs": record.get("wurcs"), "stored_iupac": record.get("iupac")}
        if crab is not None:
            row["wurcs"], row["name_error"] = crab.to_wurcs(full, "glycam")
            # the archetype's own record: the WURCS of the glycan with the reducing end open
            if variant == "archetype":
                row["open_wurcs"], _ = crab.to_wurcs(archetype.get("glycam"), "glycam")
            row["stored_normalised"] = crab.normalise(record.get("wurcs"))
            expected = row.get("open_wurcs") if variant == "archetype" else row["wurcs"]
            row["expected"] = expected
            row["stored_equal"] = (row["stored_normalised"] == expected) if expected and row["stored_normalised"] else None
            iupac_wurcs, iupac_error = crab.to_wurcs(record.get("iupac"), "iupac-condensed")
            row["iupac_wurcs"], row["iupac_error"] = iupac_wurcs, iupac_error
            row["iupac_equal"] = (iupac_wurcs == expected) if iupac_wurcs and expected else None
        out["variants"][variant] = row
    return out


def tree_facts(data: Mapping[str, Any], notation: Mapping[str, Any]) -> Dict[str, Any]:
    """Residue tree, class and the flags of the class rules (``coverage_data.py``, ``tree_of`` / ``classify``).

    The tree comes from the WURCS of the GLYCAM name (the fixed crabWURCS), else from the stored archetype
    WURCS as the thesis read it.
    """
    archetype = data.get("archetype") or {}
    tree, source, errors = None, None, []
    for source_name, wurcs in (("name", (notation["variants"].get("archetype") or {}).get("open_wurcs")),
                               ("stored archetype WURCS", archetype.get("wurcs"))):
        if not wurcs:
            continue
        try:
            tree, source = classes.parse_wurcs(wurcs), source_name
            break
        except (ValueError, KeyError) as exc:
            errors.append(f"{source_name}: {exc}"[:200])
    if tree is None:
        return {"residues": None, "source": None, "errors": errors}
    t = classes.Tree(tree)
    result = classes.classify(tree)
    detail = result.get("detail") or {}
    out = {
        "source": source, "errors": errors, "residues": len(tree),
        "iupac": classes.condensed(tree),
        "class": result["class"], "subclass": result["subclass"], "also": result["also"],
        "n_glycan_like": bool(detail.get("n_glycan_like")),
        "unusual_on_beta_man": detail.get("unusual_on_beta_man") or [],
        "open_linkages": detail.get("open_linkages") or [],
        "groups": {"sulfate": sum(classes.sulfate_groups(r) for r in tree),
                   "sialic": sum(1 for r in tree if r.stem in ("Neu", "Kdn")),
                   "uronic": sum(1 for r in tree if r.base in ("GlcA", "IdoA", "GalA") or r.stem == "Kdo"),
                   "phosphocholine": sum(1 for r in tree if "PCho" in r.name)},
        "bonds": [list(b) for b in t.bonds()],
    }
    return out


def _element_counts(atoms: structure.Atoms) -> Dict[str, int]:
    return dict(Counter(e.capitalize() for e in atoms.elements))


def _name_disagrees(info: Optional[Mapping[str, Any]], iupac: Optional[str], sugar_numbers: Iterable[int]
                    ) -> Optional[Dict[str, Any]]:
    """Port of ``representatives_data.py`` ``name_disagrees``: the linkages of the structure (anomer from the
    residue names of the phi torsion, positions from its atoms) against those of the IUPAC name, reported only
    when both give the same number of linkages and the two lists differ."""
    if not info or not iupac:
        return None
    sugars = set(sugar_numbers)
    groups: Dict[Tuple[int, int], List[Mapping[str, Any]]] = defaultdict(list)
    for meta in (info.get("torsions") or {}).get("glycosidic") or []:
        m = re.match(r"^(\d+)_(\d+)_(phi|psi|omega)", str(meta.get("canonical_name", "")))
        if m:
            groups[(int(m.group(1)), int(m.group(2)))].append(meta)
    built: Counter = Counter()
    for (child, parent), metas in groups.items():
        if child not in sugars or parent not in sugars:
            continue
        phi = next((m for m in metas if m.get("subtype") == "phi"), None)
        if phi is None:
            continue
        rn = phi["residue_names"][0]
        an = phi["atom_names"]
        ab = {"A": "a", "B": "b"}.get(rn[2:3]) or ANOMER_EXCEPTIONS.get(rn[1:3], "?")
        built[f"{ab}{an[1][1:]}-{an[2][1:]}"] += 1
    named = Counter(f"{a}{c}-{p}" for a, c, p in re.findall(r"\(([ab])(\d)-(\d)\)", iupac))
    if not named or sum(named.values()) != sum(built.values()) or named == built:
        return None
    return {"in_name_not_in_structure": dict(named - built), "in_structure_not_in_name": dict(built - named)}


def structure_facts(archive_dir: Optional[Path], static_dir: Optional[Path], data: Mapping[str, Any],
                    notation: Mapping[str, Any]) -> Dict[str, Any]:
    """The simulated structure (first frame of the raw entry) and the published Level 1 files, read from
    their coordinates and compared with the name."""
    out: Dict[str, Any] = {}
    archetype = data.get("archetype") or {}
    name_wurcs = (notation["variants"].get("archetype") or {}).get("wurcs")
    topology = archive_dir / "topology.pdb" if archive_dir is not None else None
    atoms = None
    if topology is not None and topology.is_file():
        atoms = structure.read_first_model(topology)
        source = "raw topology.pdb (first frame of the source)"
    elif static_dir is not None and (static_dir / "output/level_1/PDB/alpha.pdb").is_file():
        atoms = None   # the Level 1 files are read below; the simulated structure is unknown without the raw entry
    if atoms is not None:
        reading = structure.read(atoms)
        sim: Dict[str, Any] = {"source": source, "atoms": len(atoms), "elements": _element_counts(atoms),
                               "residues": [r.describe() for r in reading.residues]}
        root = reading.residues[reading.reducing_end] if reading.reducing_end is not None else None
        if root is not None:
            code = root.residue_name
            sim["reducing_end"] = {
                **root.describe(),
                "anomer_from_residue_name": {"A": "a", "B": "b"}.get(code[2:3]) or ANOMER_EXCEPTIONS.get(code[1:3]),
            }
        if name_wurcs:
            try:
                sim["comparison"] = structure.compare(reading, structure.declared(name_wurcs))
            except ValueError as exc:
                sim["comparison"] = {"outcome": "unread", "why": str(exc)[:200]}
        else:
            sim["comparison"] = {"outcome": "unread", "why": "the name does not give a WURCS"}
        info = read_json(static_dir / "output/info.json") if static_dir is not None else None
        sim["name_disagrees"] = _name_disagrees(info, archetype.get("iupac"), [r.number for r in reading.residues])
        out["simulated"] = sim
    # the published Level 1 files
    level1 = {}
    if static_dir is not None:
        for variant in ("alpha", "beta"):
            path = static_dir / "output/level_1/PDB" / f"{variant}.pdb"
            if not path.is_file():
                continue
            file_atoms = structure.read_first_model(path)
            reading = structure.read(file_atoms)
            root = reading.residues[reading.reducing_end] if reading.reducing_end is not None else None
            level1[variant] = {"path": f"output/level_1/PDB/{variant}.pdb",
                               "reducing_end_anomer": root.anomer if root else None,
                               "reducing_end": root.describe() if root else None}
            if atoms is None and variant == "alpha":
                out["elements_from"] = "Level 1 alpha.pdb (first model)"
                out["elements"] = _element_counts(file_atoms)
    out["level1"] = level1
    if atoms is not None:
        out["elements_from"] = "raw topology.pdb"
        out["elements"] = out["simulated"]["elements"]
    return out


def _pca_source_frames(static_dir: Path) -> Optional[np.ndarray]:
    path = static_dir / "output/pca.csv"
    if not path.is_file():
        return None
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if "source_frame" in header:
            i = header.index("source_frame")
            return np.array([int(row[i]) for row in reader], dtype=np.int64)
        return np.arange(sum(1 for _ in reader), dtype=np.int64)


def frames_facts(manifest: Optional[Mapping[str, Any]], source_frames: Optional[np.ndarray],
                 length_us: Optional[float]) -> Dict[str, Any]:
    """Frames of the raw entry and of the analysis, and the runs (``lib.replicas.replica_boundaries``)."""
    from lib import replicas as replicas_lib

    out: Dict[str, Any] = {"archive": int(manifest["frames"]) if manifest else None,
                           "kept": int(len(source_frames)) if source_frames is not None else None}
    if out["archive"] is not None and out["kept"] is not None:
        out["removed"] = out["archive"] - out["kept"]
    frames = out["archive"]
    if frames:
        out["runs_if_10001"] = frames // FRAMES_PER_RUN if frames % FRAMES_PER_RUN == 0 else None
    if manifest is not None:
        boundaries = replicas_lib.replica_boundaries(manifest, frames)
        out["runs"] = {"basis": boundaries.basis, "recorded_by": boundaries.recorded_by,
                       "names": [r.name for r in boundaries.replicas], "starts": boundaries.starts,
                       "lengths": boundaries.lengths}
        if source_frames is not None:
            starts = np.array(boundaries.starts)
            rid = np.searchsorted(starts, source_frames, side="right") - 1
            out["runs"]["kept"] = [int((rid == i).sum()) for i in range(len(starts))]
    return out


def _read_frames(entry: Any, frames: Iterable[int]) -> Dict[int, np.ndarray]:
    """Frames of an archive entry by index, in angstrom (one pass over the XTC, frames in order)."""
    from lib.archive import _xtc_file_class

    out: Dict[int, np.ndarray] = {}
    handle = _xtc_file_class()(str(entry.trajectory_path), "r")
    try:
        for f in sorted(set(int(x) for x in frames)):
            handle.seek(f)
            frame, _, _, _ = handle.read(n_frames=1)
            out[f] = frame[0].astype(np.float64) * 10.0
    finally:
        handle.close()
    return out


def _mol2_bonds(path: Path, atoms: int) -> Optional[List[Tuple[int, int]]]:
    """Bonds of a MOL2 file as 0-based pairs, or None when it has none or its atoms are not the topology's."""
    if not path.is_file():
        return None
    block, count, bonds = None, 0, []
    with open(path, "r", encoding="latin-1") as handle:
        for line in handle:
            if line.startswith("@<TRIPOS>"):
                block = line.strip()
                continue
            fields = line.split()
            if block == "@<TRIPOS>ATOM" and len(fields) >= 6:
                count += 1
            elif block == "@<TRIPOS>BOND" and len(fields) >= 3:
                try:
                    a, b = int(fields[1]) - 1, int(fields[2]) - 1
                except ValueError:
                    continue
                bonds.append((min(a, b), max(a, b)))
    if count != atoms or not bonds or max(max(p) for p in bonds) >= atoms:
        return None
    return sorted(set(bonds))


def segment_facts(archive_dir: Path, source_frames: Optional[np.ndarray], runs: Optional[Mapping[str, Any]]
                  ) -> Dict[str, Any]:
    """Whether the coordinates of every run (replica segment) of a raw entry match its topology.

    For the first and the last frame of each run: the length of every bond of the topology (``structure.mol2``
    when it has bonds for the atoms of ``topology.pdb``, else the bonds perceived from ``topology.pdb`` with the
    rule of the frame-quality filter), against the perception cutoff (covalent radii + 0.35 angstrom) and the
    filter's 3.0 angstrom; and the sign of every stereocentre against ``topology.pdb`` (the first frame of the
    source).  A frame that fails on bond lengths only (a frame the filter removes, a molecule broken across the
    box) is replaced by the nearest frame of the run that the filter kept.  A run does not match when both of
    its frames fail.  Atom counts alone cannot tell: GS00276 has the topology's atoms and bonds in every frame,
    but one of its stereocentres is inverted in two of its three runs.
    """
    from lib.archive import ArchiveEntry

    entry = ArchiveEntry(archive_dir)
    atoms = structure.atoms_from_lines(entry.atom_lines())
    nb = structure.neighbours(atoms)
    bonds = _mol2_bonds(entry.mol2_path, len(atoms))
    bonds_from = "structure.mol2"
    if bonds is None:
        bonds = sorted({(i, j) for i in range(len(nb)) for j in nb[i] if i < j})
        bonds_from = "topology.pdb (perceived)"
    pairs = np.array(bonds, dtype=int).reshape(-1, 2)
    radius = np.array([structure.COVALENT_RADII.get(e, 0.8) for e in atoms.elements])
    cutoff = radius[pairs[:, 0]] + radius[pairs[:, 1]] + structure.BOND_TOLERANCE
    centres = structure.stereocentres(atoms, nb)
    reference = structure.centre_signs(atoms.xyz[None], centres)[0]
    starts = list((runs or {}).get("starts") or [0])
    lengths = list((runs or {}).get("lengths") or [entry.n_frames])
    names = list((runs or {}).get("names") or ["all"])
    kept = np.asarray(source_frames) if source_frames is not None else None

    def nearest_kept(lo: int, hi: int, near: int) -> Optional[int]:
        if kept is None:
            return None
        inside = kept[(kept >= lo) & (kept < hi)]
        return int(inside[np.argmin(np.abs(inside - near))]) if len(inside) else None

    wanted = {f for s, n in zip(starts, lengths) if n > 0 for f in (s, s + n - 1)}
    xyz = _read_frames(entry, wanted)

    def test(f: int) -> Dict[str, Any]:
        x = xyz[f]
        d = np.linalg.norm(x[pairs[:, 0]] - x[pairs[:, 1]], axis=1) if len(pairs) else np.zeros(0)
        inverted = np.flatnonzero(structure.centre_signs(x[None], centres)[0] != reference) if centres else []
        return {"frame": f, "bonds_over_cutoff": int((d > cutoff).sum()),
                "bonds_over_3A": int((d > 3.0).sum()), "longest_bond": round(float(d.max()), 2) if len(d) else None,
                "centres_inverted": [structure.centre_label(atoms, centres[k]) for k in inverted]}

    out_runs = []
    for name, s, n in zip(names, starts, lengths):
        if n <= 0:
            continue
        checks = []
        for f in sorted({s, s + n - 1}):
            t = test(f)
            if t["bonds_over_cutoff"] and not t["centres_inverted"]:
                alternative = nearest_kept(s, s + n, f)
                if alternative is not None and alternative != f:
                    if alternative not in xyz:
                        xyz.update(_read_frames(entry, [alternative]))
                    t = {**test(alternative), "instead_of": t}
            checks.append(t)
        fails = [bool(t["bonds_over_cutoff"] or t["centres_inverted"]) for t in checks]
        out_runs.append({"run": name, "first_frame": s, "last_frame": s + n - 1, "frames": checks,
                         "matches": not all(fails)})
    return {"bonds": len(pairs), "bonds_from": bonds_from, "centres": len(centres), "runs": out_runs}


def stereo_facts(archive_dir: Path, source_frames: Optional[np.ndarray], runs: Optional[Mapping[str, Any]],
                 frames_per_run: int = STEREO_FRAMES_PER_RUN) -> Dict[str, Any]:
    """The configuration of every stereocentre in every run (``robustness_data.py``, ``archive_entry``).

    The centres are found in the first retained frame; their signed volumes are read in up to
    ``frames_per_run`` retained frames of each run, evenly spaced (the thesis read every frame: a stereocentre
    cannot change sign in a run without being inverted, and no run of the collection had a centre of mixed
    sign).
    """
    from lib.archive import ArchiveEntry

    entry = ArchiveEntry(archive_dir)
    lines = entry.atom_lines()
    atoms = structure.atoms_from_lines(lines)
    if source_frames is None or not len(source_frames):
        source_frames = np.arange(entry.n_frames)
    starts = list((runs or {}).get("starts") or [0])
    names = list((runs or {}).get("names") or ["all"])
    rid = np.searchsorted(np.array(starts), source_frames, side="right") - 1
    wanted: Dict[int, List[int]] = {}
    for i in range(len(starts)):
        mine = source_frames[rid == i]
        if not len(mine):
            continue
        pick = np.unique(mine[np.linspace(0, len(mine) - 1, min(frames_per_run, len(mine))).round().astype(int)])
        wanted[i] = [int(f) for f in pick]
    xyz = _read_frames(entry, {f for fs in wanted.values() for f in fs} | {int(source_frames[0])})
    first = atoms.with_coordinates(xyz[int(source_frames[0])])
    centres = structure.stereocentres(first)
    out_centres = []
    if centres:
        positive = {i: structure.centre_signs(np.stack([xyz[f] for f in fs]), centres).mean(axis=0)
                    for i, fs in wanted.items()}
        for k, centre in enumerate(centres):
            i_atom, nb3 = centre
            out_centres.append({
                "label": structure.centre_label(atoms, centre), "atom": atoms.names[i_atom],
                "residue": atoms.residue_names[i_atom][:3], "residue_number": atoms.residue_keys[i_atom][1],
                "neighbours": [atoms.names[j] for j in nb3],
                "positive": [None if i not in positive else round(float(positive[i][k]), 4) for i in range(len(starts))],
            })
    return {"runs": names, "frames_read": {names[i]: len(fs) for i, fs in wanted.items()},
            "frames_per_run": frames_per_run, "centres": out_centres}


def raw_facts(archive_dir: Optional[Path], manifest: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if manifest is None:
        return out
    out["source_irregularities"] = manifest.get("source_irregularities")
    out["warnings"] = manifest.get("warnings") or []
    out["glycam_name"] = manifest.get("glycam_name")
    info_path = archive_dir / "info.json" if archive_dir is not None else None
    if info_path is not None and info_path.is_file():
        text = info_path.read_text(encoding="utf-8", errors="replace")
        found = sorted(set(_EMAIL.findall(text)))
        fields = []
        try:
            info = json.loads(text)
            if isinstance(info, dict):
                fields = sorted(k for k, v in info.items() if isinstance(v, str) and _EMAIL.search(v))
                fields += sorted(k for k in info if k.lower() in ("username", "user", "submitter", "name_of_submitter")
                                 and info.get(k))
        except ValueError:
            pass
        out["info_json"] = {"emails": [mask_email(e) for e in found], "fields": sorted(set(fields))}
    return out


def replicas_facts(document: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not document:
        return None
    verdict = document.get("verdict") or {}
    return {"verdict": verdict.get("value"),
            "reasons": [{k: v for k, v in r.items()} for r in verdict.get("reasons", [])],
            "basis": (document.get("boundaries") or {}).get("basis"),
            "replicas": (document.get("boundaries") or {}).get("replicas"),
            "full_length_replicas": (document.get("boundaries") or {}).get("full_length_replicas"),
            "population_tv_mean": (document.get("between") or {}).get("population_tv_mean"),
            "population_tv_floor": (document.get("floor") or {}).get("population_tv")}


def entry_facts(glycoshape_id: str, data: Mapping[str, Any], *, static_dir: Optional[PathLike] = None,
                archive_dir: Optional[PathLike] = None, replicas_document: Optional[Mapping[str, Any]] = None,
                registry_record: Optional[Mapping[str, Any]] = None, crab: Optional[Crab] = None,
                trajectory: bool = True) -> Dict[str, Any]:
    """Everything the rules need about one entry.  Only reads.

    ``data`` is the inventory record (``data.json``); ``static_dir`` the entry's static folder
    (``static/GSxxxxx/``), ``archive_dir`` its raw archive folder, ``replicas_document`` its ``replicas.json``,
    ``registry_record`` its record in the old registry (``submission_ids.json``).  ``trajectory`` reads frames
    of the raw trajectory for the stereocentres of every run.
    """
    static_dir = Path(static_dir) if static_dir is not None else None
    archive_dir = Path(archive_dir) if archive_dir is not None else None
    archetype = data.get("archetype") or {}
    facts: Dict[str, Any] = {"glycoshape_id": glycoshape_id, "errors": []}
    facts["entry"] = {
        "name": archetype.get("name"), "glycam": archetype.get("glycam"), "iupac": archetype.get("iupac"),
        "glytoucan": {v: (data.get(v) or {}).get("glytoucan") for v in VARIANTS},
        "mass": archetype.get("mass"), "components": archetype.get("components"),
        "length_us": archetype.get("length"), "package": archetype.get("package"),
        "forcefield": archetype.get("forcefield"), "temperature": archetype.get("temperature"),
        "pressure": archetype.get("pressure"), "salt": archetype.get("salt"),
        "pca_variance_values": len((archetype.get("pca_variance") or {}).get("explained_variance_ratio") or []),
    }
    facts["inputs"] = {"static": static_dir is not None and static_dir.is_dir(),
                       "raw": archive_dir is not None and (archive_dir / "manifest.json").is_file(),
                       "replicas": replicas_document is not None, "registry": registry_record is not None}

    def guarded(key: str, function: Callable[[], Any]) -> Any:
        try:
            return function()
        except Exception as exc:  # noqa: BLE001 - one failing check must not hide the others
            facts["errors"].append({"check": key, "error": f"{type(exc).__name__}: {exc}"[:300]})
            return None

    facts["notation"] = guarded("notation", lambda: notation_facts(data, crab)) or {"variants": {}}
    facts["tree"] = guarded("tree", lambda: tree_facts(data, facts["notation"]))
    manifest = read_json(archive_dir / "manifest.json") if archive_dir is not None else None
    facts["structure"] = guarded("structure", lambda: structure_facts(archive_dir if facts["inputs"]["raw"] else None,
                                                                      static_dir, data, facts["notation"]))
    if static_dir is not None and static_dir.is_dir():
        landscape = read_json(static_dir / "landscape/landscape_metadata.json") or {}
        facts["entry"]["pca_model_values"] = len(landscape.get("explained_variance_ratio") or [])
    source_frames = guarded("pca", lambda: _pca_source_frames(static_dir)) if static_dir is not None else None
    facts["frames"] = guarded("frames", lambda: frames_facts(manifest, source_frames, _number(archetype.get("length"))))
    if trajectory and facts["inputs"]["raw"] and facts["frames"] is not None:
        facts["stereo"] = guarded("stereo", lambda: stereo_facts(archive_dir, source_frames, facts["frames"].get("runs")))
        facts["segments"] = guarded("segments", lambda: segment_facts(archive_dir, source_frames,
                                                                      facts["frames"].get("runs")))
    facts["raw"] = guarded("raw", lambda: raw_facts(archive_dir, manifest))
    facts["replicas"] = replicas_facts(replicas_document)
    if registry_record is not None:
        facts["registry"] = {k: registry_record.get(k) for k in
                             ("glycoshape_id", "glycam_name", "simulation_length", "md_package", "force_field",
                              "salt_concentration", "source_type", "status", "created")}
    return facts


# ---------------------------------------------------------------------------
# What only the collection shows
# ---------------------------------------------------------------------------
def registry_names(seed: Optional[Mapping[str, Any]], exports: Sequence[Mapping[str, Any]] = ()) -> Dict[str, List[Dict[str, Any]]]:
    """GLYCAM name -> the GlycoShape IDs the registry has for it, with their status and where each was seen.

    ``seed`` is the registry seed (``registry-seed.json``: ``entries`` with ``glycoshape_id``, ``glycam_name``,
    ``status``); ``exports`` are exports of the old registry (``submission_ids.json``, ``glycans.json``:
    ``records`` with ``glycoshape_id`` and ``glycam_name``), which hold a few IDs the seed does not.
    Values that are not GlycoShape IDs (``GS11E892``) are kept, marked ``valid: false``.
    """
    from lib.id_registry import is_id

    names: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in (seed or {}).get("entries", []):
        gid, name = row.get("glycoshape_id"), row.get("glycam_name")
        if gid and name:
            names[name][gid] = {"glycoshape_id": gid, "status": row.get("status"), "seen_in": ["registry seed"],
                                "valid": is_id(gid)}
    for export in exports:
        label = export.get("collection") or export.get("from") or "registry export"
        for row in export.get("records", []):
            gid, name = row.get("glycoshape_id"), row.get("glycam_name")
            if not gid or not name:
                continue
            item = names[name].setdefault(gid, {"glycoshape_id": gid, "status": "only in the old records",
                                                 "seen_in": [], "valid": is_id(gid)})
            if label not in item["seen_in"]:
                item["seen_in"].append(label)
    return {name: sorted(ids.values(), key=lambda x: x["glycoshape_id"]) for name, ids in names.items()}


def collection_context(facts: Mapping[str, Mapping[str, Any]], *,
                       registry: Optional[Mapping[str, List[Dict[str, Any]]]] = None,
                       inventory: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Dict[str, Any]:
    """Duplicates, the configurations of same-named stereocentres across entries and the registry conflicts.

    ``facts`` are the facts of every entry of the collection (or of those at hand); ``inventory`` adds the
    inventory records of entries whose facts are not at hand, for the duplicates.
    """
    # duplicates: the same glycan (archetype WURCS, reducing end open) and the same simulated anomer
    groups: Dict[Tuple[str, Optional[str]], List[str]] = defaultdict(list)
    keyed = set()
    for gid, f in facts.items():
        v = ((f.get("notation") or {}).get("variants") or {}).get("archetype") or {}
        wurcs = v.get("open_wurcs") or v.get("stored_normalised") or v.get("stored_wurcs")
        anomer = declared_anomer((f.get("entry") or {}).get("name"))
        if wurcs:
            groups[(wurcs, anomer)].append(gid)
            keyed.add(gid)
    for gid, record in (inventory or {}).items():
        if gid in keyed:
            continue
        a = record.get("archetype") or {}
        if a.get("wurcs"):
            groups[(a["wurcs"], declared_anomer(a.get("name")))].append(gid)
    duplicates = {gid: sorted(set(ids) - {gid}) for ids in groups.values() if len(set(ids)) > 1 for gid in ids}
    # stereocentres: the sign of every centre type in every entry (the majority over its runs)
    signs: Dict[str, Dict[str, set]] = defaultdict(lambda: {"+": set(), "-": set()})
    for gid, f in facts.items():
        for c in ((f.get("stereo") or {}).get("centres") or []):
            p = [x for x in c["positive"] if x is not None]
            for x in p:
                signs[c["label"]]["+" if x > 0.5 else "-"].add(gid)
    minority: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for label, s in signs.items():
        if s["+"] and s["-"] and len(s["+"]) != len(s["-"]):
            small = "+" if len(s["+"]) < len(s["-"]) else "-"
            for gid in s[small]:
                minority[gid].append({"centre": label, "entries_with_this_sign": len(s[small]),
                                      "entries_with_the_other_sign": len(s["-" if small == "+" else "+"])})
    # registry: names with several IDs
    several = {name: ids for name, ids in (registry or {}).items() if len(ids) > 1}
    return {"duplicates": duplicates, "stereo_minority": dict(minority), "names_with_several_ids": several,
            "entries": len(facts)}


# ---------------------------------------------------------------------------
# Rules: facts -> issues
# ---------------------------------------------------------------------------
def _issue(code: str, title: str, evidence: Mapping[str, Any], *, actions: Sequence[Mapping[str, Any]],
           files: Sequence[Mapping[str, Any]] = (), severity: Optional[str] = None, key: Optional[str] = None,
           resolved_by_reanalysis: Optional[bool] = None, needs_owner: Optional[bool] = None) -> Dict[str, Any]:
    t = ISSUE_TYPES[code]
    for a in actions:
        if a["action"] not in ACTIONS:
            raise ValueError(f"unknown action {a['action']!r}")
    return {
        "key": key or code, "code": code, "severity": severity or t["severity"], "title": title,
        "evidence": dict(evidence), "files": [dict(f) for f in files], "actions": [dict(a) for a in actions],
        "resolved_by_reanalysis": t["resolved_by_reanalysis"] if resolved_by_reanalysis is None else resolved_by_reanalysis,
        "needs_owner": t["needs_owner"] if needs_owner is None else needs_owner,
        "found_by": t["found_by"],
    }


def _act(action: str, why: str, **options: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"action": action, "why": why}
    if options:
        out["options"] = {k: v for k, v in options.items() if v is not None}
    return out


ANOMER_WORD = {"a": "alpha", "b": "beta"}


def _residue_word(code: str, named_codes: Sequence[str]) -> str:
    """A residue code read from the atoms (``a2122h_1-5``) as a residue name, with the substituents of the name
    when the name has a residue of that skeleton and ring (``GlcNAc``), else from the skeleton alone (``L-GulA``)."""
    bare = code.replace("-1a", "").replace("-1b", "").replace("-2a", "").replace("-2b", "")
    skeleton, _, ring = bare.partition("_")
    for full in named_codes:
        head, _, rest = full.partition("_")
        if head.split("-")[0] == skeleton and rest.split("_")[0] == ring.split("_")[0]:
            try:
                d = classes.decode_wurcs_residue(full)
                return classes.make_name(d["stem"], d["config"], d["ring"], d["core_mod"], d["subs"])[0]
            except (ValueError, KeyError):
                break
    try:
        d = classes.decode_wurcs_residue(f"{skeleton}_{ring}" if ring else skeleton)
        return classes.make_name(d["stem"], d["config"], d["ring"], d["core_mod"], d["subs"])[0]
    except (ValueError, KeyError):
        return code


def describe_difference(comparison: Mapping[str, Any], name_wurcs: Optional[str]) -> str:
    """The difference between a structure and its name in plain words (residue names where they can be told)."""
    named = re.findall(r"\[([^\]]*)\]", name_wurcs or "")
    word = lambda c: _residue_word(c, named)  # noqa: E731
    parts = []
    ends = comparison.get("reducing_end_anomer") or {}
    if ends.get("structure") and ends.get("name") in ("a", "b") and ends["structure"] != ends["name"]:
        parts.append(f"the reducing end is {ANOMER_WORD[ends['structure']]} where the name says {ANOMER_WORD[ends['name']]}")
    skeleton = lambda c: (c.split("-")[0], c.partition("_")[2])  # noqa: E731
    rs = comparison.get("residues_in_structure_not_in_name") or []
    rn = comparison.get("residues_in_name_not_in_structure") or []
    only_s = sorted(c for c in rs if skeleton(c) not in {skeleton(x) for x in rn})
    only_n = sorted(c for c in rn if skeleton(c) not in {skeleton(x) for x in rs})
    key = lambda x: (x[2], x[3], x[4], x[0])  # noqa: E731
    mine = sorted(comparison.get("linkages_in_structure_not_in_name") or [], key=key)
    theirs = sorted(comparison.get("linkages_in_name_not_in_structure") or [], key=key)
    fmt = lambda x: f"{word(x[0])}({x[1]}{x[2]}-{x[3]}){word(x[4])}"  # noqa: E731
    for s, n in zip(mine, theirs):
        if list(s[1:4]) == list(n[1:4]) and (only_s or only_n):
            continue                # the same linkage between residues that differ: said below
        parts.append(f"{fmt(s)} where the name has {fmt(n)}")
    for s, n in zip(only_s, only_n):
        text = f"{word(s)} where the name has {word(n)}"
        if text not in parts:
            parts.append(text)
    if len(only_s) != len(only_n):
        parts.append(f"{len(rs)} residue(s) of the structure and {len(rn)} of the name do not pair up")
    out = []
    for p in parts:                 # one mention of each difference
        if p not in out:
            out.append(p)
    return "; ".join(out)


def _files(gid: str, raw_name: Optional[str]):
    def static(path: str, what: str) -> Dict[str, str]:
        return {"store": "static", "path": f"{gid}/{path}", "what": what}

    def raw(path: str, what: str) -> Optional[Dict[str, str]]:
        return {"store": "raw", "path": f"{raw_name}/{path}", "what": what} if raw_name else None

    return static, raw


def identity_issues(f: Mapping[str, Any], ctx: Mapping[str, Any]) -> List[Dict[str, Any]]:
    gid = f["glycoshape_id"]
    out: List[Dict[str, Any]] = []
    entry = f.get("entry") or {}
    name = entry.get("name")
    raw_name = (f.get("raw") or {}).get("glycam_name")
    static, raw = _files(gid, raw_name)
    st = f.get("structure") or {}
    sim = st.get("simulated") or {}
    comparison = sim.get("comparison") or {}
    declared = declared_anomer(name)
    level1 = st.get("level1") or {}
    level1_read = {v: x.get("reducing_end_anomer") for v, x in level1.items()}
    level1_files = [static(x["path"], f"published Level 1 {v} file") for v, x in level1.items()]
    outcome = comparison.get("outcome")
    root = sim.get("reducing_end") or {}
    if outcome == "other_anomer" and declared:
        simulated = root.get("anomer")
        title = (f"The reducing end was simulated as the {ANOMER_WORD.get(simulated, simulated)} anomer, but the name "
                 f"declares {ANOMER_WORD.get(declared, declared)}, so the alpha and beta files are exchanged.")
        evidence = {
            "declared_anomer": declared, "simulated_anomer": simulated,
            "reducing_end_residue": root.get("residue"), "anomer_from_residue_name": root.get("anomer_from_residue_name"),
            "anomer_from_coordinates": simulated, "reference_carbon": root.get("reference_carbon"),
            "read_from": sim.get("source"), "level1_files_read_as": level1_read,
            "name_with_the_simulated_anomer": other_anomer_name(name or ""),
            "rest_of_the_glycan": "as named",
        }
        out.append(_issue("anomer-exchanged", title, evidence, files=[x for x in [raw("topology.pdb", "simulated structure (first frame)")] if x] + level1_files,
                          actions=[_act("reanalyse", "analyse under the anomer that was simulated: the entry becomes "
                                                     "the glycan it holds", declare_simulated_anomer=simulated),
                                   _act("rename", "rename to the simulated anomer (same effect, decided by name)",
                                        to=other_anomer_name(name or "")),
                                   _act("redo", "simulate the declared anomer"),
                                   _act("withdraw", "take the entry out")]))
    elif outcome == "different":
        disagrees = sim.get("name_disagrees")
        words = describe_difference(comparison, (f.get("notation") or {}).get("variants", {}).get("archetype", {}).get("wurcs"))
        title = "The simulated structure is not the glycan its name describes: " + (words or "residues or linkages differ") + "."
        evidence = {k: comparison.get(k) for k in ("kinds", "reducing_end_anomer", "residues_in_structure_not_in_name",
                                                   "residues_in_name_not_in_structure",
                                                   "linkages_in_structure_not_in_name",
                                                   "linkages_in_name_not_in_structure")}
        evidence["codes"] = "WURCS residue codes: skeleton (a anomeric carbon, 1 left, 2 right in the Fischer " \
                            "projection, d deoxy, h CH2-O, m CH3, A carboxyl), anomer and ring; linkages as (donor, " \
                            "anomer, donor position, acceptor position, acceptor)"
        evidence["read_from"] = sim.get("source")
        if disagrees:
            evidence["linkages_from_residue_names_and_torsions"] = disagrees
        out.append(_issue("structure-contradicts-name", title, evidence,
                          files=[x for x in [raw("topology.pdb", "simulated structure (first frame)")] if x]
                          + [static("output/info.json", "torsion metadata (residue names of each linkage)")],
                          actions=[_act("rename", "name the entry after the structure that was simulated"),
                                   _act("redo", "simulate the glycan of the name"),
                                   _act("withdraw", "take the entry out")]))
    elif sim.get("name_disagrees"):
        # the torsion tables disagree with the name while the coordinates agree: report what was found
        disagrees = sim["name_disagrees"]
        out.append(_issue("structure-contradicts-name",
                          "The residue names of a linkage disagree with the name, while the coordinates agree with it.",
                          {"linkages_from_residue_names_and_torsions": disagrees, "coordinates": outcome},
                          severity=SHOULD_FIX, files=[static("output/info.json", "torsion metadata")],
                          actions=[_act("reanalyse", "recompute the torsion tables"), _act("none", "keep")]))
    # Level 1 files that read as the other anomer although the simulated structure is what its name says
    if outcome == "match":
        for variant, read in level1_read.items():
            if not read or read == variant[0]:
                continue
            end = level1[variant].get("reducing_end") or {}
            ketose = (end.get("ring") or "").startswith("2-")
            if declared is None:
                cause, fixed_by = ("the name does not state the anomer, and GAP then treats the simulated one as "
                                   "alpha", "rename")
            elif ketose:
                cause, fixed_by = ("the anomeric carbon is C2 (a ketose); GAP writes the other anomer by exchanging "
                                   "O1 and H1 of C1 (lib/v2.py _flip_anomer_model), which a ketose does not have, so "
                                   "the file keeps the simulated anomer under the other residue name", "code")
            elif variant[0] != declared:
                cause, fixed_by = ("GAP did not find C1 and H1 of residue 2 or O1 and HO1 of residue 1 "
                                   "(lib/v2.py _flip_anomer_model), so the file keeps the simulated anomer", "code")
            else:
                cause, fixed_by = ("not established", "reanalysis")
            out.append(_issue("level1-file-anomer",
                              f"The published Level 1 {variant} file holds the {ANOMER_WORD.get(read, read)} anomer at "
                              f"the reducing end.",
                              {"file": level1[variant]["path"], "file_says": variant, "atoms_say": read,
                               "simulated_anomer": declared, "reducing_end": end, "cause": cause},
                              key=f"level1-file-anomer:{variant}",
                              resolved_by_reanalysis=fixed_by == "reanalysis", needs_owner=fixed_by == "rename",
                              files=[static(level1[variant]["path"], f"Level 1 {variant} file")],
                              actions=([_act("rename", "give the entry a GLYCAM name with its anomer")] if fixed_by == "rename" else [])
                              + [_act("reanalyse", "write the Level 1 files again"
                                      + (" once GAP flips the anomeric carbon of this residue" if fixed_by == "code" else ""))]))
    # runs that are different stereoisomers
    stereo = f.get("stereo") or {}
    runs = (f.get("frames") or {}).get("runs") or {}
    differ = []
    for c in stereo.get("centres") or []:
        p = [x for x in c["positive"] if x is not None]
        if len(p) > 1 and min(p) < STEREO_LOW and max(p) > STEREO_HIGH:
            differ.append({**{k: c[k] for k in ("atom", "residue", "residue_number", "neighbours")},
                           "sign_per_run": "".join("+" if x is not None and x > 0.5 else ("-" if x is not None else "?")
                                                   for x in c["positive"]),
                           "positive_fraction_per_run": c["positive"]})
    if differ and runs.get("basis") in ("recorded", "assumed"):
        names = stereo.get("runs") or []
        # the runs to leave out: those in the minority for the centres that differ
        minority_runs = set()
        for d in differ:
            plus = [i for i, s in enumerate(d["sign_per_run"]) if s == "+"]
            minus = [i for i, s in enumerate(d["sign_per_run"]) if s == "-"]
            minority_runs.update(plus if len(plus) < len(minus) else minus)
        exclude = [names[i] for i in sorted(minority_runs)] if len(minority_runs) < len(names) else None
        centres = "; ".join(f"{d['atom']} of {d['residue']} {d['residue_number']} ({d['sign_per_run']})" for d in differ)
        out.append(_issue("stereo-differs-between-runs",
                          f"The runs are not the same stereoisomer: {centres} is inverted between runs, so the "
                          f"populations pool two molecules.",
                          {"centres": differ, "runs": names, "runs_basis": runs.get("basis"),
                           "frames_read_per_run": stereo.get("frames_read"),
                           "runs_that_do_not_match_the_topology": [r["run"] for r in (f.get("segments") or {}).get("runs") or []
                                                                   if not r["matches"]],
                           "test": "sign of the signed volume of the carbon and three heavy neighbours: positive in "
                                   f"fewer than {STEREO_LOW:.0%} of the frames read of one run and more than "
                                   f"{STEREO_HIGH:.0%} of another"},
                          files=[x for x in [raw("trajectory.xtc", "trajectory (frames read)"),
                                             raw("manifest.json", "run boundaries")] if x],
                          actions=[_act("reanalyse", "leave out the runs of the other stereoisomer",
                                        exclude_replicas=exclude),
                                   _act("redo", "rerun the affected runs")]))
    elif differ:
        out.append(_issue("stereo-differs-between-runs",
                          "Parts of the trajectory are different stereoisomers, but its runs are not recorded.",
                          {"centres": differ, "runs_basis": runs.get("basis")}, severity=BLOCKING,
                          files=[x for x in [raw("trajectory.xtc", "trajectory")] if x],
                          actions=[_act("redo", "rerun the entry")]))
    # a centre type whose configuration is the minority in the collection
    # (not when the read-back from the coordinates already settled it: the entry's structure is reported above,
    # or its residues are confirmed to be those of the name and the entries of the other sign are the ones that
    # differ; nor for a centre that differs between the runs of the entry)
    explained = ({i["code"] for i in out} & {"anomer-exchanged", "structure-contradicts-name"}) or outcome == "match"
    differing = {f"{d['residue']} {d['atom']} ({', '.join(d['neighbours'])})" for d in differ}
    for m in (ctx.get("stereo_minority") or {}).get(gid, []):
        if explained or m["centre"] in differing:
            continue
        out.append(_issue("stereo-family-minority",
                          f"The stereocentre {m['centre']} has the other configuration than in "
                          f"{m['entries_with_the_other_sign']} other entries; check that the residue is what its name says.",
                          dict(m), key=f"stereo-family-minority:{m['centre']}",
                          files=[x for x in [raw("topology.pdb", "structure")] if x],
                          actions=[_act("none", "the configuration is right for this residue"),
                                   _act("redo", "the residue was built wrong")]))
    return out


def _wurcs_kind(stored: Optional[str], expected: Optional[str]) -> str:
    """What differs between a stored WURCS and the one of the name: ``ring`` (a furanose written with the ring
    of a pyranose, or the reverse), ``substituent missing``, ``substituent added`` or ``other``."""
    if not stored or not expected:
        return "missing"

    def residues(w: str) -> List[str]:          # the unique-residue code of every residue, in order
        m = re.match(r"^WURCS=2\.0/\d+,\d+,\d+/((?:\[[^\]]*\])+)/([^/]*)/", w)
        if not m:
            return []
        unique = re.findall(r"\[([^\]]*)\]", m.group(1))
        return [unique[int(i) - 1] for i in m.group(2).split("-") if i.isdigit() and int(i) <= len(unique)]

    rings = lambda w: Counter(p for c in residues(w) for p in c.split("_")[1:] if re.match(r"^\d-\d$", p))  # noqa: E731
    if rings(stored) != rings(expected):
        return "ring"
    subs = lambda w: Counter(p for c in residues(w) for p in c.split("_")[1:] if "*" in p)  # noqa: E731
    if subs(expected) - subs(stored):
        return "substituent missing"
    if subs(stored) - subs(expected):
        return "substituent added"
    return "other"


def notation_issues(f: Mapping[str, Any], ctx: Mapping[str, Any]) -> List[Dict[str, Any]]:
    gid = f["glycoshape_id"]
    out: List[Dict[str, Any]] = []
    entry = f.get("entry") or {}
    notation = f.get("notation") or {}
    variants = notation.get("variants") or {}
    raw_name = (f.get("raw") or {}).get("glycam_name")
    static, raw = _files(gid, raw_name)
    name = entry.get("name")
    # the name itself
    if notation and notation.get("name_is_glycam") is False:
        reason = "a GlyTouCan accession" if notation.get("name_is_accession") else "not read as a GLYCAM sequence"
        rename_to = raw_name if raw_name and raw_name != name else None
        title = (f"The entry's name '{name}' is {reason}, so its residues cannot be read from it"
                 + (f"; the raw archive has it as '{raw_name}'." if rename_to else "."))
        out.append(_issue("name-not-glycam", title,
                          {"name": name, "why": reason, "parser": notation.get("name_error"),
                           "crabwurcs": (variants.get("archetype") or {}).get("name_error") or None,
                           "glycam_name_in_raw_archive": raw_name},
                          files=[static("data.json", "inventory record")] + ([raw("manifest.json", "raw entry (its GLYCAM name)")] if raw_name else []),
                          actions=[_act("rename", "give the entry its GLYCAM name", to=rename_to)]))
    # stored WURCS
    wrong = []
    for variant in VARIANTS:
        v = variants.get(variant) or {}
        if v.get("stored_equal") is False or (v.get("expected") and not v.get("stored_normalised") and v.get("stored_wurcs")):
            wrong.append({"variant": variant, "stored": v.get("stored_wurcs"), "from_the_name": v.get("expected"),
                          "kind": _wurcs_kind(v.get("stored_normalised") or v.get("stored_wurcs"), v.get("expected")),
                          "glytoucan_stored": v.get("glytoucan")})
    if wrong:
        kinds = sorted({w["kind"] for w in wrong})
        which = ", ".join(w["variant"] for w in wrong)
        out.append(_issue("stored-wurcs-mismatch",
                          f"The stored {which} WURCS do not describe the glycan of the name ({', '.join(kinds)}); "
                          "the accessions found for them name that structure.",
                          {"variants": wrong, "kinds": kinds,
                           "crabwurcs": "computed from the GLYCAM name with the fixed crabWURCS (no bundled table)"},
                          files=[static("data.json", "inventory record")],
                          actions=[_act("reanalyse", "the re-analysis writes the WURCS again and looks the accessions up")]))
    # stored IUPAC
    iupac_wrong = []
    for variant in VARIANTS:
        v = variants.get(variant) or {}
        if v.get("iupac_equal") is False:
            iupac_wrong.append({"variant": variant, "stored_iupac": v.get("stored_iupac"),
                                "reads_as": v.get("iupac_wurcs"), "name_gives": v.get("expected")})
    if iupac_wrong:
        out.append(_issue("stored-iupac-mismatch",
                          f"The stored IUPAC name of {', '.join(x['variant'] for x in iupac_wrong)} reads as another "
                          "glycan than the GLYCAM name.",
                          {"variants": iupac_wrong}, files=[static("data.json", "inventory record")],
                          actions=[_act("reanalyse", "the re-analysis writes the IUPAC names again")]))
    # inventory fields
    tree = f.get("tree") or {}
    broken = {}
    components = entry.get("components")
    if isinstance(components, dict) and tree.get("residues") is not None:
        total = sum(v for v in components.values() if isinstance(v, (int, float)))
        if total != tree["residues"]:
            broken["components"] = {"stored": components, "residues_stored": total, "residues_of_the_glycan": tree["residues"]}
    arche = variants.get("archetype") or {}
    if arche.get("stored_iupac") and arche.get("iupac_wurcs") is None and arche.get("iupac_error"):
        broken["iupac"] = {"stored": arche.get("stored_iupac"), "not_read": arche.get("iupac_error")}
    if entry.get("mass") in (None, "", 0):
        broken["mass"] = {"stored": entry.get("mass")}
    if broken:
        renamed_first = notation.get("name_is_glycam") is False
        out.append(_issue("inventory-fields-broken",
                          f"The inventory fields {', '.join(sorted(broken))} are broken"
                          + (" because the name is not a GLYCAM sequence." if renamed_first else "."),
                          {"fields": broken}, files=[static("data.json", "inventory record")],
                          resolved_by_reanalysis=not renamed_first, needs_owner=renamed_first,
                          actions=([_act("rename", "fields follow from a GLYCAM name")] if renamed_first else [])
                          + [_act("reanalyse", "the re-analysis derives the fields from the name again")]))
    # mass of the structure against the inventory (the name)
    elements = (f.get("structure") or {}).get("elements")
    mass = _number(entry.get("mass"))
    if elements and mass and tree.get("groups"):
        g = tree["groups"]
        anions = g["sulfate"] + g["sialic"] + g["uronic"] + g["phosphocholine"]
        structure_mass = sum(MONOISOTOPIC.get(e.upper() if len(e) == 1 else e, 0.0) * n for e, n in elements.items()) \
            + anions * MONOISOTOPIC["H"]
        diff = structure_mass - mass
        if abs(diff) > MASS_TOLERANCE:
            explanation = None
            for n_so3 in (0, 1, -1, 2, -2):
                for k in range(-3, 4):
                    if (n_so3, k) != (0, 0) and abs(diff - n_so3 * SO3 - k * MONOISOTOPIC["H"]) <= MASS_TOLERANCE:
                        explanation = (n_so3, k)
                        break
                if explanation:
                    break
            words = []
            if explanation:
                n_so3, k = explanation
                if n_so3:
                    words.append(f"{abs(n_so3)} sulfate group{'s' if abs(n_so3) > 1 else ''} {'more' if n_so3 > 0 else 'fewer'}")
                if k:
                    words.append(f"{abs(k)} hydrogen{'s' if abs(k) > 1 else ''} {'more' if k > 0 else 'fewer'}")
            what = " and ".join(words) if words else f"{diff:+.2f} Da"
            out.append(_issue("mass-mismatch",
                              f"The structure file holds {what} than the glycan of the name (mass {mass:.2f} by the "
                              f"name, {structure_mass:.2f} by the atoms).",
                              {"mass_inventory": mass, "mass_structure": round(structure_mass, 2),
                               "difference": round(diff, 2), "elements": elements,
                               "sulfur_atoms": elements.get("S", 0), "sulfate_groups_in_name": g["sulfate"],
                               "anionic_groups_in_name": anions,
                               "rule": "monoisotopic mass of the atoms plus one hydrogen per anionic group of the name "
                                       "(sulfate, sialic acid, uronic acid, phosphocholine) against archetype.mass",
                               "elements_from": (f.get("structure") or {}).get("elements_from")},
                              files=[x for x in [raw("topology.pdb", "simulated structure")] if x]
                              + [static("output/level_1/PDB/alpha.pdb", "Level 1 structure")],
                              actions=[_act("rename", "the name misses what the structure has"),
                                       _act("redo", "the structure was built wrong"),
                                       _act("none", "the protonation state is intended")]))
    # duplicates
    others = (ctx.get("duplicates") or {}).get(gid)
    if others:
        out.append(_issue("duplicate-entry",
                          f"The same glycan with the same reducing-end anomer is also published as {', '.join(others)}.",
                          {"also_published_as": others, "reducing_end_anomer": declared_anomer(name),
                           "archetype_wurcs": arche.get("open_wurcs") or arche.get("stored_wurcs")},
                          files=[static("data.json", "inventory record")],
                          actions=[_act("merge-ids", "keep one ID for the glycan"),
                                   _act("withdraw", "take this entry out"),
                                   _act("none", "keep both (for instance as two independent simulations)")]))
    # class rules
    if tree.get("residues"):
        flags = []
        severity = None
        if tree.get("n_glycan_like"):
            flags.append("N-glycan with a core that is not Man(b1-4)GlcNAc(b1-4)GlcNAc")
            severity = SHOULD_FIX
        beta_arm = [x for x in tree.get("unusual_on_beta_man") or [] if re.match(r"^Man\(b", x)]
        other_unusual = [x for x in tree.get("unusual_on_beta_man") or [] if x not in beta_arm]
        if beta_arm:
            flags.append("a beta-Man arm on the core beta-Man: " + ", ".join(beta_arm))
            severity = SHOULD_FIX
        if other_unusual:
            flags.append("unusual substituent on the core beta-Man: " + ", ".join(other_unusual))
            severity = severity or NOTE
        if tree.get("open_linkages"):
            flags.append("linkage positions left open: " + ", ".join(tree["open_linkages"]))
            severity = severity or NOTE
        if flags:
            out.append(_issue("probable-input-error",
                              "The glycan is not canonical: " + "; ".join(flags) + ".",
                              {"flags": flags, "class": tree.get("class"), "subclass": tree.get("subclass"),
                               "iupac_from_the_tree": tree.get("iupac"), "tree_from": tree.get("source")},
                              severity=severity, files=[static("data.json", "inventory record")],
                              actions=[_act("rename", "correct the name"), _act("redo", "build and simulate the intended glycan"),
                                       _act("withdraw", "take the entry out"), _act("none", "the glycan is meant so")]))
    if not (entry.get("glytoucan") or {}).get("archetype"):
        out.append(_issue("glytoucan-missing", "The archetype has no GlyTouCan accession.",
                          {"glytoucan": entry.get("glytoucan")}, files=[static("data.json", "inventory record")],
                          actions=[_act("none", "register the glycan with GlyTouCan; a later re-analysis finds the accession")]))
    return out


def metadata_issues(f: Mapping[str, Any], ctx: Mapping[str, Any]) -> List[Dict[str, Any]]:
    gid = f["glycoshape_id"]
    out: List[Dict[str, Any]] = []
    entry = f.get("entry") or {}
    frames = f.get("frames") or {}
    raw_name = (f.get("raw") or {}).get("glycam_name")
    static, raw = _files(gid, raw_name)
    data_file = [static("data.json", "inventory record")]
    length = _number(entry.get("length_us")) or 0.0
    k = frames.get("runs_if_10001")
    if length == 0:
        by_frames = round(k * US_PER_RUN, 3) if k else None
        out.append(_issue("sim-time-missing",
                          "No simulated time is recorded"
                          + (f"; the frames are {k} runs of 10,001 frames ({by_frames} microseconds at 50 ps per frame)." if k else "."),
                          {"length_us": entry.get("length_us"), "frames": frames.get("archive"), "runs_if_10001": k,
                           "time_by_frames_us": by_frames,
                           "assumption": "500 ns written every 50 ps is 10,001 frames (the GOTW recipe)"},
                          files=data_file, actions=[_act("edit-metadata", "record the simulated time", length_us=by_frames)]))
    elif k and abs(length - US_PER_RUN * k) > 1e-9:
        out.append(_issue("sim-time-frames-disagree",
                          f"{length:g} microseconds are recorded, but the frames are {k} runs of 10,001 frames "
                          f"({US_PER_RUN * k:g} microseconds at 50 ps per frame).",
                          {"length_us": length, "frames": frames.get("archive"), "runs_if_10001": k,
                           "time_by_frames_us": round(US_PER_RUN * k, 3),
                           "assumption": "500 ns written every 50 ps is 10,001 frames (the GOTW recipe)"},
                          files=data_file + [x for x in [raw("manifest.json", "frames of the raw entry")] if x],
                          actions=[_act("edit-metadata", "correct the recorded time", length_us=round(US_PER_RUN * k, 3)),
                                   _act("none", "the runs were longer than 500 ns")]))
    if 0 < length < SHORT_TIME_US:
        out.append(_issue("sim-time-short", f"Only {length:g} microseconds of simulation are recorded.",
                          {"length_us": length, "frames": frames.get("archive"), "reference_us": SHORT_TIME_US},
                          files=data_file, actions=[_act("request-replicas", "simulate more"), _act("none", "enough for this glycan")]))
    registry = f.get("registry") or {}
    reg_length = _number(registry.get("simulation_length"))
    if reg_length is not None and reg_length > REGISTRY_LENGTH_NS:
        out.append(_issue("sim-time-unit",
                          f"The registry records {reg_length:g} in its microsecond field, a value in nanoseconds "
                          f"(the inventory has {entry.get('length_us')} microseconds).",
                          {"registry_simulation_length": reg_length, "inventory_length_us": entry.get("length_us"),
                           "registry_value_as_microseconds": reg_length / 1000},
                          files=[{"store": "registry", "path": "submission_ids.json", "what": f"record of {gid}"}],
                          actions=[_act("edit-metadata", "record the registry's time in microseconds",
                                        length_us=_number(entry.get("length_us")) or reg_length / 1000)]))
    blank = [k2 for k2 in ("package", "temperature", "pressure") if entry.get(k2) in (None, "", 0, 0.0)]
    if blank:
        out.append(_issue("sim-metadata-blank", f"The simulation's {', '.join(blank)} {'is' if len(blank) == 1 else 'are'} not recorded.",
                          {"blank": blank, **{k2: entry.get(k2) for k2 in ("package", "forcefield", "temperature", "pressure", "salt", "length_us")}},
                          files=data_file, actions=[_act("edit-metadata", "record the simulation conditions")]))
    ff = entry.get("forcefield")
    if ff in FORCEFIELD_PLACEHOLDERS or not ff:
        out.append(_issue("forcefield-placeholder",
                          f"The force field is recorded as '{ff}'." if ff else "No force field is recorded.",
                          {"forcefield": ff, "package": entry.get("package")}, files=data_file,
                          actions=[_act("edit-metadata", "record the force field")]))
    salt = _number(entry.get("salt"))
    if salt is not None and salt not in SALTS_OF_THE_COLLECTION and salt > 0:
        out.append(_issue("salt-implausible",
                          f"The salt concentration is recorded as {salt:g}, not one of 0, 150 or 200 mM.",
                          {"salt": salt, "collection_values_mM": list(SALTS_OF_THE_COLLECTION),
                           "registry": registry.get("salt_concentration")},
                          files=data_file, actions=[_act("edit-metadata", "record the concentration in mM")]))
    elif salt == 0 and entry.get("package"):
        out.append(_issue("salt-zero", "The salt concentration is recorded as 0: no salt, or not recorded.",
                          {"salt": salt, "package": entry.get("package")}, files=data_file,
                          actions=[_act("none", "simulated without salt"), _act("edit-metadata", "record the concentration")]))
    return out


def raw_issues(f: Mapping[str, Any], ctx: Mapping[str, Any]) -> List[Dict[str, Any]]:
    gid = f["glycoshape_id"]
    out: List[Dict[str, Any]] = []
    rawf = f.get("raw") or {}
    raw_name = rawf.get("glycam_name")
    static, raw = _files(gid, raw_name)
    irregular = rawf.get("source_irregularities")
    if irregular:
        truncated = bool(irregular.get("truncated"))
        items = irregular.get("items") or []
        kinds = sorted({i.get("kind") for i in items if i.get("kind")})
        out.append(_issue("raw-source-repaired",
                          f"The source trajectory was archived with a repair ({', '.join(kinds)}): "
                          f"{irregular.get('frames_kept')} of {irregular.get('frames_in_source')} frames kept"
                          + ("; the source file was cut short." if truncated else "."),
                          {"frames_in_source": irregular.get("frames_in_source"), "frames_kept": irregular.get("frames_kept"),
                           "frames_dropped": irregular.get("frames_dropped"), "bytes_ignored": irregular.get("bytes_ignored"),
                           "truncated": truncated, "counts": irregular.get("counts"),
                           "items": [{k: v for k, v in i.items() if k in ("kind", "action", "source_frame", "model_record",
                                                                           "atoms_found", "atoms_expected", "followed_by",
                                                                           "file_truncated", "file_size", "count")} for i in items]},
                          severity=SHOULD_FIX if truncated else NOTE, needs_owner=truncated,
                          files=[x for x in [raw("manifest.json", "source_irregularities")] if x],
                          actions=([_act("request-replicas", "the end of the trajectory is lost; ask for the complete file "
                                                             "or more sampling")] if truncated else [])
                          + [_act("none", "the archived frames are intact")]))
    frames = f.get("frames") or {}
    if frames.get("archive") and frames.get("removed") is not None and frames["removed"] > 0:
        fraction = frames["removed"] / frames["archive"]
        if fraction > FRAMES_REMOVED_NOTE:
            runs = frames.get("runs") or {}
            per_run = None
            exclude = None
            if runs.get("kept") and runs.get("basis") in ("recorded", "assumed"):
                per_run = [{"run": n, "frames": l, "kept": kk} for n, l, kk in zip(runs["names"], runs["lengths"], runs["kept"])]
                exclude = [r["run"] for r in per_run if r["frames"] and r["kept"] / r["frames"] < 0.5] or None
            heavy = fraction > FRAMES_REMOVED_SHOULD_FIX
            out.append(_issue("frames-removed",
                              f"The frame-quality filter removed {frames['removed']:,} of {frames['archive']:,} frames "
                              f"({fraction:.1%}).",
                              {"removed": frames["removed"], "of": frames["archive"], "fraction": round(fraction, 4),
                               "per_run": per_run, "filter": "a covalent bond longer than 3.0 angstrom or a non-finite coordinate"},
                              severity=SHOULD_FIX if heavy else NOTE, needs_owner=heavy,
                              files=[static("output/pca.csv", "frames kept (source_frame)")]
                              + [x for x in [raw("trajectory.xtc", "trajectory")] if x],
                              actions=([_act("reanalyse", "leave out the runs that lost most frames", exclude_replicas=exclude)] if heavy and exclude else [])
                              + ([_act("request-replicas", "replace the damaged runs"), _act("redo", "rerun the entry")] if heavy else [])
                              + [_act("none", "the remaining frames are enough")]))
    segments = f.get("segments") or {}
    bad = [r for r in segments.get("runs") or [] if not r["matches"]]
    if bad:
        all_runs = [r["run"] for r in segments.get("runs") or []]
        why = set()
        for r in bad:
            for t in r["frames"]:
                if t["centres_inverted"]:
                    why.add("a stereocentre is inverted against the topology")
                if t["bonds_over_cutoff"]:
                    why.add("bonds of the topology are stretched")
        frames_text = ", ".join(f"{r['run']} (frames {r['first_frame']:,} to {r['last_frame']:,})" for r in bad)
        exclude = [r["run"] for r in bad] if len(bad) < len(all_runs) else None
        out.append(_issue("run-topology-mismatch",
                          f"The coordinates of {frames_text} do not match the entry's topology: "
                          + " and ".join(sorted(why)) + ".",
                          {"runs": bad, "runs_of_the_entry": all_runs,
                           "runs_basis": ((f.get("frames") or {}).get("runs") or {}).get("basis"),
                           "bonds_checked": segments.get("bonds"), "bonds_from": segments.get("bonds_from"),
                           "stereocentres_checked": segments.get("centres"),
                           "test": "first and last frame of each run: topology bonds longer than the covalent radii "
                                   "+ 0.35 angstrom (and than 3.0 angstrom, the frame-quality filter), stereocentres "
                                   "whose signed volume has the other sign than in topology.pdb"},
                          files=[x for x in [raw("trajectory.xtc", "trajectory"), raw("topology.pdb", "topology"),
                                             raw("structure.mol2", "bonds"), raw("manifest.json", "runs")] if x],
                          actions=([_act("reanalyse", "leave out the runs that do not match", exclude_replicas=exclude)]
                                   if exclude else [])
                          + [_act("redo", "rerun the entry, or the runs that do not match")]))
    if frames.get("archive") is not None and frames["archive"] < SHORT_TRAJECTORY_FRAMES:
        length = _number((f.get("entry") or {}).get("length_us"))
        out.append(_issue("trajectory-short",
                          f"The trajectory has only {frames['archive']} frames"
                          + (f" for {length:g} microseconds recorded." if length else "."),
                          {"frames": frames["archive"], "length_us": length,
                           "ps_per_frame_implied": round(length * 1e6 / max(frames["archive"] - 1, 1), 1) if length else None,
                           "threshold_frames": SHORT_TRAJECTORY_FRAMES},
                          files=[x for x in [raw("manifest.json", "frames")] if x],
                          actions=[_act("request-replicas", "ask for the full trajectory or more sampling"),
                                   _act("redo", "simulate again"), _act("withdraw", "take the entry out")]))
    info = rawf.get("info_json") or {}
    if info.get("emails"):
        out.append(_issue("personal-data-in-raw",
                          "The raw archive's info.json holds an e-mail address"
                          + (f" (fields: {', '.join(info.get('fields') or [])})." if info.get("fields") else "."),
                          {"emails_masked": info["emails"], "fields": info.get("fields")},
                          files=[x for x in [raw("info.json", "raw info.json")] if x],
                          actions=[_act("edit-metadata", "remove the personal fields from the raw info.json before "
                                                         "the raw archive is shared")]))
    entry = f.get("entry") or {}
    n_inv, n_model = entry.get("pca_variance_values"), entry.get("pca_model_values")
    if n_inv and n_model and n_inv > n_model:
        out.append(_issue("pca-variance-accumulated",
                          f"archetype.pca_variance holds {n_inv} explained-variance values for a model of {n_model} components.",
                          {"values_in_inventory": n_inv, "components_in_model": n_model},
                          files=[static("data.json", "archetype.pca_variance"),
                                 static("landscape/landscape_metadata.json", "the saved model")],
                          actions=[_act("reanalyse", "the re-analysis writes the variance of its own model")]))
    return out


def registry_issues(f: Mapping[str, Any], ctx: Mapping[str, Any]) -> List[Dict[str, Any]]:
    gid = f["glycoshape_id"]
    name = (f.get("entry") or {}).get("name")
    raw_name = (f.get("raw") or {}).get("glycam_name")
    several = ctx.get("names_with_several_ids") or {}
    ids = several.get(raw_name) or several.get(name)
    if not ids:
        return []
    others = [i for i in ids if i["glycoshape_id"] != gid]
    if not others:
        return []
    return [_issue("id-several-for-name",
                   f"The registry has {len(ids)} GlycoShape IDs for this GLYCAM name: "
                   + ", ".join(f"{i['glycoshape_id']} ({i['status']})" for i in ids) + ".",
                   {"glycam_name": raw_name or name, "ids": ids},
                   files=[{"store": "registry", "path": "registry-seed.json", "what": "conflicts.name_with_several_ids"}],
                   actions=[_act("merge-ids", f"keep {gid} for the glycan and retire the others", keep=gid)])]


def replica_issues(f: Mapping[str, Any], ctx: Mapping[str, Any]) -> List[Dict[str, Any]]:
    gid = f["glycoshape_id"]
    r = f.get("replicas")
    if not r or r.get("verdict") in (None, "ok"):
        return []
    code = {"more_sampling_needed": "replicas-more-sampling-needed",
            "more_sampling_suggested": "replicas-more-sampling-suggested",
            "not_assessable": "replicas-not-assessable"}.get(r["verdict"])
    if code is None:
        return []
    reasons = r.get("reasons") or []
    named = set()
    for reason in reasons:
        for item in reason.get("clusters") or []:
            named.update(item.get("replicas") or [])
        for item in reason.get("replicas") or []:
            if isinstance(item, dict) and item.get("replica"):
                named.add(item["replica"])
    texts = [reason.get("text") for reason in reasons if reason.get("text")]
    title = {"replicas-more-sampling-needed": "The independent runs disagree on populations, clusters and rotamers: more sampling is needed.",
             "replicas-more-sampling-suggested": "The independent runs disagree on at least one criterion: more sampling is suggested.",
             "replicas-not-assessable": "The runs of this entry cannot be compared."}[code]
    actions = []
    if code != "replicas-not-assessable":
        actions.append(_act("request-replicas", "more independent runs"))
        if named and len(named) < (r.get("replicas") or 0):
            actions.append(_act("reanalyse", "leave out the runs that disagree (only if they are known to be faulty)",
                                exclude_replicas=sorted(named)))
        actions.append(_act("none", "the disagreement is understood"))
    else:
        actions.append(_act("none", "nothing to compare"))
    return [_issue(code, title, {"verdict": r["verdict"], "reasons": [x.get("code") for x in reasons], "texts": texts,
                                 "basis": r.get("basis"), "replicas": r.get("replicas"),
                                 "full_length_replicas": r.get("full_length_replicas"),
                                 "population_tv_mean": r.get("population_tv_mean"),
                                 "population_tv_floor": r.get("population_tv_floor"),
                                 "replicas_named_in_reasons": sorted(named)},
                   files=[{"store": "replicas", "path": f"{gid}.json", "what": "replicas.json"}], actions=actions)]


RULES = (identity_issues, notation_issues, metadata_issues, raw_issues, registry_issues, replica_issues)
_ORDER = {s: i for i, s in enumerate(SEVERITIES)}


def entry_issues(facts: Mapping[str, Any], context: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    """The issues of one entry, most severe first (then by code)."""
    context = context or {}
    issues: List[Dict[str, Any]] = []
    for rule in RULES:
        issues.extend(rule(facts, context))
    keys = Counter(i["key"] for i in issues)
    seen: Counter = Counter()
    for issue in issues:                      # keys stay unique within an entry
        if keys[issue["key"]] > 1:
            seen[issue["key"]] += 1
            issue["key"] = f"{issue['key']}#{seen[issue['key']]}"
    issues.sort(key=lambda i: (_ORDER[i["severity"]], i["code"], i["key"]))
    return issues


def document(facts: Mapping[str, Any], issues: Sequence[Mapping[str, Any]], *,
             collection: bool = False) -> Dict[str, Any]:
    """The ``issues.json`` of an entry."""
    entry = facts.get("entry") or {}
    counts = Counter(i["severity"] for i in issues)
    return {
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
        "glycoshape_id": facts["glycoshape_id"], "glycam_name": entry.get("name"),
        "glytoucan": (entry.get("glytoucan") or {}).get("archetype"),
        "generator": GENERATOR,
        "inputs": {**(facts.get("inputs") or {}), "collection": collection},
        "checks_failed": facts.get("errors") or [],
        "counts": {s: counts.get(s, 0) for s in SEVERITIES},
        "issues": list(issues),
    }


def encode(doc: Mapping[str, Any]) -> bytes:
    return (json.dumps(doc, ensure_ascii=False, indent=1, allow_nan=False) + "\n").encode("utf-8")


def write(doc: Mapping[str, Any], path: PathLike) -> Path:
    """Write a JSON document atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(encode(doc))
    os.replace(temporary, path)
    return path
