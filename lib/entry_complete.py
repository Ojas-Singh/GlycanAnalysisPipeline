"""Complete the static-ready result of one entry into a published glycan folder.

The analysis of an entry on a compute node (``scripts/run_entry.py``) stops
before everything that needs an identifier, submission metadata or the
network.  It reaches the publishing side as an extracted and validated
*bundle* directory::

    bundle.json    schema "glycoshape-entry-bundle": glycam_name, pipeline, levels,
                   files {relative path: {size, sha256}}, source, extends
    archive/       the raw archive entry (lib/archive.py)
    static/        the static-ready folder (output/, landscape/)
    result.json    the result of the run
    run.log

:func:`complete_entry` turns it into ``GSxxxxx/`` exactly as the static stage
of ``main.py`` writes that folder, from the bundle and a *submission record*
alone: no other glycan, no PocketBase.  The record says what the registry of
the caller knows (:class:`Submission`)::

    {
      "schema": "glycoshape-submission", "schema_version": 1,
      "glycoshape_id": "GS01067",
      "glycam_name": "DManpa1-2DManpa1-OH",
      "simulation": {
        "simulation_length_ns": 2500.0,       aggregated over all replicas; or "simulation_length_us"
        "md_package": "GROMACS", "force_field": "GLYCAM_06j",
        "temperature_k": 300.0, "pressure_bar": 1.0, "salt_concentration_mm": 150.0,
        "source_type": "manual", "comments": ""
      },
      "glytoucan_id": "",               optional: the accession to use when none is found
      "curated": {"common_names": [], "aliases": [], "description": "", "keywords": []}
    }

The simulation fields have the names they have in ``source`` of
``bundle.json`` (:func:`submission_from_bundle` copies them from there).
Every quantity carries its unit in its name, and a name without one
(``simulation_length``, ``temperature``, ``salt_concentration``...) is
refused: the length of the simulations has been entered in nanoseconds by
one client and in microseconds by another.  The units are converted here,
once, to those of ``data.json`` (:meth:`Submission.simulation`): ``length``
in microseconds (``simulation_length_ns / 1000``), ``temperature`` in
kelvin, ``pressure`` in bar, ``salt`` in mM.  The length is the aggregated
one of the entry: for an entry that extends a published one, the published
length plus that of the new run (``source`` of a bundle states the new run
only).

The shared code is ``lib.glystatic.write_identity_files`` (names, WURCS,
accessions, ``data.json``, ``glygen.json``, ``glycosmos.json``, ``snfg.svg``),
``lib.static_ready`` (the files of the folder and the REMARK block),
``lib.glymeta`` (keywords, common names, description) and
``lib.glybake.write_archive`` (``GSxxxxx.zip``).  The web lookups are bounded
and recorded (``lib.lookups``); ``offline=True`` makes none.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SUBMISSION_SCHEMA = "glycoshape-submission"
SUBMISSION_SCHEMA_VERSION = 1
BUNDLE_SCHEMA = "glycoshape-entry-bundle"
BUNDLE_SCHEMA_VERSION = 1
BUNDLE_FILE = "bundle.json"
#: Files of a bundle that ``bundle.json`` does not have to list.
UNLISTED_BUNDLE_FILES = (BUNDLE_FILE, "run.log")

_GLYCOSHAPE_ID = re.compile(r"GS\d{5}")
_GLYTOUCAN_ID = re.compile(r"G\d{5}[A-Z]{2}")
_SIMULATION_KEYS = (
    "simulation_length_ns", "simulation_length_us", "md_package", "force_field", "temperature_k", "pressure_bar",
    "salt_concentration_mm", "source_type", "comments",
)
#: The fields of ``source`` of a bundle that a submission takes over as they are.
BUNDLE_SOURCE_KEYS = (
    "simulation_length_ns", "md_package", "force_field", "temperature_k", "pressure_bar", "salt_concentration_mm",
    "comments",
)
#: Spellings without a unit and the key to use instead.
_REFUSED_SIMULATION_KEYS = {
    "simulation_length": "simulation_length_ns or simulation_length_us",
    "length": "simulation_length_ns or simulation_length_us",
    "simulationLength": "simulation_length_ns",
    "temperature": "temperature_k", "pressure": "pressure_bar",
    "salt_concentration": "salt_concentration_mm", "salt": "salt_concentration_mm",
}
_CURATED_KEYS = ("common_names", "aliases", "description", "keywords")
#: Outside these ranges a value is reported: it is more often a unit mistake than a simulation.
_PLAUSIBLE = {
    "length_us": (0.01, 1000.0, "the aggregated length", "microseconds", "simulation_length_ns"),
    "temperature_k": (200.0, 500.0, "the temperature", "K", "temperature_k"),
    "pressure_bar": (0.1, 1000.0, "the pressure", "bar", "pressure_bar"),
    "salt_mm": (0.0, 5000.0, "the salt concentration", "mM", "salt_concentration_mm"),
}
_PROCESS_LOCK = threading.Lock()


class SubmissionError(ValueError):
    """The submission record cannot be used; the message lists every reason."""


class BundleError(Exception):
    """The bundle directory is not a complete, consistent bundle; the message lists every reason."""


class CompletionError(Exception):
    """The glycan folder could not be completed."""


# ---------------------------------------------------------------------------
# The submission record
# ---------------------------------------------------------------------------
def _number(value: Any, name: str, errors: List[str], positive: bool = False) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        errors.append(f"simulation.{name} must be a number, not {value!r}")
        return None
    if value < 0 or (positive and value == 0):
        errors.append(f"simulation.{name} must be {'positive' if positive else 'zero or more'}, not {value!r}")
        return None
    return float(value)


def _text(value: Any, name: str, errors: List[str]) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        errors.append(f"{name} must be text, not {value!r}")
        return ""
    return value.strip()


def _text_list(value: Any, name: str, errors: List[str]) -> Tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        errors.append(f"{name} must be a list of texts")
        return ()
    seen, kept = set(), []
    for item in (item.strip() for item in value):
        if item and item.casefold() not in seen:
            seen.add(item.casefold())
            kept.append(item)
    return tuple(kept)


@dataclass(frozen=True)
class Submission:
    """What the registry of the caller says about one entry; see the module documentation.

    Built with :meth:`parse`, which validates the record and converts the
    units.  ``warnings`` are values that are accepted and look like a
    mistake (shown to the administrator before approval).
    """

    glycoshape_id: str
    glycam_name: str
    length_us: Optional[float] = None
    package: str = ""
    forcefield: str = ""
    temperature_k: Optional[float] = None
    pressure_bar: Optional[float] = None
    salt_mm: Optional[float] = None
    source_type: str = ""
    comments: str = ""
    glytoucan_id: str = ""
    common_names: Tuple[str, ...] = ()
    aliases: Tuple[str, ...] = ()
    description: str = ""
    keywords: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = field(default=(), compare=False)

    @classmethod
    def parse(cls, record: Any) -> "Submission":
        """A submission from its JSON record; raises :class:`SubmissionError` naming every problem."""
        if isinstance(record, Submission):
            return record
        if not isinstance(record, dict):
            raise SubmissionError("the submission record is not a JSON object")
        errors: List[str] = []
        warnings: List[str] = []
        if record.get("schema", SUBMISSION_SCHEMA) != SUBMISSION_SCHEMA:
            errors.append(f"schema is {record.get('schema')!r}, expected {SUBMISSION_SCHEMA!r}")
        if record.get("schema_version", SUBMISSION_SCHEMA_VERSION) != SUBMISSION_SCHEMA_VERSION:
            errors.append(f"schema_version is {record.get('schema_version')!r}, this code reads "
                          f"{SUBMISSION_SCHEMA_VERSION}")
        known = ("schema", "schema_version", "glycoshape_id", "glycam_name", "simulation", "glytoucan_id", "curated")
        for key in sorted(set(record) - set(known)):
            errors.append(f"unknown field {key!r}")

        glycoshape_id = record.get("glycoshape_id")
        if not isinstance(glycoshape_id, str) or not _GLYCOSHAPE_ID.fullmatch(glycoshape_id):
            errors.append(f"glycoshape_id must be GS and five digits, not {glycoshape_id!r}")
            glycoshape_id = ""
        glycam_name = record.get("glycam_name")
        if not isinstance(glycam_name, str) or not glycam_name or glycam_name != glycam_name.strip() \
                or re.search(r"[\x00-\x1f\x7f]", glycam_name):
            errors.append(f"glycam_name must be the full GLYCAM name, not {glycam_name!r}")
            glycam_name = ""

        simulation = record.get("simulation", {})
        if not isinstance(simulation, dict):
            errors.append("simulation must be an object")
            simulation = {}
        for key in sorted(set(simulation) - set(_SIMULATION_KEYS)):
            if key in _REFUSED_SIMULATION_KEYS:
                errors.append(f"simulation.{key} is not accepted: use {_REFUSED_SIMULATION_KEYS[key]} "
                              "(every quantity names its unit)")
            else:
                errors.append(f"unknown field simulation.{key}")
        length_ns = _number(simulation.get("simulation_length_ns"), "simulation_length_ns", errors)
        length_us = _number(simulation.get("simulation_length_us"), "simulation_length_us", errors)
        if length_ns is not None:
            converted = length_ns / 1000.0
            if length_us is not None and not math.isclose(converted, length_us, rel_tol=1e-9, abs_tol=1e-12):
                errors.append(f"simulation.simulation_length_ns ({length_ns:g} ns) and "
                              f"simulation.simulation_length_us ({length_us:g} us) do not agree")
            length_us = converted
        values = {
            "length_us": length_us,
            "temperature_k": _number(simulation.get("temperature_k"), "temperature_k", errors, positive=True),
            "pressure_bar": _number(simulation.get("pressure_bar"), "pressure_bar", errors),
            "salt_mm": _number(simulation.get("salt_concentration_mm"), "salt_concentration_mm", errors),
        }
        for key, (low, high, what, unit, field_name) in _PLAUSIBLE.items():
            value = values[key]
            if value is None:
                warnings.append(f"simulation.{field_name} is not given")
            elif not low <= value <= high:
                warnings.append(f"{what} is {value:g} {unit}: check the unit")

        curated = record.get("curated", {})
        if not isinstance(curated, dict):
            errors.append("curated must be an object")
            curated = {}
        for key in sorted(set(curated) - set(_CURATED_KEYS)):
            errors.append(f"unknown field curated.{key}")
        glytoucan_id = _text(record.get("glytoucan_id"), "glytoucan_id", errors)
        if glytoucan_id and not _GLYTOUCAN_ID.fullmatch(glytoucan_id):
            errors.append(f"glytoucan_id is not a GlyTouCan accession: {glytoucan_id!r}")
        submission = cls(
            glycoshape_id=glycoshape_id,
            glycam_name=glycam_name,
            package=_text(simulation.get("md_package"), "simulation.md_package", errors),
            forcefield=_text(simulation.get("force_field"), "simulation.force_field", errors),
            source_type=_text(simulation.get("source_type"), "simulation.source_type", errors),
            comments=_text(simulation.get("comments"), "simulation.comments", errors),
            glytoucan_id=glytoucan_id,
            common_names=_text_list(curated.get("common_names"), "curated.common_names", errors),
            aliases=_text_list(curated.get("aliases"), "curated.aliases", errors),
            description=_text(curated.get("description"), "curated.description", errors),
            keywords=_text_list(curated.get("keywords"), "curated.keywords", errors),
            warnings=tuple(warnings),
            **values,
        )
        if errors:
            raise SubmissionError("the submission record cannot be used: " + "; ".join(errors))
        return submission

    def simulation(self) -> Dict[str, Any]:
        """The simulation fields of ``data.json`` (``lib.glystatic.SIMULATION_FIELDS``), in its units."""
        return {
            "length": self.length_us,            # microseconds
            "package": self.package,
            "forcefield": self.forcefield,
            "temperature": self.temperature_k,   # kelvin
            "pressure": self.pressure_bar,       # bar
            "salt": self.salt_mm,                # mM
        }

    def enrichment(self) -> Dict[str, Any]:
        """The curated fields as ``lib.glymeta`` merges them into ``search_meta``."""
        from lib.glymeta import search_enrichment

        return search_enrichment(
            keywords=list(self.keywords),
            common_names=list(self.common_names),
            description=self.description,
            package=self.package,
            forcefield=self.forcefield,
            source_type=self.source_type,
            comments=self.comments,
        )

    def record(self) -> Dict[str, Any]:
        """The record in its normal form: every field, the length in nanoseconds."""
        simulation: Dict[str, Any] = {
            "simulation_length_ns": None if self.length_us is None else self.length_us * 1000.0,
            "md_package": self.package,
            "force_field": self.forcefield,
            "temperature_k": self.temperature_k,
            "pressure_bar": self.pressure_bar,
            "salt_concentration_mm": self.salt_mm,
            "source_type": self.source_type,
            "comments": self.comments,
        }
        return {
            "schema": SUBMISSION_SCHEMA,
            "schema_version": SUBMISSION_SCHEMA_VERSION,
            "glycoshape_id": self.glycoshape_id,
            "glycam_name": self.glycam_name,
            "simulation": simulation,
            "glytoucan_id": self.glytoucan_id,
            "curated": {
                "common_names": list(self.common_names),
                "aliases": list(self.aliases),
                "description": self.description,
                "keywords": list(self.keywords),
            },
        }

    def sha256(self) -> str:
        """Digest of the normal form: two records with the same content have the same one."""
        payload = json.dumps(self.record(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# The bundle
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> Tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def _levels(value: Any) -> Optional[Tuple[int, ...]]:
    """``(1, 2)`` from ``[1, 2]`` or ``["level_1", "level_2"]``; None when it is neither."""
    if not isinstance(value, list) or not value:
        return None
    levels = []
    for item in value:
        if isinstance(item, int) and not isinstance(item, bool):
            levels.append(item)
        elif isinstance(item, str) and re.fullmatch(r"(level_)?\d+", item):
            levels.append(int(item.rsplit("_", 1)[-1]))
        else:
            return None
    return tuple(sorted(set(levels)))


@dataclass
class Bundle:
    """An extracted bundle directory, read and checked by :func:`load_bundle`."""

    path: Path
    document: Dict[str, Any]
    sha256: str                         # of bundle.json
    glycam_name: str
    levels: Tuple[int, ...]
    files: Dict[str, Dict[str, Any]]
    manifest: Dict[str, Any]            # archive/manifest.json
    manifest_sha256: str
    result: Dict[str, Any]              # result.json
    hashes_verified: bool

    @property
    def archive_dir(self) -> Path:
        return self.path / "archive"

    @property
    def static_dir(self) -> Path:
        return self.path / "static"

    @property
    def pipeline(self) -> Dict[str, Any]:
        value = self.document.get("pipeline")
        return value if isinstance(value, dict) else {}

    @property
    def source(self) -> Dict[str, Any]:
        value = self.document.get("source")
        return value if isinstance(value, dict) else {}

    @property
    def extends(self) -> Optional[Dict[str, Any]]:
        value = self.document.get("extends")
        return value if isinstance(value, dict) else None

    def summary(self) -> Dict[str, Any]:
        """What a report says about the bundle."""
        return {
            "path": str(self.path),
            "bundle_sha256": self.sha256,
            "created": self.document.get("created"),
            "pipeline": self.pipeline,
            "levels": list(self.levels),
            "files": len(self.files),
            "bytes": sum(item["size"] for item in self.files.values()),
            "frames": self.manifest.get("frames"),
            "atoms": self.manifest.get("atoms"),
            "archive_manifest_sha256": self.manifest_sha256,
            "extends": self.extends,
            "hashes_verified": self.hashes_verified,
        }


def load_bundle(bundle_dir, verify_hashes: bool = True) -> Bundle:
    """Read a bundle directory and check that it is what its ``bundle.json`` describes.

    Checked: schema and version; that the directory holds exactly the listed
    files (``run.log`` may be unlisted) with the listed sizes and, with
    ``verify_hashes``, SHA-256; that the listed files are those of an archive
    entry and of a static-ready folder for the levels of the bundle; that
    the archive manifest, ``result.json`` and ``bundle.json`` name the same
    glycan and the same levels; that the run succeeded, was computed from
    the files of this archive entry and produced the static files of this
    bundle.  Raises :class:`BundleError` listing everything that fails.
    """
    from lib import archive, entry_run, static_levels

    root = Path(bundle_dir)
    errors: List[str] = []
    try:
        payload = (root / BUNDLE_FILE).read_bytes()
        document = json.loads(payload)
    except (OSError, ValueError) as exc:
        raise BundleError(f"{root / BUNDLE_FILE} cannot be read: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema") != BUNDLE_SCHEMA:
        raise BundleError(f"{root / BUNDLE_FILE} is not a {BUNDLE_SCHEMA} document")
    if document.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise BundleError(f"{BUNDLE_FILE} has schema version {document.get('schema_version')!r}, "
                          f"this code reads {BUNDLE_SCHEMA_VERSION}")
    glycam_name = document.get("glycam_name")
    if not isinstance(glycam_name, str) or not glycam_name:
        errors.append("bundle.json has no glycam_name")
        glycam_name = ""
    levels = _levels(document.get("levels"))
    if levels is None:
        errors.append(f"bundle.json levels are not a list of levels: {document.get('levels')!r}")
        levels = ()
    if not isinstance(document.get("pipeline"), dict):
        errors.append("bundle.json has no pipeline record")
    extends = document.get("extends")
    if extends is not None and not (
        isinstance(extends, dict) and isinstance(extends.get("glycam_name"), str)
        and isinstance(extends.get("manifest_sha256"), str) and isinstance(extends.get("frames"), int)
    ):
        errors.append("bundle.json extends is neither null nor {glycam_name, manifest_sha256, frames}")

    files = document.get("files")
    if not isinstance(files, dict) or not files:
        raise BundleError("bundle.json lists no files" + ("; " + "; ".join(errors) if errors else ""))
    for relative, recorded in files.items():
        parts = relative.split("/") if isinstance(relative, str) else [""]
        if any(part in ("", ".", "..") for part in parts) or "\\" in relative or relative in UNLISTED_BUNDLE_FILES[:1]:
            errors.append(f"bundle.json lists an unusable path: {relative!r}")
        elif not isinstance(recorded, dict) or not isinstance(recorded.get("size"), int) \
                or isinstance(recorded.get("size"), bool) or not isinstance(recorded.get("sha256"), str):
            errors.append(f"bundle.json has no size and SHA-256 for {relative}")
    if errors:
        raise BundleError("; ".join(errors))

    present = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    for relative in sorted(present - set(files) - set(UNLISTED_BUNDLE_FILES)):
        errors.append(f"{relative} is in the bundle directory and not in bundle.json")
    for relative in sorted(set(files) - present):
        errors.append(f"{relative} is listed in bundle.json and missing")
    for relative in sorted(set(files) & present):
        if (root / relative).is_symlink():
            errors.append(f"{relative} is a link")
        elif (root / relative).stat().st_size != files[relative]["size"]:
            errors.append(f"{relative} does not have the size bundle.json records")
        elif verify_hashes and _sha256(root / relative)[1] != files[relative]["sha256"].lower():
            errors.append(f"{relative} does not have the SHA-256 bundle.json records")

    needed = ["result.json", "archive/manifest.json"]
    needed += [f"archive/{name}" for name in (archive.TRAJECTORY_FILE, archive.TOPOLOGY_FILE, archive.MOL2_FILE)]
    needed += [f"static/{name}" for name in static_levels.required_files(levels) if name != "data.json"]
    for relative in needed:
        if relative not in files:
            errors.append(f"bundle.json does not list {relative}")
    allowed_archive = {archive.MANIFEST_FILE, archive.TRAJECTORY_FILE, archive.TOPOLOGY_FILE, archive.MOL2_FILE,
                       archive.INFO_FILE, archive.NAME_FILE}
    for relative in sorted(files):
        if relative.startswith("archive/") and relative[len("archive/"):] not in allowed_archive:
            errors.append(f"{relative} is not a file of an archive entry")
    if errors:
        raise BundleError("; ".join(errors))

    # The three documents describe one entry.
    try:
        manifest = archive.load_manifest(root / "archive")
    except archive.ArchiveError as exc:
        raise BundleError(f"archive/manifest.json: {exc}") from exc
    try:
        result = json.loads((root / "result.json").read_bytes())
    except ValueError as exc:
        raise BundleError(f"result.json is not JSON: {exc}") from exc
    if not isinstance(result, dict) or result.get("schema") != entry_run.RESULT_SCHEMA \
            or result.get("schema_version") != entry_run.RESULT_SCHEMA_VERSION:
        raise BundleError(f"result.json is not a {entry_run.RESULT_SCHEMA} document of version "
                          f"{entry_run.RESULT_SCHEMA_VERSION}")
    if result.get("status") != "ok":
        errors.append(f"the run did not succeed: result.json has status {result.get('status')!r}")
    for where, name in (("archive/manifest.json", manifest.get("glycam_name")), ("result.json", result.get("glycam_name"))):
        if name != glycam_name:
            errors.append(f"{where} is of another glycan than bundle.json: {str(name)[:80]!r}")
    manifest_files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    for name, recorded in manifest_files.items():
        listed = files.get(f"archive/{name}")
        if listed is None or (listed["size"], listed["sha256"].lower()) != (recorded.get("size"), recorded.get("sha256")):
            errors.append(f"archive/{name} is not the file its manifest records")
    for relative in files:
        if relative.startswith("archive/") and relative != "archive/manifest.json" \
                and relative[len("archive/"):] not in manifest_files:
            errors.append(f"{relative} is not in the manifest of the archive entry")
    settings = result.get("settings") if isinstance(result.get("settings"), dict) else {}
    if _levels(settings.get("cluster_levels")) != levels:
        errors.append(f"result.json was computed for levels {settings.get('cluster_levels')!r}, "
                      f"bundle.json says {list(levels)}")
    described = result.get("input") if isinstance(result.get("input"), dict) else {}
    computed_from = described.get("files") if isinstance(described.get("files"), dict) else {}
    if described.get("kind") == "archive":
        if computed_from != manifest_files:
            errors.append("result.json was not computed from the files of this archive entry")
    elif (computed_from.get(archive.SOURCE_TRAJECTORY_FILE) or {}).get("sha256") != (manifest.get("source") or {}).get("sha256"):
        errors.append("result.json was not computed from the source of this archive entry")
    produced = ((result.get("produced") or {}).get("static") or {}).get("files") if isinstance(result.get("produced"), dict) else None
    if not isinstance(produced, list):
        errors.append("result.json does not list the static files it produced")
    else:
        recorded_static = {f"static/{item.get('path')}": (item.get("size"), item.get("sha256")) for item in produced
                           if isinstance(item, dict)}
        listed_static = {relative: (item["size"], item["sha256"].lower()) for relative, item in files.items()
                         if relative.startswith("static/")}
        if recorded_static != listed_static:
            odd = sorted(set(recorded_static.items()) ^ set(listed_static.items()))[:3]
            errors.append(f"the static files are not the ones result.json produced (for example {[name for name, _ in odd]})")
    if errors:
        raise BundleError("; ".join(errors))
    return Bundle(
        path=root,
        document=document,
        sha256=hashlib.sha256(payload).hexdigest(),
        glycam_name=glycam_name,
        levels=levels,
        files={relative: {"size": item["size"], "sha256": item["sha256"].lower()} for relative, item in files.items()},
        manifest=manifest,
        manifest_sha256=_sha256(root / "archive" / archive.MANIFEST_FILE)[1],
        result=result,
        hashes_verified=bool(verify_hashes),
    )


def submission_from_bundle(
    bundle: Bundle,
    glycoshape_id: str,
    curated: Optional[Dict[str, Any]] = None,
    source_type: str = "",
    published_length_ns: Optional[float] = None,
    **simulation: Any,
) -> Dict[str, Any]:
    """A submission record for the entry of a bundle, with the run metadata the bundle carries.

    The simulation fields are those of ``source`` of ``bundle.json``
    (:data:`BUNDLE_SOURCE_KEYS`), replaced by ``simulation`` where the
    registry knows better.  ``source`` states the length of the run the
    bundle was made of: for a bundle that extends an entry, pass the length
    that is published for it as ``published_length_ns`` and the two are
    added (or give ``simulation_length_ns`` yourself); without either the
    record is refused, because the sum cannot be known here.
    """
    source = bundle.source
    fields = {key: source[key] for key in BUNDLE_SOURCE_KEYS if source.get(key) not in (None, "")}
    if bundle.extends is not None and "simulation_length_ns" not in simulation and "simulation_length_us" not in simulation:
        if published_length_ns is None:
            raise SubmissionError(
                "the bundle extends a published entry and states the length of the new run only: give "
                "published_length_ns (they are added) or simulation_length_ns (the aggregated length)")
        if "simulation_length_ns" in fields:
            fields["simulation_length_ns"] = float(published_length_ns) + float(fields["simulation_length_ns"])
    if source_type:
        fields["source_type"] = source_type
    fields.update(simulation)
    if "simulation_length_us" in simulation:
        fields.pop("simulation_length_ns", None)
    return {
        "schema": SUBMISSION_SCHEMA,
        "schema_version": SUBMISSION_SCHEMA_VERSION,
        "glycoshape_id": glycoshape_id,
        "glycam_name": bundle.glycam_name,
        "simulation": fields,
        "glytoucan_id": source.get("glytoucan_id") or "",
        "curated": dict(curated or {}),
    }


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------
IDENTITY_FIELDS = ("iupac", "iupac_extended", "glytoucan", "wurcs")


def identity_gaps(data: Dict[str, Any]) -> List[str]:
    """The identity fields of ``data.json`` that are empty, as ``variant.field``.

    These are what a lookup would have filled: names and accessions of the
    three variants, and for the archetype also the fields that come from
    PubChem and from the motif service.
    """
    gaps = []
    for variant in ("archetype", "alpha", "beta"):
        entry = data.get(variant) if isinstance(data.get(variant), dict) else {}
        fields = IDENTITY_FIELDS + (
            ("glycoct", "smiles", "mass", "motifs", "rot_bonds", "hbond_donor", "hbond_acceptor")
            if variant == "archetype" else ())
        gaps.extend(f"{variant}.{name}" for name in fields if entry.get(name) in (None, ""))
    return gaps


def _folder_files(folder: Path) -> Dict[str, Dict[str, Any]]:
    files = {}
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        size, digest = _sha256(path)
        files[path.relative_to(folder).as_posix()] = {"size": size, "sha256": digest}
    return files


def _extra_aliases(processor: Any, data: Dict[str, Any], aliases: Iterable[str]) -> List[str]:
    """Curated aliases that the names of the entry do not already give."""
    common_names = (data.get("search_meta") or {}).get("common_names") or []
    known = {name.casefold() for name in processor._build_aliases(data, list(common_names))}
    return [alias for alias in aliases if alias.casefold() not in known]


def complete_entry(
    bundle_dir,
    submission: Any,
    output_dir,
    offline: Optional[bool] = None,
    lookup_timeout: Optional[float] = None,
    lookup_budget: Optional[float] = 300.0,
    verify_hashes: bool = True,
    bundle: Optional[Bundle] = None,
) -> Dict[str, Any]:
    """Write ``<output_dir>/<GlycoShape ID>/``, the published folder of the entry of a bundle.

    ``submission`` is the record described in the module documentation (or a
    :class:`Submission`).  The folder holds what the static stage writes:
    ``data.json`` with the search metadata, ``snfg.svg``, ``glygen.json``,
    ``glycosmos.json``, ``output/``, ``landscape/`` with the REMARK block on
    the PDB files, and ``GSxxxxx.zip``.  An existing folder of that name in
    ``output_dir`` is replaced; nothing else in ``output_dir`` is read or
    written, and the bundle is only read.

    ``offline`` (default: ``GLYCOSHAPE_OFFLINE``) makes no web lookup;
    ``lookup_timeout`` caps every request and ``lookup_budget`` the time of
    all of them.  The report says what was looked up and what was not
    answered (``lookups``), which identity fields are empty (``gaps``), what
    stands in the way of publishing the folder (``problems``: a missing
    required file, neither WURCS nor accession, no ``snfg.svg``) and what
    only looks wrong (``warnings``).  Raises :class:`SubmissionError`,
    :class:`BundleError` or :class:`CompletionError` when no folder can be
    written.  One completion at a time per process.
    """
    import lib.config as config
    from lib import glystatic, lookups, replicas, static_levels
    from lib.glybake import GlycoShapeBaker
    from lib.glymeta import GlycanMetadataProcessor
    from lib.storage import local_storage

    submission = Submission.parse(submission)
    if bundle is None:
        bundle = load_bundle(bundle_dir, verify_hashes=verify_hashes)
    if bundle.glycam_name != submission.glycam_name:
        raise SubmissionError(
            f"the submission is for {submission.glycam_name[:80]!r} and the bundle holds {bundle.glycam_name[:80]!r}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings = list(submission.warnings)
    problems: List[str] = []

    # What the entry says about itself, against the submission.
    stated = _stated_length_ns(bundle)
    if stated is not None and submission.length_us is not None \
            and not math.isclose(stated / 1000.0, submission.length_us, rel_tol=1e-6):
        warnings.append(f"the entry says {stated:g} ns of simulation and the submission {submission.length_us * 1000:g} ns")
    simulation_record = submission.record()["simulation"]
    for key in BUNDLE_SOURCE_KEYS:
        value, given = bundle.source.get(key), simulation_record.get(key)
        if value in (None, "") or given in (None, "") or value == given:
            continue
        if key == "simulation_length_ns" and bundle.extends is not None:
            if given < value:
                warnings.append(f"the submission says {given:g} ns for an entry whose last run alone has {value:g} ns")
            continue                                # the aggregated length is more than that of the new run
        if isinstance(value, (int, float)) and isinstance(given, (int, float)) and math.isclose(value, given, rel_tol=1e-9):
            continue
        warnings.append(f"bundle.json source.{key} is {value!r} and the submission says {given!r}")
    if bundle.extends is not None and bundle.source.get("simulation_length_ns") not in (None, "") \
            and simulation_record["simulation_length_ns"] == bundle.source["simulation_length_ns"]:
        warnings.append("the bundle extends a published entry and the submission gives the length of the new run "
                        "only: the published length has to be added")

    with _PROCESS_LOCK:
        previous_levels = config.cluster_levels
        config.cluster_levels = tuple(bundle.levels)
        try:
            with local_storage(), lookups.recording(offline=offline, timeout=lookup_timeout, budget=lookup_budget) as log:
                completed = glystatic.complete_static_ready(
                    bundle.static_dir,
                    bundle.glycam_name,
                    submission.glycoshape_id,
                    output_dir,
                    submission.simulation(),
                    inventory_glytoucan=submission.glytoucan_id or None,
                )
                folder = Path(completed["final_output_dir"])
                processor = GlycanMetadataProcessor(output_dir, pb_client=None)
                if not processor.update_glycan_metadata(folder / "data.json", enrichment=submission.enrichment()):
                    raise CompletionError(f"the search metadata of {folder / 'data.json'} could not be written")
                data = json.loads((folder / "data.json").read_bytes())
                extra = _extra_aliases(processor, data, submission.aliases)
                if extra:
                    # Names the entry cannot be found under otherwise.  The key is only written for an
                    # entry that has such aliases, so every other data.json is the one of the static stage.
                    data["search_meta"]["aliases"] = extra
                    (folder / "data.json").write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
                if not GlycoShapeBaker(output_dir).create_individual_glycan_archive(folder):
                    raise CompletionError(f"{folder.name}.zip could not be built")
        except (SubmissionError, BundleError, CompletionError):
            raise
        except Exception as exc:
            raise CompletionError(f"{type(exc).__name__}: {exc}") from exc
        finally:
            config.cluster_levels = previous_levels

    files = _folder_files(folder)
    for name in static_levels.required_files(bundle.levels):
        if name not in files:
            problems.append(f"{name} is missing")
    for name in ("snfg.svg", "glygen.json", "glycosmos.json", f"{folder.name}.zip"):
        if name not in files:
            problems.append(f"{name} is missing" + (" (the SNFG renderer did not run)" if name == "snfg.svg" else ""))
    archetype = data.get("archetype", {})
    if not archetype.get("iupac"):
        problems.append("the archetype has no IUPAC name")
    if not archetype.get("wurcs") and not archetype.get("glytoucan"):
        problems.append("the archetype has neither a WURCS nor a GlyTouCan accession "
                        "(the static stage of main.py calls such a folder incomplete)")
    used = {f"static/{name}" for name in files}
    if not replicas.public_placement():
        used.add(f"static/{replicas.STATIC_PATH}")      # kept by the publisher for the administrators
    ignored = sorted(relative[len("static/"):] for relative in bundle.files
                     if relative.startswith("static/") and relative not in used)
    if ignored:
        warnings.append(f"{len(ignored)} file(s) of the static-ready folder are not part of a published folder: "
                        + ", ".join(ignored[:5]))
    summary = log.summary()
    if summary["unanswered"]:
        warnings.append("lookups without an answer: " + ", ".join(summary["unanswered"]))
    if summary["tools"].get("crabwurcs", {}).get("available") is False:
        warnings.append("crabWURCS could not be run: " + summary["tools"]["crabwurcs"]["detail"])
    return {
        "glycoshape_id": submission.glycoshape_id,
        "glycam_name": submission.glycam_name,
        "folder": str(folder),
        "levels": list(bundle.levels),
        "files": files,
        "file_count": len(files),
        "total_size": sum(item["size"] for item in files.values()),
        "identity": {
            variant: {name: (data.get(variant) or {}).get(name) for name in ("glycam",) + IDENTITY_FIELDS}
            for variant in ("archetype", "alpha", "beta")
        },
        "simulation": {name: archetype.get(name) for name in glystatic.SIMULATION_FIELDS},
        "search_meta": data.get("search_meta"),
        "gaps": identity_gaps(data),
        "lookups": summary,
        "problems": problems,
        "warnings": warnings,
        "submission": submission.record(),
        "submission_sha256": submission.sha256(),
        "bundle": bundle.summary(),
    }


def _stated_length_ns(bundle: Bundle) -> Optional[float]:
    """The simulation length in nanoseconds that ``info.json`` of the archive entry states, if it does."""
    path = bundle.archive_dir / "info.json"
    if not path.is_file():
        return None
    try:
        info = json.loads(path.read_bytes())
    except ValueError:
        return None
    if not isinstance(info, dict):
        return None
    for key in ("simulation_length_ns", "simulationLength"):
        value = info.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
            return float(value)
    return None
