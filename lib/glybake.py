import lib.config as config
from pathlib import Path
import os
import shutil
import logging
import hashlib
from datetime import datetime, timezone
import zipfile
import glob
import json
import requests
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union, Optional
from lib import name_utils
from lib.storage import get_storage_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The root files as functions of their content.
#
# The baker below writes them for a whole static directory; lib.static_delta
# derives them for one added, replaced or withdrawn glycan from the stored
# ones.  Both use the functions of this section, so the two results are the
# same bytes for the same database.
# ---------------------------------------------------------------------------
#: Time stamp of every member of an archive: the earliest one a zip file can hold.
ARCHIVE_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
#: Unix mode of every member (a regular file, rw-r--r--) and the deflate level.
ARCHIVE_FILE_MODE = 0o100644
ARCHIVE_COMPRESSLEVEL = 6
MISSING_WURCS_HEADER = "glycoshape_id\tvariant\tiupac\tglycam\n"


def write_archive(path: Union[str, Path], members: Iterable[Tuple[str, Any]]) -> None:
    """Write a zip archive that is a function of the names and contents of its members.

    ``members`` are ``(name, content)``; the content is the bytes, a path to
    read them from, or a function without arguments that returns them (read
    one member at a time, when it is written).

    Members are stored in the order of their names, each with the time stamp
    :data:`ARCHIVE_TIMESTAMP`, the mode :data:`ARCHIVE_FILE_MODE` and deflate
    level :data:`ARCHIVE_COMPRESSLEVEL`: neither the times and permissions of
    the files nor the order in which a directory lists them reach the
    archive.  Building it again from the same members gives the same bytes
    (with the same zlib, whose deflate decides the compressed bytes).  The
    archive appears under its name when it is complete.
    """
    path = Path(path)
    items = sorted(members, key=lambda item: item[0])
    names = [name for name, _ in items]
    if len(set(names)) != len(names):
        raise ValueError(f"an archive member is named twice in {path.name}")
    temporary = path.with_name(path.name + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=ARCHIVE_COMPRESSLEVEL) as archive:
            for name, content in items:
                info = zipfile.ZipInfo(name, date_time=ARCHIVE_TIMESTAMP)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3                      # Unix, whatever the platform that writes it
                info.external_attr = ARCHIVE_FILE_MODE << 16
                if callable(content):
                    content = content()
                elif isinstance(content, (str, Path)):
                    content = Path(content).read_bytes()
                archive.writestr(info, bytes(content), compresslevel=ARCHIVE_COMPRESSLEVEL)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def archive_holds(path: Union[str, Path], members: Dict[str, bytes]) -> bool:
    """Whether the zip archive at ``path`` holds exactly these members with these contents."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(members) or set(names) != set(members):
                return False
            return all(archive.read(name) == content for name, content in members.items())
    except (OSError, zipfile.BadZipFile, KeyError, RuntimeError, ValueError):
        return False


def inventory_bytes(inventory: Dict[str, Any]) -> bytes:
    """``GLYCOSHAPE.json`` for a mapping of GlycoShape ID to the content of its ``data.json``."""
    return json.dumps(dict(sorted(inventory.items())), indent=2, ensure_ascii=False).encode("utf-8")


def missing_reports(
    inventory: Dict[str, Any],
    calculate: Optional[Callable[[str, str, str], Tuple[Optional[str], Optional[str]]]] = None,
    resolve: Optional[Callable[[List[str]], Dict[str, Optional[str]]]] = None,
    log: Optional[logging.Logger] = None,
) -> Tuple[List[str], List[Dict[str, str]]]:
    """What ``missing_glytoucan.txt`` and ``missing_wurcs.txt`` list for an inventory.

    Returns the WURCS strings without a GlyTouCan accession, each once and in
    the order of the inventory, and one row per variant that has neither a
    WURCS nor an accession.  Two steps need services and are optional:
    ``calculate(glycan_id, variant, iupac)`` gives ``(accession, WURCS)`` for
    a variant that has neither, and ``resolve(wurcs_list)`` maps the WURCS
    that would be listed to the accession GlyTouCan has for them by now.
    Without the two, the result is a function of the inventory alone.
    """
    missing_entries: List[str] = []
    unresolved_wurcs_entries: List[Dict[str, str]] = []

    # Check both archetype and alpha/beta entries
    for glycan_id, glycan_data in inventory.items():
        entries_to_check = ['archetype']

        # Add alpha/beta if they exist
        if 'alpha' in glycan_data:
            entries_to_check.append('alpha')
        if 'beta' in glycan_data:
            entries_to_check.append('beta')

        for entry_type in entries_to_check:
            entry = glycan_data.get(entry_type, {})
            wurcs = entry.get("wurcs")
            glytoucan = entry.get("glytoucan")
            iupac = entry.get("iupac")

            if not wurcs and not glytoucan and iupac and calculate is not None:
                try:
                    calculated_glytoucan, calculated_wurcs = calculate(glycan_id, entry_type, iupac)
                except Exception as exc:
                    calculated_glytoucan, calculated_wurcs = None, None
                    if log is not None:
                        log.warning(
                            f"Failed to calculate WURCS for {glycan_id} ({entry_type}) from IUPAC {iupac}: {exc}"
                        )
                if calculated_glytoucan:
                    glytoucan = calculated_glytoucan
                if calculated_wurcs:
                    wurcs = calculated_wurcs

            if wurcs and not glytoucan:
                if wurcs not in missing_entries:
                    missing_entries.append(wurcs)
            elif not wurcs and not glytoucan:
                unresolved_wurcs_entries.append({
                    "glycoshape_id": glycan_id,
                    "variant": entry_type,
                    "iupac": entry.get("iupac") or "",
                    "glycam": entry.get("glycam") or entry.get("name") or "",
                })

    if resolve is not None:
        # Stored static JSON may predate an accession, and generated WURCS
        # can be structurally valid but non-canonical. Recheck canonical
        # WURCS in batches before calling anything a registration candidate.
        lookup_candidates = [
            wurcs for wurcs in missing_entries
            if str(wurcs).startswith("WURCS=2.0/")
        ]
        resolved_by_wurcs = resolve(lookup_candidates)
        missing_entries = [
            wurcs for wurcs in missing_entries
            if not resolved_by_wurcs.get(wurcs)
        ]
    return missing_entries, unresolved_wurcs_entries


class WurcsCalculator:
    """``calculate`` of :func:`missing_reports`: a WURCS for a variant that has none, from its IUPAC name.

    crabWURCS and the format service first, then SMILES and MolWURCS.  The
    variants of a glycan are derived together, so what the archetype gave is
    used for its anomers.
    """

    def __init__(self, log: Optional[logging.Logger] = None) -> None:
        self.logger = log or logger
        self.calculated_wurcs_by_glycan: Dict[str, Dict[str, str]] = {}

    def calculate_wurcs_from_iupac(self, glycan_id: str, entry_type: str, iupac: str):
        calculated_glytoucan, calculated_wurcs = name_utils.iupac2wurcs_glytoucan(iupac)
        if calculated_wurcs:
            return calculated_glytoucan, calculated_wurcs

        try:
            smiles = name_utils.iupac2smiles(iupac)
        except Exception as exc:
            self.logger.warning(
                f"Failed to calculate SMILES for {glycan_id} ({entry_type}) from IUPAC {iupac}: {exc}"
            )
            return calculated_glytoucan, None

        if not smiles:
            return calculated_glytoucan, None

        calculated_wurcs = name_utils.smiles2wurcs(smiles)
        if calculated_wurcs:
            variants = name_utils.get_wurcs_variants(calculated_wurcs)
            if variants:
                self.calculated_wurcs_by_glycan[glycan_id] = {
                    "archetype": variants[0],
                    "alpha": variants[1],
                    "beta": variants[2],
                }
                calculated_wurcs = self.calculated_wurcs_by_glycan[glycan_id].get(entry_type) or calculated_wurcs
            calculated_glytoucan = calculated_glytoucan or name_utils.wurcs2glytoucan(calculated_wurcs)

        return calculated_glytoucan, calculated_wurcs

    def __call__(self, glycan_id: str, entry_type: str, iupac: str):
        calculated_wurcs = self.calculated_wurcs_by_glycan.get(glycan_id, {}).get(entry_type)
        if calculated_wurcs:
            return None, calculated_wurcs
        return self.calculate_wurcs_from_iupac(glycan_id, entry_type, iupac)


def resolve_missing_accessions(candidates: List[str]) -> Dict[str, Optional[str]]:
    """``resolve`` of :func:`missing_reports`: the accessions GlyTouCan has by now (one batch request)."""
    return name_utils.wurcs2glytoucan_many(candidates)


def missing_glytoucan_text(missing_entries: List[str]) -> str:
    """``missing_glytoucan.txt``: one WURCS per line."""
    return "\n".join(missing_entries) + ("\n" if missing_entries else "")


def missing_wurcs_text(unresolved_wurcs_entries: List[Dict[str, str]]) -> str:
    """``missing_wurcs.txt``: a header and one tab-separated row per unresolved variant."""
    return MISSING_WURCS_HEADER + "".join(
        "\t".join([row["glycoshape_id"], row["variant"], row["iupac"], row["glycam"]]) + "\n"
        for row in unresolved_wurcs_entries
    )


def utc_timestamp(seconds: Optional[float] = None) -> str:
    """A time of the manifest (``generated_at``, ``mtime``): UTC, ISO 8601, ending in ``Z``; now by default."""
    moment = datetime.now(timezone.utc) if seconds is None else datetime.fromtimestamp(seconds, timezone.utc)
    return moment.isoformat().replace("+00:00", "Z")


def aggregate_hash(entries: Iterable[str]) -> str:
    """Hash sorted manifest entry lines into one deterministic digest."""
    digest = hashlib.sha256()
    for entry in sorted(entries):
        digest.update(entry.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def manifest_glycan_entry(glycan_id: str, files: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """The ``glycans[<id>]`` entry of the sync manifest for the files of one glycan folder.

    ``files`` maps the path inside the folder to ``{"path", "sha256", "size",
    "mtime"}`` (``path`` from the database root).  ``hash`` is the key the
    website compares to decide whether the folder changed: it covers the
    name, SHA-256 and size of every file, and nothing else.
    """
    return {
        "hash": aggregate_hash(f"{name}\0{info['sha256']}\0{info['size']}" for name, info in files.items()),
        "file_count": len(files),
        "total_size": sum(info["size"] for info in files.values()),
        "data_json_sha256": files.get("data.json", {}).get("sha256"),
        "zip_sha256": files.get(f"{glycan_id}.zip", {}).get("sha256"),
        "files": files,
    }


def manifest_global_hash(root_files: Dict[str, Dict[str, Any]], glycans: Dict[str, Dict[str, Any]]) -> str:
    """``global_hash``: one digest over every root file and every glycan entry.

    The website skips a synchronisation when it already has this value, so
    it has to change whenever a file of the database changes, and only then:
    it covers names, SHA-256 and sizes, not times.
    """
    return aggregate_hash([
        f"root/{name}\0{info['sha256']}\0{info['size']}"
        for name, info in root_files.items()
    ] + [
        f"glycan/{glycan_id}\0{info['hash']}\0{info['file_count']}\0{info['total_size']}"
        for glycan_id, info in glycans.items()
    ])


def build_manifest(
    root_files: Dict[str, Dict[str, Any]],
    glycans: Dict[str, Dict[str, Any]],
    output_name: str = "sync_manifest.json",
    generated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """The sync manifest for these root files and glycan entries."""
    return {
        "schema_version": 1,
        "generated_at": generated_at or utc_timestamp(),
        "hash_algorithm": "sha256",
        "static_prefix": "static",
        "manifest_path": output_name,
        "manifest_excludes": [output_name],
        "global_hash": manifest_global_hash(root_files, glycans),
        "glycan_count": len(glycans),
        "root_files": root_files,
        "glycans": glycans,
    }


def manifest_bytes(manifest: Dict[str, Any]) -> bytes:
    """The bytes of ``sync_manifest.json``."""
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


def glycan_archive_members(glycan_dir: Path, output_levels: Iterable[str] = ("level_1",)) -> List[Tuple[str, Path]]:
    """Member name and file of everything ``GSxxxxx.zip`` holds for a glycan folder.

    ``data.json``, ``snfg.svg``, ``glycosmos.json`` and ``glygen.json``, the
    format folders of older databases, and the files of the representative
    output levels under their path inside the level folder.
    """
    glycan_dir = Path(glycan_dir)
    members: List[Tuple[str, Path]] = []
    for file_name in ("data.json", "snfg.svg", "glycosmos.json", "glygen.json"):
        if (glycan_dir / file_name).exists():
            members.append((file_name, glycan_dir / file_name))
    for format_dir in (
        "PDB_format_HETATM", "CHARMM_format_HETATM", "GLYCAM_format_HETATM",
        "PDB_format_ATOM", "CHARMM_format_ATOM", "GLYCAM_format_ATOM",
    ):
        format_path = glycan_dir / format_dir
        if format_path.exists() and format_path.is_dir():
            for file in format_path.rglob('*'):
                if file.is_file():
                    members.append((format_dir + '/' + str(file.relative_to(format_path)), file))
    output_dir = glycan_dir / "output"
    for level_name in output_levels:
        level_dir = output_dir / level_name
        if output_dir.exists() and level_dir.is_dir():
            for file in sorted(level_dir.rglob('*')):
                if file.is_file():
                    members.append((str(file.relative_to(level_dir)), file))
    return members


class BakeError(RuntimeError):
    """A root file was not written because some glycan folders could not be read or archived.

    ``failures`` lists ``(glycan ID, reason)`` of every one of them, so that a
    partial database is never published as if it were complete.
    """

    def __init__(self, what: str, failures: List[Tuple[str, str]]):
        self.failures = list(failures)
        shown = "; ".join(f"{glycan_id}: {reason}" for glycan_id, reason in self.failures[:5])
        more = f" (and {len(self.failures) - 5} more)" if len(self.failures) > 5 else ""
        super().__init__(f"{what} not written: {len(self.failures)} glycan(s) failed: {shown}{more}")


class GlycoShapeBaker:
    """Main class for creating GlycoShape database archives and files."""
    ARCHIVE_OUTPUT_LEVELS = ("level_1",)

    def __init__(self, database_dir: Union[str, Path]):
        """Initialize the baker with database directory.

        Args:
            database_dir: Path to the database directory containing glycan folders
        """
        self.database_dir = Path(database_dir)
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _file_hash_info(path: Path) -> Dict[str, Any]:
        """Return stable hash metadata for a file."""
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)

        stat = path.stat()
        return {
            "sha256": digest.hexdigest(),
            "size": size,
            "mtime": utc_timestamp(stat.st_mtime),
        }

    @staticmethod
    def _aggregate_hash(entries: List[str]) -> str:
        """Hash sorted manifest entry lines into one deterministic digest."""
        return aggregate_hash(entries)

    @classmethod
    def glycan_folder_files(cls, glycan_dir: Path, root: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
        """The ``files`` of a glycan folder for the sync manifest: path, SHA-256, size and time of each file."""
        glycan_dir = Path(glycan_dir)
        root = glycan_dir.parent if root is None else Path(root)
        files: Dict[str, Dict[str, Any]] = {}
        for file_path in sorted(p for p in glycan_dir.rglob("*") if p.is_file()):
            info = cls._file_hash_info(file_path)
            files[file_path.relative_to(glycan_dir).as_posix()] = {
                "path": file_path.relative_to(root).as_posix(),
                "sha256": info["sha256"],
                "size": info["size"],
                "mtime": info["mtime"],
            }
        return files

    def create_sync_manifest(self, output_name: str = "sync_manifest.json") -> Path:
        """Create a static sync manifest for fast Oracle-to-host updates.

        The manifest is intentionally self-contained:
        - root_files covers top-level static artifacts such as GLYCOSHAPE.json
        - glycans[GSxxxxx].hash is the quick folder-level update key
        - glycans[GSxxxxx].files is the exact file list for download/verification
        """
        root = self.database_dir
        manifest_path = root / output_name

        root_files: Dict[str, Dict[str, Any]] = {}
        for file_path in sorted(root.iterdir()):
            if not file_path.is_file() or file_path.name == output_name:
                continue
            info = self._file_hash_info(file_path)
            root_files[file_path.name] = {
                "path": file_path.name,
                "sha256": info["sha256"],
                "size": info["size"],
                "mtime": info["mtime"],
            }

        glycans: Dict[str, Dict[str, Any]] = {}
        for glycan_dir in sorted(root.glob("GS*")):
            if not glycan_dir.is_dir():
                continue
            glycans[glycan_dir.name] = manifest_glycan_entry(
                glycan_dir.name, self.glycan_folder_files(glycan_dir, root))

        manifest = build_manifest(root_files, glycans, output_name)

        tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        tmp_path.write_bytes(manifest_bytes(manifest))
        tmp_path.replace(manifest_path)

        self.logger.info(
            "Created sync manifest %s with %d glycans and %d root files",
            manifest_path,
            len(glycans),
            len(root_files),
        )
        return manifest_path

    def create_individual_glycan_archive(self, glycan_dir: Path) -> bool:
        """Create archive for individual glycan with new structure.

        The archive is a function of the files it holds (:func:`write_archive`):
        building it again gives the same bytes.  An archive that is already
        there with exactly these members and contents is kept as it is, so
        the archives of a database that was baked before stay what they are.

        Args:
            glycan_dir: Path to glycan directory (e.g., GS00445/)

        Returns:
            bool: True if successful, False otherwise
        """
        if not glycan_dir.is_dir():
            self.logger.warning(f"Directory {glycan_dir} does not exist")
            return False

        glycan_id = glycan_dir.name
        archive_path = glycan_dir / f"{glycan_id}.zip"

        try:
            members = {
                name: path.read_bytes()
                for name, path in glycan_archive_members(glycan_dir, self.ARCHIVE_OUTPUT_LEVELS)
            }
            output_dir = glycan_dir / "output"
            if not any((output_dir / level_name).is_dir() for level_name in self.ARCHIVE_OUTPUT_LEVELS):
                expected_levels = ", ".join(self.ARCHIVE_OUTPUT_LEVELS)
                self.logger.warning(f"No archive output levels ({expected_levels}) found in {glycan_dir}")

            if archive_path.exists():
                if archive_holds(archive_path, members):
                    self.logger.info(f"Archive for {glycan_id} is up to date")
                    return True
                # Remove existing archive
                archive_path.unlink()
                self.logger.info(f"Removed existing {glycan_id}.zip")

            write_archive(archive_path, members.items())
            self.logger.info(f"Successfully created archive for {glycan_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create archive for {glycan_id}: {str(e)}")
            return False

    def create_master_archive(self, output_file: str = "GlycoShape.zip") -> None:
        """Create master GlycoShape archive from all individual glycan archives.

        Its members are the ``GSxxxxx.zip`` of every glycan, in the order of
        their names (:func:`write_archive`).  When the archive of a glycan
        cannot be built, no master archive is written (an old one is removed
        first) and :class:`BakeError` lists the glycans.

        Args:
            output_file: Name of the master archive file
        """
        try:
            output_path = self.database_dir / output_file

            # Remove existing master archive
            if output_path.exists():
                output_path.unlink()
                self.logger.info(f"Removed existing {output_file}")

            # Create individual archives first
            self.logger.info("Creating individual glycan archives...")
            failed: List[Tuple[str, str]] = []
            glycan_dirs = sorted(path for path in self.database_dir.glob("GS*") if path.is_dir())

            for glycan_dir in glycan_dirs:
                if not self.create_individual_glycan_archive(glycan_dir):
                    failed.append((glycan_dir.name, "its archive could not be built"))
                elif not (glycan_dir / f"{glycan_dir.name}.zip").is_file():
                    failed.append((glycan_dir.name, "its archive is missing"))

            self.logger.info(f"Created {len(glycan_dirs) - len(failed)}/{len(glycan_dirs)} individual archives")
            if failed:
                raise BakeError(output_file, failed)

            # Create master archive
            self.logger.info("Creating master archive...")
            write_archive(output_path, (
                (f"{glycan_dir.name}.zip", glycan_dir / f"{glycan_dir.name}.zip") for glycan_dir in glycan_dirs
            ))

            self.logger.info(f"Successfully created master archive: {output_file}")

        except Exception as e:
            self.logger.error(f"Failed to create master archive: {str(e)}")
            raise

    def create_consolidated_json(self, output_file: str = "GLYCOSHAPE.json") -> None:
        """Create consolidated JSON file from all data.json files.

        Every ``GSxxxxx`` folder has to have a readable ``data.json`` that
        holds a JSON object.  When one does not, nothing is written and
        :class:`BakeError` lists every such glycan: an inventory without a
        glycan whose folder is published would be a partial database.

        Args:
            output_file: Name of output JSON file
        """
        try:
            output_path = self.database_dir / output_file
            consolidated_data = {}
            failed: List[Tuple[str, str]] = []
            storage = get_storage_manager()

            # Process each glycan directory
            for glycan_dir in sorted(self.database_dir.glob("GS*")):
                if not glycan_dir.is_dir():
                    continue

                glycan_id = glycan_dir.name
                json_path = glycan_dir / "data.json"

                if not json_path.exists():
                    self.logger.error(f"No data.json in {glycan_dir}")
                    failed.append((glycan_id, "no data.json"))
                    continue
                self.logger.info(f"Processing JSON for {glycan_id}")
                try:
                    with storage.open(str(json_path), 'r') as f:
                        data = json.load(f)
                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON in {json_path}: {str(e)}")
                    failed.append((glycan_id, f"invalid JSON in data.json: {e}"))
                    continue
                except Exception as e:
                    self.logger.error(f"Error reading {json_path}: {str(e)}")
                    failed.append((glycan_id, f"data.json could not be read: {e}"))
                    continue
                if not isinstance(data, dict):
                    self.logger.error(f"{json_path} does not hold a JSON object")
                    failed.append((glycan_id, "data.json does not hold a JSON object"))
                    continue
                consolidated_data[glycan_id] = data

            if failed:
                raise BakeError(output_file, failed)

            # Write consolidated JSON
            self.logger.info(f"Writing consolidated JSON to {output_file}")
            with storage.open(output_path, 'w') as f:
                f.write(inventory_bytes(consolidated_data).decode("utf-8"))

            self.logger.info(f"Successfully created consolidated JSON with {len(consolidated_data)} entries")

        except Exception as e:
            self.logger.error(f"Failed to create consolidated JSON: {str(e)}")
            raise

    def extract_missing_glytoucan_ids(
        self,
        json_file: str = "GLYCOSHAPE.json",
        output_file: str = "missing_glytoucan.txt"
    ) -> None:
        """Extract WURCS entries that need GlyTouCan registration.

        Args:
            json_file: Input JSON file to analyze
            output_file: Output file for missing GlyTouCan entries
        """
        try:
            json_path = self.database_dir / json_file

            storage = get_storage_manager()
            json_path_str = str(json_path)
            with storage.open(json_path_str, 'r') as f:
                data = json.load(f)

            missing_entries, unresolved_wurcs_entries = missing_reports(
                data,
                calculate=WurcsCalculator(self.logger),
                resolve=resolve_missing_accessions,
                log=self.logger,
            )

            # Write missing entries to file
            output_path = f"{self.database_dir}/{output_file}"
            storage = get_storage_manager()
            with storage.open(output_path, 'w') as f:
                f.write(missing_glytoucan_text(missing_entries))

            unresolved_output_path = f"{self.database_dir}/missing_wurcs.txt"
            with storage.open(unresolved_output_path, 'w') as f:
                f.write(missing_wurcs_text(unresolved_wurcs_entries))

            self.logger.info(f"Found {len(missing_entries)} WURCS entries missing GlyTouCan IDs")
            if unresolved_wurcs_entries:
                self.logger.warning(
                    f"Found {len(unresolved_wurcs_entries)} entries missing both WURCS and GlyTouCan; "
                    f"wrote {unresolved_output_path}"
                )

        except Exception as e:
            self.logger.error(f"Failed to extract missing GlyTouCan IDs: {str(e)}")
            raise

    def submit_wurcs_to_glytoucan(
        self, 
        contributor_id: str, 
        api_key: str, 
        file_path: Union[str, Path]
    ) -> None:
        """Submit WURCS data to GlyTouCan API for registration.
        
        Args:
            contributor_id: GlyTouCan contributor ID
            api_key: GlyTouCan API key
            file_path: Path to file containing WURCS data
        """
        url = "https://glytoucan.org/api/bulkload/wurcs"
        file_path = Path(file_path)
        
        try:
            storage = get_storage_manager()
            with storage.open(file_path, 'rb') as file:
                response = requests.post(
                    url,
                    files={"file": file},
                    auth=(contributor_id, api_key),
                    timeout=300  # 5 minute timeout
                )
            
            if response.status_code == 200:
                self.logger.info("Successfully submitted WURCS data to GlyTouCan")
                self.logger.info(f"Response: {response.text}")
            else:
                self.logger.error(f"Failed to submit WURCS data: {response.status_code}")
                self.logger.error(f"Response: {response.text}")
                response.raise_for_status()
        
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error during WURCS submission: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error during WURCS submission: {str(e)}")
            raise

    def calculate_total_simulation_length(self, json_file: str = "GLYCOSHAPE.json") -> float:
        """Calculate total simulation length from all entries.
        
        Args:
            json_file: JSON file to analyze
            
        Returns:
            Total simulation length in microseconds
        """
        try:
            json_path = self.database_dir / json_file
            
            storage = get_storage_manager()
            json_path_str = str(json_path)
            with storage.open(json_path_str, 'r') as f:
                data = json.load(f)
            
            total_length = 0.0
            processed_entries = 0
            
            for glycan_id, glycan_data in data.items():
                archetype = glycan_data.get("archetype", {})
                length = archetype.get("length")
                
                if length is not None:
                    try:
                        total_length += float(length)
                        processed_entries += 1
                    except (ValueError, TypeError):
                        self.logger.warning(f"Invalid length value for {glycan_id}: {length}")
            
            self.logger.info(f"Total simulation length: {total_length} μs from {processed_entries} entries")
            return total_length
        
        except Exception as e:
            self.logger.error(f"Failed to calculate total simulation length: {str(e)}")
            raise

    def validate_database_structure(self) -> Dict[str, List[str]]:
        """Validate the database structure and report any issues.
        
        Returns:
            Dictionary with validation results
        """
        issues = {
            "missing_data_json": [],
            "missing_snfg_svg": [],
            "missing_output_dir": [],
            "missing_format_dirs": [],
            "invalid_json": []
        }
        
        required_files = ["data.json", "snfg.svg"]
        required_format_dirs = [
            "PDB_format_HETATM", "CHARMM_format_HETATM", "GLYCAM_format_HETATM",
            "PDB_format_ATOM", "CHARMM_format_ATOM", "GLYCAM_format_ATOM"
        ]
        
        for glycan_dir in self.database_dir.glob("GS*"):
            if not glycan_dir.is_dir():
                continue
                
            glycan_id = glycan_dir.name
            
            # Check required files
            for file_name in required_files:
                if not (glycan_dir / file_name).exists():
                    issues[f"missing_{file_name.replace('.', '_')}"].append(glycan_id)
            
            # Check data.json validity
            data_json = glycan_dir / "data.json"
            if data_json.exists():
                try:
                    storage = get_storage_manager()
                    data_json_str = str(data_json)
                    with storage.open(data_json_str, 'r') as f:
                        json.load(f)
                except json.JSONDecodeError:
                    issues["invalid_json"].append(glycan_id)
            
            # Check output directory
            if not (glycan_dir / "output").exists():
                issues["missing_output_dir"].append(glycan_id)
            
            # Check format directories
            missing_formats = []
            for format_dir in required_format_dirs:
                if not (glycan_dir / format_dir).exists():
                    missing_formats.append(format_dir)
            
            if missing_formats:
                issues["missing_format_dirs"].append(f"{glycan_id}: {', '.join(missing_formats)}")
        
        # Log summary
        total_issues = sum(len(issue_list) for issue_list in issues.values())
        if total_issues == 0:
            self.logger.info("Database structure validation passed - no issues found")
        else:
            self.logger.warning(f"Database validation found {total_issues} issues")
            for issue_type, issue_list in issues.items():
                if issue_list:
                    self.logger.warning(f"{issue_type}: {len(issue_list)} entries")
        
        return issues

    def bake_all(self, include_wurcs_submission: bool = False) -> None:
        """Execute all baking operations in the correct order.
        
        Args:
            include_wurcs_submission: Whether to submit WURCS data to GlyTouCan
        """
        self.logger.info("Starting GlycoShape database baking process...")
        
        try:
            # Validate database structure first
            self.logger.info("Validating database structure...")
            validation_results = self.validate_database_structure()
            
            # (FAQ creation removed)
            
            # Create consolidated JSON
            self.logger.info("Creating consolidated JSON...")
            self.create_consolidated_json()
            
            # Calculate simulation statistics
            self.logger.info("Calculating simulation statistics...")
            total_length = self.calculate_total_simulation_length()
            
            # Create archives
            self.logger.info("Creating archives...")
            self.create_master_archive()
            
            # Extract missing GlyTouCan IDs
            self.logger.info("Extracting missing GlyTouCan IDs...")
            self.extract_missing_glytoucan_ids()
            
            # Optionally submit WURCS data
            if include_wurcs_submission:
                if hasattr(config, 'contributor_id') and hasattr(config, 'api_key'):
                    self.logger.info("Submitting WURCS data to GlyTouCan...")
                    self.submit_wurcs_to_glytoucan(
                        config.contributor_id,
                        config.api_key,
                        self.database_dir / "missing_glytoucan.txt"
                    )
                else:
                    self.logger.warning("GlyTouCan credentials not configured - skipping WURCS submission")
            
            self.logger.info("GlycoShape database baking completed successfully!")
            self.logger.info(f"Total simulation time: {total_length} μs")
            
        except Exception as e:
            self.logger.error(f"Database baking failed: {str(e)}")
            raise


def main():
    """Main function to run the baking process."""
    baker = GlycoShapeBaker(config.output_path)
    baker.bake_all(include_wurcs_submission=False)  # Set to True to submit WURCS data


if __name__ == "__main__":
    main()
