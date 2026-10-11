#!/usr/bin/env python3

import json
import shutil
import subprocess
import tempfile
import pandas as pd
from io import BytesIO
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors
import re

from glycowork.motif.processing import canonicalize_iupac
from glycowork.motif.tokenization import glycan_to_composition
from glycowork.motif.draw import GlycoDraw

import logging
from lib import glypdbio, config, name_utils
from lib.storage import get_storage_manager
# The files that need neither identifiers nor the network are written by
# lib.static_ready, which the one-entry mode uses on its own. The writer of
# data.json is there too, for lib.static_levels, which must not need the
# chemistry libraries imported here.
from lib.static_ready import (
    copy_analysis_files,
    copy_and_convert_pdbs,
    copy_static_ready,
    extract_cluster_data,
    load_info_json,
    make_serializable,
    save_glycan_data,
)
import warnings
from urllib3.exceptions import InsecureRequestWarning

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress only the specific InsecureRequestWarning
warnings.filterwarnings("ignore", category=InsecureRequestWarning)


def _render_snfg_svg(glycam_name: str, iupac: str, destination: Path) -> str:
    """Render an SNFG SVG with the configured backend and return its name."""
    destination = Path(destination)

    if config.snfg_renderer == "glycodraw":
        GlycoDraw(iupac, filepath=str(destination), show_linkage=True)
        renderer = "GlycoDraw"
    else:
        command = [
            config.crabwurcs_executable,
            "render",
            "--from",
            "glycam",
            "--output",
            str(destination),
        ]
        try:
            result = subprocess.run(
                command,
                input=glycam_name,
                text=True,
                capture_output=True,
                timeout=60,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"crabWURCS executable not found: {config.crabwurcs_executable!r}. "
                "Install crabwurcs-cli or set GLYCOSHAPE_CRABWURCS_BIN."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("crabWURCS SNFG rendering timed out after 60 seconds") from exc

        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "no error output"
            raise RuntimeError(f"crabWURCS SNFG rendering failed: {detail}")
        renderer = "crabWURCS"

    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"{renderer} did not produce an SVG file")
    if "<svg" not in destination.read_text(encoding="utf-8", errors="ignore"):
        raise RuntimeError(f"{renderer} output is not an SVG document")

    return renderer


def _apply_glytoucan_fallbacks(
    glytoucan: str | None,
    glytoucan_alpha: str | None,
    glytoucan_beta: str | None,
    inventory_glytoucan: str | None,
) -> tuple[str | None, str | None, str | None]:
    """Apply only the inventory fallback for the archetype accession.

    Alpha and beta are distinct structures with distinct WURCS strings.  An
    unresolved variant must remain unresolved rather than inheriting the
    archetype accession.
    """
    if not glytoucan and inventory_glytoucan:
        glytoucan = inventory_glytoucan
    return glytoucan, glytoucan_alpha, glytoucan_beta


def get_glycan_metadata(glycam_name: str) -> dict:
    """Get glycan metadata, trying PocketBase first then falling back to CSV.

    Args:
        glycam_name: GLYCAM name of the glycan

    Returns:
        Dictionary containing metadata from PocketBase or the inventory CSV
    """
    if config.use_pocketbase:
        try:
            from lib.pocketbase import GLYCAN_COLLECTION, get_pocketbase_client, extract_inventory_metadata
            client = get_pocketbase_client()
            if client.is_available():
                record = client.get_record_by_glycam_name(glycam_name)
                if record is not None:
                    metadata = extract_inventory_metadata(record)
                    glycan_record = client.get_record_by_glycam_name(
                        glycam_name,
                        collection=GLYCAN_COLLECTION,
                    )
                    if glycan_record is None and metadata.get("ID"):
                        glycan_record = client.get_record_by_glycoshape_id(
                            metadata["ID"],
                            collection=GLYCAN_COLLECTION,
                        )
                    if not metadata.get("ID") and glycan_record:
                        metadata["ID"] = glycan_record.get("glycoshape_id", "")
                    if not metadata.get("glytoucan_id") and glycan_record:
                        metadata["glytoucan_id"] = glycan_record.get("glytoucan_id", "")
                    logger.info(f"Found metadata for {glycam_name} from PocketBase: ID={metadata['ID']}")
                    return metadata
                else:
                    logger.warning(f"Glycan '{glycam_name}' not found in PocketBase, falling back to CSV")
            else:
                logger.warning("PocketBase not available, falling back to CSV")
        except Exception as e:
            logger.warning(f"PocketBase lookup failed for {glycam_name}: {e}, falling back to CSV")

    return _get_metadata_from_csv(glycam_name)


def get_glycan_metadata_from_inventory(glycam_name: str) -> dict:
    """Backward-compatible wrapper for submission metadata lookup."""
    return get_glycan_metadata(glycam_name)


def _safe_simulation_length_ns(value) -> float | None:
    """Normalize an explicit nanosecond duration from source metadata."""
    if value is None or value == "":
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    return value


def _source_simulation_length_ns(glycam_name: str, storage) -> float | None:
    """Read an uploaded source duration expressed in nanoseconds.

    Manual upload ``info.json`` files use the camel-case ``simulationLength``
    field.  The explicit snake-case spelling is also accepted for newer
    source metadata.  Missing or malformed source metadata is intentionally
    ignored so legacy CSV/PocketBase μs values continue to work.
    """
    if storage is None:
        return None

    source_info_path = Path(config.data_dir) / glycam_name / "info.json"
    try:
        if not storage.exists(source_info_path):
            return None
        with storage.open(source_info_path, "r") as handle:
            source_info = json.load(handle)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.debug("Could not read source simulation metadata for %s: %s", glycam_name, exc)
        return None

    if not isinstance(source_info, dict):
        return None
    for key in ("simulation_length_ns", "simulationLength"):
        length_ns = _safe_simulation_length_ns(source_info.get(key))
        if length_ns is not None:
            return length_ns
    return None


def _simulation_length_us(metadata: dict, glycam_name: str, storage) -> float | None:
    """Resolve the export duration in μs from explicit ns or legacy μs data."""
    # Source metadata is authoritative.  This keeps the export independent of
    # mutable submission-progress records in PocketBase.
    length_ns = _source_simulation_length_ns(glycam_name, storage)
    if length_ns is None:
        length_ns = _safe_simulation_length_ns(metadata.get("simulation_length_ns"))
    if length_ns is not None:
        length_us = length_ns / 1000.0
        logger.info(
            "Using explicit simulation duration for %s: %.6g ns -> %.6g μs",
            glycam_name,
            length_ns,
            length_us,
        )
        return length_us

    # Existing inventory rows are already stored in μs.  Preserve that
    # contract until their source metadata is migrated to an explicit ns
    # field.
    return metadata.get("length")


def _get_metadata_from_csv(glycam_name: str) -> dict:
    """Get glycan metadata from the inventory CSV file (fallback).

    Args:
        glycam_name: GLYCAM name of the glycan

    Returns:
        Dictionary containing metadata from the inventory
    """
    try:
        # Read the inventory CSV (works for local and Oracle PAR)
        storage_manager = get_storage_manager()
        inv_path = str(config.inventory_path)
        if not storage_manager.exists(inv_path):
            # Fallback: try to locate the inventory CSV anywhere in the bucket
            candidates = []
            try:
                # Root listing (Oracle backend ignores pattern and uses prefix)
                for obj in storage_manager.list_files("", "*"):
                    name = obj.as_posix() if hasattr(obj, 'as_posix') else str(obj)
                    if name.lower().endswith("glycoshape_inventory.csv"):
                        candidates.append(name)
            except Exception:
                pass
            if candidates:
                # Prefer shortest path (closest to root)
                candidates.sort(key=lambda s: len(s))
                inv_path = candidates[0]
            else:
                raise FileNotFoundError(f"Inventory file not found: {inv_path}")
        # Read via storage layer to avoid direct FS coupling
        csv_bytes = storage_manager.read_binary(inv_path)
        from io import BytesIO
        df = pd.read_csv(BytesIO(csv_bytes))
        
        # Search for the glycam name in the inventory
        # The column name is 'Full GLYCAM name of glycan being submitted.'
        glycam_col = 'Full GLYCAM name of glycan being submitted.'
        
        # Find matching row
        matching_rows = df[df[glycam_col] == glycam_name]
        
        if matching_rows.empty:
            raise ValueError(f"Glycam name '{glycam_name}' not found in inventory")
        
        if len(matching_rows) > 1:
            logger.warning(f"Multiple entries found for {glycam_name}, using the first one")
        
        # Get the first matching row
        row = matching_rows.iloc[0]
        
        # Extract metadata
        metadata = {
            "ID": row['ID'],
            "timestamp": row['Timestamp'],
            "email": row['Email address'],
            "glycam_name": row[glycam_col],
            "transfer_method": row['How will the data be transferred?'],
            "length": row['What is the aggregated length of the simulations?'],
            "package": row['What MD package was used for the simulations?'],
            "forcefield": row['What force field was used for the simulations?'],
            "temperature": row['What temperature target was used for the simulations? '],
            "pressure": row['What pressure target was used for the simulations?'],
            "salt": row['What NaCl concentration was used for the simulations?'],
            "comments": row['Any comments that should be noted with the submission?'],
            "glytoucan_id": row['What is the GlyTouCan ID of the glycan?']
        }
        
        logger.info(f"Found metadata for {glycam_name}: ID={metadata['ID']}")
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to get metadata from inventory for {glycam_name}: {e}")
        raise


def generate_glycan_id(glycam_name: str) -> str:
    """Generate a simple ID from glycam name as fallback."""
    # Simple ID generation - you can make this more sophisticated if needed
    import hashlib
    # Create a hash of the glycam name for a consistent ID
    hash_obj = hashlib.md5(glycam_name.encode())
    return f"GS{hash_obj.hexdigest()[:6].upper()}"


#: The simulation fields of ``archetype`` in ``data.json``, in the units the website shows:
#: ``length`` in microseconds, ``temperature`` in kelvin, ``pressure`` in bar, ``salt`` in mM.
SIMULATION_FIELDS = ("length", "package", "forcefield", "temperature", "pressure", "salt")


def checked_glycoshape_id(ID, glycam_name: str) -> str:
    """The GlycoShape ID as the name of a static folder; raises when it is not one."""
    ID = str(ID or "").strip()
    if not re.fullmatch(r"GS\d+", ID):
        raise ValueError(
            f"Invalid or missing GlycoShape ID {ID!r} for '{glycam_name}'. "
            "Refusing to write static output because the destination must be a "
            "registered static/GSxxxxx child directory."
        )
    return ID


def fresh_output_dir(output_static_dir, ID: str, glycam_name: str, storage) -> str:
    """Create ``<output_static_dir>/<ID>`` empty and return it; an existing local folder is replaced."""
    output_dir = Path(output_static_dir)
    final_output_path = output_dir / ID
    if final_output_path.parent.resolve() != output_dir.resolve():
        raise ValueError(
            f"Unsafe static output destination for '{glycam_name}': {final_output_path}"
        )
    final_output_dir = str(final_output_path)
    # Remove existing directory if it exists (for both local and Oracle)
    if storage.exists(final_output_dir):
        if hasattr(storage, '_local_backend') and storage._local_backend.exists(final_output_dir):
            # For local backend, remove the directory
            import shutil
            shutil.rmtree(storage._local_backend._resolve_path(final_output_dir))
        # For Oracle, individual files will be overwritten
    storage.mkdir(final_output_dir)
    return final_output_dir


def write_identity_files(
    final_output_dir: str,
    glycam_name: str,
    ID: str,
    cluster_data: dict,
    simulation: dict,
    inventory_glytoucan: str | None = None,
    storage=None,
) -> dict:
    """``snfg.svg``, ``data.json``, ``glygen.json`` and ``glycosmos.json`` of a static folder.

    The part of ``static/GSxxxxx/`` that needs the GlycoShape ID, the
    submission metadata and the name services; both ways of building a
    static folder go through it (:func:`process_glycan` from a process
    folder, :func:`complete_static_ready` from a static-ready one).
    ``simulation`` holds :data:`SIMULATION_FIELDS` in the published units.

    Returns ``glycan_data`` (the content of ``data.json``), ``cluster_levels``
    (the level names of the folder) and ``remarks``: the accessions and IUPAC
    names for the REMARK block of the PDB files, as the keyword arguments of
    ``copy_and_convert_pdbs``.
    """
    if storage is None:
        storage = get_storage_manager()
    length, package, forcefield, temperature, pressure, salt = (simulation.get(name) for name in SIMULATION_FIELDS)

    # Get basic glycan info
    glycam_tidy, end_pos, end_link = name_utils.glycam_info(glycam_name)
    try:
        iupac = canonicalize_iupac(glycam_name)
    except Exception as e:
        logger.warning(f"Could not canonicalize {glycam_name}; using original glycan string: {e}")
        iupac = glycam_name

    # Generate SNFG diagram. Always render to a temporary local file so both
    # local and Oracle storage use the same backend path.
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_file:
            temp_path = Path(tmp_file.name)
        renderer = _render_snfg_svg(glycam_name, iupac, temp_path)
        storage.write_binary(f"{final_output_dir}/snfg.svg", temp_path.read_bytes())
        logger.info(
            "Generated SNFG diagram with %s: %s/snfg.svg",
            renderer,
            final_output_dir,
        )
    except Exception as e:
        logger.warning(f"Could not generate SNFG diagram: {e}")
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    # Process glycan chemical data
    iupac_alpha = f"{iupac}(a{end_pos}-"
    iupac_beta = f"{iupac}(b{end_pos}-"
    glycam_alpha = f"{glycam_tidy}a{end_pos}-OH"
    glycam_beta = f"{glycam_tidy}b{end_pos}-OH"

    logger.info(f"Processing glycan: {glycam_name}, IUPAC: {iupac}, GlyCAM: {glycam_tidy}")

    # Get chemical properties
    try:
        mass, tpsa, rot_bonds, hbond_donor, hbond_acceptor = name_utils.iupac2properties(iupac)
    except Exception as e:
        logger.warning(f"Could not retrieve properties for {iupac}: {e}")
        mass, tpsa, rot_bonds, hbond_donor, hbond_acceptor = None, None, None, None, None

    try:
        composition_glycowork = glycan_to_composition(iupac)
    except Exception:
        composition_glycowork = None

    # Get additional data. Some submitted GLYCAM strings contain decorations
    # that glycowork/glypy cannot parse; keep static generation best-effort.
    try:
        smiles = name_utils.iupac2smiles(iupac)
    except Exception as e:
        logger.warning(f"Could not retrieve SMILES for {iupac}: {e}")
        smiles = None
    try:
        glycoct = name_utils.iupac2glycoct(iupac)
    except Exception as e:
        logger.warning(f"Could not retrieve GlycoCT for {iupac}: {e}")
        glycoct = None
    try:
        composition = name_utils.iupac2composition(iupac)
    except Exception as e:
        logger.warning(f"Could not retrieve composition for {iupac}: {e}")
        composition = None
    try:
        oxford = name_utils.get_oxford(iupac)
    except Exception as e:
        logger.warning(f"Could not retrieve Oxford notation for {iupac}: {e}")
        oxford = None

    try:
        termini = name_utils.iupac2termini(iupac)
    except Exception as e:
        logger.warning(f"Could not retrieve termini for {iupac}: {e}")
        termini = None

    # Get WURCS and GlyTouCan IDs
    try:
        glytoucan, wurcs = name_utils.iupac2wurcs_glytoucan(iupac)
    except Exception as e:
        logger.warning(f"Could not retrieve WURCS/GlyTouCan for {iupac}: {e}")
        glytoucan, wurcs = None, None
    # Generate reducing-end variants from the complete GLYCAM strings.
    # Unlike the legacy open-ended IUPAC spellings, these preserve the
    # alpha/beta anomer and reducing-residue modifications in crabWURCS
    # 0.2.2 (particularly for sulfated GAGs).
    wurcs_alpha = name_utils.crabwurcs_convert(glycam_alpha, "glycam", "wurcs")
    if wurcs_alpha:
        glytoucan_alpha = name_utils.wurcs2glytoucan(wurcs_alpha)
    else:
        try:
            glytoucan_alpha, wurcs_alpha = name_utils.iupac2wurcs_glytoucan(iupac_alpha)
        except Exception as e:
            logger.warning(f"Could not retrieve alpha WURCS/GlyTouCan for {iupac_alpha}: {e}")
            glytoucan_alpha, wurcs_alpha = None, None

    wurcs_beta = name_utils.crabwurcs_convert(glycam_beta, "glycam", "wurcs")
    if wurcs_beta:
        glytoucan_beta = name_utils.wurcs2glytoucan(wurcs_beta)
    else:
        try:
            glytoucan_beta, wurcs_beta = name_utils.iupac2wurcs_glytoucan(iupac_beta)
        except Exception as e:
            logger.warning(f"Could not retrieve beta WURCS/GlyTouCan for {iupac_beta}: {e}")
            glytoucan_beta, wurcs_beta = None, None

    # If GlyCosmos could not convert the name, use crabWURCS locally before
    # falling back to the older SMILES/MolWURCS route.
    if wurcs is None:
        wurcs = name_utils.crabwurcs_convert(iupac, "iupac-condensed", "wurcs")
        if wurcs:
            logger.info("Recovered WURCS locally with crabWURCS")
            variants = name_utils.get_wurcs_variants(wurcs)
            if variants:
                wurcs, wurcs_alpha, wurcs_beta = variants
            glytoucan = name_utils.wurcs2glytoucan(wurcs)
            glytoucan_alpha = name_utils.wurcs2glytoucan(wurcs_alpha) if wurcs_alpha else None
            glytoucan_beta = name_utils.wurcs2glytoucan(wurcs_beta) if wurcs_beta else None

    if wurcs is None and smiles:
        try:
            logger.info("WURCS is None after crabWURCS, trying MolWURCS")
            wurcs = name_utils.smiles2wurcs(smiles)
            variants = name_utils.get_wurcs_variants(wurcs)
            if variants:
                wurcs, wurcs_alpha, wurcs_beta = variants
            else:
                logger.warning(f"Could not derive WURCS variants for {iupac}; keeping archetype WURCS for registration")
            glytoucan = name_utils.wurcs2glytoucan(wurcs)
            glytoucan_alpha = name_utils.wurcs2glytoucan(wurcs_alpha) if wurcs_alpha else None
            glytoucan_beta = name_utils.wurcs2glytoucan(wurcs_beta) if wurcs_beta else None
        except Exception as e:
            logger.warning(f"Could not retrieve WURCS from Mol2WURCS for {iupac}: {e}")
            wurcs = wurcs_alpha = wurcs_beta = None
            glytoucan = glytoucan_alpha = glytoucan_beta = None

    glytoucan, glytoucan_alpha, glytoucan_beta = _apply_glytoucan_fallbacks(
        glytoucan,
        glytoucan_alpha,
        glytoucan_beta,
        inventory_glytoucan,
    )

    # Get motifs
    try:
        motifs = name_utils.glytoucan2motif(glytoucan) if glytoucan else None
    except Exception as e:
        logger.warning(f"Could not retrieve motifs for {iupac}: {e}")
        motifs = None

    # Calculate molecular weight
    mol_weight = None
    if smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mol_weight = round(Descriptors.ExactMolWt(mol), 2)
        except Exception as e:
            logger.warning(f"Could not calculate molecular weight: {e}")

    # Extract cluster-level summaries for static JSON.
    level_1_main_clusters = {}
    level_2_coverage_clusters = {}
    cluster_levels = {}
    torsion_summary = {}

    # Extract PCA clustering metadata (silhouette scores, coverage mapping)
    silhouette_scores = {}
    coverage_clusters_per_main = {}
    pca_variance = {}

    if "pca" in cluster_data:
        pca_data = cluster_data["pca"]
        if "main_clustering" in pca_data:
            silhouette_scores = pca_data["main_clustering"].get("silhouette_scores", {})
            coverage_clusters_per_main = pca_data["main_clustering"].get("coverage_clusters_per_main", {})

        # Extract PCA variance information
        if "pca_analysis" in pca_data:
            pca_analysis = pca_data["pca_analysis"]
            pca_variance = {
                "explained_variance_ratio": pca_analysis.get("explained_variance_ratio", []),
                "cumulative_variance": pca_analysis.get("cumulative_variance", []),
                "n_components": pca_analysis.get("n_components", 10)
            }

    torsions_block = cluster_data.get("torsions", {}) if isinstance(cluster_data, dict) else {}
    family_summary = torsions_block.get("family_summary", {}) if isinstance(torsions_block, dict) else {}
    if torsions_block:
        torsion_summary = {
            "count": torsions_block.get("count"),
            "glycosidic": len(torsions_block.get("glycosidic", [])),
            "non_glycosidic": len(torsions_block.get("non_glycosidic", torsions_block.get("other", []))),
            "by_family": family_summary.get("by_family", {}),
            "by_subtype": family_summary.get("by_subtype", {}),
            "by_common_name": family_summary.get("by_common_name", {}),
        }

    if "levels" in cluster_data:
        for level_key, level_data in cluster_data["levels"].items():
            cluster_pct_map = {}
            precision = 2 if level_key == "level_1" else 4
            for cluster in level_data.get("clusters", []):
                cluster_id = cluster["cluster_id"]
                cluster_name = f"Cluster {cluster_id}"
                cluster_pct_map[cluster_name] = round(cluster["cluster_size_pct"], precision)

            cluster_levels[level_key] = {
                "n_clusters": level_data.get("n_clusters"),
                "clusters": cluster_pct_map,
            }

            if level_key == "level_1":  # Main clustering (5 clusters)
                level_1_main_clusters = dict(cluster_pct_map)
            elif level_key == "level_2":  # Coverage clustering
                level_2_coverage_clusters = dict(cluster_pct_map)

    # Build glycan data structure
    glycan_data = {
        "archetype": {
            # General info
            "ID": ID,
            "name": glycam_name,
            "glycam": glycam_tidy,
            "iupac": iupac,
            "iupac_extended": name_utils.wurcs2extendediupac(wurcs) if wurcs else None,
            "glytoucan": glytoucan,
            "wurcs": wurcs,
            "glycoct": glycoct,
            "smiles": smiles,
            "oxford": oxford,

            # Chemical properties
            "mass": mol_weight,
            "motifs": motifs,
            "termini": termini,
            "components": composition,
            "composition": composition_glycowork,
            "rot_bonds": rot_bonds,
            "hbond_donor": hbond_donor,
            "hbond_acceptor": hbond_acceptor,

            # cluster analysis data
            "entropy": cluster_data.get("entropy_nats"),
            "clusters": level_1_main_clusters,
            "coverage_clusters": level_2_coverage_clusters,
            "cluster_levels": cluster_levels,
            "torsion_summary": torsion_summary,
            "silhouette_scores": silhouette_scores,
            "coverage_clusters_per_main": coverage_clusters_per_main,
            "pca_variance": pca_variance,

            # Molecular Dynamics info from inventory
            "length": length,
            "package": package,
            "forcefield": forcefield,
            "temperature": temperature,
            "pressure": pressure,
            "salt": salt,
        },

        "alpha": {
            # General info
            "ID": ID,
            "name": glycam_name,
            "glycam": glycam_alpha,
            "iupac": iupac_alpha,
            "iupac_extended": name_utils.wurcs2extendediupac(wurcs_alpha) if wurcs_alpha else None,
            "glytoucan": glytoucan_alpha,
            "wurcs": wurcs_alpha,

        },

        "beta": {
            # General info
            "ID": ID,
            "name": glycam_name,
            "glycam": glycam_beta,
            "iupac": iupac_beta,
            "iupac_extended": name_utils.wurcs2extendediupac(wurcs_beta) if wurcs_beta else None,
            "glytoucan": glytoucan_beta,
            "wurcs": wurcs_beta,


        },

        "search_meta": {
            "common_names": [],
            "description": "",
            "keywords": [],
        }
    }

    # Save data files
    save_glycan_data(glycan_data, final_output_dir, "data")
    glygen_data = name_utils.glytoucan2glygen(glytoucan) if glytoucan else {}
    glycosmos_data = name_utils.glytoucan2glycosmos(glytoucan) if glytoucan else {}
    save_glycan_data(glygen_data, final_output_dir, "glygen")
    save_glycan_data(glycosmos_data, final_output_dir, "glycosmos")

    return {
        "glycan_data": glycan_data,
        "cluster_levels": cluster_levels,
        "remarks": {
            "archetype_glytoucan": glytoucan,
            "archetype_iupac": iupac,
            "alpha_glytoucan": glytoucan_alpha,
            "alpha_iupac": iupac_alpha,
            "beta_glytoucan": glytoucan_beta,
            "beta_iupac": iupac_beta,
        },
    }


def process_glycan(folder_path: str, glycam_name: str, output_static_dir: str) -> dict:
    """
    Process a single glycan folder and generate static data.

    Args:
        folder_path: Path to the glycan folder containing output/ subdirectory
        glycam_name: GLYCAM name of the glycan
        output_static_dir: Directory where static data will be created
    """
    from lib.storage import get_storage_manager

    folder_path = Path(folder_path)
    storage = get_storage_manager()

    logger.info(f"Processing glycan: {glycam_name} from {folder_path}")

    # Use storage abstraction layer for validation
    storage = get_storage_manager()

    # Validate input using storage abstraction
    if not storage.exists(folder_path):
        raise FileNotFoundError(f"Folder path does not exist: {folder_path}")

    output_subdir = folder_path / "output"
    if not storage.exists(output_subdir):
        raise FileNotFoundError(f"Output subdirectory not found: {output_subdir}")

    # Load analysis data from info.json
    try:
        info_data = load_info_json(folder_path)
        cluster_data = extract_cluster_data(info_data, folder_path)
    except Exception as e:
        logger.error(f"Failed to load info.json: {e}")
        raise

    # Get metadata from inventory CSV
    try:
        metadata = get_glycan_metadata(glycam_name)
        ID = metadata["ID"]
        simulation = {
            "length": _simulation_length_us(metadata, glycam_name, storage),
            "package": metadata["package"],
            "forcefield": metadata["forcefield"],
            "temperature": metadata["temperature"],
            "pressure": metadata["pressure"],
            "salt": metadata["salt"],
        }
        comments = metadata["comments"]
        inventory_glytoucan = metadata["glytoucan_id"]
    except Exception as e:
        # A GlycoShape ID is assigned by the submissions inventory/PocketBase.
        # Never manufacture one from a name: it looks legitimate (GSxxxxxx)
        # but cannot be reconciled with PocketBase or the public database.
        raise ValueError(
            f"No registered GlycoShape submission metadata for '{glycam_name}'. "
            "Create its PocketBase submission entry with an assigned GS00xxx ID, "
            "then rerun the static stage."
        ) from e

    ID = checked_glycoshape_id(ID, glycam_name)

    logger.info(f"Using ID: {ID}")

    # Create output directory
    final_output_dir = fresh_output_dir(output_static_dir, ID, glycam_name, storage)

    try:
        # snfg.svg, data.json, glygen.json, glycosmos.json
        identity = write_identity_files(
            final_output_dir,
            glycam_name,
            ID,
            cluster_data,
            simulation,
            inventory_glytoucan=inventory_glytoucan,
            storage=storage,
        )

        # Copy and convert PDB files
        copy_and_convert_pdbs(
            folder_path,
            final_output_dir,
            glycam_name,
            cluster_data=cluster_data,
            **identity["remarks"],
        )

        # Tables, info.json, torparts.npz and the frozen PCA landscape
        copy_analysis_files(folder_path, final_output_dir, identity["cluster_levels"].keys())

        logger.info(f"Successfully processed {glycam_name} -> {final_output_dir}")
        return {
            "glycoshape_id": ID,
            "final_output_dir": str(final_output_dir),
            "local_output_dir": str(final_output_dir),
        }

    except Exception as e:
        logger.error(f"Failed to process glycan data for {glycam_name}: {e}")
        raise


def complete_static_ready(
    static_ready_dir,
    glycam_name: str,
    glycoshape_id: str,
    output_static_dir,
    simulation: dict,
    inventory_glytoucan: str | None = None,
) -> dict:
    """``static/GSxxxxx/`` from a static-ready folder and an explicit submission.

    ``static_ready_dir`` is what ``lib.static_ready.write_static_ready`` (the
    one-entry mode, ``scripts/run_entry.py``) wrote: ``output/`` and
    ``landscape/`` without identifiers.  The GlycoShape ID and the simulation
    metadata are arguments here instead of being looked up, and no process
    folder and no other glycan is needed.  The folder that comes out is the
    one :func:`process_glycan` writes from the process folder the
    static-ready one was made of, for the same ID and metadata: both write
    the identifier files with :func:`write_identity_files`, and the shared
    files are those of ``lib.static_ready`` under the same REMARK block.

    ``simulation`` holds :data:`SIMULATION_FIELDS` in the published units.
    Local folders only.  Returns the keys of :func:`process_glycan` plus
    ``glycan_data``.
    """
    from lib.storage import get_storage_manager

    source = Path(static_ready_dir)
    if not (source / "output").is_dir():
        raise FileNotFoundError(f"Output subdirectory not found: {source / 'output'}")
    storage = get_storage_manager()
    try:
        cluster_data = extract_cluster_data(load_info_json(source), source)
    except Exception as e:
        logger.error(f"Failed to load info.json: {e}")
        raise
    ID = checked_glycoshape_id(glycoshape_id, glycam_name)
    logger.info(f"Using ID: {ID}")
    final_output_dir = fresh_output_dir(output_static_dir, ID, glycam_name, storage)
    try:
        identity = write_identity_files(
            final_output_dir,
            glycam_name,
            ID,
            cluster_data,
            simulation,
            inventory_glytoucan=inventory_glytoucan,
            storage=storage,
        )
        copy_static_ready(
            source,
            final_output_dir,
            cluster_data=cluster_data,
            **identity["remarks"],
        )
        logger.info(f"Successfully completed {glycam_name} -> {final_output_dir}")
        return {
            "glycoshape_id": ID,
            "final_output_dir": str(final_output_dir),
            "local_output_dir": str(final_output_dir),
            "glycan_data": identity["glycan_data"],
        }
    except Exception as e:
        logger.error(f"Failed to complete the static folder of {glycam_name}: {e}")
        raise


def main():
    """Example usage of the process_glycan function."""
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python glystatic.py <folder_path> <glycam_name> <output_static_dir>")
        print("Example: python glystatic.py ./DManpa1-2DManpa1-OH DManpa1-2DManpa1-OH ./static_output")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    glycam_name = sys.argv[2] 
    output_static_dir = sys.argv[3]
    
    try:
        process_glycan(folder_path, glycam_name, output_static_dir)
        print(f"Successfully processed {glycam_name}")
    except Exception as e:
        print(f"Error processing {glycam_name}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
