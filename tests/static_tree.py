"""A small static database for the tests of the root files: glycan folders as the static stage leaves them.

The folders hold what the finalization reads (``data.json`` with the three variants, the files of
``GSxxxxx.zip``) and a few files it only lists; the content is made up and small.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def variant(glycam: str, wurcs: Optional[str], glytoucan: Optional[str], iupac: str = "Man(a1-2)Man") -> Dict[str, Any]:
    return {"ID": None, "name": None, "glycam": glycam, "iupac": iupac, "iupac_extended": None,
            "glytoucan": glytoucan, "wurcs": wurcs}


def glycan_data(glycan_id: str, name: str, seed: str = "", **variants: Dict[str, Any]) -> Dict[str, Any]:
    """``data.json`` of a glycan; ``variants`` replace fields of ``archetype``, ``alpha`` and ``beta``."""
    stem = name[:-5] if name.endswith("-OH") else name
    number = int(glycan_id[2:])
    data: Dict[str, Any] = {
        "archetype": {
            **variant(stem, f"WURCS=2.0/{number}/[u]{seed}", f"G{number:05d}AA"),
            "oxford": None, "mass": 342.12 + number, "motifs": [{"motif": "G00026MO", "motif_label": "core"}],
            "termini": ["Man(a1-2)"], "components": {"Man": 2}, "composition": {"Hex": 2},
            "rot_bonds": 4, "hbond_donor": 8, "hbond_acceptor": 11, "entropy": 5.25 + number / 7,
            "clusters": {"Cluster 0": 61.5, "Cluster 1": 38.5},
            "coverage_clusters": {"Cluster 0": 40.25, "Cluster 1": 35.5, "Cluster 2": 24.25},
            "length": 2.5, "package": "GROMACS", "forcefield": "GLYCAM_06j",
            "temperature": 300.0, "pressure": 1.0, "salt": 150.0,
        },
        "alpha": variant(stem + "a1-OH", f"WURCS=2.0/{number}/[a]{seed}", f"G{number:05d}BB"),
        "beta": variant(stem + "b1-OH", f"WURCS=2.0/{number}/[b]{seed}", f"G{number:05d}CC"),
        "search_meta": {"common_names": [], "description": f"{glycan_id} is a glycan. μ→β {seed}", "keywords": ["MD:GROMACS"]},
    }
    for key, changes in variants.items():
        data[key].update(changes)
    for key in ("archetype", "alpha", "beta"):
        data[key]["ID"], data[key]["name"] = glycan_id, name
    return data


def write_glycan(root: Path, glycan_id: str, name: str = "DManpa1-2DManpa1-OH", seed: str = "", archive: bool = True,
                 **variants: Dict[str, Any]) -> Path:
    """Write the folder of a glycan below ``root``; with ``archive`` also its ``GSxxxxx.zip``."""
    from lib.glybake import GlycoShapeBaker

    folder = Path(root) / glycan_id
    if folder.exists():
        shutil.rmtree(folder)
    files = {
        "data.json": json.dumps(glycan_data(glycan_id, name, seed, **variants), indent=4, ensure_ascii=False).encode("utf-8"),
        "snfg.svg": f"<svg><!-- {name} {seed} --></svg>".encode(),
        "glygen.json": b"{}",
        "glycosmos.json": b"[]",
        "output/pca.csv": f"pc1,pc2\n{glycan_id},{seed}\n".encode() * 20,
        "output/info.json": json.dumps({"levels": [{"level": 1}, {"level": 2}], "seed": seed}).encode(),
        "output/level_1/dist.svg": b"<svg>level 1</svg>",
        "output/level_2/dist.svg": b"<svg>level 2</svg>",
        "landscape/landscape_metadata.json": b"{}",
    }
    for level in (1, 2):
        for fmt in ("GLYCAM", "PDB"):
            for anomer in ("alpha", "beta"):
                files[f"output/level_{level}/{fmt}/{anomer}.pdb"] = (
                    f"REMARK {glycan_id} {fmt} {anomer} level {level} {seed}\n" + "ATOM      1  C1  MAN A   1\n" * 40).encode()
    for relative, content in files.items():
        target = folder / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    if archive:
        assert GlycoShapeBaker(root).create_individual_glycan_archive(folder)
    return folder


def write_tree(root: Path, glycan_ids: Iterable[str] = ("GS00001", "GS00002", "GS00005")) -> Path:
    """A static directory of a few glycans, not yet baked."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for index, glycan_id in enumerate(glycan_ids):
        write_glycan(root, glycan_id, name=f"DGalpb1-{index + 2}DGlcpb1-OH", archive=False)
    (root / "missing_glytoucan_no_wurcs_gags.tsv").write_bytes(b"glycoshape_id\tvariant\tiupac\n")
    return root


def full_bake(root: Path) -> None:
    """The finalization of ``main.py`` (``run_database_finalization``) on a local tree."""
    from lib.glybake import GlycoShapeBaker
    from lib.json2rdf import convert_glycoshape_to_rdf
    from lib.storage import local_storage

    baker = GlycoShapeBaker(root)
    with local_storage():
        baker.bake_all(include_wurcs_submission=False)
        assert convert_glycoshape_to_rdf(str(Path(root) / "GLYCOSHAPE.json"), str(Path(root) / "GLYCOSHAPE_RDF.ttl")) is not None
    baker.create_sync_manifest()


def root_files(root: Path) -> Dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(Path(root).iterdir()) if path.is_file()}


def manifest_without_times(payload: bytes) -> Dict[str, Any]:
    """The manifest without what is not a function of the tree: ``generated_at`` and the ``mtime`` values."""
    manifest = json.loads(payload)
    manifest.pop("generated_at")
    for entry in manifest["root_files"].values():
        entry.pop("mtime")
    for glycan in manifest["glycans"].values():
        for entry in glycan["files"].values():
            entry.pop("mtime")
    return manifest
