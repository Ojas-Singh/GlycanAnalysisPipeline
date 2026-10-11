"""GlycoShape IDs: helpers for the registry of the service that publishes entries.

The registry itself (which ID belongs to which glycan, published or only
issued) is kept by the caller.  These are the pure functions it needs to
start one from what exists and to issue the next ID; nothing here stores
anything or reads a file.

**The key of the registry is the full GLYCAM name**, the one the raw archive
uses (``glycam_name`` of an archive manifest, ``raw/<GLYCAM name>/``).  The
published inventory (``GLYCOSHAPE.json``) holds it as follows, by
``lib.name_utils.glycam_info`` and ``lib.glystatic.write_identity_files``:
for a name ``N`` that ends in ``-OH`` (``...DGalpNAca1-OH``)

* ``archetype.name`` (and ``alpha.name``, ``beta.name``) is ``N``;
* ``archetype.glycam`` is ``N`` without its last five characters, the
  anomer, the position and ``-OH`` of the reducing end (``...DGalpNAc``);
* ``alpha.glycam`` and ``beta.glycam`` are that stem plus ``a<position>-OH``
  and ``b<position>-OH``, so one of the two is ``N`` again.

For a name without that ending, ``archetype.glycam`` is the name itself and
the two anomers end in ``aNone-OH`` and ``bNone-OH``.  So the full name is
``archetype.name``, never ``archetype.glycam`` (:func:`full_glycam_name`).
A few published entries were built from a folder named by a GlyTouCan
accession: their ``archetype.name`` is that accession, and their GLYCAM name
is known to the raw archive only (:func:`names_from_archive_index`).
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional

ID_DIGITS = 5
_ID = re.compile(r"GS(\d{%d})\Z" % ID_DIGITS)
_GLYTOUCAN_ID = re.compile(r"G\d{5}[A-Z]{2}\Z")
PUBLISHED, ISSUED = "published", "issued"


class RegistryError(ValueError):
    """An ID that is not one, or no ID left."""


def id_number(glycoshape_id: Any) -> int:
    """The number of ``GS`` plus five digits; raises :class:`RegistryError` for anything else."""
    match = _ID.match(glycoshape_id) if isinstance(glycoshape_id, str) else None
    if match is None:
        raise RegistryError(f"not a GlycoShape ID (GS and {ID_DIGITS} digits): {glycoshape_id!r}")
    return int(match.group(1))


def format_id(number: int) -> str:
    if not 0 <= number < 10 ** ID_DIGITS:
        raise RegistryError(f"no GlycoShape ID for the number {number}: an ID has {ID_DIGITS} digits")
    return f"GS{number:0{ID_DIGITS}d}"


def is_id(glycoshape_id: Any) -> bool:
    return isinstance(glycoshape_id, str) and _ID.match(glycoshape_id) is not None


def next_id(issued: Iterable[Any], inventory: Optional[Mapping[str, Any]] = None) -> str:
    """The next ID to issue: one more than the highest that was ever issued or published.

    ``issued`` are all IDs the registry ever gave out, whether their entry
    was published, withdrawn or never arrived; ``inventory`` adds the IDs of
    the published database.  A gap below the highest one is never filled:
    an ID that was issued may be cited somewhere.  Values that are not IDs
    (the empty ID of a record that never got one) are ignored.
    """
    numbers = [id_number(value) for value in issued if is_id(value)]
    numbers += [id_number(value) for value in (inventory or {}) if is_id(value)]
    return format_id(max(numbers, default=0) + 1)


def full_glycam_name(entry: Mapping[str, Any]) -> Optional[str]:
    """The full GLYCAM name of an entry of the inventory (the content of a ``data.json``).

    ``archetype.name``; see the module documentation for why not
    ``archetype.glycam``.  For an entry without a ``name`` the anomer of the
    reducing end cannot be told, and ``None`` is returned unless the entry
    is of a name without a reducing-end suffix.
    """
    archetype = entry.get("archetype") if isinstance(entry, Mapping) else None
    if not isinstance(archetype, Mapping):
        return None
    name = archetype.get("name")
    if isinstance(name, str) and name:
        return name
    alpha = entry.get("alpha") if isinstance(entry.get("alpha"), Mapping) else {}
    stem = archetype.get("glycam")
    if isinstance(stem, str) and stem and alpha.get("glycam") == f"{stem}aNone-OH":
        return stem
    return None


def name_is_consistent(entry: Mapping[str, Any]) -> bool:
    """Whether the three ``glycam`` fields of an entry are those the static stage derives from its name."""
    name = full_glycam_name(entry)
    if name is None:
        return False
    glycam = {variant: (entry.get(variant) or {}).get("glycam") for variant in ("archetype", "alpha", "beta")}
    if name.endswith("-OH"):
        stem, position = name[:-5], name[-4]
        return glycam == {"archetype": stem, "alpha": f"{stem}a{position}-OH", "beta": f"{stem}b{position}-OH"}
    return glycam == {"archetype": name, "alpha": f"{name}aNone-OH", "beta": f"{name}bNone-OH"}


def names_from_archive_index(index: Mapping[str, Any]) -> Dict[str, str]:
    """Source folder name -> GLYCAM name, for the entries of ``raw/index.json`` where the two differ.

    This is the GLYCAM name of an entry that was published under the name of
    its source folder (a GlyTouCan accession), for :func:`seed_registry`.
    """
    names: Dict[str, str] = {}
    for record in index.get("records") or []:
        if isinstance(record, Mapping) and record.get("source_folder") and record.get("glycam_name") \
                and record["source_folder"] != record["glycam_name"]:
            names[record["source_folder"]] = record["glycam_name"]
    return names


def seed_registry(
    inventory: Mapping[str, Any],
    issued: Iterable[Mapping[str, Any]],
    names: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """The start of a registry from the published inventory and the records of every ID ever issued.

    ``inventory`` is ``GLYCOSHAPE.json``; ``issued`` the records of the old
    registry, each with ``glycoshape_id`` and ``glycam_name`` (other fields
    are ignored); ``names`` maps a name that is not a GLYCAM name to the
    GLYCAM name (:func:`names_from_archive_index`).

    Returns ``entries`` (one per ID, sorted: ``glycoshape_id``,
    ``glycam_name``, ``status`` ``published`` or ``issued``, and
    ``published_as`` where the inventory holds another name), ``by_name``
    (GLYCAM name -> ID, for the names that have exactly one ID),
    ``next_id`` and ``conflicts``, which lists what a person has to decide:

    ``name_with_several_ids``  a GLYCAM name that was given more than one ID
    ``id_with_several_names``  an ID that was issued for one name and is published or issued for another
    ``issued_without_id``      records of the old registry that never got an ID
    ``invalid_id``             IDs that are not ``GS`` and five digits
    ``published_not_issued``   published IDs that the old registry does not know
    ``name_not_glycam``        entries whose name is a GlyTouCan accession for which no GLYCAM name is known
    ``inconsistent_entry``     published entries whose ``glycam`` fields are not those of their name
    """
    names = dict(names or {})
    conflicts: Dict[str, List[Any]] = {key: [] for key in (
        "name_with_several_ids", "id_with_several_names", "issued_without_id", "invalid_id", "published_not_issued",
        "name_not_glycam", "inconsistent_entry")}
    entries: Dict[str, Dict[str, Any]] = {}
    claims: Dict[str, List[str]] = {}          # ID -> every name it was recorded for, in order

    def glycam(name: str) -> str:
        return names.get(name, name)

    for glycan_id in sorted(inventory):
        entry = inventory[glycan_id]
        if not is_id(glycan_id):
            conflicts["invalid_id"].append({"glycoshape_id": glycan_id, "status": PUBLISHED})
            continue
        published_as = full_glycam_name(entry)
        if published_as is None:
            conflicts["inconsistent_entry"].append({"glycoshape_id": glycan_id, "reason": "no archetype.name"})
            continue
        if not name_is_consistent(entry):
            conflicts["inconsistent_entry"].append(
                {"glycoshape_id": glycan_id, "reason": "the glycam fields are not those of archetype.name"})
        name = glycam(published_as)
        if _GLYTOUCAN_ID.match(name):
            conflicts["name_not_glycam"].append({"glycoshape_id": glycan_id, "name": name})
        entries[glycan_id] = {"glycoshape_id": glycan_id, "glycam_name": name, "status": PUBLISHED}
        if name != published_as:
            entries[glycan_id]["published_as"] = published_as
        claims.setdefault(glycan_id, []).append(name)

    seen_issued = set()
    for record in issued:
        glycan_id = record.get("glycoshape_id") if isinstance(record, Mapping) else None
        raw_name = record.get("glycam_name") if isinstance(record, Mapping) else None
        name = glycam(raw_name) if isinstance(raw_name, str) else ""
        if not glycan_id:
            conflicts["issued_without_id"].append({"glycam_name": name})
            continue
        if not is_id(glycan_id):
            conflicts["invalid_id"].append({"glycoshape_id": glycan_id, "glycam_name": name, "status": ISSUED})
            continue
        seen_issued.add(glycan_id)
        if name and name not in claims.setdefault(glycan_id, []):
            claims[glycan_id].append(name)
        if glycan_id not in entries:
            entries[glycan_id] = {"glycoshape_id": glycan_id, "glycam_name": name, "status": ISSUED}
    for glycan_id in sorted(set(entries) - seen_issued):
        if entries[glycan_id]["status"] == PUBLISHED:
            conflicts["published_not_issued"].append(glycan_id)
    for glycan_id, claimed in sorted(claims.items()):
        if len(claimed) > 1:
            conflicts["id_with_several_names"].append(
                {"glycoshape_id": glycan_id, "status": entries[glycan_id]["status"], "glycam_names": claimed})

    by_name: Dict[str, List[str]] = {}
    for glycan_id in sorted(entries):
        by_name.setdefault(entries[glycan_id]["glycam_name"], []).append(glycan_id)
    for name, ids in sorted(by_name.items()):
        if name and len(ids) > 1:
            conflicts["name_with_several_ids"].append({
                "glycam_name": name,
                "ids": [{"glycoshape_id": glycan_id, "status": entries[glycan_id]["status"]} for glycan_id in ids],
            })
    return {
        "entries": [entries[glycan_id] for glycan_id in sorted(entries)],
        "by_name": {name: ids[0] for name, ids in sorted(by_name.items()) if name and len(ids) == 1},
        "next_id": next_id(entries),
        "published": sum(1 for entry in entries.values() if entry["status"] == PUBLISHED),
        "issued_only": sum(1 for entry in entries.values() if entry["status"] == ISSUED),
        "conflicts": {key: value for key, value in conflicts.items() if value},
    }
