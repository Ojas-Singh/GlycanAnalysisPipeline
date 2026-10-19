"""GlyTouCan accessions of published entries: which are missing, what a lookup finds now, and the
changes record that sets them.

An entry is published with the accessions its lookups found when it was
completed.  A lookup that failed (a service down, or refusing the server:
GlyCosmos answers the production server with HTTP 403) leaves the accession
empty, and a structure that was registered with GlyTouCan after the entry was
published has one now.  :func:`refresh` looks the WURCS of entries up again
(``lib.name_utils.lookup_accessions_many``: GlyCosmos, then GlyTouCan's own
SPARQL endpoint) and says, per variant (archetype, alpha, beta):

``unchanged``      the published accession is the one found
``fill``           none is published and one is found: the changes record sets it
``correct``        another one is published than the one found for the WURCS: the
                   changes record sets the one found (``differs`` when only missing
                   accessions are filled)
``unconfirmed``    one is published and GlyTouCan has none for the WURCS: kept
``not_registered`` none is published and GlyTouCan has none: the structure has to
                   be registered (its WURCS and names are in the report)
``no_answer``      no service answered: look it up again later
``no_wurcs``       the variant has no WURCS to look up

For each entry with something to set, the report has an ``update-metadata``
changes record (schema ``glycoshape-entry-changes``, ``lib.entry_ops``) with
the accessions in ``glytoucan``.  Nothing here writes to the storage: the
record goes through ``publish_entry.py update-metadata`` (its dry run, then
the publication) like any other change of an entry's metadata.
"""
from __future__ import annotations

import datetime
import json
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from lib import id_registry, lookups

SCHEMA = "glycoshape-accession-refresh"
SCHEMA_VERSION = 1
VARIANTS = ("archetype", "alpha", "beta")
CHANGES_SCHEMA = "glycoshape-entry-changes"
#: What happens to the accession of a variant (see the module documentation).
OUTCOMES = ("unchanged", "fill", "correct", "differs", "unconfirmed", "not_registered", "no_answer", "no_wurcs")
#: The outcomes that put an accession in the changes record.
CHANGING = ("fill", "correct")
#: Where GlyTouCan registers a structure (a GlyTouCan account is needed; the accession follows after a while).
REGISTRATION_URL = "https://glytoucan.org/Users/structure"


def _text(value: Any) -> Optional[str]:
    text = str(value).strip() if value is not None else ""
    return text if text and text.lower() not in ("null", "none") else None


def published(data: Mapping[str, Any]) -> Dict[str, Dict[str, Optional[str]]]:
    """The accession, WURCS and names of each variant of an entry (its ``data.json``, or its record of the inventory)."""
    variants = {}
    for variant in VARIANTS:
        entry = data.get(variant) if isinstance(data.get(variant), Mapping) else {}
        variants[variant] = {
            "glytoucan": _text(entry.get("glytoucan")), "wurcs": _text(entry.get("wurcs")),
            "glycam": _text(entry.get("glycam")), "iupac": _text(entry.get("iupac")),
        }
    return variants


def missing(data: Mapping[str, Any]) -> List[str]:
    """The variants of an entry that have no accession."""
    return [variant for variant, fields in published(data).items() if not fields["glytoucan"]]


def select(entries: Mapping[str, Mapping[str, Any]], ids: Optional[Iterable[str]] = None,
           every: bool = False) -> Tuple[List[str], List[str], List[str]]:
    """Which entries to look up: ``(selected, complete, unknown)``.

    ``ids`` are looked up whatever they have (``unknown``: not among the
    entries); without them, every entry that lacks an accession of a variant
    (``every``: every entry).  ``complete`` are the entries that are not
    selected and have all three accessions.
    """
    if ids:
        wanted = list(dict.fromkeys(str(item).strip().upper() for item in ids if str(item).strip()))
        return [item for item in wanted if item in entries], [], [item for item in wanted if item not in entries]
    chosen = [gid for gid in sorted(entries) if every or missing(entries[gid])]
    complete = [gid for gid in sorted(entries) if gid not in chosen and not missing(entries[gid])]
    return chosen, complete, []


def _outcome(published_accession: Optional[str], wurcs: Optional[str], lookup: Any, only_missing: bool) -> str:
    if not wurcs:
        return "no_wurcs"
    status = getattr(lookup, "status", lookups.NO_ANSWER)
    found = getattr(lookup, "accession", None)
    if status == lookups.NO_ANSWER:
        return "no_answer"
    if status == lookups.NOT_REGISTERED or not found:
        return "unconfirmed" if published_accession else "not_registered"
    if not published_accession:
        return "fill"
    if published_accession == found:
        return "unchanged"
    return "differs" if only_missing else "correct"


def changes_record(glycoshape_id: str, glycam_name: Optional[str], accessions: Mapping[str, str], author: str = "",
                   note: str = "") -> Dict[str, Any]:
    """The ``update-metadata`` changes record that sets these accessions (``{variant: accession}``)."""
    record: Dict[str, Any] = {
        "schema": CHANGES_SCHEMA, "schema_version": 1, "glycoshape_id": glycoshape_id,
        "glytoucan": {variant: accessions[variant] for variant in VARIANTS if accessions.get(variant)},
        "curator_note": note or ("GlyTouCan accessions looked up again: " + ", ".join(
            f"{variant} {accessions[variant]}" for variant in VARIANTS if accessions.get(variant))),
        "author": author,
    }
    if glycam_name:
        record["expect"] = {"glycam_name": glycam_name}
    return record


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def refresh(
    entries: Mapping[str, Mapping[str, Any]],
    ids: Optional[Iterable[str]] = None,
    every: bool = False,
    only_missing: bool = False,
    author: str = "",
    source: str = "",
    lookup_many: Optional[Callable[[List[str]], Dict[str, Any]]] = None,
    offline: Optional[bool] = None,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Look the accessions of entries up again; the report (schema :data:`SCHEMA`), nothing written.

    ``entries`` are ``{GlycoShape ID: data.json}`` (or the records of the
    inventory, which have the same variants); ``ids``, ``every``: as
    :func:`select`.  ``only_missing``: an accession that is published is
    never changed, also where the lookup finds another one (``differs``).
    ``lookup_many`` replaces ``name_utils.lookup_accessions_many`` (tests).
    """
    if lookup_many is None:
        from lib import name_utils

        lookup_many = name_utils.lookup_accessions_many
    selected, complete, unknown = select(entries, ids, every)
    variants = {gid: published(entries[gid]) for gid in selected}
    wanted = [fields["wurcs"] for gid in selected for fields in variants[gid].values() if fields["wurcs"]]
    with lookups.recording(offline=offline, timeout=timeout) as log:
        found = lookup_many(wanted) if wanted else {}
    counts = {outcome: 0 for outcome in OUTCOMES}
    rows = []
    to_register = []
    for gid in selected:
        name = id_registry.full_glycam_name(entries[gid])
        row: Dict[str, Any] = {"glycoshape_id": gid, "glycam_name": name, "variants": {}, "changes": None}
        setting: Dict[str, str] = {}
        for variant, fields in variants[gid].items():
            result = found.get(fields["wurcs"]) if fields["wurcs"] else None
            outcome = _outcome(fields["glytoucan"], fields["wurcs"], result, only_missing)
            counts[outcome] += 1
            row["variants"][variant] = {
                "wurcs": fields["wurcs"], "glycam": fields["glycam"], "iupac": fields["iupac"], "published": fields["glytoucan"],
                "found": getattr(result, "accession", None), "lookup": getattr(result, "status", None),
                "source": getattr(result, "source", None), "outcome": outcome,
            }
            if outcome in CHANGING:
                setting[variant] = row["variants"][variant]["found"]
            if outcome == "not_registered":
                to_register.append({"glycoshape_id": gid, "variant": variant, "wurcs": fields["wurcs"],
                                    "glycam": fields["glycam"], "iupac": fields["iupac"]})
        if setting:
            row["changes"] = changes_record(gid, name, setting, author)
        row["missing_after"] = [variant for variant, item in row["variants"].items()
                                if not (item["published"] or item["outcome"] in CHANGING)]
        rows.append(row)
    summary = log.summary()
    with_changes = [row for row in rows if row["changes"]]
    return {
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION, "generated_at": _utc_now(), "dry_run": True,
        "source": source,
        "selection": {"ids": list(ids) if ids else None, "mode": "ids" if ids else "every" if every else "missing",
                      "only_missing": only_missing, "entries": len(entries), "selected": len(selected),
                      "complete": complete, "unknown": unknown},
        "summary": {
            "entries_looked_up": len(selected),
            "entries_with_changes": len(with_changes),
            "entries_complete_after": sum(1 for row in rows if not row["missing_after"]) + len(complete),
            "anomers_getting_an_accession": sum(len(row["changes"]["glytoucan"]) for row in with_changes),
            "outcomes": counts,
            "not_registered": len(to_register),
        },
        "registration": {"url": REGISTRATION_URL, "structures": to_register},
        "lookups": {key: summary[key] for key in ("offline", "counts", "unanswered", "accession_counts")},
        "entries": rows,
    }


def describe(report: Mapping[str, Any]) -> List[str]:
    """The report in a few lines, for a terminal."""
    summary = report["summary"]
    outcomes = summary["outcomes"]
    lines = [
        f"{summary['entries_looked_up']} entr{'y' if summary['entries_looked_up'] == 1 else 'ies'} looked up "
        f"({report['selection']['mode']}; {len(report['selection']['complete'])} others have all three accessions)",
        f"{summary['entries_with_changes']} would change: {summary['anomers_getting_an_accession']} accession(s) set "
        f"({outcomes['fill']} missing filled, {outcomes['correct']} corrected)",
        f"not registered in GlyTouCan: {outcomes['not_registered']}; lookup failed: {outcomes['no_answer']}; "
        f"unchanged: {outcomes['unchanged']}; unconfirmed: {outcomes['unconfirmed']}; differs (kept): {outcomes['differs']}; "
        f"no WURCS: {outcomes['no_wurcs']}",
    ]
    if report["selection"]["unknown"]:
        lines.append(f"not published: {', '.join(report['selection']['unknown'])}")
    for row in report["entries"]:
        parts = []
        for variant, item in row["variants"].items():
            if item["outcome"] == "unchanged":
                continue
            what = {"fill": f"-> {item['found']}", "correct": f"{item['published']} -> {item['found']}",
                    "differs": f"{item['published']} (GlyTouCan: {item['found']}, kept)",
                    "unconfirmed": f"{item['published']} (not found for its WURCS, kept)"}.get(
                item["outcome"], item["outcome"].replace("_", " "))
            parts.append(f"{variant} {what}")
        if parts:
            lines.append(f"  {row['glycoshape_id']}: " + "; ".join(parts))
    return lines


def encode(report: Mapping[str, Any]) -> bytes:
    return (json.dumps(report, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
