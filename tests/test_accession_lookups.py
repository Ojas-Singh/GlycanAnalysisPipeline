"""GlyTouCan accessions: GlyCosmos, then GlyTouCan's SPARQL endpoint; "not registered" told from "no answer".

The answers are the recorded ones of tests/fixtures/accessions/ (GS00094: its archetype and beta
are registered, its alpha is not; GlyCosmos' 403 is what it answers the production server). No
test reaches a service: ``requests`` is replaced, and the ``no_internet`` fixture fails a test
that tries anyway.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from lib import accessions, entry_complete, entry_ops, lookups, name_utils

pytestmark = pytest.mark.usefixtures("no_internet")
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "accessions"


def recorded(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


CONVERTER = recorded("glycosmos_wurcs2wurcs_GS00094.json")
EXACT_ALPHA = recorded("glycosmos_wurcs2gtcids_GS00094_alpha.json")
SPARQL = recorded("glytoucan_sparql_GS00094.json")
CHALLENGE = recorded("glycosmos_403_challenge.json")
ARCHETYPE, ALPHA, BETA = CONVERTER["request"]["json"]


class Response:
    def __init__(self, payload, status=200):
        self.payload, self.status_code, self.encoding = payload, status, "utf-8"

    def json(self):
        if isinstance(self.payload, str):
            raise ValueError("not JSON")
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} Client Error: Forbidden", response=self)


class Services:
    """GlyCosmos and GlyTouCan as recorded, or refusing (``glycosmos="refuse"``) or down (``"down"``)."""

    def __init__(self, monkeypatch, glycosmos="answer", glytoucan="answer"):
        self.calls = []
        self.glycosmos, self.glytoucan = glycosmos, glytoucan

        def call(method):
            def send(url, **kwargs):
                self.calls.append((method, url, kwargs))
                return self.answer(method, url, **kwargs)
            return send

        monkeypatch.setattr(requests, "get", call("GET"))
        monkeypatch.setattr(requests, "post", call("POST"))
        monkeypatch.setattr(name_utils, "crabwurcs_normalize", lambda wurcs, timeout=30: wurcs)

    def answer(self, method, url, **kwargs):
        if "glycosmos.org" in url:
            if self.glycosmos == "refuse":
                return Response(CHALLENGE["body"], CHALLENGE["status"])
            if self.glycosmos == "down":
                raise requests.exceptions.ConnectTimeout("no answer from api.glycosmos.org")
            if url == name_utils.GLYCOSMOS_WURCS2WURCS:
                by_wurcs = {item["WURCS"]: item for item in CONVERTER["body"]}
                return Response([by_wurcs.get(w, {"message": "Accession not found.", "WURCS": w}) for w in kwargs["json"]])
            if url == name_utils.GLYCOSMOS_WURCS2GTCIDS:
                return Response(EXACT_ALPHA["body"])
        if url == name_utils.GLYTOUCAN_SPARQL_ENDPOINT:
            if self.glytoucan == "down":
                raise requests.exceptions.ConnectionError("Temporary failure in name resolution")
            asked = kwargs["data"]["query"]
            bindings = [binding for binding in SPARQL["body"]["results"]["bindings"]
                        if json.dumps(binding["sequence"]["value"]) in asked]
            return Response({"head": SPARQL["body"]["head"], "results": {"bindings": bindings}})
        raise AssertionError(f"unexpected request: {method} {url}")

    def urls(self):
        return [url for _, url, _ in self.calls]


def test_the_recorded_answers_are_those_of_gs00094():
    assert [item.get("id") for item in CONVERTER["body"]] == ["G67924FL", None, "G16828VN"]
    assert CONVERTER["body"][1]["message"] == "Accession not found."
    assert EXACT_ALPHA["body"] == [{"message": "Accession not found."}]
    assert sorted(b["id"]["value"] for b in SPARQL["body"]["results"]["bindings"]) == ["G16828VN", "G67924FL"]
    assert CHALLENGE["status"] == 403 and "Just a moment" in CHALLENGE["body"]


def test_glycosmos_answers_from_the_spark(monkeypatch):
    services = Services(monkeypatch)
    with lookups.recording() as log:
        assert name_utils.wurcs2glytoucan(ARCHETYPE) == "G67924FL"
        alpha = name_utils.lookup_accession(ALPHA)
    assert services.urls()[0] == name_utils.GLYCOSMOS_WURCS2WURCS
    # GlyCosmos said it has none: GlyTouCan is asked (it may have what GlyCosmos has not synchronised), and has none.
    assert (alpha.status, alpha.accession, alpha.source) == ("not_registered", None, "glytoucan")
    assert name_utils.GLYCOSMOS_WURCS2GTCIDS not in services.urls()        # "not found" needs no exact-match list
    summary = log.summary()
    assert summary["unanswered"] == [] and summary["accession_counts"] == {"found": 1, "not_registered": 1, "no_answer": 0}


def test_glycosmos_refuses_the_server_and_glytoucan_answers(monkeypatch):
    services = Services(monkeypatch, glycosmos="refuse")
    with lookups.recording() as log:
        found = [name_utils.lookup_accession(w) for w in (ARCHETYPE, ALPHA, BETA)]
    assert [(r.status, r.accession, r.source) for r in found] == [
        ("found", "G67924FL", "glytoucan"), ("not_registered", None, "glytoucan"), ("found", "G16828VN", "glytoucan")]
    # After a 403 the exact-match list of the same service is not asked.
    assert name_utils.GLYCOSMOS_WURCS2GTCIDS not in services.urls()
    summary = log.summary()
    assert summary["unanswered"] == []                                   # GlyTouCan answered every question
    refused = [entry for entry in summary["requests"] if entry["lookup"] == "wurcs2glytoucan"]
    assert {(entry["status"], entry["http_status"], entry["covered_by"]) for entry in refused} == {("http_error", 403, "glytoucan")}
    assert summary["accession_counts"] == {"found": 2, "not_registered": 1, "no_answer": 0}


def test_no_service_answers(monkeypatch):
    Services(monkeypatch, glycosmos="down", glytoucan="down")
    with lookups.recording() as log:
        result = name_utils.lookup_accession(ALPHA)
    assert (result.status, result.accession, result.source) == ("no_answer", None, None)
    assert len(result.errors) == 3                                       # converter, exact list, GlyTouCan
    assert log.unanswered() == ["glytoucan_sparql", "wurcs2glytoucan", "wurcs2glytoucan_exact"]


def test_glycosmos_says_none_and_glytoucan_is_down_is_not_registered(monkeypatch):
    Services(monkeypatch, glytoucan="down")
    with lookups.recording() as log:
        result = name_utils.lookup_accession(ALPHA)
    assert (result.status, result.source) == ("not_registered", "glycosmos")
    assert log.unanswered() == []


def test_glytoucan_is_asked_for_the_canonical_spelling_too(monkeypatch):
    services = Services(monkeypatch, glycosmos="refuse")
    written_otherwise = "WURCS=2.0/as-written-by-another-tool"
    monkeypatch.setattr(name_utils, "crabwurcs_normalize", lambda wurcs, timeout=30: BETA if wurcs == written_otherwise else wurcs)
    with lookups.recording():
        result = name_utils.lookup_accession(written_otherwise)
    assert (result.status, result.accession, result.wurcs) == ("found", "G16828VN", written_otherwise)
    query = services.calls[-1][2]["data"]["query"]
    assert json.dumps(written_otherwise) in query and json.dumps(BETA) in query


def test_many_at_once_when_glycosmos_refuses(monkeypatch):
    services = Services(monkeypatch, glycosmos="refuse")
    with lookups.recording() as log:
        found = name_utils.wurcs2glytoucan_many([ARCHETYPE, ALPHA, BETA, ALPHA])
    assert found == {ARCHETYPE: "G67924FL", ALPHA: None, BETA: "G16828VN"}
    assert services.urls() == [name_utils.GLYCOSMOS_WURCS2WURCS, name_utils.GLYTOUCAN_SPARQL_ENDPOINT]   # one batch each
    assert log.unanswered() == [] and log.summary()["accession_counts"]["not_registered"] == 1


def test_many_at_once_from_the_spark(monkeypatch):
    services = Services(monkeypatch)
    results = name_utils.lookup_accessions_many([ARCHETYPE, ALPHA, BETA])
    assert {w: (r.status, r.source) for w, r in results.items()} == {
        ARCHETYPE: ("found", "glycosmos"), ALPHA: ("not_registered", "glytoucan"), BETA: ("found", "glycosmos")}
    assert services.calls[1][2]["data"]["query"].count("WURCS=") == 1                # only the alpha is asked of GlyTouCan


def test_every_request_names_the_pipeline(monkeypatch):
    services = Services(monkeypatch, glycosmos="refuse")
    name_utils.lookup_accession(ARCHETYPE)
    assert {kwargs["headers"]["User-Agent"] for _, _, kwargs in services.calls} == {lookups.USER_AGENT}
    assert services.calls[-1][2]["headers"]["Accept"] == "application/sparql-results+json"


def test_offline_is_no_answer(monkeypatch):
    services = Services(monkeypatch)
    with lookups.recording(offline=True) as log:
        result = name_utils.lookup_accession(ALPHA)
    assert result.status == "no_answer" and services.calls == []
    assert "wurcs2glytoucan" in log.unanswered() and "glytoucan_sparql" in log.unanswered()


def test_normalize_wurcs_is_local_first(monkeypatch):
    services = Services(monkeypatch)
    monkeypatch.setattr(name_utils, "crabwurcs_normalize", lambda wurcs, timeout=30: "WURCS=2.0/canonical")
    assert name_utils.normalize_wurcs("WURCS=2.0/other") == "WURCS=2.0/canonical" and services.calls == []


# --- the report of a completion -------------------------------------------------------------------

def gs00094(alpha=None, archetype="G67924FL", beta="G16828VN"):
    return {
        "archetype": {"ID": "GS00094", "name": "DGalpb1-4DGlcpNAcb1-2DManpa1-3[DManpa1-3[DManpa1-6]DManpa1-6]DManpb1-4DGlcpNAcb1-4DGlcpNAcb1-OH",
                      "glycam": "DGalpb1-4DGlcpNAcb1-2DManpa1-3[DManpa1-3[DManpa1-6]DManpa1-6]DManpb1-4DGlcpNAcb1-4DGlcpNAc",
                      "iupac": "Gal(b1-4)GlcNAc(b1-2)Man(a1-3)[Man(a1-3)[Man(a1-6)]Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
                      "wurcs": ARCHETYPE, "glytoucan": archetype},
        "alpha": {"glycam": "DGalpb1-4DGlcpNAcb1-2DManpa1-3[DManpa1-3[DManpa1-6]DManpa1-6]DManpb1-4DGlcpNAcb1-4DGlcpNAca1-OH",
                  "iupac": "Gal(b1-4)GlcNAc(b1-2)Man(a1-3)[Man(a1-3)[Man(a1-6)]Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc(a1-",
                  "wurcs": ALPHA, "glytoucan": alpha},
        "beta": {"glycam": "DGalpb1-4DGlcpNAcb1-2DManpa1-3[DManpa1-3[DManpa1-6]DManpa1-6]DManpb1-4DGlcpNAcb1-4DGlcpNAcb1-OH",
                 "iupac": "Gal(b1-4)GlcNAc(b1-2)Man(a1-3)[Man(a1-3)[Man(a1-6)]Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc(b1-",
                 "wurcs": BETA, "glytoucan": beta},
    }


def test_the_report_of_a_completion_says_not_registered_and_lookup_failed(monkeypatch):
    Services(monkeypatch, glycosmos="refuse")
    with lookups.recording() as log:
        for wurcs in (ARCHETYPE, ALPHA):
            name_utils.lookup_accession(wurcs)
    data = gs00094(beta=None)                 # the beta was not looked up in this completion
    states = entry_complete.accession_states(data, log.summary())
    assert states == {
        "archetype": {"accession": "G67924FL", "status": "found", "source": "glytoucan"},
        "alpha": {"accession": None, "status": "not_registered", "source": "glytoucan"},
        "beta": {"accession": None, "status": "not_looked_up", "source": None},
    }
    warnings = entry_complete.accession_warnings(states)
    assert warnings == ["GlyTouCan accession not registered: alpha (GlyTouCan answered and has none for its WURCS; "
                        "register the structure, then look it up again)"]

    Services(monkeypatch, glycosmos="refuse", glytoucan="down")
    with lookups.recording() as log:
        name_utils.lookup_accession(ALPHA)
    states = entry_complete.accession_states(gs00094(archetype="G00001AA"), log.summary())
    assert states["alpha"]["status"] == "no_answer" and states["archetype"]["status"] == "given"
    assert entry_complete.accession_warnings(states)[0].startswith("GlyTouCan accession lookup failed: alpha")


# --- looking the accessions of published entries up again ------------------------------------------

def test_refresh_fills_corrects_and_lists_what_to_register(monkeypatch):
    Services(monkeypatch, glycosmos="refuse")
    entries = {
        "GS00094": gs00094(archetype=None, beta=None),                   # as published on 2026-10-17
        "GS00002": gs00094(alpha="G11111AA"),                            # all three: not looked up
        "GS00003": gs00094(archetype="G93633XO"),                        # another accession than GlyTouCan's
    }
    entries["GS00003"]["beta"]["wurcs"] = None
    report = accessions.refresh(entries, author="admin")
    assert report["selection"]["mode"] == "missing" and report["selection"]["complete"] == ["GS00002"]
    rows = {row["glycoshape_id"]: row for row in report["entries"]}
    assert sorted(rows) == ["GS00003", "GS00094"]
    assert {v: item["outcome"] for v, item in rows["GS00094"]["variants"].items()} == {
        "archetype": "fill", "alpha": "not_registered", "beta": "fill"}
    assert {v: item["outcome"] for v, item in rows["GS00003"]["variants"].items()} == {
        "archetype": "correct", "alpha": "not_registered", "beta": "no_wurcs"}
    changes = rows["GS00094"]["changes"]
    assert changes["glytoucan"] == {"archetype": "G67924FL", "beta": "G16828VN"}
    assert changes["expect"]["glycam_name"].endswith("DGlcpNAcb1-OH") and changes["author"] == "admin"
    parsed = entry_ops.Changes.parse(changes)                            # what update-metadata takes
    assert parsed.glytoucan == {"archetype": "G67924FL", "beta": "G16828VN"} and parsed.glycoshape_id == "GS00094"
    assert rows["GS00094"]["missing_after"] == ["alpha"]
    assert report["summary"]["anomers_getting_an_accession"] == 3 and report["summary"]["not_registered"] == 2
    assert [item["glycoshape_id"] for item in report["registration"]["structures"]] == ["GS00003", "GS00094"]
    assert report["registration"]["structures"][1]["wurcs"] == ALPHA and report["lookups"]["unanswered"] == []
    only = accessions.refresh(entries, only_missing=True)
    assert {row["glycoshape_id"]: row["variants"]["archetype"]["outcome"] for row in only["entries"]}["GS00003"] == "differs"
    assert [row["glycoshape_id"] for row in only["entries"] if row["changes"]] == ["GS00094"]


def test_refresh_without_an_answer_changes_nothing(monkeypatch):
    Services(monkeypatch, glycosmos="down", glytoucan="down")
    report = accessions.refresh({"GS00094": gs00094(archetype=None, beta=None)}, ids=["GS00094", "GS09999"])
    row = report["entries"][0]
    assert {v: item["outcome"] for v, item in row["variants"].items()} == {
        "archetype": "no_answer", "alpha": "no_answer", "beta": "no_answer"}
    assert row["changes"] is None and report["selection"]["unknown"] == ["GS09999"]
    assert report["summary"]["outcomes"]["no_answer"] == 3 and "glytoucan_sparql" in report["lookups"]["unanswered"]


def refresh_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location("refresh_accessions", Path(__file__).resolve().parents[1] / "scripts" / "refresh_accessions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_script_is_a_dry_run_unless_told_to_write(monkeypatch, tmp_path, capsys):
    refresh_accessions = refresh_script()
    Services(monkeypatch, glycosmos="refuse")
    static = tmp_path / "static"
    (static / "GS00094").mkdir(parents=True)
    (static / "GS00094" / "data.json").write_text(json.dumps(gs00094(archetype=None, beta=None)), encoding="utf-8")
    fetched = tmp_path / "GS00095.json"
    fetched.write_text(json.dumps({**gs00094(alpha=None), "archetype": {**gs00094()["archetype"], "ID": "GS00095"}}), encoding="utf-8")
    assert refresh_accessions.main(["--static-dir", str(static), "--entry", str(fetched), "--report", str(tmp_path / "r.json")]) == 0
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["dry_run"] is True and report["written"] == [] and report["summary"]["entries_with_changes"] == 1
    assert "dry run" in capsys.readouterr().out
    assert refresh_accessions.main(["--static-dir", str(static), "--write-changes", str(tmp_path / "changes"), "--author", "x"]) == 0
    written = json.loads((tmp_path / "changes" / "GS00094.changes.json").read_text())
    assert written["glytoucan"] == {"archetype": "G67924FL", "beta": "G16828VN"} and written["author"] == "x"
    with pytest.raises(SystemExit) as stopped:                           # argparse: exit status 2
        refresh_accessions.main(["--id", "GS1", "--static-dir", str(static)])
    assert stopped.value.code == 2
    assert refresh_accessions.main(["--static-dir", str(tmp_path / "nothing")]) == 1
