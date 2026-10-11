"""Web lookups of the static stage: bounded, optional, recorded (lib/lookups.py, lib/name_utils.py).

No test here reaches a service: ``requests`` is replaced, and the ``no_internet`` fixture fails a
test that tries anyway.
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.error import URLError

import pytest
import requests

import lib.config as config
from lib import lookups, name_utils

pytestmark = pytest.mark.usefixtures("no_internet")
REPO_ROOT = Path(__file__).resolve().parents[1]


class Response:
    def __init__(self, payload, status=200):
        self.payload, self.status_code, self.encoding = payload, status, "utf-8"

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} Server Error")


class Services:
    """``requests.get`` and ``requests.post`` replaced by one function; every call is kept."""

    def __init__(self, monkeypatch, answer):
        self.calls = []

        def call(method):
            def send(url, **kwargs):
                self.calls.append((method, url, kwargs))
                return answer(method, url, **kwargs)
            return send

        monkeypatch.setattr(requests, "get", call("GET"))
        monkeypatch.setattr(requests, "post", call("POST"))


def unreachable(method, url, **kwargs):
    raise AssertionError(f"a request was made: {method} {url}")


def test_every_request_of_the_name_services_goes_through_the_lookups_module():
    source = (REPO_ROOT / "lib" / "name_utils.py").read_text()
    code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
    assert not re.search(r"requests\.(get|post|request|Session)\(", code)
    assert code.count("lookups.get(") + code.count("lookups.post(") == 12
    assert "timeout=MOLWURCS_TIMEOUT_SECONDS" in code            # the Java fallback is bounded as well


def test_the_pipeline_has_an_offline_switch():
    source = (REPO_ROOT / "main.py").read_text()
    assert "'--offline'" in source and "config.lookups_offline = True" in source
    assert config.lookups_offline is False and config.lookup_timeout == 30.0     # the defaults


def test_a_request_without_a_timeout_of_its_own_gets_the_configured_one(monkeypatch):
    services = Services(monkeypatch, lambda method, url, **kw: Response([{"taxon": "9606"}]))
    assert name_utils.glytoucan2glycosmos("G00001AA") == [{"taxon": "9606"}]
    assert services.calls[0][2]["timeout"] == config.lookup_timeout == 30.0
    monkeypatch.setattr(config, "lookup_timeout", 7.0)
    name_utils.glytoucan2motif("G00001AA")
    assert services.calls[1][2]["timeout"] == 7.0
    services.calls.clear()
    name_utils.wurcs2glytoucan_many(["WURCS=2.0/a"])                 # names its own: 60 s for the batch
    assert services.calls[0][2]["timeout"] == 60


def test_lookups_are_recorded_with_what_was_asked_and_how_it_ended(monkeypatch):
    def answer(method, url, **kwargs):
        if "glygen" in url:
            raise requests.exceptions.ConnectTimeout("no answer from api.glygen.org")
        if "Glycan-taxonomy" in url:
            return Response({}, status=503)
        return Response([{"id": "G12345AB"}])

    Services(monkeypatch, answer)
    with lookups.recording(timeout=5) as log:
        assert name_utils.wurcs2glytoucan("WURCS=2.0/x") == "G12345AB"
        assert name_utils.glytoucan2glygen("G12345AB") is None          # as when the service is down
        assert name_utils.glytoucan2glycosmos("G12345AB") is None
    assert [(entry["lookup"], entry["service"], entry["subject"], entry["status"]) for entry in log.requests] == [
        ("wurcs2glytoucan", "api.glycosmos.org", "WURCS=2.0/x", "ok"),
        ("glytoucan2glygen", "api.glygen.org", "G12345AB", "failed"),
        ("glytoucan2glycosmos", "api.glycosmos.org", "G12345AB", "http_error"),
    ]
    assert log.requests[0]["http_status"] == 200 and log.requests[2]["http_status"] == 503
    assert log.requests[1]["error"].startswith("ConnectTimeout: no answer")
    summary = log.summary()
    assert summary["counts"] == {"ok": 1, "http_error": 1, "failed": 1, "offline": 0, "skipped": 0}
    assert summary["unanswered"] == ["glytoucan2glycosmos", "glytoucan2glygen"]
    assert summary["timeout_seconds"] == 5 and summary["offline"] is False and len(summary["requests"]) == 3
    assert lookups.active_log() is None                                 # nothing is kept outside the block


def test_the_timeout_of_a_block_caps_the_timeout_of_its_requests(monkeypatch):
    services = Services(monkeypatch, lambda method, url, **kw: Response([{"id": "G1"}]))
    with lookups.recording(timeout=5):
        name_utils.wurcs2glytoucan("WURCS=2.0/x")                       # asks for 30 s
    assert services.calls[0][2]["timeout"] == 5


def test_offline_makes_no_request_and_answers_as_a_service_that_is_down(monkeypatch):
    Services(monkeypatch, unreachable)
    monkeypatch.setattr(name_utils, "crabwurcs_convert", lambda *args, **kwargs: None)
    with lookups.recording(offline=True) as log:
        assert name_utils.wurcs2glytoucan("WURCS=2.0/x") is None
        assert name_utils.wurcs2glytoucan_many(["WURCS=2.0/x"]) == {"WURCS=2.0/x": None}
        assert name_utils.iupac2wurcs_glytoucan("Man(a1-2)Man") == (None, None)
        assert name_utils.wurcs2extendediupac("WURCS=2.0/x") is None
        assert name_utils.glytoucan2glygen("G1") is None and name_utils.glytoucan2glycosmos("G1") is None
        assert name_utils.glytoucan2motif("G1") is None and name_utils.glytoucan2iupac("G1") is None
        assert name_utils.normalize_wurcs("WURCS=2.0/x") is None and name_utils.validate_wurcs("WURCS=2.0/x") is None
        assert name_utils.wurcsRDFmatch("WURCS=2.0/x") == []
    assert log.requests and {entry["status"] for entry in log.requests} == {"offline"}
    assert log.summary()["offline"] is True
    assert "wurcs2glytoucan" in log.unanswered() and "glytoucan2glygen" in log.unanswered()


def test_the_offline_setting_applies_without_a_recording_block(monkeypatch):
    Services(monkeypatch, unreachable)
    monkeypatch.setattr(config, "lookups_offline", True)
    assert lookups.is_offline() and name_utils.wurcs2glytoucan("WURCS=2.0/x") is None
    with lookups.recording() as log:                                    # the default of a block is the setting
        assert name_utils.glytoucan2glycosmos("G1") is None
    assert log.offline and log.requests[0]["status"] == "offline"
    with lookups.recording(offline=False):                              # and a block can say otherwise
        assert not lookups.is_offline()


def test_a_used_up_budget_skips_the_remaining_lookups(monkeypatch):
    services = Services(monkeypatch, lambda method, url, **kw: Response([{"id": "G1"}]))
    with lookups.recording(budget=3600) as log:
        assert name_utils.wurcs2glytoucan("WURCS=2.0/x") == "G1"
        log.budget = 0.0                                                # the time is up
        assert name_utils.glytoucan2glycosmos("G1") is None
    assert len(services.calls) == 1
    assert [entry["status"] for entry in log.requests] == ["ok", "skipped"]
    assert "budget" in log.requests[1]["error"] and log.unanswered() == ["glytoucan2glycosmos"]


def test_requests_of_pubchempy_are_bounded_recorded_and_follow_the_offline_switch(monkeypatch):
    pubchempy = pytest.importorskip("pubchempy")
    seen = []

    class Answer:
        status = 200

    def fake_urlopen(url, data=None, **kwargs):
        seen.append((url, kwargs))
        return Answer()

    monkeypatch.setattr(pubchempy, "urlopen", fake_urlopen)             # put back after the test, wrapper included
    lookups.bound_pubchem()
    wrapped = pubchempy.urlopen
    lookups.bound_pubchem()
    assert pubchempy.urlopen is wrapped                                 # installed once
    address = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/JSON"
    with lookups.recording() as log, lookups.labelled("pubchem_properties", "Man(a1-2)Man"):
        assert pubchempy.urlopen(address, b"smiles=C", context=None).status == 200
    assert seen[0][1]["timeout"] == 30.0 and seen[0][1]["context"] is None
    assert log.requests == [{
        "lookup": "pubchem_properties", "service": "pubchem.ncbi.nlm.nih.gov", "subject": "Man(a1-2)Man",
        "status": "ok", "seconds": log.requests[0]["seconds"], "http_status": 200,
    }]
    with lookups.recording(offline=True) as log:
        with pytest.raises(URLError):
            pubchempy.urlopen(address, b"smiles=C")
    assert len(seen) == 1 and log.requests[0]["status"] == "offline"


def test_a_motif_sequence_that_was_not_found_is_asked_for_again(monkeypatch):
    answers = [requests.exceptions.ConnectionError("down"), Response({"results": {"bindings": []}}),
               Response({"results": {"bindings": [{"sequence": {"value": " Man(b1-4)GlcNAc "}}]}})]

    def answer(method, url, **kwargs):
        item = answers.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    services = Services(monkeypatch, answer)
    monkeypatch.setattr(name_utils, "_GLYTOUCAN_IUPAC_CACHE", {})
    assert name_utils.glytoucan2iupac("G00026MO") is None               # the service is down
    assert name_utils.glytoucan2iupac("G00026MO") is None               # it has no sequence
    assert name_utils.glytoucan2iupac("G00026MO") == "Man(b1-4)GlcNAc"
    assert name_utils.glytoucan2iupac("G00026MO") == "Man(b1-4)GlcNAc" and len(services.calls) == 3   # kept now


def test_termini_are_listed_once_in_the_order_they_are_found(monkeypatch):
    monkeypatch.setattr(name_utils, "get_terminal_structures", lambda iupac: ["Neu5Ac(a2-6)", "Fuc(a1-6)", "Neu5Ac(a2-6)", "GlcNAc(b1-4)"])
    assert name_utils.iupac2termini("any") == ["Neu5Ac(a2-6)", "Fuc(a1-6)", "GlcNAc(b1-4)"]
    monkeypatch.setattr(name_utils, "get_terminal_structures", lambda iupac: [])
    assert name_utils.iupac2termini("any") is None


def test_a_missing_crabwurcs_is_noted(monkeypatch):
    monkeypatch.setattr(config, "crabwurcs_executable", "/nonexistent/crabwurcs")
    with lookups.recording() as log:
        assert name_utils.crabwurcs_convert("Man", "iupac-condensed", "wurcs") is None
    assert log.summary()["tools"] == {
        "crabwurcs": {"available": False, "detail": "executable not found: /nonexistent/crabwurcs"}}
