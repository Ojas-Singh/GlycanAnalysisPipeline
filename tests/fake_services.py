"""The web services and local tools of the static stage, replaced for tests.

``install(monkeypatch)`` makes ``requests.get`` / ``requests.post`` answer with a fixed function of
the request (so two code paths that ask the same things get the same answers), and replaces the
crabWURCS executable and PubChem, which a test machine may not have.  ``install(monkeypatch,
online=False)`` makes every request fail the way it does on a machine without network.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, List, Tuple

import requests


def fake_accession(text: Any) -> str:
    """A GlyTouCan-shaped accession that is a function of ``text``."""
    digest = hashlib.sha256(str(text).encode()).hexdigest()
    return "G%05d%s" % (int(digest[:8], 16) % 100000, "".join(chr(65 + int(c, 16) % 26) for c in digest[8:10]))


class Response:
    def __init__(self, payload: Any, status: int = 200) -> None:
        self.payload, self.status_code, self.encoding = payload, status, "utf-8"

    def json(self) -> Any:
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} Server Error")


def answer(method: str, url: str, **kwargs: Any) -> Response:
    if "wurcs2wurcs" in url:
        return Response([{"id": fake_accession(w), "WURCS": w} for w in kwargs["json"]])
    if "wurcs2gtcids" in url:
        return Response([{"id": fake_accession(kwargs["params"]["wurcs"])}])
    if "iupaccondensed2wurcs" in url:
        name = url.rsplit("/", 1)[-1]
        return Response({"id": fake_accession(name), "WURCS": "WURCS=2.0/9,9,9/[remote]/" + name[:40]})
    if "wurcs2iupacextended" in url:
        return Response({"IUPACextended": "α-ext:" + hashlib.sha256(kwargs["json"]["input"].encode()).hexdigest()[:12]})
    if "get_glycan_motif" in url:
        return Response([{"motif": "G00026MO", "motif_label": "N-Glycan core basic"}, {"motif": "G00031MO"}])
    if "ts.glycosmos.org/sparql" in url:
        return Response({"results": {"bindings": [{"sequence": {"value": "Man(b1-4)GlcNAc(b1-4)GlcNAc"}}]}})
    if "api.glygen.org" in url:
        return Response({"glytoucan": {"glytoucan_ac": kwargs["json"]["glytoucan_ac"]}, "mass": 1234.5, "species": []})
    if "Glycan-taxonomy" in url:
        return Response([{"taxon": "http://identifiers.org/taxonomy/9606", "label": "Homo sapiens"}])
    raise AssertionError(f"unexpected request: {method} {url}")


def fake_crabwurcs(value: Any, source_format: str, target_format: str, timeout: int = 30) -> Any:
    """``name_utils.crabwurcs_convert``: a WURCS for every name, an extended IUPAC for no WURCS."""
    if target_format == "wurcs":
        return f"WURCS=2.0/local/{source_format}/{hashlib.sha256(str(value).encode()).hexdigest()[:16]}"
    return None


def fake_snfg(glycam_name: str, iupac: str, destination: Any) -> str:
    Path(destination).write_text(f"<svg><!-- {glycam_name} --></svg>", encoding="utf-8")
    return "test"


def install(monkeypatch: Any, online: bool = True, crabwurcs: bool = True, snfg: bool = True) -> List[Tuple[str, str]]:
    """Replace the services for the test; returns the list that collects ``(method, url)`` of every request."""
    from lib import glystatic, name_utils

    calls: List[Tuple[str, str]] = []

    def send(method: str):
        def call(url: str, **kwargs: Any) -> Response:
            calls.append((method, url.split("?")[0]))
            if not online:
                raise requests.exceptions.ConnectionError("Temporary failure in name resolution")
            return answer(method, url, **kwargs)
        return call

    monkeypatch.setattr(requests, "get", send("GET"))
    monkeypatch.setattr(requests, "post", send("POST"))
    monkeypatch.setattr(name_utils, "_GLYTOUCAN_IUPAC_CACHE", {})
    if online:
        monkeypatch.setattr(name_utils, "iupac2properties", lambda iupac: (342.1162, 189.5, 4, 8, 11))
    else:
        monkeypatch.setattr(name_utils, "iupac2properties", lambda iupac: None)   # what it returns when PubChem is down
    monkeypatch.setattr(name_utils, "crabwurcs_convert", fake_crabwurcs if crabwurcs else (lambda *a, **k: None))
    monkeypatch.setattr(name_utils, "crabwurcs_normalize", (lambda wurcs, timeout=30: wurcs) if crabwurcs else (lambda *a, **k: None))
    if not crabwurcs:                                    # nor the Java fallback, which a machine may have
        monkeypatch.setattr(name_utils, "smiles2wurcs", lambda smiles: None)
    if snfg:
        monkeypatch.setattr(glystatic, "_render_snfg_svg", fake_snfg)
    else:
        def no_renderer(glycam_name: str, iupac: str, destination: Any) -> str:
            raise RuntimeError("crabWURCS executable not found")
        monkeypatch.setattr(glystatic, "_render_snfg_svg", no_renderer)
    return calls
