"""Web lookups of the static stage: bounded, optional and recorded.

Names, accessions and cross references of an entry come from web services
(GlyCosmos and GlyTouCan, GlyGen, PubChem).  They are best effort: an entry
is written without what a service did not answer.  This module is the one
place the requests go through, so that

* every request has a timeout (``GLYCOSHAPE_LOOKUP_TIMEOUT``, 30 s, for the
  requests that name none), and a group of lookups can have a time budget
  after which the rest is not attempted;
* lookups can be switched off (``GLYCOSHAPE_OFFLINE=1``, or ``offline=True``
  of :func:`recording`): no request is made and the caller sees the failure
  it sees when a service is down;
* what was asked and how it ended can be recorded, so that an entry
  published while a service was down can be refreshed later::

      with lookups.recording(offline=False, budget=300) as log:
          ...                      # anything that calls lib.name_utils
      log.summary()                # what was looked up, what failed

Without :func:`recording` nothing is kept and the behaviour is the one of
the settings.  The log belongs to the thread (context) that opened it.

Every request names the pipeline in its ``User-Agent`` (:data:`USER_AGENT`).

GlyTouCan accessions are looked up in more than one service (GlyCosmos,
then GlyTouCan's own SPARQL endpoint; ``lib.name_utils.lookup_accession``).
The outcome of each such lookup is kept apart from its requests
(:func:`settle_accession`): ``found``, ``not_registered`` (a service
answered and has no accession for the WURCS) or ``no_answer`` (none
answered).  A request that failed while another service answered for the
same WURCS is marked ``covered_by`` that service and is not unanswered.
"""
from __future__ import annotations

import contextlib
import contextvars
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional
from urllib.parse import urlsplit

import requests

import lib.config as config

#: Statuses of a recorded request.  All but ``ok`` mean that the answer is missing.
OK, HTTP_ERROR, FAILED, OFFLINE, SKIPPED = "ok", "http_error", "failed", "offline", "skipped"
UNANSWERED = (HTTP_ERROR, FAILED, OFFLINE, SKIPPED)
#: Outcomes of an accession lookup (:func:`settle_accession`).
FOUND, NOT_REGISTERED, NO_ANSWER = "found", "not_registered", "no_answer"
ACCESSION_OUTCOMES = (FOUND, NOT_REGISTERED, NO_ANSWER)
#: The lookups that ask for a GlyTouCan accession (lib.name_utils): their outcome is the one of the WURCS.
ACCESSION_LOOKUPS = ("wurcs2glytoucan", "wurcs2glytoucan_exact", "wurcs2glytoucan_many", "glytoucan_sparql")
#: What every request of the pipeline says it is (GlyCosmos refuses some default user agents).
USER_AGENT = "GlycoShape-GlycanAnalysisPipeline/1.0 (+https://github.com/Ojas-Singh/GlycanAnalysisPipeline)"
_SUBJECT_LIMIT = 300
_ERROR_LIMIT = 300


class LookupUnavailable(requests.exceptions.RequestException):
    """A lookup that was not attempted: lookups are offline, or the time budget is used up."""


@dataclass
class LookupLog:
    """The lookups made while it was active (:func:`recording`)."""

    offline: bool = False
    timeout: Optional[float] = None
    budget: Optional[float] = None
    requests: List[Dict[str, Any]] = field(default_factory=list)
    tools: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    #: The outcome of every accession lookup: ``{"wurcs", "status", "accession", "source"}``.
    accessions: List[Dict[str, Any]] = field(default_factory=list)
    _started: float = field(default_factory=time.monotonic, repr=False)

    def remaining(self) -> Optional[float]:
        """Seconds left of the budget; ``None`` without one."""
        if self.budget is None:
            return None
        return self.budget - (time.monotonic() - self._started)

    def unanswered(self) -> List[str]:
        """Names of the lookups of which at least one request got no answer, sorted.

        A request that failed while another service answered the same question
        (``covered_by``) is not counted.
        """
        return sorted({entry["lookup"] for entry in self.requests
                       if entry["status"] in UNANSWERED and not entry.get("covered_by")})

    def summary(self) -> Dict[str, Any]:
        """A JSON document of the log: settings, counts per status, what to retry, every request.

        ``accessions`` are the outcomes of the accession lookups, one per WURCS
        (the last one when a WURCS was looked up twice), and
        ``accession_counts`` their counts.
        """
        counts = {status: 0 for status in (OK,) + UNANSWERED}
        for entry in self.requests:
            counts[entry["status"]] += 1
        outcomes = {item["wurcs"]: item for item in self.accessions}
        accession_counts = {status: 0 for status in ACCESSION_OUTCOMES}
        for item in outcomes.values():
            accession_counts[item["status"]] += 1
        return {
            "offline": self.offline,
            "timeout_seconds": self.timeout if self.timeout is not None else float(config.lookup_timeout),
            "budget_seconds": self.budget,
            "counts": counts,
            "unanswered": self.unanswered(),
            "accession_counts": accession_counts,
            "accessions": [dict(item) for item in outcomes.values()],
            "tools": {name: dict(state) for name, state in sorted(self.tools.items())},
            "requests": [dict(entry) for entry in self.requests],
        }


_ACTIVE: contextvars.ContextVar[Optional[LookupLog]] = contextvars.ContextVar("glycoshape_lookups", default=None)
_LABEL: contextvars.ContextVar[Optional[tuple]] = contextvars.ContextVar("glycoshape_lookup_label", default=None)


def active_log() -> Optional[LookupLog]:
    """The log of this thread, when lookups are being recorded."""
    return _ACTIVE.get()


def is_offline() -> bool:
    """Whether a lookup made now would be refused."""
    log = _ACTIVE.get()
    return log.offline if log is not None else bool(config.lookups_offline)


@contextlib.contextmanager
def recording(
    offline: Optional[bool] = None,
    timeout: Optional[float] = None,
    budget: Optional[float] = None,
) -> Iterator[LookupLog]:
    """Record the lookups of the block, with its own settings.

    ``offline`` defaults to ``GLYCOSHAPE_OFFLINE``.  ``timeout`` caps the
    timeout of every request of the block; ``budget`` is the time all of them
    may take together: once it is used up the remaining lookups are recorded
    as ``skipped`` and fail at once.
    """
    log = LookupLog(
        offline=bool(config.lookups_offline) if offline is None else bool(offline),
        timeout=timeout,
        budget=budget,
    )
    token = _ACTIVE.set(log)
    try:
        yield log
    finally:
        _ACTIVE.reset(token)


@contextlib.contextmanager
def labelled(lookup: str, subject: Any = None) -> Iterator[None]:
    """Name the requests that a library makes on its own inside the block (PubChem through pubchempy)."""
    token = _LABEL.set((lookup, subject))
    try:
        yield
    finally:
        _LABEL.reset(token)


def tool(name: str, available: bool, detail: str = "") -> None:
    """Note whether a local tool that replaces a web service (crabWURCS) could be run."""
    log = _ACTIVE.get()
    if log is None:
        return
    state = log.tools.setdefault(name, {"available": True, "detail": ""})
    if not available:
        state.update(available=False, detail=str(detail)[:_ERROR_LIMIT])
    elif state["available"] and detail:
        state["detail"] = str(detail)[:_ERROR_LIMIT]


def mark() -> int:
    """Where the log of this thread stands now (the number of requests it holds; 0 without a log).

    Give it to :func:`settle_accession` as ``since``: the requests made after
    it are the ones of that lookup.
    """
    log = _ACTIVE.get()
    return len(log.requests) if log is not None else 0


def settle_accession(wurcs: str, status: str, accession: Optional[str] = None, source: Optional[str] = None,
                     since: Optional[int] = None) -> None:
    """Record how the accession lookup of ``wurcs`` ended.

    ``status`` is :data:`FOUND`, :data:`NOT_REGISTERED` or :data:`NO_ANSWER`;
    ``source`` the service that answered.  When a service answered, the
    accession requests since ``since`` that got no answer are marked
    ``covered_by`` it: another service failed, and the question was answered.
    """
    if status not in ACCESSION_OUTCOMES:
        raise ValueError(f"not an outcome of an accession lookup: {status!r}")
    log = _ACTIVE.get()
    if log is None:
        return
    if status != NO_ANSWER and since is not None:
        cover(since, source or "another service")
    log.accessions.append({"wurcs": str(wurcs), "status": status, "accession": accession, "source": source})


def cover(since: int, by: str) -> None:
    """Mark the accession requests since ``since`` that got no answer as answered by ``by``."""
    log = _ACTIVE.get()
    if log is None:
        return
    for entry in log.requests[since:]:
        if entry["status"] in UNANSWERED and entry["lookup"] in ACCESSION_LOOKUPS:
            entry.setdefault("covered_by", by)


def _short(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _attempt(lookup: str, subject: Any, service: str, timeout: Optional[float], call: Callable[[float], Any]) -> Any:
    log = _ACTIVE.get()
    entry: Dict[str, Any] = {"lookup": lookup, "service": service, "subject": _short(subject, _SUBJECT_LIMIT)}

    def record(status: str, **more: Any) -> None:
        if log is not None:
            log.requests.append({**entry, "status": status, **more})

    if is_offline():
        record(OFFLINE)
        raise LookupUnavailable(f"{lookup}: lookups are offline")
    limit = timeout or float(config.lookup_timeout)
    if log is not None:
        if log.timeout is not None:
            limit = min(limit, log.timeout)
        remaining = log.remaining()
        if remaining is not None:
            if remaining <= 0:
                record(SKIPPED, error="the time budget for lookups is used up")
                raise LookupUnavailable(f"{lookup}: the time budget for lookups is used up")
            limit = min(limit, remaining)
    started = time.monotonic()
    try:
        response = call(limit)
    except Exception as exc:
        record(FAILED, error=_short(f"{type(exc).__name__}: {exc}", _ERROR_LIMIT),
               seconds=round(time.monotonic() - started, 3))
        raise
    code = getattr(response, "status_code", None)
    if not isinstance(code, int):
        code = getattr(response, "status", None)
    more: Dict[str, Any] = {"seconds": round(time.monotonic() - started, 3)}
    if isinstance(code, int):
        more["http_status"] = code
    record(HTTP_ERROR if isinstance(code, int) and code >= 400 else OK, **more)
    return response


def _headers(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """The request's arguments with the pipeline's ``User-Agent`` (unless the caller gave one)."""
    headers = dict(kwargs.get("headers") or {})
    if not any(str(name).lower() == "user-agent" for name in headers):
        headers["User-Agent"] = USER_AGENT
    return {**kwargs, "headers": headers}


def get(url: str, *, lookup: str, subject: Any = None, timeout: Optional[float] = None, **kwargs: Any) -> Any:
    """``requests.get`` with a timeout, refused when offline, recorded when a log is active."""
    kwargs = _headers(kwargs)
    return _attempt(lookup, subject, urlsplit(url).netloc, timeout,
                    lambda limit: requests.get(url, timeout=limit, **kwargs))


def post(url: str, *, lookup: str, subject: Any = None, timeout: Optional[float] = None, **kwargs: Any) -> Any:
    """``requests.post`` with a timeout, refused when offline, recorded when a log is active."""
    kwargs = _headers(kwargs)
    return _attempt(lookup, subject, urlsplit(url).netloc, timeout,
                    lambda limit: requests.post(url, timeout=limit, **kwargs))


def bound_pubchem() -> None:
    """Give the requests of ``pubchempy`` a timeout, and make them follow the offline switch and the log.

    ``pubchempy`` (used by glycowork for the molecular properties) calls
    ``urlopen`` without a timeout, so a PubChem that accepts the connection
    and does not answer would hold the static stage for ever.  Its
    ``urlopen`` is replaced, once, by one that goes through this module.
    """
    try:
        import pubchempy
    except ImportError:
        return
    original = pubchempy.urlopen
    if getattr(original, "_glycoshape_bounded", False):
        return

    def bounded(url: Any, data: Any = None, *args: Any, **kwargs: Any) -> Any:
        from urllib.error import URLError

        lookup, subject = _LABEL.get() or ("pubchem", None)
        address = url if isinstance(url, str) else getattr(url, "full_url", "")

        def call(limit: float) -> Any:
            kwargs.setdefault("timeout", limit)
            return original(url, data, *args, **kwargs)

        try:
            return _attempt(lookup, subject, urlsplit(address).netloc, None, call)
        except LookupUnavailable as exc:
            raise URLError(str(exc)) from exc

    bounded._glycoshape_bounded = True   # type: ignore[attr-defined]
    pubchempy.urlopen = bounded
