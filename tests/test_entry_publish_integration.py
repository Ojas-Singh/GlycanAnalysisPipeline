"""Publishing one entry against a real S3-compatible server on this machine (MinIO, moto).

Skipped when there is none (see tests/conftest.py).  What the in-memory client cannot show: the real
client and its errors, conditional writes as a server answers them, the command-line tool end to end
in processes of its own, two of them at once, and that no output carries a credential.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest

from lib import entry_publish, static_publish
from lib.entry_publish import PublishEntryError, Target

import fake_services
import publish_world
from bundle_fixtures import renamed_bundle, submission_record
from publish_world import PUBLISHED, Reader, quiet, stored_manifest
from s3_fakes import Interrupted
from synthetic_glycan import GLYCAM_NAME

pytestmark = pytest.mark.usefixtures("no_internet")
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "publish_entry.py"
GS, SECOND_GS, SECOND_NAME = "GS01067", "GS01068", "DManpa1-2DManpb1-OH"
STATE = "publish-state/static"


@pytest.fixture
def published(s3_bucket, tmp_path, monkeypatch):
    """A published database of three glycans and a raw archive in a bucket of the test server."""
    backend, client, bucket = s3_bucket
    fake_services.install(monkeypatch)
    publish_world.build(tmp_path / "published", backend, raw_source=publish_world.other_archive(tmp_path / "stored-archive"))
    target = Target(static=backend, raw=backend, workers=4)
    return {"backend": backend, "bucket": bucket, "target": target, "reader": Reader(backend, tmp_path / "website")}


def test_publish_replace_and_withdraw_on_a_real_server(published, bundle_dir, tmp_path, s3_server):
    backend, target, reader = published["backend"], published["target"], published["reader"]
    assert reader.try_sync() is None and reader.glycans() == sorted(PUBLISHED)
    record = submission_record(GS, GLYCAM_NAME)

    dry = entry_publish.publish_entry("publish", bundle_dir, record, target, dry_run=True, work_dir=tmp_path / "work", log=quiet)
    assert dry["ok"] and stored_manifest(backend)["global_hash"] == reader.global_hash
    assert [item["key"] for item in backend.list_objects("publish-")] == []          # nothing was written

    report = entry_publish.publish_entry("publish", bundle_dir, record, target, work_dir=tmp_path / "work",
                                         expect_hash=dry["entry"]["hash"], log=quiet)
    assert report["status"] == "published"
    assert reader.try_sync() is None and reader.glycans() == sorted(PUBLISHED + (GS,))
    assert reader.fresh(tmp_path / "behind").try_sync() is None
    assert static_publish.check_published(backend, "static", log=quiet)["ok"]
    assert [item["key"] for item in backend.list_objects("publish-")] == [f"{STATE}/entries/{GS}.json"]
    assert entry_publish.publish_entry("publish", bundle_dir, record, target, work_dir=tmp_path / "work", log=quiet)["status"] == "already-published"
    # what the server makes of a conditional write is found out, not assumed
    honoured = entry_publish.conditional_writes(target, entry_publish.Store(backend, entry_publish.new_stats()))
    print(f"{s3_server['kind']}: conditional writes honoured: {honoured}")
    assert [item["key"] for item in backend.list_objects(f"{STATE}/probe-")] == []

    changed = submission_record(GS, GLYCAM_NAME, curated={"description": "Replaced on a real server."})
    report = entry_publish.publish_entry("replace", bundle_dir, changed, target, work_dir=tmp_path / "work-replace", log=quiet)
    assert report["status"] == "published" and len(report["static"]["stage_and_switch"]) == 5
    assert reader.try_sync() is None
    assert json.loads((reader.folder / GS / "data.json").read_text())["search_meta"]["description"] == "Replaced on a real server."

    report = entry_publish.withdraw_entry(GS, target, tmp_path / "work-withdraw", reason="test", log=quiet)
    assert report["status"] == "withdrawn"
    assert reader.try_sync() is None and reader.glycans() == sorted(PUBLISHED)
    assert [item["key"] for item in backend.list_objects(f"static/{GS}/")] == []
    assert backend.exists(f"raw/{GLYCAM_NAME}/manifest.json")                         # the raw entry stays
    assert static_publish.check_published(backend, "static", log=quiet)["ok"]
    assert entry_publish.status(target, GS, check=True)["entry"]["state"] == "withdrawn"


def test_an_interrupted_switch_on_a_real_server_is_completed_by_recover(published, bundle_dir, tmp_path):
    backend, target, reader = published["backend"], published["target"], published["reader"]
    reader.sync()
    old_hash = reader.global_hash

    def stop_in_the_window(name):
        if name == "switched:static/GLYCOSHAPE.json":
            raise Interrupted("stopped after the first file was switched")

    with pytest.raises(PublishEntryError, match="stopped while the changed files were switched"):
        entry_publish.publish_entry("publish", bundle_dir, submission_record(GS, GLYCAM_NAME), target,
                                    work_dir=tmp_path / "work", checkpoint=stop_in_the_window, log=quiet)
    assert stored_manifest(backend)["global_hash"] == old_hash
    assert reader.try_sync() is None and reader.global_hash == old_hash               # up to date: not affected
    assert "download verification failed for GLYCOSHAPE.json" in reader.fresh(tmp_path / "behind").try_sync()
    assert entry_publish.status(target)["pending"]["glycoshape_id"] == GS
    assert entry_publish.recover(target, log=quiet)["manifest_written"]
    assert reader.try_sync() is None and reader.glycans() == sorted(PUBLISHED + (GS,))
    assert reader.fresh(tmp_path / "behind").try_sync() is None
    assert static_publish.check_published(backend, "static", log=quiet)["ok"]
    assert [item["key"] for item in backend.list_objects("publish-")] == [f"{STATE}/entries/{GS}.json"]


# ---------------------------------------------------------------------------
# The command-line tool, in processes of its own
# ---------------------------------------------------------------------------
_RAN_AGAINST = {}


def once(name, s3_server):
    """The command-line tests start a dozen processes each: one server is enough for them."""
    if name in _RAN_AGAINST:
        pytest.skip(f"ran against the {_RAN_AGAINST[name]} server")
    _RAN_AGAINST[name] = s3_server["kind"]


class Tool:
    """``scripts/publish_entry.py`` with the settings of the test server in its environment."""

    def __init__(self, s3_server, bucket, tmp_path):
        self.secrets = [s3_server["secret_key"], s3_server["access_key"]]
        self.outputs = []
        self.environment = {key: value for key, value in os.environ.items()
                            if not key.startswith(("GLYCOSHAPE_", "POCKETBASE", "AWS_"))}
        self.environment.update({
            "GLYCOSHAPE_S3_ENDPOINT": s3_server["endpoint"], "GLYCOSHAPE_S3_REGION": s3_server["region"],
            "GLYCOSHAPE_S3_ACCESS_KEY_ID": s3_server["access_key"],
            "GLYCOSHAPE_S3_SECRET_ACCESS_KEY": s3_server["secret_key"],
            "GLYCOSHAPE_S3_STATIC_BUCKET": bucket, "GLYCOSHAPE_S3_RAW_BUCKET": bucket,
            "GLYCOSHAPE_S3_MAX_ATTEMPTS": "3", "GLYCOSHAPE_CRABWURCS_BIN": str(tmp_path / "no-crabwurcs"),
            # a machine without network would behave the same; the proxies make sure nothing leaves this one
            "http_proxy": "http://127.0.0.1:9", "https_proxy": "http://127.0.0.1:9", "no_proxy": "127.0.0.1,localhost",
        })
        self.cwd = tmp_path

    def command(self, *arguments):
        return [sys.executable, str(SCRIPT), *[str(argument) for argument in arguments]]

    def run(self, *arguments, expect=0):
        done = subprocess.run(self.command(*arguments), env=self.environment, cwd=str(self.cwd),
                              capture_output=True, text=True, timeout=600)
        self.outputs += [done.stdout, done.stderr]
        assert done.returncode == expect, f"exit {done.returncode}\n{done.stdout[-3000:]}\n{done.stderr[-3000:]}"
        return done

    def assert_no_credential_anywhere(self, *folders):
        texts = list(self.outputs)
        for folder in folders:
            texts += [path.read_bytes().decode("utf-8", "replace") for path in Path(folder).rglob("*")
                      if path.is_file() and path.stat().st_size < 5_000_000]
        assert len(texts) > 10
        for text in texts:
            for secret in self.secrets:
                assert secret not in text


def test_the_command_line_tool_end_to_end(published, bundle_dir, tmp_path, s3_server):
    once("tool", s3_server)
    backend, reader = published["backend"], published["reader"]
    tool = Tool(s3_server, published["bucket"], tmp_path)
    submission = tmp_path / "submission.json"
    submission.write_text(json.dumps(submission_record(GS, GLYCAM_NAME)))
    work, reports = tmp_path / "work", tmp_path / "reports"
    reports.mkdir()
    common = ("--bundle", bundle_dir, "--submission", submission, "--work-dir", work, "--offline", "--allow-incomplete")

    # without --allow-incomplete an entry that could not be identified offline is refused: exit status 2
    refused = tool.run("publish", "--bundle", bundle_dir, "--submission", submission, "--work-dir", work, "--offline",
                       "--dry-run", expect=2)
    assert "REFUSED the entry is not complete" in refused.stderr
    dry = tool.run("publish", *common, "--dry-run", "--report", reports / "dry.json")
    assert "dry run: nothing was written to the storage; this entry would be published" in dry.stdout
    assert [item["key"] for item in backend.list_objects("publish-")] == []
    assert not backend.exists(f"static/{GS}/data.json") and not backend.exists(f"raw/{GLYCAM_NAME}/manifest.json")
    approved = json.loads((reports / "dry.json").read_text())["entry"]["hash"]
    assert approved in dry.stdout

    done = tool.run("publish", *common, "--expect-hash", approved, "--report", reports / "publish.json")
    assert f"publish: published {GS}" in done.stdout and "transfers: up" in done.stdout
    assert reader.try_sync() is None and reader.glycans() == sorted(PUBLISHED + (GS,))
    again = tool.run("publish", *common)
    assert f"publish: already-published {GS}" in again.stdout
    status = tool.run("status", "--id", GS, "--check", "--report", reports / "status.json")
    assert f"{GS}: published; 24 object(s) stored" in status.stdout and "pending publication: none" in status.stdout
    # what the entry was published without, for a later refresh: offline, no service was asked
    assert "lookups without an answer at publication: " in status.stdout and "pubchem_properties" in status.stdout
    # the same glycan under another ID: refused, nothing written
    other = tmp_path / "other.json"
    other.write_text(json.dumps(submission_record("GS01099", GLYCAM_NAME)))
    refused = tool.run("publish", "--bundle", bundle_dir, "--submission", other, "--work-dir", tmp_path / "work-other",
                       "--offline", "--allow-incomplete", expect=2)
    assert f"REFUSED this glycan is published as {GS}" in refused.stderr

    # a lock of a publisher on another machine: exit status 3
    lock = entry_publish.acquire_lock(published["target"], entry_publish.Store(backend, entry_publish.new_stats()),
                                      "elsewhere:1:abcdef12", "publish", "GS09999", ttl=600, log=quiet)
    backend.write_binary(f"{STATE}/lock.json", json.dumps({**lock, "host": "elsewhere"}).encode())
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(submission_record(GS, GLYCAM_NAME, curated={"description": "Replaced by the tool."})))
    replace = ("replace", "--bundle", bundle_dir, "--submission", changed, "--work-dir", tmp_path / "work-replace",
               "--offline", "--allow-incomplete")
    locked = tool.run(*replace, expect=3)
    assert "the database is locked by elsewhere:1:abcdef12" in locked.stderr
    done = tool.run(*replace, "--break-lock")
    assert f"replace: published {GS}" in done.stdout and "taking over the lock" in done.stdout
    assert reader.try_sync() is None
    assert json.loads((reader.folder / GS / "data.json").read_text())["search_meta"]["description"] == "Replaced by the tool."

    done = tool.run("withdraw", "--id", GS, "--glycam-name", GLYCAM_NAME, "--work-dir", tmp_path / "work-withdraw",
                    "--offline", "--reason", "test")
    assert f"withdraw: withdrawn {GS}" in done.stdout
    assert reader.try_sync() is None and reader.glycans() == sorted(PUBLISHED)
    status = tool.run("status", "--id", GS)
    assert f"{GS}: withdrawn; 0 object(s) stored" in status.stdout and "raw entry stored" in status.stdout
    assert "nothing to recover" in tool.run("recover").stdout
    assert static_publish.check_published(backend, "static", log=quiet)["ok"]
    tool.assert_no_credential_anywhere(work, tmp_path / "work-replace", tmp_path / "work-withdraw", reports)


#: A stand-in for the crabwurcs executable: ``render`` writes an SVG, ``convert ... --to wurcs`` answers with a WURCS.
RENDERER = """#!/bin/sh
name=$(cat)
if [ "$1" = render ]; then
    printf '<svg xmlns="http://www.w3.org/2000/svg"><!-- %s --></svg>' "$name" > "$5"
elif [ "$1" = convert ] && [ "$5" = wurcs ]; then
    printf 'WURCS=2.0/stand-in/%s' "$name"
else
    exit 1
fi
"""


def test_the_tool_completes_a_refused_entry_again_once_the_renderer_is_there(published, bundle_dir, tmp_path, s3_server):
    """As in the first trial: crabwurcs was missing, it was installed, and the same command ran with the same work folder."""
    once("renderer", s3_server)
    tool = Tool(s3_server, published["bucket"], tmp_path)
    submission = tmp_path / "submission.json"
    submission.write_text(json.dumps(submission_record(GS, GLYCAM_NAME)))
    command = ("publish", "--bundle", bundle_dir, "--submission", submission, "--work-dir", tmp_path / "work",
               "--offline", "--dry-run")
    refused = tool.run(*command, expect=2)
    assert "REFUSED the entry is not complete: snfg.svg is missing (the SNFG renderer did not run)" in refused.stderr
    assert not (tmp_path / "work" / "static" / GS / "snfg.svg").exists()

    renderer = tmp_path / "crabwurcs"
    renderer.write_text(RENDERER)
    renderer.chmod(0o755)
    tool.environment["GLYCOSHAPE_CRABWURCS_BIN"] = str(renderer)
    dry = tool.run(*command)
    assert "was not complete (snfg.svg is missing" in dry.stdout and "completing it again" in dry.stdout
    assert "dry run: nothing was written to the storage; this entry would be published" in dry.stdout
    assert (tmp_path / "work" / "static" / GS / "snfg.svg").is_file()
    again = tool.run(*command)                                  # complete now: the folder is kept
    assert "using the folder completed at" in again.stdout and "this entry would be published" in again.stdout

    # a bundle that does not state its length, published with --id: said in words, and a warning
    by_id = tool.run("publish", "--bundle", bundle_dir, "--id", GS, "--work-dir", tmp_path / "work-id", "--offline", "--dry-run")
    assert "  simulation: length not given, " in by_id.stdout and "None us" not in by_id.stdout
    assert "WARNING simulation.simulation_length_ns is not given" in by_id.stdout
    assert [item["key"] for item in published["backend"].list_objects("publish-")] == []          # dry runs: nothing written


def test_two_processes_publishing_at_once_do_not_lose_an_entry(published, bundle_dir, tmp_path, s3_server):
    once("two processes", s3_server)
    backend, reader = published["backend"], published["reader"]
    tool = Tool(s3_server, published["bucket"], tmp_path)
    second_bundle = renamed_bundle(tmp_path / "second-bundle", bundle_dir, SECOND_NAME)
    jobs = {}
    for name, bundle, glycan_id, glycam_name in (("first", bundle_dir, GS, GLYCAM_NAME), ("second", second_bundle, SECOND_GS, SECOND_NAME)):
        submission = tmp_path / f"{name}.json"
        submission.write_text(json.dumps(submission_record(glycan_id, glycam_name)))
        jobs[name] = ("publish", "--bundle", bundle, "--submission", submission, "--work-dir", tmp_path / f"work-{name}",
                      "--offline", "--allow-incomplete")
    running = {name: subprocess.Popen(tool.command(*arguments), env=tool.environment, cwd=str(tmp_path),
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
               for name, arguments in jobs.items()}
    codes = {}
    for name, process in running.items():
        out, err = process.communicate(timeout=600)
        tool.outputs += [out, err]
        codes[name] = process.returncode
        assert process.returncode in (0, 1, 3), f"{name}: exit {process.returncode}\n{out[-2000:]}\n{err[-2000:]}"
    # one that was held back by the lock (3), or stopped before it switched anything (1), runs again
    for name, code in codes.items():
        if code != 0:
            tool.run(*jobs[name])
    print(f"{s3_server['kind']}: exit statuses of the two publishers at once: {codes}")
    assert sorted(stored_manifest(backend)["glycans"]) == sorted(PUBLISHED + (GS, SECOND_GS))
    assert reader.try_sync() is None and reader.glycans() == sorted(PUBLISHED + (GS, SECOND_GS))
    assert reader.fresh(tmp_path / "behind").try_sync() is None
    assert static_publish.check_published(backend, "static", log=quiet)["ok"]
    assert sorted(item["key"] for item in backend.list_objects("publish-")) == [
        f"{STATE}/entries/{GS}.json", f"{STATE}/entries/{SECOND_GS}.json"]
    tool.assert_no_credential_anywhere(tmp_path / "work-first", tmp_path / "work-second")
