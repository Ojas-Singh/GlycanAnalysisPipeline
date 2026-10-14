"""The bulk re-analysis driver (lib/bulk_reanalysis.py, scripts/bulk_reanalyse.py) on a tiny archive, without Slurm.

The raw archive holds the synthetic glycan (published as GS01067) and an entry that is not published;
the published database has two more glycans without a raw entry.  Every stage runs here as it runs on
the cluster and the login node: plan, analyse (two shards), complete, finalize, compare, and publish
into an in-memory bucket under the lock of the entry publisher.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest

from lib import archive, bulk_reanalysis as bulk, entry_complete, entry_publish, static_publish
from lib.entry_publish import Target
from lib.storage import S3StorageBackend

import fake_services
from bundle_fixtures import make_bundle, renamed_bundle, submission_record
from publish_world import Reader, quiet, stored_manifest
from s3_fakes import FakeS3Client
from static_tree import full_bake, write_tree
from synthetic_glycan import GLYCAM_NAME

pytestmark = pytest.mark.usefixtures("no_internet")
REPO_ROOT = Path(__file__).resolve().parents[1]
GS = "GS01067"
UNPUBLISHED = "DManpa1-6DManpa1-OH"


@pytest.fixture(scope="module")
def site(tmp_path_factory, synthetic_entry):
    """A raw archive of two entries and the published database that has one of them (and two others)."""
    root = tmp_path_factory.mktemp("bulk-site")
    with pytest.MonkeyPatch.context() as patching:
        fake_services.install(patching)
        archive_dir = root / "archive"
        shutil.copytree(synthetic_entry["archive"], archive_dir / GLYCAM_NAME)
        bundle = make_bundle(root / "bundle", synthetic_entry["result"], synthetic_entry["archive"])
        other = renamed_bundle(root / "other", bundle, UNPUBLISHED)
        shutil.copytree(other / "archive", archive_dir / UNPUBLISHED)
        (archive_dir / UNPUBLISHED / "manifest.json").write_text(
            (other / "archive" / "manifest.json").read_text())
        index, problems = archive.write_index(archive_dir)
        assert not problems and index["entries"] == 2
        published = write_tree(root / "published", ("GS00001", "GS00002"))
        entry_complete.complete_entry(bundle, submission_record(GS, GLYCAM_NAME, curated={
            "common_names": ["Mannobiose"], "aliases": [], "description": "", "keywords": ["Curated:yes"]}), published)
        full_bake(published)
    return {"archive": archive_dir, "published": published}


@pytest.fixture
def run(site, tmp_path, monkeypatch):
    fake_services.install(monkeypatch)
    return tmp_path / "run"


def _cli(argv):
    import importlib.util

    spec = importlib.util.spec_from_file_location("bulk_cli", REPO_ROOT / "scripts" / "bulk_reanalyse.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.main(argv)


def test_the_whole_run_on_a_tiny_archive(site, run, tmp_path):
    plan = bulk.make_plan(site["archive"], site["published"], run, shards=2, workers=1, log=quiet)
    rows = {row["glycam_name"]: row for row in plan["entries"]}
    assert rows[GLYCAM_NAME]["glycoshape_id"] == GS and rows[GLYCAM_NAME]["shard"] in (1, 2)
    assert rows[UNPUBLISHED]["glycoshape_id"] is None and rows[UNPUBLISHED]["skip"].startswith("not published")
    assert plan["published_without_raw_entry"] == ["GS00001", "GS00002"]
    assert plan["base_global_hash"] == json.loads((site["published"] / "sync_manifest.json").read_text())["global_hash"]
    assert plan["estimate"]["entries"] == 1 and plan["estimate"]["entry_seconds"] >= 30

    for shard in (1, 2):                                   # what the two array tasks do
        summary = bulk.analyse(run, shard=shard, jobs=2, workers=1, log=quiet)
        assert summary["failed"] == []
    status = json.loads((run / "entries" / GLYCAM_NAME / "status.json").read_text())
    assert status["status"] == "ok" and status["retained"] == 480 and status["seconds"] > 0
    assert not (run / "entries" / GLYCAM_NAME / "result" / "process").exists()          # pruned
    assert (run / "entries" / GLYCAM_NAME / "analysis.log").is_file()
    assert json.loads((run / "logs" / f"analyse-{rows[GLYCAM_NAME]['shard']}.json").read_text())["ok"] == 1
    again = bulk.analyse(run, shard=rows[GLYCAM_NAME]["shard"], jobs=2, workers=1, log=quiet)
    assert again["resumed"] == 1                                                          # resume

    done = bulk.complete(run, jobs=1, log=quiet)
    assert done["ok"] == 1 and done["failed"] == [] and done["incomplete"] == []
    data = json.loads((run / "tree" / GS / "data.json").read_text())
    assert data["archetype"]["name"] == GLYCAM_NAME
    assert "Curated:yes" in data["search_meta"]["keywords"] and "Mannobiose" in data["search_meta"]["common_names"]
    assert (run / "replicas" / f"{GS}.json").is_file()
    assert bulk.complete(run, jobs=1, log=quiet)["resumed"] == 1

    final = bulk.finalize(run, log=quiet)                         # two IDs have no raw entry: kept as published
    assert final["glycans"] == 3 and final["reanalysed"] == 1
    assert final["kept_published"] == [
        {"glycoshape_id": glycan_id, "reason": "no raw entry in the archive that was analysed"}
        for glycan_id in ("GS00001", "GS00002")]
    manifest = static_publish.parse_manifest((run / "tree" / "sync_manifest.json").read_bytes(), "tree")
    assert sorted(manifest["glycans"]) == ["GS00001", "GS00002", GS]
    assert "missing_glytoucan_no_wurcs_gags.tsv" in manifest["root_files"]               # copied from the published tree

    compared = bulk.compare(run, log=quiet)
    assert compared["ok"], compared["blocking"]
    assert compared["counts"]["unchanged_folders"] == 2 and compared["counts"]["renamed"] == 0
    assert compared["new"]["glycans"] == compared["published"]["glycans"] == 3
    assert GS in compared["entries"] or compared["counts"]["unchanged_folders"] == 3
    overall = bulk.summary(run)
    assert overall["counts"]["analysed"] == 1 and overall["finalized"] and overall["compared"]


def test_publishing_the_tree_under_the_lock_and_only_onto_its_base(site, run, tmp_path):
    bulk.make_plan(site["archive"], site["published"], run, shards=1, workers=1, log=quiet)
    bulk.analyse(run, shard=1, jobs=1, workers=1, log=quiet)
    bulk.complete(run, jobs=1, log=quiet)
    bulk.finalize(run, keep_published=["all"], log=quiet)
    plan = bulk.load_plan(run)

    client = FakeS3Client("bucket")
    backend = S3StorageBackend(client, bucket="bucket", sleep=lambda seconds: None)
    static_publish.publish(site["published"], backend, "static", workers=2, log=quiet)   # the database as it is today
    target = Target(static=backend, raw=backend, workers=2)
    reader = Reader(backend, tmp_path / "website")
    reader.sync()
    assert bulk.check_base(target, plan["base_global_hash"])["safe"]

    dry = bulk.publish(run, target, plan["base_global_hash"], workers=2, dry_run=True, log=quiet)
    assert dry["dry_run"] and stored_manifest(backend)["global_hash"] == plan["base_global_hash"]
    report = bulk.publish(run, target, plan["base_global_hash"], workers=2, log=quiet)
    assert stored_manifest(backend)["global_hash"] == json.loads((run / "tree" / "sync_manifest.json").read_text())["global_hash"]
    assert reader.try_sync() is None and reader.fresh(tmp_path / "fresh").try_sync() is None
    assert report["changed_glycans"] >= 1
    keys = client.keys("bucket", "publish-state/static/")
    assert f"publish-state/static/entries/{GS}.json" in keys and f"publish-state/static/replicas/{GS}.json" in keys
    assert any(key.startswith(f"publish-state/static/history/{GS}/") and key.endswith("-bulk-reanalyse.json") for key in keys)
    assert any(key.startswith("publish-state/static/history/bulk/") for key in keys)
    assert "publish-state/static/lock.json" not in keys                                   # released

    # a copy of the old published tree is brought up to the stored database, as the base of a next run
    local = tmp_path / "local-published"
    shutil.copytree(site["published"], local)
    synced = bulk.sync_published(target, local, workers=2, records_dir=tmp_path / "records", log=quiet)
    assert synced["global_hash"] == stored_manifest(backend)["global_hash"] and synced["downloaded"] >= 1
    assert (local / "sync_manifest.json").read_bytes() == (run / "tree" / "sync_manifest.json").read_bytes()
    assert (local / GS / "data.json").read_bytes() == (run / "tree" / GS / "data.json").read_bytes()
    assert synced["records"] >= 1 and (tmp_path / "records" / f"{GS}.json").is_file()
    assert (site["published"] / "sync_manifest.json").read_bytes() != (local / "sync_manifest.json").read_bytes()

    # the stored database moved on: the tree is not published onto it
    report = bulk.check_base(target, plan["base_global_hash"])
    assert not report["safe"] and "not the base of the run" in report["problems"][0]
    with pytest.raises(bulk.BulkError):
        bulk.publish(run, target, plan["base_global_hash"], workers=2, log=quiet)
    # nor while another publisher holds the lock
    store = entry_publish.Store(backend, entry_publish.new_stats())
    entry_publish.acquire_lock(target, store, "elsewhere:1:abcd", "publish", "GS01070", ttl=600, log=quiet)
    with pytest.raises(entry_publish.LockHeld):
        bulk.publish(run, target, stored_manifest(backend)["global_hash"], workers=2, log=quiet)


def test_the_command_line_of_the_driver(site, run, tmp_path):
    fixes = tmp_path / "fixes.json"
    fixes.write_text(json.dumps({GS: {"exclude": ["run_all"]}, "GS09999": {"exclude": ["x"]}}))
    module, status = _cli(["plan", "--archive-dir", str(site["archive"]), "--published-dir", str(site["published"]),
                           "--run-dir", str(run), "--shards", "1", "--fixes", str(fixes)])
    assert status == 0
    plan = bulk.load_plan(run)
    assert plan["unused_fixes"] == ["GS09999"]
    # excluding the only replica of an entry is an analysis that fails: reported per entry, exit status 1
    _, status = _cli(["analyse", "--run-dir", str(run), "--shard", "1", "--jobs", "1", "--workers", "1"])
    assert status == 1
    failed = json.loads((run / "logs" / "analyse-1.json").read_text())["failed"]
    assert failed and failed[0]["glycam_name"] == GLYCAM_NAME
    _, status = _cli(["summary", "--run-dir", str(run)])
    assert status == 1
    # the run goes on: the entry that failed keeps its published version, and the summaries say why
    _, status = _cli(["complete", "--run-dir", str(run), "--jobs", "1"])
    assert status == 0
    _, status = _cli(["finalize", "--run-dir", str(run)])
    assert status == 0
    kept = {item["glycoshape_id"]: item["reason"]
            for item in json.loads((run / "logs" / "finalize.json").read_text())["kept_published"]}
    assert sorted(kept) == ["GS00001", "GS00002", GS] and kept[GS].startswith("analysis failed: ")
    published = static_publish.parse_manifest((site["published"] / "sync_manifest.json").read_bytes(), "published")
    new = static_publish.parse_manifest((run / "tree" / "sync_manifest.json").read_bytes(), "tree")
    assert new["glycans"][GS]["hash"] == published["glycans"][GS]["hash"]          # unchanged, not dropped
    _, status = _cli(["compare", "--run-dir", str(run)])
    assert status == 0
    compared = json.loads((run / "compare.json").read_text())
    assert compared["counts"]["unchanged_folders"] == 3 and len(compared["kept_published"]) == 3
    _, status = _cli(["finalize", "--run-dir", str(run / "nowhere")])
    assert status == 2


def test_an_entry_that_was_not_analysed_stops_the_finalization(site, run):
    bulk.make_plan(site["archive"], site["published"], run, shards=1, workers=1, log=quiet)
    with pytest.raises(bulk.BulkError) as stopped:
        bulk.finalize(run, log=quiet)
    assert "not analysed yet" in str(stopped.value) and GS in str(stopped.value)
    final = bulk.finalize(run, keep_published=[GS], log=quiet)
    assert {item["glycoshape_id"] for item in final["kept_published"]} == {"GS00001", "GS00002", GS}


def test_the_slurm_script_runs_a_shard_without_slurm(site, run, tmp_path):
    import subprocess
    import sys

    bulk.make_plan(site["archive"], site["published"], run, shards=1, workers=1, log=quiet)
    environment = dict(os.environ, RUN_DIR=str(run), PYTHON=sys.executable, SLURM_SUBMIT_DIR=str(REPO_ROOT),
                       SLURM_CPUS_PER_TASK="2", SLURM_ARRAY_TASK_ID="1", SLURM_ARRAY_TASK_COUNT="1", JOBS="1",
                       WORKERS="1", MEMORY_GB="8")
    done = subprocess.run(["bash", str(REPO_ROOT / "slurm" / "bulk_reanalyse_amd.sh")], env=environment,
                          capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "task 1/1" in done.stdout and json.loads((run / "logs" / "analyse-1.json").read_text())["ok"] == 1
    environment.update(JOBS="4")                                  # 4 x 1 > 2 cores: refused before anything runs
    refused = subprocess.run(["bash", str(REPO_ROOT / "slurm" / "bulk_reanalyse_amd.sh")], env=environment,
                             capture_output=True, text=True, check=False)
    assert refused.returncode == 2 and "exceeds the 2 allocated cores" in refused.stderr
