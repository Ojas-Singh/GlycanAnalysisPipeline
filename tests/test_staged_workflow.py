"""Tests of how the staged workflow picks its storage: the shell helper and main.py."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SELECT = "from lib import config; print(config.storage_backend)"
PAR_URL = "https://objectstorage.example/p/token/n/ns/b/bucket/o/"


def clean_env(tmp_path, **extra):
    """The environment without any GLYCOSHAPE setting, and with every default path of main.py in tmp_path."""
    env = {name: value for name, value in os.environ.items() if not name.startswith(("GLYCOSHAPE_", "POCKETBASE_"))}
    env.update(
        GLYCOSHAPE_GAP_LOG=str(tmp_path / "run.log"),
        GLYCOSHAPE_RDF_DIR=str(tmp_path / "rdf"),
        GLYCOSHAPE_INVENTORY_PATH=str(tmp_path / "inventory.csv"),
        **extra,
    )
    return env


@pytest.mark.parametrize("settings, expected", [
    ({}, "local"),
    ({"GLYCOSHAPE_ORACLE_PAR_URL": PAR_URL}, "oracle"),
    ({"GLYCOSHAPE_ORACLE_PAR_URL": PAR_URL, "GLYCOSHAPE_STORAGE": "s3"}, "s3"),
    ({"GLYCOSHAPE_ORACLE_PAR_URL": PAR_URL, "GLYCOSHAPE_STORAGE": "local"}, "local"),
    ({"GLYCOSHAPE_STORAGE": "S3"}, "s3"),
])
def test_the_selection_the_shell_scripts_read(tmp_path, settings, expected):
    run = subprocess.run([sys.executable, "-c", SELECT], capture_output=True, text=True, cwd=REPO_ROOT,
                         env=clean_env(tmp_path, **settings))
    assert run.returncode == 0 and run.stdout.strip() == expected, run.stderr
    assert "STORAGE=\"$(" in (REPO_ROOT / "scripts" / "continue_rebuild_after_slurm.sh").read_text()
    for script in ("run_staged_pipeline.sh", "continue_rebuild_after_slurm.sh"):
        assert f"-c '{SELECT}'" in (REPO_ROOT / "scripts" / script).read_text()


@pytest.fixture
def staged(tmp_path):
    """A checkout with the helper script and a stand-in for .venv/bin/python that records its calls."""
    (tmp_path / "scripts").mkdir()
    shutil.copy(REPO_ROOT / "scripts" / "run_staged_pipeline.sh", tmp_path / "scripts")
    python = tmp_path / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(
        '#!/usr/bin/env bash\n'
        'if [[ "$1" != "-c" ]]; then echo "$*" >> calls.log; exit 0; fi\n'
        'if [[ "${FAKE_STORAGE}" == "invalid" ]]; then echo "ValueError: Invalid GLYCOSHAPE_STORAGE" >&2; exit 1; fi\n'
        'echo "${FAKE_STORAGE}"\n')
    python.chmod(0o755)

    def run(storage, *arguments, **env):
        (tmp_path / "calls.log").unlink(missing_ok=True)
        done = subprocess.run(
            ["bash", str(tmp_path / "scripts" / "run_staged_pipeline.sh"), *arguments],
            capture_output=True, text=True, cwd="/", env={**os.environ, "FAKE_STORAGE": storage, **env})
        calls = (tmp_path / "calls.log").read_text().splitlines() if (tmp_path / "calls.log").exists() else []
        return done.returncode, calls

    return run


def test_the_helper_uses_the_scripts_of_the_selected_storage(staged):
    build = "main.py --mode static-only --process-dir process --output-dir static"
    assert staged("s3", "download", "--glycan", "DGlcpa1-OH") == (0, [
        "scripts/archive_s3.py download --archive-dir data --glycan DGlcpa1-OH"])
    assert staged("s3", "publish", "--dry-run") == (0, [build, "scripts/static_s3.py publish --static-dir static --dry-run"])
    assert staged("s3", "publish", FORCE_UPDATE="1") == (0, [
        build + " --update", "scripts/static_s3.py publish --static-dir static"])

    # with Oracle, and without any storage, the commands are the ones the helper always ran
    for storage in ("oracle", "local"):
        assert staged(storage, "download", "--glycan", "DGlcpa1-OH") == (0, [
            "scripts/download_data_from_oracle.py --data-dir data --glycan DGlcpa1-OH"])
        assert staged(storage, "publish", "--skip-existing") == (0, [
            build, "scripts/upload_static_to_oracle.py --static-dir static --skip-existing"])
    assert staged("s3", "nonsense") == (2, [])
    # a storage setting that cannot be read stops the helper; it does not fall back to the other storage
    assert staged("invalid", "download") == (1, []) and staged("invalid", "publish") == (1, [])


def run_main(tmp_path, *arguments, **settings):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "main.py"), *arguments,
         "--data-dir", str(tmp_path / "data"), "--process-dir", str(tmp_path / "process"),
         "--output-dir", str(tmp_path / "static")],
        capture_output=True, text=True, cwd=tmp_path, env=clean_env(tmp_path, **settings))


def test_main_runs_the_local_modes_with_s3_selected_and_refuses_the_others(tmp_path):
    # GLYCOSHAPE_S3_* are not set: any attempt to open the S3 storage would fail
    run = run_main(tmp_path, "--mode", "process-only", "--status", GLYCOSHAPE_STORAGE="s3")
    assert run.returncode == 0, run.stdout + run.stderr
    assert "Storage mode: local" in run.stderr and "No glycans found" in run.stdout

    refused = tmp_path / "refused"
    refused.mkdir()
    for mode in ("incremental", "refresh-static", "fresh"):
        run = run_main(refused, "--mode", mode, GLYCOSHAPE_STORAGE="s3")
        assert run.returncode == 2, run.stdout + run.stderr
        assert "GLYCOSHAPE_STORAGE=s3 supports --mode process-only and --mode static-only" in run.stderr
    assert [path.name for path in refused.iterdir()] == ["run.log"]   # refused before anything was set up
