#!/usr/bin/env python3
"""Analyse one trajectory entry on the node that produced it (no network, no services).

    python scripts/run_entry.py ENTRY OUTPUT [--workers N] [--levels 1,2] [--force]

ENTRY is an archive entry or a simulation.pdb folder; OUTPUT becomes the
result folder (result.json, run.log, input/, process/, static/). The worker
count is --workers, else SLURM_CPUS_PER_TASK. See lib/entry_run.py and the
README section "One entry on a compute node".
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib import entry_run

if __name__ == "__main__":
    raise SystemExit(entry_run.main())
