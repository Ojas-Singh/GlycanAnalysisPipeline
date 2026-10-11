#!/usr/bin/env python3
"""Turn the dry trajectories of a simulation run into one entry bundle, on the node (no network, no services).

    python scripts/make_entry.py --glycam-name NAME --workers N --output BUNDLE.tar REPLICA [REPLICA ...]
                                 [--extend ENTRY] [--levels 1,2] [--info info.json] [--frame-stride 5] ...

REPLICA is a replica folder (glycan.dry.pdb + glycan.dry.mol2) or a PDB file. The replicas become one
archive entry (each a segment), the entry is analysed, and archive entry, static-ready files, result.json
and run.log are written as one tar file with bundle.json first. --extend ENTRY puts the frames of an
existing entry before the replicas. See lib/entry_make.py and the README section "A run on the cluster".
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib import entry_make

if __name__ == "__main__":
    raise SystemExit(entry_make.main())
