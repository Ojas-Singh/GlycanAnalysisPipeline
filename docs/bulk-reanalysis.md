# Bulk re-analysis of the published database

Re-analyse every entry of the raw archive with the current GAP, derive the whole static database (Levels 1 and 2,
`replicas.json` for the administrators) from it and the curated metadata of every GlycoShape ID, check it against
the published one, and publish it to R2 in one publication. The analysis runs on Iridis X as `os1e24`; everything
that needs the network or the R2 keys runs on the Spark (the keys never go to Iridis).

Driver: `scripts/bulk_reanalyse.py` (`lib/bulk_reanalysis.py`); Slurm script: `slurm/bulk_reanalyse_amd.sh`.
Every stage can be run again and skips what is done; every stage writes a summary to `<run>/logs/`.

| Stage | Where | What | Measured / expected |
|---|---|---|---|
| `sync-published` | Spark | the local published tree = what R2 serves (only when something was published since the copy) | |
| `plan` | Spark | archive entry -> published ID, cost, memory, shards | 19 s for the 1,004 entries |
| `check-base` | Spark | R2 still holds the database the run starts from | seconds |
| `analyse` | Iridis X, `amd` nodes | `run_entry.py` per entry, no network | 11.6 entry-hours at 16 workers: about 1 to 1.5 h on 4 nodes, 4 to 5 h on one |
| `complete` | Spark (network) | the published folder of every entry, with its curated metadata | about 6 s per entry offline; with the web lookups 10 to 30 s; 8 at a time |
| `finalize` | Spark | root files, `sync_manifest.json` last | 100 s, 1 GB for the 1,001 entries |
| `compare` | Spark | the new tree against the published one | 16 s |
| `publish` | Spark | one publication under the lock of the entry publisher | about 20 min upload, then the switch |

## Decision: one publication, not 1,001 replaces

The tree is published in **one** `static_publish` publication (the code of `scripts/static_s3.py publish`), wrapped
by `bulk_reanalyse.py publish`:

* it takes the lock of `publish_entry.py` (`publish-state/static/lock.json`), so the incremental publisher (the
  worker) waits (exit status 3, it retries) and cannot publish in between;
* with the lock held it checks that the stored `static/sync_manifest.json` has the `global_hash` the run started from
  (`--expect-base`, the `base_global_hash` of `plan.json`) and that no publication is pending; otherwise nothing is
  written;
* afterwards it writes, for every glycan whose folder changed, the record of the entry
  (`publish-state/static/entries/GSxxxxx.json`), a version in its history
  (`publish-state/static/history/GSxxxxx/<global hash>-bulk-reanalyse.json`) and the administrators' copy of
  `replicas.json`, and one record of the whole run (`publish-state/static/history/bulk/<time>.json`, with the list of
  entries kept as published and why).

Per-entry `replace` of all 1,001 entries is the alternative and is not used: every replace derives and uploads the
root files again (`GLYCOSHAPE.json`, `GlycoShape.zip`, the RDF: about 140 MB), so 1,001 replaces upload about
140 GB, make 1,001 versions of the database that the website downloads one after the other (its root files each
time), and take 6 to 17 h. One publication uploads the changed files once (about 23 GB) and the website
synchronises once.

**When a bulk publication is safe, and how to verify it.** It replaces the whole `static/` prefix by the tree, so it
is safe only while R2 holds exactly the database the run started from: any publication since (a new GOTW entry, a
`replace`, an `update-metadata`, a `withdraw`) changes the stored `global_hash`, and the tree does not have it.
`bulk_reanalyse.py check-base` verifies this (stored hash, pending publication, lock, and with `--archive-dir` that
`raw/index.json` is the index of the analysed archive), and `publish` checks it again under the lock. If it fails:

```bash
bulk_reanalyse.py sync-published --published-dir ~/gap-work/static-l12 --records-dir ~/gap-work/bulk-2026-10/records
bulk_reanalyse.py plan ...           # the same command: the new base; analysed entries are resumed
bulk_reanalyse.py complete --run-dir R --curation-dir ~/gap-work/bulk-2026-10/records ...
bulk_reanalyse.py finalize ... ; bulk_reanalyse.py compare ... ; bulk_reanalyse.py publish --expect-base <new base>
```

An entry published one at a time after the archive copy has no raw entry in it and is kept as published (it was
analysed by the current code anyway). `sync-published` replaces files by rename, so it never writes into a file
that is a hard link of another tree (`~/gap-work/static-l12` shares files with `~/gap-work/prod-static`).

**Raw side.** Nothing of `raw/` changes in a bulk run. Fixes that rename an entry (the anomer that was simulated)
are not part of it: they are `publish_entry.py reanalyse --simulated-anomer` per entry afterwards (see below).

## Entries that fail

An entry whose analysis fails (for example GS00276, whose runs have coordinates that do not match its topology) or
whose completion fails is **not** dropped: the driver records the failure with its reason
(`<run>/entries/<folder>/status.json` or `complete.json`), goes on with the others, and `finalize` puts the
**published folder of that entry, unchanged**, into the tree. `logs/finalize.json` lists every such entry under
`kept_published` with the reason; `compare.json` repeats the list; the record of the publication in
`publish-state/static/history/bulk/` keeps it for the admin page. The same holds for a published ID without a raw
entry. An entry that was never analysed (a shard that did not run) stops `finalize` instead, unless
`--keep-published ID` (or `all`) says to keep it.

## Before the run

* **Code.** A GAP commit on `main` with the fixes the run is for (for example PR #9, the torsion definitions). Pin
  it: the run records `code_sha256` in every `status.json`, and an entry is analysed again when the code changes.
* **crabWURCS.** The completion writes WURCS, IUPAC and the SNFG images with crabWURCS. The problem list (section 4)
  says that 0.3.2 regenerates the wrong alpha/beta WURCS of 25 furanose entries and of GS00237/238/314/386/667:
  wait for that release and put it on the machine that runs `complete` (`GLYCOSHAPE_CRABWURCS_BIN`).
* **Raw archive.** Use `~/gap-work/archive` on the Spark: all 1,004 entries, the current converter, identical to
  R2 `raw/` (`archive_s3.py compare` was clean on 2026-10-11; run it again read-only:
  `scripts/archive_s3.py compare --archive-dir ~/gap-work/archive`). Do **not** use
  `/scratch/os1e24/gap-next/archive`: it lacks the three repaired entries (GS00968, GS00055, GS00741) and was written
  by the older converter code. Copy it to Iridis (40 GB, about 10 min):
  `rsync -a --info=progress2 ~/gap-work/archive/ iridis-x1:/scratch/os1e24/gap-bulk/archive/`, then on the login
  node `convert_data_to_archive.py --verify-only --archive-dir /scratch/os1e24/gap-bulk/archive --workers 8`
  (files against their manifests).
* **Published tree.** `~/gap-work/static-l12` is what was uploaded to R2 on 2026-10-11 (1,001 glycans,
  `global_hash 219e9619...`). `check-base` says whether R2 still holds it; if not, `sync-published` first.

## Steps

On the **Spark** (from the checkout of the pinned commit, with `~/envs/gap-r2` or the GAP venv):

```bash
R=~/gap-work/bulk-2026-10/run
python scripts/bulk_reanalyse.py plan --archive-dir ~/gap-work/archive --published-dir ~/gap-work/static-l12 \
    --run-dir $R --shards 4 [--fixes fixes.json]
set -a; . ~/gap-work/r2-test/r2env.sh; set +a          # read-only use here
python scripts/bulk_reanalyse.py check-base --run-dir $R --archive-dir ~/gap-work/archive
rsync -a $R/plan.json iridis-x1:/scratch/os1e24/gap-bulk/run/
```

`plan` prints the entries it skips: the three raw entries without a published ID (GS01064 to GS01066, waiting for
publication with IDs from the registry) and entries whose fix renames them. `fixes.json` maps an ID to
`{"exclude": ["run_2"]}` (replica names as in `replicas.json`) for entries whose replicas should be left out.
The plan of 2026-10-14 on the real archive: 1,001 entries, median 35 s, 90th percentile 59 s, longest 220 s (one
entry, 16 workers); memory median 3.9 GB, largest 33 GB.

On the **Iridis login node**, as `os1e24` (the home quota is full: keep the uv cache and Python on scratch):

```bash
cd /scratch/os1e24/gap-bulk
git clone https://github.com/Ojas-Singh/GlycanAnalysisPipeline && cd GlycanAnalysisPipeline && git checkout <commit>
export UV_CACHE_DIR=/scratch/os1e24/gap-bulk/uv-cache UV_PYTHON_INSTALL_DIR=/scratch/os1e24/gap-bulk/uv-python
uv venv --python 3.10 .venv && uv pip install --python .venv/bin/python -r requirements-node.txt
mkdir -p logs/slurm
RUN_DIR=/scratch/os1e24/gap-bulk/run ARCHIVE_DIR=/scratch/os1e24/gap-bulk/archive \
    sbatch --array=1-4 slurm/bulk_reanalyse_amd.sh
```

The array has as many tasks as the plan has shards; each task is one 64-core `amd` node, 4 entries at a time with
16 workers each (16 workers were faster than 64 per entry in job 1797685), within the memory of the node
(`MEMORY_GB`, default all but 16 GB; an entry waits until its estimate fits). Per entry:
`run/entries/<folder>/analysis.log` and `status.json`; per task: `run/logs/analyse-<task>.json`. After a failure,
a timeout or a pre-emption submit the same command: done entries are skipped. One entry again:
`ENTRY="GS00276" sbatch --array=1 slurm/bulk_reanalyse_amd.sh`.

```bash
.venv/bin/python scripts/bulk_reanalyse.py summary --run-dir /scratch/os1e24/gap-bulk/run
```

Back on the **Spark**: copy the results (the static-ready halves and the logs; `process/` is removed after each
successful entry), then complete with the network and the crabWURCS release, finalize and compare:

```bash
rsync -a iridis-x1:/scratch/os1e24/gap-bulk/run/ $R/
python scripts/bulk_reanalyse.py complete --run-dir $R --jobs 8 [--curation-dir <records>]
python scripts/bulk_reanalyse.py finalize --run-dir $R
python scripts/bulk_reanalyse.py compare --run-dir $R
python scripts/bulk_reanalyse.py summary --run-dir $R
```

## What to check before publishing

`compare` (exit status 0) and `compare.json`:

* **counts**: as many glycans as published, the same IDs (blocking otherwise);
* **names**: no `archetype.name` changes unless intended (`plan --rename-to-raw-names` gives the three entries
  published under a GlyTouCan accession, GS00722, GS00723, GS00728, their GLYCAM names; `--intended-rename ID` for
  others); an unintended rename is blocking;
* **identity**: every entry has a WURCS or an accession (blocking); `accession_lost` should be 0 after an online
  `complete` (an offline one loses the alpha/beta accessions); changed accessions and WURCS are expected where
  crabWURCS was fixed: look at them;
* **files and sizes**: no file the website needs is missing (blocking); folder sizes within 0.5 to 2 times the
  published ones (`size_outside` names the others);
* **analysis**: Level 1 population differences (`level_1_population_tv`) and representative frames that changed;
  large differences come from analysis fixes and should be explainable;
* **kept_published**: the entries that keep their published version, with the reason: decide whether that is
  acceptable for each, and put them on the admin page.

Then look at a few entries by hand (`entries.<ID>.diff` in `compare.json`), and publish:

```bash
set -a; . ~/gap-work/r2-test/r2env.sh; set +a
python scripts/bulk_reanalyse.py check-base --run-dir $R
python scripts/bulk_reanalyse.py publish --run-dir $R --expect-base <base_global_hash of plan.json> --dry-run
python scripts/bulk_reanalyse.py publish --run-dir $R --expect-base <base_global_hash of plan.json> --workers 32
python scripts/static_s3.py check
```

During the switch (the copies of the changed files inside R2, after the upload) a website that is behind fails its
synchronisation and retries; one that is up to date is not affected. With nearly every file changed the window is
the time of about 21,000 copies, so use `--workers 32`; the dev site then downloads the whole database once (76 min
from R2 for 23.6 GB on 2026-10-11).

## Measured on the Spark (2026-10-14, read-only, main 4a9a775 plus this branch)

Three real entries through every stage, the other 998 kept as published: GS00020 (45,000 frames) 110 s and 5.6 GB
peak with 4 workers on a loaded 20-core machine, GS00002 30 s, GS00276 56 s (it does not fail with this code; the
failure is expected with the torsion and MOL2 fixes); `complete --offline` 7 s for the three; `finalize` of the
whole tree 100 s and 1 GB; `compare` 16 s. The re-analysis reproduced the published Level 1 populations and
representative frames of all three exactly; 21 of 25 files per folder changed; GS00020 gained the class keywords
`Complex N-Glycan` and `Tetraantennary N-Glycan`; offline, the alpha/beta accessions were missing (expected).

## After the bulk publication

* Entries whose files exchange alpha and beta (problem list, section 1; for example GS00020, GS00021, GS00028):
  `publish_entry.py reanalyse --id GS00020 --simulated-anomer alpha --note "..." --work-dir ... --dry-run`, then the
  same with `--expect-hash`. This renames the entry to the anomer that was simulated and moves its raw entry.
* Entries whose replicas are different molecules (G04196SD, G71956IS, G97921YP):
  `reanalyse --exclude-replica run_3`.
* The two entries estimated over 32 GB are refused by a worker limited to 32 GB (`--memory-limit-gb`); a later
  single-entry `reanalyse` of them runs on the Spark (128 GB, where the R2 keys are) with the same command.
