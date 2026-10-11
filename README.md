# GlycanAnalysisPipeline 🧬

Glycan conformations analysis pipeline for [GlycoShape Database](https://glycoshape.org).

<img src="docs/Figure.jpg" alt="Schematic overview of Glycan Analysis Pipeline (GAP) used to build the GlycoShape Glycan Database (GDB). Panel a) Multiple uncorrelated replica molecular dynamics (MD) simulations are performed for each glycan in the GDB, to comprehensively sample its structural dynamics. The resulting MD frames are then transformed into a graph matrix representation, as depicted in Panel b), simplified by flattening the lower half as shown in Panel c). This step enables a dimensionality reduction via principal component analysis (PCA), shown in Panel d). These data are clustered by Gaussian Mixture Model (GMM) and the results of which are displayed in terms of cluster distributions, see Panel e). Panel f) Representative 3D structures for each cluster are selected based on KDE maxima, along with comprehensive torsion angle profiles for the highest populated clusters, showing the wide breadth of the conformational space covered by GAP. Panel g) Structures derived from GAP are clearly presented on the GlycoShape GDB web platform, in addition to biological and chemical information." style="zoom: 33%;" />

## At a glance 🗺️

GAP takes the trajectories of one glycan (all its simulations, dry) and produces a database entry: the conformational landscape (PCA), clusters at two levels of detail with their populations, representative structures (Level 1: the main conformers; Level 2: a finer coverage of the landscape), torsion statistics and the files the GlycoShape website serves.

| Stage | What it does | Where it runs | Section |
|---|---|---|---|
| Raw archive | One entry per glycan, named by its GLYCAM name: `trajectory.xtc` at the precision of the source, `topology.pdb`, `structure.mol2`, `manifest.json` with checksums | Anywhere; object storage keeps it under `raw/<GLYCAM name>/` | [Raw trajectory archive](#raw-trajectory-archive) |
| Analysis of one entry | Store, PCA, clustering, torsions, representative structures; a self-contained result folder with `result.json` | A compute node, without network | [One entry on a compute node](#one-entry-on-a-compute-node) |
| A run on the cluster | The dry trajectories of the replicas of a run become one archive entry, its analysis and one bundle file to upload; an existing entry can be extended | Inside the simulation job, without network | [A run on the cluster](#a-run-on-the-cluster) |
| Whole-database runs | The same analysis over a data directory, then the static folder with identifiers and metadata | Login node and Slurm array | [Running](#running-), [Staged Oracle/Slurm workflow](#staged-oracleslurm-workflow) |
| Publishing | The static database with `sync_manifest.json`, which the website synchronises from | Login node; S3-compatible storage (Cloudflare R2) or Oracle | [S3 storage (Cloudflare R2)](#s3-storage-cloudflare-r2) |
| Publishing one entry | The bundle of one analysed entry becomes `static/GSxxxxx/` and `raw/<GLYCAM name>/`; the root files and `sync_manifest.json` follow from the stored ones. Also: replace, withdraw | The server that approves entries; S3-compatible storage | [One entry: `scripts/publish_entry.py`](#one-entry-scriptspublish_entrypy) |

Level 3 (a coverage set of up to 512 structures) is no longer produced by default; `GLYCOSHAPE_CLUSTER_LEVELS=1,2,3` brings it back, and folders that still contain it are accepted.

## Installation 💻

The pipeline uses Python 3.10 and uv (a fast Python package manager) for dependency management.

### Linux/macOS 🐧🍏

1. Make the installation script executable:
   ```bash
   chmod +x install.sh
   ```

2. Run the installation script:
   ```bash
   ./install.sh
   ```

3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

### Manual Installation (Alternative) 🛠️

If you prefer manual installation:

1. Install Python 3.10 (if not already installed)
2. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. Create virtual environment: `uv venv --python 3.10 .venv`
4. Activate environment:
   - Linux/macOS: `source .venv/bin/activate`
   - Windows: `.venv\Scripts\activate`
5. Install dependencies: `uv pip install -r requirements.txt`
6. Install the crabWURCS CLI (requires Rust/Cargo):
   `cargo install --git https://github.com/Ojas-Singh/crabWURCS --rev d6145803ce5977eb4fb9ab785f3a242f63e0036b --locked crabwurcs-cli`

## Configure ⚙️

Before running the pipeline, ensure the following environment variables are set up.

```bash
export GLYCOSHAPE_ORACLE_PAR_URL=https://object-storage-par-url/
export GLYCOSHAPE_ORACLE_PROCESS_PREFIX=process
export GLYCOSHAPE_ORACLE_OUTPUT_PREFIX=static
export GLYCOSHAPE_DATA_DIR=data
export GLYCOSHAPE_PROCESS_DIR=/scratch/os1e24/GlycanAnalysisPipeline/process
export GLYCOSHAPE_OUTPUT_DIR=/scratch/os1e24/GlycanAnalysisPipeline/static
export GLYCOSHAPE_INVENTORY_PATH=/path/to/glycoshape/inventory
export GLYCOSHAPE_DB_UPDATE=True
export GLYTOUCAN_CONTRIBUTOR_ID=your_contributor_id
export GLYTOUCAN_API_KEY=your_api_key
export GLYCOSHAPE_RDF_DIR=/scratch/os1e24/GlycanAnalysisPipeline/GLYCOSHAPE_RDF
export GLYCOSHAPE_GAP_LOG=/scratch/os1e24/GlycanAnalysisPipeline/logs/GlycoShape_GAP.log
export GLYCOSHAPE_MAX_WORKERS=1
export GLYCOSHAPE_STATIC_WORKERS=8
export GLYCOSHAPE_UPLOAD_WORKERS=1
export GLYCOSHAPE_STORE_FRAMES_BUFFER=128
export GLYCOSHAPE_STORE_FRAME_CHUNK=128
export GLYCOSHAPE_CLUSTER_LEVELS=1,2
export GLYCOSHAPE_MAX_COVALENT_BOND_ANGSTROM=3.0
export GLYCOSHAPE_FRAME_QUALITY_BLOCK_FRAMES=1024
export GLYCOSHAPE_LOOKUP_TIMEOUT=30     # seconds per request to GlyCosmos, GlyTouCan, GlyGen, PubChem
export GLYCOSHAPE_OFFLINE=0             # 1: make no such request; entries are written without what they would have answered
export GLYCOSHAPE_SNFG_RENDERER=crabwurcs
# Optional when crabwurcs is not on PATH:
export GLYCOSHAPE_CRABWURCS_BIN=/path/to/crabwurcs
export POCKETBASE_URL=http://localhost:8090
export POCKETBASE_TOKEN=your_pocketbase_token
```

Ensure the required directories and files exist, or modify the paths as needed.

`GLYCOSHAPE_SNFG_RENDERER` controls static `snfg.svg` generation. It defaults to
`crabwurcs`, which renders the submitted GLYCAM notation directly. Set it to
`glycodraw` to retain the previous glycowork `GlycoDraw` renderer. `install.sh`
installs a project-local crabWURCS binary when the default renderer is selected;
`GLYCOSHAPE_CRABWURCS_BIN` can instead point to an existing installation.
The same executable is also used as a local fallback for IUPAC/SMILES → WURCS
and WURCS → extended-IUPAC conversion when the remote format service cannot
represent a glycan.

With `GLYCOSHAPE_ORACLE_PAR_URL` set, use the staged workflow below: download the bucket `data/` prefix to local storage on a login node, process local data on AMD Slurm nodes, then build and upload only `static/` from a login node. `process/` is deliberately local-only and must not be uploaded to Oracle.

`GLYCOSHAPE_CLUSTER_LEVELS` (or `--levels`) is the set of clustering levels
that is computed, required by the completeness checks and exported: `1,2`
(the default) or `1,2,3`. See [Clustering levels](#clustering-levels).

`GLYCOSHAPE_STORE_FRAME_CHUNK` is the number of frames per Zarr chunk of
`process/<glycan>/data/coords.zarr`; `0` (the default) is one chunk for the
whole trajectory. `GLYCOSHAPE_STORE_FRAMES_BUFFER` is only how many parsed
frames are handed over at a time: whatever its value, every chunk is
compressed and written once.

`GLYCOSHAPE_MAX_WORKERS` controls numerical fan-out inside one glycan.
`GLYCOSHAPE_STATIC_WORKERS` separately controls how many glycans static-only
mode rebuilds concurrently and defaults to 8. Set it to `1` for serial
debugging or lower it on a memory-constrained login node.

When `POCKETBASE_URL` and either `POCKETBASE_TOKEN` or `POCKETBASE_ADMIN_TOKEN` are set, the pipeline uses PocketBase `glycan_submission` records as the submission metadata source for `glystatic` and related metadata helpers. The CSV inventory remains the fallback when PocketBase is unset, unavailable, or missing a record.

## Running 🚀

```bash
python main.py --mode incremental
```

Run modes:

- `incremental` (default): new or incomplete glycans run fully; completed glycans skip v2/process/glystatic and only sync static locally when needed.
- `refresh-static`: completed glycans sync static locally and refresh metadata without process downloads.
- `fresh`: recomputes v2 and glystatic even when bucket outputs already exist. `--update` is kept as an alias for this mode except when `--mode process-only` or `--mode static-only` is explicitly selected.
- `process-only`: runs local `data/` to local `process/` only, skips glystatic/glymeta/PocketBase/Oracle upload/finalization, and is suitable for Slurm nodes without internet access. For a single entry with a self-contained result folder see [One entry on a compute node](#one-entry-on-a-compute-node).
- `static-only`: rebuilds local `static/` from existing local `process/` outputs, skips already complete GS folders unless `--update` is passed, skips v2/process and Oracle sync/upload, runs glymeta best-effort for rebuilt glycans, and refreshes final database artifacts.

### Clustering levels

Level 1 is the GMM clustering into five clusters, Level 2 the k-center
coverage set (5 to 128 structures, from the conformational entropy). Level 3,
the k-center set of up to 512 structures, is no longer published and is off by
default: with levels `1,2` nothing of it is computed, written, required or
exported, in the process stage and in the static stage. Levels 1 and 2 do not
depend on it (each k-center run seeds the generator itself), so they are the
same with and without it.

- `--levels 1,2,3` (or `GLYCOSHAPE_CLUSTER_LEVELS=1,2,3`) restores Level 3
  everywhere: clustering, `output/level_3/`, `torsion_level_3_reps`, the
  `level_3_cluster` / `level_3_center` columns of `torsion_all`, `info.json`
  and the completeness checks.
- Existing `process/` and `static/` folders that contain Level 3 are complete
  under either setting and are not recomputed by an incremental run.
- The static stage writes only the configured levels: `--mode static-only
  --update` with levels `1,2` rebuilds `static/GSxxxxx/` without Level 3 from a
  `process/` folder that has it (no `output/level_3/`, no
  `torsion_level_3_reps.csv`, no Level 3 columns in `torsion_all.csv`, no
  Level 3 entries in `info.json` and `data.json`).
- A `process/` folder made with levels `1,2` is incomplete under `1,2,3`; a
  process run then adds Level 3 and rewrites what is indexed by clusters.
- A forced process run (`--update`) with levels `1,2` on a folder that has
  Level 3 leaves the old `output/level_3/` and `torsion_level_3_reps.parquet`
  files in place; they are ignored.
- A published `static/` database that has Level 3 is turned into the one
  without it, with no `process/` data and no recomputation, by
  [`scripts/static_drop_levels.py`](#static-database-without-level-3-scriptsstatic_drop_levelspy).

Process-only runs keep `process/` Hugging Face friendly: tabular analysis files are written as Parquet under `embedding/` and `embedding/exports/`, while `process/output/` keeps only static-ready non-tabular artifacts (`info.json`, `torparts.npz`, representative PDBs, and SVG plots). Static generation later converts those Parquet exports back to the website CSV files under `static/`.

Before PCA, the store step repairs connectivity using chemically plausible
bonds observed at covalent distance in the trajectory. It then excludes only
frames containing non-finite coordinates or a recovered covalent bond longer
than `GLYCOSHAPE_MAX_COVALENT_BOND_ANGSTROM` (default 3.0 Å). Inter-frame atom
movement and whole-molecule translation are diagnostic only. Frame provenance
is recorded in `data/frame_quality.parquet`, `data/frame_map.parquet`, and
`data/frame_filter.json`.

```bash
python main.py --mode process-only --data-dir data --process-dir process
python main.py --mode process-only --glycan DGalpa1-OH --data-dir data --process-dir process --update
```

Audit old data and optionally rebuild only entries with rejected frames,
repaired topology, or stale/missing frame-quality metadata:

```bash
.venv/bin/python scripts/audit_and_rebuild_broken_frames.py \
  --data-dir data --process-dir process --report frame_quality_audit.json

.venv/bin/python scripts/audit_and_rebuild_broken_frames.py \
  --data-dir data --process-dir process --rebuild

# AMD Slurm equivalent; set REBUILD=1 after reviewing the audit.
sbatch slurm/audit_rebuild_broken_frames_amd.sh
REBUILD=1 sbatch slurm/audit_rebuild_broken_frames_amd.sh
```

After process-only jobs finish, rebuild website-serving files on a login node:

```bash
python main.py --mode static-only --glycan DGalpa1-OH --process-dir process --output-dir static
python main.py --mode static-only --process-dir process --output-dir static
```

Static-only mode discovers complete local `process/<glycan>/` folders, skips incomplete ones, and rebuilds independent glycans concurrently (8 at a time by default). It auto-skips already complete `static/GSxxxxx/` folders, converts Parquet exports back to the website CSV filenames for rebuilt glycans, copies representative structures/SVGs and PCA landscape files, then rebuilds `static/GLYCOSHAPE.json`, `static/GLYCOSHAPE_RDF.ttl`, and archive artifacts. Pass `--update` to force overwriting/refreshing existing GS folders. It ignores `GLYCOSHAPE_ORACLE_PAR_URL`; no Oracle upload or sync is performed.

### Static database without Level 3: `scripts/static_drop_levels.py`

```bash
.venv/bin/python scripts/static_drop_levels.py --source static --dest static-l12 --workers 8
.venv/bin/python scripts/static_s3.py publish --static-dir static-l12 --dry-run     # then publish it
```

Writes a static database that was built with Levels 1, 2 and 3 as the one the pipeline builds with levels `1,2` (`--levels`, default `1,2`), into a new folder. Nothing is recomputed, no `process/` data, service or network is needed, GlycoShape IDs and metadata stay as they are, and the source is only read (a source from which the `output/level_3/` folders were already deleted gives the same result).

Per glycan folder, in `--workers` processes:

| Files | What happens |
|---|---|
| `glygen.json`, `glycosmos.json`, `snfg.svg`, `landscape/*`, `output/torparts.npz`, `output/pca.csv`, `output/torsion_glycosidic.csv`, `output/level_{1,2}/*`, `output/torsion_level_{1,2}_reps.csv` | kept: hard-linked (copied on another file system, or with `--copy`) |
| `output/info.json` | rewritten without the Level 3 entries (`lib.static_ready.info_json_for_levels`) |
| `output/torsion_all.csv` (any table with `level_3_cluster` / `level_3_center` columns) | rewritten without those columns (`lib.static_ready.csv_for_levels`) |
| `data.json` | rewritten without `archetype.cluster_levels.level_3` (`lib.static_ready.save_glycan_data`) |
| `output/level_3/*`, `output/torsion_level_3_reps.csv` | dropped |
| `GSxxxxx.zip` | built again at the end: it holds `data.json` (and Level 1 only) |
| anything else | kept and listed as `extra` in the report; hidden files are left out, because the website refuses a manifest that lists one |

The three rewritten files are written by the functions the static stage uses, so they are byte for byte what `main.py --mode static-only --update` writes with levels `1,2` from the process folder the published one was made of (`data.json`: the published one without that entry, in the same format). A folder is built in a staging folder, checked for every file the pipeline requires for these levels, and renamed into the destination: the destination only ever holds finished folders.

When every folder is there the root files are written in the order of the finalization, with its functions: `GLYCOSHAPE.json` (from the `data.json` files), every `GSxxxxx.zip` and `GlycoShape.zip`, and `sync_manifest.json` last. `GLYCOSHAPE_RDF.ttl` and the `missing_*` lists do not depend on the levels and are copied as they are; so is any other root file, which the report lists. The manifest is then validated with the rules of the publisher and of the website, and has to list exactly the files of the new folder.

- **Resume** by running the same command again: a glycan folder that is in the destination is finished and skipped. A glycan that fails (for example an unreadable `info.json`, or a folder that lacks a Level 1 or 2 file) is reported and left out, the others are done, the root files are not written and the exit status is 1. Delete `sync_manifest.json` in the destination to have the root files written again.
- **Nothing else is written into the destination.** Staging folders, the per-glycan records, a lock and `report.json` (per glycan: files kept, rewritten, dropped, bytes before and after) are in the work folder, `<dest>.work` unless `--work-dir` names another (on the file system of the destination). It can be deleted afterwards.
- **Hard links.** A kept file of a glycan folder is one file with two names. This tool never writes into one, and neither do the static stage and the finalization run on local folders (the files they rewrite in place, which are the root files and `data.json`, are copied here, not linked). Anything else that edits such a file in place in one database changes the other: delete the source afterwards, or use `--copy`.
- **Reproducible but for the times in the manifest**: `sync_manifest.json` carries its `generated_at` and the modification times of the files, which no hash covers. Everything else is a function of the source, the archives included (see [Reproducible root files](#reproducible-root-files)).
- **The source is checked on the way.** Every file that is taken from it, kept or read to be rewritten, is compared with the SHA-256 that the `sync_manifest.json` of the source records, and the new `GLYCOSHAPE.json` with the one of the source (equal but for `cluster_levels.level_3`). A difference is a warning in the output and in the report: the source is then not the database its manifest describes (an incomplete copy, a file changed afterwards).

A database of the size of the published one (1,001 glycans, 29 GB without the Level 3 structures, of which 13 GB `torsion_all.csv` and 4.5 GB `info.json`) took one minute with 8 workers and 2 GB of memory, half of it for the glycan folders and half for the archives and the manifest; eight glycans of the largest size in work together took 5 GB. `--skip-finalize` stops after the glycan folders.

## Staged Oracle/Slurm workflow

Run these commands from the repository checkout. The download and publish stages run on a login node; only the processing stage runs on the AMD Slurm partition.

```bash
# Login node: mirror Oracle data/ locally. This only downloads source files.
./scripts/run_staged_pipeline.sh download

# Login node: submit local data/ -> local process/ jobs on four AMD nodes.
sbatch slurm/process_only_amd.sh

# Login node, after every array task succeeds: local process/ -> local static/,
# then publish only static/ to Oracle.
./scripts/run_staged_pipeline.sh publish
```

The script defaults to `#SBATCH --array=1-4`. A Slurm array creates four independent array tasks (`1`, `2`, `3`, and `4`); each task requests one AMD node with 64 CPUs, so this means four machines and 256 CPUs when all tasks run concurrently. The array number is a shard ID, not a glycan ID. The sorted glycan list is distributed round-robin: task 1 receives entries 1, 5, 9, …; task 2 receives 2, 6, 10, …; and so on. With the current defaults, each node processes 16 glycans at once and assigns four CPUs to each glycan (`16 × 4 = 64`). Across four simultaneously running nodes, up to 64 glycans are processed at once.

The plain `sbatch slurm/process_only_amd.sh` command uses that four-node default. To choose a different number of shards/nodes, override the embedded array setting—for example, `sbatch --array=1-8 slurm/process_only_amd.sh` creates eight tasks. Actual simultaneous execution still depends on scheduler capacity and account limits; Slurm may queue some array tasks. `NJOBS` changes concurrent glycans per node and `WORKERS` changes CPUs per glycan, but `NJOBS × WORKERS` must not exceed the 64 allocated CPUs.

The normal workflow is incremental: `download` preserves existing local source files; the Slurm stage reuses complete local `process/` results; and `static-only` skips complete `static/GSxxxxx/` folders. To deliberately recompute assigned glycans, submit with `FORCE_UPDATE=1 sbatch slurm/process_only_amd.sh`. For a subset, pass one or more `--glycan <name>` arguments to `download`; then submit the normal Slurm script. `publish --skip-existing` retains existing Oracle objects, while the default overwrites files in the `static/` prefix.

The login-node scripts are deliberately scoped as follows:

- `scripts/download_data_from_oracle.py`: Oracle `data/` → local `data/` only.
- `slurm/process_only_amd.sh`: local `data/` → local `process/` only; it unsets the Oracle PAR URL.
- `scripts/upload_static_to_oracle.py`: local `static/` → Oracle `static/` only.

Source directory names under `data/` are opaque storage keys and need not be GLYCAM names. When present, `data/<storage-key>/name.json` supplies the authoritative `indexOrderedSequence` used by processing. Use `scripts/upload_data_to_oracle.py --glycan <storage-key> --file name.json` to publish a corrected mapping without re-uploading a simulation PDB.

## One entry on a compute node

After a simulation job has written its trajectories, the same job can analyse
them on its node, without network access and without any service (no
PocketBase, no GlyTouCan or format services, no object storage):

```bash
.venv/bin/python scripts/run_entry.py <entry> <result-folder> [--workers N] [--levels 1,2] [--force]
```

`<entry>` is an archive entry (`manifest.json`, `trajectory.xtc`,
`topology.pdb`, `structure.mol2`) or a folder with `simulation.pdb` and
`structure.mol2`. The files of an archive entry are checked against the sizes
and SHA-256 of its manifest first (`--no-verify-input` checks sizes only). The
worker count is `--workers`, else `SLURM_CPUS_PER_TASK`, else
`GLYCOSHAPE_MAX_WORKERS`, else the cores the process may run on; it sets the
BLAS/OpenMP threads and the worker processes of the run, whatever the job
environment says. In a job script:

```bash
#SBATCH --cpus-per-task=64
...                                             # the simulation and the conversion to an archive entry
"$GAP/.venv/bin/python" "$GAP/scripts/run_entry.py" "$ARCHIVE/$FOLDER" "$RESULTS/$FOLDER" || exit 1
```

`slurm/run_entry_amd.sh` runs it as a job of its own. Exit status: `0` done,
`1` the analysis failed (the step and the message are printed, and are in
`result.json` and `run.log`), `2` the input or the result folder cannot be
used (message printed; nothing is computed). For the replicas of a simulation
run, `scripts/make_entry.py` does the conversion to an entry, this analysis
and the bundle to upload in one command: [A run on the cluster](#a-run-on-the-cluster).

The result folder is self-contained:

```
<result-folder>/
  result.json           status; input (kind, path, atoms, frames, size and SHA-256 of every file; for an
                        archive entry these are the hashes of its manifest, with the SHA-256 of the
                        coordinates and of the source simulation.pdb); pipeline (commit, SHA-256 of the
                        sources, package versions); settings (levels, workers, thresholds); seconds per step;
                        frame-quality summary (frames read, kept, rejected, bonds recovered); analysis summary
                        (entropy, clusters, sizes and representative frames per level); size and SHA-256 of
                        every file produced; what is left to publishing
  run.log               the log of the run
  input/                manifest.json, info.json, name.json of the entry (the small files, not the trajectory)
  process/<folder>/     the processed data: data/, embedding/, output/ exactly as in process/<glycan>/
  static/               the part of static/GSxxxxx/ that follows from the trajectory alone:
    output/info.json, torparts.npz
    output/pca.csv, torsion_glycosidic.csv, torsion_all.csv, torsion_level_1_reps.csv, torsion_level_2_reps.csv
    output/level_1/, level_2/     dist.svg, GLYCAM/alpha.pdb, GLYCAM/beta.pdb, PDB/alpha.pdb, PDB/beta.pdb
    landscape/pca_model.joblib, landscape_metadata.json, preprocessing.npz
```

`result.json` has status `running` while the run is in progress, so a job that
is killed leaves a folder that says so. Running the command again on a result
folder returns at once when the result is current (same input files, levels,
bond threshold and code); otherwise `process/`, `static/` and `input/` in it
are replaced, never completed. A folder that is not empty and holds no
`result.json` of this tool is refused.

Deliberately left to publishing, because it needs identifiers, submission
metadata or the network:

- the GlycoShape ID and with it the folder name `static/GSxxxxx`;
- `data.json` (names, IUPAC, WURCS, GlyTouCan accessions, composition, motifs,
  simulation metadata), `glygen.json`, `glycosmos.json`, `snfg.svg`;
- the GlycoShape `REMARK` block with the GlyTouCan ID and the IUPAC name at
  the top of the PDB files: the files in `static/output/level_*/` are exactly
  the published files without that block (their first line is `REMARK
  GLYCAN MULTI-MODEL PDB ...`);
- `GLYCOSHAPE.json`, `GlycoShape.zip`, `GLYCOSHAPE_RDF.ttl`,
  `sync_manifest.json`, and any upload.

Publishing can start from either half of the result: run the static stage on
the processed data (`python main.py --mode static-only --process-dir
<result-folder>/process ...`, which writes the same files plus the ones
above), or hand the static-ready half over as a bundle and let
[`scripts/publish_entry.py`](#one-entry-scriptspublish_entrypy) add the files
above and publish the entry. The two give the same folder: `lib/static_ready.py`
is the single implementation of the shared files and
`lib.glystatic.write_identity_files` of the identifier files; both ways call
them. The static stage takes the GLYCAM name from the name of
the process folder, which for an archive entry with a long name is the
shortened folder name: the name to publish under is `glycam_name` in
`result.json`.

The analysis of one entry needs memory for one matrix of 8 bytes x frames x
atom pairs of the PCA landscape (pairs of the heavy atoms outside the reducing
`ROH`, at most 300 atoms): 18.8 GB for the 159,940 retained frames of a
329-atom glycan (172 such atoms, 14,706 pairs), 0.8 GB for 50,005 frames of a
97-atom one (62 atoms, 1,891 pairs). Everything else is small next to it.

## A run on the cluster

A simulation run leaves one dry multi-model PDB per replica. One command, run in the job that simulated them, turns the replicas into what publishing needs: the archive entry, its analysis and one file to upload. It uses no network and no service.

```bash
# Once, on the login node (needs the network): the environment of the node command
uv venv --python 3.10 "$GAP/.venv-node"
uv pip install --python "$GAP/.venv-node/bin/python" -r "$GAP/requirements-node.txt"

# In the job, after the simulations and their post-processing (192 cores, --cpus-per-task 1)
"$GAP/.venv-node/bin/python" "$GAP/scripts/make_entry.py" \
    --glycam-name "$GLYCAM_NAME" --workers 16 --output "$RUN/bundle.tar" \
    --replica-map "$RUN/replica_map.txt" --frame-stride 5 \
    --md-package GROMACS --force-field GLYCAM_06j --temperature-k 300 --pressure-bar 1 \
    --salt-concentration-mm 200 --run-id "$RUN_ID" --summary "$RUN/bundle.summary.json" || exit 1
```

The replicas are given as arguments (a folder with `glycan.dry.pdb` and `glycan.dry.mol2`, or `simulation.pdb` and `structure.mol2`; or the PDB file, its MOL2 next to it; `NAME=PATH` names one) or with `--replica-map`, the `folder|name` file a run already has. The command

1. checks on the first frames that all replicas have the same atoms in the same order, and their MOL2 files the same atoms and bonds;
2. converts them into one archive entry in `--workers` processes, each replica verified against its own text and recorded as a [segment](#raw-trajectory-archive) (no concatenated `simulation.pdb` is written);
3. analyses the entry as [`run_entry.py`](#one-entry-on-a-compute-node) does;
4. writes the bundle and reads it back the way the receiving side does.

| Option | |
|---|---|
| `--glycam-name` (required) | the full GLYCAM name; it names the entry and is written as `name.json`. `info.json` is taken from `--info`, else from the first replica folder that has one, and archived unchanged (a different `indexOrderedSequence` in it is a warning in the manifest) |
| `--workers` (required) | cores to use. There is no default: a simulation job has `--cpus-per-task 1`, so `SLURM_CPUS_PER_TASK` says nothing useful. The command has to be started where it may use that many cores (in the batch script itself, or with `srun --exact --ntasks=1 --cpus-per-task=N`, not as one task of one CPU); `source.host.cores_available` of `bundle.json` records what it was given, with a warning when that is fewer than `--workers` |
| `--output` (required) | the bundle; it appears under this name only when it is complete, and a bundle that an earlier run left there is removed first: after a run there is its bundle, or none |
| `--frame-stride N`, or `--time-per-frame-ps T` | the dry trajectory holds every N-th saved frame (`trajin ... 1 last 5` is 5); with `production.mdp` in the replica folder the time per frame is `dt x nstxout-compressed x N`. Recorded per segment; without either it is null |
| `--md-package`, `--force-field`, `--temperature-k`, `--pressure-bar`, `--salt-concentration-mm`, `--comments`, `--glytoucan-id`, `--run-id`, `--simulation-length-ns` | metadata of the run for publishing, copied into `bundle.json` with the unit in the name of the key; the run identifier, package, force field, temperature, pressure and salt concentration are also recorded with every segment of the manifest (`run`), which outlives the bundle |
| `--replica-prefix TEXT` | put before the name of every replica, for example the identifier of the run; the segments of an entry are told apart by their names |
| `--extend ENTRY` | merge and extend, see below |
| `--levels`, `--mol2`, `--source-name`, `--repair`, `--work-dir`, `--keep-work`, `--force`, `--summary FILE`, `--verbose` | as their help says |

Standard output is one line (`make_entry: ok: <name>: <frames> frames of <atoms> atoms (...), <kept> kept; clusters ...; bundle <path> <MB> sha256 <hash>; <seconds> s with <n> workers`); everything else goes to standard error, and `--summary` writes the same as JSON. Exit status: `0` the bundle was written, `1` the analysis or the bundle failed (the step and the message are printed; `result.json` and `run.log` are in the work folder), `2` the input cannot be used (the message says why, for example which atom of which replica differs or which replica ends inside a frame; nothing is computed).

The work is done in `<output>.work` (`--work-dir`), which is removed when the bundle is there. After an interruption, or a failure, run the same command again: an entry that was converted from exactly these files (they are hashed: a replica that was simulated again has the same size) and an analysis that is current for it are reused, anything else is done again. Run one command per output at a time.

### The bundle

A plain `tar` file (ustar, not compressed) with these members, in this order:

```
bundle.json     schema "glycoshape-entry-bundle", schema_version 1, glycam_name, created (UTC), pipeline
                {commit, uncommitted_changes, version, converter_version, python, packages}, levels,
                files {path: {size, sha256}} for every other member, source {...}, extends
archive/        trajectory.xtc, topology.pdb, structure.mol2, manifest.json, info.json, name.json: the archive
                entry (in a bundle its folder is "archive", not the folder its manifest names)
static/         output/..., landscape/...: the static-ready files of run_entry.py
result.json     as run_entry.py writes it
run.log
```

`pipeline.version` is the SHA-256 of `main.py` and `lib/*.py` (`code_sha256` of `result.json`); `commit` is null where the code is not a git checkout. `source` says what the entry was made from: `replicas` (name, frames, first frame, time per frame, size and SHA-256 of the PDB and the MOL2, the settings read from `production.mdp`), `trajectory` (atoms, frames, how many of them from this run and from an extended entry), and the metadata of the run:

| Key of `source` | Unit |
|---|---|
| `simulation_length_ns` | nanoseconds, summed over the replicas of this run (not of an entry that is extended). From the frames, `(frames - 1) x time_per_frame_ps`, when the time per frame is known, else `dt x nsteps` of every `production.mdp`, else null; `simulation_length_from` says which. `--simulation-length-ns` overrides it |
| `time_per_frame_ps` | picoseconds |
| `temperature_k`, `pressure_bar`, `salt_concentration_mm` | kelvin, bar, millimolar NaCl |
| `md_package`, `force_field`, `comments`, `glytoucan_id`, `run_id` | text |

`extends` is null, or `{glycam_name, manifest_sha256, frames}` of the entry that was extended.

It is not compressed: `trajectory.xtc` is 60 to 64 % of a bundle and gzip takes 3 to 10 % off it; the CSV files (a third of a bundle) halve. On the three glycans below gzip would save 22 to 29 % (97 of the 441 MB of the largest, for 13 s of one core on the node). The receiving side would then have to decompress untrusted input before it can check anything, and a member could no longer be read at its offset in the file. A bundle is already a tenth of the `simulation.pdb` that is uploaded today (441 MB for 4.29 GB).

`lib.entry_bundle.open_bundle(tar_path, extract_to)` is the receiving side. It treats the bundle as untrusted (it is made under a member's account and unpacked on a server) and returns the folder, with exactly the layout above, only when everything checks; otherwise it raises `BundleError` and leaves nothing behind. It needs the standard library only. Checked are: a plain tar whose first member is `bundle.json` of this schema; after it exactly the listed members in order, each a regular file with a plain header (no link, device, directory, sparse file, extended or long-name header) of the listed size and SHA-256, and nothing after them; every path relative, without `..`, hidden or unusual names, and one of `archive/<file of an entry>`, `static/...`, `result.json`, `run.log`; no member above `max_member_bytes` and all together not above `max_bytes` (16 GiB), before anything is read; the files a publishable entry needs for its levels are there; `archive/manifest.json` and `result.json` are of the GLYCAM name of `bundle.json` (and of `expected_glycam_name`), the manifest lists the archive files of the bundle with their sizes and hashes, the result was computed from exactly that manifest and those files with those levels and records exactly the static files of the bundle; `extends` agrees with the manifest. `check_bundle` does the same without unpacking, `load_bundle_dir` checks an unpacked folder again. `lib.archive.verify_entry(<folder>/archive, check_folder=False)` additionally decodes the trajectory.

### Merge and extend

When the glycan is already in the database, the new trajectories can be added to the stored ones and the union analysed:

```bash
.venv/bin/python scripts/archive_s3.py download --archive-dir "$RUN/stored" --glycan "$GLYCAM_NAME"   # login node
"$GAP/.venv-node/bin/python" "$GAP/scripts/make_entry.py" --glycam-name "$GLYCAM_NAME" --workers 16 \
    --output "$RUN/bundle.tar" --extend "$RUN/stored/<entry folder>" --replica-map "$RUN/replica_map.txt" \
    --replica-prefix "$RUN_ID-" --frame-stride 5
```

The frames of the stored entry come first and are taken over as the bytes they are (see [Extending an entry](#raw-trajectory-archive)); the replicas follow. The stored entry is checked against its manifest first and is only read. Its name has to be `--glycam-name`, and the replicas have to have its atoms, the same names, residues and order, or the command stops with exit status 2 and says which atom differs. A replica may not have the name of a segment the stored entry already has (runs name their replicas alike: use `--replica-prefix`). `bundle.json` then has `extends` with the SHA-256 of the manifest of the stored entry: publishing replaces the stored entry only if that is still the manifest in the storage. `source.simulation_length_ns` covers the new replicas only; the length of the whole is the stored length plus it. An entry that was made from a damaged source keeps the record of that in its segment and stays listed as repaired.

### Time and memory

Measured on a 20-core ARM machine with the replicas on a local disk: the whole command under `/usr/bin/time`, each run started while nothing else used the machine. The memory is the larger of the maximum resident set of the main process and the peak of the summed proportional set sizes of all processes of the command (sampled twice a second).

| Glycan | Replicas | `--workers` | Time | Peak memory | Bundle |
|---|---|---|---|---|---|
| 97 atoms, 50,005 frames | 3 (384 MB of text) | 8 | 14 s | 1.9 GiB | 48 MB |
| | | 16 | 16 s | 3.0 GiB | |
| 231 atoms, 40,004 frames | 2 (731 MB) | 8 | 18 s | 3.1 GiB | 82 MB |
| | | 16 | 20 s | 3.6 GiB | |
| 329 atoms, 160,016 frames | 4 (4.29 GB) | 8 | 91 s | 19.0 GiB | 441 MB |
| | | 16 | 90 s | 19.4 GiB | |

Sixteen workers were not faster than eight there. Of the 91 s of the largest, 6 s are the conversion (four replicas in four processes), 84 s the analysis (clustering 37 s, PCA 18 s, figures 14 s) and 1 s the bundle with reading it back. The memory is that of the analysis ([One entry on a compute node](#one-entry-on-a-compute-node)): one matrix of 8 bytes x retained frames x atom pairs. Extending a stored entry of 120,012 frames with 5,000 new ones took 64 s and 14.4 GiB with 8 workers (346 MB bundle).

`requirements-node.txt` is the part of `requirements.txt` that the node command loads: 13 packages, 36 with what they depend on, 0.87 GB installed (the environment of `requirements.txt` with the test tools is 1.3 GB). The command fails without any one of the 13. Importing everything it loads takes 0.8 s and 0.28 GB; on a trajectory of 480 frames the whole command takes 2 to 4 s and 0.35 GB. `lib/entry_bundle.py`, the receiving side, imports the standard library only.

## Raw trajectory archive

The raw data can be kept in a compact archive instead of multi-model PDB text. One entry is one glycan:

```
<archive>/
  index.json                 one record per entry (names, folders, counts, sizes, SHA-256); "repaired" names
                             the entries of damaged sources
  conversion_report.json     the last converter run; entries that failed are under "failed" with the reason,
                             entries of damaged sources under "repaired"
  <folder>/
    trajectory.xtc           all frames in source order; XTC precision 10000 (0.0001 nm = 0.001 Å)
    topology.pdb             header, first frame and records after the last frame (CONECT, ...), verbatim lines
    structure.mol2           unchanged
    info.json, name.json     unchanged, when the source has them
    manifest.json            GLYCAM name, atoms, frames, box, size and SHA-256 of every file and of the
                             source simulation.pdb, what the verification measured, converter version/date;
                             for a damaged source also "source_irregularities"; for an entry made of
                             several trajectories also "segments", for one that extends another "extends"
```

- **Coordinates are exact.** They go from the PDB columns to the XTC as integers of 0.001 Å, so the archive holds the digits of the source; `-0.000` is stored as `0.000` and counted in the manifest. Step and time of a frame are its index from 0 (the source has no times). The box of each frame is the `CRYST1` in force at that frame, so a file made of several runs written one after the other keeps the box of each run, and the manifest lists the frame at which each starts (as it does for `END` and other records found between frames). The 1 × 1 × 1 Å placeholder cell that PDB writers emit when there is no box is recorded in the manifest and `topology.pdb` but not written as a box.
- **Names.** An entry is identified by its full GLYCAM name: `glycam_name` in the manifest and the index, and `raw/<GLYCAM name>/` as object-storage key prefix (`lib.archive.object_key_prefix`). The name comes from `name.json` when the source folder has one, from `info.json` when the folder is named by a GlyTouCan ID, and is the folder name otherwise. `<folder>` is the GLYCAM name when that is a usable directory name (at most 255 bytes, no `/`); otherwise it is the first 200 bytes of the name, `~`, and the first 16 hex digits of the SHA-256 of the name (`lib.archive.entry_folder_name`). Always read the name from `manifest.json`, not from the folder.
- **Names from outside.** `--names FILE` gives the converter a JSON object of source folder name to GLYCAM name, for example `{"G60371DN": "DNeup5Aca2-3[...]DGlcpNAcb1-OH"}` for a folder whose `info.json` spells the name differently. Such a name takes precedence over `name.json`, `info.json` and the folder name, and the manifest records `"glycam_name_from": "names-file"` (a different name in `name.json` or `info.json` is listed under `warnings`). A folder that is not under `--data-dir` is an error. If the same source folder was converted before under another name, that entry stays and the run ends with a non-zero status (`SUPERSEDED ...` on stderr, `superseded` in the report, and the index reports two entries for one source folder); add `--remove-superseded` to delete it once the new entry is in place. Pass the same `--names` to `--verify-only --data-dir`; without it the name of such an entry is not compared with the source (`glycam_name_checked` is false in the report) and everything else is.
- **Verification.** Before an entry gets its final name the XTC is read back and every coordinate is compared with the source text; `manifest.json` records the number compared, the number that differ (always 0 for a published entry) and the largest difference in Å between the stored float and the decimal number of the text. A sample of frames (`--check-frames`, default 100) is also re-read from the source with plain `float()` as an independent check. A source that cannot be represented is not converted and is listed in the report: a frame with another atom count or other atom records, a truncated file, a coordinate that the XTC does not return exactly (possible only from 5120 Å on).
- **Damaged sources (`--repair`).** Without `--repair` a source has to be regular. With it the converter also archives a source that is damaged in one of these ways, and in no other:
  - *A frame without `ENDMDL`*: the next `MODEL` record, or the end of the file, comes first. Its records have to be those of the first frame, in order, as far as they go, each a whole line; after them there may be one line that is the beginning of the record expected there (the line the cut went through), which never counts as an atom record. With fewer atom records than the first frame the frame is **dropped** (`incomplete_frame`) and every complete frame is kept in source order; with all of them it is **kept** (`frame_without_endmdl`), since only the `ENDMDL` is missing and its coordinates are checked like those of any frame. When one run was cut in the middle of a line and the next run appended, the `CRYST1` of the next run stands on that line (`ATOCRYST1 ...`); it is read as the record it is, so the frames after the cut get their own box. This holds wherever the cut fell in an atom, `TER` or `MODEL` record; text on the line of a whole `ENDMDL` or `END` record is not looked at, with or without `--repair` (a run cut exactly behind that word would lose the `CRYST1` of the next one unnoticed: such a file has one `MODEL` restart more than `CRYST1` records).
  - *A `MODEL` record that no atom record follows* (`model_record_without_frame`), for example at the end of the file.
  - *Frames that `ENDMDL` ends and no `MODEL` record starts* (`frame_without_model_record`).

  Refused with `--repair` too: a frame that `ENDMDL` ends with another atom count, atom records that are not those of the first frame (also in a cut frame), an unreadable coordinate, a first frame that is not complete, atom records with neither `MODEL` before nor `ENDMDL` after them, text other than `CRYST1` on a cut line, and a source in which more than 1000 frames or `MODEL` records are of the first two kinds (each is listed in the manifest). The manifest of a repaired entry has `source_irregularities`: `frames_in_source`, `frames_kept`, `frames_dropped`, `bytes_ignored`, `truncated` (the file ends inside a frame) and `items`, one per frame or record in source order with `kind`, `action` (`dropped`, `kept`, `ignored`), `source_frame` (its index among the frames of the source), `frames_kept_before` (the frames of the archive before it, which is the index of the next frame of the archive), `offset` and `bytes` (its place in `simulation.pdb`), the SHA-256 of those bytes, `bytes_ignored`, `atoms_found` against `atoms_expected`, `cut_line_bytes`, `followed_by`, and for the frame the file ends in `file_truncated` and `file_size`; the frames without `MODEL` record follow as one item with their count, indices and offsets. `frames`, the steps of the XTC, `box.cells[].first_frame` and `records_between_frames.*.before_frame` count the frames of the archive; `warnings` says that the entry was repaired. A manifest without `source_irregularities` is that of a regular source, and a regular source is converted to the same bytes with and without `--repair`. Verification is the same for the frames that are kept, and the record is checked against the file a second way: the `MODEL` and `ENDMDL` records are located by a search of its bytes, their numbers have to be those the record implies, the n-th `ENDMDL` has to lie in the bytes the n-th frame of the archive was read from (frames kept without one aside), and every item is read at its offset (its place among the frames, no `ENDMDL`, SHA-256, number of whole atom records). `index.json` lists the repaired entries under `repaired` and gives the record of each a `repaired` summary; the report has `repaired`, and the run prints a `REPAIRED` line per entry. A rerun skips a repaired entry like any other, with or without `--repair`; `--verify-only --data-dir` reads each source the way its manifest says.
- **MODEL numbers** are not kept: a frame is its position in the trajectory, whatever the number on its `MODEL` record (they wrap, restart with every run, or are missing). The manifest summarises them under `source.model_numbers` (first, last, largest, whether they count up by one); `-1` there stands for a record without a number, and for a repaired source such records are counted under `model_records_without_number` with an example of their text.
- **Several trajectories in one entry.** `lib.archive_segments.convert_segments` writes one entry from the replicas of a run, each a multi-model PDB of its own, and optionally after the frames of an existing entry ([A run on the cluster](#a-run-on-the-cluster) uses it). Every replica is read and verified against its own text as above, in parallel, into a part of the trajectory that carries the step numbers it has in the whole; the parts are appended byte for byte (an XTC has no header and its frames do not refer to each other), so nothing is concatenated as text and the trajectory is the one the converter makes of the concatenated file. The manifest has every key above and in addition `segments`, one record per trajectory in frame order: `name`, `first_frame`, `frames`, `coordinates_sha256` of its frames, its bytes in `trajectory.xtc` (`trajectory_offset`, `trajectory_bytes`), `time_per_frame_ps` (null when unknown), `run` (what the caller knew about the simulation), its `box`, `source` (file name, size, SHA-256, `model_numbers`, ...), its own `verification`, and `source_irregularities` when that replica was archived with repair. `source` of the manifest then describes all of them: `file` is null, `size` the sum, `sha256` the SHA-256 of the SHA-256 values of the source files in order. Every replica has to have the same atoms as the first (the same number, and name, residue name and residue number of every atom, in order), else nothing is written and the message names the first atom that differs; a difference in other columns of the atom records (serial number, chain, element) is a warning and is recorded as `atom_records` of the segment. A replica without `CRYST1` has no box (in a concatenated file the cell of the run before it would stay in force). A manifest without `segments` is that of one `simulation.pdb`; such entries are written exactly as before. Readers need not know about segments: `ArchiveEntry`, the store step, `index.json` and the transfers read the entry like any other, `verify_entry` also checks the coordinates of every segment against its record, and `repair_summary` (and with it `repaired` in the index) adds up the segments that came from a damaged source.
- **Extending an entry.** With `extend=<entry folder>` the frames of that entry come first, as the bytes they are: its `trajectory.xtc` is the beginning of the new one and is not decoded and encoded again, and its `topology.pdb`, `structure.mol2` and `info.json` are kept. The new trajectories have to have its atoms (as above; an error says what differs). The manifest gets `extends` (`glycam_name`, `manifest_sha256` of the extended entry, `frames`, the size and hashes of its trajectory, its `files`, `previous` for an entry that was extended before), and the extended entry becomes the first segment (its segments, if it had any) with `carried_from`; an entry that was repaired keeps its `source_irregularities` in that segment. `lib.archive_segments.check_extension(new, old)` checks an extension against the entry it claims to extend: the SHA-256 of the manifest, the first bytes of the trajectory, and every frame decoded from both.

```bash
# Convert (resumable: entries whose manifest verifies are skipped; the source is only read)
.venv/bin/python scripts/convert_data_to_archive.py --data-dir data --archive-dir archive --workers 8

# Give a source folder its GLYCAM name from outside and drop the entry it had under the old name
.venv/bin/python scripts/convert_data_to_archive.py --data-dir data --archive-dir archive \
  --glycan G60371DN --names names.json --remove-superseded

# Archive damaged sources as well (here from a directory of their own, into an archive that has the other
# entries): index.json is written again for the whole archive, --report leaves its conversion_report.json alone
.venv/bin/python scripts/convert_data_to_archive.py --data-dir damaged --archive-dir archive --workers 3 \
  --repair --report archive/conversion_report.repaired.json

# One 64-core AMD node; DATA_DIR and ARCHIVE_DIR are at the top of the script.
# VERIFY=1 runs the verification below instead, with every frame re-read from the source.
# NAMES_FILE=names.json passes --names, REPAIR=1 passes --repair.
mkdir -p logs/slurm && sbatch slurm/convert_data_to_archive_amd.sh

# Verify an archive: files against manifests, and with --data-dir every coordinate against the source
.venv/bin/python scripts/convert_data_to_archive.py --verify-only --archive-dir archive --data-dir data --workers 8
```

The pipeline reads either layout: point `--data-dir` at the archive and the store step streams frames from `trajectory.xtc` (`lib.glystore.store_from_archive`) with the same result as from `simulation.pdb`; folders with `simulation.pdb` keep working. Archive entries are read from the local filesystem only. In Python, `lib.archive.ArchiveEntry(path).iter_frames(chunk=1024)` yields `(frames, atoms, 3)` float32 arrays in Å, and `lib.archive.verify_entry(path, source_dir)` checks one entry. The XTC is written with MDTraj (`mdtraj==1.10.3` in `requirements.txt`); an existing `.venv` needs `uv pip install -r requirements.txt` once.

## S3 storage (Cloudflare R2)

`GLYCOSHAPE_STORAGE=s3` selects S3-compatible object storage instead of the Oracle PAR bucket. Without it nothing changes: Oracle when `GLYCOSHAPE_ORACLE_PAR_URL` is set, local folders otherwise. `GLYCOSHAPE_STORAGE=local` forces local folders. Put the settings in the environment or in `.env` (not committed); the secret goes to the client only and is never logged. The client is `boto3` (pinned in `requirements.txt`; an existing `.venv` needs `uv pip install -r requirements.txt` once; it adds four packages, 16 MB to download and 31 MB installed, nearly all of it `botocore`).

| Variable | Default | |
|---|---|---|
| `GLYCOSHAPE_STORAGE` | unset | `s3`, `oracle` or `local` |
| `GLYCOSHAPE_S3_ENDPOINT` | required | `https://<account id>.r2.cloudflarestorage.com` |
| `GLYCOSHAPE_S3_ACCESS_KEY_ID`, `GLYCOSHAPE_S3_SECRET_ACCESS_KEY` | required | an R2 API token that may read and write objects in the buckets below |
| `GLYCOSHAPE_S3_REGION` | `auto` | |
| `GLYCOSHAPE_S3_ADDRESSING_STYLE` | `path` | `path`, `virtual` or `auto` |
| `GLYCOSHAPE_S3_RAW_BUCKET`, `GLYCOSHAPE_S3_RAW_PREFIX` | `glycoshape-archive`, `raw` | archive entries `raw/<GLYCAM name>/...` and `raw/index.json` |
| `GLYCOSHAPE_S3_PROCESS_BUCKET`, `GLYCOSHAPE_S3_PROCESS_PREFIX` | `glycoshape-archive`, `process` | kept for `process/<GLYCAM name>/...`; no command uploads there yet |
| `GLYCOSHAPE_S3_STATIC_BUCKET`, `GLYCOSHAPE_S3_STATIC_PREFIX` | `glycoshape-database`, `static` | the published static database |
| `GLYCOSHAPE_S3_STAGING_PREFIX` | `publish-staging` | in the static bucket, outside the static prefix: where changed files wait during a publish |
| `GLYCOSHAPE_S3_MAX_ATTEMPTS` | `8` | attempts per request; connection errors, throttling (HTTP 429 included) and 5xx answers are retried with exponential backoff and jitter |
| `GLYCOSHAPE_S3_MULTIPART_THRESHOLD_MB`, `GLYCOSHAPE_S3_MULTIPART_CHUNK_MB` | `64`, `32` | a larger file is uploaded in parts of this size |
| `GLYCOSHAPE_S3_TRANSFER_CONCURRENCY` | `4` | parts of one file in transfer at the same time |
| `GLYCOSHAPE_S3_CHECKSUMS` | `when_required` | `when_supported` lets the client add a CRC32 to every upload, where the service accepts that |

Every object is uploaded with its SHA-256 as metadata (`x-amz-meta-sha256`); "unchanged" below always means that the stored size and this value are those of the local manifest. Files are streamed, never read into memory.

With S3 selected, `main.py` runs `--mode process-only` and `--mode static-only`, on local folders as always, and refuses the modes that read and publish through the storage layer. Use the staged workflow:

```bash
# Login node: raw entries -> data/ (all entries of raw/index.json, or --glycan '<GLYCAM name>'; repeatable)
./scripts/run_staged_pipeline.sh download --workers 8

# Login node: submit local data/ -> local process/, as for Oracle
sbatch slurm/process_only_amd.sh

# Login node, after every task succeeded: local process/ -> local static/, then publish what changed
./scripts/run_staged_pipeline.sh publish
.venv/bin/python scripts/static_s3.py check
```

`run_staged_pipeline.sh` (and `continue_rebuild_after_slurm.sh`, `publish_after_slurm.sh`) ask `lib/config.py` which storage is selected, so the same commands serve Oracle and S3; the compute-node jobs set `GLYCOSHAPE_STORAGE=local`.

### Raw archive: `scripts/archive_s3.py`

```bash
# Upload an archive directory (the one convert_data_to_archive.py wrote, with its index.json)
.venv/bin/python scripts/archive_s3.py upload --archive-dir archive --workers 8 --dry-run
.venv/bin/python scripts/archive_s3.py upload --archive-dir archive --workers 8
.venv/bin/python scripts/archive_s3.py compare --archive-dir archive        # listing against index.json, any time

# Download: every entry of the stored index, or entries by GLYCAM name
.venv/bin/python scripts/archive_s3.py download --archive-dir data --workers 8
.venv/bin/python scripts/archive_s3.py download --archive-dir data --glycan 'DManpa1-2DManpa1-OH'
```

- **Keys.** `raw/<GLYCAM name>/<file>`, with the name of the entry's `manifest.json`, not of its local folder (which is shortened for long names). `--glycan` takes the GLYCAM name; for `upload` also the entry folder or the source folder.
- **Upload.** A file is skipped when the stored object is unchanged, hashed before it is sent and refused when it is not what its manifest says. The files of an entry go first and its `manifest.json` last, and when a file of an entry that is already stored has to be replaced, the stored manifest is deleted first; so an entry with a manifest in the storage is complete. Afterwards the listing of `raw/` is compared with `index.json` (keys and sizes) and the stored manifest of every entry with the SHA-256 that the index records; `raw/index.json` is uploaded only when every entry of the index is there. The exit status is non-zero when one of the entries that were to be uploaded failed or is not complete in the storage. `index.json` has to describe the archive: run the converter again after changing an entry.
- **Download.** Each file is downloaded into `data/.staging/`, hashed and compared with the manifest; the entry folder (`lib.archive.entry_folder_name`) appears only when all of it verified. Without `--glycan` the stored index says which entries there are and pins the manifest of each; an entry named with `--glycan` is verified against its own manifest only, so it can be fetched while the index is still the one of an earlier upload. The folder is an archive entry that `main.py --mode process-only --data-dir data --glycan <folder>` processes; the command prints it. An entry that is already there with the same manifest and whose files verify is skipped (`--trust-local` compares sizes instead of hashing them again). A folder that is not an archive entry is never written into.
- **Resume** by running the same command again. `--report FILE` writes what was done as JSON. An interrupted multipart upload leaves parts in the bucket that the service removes after its lifecycle period (seven days by default on R2).

### Static database: `scripts/static_s3.py`

```bash
.venv/bin/python scripts/static_s3.py publish --static-dir static --dry-run   # what would be sent
.venv/bin/python scripts/static_s3.py publish --static-dir static --workers 16
.venv/bin/python scripts/static_s3.py check              # read back as the website does: download and hash everything
.venv/bin/python scripts/static_s3.py check --quick      # sizes and SHA-256 metadata only
```

The website reads `static/sync_manifest.json` and then downloads the files it lists, comparing size and SHA-256 of each; one mismatch or missing object fails its run (and leaves its own database alone). `publish` therefore sends only what `sync_manifest.json` lists, refuses a manifest the website would refuse (for example one that lists a hidden file) or that does not describe the local files, and works in this order:

1. **New files** (no object under that key) are uploaded to their keys. No published manifest lists them, so no reader fetches them.
2. **Changed files** are uploaded to `publish-staging/static/<SHA-256>`, outside the published prefix.
3. Changed files are **copied inside the storage** from the staging prefix onto their keys.
4. The listing is compared with the manifest and **`sync_manifest.json` is written last**, as one object.
5. The staging objects are deleted; `--delete-stale` also deletes objects under `static/` that the manifest no longer lists (they are only counted otherwise).

If the upload is interrupted in step 1 or 2, which is where the time goes, the published prefix is exactly what it was: the old manifest and every file it lists. Step 3 cannot be made invisible, because two objects cannot be replaced at once: from the first copy until the manifest is written, the published manifest describes files that have been replaced. A website that already has the published `global_hash` does nothing in that window; one that is behind fails its verification, keeps its database and retries. Run `publish` again after any interruption: it continues from what is stored. Without a published manifest (the first publish) there is no reader, and files are replaced in place.

`publish` exits non-zero when it stopped, and also when the manifest was written but step 5 failed (it says which). `check` exits non-zero and lists the files that are missing or differ. `--quick` trusts the SHA-256 metadata, so it does not notice an object whose content was changed by other means; the full check does.

### One entry: `scripts/publish_entry.py`

A new glycan does not need the whole pipeline on a login node any more. The cluster job analyses the entry ([One entry on a compute node](#one-entry-on-a-compute-node)) and hands the result over as a bundle; the server that approves entries publishes it. Nothing here asks PocketBase: the caller owns the registry of GlycoShape IDs and passes the ID in.

```bash
.venv/bin/python scripts/publish_entry.py publish  --bundle run-17/ --submission run-17.json --dry-run   # what an administrator approves
.venv/bin/python scripts/publish_entry.py publish  --bundle run-17/ --submission run-17.json --expect-hash <hash of the dry run>
.venv/bin/python scripts/publish_entry.py replace  --bundle run-18/ --submission run-18.json              # same ID, new content
.venv/bin/python scripts/publish_entry.py withdraw --id GS01067 --work-dir withdraw-GS01067
.venv/bin/python scripts/publish_entry.py status   --id GS01067 --check
.venv/bin/python scripts/publish_entry.py recover                                                        # finish an interrupted switch
```

**Input.** An extracted, validated bundle directory (`bundle.json` with the size and SHA-256 of every file, `archive/` the raw entry, `static/` the static-ready folder, `result.json`, `run.log`) and a submission record:

```json
{
  "schema": "glycoshape-submission", "schema_version": 1,
  "glycoshape_id": "GS01067",
  "glycam_name": "DManpa1-2DManpa1-OH",
  "simulation": {
    "simulation_length_ns": 2500.0,
    "md_package": "GROMACS", "force_field": "GLYCAM_06j",
    "temperature_k": 300.0, "pressure_bar": 1.0, "salt_concentration_mm": 150.0,
    "source_type": "manual", "comments": ""
  },
  "glytoucan_id": "",
  "curated": {"common_names": [], "aliases": [], "description": "", "keywords": []}
}
```

The simulation fields are named as in `source` of `bundle.json` (`--id GS01067` instead of `--submission` takes them from there). Every quantity has its unit in its name; `simulation_length`, `temperature`, `pressure` or `salt_concentration` without one is refused, because the length has been entered in nanoseconds by one client and in microseconds by another. `simulation_length_us` is accepted in place of `simulation_length_ns`. The units are converted once, in `lib/entry_complete.py`, to those of `data.json`: `length` in microseconds (`simulation_length_ns / 1000`), `temperature` in kelvin, `pressure` in bar, `salt` in mM. The length is the aggregated one of the entry: for a bundle that extends a published entry, `source` states the new run only and the published length has to be added. Values that look like a unit mistake (a length under 10 ns or over 1 ms, a temperature outside 200 to 500 K) are accepted and reported as warnings.

**The glycan folder** is completed in the work folder (`<bundle>.publish`, or `--work-dir`) by `lib.entry_complete.complete_entry`: `data.json` with the search metadata, `snfg.svg`, `glygen.json`, `glycosmos.json`, `output/` and `landscape/` with the REMARK block on the PDB files, and `GSxxxxx.zip`. It is the folder `main.py --mode static-only` writes for the same ID and metadata: both call the same functions (`lib.glystatic.write_identity_files`, `lib.static_ready`, `lib.glymeta`, `lib.glybake.write_archive`). Curated aliases that the names of the entry do not already give are kept under `search_meta.aliases`, a key that only such an entry has.

**Web lookups** (GlyCosmos and GlyTouCan, GlyGen, PubChem) go through `lib/lookups.py`: each request has a timeout (`GLYCOSHAPE_LOOKUP_TIMEOUT`, `--lookup-timeout`), all of them together a budget (`--lookup-budget`, 300 s), and `--offline` (or `GLYCOSHAPE_OFFLINE=1`, also `main.py --offline`) makes none. An entry is published with what was answered. What was asked and what was not answered is in the report, and stays with the entry in the storage (`publish-state/static/entries/GSxxxxx.json`: `lookups`, and `gaps`, the identity fields that are empty), so a later `replace --refresh-metadata` knows what to retry; `status --id` prints it. An entry without WURCS and accession, or without its SNFG image, is refused unless `--allow-incomplete` is given (the static stage calls such a folder incomplete as well).

**The root files** are not built from all glycan folders but derived from the stored ones (`lib/static_delta.py`): `GLYCOSHAPE.json` with one key inserted, replaced or removed; `GlycoShape.zip` with one member more or less; `GLYCOSHAPE_RDF.ttl` converted from the new inventory; the two `missing_*` reports with the lines of the changed glycan exchanged; `sync_manifest.json` with one glycan entry and the entries of these root files changed and `global_hash` recomputed. No other glycan folder is read: the publication works on a bucket that holds only the root objects. The result is byte for byte what the finalization writes for the changed tree, apart from `generated_at` and the `mtime` values of the manifest.

**Checked before anything is written** to the raw or the published prefix (the lock is the one object written before): the hashes of the bundle; that the archive entry, the result and the bundle are of one glycan and that the result was computed from these raw files; the levels (`--levels`) and the pipeline of the bundle (`--accept-pipeline-version`, `--accept-pipeline-commit`, converter version, `--require-clean`); that the ID is not the ID of another glycan and the glycan not published under another ID; for `publish` that the ID is new, for `replace` that it is published; that a withdrawn ID comes back only for its own glycan; that the stored root files are those of the stored manifest; that the stored raw entry is the one the bundle extends, or the same (`--overwrite-raw` for another one); `--expect-hash`. Exit status 2 and every reason, on stderr and in `--report`.

**Order**, and what the website sees:

| Step | What is written | A reader of `static/` sees | Interrupted here |
|---|---|---|---|
| 1 | `raw/<GLYCAM name>/` files, `manifest.json` last; then `raw/index.json` | the old database | the raw entry has no manifest (incomplete) until the run is repeated |
| 2 | files of `static/GSxxxxx/` under keys no stored manifest lists | the old database | objects that nothing lists |
| 3 | replacements (root files, changed files of a replaced glycan) and the new manifest under `publish-staging/static/<SHA-256>` | the old database | staged objects; removed by the next run |
| 4 | `publish-state/static/pending.json`: what is about to be switched | the old database | the next run (or `recover`) completes the switch from the storage |
| 5 | staged files are copied onto their keys inside the storage, `--workers` at a time: first the changed files of the glycan folder, then the root files | old manifest, some files already new: a reader that is behind fails its verification, keeps its database and retries at its next synchronisation; one that is up to date does nothing | as during the step, until the next run or `recover` |
| 6 | `sync_manifest.json` is copied into place, alone, when every copy of step 5 has succeeded | the new database | |
| 7 | stale files of the glycan are deleted (the whole folder for a withdrawal), the record of the entry and `withdrawn.json` are written, the staged objects and then `pending.json` are removed | the new database | objects that nothing lists; removed by the next run |

Every step skips an object that is stored with the same size and SHA-256, and the journal in the work folder (`journal.json`) keeps the completed folder and the derived root files, so the same command continues an interrupted run and publishes the same bytes. Any later command first completes a pending switch. `status` shows the lock, a pending publication and the state of one entry.

**The switch** (steps 5 and 6) is the only time at which a reader can fail, so it is kept short. In a first trial against R2 a copy inside the storage took one to two seconds, and a replacement that switched 23 files one after another kept this window open for 46 s; the copies of a group are therefore made `--workers` at a time. What is kept in order, and why:

- *Before the first copy* every file of the switch is looked at: which ones are not at their keys yet, and that each of those, and the manifest, is staged. A switch that could not be finished from the storage is not begun.
- *The changed files of the glycan folder go first.* A reader fetches the files of a glycan only when it does not have the published folder of that glycan, so while these copies run a website that is behind for another reason (it missed the publication of another glycan) still synchronises, to the old database. Only a reader without the folder (a new website, or one that missed the previous publication of this glycan) fails from here on.
- *The root files go second*, when every copy of the folder has returned. Every reader that is behind fetches every root file, so once the first of these copies is done all of them fail: the group that affects everybody is the last and the short one (three to five files, one round of copies), however many files of the folder there are and however long they take.
- *`sync_manifest.json` goes last and alone*, when every copy before it has succeeded and none is under way. The stored manifest therefore never lists a file that is not at its key: the files it lists that the old one did not were uploaded in step 2, and the files it no longer lists are deleted in step 7.

Inside a group the order is free, so an interruption leaves any subset of the group switched; all of these are the state of step 5 in the table, and whoever runs next looks at every key and makes the copies that are missing. A copy that fails stops the switch in the same way: the copies under way are waited for, the others are not made, the manifest is not written, `pending.json` stays. The window is about ⌈folder files / workers⌉ + ⌈root files / workers⌉ + 1 copies long (in a test storage in which every copy takes 1.5 s, the 24 copies of such a replacement: 7.5 s with 8 workers and 4.5 s with 24, against 36 s one after another), and the last two terms bound the part in which every reader that is behind fails. The journal (`steps.static`) and the report (`switch`) record both: `switched` (the copies made, without the manifest), `switch_seconds` (from the first copy until the manifest is in place) and `switch_root_seconds` (the same from the moment the root files began to be copied).

**Replace.** Files that did not change are not sent. A bundle whose `extends` names the stored raw manifest replaces the raw entry in place: the previous trajectory is dropped, since the new one holds its frames, and the previous `manifest.json` is kept under `raw-history/<GLYCAM name>/<its SHA-256>/` (the record of what the earlier publication was computed from). A raw entry that the bundle does not extend is replaced only with `--overwrite-raw`, and then all of its files are copied to the history prefix first (`--keep-previous-raw`).

**Withdraw** publishes a manifest without the entry first, then deletes `static/GSxxxxx/`. The raw entry stays. The ID is recorded in `publish-state/static/withdrawn.json`: it can come back for the same glycan and is refused for any other.

**Two publishers.** One process publishes at a time; a single service serialises its publications this way. Against a second process there are a lock object (`publish-state/static/lock.json`: owner, host, process, expiry; exit status 3 for whoever finds it), the claim of `pending.json` before the switch, and a comparison of the stored manifest with the one the change was derived from. Where the storage honours conditional writes (`If-None-Match`; the tool probes for it and says when it does not) a free lock and `pending.json` are created by one publisher only, and no change is lost. Where it does not, they are written and read back, which does not separate two publishers that take the same step at the same moment. A lock is taken over when it has expired (`--lock-ttl`, one hour), when its process on this machine no longer exists, or with `--break-lock`. `scripts/static_s3.py publish` and `scripts/archive_s3.py upload` do not take the lock and must not run during a publication. Once entries are published this way the storage is the database: `static_s3.py publish` from a local tree that does not have them would publish a manifest without them (and delete their folders with `--delete-stale`), and `archive_s3.py upload` from a local archive without them would write an `index.json` that no longer lists them (their objects stay and are reported as extra). Bring a local tree up to date from the storage before using either.

**Cost.** Adding one glycan (231 atoms, 40,004 frames; bundle 82 MB) to the published database of 1,001 glycans, against a local MinIO that held only the root objects: 17 s for the dry run (1.1 GB peak memory, of which the RDF conversion needs most; 127 MB downloaded) and 5 s for the publication that follows it in the same work folder (229 MB uploaded: 50 MB raw entry, 32 MB folder, 139 MB root files and 7 MB manifest; 146 MB copied inside the storage; the switch took 0.5 s). `--cache-dir` keeps the root files of the last publication, so the next one downloads only the manifest.

#### GlycoShape IDs

`lib/id_registry.py` has what the registry of the caller needs and stores nothing itself: `next_id(issued, inventory)` is one more than the highest ID ever issued or published (GS01067 today: the old registry issued up to GS01066 while GS01063 is the highest published one; a gap is never filled), and `seed_registry(inventory, issued, names)` pairs GLYCAM names and IDs from `GLYCOSHAPE.json` and the records of the old registry and lists what a person has to decide (a name with several IDs, an ID with several names, records without an ID). The key is the full GLYCAM name, `archetype.name` of the inventory; `archetype.glycam` lacks the last five characters of it, the anomer, position and `-OH` of the reducing end.

#### Reproducible root files

`GSxxxxx.zip` and `GlycoShape.zip` are written with their members in the order of their names and one fixed time stamp (`lib.glybake.write_archive`), so an archive is a function of what it holds. An archive that already holds exactly the files of its folder is kept as it is: the archives of the published database keep their bytes until the content of their folder changes. `GlycoShape.zip` is always written anew, in the new form. `GLYCOSHAPE_RDF.ttl` is a function of `GLYCOSHAPE.json` (blank nodes are numbered in the order of the input). `archetype.termini` of `data.json` keeps the order glycowork returns (it came from a set before and changed from run to run). What remains outside any hash are the times in `sync_manifest.json`.

### Tests

`python -m pytest tests` runs the unit tests of the storage, the transfers and the publish order on an in-memory client. The integration tests need boto3 and a local S3-compatible server and are skipped without one: a MinIO container is started for the session when Docker has a MinIO image (it is removed afterwards and nothing is pulled), and the moto server when `moto[s3]`, `flask` and `flask-cors` are installed. MinIO refuses a key with a segment longer than 255 bytes, so the 267-character GLYCAM name is tested against moto. `GLYCOSHAPE_TEST_S3_ENDPOINT`, `GLYCOSHAPE_TEST_S3_ACCESS_KEY_ID` and `GLYCOSHAPE_TEST_S3_SECRET_ACCESS_KEY` name another throwaway server instead (see `tests/conftest.py`).

## Login Node Run

Run from the checkout on the login node:

```bash
cd /scratch/os1e24/GlycanAnalysisPipeline
./install.sh --run -- --mode incremental
```

To run and safely close SSH:

```bash
./install.sh --run --background -- --mode incremental
tail -f logs/GlycoShape_GAP.log

# Normal (cleans up after upload)
./install.sh --run --background -- --mode incremental

# Keep everything local
./install.sh --run --background -- --mode incremental --keep-local
```

The background launcher writes the process ID to `logs/GlycoShape_GAP.pid`. The old automatic poweroff behavior has been removed.

## SPARQL Endpoint 🕸️

```bash

# Set or override the RDF folder (defaults to GLYCOSHAPE_OUTPUT_DIR/GLYCOSHAPE_RDF)
export GLYCOSHAPE_RDF_DIR="${GLYCOSHAPE_RDF_DIR:-${GLYCOSHAPE_OUTPUT_DIR}/GLYCOSHAPE_RDF}"

# Load and serve the RDF dataset
oxigraph load --location "$GLYCOSHAPE_RDF_DIR" --file "$GLYCOSHAPE_OUTPUT_DIR/GLYCOSHAPE_RDF.ttl"
oxigraph serve --location "$GLYCOSHAPE_RDF_DIR"
```

If your GLYCOSHAPE_RDF folder is in a different path, set GLYCOSHAPE_RDF_DIR to that path before running the commands.

This will host the SPARQL endpoint at http://localhost:7878

## Deactivation 📴

To deactivate the virtual environment:
```bash
deactivate
```

## Citation 📚🔬

All data are freely available for academic use under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0) license. For commercial licensing inquiries, contact: elisa.fadda@soton.ac.uk.

**🔔 Published in Nature Methods (2024)** — please cite the work as:

Callum M. Ives, Ojas Singh, Silvia D’Andrea, Carl A. Fogarty, Aoife M. Harbison, Akash Satheesan, Beatrice Tropea & Elisa Fadda. Restoring Protein Glycosylation with GlycoShape. Nature Methods (2024). https://doi.org/10.1038/s41592-024-02464-7

Preferred short citation: Ives CM, Singh O, et al., Nat Methods (2024).
