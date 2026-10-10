# GlycanAnalysisPipeline 🧬

Glycan conformations analysis pipeline for [GlycoShape Database](https://glycoshape.org).

<img src="docs/Figure.jpg" alt="Schematic overview of Glycan Analysis Pipeline (GAP) used to build the GlycoShape Glycan Database (GDB). Panel a) Multiple uncorrelated replica molecular dynamics (MD) simulations are performed for each glycan in the GDB, to comprehensively sample its structural dynamics. The resulting MD frames are then transformed into a graph matrix representation, as depicted in Panel b), simplified by flattening the lower half as shown in Panel c). This step enables a dimensionality reduction via principal component analysis (PCA), shown in Panel d). These data are clustered by Gaussian Mixture Model (GMM) and the results of which are displayed in terms of cluster distributions, see Panel e). Panel f) Representative 3D structures for each cluster are selected based on KDE maxima, along with comprehensive torsion angle profiles for the highest populated clusters, showing the wide breadth of the conformational space covered by GAP. Panel g) Structures derived from GAP are clearly presented on the GlycoShape GDB web platform, in addition to biological and chemical information." style="zoom: 33%;" />

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
used (message printed; nothing is computed).

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
above), or copy `static/` to `static/GSxxxxx/` and add the files above.
`lib/static_ready.py` is the single implementation of the shared files; the
static stage calls it. The static stage takes the GLYCAM name from the name of
the process folder, which for an archive entry with a long name is the
shortened folder name: the name to publish under is `glycam_name` in
`result.json`.

The analysis of one entry needs memory for one matrix of 8 bytes x frames x
atom pairs of the PCA landscape (pairs of the heavy atoms outside the reducing
`ROH`, at most 300 atoms): 18.8 GB for the 159,940 retained frames of a
329-atom glycan (172 such atoms, 14,706 pairs), 0.8 GB for 50,005 frames of a
97-atom one (62 atoms, 1,891 pairs). Everything else is small next to it.

## Raw trajectory archive

The raw data can be kept in a compact archive instead of multi-model PDB text. One entry is one glycan:

```
<archive>/
  index.json                 one record per entry (names, folders, counts, sizes, SHA-256)
  conversion_report.json     the last converter run; entries that failed are under "failed" with the reason
  <folder>/
    trajectory.xtc           all frames in source order; XTC precision 10000 (0.0001 nm = 0.001 Å)
    topology.pdb             header, first frame and records after the last frame (CONECT, ...), verbatim lines
    structure.mol2           unchanged
    info.json, name.json     unchanged, when the source has them
    manifest.json            GLYCAM name, atoms, frames, box, size and SHA-256 of every file and of the
                             source simulation.pdb, what the verification measured, converter version/date
```

- **Coordinates are exact.** They go from the PDB columns to the XTC as integers of 0.001 Å, so the archive holds the digits of the source; `-0.000` is stored as `0.000` and counted in the manifest. Step and time of a frame are its index from 0 (the source has no times). The box of each frame is the `CRYST1` in force at that frame, so a file made of several runs written one after the other keeps the box of each run, and the manifest lists the frame at which each starts (as it does for `END` and other records found between frames). The 1 × 1 × 1 Å placeholder cell that PDB writers emit when there is no box is recorded in the manifest and `topology.pdb` but not written as a box.
- **Names.** An entry is identified by its full GLYCAM name: `glycam_name` in the manifest and the index, and `raw/<GLYCAM name>/` as object-storage key prefix (`lib.archive.object_key_prefix`). The name comes from `name.json` when the source folder has one, from `info.json` when the folder is named by a GlyTouCan ID, and is the folder name otherwise. `<folder>` is the GLYCAM name when that is a usable directory name (at most 255 bytes, no `/`); otherwise it is the first 200 bytes of the name, `~`, and the first 16 hex digits of the SHA-256 of the name (`lib.archive.entry_folder_name`). Always read the name from `manifest.json`, not from the folder.
- **Verification.** Before an entry gets its final name the XTC is read back and every coordinate is compared with the source text; `manifest.json` records the number compared, the number that differ (always 0 for a published entry) and the largest difference in Å between the stored float and the decimal number of the text. A sample of frames (`--check-frames`, default 100) is also re-read from the source with plain `float()` as an independent check. A source that cannot be represented is not converted and is listed in the report: a frame with another atom count or other atom records, a truncated file, a coordinate that the XTC does not return exactly (possible only from 5120 Å on).

```bash
# Convert (resumable: entries whose manifest verifies are skipped; the source is only read)
.venv/bin/python scripts/convert_data_to_archive.py --data-dir data --archive-dir archive --workers 8

# One 64-core AMD node; DATA_DIR and ARCHIVE_DIR are at the top of the script.
# VERIFY=1 runs the verification below instead, with every frame re-read from the source.
mkdir -p logs/slurm && sbatch slurm/convert_data_to_archive_amd.sh

# Verify an archive: files against manifests, and with --data-dir every coordinate against the source
.venv/bin/python scripts/convert_data_to_archive.py --verify-only --archive-dir archive --data-dir data --workers 8
```

The pipeline reads either layout: point `--data-dir` at the archive and the store step streams frames from `trajectory.xtc` (`lib.glystore.store_from_archive`) with the same result as from `simulation.pdb`; folders with `simulation.pdb` keep working. Archive entries are read from the local filesystem only. In Python, `lib.archive.ArchiveEntry(path).iter_frames(chunk=1024)` yields `(frames, atoms, 3)` float32 arrays in Å, and `lib.archive.verify_entry(path, source_dir)` checks one entry. The XTC is written with MDTraj (`mdtraj==1.10.3` in `requirements.txt`); an existing `.venv` needs `uv pip install -r requirements.txt` once.

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
