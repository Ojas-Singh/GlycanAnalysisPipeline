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
export GLYCOSHAPE_UPLOAD_WORKERS=1
export GLYCOSHAPE_STORE_FRAMES_BUFFER=128
export GLYCOSHAPE_STORE_FRAME_CHUNK=128
export POCKETBASE_URL=http://localhost:8090
export POCKETBASE_TOKEN=your_pocketbase_token
```

Ensure the required directories and files exist, or modify the paths as needed.

With `GLYCOSHAPE_ORACLE_PAR_URL` set, the pipeline reads input data from the bucket `data` prefix, stages process files locally under `process/`, writes final static files locally under `static/`, and uploads generated artifacts back to the bucket `process/` and `static/` prefixes. The serving machine should mirror the bucket `static/` prefix independently; this pipeline no longer assumes the serving instance can directly read `GLYCOSHAPE_OUTPUT_DIR`.

For the current login-node limits, keep worker fan-out low. The defaults in `.env` use one worker and smaller frame buffers so PCA, clustering, and uploads run without changing the analysis methods.

When `POCKETBASE_URL` and either `POCKETBASE_TOKEN` or `POCKETBASE_ADMIN_TOKEN` are set, the pipeline uses PocketBase `glycan_submission` records as the submission metadata source for `glystatic` and related metadata helpers. The CSV inventory remains the fallback when PocketBase is unset, unavailable, or missing a record.

## Running 🚀

```bash
python main.py --mode incremental
```

Run modes:

- `incremental` (default): new or incomplete glycans run fully; completed glycans skip v2/process/glystatic and only sync static locally when needed.
- `refresh-static`: completed glycans sync static locally and refresh metadata without process downloads.
- `fresh`: recomputes v2 and glystatic even when bucket outputs already exist. `--update` is kept as an alias for this mode.

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
