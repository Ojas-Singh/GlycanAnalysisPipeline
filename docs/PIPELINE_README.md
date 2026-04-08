# Updated Glycan Analysis Pipeline

## Overview

The updated `main.py` provides a comprehensive pipeline that processes glycans through multiple analysis stages with an improved programmatic API:

1. **v2 Analysis**: Individual glycan analysis including PCA, clustering, and torsion analysis
2. **glystatic**: Generate static database entries with structural and metadata information  
3. **glymeta**: Update metadata and search terms for each glycan
4. **glybake**: Create final database archives and consolidated files

## Usage

### Process All Glycans
```bash
python main.py
```

### Process with Force Update
```bash
python main.py --update
```

### Check Status of All Glycans
```bash
python main.py --status
```

### Process Single Glycan (for testing)
```bash
python main.py --glycan "DManpa1-2DManpa1-OH"
```

### Check Status of Single Glycan
```bash
python main.py --glycan "DManpa1-2DManpa1-OH" --status
```

### Custom Directories
```bash
python main.py --data-dir /path/to/input --output-dir /path/to/output
```

## Command Line Arguments

- `--data-dir`: Directory containing glycan input data (overrides config)
- `--output-dir`: Directory for database output (overrides config)  
- `--update`: Force recomputation of existing results
- `--glycan`: Process only a specific glycan (for testing)
- `--status`: Check status instead of running pipeline

## New v2 API

The v2 module now provides a clean programmatic API:

### Class-Based Interface
```python
from lib.v2 import GlycanAnalysisPipeline

# Initialize pipeline
pipeline = GlycanAnalysisPipeline(data_dir="/path/to/data")

# Run analysis
results = pipeline.run_analysis("DManpa1-2DManpa1-OH", force_update=False)

if results['success']:
    print(f"Completed steps: {results['steps_completed']}")
else:
    print(f"Failed: {results['error_message']}")

# Check status
status = pipeline.get_analysis_status("DManpa1-2DManpa1-OH")
print(f"Input files exist: {status['input_files_exist']}")
for step, completed in status['steps_status'].items():
    print(f"{step}: {'✓' if completed else '✗'}")
```

### Convenience Functions
```python
from lib.v2 import run_glycan_analysis, get_glycan_status

# Run analysis
results = run_glycan_analysis("DManpa1-2DManpa1-OH", "/path/to/data")

# Check status
status = get_glycan_status("DManpa1-2DManpa1-OH", "/path/to/data")
```

### v2 Command Line (Enhanced)
```bash
# Run analysis
python -m lib.v2 --name "DManpa1-2DManpa1-OH"

# Check status only
python -m lib.v2 --name "DManpa1-2DManpa1-OH" --status

# Force update
python -m lib.v2 --name "DManpa1-2DManpa1-OH" --update
```

## Pipeline Steps

### For Each Glycan:

1. **Discovery**: Automatically discovers glycan directories in `data_dir`
   - Requires `{glycan_name}.pdb` and `{glycan_name}.mol2` files

2. **v2 Analysis**: 
   - **Smart Skipping**: Checks completion status and skips if already done (unless `--update`)
   - **Detailed Progress**: Reports which steps completed/failed
   - **Robust Error Handling**: Continues even if some steps fail
   - Generates trajectory data, PCA, clustering results
   - Calculates torsion angles and statistics
   - Creates representative structures with alpha/beta variants
   - Outputs to `{glycan_name}/output/` with hierarchical structure

3. **glystatic Processing**:
   - Skips if valid `data.json` already exists (unless `--update` used)
   - Generates structural data and metadata
   - Creates database entry in output directory

4. **glymeta Processing**:
   - Updates search metadata and keywords
   - Adds common names and glycan classification
   - Enhances `data.json` with search capabilities
   - Prefills the PocketBase `glycans` collection when configured
   - Derives `name_variants` from archetype, alpha, and beta naming forms for cross-variant search

### Database Finalization:

5. **glybake Operations**:
   - Validates metadata across all processed glycans
   - Creates individual glycan archives
   - Generates master `GlycoShape.zip` archive
   - Creates consolidated `GLYCOSHAPE.json`
   - Generates FAQ and missing GlyTouCan ID files

## Improved Features

### Intelligent Status Checking
- **v2 Steps**: Checks completion of store, PCA, clustering, torsions, structures, export
- **glystatic**: Validates data.json structure and required fields
- **glymeta**: Checks for search_meta section in data.json

### Better Error Handling
- **Graceful Failures**: Individual glycan failures don't stop the pipeline
- **Partial Completion**: Reports which steps succeeded before failure
- **Detailed Logging**: Comprehensive progress and error information

### Smart Skipping Logic
- **v2 Analysis**: Checks multiple output files to determine completion
- **glystatic**: Validates existing data.json structure
- **Force Override**: `--update` flag forces recomputation

## Directory Structure

### Input Structure (`data_dir`):
```
data_dir/
├── DManpa1-2DManpa1-OH/
│   ├── DManpa1-2DManpa1-OH.pdb
│   ├── DManpa1-2DManpa1-OH.mol2
│   └── [other files...]
└── [other glycans...]
```

### Output Structure (`output_dir`):
```
output_dir/
├── GS00445/
│   ├── data.json
│   ├── snfg.svg
│   ├── output/
│   │   ├── level_1/
│   │   │   ├── PDB/
│   │   │   │   ├── alpha/
│   │   │   │   └── beta/
│   │   │   ├── CHARMM/
│   │   │   ├── GLYCAM/
│   │   │   └── dist.svg
│   │   ├── level_2/
│   │   └── [other levels...]
│   │   ├── pca.csv
│   │   ├── torsion_glycosidic.csv
│   │   └── torparts.npz
│   ├── PDB_format_HETATM/
│   ├── CHARMM_format_HETATM/
│   ├── GLYCAM_format_HETATM/
│   ├── [ATOM format directories...]
│   └── GS00445.zip
├── [other glycans...]
├── GlycoShape.zip
├── GLYCOSHAPE.json
├── faq.json
└── missing_glytoucan.txt
```

## Testing

### Comprehensive Test Suite
```bash
# Run all tests
python test_pipeline.py

# Test just v2 API
python -c "from test_pipeline import test_v2_api; test_v2_api()"
```

### Status Monitoring
```bash
# Check completion rates across all glycans
python main.py --status

# Monitor specific glycan
python main.py --glycan "DManpa1-2DManpa1-OH" --status
```

## Pipeline Statistics

The pipeline provides detailed statistics including:
- Total glycans processed
- Success/failure rates per step
- Overall completion statistics
- List of failed glycans with specific error details
- Step-by-step completion tracking

## Configuration

The pipeline uses settings from `lib/config.py`:
- `data_dir`: Input directory path
- `output_path`: Output directory path  
- `inventory_path`: Glycan inventory CSV path
- Other configuration parameters

These can be overridden using command line arguments.

## API Integration Benefits

1. **No sys.argv Manipulation**: Clean programmatic interface
2. **Detailed Return Values**: Comprehensive results and error information
3. **Status Checking**: Built-in completion status validation
4. **Better Error Handling**: Structured error reporting
5. **Flexible Usage**: Can be used as library or command-line tool
