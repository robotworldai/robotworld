# Analysis and Results Parsing

This page covers the scripts in `analysis/` for summarizing, comparing, and auditing experiment results after evaluation runs. For the output directory structure, HDF5 layout, and episode result fields, see [Data Storage and Output](data.md).

## `analysis/read_results.py`

The primary script for reading and summarizing experiment results from `episode_results.jsonl` (or legacy `.json`). It supports multiple summarization modes, filtering, CSV export, and multi-folder aggregation.

### Basic Usage

```bash
python analysis/read_results.py <folder> [<folder> ...]
```

`<folder>` can be:
- A folder name relative to the default output directory (e.g., `2025-09-02_13-15-34`)
- An absolute path (e.g., `/data/experiments/my_run`)
- A glob pattern (e.g., `pi0_*`), which prompts for confirmation before proceeding

Multiple folders can be passed to aggregate results across runs.

### Summarization Modes

By default, the script prints a per-task summary table with success rate, score, and trajectory metrics. Additional modes provide different views of the same data:

| Flag | Description |
|------|-------------|
| *(default)* | Per-task table with success/failure counts, percentages, scores, and trajectory metrics |
| `--by-attributes` | Groups tasks by benchmark categories (visual, relational, procedural) with attribute breakdowns |
| `--by-difficulty` | Summarizes results grouped by difficulty label (simple, moderate, complex) |
| `--by-scene` | Aggregates results by scene instead of by task |
| `--by-wrong-objects` | Per-task breakdown of wrong object grasps: success count, fail count, and which objects were grabbed |
| `--by-instruction-type` | Pivot table comparing success rates across instruction types (default, vague, specific, etc.) |
| `--show-episodes` | Appends a detailed per-episode table after the summary |

### Filtering

| Flag | Description |
|------|-------------|
| `--task TASK [TASK ...]` | Show only the specified task name(s) |
| `--filter-pattern PATTERN` | Glob-style pattern to filter results (e.g., `pick_*`, `*cube*`) |
| `--filter-field FIELD` | Field to apply the filter on. Default: `env_name`. Other options: `task_name`, `scene`, `attributes` |

### Output Format

| Flag | Description |
|------|-------------|
| `--csv` | Print results in CSV format (tab-separated) for copy-pasting into spreadsheets |
| `--show-stddev-compact` | Stddev shown inline as `value (± stddev)` instead of separate columns. CSV mode only; behaves like `--show-stddev` in non-CSV mode (implies `--csv`) |
| `--output-csv FILE` | Write CSV output to a file instead of stdout. If the path is relative, it is placed inside the first data folder (implies `--csv`) |

### Display Options

| Flag | Description |
|------|-------------|
| `--verbose` | Show all metrics + stddev columns (equivalent to `--metrics all --show-stddev`) |
| `--metrics NAME [NAME ...]` | Pick which optional columns to show. Names: `score`, `time`, `wrongobj`, `sparc`, `pathlen`, `speed`, `timing`, `succ-eps`, `all`. Default: `score time sparc pathlen speed`. Note: `sparc`/`pathlen`/`speed` are grouped — selecting any one shows all three. |
| `--show-stddev` | Show stddev as separate columns next to value columns |
| `--exclude-containers` | Exclude container objects (bin, crate, box, etc.) from wrong-object-grabbed counts |

The success rate is always shown alongside its 95% Beta-posterior credible interval (`[lcb-ucb]` in human-readable mode; `LCB %` and `UCB %` columns in CSV mode). The interval comes from `Beta(k+1, n-k+1)` with a uniform prior — wide at small N (e.g. 10/10 → `[71.5-99.8]`), tight at large N. See [Statistical Significance and Adaptive Sampling](statistical_significance.md) for details and for the `--num-episodes-adaptive` stopping rule that targets a fixed CI width.

### Examples

```bash
# Basic summary for a single run
python analysis/read_results.py 2025-09-02_13-15-34

# Verbose summary with all details
python analysis/read_results.py 2025-09-02_13-15-34 --verbose

# Aggregate results across multiple runs
python analysis/read_results.py pi0_run1 pi0_run2 pi0_run3

# Aggregate with glob pattern
python analysis/read_results.py "pi0_*"

# Filter to specific tasks
python analysis/read_results.py 2025-09-02_13-15-34 --task RubiksCubeTask BananaInBowlTask

# Filter by env_name pattern
python analysis/read_results.py 2025-09-02_13-15-34 --filter-pattern "*cube*"

# Group results by benchmark category
python analysis/read_results.py 2025-09-02_13-15-34 --by-attributes

# Compare instruction types
python analysis/read_results.py 2025-09-02_13-15-34 --by-instruction-type

# Export to CSV file
python analysis/read_results.py 2025-09-02_13-15-34 --output-csv summary.csv

# Compact CSV for spreadsheets (stddev inline as 'value (± stddev)')
python analysis/read_results.py 2025-09-02_13-15-34 --csv --show-stddev-compact

# Summary with only score and time columns (no trajectory metrics)
python analysis/read_results.py 2025-09-02_13-15-34 --metrics score time

# Show all columns + stddev
python analysis/read_results.py 2025-09-02_13-15-34 --metrics all --show-stddev

# Wrong object analysis, excluding containers
python analysis/read_results.py 2025-09-02_13-15-34 --by-wrong-objects --exclude-containers
```

### Sample Output

The default output includes the success rate, its 95% Beta credible interval, and trajectory metrics columns (EE SPARC, Path Length, Speed):

```
---------------------------------------------- EXPERIMENT SUMMARY ----------------------------------------------
Task Name                Success    %     95% CI         Score(total) Score(fail) Time(s) EE SPARC PathLen(m) Speed(cm/s)
----------------------------------------------------------------------------------------------------------------
TOTAL (2 tasks)          6/20      30.0%  [13.7-50.7]    0.400        0.143       65.59   -12.86   7.33       2.9
----------------------------------------------------------------------------------------------------------------
AnimalsInBinTask         0/10      0.0%   [0.2-28.5]     0.000        0.000       -       -7.49    2.02       2.2
AppleAndYogurtInBowlTask 6/10      60.0%  [30.8-83.3]    0.800        0.500       65.59   -18.23   12.63      3.5
----------------------------------------------------------------------------------------------------------------
```

Score columns:
- **`Score(total)`**: mean per-episode score across all episodes (successes contribute 1.0; failures contribute their fractional subtask progress in `[0, 1)`).
- **`Score(fail)`**: mean per-episode score over failed episodes only — "how close did the failures get."

`Score(total) = success_rate + (1 − success_rate) · Score(fail)`.

`EE SPARC` is the spectral arc length (smoothness) metric; more negative = less smooth. Stationary trajectories return NaN and are excluded from the average. Use `--metrics score time` (or any subset omitting `sparc`/`pathlen`/`speed`) to hide the trajectory metrics columns.

---

## `analysis/check_results.py`

Validates that episode results are consistent with `run_*.hdf5` files — checks that every episode entry has a matching demo in the HDF5, and reports missing or corrupt data.

**Usage:**
```bash
python analysis/check_results.py <folder> [<folder> ...] [--verbose] [--diagnose]
```

**Arguments:**
| Flag | Description | Default |
|------|-------------|---------|
| `folder` (positional) | Folder(s) or absolute path(s) containing results | *(required)* |
| `--verbose` | Print status for every episode, not only errors | `False` |
| `--diagnose` | Extra HDF5 diagnostics (available demos, numbering gaps, etc.) | `False` |

**Example:**
```bash
# Quick sanity check
python analysis/check_results.py 2025-09-02_13-15-34

# Full diagnostics
python analysis/check_results.py 2025-09-02_13-15-34 --verbose --diagnose
```

---

## `analysis/compile_results.py`

Compile and merge experiment results. Supports two modes:

### Mode 1: Compile results to a single file

Reads `episode_results.jsonl` (or legacy `.json`) from one or more folders and writes a single output file.

```bash
python analysis/compile_results.py "pi05_batch*" -o results.jsonl
python analysis/compile_results.py "pi05_batch*" -o results.json   # JSON array format
python analysis/compile_results.py "pi05_batch*" -o results        # defaults to .jsonl
```

### Mode 2: Merge folders

Moves task subdirectories and merges results into a single output folder. Aborts if any task folder appears in multiple sources (conflict). Source folders are removed after merge by default.

```bash
python analysis/compile_results.py "pi05_batch*" --merge output_folder
python analysis/compile_results.py "pi05_batch*" --merge output_folder --keep  # preserve sources
```

**Arguments:**
| Flag | Description | Default |
|------|-------------|---------|
| `folders` (positional) | Folders to compile/merge (glob patterns supported) | *(required)* |
| `-o` / `--output` | Output file path (compile mode). Extension determines format. | — |
| `--merge` | Output folder path (merge mode). Moves task folders + merges results. | — |
| `--keep` | Keep source folders after merge | `False` (remove) |
| `-y` / `--yes` | Skip confirmation when globs expand to many folders | `False` |
| `--task FILTER` | Filter episodes (e.g., `wrong object`) | `None` |

**Examples:**
```bash
# Compile batch results into one file
python analysis/compile_results.py run_1 run_2 run_3 -o combined.jsonl

# Merge batch folders into one folder
python analysis/compile_results.py "pi05_batch*" --merge pi05_merged
```

---

## `analysis/extract_initial_poses.py`

Extracts initial camera and object poses from HDF5 files and writes `episode_initial_poses.json`. Useful for analyzing pose distributions or debugging scene initialization.

**Usage:**
```bash
python analysis/extract_initial_poses.py <folder> [<folder> ...]
```

**Arguments:**
| Flag | Description | Default |
|------|-------------|---------|
| `folder` (positional) | Folder(s) or absolute path(s) containing results | *(required)* |
| `--overwrite` | Recompute even if `episode_initial_poses.json` exists | `False` |
| `--csv` | CSV-style output | `False` |
| `--summary` | Summary table (counts) instead of per-episode detail | `False` |
| `--all` | Include all pose columns (all cameras/objects) | `False` |
| `--compact` | Compact poses (xyz only, no orientation) | `False` |
| `--output-file FILE` | Write CSV to this path instead of stdout | `None` |

**Example:**
```bash
# Extract poses and print summary
python analysis/extract_initial_poses.py 2025-09-02_13-15-34 --summary

# Export all poses as CSV
python analysis/extract_initial_poses.py 2025-09-02_13-15-34 --csv --all --output-file poses.csv
```

---

## `scripts/read_subtask_status_from_hdf5.py`

Reads and displays subtask completion status directly from an HDF5 data file. Extracts timing, status codes, completion flags, and scores for each subtask step during episode execution.

**Usage:**
```bash
python scripts/read_subtask_status_from_hdf5.py <hdf5_file> [-e EPISODE]
```

**Arguments:**
| Flag | Description | Default |
|------|-------------|---------|
| `file` (positional) | Path to the HDF5 data file | *(required)* |
| `-e` / `--episode` | Episode index (e.g., `0` for `demo_0`). If omitted, shows all episodes | `None` |

**Example:**
```bash
# Display all episodes
python scripts/read_subtask_status_from_hdf5.py output/2025-09-02_13-15-34/RubiksCubeTask/run_0.hdf5

# Display specific episode
python scripts/read_subtask_status_from_hdf5.py output/2025-09-02_13-15-34/RubiksCubeTask/run_0.hdf5 -e 0
```

---

## See Also

- [Data Storage and Output](data.md) — Output directory structure, HDF5 layout, and episode result fields
