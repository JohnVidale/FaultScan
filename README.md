# FaultScan

FaultScan analyzes dense local seismic-array recordings from the 2022 Anza/San
Jacinto earthquake sequence. It aligns three-component waveforms, measures
station and event timing corrections, builds coherent stacks, and produces
plots and workbooks for evaluating subtle changes near the S wave.

The long-term scientific objective is to determine whether the subsurface near
the San Jacinto fault changed across the M3.4 mainshock and during its early
aftershock sequence. The current alignment, relocation, screening, and static
correction work is intended to separate possible temporal structural changes
from event-location, origin-time, station, site, and processing effects.

## Main workflow

1. `align_stack.py` reads `rp_input.json`, the event catalog, station metadata,
   and waveform snippets or long traces.
2. `align_utils.py` supplies waveform reading, station rejection, horizontal
   rotation, TauP travel times, staged alignment, screening, and stacking.
3. DP1/DP2 horizontals are rotated to radial (R) and transverse (T) components
   using the source-receiver geometry and the project's orientation correction.
4. Traces are aligned relative to predicted P- or S-wave timing, screened by
   windowed correlation, and combined into event stacks.
5. The pipeline writes per-event products, station residuals, all-event stack
   plots, alignment workbooks, and a snapshot of the run parameters.
6. The post-processing programs summarize station statics, compare components,
   and optionally apply validated event-shift residuals to the catalog.

The pipeline currently uses the shared catalog column `time shift` for event
origin corrections on all waveform components. Event-stack cross-correlation
is a separate optional residual-alignment stage.

## Repository layout

| Path | Purpose |
| --- | --- |
| `align_stack.py` | Main waveform alignment, screening, stacking, plotting, and export pipeline. |
| `align_utils.py` | Shared waveform, geometry, travel-time, alignment, and plotting helpers. |
| `rp_input.json` | Active run configuration loaded automatically by `align_stack.py`. |
| `TASKS.md` | Curated queue of active, upcoming, waiting, proposed, and recently closed work. |
| `plot_statics_by_station.py` | Robust event-baseline correction, station-static estimation, plots, and Excel summaries. |
| `augment_stations_with_statics.py` | Merges robust station-static estimates into a station workbook. |
| `plot_event_rt_snippets.py` | Produces distance-ordered, overlaid radial/transverse snippet figures for one event. |
| `plot_event_time_shifts.py` | Compares radial- and transverse-derived event shifts stored in the catalog. |
| `plot_station_static_basis_comparison.py` | Compares R- and T-basis station-static estimates for each waveform component. |
| `plot_station_component_static_comparison.py` | Compares Z, R, and T station statics within each correction basis. |
| `tools/update_catalog_time_shifts.py` | Validates and applies measured event-stack residuals to a catalog workbook. |
| `stacker.py` | Compares and plots per-event component stacks from one explicit `align_stack` run. |
| `mapper.py` | Independent USGS seismicity download and mapping utility for the Pinyon Flat area. |
| `scripts/` | Data-preparation and output-regression utilities. |
| `tests/` | Unit and smoke tests for pipeline branches, helpers, and catalog updating. |
| `Notes/` | Scientific goals, current status, decisions, review findings, and conversation summaries. |

The JSON files under `reserve json files/` and `rp_input real.json` are retained
configuration variants. They are not loaded by the main pipeline unless they
are deliberately copied or supplied as the active configuration.

## Data and outputs

The code and the research data are stored separately. Current defaults expect
the research tree at:

```text
/Users/jvidale/Documents/Research/FaultScanR/
├── event_sta_info/
│   ├── catalog_local_hand.xlsx
│   ├── stations.txt
│   └── stations.xlsx
├── Sgrams/
│   ├── 20220930_<rate>Hz/
│   └── Snippets_<rate>Hz/
└── stack_output/
```

The principal inputs are:

- `catalog_local_hand.xlsx`: event IDs, origin times, locations, catalog time
  shifts, and event-selection metadata.
- `stations.txt` and `stations.xlsx`: station coordinates, identifiers, and
  optional station-static columns.
- MiniSEED waveforms: DPZ vertical data and DP1/DP2 horizontal data at the
  configured sample rate.
- `rp_input.json`: frequency band, time windows, phase, components, event list,
  input mode, alignment controls, and plotting options.

`align_stack.py` writes directly to `stack_output/`, with one subdirectory per
event. A later run replaces files with the same names but does not delete
unrelated event directories. Depending on configuration, products include:

- Per-event aligned component stacks and MiniSEED files.
- Record-section, stage-stack, envelope, correlation, and station-map figures.
- A JSON snapshot of the parameters used for the run.
- All-event component-stack figures.
- Cross-event alignment workbooks and offset-stack figures.
- Per-event station-residual workbooks under `stack_output/Statics` when
  `station_static_mode` is `cross_correlation`.

Many scripts contain user-specific absolute default paths. Check `rp_input.json`
and each command's `--help` output before using the repository with a different
research-data location.

## Environment

Run the project in the `vidale_main` Conda environment. The main dependencies
include Python, NumPy, pandas, SciPy, Matplotlib, ObsPy, and openpyxl.

```bash
conda run -n vidale_main python align_stack.py
```

The active configuration is read automatically from `rp_input.json` beside
`align_stack.py`. A run should normally start by reviewing at least:

- `events`, `input_mode`, and `analysis_hz`
- `component`, `all_channels`, and `align_phase`
- `min_freq`, `max_freq`, and the time/correlation windows
- `use_json_event_location`
- `use_event_static_correction`
- `station_static_mode` and, for `tabulated`, its workbook/column
- Plotting switches

When `all_channels` is true, Z, R, and T are processed; `component` does not
limit the run to a single component.

Event and station corrections are configured independently:

- `use_event_static_correction: true` applies the catalog `time shift` values;
  `false` leaves the catalog origin times uncorrected and measures cross-event
  stack residuals from the waveforms.
- `station_static_mode: "none"` uses only the relative TauP-predicted station
  travel times and performs no station residual-lag search.
- `station_static_mode: "tabulated"` adds the configured station-static
  workbook column to the TauP shifts.
- `station_static_mode: "cross_correlation"` adds residual lags estimated from
  the waveforms to the TauP shifts and writes station-residual workbooks.

For compatibility with older configuration files,
`use_station_static_correction: true` maps to `tabulated` and `false` maps to
`cross_correlation`. New configurations should use `station_static_mode`.

## Common commands

Run the alignment pipeline:

```bash
conda run -n vidale_main python align_stack.py
```

Summarize the statics matching the active component and phase in
`rp_input.json`:

```bash
conda run -n vidale_main python plot_statics_by_station.py
```

Select an explicit case:

```bash
conda run -n vidale_main python plot_statics_by_station.py \
  --component R --phase S
```

Plot R/T snippets for one event:

```bash
conda run -n vidale_main python plot_event_rt_snippets.py \
  --event CI_40353864
```

The Z, R, and T traces for each station are shifted so their TauP-predicted S
pick is at 0 s and is marked in green; Z is gray. Each station's configured
static from `stations.xlsx` is marked and labeled in purple. Edit `START_TIME`
and `END_TIME` near the top of `plot_event_rt_snippets.py` to change the
S-relative display window; `MIN_FREQ` and `MAX_FREQ` control the default
3–10 Hz bandpass. Corresponding command-line options override these defaults.
The plot also reads `win_pre`, `win_post`, and `move_limit_sec` from
`rp_input.json` and marks the correlation window and its expanded shift-search
limits. `APPLY_STATION_STATICS_R`, `APPLY_STATION_STATICS_T`, and
`APPLY_STATION_STATICS_Z` independently control whether each component is
shifted left by its value from `stations.xlsx`. The corresponding command-line
controls are `--shift-r`, `--shift-t`, and `--shift-z`, with `--no-shift-r`,
`--no-shift-t`, and `--no-shift-z` disabling individual components.
`--no-time-shift` disables statics globally, while an individual component
option can override it. Use `--no-z` to omit the vertical-data read and produce
R/T-only plots; `--include-z` enables Z/R/T behavior.

Compare Z and T stacks written by `align_stack.py`. `stacker.py` reads from and
writes plots directly to
`/Users/jvidale/Documents/Research/FaultScanR/stack_output/`:

```bash
conda run -n vidale_main python stacker.py \
  --components Z T
```

Running `stacker.py` with the VS Code triangle and no arguments uses the
editable `DEFAULT_*` settings near the top of the file. Command-line options
override those settings when supplied.

Preview a catalog shift update before modifying the workbook:

```bash
conda run -n vidale_main python tools/update_catalog_time_shifts.py \
  R /path/to/stack_output --dry-run
```

Only remove `--dry-run` after verifying the reported source column, baseline,
residual, destination column, event list, and proposed values. The updater
rejects alignment workbooks without measured cross-correlation residuals and
guards against stale or ambiguous catalog provenance.

Compare two run-output directories for regressions:

```bash
conda run -n vidale_main python scripts/compare_outputs.py \
  /path/to/baseline /path/to/candidate
```

Use `python <program> --help` for the complete options of programs that expose
a command-line interface.

## Tests

Run the complete test suite with:

```bash
conda run -n vidale_main python -m unittest discover -s tests
```

The focused alignment smoke suite is:

```bash
conda run -n vidale_main python -m unittest tests.test_align_stack_smoke
```

Tests exercise important contracts, but they do not replace inspection of the
scientific inputs, waveform alignment figures, workbook provenance, or output
consistency from a representative run.

## Scientific and project notes

- `Notes/faultscan_science_goals.md` explains the experiment, target signals,
  location controls, and criteria for a convincing structural observation.
- `Notes/faultscan_project_status.md` records the current processing state,
  active decisions, results, next steps, and open questions.
- `Notes/faultscan_project_chat_summary.md` preserves implementation history,
  scientific reasoning, recovery decisions, and detailed review findings.
- `statics_results_explanation.html` is a static explanatory snapshot of an
  earlier set of station-static plots; it is not an automatically regenerated
  report.

Use this README for the stable overview and normal entry points. Use `Notes/`
for evolving scientific interpretation, project history, and unresolved
decisions.

## Current scientific cautions

- Station and event corrections are enabling measurements, not final evidence
  of a time-dependent fault-zone change.
- Event location, origin time, source duration, magnitude, component
  orientation, site response, and waveform quality can all imitate or obscure
  the target signal.
- R- and T-derived timing/static results should not be treated as
  interchangeable without a documented physical and statistical justification.
- Generated reports and comparison workbooks are snapshots and should be
  regenerated after their source workbooks or configuration change.

## Authorship

FaultScan is a research codebase developed by John Vidale with AI-assisted
implementation and documentation. Scientific conclusions remain subject to
data-quality review and independent validation.
