# FaultScan Project Status

*Updated 2026-07-22*

## Project purpose

FaultScan analyzes dense local seismic-array recordings from the Anza/San
Jacinto swarm to measure and interpret small timing differences among stations
and events. The main near-term goal is to obtain reliable station-static and
event-alignment corrections so that waveform stacks, phase timing, and spatial
comparisons are physically meaningful.

## Data

- Local event catalog: `event_sta_info/catalog_local_hand.xlsx`.
- Station locations and metadata: `event_sta_info/stations.txt` and
  `event_sta_info/stations.xlsx`.
- Waveforms: 50, 100, and 250 Hz long traces plus 250 Hz snippets, stored under
  `/Users/jvidale/Documents/Research/FaultScanR`.
- The current analyses use physical node IDs, with available station metadata
  and waveform data spanning nodes 36--333. Nodes 0--35 are not present in the
  inspected inputs.

## Processing workflow

1. `align_stack.py` reads `rp_input.json`, event metadata, station metadata,
   and waveform snippets.
2. Horizontal DP1/DP2 recordings are rotated to radial (R) and transverse (T)
   components using station geometry and the existing -11 degree orientation
   correction.
3. Traces are screened, aligned in stages against predicted TauP arrival
   timing, and stacked. Catalog `time shift` values are applied to all Z, R,
   and T processing.
4. Residual station shifts are written as per-event, component/phase-aware
   static workbooks in `output/Statics`, unless precomputed station-static
   correction is enabled.
5. `plot_statics_by_station.py` removes robust per-event baselines, rejects
   outliers with MAD screening, estimates per-station median statics, and
   produces plots and Excel summaries.
6. Supporting scripts compare components/bases, merge station medians into
   `stations.xlsx`, and create R/T event snippet figures.

## Current configuration and decisions

- Use the `vidale_main` Conda environment for running the pipeline and tests.
- `rp_input.json` is the source of run parameters and overrides in-code
  defaults.
- The current reviewed configuration uses 250 Hz snippets, a 3--10 Hz filter,
  T-component S alignment, `all_channels: true`, JSON event-location override
  enabled, event-static correction enabled, and station-static correction
  disabled. Event-stack cross-correlation is therefore disabled. Its active
  timing window is 0--12 s, with 0.05 s pre-window and 0.15 s post-window.
- The shared catalog field `time shift` remains the correction applied across
  all components. T-derived shifts are stored separately in `time shift T` as
  a comparison product.
- The following stations are excluded before waveform reading and all later
  processing: 00050, 00058, 00074, 00085, 00107, 00154, 00161, 00171, 00179,
  00303, and 00331.

## Current outputs and results

- Per-event Z/R/T P- or S-phase static workbooks in the research output
  directory, named to identify component, phase, and catalog-shift basis.
- Station-static plots and Excel workbooks for raw shifts, event-baseline
  corrected shifts, per-station medians, and event-alignment information.
- Six-case summaries are available for `Z_P`, `Z_S`, `R_P`, `R_S`, `T_P`, and
  `T_S`; each analyzed output contains 289 stations.
- The R-S versus T-S comparison used 289 common stations and found essentially
  no correlation (about 0.008), with a median absolute difference of about
  0.0127 s. This indicates that the two component results should not yet be
  treated as interchangeable station corrections.
- All-event component-stack alignment plots and Excel workbooks are available
  for R and T. They use `CI_40353472` as the reference event and distinguish
  catalog-predicted shifts from optional waveform residual shifts.
- `statics_results_explanation.html` is a static explanatory snapshot of
  earlier station-static plots; it is not an automatically regenerated report.

## Plotting

- With `show_record_section_plot: true`, the pipeline saves aligned
  record-section/stack images such as `<event>_<component>_<phase>.png`.
- Individual seismogram pages are disabled when
  `show_individual_seismograms: false`.
- The pipeline does not currently create an explicit side-by-side plot of
  waveforms before versus after alignment. Existing plots show aligned stacks
  and correlation-window comparisons.

## Next steps and goals

1. Decide the scientific role of T-derived event shifts: retain them solely as
   a diagnostic, replace the shared R-derived `time shift`, or combine the two
   with an explicitly justified method.
2. Rerun `align_stack.py` whenever a new component/phase case is selected, then
   regenerate station-static summaries, six-case comparisons, and the HTML
   explanation so all derived products reflect the same inputs.
3. Evaluate why R-S and T-S station statics disagree, including component
   orientation, local site response, phase selection, signal quality, and
   event-dependent effects.
4. Review the current unstaged changes and decide which source files,
   configuration variants, notes, and generated artifacts belong in version
   control. Avoid committing `.DS_Store` and large disposable run outputs.
5. Keep the project `README.md` code index current as program responsibilities,
   inputs, outputs, and normal command sequences change.
6. Extend the durable research notebook with station-quality observations,
   interpretation of notable results, hypotheses, and reasons for decisions.

## Open questions

- Which station-static correction is most appropriate for future multi-component
  processing: R, T, a component-specific correction, or a combined estimate?
- Are the rejected stations affected by instrument problems, local site
  response, or both?
- Should an automated, reproducible generator replace the hand-/AI-authored
  `statics_results_explanation.html` snapshot?
- Should a direct before/after alignment waveform figure be added to support
  visual quality control?
