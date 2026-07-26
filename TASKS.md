# FaultScan Tasks

This is the short, actively curated work list for FaultScan. Long-term
scientific direction belongs in `Notes/faultscan_science_goals.md`; project
history and interpretation belong in the other files under `Notes/`.

## Working conventions

- Keep no more than three tasks in **Now**.
- Move the complete task entry from **Next** to **Now** when starting it.
- Add `Started: YYYY-MM-DD` when a task moves to **Now**.
- A task is complete only when its stated completion criteria are satisfied.
- When closing a task, check it, add the date, result, and verification, and
  move it to **Closed**.
- Use `Status: Cancelled`, `Superseded`, or `No longer needed` when appropriate
  instead of silently deleting meaningful work.
- Move older closed entries to `Notes/task_archive.md` when this file becomes
  cumbersome.

## Now

- [ ] **FS-003 — check that static shifts**
  - Purpose: check that static shifts are coded correctly.
  - Approach: compare a representative event with correlation shift and static shift. Check waveforms and number of stations used.  Check other two components.  Check various frequencies.
  - Added: 2026-07-23

- [ ] **FS-004 — check static shifts in station table**
  - Purpose: Determine whether the stored station statics improve waveform
    alignment across frequency bands.
  - Done when: Matched runs with and without station-static correction have
    been compared at representative frequencies, and changes in alignment,
    correlation, and stack coherence have been summarized.
  - Related: `align_stack.py`, `rp_input.json`,
    `event_sta_info/stations.xlsx`, `output/Statics`
  - Approach: Hold the event set and all non-static parameters fixed, run each
    frequency band with `station_static_mode` set to `none`, `tabulated`, and
    `cross_correlation`, and compare the same plots and quantitative quality
    measures.
  - Added: 2026-07-23

- [ ] **FS-005 — add P shifts to station file**
  - Purpose: Estimate and store P-wave station corrections using one or more
    events with sharp, coherent P arrivals.
  - Done when: Suitable calibration events have been selected, P statics have
    been validated, and a clearly named P-static column has been added without
    overwriting the existing S correction.
  - Related: `align_stack.py`, `plot_statics_by_station.py`,
    `augment_stations_with_statics.py`, `rp_input.json`,
    `event_sta_info/stations.xlsx`
  - Approach: Inspect candidate-event vertical traces, run P alignment for the
    best events, reject unstable station estimates, calculate robust station
    values, and merge them into a new workbook column.
  - Added: 2026-07-23

## Next

- [ ] **FS-006 — Search for waveform differences**
  - Purpose: Superimpose comparable foreshock and aftershock waveforms to
    identify differences across the mainshock.
  - Done when: A reproducible set of corrected and normalized event comparisons
    has been generated, and candidate differences have been measured and
    recorded.
  - Related: `align_stack.py`, `rp_input.json`,
    `event_sta_info/catalog_local_hand.xlsx`,
    `Notes/faultscan_science_goals.md`
  - Approach: Select the best-colocated and most coherent events, apply a
    common processing configuration, normalize amplitude and source-duration
    effects, and compare aligned waveforms, spectra, polarization, and early
    S-coda residuals.
  - Added: 2026-07-23

- [ ] **FS-007 — Examine differences**
  - Purpose: Distinguish temporal waveform changes from differences caused by
    event location, radiation pattern, propagation path, or fault structure.
  - Done when: Each candidate difference has been tested against event time and
    spatial/geometric controls, with the supported and rejected explanations
    documented.
  - Related: `Notes/faultscan_science_goals.md`,
    `Notes/faultscan_project_status.md`,
    `event_sta_info/catalog_local_hand.xlsx`, aligned-stack outputs
  - Approach: Group comparisons by time relative to the mainshock, relative
    hypocentral separation, azimuth, and component; then test which variables
    best explain the measured waveform differences.
  - Added: 2026-07-23

## Waiting

- [ ] **FS-010 — Incorporate updated relative earthquake relocations**
  - Purpose: Test whether observed waveform differences can be explained by
    improved relative earthquake locations.
  - Waiting for: A reviewed updated relative-location catalog.
  - Done when: The revised locations are incorporated reproducibly and the
    principal waveform comparisons are repeated using the best-colocated event
    pairs.
  - Related: `event_sta_info/catalog_local_hand.xlsx`,
    `Notes/faultscan_science_goals.md`, waveform-comparison outputs
  - Approach: Preserve the current catalog, validate event-ID and coordinate
    mappings, import the revised locations, recompute geometry-dependent
    products, and compare results before and after relocation.
  - Added: 2026-07-23

## Ideas

- [ ] **FS-011 — Measure polarization and shear-wave splitting changes**
  - Purpose: Test whether component relationships change across the mainshock.
  - Done when: Polarization and splitting measurements with uncertainties have
    been compared across suitable foreshock and aftershock event pairs.
  - Related: `align_stack.py`, aligned R/T/Z traces,
    `Notes/faultscan_science_goals.md`
  - Approach: Select high-coherence windows around S, estimate polarization
    direction and splitting parameters consistently, and test their dependence
    on event time and location.
  - Added: 2026-07-23

- [ ] **FS-012 — Automate regeneration of the statics explanation**
  - Purpose: Replace the static hand-/AI-authored HTML snapshot with a
    reproducible report derived from current workbooks.
  - Done when: One documented command regenerates the report, figures, tables,
    provenance, and summary statistics from selected statics workbooks.
  - Related: `plot_statics_by_station.py`,
    `statics_results_explanation.html`, `statics_results_assets/`
  - Approach: Extract the report's calculations and layout into a script,
    require explicit input cases, embed run/configuration provenance, and test
    generation in a temporary output directory.
  - Added: 2026-07-23

- [ ] **FS-013 — Generate an automatic station-quality Markdown report**
  - Purpose: Record objective processing counts, exclusions, timing problems,
    clipping, gaps, and stack statistics for each run.
  - Done when: Each pipeline run can write a deterministic Markdown report
    containing the agreed quality fields and links to its supporting products.
  - Related: `align_stack.py`, `align_utils.py`, run parameter snapshots,
    timestamped output directories
  - Approach: Define a small structured quality-record schema, collect values
    during processing, and render a run-scoped Markdown summary from that
    record rather than parsing console text.
  - Added: 2026-07-23

## Closed

- [x] **FS-002 — Create a repository overview and program index**
  - Status: Completed
  - Completed: 2026-07-22
  - Result: Created `README.md` with the scientific purpose, workflow, program
    index, inputs, outputs, common commands, and testing instructions.
  - Verification: Markdown links and referenced repository paths were checked.
