# FaultScan Project Chat Summary

## Scope

This note summarizes the decisions and implementation work available in the
current Coding the Shifts conversation, together with the repository files
reviewed here. It does not claim to include details that appear only in other
project conversation threads.

## Runtime and Inputs

- Run Python commands in the Conda environment `vidale_main`.
- The active pipeline configuration is `rp_input.json`, loaded automatically by `align_stack.py`.
- Research inputs and outputs are rooted at `/Users/jvidale/Documents/Research/FaultScanR`.
- The catalog is `event_sta_info/catalog_local_hand.xlsx`; station coordinates are in `event_sta_info/stations.txt` and `stations.xlsx`.

## Event Location and Time Shifts

- `use_json_event_location` controls only whether `event_lat`, `event_lon`, and `event_depth` in `rp_input.json` override catalog event locations.
- When `use_event_static_correction` is enabled, the catalog column
  `time shift` is applied independently of `use_json_event_location`.
- The same enabled catalog event shift is applied for Z, R, and T processing.
- `use_event_static_correction` controls event corrections independently of
  the three-way station-static mode.
  - `true`: applies the catalog event origin shifts and does not measure an additional cross-event waveform residual.
  - `false`: leaves catalog event statics unapplied and measures cross-event stack residuals from the waveforms.

## Station Processing and Statics

- DP1/DP2 horizontals are rotated to R/T using station geometry and the existing `-11 degree` orientation correction.
- In `cross_correlation` station-static mode, the pipeline computes fresh
  station residual shifts relative to theoretical TauP timing and writes
  per-event/component static workbooks to `output/Statics`.
- `plot_statics_by_station.py` calculates robust event baselines and robust per-station median statics using MAD outlier rejection.
- `stations.xlsx` can hold station static columns. The current shared-column convention is `station static s`.
- `station_static_mode` explicitly selects `none`, `tabulated`, or
  `cross_correlation` station alignment. `none` uses relative TauP travel
  times alone; `tabulated` applies the `station static s` values from
  `stations.xlsx`; and `cross_correlation` estimates waveform residuals.
  In `tabulated` mode:
  - Station statics are converted to be relative to the selected reference station.
  - The correction is used in all three alignment stages.
  - Station correlation is still evaluated for screening, but no new residual lag is searched.
  - New station-static workbooks are not written in this mode.
  - Any processed, non-rejected station lacking a finite static causes an error rather than being silently used without a correction.

## Rejected Stations

The shared waveform reader excludes these stations before reading, rotation, alignment, stacking, static calculations, and plotting:

`00050, 00058, 00074, 00085, 00107, 00154, 00161, 00171, 00179, 00303, 00331`

## Station-Static Analysis and Plots

- `plot_statics_by_station.py`: raw/corrected station-static scatter plots plus baseline, station-median, and event-alignment Excel workbooks.
- `augment_stations_with_statics.py`: merges robust station medians from R/T-basis runs into `stations.xlsx`.
- `plot_station_static_basis_comparison.py`: compares the same component's R-basis and T-basis station statics.
- `plot_station_component_static_comparison.py`: compares R/T, R/Z, and T/Z component station statics within each correlation basis.
- Earlier R/T-basis labeling described the historical catalog-shift basis, not cross-correlation of an R trace against a T trace. With the current shared `time shift` column, that distinction is no longer needed for new runs.

## Event Snippet Plotter

`plot_event_rt_snippets.py` is a standalone R/T snippet plotting tool.

- Defaults currently target event `CI_40353864`.
- Uses snippets at 250 Hz, a 2--12 s time window, and a 1--4 Hz bandpass.
- Plots R (blue) and T (orange), omitting Z.
- Uses 20 station pairs per frame, sorted near to far from top to bottom.
- Draws per-station green predicted S-arrival markers from the `iasp91` model.
- Uses one amplitude scale across all R and T traces in the event, preserving station-to-station and component-to-component amplitude differences.

## Verification

- The `align_stack` smoke suite has been run repeatedly with `conda run -n vidale_main python -m unittest tests.test_align_stack_smoke`.
- The latest implementation checkpoint reported 23 passing smoke tests.

## 2026-07-22 Chat Addendum: Component/Phase Statics Review

### Configuration and Precedence

- `rp_input.json` is the intended source of run parameters; values declared there override the in-file defaults in `align_stack.py`.
- `plot_statics_by_station.py` was updated to follow the same rule for both component and phase: without explicit CLI flags, it reads `component` and `align_phase` from `rp_input.json`. Missing JSON values fall back to `R` and `S`, respectively.
- Explicit plotting arguments still take precedence, for example `--component Z --phase P`.
- The active configuration at this update uses 250 Hz snippets, a 3--10 Hz band, `all_channels: true`, `component: T`, `align_phase: S`, JSON event-location override enabled, event-stack cross-correlation disabled, and station-static correction disabled. The active time window is 4.75--5.75 s with a 0.05 s pre-window and 0.15 s post-window.

### Static Export and Plotting Behavior

- Per-event static workbooks are component/phase aware: Z, R, and T can be written for P or S alignment, rather than limiting static export to radial S.
- The statics plotter filters its input workbooks by the selected component and phase and uses component-specific titles and output prefixes. This fixed the earlier situation in which a T alignment run could be followed by an R-labeled summary because the plotter independently defaulted to R.
- Plot interpretation was confirmed: each point is one event/station residual static; the raw plot marks the per-station raw median; the corrected plot removes a robust per-event baseline; station summary workbooks use the corrected values with MAD outlier rejection.

### Six-Case Comparison and Outputs

- The six available station-static cases are `Z_P`, `Z_S`, `R_P`, `R_S`, `T_P`, and `T_S`, each with 289 stations in the analyzed outputs.
- A session workbook, `six_case_station_statics_summary.xlsx`, was generated with `combined_long`, `case_summary`, `median_static_wide`, `uncertainty_wide`, and `n_outliers_wide` sheets. It contains `median_event_baseline_corrected_static_seconds`, `static_uncertainty_seconds`, and `n_outliers` for every station/case.
- An R-S versus T-S station-static scatter plot, `r_s_vs_t_s_station_static_scatter.png`, was generated from the corrected station medians. It used 289 common stations; the reported R/T correlation was about 0.008, the median absolute difference about 0.0127 s, and the maximum absolute difference about 0.1216 s.
- `statics_results_explanation.html` and its `statics_results_assets/` directory provide a self-contained explanatory report with the R/T raw and event-baseline-corrected plots and the then-current summary statistics.

### Data Availability and Follow-Up

- Station identifiers are physical node IDs, not zero-based row numbers. The available metadata and waveform trees begin at station 36 and span 36--333; stations 0--35 are absent from `stations.txt`, 50/100/250 Hz long traces, and the 250 Hz snippet inputs.
- Before plotting a newly selected component/phase (for example Z/P), rerun `align_stack.py` after updating `rp_input.json`; `plot_statics_by_station.py` can only summarize matching per-event statics workbooks that already exist.
- Regenerate the HTML report and six-case comparison artifacts after changing the underlying statics workbooks; they are snapshots, not automatically refreshed products.

## 2026-07-22 Chat Addendum: Swarm Station Quality and Fault Geometry Notes

### Scope and Changes

- This was explicitly a notes-only discussion. It made no code changes, configuration changes, output files, or plots.
- The rejected-station set already documented above was reaffirmed for `align_stack`; this chat added the observational reasons for those exclusions rather than changing the set.

### Station Observations During the Swarm

- Station 50: bad T.
- Station 58: noisy.
- Stations 74 and 85: small R.
- Stations 107 and 331: bad R.
- Stations 154, 161, and 303: bad T.
- Stations 171 and 179: both horizontal components dead.
- Some of these observations may reflect poor site conditions that distort the waveform rather than instrument failure. The cause remains unresolved and should not be inferred solely from these waveform-quality labels.

### Largest Event and Local Fault Geometry

- The largest event in `catalog_local_hand.xlsx` is `CI_40353408`, M3.41, at 2022-09-30 11:50:11.75 UTC near 33.4803 N, 116.5132 W and 9.3 km catalog depth.
- The published CI/USGS focal-mechanism product is predominantly strike-slip with a small reverse component. Its nodal planes are strike/dip/rake 297/76/171 degrees and 29/81/14 degrees.
- The focal-mechanism product is automatic and preliminary, so its detailed geometry should be treated cautiously.
- Measurement of the mapped USGS Quaternary Fault and Fold Database trace at the swarm location gives a local Anza-section/Clark-strand strike of roughly 120--123 degrees (equivalently 300--303 degrees) over a 0.5--1 km scale; the trace bends locally and is about 111 degrees over a roughly 250 m scale.
- The 297-degree nodal plane is therefore approximately parallel to the local San Jacinto fault trace and is the favored interpretation: right-lateral slip on the WNW--ESE plane. The focal mechanism alone does not uniquely identify the physical fault plane, so the conjugate NNE--SSW left-lateral plane remains the alternative.

## 2026-07-22 Chat Addendum: Persistent Project Memory Workflow

### Scope and Decisions

- This was a workflow and project-organization discussion. It made no code changes, configuration changes, output files, or plots.
- FaultScan should be treated as the persistent home for the scientific development of the project, not merely as a Python repository. Its long-term scope may include source code, Git history, research notes, manuscripts, reviewer responses, relevant PDFs, figures, and project-related AI conversations.
- Codex remains the engineering environment, while VS Code remains the primary environment for execution, debugging, Git work, and visualization.
- Durable scientific memory should live in real Markdown files accessible to both Python and AI. AI conversations can help with interpretation, planning, and synthesis, but should not be the sole permanent research notebook.
- The project memory should preserve the reasoning that is hardest to reconstruct later: what was noticed, why a result was trusted or rejected, which hypotheses were considered, and why the research direction changed.

### Separation of Records

- Python-generated Markdown should provide objective processing records, including stations processed or rejected, rejection reasons, timing problems, clipping, data gaps, and stack statistics.
- Human-authored research notes should record scientific interpretation: exclusion rationale, new hypotheses, unexpected observations, future tests, unanswered questions, and decisions with their reasoning.
- The proposed daily workflow is to run the analysis, generate an automatic station-quality summary, add only important scientific observations to the research notebook, and then use ChatGPT or Codex to discuss implications, plan follow-up analyses, and identify longer-term patterns.

### Intended Long-Term Capability

- The accumulated notes should eventually make it possible to answer when a station problem was first noticed, whether stations have failed repeatedly, whether an analysis was already attempted, why an approach was rejected, and which papers or reviewer comments relate to an issue.

### Unresolved Organization Choices

- The precise notebook structure remains open. Candidate layouts include topic files such as `notes/research_log.md`, `notes/station_quality.md`, and `notes/paper_ideas.md`, or dated files such as `notes/daily/2026-07-21.md`.
- The preferred location is inside the project and under Git, or at minimum immediately alongside the project and visible to both Codex and ChatGPT. The final placement and version-control policy have not yet been selected.
- A concise workflow overview could live at `notes/README.md` or `PROJECT_WORKFLOW.md`; no such file was created in this chat.

## 2026-07-22 Chat Addendum: Working-Version Recovery and Event-Stack Shift Outputs

### Recovery Decision and Git History

- An apparent regression was first diagnosed as an incomplete uncommitted refactor: event-location override handling and per-event station-static export had been removed while related JSON keys and tests remained; an original-trace overlay option was present in JSON but not honored by plotting; and `write_run_parameter_snapshot` had stopped accepting string paths used by smoke tests.
- The repository was initially returned conservatively to commit `4f067ef` (`Working on statics`, 2026-06-09), but later run artifacts showed that newer uncommitted code had completed successfully. VS Code local history then provided the exact newer source snapshot: `align_stack.py` saved 2026-07-13 14:35:36 was byte-for-byte identical to the preserved stash, and the matching `align_utils.py` history copy was also recovered.
- Two July 13 runs immediately after that source save produced the final all-event R plots, demonstrating that this was a working pipeline version rather than merely a syntactically valid edit. At the time, the runs included `all_events_R_offset_stacks_S.png` and `all_events_component_stacks_S.png`; the second run produced 362 files. Those original timestamped run directories were subsequently removed or relocated and are no longer present at their original paths.
- The recovered July 13 source and its R/S 250 Hz run configuration were committed and pushed to `main` as `e674e9d` with message `restarted editing with new chatgpt set up, trying to align events next`. Before pushing, `r_s_vs_t_s_station_static_scatter.png` and `six_case_station_statics_summary.xlsx` were moved out of the repository and removed from the amended commit. The local branch and `origin/main` then pointed to the same commit.
- Two safety stashes remain available: `backup before regression to last working state 2026-07-18` (created 10:07:46 PDT) and `backup selective restoration before returning to July 13 working version` (created 10:49:40 PDT). They preserve the superseded intermediate states and should not be dropped until their contents are no longer needed.

### Cross-Event Stack Alignment and Catalog Shift Work

- Subsequent unstaged development added `compute_event_stack_alignment_shifts` and `plot_all_events_component_offsets_aligned`. Event stacks are compared with reference event `CI_40353472`; the optional waveform residual search is limited by `event_stack_alignment_max_shift_sec`. The control was initially named `use_event_stack_xcorr_alignment` and was later replaced by the inverse-convention `use_event_static_correction` flag.
- The aligned-stack products include a vertically offset PNG and an Excel workbook named `all_events_<component>_stack_xcorr_alignment_to_CI_40353472.xlsx`. The workbook records predicted and residual lags, waveform correlation, left/right alignment shifts, whether a residual was measured, the reference-event flag, and the catalog time shift used for plotting.
- Repeated R and T products currently exist in later run directories, including `20260720_163908_1030`, `20260720_181451_5781`, `20260720_181740_9915`, `20260721_181654_3637`, and `20260721_185210_2337`. Each contains the relevant component alignment workbook and the corresponding `all_events_<component>_offset_stacks_S_xcorr_aligned_to_CI_40353472.png`; the latest observed T pair is in `20260721_185210_2337`.
- `tools/update_catalog_time_shifts.mjs` was generalized to accept a component and run-output directory. R results update the shared catalog column `time shift`; T results are preserved separately in `time shift T`. `catalog_local_hand.xlsx` currently contains both columns, and the T column has populated event shifts (for example, `CI_40353272 = -0.052 s` and reference event `CI_40353472 = 0.000 s`). The alignment pipeline still intentionally applies the shared `time shift` column to Z, R, and T; `time shift T` is retained as a comparison/result column rather than automatically replacing the shared correction.

### Configuration and Outputs Added During This Work

- JSON configuration gained explicit controls for `event_alignment_reference`, `event_stack_alignment_max_shift_sec`, `use_json_event_location`, `use_event_static_correction`, and the three-way `station_static_mode`. Legacy `use_station_static_correction` configurations remain accepted. Event-location override validation requires all three coordinates/depth values when enabled and otherwise retains catalog metadata.
- Per-event station-shift export was restored and generalized. Workbooks now identify the waveform component/phase and the catalog-shift basis (`R` or `T`) and use filenames ending in `shiftR_xcorr_statics.xlsx` or `shiftT_xcorr_statics.xlsx`.
- `plot_statics_by_station.py` gained event-baseline differences relative to a configurable reference event and writes an event-alignment workbook in addition to raw, corrected, station-median, and baseline products.

### Verification and Unresolved Items

- The default system Python lacked ObsPy; project tests and pipeline commands must use `conda run -n vidale_main python`. Earlier failures under the default interpreter were environmental, not pipeline regressions.
- The exact July 13 recovery files compiled successfully. A temporary selective restoration of the all-event offset plot was also covered by a focused test, and that temporary state reached 28 passing tests before the evidence-backed July 13 version was restored. Later smoke coverage was expanded for event-location selection, stack-shift calculation, component/phase static export, and R/T catalog-shift bases; the current summary's latest checkpoint remains the authoritative verification count.
- The request to repeat the catalog-writing process with T is now represented by the T alignment workbooks and populated `time shift T` column. A remaining scientific decision is whether T-derived shifts should remain a diagnostic comparison, replace the shared `time shift`, or be combined with R; current processing continues to use the shared `time shift` for every waveform component.
- Much of the post-`e674e9d` implementation remains unstaged/uncommitted in the working tree, together with new scripts and notes. Before the next push, review the broad diff, decide which generated/config reserve files belong under version control, and avoid committing `.DS_Store`.

## 2026-07-22 Chat Addendum: Configuration Audit and Plot-Window Discussion

### Scope and Verification

- This chat performed a read-only inspection of `align_stack.py`, `align_utils.py`, `stacker.py`, and the active JSON configuration. It made no code or configuration changes and created no FaultScan output files or plots.
- `align_stack.py` and `align_utils.py` passed `python -m py_compile` during the inspection. Ruff reported only four unused imports in `align_stack.py`: `UTCDateTime`, `gps2dist_azimuth`, `ensure_utc_datetime`, and `make_event_output_dir`. These are cleanup items rather than syntax or runtime failures.
- At the inspected checkpoint, `all_channels: true` meant the separate `component` value did not restrict processing to that one component. This interaction should be kept in mind when interpreting or editing `rp_input.json`.

### Plot-Window Placement

- Interactive Matplotlib figure windows can in principle be positioned with a small backend-specific helper using the GUI window manager (`window.move(x, y)` for Qt or `window.wm_geometry(...)` for Tk). Pixel coordinates are measured from the display's upper-left corner.
- No plot-positioning helper was added. The behavior remains unresolved for the macOS native Matplotlib backend, and figures that are saved and immediately closed cannot be positioned as persistent on-screen windows.

### Other Conversation Topics

- The chat also included general Apple Music and Codex-product questions. Those discussions produced no FaultScan decisions, code changes, configuration changes, outputs, or plots and are therefore not incorporated into the technical project record.

## 2026-07-22 Chat Addendum: Repository Status, HTML Provenance, and Plot Products

### Scope

- This was a repository-inspection and project-memory update chat. Apart from this notes update, it made no code changes, configuration changes, output files, or plots.
- The chat clarified what Git could and could not say about recent changes, and distinguished tracked committed work from untracked local artifacts.

### Git Status and Recent Changes Discussed

- `untracked` was clarified to mean that Git sees a file in the working directory but is not yet managing it in version history. Such files can be added with `git add`, ignored with `.gitignore`, or left untracked if they are temporary/generated outputs.
- Earlier in the inspection, the repository had no modified tracked files and two untracked analysis artifacts:
  - `r_s_vs_t_s_station_static_scatter.png`
  - `six_case_station_statics_summary.xlsx`
- The most recent committed code changes discussed were from commit `4f067ef` (`Working on statics`, 2026-06-09), which generalized component/phase station-static export, updated `plot_statics_by_station.py` to read component/phase defaults from `rp_input.json`, changed the active JSON from radial/S-style defaults toward T/P in that historical checkpoint, updated tests, and added `statics_results_explanation.html` plus `statics_results_assets/`.

### Waveform and Static Plot Products

- With `show_record_section_plot: true`, `align_stack.py` saves aligned record-section/stack plots named like `<event>_<component>_<phase>.png`.
- With `all_channels: false`, the pipeline also runs stage-stack plotting through `plot_stage_stacks`.
- With `show_individual_seismograms: false`, individual seismogram pages are not generated.
- No explicit side-by-side waveform plot of "before alignment" versus "after alignment" was found in the inspected code. Existing comparison plots are aligned record sections/stacks and correlation-window pass/fail snippet comparisons, not a direct unaligned-vs-aligned waveform figure.

### Python-Code Documentation Gap

- No general project README or code-index file describing all Python scripts was found in the repository.
- `statics_results_explanation.html` is the closest current explanatory document, but it explains statics results rather than the whole codebase.
- Several scripts have useful top-level docstrings or self-describing names (`mapper.py`, `scripts/compare_outputs.py`, `scripts/create_dirs_from_excel_entries.py`, `scripts/populate_snippets_from_source.py`), but core pipeline files such as `align_stack.py`, `align_utils.py`, `plot_statics_by_station.py`, and `stacker.py` do not yet have a single shared overview document.
- Unresolved item: create a concise README or notes page that explains the purpose, inputs, outputs, and normal command sequence for each Python script.

### `statics_results_explanation.html` Provenance and Viewing

- `statics_results_explanation.html` can be viewed as a rendered report by opening it in a web browser; opening it inside the editor shows the HTML source.
- Git history showed the file first appeared in commit `4f067ef` with 405 inserted lines, alongside four PNG assets:
  - `statics_results_assets/r_s_event_baseline_corrected_statics_by_station.png`
  - `statics_results_assets/r_s_statics_by_station.png`
  - `statics_results_assets/t_s_event_baseline_corrected_statics_by_station.png`
  - `statics_results_assets/t_s_statics_by_station.png`
- No Python, shell, Markdown, or JSON generator reference for `statics_results_explanation.html` was found in the repository. The report appears to be a hand-authored or AI-authored snapshot based on `plot_statics_by_station.py` outputs rather than a reproducible product generated by a tracked script.
- It remains unresolved whether the HTML came from the ChatGPT app, VS Code chat, Codex, or manual editing. Git records when it entered the repository but not which app or prompt created it. VS Code Timeline or chat history around 2026-06-09 may be the best remaining place to look.

## 2026-07-22 Change Review: Alignment Tools Integration

### Review Scope and Verification

- The reviewed change is commit `fdf9001` (`Add alignment tools and project status notes`) relative to `e674e9d`. It adds the cross-event alignment products, station-static tooling, plotting utilities, catalog-update script, configuration changes, tests, and project notes summarized above.
- All changed Python files compile in the `vidale_main` Conda environment.
- The focused command `python -m unittest tests.test_align_stack_smoke` passes all 23 tests.
- Full discovery with `python -m unittest discover -s tests` runs 34 tests but fails four pipeline smoke tests. In each failure, mocked event context returns `save_dir` as a string and `write_run_parameter_snapshot` calls `.mkdir()` on it. The snapshot helper should normalize its argument with `Path(output_dir)`, or the repository-wide path contract and tests should be changed consistently. The full suite is therefore not currently green.

### Review Findings Requiring Follow-Up

- At the reviewed commit, `tools/update_catalog_time_shifts.mjs` overwrote catalog values with `shift_left_to_align_waveform_seconds` without checking `xcorr_residual_measured`. That finding was subsequently addressed by the validated Python updater, which rejects non-measured workbooks and combines each residual with the catalog baseline recorded by the run.
- Catalog updating is not safely repeatable. Event origins are processed using the catalog's existing `time shift`, while the alignment workbook reports the remaining alignment measured from that processed run. Treating the reported value as a new absolute catalog value can discard the prior correction on a subsequent iteration. The update rule needs explicit baseline provenance and should normally combine a measured residual with the shift used by that run, rather than blindly replacing it.
- Static-workbook provenance is inconsistent with the processing path. `catalog_time_shift_column()` always selects the shared `time shift` column, but `write_component_phase_time_shifts()` labels and names outputs as `shiftR` or `shiftT` from the configured `component`. With `all_channels: true` and `component: T`, Z/R/T workbooks are consequently labeled as T-basis even though all components used the shared shift column. Downstream R-versus-T filtering and station-workbook augmentation can therefore compare labels rather than genuinely different correction bases. The provenance field should record the actual catalog column used, or the obsolete basis distinction should be removed.
- The catalog updater imports `@oai/artifact-tool` from a versioned absolute cache path under one user's home directory. This makes the committed tool dependent on a transient local runtime layout; a stable declared dependency or a Python workbook implementation would make the workflow reproducible.

### Working-Tree Note

- The only uncommitted file observed during this review was `.DS_Store`; it was not modified or included in the notes update.

## 2026-07-26 Chat Addendum: Cross-Task Summary Consolidation

### Scope and Outcome

- The user revisited most FaultScan project chats and issued the summary-merge request in each. Their contributions are now consolidated into this project note rather than being left only in separate chat histories.
- This addendum records a project-memory update only. It made no code changes, configuration changes, analysis outputs, plots, or scientific interpretations.

### Limitation

- A chat can contribute only information it explicitly writes into this Markdown file. The file is therefore the durable consolidated record, but it cannot guarantee inclusion of details from a project chat that was not asked to update it or did not complete its update.

## 2026-07-26 Chat Addendum: Archiving and Per-Chat Memory Capture

### Scope and Decisions

- This was a project-memory workflow discussion. Apart from this notes update, it made no code changes, configuration changes, output files, or plots.
- Archiving a chat preserves it in the account so it can be searched, reopened, continued, or unarchived, but removes it from normal sidebar view and from automatic chat-history reference in other chats. Any separately saved memory remains, and information already copied into project files remains available independently of the archived conversation.
- Archived chats can be found through the same search opened by the sidebar magnifying glass or `Command-K` on macOS. They can also be managed and unarchived through the application's Archived Chats settings.
- For FaultScan, project-relevant details should be transferred to `Notes/faultscan_project_chat_summary.md` before or after archiving. This makes the durable record independent of whether the originating chat remains active.

### Agreed Capture Procedure

- There is no built-in Codex command that should be relied upon to sweep and accurately reconstruct every other project chat. Instead, open each relevant active or archived chat and ask that chat to contribute its own information to the cumulative summary.
- The reusable prompt is:

  > Review this entire chat and update Notes/faultscan_project_chat_summary.md with any decisions, code changes, configuration changes, output files, plots, scientific interpretations, and unresolved items that are not already recorded. Keep existing accurate content, avoid duplication, and add a dated entry describing what this chat contributed. If the chat contains no project-relevant information, do not modify the file.

- This prompt should be run once near the end of each relevant conversation. Existing archived chats can be located with `Command-K`, opened, and processed the same way without permanently unarchiving them.
- No custom slash command, skill, hook, or automation was created for this workflow. Automating the per-chat capture remains an optional future improvement, but any implementation must avoid assuming that one chat has complete access to the full contents of all others.

## 2026-07-26 Chat Addendum: Daily Research Log Initiation

### Scope and Files

- This chat put the previously proposed daily-notebook structure into practice
  by creating `Notes/daily/2026-07-22.md`. The file was initially untracked but
  was later added to Git in commit `4348661`.
- The adopted naming convention is `Notes/daily/YYYY-MM-DD.md`. The initial
  template separates work completed, results and observations, scientific
  interpretation, decisions and reasoning, problems or uncertainties, and
  next steps.
- The daily log is intended to remain selective: automatic processing reports
  hold detailed objective records, while the daily note preserves important
  observations, conclusions, questions, and reasons for decisions.
- Apart from the Markdown daily log and this summary update, this chat made no
  code changes, configuration changes, research output files, or plots.

### July 22 Review Captured in the Log

- The first daily log consolidated the alignment-tool integration,
  documentation work, event/static correction decisions, outputs reviewed,
  scientific interpretations, and unresolved questions already described in
  the preceding addenda.
- At the configuration checkpoint recorded by that log, processing used 250 Hz
  snippets, all three components, S alignment, a 3--10 Hz band, a 0--12 s
  interval, JSON event-location override, catalog event-static correction,
  imposed station statics from `station static s`, and enabled individual and
  record-section plotting. This is a historical snapshot; subsequent work
  replaced the Boolean station-static setting with `station_static_mode` and
  changed the active plotting and time-window values.
- Full test discovery in the `vidale_main` environment passed all 40 tests at
  that checkpoint. The tests wrote only to temporary directories and did not
  create a new research processing run. This superseded the earlier review
  checkpoint at which 34 tests ran with four pipeline-smoke failures.

### Remaining Notebook Decision

- Daily logging is now an established part of the repository workflow.
  It remains optional to promote recurring subjects into durable topic notes
  such as `station_quality.md` or `paper_ideas.md` when a subject becomes too
  broad or long-lived for chronological daily entries alone.

## 2026-07-22 Chat Addendum: Project Status Document and GitHub Sync

### Documentation and Synchronization

- This chat created `Notes/faultscan_project_status.md`, a concise scientific
  status document distilled from the cumulative chat summary. It records the
  project purpose, data sources, processing workflow, active configuration,
  current outputs and interpretations, plotting behavior, next steps, and open
  questions. It is intended as a quick orientation document rather than a
  replacement for the detailed chronological summary or daily log.
- The working project update was tested with
  `conda run -n vidale_main python -m unittest tests.test_align_stack_smoke`;
  all 23 smoke tests passed. The update was committed as `fdf9001` (`Add
  alignment tools and project status notes`) and pushed to `origin/main` on
  2026-07-22.
- `.DS_Store` was deliberately left uncommitted as local macOS metadata. No
  research processing run, configuration change, or scientific result was
  produced solely by the documentation-and-sync portion of this chat.

## 2026-07-26 Chat Addendum: Station-Static Modes, Stacker Workflow, and Stack Screening

### Repository Organization and Configuration Decisions

- This chat established `README.md` as the concise program and workflow
  overview and `TASKS.md` as the short curated queue. Task entries were
  expanded with purpose, completion criteria, relevant files, and suggested
  approach without implementing the tasks. Daily logs use
  `Notes/daily/YYYY-MM-DD.md`; completed tasks may be checked and either
  retained, archived, or deleted when their history is not useful.
- The former Boolean station-static switch was replaced by
  `station_static_mode` with three explicit values:
  - `none`: apply relative TauP station travel times only and perform no
    station residual-lag search.
  - `tabulated`: add the configured station-table correction to the TauP
    shift and evaluate correlation at that fixed alignment.
  - `cross_correlation`: add a waveform-estimated residual lag to the TauP
    shift and write measured station-static workbooks.
- Legacy JSON remains supported: `use_station_static_correction: true` maps
  to `tabulated`, and `false` maps to `cross_correlation`. The active
  configuration was migrated to `station_static_mode: "tabulated"`.
- Station and event correction choices remain independent. In tabulated
  station mode there is no varying-lag station search; the correlation used
  for station quality is measured at the imposed TauP-plus-table alignment.
  Cross-event varying-lag alignment is controlled separately by
  `use_event_static_correction`.
- These changes, the documentation/task files, daily logs, static-comparison
  artifact, tests, and revised stacker workflow were committed and pushed as
  `4348661` (`Add static alignment modes and stacker workflow`) on
  2026-07-25.

### Static-Shift Comparison and Interpretation

- The durable comparison artifact
  `outputs/static_comparison_20260724/CI_40353552_static_shift_comparison_3-10Hz.xlsx`
  compares stored and correlation-estimated S station shifts for event
  `CI_40353552` at 3--10 Hz. It contains 278 matched stations per component;
  251 Z, 278 R, and 274 T traces passed its correlation screen.
- The workbook reports stored-versus-estimated correlations of 0.315 for Z,
  0.962 for R, and -0.064 for T. For this event and report, the stored S
  correction closely follows the radial estimate but not the vertical or
  transverse estimates. This is a limited one-event observation, not a
  validated component-independent station correction.
- Exploratory comparisons were also discussed at 1--4 Hz and 8--20 Hz,
  including baseline removal before comparing shifts. No durable matching
  output artifact for those bands was confirmed in the repository, so those
  cases should be rerun before their numerical results are treated as project
  findings.

### Corrected `stacker.py` Workflow

- The earlier `stacker.py` read fixed continuous files under a frequency-named
  directory and was not connected to `align_stack` output. It was replaced
  with a postprocessor that reads per-event
  `<event>/<event>_<component>_stack.mseed` files and parameter snapshots from
  one explicit timestamped `align_stack` run.
- The research output prefix
  `/Users/jvidale/Documents/Research/FaultScanR/output/2026` is implicit. For
  example, `--run 0724_153034_7601` selects
  `output/20260724_153034_7601`. `--components Z R T`, `Z T`, or a single
  component processes exactly the requested components. Stacker does not
  refilter the inputs; its usable passband is inherited from the selected
  `align_stack` run.
- Stacker cross-aligns the event stacks to the configured reference event
  within the run's bounded event-stack lag, normalizes each event stack, and
  draws individual traces plus a point-by-point `nanmedian` as the heavy black
  trace.
- Event selection was tightened to accept only catalog rows whose `skip`
  value is numerically zero. A `smaller` masking policy replaces the fixed
  4.6--6.0 s window after each smaller interfering event origin with `NaN`;
  each masked segment is 1.4 s. Other policies are
  `comparable_or_larger`, `all`, and `none`.
- Editable `DEFAULT_*` values near the top of `stacker.py` make the VS Code
  Run triangle reproduce the chosen run, components, masking, and display
  behavior without command-line arguments. At the recorded checkpoint the
  defaults select run `0724_153034_7601`, Z/R/T, smaller-event masking,
  overlay-only output, an interactive Matplotlib window, and no PNG save.
  Command-line arguments still override every default.

### Runs, Plots, and Performance Observation

- Stacker was run on `align_stack` output
  `20260724_153034_7601`, a 3--10 Hz Z/R/T run. The saved stacker directory
  contains:
  - `stack_segments_overlay.png`
  - `stack_segments_median.png`
  - `stack_segments_offset_pre_mask.png`
  - `stack_segments_offset_post_mask.png`
  - `stack_segments_overlay_smaller_events_masked.png`
- The stricter `skip = 0` selection identified 27 events with all three
  component stacks and was displayed interactively with saving disabled.
  The saved smaller-event-masked PNG predates that final no-save interactive
  invocation and should not be assumed to document the strict 27-event
  selection without regeneration.
- The selected `align_stack` run produced 639 PNGs and 105 component-stack
  MiniSEED files; 315 PNGs were individual-seismogram pages. Its first and
  last event snapshots were about 7 minutes 35 seconds apart. This supports
  the implementation-level inference that figure creation and saving,
  especially with individual and record-section plotting enabled, dominate
  runtime more than fixed-alignment correlation and averaging. Waveform
  reading, filtering, TauP calculations, and horizontal rotation remain the
  other principal costs.

### Correlation Screening Fix and Verification

- Review found that stations below `r_window_min` were marked rejected and
  excluded from the Stage 2 selected stack, but were still appended to
  `aligned_bank_all` and therefore contributed to the final event mean.
- The final-stack path was corrected to average only `aligned_bank`, which
  contains stations passing the correlation screen. Rejected traces remain
  available for diagnostic plots, station correlations, and rejected-row
  reporting but no longer affect the event-stack MiniSEED used by stacker.
- Regression coverage confirms that adding a rejected trace cannot change the
  final stack. At the implementation checkpoint, focused alignment tests
  passed 42 tests and full discovery passed all 58 tests in `vidale_main`.

### Unresolved and Required Follow-Up

- Existing run `20260724_153034_7601` and its stacker figures were generated
  before the final rejected-trace exclusion was established. Rerun
  `align_stack.py`, update stacker's `DEFAULT_RUN`, and rerun stacker before
  interpreting regenerated event medians scientifically.
- Complete FS-003 and FS-004 by validating `none`, `tabulated`, and
  `cross_correlation` across representative events, components, frequency
  bands, station counts, alignment correlations, and stack coherence.
- Decide whether stacker's fixed 4.6--6.0 s interfering-event mask should
  remain a common S-window or be replaced by event-specific predicted arrival
  windows.
- The stored correction's strong agreement with R but disagreement with Z/T
  remains unexplained; orientation, phase selection, site response, signal
  quality, and event dependence remain candidate causes.

## 2026-07-26 Chat Addendum: Runtime Repair, Plot Limits, and Pre-P Noise Screening

### Runtime Repair and Stacker Plot Control

- A partial `aligned_bank_all` to `aligned_bank` refactor caused a real
  `KeyError` in `align_stack.py`: Stage 3 returned `aligned_bank`, while the
  final-stack caller still requested the removed `aligned_bank_all` key. The
  caller was corrected to use and pass `aligned_bank`, matching the
  correlation-screened final-stack behavior described above.
- `stacker.py` gained `--amplitude-limits MIN MAX`, defaulting to `-0.1 0.1`.
  It sets the visible y-axis range for overlay and median plots; offset plots
  retain their unbounded offset scale so stacked traces remain readable.
- The runtime repair and focused stacker/alignment tests were run in
  `vidale_main`; they used temporary test outputs only. No new scientific
  processing run, research output, or plot interpretation was produced by
  these fixes.

### Pre-P Trace-Quality Rejection

- `align_stack.py` and `align_utils.py` now apply an additional station-screen
  before a trace enters the final event stack. For each station with a
  computable TauP P arrival, the code calculates
  `max(abs(entire trace)) / median(abs(pre-P samples))`.
- The active `rp_input.json` setting is
  `trace_peak_to_pre_p_median_min: 10.0`. A trace is rejected when this ratio
  is **less than** 10, indicating that its strongest waveform amplitude is
  insufficiently above the pre-P background level. A station without a
  computable P arrival is not rejected by this additional rule. Setting the
  threshold to `null` disables it.
- The threshold is written into run-parameter snapshots. Per-component console
  output now separately reports accepted traces, traces rejected for any
  reason, correlation-threshold failures, and
  trace-peak/pre-P-median-threshold failures. A trace that fails both tests is
  included in both criterion counts but only once in the overall rejection
  total.
- The requested interpretation is that this is a background-noise screen, not
  a rejection of unusually large signals: high pre-P noise raises the
  denominator and lowers the ratio.

### Verification and Follow-Up

- Focused alignment tests, including regression coverage for the ratio
  calculation and separate rejection counts, passed 45 tests in
  `vidale_main` after the final `< 10` criterion was implemented.
- The value 10.0 is an operational threshold, not yet a scientifically
  validated quality criterion. FS-003/FS-004 should measure how it changes
  accepted station counts, correlations, and stack coherence across events,
  components, and frequency bands before it is treated as a settled setting.
- VS Code can warn when its editor buffer conflicts with a file changed on
  disk. In that situation, compare or reload the disk version before saving;
  keeping the stale editor version can overwrite these uncommitted changes.

## 2026-07-26 Chat Addendum: Durable Codex Guidance and Documentation Sync

### Decisions and Repository Context

- A concise repository-root `AGENTS.md` was added to give Codex durable
  FaultScan-specific guidance. It directs agents to preserve unrelated working
  tree changes, avoid scientific processing and input/output mutations unless
  explicitly requested, use the README/status/goals/tasks/daily-note files for
  their respective purposes, and distinguish implementation state from
  validated scientific results.
- The guidance records the `vidale_main` full-test command and requires tests
  and scientific processing to be reported separately. It also requires
  provenance for configuration and outputs, and prohibits overwriting
  station-static tables or catalog time shifts without an explicit request and
  validation plan.
- Automation-local `memory.md` remains useful only for continuity of the
  daily-log automation on this machine. It is not a substitute for
  repository-tracked guidance or project notes.

### Documentation Synchronization

- Daily logs for 2026-07-25 and 2026-07-26 were committed and pushed as
  `e60e73b` (`Add July 25 and 26 daily logs`). The synchronization contained
  documentation only; it did not publish the unrelated pending source,
  configuration, test, or research-processing changes.
- This addendum records project workflow and documentation decisions only. It
  introduces no scientific result, output file, plot, or completed task.

## 2026-07-27 Chat Addendum: Documentation Review and GitHub Synchronization

### Scope and Synchronization

- This chat requested an evidence-based daily log, an update to this durable
  summary, and synchronization with the configured GitHub remote. `git fetch
  origin` confirmed that local `main` and `origin/main` were aligned at
  `997d177` before the documentation updates; no remote-only commits were
  found.
- The review covered the repository state, recent commits and reflog, task
  queue, active configuration, README, project status, prior daily log, and
  this summary. It did not run tests or the scientific processing pipeline,
  modify scientific inputs or outputs, or complete a task.

### Current Pending Implementation State

- The active pending `rp_input.json` selects 250 Hz snippets, all channels,
  S alignment, a 1--4 Hz band, a -2--12 s interval, enabled catalog event
  corrections, `station_static_mode: "tabulated"`, a 0.2 correlation
  threshold, a 0.05 s maximum station move, and
  `trace_peak_to_pre_p_median_min: 20.0`. JSON event-location override and
  individual/overlay/record-section plotting are enabled.
- The uncommitted worktree also contains changes to alignment utilities and
  pipeline code, stacker, the R/T snippet plotter, tests, README, and the
  project-status note, plus a new untracked snippet-plotter test. The working
  tree also contains local `.DS_Store` metadata, which is not project evidence.
- These changes are implementation state only. The earlier recorded 45-test
  focused checkpoint and 58-test full-discovery checkpoint remain historical
  claims; they were not rerun or independently reviewed in this chat. No new
  scientific processing or result was established.

### Durable Follow-Up

- Keep FS-003, FS-004, and FS-005 open until their explicit representative
  validation criteria are met. In particular, evaluate station-static modes
  with matched runs and do not treat the current 1--4 Hz tabulated
  configuration as a scientific conclusion.
