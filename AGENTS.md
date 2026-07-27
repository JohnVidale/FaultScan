# FaultScan repository guidance

## Scope and safety

- Preserve unrelated working-tree changes. Stage, commit, or edit only files
  required for the requested task.
- Do not run the scientific waveform-processing pipeline, alter scientific
  inputs, or generate/overwrite research outputs unless the user explicitly
  asks for that work.
- Treat source changes, configuration changes, and generated files as
  implementation state—not as validated scientific results—until supporting
  evidence is recorded.

## Authoritative project context

- Use `README.md` for repository workflow and program orientation.
- Use `Notes/faultscan_project_status.md` for current processing status and
  open technical questions.
- Use `Notes/faultscan_science_goals.md` for scientific objectives and
  interpretation criteria.
- Use `TASKS.md` as the curated work queue. A task is not complete merely
  because it appears there; follow its stated completion and verification
  criteria.
- Use `Notes/daily/YYYY-MM-DD.md` for dated, evidence-based activity records.
  Keep current-day observations distinct from historical results.

## Verification and reporting

- Prefer repository evidence: commits, diffs, configurations, tracked output
  provenance, and recorded test results. State uncertainty rather than infer
  processing or validation that is not evidenced.
- When tests are requested, use the `vidale_main` environment. The full suite
  command is:

  ```bash
  conda run -n vidale_main python -m unittest discover -s tests
  ```

- Report tests and scientific processing separately. If either was not run,
  say so explicitly.
- For changes affecting alignment, statics, screening, or stack selection,
  distinguish code-level verification from representative scientific
  validation across the relevant events, components, frequencies, and station
  counts.

## Configuration and outputs

- Treat `rp_input.json` as the active alignment configuration; describe its
  settings accurately when they affect a result.
- Preserve output provenance, especially run parameters, event/component/phase
  selection, and whether a report derives from a historical run or a rerun.
- Do not overwrite station-static tables or catalog time shifts without an
  explicit request and validation plan.
