# Dynamic Gating analysis

This repository develops a publication-oriented analysis of reward-dependent
visual behavior in the 99 session-level Neuropixels recordings from
[DANDI:001051/0.260825.2232](https://doi.org/10.48324/dandi.001051/0.260825.2232).
The immutable DANDI version contains recordings from 27 mice. Probe-level LFP
files are present too, but the primary analysis uses spikes from session NWBs.

The approved scientific workflow is recorded in
[`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md), with technical details in the
[`implementation specification`](docs/ANALYSIS_IMPLEMENTATION_SPEC.md).
Reusable, read-only access and quality control live in `src/dg`; analysis
orchestration remains in scripts and marimo notebooks. Decisions D01--D15 in
[`config/analysis_lock.yaml`](config/analysis_lock.yaml) are approved. A
preliminary discovery-only neural-transition Figure 2 now uses the fixed
outcome-blind mouse split and existing behavior QC; confirmation identities and
neural data remain sealed. The one-time confirmation run still requires frozen
discovery-dependent windows, representations, regions, and models. Figure 1 and
Supplementary Figure S1 remain exploratory because they were developed before
approval. Discovery-only Figures 3--6 are complete; Figure 5 reports the
familywise-null simultaneous-network screen rather than a supported pathway.

## Dependency status

The project pins the public `lazynwb` commit whose package version is exactly
`1.0.0.dev8`. The commit pin is necessary because this prerelease is not
published on PyPI; it also makes the implementation used by the manuscript
unambiguous.

Neural scripts do not open NWBs with a direct legacy file accessor. Numeric
unit metadata, anatomy strings, and ragged `spike_times` use lazynwb's custom
reader over obstore byte ranges. The pinned dev8 public materializer does not
yet expose contiguous variable-length strings, so `dg.lazynwb_obstore` bridges
its range reader and HDF5 parser without leaving the obstore path. Spike
histories are reduced one session at a time in bounded 32-unit batches; no
pipeline materializes spike arrays across sessions.

## Access pattern

Keep NWB table reads lazy until the final collection step so scalar predicates
and column projection are pushed down to the cloud read. Metadata and internal
path discovery are intentionally eager because they are small:

```python
import dg.data
import dg.quality

sources = dg.data.get_session_nwb_sources(max_sessions=1)

trials = dg.quality.add_trial_response_from_licks(
    dg.data.scan_trials_with_reward_epochs(
        sources,
        columns=dg.data.BEHAVIOR_TRIAL_COLUMNS,
    )
)

session_qc = (
    dg.quality.summarize_session_behavior(trials, response_column="response_in_window")
    .pipe(dg.quality.add_good_session_flag)
    .collect()
)

units = (
    dg.data.scan_units(sources, well_isolated=True)
    .select(
        "id",
        "peak_channel_id",
        "_nwb_path",
        "_table_path",
        "_table_index",
    )
    .collect()
)
```

The unit screen requires `isi_violations < 0.5`, `amplitude_cutoff < 0.1`, and
the authors' `quality == "good"` label. The session screen defaults to the
final ten minutes of the no-reward block and
excludes the last ten minutes of the second engaged block. It requires engaged
go responses, catch-trial discrimination (loglinear-corrected d-prime), and
suppressed no-reward go responses. Its numeric cutoffs are frozen by the
approved analysis lock.
Responses are derived from exact-deduplicated raw trial `lick_times` in the
common (150, 750] ms window, with raw and duplicate counts retained for audit.
The online `hit` and `false_alarm` events are retained for agreement
audits but are not used as cross-state response labels because outcome events
can be absent while licks continue in the no-reward epoch.
`dg.quality.summarize_coarse_engaged_rate_stability` is explicitly a
diagnostic/sensitivity screen, not the approved primary condition-matched unit
stability filter.

All multi-file joins must include `_nwb_path`. By default, raw table-scan
helpers preserve `_table_path` and `_table_index`, allowing each scanned row to
be traced to its NWB table. Session- and unit-level aggregates retain their
source/key columns; analysis scripts must also write the input manifest and
transformation metadata specified in the analysis plan rather than implying
that an aggregate maps to one source-table row.

Task-presentation materialization is fail-closed. For every canonical session,
the projected `_table_index` must be the complete contiguous sequence from zero
through `n - 1`; a missing, duplicate, or out-of-range row invalidates the whole
audit. Trial `change_frame` can be task-block-relative while presentation
`start_frame` is session-global, so alignment infers one integer origin offset
independently per session. The offset must be the unique solution that maps
every finite trial anchor one-to-one using exhaustive exact non-omitted image-
token and physical-change matches. After correction, every matched presentation
start must fall strictly within its trial interval. Timestamps are retained as
descriptive diagnostics only and never infer, rank, or select the offset.

## Reproduce the current results

The scripts contain inline dependency metadata. From the repository root, run:

```sh
uv run scripts/00_freeze_and_audit.py
uv run scripts/01_analyze_behavior.py
uv run scripts/03_allocate_cohorts.py
uv run scripts/04_analyze_behavior_transitions.py
uv run scripts/02_plot_behavior.py
uv run scripts/05_plot_behavior_transitions.py
uv run scripts/06_analyze_neural_transitions.py
uv run scripts/07_plot_neural_transitions.py
uv run scripts/08_analyze_functional_populations.py
uv run scripts/09_plot_functional_populations.py
uv run scripts/10_analyze_anatomy.py
uv run scripts/11_plot_anatomy.py
uv run scripts/12_analyze_network.py
uv run scripts/13_plot_network.py
uv run scripts/14_analyze_perturbations.py
uv run scripts/15_plot_perturbations.py
make -C manuscript
```

Run the spike-reading analyses (`06`, `08`, `12`, and `14`) serially. They
checkpoint by session and use 32-unit obstore batches so an interrupted run can
resume without accumulating neural data from multiple sessions.

The audit and behavior commands read the immutable remote release and may take time;
the first D05 allocation also retrieves content-pinned Allen ontology resources. If a
completed real-data `results/tables/behavior_trials.parquet` is already present,
the behavior summaries can be regenerated without rescanning NWB trial tables:

```sh
uv run scripts/01_analyze_behavior.py --reuse-behavior-trials
```

That recovery path verifies exact inventory coverage before reuse; it does not
generate or substitute synthetic data. Reuse also requires
`results/manifests/behavior_trials_checkpoint.json`, written only by a completed
direct remote-NWB run, and verifies the exact trial parquet, session inventory,
and pinned companion-table hashes. A reuse run never refreshes that trusted
checkpoint. The behavior run manifest records hashes and byte sizes for every
output plus the generator and all local analysis modules.

## Current outputs

- Audit provenance and frozen inputs: `results/manifests/`, including
  `audit_run.json`, `dandiset_assets.parquet`, exact per-session schema and clock
  metadata audits, projected scalar unit/stimulus/trial inventories, row-level
  trial--presentation alignment, and coverage summaries.
- Behavior analysis tables: `results/tables/behavior_statistics.csv`,
  `session_attrition.csv`, mouse/session block summaries, response audits, and
  the row-level `behavior_trials.parquet`.
- Outcome-blind D05 allocation: content-addressed mouse/session assignments,
  metadata and official-major-division balance tables, ontology provenance, and
  `results/manifests/cohort_allocation_run.json`. Only the authenticated artifact
  selector can activate discovery sessions; confirmation flags remain sealed.
- Behavior-transition QC: real- and matched-pseudo-boundary trajectories, exact
  effect-window support, mouse-level effects, and
  `results/tables/behavior_transition_statistics.csv`.
- Figure 1: SVG, PDF, and PNG in `results/figures/main/`, with a manifest and
  panel-level source-data CSVs beside them. The plotted symmetric-score row is
  `gating_summary_statistic.csv`; it is a summary, not a primary inferential
  test.
- Supplementary Figure S1: SVG, PDF, and PNG in
  `results/figures/supplementary/`, with source data and a fail-closed manifest.
- Figure 2: discovery-only reward-state-axis trajectories and mouse effects as
  SVG, PDF, and PNG in `results/figures/main/`; source-data CSVs and a figure
  manifest sit beside them. The analysis tables include all-unit QC and axis
  weights, trial scores, session/mouse summaries, and
  `results/tables/neural_transition_statistics.csv`. The run manifest is
  `results/manifests/neural_transition_analysis_run.json`.
- Figure 3: all-unit functional profiles, cross-half response labels,
  mouse-level timing effects, running-clock QC, and the movement-adjusted
  statistics in `results/tables/functional_population_statistics.csv`.
- Figure 4: CCFv3 coverage and mouse-level enrichment of Figure 2 axis-loading
  magnitude, including the session-preserving permutation null and
  `results/tables/anatomy_enrichment_statistics.csv`.
- Figure 5: the complete 20-pair screen, nested-CV block performance, nominated
  Midbrain-to-Isocortex visualization, and familywise-null inference in
  `results/tables/network_statistics.csv`.
- Figure 6: held-out reduced-contrast and designated-identity behavioral/neural
  probes with `results/tables/perturbation_statistics.csv`.
- Preliminary manuscript: `manuscript/results.typ`; compilation writes
  `manuscript/build/results.pdf`.

The authoritative status column for current Figure 1 statistics is `pass`
under the approved lock. The analysis remains exploratory because it was
developed before approval. The approved threshold selects 34 sessions from 18
mice as a QC view; the preliminary behavior result uses all 99 technically
valid, estimable sessions from 27 mice. The
directional withdrawal and restoration contrasts are the primary tests; the
symmetric gating score plotted in Figure 1 is summary-only. Milestone 0 remains
explicitly partial: full timestamp-vector
validation, protocol-level novelty history, optotagging/RF/gabor/flash inventories,
generic embedded-video detection, and the frozen primary CCF mapping remain open.
The discovery vertical slices record their narrower inputs and limitations
directly. D04 engaged-block consistency is not yet applied; Figure 3 adjusts
measured licking and running, but pupil/orofacial controls remain incomplete,
and Figure 5 does not support a circuit interaction after correction.
Confirmation identities and neural data remain sealed.
