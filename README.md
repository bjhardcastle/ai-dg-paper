# Dynamic Gating analysis

This repository develops a publication-oriented analysis of reward-dependent
visual behavior in the 99 session-level Neuropixels recordings from
[DANDI:001051/0.260825.2232](https://doi.org/10.48324/dandi.001051/0.260825.2232).
The immutable DANDI version contains recordings from 27 mice. Probe-level LFP
files are present too, but the primary analysis uses spikes from session NWBs.

The scientific workflow is proposed in
[`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md), with technical details in the
[`implementation specification`](docs/ANALYSIS_IMPLEMENTATION_SPEC.md).
Reusable, read-only access and quality control live in `src/dg`; analysis
orchestration will remain in scripts and marimo notebooks while the workflow is
being reviewed.

## Dependency status

The project pins the public `lazynwb` commit whose package version is exactly
`1.0.0.dev8`. The commit pin is necessary because this prerelease is not
published on PyPI; it also makes the implementation used by the manuscript
unambiguous.

## Access pattern

Keep NWB table reads lazy until the final collection step so scalar predicates
and column projection are pushed down to the cloud read. Metadata and internal
path discovery are intentionally eager because they are small:

```python
import dg.data
import dg.quality

sources = dg.data.get_session_nwb_sources(max_sessions=1)

session_qc = (
    dg.quality.summarize_session_behavior(dg.data.scan_trials_with_reward_epochs(sources))
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

The session screen defaults to the final ten minutes of the no-reward block and
excludes the last ten minutes of the second engaged block. It requires engaged
go responses, catch-trial discrimination (loglinear-corrected d-prime), and
suppressed no-reward go responses. Its numeric cutoffs remain proposals until
the human-review decisions in the analysis plan are locked.
`dg.quality.summarize_coarse_engaged_rate_stability` is explicitly a
diagnostic/sensitivity screen, not the proposed primary condition-matched unit
stability filter.

All multi-file joins must include `_nwb_path`. By default, raw table-scan
helpers preserve `_table_path` and `_table_index`, allowing each scanned row to
be traced to its NWB table. Session- and unit-level aggregates retain their
source/key columns; analysis scripts must also write the input manifest and
transformation metadata specified in the analysis plan rather than implying
that an aggregate maps to one source-table row.
