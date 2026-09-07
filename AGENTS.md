## Dynamic Gating dataset 

## Your mission
- You are a world-class neuroscience researcher, tasked with creating a manuscript suitable for publication in a peer-reviewed neuroscience journal (e.g. Nature Neuroscience).
- each figure should convey one clear result with as many subpanels as are absolutely necessary to convince the reader.

## Data Sources

- NWBs on DANDI: https://dandiarchive.org/dandiset/001051 also has a readme with a description of the dataset and experimental design.


## Ideas for exploration 
- The overarching goal is to identify a candidate circuit by which stimulus-driven licking behavior is gated by reward availability. 

Focus on the following scientific priorities:
    - Finding functional response clusters with interesting dynamics at the beginning (or end) of the no-reward block.
    - Finding functional response clusters with sensory and/or motor activity modulated by the availability of reward.
    - Identifying regions that are enriched in these functional clusters.
    - Identifying network interactions between simultaneously recorded areas that might explain how sensory activity drives behavioral responses only when rewards are available.
    - Note that there are stimulus perturbations (novelty and contrast) that may be useful in characterizing functional clusters and disambiguating competing theories of stimulus gating.

## Analysis methods
- Good session filter: only use sessions during which the mouse licked reliably for image changes in both engaged blocks, but stopped licking during the no-reward block.

- Good unit filter: only analyze units that are well-isolated (isi violation ratio < 0.5, amplitude cutoff ratio < 0.1, quality == "good"). In addition, create a filter that excludes units for which activity during the two engaged blocks is not consistent. However, allow for the possibility that the mouse disengages from the task in the last 10 minutes.

- Always write tables of statistics for every analysis. Where possible aggregate on the mouse level. If not possible, use hierarchical bootstrapping. Cross-validate metrics when possible.

- For neural analysis, we are mainly interested in spiking data.
- cite relevant literature for methods and scientific motivation
- do not create any synthetic data.
- save results and figures for all units, so others can browse.

## software preferences 
- import modules, not functions or classes (ie, use `import polars as pl` instead of `from polars import read_csv`). The exception is certain stdlib libraries like `from typing import ...`
- lazynwb==1.0.0dev8 for data access
```python
# /// script
# dependencies = [
#     "lazynwb==1.0.0dev8", # adds upath and polars
# ]
# requires-python = ">=3.11"
# ///
import polars as pl
import lazynwb
import upath

lazynwb.config.anon = True

```
- make high quality, efficient fundamental data access functions in the library, but prefer orchestration in scripts and mariom notebooks for now - we will solidify the library's role in running an efficient workflow at a later point. use in-line dependency metadata.



## Workflow rules
- When the report or figures change, commit the updated report pdf and figure pngs as "WIP" and push to github.

## Hard-won dataset facts
- DANDI `001051/0.260825.2232`: 99 session NWBs, 27 mice, 66,130 trials; the 567 probe-LFP NWBs are separate assets.
- The dataset mixes 63 `DynamicGating` and 36 `DynamicGating_DEV` sessions; expect cohort-linked schema differences.
- `no_reward_epoch`, `is_sham_change`, and `omitted_reward` are absent together in 22/36 DEV NWBs but present in all 63 non-dev NWBs.
- The commit-pinned companion trial table covers 96/99 sessions overall and resolves all 13,004 trials in the 22 missing-flag sessions; current audit has zero unresolved labels or NWB/companion conflicts. Never infer blocks from nominal times, `active`, `rewarded`, or `stimulus_block`.
- Root `identifier` supplies the ecephys session ID; standard NWB `session_id` is often null. Multi-file joins with lazynwb must include `_nwb_path`.
- Task presentations live at five mutually exclusive table paths: one generic DEV path (36 sessions) and four image-set-specific paths (14--17 sessions each). Discover per session; never hard-code one.
- Trial `change_frame` may be task-relative while presentation `start_frame` is session-global. Infer one unique integer origin per session from exact image/change matches; timestamps are diagnostics only.
- `is_change` is not a physical-change flag: 2,913 catch/sham trials have `is_change == true` without an identity change, and 535 auto-rewarded physical changes are not `go`. Compare image identities explicitly.
- Contrast is encoded in image tokens, not a numeric trial column; observed trial tokens include `im115_r-0.7` and `im115_r-1.0`. The later passive-flash table is unrelated to task contrast. `is_image_novel` labels identity, not a novelty epoch; novelty is confounded with image/day.
- `auto_rewarded` cannot identify E2 onset; at least one audited session has auto-rewarded trials only in E1.
- All 99 NWBs declare the same strict response window, `(0.150, 0.750]` s. Derive responses from exact-deduplicated raw `lick_times`, not online outcomes; engaged online/raw labels agree only as an audit (~99.3--99.4%).
- In NR, online outcomes are absent for 1,007/14,305 completed go/catch trials despite continued licking. One session alone contains all 256 duplicated lick timestamps across 255 trials; no invalid or unsorted lick vectors were found.
- All 99 sessions have valid E1 -> NR -> E2 structure; median block durations are about 15, 25, and 25 min. All are technically valid/contrast-estimable, but the locked behavior threshold retains only 34 sessions from 18 mice.
- Core lick/reward/running/stimulus/optotagging clocks exist in all sessions; eye tracking is absent only in sessions `1187475832` and `1202438738`. Clock audits are metadata-only: timestamp vectors remain unvalidated.
- No raw-video asset/path was name-matched, but that audit inspected metadata/path names only; do not treat it as proof that video values are absent.
- `lazynwb==1.0.0.dev8` must be git-pinned (not PyPI). Its public materializer misses contiguous VLEN `quality` strings; use `dg.lazynwb_obstore`. Stream ragged `spike_times` in bounded batches (current neural run uses 32 units).
- Current behavior signal: mean go-response probability is E1 `0.666`, late-NR `0.151`, early-E2 `0.561`; recovery is strong but incomplete (E2 - E1 `-0.104`).
- Boundary-local behavior exceeds midpoint pseudo-boundaries at withdrawal (`+0.144`) and restoration (`+0.500`). Change-vs-catch specificity is uncertain at withdrawal but strong at restoration (`+0.402`).
- Current discovery-only neural signal: 20,667/48,600 units pass isolation QC in 23 sessions; early E1-vs-NR activity decodes state out of fold (mouse-mean AUC `0.835`). The axis jumps at restoration beyond midpoint drift (`0.204`) but not withdrawal (`0.008`); E2 recovery is suggestive, not multiplicity-significant.