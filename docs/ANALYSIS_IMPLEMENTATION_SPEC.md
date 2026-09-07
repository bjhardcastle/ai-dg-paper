# Dynamic Gating: detailed analysis implementation specification

- Status: **technical companion to the high-level plan; not yet preregistered or locked**
- Dataset: [DANDI:001051/0.260825.2232](https://dandiarchive.org/dandiset/001051/0.260825.2232), immutable published version
- Primary modality: extracellular spikes
- Scientific scope: identify a candidate distributed pathway by which visual changes drive licking when reward is available, but fail to do so when reward is unavailable.

This plan is intentionally stricter than an exploratory notebook. It separates discovery from confirmation, treats the mouse as the biological replicate, specifies falsification and robustness analyses, and requires a machine-readable statistics table for every reported result. It does not authorize looking at confirmatory neural outcomes before the decision gates below are closed.

## 1. Proposed central claim and causal limits

The strongest claim this observational dataset can support is:

> Reward availability is associated with a reversible reconfiguration of a reproducible, anatomically enriched, inter-regional neural pathway that preserves early visual information while changing its predictive relationship to lick-related population activity.

The working primary hypothesis is therefore a **routing-gate hypothesis**, not a uniform suppression hypothesis: early sensory responses remain detectable when rewards are unavailable, whereas late sensory/action responses and cross-area predictive coupling weaken. A candidate pathway should satisfy all of the following:

1. A source population encodes the image/change before the lick response window in both rewarded and no-reward blocks.
2. A downstream population contains reward-state-dependent late sensory or pre-lick activity and lick-related activity.
3. Source activity improves out-of-sample prediction of downstream activity beyond stimulus, movement, arousal, session time, and the downstream population's own recent activity.
4. This incremental prediction changes reversibly across the first engaged block, no-reward block, and second engaged block.
5. The functional populations and interaction occur in a consistent set of regions and replicate across mice.
6. Where perturbation factors pass an estimability/counterbalancing audit, held-out novelty and contrast responses support the pattern predicted by the proposed computation; otherwise they constrain interpretation without selecting a winner.

Failure of any link weakens the interpretation. Because reward availability was manipulated but neural nodes were not, words such as **association**, **predictive interaction**, and **candidate pathway** are appropriate. “Causal circuit,” “information flow,” “mediation,” and anatomical direction should not be used as conclusions without a neural perturbation or another design that identifies causality.

The motivation follows work showing that motivational state can gate brain-wide cue-to-action dynamics ([Allen et al., 2019](https://doi.org/10.1126/science.aav3932)), that action and engagement signals are widespread while visual and choice signals are more regionally structured ([Steinmetz et al., 2019](https://doi.org/10.1038/s41586-019-1787-x)), and that inter-area prediction can occupy a low-dimensional communication subspace ([Semedo et al., 2019](https://doi.org/10.1016/j.neuron.2019.01.026)).

## 2. Dataset ground truth to establish before analysis

### 2.1 Provisional inventory

The following facts are provisional and must be regenerated from the frozen DANDI version rather than copied into the manuscript:

- The [published DANDI metadata](https://api.dandiarchive.org/api/dandisets/001051/versions/0.260825.2232/) describes 99 electrophysiology sessions from 27 mice, 666 files, and approximately 1.81 TB. It describes multi-Neuropixels recordings throughout the left hemisphere and a middle no-reward block with the lick spout still available. This release was published on August 25, 2026, with DOI [10.48324/dandi.001051/0.260825.2232](https://doi.org/10.48324/dandi.001051/0.260825.2232).
- The public [session metadata table at the pinned companion commit](https://github.com/AllenInstitute/SHIELD_Dynamic_Gating_Analysis/blob/3c09a0dc2972381f06393ddf5c000e16462c7037/metadata_tables/dynamic_gating_session_metadata.csv) also has 99 sessions across four recording days (`EPHYS_1` through `EPHYS_4`), with repeated sessions per mouse.
- A schema audit found trial-level `no_reward_epoch`, `is_sham_change`, and `omitted_reward` in 77 session NWBs and absent from 22 NWBs spanning six mice. For those files, use the authors' commit-pinned master stimulus table as an audited fallback; it covers 96 of 99 sessions and the preliminary join covers all 22 missing-flag NWBs. Regenerate that join audit, and exclude any unresolved session with a reason rather than assigning it from nominal task time.
- The nominal `no_reward` task attribute is `[900, 2400]` seconds, but observed trial flags transition later (approximately 925 and 2426 seconds in an inspected session). Programmed times therefore cannot substitute for trial-level block labels.
- Task-image presentations occur under five session-dependent interval-table names. The later `flash_250ms_presentations` table is passive and its `contrast` field is not the task contrast perturbation. Task contrast is encoded in names such as `im115_r-0.7` versus `im115_r-1.0`; physical changes must be derived from initial and changed image identities because `is_change` includes go and catch/sham events.
- `is_image_novel` is a presentation-level identity label, not by itself a novelty epoch. Day-specific novel-image and cumulative-exposure semantics must be verified from the protocol before use as factors.
- The public master stimulus table covers 96 of the 99 sessions, and public session-level unit counts do not exactly match the number of rows in the public units table. These discrepancies are an audit item, not an exclusion rule.
- Ninety-seven sessions are marked as having LFP, but the planned primary analyses use spikes. LFP should remain a secondary, separately reviewed analysis.

The dataset was introduced with the SHIELD preparation in [Bennett et al. (2024)](https://doi.org/10.1016/j.neuron.2024.06.015). The related Visual Behavior Neuropixels task and neural response framework are described by [Bennett et al. (2026)](https://doi.org/10.1016/j.cell.2026.06.025) and [Siegle et al. (2021)](https://doi.org/10.1038/s41586-020-03171-x).

### 2.2 Required immutable data manifest

Before any neural result is considered final:

1. Use immutable DANDI version `0.260825.2232`; do not substitute `draft` in the manuscript workflow.
2. Save Dandiset version, DOI, retrieval date, every asset ID, path, size, etag/checksum, subject, acquisition date, and session ID.
3. Identify the session NWB versus per-probe LFP NWBs explicitly.
4. Reconcile the 99-session inventory against NWB assets and the public metadata tables; document missing, duplicated, malformed, or superseded sessions.
5. Store raw region acronyms/CCF coordinates and the exact CCF ontology version used to derive parent regions. Use the Allen CCFv3 framework ([Wang et al., 2020](https://doi.org/10.1016/j.cell.2020.04.007)).
6. Audit schemas across every session before collecting array-valued columns. Record missing and type-drifting columns.

The access implementation uses `lazynwb==1.0.0.dev8`, pinned to public commit `387c250bee6a6fddd5c96b9cf1490b8f02c292f8` because that prerelease is not published on PyPI. Use Polars lazy scans in the order scan -> filter -> select -> collect. Multi-table joins must include the NWB path as a key in addition to table indices/IDs. The audit must verify, rather than assume, the exact paths and meanings of:

- units and spike times;
- trials and stimulus presentations;
- lick and reward timestamps;
- running, pupil, and eye/face measurements if present;
- probe, electrode/channel, CCF coordinate, and structure fields;
- auto-reward, omission, change/catch, image identity, contrast, active/no-reward, and block-boundary fields;
- optotagging epochs and receptive-field mapping stimuli;
- acquisition-clock alignment and dropped/duplicated timestamps.

No synthetic observations, synthetic sessions, or imputed spike trains will be created. Missing behavioral covariates may be handled by prespecified missingness strata or complete-case sensitivity analyses, never by inventing measurements.

Reward availability must come from the audited trial flag (NWB first, pinned companion fallback). The presentation fields `active`, `rewarded`, or `stimulus_block` must not be repurposed as reward-state labels.

## 3. Decisions that require human approval

Every row below must receive an explicit decision in a dated `analysis_lock` document. Recommended defaults are proposals, not silent choices.

| ID | Decision needing human review | Recommended default | Consequence if unresolved |
|---|---|---|---|
| D01 | Immutable DANDI version | Use published version `0.260825.2232` and analyze only its asset manifest | Substituting the draft would make results non-reproducible |
| D02 | Task-event semantics | Confirm block boundaries, response window, catch definition, auto-reward sequence, novelty, and contrast against NWBs and experimenter documentation | Factors and behavioral QC may be mislabeled |
| D03 | Session inclusion thresholds | Lock thresholds using behavior only; use the proposed two-engaged-block/late-no-reward rule below | Neural selection could become outcome-dependent |
| D04 | Engaged-block unit consistency filter | Use a noise-calibrated equivalence filter based only on shared familiar stimuli and pre-stimulus activity; exclude the last 10 minutes | Stability filtering could erase biology or admit drift |
| D05 | Discovery/confirmation split | Split by mouse after behavior QC and before inspecting neural outcomes; target approximately 2/3 discovery and 1/3 confirmation | Unit-level leakage would invalidate confirmation |
| D06 | Primary time windows | Validate 25-ms modeling bins and prespecify early sensory, late/pre-lick, lick, and baseline windows | Window selection could follow the observed result |
| D07 | Functional representation | Ridge-regularized event-kernel spike model with out-of-fold deviance/dropouts; also report model-free PSTHs | Cluster inputs otherwise remain underspecified |
| D08 | Discrete clusters versus continuous axes | Permit discrete labels only if mouse-bootstrap stability and held-out mapping pass; otherwise report continuous functional axes | Unstable clusters could be overinterpreted |
| D09 | Anatomical granularity and coverage | Lock one CCF parent level and minimum mouse/session/unit coverage before enrichment testing | Region fragmentation creates unstable multiple testing |
| D10 | Primary interaction analysis | Lagged regularized reduced-rank prediction of target residual activity, with target history and measured covariates | “Network interaction” remains too flexible |
| D11 | Candidate source/target regions | Nominate at most a small number in discovery, then test unchanged in held-out mice | Region-pair fishing inflates evidence |
| D12 | Genotype, sex, day, and DEV cohort handling | Treat as prespecified strata/covariates; make cell-type claims only with validated optotagging | Cohort composition may masquerade as anatomy/state |
| D13 | Missing pupil/video policy | Running and licks are core; pupil/video are adjusted where valid and tested in matched complete-case analyses | Different session sets could drive covariate results |
| D14 | Confirmatory viability threshold | Require enough eligible held-out mice and simultaneous region pairs; otherwise label the complete study exploratory | Underpowered “confirmation” would be misleading |
| D15 | Scope of LFP analyses | Keep outside the primary manuscript until the spike results and a separate LFP plan are reviewed | Scope expansion could delay or dilute the central result |
**Gate:** do not unlock confirmatory neural labels or run inferential tests in the held-out mice until D01-D14 are recorded and analysis code passes frozen tests on discovery data.

## 4. Experimental contrasts and hypotheses

Use three block labels throughout:

- `E1`: first reward-available/behaviorally engaged block;
- `NR`: reward-unavailable block;
- `E2`: second reward-available block, beginning at the audited trial-level return of reward availability. Reward or cue events around that boundary must be labeled separately after their semantics are validated.

Do not call all `NR` activity “disengaged” merely because licking falls. Reward availability is experimentally defined; engagement is a behavioral/latent state that must be measured.

Two orthogonal block contrasts help distinguish reversible gating from monotonic drift:

- reversible reward-availability contrast: `0.5 * E1 - NR + 0.5 * E2`;
- session-time/hysteresis contrast: `-0.5 * E1 + 0.5 * E2`.

For unequal block lengths, estimate these contrasts in matched-duration or matched-trial windows and in a model with continuous session time. A convincing reward-availability effect should differ in `NR` and return toward the `E1` state in `E2`; a monotonic E1-to-E2 trend is not sufficient.

### H1: a reward-state signal appears around block transitions

Prediction: a reproducible subset of units changes tonic or stimulus-linked firing near the start of `NR`, changes with behavior across the no-reward block, and reverses near the audited onset of `E2`. Onset and recovery need not have identical time constants; any coincident reward/cue response must be modeled separately.

Falsification: apparent transition responses also occur at time-matched pseudo-boundaries, follow a smooth session-time trend, are explained by licks/running/pupil, or do not reverse at `E2`.

### H2: reward availability changes sensory response gain

Prediction: early image/change responses or image decoding change between the engaged and no-reward blocks, possibly in a region- and cluster-specific manner. A global sensory-gain account predicts similar scaling across identities and contrast levels.

Falsification: early sensory kernels and image decoding generalize across state after controlling for arousal and movement, while only later activity changes.

### H3 (primary): reward availability changes sensory-to-action routing

Prediction: early sensory encoding remains relatively stable, but late/pre-lick activity, downstream lick prediction, and source-to-target incremental predictive coupling are larger in `E1/E2` than `NR`. The coupling change should be reversible and visible on no-lick trials or before the first lick.

Falsification: the interaction effect is eliminated by stimulus/behavior covariates, trial shifting, equalized population sizes, or mouse-held-out testing; or it is entirely explained by loss of sensory signal at the source.

### H4: gating is downstream motor suppression rather than altered inter-area routing

Prediction: sensory representations and source-to-target coupling remain stable, but lick-related/preparatory populations change state or gain. Spontaneous and false-alarm licks should recruit the motor population similarly across reward conditions if the effect is specific to action initiation rather than execution.

Falsification: downstream activity is altered well before movement and the source-to-target incremental prediction changes independently of lick occurrence.

### H5: identifiable novelty and contrast effects constrain competing accounts

Predictions:

- A global gain mechanism scales responses similarly for familiar, novel, full-contrast, and reduced-contrast inputs.
- A surprise/adaptation mechanism preferentially affects novel or low-probability events and decays with exposure, including during `NR`.
- A learned sensorimotor-routing mechanism preserves early identity/contrast coding but changes late change/lick coding and coupling according to reward availability.
- A motor-only mechanism follows lick probability/latency more closely than novelty or physical contrast.

Novelty is confounded with image identity and recording day unless the NWB design provides stronger counterbalancing than the public table indicates. Treat within-session exposure dynamics and mouse-level replication as primary; do not interpret a day effect as pure novelty.

## 5. Cohort definition and quality control

### 5.1 Session filter

Session inclusion must be computed without neural data. The NWB task parameters specify a 150-750 ms response window; verify this against task labels and the dataset's associated analysis ([Bennett et al., 2026](https://doi.org/10.1016/j.cell.2026.06.025)) before freezing it.

For each block/window, calculate:

- change-evoked response probability;
- catch/sham-change response probability and signal-detection d-prime where valid;
- lick latency distribution;
- lick bouts and total lick rate;
- rewarded and auto-reward counts;
- running and available pupil/eye summaries;
- number of analyzable change, catch, omission, and perturbation events.

Proposed primary rule, subject to D03:

1. `E1` and the portion of `E2` before its final 10 minutes each show reliable change detection: at least a minimum number of go/catch trials, change-response probability at least 0.5, and d-prime at least 1.0. For the nominal 25-minute E2 block this is its first 15 minutes; the relative rule remains defined for sessions with atypical duration.
2. The final 10 minutes of `NR` show behavioral suppression: change-response probability at most 0.2 and at least a 0.3 absolute decrease relative to the less responsive of `E1` and early `E2`.
3. The animal resumes after the audited E2 transition. Semantically validated reward/cue trials are excluded from ordinary contingent performance and modeled separately. Do not require the NWB `auto_rewarded` flag as an E2 marker: an inspected session carries that flag only on E1 trials.
4. The last 10 minutes of `E2` do not determine inclusion, so late task disengagement is allowed as requested.
5. Minimum trial counts and binomial confidence intervals are reported, and sessions with hardware/timing failures are excluded by reason codes independent of behavior.

Why late `NR`: animals may initially persist in licking while learning that reward has been removed. Using the full no-reward block as an exclusion criterion would select directly on extinction speed, the transition phenomenon of interest.

These numeric thresholds are starting proposals. Plot the distribution of each behavior-only criterion and have a human approve the cut points before neural outcomes are viewed. Keep a continuous “behavioral gating score” for sensitivity analyses so the conclusions do not depend entirely on dichotomization.

Required session-QC outputs:

- one row per session with every metric, threshold, pass/fail flag, and exclusion reason;
- one row per mouse/day showing whether repeated-session coverage is complete;
- an attrition table from 99 inventory sessions to analyzable sessions;
- a per-session behavior report around both transitions;
- a comparison of included and excluded sessions on mouse, sex, genotype, project cohort, day, probe count, and region coverage.

### 5.2 Unit filter

The mandatory primary isolation thresholds are:

- `isi_violations < 0.5`;
- `amplitude_cutoff < 0.1`.

Use strict `<`, not `<=`, unless D04 explicitly changes it. Do not silently add `quality == "good"`, firing-rate, SNR, presence-ratio, receptive-field, or region filters to the primary definition. Such metrics should be reported and may form prespecified sensitivity analyses. The Allen visual-system literature provides context for objective Neuropixels quality filtering ([Siegle et al., 2021](https://doi.org/10.1038/s41586-020-03171-x)).

### 5.3 Engaged-block consistency filter

This filter is meant to reject unstable recordings, not neurons that show the biological `NR` effect. It must therefore use only `E1` and early `E2`, never `NR`, and only conditions shared across those blocks.

Proposed noise-calibrated procedure, subject to D04:

1. Exclude the last 10 minutes of `E2`.
2. Use the two shared full-contrast familiar images on non-change presentations plus pre-stimulus bins. Exclude perturbation, omission, auto-reward, reward, and peri-lick bins.
3. Match `E1` and `E2` samples on image, time since last image/change/reward, running-speed bin, and lick-free status.
4. Construct a compact per-unit fingerprint: baseline rate plus early and late response bins for each shared image.
5. Estimate within-block split-half discrepancy and reliability with condition-stratified resampling. Compare the E1-to-E2 discrepancy with the unit's own within-block noise.
6. Require firing to be observed across multiple non-overlapping time bins in both blocks and require practical equivalence of overall rates. A candidate equivalence margin is a two-fold rate ratio plus a cross-block discrepancy no larger than a prespecified noise-calibrated bound.
7. Units too sparse to evaluate are labeled `stability_indeterminate`, not automatically stable.

Lock the exact reliability statistic, number of resamples, equivalence margin, and indeterminate-unit policy after examining reliability distributions **without region names, cluster labels, or NR activity**. The primary cohort combines isolation and stability requirements; repeat central analyses with isolation-only QC as a sensitivity analysis. A conclusion that exists only after aggressive stability filtering is fragile.

### 5.4 Anatomical QC and analysis eligibility

- Preserve raw CCF acronyms and coordinates for the unit browser.
- Exclude `out of brain`, `No Area`, root, ventricles, and fiber tracts from regional enrichment tests, but retain them with reason flags in all-unit outputs.
- Map raw acronyms to a single locked ontology level for primary inference; retain fine structures descriptively.
- Proposed minimum for a primary region: represented in at least 5 mice and 8 sessions, with at least 10 eligible units in the median represented mouse. Lock after a coverage-only audit.
- Report probe/session coverage and the anatomical sampling denominator. Cluster enrichment is always relative to eligible recorded units, not to the biological neuron population of a region.

## 6. Discovery, cross-validation, and confirmation

### 6.1 Split policy

After behavior-only session QC but before neural inspection, assign entire mice to discovery or confirmation using a fixed seed and a metadata-only stratification objective. Balance sex, genotype, DEV versus production cohort, number of eligible sessions, recording days, and coarse regional coverage as far as possible.

Recommended default:

- approximately two-thirds of eligible mice for discovery;
- approximately one-third for a one-time confirmation;
- no unit or session from a mouse may cross the boundary.

If fewer than 18 mice remain overall, fewer than 6 remain in confirmation, or a nominated interaction has fewer than 5 confirmatory mice with simultaneous source/target coverage, do not describe a holdout analysis as confirmatory. Use grouped nested cross-validation on all mice, label the study exploratory, and reserve confirmation for future data.

### 6.2 What discovery may change

Discovery data may be used to:

- validate event construction and QC implementation;
- choose time-basis complexity and regularization ranges;
- determine whether functional structure is discrete or continuous;
- nominate a small set of clusters, anatomical regions, and region pairs;
- refine visualization and select one primary interaction estimand;
- perform power/precision calculations based on mouse-level effect distributions.

Before confirmation, freeze code, containers/lockfiles, thresholds, seeds, feature transforms, cluster templates, region mapping, model hyperparameter search spaces, tests, figures, and table schemas. Map held-out units to frozen clusters/templates; never recluster the combined dataset and call the result confirmed.

### 6.3 Cross-validation rules

- All preprocessing learned from data—normalization, dimensionality reduction, feature selection, cluster centroids, region-pair selection, and hyperparameters—must be fit inside the appropriate training fold.
- Neural time-series models use contiguous temporal/trial blocks rather than randomly interleaved bins to reduce leakage from autocorrelation and slow state.
- Population generalization is evaluated with outer folds grouped by mouse. Within-session tuning may use inner contiguous folds.
- Repeated sessions from one mouse remain in the same outer fold.
- Trial-level splits shared by simultaneously recorded units preserve population vectors.
- Any pseudopopulation analysis is secondary to simultaneously recorded population analysis and cannot establish interaction.
- Use nested rather than single-loop cross-validation for tuned models; selection and error estimation in one loop is optimistically biased ([Varma and Simon, 2006](https://doi.org/10.1186/1471-2105-7-91)).

## 7. Canonical derived data

Build a small number of versioned, analysis-ready tables from NWB rather than repeatedly interpreting raw fields in notebooks.

### 7.1 Session and event tables

Create one canonical event table with stable keys and at least:

- DANDI version/asset/NWB path, mouse, session, acquisition date, and recording day;
- event and trial IDs, timestamps, block (`E1`, `NR`, `E2`), exact block-relative time, and transition-relative event index;
- image identity, validated novelty/familiarity, contrast, change/catch, omission, reward availability, response, response latency, reward, and auto-reward;
- previous images, repetition number, time since prior change/lick/reward, and cumulative image exposure;
- running, pupil/eye/video validity and summaries in prespecified windows;
- session-QC flag, discovery/confirmation label, and exclusion reasons.

Build a parallel continuous-time covariate table on one validated clock. Report timestamp monotonicity, missing intervals, and alignment errors.

### 7.2 Unit and spike-feature tables

Create one all-unit table with:

- raw unit metrics and all QC flags/reasons;
- raw and parent anatomy, coordinates, probe, mouse, and session;
- engaged-block consistency metrics and uncertainty;
- event counts available for every condition;
- out-of-fold model performance and every sensory/motor/state/transition feature;
- functional cluster posterior/distance or continuous-axis scores;
- discovery/confirmation status;
- links to model-free and model-based per-unit figures.

Store canonical large tables as Parquet and publication-facing statistics as CSV plus Markdown. No qualifying unit should disappear because it was not selected for a main figure.

## 8. Analysis modules

### A. Establish reversible behavioral gating

This analysis is both a result and the session-selection foundation.

1. Plot lick probability, latency, lick rate, reward rate, running, and pupil (where valid) over normalized session time and over change trials around both boundaries.
2. Estimate per-mouse `E1`, late `NR`, early `E2`, and late `E2` change/catch response rates.
3. Test the reversible block contrast at the mouse level. Use a binomial/logistic mixed model with mouse and session structure as a secondary trial-level check.
4. Estimate extinction and reacquisition time constants or change points per session with uncertainty; avoid assigning a transition latency when the data do not identify one.
5. Compare included and excluded sessions and report results under the continuous behavioral gating score.

Primary behavioral result: across technically valid sessions, mice tend to respond selectively to changes in both reward-available blocks, suppress responses late in `NR`, and recover after the audited E2 transition. The threshold-selected cohort is a QC characterization, not the inferential sample for establishing this population effect.

### B. Fit interpretable single-unit encoding models

Use a regularized spike-count model with temporal basis functions. A 25-ms bin is a reasonable starting point because the related VBN spike GLM used 25-ms bins and event kernels ([Bennett et al., 2026](https://doi.org/10.1016/j.cell.2026.06.025)); D06 must lock the final resolution after timing validation.

Candidate predictors:

- familiar/novel image identity and contrast;
- image onset, change, catch, omission, reward, and auto-reward;
- block/reward availability, continuous session time, and time/event index from both boundaries;
- lick events/bouts and response latency;
- running speed/acceleration and pupil/eye/video covariates where valid;
- stimulus history, repetition count, time since change/lick/reward;
- interactions of block with sensory and motor kernels.

Fit both:

1. a **total state-associated model**, excluding downstream behavioral covariates that could absorb the pathway from reward availability to action; and
2. a **movement-adjusted model**, including measured lick/running/arousal variables to ask what association remains beyond those measurements.

Do not call the second model a causal “direct effect.” Licking and arousal are consequences as well as correlates of reward state, and unmeasured orofacial movement may remain. Movement can explain widespread neural variance ([Musall et al., 2019](https://doi.org/10.1038/s41593-019-0502-4); [Stringer et al., 2019](https://doi.org/10.1126/science.aav7893)), while locomotion itself modulates visual responses ([Niell and Stryker, 2010](https://doi.org/10.1016/j.neuron.2010.01.033)).

Use held-out Poisson deviance explained (or another locked count-model score) and refit reduced models for grouped feature-dropout estimates. Report model-free PSTHs and spike counts beside model-based effects. Key per-unit features should include:

- early image/change sensory response;
- late sensory/change response before typical lick onset;
- lick-aligned and reward-aligned response;
- reversible state modulation of each response;
- tonic transition modulation;
- novelty, contrast, and exposure interactions;
- cross-validated model reliability.

Pre-lick analysis should include both a population-fixed early window and event censoring at the first lick. Compare change trials with the same stimulus but different response outcomes; analyze false-alarm/spontaneous licks to distinguish movement execution from change detection. Matching must not replace reporting the unmatched total association.

### C. Characterize dynamics at no-reward onset and offset

Analyze at least four anchors separately:

1. first audited trial-level transition into `NR` (with nominal programmed time only as a secondary alignment);
2. first expected-but-omitted reward after a change response in `NR`;
3. audited trial-level onset of `E2`, with validated reward/cue events modeled as separate anchors;
4. first subsequent contingent reward/criterion-level return of licking.

For each unit, model tonic rate, familiar-image response, change response, lick response, and model residuals as smooth functions of boundary-relative time or change index. Use piecewise splines/change-point models only when supported by out-of-sample performance; otherwise report prespecified early/middle/late windows.

Controls:

- identical analyses at many within-block pseudo-boundaries;
- time-reversed or circularly shifted boundaries within session;
- matched licks, running, pupil, image identity, contrast, and stimulus history;
- reward/auto-reward event kernels so the `E2` boundary is not mistaken for a pure state transition;
- the E1-to-E2 session-time contrast and smooth drift model.

Primary transition result: one or more reproducible response motifs track loss and restoration of reward availability more closely than smooth time, overt movement, or isolated reward delivery.

### D. Discover and validate functional response structure

Cluster only cross-validated, interpretable response features; do not cluster raw firing rate or anatomy. Standardize features within the training data, cap the influence of high-rate units, and keep region labels hidden during cluster construction.

Recommended feature families:

- model-free normalized PSTHs for familiar non-change, change, lick, reward, and omission events in each block;
- fitted sensory, motor, state, and transition kernels;
- reversible state contrasts and transition time-course coefficients;
- novelty/contrast effects only in a secondary characterization layer, so the primary clusters are not defined by the perturbation used to test them.

Recommended discovery sequence:

1. Remove units whose cross-validated model performance is indistinguishable from a time/stimulus-preserving null from discrete clustering; retain them as `unclassified` and in all outputs.
2. Reduce discovery features with a linear, fold-fitted method such as PCA. dPCA may be used descriptively to expose state, stimulus, action, and time axes ([Kobak et al., 2016](https://doi.org/10.7554/eLife.10989)). UMAP may visualize but must not determine inference on its own.
3. Compare a small prespecified family of clustering models, such as Gaussian mixtures and Ward/k-means partitions.
4. Choose complexity using held-out likelihood/separation and stability under resampling entire mice, sessions within mice, trials, and feature windows.
5. Align labels across resamples and report assignment probability, adjusted Rand index, centroid correlations, cluster size by mouse, and sensitivity to algorithm/hyperparameters.
6. Freeze centroids/classifier in discovery and map held-out units without refitting. Within each unit, use independent trial folds for cluster assignment and evaluation of any state/transition feature that helped define the cluster; otherwise label those dynamics descriptive rather than confirmatory.

The Allen Visual Coding survey provides a precedent for reliability-based functional classification ([de Vries et al., 2020](https://doi.org/10.1038/s41593-019-0550-9)), and the related VBN analysis used event-kernel GLMs to separate sensory and action response classes ([Bennett et al., 2026](https://doi.org/10.1016/j.cell.2026.06.025)). This study must improve on a visually selected cluster count by making stability and held-out mapping explicit.

**Predeclared fallback:** if no discrete solution is stable across mouse bootstraps and held-out mapping, do not force named clusters. Report reproducible continuous functional axes and test regional distributions of those scores.

### E. Test anatomical enrichment

For every locked cluster or functional axis:

1. Compute the fraction/distribution among all eligible recorded units in each region, first within session and then within mouse.
2. Estimate mouse-level enrichment with uncertainty and show every mouse, not only pooled unit counts.
3. Use a hierarchical/multinomial model or hierarchical bootstrap when region coverage is incomplete.
4. Build a session-preserving null by permuting cluster labels within session, optionally within broad firing-rate/reliability strata. This preserves probe sampling and mouse/session composition.
5. Correct across the predeclared cluster-by-region family and report both raw and adjusted p-values.
6. Confirm only discovery-nominated enrichments in held-out mice; all full-brain screens remain explicitly exploratory.

Report odds ratios or standardized score differences, confidence intervals, number of mice, sessions, units, and coverage denominators. A large pooled-unit enrichment with inconsistent mouse effects is not a positive result.

### F. Identify state-dependent interactions between simultaneous regions

Only simultaneously recorded populations can support the primary network analysis. Begin with a coverage matrix of region pairs by mouse/session and lock minimum population size and coverage before inspecting interaction effects.

#### Primary model

For each nominated source-target pair and session:

1. Bin spikes at the locked resolution and fit target activity from stimulus/task covariates, measured behavior/arousal, smooth session time, target population history, and state.
2. Add lagged source population activity through ridge-regularized linear reduced-rank regression (RRR) on variance-stabilized population activity, with dimensionality and regularization selected in inner contiguous folds.
3. Define interaction strength as the incremental held-out target R-squared explained by source activity. A generalized Poisson low-rank model may be a separately locked sensitivity analysis and would instead use held-out deviance.
4. Estimate the reversible block contrast in that increment, using equal-duration/trial samples and repeated equal-size neuron subsampling across states and regions.
5. Aggregate session estimates within mouse for primary inference.

RRR is motivated by the communication-subspace framework ([Semedo et al., 2019](https://doi.org/10.1016/j.neuron.2019.01.026)), but predictive subspaces do not prove direct connectivity.

#### Mandatory controls

- trial-shift source activity within block while preserving stimulus/state statistics;
- circular shifts larger than the modeled lag;
- reverse source/target direction;
- zero-lag versus strictly positive-lag models;
- target-history-only and behavior-only baselines;
- equal neuron counts, equal trial counts, and firing-rate-matched subsamples;
- source and target regions on the same versus different probes where possible;
- residual coupling after excluding all peri-lick time and on no-lick changes;
- pseudo-boundary and E1-to-E2 drift contrasts;
- leave-one-mouse-out influence analysis.

#### Complementary population tests

- Cross-state image/change decoding: train in `E1`, test in `NR` and `E2`, and rotate the train/test state.
- Cross-state lick decoding from pre-lick activity.
- Geometry of sensory axes relative to the predictive source subspace.
- Time-resolved source sensory decoding followed by target/action decoding to constrain temporal order.

The main network result should be one nominated pathway with a clear mouse-level effect, not a dense matrix of selectively reported region pairs. The full screened matrix, including null pairs, belongs in supplementary data.

### G. Use novelty and contrast as out-of-sample mechanistic probes

After D02 validates stimulus semantics, fit a factorial model containing reward availability, image identity, novelty, contrast, change status, cumulative exposure, and their targeted interactions. Analyze physical contrast within the same image identity wherever possible.

Required analyses:

1. Behavioral psychometrics: response probability and latency by state, contrast, novelty, change status, and exposure.
2. Early sensory gain: response amplitude/decoding for full versus reduced contrast by block.
3. Novelty dynamics: initial response and decay over presentations, separated from recording day and image identity as far as the design permits.
4. Cluster characterization: perturbation responses for already frozen clusters/axes.
5. Network generalization: whether the nominated predictive subspace and state-dependent interaction generalize to novel and reduced-contrast stimuli.
6. Model comparison: quantitative predictions for global sensory gain, surprise/adaptation, routing, and motor-only accounts, evaluated on held-out perturbation trials.

Never define clusters on all novelty/contrast trials and then use those same trials as an independent test. Hold perturbation trials out of cluster discovery or use nested folds. Report the limitations of any identity-day-novelty confounding explicitly.

### H. Integrate evidence into a candidate pathway

Construct an evidence table, not a narrative collage. For each nominated pathway, include:

- source and target region coverage;
- source sensory encoding in every state;
- target sensory/motor cluster enrichment;
- interaction-strength contrast and null controls;
- timing relative to stimulus and lick;
- novelty/contrast model predictions and observed results;
- discovery and confirmation mouse-level effects;
- robustness and failed checks.

Call a pathway “supported” only if a prespecified conjunction of evidence passes. A suggested conjunction is: held-out source sensory decoding above chance, positive source increment in target prediction, positive reversible interaction contrast, consistent direction in at least two-thirds of confirmatory mice, and no failure of the primary trial-shift or time-drift controls. D14 must lock the final rule.

## 9. Statistical framework

### 9.1 Unit of inference

The mouse is the primary independent biological replicate. Trials, units, probes, and repeated sessions are nested observations and must not be treated as independent mice. Pooled-unit tests may be shown only as descriptive or explicitly labeled sensitivity analyses.

Preferred order:

1. estimate unit/session effects without thresholding on significance;
2. summarize within session/region;
3. summarize repeated sessions within mouse using a prespecified weighting rule;
4. test/report the mouse-level distribution.

When mouse aggregation discards essential incomplete nesting, use a hierarchical bootstrap that resamples mice, then sessions within mouse, then units/population samples or trials at the relevant level. The rationale and Type-I error problem are described by [Saravanan et al. (2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7906290/). Refit the full relevant pipeline—including clustering or dimension selection—inside bootstrap replicates when feasible.

### 9.2 Estimands and uncertainty

Every confirmatory result must name one estimand before its test, for example:

- paired mouse difference in change-response probability, `0.5(E1+E2)-NR`;
- paired mouse difference in cluster-specific sensory kernel amplitude;
- regional log-odds enrichment of a frozen cluster;
- paired mouse difference in source-added held-out target R-squared;
- state-by-contrast or state-by-novelty interaction.

Report effect estimate, 95% confidence interval, exact number of mice/sessions/units/trials, test/model, sidedness, multiplicity family, raw p-value, adjusted q-value, and analysis version. Use two-sided tests unless a directional hypothesis and failure criterion are locked in advance. Emphasize effect magnitude and uncertainty rather than a p-value threshold.

### 9.3 Multiplicity

Define families before testing:

- behavior/QC confirmation;
- nominated cluster effects;
- nominated region enrichments;
- nominated region-pair interaction effects;
- perturbation interactions.

Use hierarchical testing where a global family test precedes local contrasts, and control false discovery rate within exploratory screens. Confirmation should use only the small frozen list and report all tests. Time-resolved inference should use a prespecified window/summary or cluster-based permutation with the whole time family preserved; do not test every bin and report the best one.

### 9.4 Missingness and weighting

- Report missing covariates and region coverage by block/session/mouse.
- Do not let mice with more sessions, probes, or units dominate the primary estimate.
- Default to equal mouse weighting; compare precision weighting only as a sensitivity analysis.
- For network models, balance trials and neuron counts across states before comparing predictive performance.
- Separate “not recorded,” “recorded but failed QC,” “insufficient events,” and “model failed” in every table.

## 10. Planned result and figure structure

Each main figure should support one sentence-level result. Subpanels are included only when necessary to establish the result or rule out its closest alternative.

### Figure 1 — Reward availability shifts licking across technically valid sessions

Claim: across all technically valid sessions, continuous within-mouse behavior shows loss and restoration of stimulus-driven licking. The threshold-selected neural-analysis cohort is shown as QC characterization, not used as independent evidence for this claim.

Minimum panels:

1. task/block schematic with actual median timing and the session filter;
2. mouse-level change/catch response trajectories around both boundaries in all technically valid sessions;
3. paired mouse continuous-gating contrasts, with the threshold rule and inclusion/attrition counts overlaid descriptively.

Statistics table: all behavior contrasts, trial counts, effect sizes, confidence intervals, and QC attrition.

### Figure 2 — Reproducible neural motifs track loss and restoration of reward

Claim: stable functional motifs show distinct onset/offset dynamics that exceed pseudo-boundary, movement, and session-drift controls.

Minimum panels:

1. frozen cluster/axis response summaries around both transitions;
2. mouse-level reversible transition effects and pseudo-boundary null;
3. held-out mapping/stability evidence.

Statistics table: cluster stability, assignment uncertainty, transition effects, and null controls.

### Figure 3 — Reward availability selectively modulates sensory-to-action response classes

Claim: early sensory coding is more state-invariant than late/pre-lick and action-related activity.

Minimum panels:

1. model-free and model-based time courses for frozen sensory/action classes;
2. mouse-level early-versus-late reversible state contrasts;
3. total versus movement-adjusted estimates.

Statistics table: kernel/dropout contrasts, model performance, timing, and covariate controls.

### Figure 4 — Gating-related response classes are anatomically enriched

Claim: a small set of well-sampled regions contains reproducible enrichment for the state/transition or sensory-action classes.

Minimum panels:

1. CCF map/coverage of tested regions;
2. mouse-level forest plot of frozen region enrichments;
3. session-preserving permutation comparison.

Statistics table: all tested region-by-class effects, denominators, p/q values, and held-out status.

### Figure 5 — A simultaneous inter-regional interaction tracks reward state

Claim: one discovery-nominated source-target population interaction is stronger when reward is available and survives target-history, behavior, trial-shift, and time-drift controls.

Minimum panels:

1. predictive-model schematic and temporal ordering;
2. held-out source-added target prediction by block, per mouse;
3. primary null/control comparison.

Statistics table: coverage, model ranks/hyperparameters, out-of-fold performance, state contrasts, directions, and nulls for every screened pair.

### Figure 6 — Identifiable perturbations constrain the gating mechanism

Claim: after an estimability/counterbalancing audit, held-out perturbation responses favor one account or constrain the viable global-gain, surprise/adaptation, routing, and motor-only alternatives. Novelty is not interpreted separately when its identity/day confound is unresolved.

Minimum panels:

1. prespecified model predictions;
2. behavioral and early/late neural state-by-perturbation effects;
3. nominated pathway's perturbation generalization/model comparison.

Statistics table: every factorial interaction, model-comparison score, and identity/day limitation.

### Supplementary figures

- full dataset/session/probe/region inventory and missingness;
- unit QC and engaged-block consistency diagnostics, including all threshold sensitivities;
- all cluster/axis response profiles and all-unit model performance;
- continuous-axis fallback and alternative clustering solutions;
- complete region enrichment screen and ontology sensitivity;
- complete simultaneous region-pair coverage and interaction matrix;
- all movement/arousal, time-drift, shuffled-boundary, and neuron/trial-balancing controls;
- genotype, sex, recording day, project-cohort, and leave-one-mouse-out analyses;
- novelty identity/exposure diagnostics;
- representative and failure-case unit reports.

## 11. Required outputs and statistics tables

Suggested artifact contract (paths are targets, not yet-created files):

```text
results/
  manifests/
    dandiset_assets.parquet
    schema_audit.parquet
    analysis_lock.yaml
    software_environment.txt
  tables/
    session_qc.parquet
    session_attrition.csv
    event_table.parquet
    all_units.parquet
    unit_qc_attrition.csv
    behavior_statistics.csv
    unit_model_statistics.parquet
    cluster_membership.parquet
    cluster_stability.csv
    region_coverage.csv
    region_enrichment_statistics.csv
    region_pair_coverage.csv
    network_statistics.csv
    perturbation_statistics.csv
    robustness_statistics.csv
    figure_source_manifest.csv
  models/
    frozen_feature_transform.*
    frozen_cluster_model.*
    nominated_network_models/
  figures/
    main/
    supplementary/
    all_units/
  unit_browser/
    index.html
    units/
```

Every statistics table must contain, where applicable:

- analysis/result/contrast ID and plain-language hypothesis;
- data/DANDI/code version and seed;
- discovery, confirmation, or exploratory label;
- inclusion definition and missingness stratum;
- estimate, scale/unit, confidence interval, test statistic, p, q, multiplicity family, and sidedness;
- mouse, session, probe, unit, and trial counts for every group;
- aggregation/bootstrap/model formula and cross-validation grouping;
- status (`pass`, `null`, `fragile`, `not_estimable`, `failed`) and machine-readable reason.

The all-unit browser must include every inventoried unit, including excluded and unclassified units, with explicit reason flags. For analyzable units it should show:

- waveforms and QC metrics available from NWB;
- location/probe/CCF coordinates;
- firing rate over the complete session;
- E1/NR/E2 PSTHs for familiar image, change, omission, lick, reward, novelty, and contrast events where available;
- transition-aligned activity;
- observed versus cross-validated model prediction and feature contributions;
- stability metrics, cluster probability/axis scores, and all analysis links.

Figures must have adjacent source-data tables sufficient to redraw every point. Save vector PDF/SVG for manuscripts and high-resolution PNG for browsing. Never save only group averages; preserve mouse/session/unit-level plotted data.

## 12. Robustness and negative-control matrix

Every central result should be tested against the following alternatives and summarized in `robustness_statistics.csv` rather than selectively mentioned.

| Threat | Required check |
|---|---|
| No-reward is always in the middle | Reversible `[0.5, -1, 0.5]` contrast; continuous session-time spline; E1-to-E2 contrast; pseudo-boundaries |
| Blocks have unequal duration/trial counts | Equal-duration and equal-trial subsampling; early/middle/late matched windows |
| Late E2 disengagement | Primary early-E2 analysis; repeat including and excluding final 10 minutes; continuous engagement score |
| Session selection induces a collider/generalization limit | Repeat key neural estimates in all technically valid sessions with continuous behavioral gating interaction |
| Spike-sorting drift | Engaged-block consistency filter; isolation-only analysis; presence-ratio/SNR/drift sensitivity; firing-rate trajectories |
| Aggressive unit filter selects the effect | Threshold grid and all-unit continuous quality weighting as secondary checks |
| Licking explains neural differences | Pre-lick/censored windows; no-lick changes; matched lick trials; false-alarm/spontaneous lick analyses |
| Running/arousal explains neural differences | Covariate adjustment, matched strata, and complete-case pupil analyses; report residual missing movement caveat |
| Reward delivery explains E2 transition | Separate auto-reward and reward kernels; compare later recovery; reward-matched controls |
| Stimulus/adaptation history differs | Match identity/contrast/repetition/time since change; history kernels; exposure trajectories |
| Region coverage drives enrichment | Mouse/session aggregation; within-session permutation; coverage threshold; equal-unit subsampling |
| Large regions dominate cluster discovery | Region-blind features; equal-mouse or equal-session resampling; test cluster stability by mouse |
| Population size drives network performance | Repeated equal-neuron subsampling and rank/regularization nested in training data |
| Shared stimulus drive mimics coupling | Full task-covariate residualization, trial shifts within condition, positive-lag requirement, target-history baseline |
| Common input mimics direction | Reverse-direction and zero-lag controls; explicitly avoid causal direction claims |
| One mouse drives the result | Leave-one-mouse-out estimates and influence plot |
| Repeated recording day/genotype/cohort confounds | Stratified summaries and interactions; exclude each cohort in turn; never treat sessions as independent mice |
| Novelty is image/day confounded | Within-session exposure effects, image identity terms, day-stratified replication, narrow interpretation |
| Cluster solution is arbitrary | Mouse-bootstrap consensus, alternative algorithms/features, held-out label mapping, continuous-axis fallback |
| Time-window tuning inflates effects | Freeze windows in discovery; show full time course; confirm with unchanged summaries |

## 13. Staged milestones and stop/go gates

### Milestone 0 — data freeze and audit

Deliver:

- immutable asset manifest;
- NWB schema/path report;
- reconciled session/unit/stimulus inventories;
- validated event dictionary and clock checks;
- coverage-only summaries.

Gate: human approval of D01-D02. Stop if task state, novelty, or contrast cannot be reconstructed reliably.

### Milestone 1 — behavior and QC lock

Deliver:

- blinded behavior-only QC dashboard;
- session and unit attrition tables;
- engaged-block consistency proposal/distributions without region or `NR` neural outcomes;
- discovery/confirmation allocation and precision assessment.

Gate: human approval of D03-D06 and D12-D14. If the behavior-defined cohort is too small, remove confirmatory language before neural analysis.

### Milestone 2 — discovery encoding and functional structure

Deliver:

- canonical event/all-unit tables;
- cross-validated spike models and model-free PSTHs;
- transition analysis;
- cluster stability comparison and continuous-axis fallback;
- frozen feature/cluster specification.

Gate: approve D07-D08. Stop discrete-cluster claims if mouse-bootstrap and held-out-style discovery folds are unstable.

### Milestone 3 — discovery anatomy and interactions

Deliver:

- locked region hierarchy and coverage thresholds;
- exploratory enrichment screen;
- simultaneous region-pair matrix;
- network model comparison and null controls;
- at most a small number of nominated regions/pairs.

Gate: approve D09-D11 and freeze the primary candidate-pathway estimand. Stop pathway claims if no region pair has adequate independent mouse coverage or no source-added prediction survives trial-shift controls.

### Milestone 4 — perturbation model discrimination

Deliver:

- validated novelty/contrast design matrix;
- held-out-within-discovery perturbation predictions;
- frozen predictions for the competing mechanisms;
- complete identity/day/exposure limitations.

Gate: lock the perturbation estimability audit and any resulting Figure 6 tests before confirmatory mice are accessed.

### Milestone 5 — one-time confirmation

Run the frozen pipeline once on held-out mice. Deliver every nominated result, including null and failed estimates. Do not retune and rerun confirmation. Any post-unblinding modifications create a new exploratory analysis label.

Gate: apply the locked conjunction rule for a supported candidate pathway. A null confirmation is a publishable null, not permission to search for a replacement pathway in the same holdout.

### Milestone 6 — manuscript and reproducible release

Deliver:

- minimal main figures and complete source data;
- all supplementary/QC/robustness outputs;
- all-unit browser;
- machine-readable statistics tables;
- executable scripts/notebooks with inline dependencies and pinned environment;
- methods text generated from the analysis lock and manifests;
- limitations section distinguishing reward availability, behavioral engagement, predictive interaction, and causality.

Fundamental I/O, filtering, event alignment, statistics-table writing, and plotting primitives should live in the library with tests. Dataset orchestration and scientific iteration should remain in scripts and marimo notebooks until the workflow stabilizes. Code should import modules rather than functions/classes except where a standard-library exception is justified.

## 14. Primary literature and source links

Dataset-specific sources:

- [DANDI:001051/0.260825.2232 — Large-scale Neuropixels recordings through SHIELD implant during visual change detection with dynamic gating](https://doi.org/10.48324/dandi.001051/0.260825.2232)
- [Allen Institute SHIELD Dynamic Gating analysis repository and metadata tables, pinned commit](https://github.com/AllenInstitute/SHIELD_Dynamic_Gating_Analysis/tree/3c09a0dc2972381f06393ddf5c000e16462c7037)
- [Bennett et al., 2024, *Neuron* — SHIELD](https://doi.org/10.1016/j.neuron.2024.06.015)
- [Bennett et al., 2026, *Cell* — Map of spiking activity underlying change detection](https://doi.org/10.1016/j.cell.2026.06.025)

Scientific motivation and task/state confounds:

- [Allen et al., 2019, *Science* — Thirst regulates motivated behavior through brain-wide dynamics](https://doi.org/10.1126/science.aav3932)
- [Steinmetz et al., 2019, *Nature* — Distributed coding of choice, action, and engagement](https://doi.org/10.1038/s41586-019-1787-x)
- [Siegle et al., 2021, *Nature* — Functional hierarchy in the mouse visual system](https://doi.org/10.1038/s41586-020-03171-x)
- [Niell and Stryker, 2010, *Neuron* — Behavioral-state modulation of visual responses](https://doi.org/10.1016/j.neuron.2010.01.033)
- [Musall et al., 2019, *Nature Neuroscience* — Movement-dominated single-trial dynamics](https://doi.org/10.1038/s41593-019-0502-4)
- [Stringer et al., 2019, *Science* — Multidimensional behavior and brain-wide activity](https://doi.org/10.1126/science.aav7893)
- [Garrett et al., 2020, *eLife* — Experience, novelty, and VIP-cell response dynamics](https://doi.org/10.7554/eLife.50340)
- [Henschke et al., 2020, *Current Biology* — Reward association and stimulus representations in V1](https://doi.org/10.1016/j.cub.2020.03.018)

Methods:

- [Semedo et al., 2019, *Neuron* — Communication subspaces and reduced-rank regression](https://doi.org/10.1016/j.neuron.2019.01.026)
- [de Vries et al., 2020, *Nature Neuroscience* — Reliability-based functional classes](https://doi.org/10.1038/s41593-019-0550-9)
- [Kobak et al., 2016, *eLife* — Demixed principal component analysis](https://doi.org/10.7554/eLife.10989)
- [Saravanan et al., 2020, *Neurons, Behavior, Data Analysis, and Theory* — Hierarchical bootstrap](https://pmc.ncbi.nlm.nih.gov/articles/PMC7906290/)
- [Varma and Simon, 2006, *BMC Bioinformatics* — Nested cross-validation and model-selection bias](https://doi.org/10.1186/1471-2105-7-91)
- [Wang et al., 2020, *Cell* — Allen Mouse Brain CCFv3](https://doi.org/10.1016/j.cell.2020.04.007)
