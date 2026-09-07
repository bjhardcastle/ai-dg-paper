# Dynamic Gating: high-level analysis plan

**Status:** D01-D15 were approved by the user on 2026-09-07 and are recorded in `config/analysis_lock.yaml`. Figure 1 remains exploratory because it was developed before approval. Preliminary discovery-only Figures 2--6 are complete: the functional and anatomical results are positive, the simultaneous-network screen is null after correction, and held-out contrast/identity probes constrain interpretation. These vertical slices use isolation-only unit QC and do not replace D04, the remaining controls, or confirmation. Discovery-nominated windows, representations, regions, and models must still be frozen before the one-time confirmation run.

**Dataset:** immutable [DANDI:001051/0.260825.2232](https://doi.org/10.48324/dandi.001051/0.260825.2232), 99 session NWBs from 27 mice. Spiking is the primary neural modality.

**Goal:** identify a candidate distributed pathway by which visual changes drive licking when reward is available but fail to do so when reward is unavailable.

The companion [implementation specification](docs/ANALYSIS_IMPLEMENTATION_SPEC.md) contains feature definitions, model controls, output schemas, robustness checks, and milestone deliverables. This document is the shorter decision surface for scientific review.

## Proposed central claim

The strongest claim supported by this dataset would be:

> Reward availability is associated with a reversible reconfiguration of a reproducible, anatomically enriched inter-regional pathway that preserves early visual information while changing its predictive relationship to lick-related population activity.

This is a routing-gate hypothesis, not an assumption. A supported pathway must satisfy a prespecified conjunction:

1. A source population encodes image/change information before the lick window in both reward states.
2. A target population has reward-state-dependent late sensory, preparatory, or lick-related activity.
3. Source activity improves held-out prediction of target activity beyond stimulus, movement, arousal, session time, and target history.
4. The source-added prediction weakens in the no-reward block and returns in the second engaged block.
5. The functional populations, anatomy, and interaction reproduce across mice.
6. Where perturbation factors pass an estimability/counterbalancing audit, held-out novelty and contrast responses support the same computational account; otherwise they only constrain interpretation.

Because neural nodes were not perturbed, the manuscript should say **candidate pathway**, **association**, and **predictive interaction**, not causal circuit, mediation, or information flow. This framing is motivated by brain-wide state-dependent cue-to-action dynamics ([Allen et al., 2019](https://doi.org/10.1126/science.aav3932)), widespread action/engagement signals alongside more structured sensory signals ([Steinmetz et al., 2019](https://doi.org/10.1038/s41586-019-1787-x)), and low-dimensional inter-area communication subspaces ([Semedo et al., 2019](https://doi.org/10.1016/j.neuron.2019.01.026)).

## Dataset facts and required audit

Use only published DANDI version `0.260825.2232` and save an immutable asset manifest. Session NWBs must be distinguished from probe-level LFP files. All derived rows retain NWB path, table path, and table index.

A preliminary whole-dataset schema audit found several issues that are analysis-critical:

- Trial fields `no_reward_epoch`, `is_sham_change`, and `omitted_reward` exist in 77 session NWBs and are absent from 22. For missing flags, use the authors' [commit-pinned companion table](https://github.com/AllenInstitute/SHIELD_Dynamic_Gating_Analysis/tree/3c09a0dc2972381f06393ddf5c000e16462c7037), which covers 96 of 99 sessions and, in the preliminary join, all 22 missing-flag sessions. Preserve the NWB value when present, regenerate the coverage check, record disagreements, and exclude unresolved sessions.
- Nominal task-parameter times do not coincide exactly with trial transitions. Never infer confirmatory block labels from nominal time, `active`, `rewarded`, or `stimulus_block`.
- Task-image presentations use five table paths. The later passive-flash table is not the task contrast manipulation. Task contrast is encoded in image names such as `im115_r-0.7`; `is_image_novel` is an identity label, not a novelty epoch.
- `is_change` includes go and catch/sham events. Define a physical identity change from `initial_image_name != change_image_name`, while retaining go/catch labels as separate task factors.
- Trial `change_frame` can be relative to the task block while presentation `start_frame` is session-global. Infer one integer frame-origin offset independently per session from exhaustive exact matches on the non-omitted raw image token and physical-change label. Accept an offset only when it is the unique solution that maps every finite trial anchor one-to-one, then require each matched presentation start to fall strictly within its trial interval. Presentation timestamps are descriptive checks only; they must never infer, rank, or select the frame offset.
- A materialized task-presentation table is complete only when every canonical session contains the contiguous source indices `_table_index = 0, ..., n - 1`. A missing, duplicate, or out-of-range row invalidates the whole audit rather than yielding a partial multi-session table.
- Online `hit`, `miss`, `false_alarm`, and `correct_reject` events are not state-independent response labels: during the no-reward epoch, raw response-window licks can occur on trials with no online outcome event. Derive the cross-state behavioral response and latency from exact-deduplicated raw `lick_times` in the audited `(150, 750]` ms window, report duplicate burden, and retain paired online outcome labels only for an agreement audit in reward-available blocks.
- The NWB root `identifier`, not the often-null standard `session_id`, supplies the ecephys session ID in inspected files.

Before neural analysis, regenerate these counts, audit every session schema, validate acquisition clocks and event alignment, reconcile the 99 sessions with companion metadata, and freeze the CCF ontology level used for anatomy. No synthetic observations, sessions, or spike trains will be created.

## Approved decision framework

The user approved the choices below using behavior and coverage only, before inspection of reward-state neural effects. `config/analysis_lock.yaml` is authoritative. Choices that intentionally depend on discovery (for example exact response windows, stable functional representation, and nominated region pairs) still require a versioned discovery freeze before confirmation.

| Decision | Recommended choice |
|---|---|
| Data freeze | DANDI `0.260825.2232`; save asset IDs, paths, subject/session IDs, and retrieval date |
| Event semantics | NWB trial flags first; pinned companion fallback with conflict/coverage audit; no nominal-time fallback |
| Session thresholds | Rules below: engaged hit rate >= 0.5, engaged d-prime >= 1.0, late no-reward hit rate <= 0.2, with locked minimum counts |
| Unit consistency | Condition-matched equivalence test between engaged blocks, excluding final 10 minutes; coarse rate ratio is diagnostic only |
| Discovery/confirmation | Split by mouse after behavior QC, approximately 2/3 discovery and 1/3 confirmation; one final holdout evaluation |
| Functional representation | Cross-validated event-kernel spike model plus model-free PSTHs; discrete clusters only if stable across mouse bootstraps |
| Anatomy/network scope | Lock CCF parent level, coverage minima, a small discovery-nominated region/pair set, and primary lag/window before confirmation |
| Confirmatory viability | Require adequate independent mice and simultaneous populations; otherwise label the study exploratory |

Also approve handling of sex, genotype, recording day/cohort, missing pupil/video, LFP scope, primary time bins/windows, and the exact candidate-pathway conjunction. The detailed specification tracks these as D01-D15.

## Cohort and quality control

### Sessions

Compute eligibility without neural data. Use the NWB-defined response window with the task software's strict lower bound, `(150, 750]` ms, after validating it against raw lick timestamps, paired reward-available outcome labels, and the related task description ([Bennett et al., 2026](https://doi.org/10.1016/j.cell.2026.06.025)). Define response and latency from exact-deduplicated raw `lick_times` for every reward state, retain duplicate counts, and never treat an absent online outcome event in `NR` as a non-response. Label the blocks:

- `E1`: first reward-available block;
- `NR`: no-reward block;
- `E2`: second reward-available block after the audited return of reward availability.

Approved behavior-only rule (D03):

1. In `E1` and the portion of `E2` before its final 10 minutes, require enough completed, non-auto-rewarded go and catch trials, go-response probability >= 0.5, and loglinear-corrected d-prime >= 1.0.
2. In the final 10 minutes of `NR`, require enough go trials and go-response probability <= 0.2. These bounds imply at least a 0.3 response-rate drop from either qualifying engaged block.
3. Require valid E1 -> NR -> E2 structure, unique trial indices, finite ordered timestamps, positive trial durations, resolved reward labels, and no source conflicts.
4. Exclude any semantically validated auto-reward/cue trials from ordinary contingent performance. The E2 metric must demonstrate post-transition reacquisition, but do not assume the NWB `auto_rewarded` field encodes the E2 transition: an inspected session has auto-rewarded trials only in E1. Allow disengagement in the last 10 minutes as requested.

Late NR avoids selecting on early extinction speed, which is itself a target phenomenon. Report every rate, d-prime, count, confidence interval, threshold flag, and exclusion reason. Retain a continuous behavioral-gating score for sensitivity analyses.

### Units

The mandatory isolation filter is exactly:

- `isi_violations < 0.5`;
- `amplitude_cutoff < 0.1`;
- `quality == "good"`.

Reject null, nonfinite, and negative numeric values and missing quality labels. Do not silently add firing rate, SNR, presence ratio, receptive-field, or anatomy requirements to the primary definition.

The primary unit cohort combines the isolation thresholds with an engaged-block consistency filter. That filter should compare lick-free baseline and responses to shared full-contrast familiar stimuli in E1 versus E2, excluding the final 10 minutes. Estimate split-half noise, cross-validate the consistency score, and use an equivalence margin calibrated from within-block variability. A whole-block firing-rate ratio is retained only as an auditable diagnostic because task events and behavior differ across blocks. Repeat central results with isolation-only units as a sensitivity analysis.

## Analysis sequence

### 1. Establish behavior and transitions

Estimate mouse-level response probability, false-alarm probability, d-prime, latency, lick bouts, running, and available pupil measures through E1 -> NR -> E2. Establish the population behavioral effect in all technically valid sessions using continuous gating metrics; treat the threshold-selected cohort as QC characterization, not independent proof of gating. Model trial time continuously around both transitions and compare real boundaries with matched pseudo-boundaries, without requiring identical extinction kinetics.

### 2. Find functional response motifs

Fit cross-validated spike-count models with sensory, change/catch, lick, reward/omission, reward-state, transition, slow-time, and measured movement/arousal predictors. Use contiguous trial folds and inner folds for regularization. Report held-out performance and single-predictor-family dropouts alongside model-free PSTHs.

Cluster only held-out response features, not anatomy or raw firing rate. Resample mice, sessions, trials, and feature windows. Freeze the discovery representation and classifier before mapping confirmatory units. Use separate within-unit trial folds for cluster assignment and testing state dynamics; otherwise those defining dynamics are descriptive. If no discrete solution is stable, report continuous functional axes rather than named clusters.

Primary comparisons distinguish:

- reversible reward state: `0.5 * E1 - NR + 0.5 * E2`;
- monotonic drift/hysteresis: `-0.5 * E1 + 0.5 * E2`;
- true transitions versus time-matched pseudo-boundaries.

### 3. Test anatomical enrichment

For each locked cluster or functional axis, calculate its distribution relative to all eligible recorded units in each region. Aggregate within mouse first. Use mouse-level paired estimates when coverage permits; otherwise use a hierarchy-preserving bootstrap or multilevel model. Preserve session/probe sampling in permutation nulls, enforce region coverage minima, correct the predeclared family, and show every mouse. A pooled-unit effect with inconsistent mouse effects is not positive.

### 4. Identify state-dependent network interactions

Restrict the primary network analysis to simultaneously recorded, discovery-nominated region pairs with adequate mouse coverage. Predict variance-stabilized target population activity from stimulus/task variables, measured behavior/arousal, smooth session time, target history, and reward state. Then add lagged source population activity using ridge-regularized linear reduced-rank regression. Select rank and regularization in inner contiguous folds and quantify interaction as incremental held-out target R-squared.

Test whether source-added prediction follows the reversible state contrast after matching duration, trial counts, firing rates, and neuron counts. Mandatory controls include within-block trial shifts, circular time shifts, reverse direction, positive-lag versus zero-lag models, exclusion of peri-lick periods, no-lick change trials, pseudo-boundaries, target-history-only baselines, and leave-one-mouse-out influence. Predictive gain does not establish direct connectivity.

### 5. Use novelty and contrast as out-of-sample probes

Exclude perturbation trials from cluster discovery. First audit whether contrast and novelty effects are separately estimable given image, day, and exposure. Where they are, test reward-state interactions with image identity, physical contrast, novel identity, and cumulative exposure. Otherwise use only identified within-session contrasts and describe the observations as constraints. Compare four accounts:

- global sensory gain: proportional early scaling across identities and contrasts;
- surprise/adaptation: stronger novelty/rarity effects that decay with exposure;
- learned routing: preserved early encoding with state-dependent late activity/coupling;
- motor suppression: effects track lick preparation/execution more than stimulus factors.

Novel identity is confounded with day and image identity, so prioritize within-session exposure and mouse replication. Reward learning can alter even early visual representations ([Henschke et al., 2020](https://doi.org/10.1016/j.cub.2020.03.018)); preserved sensory coding must therefore be tested, not presumed.

## Inference and reporting

The mouse is the primary biological replicate. Aggregate session estimates within mouse and use paired mouse-level effects with bootstrap confidence intervals and randomization tests. When a mouse-level statistic is impossible, use a hierarchical bootstrap that resamples mice, then sessions, then trials or units; never treat units as independent biological replicates. Cross-validation splits by mouse for generalization and by contiguous time/trials within session for prediction. All preprocessing and hyperparameter selection occur inside training folds.

Every analysis writes a tidy statistics table containing the hypothesis/contrast ID, estimate, uncertainty, raw and adjusted p-values where applicable, sample sizes at mouse/session/unit/trial levels, inclusion rule, model/fold/null identifiers, and code/data provenance. Report effect sizes and intervals even for null results. Keep a complete multiplicity ledger and distinguish exploratory from confirmatory tests.

## Main-figure claims

Each figure should defend one result with only necessary panels:

1. **Behavior:** continuous within-mouse licking changes reversibly with reward availability across all technically valid sessions; cohort selection is shown as QC.
2. **Transitions:** specific neural motifs track loss/restoration of availability beyond smooth drift and movement.
3. **Functional populations:** sensory/action dynamics are selectively state-modulated and reproduce across mice.
4. **Anatomy:** the locked functional populations are enriched in specific regions at the mouse level.
5. **Network:** a nominated simultaneous source-target interaction weakens in NR and returns in E2 after controls.
6. **Mechanistic constraint:** where identifiable, held-out contrast/novelty trials favor one gating account; otherwise they bound the viable alternatives.

Each figure receives a machine-readable statistics table. Null or unstable steps remain reportable and should shorten the causal narrative rather than trigger a replacement search in held-out mice.

## Reproducibility and review gates

1. **Audit gate:** approve data version and event semantics; stop if reward state, novelty, or contrast cannot be reconstructed reliably.
2. **Behavior/QC gate:** approve thresholds, unit consistency, coverage, and mouse split using no neural state outcomes.
3. **Discovery gate:** freeze features, clusters/axes, regions, region pairs, windows, nulls, and the conjunction rule.
4. **Confirmation gate:** run the unchanged pipeline once in held-out mice. Any post-unblinding revision is exploratory.
5. **Release gate:** publish executable scripts/notebooks with inline dependencies, pinned environment, manifests, complete statistics tables, minimal figures, and a browser containing every inventoried unit. Excluded units receive provenance, QC metrics, and reason flags; analyzable units additionally receive response and model figures.

Fundamental access, filtering, event alignment, statistics writing, and plotting primitives belong in the tested library. Dataset orchestration remains in scripts and marimo notebooks while the workflow is being reviewed. The detailed implementation and literature basis are in the companion specification.
