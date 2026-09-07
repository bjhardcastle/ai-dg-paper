#set document(
  title: "Dynamic gating of visually driven behavior",
  author: ("Author list pending",),
  keywords: ("reward availability", "visual behavior", "Neuropixels"),
)
#set page(
  paper: "us-letter",
  margin: (x: 0.9in, y: 0.85in),
  numbering: "1",
)
#set text(size: 10.5pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")

#align(center)[
  #text(17pt, weight: "bold")[Dynamic gating of visually driven behavior]
  #v(0.35em)

  #text(11pt)[Preliminary Results]
  #v(0.15em)

  #emph[Author list pending]
]

#v(0.8em)

#block(
  width: 100%,
  inset: 10pt,
  radius: 3pt,
  fill: rgb("#fff6dd"),
  stroke: 0.8pt + rgb("#b98500"),
)[
  *Evidence status.* Figures 1--6 and Supplementary Figure S1 are generated from
  real data in the immutable DANDI release. The two behavior analyses remain
  exploratory because they were developed before approval. Figures 2--6 are
  preliminary discovery-only neural or neural-linked results produced after
  decisions D01--D15 were approved. Confirmation identities and neural data
  remained sealed. D04 engaged-block unit stability and several broader
  movement/network controls remain outstanding.
]

= Results

We first asked whether stimulus-driven licking tracked reward availability,
then tested how early and late population activity, anatomical loading, and
simultaneous inter-regional prediction changed across the same transitions in
discovery mice. Finally, held-out contrast and designated-identity trials
constrained the interpretation. The mouse was the biological replicate: session
estimates were averaged within mouse and inference treated mice equally. All
neural results remain preliminary and discovery-only.

== Behavior across reward-availability blocks (Figure 1)

- *Evidence status:* Exploratory because the analysis was developed before lock approval; D01--D15 are now approved and the current statistics rows have status #raw("pass").
- *Cohort and estimability:* All 99 inventoried sessions from 27 mice were technically valid and contributed an estimable E1/late-NR/early-E2 contrast. The approved threshold filter retained 34 sessions from 18 mice, but the results below use all technically valid, complete sessions; threshold attrition is shown only as QC.
- *Blockwise behavior:* Mean change-trial response probability was 0.666 in E1, 0.151 in late NR, and 0.561 in early E2. Responses were computed from exact-deduplicated raw lick timestamps in the #raw("(150, 750]") ms post-change window.
- *Loss and return:* The response probability decreased from E1 to late NR by 0.515 (95% mouse-bootstrap CI #raw("[0.444, 0.584]")) and increased from late NR to early E2 by 0.410 #raw("[0.323, 0.502]"). Both two-sided mouse sign-flip tests had Holm-adjusted #raw("p = 0.000020") within a declared exploratory Figure 1 family; this family was not prespecified.
- *Reversibility and drift:* The symmetric summary, #raw("0.5 × E1 - late NR + 0.5 × early E2"), was 0.462 #raw("[0.386, 0.540]"). Recovery was incomplete: early E2 remained 0.104 below E1 (E2 - E1 = -0.104 #raw("[-0.149, -0.064]")).
- *Specificity control:* Go-minus-catch response specificity decreased from E1 to late NR by 0.441 #raw("[0.369, 0.512]") and increased in early E2 by 0.362 #raw("[0.279, 0.449]") (both Holm-adjusted #raw("p = 0.000020"), exploratory), arguing against an explanation based only on indiscriminate licking.
- *Limit:* Figure 1 alone resolves only three block windows. Supplementary Figure S1 adds boundary-centered trajectories and a matched within-state midpoint control, but a single midpoint does not exclude nonlinear within-state drift.

#figure(
  image("../results/figures/main/figure_1_behavior.svg", width: 100%),
  caption: [
    *Reward availability shifts licking across technically valid sessions.*
    *(A)* Median timing of the first engaged (E1), no-reward (NR), and second
    engaged (E2) blocks across 99 sessions from 27 mice. *(B)* Equal-session
    mouse means for change-trial (go) and catch-trial response probabilities in
    E1, late NR, and early E2; gray lines connect complete mouse-level block
    estimates. *(C)* Per-mouse symmetric gating contrast and its equal-mouse
    mean with 95% percentile bootstrap CI. Session QC distinguishes technical
    validity and contrast estimability (99 sessions, 27 mice) from the approved
    behavior-threshold cohort (34 sessions, 18 mice). The analysis remains
    exploratory because it was developed before approval; current statistics
    rows have status #raw("pass"). Source data and statistics:
    #raw("results/figures/main/figure_1_behavior_source_data/") and
    #raw("results/tables/behavior_statistics.csv").
  ],
) <fig-behavior>

== Behavior at reward-state transitions (Supplementary Figure S1)

Beyond matched within-state midpoint changes, change-trial licking changed in
the direction predicted by both reward withdrawal and reward restoration. This
exploratory, behavior-only QC analysis is distinct from the discovery-only neural transition
analysis in Figure 2. Fixed 3-minute pre- and post-anchor windows defined each
effect. The real-boundary step was compared with a pseudo-step at the temporal
midpoint of the respective source state in the same session; session estimates
were averaged within mouse and inference weighted mice equally.

- *Go responses:* The direction-corrected real-minus-midpoint-pseudo effect was 0.144 at reward withdrawal (95% mouse-bootstrap CI #raw("[0.077, 0.210]"), Holm-adjusted #raw("p = 0.000350"), 27 mice, 99 sessions) and 0.500 at reward restoration (#raw("[0.402, 0.600]"), Holm-adjusted #raw("p = 0.000020"), 27 mice, 97 sessions).
- *Response specificity:* For go-minus-catch responses, the corresponding withdrawal effect was 0.102 (#raw("[-0.011, 0.209]"), Holm-adjusted #raw("p = 0.0853"), 27 mice, 97 sessions), whereas the restoration effect was 0.402 (#raw("[0.277, 0.536]"), Holm-adjusted #raw("p = 0.000020"), 27 mice, 95 sessions). Thus, restoration was specific to stimulus-change responding, while withdrawal specificity remained uncertain.
- *Multiplicity and limit:* Holm correction was applied across withdrawal and restoration in two separate two-transition families, one for go responses and one for go-minus-catch specificity. The midpoint comparison addresses a matched within-state temporal change, but a single midpoint does not exclude nonlinear drift.

#figure(
  image(
    "../results/figures/supplementary/figure_s1_behavior_transitions.svg",
    width: 100%,
  ),
  numbering: none,
  caption: [
    *Supplementary Figure S1.*
    *Behavior changes at audited reward-state boundaries beyond matched
    within-state midpoint changes.* *(A, B)* Equal-mouse response-probability
    trajectories around reward withdrawal and restoration, displayed from -6
    to +6 minutes in nonoverlapping 2-minute bins; shading denotes 95%
    mouse-bootstrap CIs. *(C)* Direction-corrected real-boundary minus
    source-state-midpoint pseudo effects for go responses and go-minus-catch
    specificity. Points show individual mice; larger markers and error bars show
    equal-mouse means and 95% percentile bootstrap CIs. Positive values indicate
    changes in the reward-state-consistent direction. Effects used fixed
    3-minute pre- and post-anchor windows, with equal-session averaging within
    mouse and equal-mouse inference. Holm correction was applied separately to
    the two-transition go and go-minus-catch families. This is exploratory
    behavior-only QC, not the discovery-only neural Figure 2 analysis. A single
    midpoint control does not exclude nonlinear drift. Source data and
    statistics:
    #raw("results/figures/supplementary/figure_s1_behavior_transitions_source_data/")
    and #raw("results/tables/behavior_transition_statistics.csv").
  ],
) <fig-behavior-transitions>

== Early stimulus-linked population activity tracks reward state (Figure 2)

A session-specific ridge axis was learned only from familiar, full-contrast,
physical image changes without a lick in the closed #raw("[-150, +150]") ms
interval around the change. Each unit contributed the square-root-transformed
spike-count change from #raw("[-150, 0)") to #raw("[0, 150)") ms. E1 and NR
trials were scored out of fold using contiguous-within-state nested folds; E2
was never used for fitting, standardization, calibration, or tuning.

- *Cohort and unit screen:* The discovery slice contained 23 behavior-eligible sessions from 12 mice, 20,667 units passing #raw("isi_violations < 0.5"), #raw("amplitude_cutoff < 0.1"), and #raw("quality == good"), and 2,348 eligible trials. The approved D04 engaged-block consistency filter is not yet implemented, so this is an isolation-only preliminary result.
- *Out-of-fold state information:* The E1-versus-NR axis achieved a mouse-mean out-of-fold ROC AUC of 0.835 (95% mouse-bootstrap CI #raw("[0.790, 0.877]"), Holm-adjusted #raw("p = 0.000977"), 12 mice, 23 sessions, 20,667 units, 1,498 E1/NR trials).
- *Held-out E2 transfer:* With E2 excluded from every fitting and tuning step, its mean score exceeded the NR score by 0.051 axis units (#raw("[0.003, 0.101]"), Holm-adjusted #raw("p = 0.0747"), 12 mice, 23 sessions). The bootstrap interval excluded zero, but the discrete mouse-level randomization test did not pass 0.05 after correction; recovery on this summary is therefore suggestive rather than decisive.
- *Transition specificity:* The direction-corrected six-minute real-minus-pseudo step was 0.008 at reward withdrawal (#raw("[-0.083, 0.097]"), Holm-adjusted #raw("p = 0.869"), 12 mice, 21 sessions) and 0.204 at reward restoration (#raw("[0.123, 0.285]"), Holm-adjusted #raw("p = 0.00195"), 11 mice, 19 sessions). Thus, this axis changed abruptly beyond matched midpoint drift at restoration, but not at withdrawal.
- *Limits:* Peristimulus lick exclusion does not control running, pupil, or unmeasured orofacial movement. One midpoint pseudo-boundary does not exclude nonlinear drift. This slice establishes neither discrete response motifs nor anatomical enrichment, circuit interaction, or confirmation.

#counter(figure).update(1)
#figure(
  image("../results/figures/main/figure_2_neural_transitions.svg", width: 100%),
  caption: [
    *A discovery-only stimulus-linked population axis changes at reward
    restoration but not reward withdrawal beyond a within-state midpoint
    control.* *(A)* Equal-mouse axis trajectories around reward withdrawal and
    restoration from -6 to +6 minutes in nonoverlapping 2-minute bins. Colored
    curves mark the real boundary, gray curves mark the temporal midpoint of
    the source reward state, and shading denotes 95% mouse-bootstrap CIs.
    *(B)* Individual-mouse direction-corrected real-minus-pseudo effects using
    fixed six-minute windows on each side; larger markers and error bars show
    equal-mouse means and 95% percentile bootstrap CIs. Positive values are in
    the reward-state-consistent direction. Sessions were averaged equally
    within mouse, mice were weighted equally, and E2 was held out from all axis
    fitting and tuning. The analysis is preliminary and discovery-only; D04
    unit stability and broader movement controls remain outstanding. Source
    data and statistics:
    #raw("results/figures/main/figure_2_neural_transitions_source_data/"),
    #raw("results/tables/neural_transition_statistics.csv"),
    #raw("results/tables/neural_transition_trajectory.csv"), and
    #raw("results/tables/mouse_neural_transition_effects.csv").
  ],
) <fig-neural-transitions>

== Reward-state-associated modulation is stronger late (Figure 3)

We assigned each unit to a descriptive fixed response archetype according to
its dominant early sensory, late pre-lick, or lick-aligned
pre-lick/action-related response using alternating trials and required
agreement in archetype and response sign across the two halves. We then
quantified the reversible state contrast, #raw("0.5 × E1 - NR + 0.5 × E2"),
in early #raw("[25, 150)") ms and late #raw("[150, 600)") ms windows. Late
responses were censored at the first exact-deduplicated raw response-window
lick. Nested contiguous-fold ridge models included image identity, linear and
quadratic normalized session time, and engaged-block drift; the adjusted model
additionally included raw-lick response and latency, baseline running, and
window-minus-baseline running change.

- *Functional-response stability:* Cross-half class and sign agreed for 14,791 of 20,667 isolation-qualified units (71.6%) across 23 discovery sessions from 12 mice. The class-specific PSTHs are descriptive summaries; the primary timing tests below average units within session, sessions within mouse, and mice equally.
- *Model-free timing:* Reversible state modulation was larger late than early by 0.195 sign-aligned Anscombe-rate units (95% mouse-bootstrap CI #raw("[0.141, 0.260]"), Holm-adjusted #raw("p = 0.00244"), 12 mice, 23 sessions, 2,392 trials).
- *Adjusted timing:* The late-minus-early association persisted after lick/running adjustment, but was much smaller: 0.023 #raw("[0.0069, 0.0375]") (Holm-adjusted #raw("p = 0.0195")). The adjusted late-window state contrast itself was 0.038 #raw("[0.025, 0.051]") (Holm-adjusted #raw("p = 0.00244")). For context, the corresponding total-model late estimate was 0.188 #raw("[0.151, 0.227]"); the drop after adjustment is descriptive because the total-minus-adjusted difference was not a separate primary test.
- *Interpretation and limit:* Reward-state-associated modulation was concentrated in late sensory-to-action activity, and the association was strongly attenuated but remained positive after adjustment for measured licking and running. Pupil and unmeasured orofacial movement remain uncontrolled, and D04 engaged-block unit stability has not yet been applied.

#figure(
  image("../results/figures/main/figure_3_functional_populations.svg", width: 100%),
  caption: [
    *Reward-state-associated modulation is stronger in late sensory-to-action
    activity.* *(A)* Equal-mouse sign-aligned image-change PSTHs for
    cross-half-stable early-sensory and pooled late/action response classes in
    E1, NR, and E2; shading denotes 95% mouse-bootstrap CIs. These class traces
    are descriptive. *(B)* Per-mouse adjusted reversible state contrasts in the
    early and first-response-window-lick-censored late windows. *(C)* Per-mouse total and
    lick/running-adjusted late-window contrasts. Large horizontal markers show
    equal-mouse means. Primary inference was across five timing statistics with
    Holm correction. The discovery slice includes 20,667 isolation-qualified
    units and 2,392 trials from 23 sessions and 12 mice; 14,791 units had stable
    cross-half response labels. D04 stability, pupil, and broader orofacial
    controls remain pending. Source data and statistics:
    #raw("results/figures/main/figure_3_functional_populations_source_data/")
    and #raw("results/tables/functional_population_statistics.csv").
  ],
) <fig-functional-populations>

== Reward-state-axis magnitude is anatomically structured (Figure 4)

We next asked whether the absolute magnitude of each unit's Figure 2
E1-versus-NR decoder weight was concentrated in particular Allen CCFv3 major
divisions. We first took the absolute value of each unit's weight and then
z-scored that magnitude within session, compared each region with the same
sessions' full well-isolated population, averaged sessions within mouse, and
preserved session sampling in the anatomy-label permutation null.

- *Coverage:* Of 20,667 isolation-qualified units in 23 discovery sessions from 12 mice, 19,709 mapped to an official CCFv3 major division. Five divisions passed the locked coverage minimum of five mice, eight sessions, and a median of ten units per represented mouse.
- *Positive enrichment:* Absolute loading was elevated in Isocortex by 0.088 within-session SD units (95% mouse-bootstrap CI #raw("[0.067, 0.109]"), Holm-adjusted #raw("p = 0.0010"), 12 mice, 23 sessions, 10,069 units) and in Striatum by 0.190 (#raw("[0.066, 0.294]"), Holm-adjusted #raw("p = 0.0010"), 11 mice, 20 sessions, 2,046 units).
- *Depletion and direction:* Absolute loading was lower in Thalamus (-0.273, #raw("[-0.301, -0.243]"), Holm-adjusted #raw("p = 0.0010")) and Hippocampal formation (-0.081, #raw("[-0.126, -0.032]"), Holm-adjusted #raw("p = 0.0035")); Midbrain was uncertain (0.011, #raw("[-0.085, 0.133]"), adjusted #raw("p = 1.0")). No signed-loading regional contrast survived correction. Thus, anatomy localized reward-state-axis loading magnitude without supporting a common regional sign.
- *Limit:* These are discovery nominations based on the Figure 2 isolation-only axis, not independent confirmation or evidence that any region causally gates behavior. D04 engaged-block unit stability remains outstanding.

#figure(
  image("../results/figures/main/figure_4_anatomy.svg", width: 100%),
  caption: [
    *The magnitude of reward-state-axis loading is enriched in Isocortex and
    Striatum and depleted in Thalamus and Hippocampal formation.* *(A)* AP--DV
    coverage of isolation-qualified discovery units, colored by CCFv3 major
    division. *(B)* Within-mouse regional differences in absolute standardized
    Figure 2 decoder loading relative to the full eligible population in the
    same sessions. Large points and bars show equal-mouse means and 95%
    mouse-bootstrap CIs; small points show mice. *(C)* The observed Striatum
    enrichment against the session-preserving anatomy-label permutation null.
    Holm correction covered one ten-test family: five coverage-eligible
    divisions crossed with absolute and signed loading. Signed loading yielded
    no corrected regional effect.
    The analysis is discovery-only and D04 stability filtering remains pending.
    Source data and statistics:
    #raw("results/figures/main/figure_4_anatomy_source_data/") and
    #raw("results/tables/anatomy_enrichment_statistics.csv").
  ],
) <fig-anatomy>

== The inter-regional screen does not identify a robust gating interaction (Figure 5)

We screened 20 adequately covered ordered region pairs for positive-lag
prediction on familiar full-contrast changes without a lick from -150 through
+300 ms. For each session and state, a ridge reduced-rank model predicted target
activity at 150--300 ms from source activity at 0--150 ms after accounting for
image identity, session time, eventual response and latency, and target early
history. Nested contiguous folds selected model complexity, and states were
matched for trial and neuron counts. The primary control shifted source trials
within reward block while retaining identical model complexity.

- *Discovery nomination:* The largest controlled reversible contrast was Midbrain to Isocortex. This pair was supported by 10 mice, 13 sessions, 580 region-pair unit contributions, and 738 balanced trial rows; its selection from the same 20-pair screen makes it explicitly selection-biased.
- *Held-out performance:* Source-added held-out target #raw("R²") was -0.012 in E1 (95% mouse-bootstrap CI #raw("[-0.019, -0.006]")), -0.011 in NR (#raw("[-0.024, -0.002]")), and -0.008 in E2 (#raw("[-0.019, 0.004]")). Thus, the added source term did not improve mean prediction beyond the controlled baseline in any block.
- *State interaction and null:* The real-minus-within-block-trial-shift reversible contrast was 0.0102 #raw("R²") units (#raw("[-0.0025, 0.0266]"), raw #raw("p = 0.270"), Holm-adjusted #raw("p = 1.0")). No ordered pair survived correction across the 20-pair discovery family. The network step therefore yields a reportable null rather than support for a candidate pathway.
- *Limits:* This preliminary isolation-only slice excludes peri-lick activity and controls eventual lick response, but does not yet adjust running or pupil, apply D04 stability, test every planned lag/reverse-direction control, or establish connectivity or causality.

#figure(
  image("../results/figures/main/figure_5_network_interaction.svg", width: 100%),
  caption: [
    *A discovery screen does not identify a robust state-dependent
    inter-regional predictive interaction.* *(A)* Positive-lag model and
    within-block trial-shift control. The full model adds residual Midbrain
    activity at 0--150 ms to a baseline predicting Isocortex activity at
    150--300 ms. *(B)* Source-added held-out target #raw("R²") across E1, NR,
    and E2 for the top discovery nomination; gray lines are mice and colored
    points and bars are equal-mouse means and 95% mouse-bootstrap CIs. *(C)*
    Per-mouse reversible contrasts for real and shifted-source models. The
    controlled difference was 0.0102 #raw("R²") units
    #raw("[-0.0025, 0.0266]") and was not distinguishable from zero (raw
    #raw("p = 0.270"), Holm-adjusted #raw("p = 1.0")).
    Prediction used nested contiguous folds and balanced trials and neurons.
    The analysis is discovery-only; running, pupil, D04 stability, and broader
    direction/lag controls remain pending. Source data and statistics:
    #raw("results/figures/main/figure_5_network_interaction_source_data/") and
    #raw("results/tables/network_statistics.csv").
  ],
) <fig-network>

== Held-out contrast and identity probes constrain the gating account (Figure 6)

The Figure 2 reward-state axis was fit only to familiar, full-contrast changes.
We then scored reduced-contrast changes and changes to the designated novel
identity without refitting. Trials were completed physical changes with no lick
in the closed #raw("[-150, +150]") ms interval; inference again averaged
sessions within mouse and treated mice equally.

- *Contrast changes behavioral gating:* The reversible behavioral contrast was 0.618 at full contrast (95% mouse-bootstrap CI #raw("[0.534, 0.702]")) and 0.408 at 0.7 contrast (#raw("[0.306, 0.517]")). The reduced-minus-full state-by-contrast interaction was -0.210 (#raw("[-0.327, -0.095]"), Holm-adjusted #raw("p = 0.0117"), 12 mice, 23 sessions), showing weaker reward-state gating for the lower-contrast image.
- *No behavioral-gating difference was detected for designated identity:* The reversible behavioral contrast was 0.585 for familiar identities (#raw("[0.504, 0.668]")) and 0.581 for the designated novel identity (#raw("[0.507, 0.660]")). Their interaction was -0.0037 (#raw("[-0.0470, 0.0492]"), Holm-adjusted #raw("p = 0.900"), 12 mice, 23 sessions); this was not an equivalence test.
- *Out-of-fit neural transfer:* The frozen early axis retained a positive reversible reward-state contrast on both excluded trial classes: 0.130 axis units for reduced contrast (#raw("[0.081, 0.176]"), Holm-adjusted #raw("p = 0.00195")) and 0.164 for the designated novel identity (#raw("[0.116, 0.212]"), Holm-adjusted #raw("p = 0.00195")). Familiar/full-contrast traces are fit-condition references, so their larger values are not a quantitative held-out comparison.
- *Interpretive bound:* The designated identity is confounded with image identity and recording day and is not a pure first-exposure manipulation. These results therefore constrain, but do not identify, a novelty mechanism; they show that the early reward-state signal generalizes to two trial classes excluded from fitting. Behavioral gating was attenuated at lower contrast, whereas the designated-identity interaction was near zero and uncertain.

#figure(
  image("../results/figures/main/figure_6_perturbations.svg", width: 100%),
  caption: [
    *Behavioral gating is attenuated at lower contrast, whereas an early
    reward-state axis transfers to both held-out perturbation classes.* *(A)*
    Equal-mouse responses across E1, NR, and E2 for full/70% contrast and
    familiar/designated-novel identities. *(B)* Frozen Figure 2 axis scores;
    full/familiar curves are fit-condition references, whereas the other
    classes were excluded from fitting. *(C)* Behavioral state-by-perturbation
    interactions. *(D)* Reversible axis contrasts for the held-out classes.
    Points and bars show equal-mouse means and 95% mouse-bootstrap CIs. Novel
    identity is confounded with identity and day. This analysis is
    discovery-only and D04 remains pending. Source data and statistics:
    #raw("results/figures/main/figure_6_perturbations_source_data/") and
    #raw("results/tables/perturbation_statistics.csv").
  ],
) <fig-perturbations>

== Candidate-pathway synthesis

The discovery results support a reversible neural correlate of reward
availability at both an early lick-free population axis and a predominantly
late sensory-to-action response component. The early-axis loading magnitude is
anatomically nonuniform and its state contrast transfers to reduced-contrast
and designated-novel-identity trials. Lower contrast attenuates the behavioral
state interaction, whereas no designated-identity difference was detected.

The prespecified candidate-pathway conjunction is nevertheless not satisfied.
The 20-pair simultaneous-recording screen found no source-added interaction
that survived the within-block shift control and familywise correction, and
the selected Midbrain-to-Isocortex direction did not improve held-out target
prediction on average. The current data therefore nominate functional and
anatomical targets for follow-up but do not identify a distributed pathway.
Confirmation, D04 stability filtering, and the remaining movement and network
controls are required before a stronger circuit-level claim.
