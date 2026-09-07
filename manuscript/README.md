# Preliminary Results manuscript

`results.typ` is a compilation-ready preliminary Results document. Figures
1--6 and Supplementary Figure S1 are backed by real artifacts. Figures 2--6 are
preliminary discovery-only analyses; the Figure 5 network screen is reported as
a null after familywise correction. Decisions D01--D15 are approved.
Confirmation remains sealed until the discovery-dependent specification is
frozen. Figure 1 and S1 remain exploratory because they were developed before
approval, and D04 unit stability remains pending.

## Build

Typst 0.15 or newer is recommended. From the repository root:

```sh
make -C manuscript
```

The PDF is written to `manuscript/build/results.pdf`. The repository-wide
`build/` ignore rule keeps compiled output out of version control.

For live preview while editing:

```sh
make -C manuscript watch
```

The source has no package or network dependency, so compilation does not fetch
templates or fonts.

## Artifact inclusion contract

The analysis pipeline remains authoritative. Manuscript prose must not be used
to calculate, copy-edit, or silently replace statistics.

| Result | Vector figure target | Authoritative statistics source |
|---|---|---|
| Behavior | `results/figures/main/figure_1_behavior.svg` | `results/tables/behavior_statistics.csv` |
| Behavior-transition QC | `results/figures/supplementary/figure_s1_behavior_transitions.svg` | `results/tables/behavior_transition_statistics.csv` |
| Neural transitions | `results/figures/main/figure_2_neural_transitions.svg` | `results/tables/neural_transition_statistics.csv` |
| Functional populations | `results/figures/main/figure_3_functional_populations.svg` | `results/tables/functional_population_statistics.csv` |
| Anatomy | `results/figures/main/figure_4_anatomy.svg` | `results/tables/anatomy_enrichment_statistics.csv` |
| Network | `results/figures/main/figure_5_network_interaction.svg` | `results/tables/network_statistics.csv` |
| Perturbations | `results/figures/main/figure_6_perturbations.svg` | `results/tables/perturbation_statistics.csv` |

These figure filenames specialize the `results/figures/main/` directory already
specified by the analysis plan.

For each result:

1. Verify that the statistics table records provenance, analysis status, effect
   estimate, interval, multiplicity information, and sample sizes.
2. Verify that the adjacent figure source-data table can redraw every plotted
   point and retains mouse/session/unit identifiers as appropriate.
3. Embed a Typst `figure(image(...))` call only after the vector figure exists.
4. Write terse, outcome-bearing bullets from the
   checked statistics rows. Preserve null, failed, and non-estimable
   outcomes.
5. Compile and inspect the PDF after every figure insertion.

The current Typst source embeds all six main-figure SVGs and Supplementary
Figure S1. Figures 2--6 and S1 have adjacent manifests; exploratory Figure 1
has panel-level source data but no authoritative figure manifest. Every
inferential analysis has a canonical statistics CSV in `results/tables/`.
