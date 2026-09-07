TYPST ?= typst
SOURCE := results.typ
OUTPUT := build/results.pdf
FIGURE_1 := ../results/figures/main/figure_1_behavior.svg
FIGURE_2 := ../results/figures/main/figure_2_neural_transitions.svg
FIGURE_3 := ../results/figures/main/figure_3_functional_populations.svg
FIGURE_4 := ../results/figures/main/figure_4_anatomy.svg
FIGURE_5 := ../results/figures/main/figure_5_network_interaction.svg
FIGURE_6 := ../results/figures/main/figure_6_perturbations.svg
FIGURE_S1 := ../results/figures/supplementary/figure_s1_behavior_transitions.svg

.PHONY: all watch

all: $(OUTPUT)

$(OUTPUT): $(SOURCE) $(FIGURE_1) $(FIGURE_2) $(FIGURE_3) $(FIGURE_4) $(FIGURE_5) $(FIGURE_6) $(FIGURE_S1)
	@mkdir -p build
	$(TYPST) compile --root .. $(SOURCE) $(OUTPUT)

watch:
	@mkdir -p build
	$(TYPST) watch --root .. $(SOURCE) $(OUTPUT)
