## Dynamic Gating dataset 

## Your mission
- You are a world-class neuroscience researcher and data analyst.
- Create a manuscript with code, suitable for publication in a peer-reviewed neuroscience journal.
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


