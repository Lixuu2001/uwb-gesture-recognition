# Three-radar trajectory claim: dataset-paper prior-art and feasibility check

Date: 2026-10-11  
Branch: `codex/uwb-direction1-cloud-20260930`  
Observed branch head before this update: `050e1e56dd50b10573d0bfe791089482c0c80390`

## Primary-source finding

The original UWB-Gestures data descriptor (Ahmed et al., Scientific Data 8, 102, 2021, DOI 10.1038/s41597-021-00876-0) explicitly describes three independent monostatic XeThru X4 radars placed left/top/right, with each radar's data saved separately. It says this design permits use of one or multiple radars and identifies hand tracking and trajectory prediction as an intended use of the multi-radar data. The paper describes the setup geometry (left/right separation 1.1 m; top radar 0.55 m from the horizontal midpoint) and 90 slow-time frames per sample, about 4.5 seconds. Source: [publisher full text](https://www.nature.com/articles/s41597-021-00876-0); corroborating [PMC record](https://pmc.ncbi.nlm.nih.gov/articles/PMC8041886/).

This is **dataset-scope prior art**, not evidence that the descriptor presents or validates a particular tracking algorithm. It does, however, close “three-radar hand tracking/trajectory prediction” as a standalone problem statement or novelty claim. Any new method would need a clear algorithmic distinction and a controlled evaluation beyond the dataset's stated intended use.

## Feasibility against the current cloud copy

The present restored input remains the clutter-removed range-time cache; the current inventory does not expose raw complex echoes, a shared frame clock/trigger, per-view calibration files, or synchronized independent hand-pose truth. The paper's nominal sensor layout does not by itself establish sample-level synchronization, calibrated sensor extrinsics in the cloud package, or a measured 3-D trajectory. The existing event pairing is by participant/gesture/repetition and is soft-synchronized, per the project provenance review.

Thus no calibrated trajectory/triangulation experiment is supported by this cache. Do not infer a 3-D hand path from three independent range-time images or from class labels. A new acquisition would need a shared trigger, surveyed X4 positions/orientations, raw echo/clock records, and synchronized motion ground truth before a geometry-based method could be evaluated.

## Execution

Restore path exists. Process inspection found only defunct historical Python PIDs 598, 607, and 4719; no active training. No radar device node was found. No data values, labels, or checkpoint contents were read, and no fitting/training ran. No data was downloaded or modified.

## Decision

Keep the research state `candidate_novelty_and_access_gates_unresolved`. No method clears the gate from this result; do not start trajectory prediction, 3-D reconstruction, or another architecture grid. Next useful work is new calibrated/synchronized acquisition or an independently annotated compatible IR-UWB cohort. This report records a literature/feasibility exclusion, not a validated contribution or a claim that all possible multi-radar tracking methods are preempted.
