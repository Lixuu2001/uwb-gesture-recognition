# MVRADAR author-code scope audit

**Date:** 2026-10-28 UTC  
**Parent branch commit:** 6c2b2c87c278b23b5ad0ff5761666f5006815489  
**Scope:** public author repository source tree and selected implementation files; no paper PDF located or read. No UWB data, labels, model outputs, or checkpoints accessed. No fitting or training.

## What the author repository implements

The public `Tkwer/MVRADAR` repository at branch `mutliview_dev2` has a source tree with separate encoders for RT/DT and RDT/ERT/ART feature representations, then fuses those encoded representations. The model's `selected_features` are those five signal products; its Dempster–Shafer path calls a multi-view DS fusion decoder. The optional domain-adversarial head receives a configured number of domains. The dataset implementation assigns each configured root directory a dataset/domain label and loads the five feature products for each sample. The README also describes the views as radar feature representations and recommends a companion MIMO radar dataset.

This supports a narrower reading of the paper's implemented “multi-view”: in the public code inspected, the fused views are heterogeneous radar representations (RT, DT, RDT, ERT, ART), while the domain labels come from data roots/collections. The code does **not** demonstrate that the method explicitly treats three separately positioned physical X4 radars (Left/Top/Right) as its view axis. This refines the prior-art boundary; it does not establish that the paper's full manuscript has no such experiment, and it is not a substitute for the still-unavailable TIM full text.

## Relevance to the UWB-Gestures candidate

The author implementation is not direct evidence that Chen et al.'s method solves held-physical-sensor-view generalization. However, the UWB-Gestures corpus already has a same-corpus three-physical-view fusion paper, and prior work covers domain adversarial learning and view fusion. Therefore, leave-one-physical-view-out evaluation alone still contributes a protocol/question, not a method. A method candidate would need a distinct, physically justified mechanism that can be computed from the current clutter-removed real-valued maps, with a frozen participant-held-out protocol and controls. No such mechanism was identified here.

## Decision

- Keep the held-physical-view route closed as a **method** experiment until both the TIM full-text gate and a specific mechanism gate clear.
- Do not interpret the MVRADAR implementation's five feature representations as proof of three physical-radar view-domain training.
- No arrays or labels were read; no experiment or training was run.
- Research-contribution goal remains unresolved.

## Source record

- Author repository: https://github.com/Tkwer/MVRADAR/tree/mutliview_dev2
- Files reviewed: `README.md`, `code/models/model.py`, `code/models/encoder.py`, `code/models/decoder.py`, `code/models/methods/DScombine.py`, `code/models/methods/reverse_gradient.py`, `code/bin/dataset.py`, `code/examples/conf1.yaml`.
