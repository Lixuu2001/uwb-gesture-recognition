# Identity auxiliary supervision prior-art gate

**Date:** 2026-10-26 15:41:25 UTC UTC  
**Decision:** generic identity-as-auxiliary gesture supervision is not cleared as a method contribution. No model experiment was run or authorized.

## Candidate screened

Question: does adding participant identity as an auxiliary source-training target improve gesture prediction on a held-out participant? This is a plausible cross-user diagnostic because the existing project metadata contains participant IDs, but using ID supervision alone does not define a new radar-signal mechanism and may encourage identity-sensitive features instead of invariance.

## Prior art and scope

Wu et al., “A Joint Gesture-Identity Recognition Framework Based on 4D Millimeter-Wave Radar Sensing,” *Sensors* 25(23), 7249 (2025), DOI [10.3390/s25237249](https://doi.org/10.3390/s25237249). The publisher/primary-source indexed record and article excerpts describe a 4D FMCW dataset with seven volunteers and seven gestures, three representations (micro-Doppler, azimuth-time, elevation-time), and a joint gesture/identity network with task-specific outputs and cross-modal rectification. It therefore preempts a broad claim that adding identity as an auxiliary task or jointly predicting identity and gesture is itself new. The official full article page returned HTTP 429 in the web reader and PMC access presented a browser verification page; this note does not claim full-text review.

The project’s current target is user-held-out gesture recognition on three separate X4 IR-UWB views; identity is not available at inference. Existing bounded project records already document subject-invariance/adversarial approaches in radar gesture recognition, including same-corpus prior art. The different radar modality and dataset mean the 2025 paper does not establish performance or exact protocol equivalence here; it does establish that the generic auxiliary-task recipe is known.

## Disposition and limits

Close only the generic identity-auxiliary recipe as a standalone novelty claim. A materially distinct physical signal mechanism would still need to be specified and directly observable in the clutter-removed real-valued maps, then reviewed against primary literature and evaluated under a frozen held-user protocol. No user labels, array values, logits, or checkpoints were read; no fitting, training, data download, or source-snapshot change occurred. The research contribution remains unvalidated.
