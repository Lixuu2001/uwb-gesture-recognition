# Range-resolved recurrence candidate: direct prior-art gate

**Date:** 2026-10-10  
**Branch parent:** 7525e185744029c259e085a8335aa1776793c996  
**Decision:** do not fit/train a recurrence-plot or RQA classifier on the restored UWB-Gestures cache as a claimed new method.

## New evidence reviewed

Full text of He, Aubry, Le Chevalier, and Yarovoy, “Self-similarity matrix based slow-time feature extraction for human target in high-resolution radar,” *International Journal of Microwave and Wireless Technologies* (2014), DOI [10.1017/S1759078714000087](https://doi.org/10.1017/S1759078714000087), was reviewed from the TU Delft repository PDF (12 pages). It constructs slow-time self-similarity matrices for radar high-resolution range profiles (HRRP), micro-Doppler images, and range-Doppler video sequences. It treats each as a multidimensional time series and uses mutual information as its similarity measure; the paper studies human walking/running periodicity and angle invariance.

This is direct prior art for a range-profile-resolved radar self-similarity mechanism. Its task is gait/slow-time characterization, not fine-grained hand-gesture classification on the UWB-Gestures corpus, and its similarity construction is not identical to every thresholded RQA variant. The task difference may justify a new application study, but simply mapping RQA/self-similarity onto the existing range-time cache does not establish an original method.

Rani et al., “Exploiting Unique State Transitions to Capture Micro-Doppler Signatures of Human Actions Using CW Radar,” *IEEE Sensors Journal* (2021), DOI [10.1109/JSEN.2021.3126436](https://doi.org/10.1109/JSEN.2021.3126436), has an accessible abstract/metadata describing recurrence quantification and Poincaré features derived from cadence-velocity diagrams for radar action classification. Its full text was not obtained, so no exact implementation or protocol equivalence is claimed.

The restored dataset already provides clutter-removed range-time inputs; the raw echo/background/calibration files needed to claim a new physical front end are absent from this restore. No independent synchronized hand-motion truth is available. These limits prevent turning the representation change into a measurement-grounded contribution here.

## Gate outcome

- Generic radar recurrence/RQA is prior art.
- Range-resolved radar self-similarity is also prior art; the proposed narrow variant is not novelty-cleared.
- The exact task/protocol differs from the cited radar studies, but a task change alone is not evidence of method novelty.
- Do not run an RQA parameter grid, classifier fit, or participant-fold experiment under this candidate.
- No project labels or array values were read; no fitting/training started; no score is reported.
- The restored data remain at `/workspace/uwb-restored-20260930`; no restore/download was repeated. Process inspection found only defunct historical Python entries and no active research job. The 789-file source snapshot was left unchanged.

## Next research condition

The current cache-only candidate set has no novelty-cleared method. Resume experiments only if a materially distinct, literature-reviewed mechanism can be specified against available inputs, or new calibrated radar measurements and synchronized motion truth become available. Keep the project contribution marked unvalidated.
