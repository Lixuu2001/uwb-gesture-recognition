# UWB open-set and multiview prior-art check — 2026-10-08

## Scope and decision

This is a targeted prior-art review, not an exhaustive survey and not an experiment. It checks whether open-set recognition, subject disentanglement, and separate-branch multiradar fusion remain available as a novelty direction for the restored UWB-Gestures data.

**Decision:** these broad directions do not clear the novelty gate. A 2026 paper now directly reports open-set UWB gesture recognition on UWB-Gestures using graph prototypes and adversarial subject disentanglement; a 2023 paper already reports three-X4 multi-input CNN/ExtraTrees fusion on the same dataset. Do not reimplement those combinations and describe them as original. Their published results are not comparable to this project's eight-fold participant-disjoint scores because the 2023 paper reports a random 80/20 sample split, and the 2026 abstract's five class splits are a different task protocol.

## Primary sources checked

1. Yang et al., “Spatiotemporal graph prototype learning with adversarial training for UWB open-set gesture recognition,” *Measurement Science and Technology* 37, 286102 (published 2026-07-10), DOI [10.1088/1361-6501/ae7fab](https://doi.org/10.1088/1361-6501/ae7fab). The publisher record/abstract identifies UWB-Gestures, open-set recognition, a hybrid dilated-spatiotemporal/transformer extractor, graph prototypes, and adversarial subject disentanglement. It reports mean AUROC 0.9291 and OSCR 0.9201 across five gesture-class splits. The publisher landing page could not be opened as full text in this review; do not describe this as a full-paper review or exact reproduction.
2. Korti & Slimane, “Unobtrusive hand gesture recognition using ultra-wide band radar and deep learning,” *International Journal of Electrical and Computer Engineering* 13(6), 6872–6881 (2023), DOI [10.11591/ijece.v13i6.pp6872-6881](https://doi.org/10.11591/ijece.v13i6.pp6872-6881). The full publisher PDF was reviewed. It uses the public eight-volunteer, 12-gesture, three-XeThru-X4 UWB-Gestures data and a three-branch CNN feature extractor followed by ExtraTrees (MIMO-CxT); it reports 98.86% test accuracy. Its methods state a random 80/20 sample split, so that score is not a participant-independent comparison to this project. The paper directly preempts generic claims about separate per-radar branches, feature concatenation, ExtraTrees fusion, and single-versus-three view ablations on this dataset.
3. Ahmed et al., “UWB-gestures, a public dataset of dynamic hand gestures acquired using impulse radar sensors,” *Scientific Data* 8, 102 (2021), DOI [10.1038/s41597-021-00876-0](https://doi.org/10.1038/s41597-021-00876-0). Dataset descriptor: three independently stored radar views; it motivates radar gesture benchmarking but does not provide the missing external pose/onset truth or new placement/session cohorts in the current restore.

## Consequences for the next experiment

- Do not reopen generic open-set graph-prototype, adversarial subject-invariance, three-branch fusion, ExtraTrees readout, or view-count comparison grids on the same fixed cache.
- Existing eight-participant scores remain exploratory within this repeatedly used cohort; they do not validate cross-session, cross-placement, or new-user performance.
- The current segmented archive has no continuous pre-gesture background or independently annotated onset, and no external hand-motion trajectory truth. It cannot support online-onset or physical-trajectory claims by renaming a class split or a crop proxy.
- A potentially distinct claim would need a new endpoint and new measurement conditions with external truth. It must pass a further scoped review before capture, then use a frozen participant/session/placement split. The cloud currently has no X4 acquisition interface or new synchronized measurement package, so no training was started.

## Search record

Queries covered IR-UWB open-set recognition, UWB-Gestures multiview/open-set prior art, and cross-domain radar gesture recognition. The mmWave domain-generalization/transfer papers surfaced by the search establish that generic “domain-independent gesture recognition” is also an old direction, but their FMCW data are not treated as direct IR-UWB results. Search results were used to locate the primary publisher records; the conclusions above cite publisher/author primary sources.

This review did not read gesture labels or radar arrays and performed no fitting or training. The requested research contribution remains unvalidated.
