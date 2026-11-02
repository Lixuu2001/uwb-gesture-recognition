# Same-corpus open-set and subject-generalization prior art review

Date: 2026-10-11 (UTC)

## Question and decision

Does the same UWB-Gestures cohort already have a method that combines open-set gesture recognition with subject-related feature disentanglement, making a generic graph-prototype OSR or adversarial subject-invariant classifier an unsubstantiated contribution here?

**Decision: yes. Do not claim either generic mechanism as new on this cohort.** A 2026 Measurement Science and Technology paper explicitly reports experiments on UWB-Gestures and proposes graph prototype learning plus gradient-reversal adversarial feature disentanglement. This is material same-corpus prior art. It does not establish that its subject split matches this project's fixed eight-fold held-user protocol.

## Source-backed scope

Yang, Yin, Sun, Wang, and Zhang, “Spatiotemporal graph prototype learning with adversarial training for UWB open-set gesture recognition,” *Measurement Science and Technology* 37 (2026), 286102, DOI [10.1088/1361-6501/ae7fab](https://doi.org/10.1088/1361-6501/ae7fab), published July 10, 2026.

The publisher abstract/preview describes:
- a dual-stream range-time encoder using atrous spatial pyramid pooling and a Transformer;
- graph prototype learning to calibrate known-class representations and separate known from unknown inputs;
- a gradient-reversal adversarial module intended to suppress subject-related information;
- closed-set and open-set evaluation on UWB-Gestures, with mean AUROC 0.9291 and OSCR 0.9201 over five gesture-class splits;
- additional cross-subject evaluation and ablations.

The accessible publisher preview supports the above method scope and same-corpus claim. The IOP full-text page was blocked to this reader; the full methods/results text and split construction were not available. Therefore the paper's held-subject split, fold construction, leakage controls, and score comparability are **unverified**. The reported OSR scores are not comparable to the project's eight-fold closed-set macro-accuracy protocol.

## Consequence for the project

1. Generic graph-prototype/open-set rejection is not cleared as a contribution on this data.
2. Generic adversarial subject disentanglement/domain invariance is not cleared either.
3. Combining the above with generic multi-scale or temporal models would not repair the novelty gap.
4. A candidate must identify a narrower, measurable IR-UWB signal mechanism absent from the paper, state its required input explicitly, and pass a pre-registered distinction and feasibility screen before training.
5. The processed cache still lacks raw pulse/complex echoes, per-view background captures, calibrated motion truth, and measured per-view acquisition costs. No training, test-label fitting, or data-value inspection was performed for this literature review.

## Access and search scope

This was a bounded same-corpus prior-art check, not an exhaustive literature review. IOP's article page was inaccessible beyond indexed publisher metadata/abstract; ResearchGate exposes only the publisher preview and a request-full-text control. No request was sent to the authors, no access controls were bypassed, and no paper file was downloaded. Claims above are limited to the accessible abstract/preview.

## Sources

- Publisher DOI record and abstract: https://doi.org/10.1088/1361-6501/ae7fab
- IOP article page: https://iopscience.iop.org/article/10.1088/1361-6501/ae7fab
- Publisher preview hosted on ResearchGate: https://www.researchgate.net/publication/407327581_Spatiotemporal_graph_prototype_learning_with_adversarial_training_for_UWB_open-set_gesture_recognition
