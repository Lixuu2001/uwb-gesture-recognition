# KIT cross-sensor transfer and method-prior-art gate

Date: 2026-10-11 12:16 UTC

## Question

Does the KIT FC-FMCW gesture corpus provide a usable independent external test for the current IR-UWB project, and does the current literature leave a method candidate sufficiently distinct to justify downloading the corpus or starting another experiment?

## Primary-source findings

- KITopen describes a separate dataset collected with one Infineon BGT60TR13C FC-FMCW radar: 2,479 recordings from 25 people, seven gesture classes plus zero-class measurements. Part 1 has 924 records from 22 people; Part 2 has 1,555 records from four people and five of the gestures plus zero-class cases. KIT says a legend accompanies the recordings and lists CC BY 4.0. The RADAR4KIT metadata lists a 14.7 GB archive. Source: [KITopen dataset record](https://publikationen.bibliothek.kit.edu/1000172790), DOI [10.35097/hCNNxXlJBdFTSYse](https://doi.org/10.35097/hCNNxXlJBdFTSYse).
- Park and Jeong's 2025 Sensors paper uses this KIT cohort and specifies entropy-based adaptive Top-K selection, a multi-stream EfficientNetV2 over range/Doppler trajectories, plus radar-specific augmentation/optimization. The paper excludes its two-/four-finger classes as similar to the one-finger class and reports a six-class configuration. It reports a five-repeat mean accuracy of 99.50% and a best run of 99.77% under its own split. These figures are not comparable to the project's eight-fold held-user protocol. Source: [paper full-text record](https://www.mdpi.com/1424-8220/25/20/6324), [author-hosted PDF record](https://kumoh.scholarworks.kr/bitstream/2020.sw.kumoh/29851/1/Radar-Based%20Gesture%20Recognition%20Using%20Adaptive%20Top-K.pdf).
- Current project data are clutter-removed IR-UWB X4 range-time maps. The KIT collection is a different radar modality and its public metadata does not provide a verified mapping to the project's gesture vocabulary or the exact representation/preprocessing needed for a matched external test. KIT explicitly points users to its legend for that mapping. The dataset is therefore a possible future cross-sensor transfer resource, not a currently protocol-compatible external cohort.

## Decision

Do not download the 14.7 GB archive or launch a KIT experiment in this continuation. The metadata and current access path do not yet establish a shared class map or a defensible matched input/evaluation protocol. Independently, generic entropy-based sensor/bin selection, multi-stream range/Doppler learning, and radar-specific augmentation are already method-level precedents; combining these modules would not clear novelty. A future KIT study would need a verified common-class map and a preregistered cross-sensor question with a genuinely distinct, feasible method. An accuracy comparison to the published result would not suffice.

This is a literature/access gate result, not a trained model or a validated contribution. No KIT archive was downloaded. No project labels, radar-array values, or checkpoints were read. No fitting or training was run. No code or original data/source snapshot was changed. The verified restore remains in place; current environment is CPU-only, with no acquisition device. No active training process was found in the current continuation preflight.

## Reproducibility and scope limits

The primary dataset metadata and paper records were reviewed through public KIT/author/publisher pages on 2026-10-11. The paper's detailed class-name legend is not included in the KITopen metadata excerpt; no class mapping is asserted. The bounded review is not exhaustive, and it does not show that all possible cross-sensor transfer methods have prior art. It only closes the generic method combinations listed above as standalone novelty claims for this task.

