# UWB open-set gesture prior-art gate — 2026-10-09

## Scope and execution
A read-only primary-literature search checked whether unknown-gesture rejection could be a fresh research route on the restored UWB-Gestures corpus. No UWB labels or radar arrays were read, no files were downloaded, and no model was fitted or trained. The current executor remains CPU-only with restricted shell egress; the restored dataset and 789-file source snapshot are already available.

## Evidence
- Eom, Jeon & Jeong, “UWB Impulse Radar-Based Open-Set Gesture Recognition Using Transformer and One-Versus-Rest Classifier,” *IEEE Internet of Things Journal* 12(13), 2025, DOI 10.1109/JIOT.2025.3556901. The university research record describes UWB impulse radar open-set recognition with a temporal/spatial Transformer feature extractor and per-known-class binary inspectors. [IEEE Xplore record](https://ieeexplore.ieee.org/abstract/document/10947050); [Seoul National University research record](https://snu.elsevierpure.com/en/publications/uwb-impulse-radar-based-open-set-gesture-recognition-using-transf/).
- Yang et al., “Spatiotemporal graph prototype learning with adversarial training for UWB open-set gesture recognition,” *Measurement Science and Technology* 37 (2026), 286102, DOI 10.1088/1361-6501/ae7fab. The publisher record describes UWB open-set recognition; its abstract/indexed publisher text says it evaluates on UWB-Gestures, reports five held-out gesture-class splits, and includes further cross-subject analysis. It proposes a dual-stream spatiotemporal encoder, graph prototypes, and adversarial subject disentanglement. [IOP Publishing article](https://doi.org/10.1088/1361-6501/ae7fab).
- The original public dataset contains 12 dynamic hand-gesture classes from eight subjects using impulse UWB radar. [Scientific Data descriptor](https://www.nature.com/articles/s41597-021-00876-0).

## Decision
Generic UWB open-set gesture recognition is now directly preempted, including a recent study explicitly using this same dataset and adding a cross-subject analysis. Merely holding out classes differently, replacing its Transformer/prototype components, or adding a rejection threshold would not clear the requested method contribution. The two papers' exact split details and numerical results are not compared here, and this review does not establish equivalence to the project's eight-fold participant-held evaluation.

No experiment was started. There is no remaining evidence-cleared candidate in the current branch that can be responsibly trained on this fixed cohort without first stating a materially distinct question and protocol. A new synchronized cohort with external motion truth remains blocked by unavailable capture hardware; cross-dataset transfer remains behind independent semantics, archive/license association, and participant-disjoint gates.

## Research sources
- https://ieeexplore.ieee.org/abstract/document/10947050
- https://snu.elsevierpure.com/en/publications/uwb-impulse-radar-based-open-set-gesture-recognition-using-transf/
- https://doi.org/10.1088/1361-6501/ae7fab
- https://www.nature.com/articles/s41597-021-00876-0
