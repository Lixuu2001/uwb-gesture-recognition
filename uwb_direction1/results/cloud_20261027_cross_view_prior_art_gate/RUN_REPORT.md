# Held-radar-view generalization: bounded prior-art gate

**Date:** 2026-10-27 01:10 UTC  
**Decision:** do not start a held-view experiment as a novel-method study. The evaluation question is testable in principle, but current evidence does not clear a distinct method contribution. This is a literature/protocol gate, not a scientific null result.

## Candidate

Outer leave-one-participant-out evaluation with one physical radar view held out: train on two views (Left/Top/Right) from source participants, validate only on the predetermined source participant using those same two views, then evaluate the held participant on the held view. Rotate the held view. This asks whether a model transfers to an unseen X4 radar viewpoint; it is distinct from ordinary participant-held-out testing but is an evaluation setting, not itself a new algorithm.

## Literature checked

- Khan et al., “UWB-gestures, a public dataset of dynamic hand gestures acquired using impulse radar sensors,” *Scientific Data* (2021), DOI [10.1038/s41597-021-00876-0](https://doi.org/10.1038/s41597-021-00876-0). The dataset paper describes three separately accessible impulse-radar views (Left, Top, Right), explicitly supporting differing hardware placement and view-selection research. Thus the three-view data setup is established.
- Xue et al., “Neuromorphic-inspired multi-view global-local fusion for IR-UWB radar dynamic gesture recognition,” *Frontiers in Neuroscience* (2026), DOI [10.3389/fnins.2026.1864448](https://doi.org/10.3389/fnins.2026.1864448). Full text was reviewed in the earlier project gate. It uses these three views on the same corpus, evaluates single-view and multi-view configurations, and uses a subject-independent 6/2 participant split. This preempts generic same-corpus multi-view fusion and accuracy comparisons; its reported protocol does not establish held-radar-view testing.
- Chen et al., “Domain-Generalized Gesture Recognition via mmWave Radar Signal Multi-View Learning,” *IEEE Transactions on Instrumentation and Measurement* (2025/2026), DOI [10.1109/TIM.2025.3637962](https://doi.org/10.1109/TIM.2025.3637962). The bounded accessible material includes the abstract and author implementation README, not the article full text. The paper claims domain-generalized multi-view radar-gesture learning with domain-knowledge/evidence fusion; the author repository documents multi-view radar representations, fusion strategies, and cross-domain learning ([MVRADAR](https://github.com/Tkwer/MVRADAR)). The exact sensor-view protocol and method details remain unverified, so this source is a strong generic DG/multiview prior, not a claim of exact protocol equivalence.
- Chen et al., “Cross-Position and Cross-Orientation Human Gesture Recognition via mmWave Radar,” IEEE IWS 2026, DOI [10.1109/IWS69027.2026.11674697](https://doi.org/10.1109/IWS69027.2026.11674697). Publisher abstract/metadata describes parameter mapping based on spatial physical features for position/orientation variation followed by multi-dimensional fusion. Full text was not reviewed. This further cautions against claiming generic physical view alignment as new.

This is a bounded search; it does not establish exhaustive novelty.

## Data and validity constraints

The current restored input is a real-valued clutter-removed tensor `X[9593,3,90,189]`, not raw complex echoes. View/repetition pairing is keyed by participant, gesture, and repetition index; source metadata calls records the same event, but external timestamps/shared trigger and phase calibration have not been verified. Therefore a held-view score can be evaluated as a corpus protocol, but paired-view consistency cannot be interpreted as synchronized physical motion or causal cross-radar transfer.

The same eight participants have prior method-development exposure. Any result would be exploratory and cannot support independent-user generalization. A fair experiment would require frozen preprocessing, participant splits, quality masks, model family, hyperparameters, checkpoint-selection rule, and matched controls before inspecting held-view results. No such experiment was authorized here because the method novelty gate did not clear.

## Runtime and data handling

The cloud executor is running at revision 1437: 5 CPU cores; NumPy, SciPy, and scikit-learn available; PyTorch unavailable; restricted networking enforced. The restored data path and 789-file source snapshot are present. Process/checkpoint scan found no active UWB/training job and no common serialized checkpoint files within workspace depth 7. No sample arrays or label values were read; no data were downloaded; no fitting or training ran.

## Disposition and next step

Do not present held-view evaluation, multi-view fusion, domain-generalization, or spatial mapping alone as a new method. Before any held-view run, obtain and lawfully review the TIM 2025/2026 full text and determine whether its mechanism already covers this setting. If it does, close the method route. If it does not, a protocol-only benchmark may be considered separately, but it would not satisfy the current original-method goal without a distinct signal mechanism and honest limits on the eight-user corpus. Current contribution goal remains unresolved.
