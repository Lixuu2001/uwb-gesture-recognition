# Direct prior-art update: same-dataset three-view IR-UWB recognition

Date: 2026-10-08. This is a bounded literature and data-integrity audit, not a new recognition experiment. No gesture labels or radar arrays were read in this update; no fitting or training was run.

## Direct same-dataset paper

Xue, Yang, and Zhu published “Neuromorphic-inspired multi-view global-local fusion for IR-UWB radar dynamic gesture recognition” in *Frontiers in Neuroscience* on 2026-06-19 (DOI 10.3389/fnins.2026.1864448). The paper explicitly uses the public Ahmed et al. UWB-Gestures dataset: 12 classes, 8 participants, 9,600 samples, and three left/top/right X4 radars. It reports a subject-independent split with six participants for training and two for testing and 98.29% average accuracy.

Its method is direct prior art for generic three-view fusion claims: it applies MTI to each 90×189 time-range view, resizes each to 64×128, early-fuses all three, then combines local ConvMixer and global CNN–Transformer branches with adaptive gated first-order and bilinear second-order fusion. It also reports single-, dual-, and tri-view comparisons. Therefore, early fusion, global/local branches, adaptive view/feature fusion, and bilinear interaction on this dataset are not defensible as this project’s original contribution without a materially different, evidence-backed question.

The article’s main text gives a single 6/2 subject split but does not name the held-out participants or report a split seed in the dataset/experiment sections. It specifies 100 epochs and shows a test-accuracy-over-epoch curve; the text inspected does not explain a validation-only checkpoint-selection rule. Its data-availability statement points to article/supplementary material, and the article page has no code-repository link. These are reproducibility limits, not proof of leakage or invalid results. The reported 98.29% must not be directly compared to this project’s existing eight-fold/readout figures without the exact split and matched pipeline.

## Duplicate integrity check relevant to participant-independent splits

The earlier label-free screen identified 100 exact duplicate triplets among the restored 9,593 rows. This follow-up read only `subj_clutterremoved.npy` and those already verified duplicate row indices. All 100 pairs are within participant 2; none crosses participant IDs. Thus these known duplicates do not cross any participant-disjoint split. They could still contaminate a random row-level split if duplicate members are separated. The current audit does not establish whether any other near-duplicates exist.

The reproducible result is `DUPLICATE_SUBJECT_AUDIT.json`; code is `tools/audit_duplicate_subject_overlap.py`. The gesture label vector and radar arrays were not opened for this check.

Reproduce the audit from the existing restore:

```bash
python tools/audit_duplicate_subject_overlap.py \
  --data /workspace/uwb-restored-20260930 \
  --diagnostic-summary results/cloud_20261008_paired_time_alignment/SUMMARY.json \
  --out /workspace/uwb-duplicate-subject-audit-20261008/AUDIT.json
```

## Decision and next evidence

Do not implement MV-GLFNet by recombining repository modules and present it as novelty. A faithful numerical comparison to the paper is not currently exact-reproducible from its main text because the particular two held-out participant IDs and code are not supplied there. A 28-pair rerun over the same eight development participants could characterize split sensitivity, but would be an internal evaluation of known architecture families—not an independent confirmation or a new radar method. It is not launched here.

The project still lacks a validated novel method. The most defensible next route is a new cohort/session/placement evaluation with frozen train/validation/test roles, or a separately measured physical target (for trajectory work, synchronized external motion/pose truth). Until such evidence is present, no further architecture/descriptor sweep on the fixed cache is justified. This is an evidence boundary, not a claim that the overall research is impossible.

## Sources

- Xue, Yang & Zhu (2026), *Frontiers in Neuroscience*, DOI: https://doi.org/10.3389/fnins.2026.1864448.
- Ahmed et al. (2021), *Scientific Data* 8:102, UWB-Gestures dataset descriptor, DOI: https://doi.org/10.1038/s41597-021-00876-0.
