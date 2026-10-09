# Continuation addendum — Pantomime endpoint review (2026-10-09)

The targeted review found that the institutional records substantially clarify Pantomime licensing/provenance: Aalto lists CC-BY-4.0 with DOI 10.5281/zenodo.3772659; Aalto catalog listings and Research.fi pair that DOI with Zenodo 4459969, and Research.fi marks the dataset CC BY 4.0. The Zenodo 4459969 page itself shows a blank license field. Its archive member manifest and file-level license association remain unchecked; neither Pantomime archive was downloaded or extracted. The source-code archive has a separate LICENSE, whose contents were not read.

Exact-endpoint prior-art review adds:
- Feng et al. (IEEE TMC 2026) transfer image-domain gesture knowledge to unlabeled FMCW/mmWave gesture signals using trajectory-image representation, adversarial/contrastive alignment and target self-training. This closes off broad heterogeneous gesture/image-to-radar transfer claims. Their paper does not report the UWB-Gestures/X4 → Pantomime/point-cloud endpoint.
- GesturePrint (ICDCS 2024) evaluates Pantomime and two other datasets independently as benchmarks; it does not transfer a UWB-Gestures-trained model to Pantomime.
- UWB-Gestures and Pantomime use different radar modalities and gesture vocabularies. The coarse swipe/circular name map is not evidence of kinematic equivalence.

A bounded targeted search did not find an exact UWB-Gestures/X4-to-Pantomime-point-cloud family-transfer study; this is not exhaustive clearance. The endpoint remains unevaluated, but the semantic label aggregation alone is too generic to claim a method or contribution. Do not start transfer training until a substantive question beyond broad label transfer, target-label-independent semantic justification, manifest/release verification, and a participant-disjoint protocol with matched baseline are all clear. If label correspondence cannot be justified without examining target test labels, drop this endpoint.

No labels or radar arrays were opened, no model was fit, and no training started. Executor remains CPU-only with enforced restricted egress and no Zenodo shell access; no active experiment process was found. See research/PANTOMIME_ENDPOINT_PRIOR_ART_20261009.md and results/cloud_20261009_endpoint_review/RUN_REPORT.md.
