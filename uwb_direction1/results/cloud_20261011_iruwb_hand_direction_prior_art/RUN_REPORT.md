# IR-UWB hand-direction/area prior-art screen

**Date:** 2026-10-11  
**Decision:** generic hand-direction or apparent-area canonicalization is not cleared as a new contribution; no experiment started.

## Candidate family screened

Following the rejection of generic multi-view contrastive alignment, this screen checked whether a simple signal-level contribution could be framed around direction or apparent hand area in IR-UWB returns, especially to normalize across physical radar views.

## Primary-source evidence and access limits

1. **Kim et al., “Hand-Based Gesture Recognition for Vehicular Applications Using IR-UWB Radar,” Sensors (2017), PMID 28398267.** PubMed's indexed abstract says the work proposes an IR-UWB hand-gesture algorithm intended to remain robust to changes in distance or direction and to reject meaningless motions. The linked PMC full text triggered a browser-verification page in this executor, so the detailed algorithm and protocol were not independently reviewed here. [PubMed record](https://pubmed.ncbi.nlm.nih.gov/28398267/); [PMC full text endpoint](https://pmc.ncbi.nlm.nih.gov/articles/PMC5422194/).
2. **“Algorithm for Gesture Recognition Using an IR-UWB Radar Sensor,” Journal of Computer and Communications (2016), DOI 10.4236/jcc.2016.43015.** The publisher-indexed abstract describes using hand movement direction and change in the frontal hand surface area facing the radar. The DOI page returned an access error in this executor; this is abstract-level evidence, not a full-text review. [DOI](https://doi.org/10.4236/jcc.2016.43015); [publisher PDF](https://www.scirp.org/pdf/jcc_2016030217032723.pdf).
3. The more recent X4 corpus paper, **Xue et al. 2026 MV-GLFNet**, already studies the same three physical views and multi-view fusion. Prior report `uwb_direction1/results/cloud_20261011_cross_view_contrastive_prior_art/RUN_REPORT.md` records that boundary.

## Consequence

A method stated only as extracting/normalizing motion direction, radar-facing hand area, or combining those cues with generic view-invariant learning would be too close to established IR-UWB gesture concepts. The old papers use different setups/tasks, so this does not prove that their numerical results transfer to the UWB-Gestures corpus; it does rule out a broad novelty claim based only on those feature ideas. A future candidate needs a narrower signal-level hypothesis and evidence that it handles a documented failure mode not already addressed by these papers.

No UWB labels, tensor values, or checkpoints were read. No data was downloaded, fit, or trained. The cloud executor remains CPU-only and current network policy reports an unknown state with no listed custom hosts; no policy bypass was attempted.

## Next action

Keep screening only for a distinct, source-supported method tied to available X4 representations and labels. Do not launch a training grid or present simple physical descriptors as novel.
