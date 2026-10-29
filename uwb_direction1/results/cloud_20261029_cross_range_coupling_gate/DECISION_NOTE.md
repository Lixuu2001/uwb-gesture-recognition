# 2026-10-29 — cross-range-bin co-modulation candidate gate

## Candidate screened
A single-view, phase-free descriptor based on lagged co-modulation among neighboring range-bin envelopes in the existing clutter-removed range-time maps. The proposed mechanism is that a moving hand may produce a structured sequence of jointly active adjacent range bins. This differs mathematically from temporal self-similarity, the previously screened range-ridge/path summaries, and cross-view fusion. It is observable in the current real-valued maps, so this was worth a source-only screen.

## Literature check and disposition
A bounded targeted search did not identify an exact IR-UWB hand-gesture paper naming this precise neighboring-bin covariance descriptor. That absence is not novelty clearance. The descriptor is also a generic second-order texture/correlation statistic: Khan et al., *Sensors* 17(4), 833 (2017), DOI 10.3390/s17040833, already uses magnitude-distribution, frequency, and TOA-variation descriptors for IR-UWB gestures; Leem et al., *IEEE TIM* 69(4), 1066–1081 (2020), DOI 10.1109/TIM.2019.2909249, studies IR-UWB hand localization/tracking; and Xue et al., *Frontiers in Neuroscience* 20 (2026), DOI 10.3389/fnins.2026.1864448, uses the same UWB-Gestures corpus with local ConvMixer and global temporal modeling of time-range maps plus gated/bilinear fusion. A 2026 fine-grained radar HGR paper also uses range-time, Doppler-time, and angle-time representations with range-bin normalization (DOI 10.3390/electronics15020437).

The proposed statistic therefore lacks a clear algorithmic distinction from standard map texture/correlation and the same-corpus model's spatiotemporal interaction modeling. Without hand-motion truth, a performance gain would not establish the claimed physical co-modulation mechanism. The bounded search is not exhaustive. **Candidate not cleared for a research-contribution experiment.** Do not turn a standard feature ablation into an originality claim.

## Execution disposition
No source data values, labels, predictions, or test users were read. No fitting, training, or parameter search ran. The current verified clutter-removed cache remains the only relevant input; the current status says raw complex echoes and calibrated/synchronized motion truth are absent. The cloud preflight found no active UWB run or model checkpoint; NumPy, SciPy, and scikit-learn are present, PyTorch is absent, and the executor has five CPU cores. No download or restore was needed.

This is a literature/feasibility no-go for this candidate, not a negative result about its possible predictive utility and not proof that all new methods are impossible. Continue only if new primary-source evidence or a materially distinct observable mechanism changes the novelty case; do not run another feature grid to rescue this candidate.

## Bounded sources
- Ahmed et al., *Scientific Data* 8, 102 (2021), DOI 10.1038/s41597-021-00876-0 — dataset/acquisition and separate monostatic views.
- Khan et al., *Sensors* 17(4), 833 (2017), DOI 10.3390/s17040833 — IR-UWB gesture descriptors.
- Leem et al., *IEEE Transactions on Instrumentation and Measurement* 69(4), 1066–1081 (2020), DOI 10.1109/TIM.2019.2909249 — IR-UWB localization/tracking.
- Xue, Yang & Zhu, *Frontiers in Neuroscience* 20 (2026), DOI 10.3389/fnins.2026.1864448 — same-corpus tri-view/global-local time-range method.
- Fine-Grained Radar Hand Gesture Recognition Method Based on Variable-Channel DRSN, *Electronics* 15(2), 437 (2026), DOI 10.3390/electronics15020437 — RT/DT/AT map features and range normalization.
