# Paired-view timing observability: label-free feasibility protocol

Date: 2026-10-08. Scope: a data-feasibility diagnostic only; no classifier, labels, predictions, or fitted model.

## Question and boundary

The preserved UWB-Gestures project pairs Left/Top/Right views by sample index. The public data descriptor says the three monostatic radars operate independently and stores each radar's files separately; it defines 90 slow-time frames as a 4.5 s gesture segment. The available local cache is a single real-valued `float32` amplitude-like tensor `[9593,3,90,189]`. No per-frame timestamps, common-trigger records, complex I/Q, or sample-level synchronized video are present in the restored artifacts. The source code calls index pairing “soft-synchronised”; this is a pairing assumption, not a timestamp audit.

Direct prior art already covers temporal alignment/fusion: dynamic time warping for online multisensor fusion and multi-sensor gesture recognition, event-based synchronization of independently recorded streams, and UWB node clock synchronization. Therefore generic DTW, correlation, or clock-alignment is not a novelty claim. This screen asks only whether the existing paired amplitude clips expose a stable, label-free temporal relationship worth a later, separately preregistered application test.

## Frozen signal and statistic

1. Read `X_clutterremoved.npy` by memory map and `subj_clutterremoved.npy`; do not load or inspect `y_*`, `quality.npy`, labels, logits, or model features. Exclude only exact duplicate triplets after verifying bytewise equality of all three views; report the excluded count. Keep all other finite records.
2. For each view/sample, compute a fixed nonnegative motion-envelope sequence `m[t] = sqrt(mean((X[t+1,:]-X[t,:])**2))`, `t=0..88`, then z-normalize the 89 values. A zero-variance trace is invalid and counted; no adaptive filter, threshold, smoothing, or learned transform.
3. For each of the three view pairs, report Pearson correlation at lag 0 and the maximum correlation over integer lags `[-4,+4]` (±200 ms at the documented 20 Hz). Report the maximizing lag, valid fraction, boundary-peak fraction, per-user medians, and the distribution across records. Do not convert lag to physical time offset or velocity.
4. Negative controls: (a) within-user shuffled sample pairing for one view, 20 fixed permutations, to test same-index event correspondence; (b) artificial ±1…±4-frame shifts on an independently selected 512-record subset, to verify the estimator's ability to resolve injected offsets. Both are diagnostics, not ground truth.

## Frozen feasibility gate

Continue to a separately preregistered application experiment only if all hold:

- at least 95% of unique triplets yield finite nonconstant motion traces;
- for at least two of the three view pairs, median matched lag-zero correlation is at least 0.30 and median matched best-lag correlation exceeds the same-user permutation-null median by at least 0.10;
- for those same pairs, at least 6/8 user medians have `|best_lag| <= 1` frame and fewer than 5% of matched records peak at the ±4-frame search boundary;
- the injected-shift control recovers at least 90% of nonzero shifts exactly.

Passing this gate would establish only a potentially measurable same-event envelope relation. It cannot establish actual clock offset or alignment accuracy without timestamp/video ground truth. Even a pass does not justify an originality claim; standard temporal alignment is prior art, and any recognition follow-up would need a fixed strong baseline, time-shuffle control, validation-only choice, full eight-user protocol, and direct prior-art comparison. A failed gate closes this candidate without expanding lag range, changing the envelope, or training a classifier.

## Sources checked

- Ahmed et al., “UWB-gestures, a public dataset of dynamic hand gestures acquired using impulse radar sensors,” *Scientific Data* 8, 102 (2021), https://doi.org/10.1038/s41597-021-00876-0. The descriptor specifies three independently operating monostatic radars, separate files, 90 slow-time frames / 4.5 s; it does not document a common trigger or sample-level timestamps in the data description.
- Ko et al., “Using dynamic time warping for online temporal fusion in multisensor systems,” *Information Fusion* 9(3), 370–388 (2008), https://doi.org/10.1016/j.inffus.2006.08.002.
- Choi & Kim, “Combined Dynamic Time Warping with Multiple Sensors for 3D Gesture Recognition,” *Sensors* 17(8), 1893 (2017), https://doi.org/10.3390/s17081893.
- “Automatic Event-Based Synchronization of Multimodal Data Streams from Wearable and Ambient Sensors” (2008), https://doi.org/10.1007/978-3-642-04471-7_11.
- Ledergerber & D’Andrea, “A Multi-Static Radar Network with Ultra-Wideband Radio-Equipped Devices,” *Sensors* 20(6), 1599 (2020), https://doi.org/10.3390/s20061599. It discusses independently clocked UWB nodes; it is adjacent timing context, not the same dataset setup.

## Provenance and limits

Protocol freezes the no-label scope, statistic, controls, and gate before running the signal diagnostic. The data are historical development data used in earlier work. A pass is exploratory feasibility, not independent-user evidence. No raw arrays, extracted feature arrays, or sample IDs are to be committed to Git.
