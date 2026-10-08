# Paired-view timing observability: label-free feasibility result

Date: 2026-10-08. Result: **fixed feasibility gate failed; close this candidate.** This is not a classifier result, not a clock-offset estimate, and not evidence that the sensors were physically unsynchronized.

## Question and prior art

The UWB-Gestures article describes three independently operating monostatic radars and separately stored sensor files, with 90 slow-time rows spanning 4.5 seconds. Its data description provides no frame timestamps or common-trigger record. The preserved project code pairs samples by repetition index and calls this “soft-synchronised.” The local restored cache contains only a real-valued float32 `[9593,3,90,189]` tensor and participant IDs; it has no complex I/Q, frame timestamps, or sample-level synchronized video.

Temporal alignment and DTW fusion are established prior art: Ko et al. studied online multisensor temporal fusion with DTW, Choi & Kim studied multi-sensor gesture recognition with DTW, and Ledergerber & D’Andrea discuss clock behavior in UWB radar networks. We therefore screened whether this particular cache has an identifiable sample-paired envelope lag; we did not propose generic DTW as new.

## Frozen procedure

Protocol `research/PAIRWISE_TIME_ALIGNMENT_FEASIBILITY_PROTOCOL_20261008.md` was committed before reading signal statistics (commit `51a622cbfe54ea5ef8b2f4f4d90bc69335e243bf`). It defines the sole statistic as the per-view RMS of consecutive slow-time differences across fast-time bins, z-normalized within each clip. The only fitted quantity is no model: report zero-lag Pearson correlation and the maximum over fixed lags −4…+4 frames. Controls are a 20-repeat within-participant sample-index shuffle and known synthetic ±1…±4-frame offsets. No label, class file, quality-label mask, prediction, or learned feature was read; there was no model fitting or training.

The script verified all 100 source-manifest exact-duplicate triplets by comparing all three arrays and excluded the duplicate row from each. Of 9,493 unique triplets, 9,493 (100%) had finite, nonconstant envelope traces. The input array hashes match the preserved data manifest.

## Results

| View pair | Median zero-lag corr. | Median best-lag corr. | Matched minus shuffled median best corr. | Median absolute best lag | Best-lag boundary rate |
|---|---:|---:|---:|---:|---:|
| L–T | 0.158947 | 0.285537 | +0.145539 | 3 frames | 48.46% |
| L–R | 0.183686 | 0.289205 | +0.146648 | 3 frames | 40.61% |
| T–R | 0.117728 | 0.236390 | +0.103430 | 3 frames | 46.30% |

The same-user shuffle contrast exceeded +0.10 for all three pairs, but none met the required zero-lag correlation ≥0.30. No pair met the lag-stability rule: zero of eight users per pair had median absolute lag ≤1 frame, while 40.61–48.46% of sample maxima fell on the ±4-frame search boundary. The fixed estimator recovered all injected shifts exactly for all three pairs (100%); that validates only the implementation's sensitivity to artificial offsets.

The precommitted joint continuation gate is **false**. Do not widen the lag search, alter the envelope, promote a different pair, or run recognition training from this screen. High boundary rate and weak lag-zero correspondence mean this observable does not identify a stable paired-view delay under the frozen method. Since there is no timestamp or synchronized motion reference, even a strong envelope match could not certify a physical clock offset.

## Independent audit and reproducibility

An independent NumPy `corrcoef` audit replayed 128 records × 3 view pairs (384 lag vectors): maximum absolute correlation error `2.94e−8`, exact best-lag match `100%`; it rechecked all 100 duplicate triplets, all synthetic-shift recoveries, and the frozen gate arithmetic. See `AUDIT.json`, `SUMMARY.json`, and `SOURCE_HASHES.json`.

Reproduce from the existing cloud restore without another download:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python tools/paired_time_alignment_diagnostic.py \
  --data /workspace/uwb-restored-20260930 \
  --out /workspace/uwb-pairwise-time-alignment-20261008

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python tools/audit_paired_time_alignment_diagnostic.py \
  --data /workspace/uwb-restored-20260930 \
  --results /workspace/uwb-pairwise-time-alignment-20261008 \
  --out /workspace/uwb-pairwise-time-alignment-audit-20261008
```

Sample-level lag readings and IDs remain in `/workspace/uwb-pairwise-time-alignment-20261008/PAIRED_VIEW_READINGS.npz`, outside Git. The immutable 789-file source snapshot and restored source arrays were not modified; no raw arrays were added to Git. No training process is running. The user’s research goal remains unmet.

## Next justified evidence

The present fixed dataset cannot validate time alignment or 3-D trajectories. A future contribution in that direction requires a new capture with hardware trigger/per-frame timestamps and an independently synchronized visual or motion reference. Otherwise continuation must find another observable with independent ground truth; do not launch another amplitude-descriptor/readout sweep on this cache.

## Sources

- Ahmed et al., *Scientific Data* 8, 102 (2021), UWB-Gestures data descriptor: https://doi.org/10.1038/s41597-021-00876-0.
- Ko et al., *Information Fusion* 9(3), 370–388 (2008): https://doi.org/10.1016/j.inffus.2006.08.002.
- Choi & Kim, *Sensors* 17(8), 1893 (2017): https://doi.org/10.3390/s17081893.
- Ledergerber & D’Andrea, *Sensors* 20(6), 1599 (2020): https://doi.org/10.3390/s20061599.
