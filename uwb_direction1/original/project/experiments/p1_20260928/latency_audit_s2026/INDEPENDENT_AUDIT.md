# Independent latency audit

**PASS** — 23218 checks; 0 issues. CPU-only; no GPU timing rerun.

Frozen files: 19/19 SHA-256 matches; upstream inventory: 308 byte hashes; 279 clips, 1953 method observations, 17 GPU-environment records.

| Method | Mean ms | Median ms | p95 ms | Mean actual views | Offline prediction/path differences |
|---|---:|---:|---:|---:|---:|
| baselineRight | 6.221236 | 5.9446 | 8.33300 | 1.000000 | 1/0 |
| fixed1 | 6.546686 | 6.2586 | 8.66670 | 1.000000 | 0/0 |
| fixed2 | 12.411842 | 11.9032 | 16.08250 | 2.000000 | 0/0 |
| fixed3 | 18.024117 | 17.3396 | 23.10739 | 3.000000 | 0/0 |
| frozen_budget2_cascade | 13.180557 | 12.5229 | 20.58804 | 2.017921 | 0/0 |
| class_gain | 11.918466 | 12.3305 | 18.65070 | 1.777778 | 0/0 |
| extra_trees | 24.347789 | 26.3057 | 35.11182 | 1.992832 | 0/0 |

Every per-fold and pooled mean, median, NumPy-linear p95, per-user mean, user macro mean, view histogram, and pooled physical-view cost was recomputed from the observations. All sampled IDs were regenerated from labels/subjects and the frozen RNG; all 837 original preprocessing outputs were rerun on CPU and compared bitwise with prepared data. All row path/access/count/logit assertions and the frozen offline prediction/path fields were checked.

The saved source and run metadata specify at least 10 warmups per method and forced path on each fold. The archive does not include a separate trace of each warmup call.

The recorded 17 environment checks consist of preflight and before/after checks for every fold. Each has two process snapshots, five utilization samples and last-three readings <=5%; the same QQ.exe PID and process identity remain throughout the records. This was a nonexclusive desktop-context measurement. Sampling cannot establish continuous absence of interference.

The synchronized elapsed time includes preprocessing and float16 packing for requested views, CPU-to-GPU transfer, incremental view encoding, fusion/output transfer, and CPU policy decisions. Host wall-clock stage measurements overlap, so their individual means should not be summed. For ExtraTrees, recorded routing/harness mean is 11.359 ms versus 0.060 ms for fixed3; this is descriptive and does not isolate CPU decision time.

The frozen formal 9300-clip policy result remains STOP_VALUE_POLICY_MECHANISM. The ExtraTrees route processed fewer views on this 279-clip timing set but had a larger measured mean than fixed3. The simple cascade is a measured local tradeoff; its formal eight-user mean cost is 2.004615530303 (>2), and its accuracy is more than 1 percentage point below fixed three-view. Fixed2 is the validation-selected two-view combination, not an after-the-fact choice from these latency rows.

A 90-frame, 20 Hz clip already represents about 4.5 seconds of acquisition before this measurement begins. The result is processing latency after a host-resident, clutter-removed clip is available; it excludes sensing, file I/O, earlier clutter removal and model loading. It does not establish full interaction latency, causal-prefix recognition, power savings or a second independent hardware measurement.

## Limits of independent verification

- Upstream and frozen hashes match the saved inventories today; this is internal byte consistency, not an external timestamp witness.
- Direct-versus-batch-one replay flags were structurally audited and tied to saved visited-logit hashes. The GPU logits were not regenerated.
- Original frozen preprocessing definitions were reexecuted and compared with prepared data. This checks reproducibility of the frozen transform, not a separately authored transform.
