# Fixed input Mixup transfer: repair02

## 2026-11-05 Mixup synthetic optimizer-state alias failure diagnosed and repaired

After the prior import fix, native resume equality test returned actual exit1 after19.018124wall seconds. A separate synthetic diagnosis proves restored optimizer state shares mutable tensor references with the saved in-memory fixture: fixture SHA changed after the left update. Both branches' Beta/permutation/mixed input/loss were identical; restored optimizer-state comparison failed. This is an engineering state-isolation defect, not an accuracy negative. No UWB arrays/labels or fitting used.

Restore now deep-copies the optimizer-state mapping before native load_state_dict; synthetic test additionally asserts the fixture digest stays unchanged. This copies state values without changing native update arithmetic. Corrected frozen snapshot research/cloud_20261105_mixup_timemil64_repair02 supersedes repair01 for execution; both failed logs and actual terminations and causal diagnosis are retained. Alpha1/full TimeMIL64/B64/seed20261029, roles/selection/accuracy gates/resource caps unchanged; no rescue configuration. Rerun author/gradient/full-state/RNG/journal tests and CPU capacity before any UWB fit. No active model job or new accuracy result.

