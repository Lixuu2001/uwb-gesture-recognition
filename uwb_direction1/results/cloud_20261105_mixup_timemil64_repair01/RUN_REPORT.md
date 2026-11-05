# Fixed input Mixup transfer: repair01

## 2026-11-05 Mixup synthetic import failure repaired before any UWB fitting

Preregistered synthetic validation returned actual exit1 after2.001909wall seconds: local profile.py shadowed Python stdlib profile during PyTorch Dynamo/cProfile import, AttributeError(no run). This is an execution defect, not an accuracy negative. No UWB arrays/labels opened, model fitting/update not reached. Original failed stdout/termination retained.

Engineering repair renames local file to capacity_profile.py, updates its controller invocation, preserves old execution directory, and uses execution_repair01 for the rerun. Exact alpha1/TimeMIL850973params/64epochs/B64/seed20261029, roles/selection/diagnostic gate/caps unchanged. Corrected frozen runnable snapshot is research/cloud_20261105_mixup_timemil64_repair01, superseding the original preparation's runnable file names; original preregistration and failure preserved. Synthetic author-formula/gradient/native RNG/journal tests and full-batch resource measurement still pending; no background model job or UWB accuracy result.

