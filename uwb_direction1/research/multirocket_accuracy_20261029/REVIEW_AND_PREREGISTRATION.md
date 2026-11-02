# Full-size MultiRocket source pilot: prepared, not launched

This is a published-method accuracy experiment, not a proposed original method. The user's latest instruction permits this distinction. The ongoing MV-GLFNet-style full/local CNN experiment keeps priority; this runner refuses to open UWB arrays or create a run directory until that entire queue is complete and its independent audit passes. No new real-data experiment is launched by this registration.

## Why this configuration has not been rejected

The 2026-10-04 response-pooling report closed its small independent bank/range subset/altered statistics configuration, and explicitly states that this does not test full MultiRocket. Today's full MiniRocket and HYDRA outcomes also do not evaluate MultiRocket's full base-and-difference, four-statistic transform. No equivalent completed full-size MultiRocket experiment was found in the current project memory and STATUS records. This is an accuracy-baseline rationale; it makes no novelty argument.

Tan, Dempster, Bergmeir and Webb, *MultiRocket*, Data Mining and Knowledge Discovery36,1623–1646(2022), DOI10.1007/s10618-022-00844-1, is the original method source. Read scope in this continuation: publisher HTML abstract, introduction and method Sections3.1–3.5, plus the original multivariate implementation and driver. The original benchmark is mainly univariate UCR, and is not evidence of improved UWB recognition. We have not claimed to review the complete paper/appendices. Original repository pinned at3ccaa4f8ed11769b904af885e17018c39c89a2ac.

## Implemented variant and upstream defect

Review identified missing first-difference channel-index advancement in the original multivariate transform. The public aeon issue2967 and merged fix3003 document the same problem; author Angus Dempster confirms the advancement defect. The separate choice to reuse the base-series channel selections is considered acceptable in that discussion, and is preserved by the corrected implementation. Sources:

- https://github.com/aeon-toolkit/aeon/issues/2967
- https://github.com/aeon-toolkit/aeon/pull/3003
- https://github.com/ChangWeiTan/MultiRocket/blob/3ccaa4f8ed11769b904af885e17018c39c89a2ac/multirocket/multirocket_multivariate.py

Use unedited aeon1.6.0 core, n_kernels6250, four features/kernel, two representations, max32dilations, no input normalization, seed20261029, one thread:49,728 features. This is the published default feature scale with the maintained multivariate fix; it is not bitwise reproduction of the original buggy code. Adapter checks the raw core output for finite values before aeon's public API filling fallback. Synthetic public-API parity passes.

The ridge readout is a declared protocol adapter: training-only StandardScaler, equal user/class weights, fixed alpha1, weighted intercept, float64 dual Cholesky solve with a residual check. Both full MultiRocket and the matched9996-feature MiniRocket control use this same readout. MiniRocket reuses the exact stored held1 transform parameters (fit on users3–6), not another seed. This is not reproduction of the author's classifier/inner validation setup. Historical MiniRocket LSQR solves reached160–172 iterations under a200cap, unlike the capped HYDRA case; using the same checked direct readout in both new arms removes solver-completion differences without tuning alpha or seed.

## Frozen source roles and decision

Exact configuration is in PROTOCOL.json. Reuse the existing SHA-bound6900 retained source IDs: train users3–6(4700), validation user2(1000), diagnostic user1(1200). All189range bins in all3views form567channels over90frames. These are signed, real clutter-removed RF maps. First differences are signal changes, not measured hand velocity/acceleration. No users7–8 labels are indexed. No label-based transform fitting, extra preprocessing, hyperparameter search or new seed is allowed.

All transforms, readouts and both prediction arrays must be saved with hashes before diagnostic-user labels are indexed. Validation is reported; no parameter is selected from it in this fixed pilot. Continue only if MultiRocket-minus-matchedMiniRocket >=0.5pp and MultiRocket >=max(fullCNN,localCNN)-0.5pp on the same source diagnostic user. This licenses only a separately preregistered six-source-user comparison. Failure closes this exact configuration without rescue grids. A budget/engineering stop is incomplete, not a scientific negative. This single repeatedly developed source user cannot establish independent generalization, eightfold improvement or an original contribution.

## Executable preparation and budget evidence

adapter.py and run_pilot.py implement the transform/readout and the guarded pilot. Synthetic checks cover49,728-feature shape/finite outputs, seeded fitting, batching, public API parity, serialized replay, and weighted ridge agreement against sklearn to1e-10. An initial import lacked Deprecated; it was installed only in this task target. The first smoke pass used the global Numba0.68 outside aeon's declared <0.64 requirement and is retained as preliminary evidence only. The accepted rerun uses isolated Numba0.63.1/llvmlite0.46.0 and passes the same checks. No active CNN packages were changed.

COST.json contains a single-thread synthetic estimate of924.803seconds with3x margin, including both transforms, scaled4700-row direct readouts and180seconds additional MiniRocket/IO allowance. It is an estimate, not a real-data runtime result. Execution retains independent1800-second wall and process CPU limits. An independent supervisor writes termination/incomplete receipts even after a child hard kill, and retains partial artifacts. It logs partial states and fails numerical solves instead of treating truncated optimization as accuracy evidence. The unfinished-predecessor guard was exercised: expected error before any run directory or data access.

After the original CNN completes, first run its independent complete audit and keep the required COMPLETE_AUDIT.json path. Only then, and after verifying frozen hashes, execute:

```bash
python /workspace/uwb-multirocket-20261029/launch.py --run
```

Do not run alongside the original worker. If an engineering stop occurs, inspect the exact partial artifacts and remaining aggregate budget; no automatic restart, fresh statistical trial, parameter change or reselection is authorized by a mechanical retry count. The future result needs independent artifact/metric checking before applying the gate. Raw arrays, fitted transforms, IDs and readouts remain in private cloud/Drive, outside Git.

The synthetic budget self-test separately triggered a wall timeout and a process CPU limit, verified durable termination receipts, and verified that a partial file survived. No UWB files were read. launch.py caps the child at1790wall seconds with3seconds termination grace, within the1800-second total allocation; pre-exec process CPU limits include imports. No scientific subprocess was started by this test.
