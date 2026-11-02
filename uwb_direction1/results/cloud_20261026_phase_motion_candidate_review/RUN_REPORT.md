# 已有单雷达相位运动结果：源码与先例审查

**时间：** 2026-10-26 UTC  
**分支父提交：** bd53d8f7c7c45df90136fd31d09df59669cb9ff7  
**结论：** 保留为历史实证候选；未验证原创方法贡献，未独立复跑。此轮只检查现有报告、代码和结果汇总，并做定向文献核查；没有读取数组值或标签、拟合模型、启动训练、下载 Drive 文件或改变 Drive 权限。

## 源码证据和复核范围

目标云端包含已验真的数据恢复包（86 个 payload 文件）和 789 文件原源码快照。审阅的源码记录位于不可变项目快照：
- `experiments/p1_20260905/FIRST_PART_REPORT.md`，SHA-256 `e7d43805290472655d13c0395b1d212b95a15c7dfd82dbb9f13f1db1bdf09f45`
- `experiments/p1_20260905/AUDIT.md`，SHA-256 `3496932556c3ade901c5e383d7ca1858d0a0df8619d53b3fa7ad7cac585e5e30`
- `experiments/p1_20260905/PLAN.md`，SHA-256 `43b74c634851469baaa881f8878fe5ceff51e088b96d475b670e8169a3f715b8`
- `experiments/p1_20260905/RUN_STATE.md`，SHA-256 `cf01dab023409ba8b7df80b876b64ab94d45df434334f40401a6ecf708ea7528`
- `experiments/p1_20260905/results_clean/RESULTS_LIVE.md`，SHA-256 `e531167c7b3d463b65030ba37e4fbb162b5ffe4c7367492f9f9504bfa4661ebf`
- `experiments/p1_20260905/vendor/uwb_osr/data.py`，SHA-256 `b4d9a626a87219b0a378e7b23120aa260f290dcd3310a6a1572f41756f20b98a`
- `experiments/p1_20260905/run_r_seed2027.py`，SHA-256 `d70c8df98ecefcf392820f71a6d105c1fe5b454ef058cf55c2092befd6b34112`
- `experiments/p1_20260905/phase_sign_control.py`，SHA-256 `552f5a13f2276d7f7a5bed2d4649dc388aa05897e6003e3038078846eced4425`

The historical report documents a fixed 12-class, eight-participant LOSO analysis after a predeclared quality mask retained 9,300 records. It reports Right-view amplitude-only at 68.57% and amplitude-plus-phase-motion at 72.41% (seed 2026, +3.84 pp, 8/8 users improved); the matched seed-2027 pair is 68.22% versus 72.40% (+4.18 pp, 8/8). A same-width duplicate-amplitude control was 67.86%. The sign-specific control is inconclusive: unsigned phase was 70.83%, with signed-minus-unsigned participant bootstrap CI [-0.69, +3.72] pp and only 5/8 participants positive. Thus the evidence supports the combined representation in those reported runs, not a signed-direction-specific cause.

The source implementation forms an analytic signal along fast time, removes its slow-time mean, calculates adjacent-frame complex products, and uses their principal phase angle weighted by echo magnitude; that channel is concatenated with amplitude. The source explicitly warns this is not independently calibrated velocity and may be affected by phase wrapping/aliasing, multipath and extended scatterers.

## Literature boundary

- Skaria et al., *IEEE Access* (2020), DOI [10.1109/ACCESS.2020.3037062](https://doi.org/10.1109/ACCESS.2020.3037062), describe impulse-UWB gesture signatures as range-Doppler frame sequences. This establishes UWB Doppler/motion representations as prior art; the abstract does not establish the exact same amplitude-plus-analytic-phase concatenation used here.
- Zhang et al., *IET Signal Processing* (2026), DOI [10.1049/sil2/9971257](https://doi.org/10.1049/sil2/9971257), describe adjacent-chirp phase difference as a speed cue and use RTM/DTM features for FMCW gestures. This is a related physical principle in a different radar modality and is not an exact replication.
- Ahmed et al., *Scientific Data* 8, 102 (2021), DOI [10.1038/s41597-021-00876-0](https://doi.org/10.1038/s41597-021-00876-0), document this corpus as eight volunteers and twelve gestures from three separately saved impulse-UWB radar views. It does not provide independent motion-capture truth for interpreting this project's phase feature as calibrated hand velocity.

These sources support the physical motivation and make a generic “phase/Doppler features for radar gestures” novelty claim untenable. They do not, by themselves, answer whether this specific feature adds useful information under a strict held-user protocol for this corpus. The bounded search is not an exhaustive novelty review; no claim of exhaustive full-text review is made.

## Limits on interpretation

1. The Right view was selected after earlier multiview exploration. The same eight participants have already been used in substantial project development; they are not an untouched external cohort.
2. The two seeds change both initialization and the within-user train/validation split. They are repeat runs on the same users, not independent replications.
3. The quality mask excludes 300 records (conflicting HV2 labels and unverified HV6-G4 cross-view alignment); the results concern the retained set, not all original rows.
4. The Git-tracked source snapshot has the aggregate `RESULTS_LIVE.md`, protocol, code and prose report, but not a complete per-fold phase-model prediction/checkpoint bundle under `results_clean`. The current executor has no serialized model checkpoint and no PyTorch installation. The numbers above are source-reported, not independently recomputed here.
5. No claim is made that phase motion recovers 3-D motion or true radial speed; the available processed signal and report do not supply synchronized calibrated kinematic ground truth.

## Research disposition and next falsifiable step

Keep this result as a potentially useful empirical chapter/corpus result, clearly labeled exploratory and post-selection-sensitive. Do not call its signal transform a new algorithm or mark the contribution goal complete.

The next useful decision is an artifact-level reproducibility audit: determine whether the original per-fold predictions, preprocessing cache manifest, exact code/environment record, and checkpoint/optimizer snapshots exist in the already-authorized private archive. If they can be recovered and verified, a single preregistered held-user confirmation can test whether amplitude-plus-phase beats amplitude and the matched-width control without selecting the Right view from outer-test outcomes. If they cannot, the report remains historical evidence only and the contribution claim stays unresolved. A later metadata-only audit verified the specified Drive folder remains owner-only (`shared=false`). Its 803 direct children had no filename match for `p1_20260905`, `phase`, `Right`, `amplitude`, `seed2026`, or `R_concat`. No Drive file content was fetched; opaque or differently named archives are not ruled out. This metadata check does not authorize a model run.
