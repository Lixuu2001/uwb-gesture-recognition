# Raw temporal reversal with label mapping: failed source gate; fixed screen closed

Preregistration `322775de8ccb7126eeccd21214d1cffa7574f5fe` was pushed and remote-read back before training-only timing and real fitting. Protocol SHA256 `39ebddf4e7f2baeee8de6cd3f9886a6c4354216587d005ff5c4a266b84dba06d`.

| Arm | Mean source-user macro accuracy |
|---|---:|
|ridge|93.063131%|
|mapped_mix|92.404040%|
|naive_mix|92.132576%|
|unmapped_mix|88.766414%|

- Primary vs ridge: -0.659091pp, 2/6 strict wins, worst -2.750000pp; passed=False.
- Primary vs naive_mix: +0.271465pp, 4/6 strict wins, worst -0.500000pp; passed=True.
- Primary vs unmapped_mix: +3.637626pp, 5/6 strict wins, worst -0.090909pp; passed=True.

No eightfold expansion, mixture/label mapping/lambda/normalization/seed sweep or promotion of a secondary arm. Failure is limited to this frozen-encoder intervention, not all end-to-end temporal augmentation.

Six historically reused development users3–8, four training/next validation/held diagnostic, unchanged source24-epoch averaged checkpoints and128dim original CNN features. The primary mapped_mix adds one correctly preprocessed reversed example per eligible original training clip, with labels0↔1,2↔3,4↔7,5↔6,8↔9. Classes10(push) and11(empty) are left untouched. Only pairs present within each source training user's quality-retained labels are eligible; user6 class2 is therefore not augmented after the existing class3 quality exclusion. This does not create new class support. All other training examples remain original. Original inference is unchanged and does not require a query label or reversed query.

The naive_mix control flips already normalized cached input features along time and applies the same label map. The unmapped_mix negative control uses correct raw reversal but intentionally keeps the original label. It is a deliberately invalid invariance assumption; beating it alone proves no useful contribution. The primary must beat original strong ridge by0.5pp and both controls by0.25pp, each with4/6 strict wins and no more than1pp worst paired decline.

All three new arms fit their own training-only means/scales and user/class-balanced ridge on original plus augmented rows. Matched100-per-class training counts mean each eligible class's original and synthetic mass is split equally. Fixed lambdas0.001/0.01/0.1 are selected using only next validation user macro accuracy, onehot MSE, then larger lambda. No diagnostic labels enter fitting, mapping, eligibility, preprocessing or selection; all predictions are saved before held evaluation.54 candidate fits,18 selected readouts; no new encoder training. Repeated development use prevents treating this as untouched confirmation. The eightfold strong ridge94.897727% is a different protocol, not a comparator for these six-source means.

Raw90-frame reversal requires recomputing complex analytic increments, per-clip standardization, historical first-increment duplication and float16/range pooling. Simply reversing the cached signed phase image misses conjugation and changes the wrong boundary. A synthetic independent SciPy-Hilbert derivation matched correct preprocessing exactly; naive input differed by up to4.626953125 on that synthetic example. This is an implementation distinction, not evidence of real accuracy gain or physical reversibility. Reversal changes execution speed profiles, posture/hand scattering and clip boundaries; semantic label pairs do not certify natural reversed UWB observations.

Independent audit reconstructed all23200 augmented training appearances from original RF using SciPy Hilbert plus conjugate/reordered increments, replayed46400 functional CNN forwards, and solved54 augmented-intercept normal equations. Maximum hidden error0, maximum prediction error6.92779167366e-14; all selected and candidate argmax agree. 51 artifact SHA checks, roles, IDs, weights, validation selections and gate arithmetic passed. sklearn cholesky separately matched all54 candidate validation score systems during fitting. Duplicate-original augmentation is algebraically/numerically identical to ordinary ridge under the class weights. Actual run378.868760s; training16 timing projected815.982647s under1200s budget. Independent audit240.501334s. No active background training after completion.

Literature: [Price and Damen, Retro-Actions2019](https://arxiv.org/html/1909.09422) already establishes label-altering temporal augmentation and discusses reversal artifacts. The current frozen-radar implementation is not their video-training replication. Reviewed HTML sections are specified in research/TEMPORAL_REVERSAL_REVIEW.md; CVF PDF returned403. Necessary recomputation of the radar phase representation does not make known relabeling plus ridge an original method. Original project already cautions against label-preserving reversal and exact CW/CCW reversibility. No new verified contribution is claimed.

Reproduction after restoring raw/prepared and prior private checkpoint/feature/readout receipts, using fresh output directories:

```sh
python uwb_direction1/tools/test_temporal_reversal.py
python uwb_direction1/tools/run_temporal_reversal.py --protocol uwb_direction1/research/TEMPORAL_REVERSAL_PROTOCOL.json --out /workspace/reversal-replay-cost --cost-only
python uwb_direction1/tools/run_temporal_reversal.py --protocol uwb_direction1/research/TEMPORAL_REVERSAL_PROTOCOL.json --out /workspace/reversal-replay --cost /workspace/reversal-replay-cost/COST.json
python uwb_direction1/tools/audit_temporal_reversal.py /workspace/reversal-replay
```

All source/augmented IDs, reverse hidden features, candidate coefficients, selection records, predictions, configuration, audit and logs are retained. Final private Drive receipt is authoritative; archived report may precede that receipt. Original raw arrays are not uploaded to Git;789-file source snapshot unchanged. Goal remains unmet and automatic continuation remains enabled.

私有备份已逐份下载复验：7块、37622351字节、87成员SHA全部一致，元数据shared=false且仅owner；未更改共享权限。

Documentation erratum: the preregistered review wrote89−t in its prose. With frames0–89 and increments1–89, the exact reversed increment is conjugate(d_(90−t)). Corrected in the Git review and literature receipt; executable operations and independently verified artifacts never changed. The immutable private provenance retains the earlier prose version.
