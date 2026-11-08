# Frozen IRM100/500 native LOSO6 exploratory result

GAP mean present-class macro recall 93.667172%; IRM 89.423990%; paired mean -4.243182 percentage points. Fixed spending gate passed: False.

Route: `close_exact_IRM_LOSO6_lambda100_warmup1_anneal500_nativeGAP64_seed20261029_no_rescue`. The one cohort scorer is consumed; no individual held score selected the configuration, epochs, seed or which folds were run. A failed spending gate does not erase any positive per-user effect. No original-contribution or independent-new-user validation claim.

|Held|GAP accuracy|IRM accuracy|GAP present macro|IRM present macro|Paired macro pp|n|Missing classes (0-based)|
|---|---:|---:|---:|---:|---:|---:|---|
|1|98.416667%|87.583333%|98.416667%|87.583333%|-10.833333|1200|[]|
|2|98.700000%|98.400000%|98.700000%|98.400000%|-0.300000|1000|[10, 11]|
|3|90.000000%|89.083333%|90.000000%|89.083333%|-0.916667|1200|[]|
|4|94.250000%|94.250000%|94.250000%|94.250000%|+0.000000|1200|[]|
|5|96.000000%|87.500000%|96.000000%|87.500000%|-8.500000|1200|[]|
|6|84.636364%|79.727273%|84.636364%|79.727273%|-4.909091|1100|[3]|

|Aggregate|GAP|IRM|
|---|---:|---:|
|equal_user_mean_macro_recall|93.667172%|89.423990%|
|equal_user_mean_accuracy|93.667172%|89.423990%|
|worst_user_macro_recall|84.636364%|79.727273%|
|pooled_sample_accuracy|93.652174%|89.304348%|

Strict wins: 0/6; exact ties: 1/6; worst paired macro difference: -10.833333 pp. Frozen gate: mean paired gain >=0.5pp, >=4 strict wins, worst paired decline <=2pp (registered tolerance1e-12).

Accuracy is the fraction of all records correct. Present-class macro recall averages only classes with positive true support; missing recalls are null, and predicting an absent class remains an error. Here every present class has100 records, so each user's two measures coincide numerically. This does not make the definitions interchangeable. Pooled sample accuracy is separately named and is not the equal-user mean.

|Held|Class supports (0..11)|GAP class recalls (null missing)|IRM class recalls (null missing)|
|---|---|---|---|
|1|[100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]|[1.0, 1.0, 0.99, 0.94, 1.0, 1.0, 1.0, 1.0, 0.92, 0.97, 0.99, 1.0]|[0.96, 0.96, 0.82, 0.93, 0.89, 0.98, 0.97, 0.94, 0.97, 0.95, 0.15, 0.99]|
|2|[100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 0]|[1.0, 1.0, 1.0, 1.0, 0.98, 0.94, 1.0, 1.0, 1.0, 0.95, null, null]|[1.0, 1.0, 0.97, 1.0, 1.0, 0.94, 0.99, 1.0, 0.96, 0.98, null, null]|
|3|[100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]|[1.0, 1.0, 0.98, 0.99, 1.0, 0.97, 1.0, 0.87, 1.0, 0.0, 1.0, 0.99]|[1.0, 1.0, 0.99, 1.0, 0.99, 0.96, 0.99, 0.78, 0.98, 0.0, 1.0, 1.0]|
|4|[100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]|[1.0, 1.0, 0.93, 0.99, 0.94, 1.0, 1.0, 0.98, 0.8, 1.0, 0.67, 1.0]|[1.0, 1.0, 0.98, 0.99, 0.89, 1.0, 1.0, 0.98, 0.89, 1.0, 0.58, 1.0]|
|5|[100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]|[1.0, 1.0, 1.0, 0.98, 0.98, 0.96, 0.94, 1.0, 0.69, 0.98, 1.0, 0.99]|[0.61, 1.0, 0.97, 0.8, 0.84, 0.79, 0.9, 0.99, 0.89, 0.75, 0.97, 0.99]|
|6|[100, 100, 100, 0, 100, 100, 100, 100, 100, 100, 100, 100]|[1.0, 1.0, 0.81, null, 0.93, 0.9, 0.7, 0.94, 0.45, 0.6, 0.99, 0.99]|[1.0, 1.0, 0.85, null, 0.94, 0.86, 0.55, 0.85, 0.68, 0.09, 0.95, 1.0]|

## Fixed protocol and reproducibility

Test h1..6; validation h%6+1; the other four subjects train. All role/sample IDs are disjoint within each fold, all6900 source IDs are preserved privately, all six models start from the same untrained native state with fresh optimizer/order/Python/NumPy/Torch RNG. Full GAP617612/all3 signed per-record RMS90x567/B64/64epochs/seed20261029/native global onehot12 BCE/AdamWLookahead are retained. IRM adds the published dummy-scale gradient-product term, warm weight1 before500 completed updates and100 thereafter, and resets the whole native optimizer before update501. Global BCE, native mixed-source batches, pooled BN/dropout and native optimizer reset are declared transfer differences from DomainBed; this is not an author-benchmark reproduction. Native reset is also a difference from the original GAP comparator. A positive cohort therefore needs the preregistered matched globalBCE/reset500/no-IRM six-fold control before penalty attribution or outer8.

All six full64 production and fresh64 replay trajectories, all65 epoch/journal records, every step/native model/optimizer/slow/all RNG/selected outputs were exact. Independent first-step dummy-scale autograd checked all617612 clipped gradients; full same-implementation replay alone cannot exclude common implementation bugs. Model selection was validation-only. All model/trajectory/ID/output private raw-all-member insurance and exact Git release preceded the single score. The saved label-free paired inference pass recomputed all baseline/method logits bitwise and checked parameters/buffers unchanged.

Full archive: 75 parts, 548348323 bytes, 1995 payloads, SHA256 `3436cbcc7bb738e30e5279acc9c998c3468a10954e66aa68ad9b08440494cf10`. Verification: 1996 tar members including the member manifest. Raw arrays/runtime binaries remain external via the persistent recovery chain. Extraction/end-to-end interrupted-job resume was not exercised in the intact environment.

Published sources: Arjovsky et al., Invariant Risk Minimization, arXiv1907.02893 (method sections3.1-3.2 reviewed); pinned DomainBed commit b93c22a1cfc3b2428398272c1a116c8de1f4139e. No coefficient, kernel, seed, epoch or fold rescue is permitted after this score. Users7-8 did not select this method; historical exposure to all8 users makes this exploratory, not independent validation.

## Costs

Stage costs below are differences of saved cumulative per-fold timers, not sums of cumulative timers. Production includes native loading/validation and the first-step auditor; it is not an isolated pure training kernel measurement. Independent fresh replay is reproduction cost, not extra optimization for the selected production model.

|Held|Production stage wall/CPU s|Fresh replay stage wall/CPU s|Output stage wall/CPU s|Selected epoch (validation)|
|---|---:|---:|---:|---:|
|1|2490.560512/2489.344770|2507.470335/2506.872969|11.021065/10.775330|46|
|2|2549.570262/2549.559967|2515.511078/2514.141669|10.017253/10.298878|62|
|3|2419.206439/2419.140241|2372.165493/2371.170288|11.021619/11.059570|5|
|4|2412.341797/2410.495532|2392.209057/2391.547727|11.024941/11.175357|62|
|5|2428.408536/2428.148916|2435.140124/2434.569721|11.018330/10.690938|47|
|6|2450.115450/2450.264461|2452.199260/2450.905025|10.019286/10.192894|6|

Whole IRM phase including fold supervision: wall 29509.953317s / CPU 29548.175222s; whole-phase12h wall/CPU caps were not reset between folds. Production updates27776; fresh replay updates27776. Do not add the per-fold stage totals again to this whole-phase total. Native earlier GAP artifacts retain their own historical cost/audit scope (initial/first/selected/output audits, not an asserted full64 fresh replay).

|Inference arm (saved label-free fixed pass)|Total measured wall s|Total measured CPU s|Wall ms/record|
|---|---:|---:|---:|
|baseline|15.740581|15.730491|2.281244|
|method|15.299903|15.299779|2.217377|

Full inference worker including loading/checks/warmup: wall 37.574744s / CPU 37.949419s; RSS 1789272KiB. Each saved row separately reports warmup. Scorer evaluation wall 4.032978s.

Insurance/build/network/operator cost receipts are separate artifacts. The full archive build records parent-process CPU only and does not measure child build CPU; that unknown value is not zero. Full network/remote connector CPU is unmeasured. Previous per-fold insurance and this full archive overlap in coverage and must not be called additional model fits. All prior MMD/VREx/Fishr research costs remain separately reported:
- MMD: wall 30865.434969089023s / CPU 30891.040083422995s (prior MMD held1 is a subcost, never added twice).
- VREx: wall 31157.163366941997s / CPU 31189.589142517s (prior MMD held1 is a subcost, never added twice).
- Fishr: wall 29559.94331930898s / CPU 29599.386025808s (prior MMD held1 is a subcost, never added twice).

Historical fixed TimeMIL+Shuffle94.6290% used a different64+100 budget and retained its original failed spending conclusion; it is a historical strong reference, not this matched comparator. Existing old LTR/frozen cross-user penalty and every closed MMD/VREx/Fishr configuration remain unchanged.
