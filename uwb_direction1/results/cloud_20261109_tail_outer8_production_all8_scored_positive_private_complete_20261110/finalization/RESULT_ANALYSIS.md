# UWB Direction1: prospective matched full8 fixed-tail result

Whole-cohort once score: native **96.4473%** → fixed tail **97.2845%**, **+0.8371 percentage points**. Predeclared accuracy gate **passed**; 6 wins / 1 ties / 1 losses. Worst paired accuracy delta -0.2500pp.

This is an exploratory matched transfer experiment on eight historically exposed users. It establishes the observed accuracy difference for this fixed configuration/seed/cohort; it does not establish independent new-user validation, originality, a causal mechanism, or exact reproduction of the original SWA benchmark. No test scores selected method/window/LR/seed. Users7–8 enter fixed training/validation and this final report after method selection.

## Protocol and evidence

Frozen before new users7–8 method selection: test h=1..8, validation h%8+1, other six train; original6900 development IDs plus all2400 users7/8 IDs. Three full radars/signed per-record per-radar RMS, native GAP617612, global12-class BCE/B64/64epochs/seed20261029/AdamWLookahead. Native validation checkpoint and passive dense tail post-updates epochs33–64 share one trajectory. Tail and native_sourceBN use source-only BN; native_sourceBN remains descriptive control. Every fold trained64 epochs even when its checkpoint was selected earlier.

All8 production and fresh replay64 (55936 updates each),65 epoch records per fold, native optimizer/RNG/fullstate and224 frozen-file checks, all three-arm output audits,144 owner-only private chunks/2512 payload files, and24 blind saved-model test-output recomputations preceded one complete cohort score. The exclusive once lock remains consumed. Actual sample IDs/truth/predictions stay private. No old MMD/singlefold/SWAD scorer or refit was invoked.

## Per-user metrics

|Held user|n|Native accuracy|Tail accuracy|Accuracy delta pp|Native present macro recall|Tail present macro recall|Missing 0-based classes|
|---:|---:|---:|---:|---:|---:|---:|---|
|1|1200|98.5833%|99.5000%|+0.9167|98.5833%|99.5000%|none|
|2|1000|99.7000%|99.7000%|+0.0000|99.7000%|99.7000%|10,11|
|3|1200|91.3333%|91.0833%|-0.2500|91.3333%|91.0833%|none|
|4|1200|97.2500%|98.2500%|+1.0000|97.2500%|98.2500%|none|
|5|1200|98.6667%|99.1667%|+0.5000|98.6667%|99.1667%|none|
|6|1100|92.5455%|94.9091%|+2.3636|92.5455%|94.9091%|3|
|7|1200|96.1667%|98.0000%|+1.8333|96.1667%|98.0000%|none|
|8|1200|97.3333%|97.6667%|+0.3333|97.3333%|97.6667%|none|

Accuracy is correct/total. Present-class macro recall averages per-class recalls only where true support>0; absent-class recall is null, not zero. Here each present class has100 examples, so the two metrics coincide numerically per user; their definitions remain different. Predictions to an absent class count as errors. Per-class recalls for every arm are retained in SUMMARY.json.

|Held|c0|c1|c2|c3|c4|c5|c6|c7|c8|c9|c10|c11|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|100|100|100|100|100|100|100|100|100|100|100|100|
|2|100|100|100|100|100|100|100|100|100|100|0|0|
|3|100|100|100|100|100|100|100|100|100|100|100|100|
|4|100|100|100|100|100|100|100|100|100|100|100|100|
|5|100|100|100|100|100|100|100|100|100|100|100|100|
|6|100|100|100|0|100|100|100|100|100|100|100|100|
|7|100|100|100|100|100|100|100|100|100|100|100|100|
|8|100|100|100|100|100|100|100|100|100|100|100|100|

## Aggregates and fixed gate

|Arm|Equal-user accuracy|Equal-user present macro|Worst user accuracy|Worst present macro|Pooled9300-sample accuracy|
|---|---:|---:|---:|---:|---:|
|native|96.4473%|96.4473%|91.3333%|91.3333%|96.4194%|
|tail|97.2845%|97.2845%|91.0833%|91.0833%|97.2581%|
|native_sourceBN|96.5684%|96.5684%|91.3333%|91.3333%|96.5376%|

Primary mean paired accuracy gain +0.8371pp; present-macro gain +0.8371pp. Native_sourceBN descriptive mean accuracy gain +0.1210pp. This control does not change the primary method or establish isolated causality.

Fixed gate: mean accuracy gain≥0.5pp, at least5/8 strict user improvements, worst paired accuracy degradation≤2pp, tolerance1e−12. Positive and negative effects are retained regardless of the gate. User3 drops0.25pp; worst-user accuracy also drops from91.3333% to91.0833%. Report this limitation alongside the higher mean. No original contribution claim follows automatically from the gate.

## Training, reproduction and inference costs

The phase raw ledger CPU measures training/replay/audit workers plus controllers/guards. Stage wall below derives same-boot worker startticks/HZ to waited completion, with kernel-tick/polling uncertainty; it excludes interfold idle periods. Training includes validation/checkpoint/output/sourceBN and passive collector work, not isolated gradient-only time. Standalone native-baseline runtime was not measured separately; both arms share a trajectory.

|Stage|Wall seconds|Local CPU seconds|
|---|---:|---:|
|output_audit|472.609|471.783|
|replay|32009.742|32084.607|
|train|32279.943|32376.978|

|Arm|Model-only forward wall s|RMS + forward wall s|Model-only forward CPU s|RMS + forward CPU s|
|---|---:|---:|---:|---:|
|native|19.823700|22.574888|19.821295|22.572165|
|tail|19.735081|22.423621|19.731793|22.420350|
|native_sourceBN|19.610668|22.227802|19.608837|22.226086|

Inference is one CPU/single-thread/B64 pass per saved model, no warmup/repetition distribution. Model loading/hash/input preparation and complete process-family costs are recorded separately. Blind24 process81.829945wall/82.224117CPU seconds; scoring once process3.678546wall/4.092324CPU seconds. Archive build/verification measurements, helper budget allowances, whole-phase waiting wall, and final accounting are separate; never sum cumulative stage totals or count transferwall twice. FINALIZATION_COST_SUMMARY.json is the terminal accounting record, including final private report backup. Same original129600-second wall/localCPU cap,12GiB RSS/256MiB reserve; no reset. Provider CPU unmeasured.

## Prior evidence and literature

Old GAP64 development6 mean93.6672%, fixed mixed64+100 mean94.6290% and old LTR93.4417% retain their original identities/failed spending gates and different training budgets. They are historical references, not this full8 paired baseline. Closed MMD/VREx/Fishr/IRM/GroupDRO/SWAD results stay closed; the earlier SWAD tail6 descriptive control is not retrospectively promoted. PRIOR_RESEARCH_COSTS.json retains all closed-study costs and MMD firstfold5022.569200wall/5019.332419CPU separately and in total.

Literature: [Izmailov et al., UAI2018, SWA](https://arxiv.org/abs/1803.05407v3); [pinned BSD2 author code411b2fc](https://github.com/timgaripov/swa/tree/411b2fcad59bec60c6c9eb1eb19ab906540e5ea2). Our AdamWLookahead/dense post-update averaging33–64/equal-batch momentum=None sourceBN is a documented adaptation, not the original SGD/epoch-end/sample-weighted-BN protocol. Method selection and source review precede the new full8 result.

The predeclared efficacy target passed. New fits and method/window/seed searches are paused; existing hourly task/environment/enabled state stays unchanged. Preserve this complete positive/negative exploratory evidence and all private recoveries.
