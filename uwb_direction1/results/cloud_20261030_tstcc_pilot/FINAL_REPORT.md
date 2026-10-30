Full-scale TS-TCC multivariate adaptation completed and passed full replay audit.
It improves over its matched40epoch scratch control, but does not reach the
registered strongest-reference gate. Close this exact configuration; no rescue.

| Classifier pipeline | Held-user1 present-class macro |
| --- | ---: |
| SSL40 + frozen-parameter linear40 | 85.000000% |
| SSL40 + full fine-tune40 | 97.500000% |
| Samearchitecture supervised scratch40 | 96.166667% |
| Frozen strongest prior same-held1 reference | 98.666667% |

Fine-tune-minus-scratch:+1.333333pp. Fine-tune-minus-reference:-1.166667pp.
The preregistered rule requires any named arm at least+.5pp above reference;
none passes, route=close_exact_configuration. The within-experiment gain is
useful exploratory accuracy evidence, not an improvement over the strongest
known pipeline or an independent eight-user result. SSL adds40pretraining epochs
and augmentations relative to scratch40, so this does not isolate a causal
contrastive-loss mechanism or computationally matched benefit. No originality
claim follows from combining published modules or observing this difference.

Original author HAR internal widths32/64/128, TC100/depth4/heads4/sixfuturesteps,
batch128, seed20261029, each40epochs retained. Four phases total160epochs,
5760updates/737280logged training appearances. Model247500+TC313096parameters.
Full567variables x90frames, record/view RMS; trainusers3–6=4700, val2=1000,
held1=1200, existing quality exclusions. Train36x128=4608records each epoch.
Original frozen-parameter linear readout retains train-mode dropout and BN
running-stat updates; fine-tune/scratch train all parameters. Fresh same heads
and Adam, source validation-only scheduler, final40epoch checkpoint selection.

Documented adaptations: NumPy2 ragged segment ordering, all-channel shared time
permutation replacing original channel0broadcast, explicit radar axes, dedicated
fixed sampler/RNG, training-only fixed augmentation and full1000validation.
Original code Gaussian gains/jitter/max_seg8 retained, with paper differences
explicit. Streaming augmentation agrees with corrected full reference and RNG.
This is not a physically validated radar augmentation or exact original GPU
paper replication. Both source-code parity and complete checkpoint restoration
were tested prospectively. Paper read scope is partial publisher HTML; formula
extraction and figures were not fully reviewed.

Preregistered67bf23dc7acc41636caf38dd0e5ca69068478cc4; protocol SHA
024d044ce1a619526b8072672e1b76bf9623748ddc57dbcab0d937fe2be43d3a.
All71scientific hashes unchanged. All3classifier prediction files froze before
held1labels were indexed. Users7–8 labels excluded from fit/selection/scoring.
All eight users historically developed; this is exploratory, not independent.
No comparison is made to the differently defined legacy LTR eightfold93.4417%.

Training supervisor exit0/no timeout:1054.189592306wall/1053.985326childCPU.
Worker summary1050.995685025wall/1053.134215165CPU is a separate measurement.
Controller waited for existing training; no duplicate training or audit retry.
Completed-run audit exit0/no timeout:1019.823728741wall/1019.641631childCPU.
It checked130artifact hashes, replayed all5760updates/737280sample appearances,
all4phases initial/first/10/20/30/40/last models+optimizers+scheduler+allRNG
bitwise; independently regenerated6900RMS inputs,4700fixed augmented views and
recipe, all120validation epochs and6600final validation/diagnostic logit rows.
Saved logits/predictions match exactly; independently reduced macro tolerance
1e-12 handles the linear-score oneULP difference. Audit code SHA
4b7afea885e400b6bcaa879b175614fb574b7a97621cfc9bbf3e50f564440a5c.

Completed standalone private model recovery:111188908bytes/16chunks,
150payload+manifest=151members, SHA
a54bd743b24beb6f61b94ae607bcc34a47ae272caff9daed7f639ef235d118c4.
All16chunk privacy/parent/size/MIME metadata checked, authenticatedrawfetch
include_base64=false immediately materialized and SHA/size checked, archive
reassembled and all150payload member hashes verified. Complete receipt SHA
a573dc4a62811d02c2a4712a797fa53b612b80d4c610458b4adea0fbeed48a89.
Includes all four phases' complete checkpoint states, IDs/logs/predictions,
fixed augmentation recipe, source/dependencies and audit. No dependency on the
older first-update insurance bundle. Original raw maps and large augmented
arrays excluded; existing migration/RMS and pinned recipe regenerate them.
No automatic full-job resume claim. Final human report was written after this
immutable bundle; machine SUMMARY/audit and checkpoints are inside the bundle.

No active worker remains. Accuracy/contribution goal unresolved; continuation
remains enabled. Next prepared candidate is published TimesNet official complete
UWaveGestureLibrary classification preset, not a hand-shrunk model or a reuse
of TS-TCC. Its source/synthetic/cost status must be checked before any real fit.
