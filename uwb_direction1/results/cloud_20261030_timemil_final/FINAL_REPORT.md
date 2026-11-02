# TimeMIL/GAP fixed64epoch source pilot — final audited result, 2026-10-30

Both registered arms completed64epochs at seed20261029. Train users3–6(4700records),validation2(1000),diagnostic1(1200); original quality exclusions and source-only6900manifest unchanged. Each best checkpoint selected only by validation macro/BCE/earlier epoch; both diagnostic predictions froze before diagnostic labels were scored. Users7–8 labels unused. This is a published-method accuracy transfer, not an original-method or exact300epoch paper reproduction.

| Method | Held1 present-class macro recall | Selected epoch |
|---|---:|---:|
| TimeMIL128embedding/full author architecture |98.666667%|41|
| Matched Inception backbone with GAP |98.416667%|60|
| Prior local CNN24epoch reference |96.833333%|23|
| Prior full fusion CNN24epoch reference |96.000000%|as previously audited|

TimeMIL−GAP=+0.250000pp, below the prospectively fixed+0.5pp mechanism gate. TimeMIL is above strongest prior CNN−0.5pp, but the jointgate is false. Close this exact64epoch TimeMIL configuration; no seed/epoch/LR/normalization rescue, no source-sixfold TimeMIL expansion. This does not reject the author's300epoch method or all MIL models.

GAP−strongest priorCNN=+1.583333pp; the separately prospectively registered+1pp baseline-expansion eligibility criterion passes. Proceed to a separately frozen six-source-user GAP confirmation against existing matched-role localCNN folds. That positive pilot is selection evidence, not independent confirmation. Do not compare this held1 score to the sixfold mean or legacy8fold93.4417% as a gain.

Independent audit passed44artifact/input/code hashes; exact IDs/roles, all6900NumPy preprocessing records, initial/first-update/final optimizer states (4736steps each), RNG/BN/unused parameters/wavelet aliases, validation-only checkpoint selection, all1000validation+1200diagnostic forwards for each arm, BCE/argmax/metrics and both original gates. Fixed tolerances preserved; no nonselected epoch retraining/replay. AuditorSHA09b34fa89f9e9b85685a0c500f56912a1df70f3dc4c2d6d5695c2aa8969afdcc. Audit process exited0 in39.694439wall/39.674359CPU seconds (<890wall/880soft890hardCPU).

Training exited0/no timeout in11537.791314supervisor wall/11536.040844childCPU seconds. Including preserved attempt01 pre-data failure:11540.564732wall/11538.764409CPU seconds, both below43200. First attempt missing-os failure had zero fits/samples/steps/scores and remains preserved; run02 science unchanged.

Inputs retain all189range bins×3views×90frames, per-record/view RMS, no cross-record normalization/resize. Full850973parameter TimeMIL and matched617612parameter GAP use AdamW.001/decay.0001 plus stable author Lookahead(.5,k6),unweighted12class OvRBCE,batch64,clip2,noaugmentation/scheduler. Paper/model-source Chenetal.ICML2024 PMLR235:7190–7206 https://proceedings.mlr.press/v235/chen24af.html; author7542c1c275aae3a9db35ee5c9bdfc0ccf386cd5c. Read scope and paper/CLI warmup/LR/embed/device/serialization adaptations remain in REVIEW_AND_PREREGISTRATION.md. GAP is an established Inception-family temporal baseline, not new architecture.

Source users have repeated historical development. This single source-user experiment and later sixfold confirmation remain exploratory; no independent validation, users7–8 selection or original contribution claim. Overall user goal unresolved; continuation remains enabled.
